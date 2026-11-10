# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ImageDepthIntrinsicsDataset - Optimized dataloader for depth-to-pointcloud conversion.

Performance Optimizations (vs original version):
1. **Cached Ray/Grid Computation**: Pre-compute and cache all ray arrays and backprojection
   grids. With resize_height, common sizes are pre-cached in __init__. Eliminates expensive
   per-sample recomputation.

2. **GPU-Accelerated Conversion**: All depth->pointcloud operations use vectorized torch ops
   instead of numpy loops. Optional GPU mode (use_gpu_conversion=True) for even faster processing.

3. **Fast Resizing**: Replace PIL resize with torch.nn.functional.interpolate (3-5x faster).

4. **Smart Caching**: VKitti2 backprojection grids cached by intrinsics tuple (many frames
   share the same camera parameters after rounding).

Expected speedup: 10-50x depending on dataset, resize settings, and GPU usage.

Usage:
    # CPU mode (still much faster than before)
    dataset = ImageDepthIntrinsicsDataset(..., use_gpu_conversion=False)

    # GPU mode (fastest, requires CUDA)
    dataset = ImageDepthIntrinsicsDataset(..., use_gpu_conversion=True)
"""

import json
import os
import glob
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import torchvision
import random
from torchvision.transforms import v2
import pickle

from dataloader.h5_shards import (
    H5_SHARD_DATASET_NAMES,
    find_h5_shards,
    orient_toward_camera,
    shard_data_pair,
    vkitti_benchmark_crop_image,
)
import math

import time
import bisect  # For binary search

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Paired crop classes and helper functions
from dataloader.point_transforms import (
    RandomCropPaired,
    CenterCropPaired,
    RandomCropPairedWithOffset,
    CenterCropPairedWithOffset,
    compute_distance_statistic,
    random_frame_crop,
)


# --- SceneNet-RGBD Helper Functions ---

def compute_scenenet_intrinsics(width=320, height=240):
    """
    Compute intrinsics for SceneNet from its fixed FOV values.

    SceneNet uses hfov=60°, vfov=45°.
    fx = width / (2 * tan(hfov/2))
    fy = height / (2 * tan(vfov/2))
    cx = width / 2
    cy = height / 2

    Returns: (fx, fy, cx, cy)
    """
    hfov = 60.0
    vfov = 45.0
    fx = width / (2.0 * math.tan(math.radians(hfov / 2.0)))
    fy = height / (2.0 * math.tan(math.radians(vfov / 2.0)))
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def scenenet_ray_directions(width=320, height=240, device='cpu'):
    """
    Unit ray direction per pixel under the SceneNet-RGBD camera model.

    SceneNet-RGBD stores Euclidean ray-depth rather than planar z-depth, so
    recovering points needs the direction of each pixel's ray. Directions come
    straight from the pinhole model of compute_scenenet_intrinsics() above:
    un-project the pixel centre (u, v) = (x + 0.5, y + 0.5) to
    ((u - cx) / fx, (v - cy) / fy, 1) and normalise to unit length.

    Args:
        width, height: Image dimensions
        device: 'cpu' or 'cuda'

    Returns: (H, W, 3) tensor of unit rays
    """
    fx, fy, cx, cy = compute_scenenet_intrinsics(width=width, height=height)

    x = torch.arange(width, dtype=torch.float32, device=device)
    y = torch.arange(height, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # (H, W)

    # Pixel centres, un-projected onto the z = 1 plane.
    x_vect = (xx + 0.5 - cx) / fx
    y_vect = (yy + 0.5 - cy) / fy
    z_vect = torch.ones_like(x_vect)

    rays = torch.stack([x_vect, y_vect, z_vect], dim=-1)  # (H, W, 3)
    return rays / torch.norm(rays, dim=-1, keepdim=True)


def raydepth_to_points(depth_map, ray_directions):
    """
    Convert Euclidean ray-depth to 3D camera-frame points.

    Args:
        depth_map: (H, W) torch tensor of ray-depth values in meters
        ray_directions: (H, W, 3) torch tensor of unit ray directions

    Returns:
        (H, W, 3) torch tensor of 3D points in camera coordinates
    """
    return ray_directions * depth_map.unsqueeze(-1)


# --- TartanAir and VKitti2 Helper Functions ---

def compute_backprojection_grid(width, height, fx, fy, cx, cy, device='cpu'):
    """
    Computes a grid of normalized coordinates for back-projection (GPU-accelerated).
    Used for z-depth to 3D conversion (TartanAir, VKitti2).

    Args:
        width, height: Image dimensions
        fx, fy, cx, cy: Camera intrinsics
        device: 'cpu' or 'cuda'

    Returns: (H, W, 3) tensor where:
        grid[:,:,0] = (u - cx) / fx
        grid[:,:,1] = (v - cy) / fy
        grid[:,:,2] = 1.0
    """
    u = torch.arange(width, dtype=torch.float32, device=device)
    v = torch.arange(height, dtype=torch.float32, device=device)
    vv, uu = torch.meshgrid(v, u, indexing='ij')  # (H, W)

    x_norm = (uu - cx) / fx
    y_norm = (vv - cy) / fy
    ones = torch.ones_like(x_norm)

    grid = torch.stack([x_norm, y_norm, ones], dim=-1)  # (H, W, 3)
    return grid


def points_in_camera_coords_zdepth(depth_map, backprojection_grid):
    """
    Convert z-depth map to 3D point cloud using pre-computed grid (GPU-accelerated).

    Args:
        depth_map: (H, W) torch tensor of z-depth values in meters
        backprojection_grid: (H, W, 3) torch tensor of normalized coordinates

    Returns:
        (H, W, 3) torch tensor of 3D points in camera coordinates
    """
    return backprojection_grid * depth_map.unsqueeze(-1)


# --- VKitti2 Intrinsics Parsing ---

def load_vkitti_intrinsics(intrinsic_path):
    """
    Parse Virtual KITTI 2 intrinsic.txt file.

    Returns: dict mapping (frame_idx, camera_id) -> (fx, fy, cx, cy)
    """
    intrinsics = {}
    if not os.path.exists(intrinsic_path):
        return intrinsics

    with open(intrinsic_path, 'r') as f:
        lines = f.readlines()
        # Skip header: frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
        for line in lines[1:]:
            parts = line.strip().split()
            if len(parts) < 6:
                continue

            frame_idx = int(parts[0])
            camera_id = int(parts[1])
            fx = float(parts[2])  # K[0,0]
            fy = float(parts[3])  # K[1,1]
            cx = float(parts[4])  # K[0,2]
            cy = float(parts[5])  # K[1,2]

            intrinsics[(frame_idx, camera_id)] = (fx, fy, cx, cy)

    return intrinsics


# --- TartanAir V2 Depth Decoding ---

def read_decode_tartanairv2_depth(depthpath):
    """
    Decode TartanAir V2 depth image from RGBA PNG to float32.

    TartanAir V2 stores float32 depth values in 4-channel 8-bit RGBA PNGs.
    Each pixel's 4 bytes represent a single float32 value.

    Args:
        depthpath: Path to depth PNG file

    Returns:
        (H, W) numpy array of float32 depth values in meters
    """
    import cv2
    depth_rgba = cv2.imread(depthpath, cv2.IMREAD_UNCHANGED)
    depth = depth_rgba.view("<f4")
    return np.squeeze(depth, axis=-1)


# --- OmniWorld-Game Depth Decoding ---

def load_omniworldgame_depth(depthpath):
    """
    Load and decode OmniWorld-Game depth from 16-bit PNG.

    OmniWorld-Game stores depth as 16-bit unsigned integers in [0, 65535].
    Values 0-100 are invalid (too close), 65500-65535 are sky/far.
    The depth values use a disparity-like encoding that needs conversion.

    Args:
        depthpath: Path to depth PNG file

    Returns:
        depthmap: (H, W) float32 array with z-depth in meters
        valid: (H, W) bool array, True for reliable pixels
    """
    import imageio.v2

    # Load 16-bit depth
    depthmap = imageio.v2.imread(depthpath).astype(np.float32) / 65535.0

    # Identify invalid regions
    near_mask = depthmap < 0.0015  # too close
    far_mask = depthmap > (65500.0 / 65535.0)  # sky/too far

    # Convert from disparity-like encoding to z-depth
    # Formula: depth_z = depth_normalized / (far - depth_normalized * (far - near)) / scale
    near, far = 1.0, 1000.0
    depthmap = depthmap / (far - depthmap * (far - near)) / 0.004

    # Mark invalid pixels
    # valid = ~(near_mask | far_mask)
    # only remove near for now, since we want to keep sky points for sky dome handling
    valid = ~near_mask
    depthmap[~valid] = 0.0

    return depthmap, valid


# --- Dataset Class ---

class ImageDepthIntrinsicsDataset(Dataset):
    """
    PyTorch Dataset for paired RGB and Depth data with on-the-fly pointcloud conversion.

    Supports multiple datasets with different raw file structures:
    - SceneNet-RGBD: ray-depth in .png files
    - TartanAir: z-depth in .npy files
    - TartanAir V2: z-depth in RGBA-encoded .png files (float32)
    - VKitti2: z-depth in .png files with per-frame intrinsics
    - Hypersim: z-depth in .npy files with per-frame intrinsics in .npz
    - UrbanSyn: z-depth in .npy files (preprocessed) with per-frame intrinsics in .npz
    - MVSSynth: z-depth in .npy files (preprocessed from EXR) with per-frame intrinsics in .npz
    - Synscapes: z-depth in .npy files (preprocessed from EXR) with per-frame intrinsics in .npz
    - OmniWorld-Game: z-depth in 16-bit .png files with per-split intrinsics in .json
    - IRS: z-depth in .npy files (preprocessed from disparity EXR) with per-frame intrinsics in .npz
    - Dynamic Replica: z-depth in .npy files with per-frame intrinsics in .npz
    - EDEN: z-depth in .npy files with per-frame intrinsics in .npz

    Converts (image, depth, intrinsics) -> (image, pointcloud).

    Features:
        - Scene-level splitting (Manual via .txt or Auto via uniform sampling)
        - Uniform subsampling of K frames per scene
        - Optional resize before crop (with intrinsics scaling)
        - Paired cropping (Random for Train, Center for Test)
        - Paired image/point-map augmentations
        - Same normalization pipeline

    Resize Pipeline:
        - Image: BILINEAR interpolation
        - Depth: NEAREST interpolation (preserves depth values)
        - Intrinsics: Scaled proportionally to new dimensions
        - Applied before augmentations and crop
    """

    def __init__(
        self,
        root_dir,
        dataset_name,  # 'scenenet', 'tartanair', 'tartanairv2', 'vkitti2', 'hypersim', 'urbansyn', 'mvssynth', 'synscapes', 'omniworldgame', 'irs', 'dynamic_replica', 'eden'
        split='train',
        num_test_scenes=2,
        test_scenes_file=None,
        samples_per_scene=None,
        crop_size=None,
        resize_height=None,  # NEW: Resize before crop (int or dict)
        resize_width=None,  # NEW: Resize to fixed width before crop (int or dict)
        center_shift=False,
        normalize_by_mean=False,
        num_overfit_samples=None,
        num_dataset_duplicates=None,
        max_depth=10.,
        subsample_scenes=None,
        center_shift_z_only=False,
        more_img_aug=False,
        stronger_img_aug=False,
        compute_scale_factor_only_valid=False,
        compute_scale_factor_use_std=False,
        compute_scale_factor_use_percentile=False,
        clamp_max_depth=None,
        handle_sky=False,
        use_sky_dome=False,
        sky_far_plane_value=3.0,
        sky_normal_dome=False,
        sky_instance_background=False,
        debug_timing=False,
        use_gpu_conversion=False,  # NEW: Use GPU for depth->pointcloud conversion
        remove_outliers=False,  # NEW: Remove points with extreme normalized coordinates
        outlier_threshold=3.0,  # NEW: Threshold for outlier removal (absolute value)
        min_height=None,  # NEW: Filter out samples below this height
        min_width=None,  # NEW: Filter out samples below this width
        scale_factor_augment=False,
        scale_factor_augment_range=(0.8, 1.2),
        no_scale_factor=False,
        # High-res finetuning mode parameters
        finetune_highres_mode=False,  # Enable adaptive resize for high-res finetuning
        finetune_target_height=512,   # Target height for finetuning
        finetune_target_width=None,   # Target width for finetuning (None = height-only mode)
        finetune_target_crop=512,     # Crop size for finetuning (int for square, [h,w] for rectangular)
        load_instance=False,  # h5 shards only: also return gen2seg instance ids (instance, instance_background_supervised)
        instance_counts=None,  # hypersim_h5: json of scripts/train_prepare_hypersim_instance_counts.py
        instance_min_scene_objects=0,  # gen2seg: scenes with fewer annotated objects give no instance supervision
        instance_loss=True,  # False: the ids are loaded for the batch but no sample enters the instance loss
    ):
        """
        Args:
            root_dir (str): Root directory of the dataset
            dataset_name (str): One of 'scenenet', 'tartanair', 'tartanairv2', 'vkitti2', 'hypersim', 'urbansyn', 'mvssynth', 'synscapes', 'omniworldgame', 'irs', 'dynamic_replica', 'eden'
            split (str): 'train', 'test', or 'all'
            resize_height (int or dict, optional): Resize height before crop.
                If int: applies to all datasets. If dict: per-dataset values.
                Example: 512 or {'scenenet': 256, 'tartanair': 512}
            use_gpu_conversion (bool): If True, perform depth->pointcloud conversion on GPU.
                Significantly faster but requires CUDA. Default: False
            ... (see the constructor signature below for the remaining args)

        Performance Optimizations:
            - Ray arrays and backprojection grids are cached (no recomputation)
            - Resizing uses torch.nn.functional.interpolate (faster than PIL)
            - Depth->pointcloud conversion uses vectorized torch ops (no loops)
            - Optional GPU acceleration for all conversion operations
        """
        self.root_dir = root_dir
        self.dataset_name = dataset_name.lower()
        self.crop_size = crop_size
        self.split = split.lower()
        self.num_test_scenes = num_test_scenes
        self.test_scenes_file = test_scenes_file
        self.samples_per_scene = samples_per_scene
        self.center_shift = center_shift
        self.center_shift_z_only = center_shift_z_only
        self.normalize_by_mean = normalize_by_mean
        self.max_depth = max_depth

        self.subsample_scenes = subsample_scenes
        self.more_img_aug = more_img_aug
        self.stronger_img_aug = stronger_img_aug
        self.compute_scale_factor_only_valid = compute_scale_factor_only_valid
        self.compute_scale_factor_use_std = compute_scale_factor_use_std
        self.compute_scale_factor_use_percentile = compute_scale_factor_use_percentile
        self.clamp_max_depth = clamp_max_depth
        self.handle_sky = handle_sky
        self.use_sky_dome = use_sky_dome
        self.sky_far_plane_value = sky_far_plane_value
        self.sky_normal_dome = sky_normal_dome
        self.sky_instance_background = sky_instance_background
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        self.debug_timing = debug_timing
        self.min_height = min_height
        self.min_width = min_width
        self.scale_factor_augment = scale_factor_augment
        self.scale_factor_augment_range = scale_factor_augment_range
        self.no_scale_factor = no_scale_factor

        # High-res finetuning mode
        self.finetune_highres_mode = finetune_highres_mode
        self.load_instance = load_instance
        self.instance_scene_objects = None
        self.instance_min_scene_objects = instance_min_scene_objects
        self.instance_loss = instance_loss
        if load_instance and instance_min_scene_objects > 0:
            with open(instance_counts) as handle:
                self.instance_scene_objects = json.load(handle)['scenes']
        self.finetune_target_height = finetune_target_height
        self.finetune_target_width = finetune_target_width
        self.finetune_target_crop = finetune_target_crop
        if self.finetune_highres_mode:
            print(f"[{self.dataset_name.upper()}] High-res finetuning mode: target_height={finetune_target_height}, target_width={finetune_target_width}, crop={finetune_target_crop}")

        self._timing_count = 0
        self._timing_accum = {'file_read': 0, 'depth_convert': 0, 'augment': 0, 'total': 0}

        # Resolve resize_height for this dataset
        if resize_height is not None:
            if isinstance(resize_height, dict):
                self.resize_height = resize_height.get(self.dataset_name, None)
            else:
                self.resize_height = resize_height
        else:
            self.resize_height = None

        if self.resize_height is not None:
            print(f"[{self.dataset_name.upper()}] Will resize height to {self.resize_height} before crop")

        # Resolve resize_width for this dataset
        if resize_width is not None:
            if isinstance(resize_width, dict):
                self.resize_width = resize_width.get(self.dataset_name, None)
            else:
                self.resize_width = resize_width
        else:
            self.resize_width = None

        if self.resize_width is not None:
            print(f"[{self.dataset_name.upper()}] Will resize width to {self.resize_width} before crop")

        # NEW: Setup device for conversion
        self.use_gpu_conversion = use_gpu_conversion
        if use_gpu_conversion and not torch.cuda.is_available():
            print(f"[WARNING] GPU conversion requested but CUDA not available, falling back to CPU")
            self.use_gpu_conversion = False
        self.conversion_device = torch.device('cuda' if self.use_gpu_conversion else 'cpu')
        if self.use_gpu_conversion:
            print(f"[{self.dataset_name.upper()}] Using GPU for depth->pointcloud conversion")

        # NEW: Caches for ray arrays and backprojection grids
        self._ray_cache = {}  # Key: (width, height)
        self._backproj_cache = {}  # Key: (width, height, fx, fy, cx, cy) - rounded

        # Datasets with per-frame intrinsics should NOT cache backprojection grids
        # to avoid unbounded memory growth
        variable_intrinsics_datasets = {'hypersim', 'urbansyn', 'synscapes', 'irs', 'omniworldgame', 'mvssynth', 'dynamic_replica', 'eden', *H5_SHARD_DATASET_NAMES}
        self._use_backproj_cache = self.dataset_name not in variable_intrinsics_datasets
        # Also disable camera intrinsics file caching for variable-intrinsics datasets
        self._use_cam_cache = self.dataset_name not in variable_intrinsics_datasets
        if not self._use_backproj_cache:
            print(f"[{self.dataset_name.upper()}] Backprojection and camera caching DISABLED (variable intrinsics)")

        # Precompute ray array for SceneNet (constant across all images)
        if self.dataset_name == 'scenenet':
            # Pre-cache for default size
            self._ray_cache[(320, 240)] = scenenet_ray_directions(width=320, height=240, device=self.conversion_device)

            # Pre-cache for resized size if applicable
            if self.resize_height is not None:
                # Compute target width maintaining aspect ratio (320/240 = 4/3)
                scale = self.resize_height / 240
                target_width = int(320 * scale)
                self._ray_cache[(target_width, self.resize_height)] = scenenet_ray_directions(
                    width=target_width, height=self.resize_height, device=self.conversion_device
                )
                print(f"[SceneNet] Pre-cached ray array for size ({target_width}, {self.resize_height})")

        # Precompute backprojection grid for TartanAir (fixed intrinsics)
        if self.dataset_name == 'tartanair':
            fx, fy, cx, cy = 320.0, 320.0, 320.0, 240.0

            # Pre-cache for default size
            cache_key = (640, 480, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
            self._backproj_cache[cache_key] = compute_backprojection_grid(640, 480, fx, fy, cx, cy, device=self.conversion_device)

            # Pre-cache for resized size if applicable
            if self.resize_height is not None:
                scale = self.resize_height / 480
                new_w = int(640 * scale)
                new_h = self.resize_height
                fx_scaled = fx * scale
                fy_scaled = fy * scale
                cx_scaled = cx * scale
                cy_scaled = cy * scale

                cache_key = (new_w, new_h, round(fx_scaled, 2), round(fy_scaled, 2), round(cx_scaled, 2), round(cy_scaled, 2))
                self._backproj_cache[cache_key] = compute_backprojection_grid(
                    new_w, new_h, fx_scaled, fy_scaled, cx_scaled, cy_scaled, device=self.conversion_device
                )
                print(f"[TartanAir] Pre-cached backproj grid for size ({new_w}, {new_h})")

        # Precompute backprojection grid for TartanAir V2 (fixed intrinsics, 640x640)
        if self.dataset_name == 'tartanairv2':
            # TartanAir V2 intrinsics: fx=320, fy=320, cx=320, cy=320 for 640x640 images
            fx, fy, cx, cy = 320.0, 320.0, 320.0, 320.0

            # Pre-cache for default size (640x640)
            cache_key = (640, 640, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
            self._backproj_cache[cache_key] = compute_backprojection_grid(640, 640, fx, fy, cx, cy, device=self.conversion_device)

            # Pre-cache for resized size if applicable
            if self.resize_height is not None:
                scale = self.resize_height / 640
                new_w = int(640 * scale)
                new_h = self.resize_height
                fx_scaled = fx * scale
                fy_scaled = fy * scale
                cx_scaled = cx * scale
                cy_scaled = cy * scale

                cache_key = (new_w, new_h, round(fx_scaled, 2), round(fy_scaled, 2), round(cx_scaled, 2), round(cy_scaled, 2))
                self._backproj_cache[cache_key] = compute_backprojection_grid(
                    new_w, new_h, fx_scaled, fy_scaled, cx_scaled, cy_scaled, device=self.conversion_device
                )
                print(f"[TartanAirV2] Pre-cached backproj grid for size ({new_w}, {new_h})")

        # Cache for VKitti2 intrinsics (loaded per-sequence) and grids
        if self.dataset_name == 'vkitti2':
            self.vkitti2_intrinsics_cache = {}

        # Cache for Hypersim camera files (per-frame .npz files)
        if self.dataset_name == 'hypersim':
            self.hypersim_cam_cache = {}

        # Cache for UrbanSyn camera files (per-frame .npz files, similar to Hypersim)
        if self.dataset_name == 'urbansyn':
            self.urbansyn_cam_cache = {}

        # Cache for MVSSynth camera files (per-frame .npz files, similar to Hypersim)
        if self.dataset_name == 'mvssynth':
            self.mvssynth_cam_cache = {}

        # Cache for Synscapes camera files (per-frame .npz files, similar to Hypersim)
        if self.dataset_name == 'synscapes':
            self.synscapes_cam_cache = {}

        # Cache for IRS camera files (per-frame .npz files, similar to Hypersim)
        if self.dataset_name == 'irs':
            self.irs_cam_cache = {}

        # Cache for Dynamic Replica camera files (per-frame .npz files, fixed intrinsics so caching is efficient)
        if self.dataset_name == 'dynamic_replica':
            self.dynamic_replica_cam_cache = {}

        # 1. Discover and Split Scenes
        if self.dataset_name == 'scenenet':
            self.all_data_paths = self._find_and_split_data_scenenet()
        elif self.dataset_name == 'tartanair':
            self.all_data_paths = self._find_and_split_data_tartanair()
        elif self.dataset_name == 'tartanairv2':
            self.all_data_paths = self._find_and_split_data_tartanairv2()
        elif self.dataset_name == 'vkitti2':
            self.all_data_paths = self._find_and_split_data_vkitti2()
        elif self.dataset_name == 'hypersim':
            self.all_data_paths = self._find_and_split_data_hypersim()
        elif self.dataset_name == 'urbansyn':
            self.all_data_paths = self._find_and_split_data_urbansyn()
        elif self.dataset_name == 'mvssynth':
            self.all_data_paths = self._find_and_split_data_mvssynth()
        elif self.dataset_name == 'synscapes':
            self.all_data_paths = self._find_and_split_data_synscapes()
        elif self.dataset_name == 'omniworldgame':
            self.all_data_paths = self._find_and_split_data_omniworldgame()
        elif self.dataset_name == 'irs':
            self.all_data_paths = self._find_and_split_data_irs()
        elif self.dataset_name == 'dynamic_replica':
            self.all_data_paths = self._find_and_split_data_dynamic_replica()
        elif self.dataset_name == 'eden':
            self.all_data_paths = self._find_and_split_data_eden()
        elif self.dataset_name in H5_SHARD_DATASET_NAMES:
            self.all_data_paths = find_h5_shards(self.root_dir, self.dataset_name)
        else:
            raise ValueError(f"Unknown dataset_name: {dataset_name}. Supported: 'scenenet', 'tartanair', 'tartanairv2', 'vkitti2', 'hypersim', 'urbansyn', 'mvssynth', 'synscapes', 'omniworldgame', 'irs', 'dynamic_replica', 'eden'")

        if num_overfit_samples is not None:
            self.all_data_paths = self.all_data_paths[:num_overfit_samples]

        if num_dataset_duplicates is not None:
            self.all_data_paths = self.all_data_paths * num_dataset_duplicates

        # Compute cumulative indices for fast O(log n) lookup in __getitem__
        self._compute_cumulative_indices()

        # 2. Setup the crop transformation
        # Determine effective crop size (finetune mode overrides crop_size)
        effective_crop_size = self.finetune_target_crop if self.finetune_highres_mode else self.crop_size

        if effective_crop_size is not None:
            if self.finetune_highres_mode:
                # High-res finetuning mode: use crop classes that return offsets for intrinsics adjustment
                if self.split in ['test', 'val'] or num_overfit_samples is not None:
                    print(f"[{self.split.upper()}] High-res mode: Using CenterCropPairedWithOffset of size {effective_crop_size}.")
                    self.cropper = CenterCropPairedWithOffset(size=effective_crop_size)
                else:
                    print(f"[{self.split.upper()}] High-res mode: Using RandomCropPairedWithOffset of size {effective_crop_size}.")
                    self.cropper = RandomCropPairedWithOffset(size=effective_crop_size)
            else:
                # Standard mode: use original crop classes
                if self.split in ['test', 'val'] or num_overfit_samples is not None:
                    print(f"[{self.split.upper()}] Using fixed CenterCropPaired of size {effective_crop_size}.")
                    self.cropper = CenterCropPaired(size=effective_crop_size)
                else:
                    print(f"[{self.split.upper()}] Using random RandomCropPaired of size {effective_crop_size}.")
                    self.cropper = RandomCropPaired(size=effective_crop_size)
        else:
            self.cropper = None

        # Standard Image Transforms
        self.image_to_tensor = TF.to_tensor

        if self.split in ['train', 'all']:
            # Color jitting
            self.color_jitter = torchvision.transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
            )

            if self.more_img_aug:
                aug_prob = 0.2 if self.stronger_img_aug else 0.1
                self.appearance_aug = v2.Compose([
                    v2.RandomApply([v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=aug_prob),
                    v2.RandomAutocontrast(p=aug_prob),
                    v2.RandomEqualize(p=aug_prob),
                    v2.RandomApply([v2.RandomChannelPermutation()], p=aug_prob),
                    v2.RandomApply([v2.JPEG(quality=(40, 100))], p=aug_prob),
                    v2.RandomGrayscale(p=aug_prob),
                ])

    def _compute_highres_resize_params(self, orig_h, orig_w):
        """
        Compute resize parameters for high-res finetuning mode.

        Logic (when target_width is None - height-only mode):
        - If native height < target: always resize to target (upscale)
        - If native height > target: 50/50 random choice between resize or not
        - If native height == target: no resize needed

        Logic (when target_width is specified - rectangular mode):
        - If both dimensions >= target: 50/50 random choice to resize or direct crop
        - If either dimension is smaller than target: must resize
        - Scale is computed to ensure both dimensions >= target after resize

        Returns:
            (do_resize, scale, new_h, new_w) or (False, None, None, None) if no resize needed
        """
        if not self.finetune_highres_mode:
            return False, None, None, None

        target_h = self.finetune_target_height
        target_w = self.finetune_target_width

        # Rectangular mode: check both dimensions
        if target_w is not None:
            height_ok = orig_h >= target_h
            width_ok = orig_w >= target_w

            if height_ok and width_ok:
                # Image is large enough in both dimensions: 50/50 choice
                do_resize = random.random() < 0.5
            else:
                # Image too small in at least one dimension: must resize
                do_resize = True

            if do_resize:
                # Compute scale to ensure both dims >= target
                scale_h = target_h / orig_h
                scale_w = target_w / orig_w
                scale = max(scale_h, scale_w)
                new_h = round(orig_h * scale)
                new_w = round(orig_w * scale)
                return True, scale, new_h, new_w
            else:
                return False, None, None, None
        else:
            # Height-only mode (original behavior)
            if orig_h < target_h:
                # Low-res: always upscale
                do_resize = True
            elif orig_h > target_h:
                # High-res: 50/50 random choice
                do_resize = random.random() < 0.5
            else:
                # Already at target height
                do_resize = False

            if do_resize:
                scale = target_h / orig_h
                new_h = target_h
                new_w = round(orig_w * scale)
            else:
                scale = 1.0
                new_h, new_w = orig_h, orig_w

            # Ensure both dimensions are at least as large as crop size
            # (handles portrait images where width < crop after height-based resize)
            crop_size = self.finetune_target_crop
            if crop_size is not None:
                crop_h, crop_w = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size
                if new_h < crop_h or new_w < crop_w:
                    do_resize = True
                    scale = max(crop_h / orig_h, crop_w / orig_w)
                    new_h = round(orig_h * scale)
                    new_w = round(orig_w * scale)

            if do_resize:
                return True, scale, new_h, new_w
            else:
                return False, None, None, None

    def _get_cache_path(self):
        """Generate cache file path based on dataset parameters."""
        cache_dir = os.path.join(self.root_dir, '.cache')

        subsample_str = f"subsample{self.subsample_scenes}" if self.subsample_scenes is not None else "nosubsample"
        samples_str = f"samples{self.samples_per_scene}" if self.samples_per_scene is not None else "allsamples"

        # Only add minsize suffix if filtering is enabled (backwards compatible with existing caches)
        if self.min_height is not None or self.min_width is not None:
            minsize_str = f"_minh{self.min_height}_minw{self.min_width}"
        else:
            minsize_str = ""

        cache_filename = f"{self.dataset_name}_{self.split}_{subsample_str}_{samples_str}{minsize_str}.pkl"

        return os.path.join(cache_dir, cache_filename)

    def _get_cache_path_for_split(self, split):
        """Generate cache file path for a specific split (used when loading train+val for 'all')."""
        cache_dir = os.path.join(self.root_dir, '.cache')
        subsample_str = f"subsample{self.subsample_scenes}" if self.subsample_scenes is not None else "nosubsample"
        samples_str = f"samples{self.samples_per_scene}" if self.samples_per_scene is not None else "allsamples"

        # Only add minsize suffix if filtering is enabled (backwards compatible with existing caches)
        if self.min_height is not None or self.min_width is not None:
            minsize_str = f"_minh{self.min_height}_minw{self.min_width}"
        else:
            minsize_str = ""

        cache_filename = f"{self.dataset_name}_{split}_{subsample_str}_{samples_str}{minsize_str}.pkl"
        return os.path.join(cache_dir, cache_filename)

    def _load_cache_from_path(self, cache_path):
        """Load cache from a specific path and return just the data (no validation)."""
        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)
            return cache_data.get('all_data')
        except Exception as e:
            print(f"Failed to load cache from {cache_path}: {e}")
            return None

    def _load_cache(self):
        """Load cached data if it exists and is valid."""
        cache_path = self._get_cache_path()

        if not os.path.exists(cache_path):
            return None

        try:
            with open(cache_path, 'rb') as f:
                cache_data = pickle.load(f)

            # Debug: Check each parameter
            # Normalize paths for comparison (handles symlinks and relative vs absolute paths)
            cached_root = os.path.abspath(os.path.realpath(cache_data.get('root_dir', '')))
            current_root = os.path.abspath(os.path.realpath(self.root_dir))
            root_match = cached_root == current_root

            split_match = cache_data.get('split') == self.split
            dataset_match = cache_data.get('dataset_name') == self.dataset_name
            subsample_match = cache_data.get('subsample_scenes') == self.subsample_scenes
            samples_match = cache_data.get('samples_per_scene') == self.samples_per_scene

            if (root_match and split_match and dataset_match and subsample_match and samples_match):
                print(f"[{self.split.upper()}] Loading data from cache: {cache_path}")
                return cache_data.get('all_data')
            else:
                print(f"[{self.split.upper()}] Cache invalidated due to parameter mismatch:")
                if not root_match:
                    print(f"  ✗ root_dir: cached='{cache_data.get('root_dir')}' != current='{self.root_dir}'")
                if not split_match:
                    print(f"  ✗ split: cached='{cache_data.get('split')}' != current='{self.split}'")
                if not dataset_match:
                    print(f"  ✗ dataset_name: cached='{cache_data.get('dataset_name')}' != current='{self.dataset_name}'")
                if not subsample_match:
                    print(f"  ✗ subsample_scenes: cached={cache_data.get('subsample_scenes')} != current={self.subsample_scenes}")
                if not samples_match:
                    print(f"  ✗ samples_per_scene: cached={cache_data.get('samples_per_scene')} != current={self.samples_per_scene}")
                return None

        except Exception as e:
            print(f"[{self.split.upper()}] Failed to load cache: {e}")
            return None

    def _save_cache(self, all_data):
        """Save all_data to cache file."""
        cache_path = self._get_cache_path()
        cache_dir = os.path.dirname(cache_path)

        os.makedirs(cache_dir, exist_ok=True)

        try:
            cache_data = {
                'root_dir': self.root_dir,
                'split': self.split,
                'dataset_name': self.dataset_name,
                'subsample_scenes': self.subsample_scenes,
                'samples_per_scene': self.samples_per_scene,
                'all_data': all_data
            }

            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"[{self.split.upper()}] Saved data to cache: {cache_path}")

        except Exception as e:
            print(f"[{self.split.upper()}] Failed to save cache: {e}")

    def _find_and_split_data_scenenet(self):
        """
        Find and split SceneNet-RGBD data.

        Structure: root/train/{scene}/{trajectory}/photo/{frame}.jpg
                                                   /depth/{frame}.png
        """
        # Special handling for split='all': try to load train+val caches
        if self.split == 'all':
            # Try to load both train and val caches
            train_cache_path = self._get_cache_path_for_split('train')
            val_cache_path = self._get_cache_path_for_split('val')

            train_cached = self._load_cache_from_path(train_cache_path)
            val_cached = self._load_cache_from_path(val_cache_path)

            if train_cached is not None and val_cached is not None:
                print(f"[ALL] Loaded {len(train_cached)} train samples from cache")
                print(f"[ALL] Loaded {len(val_cached)} val samples from cache")
                all_data = train_cached + val_cached
                print(f"[SceneNet-ALL] Total: {len(all_data)} samples (from cached train+val)")
                return all_data
            else:
                print(f"[ALL] Could not load both train and val caches, will rescan files...")

        # Try to load from cache first
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        # Get split-specific root (train or val)
        if self.split == 'train':
            split_dirs = ['train']
        elif self.split == 'test':
            split_dirs = ['val']
        elif self.split == 'all':
            split_dirs = ['train', 'val']
        else:
            split_dirs = [self.split]

        for split_dir in split_dirs:
            split_path = os.path.join(self.root_dir, split_dir)
            if not os.path.exists(split_path):
                continue

            # Get all scene directories
            scene_dirs = sorted([d for d in glob.glob(os.path.join(split_path, '*')) if os.path.isdir(d)])

            if self.subsample_scenes is not None:
                scene_dirs = scene_dirs[::self.subsample_scenes]

            print(f"[SceneNet-{self.split.upper()}] Found {len(scene_dirs)} scenes in {split_dir}/")

            for scene_dir in scene_dirs:
                scene_name = os.path.basename(scene_dir)

                # Get all trajectory directories
                traj_dirs = sorted([d for d in glob.glob(os.path.join(scene_dir, '*')) if os.path.isdir(d)])

                for traj_dir in traj_dirs:
                    traj_name = os.path.basename(traj_dir)

                    photo_dir = os.path.join(traj_dir, 'photo')
                    depth_dir = os.path.join(traj_dir, 'depth')

                    if not os.path.isdir(photo_dir) or not os.path.isdir(depth_dir):
                        continue

                    # Extract frame IDs from RGB filenames (lazy loading)
                    rgb_filenames = [f for f in os.listdir(photo_dir) if f.endswith('.jpg')]
                    if not rgb_filenames:
                        continue

                    frame_ids = sorted([os.path.splitext(f)[0] for f in rgb_filenames])

                    # Verify at least one depth file exists (spot check)
                    if not os.path.exists(os.path.join(depth_dir, f"{frame_ids[0]}.png")):
                        continue

                    # Apply subsampling if requested
                    num_available = len(frame_ids)
                    if self.samples_per_scene is not None and num_available > self.samples_per_scene:
                        indices = np.linspace(0, num_available - 1, self.samples_per_scene, dtype=int)
                        frame_ids = [frame_ids[i] for i in indices]

                    # Store trajectory metadata (lazy loading)
                    all_data.append({
                        'traj_dir': traj_dir,
                        'scene_name': scene_name,
                        'traj_name': traj_name,
                        'frame_ids': frame_ids,
                        'intrinsics': 'scenenet_fov',
                        'dataset': 'scenenet',
                        'lazy_load': True
                    })

        # Count total frames across all trajectories
        total_frames = sum(len(entry['frame_ids']) for entry in all_data)
        print(f"[SceneNet-{self.split.upper()}] Found {len(all_data)} trajectories with {total_frames} valid paired samples.")

        # Save to cache
        self._save_cache(all_data)

        return all_data

    def _find_and_split_data_tartanair(self):
        """
        Find and split TartanAir data.

        Structure: root/{scene}/{difficulty}/{sequence}/image_left/{frame}_left.png
                                                       /depth_left/{frame}_left_depth.npy
        """
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory {self.root_dir} does not exist.")
            return []

        # Discover all sequences
        all_sequences = []

        scene_dirs = sorted([d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)])

        for scene_dir in scene_dirs:
            scene_name = os.path.basename(scene_dir)

            diff_dirs = sorted([d for d in glob.glob(os.path.join(scene_dir, '*')) if os.path.isdir(d)])

            for diff_dir in diff_dirs:
                diff_name = os.path.basename(diff_dir)

                seq_dirs = sorted([d for d in glob.glob(os.path.join(diff_dir, '*')) if os.path.isdir(d)])

                for seq_dir in seq_dirs:
                    seq_name = os.path.basename(seq_dir)
                    rel_path = f"{scene_name}/{diff_name}/{seq_name}"
                    all_sequences.append((seq_dir, rel_path))

        total_sequences = len(all_sequences)

        if self.subsample_scenes is not None:
            all_sequences = all_sequences[::self.subsample_scenes]

        if total_sequences == 0:
            return []

        # Load test split if provided
        manual_test_seqs = set()
        use_manual_split = False

        if self.test_scenes_file:
            resolved_path = self.test_scenes_file
            if not os.path.exists(resolved_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(script_dir, self.test_scenes_file)

            if os.path.exists(resolved_path):
                print(f"Loading TartanAir test split from: {resolved_path}")
                with open(resolved_path, 'r') as f:
                    manual_test_seqs = {line.strip().replace('\\', '/') for line in f if line.strip()}
                use_manual_split = True
            else:
                print(f"Warning: File {self.test_scenes_file} not found.")

        # Filter sequences
        target_sequences = []

        auto_test_indices = set()
        if not use_manual_split and self.num_test_scenes is not None and self.num_test_scenes > 0:
            rng = np.linspace(0, total_sequences - 1, self.num_test_scenes, dtype=int)
            auto_test_indices = set(rng)

        for i, (seq_dir, rel_path) in enumerate(all_sequences):
            is_test = False

            if use_manual_split:
                if rel_path in manual_test_seqs:
                    is_test = True
            else:
                if i in auto_test_indices:
                    is_test = True

            # Apply split filter
            if self.split == 'train' and not is_test:
                target_sequences.append(seq_dir)
            elif self.split == 'test' and is_test:
                target_sequences.append(seq_dir)
            elif self.split == 'all':
                target_sequences.append(seq_dir)

        print(f"[TartanAir-{self.split.upper()}] Selected {len(target_sequences)} sequences out of {total_sequences}.")

        # MEMORY OPTIMIZATION: Gather data using lazy loading (store only sequence metadata + frame IDs)
        total_frames = 0
        for seq_dir in target_sequences:
            rgb_dir = os.path.join(seq_dir, 'image_left')
            depth_dir = os.path.join(seq_dir, 'depth_left')

            if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
                continue

            # Extract frame IDs from filenames without storing full paths
            rgb_filenames = [f for f in os.listdir(rgb_dir) if f.endswith('_left.png')]
            frame_ids = sorted([f.replace('.png', '') for f in rgb_filenames])

            # Validate depth files exist
            valid_frame_ids = [fid for fid in frame_ids if os.path.exists(os.path.join(depth_dir, f'{fid}_depth.npy'))]

            del rgb_filenames  # Free memory

            # Apply samples_per_scene subsampling
            if self.samples_per_scene is not None and len(valid_frame_ids) > self.samples_per_scene:
                indices = np.linspace(0, len(valid_frame_ids) - 1, self.samples_per_scene, dtype=int)
                valid_frame_ids = [valid_frame_ids[i] for i in indices]

            if valid_frame_ids:
                all_data.append({
                    'seq_dir': seq_dir,
                    'frame_ids': valid_frame_ids,
                    'intrinsics': 'tartanair_fixed',
                    'dataset': 'tartanair',
                    'lazy_load': True
                })
                total_frames += len(valid_frame_ids)

        print(f"[TartanAir-{self.split.upper()}] Found {len(all_data)} sequences with {total_frames} total frames (lazy loading).")

        self._save_cache(all_data)

        return all_data

    def _find_and_split_data_tartanairv2(self):
        """
        Find and split TartanAir V2 data with OPTIMIZED LAZY FRAME ENUMERATION.

        Memory-optimized version that stores only trajectory metadata instead of all file paths.
        This reduces memory from ~2GB to ~10MB for the full 4.5M sample dataset.

        Structure: root/{environment}/Data_{difficulty}/{trajectory}/image_{cam}/{frame:06d}_{cam}.png
                                                                    /depth_{cam}/{frame:06d}_{cam}_depth.png

        Cameras: lcam_front, lcam_back, lcam_left, lcam_right, lcam_top, lcam_bottom,
                 rcam_front, rcam_back, rcam_left, rcam_right, rcam_top, rcam_bottom
        """
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory {self.root_dir} does not exist.")
            return []

        print(f"[TartanAirV2-{self.split.upper()}] Using OPTIMIZED lazy frame enumeration (low memory mode)")

        # Discover all trajectories across all environments
        all_trajectories = []

        # List of all 12 camera views
        camera_views = [
            'lcam_front', 'lcam_back', 'lcam_left', 'lcam_right', 'lcam_top', 'lcam_bottom',
            'rcam_front', 'rcam_back', 'rcam_left', 'rcam_right', 'rcam_top', 'rcam_bottom'
        ]

        env_dirs = sorted([d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)])

        for env_dir in env_dirs:
            env_name = os.path.basename(env_dir)

            # Find all difficulty levels (Data_easy, Data_hard, etc.)
            diff_dirs = sorted([d for d in glob.glob(os.path.join(env_dir, 'Data_*')) if os.path.isdir(d)])

            for diff_dir in diff_dirs:
                diff_name = os.path.basename(diff_dir)

                # Find all trajectory folders (P000, P001, etc.)
                traj_dirs = sorted([d for d in glob.glob(os.path.join(diff_dir, 'P*')) if os.path.isdir(d)])

                for traj_dir in traj_dirs:
                    traj_name = os.path.basename(traj_dir)
                    rel_path = f"{env_name}/{diff_name}/{traj_name}"

                    # Each trajectory contains 12 camera views
                    for cam in camera_views:
                        cam_path = f"{rel_path}/{cam}"
                        all_trajectories.append((traj_dir, cam, cam_path))

        total_trajectories = len(all_trajectories)

        if self.subsample_scenes is not None:
            all_trajectories = all_trajectories[::self.subsample_scenes]

        if total_trajectories == 0:
            return []

        # Load test split if provided
        manual_test_trajs = set()
        use_manual_split = False

        if self.test_scenes_file:
            resolved_path = self.test_scenes_file
            if not os.path.exists(resolved_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(script_dir, self.test_scenes_file)

            if os.path.exists(resolved_path):
                print(f"Loading TartanAir V2 test split from: {resolved_path}")
                with open(resolved_path, 'r') as f:
                    manual_test_trajs = {line.strip().replace('\\', '/') for line in f if line.strip()}
                use_manual_split = True
            else:
                print(f"Warning: File {self.test_scenes_file} not found.")

        # Filter trajectories
        target_trajectories = []

        auto_test_indices = set()
        if not use_manual_split and self.num_test_scenes is not None and self.num_test_scenes > 0:
            rng = np.linspace(0, total_trajectories - 1, self.num_test_scenes, dtype=int)
            auto_test_indices = set(rng)

        for i, (traj_dir, cam, cam_path) in enumerate(all_trajectories):
            is_test = False

            if use_manual_split:
                if cam_path in manual_test_trajs:
                    is_test = True
            else:
                if i in auto_test_indices:
                    is_test = True

            # Apply split filter
            if self.split == 'train' and not is_test:
                target_trajectories.append((traj_dir, cam))
            elif self.split == 'test' and is_test:
                target_trajectories.append((traj_dir, cam))
            elif self.split == 'all':
                target_trajectories.append((traj_dir, cam))

        print(f"[TartanAirV2-{self.split.upper()}] Selected {len(target_trajectories)} camera trajectories out of {total_trajectories}.")

        # OPTIMIZATION: Store only trajectory metadata + frame IDs (not full paths)
        # This reduces memory from ~2GB to ~10MB for full dataset
        total_frames = 0
        for traj_dir, cam in target_trajectories:
            rgb_dir = os.path.join(traj_dir, f'image_{cam}')
            depth_dir = os.path.join(traj_dir, f'depth_{cam}')

            if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
                continue

            # MEMORY OPTIMIZATION: Find frame IDs without storing full paths
            # Use os.listdir() instead of glob to reduce memory
            try:
                rgb_filenames = [f for f in os.listdir(rgb_dir) if f.endswith(f'_{cam}.png')]
                frame_ids = sorted([f.split('_')[0] for f in rgb_filenames])

                # Verify at least one depth file exists
                if not frame_ids:
                    continue

                # Free memory immediately
                del rgb_filenames

                # Apply samples_per_scene subsampling if requested
                if self.samples_per_scene is not None and len(frame_ids) > self.samples_per_scene:
                    indices = np.linspace(0, len(frame_ids) - 1, self.samples_per_scene, dtype=int)
                    frame_ids = [frame_ids[i] for i in indices]

                # Store compact metadata: trajectory directory, camera, and frame ID list
                # This entry will be expanded to full paths in __getitem__()
                all_data.append({
                    'traj_dir': traj_dir,
                    'camera': cam,
                    'frame_ids': frame_ids,  # List of frame IDs (e.g., ['000000', '000001', ...])
                    'intrinsics': 'tartanairv2_fixed',
                    'dataset': 'tartanairv2',
                    'lazy_load': True  # Flag to indicate this needs path construction
                })

                total_frames += len(frame_ids)

            except Exception as e:
                print(f"Warning: Failed to process {traj_dir}/{cam}: {e}")
                continue

        print(f"[TartanAirV2-{self.split.upper()}] Found {len(all_data)} trajectories with {total_frames:,} total frames.")
        print(f"[TartanAirV2-{self.split.upper()}] Memory footprint: ~{len(all_data) * 0.01:.1f} MB (vs ~{total_frames * 0.0002:.0f} MB if storing all paths)")

        self._save_cache(all_data)

        return all_data

    def _find_and_split_data_vkitti2(self):
        """
        Find and split VKitti2 data.

        Structure: root/{scene}/{variation}/frames/rgb/Camera_{id}/rgb_{frame}.jpg
                                                   /depth/Camera_{id}/depth_{frame}.png
                                          /intrinsic.txt
        """
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory {self.root_dir} does not exist.")
            return []

        # Discover all camera sequences
        all_sequences = []

        scene_dirs = sorted([d for d in glob.glob(os.path.join(self.root_dir, 'Scene*')) if os.path.isdir(d)])

        for scene_dir in scene_dirs:
            scene_name = os.path.basename(scene_dir)

            var_dirs = sorted([d for d in glob.glob(os.path.join(scene_dir, '*')) if os.path.isdir(d)])

            for var_dir in var_dirs:
                var_name = os.path.basename(var_dir)

                # Check for camera directories
                rgb_base = os.path.join(var_dir, 'frames', 'rgb')
                if not os.path.exists(rgb_base):
                    continue

                cam_dirs = sorted([d for d in glob.glob(os.path.join(rgb_base, 'Camera_*')) if os.path.isdir(d)])

                for cam_dir in cam_dirs:
                    cam_name = os.path.basename(cam_dir)
                    rel_path = f"{scene_name}/{var_name}/{cam_name}"

                    # Store intrinsic file path for this sequence
                    intrinsic_file = os.path.join(var_dir, 'intrinsic.txt')

                    all_sequences.append((cam_dir, rel_path, var_dir, intrinsic_file))

        total_sequences = len(all_sequences)

        if self.subsample_scenes is not None:
            all_sequences = all_sequences[::self.subsample_scenes]

        if total_sequences == 0:
            return []

        # Load test split if provided
        manual_test_seqs = set()
        use_manual_split = False

        if self.test_scenes_file:
            resolved_path = self.test_scenes_file
            if not os.path.exists(resolved_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(script_dir, self.test_scenes_file)

            if os.path.exists(resolved_path):
                print(f"Loading VKitti2 test split from: {resolved_path}")
                with open(resolved_path, 'r') as f:
                    manual_test_seqs = {line.strip().replace('\\', '/') for line in f if line.strip()}
                use_manual_split = True
            else:
                print(f"Warning: File {self.test_scenes_file} not found.")

        # Filter sequences
        target_sequences = []

        auto_test_indices = set()
        if not use_manual_split and self.num_test_scenes is not None and self.num_test_scenes > 0:
            rng = np.linspace(0, total_sequences - 1, self.num_test_scenes, dtype=int)
            auto_test_indices = set(rng)

        for i, (cam_dir, rel_path, var_dir, intrinsic_file) in enumerate(all_sequences):
            is_test = False

            if use_manual_split:
                if rel_path in manual_test_seqs:
                    is_test = True
            else:
                if i in auto_test_indices:
                    is_test = True

            # Apply split filter
            if self.split == 'train' and not is_test:
                target_sequences.append((cam_dir, var_dir, intrinsic_file))
            elif self.split == 'test' and is_test:
                target_sequences.append((cam_dir, var_dir, intrinsic_file))
            elif self.split == 'all':
                target_sequences.append((cam_dir, var_dir, intrinsic_file))

        print(f"[VKitti2-{self.split.upper()}] Selected {len(target_sequences)} camera sequences out of {total_sequences}.")

        # Gather data from selected sequences using lazy loading
        total_frames = 0
        for cam_dir, var_dir, intrinsic_file in target_sequences:
            rgb_files = sorted(glob.glob(os.path.join(cam_dir, 'rgb_*.jpg')))

            # Get camera ID from directory name
            cam_name = os.path.basename(cam_dir)
            cam_id = int(cam_name.split('_')[-1])

            # Construct depth directory
            depth_dir = cam_dir.replace('/rgb/', '/depth/')

            if not os.path.isdir(depth_dir):
                continue

            # Extract frame IDs and validate
            valid_frame_ids = []
            for rgb_path in rgb_files:
                filename = os.path.basename(rgb_path)
                frame_str = filename.split('_')[1].split('.')[0]  # e.g., '00226'

                depth_filename = f"depth_{frame_str}.png"
                depth_path = os.path.join(depth_dir, depth_filename)

                if os.path.exists(depth_path):
                    valid_frame_ids.append(frame_str)

            if not valid_frame_ids:
                continue

            # Subsample frames if requested
            if self.samples_per_scene is not None and len(valid_frame_ids) > self.samples_per_scene:
                indices = np.linspace(0, len(valid_frame_ids) - 1, self.samples_per_scene, dtype=int)
                valid_frame_ids = [valid_frame_ids[i] for i in indices]

            total_frames += len(valid_frame_ids)

            # Store sequence metadata with lazy loading
            all_data.append({
                'rgb_dir': cam_dir,
                'depth_dir': depth_dir,
                'intrinsics': intrinsic_file,
                'camera_id': cam_id,
                'frame_ids': valid_frame_ids,
                'dataset': 'vkitti2',
                'lazy_load': True
            })

        print(f"[VKitti2-{self.split.upper()}] Found {len(all_data)} sequences with {total_frames} total frames (lazy loading).")

        self._save_cache(all_data)

        return all_data

    def _find_and_split_data_hypersim(self):
        """
        Find and split Hypersim data.

        Structure: root/{scene}/{subscene}/{frame:06d}_rgb.png
                                          /{frame:06d}_depth.npy
                                          /{frame:06d}_cam.npz
        Example: hypersim_rgbdepth/ai_013_002/cam_00/000000_rgb.png
        """
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory {self.root_dir} does not exist.")
            return []

        # Discover all subscene directories
        all_subscenes = []

        scene_dirs = sorted([d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)])

        for scene_dir in scene_dirs:
            scene_name = os.path.basename(scene_dir)

            subscene_dirs = sorted([d for d in glob.glob(os.path.join(scene_dir, '*')) if os.path.isdir(d)])

            for subscene_dir in subscene_dirs:
                subscene_name = os.path.basename(subscene_dir)
                rel_path = f"{scene_name}/{subscene_name}"
                all_subscenes.append((subscene_dir, rel_path, scene_name))

        total_subscenes = len(all_subscenes)

        if self.subsample_scenes is not None:
            all_subscenes = all_subscenes[::self.subsample_scenes]

        if total_subscenes == 0:
            return []

        # Load test split if provided
        manual_test_scenes = set()
        use_manual_split = False

        if self.test_scenes_file:
            resolved_path = self.test_scenes_file
            if not os.path.exists(resolved_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(script_dir, self.test_scenes_file)

            if os.path.exists(resolved_path):
                print(f"Loading Hypersim test split from: {resolved_path}")
                with open(resolved_path, 'r') as f:
                    manual_test_scenes = {line.strip() for line in f if line.strip()}
                use_manual_split = True
            else:
                print(f"Warning: File {self.test_scenes_file} not found.")

        # Filter subscenes based on split
        target_subscenes = []

        auto_test_indices = set()
        if not use_manual_split and self.num_test_scenes is not None and self.num_test_scenes > 0:
            rng = np.linspace(0, total_subscenes - 1, self.num_test_scenes, dtype=int)
            auto_test_indices = set(rng)

        for i, (subscene_dir, rel_path, scene_name) in enumerate(all_subscenes):
            is_test = False

            if use_manual_split:
                # Check if scene name is in test set
                if scene_name in manual_test_scenes:
                    is_test = True
            else:
                if i in auto_test_indices:
                    is_test = True

            # Apply split filter
            if self.split == 'train' and not is_test:
                target_subscenes.append(subscene_dir)
            elif self.split == 'test' and is_test:
                target_subscenes.append(subscene_dir)
            elif self.split == 'all':
                target_subscenes.append(subscene_dir)

        print(f"[Hypersim-{self.split.upper()}] Selected {len(target_subscenes)} subscenes out of {total_subscenes}.")

        # Gather data from selected subscenes using lazy loading
        total_frames = 0
        for subscene_dir in target_subscenes:
            rgb_files = sorted(glob.glob(os.path.join(subscene_dir, '*_rgb.png')))

            # Extract frame IDs and validate
            valid_frame_ids = []
            for rgb_path in rgb_files:
                filename = os.path.basename(rgb_path)
                # Extract frame ID from filename like "000000_rgb.png"
                frame_str = filename.split('_')[0]

                depth_path = os.path.join(subscene_dir, f"{frame_str}_depth.npy")
                cam_path = os.path.join(subscene_dir, f"{frame_str}_cam.npz")

                if os.path.exists(depth_path) and os.path.exists(cam_path):
                    valid_frame_ids.append(frame_str)

            if not valid_frame_ids:
                continue

            # Subsample frames if requested
            if self.samples_per_scene is not None and len(valid_frame_ids) > self.samples_per_scene:
                indices = np.linspace(0, len(valid_frame_ids) - 1, self.samples_per_scene, dtype=int)
                valid_frame_ids = [valid_frame_ids[i] for i in indices]

            total_frames += len(valid_frame_ids)

            # Store subscene metadata with lazy loading
            all_data.append({
                'subscene_dir': subscene_dir,
                'frame_ids': valid_frame_ids,
                'dataset': 'hypersim',
                'lazy_load': True
            })

        print(f"[Hypersim-{self.split.upper()}] Found {len(all_data)} subscenes with {total_frames} total frames (lazy loading).")

        self._save_cache(all_data)

        return all_data

    def _find_and_split_data_urbansyn(self):
        """
        Find and split UrbanSyn data.

        Structure: root/rgb/rgb_{frame:04d}.png
                       /depth/rgb_{frame:04d}.npy (preprocessed, already scaled)
                       /cam/rgb_{frame:04d}.npz (per-frame intrinsics)

        Note: Sky pixels are detected based on depth > max_depth threshold,
              no separate sky mask file is needed.
        """
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory {self.root_dir} does not exist.")
            return []

        rgb_dir = os.path.join(self.root_dir, 'rgb')
        depth_dir = os.path.join(self.root_dir, 'depth')
        cam_dir = os.path.join(self.root_dir, 'cam')

        if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir):
            print(f"Error: rgb/ or depth/ directory not found in {self.root_dir}")
            return []

        # Find all RGB files and extract valid frame IDs
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, 'rgb_*.png')))

        if len(rgb_files) == 0:
            print(f"Error: No RGB files found in {rgb_dir}")
            return []

        # Extract frame IDs and validate
        all_valid_frame_ids = []
        for rgb_path in rgb_files:
            filename = os.path.basename(rgb_path)
            # Extract basename like "rgb_0001"
            basename = filename.replace('.png', '')

            depth_path = os.path.join(depth_dir, f"{basename}.npy")
            cam_path = os.path.join(cam_dir, f"{basename}.npz")

            if os.path.exists(depth_path) and os.path.exists(cam_path):
                all_valid_frame_ids.append(basename)

        total_frames = len(all_valid_frame_ids)

        if total_frames == 0:
            print(f"Error: No matching RGB-depth-cam triplets found")
            return []

        # Apply subsampling if requested
        if self.subsample_scenes is not None:
            all_valid_frame_ids = all_valid_frame_ids[::self.subsample_scenes]
            print(f"[UrbanSyn] Subsampled from {total_frames} to {len(all_valid_frame_ids)} frames")

        # Split into train/test
        # Since this is a single "scene", we use frame-level splitting
        if self.num_test_scenes is not None and self.num_test_scenes > 0:
            # Use uniform sampling for test frames
            num_test_frames = min(self.num_test_scenes, len(all_valid_frame_ids))
            test_indices = set(np.linspace(0, len(all_valid_frame_ids) - 1, num_test_frames, dtype=int))
        else:
            test_indices = set()

        # Filter frame IDs based on split
        if self.split == 'train':
            selected_frame_ids = [all_valid_frame_ids[i] for i in range(len(all_valid_frame_ids)) if i not in test_indices]
        elif self.split == 'test':
            selected_frame_ids = [all_valid_frame_ids[i] for i in sorted(test_indices)]
        elif self.split == 'all':
            selected_frame_ids = all_valid_frame_ids
        else:
            selected_frame_ids = all_valid_frame_ids

        # Apply samples_per_scene subsampling (frame subsampling in this case)
        if self.samples_per_scene is not None and len(selected_frame_ids) > self.samples_per_scene:
            indices = np.linspace(0, len(selected_frame_ids) - 1, self.samples_per_scene, dtype=int)
            selected_frame_ids = [selected_frame_ids[i] for i in indices]

        # Store as single lazy loading entry
        all_data = [{
            'rgb_dir': rgb_dir,
            'depth_dir': depth_dir,
            'cam_dir': cam_dir,
            'frame_ids': selected_frame_ids,
            'dataset': 'urbansyn',
            'lazy_load': True
        }]

        print(f"[UrbanSyn-{self.split.upper()}] Found {len(selected_frame_ids)} frames (lazy loading, total {total_frames} frames).")

        self._save_cache(all_data)

        return all_data

    def _find_and_split_data_mvssynth(self):
        """
        Find and split MVSSynth data.

        Structure: root/{sequence}/rgb/{frame}.jpg
                                  /depth/{frame}.npy
                                  /cam/{frame}.npz
        Example: mvssynth/GTAV_720/0000/rgb/0000.jpg
        """
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory {self.root_dir} does not exist.")
            return []

        # Discover all sequence directories
        all_sequences = []

        seq_dirs = sorted([d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)])

        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            all_sequences.append((seq_dir, seq_name))

        total_sequences = len(all_sequences)

        if self.subsample_scenes is not None:
            all_sequences = all_sequences[::self.subsample_scenes]

        if total_sequences == 0:
            return []

        # Load test split if provided (sequence-level)
        manual_test_seqs = set()
        use_manual_split = False

        if self.test_scenes_file:
            resolved_path = self.test_scenes_file
            if not os.path.exists(resolved_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(script_dir, self.test_scenes_file)

            if os.path.exists(resolved_path):
                print(f"Loading MVSSynth test split from: {resolved_path}")
                with open(resolved_path, 'r') as f:
                    manual_test_seqs = {line.strip() for line in f if line.strip()}
                use_manual_split = True
            else:
                print(f"Warning: File {self.test_scenes_file} not found.")

        # Filter sequences based on split
        target_sequences = []

        auto_test_indices = set()
        if not use_manual_split and self.num_test_scenes is not None and self.num_test_scenes > 0:
            rng = np.linspace(0, total_sequences - 1, self.num_test_scenes, dtype=int)
            auto_test_indices = set(rng)

        for i, (seq_dir, seq_name) in enumerate(all_sequences):
            is_test = False

            if use_manual_split:
                # Check if sequence name is in test set
                if seq_name in manual_test_seqs:
                    is_test = True
            else:
                if i in auto_test_indices:
                    is_test = True

            # Apply split filter
            if self.split == 'train' and not is_test:
                target_sequences.append(seq_dir)
            elif self.split == 'test' and is_test:
                target_sequences.append(seq_dir)
            elif self.split == 'all':
                target_sequences.append(seq_dir)

        print(f"[MVSSynth-{self.split.upper()}] Selected {len(target_sequences)} sequences out of {total_sequences}.")

        # Gather data from selected sequences
        for seq_dir in target_sequences:
            rgb_dir = os.path.join(seq_dir, 'rgb')
            depth_dir = os.path.join(seq_dir, 'depth')
            cam_dir = os.path.join(seq_dir, 'cam')

            if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir) or not os.path.isdir(cam_dir):
                continue

            rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.jpg')))

            seq_valid_triplets = []

            for rgb_path in rgb_files:
                filename = os.path.basename(rgb_path)
                # Extract frame ID from filename like "0000.jpg"
                frame_str = os.path.splitext(filename)[0]

                depth_path = os.path.join(depth_dir, f"{frame_str}.npy")
                cam_path = os.path.join(cam_dir, f"{frame_str}.npz")

                if os.path.exists(depth_path) and os.path.exists(cam_path):
                    seq_valid_triplets.append({
                        'image': rgb_path,
                        'depth': depth_path,
                        'cam_file': cam_path,
                        'dataset': 'mvssynth'
                    })

            # Subsample frames if requested
            if self.samples_per_scene is not None and len(seq_valid_triplets) > self.samples_per_scene:
                indices = np.linspace(0, len(seq_valid_triplets) - 1, self.samples_per_scene, dtype=int)
                selected = [seq_valid_triplets[i] for i in indices]
                all_data.extend(selected)
            else:
                all_data.extend(seq_valid_triplets)

        print(f"[MVSSynth-{self.split.upper()}] Found {len(all_data)} valid paired samples.")

        self._save_cache(all_data)

        return all_data

    def _find_and_split_data_synscapes(self):
        """
        Find and split Synscapes data.

        Structure: root/rgb/{frame}.png
                       /depth/{frame}.npy
                       /cam/{frame}.npz

        Note: Synscapes is a single flat dataset (no sequences/scenes subdivision).
        We treat all frames as one "scene" for splitting purposes.
        """
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory {self.root_dir} does not exist.")
            return []

        rgb_dir = os.path.join(self.root_dir, 'rgb')
        depth_dir = os.path.join(self.root_dir, 'depth')
        cam_dir = os.path.join(self.root_dir, 'cam')

        if not os.path.isdir(rgb_dir) or not os.path.isdir(depth_dir) or not os.path.isdir(cam_dir):
            print(f"Error: Synscapes dataset missing required directories (rgb, depth, cam).")
            return []

        # Get all RGB files and extract valid frame IDs
        rgb_files = sorted(glob.glob(os.path.join(rgb_dir, '*.png')))

        if len(rgb_files) == 0:
            print(f"Error: No RGB images found in {rgb_dir}")
            return []

        # Extract frame IDs and validate
        all_valid_frame_ids = []
        for rgb_path in rgb_files:
            filename = os.path.basename(rgb_path)
            frame_str = os.path.splitext(filename)[0]

            depth_path = os.path.join(depth_dir, f"{frame_str}.npy")
            cam_path = os.path.join(cam_dir, f"{frame_str}.npz")

            if os.path.exists(depth_path) and os.path.exists(cam_path):
                all_valid_frame_ids.append(frame_str)

        total_frames = len(all_valid_frame_ids)

        if total_frames == 0:
            print(f"Error: No valid triplets found in {self.root_dir}")
            return []

        # Apply subsampling at the full dataset level if requested
        if self.subsample_scenes is not None:
            all_valid_frame_ids = all_valid_frame_ids[::self.subsample_scenes]

        # Split into train/test
        # For Synscapes (flat dataset), we use uniform sampling for test set
        test_indices = set()

        if self.test_scenes_file:
            # Manual split: load frame indices/names from file
            resolved_path = self.test_scenes_file
            if not os.path.exists(resolved_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(script_dir, self.test_scenes_file)

            if os.path.exists(resolved_path):
                print(f"Loading Synscapes test split from: {resolved_path}")
                with open(resolved_path, 'r') as f:
                    test_frames = {line.strip() for line in f if line.strip()}

                # Mark test indices based on frame IDs in file
                for i, frame_id in enumerate(all_valid_frame_ids):
                    if frame_id in test_frames:
                        test_indices.add(i)
            else:
                print(f"Warning: File {self.test_scenes_file} not found.")

        # Auto split if no manual split
        if len(test_indices) == 0 and self.num_test_scenes is not None and self.num_test_scenes > 0:
            # Interpret num_test_scenes as number of test frames for flat datasets
            rng = np.linspace(0, len(all_valid_frame_ids) - 1, self.num_test_scenes, dtype=int)
            test_indices = set(rng)

        # Filter frame IDs based on split
        if self.split == 'train':
            selected_frame_ids = [all_valid_frame_ids[i] for i in range(len(all_valid_frame_ids)) if i not in test_indices]
        elif self.split == 'test':
            selected_frame_ids = [all_valid_frame_ids[i] for i in sorted(test_indices)]
        elif self.split == 'all':
            selected_frame_ids = all_valid_frame_ids
        else:
            selected_frame_ids = all_valid_frame_ids

        # Apply per-scene sampling (though Synscapes is one "scene")
        if self.samples_per_scene is not None and len(selected_frame_ids) > self.samples_per_scene:
            indices = np.linspace(0, len(selected_frame_ids) - 1, self.samples_per_scene, dtype=int)
            selected_frame_ids = [selected_frame_ids[i] for i in indices]

        # Store as single lazy loading entry
        all_data = [{
            'rgb_dir': rgb_dir,
            'depth_dir': depth_dir,
            'cam_dir': cam_dir,
            'frame_ids': selected_frame_ids,
            'dataset': 'synscapes',
            'lazy_load': True
        }]

        print(f"[Synscapes-{self.split.upper()}] Found {len(selected_frame_ids)} frames (lazy loading, total {total_frames}).")

        self._save_cache(all_data)

        return all_data

    def _find_and_split_data_omniworldgame(self):
        """
        Find and split OmniWorld-Game data.

        Structure: root/videos/OmniWorld-Game/{scene}/color/{frame}.png
                       /annotations/OmniWorld-Game/{scene}/depth/{frame}.png
                       /annotations/OmniWorld-Game/{scene}/camera/split_{idx}.json
                       /annotations/OmniWorld-Game/{scene}/split_info.json

        Each scene contains multiple "splits" with separate camera calibrations.
        """
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory {self.root_dir} does not exist.")
            return []

        # Discover all scene directories
        videos_dir = os.path.join(self.root_dir, 'videos', 'OmniWorld-Game')
        annotations_dir = os.path.join(self.root_dir, 'annotations', 'OmniWorld-Game')

        if not os.path.isdir(videos_dir) or not os.path.isdir(annotations_dir):
            print(f"Error: OmniWorld-Game dataset missing required directories.")
            print(f"Expected: {videos_dir} and {annotations_dir}")
            return []

        scene_dirs = sorted([d for d in glob.glob(os.path.join(videos_dir, '*')) if os.path.isdir(d)])
        all_scenes = []

        for scene_dir in scene_dirs:
            scene_name = os.path.basename(scene_dir)
            annotation_scene_dir = os.path.join(annotations_dir, scene_name)

            # Check if annotation directory exists
            if not os.path.isdir(annotation_scene_dir):
                continue

            # Check if split_info.json exists
            split_info_path = os.path.join(annotation_scene_dir, 'split_info.json')
            if not os.path.exists(split_info_path):
                continue

            all_scenes.append((scene_dir, annotation_scene_dir, scene_name))

        total_scenes = len(all_scenes)

        if total_scenes == 0:
            print(f"Error: No valid OmniWorld-Game scenes found in {videos_dir}")
            return []

        # Apply scene subsampling if requested
        if self.subsample_scenes is not None:
            all_scenes = all_scenes[::self.subsample_scenes]

        # Load test split if provided (scene-level)
        manual_test_scenes = set()
        use_manual_split = False

        if self.test_scenes_file:
            resolved_path = self.test_scenes_file
            if not os.path.exists(resolved_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(script_dir, self.test_scenes_file)

            if os.path.exists(resolved_path):
                print(f"Loading OmniWorld-Game test split from: {resolved_path}")
                with open(resolved_path, 'r') as f:
                    manual_test_scenes = {line.strip() for line in f if line.strip()}
                use_manual_split = True
            else:
                print(f"Warning: File {self.test_scenes_file} not found.")

        # Filter scenes based on split
        target_scenes = []

        auto_test_indices = set()
        if not use_manual_split and self.num_test_scenes is not None and self.num_test_scenes > 0:
            rng = np.linspace(0, total_scenes - 1, self.num_test_scenes, dtype=int)
            auto_test_indices = set(rng)

        for i, (scene_dir, annotation_scene_dir, scene_name) in enumerate(all_scenes):
            is_test = False

            if use_manual_split:
                if scene_name in manual_test_scenes:
                    is_test = True
            else:
                if i in auto_test_indices:
                    is_test = True

            # Apply split filter
            if self.split == 'train' and not is_test:
                target_scenes.append((scene_dir, annotation_scene_dir, scene_name))
            elif self.split == 'test' and is_test:
                target_scenes.append((scene_dir, annotation_scene_dir, scene_name))
            elif self.split == 'all':
                target_scenes.append((scene_dir, annotation_scene_dir, scene_name))

        print(f"[OmniWorld-Game-{self.split.upper()}] Selected {len(target_scenes)} scenes out of {total_scenes}.")

        print(f"[OmniWorld-Game-{self.split.upper()}] Using OPTIMIZED lazy loading (low memory mode)")

        # OPTIMIZATION: Store scene metadata instead of individual frame paths
        # This reduces memory from ~530MB to ~27MB for full dataset
        import json
        total_frames = 0

        for scene_dir, annotation_scene_dir, scene_name in target_scenes:
            # Load split_info.json
            split_info_path = os.path.join(annotation_scene_dir, 'split_info.json')
            try:
                with open(split_info_path, 'r', encoding='utf-8') as f:
                    split_info = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load split_info for scene {scene_name}: {e}")
                continue

            color_dir = os.path.join(scene_dir, 'color')
            depth_dir = os.path.join(annotation_scene_dir, 'depth')
            camera_dir = os.path.join(annotation_scene_dir, 'camera')

            if not os.path.isdir(color_dir) or not os.path.isdir(depth_dir) or not os.path.isdir(camera_dir):
                continue

            # Verify camera files exist for all splits
            split_num = split_info['split_num']
            valid_splits = []
            for split_idx in range(split_num):
                cam_file = os.path.join(camera_dir, f'split_{split_idx}.json')
                if os.path.exists(cam_file):
                    valid_splits.append(split_idx)

            if not valid_splits:
                continue

            # Store compact scene metadata with frame indices from split_info.json
            # Paths will be constructed on-demand in __getitem__()
            all_data.append({
                'scene_name': scene_name,
                'color_dir': color_dir,
                'depth_dir': depth_dir,
                'camera_dir': camera_dir,
                'split_info': split_info,  # Contains 'split_num' and 'split' (list of frame indices per split)
                'valid_splits': valid_splits,  # Only splits with existing camera files
                'dataset': 'omniworldgame',
                'lazy_load': True  # Flag for lazy loading
            })

            # Count total frames for this scene
            scene_frames = sum(len(split_info['split'][i]) for i in valid_splits)
            total_frames += scene_frames

        print(f"[OmniWorld-Game-{self.split.upper()}] Found {len(all_data)} scenes with {total_frames:,} total frames.")
        print(f"[OmniWorld-Game-{self.split.upper()}] Memory footprint: ~{len(all_data) * 0.1:.1f} MB (vs ~{total_frames * 0.00025:.0f} MB if storing all paths)")

        self._save_cache(all_data)

        return all_data

    def _find_and_split_data_irs(self):
        """
        Find and split IRS data with OPTIMIZED LAZY FRAME ENUMERATION.

        Memory-optimized version that stores only sequence metadata instead of all file paths.
        This reduces memory usage significantly for datasets with many frames.

        Structure: root/{sequence}/rgb/{frame:05d}.png
                                  /depth/{frame:05d}.npy
                                  /cam/{frame:05d}.npz

        Example sequences: ArchVizInterior03Data, ModernClassicInterior, ConvenienceStore, etc.
        Depth is z-depth in meters (preprocessed from disparity EXR).
        Intrinsics are per-frame 3x3 matrices (fx=fy=480, cx=width/2, cy=height/2).
        """
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory {self.root_dir} does not exist.")
            return []

        print(f"[IRS-{self.split.upper()}] Using OPTIMIZED lazy frame enumeration (low memory mode)")

        # Discover all sequence directories
        all_sequences = []
        seq_dirs = sorted([d for d in glob.glob(os.path.join(self.root_dir, '*')) if os.path.isdir(d)])

        for seq_dir in seq_dirs:
            seq_name = os.path.basename(seq_dir)
            # Check if sequence has the expected structure
            rgb_dir = os.path.join(seq_dir, 'rgb')
            depth_dir = os.path.join(seq_dir, 'depth')
            cam_dir = os.path.join(seq_dir, 'cam')

            if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir) and os.path.isdir(cam_dir):
                all_sequences.append((seq_dir, seq_name))

        total_sequences = len(all_sequences)

        if self.subsample_scenes is not None:
            all_sequences = all_sequences[::self.subsample_scenes]

        if total_sequences == 0:
            print(f"Error: No valid IRS sequences found in {self.root_dir}")
            return []

        # Load test split if provided
        manual_test_seqs = set()
        use_manual_split = False

        if self.test_scenes_file:
            resolved_path = self.test_scenes_file
            if not os.path.exists(resolved_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(script_dir, self.test_scenes_file)

            if os.path.exists(resolved_path):
                print(f"Loading IRS test split from: {resolved_path}")
                with open(resolved_path, 'r') as f:
                    manual_test_seqs = {line.strip() for line in f if line.strip()}
                use_manual_split = True
            else:
                print(f"Warning: File {self.test_scenes_file} not found.")

        # Filter sequences based on split
        target_sequences = []

        auto_test_indices = set()
        if not use_manual_split and self.num_test_scenes is not None and self.num_test_scenes > 0:
            rng = np.linspace(0, total_sequences - 1, self.num_test_scenes, dtype=int)
            auto_test_indices = set(rng)

        for i, (seq_dir, seq_name) in enumerate(all_sequences):
            is_test = False

            if use_manual_split:
                if seq_name in manual_test_seqs:
                    is_test = True
            else:
                if i in auto_test_indices:
                    is_test = True

            # Apply split filter
            if self.split == 'train' and not is_test:
                target_sequences.append((seq_dir, seq_name))
            elif self.split == 'test' and is_test:
                target_sequences.append((seq_dir, seq_name))
            elif self.split == 'all':
                target_sequences.append((seq_dir, seq_name))

        print(f"[IRS-{self.split.upper()}] Selected {len(target_sequences)} sequences out of {total_sequences}.")

        # OPTIMIZATION: Store only sequence metadata + frame IDs (not full paths)
        # This reduces memory usage significantly for large datasets
        total_frames = 0
        for seq_dir, seq_name in target_sequences:
            rgb_dir = os.path.join(seq_dir, 'rgb')
            depth_dir = os.path.join(seq_dir, 'depth')
            cam_dir = os.path.join(seq_dir, 'cam')

            # MEMORY OPTIMIZATION: Find frame IDs without storing full paths
            # Use os.listdir() instead of glob to reduce memory
            try:
                rgb_filenames = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
                # Extract frame IDs from filenames (e.g., "00001.png" -> "00001")
                all_frame_ids = sorted([f.replace('.png', '') for f in rgb_filenames])

                # Verify at least one frame exists
                if not all_frame_ids:
                    continue

                # Free memory immediately
                del rgb_filenames

                # FILTER INCOMPLETE FILES: Only keep frames where RGB, depth, and cam all exist
                # This prevents FileNotFoundError when some files are missing
                frame_ids = []
                skipped_count = 0
                for frame_id in all_frame_ids:
                    depth_file = os.path.join(depth_dir, f'{frame_id}.npy')
                    cam_file = os.path.join(cam_dir, f'{frame_id}.npz')
                    if os.path.exists(depth_file) and os.path.exists(cam_file):
                        frame_ids.append(frame_id)
                    else:
                        skipped_count += 1

                if skipped_count > 0:
                    print(f"  Warning: Sequence {seq_name} has {skipped_count} incomplete frames (missing depth/cam files), skipping them.")

                # Verify at least one complete frame exists
                if not frame_ids:
                    continue

                # Apply samples_per_scene subsampling if requested
                if self.samples_per_scene is not None and len(frame_ids) > self.samples_per_scene:
                    indices = np.linspace(0, len(frame_ids) - 1, self.samples_per_scene, dtype=int)
                    frame_ids = [frame_ids[i] for i in indices]

                # Store compact metadata: sequence directory and frame ID list
                # This entry will be expanded to full paths in __getitem__()
                all_data.append({
                    'seq_dir': seq_dir,
                    'seq_name': seq_name,
                    'frame_ids': frame_ids,  # List of frame IDs (e.g., ['00001', '00002', ...])
                    'dataset': 'irs',
                    'lazy_load': True  # Flag to indicate this needs path construction
                })

                total_frames += len(frame_ids)

            except Exception as e:
                print(f"Warning: Failed to process {seq_dir}: {e}")
                continue

        print(f"[IRS-{self.split.upper()}] Found {len(all_data)} sequences with {total_frames:,} total frames.")
        print(f"[IRS-{self.split.upper()}] Memory footprint: ~{len(all_data) * 0.01:.1f} MB (vs ~{total_frames * 0.0002:.0f} MB if storing all paths)")

        self._save_cache(all_data)

        return all_data

    def _find_and_split_data_dynamic_replica(self):
        """
        Find and split Dynamic Replica data with OPTIMIZED LAZY FRAME ENUMERATION.

        Memory-optimized version that stores only sequence metadata instead of all file paths.
        This reduces memory usage significantly for datasets with many frames.

        Structure: root/{split}/{sequence}/left/rgb/{timestamp}.png
                                              /depth/{timestamp}.npy
                                              /cam/{timestamp}.npz

        Note: Only 'left' camera has data (right directories are empty).
        Timestamps are float strings (e.g., '0.0', '0.03333333333333333').
        Depth is z-depth in meters (float32).
        Intrinsics are fixed: fx=fy=700, cx=640, cy=360 at 1280x720.
        """
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory {self.root_dir} does not exist.")
            return []

        print(f"[DynamicReplica-{self.split.upper()}] Using OPTIMIZED lazy frame enumeration (low memory mode)")

        # Determine which splits to use
        if self.split == 'all':
            split_dirs = ['train', 'valid', 'test']
        elif self.split == 'train':
            split_dirs = ['train']
        elif self.split == 'val' or self.split == 'valid':
            split_dirs = ['valid']
        elif self.split == 'test':
            split_dirs = ['test']
        else:
            split_dirs = [self.split]

        # Discover all sequence directories across selected splits
        all_sequences = []
        for split_dir in split_dirs:
            split_path = os.path.join(self.root_dir, split_dir)
            if not os.path.exists(split_path):
                print(f"Warning: Split directory {split_path} does not exist, skipping.")
                continue

            seq_dirs = sorted([d for d in glob.glob(os.path.join(split_path, '*')) if os.path.isdir(d)])
            print(f"[DynamicReplica] Found {len(seq_dirs)} sequences in {split_dir}/")

            for seq_dir in seq_dirs:
                seq_name = os.path.basename(seq_dir)
                # Only use 'left' camera (right is empty in the preprocessed data)
                cam_path = os.path.join(seq_dir, 'left')
                rgb_dir = os.path.join(cam_path, 'rgb')
                depth_dir = os.path.join(cam_path, 'depth')
                cam_dir = os.path.join(cam_path, 'cam')

                if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir) and os.path.isdir(cam_dir):
                    all_sequences.append((seq_dir, seq_name, split_dir))

        total_sequences = len(all_sequences)

        if self.subsample_scenes is not None:
            all_sequences = all_sequences[::self.subsample_scenes]

        if total_sequences == 0:
            print(f"Error: No valid Dynamic Replica sequences found in {self.root_dir}")
            return []

        print(f"[DynamicReplica-{self.split.upper()}] Selected {len(all_sequences)} sequences out of {total_sequences}.")

        # OPTIMIZATION: Store only sequence metadata + frame IDs (not full paths)
        total_frames = 0
        for seq_dir, seq_name, orig_split in all_sequences:
            cam_path = os.path.join(seq_dir, 'left')
            rgb_dir = os.path.join(cam_path, 'rgb')
            depth_dir = os.path.join(cam_path, 'depth')
            cam_dir = os.path.join(cam_path, 'cam')

            # MEMORY OPTIMIZATION: Find frame IDs without storing full paths
            try:
                rgb_filenames = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
                # Extract timestamp IDs from filenames (e.g., "0.0.png" -> "0.0")
                # Sort by numeric value to ensure proper temporal ordering
                all_frame_ids = sorted([f.replace('.png', '') for f in rgb_filenames],
                                       key=lambda x: float(x))

                if not all_frame_ids:
                    continue

                del rgb_filenames

                # FILTER INCOMPLETE FILES: Only keep frames where RGB, depth, and cam all exist
                frame_ids = []
                skipped_count = 0
                for frame_id in all_frame_ids:
                    depth_file = os.path.join(depth_dir, f'{frame_id}.npy')
                    cam_file = os.path.join(cam_dir, f'{frame_id}.npz')
                    if os.path.exists(depth_file) and os.path.exists(cam_file):
                        frame_ids.append(frame_id)
                    else:
                        skipped_count += 1

                if skipped_count > 0:
                    print(f"  Warning: Sequence {seq_name} has {skipped_count} incomplete frames, skipping them.")

                if not frame_ids:
                    continue

                # Apply samples_per_scene subsampling if requested
                if self.samples_per_scene is not None and len(frame_ids) > self.samples_per_scene:
                    indices = np.linspace(0, len(frame_ids) - 1, self.samples_per_scene, dtype=int)
                    frame_ids = [frame_ids[i] for i in indices]

                # Store compact metadata
                all_data.append({
                    'seq_dir': seq_dir,
                    'seq_name': seq_name,
                    'frame_ids': frame_ids,
                    'dataset': 'dynamic_replica',
                    'lazy_load': True
                })

                total_frames += len(frame_ids)

            except Exception as e:
                print(f"Warning: Failed to process {seq_dir}: {e}")
                continue

        print(f"[DynamicReplica-{self.split.upper()}] Found {len(all_data)} sequences with {total_frames:,} total frames.")
        print(f"[DynamicReplica-{self.split.upper()}] Memory footprint: ~{len(all_data) * 0.01:.1f} MB (vs ~{total_frames * 0.0002:.0f} MB if storing all paths)")

        self._save_cache(all_data)

        return all_data

    def _find_and_split_data_eden(self):
        """
        Find and split Eden data with LAZY LOADING.

        Structure: root/{scene_id}_{lighting_mode}/rgb/{basename}.png
                                                  /depth/{basename}.npy
                                                  /cam/{basename}.npz

        Example: eden/0001_clear/rgb/A_0001.png
        Lighting modes: clear, cloudy, overcast, sunset, twilight

        Depth is z-depth in meters (float32).
        Intrinsics are per-frame 3x3 matrices stored in .npz files.
        """
        cached_data = self._load_cache()
        if cached_data is not None:
            return cached_data

        all_data = []

        if not os.path.exists(self.root_dir):
            print(f"Error: Root directory {self.root_dir} does not exist.")
            return []

        # Discover all sequence directories (e.g., 0001_clear, 0001_cloudy)
        all_sequences = []
        for seq_name in sorted(os.listdir(self.root_dir)):
            seq_dir = os.path.join(self.root_dir, seq_name)
            if os.path.isdir(seq_dir):
                # Check if sequence has the expected structure (rgb/, depth/, cam/)
                rgb_dir = os.path.join(seq_dir, 'rgb')
                depth_dir = os.path.join(seq_dir, 'depth')
                cam_dir = os.path.join(seq_dir, 'cam')

                if os.path.isdir(rgb_dir) and os.path.isdir(depth_dir) and os.path.isdir(cam_dir):
                    all_sequences.append((seq_dir, seq_name))

        total_sequences = len(all_sequences)

        if self.subsample_scenes is not None:
            all_sequences = all_sequences[::self.subsample_scenes]

        if total_sequences == 0:
            print(f"Error: No valid Eden sequences found in {self.root_dir}")
            return []

        # Load test split if provided
        manual_test_scenes = set()
        use_manual_split = False

        if self.test_scenes_file:
            resolved_path = self.test_scenes_file
            if not os.path.exists(resolved_path):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                resolved_path = os.path.join(script_dir, self.test_scenes_file)

            if os.path.exists(resolved_path):
                print(f"Loading Eden test split from: {resolved_path}")
                with open(resolved_path, 'r') as f:
                    manual_test_scenes = {line.strip() for line in f if line.strip()}
                use_manual_split = True
            else:
                print(f"Warning: File {self.test_scenes_file} not found.")

        # Filter sequences based on split
        target_sequences = []

        auto_test_indices = set()
        if not use_manual_split and self.num_test_scenes is not None and self.num_test_scenes > 0:
            rng = np.linspace(0, total_sequences - 1, self.num_test_scenes, dtype=int)
            auto_test_indices = set(rng)

        for i, (seq_dir, seq_name) in enumerate(all_sequences):
            is_test = False

            if use_manual_split:
                if seq_name in manual_test_scenes:
                    is_test = True
            else:
                if i in auto_test_indices:
                    is_test = True

            # Apply split filter
            if self.split == 'train' and not is_test:
                target_sequences.append((seq_dir, seq_name))
            elif self.split == 'test' and is_test:
                target_sequences.append((seq_dir, seq_name))
            elif self.split == 'all':
                target_sequences.append((seq_dir, seq_name))

        print(f"[Eden-{self.split.upper()}] Selected {len(target_sequences)} sequences out of {total_sequences}.")

        # Gather data from selected sequences using lazy loading
        total_frames = 0
        for seq_dir, seq_name in target_sequences:
            rgb_dir = os.path.join(seq_dir, 'rgb')
            depth_dir = os.path.join(seq_dir, 'depth')
            cam_dir = os.path.join(seq_dir, 'cam')

            # Find all RGB files and extract basenames
            rgb_filenames = [f for f in os.listdir(rgb_dir) if f.endswith('.png')]
            if not rgb_filenames:
                continue

            # Extract basenames (e.g., "A_0001.png" -> "A_0001")
            all_basenames = sorted([f.replace('.png', '') for f in rgb_filenames])

            # Validate that depth and cam files exist
            valid_frame_ids = []
            for basename in all_basenames:
                depth_path = os.path.join(depth_dir, f'{basename}.npy')
                cam_path = os.path.join(cam_dir, f'{basename}.npz')
                if os.path.exists(depth_path) and os.path.exists(cam_path):
                    valid_frame_ids.append(basename)

            if not valid_frame_ids:
                continue

            # Apply samples_per_scene subsampling if requested
            if self.samples_per_scene is not None and len(valid_frame_ids) > self.samples_per_scene:
                indices = np.linspace(0, len(valid_frame_ids) - 1, self.samples_per_scene, dtype=int)
                valid_frame_ids = [valid_frame_ids[i] for i in indices]

            total_frames += len(valid_frame_ids)

            # Store sequence metadata with lazy loading
            all_data.append({
                'seq_dir': seq_dir,
                'seq_name': seq_name,
                'frame_ids': valid_frame_ids,
                'dataset': 'eden',
                'lazy_load': True
            })

        print(f"[Eden-{self.split.upper()}] Found {len(all_data)} sequences with {total_frames} total frames (lazy loading).")

        self._save_cache(all_data)

        return all_data

    def __len__(self):
        # Calculate total length including lazy-loaded entries
        total_len = 0
        for entry in self.all_data_paths:
            if entry.get('lazy_load', False):
                dataset = entry.get('dataset', '')
                if dataset == 'tartanairv2':
                    # TartanAirV2 lazy-loaded trajectory
                    total_len += len(entry['frame_ids'])
                elif dataset == 'tartanair':
                    # TartanAir lazy-loaded sequence
                    total_len += len(entry['frame_ids'])
                elif dataset == 'scenenet':
                    # SceneNet lazy-loaded trajectory
                    total_len += len(entry['frame_ids'])
                elif dataset == 'omniworldgame':
                    # OmniWorld-Game lazy-loaded scene
                    # Count frames across all valid splits
                    for split_idx in entry['valid_splits']:
                        total_len += len(entry['split_info']['split'][split_idx])
                elif dataset == 'irs':
                    # IRS lazy-loaded sequence
                    total_len += len(entry['frame_ids'])
                elif dataset == 'dynamic_replica':
                    # Dynamic Replica lazy-loaded sequence
                    total_len += len(entry['frame_ids'])
                elif dataset == 'hypersim':
                    # Hypersim lazy-loaded subscene
                    total_len += len(entry['frame_ids'])
                elif dataset == 'vkitti2':
                    # VKitti2 lazy-loaded sequence
                    total_len += len(entry['frame_ids'])
                elif dataset == 'urbansyn':
                    # UrbanSyn lazy-loaded dataset
                    total_len += len(entry['frame_ids'])
                elif dataset == 'synscapes':
                    # Synscapes lazy-loaded dataset
                    total_len += len(entry['frame_ids'])
                elif dataset == 'eden':
                    # Eden lazy-loaded sequence
                    total_len += len(entry['frame_ids'])
                elif dataset == 'h5_shards':
                    total_len += entry['sample_count']
            else:
                # Regular sample with explicit paths
                total_len += 1
        return total_len

    def _compute_cumulative_indices(self):
        """
        Pre-compute cumulative frame counts for O(log n) index lookup.

        Similar to PyTorch's ConcatDataset.cumulative_sizes pattern.
        This enables binary search instead of linear search in __getitem__.

        Creates self._cumulative_indices: list where each element is the cumulative
        count of frames up to that entry index. Used with bisect.bisect_right()
        to find which entry contains a given global index.
        """
        cumulative = [0]
        for entry in self.all_data_paths:
            if entry.get('lazy_load', False):
                dataset = entry.get('dataset', '')

                # Count frames based on dataset type
                if dataset == 'tartanairv2':
                    num_frames = len(entry['frame_ids'])
                elif dataset == 'tartanair':
                    num_frames = len(entry['frame_ids'])
                elif dataset == 'scenenet':
                    num_frames = len(entry['frame_ids'])
                elif dataset == 'omniworldgame':
                    # Count frames across all valid splits
                    num_frames = sum(len(entry['split_info']['split'][split_idx])
                                     for split_idx in entry['valid_splits'])
                elif dataset == 'irs':
                    num_frames = len(entry['frame_ids'])
                elif dataset == 'dynamic_replica':
                    num_frames = len(entry['frame_ids'])
                elif dataset == 'hypersim':
                    num_frames = len(entry['frame_ids'])
                elif dataset == 'vkitti2':
                    num_frames = len(entry['frame_ids'])
                elif dataset == 'urbansyn':
                    num_frames = len(entry['frame_ids'])
                elif dataset == 'synscapes':
                    num_frames = len(entry['frame_ids'])
                elif dataset == 'eden':
                    num_frames = len(entry['frame_ids'])
                elif dataset == 'h5_shards':
                    num_frames = entry['sample_count']
                else:
                    # Unknown dataset - assume single frame
                    num_frames = 1

                cumulative.append(cumulative[-1] + num_frames)
            else:
                # Regular entry with explicit paths
                cumulative.append(cumulative[-1] + 1)

        self._cumulative_indices = cumulative
        print(f"[{self.dataset_name.upper()}] Computed cumulative indices: {len(cumulative)} entries, {cumulative[-1]} total frames")

    def __getitem__(self, idx, _retry=0):
        """idx: an index, or (index, height, width, side) from dataloader/aspect_batches.AspectBatchSampler."""
        _MAX_RETRIES = 10
        idx, frame = (idx[0], idx[1:]) if isinstance(idx, tuple) else (idx, None)
        try:
            return self._getitem_impl(idx, frame)
        except (UnidentifiedImageError, OSError, IOError, EOFError) as e:
            if _retry < _MAX_RETRIES:
                print(f"[WARNING] Skipping corrupted sample idx={idx}: {e}")
                new_idx = random.randint(0, len(self) - 1)
                return self.__getitem__(new_idx if frame is None else (new_idx, *frame), _retry=_retry + 1)
            else:
                raise RuntimeError(f"Too many corrupted samples (>{_MAX_RETRIES} retries). Last error: {e}")

    def _getitem_impl(self, idx, frame=None):

        t_start = time.perf_counter() if self.debug_timing else None

        # OPTIMIZATION: Use binary search for O(log n) lookup instead of O(n) linear search
        entry_idx = bisect.bisect_right(self._cumulative_indices, idx) - 1
        frame_idx = idx - self._cumulative_indices[entry_idx]

        entry = self.all_data_paths[entry_idx]

        # Handle lazy-loaded entries
        if entry.get('lazy_load', False):
            dataset = entry.get('dataset', '')

            if dataset == 'tartanairv2':
                # TartanAirV2 lazy-loaded trajectory
                traj_dir = entry['traj_dir']
                cam = entry['camera']
                frame_id = entry['frame_ids'][frame_idx]

                data_pair = {
                    'image': os.path.join(traj_dir, f'image_{cam}', f'{frame_id}_{cam}.png'),
                    'depth': os.path.join(traj_dir, f'depth_{cam}', f'{frame_id}_{cam}_depth.png'),
                    'intrinsics': entry['intrinsics'],
                    'dataset': entry['dataset']
                }

            elif dataset == 'tartanair':
                # TartanAir lazy-loaded sequence
                seq_dir = entry['seq_dir']
                frame_id = entry['frame_ids'][frame_idx]

                data_pair = {
                    'image': os.path.join(seq_dir, 'image_left', f'{frame_id}.png'),
                    'depth': os.path.join(seq_dir, 'depth_left', f'{frame_id}_depth.npy'),
                    'intrinsics': entry['intrinsics'],
                    'dataset': entry['dataset']
                }

            elif dataset == 'scenenet':
                # SceneNet lazy-loaded trajectory
                traj_dir = entry['traj_dir']
                frame_id = entry['frame_ids'][frame_idx]

                data_pair = {
                    'image': os.path.join(traj_dir, 'photo', f'{frame_id}.jpg'),
                    'depth': os.path.join(traj_dir, 'depth', f'{frame_id}.png'),
                    'intrinsics': entry['intrinsics'],
                    'dataset': entry['dataset']
                }

            elif dataset == 'omniworldgame':
                # OmniWorld-Game lazy-loaded scene - map frame_idx to correct split
                current_idx = frame_idx
                for split_idx in entry['valid_splits']:
                    frames_in_split = entry['split_info']['split'][split_idx]
                    if current_idx < len(frames_in_split):
                        # Found the correct split
                        frame_idx_global = frames_in_split[current_idx]
                        frame_str = f"{frame_idx_global:06d}"

                        data_pair = {
                            'image': os.path.join(entry['color_dir'], f'{frame_str}.png'),
                            'depth': os.path.join(entry['depth_dir'], f'{frame_str}.png'),
                            'cam_file': os.path.join(entry['camera_dir'], f'split_{split_idx}.json'),
                            'split_idx': split_idx,
                            'frame_idx_in_split': current_idx,
                            'dataset': entry['dataset']
                        }
                        break
                    else:
                        current_idx -= len(frames_in_split)

            elif dataset == 'irs':
                # IRS lazy-loaded sequence
                seq_dir = entry['seq_dir']
                frame_id = entry['frame_ids'][frame_idx]

                data_pair = {
                    'image': os.path.join(seq_dir, 'rgb', f'{frame_id}.png'),
                    'depth': os.path.join(seq_dir, 'depth', f'{frame_id}.npy'),
                    'cam_file': os.path.join(seq_dir, 'cam', f'{frame_id}.npz'),
                    'dataset': entry['dataset']
                }

            elif dataset == 'dynamic_replica':
                # Dynamic Replica lazy-loaded sequence
                seq_dir = entry['seq_dir']
                frame_id = entry['frame_ids'][frame_idx]

                # Only use 'left' camera (right is empty)
                cam_path = os.path.join(seq_dir, 'left')
                data_pair = {
                    'image': os.path.join(cam_path, 'rgb', f'{frame_id}.png'),
                    'depth': os.path.join(cam_path, 'depth', f'{frame_id}.npy'),
                    'cam_file': os.path.join(cam_path, 'cam', f'{frame_id}.npz'),
                    'dataset': entry['dataset']
                }

            elif dataset == 'hypersim':
                # Hypersim lazy-loaded subscene
                subscene_dir = entry['subscene_dir']
                frame_id = entry['frame_ids'][frame_idx]

                data_pair = {
                    'image': os.path.join(subscene_dir, f'{frame_id}_rgb.png'),
                    'depth': os.path.join(subscene_dir, f'{frame_id}_depth.npy'),
                    'cam_file': os.path.join(subscene_dir, f'{frame_id}_cam.npz'),
                    'dataset': entry['dataset']
                }

            elif dataset == 'vkitti2':
                # VKitti2 lazy-loaded sequence
                rgb_dir = entry['rgb_dir']
                depth_dir = entry['depth_dir']
                frame_id = entry['frame_ids'][frame_idx]

                data_pair = {
                    'image': os.path.join(rgb_dir, f'rgb_{frame_id}.jpg'),
                    'depth': os.path.join(depth_dir, f'depth_{frame_id}.png'),
                    'intrinsics': entry['intrinsics'],
                    'camera_id': entry['camera_id'],
                    'frame_idx': int(frame_id),
                    'dataset': entry['dataset']
                }

            elif dataset == 'urbansyn':
                # UrbanSyn lazy-loaded dataset
                rgb_dir = entry['rgb_dir']
                depth_dir = entry['depth_dir']
                cam_dir = entry['cam_dir']
                frame_id = entry['frame_ids'][frame_idx]

                data_pair = {
                    'image': os.path.join(rgb_dir, f'{frame_id}.png'),
                    'depth': os.path.join(depth_dir, f'{frame_id}.npy'),
                    'cam_file': os.path.join(cam_dir, f'{frame_id}.npz'),
                    'dataset': entry['dataset']
                }

            elif dataset == 'synscapes':
                # Synscapes lazy-loaded dataset
                rgb_dir = entry['rgb_dir']
                depth_dir = entry['depth_dir']
                cam_dir = entry['cam_dir']
                frame_id = entry['frame_ids'][frame_idx]

                data_pair = {
                    'image': os.path.join(rgb_dir, f'{frame_id}.png'),
                    'depth': os.path.join(depth_dir, f'{frame_id}.npy'),
                    'cam_file': os.path.join(cam_dir, f'{frame_id}.npz'),
                    'dataset': entry['dataset']
                }

            elif dataset == 'eden':
                # Eden lazy-loaded sequence
                seq_dir = entry['seq_dir']
                frame_id = entry['frame_ids'][frame_idx]

                data_pair = {
                    'image': os.path.join(seq_dir, 'rgb', f'{frame_id}.png'),
                    'depth': os.path.join(seq_dir, 'depth', f'{frame_id}.npy'),
                    'cam_file': os.path.join(seq_dir, 'cam', f'{frame_id}.npz'),
                    'dataset': entry['dataset']
                }

            elif dataset == 'h5_shards':
                data_pair = shard_data_pair(entry, frame_idx, self.dataset_name, with_instance=self.load_instance)
            else:
                raise ValueError(f"Unknown lazy-loaded dataset: {dataset}")
        else:
            # Regular entry with explicit paths
            data_pair = entry

        # 1. Load Image
        img_pil = Image.open(data_pair['image']).convert("RGB")

        # 2. Load Depth and Convert to Pointcloud
        if self.dataset_name == 'scenenet':
            # Load depth (16-bit PNG in mm)
            depth_img = Image.open(data_pair['depth'])
            depth_map_np = np.array(depth_img).astype(np.float32) * 0.001  # mm to meters

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # Resize if requested (using torch for speed)
            if self.resize_height is not None or self.resize_width is not None:
                orig_w, orig_h = img_pil.size
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                # Get cached ray array (or compute if not cached)
                cache_key = (new_w, new_h)
                if cache_key not in self._ray_cache:
                    self._ray_cache[cache_key] = scenenet_ray_directions(width=new_w, height=new_h, device=self.conversion_device)
                ray_array = self._ray_cache[cache_key]

                # Compute intrinsics for resized image
                fx, fy, cx, cy = compute_scenenet_intrinsics(new_w, new_h)
            else:
                # Use pre-cached default size ray array
                ray_array = self._ray_cache[(320, 240)]
                # Default intrinsics for 320x240
                fx, fy, cx, cy = compute_scenenet_intrinsics(320, 240)

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert ray-depth to 3D points (GPU accelerated)
            pc_tensor = raydepth_to_points(depth_map, ray_array)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline (temporary, will optimize further later)
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name == 'tartanair':
            # Load z-depth (NPY in meters)
            depth_map_np = np.load(data_pair['depth']).astype(np.float32)

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # TartanAir fixed intrinsics: fx=fy=320, cx=320, cy=240 for 640x480
            orig_fx, orig_fy, orig_cx, orig_cy = 320.0, 320.0, 320.0, 240.0
            orig_w, orig_h = img_pil.size

            # Resize logic
            if self.finetune_highres_mode:
                # High-res finetuning mode: adaptive resize based on native resolution
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    # Resize image (bilinear)
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics
                    fx = orig_fx * scale
                    fy = orig_fy * scale
                    cx = orig_cx * scale
                    cy = orig_cy * scale

                    # Get backprojection grid
                    cache_key = (new_w, new_h, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                    if cache_key not in self._backproj_cache:
                        self._backproj_cache[cache_key] = compute_backprojection_grid(new_w, new_h, fx, fy, cx, cy, device=self.conversion_device)
                    backproj_grid = self._backproj_cache[cache_key]
                else:
                    fx, fy, cx, cy = orig_fx, orig_fy, orig_cx, orig_cy
                    cache_key = (640, 480, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                    backproj_grid = self._backproj_cache[cache_key]

            elif self.resize_height is not None or self.resize_width is not None:
                # Standard resize mode (for backward compatibility)
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                # Scale intrinsics
                fx = orig_fx * scale
                fy = orig_fy * scale
                cx = orig_cx * scale
                cy = orig_cy * scale

                # Get cached backprojection grid (or compute if not cached)
                cache_key = (new_w, new_h, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                if cache_key not in self._backproj_cache:
                    self._backproj_cache[cache_key] = compute_backprojection_grid(new_w, new_h, fx, fy, cx, cy, device=self.conversion_device)
                backproj_grid = self._backproj_cache[cache_key]
            else:
                # Use pre-cached default size grid
                fx, fy, cx, cy = orig_fx, orig_fy, orig_cx, orig_cy
                cache_key = (640, 480, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                backproj_grid = self._backproj_cache[cache_key]

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert z-depth to 3D points (GPU accelerated)
            pc_tensor = points_in_camera_coords_zdepth(depth_map, backproj_grid)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name == 'tartanairv2':
            # Load z-depth (RGBA PNG encoding float32)
            depth_map_np = read_decode_tartanairv2_depth(data_pair['depth'])

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # TartanAir V2 fixed intrinsics: fx=fy=cx=cy=320 for 640x640
            orig_fx, orig_fy, orig_cx, orig_cy = 320.0, 320.0, 320.0, 320.0
            orig_w, orig_h = img_pil.size

            # Resize logic
            if self.finetune_highres_mode:
                # High-res finetuning mode: adaptive resize based on native resolution
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    # Resize image (bilinear)
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics
                    fx = orig_fx * scale
                    fy = orig_fy * scale
                    cx = orig_cx * scale
                    cy = orig_cy * scale

                    # Get backprojection grid
                    cache_key = (new_w, new_h, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                    if cache_key not in self._backproj_cache:
                        self._backproj_cache[cache_key] = compute_backprojection_grid(new_w, new_h, fx, fy, cx, cy, device=self.conversion_device)
                    backproj_grid = self._backproj_cache[cache_key]
                else:
                    fx, fy, cx, cy = orig_fx, orig_fy, orig_cx, orig_cy
                    cache_key = (640, 640, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                    backproj_grid = self._backproj_cache[cache_key]

            elif self.resize_height is not None or self.resize_width is not None:
                # Standard resize mode (for backward compatibility)
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                # Scale intrinsics
                fx = orig_fx * scale
                fy = orig_fy * scale
                cx = orig_cx * scale
                cy = orig_cy * scale

                # Get cached backprojection grid (or compute if not cached)
                cache_key = (new_w, new_h, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                if cache_key not in self._backproj_cache:
                    self._backproj_cache[cache_key] = compute_backprojection_grid(new_w, new_h, fx, fy, cx, cy, device=self.conversion_device)
                backproj_grid = self._backproj_cache[cache_key]
            else:
                # Use pre-cached default size grid (640x640)
                fx, fy, cx, cy = orig_fx, orig_fy, orig_cx, orig_cy
                cache_key = (640, 640, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                backproj_grid = self._backproj_cache[cache_key]

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert z-depth to 3D points (GPU accelerated)
            pc_tensor = points_in_camera_coords_zdepth(depth_map, backproj_grid)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name == 'vkitti2':
            # Load z-depth (16-bit PNG in cm)
            import cv2
            depth_raw = cv2.imread(data_pair['depth'], cv2.IMREAD_ANYDEPTH)
            depth_map_np = depth_raw.astype(np.float32) / 100.0  # cm to meters

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # Get intrinsics for this frame
            intrinsic_file = data_pair['intrinsics']

            # Cache intrinsics file parsing
            if intrinsic_file not in self.vkitti2_intrinsics_cache:
                self.vkitti2_intrinsics_cache[intrinsic_file] = load_vkitti_intrinsics(intrinsic_file)

            intrinsics_map = self.vkitti2_intrinsics_cache[intrinsic_file]

            frame_idx = data_pair['frame_idx']
            camera_id = data_pair['camera_id']

            if (frame_idx, camera_id) not in intrinsics_map:
                raise ValueError(f"Intrinsics not found for frame {frame_idx}, camera {camera_id}")

            fx, fy, cx, cy = intrinsics_map[(frame_idx, camera_id)]

            # Resize logic
            orig_w, orig_h = img_pil.size

            if self.finetune_highres_mode:
                # High-res finetuning mode: adaptive resize based on native resolution
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    # Resize image (bilinear)
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                    h, w = new_h, new_w
                else:
                    h, w = orig_h, orig_w

            elif self.resize_height is not None or self.resize_width is not None:
                # Standard resize mode (for backward compatibility)
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics (uniform scale for aspect ratio preservation)
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                h, w = new_h, new_w
            else:
                h, w = depth_tensor.shape[2], depth_tensor.shape[3]

            # Get cached backprojection grid (or compute if not cached)
            cache_key = (w, h, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
            if cache_key not in self._backproj_cache:
                self._backproj_cache[cache_key] = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)
            backproj_grid = self._backproj_cache[cache_key]

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert z-depth to 3D points (GPU accelerated)
            pc_tensor = points_in_camera_coords_zdepth(depth_map, backproj_grid)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name == 'hypersim':
            # Load z-depth (.npy in meters)
            depth_map_np = np.load(data_pair['depth']).astype(np.float32)

            # FIX: Filter invalid depth values to prevent NaN propagation
            depth_map_np[~np.isfinite(depth_map_np)] = 0.

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # Load intrinsics from .npz file (conditionally cache to avoid OOM with per-frame intrinsics)
            cam_file = data_pair['cam_file']
            if self._use_cam_cache:
                if cam_file not in self.hypersim_cam_cache:
                    self.hypersim_cam_cache[cam_file] = np.load(cam_file)
                cam_data = self.hypersim_cam_cache[cam_file]
            else:
                cam_data = np.load(cam_file)
            K = cam_data['intrinsics']  # [3, 3]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # FIX: Validate intrinsics - skip samples with invalid focal lengths
            if not (np.isfinite(fx) and np.isfinite(fy) and np.isfinite(cx) and np.isfinite(cy)):
                print(f"[WARNING] Hypersim sample has non-finite intrinsics: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                # Return a dummy sample with zeros
                dummy_img = torch.zeros(3, 256, 256)
                dummy_pc = torch.zeros(3, 256, 256)
                return {
                    'image': dummy_img,
                    'pointcloud': dummy_pc,
                    'scale_factor': torch.tensor(1.0),
                    'dataset_name': self.dataset_name,
                    'intrinsics': (fx, fy, cx, cy),
                    'valid_mask': torch.zeros(256, 256, dtype=torch.bool),
                    'sky_mask': torch.zeros(256, 256, dtype=torch.bool),
                }

            if fx <= 0 or fy <= 0:
                print(f"[WARNING] Hypersim sample has invalid focal lengths: fx={fx}, fy={fy}")
                # Return a dummy sample with zeros
                dummy_img = torch.zeros(3, 256, 256)
                dummy_pc = torch.zeros(3, 256, 256)
                return {
                    'image': dummy_img,
                    'pointcloud': dummy_pc,
                    'scale_factor': torch.tensor(1.0),
                    'dataset_name': self.dataset_name,
                    'intrinsics': (fx, fy, cx, cy),
                    'valid_mask': torch.zeros(256, 256, dtype=torch.bool),
                    'sky_mask': torch.zeros(256, 256, dtype=torch.bool),
                }

            # Resize logic
            orig_w, orig_h = img_pil.size

            if self.finetune_highres_mode:
                # High-res finetuning mode: adaptive resize based on native resolution
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    # Resize image (bilinear)
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                    h, w = new_h, new_w
                else:
                    h, w = orig_h, orig_w

            elif self.resize_height is not None or self.resize_width is not None:
                # Standard resize mode (for backward compatibility)
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics (uniform scale for aspect ratio preservation)
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                h, w = new_h, new_w
            else:
                h, w = depth_tensor.shape[2], depth_tensor.shape[3]

            # Get backprojection grid - compute on-the-fly for variable intrinsics datasets
            # to avoid unbounded cache growth (Hypersim has per-frame intrinsics)
            if self._use_backproj_cache:
                cache_key = (w, h, round(fx, 6), round(fy, 6), round(cx, 6), round(cy, 6))
                if cache_key not in self._backproj_cache:
                    self._backproj_cache[cache_key] = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)
                backproj_grid = self._backproj_cache[cache_key]
            else:
                backproj_grid = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert z-depth to 3D points (GPU accelerated)
            pc_tensor = points_in_camera_coords_zdepth(depth_map, backproj_grid)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name == 'urbansyn':
            # Load z-depth (NPY in meters, preprocessed)
            depth_map_np = np.load(data_pair['depth']).astype(np.float32)

            # Filter invalid depth values to prevent NaN propagation
            depth_map_np[~np.isfinite(depth_map_np)] = 0.

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # Load intrinsics from .npz file (conditionally cache to avoid OOM)
            cam_file = data_pair['cam_file']
            if self._use_cam_cache:
                if not hasattr(self, 'urbansyn_cam_cache'):
                    self.urbansyn_cam_cache = {}
                if cam_file not in self.urbansyn_cam_cache:
                    self.urbansyn_cam_cache[cam_file] = np.load(cam_file)
                cam_data = self.urbansyn_cam_cache[cam_file]
            else:
                cam_data = np.load(cam_file)
            K = cam_data['intrinsics']  # [3, 3]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # Resize logic
            orig_w, orig_h = img_pil.size

            if self.finetune_highres_mode:
                # High-res finetuning mode: adaptive resize based on native resolution
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    # Resize image (bilinear)
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                    h, w = new_h, new_w
                else:
                    h, w = orig_h, orig_w

            elif self.resize_height is not None or self.resize_width is not None:
                # Standard resize mode (for backward compatibility)
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics (uniform scale for aspect ratio preservation)
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                h, w = new_h, new_w
            else:
                h, w = depth_tensor.shape[2], depth_tensor.shape[3]

            # Get backprojection grid - compute on-the-fly for variable intrinsics datasets
            if self._use_backproj_cache:
                cache_key = (w, h, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                if cache_key not in self._backproj_cache:
                    self._backproj_cache[cache_key] = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)
                backproj_grid = self._backproj_cache[cache_key]
            else:
                backproj_grid = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert z-depth to 3D points (GPU accelerated)
            pc_tensor = points_in_camera_coords_zdepth(depth_map, backproj_grid)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name == 'mvssynth':
            # Load z-depth (NPY in centimeters, preprocessed from EXR)
            depth_map_np = np.load(data_pair['depth']).astype(np.float32)

            # Filter invalid depth values to prevent NaN propagation
            depth_map_np[~np.isfinite(depth_map_np)] = 0.

            # Convert from centimeters to meters
            depth_map_np = depth_map_np / 100.0

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # Load intrinsics from .npz file (conditionally cache to avoid OOM)
            cam_file = data_pair['cam_file']
            if self._use_cam_cache:
                if not hasattr(self, 'mvssynth_cam_cache'):
                    self.mvssynth_cam_cache = {}
                if cam_file not in self.mvssynth_cam_cache:
                    self.mvssynth_cam_cache[cam_file] = np.load(cam_file)
                cam_data = self.mvssynth_cam_cache[cam_file]
            else:
                cam_data = np.load(cam_file)
            K = cam_data['intrinsics']  # [3, 3]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # Resize logic
            orig_w, orig_h = img_pil.size

            if self.finetune_highres_mode:
                # High-res finetuning mode: adaptive resize based on native resolution
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    # Resize image (bilinear)
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                    h, w = new_h, new_w
                else:
                    h, w = orig_h, orig_w

            elif self.resize_height is not None or self.resize_width is not None:
                # Standard resize mode (for backward compatibility)
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics (uniform scale for aspect ratio preservation)
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                h, w = new_h, new_w
            else:
                h, w = depth_tensor.shape[2], depth_tensor.shape[3]

            # Get backprojection grid - compute on-the-fly for variable intrinsics datasets
            if self._use_backproj_cache:
                cache_key = (w, h, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                if cache_key not in self._backproj_cache:
                    self._backproj_cache[cache_key] = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)
                backproj_grid = self._backproj_cache[cache_key]
            else:
                backproj_grid = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert z-depth to 3D points (GPU accelerated)
            pc_tensor = points_in_camera_coords_zdepth(depth_map, backproj_grid)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name == 'synscapes':
            # Load z-depth (NPY in meters, preprocessed from EXR)
            depth_map_np = np.load(data_pair['depth']).astype(np.float32)

            # Filter invalid depth values to prevent NaN propagation
            depth_map_np[~np.isfinite(depth_map_np)] = 0.

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # Load intrinsics from .npz file (conditionally cache to avoid OOM)
            cam_file = data_pair['cam_file']
            if self._use_cam_cache:
                if not hasattr(self, 'synscapes_cam_cache'):
                    self.synscapes_cam_cache = {}
                if cam_file not in self.synscapes_cam_cache:
                    self.synscapes_cam_cache[cam_file] = np.load(cam_file)
                cam_data = self.synscapes_cam_cache[cam_file]
            else:
                cam_data = np.load(cam_file)
            K = cam_data['intrinsics']  # [3, 3]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # Resize logic
            orig_w, orig_h = img_pil.size

            if self.finetune_highres_mode:
                # High-res finetuning mode: adaptive resize based on native resolution
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    # Resize image (bilinear)
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                    h, w = new_h, new_w
                else:
                    h, w = orig_h, orig_w

            elif self.resize_height is not None or self.resize_width is not None:
                # Standard resize mode (for backward compatibility)
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics (uniform scale for aspect ratio preservation)
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                h, w = new_h, new_w
            else:
                h, w = depth_tensor.shape[2], depth_tensor.shape[3]

            # Get backprojection grid - compute on-the-fly for variable intrinsics datasets
            if self._use_backproj_cache:
                cache_key = (w, h, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                if cache_key not in self._backproj_cache:
                    self._backproj_cache[cache_key] = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)
                backproj_grid = self._backproj_cache[cache_key]
            else:
                backproj_grid = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert z-depth to 3D points (GPU accelerated)
            pc_tensor = points_in_camera_coords_zdepth(depth_map, backproj_grid)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name == 'omniworldgame':
            # Load z-depth (16-bit PNG with special encoding)
            depth_map_np, valid_mask = load_omniworldgame_depth(data_pair['depth'])

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # Load intrinsics from camera JSON file (conditionally cache to avoid OOM)
            cam_file = data_pair['cam_file']
            frame_idx_in_split = data_pair['frame_idx_in_split']

            if self._use_cam_cache:
                if not hasattr(self, 'omniworldgame_cam_cache'):
                    self.omniworldgame_cam_cache = {}
                if cam_file not in self.omniworldgame_cam_cache:
                    import json
                    with open(cam_file, 'r', encoding='utf-8') as f:
                        self.omniworldgame_cam_cache[cam_file] = json.load(f)
                cam_data = self.omniworldgame_cam_cache[cam_file]
            else:
                import json
                with open(cam_file, 'r', encoding='utf-8') as f:
                    cam_data = json.load(f)

            # Extract intrinsics for this specific frame
            # OmniWorld-Game has per-frame focal length (list) but constant cx/cy (scalars)
            focal = cam_data['focals'][frame_idx_in_split]
            cx = cam_data['cx']  # Constant for the split
            cy = cam_data['cy']  # Constant for the split
            fx = fy = focal  # Same focal length for x and y

            # Resize logic
            orig_w, orig_h = img_pil.size

            if self.finetune_highres_mode:
                # High-res finetuning mode: adaptive resize based on native resolution
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    # Resize image (bilinear)
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                    h, w = new_h, new_w
                else:
                    h, w = orig_h, orig_w

            elif self.resize_height is not None or self.resize_width is not None:
                # Standard resize mode (for backward compatibility)
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics (uniform scale for aspect ratio preservation)
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                h, w = new_h, new_w
            else:
                h, w = depth_tensor.shape[2], depth_tensor.shape[3]

            # Get backprojection grid - compute on-the-fly for variable intrinsics datasets
            if self._use_backproj_cache:
                cache_key = (w, h, round(fx, 6), round(fy, 6), round(cx, 6), round(cy, 6))
                if cache_key not in self._backproj_cache:
                    self._backproj_cache[cache_key] = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)
                backproj_grid = self._backproj_cache[cache_key]
            else:
                backproj_grid = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert z-depth to 3D points (GPU accelerated)
            pc_tensor = points_in_camera_coords_zdepth(depth_map, backproj_grid)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name == 'irs':
            # Load z-depth (NPY in meters, preprocessed from disparity EXR)
            depth_map_np = np.load(data_pair['depth']).astype(np.float32)

            # Filter invalid depth values to prevent NaN propagation
            depth_map_np[~np.isfinite(depth_map_np)] = 0.

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # Load intrinsics from .npz file (conditionally cache to avoid OOM)
            cam_file = data_pair['cam_file']
            if self._use_cam_cache:
                if not hasattr(self, 'irs_cam_cache'):
                    self.irs_cam_cache = {}
                if cam_file not in self.irs_cam_cache:
                    self.irs_cam_cache[cam_file] = np.load(cam_file)
                cam_data = self.irs_cam_cache[cam_file]
            else:
                cam_data = np.load(cam_file)
            K = cam_data['intrinsics']  # [3, 3]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # Resize logic
            orig_w, orig_h = img_pil.size

            if self.finetune_highres_mode:
                # High-res finetuning mode: adaptive resize based on native resolution
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    # Resize image (bilinear)
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                    h, w = new_h, new_w
                else:
                    h, w = orig_h, orig_w

            elif self.resize_height is not None or self.resize_width is not None:
                # Standard resize mode (for backward compatibility)
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics (uniform scale for aspect ratio preservation)
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                h, w = new_h, new_w
            else:
                h, w = depth_tensor.shape[2], depth_tensor.shape[3]

            # Get backprojection grid - compute on-the-fly for variable intrinsics datasets
            if self._use_backproj_cache:
                cache_key = (w, h, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                if cache_key not in self._backproj_cache:
                    self._backproj_cache[cache_key] = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)
                backproj_grid = self._backproj_cache[cache_key]
            else:
                backproj_grid = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert z-depth to 3D points (GPU accelerated)
            pc_tensor = points_in_camera_coords_zdepth(depth_map, backproj_grid)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name == 'dynamic_replica':
            # Load z-depth (NPY in meters, float32)
            depth_map_np = np.load(data_pair['depth']).astype(np.float32)

            # Filter invalid depth values to prevent NaN propagation
            depth_map_np[~np.isfinite(depth_map_np)] = 0.

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # Load intrinsics from .npz file (can cache since intrinsics are fixed at fx=fy=700, cx=640, cy=360)
            cam_file = data_pair['cam_file']
            if self._use_cam_cache:
                if not hasattr(self, 'dynamic_replica_cam_cache'):
                    self.dynamic_replica_cam_cache = {}
                if cam_file not in self.dynamic_replica_cam_cache:
                    self.dynamic_replica_cam_cache[cam_file] = np.load(cam_file)
                cam_data = self.dynamic_replica_cam_cache[cam_file]
            else:
                cam_data = np.load(cam_file)
            K = cam_data['intrinsics']  # [3, 3]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # Resize logic
            orig_w, orig_h = img_pil.size

            if self.finetune_highres_mode:
                # High-res finetuning mode: adaptive resize based on native resolution
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    # Resize image (bilinear)
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                    h, w = new_h, new_w
                else:
                    h, w = orig_h, orig_w

            elif self.resize_height is not None or self.resize_width is not None:
                # Standard resize mode (for backward compatibility)
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics (uniform scale for aspect ratio preservation)
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                h, w = new_h, new_w
            else:
                h, w = depth_tensor.shape[2], depth_tensor.shape[3]

            # Get backprojection grid - can cache since intrinsics are fixed
            if self._use_backproj_cache:
                cache_key = (w, h, round(fx, 2), round(fy, 2), round(cx, 2), round(cy, 2))
                if cache_key not in self._backproj_cache:
                    self._backproj_cache[cache_key] = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)
                backproj_grid = self._backproj_cache[cache_key]
            else:
                backproj_grid = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert z-depth to 3D points (GPU accelerated)
            pc_tensor = points_in_camera_coords_zdepth(depth_map, backproj_grid)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name == 'eden':
            # Load z-depth from NPY file (preprocessed, meters)
            depth_map_np = np.load(data_pair['depth']).astype(np.float32)

            # Filter invalid depth values to prevent NaN propagation
            depth_map_np[~np.isfinite(depth_map_np)] = 0.

            # Convert to tensor early for GPU operations
            depth_tensor = torch.from_numpy(depth_map_np).unsqueeze(0).unsqueeze(0).to(self.conversion_device)  # (1, 1, H, W)

            # Load intrinsics from NPZ file
            cam_file = data_pair['cam_file']
            cam_data = np.load(cam_file)
            K = cam_data['intrinsics']  # [3, 3]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

            # Resize logic
            orig_w, orig_h = img_pil.size

            if self.finetune_highres_mode:
                # High-res finetuning mode: adaptive resize based on native resolution
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    # Resize image (bilinear)
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                    h, w = new_h, new_w
                else:
                    h, w = orig_h, orig_w

            elif self.resize_height is not None or self.resize_width is not None:
                # Standard resize mode (for backward compatibility)
                scale_h = self.resize_height / orig_h if self.resize_height else 0
                scale_w = self.resize_width / orig_w if self.resize_width else 0
                scale = max(scale_h, scale_w) if (scale_h and scale_w) else (scale_h or scale_w)
                new_h = int(orig_h * scale)
                new_w = int(orig_w * scale)

                # OPTIMIZATION: Skip resize if dimensions already match
                if new_h != orig_h or new_w != orig_w:
                    # Resize image (bilinear) - convert to tensor for faster resize
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)  # (1, 3, H, W)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))

                    # Resize depth (nearest)
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')

                    # Scale intrinsics (uniform scale for aspect ratio preservation)
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale

                h, w = new_h, new_w
            else:
                h, w = depth_tensor.shape[2], depth_tensor.shape[3]

            # Eden: Don't cache backprojection grids (per-frame intrinsics)
            backproj_grid = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)

            # Convert to 2D tensor for pointcloud conversion
            depth_map = depth_tensor.squeeze(0).squeeze(0)  # (H, W)

            # Convert z-depth to 3D points (GPU accelerated)
            pc_tensor = points_in_camera_coords_zdepth(depth_map, backproj_grid)  # (H, W, 3)

            # Convert back to numpy for rest of pipeline
            pc_np = pc_tensor.cpu().numpy()

        elif self.dataset_name in H5_SHARD_DATASET_NAMES:
            if self.dataset_name == 'vkitti_h5':
                img_pil = vkitti_benchmark_crop_image(img_pil)
            depth_tensor = torch.from_numpy(data_pair['depth']).unsqueeze(0).unsqueeze(0).to(self.conversion_device)
            normal_tensor = torch.from_numpy(data_pair['normal']).permute(2, 0, 1).unsqueeze(0).to(self.conversion_device)
            instance_tensor = None
            if self.load_instance:
                instance_tensor = torch.from_numpy(data_pair['instance'])[None, None].float()
            fx, fy, cx, cy = data_pair['intrinsics']
            orig_w, orig_h = img_pil.size
            h, w = orig_h, orig_w
            if frame is not None:
                # the crop of dataloader/aspect_batches.py, resized to the frame size of its batch
                h, w, side = frame
                top, left, crop_h, crop_w = random_frame_crop(orig_h, orig_w, h, w, side)
                img_tensor = TF.to_tensor(img_pil)[None, :, top:top + crop_h, left:left + crop_w]
                img_tensor = torch.nn.functional.interpolate(img_tensor, size=(h, w), mode='bilinear',
                                                             align_corners=False, antialias=True)
                img_pil = TF.to_pil_image(img_tensor[0].clamp(0, 1))
                depth_tensor, normal_tensor = (torch.nn.functional.interpolate(
                    tensor[..., top:top + crop_h, left:left + crop_w], size=(h, w), mode='nearest')
                    for tensor in (depth_tensor, normal_tensor))
                if instance_tensor is not None:
                    instance_tensor = torch.nn.functional.interpolate(
                        instance_tensor[..., top:top + crop_h, left:left + crop_w], size=(h, w), mode='nearest')
                fx, cx = fx * w / crop_w, (cx - left) * w / crop_w
                fy, cy = fy * h / crop_h, (cy - top) * h / crop_h
            elif self.finetune_highres_mode:
                do_resize, scale, new_h, new_w = self._compute_highres_resize_params(orig_h, orig_w)
                if do_resize:
                    img_tensor = TF.to_tensor(img_pil).unsqueeze(0)
                    img_tensor = torch.nn.functional.interpolate(img_tensor, size=(new_h, new_w), mode='bilinear', align_corners=False)
                    img_pil = TF.to_pil_image(img_tensor.squeeze(0))
                    depth_tensor = torch.nn.functional.interpolate(depth_tensor, size=(new_h, new_w), mode='nearest')
                    normal_tensor = torch.nn.functional.interpolate(normal_tensor, size=(new_h, new_w), mode='nearest')
                    if instance_tensor is not None:
                        instance_tensor = torch.nn.functional.interpolate(instance_tensor, size=(new_h, new_w), mode='nearest')
                    fx = fx * scale
                    fy = fy * scale
                    cx = cx * scale
                    cy = cy * scale
                    h, w = new_h, new_w
            backproj_grid = compute_backprojection_grid(w, h, fx, fy, cx, cy, device=self.conversion_device)
            pc_tensor = points_in_camera_coords_zdepth(depth_tensor.squeeze(0).squeeze(0), backproj_grid)
            pc_np = pc_tensor.cpu().numpy()
            normal_np = normal_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
            if self.dataset_name in H5_SHARD_DATASET_NAMES:
                normal_np = orient_toward_camera(normal_np, pc_np)
            # the normal map rides along the point map through flip, filtering and crop
            pc_np = np.concatenate((pc_np, normal_np), axis=-1)
            # instance ids travel separately: the flip negates every third channel and depth filtering zeroes pixels
            if instance_tensor is not None:
                instance_np = instance_tensor[0, 0].numpy().astype(np.int32)

        t_depth_convert = time.perf_counter() if self.debug_timing else None

        # 3. Apply Augmentations

        # A. Color Jitter (Only applies to the Image)
        if self.split in ['train', 'all']:
            img_pil = self.color_jitter(img_pil)

            if self.more_img_aug:
                img_pil = self.appearance_aug(img_pil)

        # B. Random Horizontal Flip (Applies to BOTH)
        if self.split in ['train', 'all'] and random.random() < 0.5:
            img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            pc_np = np.flip(pc_np, axis=1).copy()
            pc_np[..., 0::3] = -pc_np[..., 0::3]  # x of every 3-vector: the point and, when present, the normal
            if self.load_instance:
                instance_np = np.flip(instance_np, axis=1).copy()
            cx = img_pil.size[0] - 1 - cx

        # C. Filter invalid points
        pc_np[~np.isfinite(pc_np)] = 0.

        if self.clamp_max_depth is not None:
            pc_np[:, :, 2] = np.minimum(pc_np[:, :, 2], self.clamp_max_depth)

        if not self.handle_sky:
            depth_mask = pc_np[..., 2] > self.max_depth
            pc_np[depth_mask] = 0.

        # D. Sanity check dimensions
        width, height = img_pil.size
        if pc_np.shape[:2] != (height, width):
            raise ValueError(
                f"Dimension mismatch at index {idx}: "
                f"Image size ({height}, {width}) does not match PC shape {pc_np.shape[:2]}"
            )

        # E. Apply Paired Crop (a frame of dataloader/aspect_batches.py is already cropped)
        if self.cropper is not None and frame is None:
            # Get dimensions before crop
            pre_crop_w, pre_crop_h = img_pil.size

            # Call cropper - may return 2 or 3 values depending on cropper type
            crop_result = self.cropper(img_pil, pc_np)

            if len(crop_result) == 3:
                # New crop classes that return exact offsets (for high-res finetuning mode)
                img_pil, pc_np, (crop_offset_x, crop_offset_y) = crop_result
                if self.load_instance:
                    crop_h, crop_w = pc_np.shape[:2]
                    instance_np = instance_np[crop_offset_y:crop_offset_y + crop_h, crop_offset_x:crop_offset_x + crop_w]
                # Adjust intrinsics using exact offsets
                cx = cx - crop_offset_x
                cy = cy - crop_offset_y
            else:
                # Legacy crop classes - use center approximation
                img_pil, pc_np = crop_result
                assert not self.load_instance, 'instance ids need a cropper that reports its exact offsets'

                # Compute crop offset (for center crop) or estimate (for random crop)
                # Note: Random crop offset is not deterministic, but we compute center offset as approximation
                if hasattr(self.cropper, 'size'):
                    target_h, target_w = self.cropper.size
                    # For CenterCropPaired, use exact center offset
                    # For RandomCropPaired, this is an approximation
                    crop_offset_x = (pre_crop_w - target_w) // 2
                    crop_offset_y = (pre_crop_h - target_h) // 2
                    # Adjust intrinsics for crop
                    cx = cx - crop_offset_x
                    cy = cy - crop_offset_y

        t_file_read = time.perf_counter() if self.debug_timing else None

        pc_np, normal_np = pc_np[..., :3], pc_np[..., 3:]

        # 4. Convert to Tensor
        image_tensor = self.image_to_tensor(img_pil)

        # Add random Gaussian noise
        aug_prob = 0.2 if self.stronger_img_aug else 0.1
        if self.more_img_aug and self.split == 'train' and random.random() < aug_prob:
            noise = torch.randn_like(image_tensor) * 0.02
            image_tensor = image_tensor + noise
            image_tensor = torch.clamp(image_tensor, 0.0, 1.0)

        # Filter invalid RGB
        image_tensor[~torch.isfinite(image_tensor)] = 0.

        # Convert pointcloud to tensor
        pointcloud_tensor = torch.from_numpy(pc_np).permute(2, 0, 1).float()
        pointcloud_unnormalized = pointcloud_tensor.clone()

        valid_mask = pointcloud_tensor[2, :, :] > 1e-6

        if self.handle_sky:
            sky_mask = pointcloud_unnormalized[2, :, :] > self.max_depth
            pointcloud_tensor[2, sky_mask] = 0.
            valid_mask = pointcloud_tensor[2, :, :] > 1e-6

        has_valid_points = valid_mask.any()

        # 5. Normalization

        # Center shift
        if self.center_shift:
            if has_valid_points:
                center = pointcloud_tensor[:, valid_mask].mean(dim=1).view(3, 1, 1)
            else:
                center = torch.zeros((3, 1, 1), device=pointcloud_tensor.device)

            if self.center_shift_z_only:
                center[:2, :, :] = 0.

            pointcloud_tensor = pointcloud_tensor - center

        # Scale normalize
        if self.no_scale_factor:
            scale_factor = torch.tensor(1.0)
        else:
            scale_factor = compute_distance_statistic(
                pointcloud_tensor,
                valid_mask=valid_mask if self.compute_scale_factor_only_valid else None,
                use_mean=self.normalize_by_mean,
                use_std=self.compute_scale_factor_use_std,
                use_percentile=self.compute_scale_factor_use_percentile
            )

            # Apply scale factor augmentation (training only)
            if self.scale_factor_augment and self.split in ['train', 'all']:
                aug_factor = random.uniform(*self.scale_factor_augment_range)
                scale_factor = scale_factor * aug_factor

            if scale_factor < 1e-6:
                pointcloud_tensor.zero_()
            else:
                pointcloud_tensor = pointcloud_tensor / scale_factor

        # Remove outliers after normalization (for flow matching stability)
        if self.remove_outliers:
            # Find points where any coordinate exceeds the threshold
            outlier_mask = (torch.abs(pointcloud_tensor[0, :, :]) > self.outlier_threshold) | \
                          (torch.abs(pointcloud_tensor[1, :, :]) > self.outlier_threshold) | \
                          (torch.abs(pointcloud_tensor[2, :, :]) > self.outlier_threshold)
            # Zero out outlier points
            pointcloud_tensor[:, outlier_mask] = 0.0
            # Update valid mask to exclude outliers
            valid_mask = valid_mask & ~outlier_mask

        # Handle sky
        if self.handle_sky:
            FAR_PLANE_VAL = self.sky_far_plane_value

            if self.use_sky_dome:
                if sky_mask.any():
                    raw_sky_vectors = pointcloud_unnormalized[:, sky_mask]
                    raw_norms = torch.norm(raw_sky_vectors, dim=0, keepdim=True)
                    sky_directions = raw_sky_vectors / (raw_norms + 1e-8)
                    pointcloud_tensor[:, sky_mask] = sky_directions * FAR_PLANE_VAL
            else:
                pointcloud_tensor[0, sky_mask] = 0.0
                pointcloud_tensor[1, sky_mask] = 0.0
                pointcloud_tensor[2, sky_mask] = FAR_PLANE_VAL

            valid_mask = valid_mask | sky_mask

        # 6. Return the dictionary
        out_dict = {
            'image': image_tensor,
            'pointcloud': pointcloud_tensor,
            'pointcloud_unnormalized': pointcloud_unnormalized,  # For visualization
            'scale_factor': scale_factor,
            'dataset_name': self.dataset_name,  # NEW: for multi-dataset training
            'intrinsics': (fx, fy, cx, cy),  # NEW: for variable resolution training
        }

        if self.center_shift:
            out_dict['center_shift'] = center
            out_dict['valid_mask'] = valid_mask

        if self.handle_sky:
            out_dict['sky_mask'] = sky_mask
        else:
            # Add dummy sky_mask for consistency in mixed dataset training
            # This ensures all samples have the same dict keys for PyTorch collation
            out_dict['sky_mask'] = torch.zeros_like(pointcloud_tensor[0, :, :], dtype=torch.bool)

        if self.dataset_name in H5_SHARD_DATASET_NAMES:
            normal_tensor = torch.from_numpy(normal_np.copy()).permute(2, 0, 1).float()
            sky = out_dict['sky_mask']
            normal_mask = (normal_tensor.norm(dim=0) > 0.5) & ~sky
            if self.sky_normal_dome:
                # the sky dome of the point map, seen from its centre: the normal faces the camera along the pixel ray
                sky_rays = pointcloud_unnormalized[:, sky]
                normal_tensor[:, sky] = -sky_rays / sky_rays.norm(dim=0, keepdim=True)
                normal_mask = normal_mask | sky
            out_dict['normal'] = normal_tensor
            out_dict['normal_mask'] = normal_mask
        if self.load_instance:
            instance = torch.from_numpy(np.ascontiguousarray(instance_np)).long()
            background_supervised = bool(data_pair['instance_background_supervised'])
            if self.sky_instance_background and not background_supervised:
                # the unlabeled sky of an otherwise unsupervised background is pulled to black (instance_target -2)
                instance[(instance < 0) & out_dict['sky_mask']] = -2
            out_dict['instance'] = instance
            out_dict['instance_background_supervised'] = background_supervised
            # gen2seg curation, applied to the instance loss only (the geometry streams keep every sample): no crop
            # without a labeled instance (this also drops unannotated frames) and no scene with too few objects
            supervised = self.instance_loss and bool((instance_np >= 0).any())
            if self.instance_scene_objects is not None:
                scene = data_pair['reference'].split('/')[1]
                supervised = supervised and self.instance_scene_objects[scene] >= self.instance_min_scene_objects
            out_dict['instance_supervised'] = supervised

        if self.split in ['val', 'test']:
            out_dict['pointcloud_unnormalized'] = pointcloud_unnormalized

        # Timing logging
        if self.debug_timing:
            t_end = time.perf_counter()
            self._timing_accum['file_read'] += t_depth_convert - t_start
            self._timing_accum['depth_convert'] += t_file_read - t_depth_convert
            self._timing_accum['augment'] += t_end - t_file_read
            self._timing_accum['total'] += t_end - t_start
            self._timing_count += 1

            if self._timing_count % 100 == 0:
                n = self._timing_count
                print(f"[File Timing] samples={n}, "
                      f"file_read={self._timing_accum['file_read']/n*1000:.2f}ms, "
                      f"depth_convert={self._timing_accum['depth_convert']/n*1000:.2f}ms, "
                      f"augment={self._timing_accum['augment']/n*1000:.2f}ms, "
                      f"total={self._timing_accum['total']/n*1000:.2f}ms")

        return out_dict
