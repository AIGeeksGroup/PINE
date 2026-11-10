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
Evaluation dataloader for depth data in MoGe format.

Reads logarithmic 16-bit PNG depth images with near/far metadata,
and returns raw (unnormalized) pointmaps for evaluation.
"""

import io
import json
import os
from pathlib import Path
from typing import Union, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def read_depth(path: Union[str, os.PathLike, io.IOBase]) -> np.ndarray:
    """
    Read a depth image in MoGe logarithmic 16-bit PNG format.
    
    The depth is stored in logarithmic scale:
    - 0: NaN (unknown/invalid)
    - 1-65534: logarithmically encoded depth values
    - 65535: Infinity
    
    Returns:
        float32 depth array of shape (H, W) in meters.
    """
    if isinstance(path, (str, os.PathLike)):
        data = Path(path).read_bytes()
    else:
        data = path.read()
    
    pil_image = Image.open(io.BytesIO(data))
    near = float(pil_image.info.get('near'))
    far = float(pil_image.info.get('far'))
    depth = np.array(pil_image)
    
    # Mark special values
    mask_nan, mask_inf = depth == 0, depth == 65535
    
    # Decode logarithmic scale: depth = near^(1-t) * far^t
    t = (depth.astype(np.float32) - 1) / 65533
    depth = near ** (1 - t) * far ** t
    
    # Legacy support for depth units
    if 'unit' in pil_image.info:
        unit = float(pil_image.info.get('unit'))
        depth = depth * unit
    
    # Apply special value masks
    depth[mask_nan] = np.nan
    depth[mask_inf] = np.inf
    
    return depth


def depth_to_pointmap(depth: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    """
    Convert depth map to 3D point map using pinhole camera model.

    Args:
        depth: (H, W) depth values in meters
        intrinsics: (3, 3) camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

    Returns:
        (H, W, 3) point map with (X, Y, Z) coordinates
    """
    H, W = depth.shape
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]

    # Create pixel coordinate grids
    u = np.arange(W, dtype=np.float32)
    v = np.arange(H, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    # Initialize output arrays with nan
    X = np.full_like(depth, np.nan, dtype=np.float32)
    Y = np.full_like(depth, np.nan, dtype=np.float32)
    Z = depth.copy()

    # Create mask for finite depth values to avoid RuntimeWarning on nan/inf
    finite_mask = np.isfinite(depth)

    # Backproject to 3D (only compute on finite values to avoid warnings)
    X[finite_mask] = (u[finite_mask] - cx) * depth[finite_mask] / fx
    Y[finite_mask] = (v[finite_mask] - cy) * depth[finite_mask] / fy

    pointmap = np.stack([X, Y, Z], axis=-1)
    return pointmap


class EvalDepthDataset(Dataset):
    """
    Evaluation dataset for depth prediction.
    
    Reads images and ground truth depth in MoGe format:
    - depth.png: 16-bit PNG with logarithmic encoding and near/far metadata
    - image.jpg: RGB image
    - meta.json: {"intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]}
    
    Uses index.txt for sample discovery, or glob fallback.
    
    Args:
        data_root: Root directory containing eval datasets
        dataset_name: Name of dataset (e.g., "NYUv2", "KITTI")
        crop_size: Optional crop size (int or tuple). If provided, applies center crop.
    
    Returns dict with:
        - image: (3, H, W) float tensor, normalized to [0, 1]
        - pointcloud: (3, H, W) raw metric coordinates
        - valid_mask: (H, W) bool tensor, True where depth is finite
        - intrinsics: (3, 3) camera matrix (adjusted for crop if applied)
        - sample_id: string identifier
    """
    
    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        crop_size: int = None,
        max_samples: int = None,
        resize_height: int = None,
    ):
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.dataset_path = self.data_root / dataset_name

        # Resize settings (applied before crop)
        self.resize_height = resize_height
        if resize_height is not None:
            print(f"[EvalDepthDataset] Will resize shortest side to {resize_height}")

        # Center crop settings
        if crop_size is not None:
            if isinstance(crop_size, int):
                self.crop_size = (crop_size, crop_size)
            else:
                self.crop_size = crop_size
            print(f"[EvalDepthDataset] Using center crop of size {self.crop_size}")
        else:
            self.crop_size = None
        
        # Try to read sample IDs from index.txt, fallback to glob discovery
        index_file = self.dataset_path / "index.txt"
        if index_file.exists():
            with open(index_file, 'r') as f:
                self.sample_ids = [line.strip() for line in f if line.strip()]
            print(f"[EvalDepthDataset] Loaded {len(self.sample_ids)} samples from {dataset_name}/index.txt")
        else:
            # Glob for directories containing depth.png
            self.sample_ids = []
            for depth_file in sorted(self.dataset_path.rglob("depth.png")):
                # Get relative path from dataset_path to the sample directory
                sample_dir = depth_file.parent
                sample_id = str(sample_dir.relative_to(self.dataset_path))
                self.sample_ids.append(sample_id)
            
            if len(self.sample_ids) == 0:
                raise FileNotFoundError(
                    f"No samples found in {self.dataset_path}. "
                    f"Expected either index.txt or directories with depth.png"
                )
            print(f"[EvalDepthDataset] Discovered {len(self.sample_ids)} samples from {dataset_name} via glob")
        
        # Subsample if max_samples is specified
        if max_samples is not None and max_samples > 0 and len(self.sample_ids) > max_samples:
            # Uniformly subsample
            stride = len(self.sample_ids) // max_samples
            self.sample_ids = self.sample_ids[::stride][:max_samples]
            print(f"[EvalDepthDataset] Subsampled to {len(self.sample_ids)} samples (max_samples={max_samples})")
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def __getitem__(self, idx: int) -> dict:
        sample_id = self.sample_ids[idx]
        sample_dir = self.dataset_path / sample_id
        
        # Load image
        image_path = sample_dir / "image.jpg"
        if not image_path.exists():
            image_path = sample_dir / "image.png"
        image_pil = Image.open(image_path).convert("RGB")
        W, H = image_pil.size
        
        # Load depth
        depth_path = sample_dir / "depth.png"
        depth = read_depth(depth_path)  # (H, W)
        
        # Load intrinsics from meta.json
        meta_path = sample_dir / "meta.json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        intrinsics = np.array(meta["intrinsics"], dtype=np.float32)  # (3, 3)
        
        # normalized and scale by image dimensions
        intrinsics[0, 0] *= W  # fx
        intrinsics[1, 1] *= H  # fy
        intrinsics[0, 2] *= W  # cx
        intrinsics[1, 2] *= H  # cy

        # Apply resize if specified (before crop)
        if self.resize_height is not None:
            # Resize shortest side to resize_height, maintain aspect ratio
            if H < W:
                # Height is shorter
                new_h = self.resize_height
                new_w = int(W * (new_h / H))
            else:
                # Width is shorter or equal
                new_w = self.resize_height
                new_h = int(H * (new_w / W))

            # Resize image with BILINEAR interpolation
            image_pil = image_pil.resize((new_w, new_h), Image.BILINEAR)

            # Resize depth with NEAREST interpolation to preserve depth values
            depth_pil = Image.fromarray(depth.astype(np.float32))
            depth_pil = depth_pil.resize((new_w, new_h), Image.NEAREST)
            depth = np.array(depth_pil)

            # Scale intrinsics
            scale_w = new_w / W
            scale_h = new_h / H
            intrinsics[0, 0] *= scale_w  # fx
            intrinsics[1, 1] *= scale_h  # fy
            intrinsics[0, 2] *= scale_w  # cx
            intrinsics[1, 2] *= scale_h  # cy

            W, H = new_w, new_h

        # Apply center crop if specified
        if self.crop_size is not None:
            target_h, target_w = self.crop_size
            
            if W >= target_w and H >= target_h:
                # Calculate crop offsets
                i = (H - target_h) // 2  # top
                j = (W - target_w) // 2  # left
                
                # Crop image
                image_pil = image_pil.crop((j, i, j + target_w, i + target_h))
                
                # Crop depth
                depth = depth[i:i + target_h, j:j + target_w]
                
                # Adjust intrinsics (cx, cy shift by crop offset)
                intrinsics[0, 2] -= j  # cx
                intrinsics[1, 2] -= i  # cy
                
                W, H = target_w, target_h
            else:
                print(f"Warning: Image size ({H}, {W}) smaller than crop size {self.crop_size}. No crop applied.")
        
        # Convert image to numpy
        image = np.array(image_pil, dtype=np.float32) / 255.0  # (H, W, 3)
        
        # Convert depth to pointmap
        pointmap = depth_to_pointmap(depth, intrinsics)  # (H, W, 3)
        
        # Create valid mask (finite depth values)
        valid_mask = np.isfinite(depth)
        
        # Convert to tensors
        image_tensor = torch.from_numpy(image).permute(2, 0, 1)  # (3, H, W)
        pointcloud_tensor = torch.from_numpy(pointmap).permute(2, 0, 1)  # (3, H, W)
        valid_mask_tensor = torch.from_numpy(valid_mask)  # (H, W)
        intrinsics_tensor = torch.from_numpy(intrinsics)  # (3, 3)
        
        return {
            'image': image_tensor,
            'pointcloud': pointcloud_tensor,
            'valid_mask': valid_mask_tensor,
            'intrinsics': intrinsics_tensor,
            'sample_id': sample_id,
            'dataset_name': self.dataset_name,
        }


class MultiEvalDepthDataset(Dataset):
    """
    Wrapper that combines multiple EvalDepthDataset instances.

    This allows evaluating on multiple datasets sequentially while tracking
    which dataset each sample comes from.

    Args:
        data_root: Root directory containing eval datasets
        dataset_names: List of dataset names (e.g., ["NYUv2", "KITTI"])
        crop_size: Optional crop size (passed to each dataset)
        max_samples: Optional max samples per dataset
    """

    def __init__(
        self,
        data_root: str,
        dataset_names: List[str],
        crop_size: int = None,
        max_samples: int = None,
        resize_height: int = None,
    ):
        self.datasets = []
        self.dataset_names = dataset_names
        self.cumulative_sizes = [0]

        for name in dataset_names:
            try:
                dataset = EvalDepthDataset(
                    data_root=data_root,
                    dataset_name=name,
                    crop_size=crop_size,
                    max_samples=max_samples,
                    resize_height=resize_height,
                )
                self.datasets.append(dataset)
                self.cumulative_sizes.append(self.cumulative_sizes[-1] + len(dataset))
                print(f"[MultiEvalDepthDataset] Loaded {name}: {len(dataset)} samples")
            except Exception as e:
                print(f"[MultiEvalDepthDataset] Failed to load {name}: {e}")

        if len(self.datasets) == 0:
            raise ValueError(f"No datasets could be loaded from {dataset_names}")

        print(f"[MultiEvalDepthDataset] Total: {self.cumulative_sizes[-1]} samples across {len(self.datasets)} datasets")

    def __len__(self) -> int:
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> dict:
        # Find which dataset this index belongs to
        dataset_idx = 0
        for i, cumsum in enumerate(self.cumulative_sizes[1:]):
            if idx < cumsum:
                dataset_idx = i
                break

        # Get the local index within that dataset
        local_idx = idx - self.cumulative_sizes[dataset_idx]

        # Get the sample (already includes 'dataset_name' from EvalDepthDataset)
        return self.datasets[dataset_idx][local_idx]


class WildImagesDataset(Dataset):
    """
    Inference-only dataset for arbitrary images without ground truth depth.
    Loads images from a flat or nested directory. Returns only image tensors.
    """
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    def __init__(
        self,
        images_dir: str,
        dataset_name: str = 'wild',
        crop_size=None,
        resize_height: int = None,
        max_samples: int = None,
    ):
        self.images_dir = Path(images_dir)
        self.dataset_name = dataset_name
        self.resize_height = resize_height

        if crop_size is not None:
            self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else tuple(crop_size)
        else:
            self.crop_size = None

        # Discover all images
        image_paths = []
        for ext in self.IMAGE_EXTENSIONS:
            image_paths.extend(self.images_dir.rglob(f'*{ext}'))
        self.image_paths = sorted(set(image_paths))

        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.images_dir}")
        print(f"[WildImagesDataset] Found {len(self.image_paths)} images in {images_dir}")

        if max_samples is not None and max_samples > 0 and len(self.image_paths) > max_samples:
            stride = len(self.image_paths) // max_samples
            self.image_paths = self.image_paths[::stride][:max_samples]
            print(f"[WildImagesDataset] Subsampled to {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __repr__(self):
        return f"WildImagesDataset({self.dataset_name}, {len(self.image_paths)} images)"

    def __getitem__(self, idx: int) -> dict:
        image_path = self.image_paths[idx]
        image_pil = Image.open(image_path).convert("RGB")

        # Resize shortest side
        if self.resize_height is not None:
            W, H = image_pil.size
            if H <= W:
                new_H = self.resize_height
                new_W = int(round(W * new_H / H))
            else:
                new_W = self.resize_height
                new_H = int(round(H * new_W / W))
            image_pil = image_pil.resize((new_W, new_H), Image.BILINEAR)

        # Center crop
        if self.crop_size is not None:
            W, H = image_pil.size
            crop_h, crop_w = self.crop_size
            left = (W - crop_w) // 2
            top = (H - crop_h) // 2
            image_pil = image_pil.crop((left, top, left + crop_w, top + crop_h))

        image = torch.from_numpy(np.array(image_pil, dtype=np.float32) / 255.0).permute(2, 0, 1)  # [3, H, W]

        try:
            sample_id = str(image_path.relative_to(self.images_dir))
        except ValueError:
            sample_id = image_path.name

        return {
            'image': image,
            'sample_id': sample_id,
            'dataset_name': self.dataset_name,
        }

