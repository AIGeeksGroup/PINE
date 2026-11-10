# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Training sets packed as HDF5 shards with a camera sidecar: Hypersim, Virtual KITTI 2, UrbanSyn and Structured3D (see
DATASETS.md and the scripts/train_prepare_*_h5.py packers)."""

import os
from functools import lru_cache
from io import BytesIO
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')  # UrbanSyn depth is float EXR
import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

H5_SHARD_DATASET_NAMES = ('hypersim_h5', 'vkitti_h5', 'urbansyn_h5', 'structured3d_h5')
CENTIMETERS_PER_METER = 100.0
millimeters_per_meter = 1000.0
NORMAL_PNG_SCALE = 127.5
MINIMUM_NORMAL_LENGTH = 1e-6
VKITTI_BENCHMARK_HEIGHT_WIDTH = (352, 1216)


@lru_cache(maxsize=None)
def open_shard(shard_path):
    return h5py.File(shard_path, 'r')


def read_indexed_bytes(shard_path, name, sample_index):
    shard = open_shard(shard_path)
    offsets = shard[f'{name}_offsets']
    return shard[f'{name}_bytes'][int(offsets[sample_index]):int(offsets[sample_index + 1])]


def load_intrinsics_by_sample_key(dataset_name):
    """(fx, fy, cx, cy) per sample key from the camera sidecar of this dataset."""
    sidecar_path = Path(os.environ['CAMERA_METADATA_ROOT']) / f"{dataset_name.removesuffix('_h5')}.h5"
    with h5py.File(sidecar_path, 'r') as sidecar:
        sample_keys = sidecar['sample_key'][:]
        intrinsic_matrices = np.linalg.inv(sidecar['camera_rays_from_pixel'][:])
    return {
        sample_key.decode(): (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2]))
        for sample_key, K in zip(sample_keys, intrinsic_matrices)
    }


def find_h5_shards(root_dir, dataset_name):
    """One lazy entry per packed shard over the samples covered by the camera sidecar."""
    intrinsics_by_key = load_intrinsics_by_sample_key(dataset_name)
    entries = []
    for shard_path in sorted(Path(root_dir).glob('*.h5')):
        with h5py.File(shard_path, 'r') as shard:
            sample_keys = [sample_key.decode() for sample_key in shard['canonical_reference'][:]]
        sample_indices = [index for index, sample_key in enumerate(sample_keys) if sample_key in intrinsics_by_key]
        entries.append({
            'lazy_load': True,
            'dataset': 'h5_shards',
            'shard': str(shard_path),
            'sample_count': len(sample_indices),
            'sample_indices': sample_indices,
            'intrinsics': [intrinsics_by_key[sample_keys[index]] for index in sample_indices],
        })
    return entries


def hypersim_z_depth(shard_path, sample_index, intrinsics):
    """Ray distance to z-depth through the pinhole ray lengths."""
    fx, fy, cx, cy = intrinsics
    ray_distance = np.asarray(open_shard(shard_path)['ray_depth'][sample_index], dtype=np.float32)
    height, width = ray_distance.shape
    horizontal = (np.arange(width, dtype=np.float32) - cx) / fx
    vertical = (np.arange(height, dtype=np.float32) - cy) / fy
    ray_length = np.sqrt(horizontal[None, :] ** 2 + vertical[:, None] ** 2 + 1.0)
    return ray_distance / ray_length


def unit_normals(normal, valid):
    """Unit-length normals on valid pixels, zeros elsewhere."""
    length = np.linalg.norm(normal, axis=-1, keepdims=True)
    return np.where(valid[..., None], normal / np.maximum(length, MINIMUM_NORMAL_LENGTH), 0.0).astype(np.float32)


def decode_normal_png(png_bytes):
    """[H, W, 3] normal map from an RGB PNG (or lossless WebP) that stores (n + 1) * 127.5. All-zero pixels carry no
    normal."""
    normal = cv2.cvtColor(cv2.imdecode(png_bytes, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return unit_normals(normal.astype(np.float32) / NORMAL_PNG_SCALE - 1.0, np.any(normal != 0, axis=-1))


def shard_normal(dataset_name, shard_path, sample_index):
    """Stored camera-frame normal map [H, W, 3] (x right, y down, z forward, up to sign), zero where undefined."""
    if dataset_name == 'hypersim_h5':
        normal = np.asarray(open_shard(shard_path)['normal'][sample_index], dtype=np.float32)
        normal[..., 0] *= -1.0
        length = np.linalg.norm(np.nan_to_num(normal), axis=-1)
        valid = np.isfinite(normal).all(axis=-1) & (length > MINIMUM_NORMAL_LENGTH)
        return unit_normals(np.nan_to_num(normal), valid)
    normal = decode_normal_png(read_indexed_bytes(shard_path, 'normal', sample_index))
    if dataset_name == 'vkitti_h5':
        normal = vkitti_benchmark_crop(normal)
    return normal


def shard_instance(dataset_name, shard_path, sample_index):
    """[H, W] int32 instance ids with -1 as background, and whether that background is supervised (gen2seg).

    Hypersim stores int16 ids with -1 on unlabeled pixels; gen2seg paints those black and pulls them to (0, 0, 0), so
    its background is supervised. vKITTI stores a palette PNG with 0 for everything but the labeled cars; gen2seg sets
    no_bg for it (the background holds unlabeled objects), so its background is not supervised. UrbanSyn and
    Structured3D store a uint16 PNG with 0 as background, supervised as their shard attribute says: not for UrbanSyn
    (unlabeled static classes), yes for Structured3D (walls, floor and ceiling)."""
    if dataset_name == 'hypersim_h5':
        return np.asarray(open_shard(shard_path)['instance'][sample_index], dtype=np.int32), True
    if dataset_name == 'vkitti_h5':
        # a palette PNG whose indices are the ids; cv2 would decode the palette colours instead
        png = BytesIO(np.asarray(read_indexed_bytes(shard_path, 'instance_segmentation', sample_index)).tobytes())
        ids = vkitti_benchmark_crop(np.asarray(Image.open(png))).astype(np.int32)
        return np.where(ids == 0, -1, ids), False
    if dataset_name in ('urbansyn_h5', 'structured3d_h5'):
        ids = cv2.imdecode(read_indexed_bytes(shard_path, 'instance_segmentation', sample_index), cv2.IMREAD_UNCHANGED)
        supervised = bool(open_shard(shard_path).attrs['instance_background_supervised'])
        return np.where(ids == 0, -1, ids.astype(np.int32)), supervised
    raise ValueError(f'No instance annotations for {dataset_name}')


def orient_toward_camera(normal, points):
    """Flip every normal whose dot product with its viewing ray is positive, so all normals face the camera."""
    facing_away = (normal * points).sum(axis=-1, keepdims=True) > 0
    return np.where(facing_away, -normal, normal)


def vkitti_benchmark_crop(array):
    """Bottom rows and centered columns of the KITTI benchmark crop."""
    crop_height, crop_width = VKITTI_BENCHMARK_HEIGHT_WIDTH
    top = array.shape[0] - crop_height
    left = (array.shape[1] - crop_width) // 2
    return array[top:top + crop_height, left:left + crop_width]


def vkitti_benchmark_crop_image(image):
    crop_height, crop_width = VKITTI_BENCHMARK_HEIGHT_WIDTH
    width, height = image.size
    top = height - crop_height
    left = (width - crop_width) // 2
    return image.crop((left, top, left + crop_width, top + crop_height))


def shard_data_pair(entry, frame_index, dataset_name, with_instance=False):
    """RGB bytes for PIL and a z-depth map in meters with invalid pixels at zero."""
    shard_path = entry['shard']
    sample_index = entry['sample_indices'][frame_index]
    intrinsics = entry['intrinsics'][frame_index]
    if dataset_name == 'hypersim_h5':
        rgb_bytes = read_indexed_bytes(shard_path, 'image', sample_index)
        depth = hypersim_z_depth(shard_path, sample_index, intrinsics)
    elif dataset_name == 'vkitti_h5':
        rgb_bytes = read_indexed_bytes(shard_path, 'rgb', sample_index)
        depth_centimeters = cv2.imdecode(read_indexed_bytes(shard_path, 'depth', sample_index), cv2.IMREAD_ANYDEPTH)
        depth = vkitti_benchmark_crop(depth_centimeters).astype(np.float32) / CENTIMETERS_PER_METER
    elif dataset_name == 'urbansyn_h5':
        rgb_bytes = read_indexed_bytes(shard_path, 'rgb', sample_index)
        depth = cv2.imdecode(read_indexed_bytes(shard_path, 'depth', sample_index), cv2.IMREAD_UNCHANGED)
        depth = depth.astype(np.float32)
    elif dataset_name == 'structured3d_h5':
        rgb_bytes = read_indexed_bytes(shard_path, 'rgb', sample_index)
        depth_millimeters = cv2.imdecode(read_indexed_bytes(shard_path, 'depth', sample_index), cv2.IMREAD_ANYDEPTH)
        depth = depth_millimeters.astype(np.float32) / millimeters_per_meter
    normal = shard_normal(dataset_name, shard_path, sample_index)
    depth[~np.isfinite(depth)] = 0.0
    pair = {
        'image': BytesIO(np.asarray(rgb_bytes).tobytes()),
        'depth': depth,
        'normal': normal,
        'intrinsics': intrinsics,
        'dataset': dataset_name,
    }
    if with_instance:
        instance, supervised = shard_instance(dataset_name, shard_path, sample_index)
        pair['instance'], pair['instance_background_supervised'] = instance, supervised
        if dataset_name == 'hypersim_h5':
            pair['reference'] = open_shard(shard_path)['canonical_reference'][sample_index].decode()
    return pair
