# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""DSINE benchmarks: legacy camera-facing crops or untouched full-frame benchmark normals and masks."""
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

NORMAL_SUFFIX = {'nyuv2': '_normal.png', 'scannet': '_normal.png', 'sintel': '_normal.exr', 'ibims': '_normal.exr'}
SPLIT_FILE = {'nyuv2': 'test.txt', 'scannet': 'test.txt', 'sintel': 'sintel.txt', 'ibims': 'ibims.txt'}
BENCHMARK_TO_CAMERA_FACING = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
PNG_NORMAL_SCALE = 127.5


def read_benchmark_normal(path, camera_facing=True):
    """Decode a DSINE normal file and its official mask, optionally converting to PointDiT's camera-facing frame."""
    raw = cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB).astype(np.float32)
    if path.suffix == '.exr':
        normal, valid = raw, np.linalg.norm(raw, axis=2) > 0.5
    else:
        normal, valid = raw / PNG_NORMAL_SCALE - 1.0, raw.sum(axis=2) > 0
    if camera_facing:
        normal = normal * BENCHMARK_TO_CAMERA_FACING
    return normal, valid


class EvalNormalDataset(Dataset):
    """One DSINE benchmark set. Returns image [3, H, W] in [0, 1], normal [3, H, W], valid_mask [H, W], intrinsics 3x3."""

    def __init__(self, data_root, split_root, dataset_name, crop_size=512, resize_height=512, max_samples=None,
                 full_frame=False, patch_size=16):
        self.dataset_name = dataset_name
        self.dataset_path = Path(data_root) / dataset_name
        self.crop_size = crop_size
        self.resize_height = resize_height
        self.full_frame = full_frame
        self.patch_size = patch_size
        with open(Path(split_root) / dataset_name / 'split' / SPLIT_FILE[dataset_name]) as f:
            self.sample_ids = [line.strip().replace('_img.png', '') for line in f if line.strip()]
        if max_samples is not None and len(self.sample_ids) > max_samples:
            stride = len(self.sample_ids) // max_samples
            self.sample_ids = self.sample_ids[::stride][:max_samples]

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, index):
        sample_id = self.sample_ids[index]
        stem = self.dataset_path / sample_id
        image = Image.open(f'{stem}_img.png').convert('RGB')
        normal, valid = read_benchmark_normal(
            Path(f'{stem}{NORMAL_SUFFIX[self.dataset_name]}'), camera_facing=not self.full_frame,
        )
        intrinsics = np.load(f'{stem}_intrins.npy').astype(np.float32)
        width, height = image.size
        if self.full_frame:
            if self.resize_height is not None:
                scale = self.resize_height / min(width, height)
                new_w = round(width * scale / self.patch_size) * self.patch_size
                new_h = round(height * scale / self.patch_size) * self.patch_size
                image = image.resize((new_w, new_h), Image.BILINEAR)
                intrinsics[0] *= new_w / width
                intrinsics[1] *= new_h / height
            return self.format_sample(image, normal, valid, intrinsics, sample_id, (height, width))
        scale = self.resize_height / min(width, height)
        new_w, new_h = round(width * scale), round(height * scale)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        normal = cv2.resize(normal, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        valid = cv2.resize(valid.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        intrinsics[0] *= new_w / width
        intrinsics[1] *= new_h / height
        top, left = (new_h - self.crop_size) // 2, (new_w - self.crop_size) // 2
        image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
        normal = normal[top:top + self.crop_size, left:left + self.crop_size]
        valid = valid[top:top + self.crop_size, left:left + self.crop_size]
        intrinsics[0, 2] -= left
        intrinsics[1, 2] -= top
        return self.format_sample(image, normal, valid, intrinsics, sample_id, (height, width))

    def format_sample(self, image, normal, valid, intrinsics, sample_id, original_size):
        return {'image': torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1),
                'normal': torch.from_numpy(normal).permute(2, 0, 1), 'valid_mask': torch.from_numpy(valid),
                'intrinsics': torch.from_numpy(intrinsics), 'sample_id': sample_id, 'dataset_name': self.dataset_name,
                'original_size': original_size}
