# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Images with class-agnostic instance ids but no geometry or camera (EntitySeg and ADE20K).
Ids are a uint16 PNG: 0 is background (-1), 1..N are instances (0..N-1) and 65535 marks pixels left out of the instance
loss (-3).

A sample carries the keys of ImageDepthIntrinsicsDataset, so both mix in one batch: the geometry is zero under empty
masks, and NaN intrinsics mark an unknown camera. The augmentation follows the geometry sets."""
import random
from io import BytesIO
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ColorJitter, v2

from dataloader.point_transforms import random_frame_crop
from dataloader.h5_shards import read_indexed_bytes


class ImageInstanceSegmentationDataset(Dataset):
    def __init__(self, root_dir, dataset_name, target_height, crop_size, instance_background_supervised,
                 resize_probability, flip_probability, color_jitter, appearance_probability, blur_kernel,
                 blur_sigma, jpeg_quality):
        self.dataset_name = dataset_name
        self.samples = []
        for shard_path in sorted(Path(root_dir).glob('*.h5')):
            with h5py.File(shard_path, 'r') as shard:
                self.samples += [(str(shard_path), index) for index in range(shard.attrs['sample_count'])]
        self.target_height = target_height
        self.crop_size = crop_size
        self.instance_background_supervised = instance_background_supervised
        self.resize_probability = resize_probability
        self.flip_probability = flip_probability
        self.color_jitter = ColorJitter(**color_jitter)
        self.appearance = v2.Compose([
            v2.RandomApply([v2.GaussianBlur(kernel_size=blur_kernel, sigma=blur_sigma)], p=appearance_probability),
            v2.RandomAutocontrast(p=appearance_probability),
            v2.RandomEqualize(p=appearance_probability),
            v2.RandomApply([v2.RandomChannelPermutation()], p=appearance_probability),
            v2.RandomApply([v2.JPEG(quality=jpeg_quality)], p=appearance_probability),
            v2.RandomGrayscale(p=appearance_probability),
        ])

    def empty_sample(self, height, width):
        """The geometry keys of ImageDepthIntrinsicsDataset, empty, for a height x width sample."""
        geometry = torch.zeros(3, height, width)
        mask = torch.zeros(height, width, dtype=torch.bool)
        return {'pointcloud': geometry, 'pointcloud_unnormalized': geometry, 'normal': geometry, 'valid_mask': mask,
                'sky_mask': mask, 'normal_mask': mask, 'center_shift': torch.zeros(3, 1, 1),
                'scale_factor': torch.tensor(0.0), 'intrinsics': (float('nan'),) * 4,
                'dataset_name': self.dataset_name,
                'instance_background_supervised': self.instance_background_supervised}

    def __len__(self):
        return len(self.samples)

    def resized_size(self, height, width):
        """Height and width after the resize policy of ImageDepthIntrinsicsDataset (height-only mode)."""
        shrink = height > self.target_height and random.random() < self.resize_probability
        scale = self.target_height / height if height < self.target_height or shrink else 1.0
        scale = max(scale, self.crop_size / height, self.crop_size / width)
        return round(height * scale), round(width * scale)

    def __getitem__(self, index):
        index, frame = (index[0], index[1:]) if isinstance(index, tuple) else (index, None)
        shard_path, sample_index = self.samples[index]
        image = Image.open(BytesIO(read_indexed_bytes(shard_path, 'rgb', sample_index).tobytes())).convert('RGB')
        ids = cv2.imdecode(read_indexed_bytes(shard_path, 'instance_segmentation', sample_index), cv2.IMREAD_UNCHANGED)
        instance = torch.from_numpy(ids.astype(np.int32))[None, None].float()
        if frame is not None:
            # the crop of dataloader/aspect_batches.py, resized to the frame size of its batch
            height, width, side = frame
            top, left, crop_height, crop_width = random_frame_crop(image.height, image.width, height, width, side)
            crop = TF.to_tensor(image)[None, :, top:top + crop_height, left:left + crop_width]
            crop = F.interpolate(crop, size=(height, width), mode='bilinear', align_corners=False, antialias=True)
            image = TF.to_pil_image(crop[0].clamp(0, 1))
            instance = F.interpolate(instance[..., top:top + crop_height, left:left + crop_width],
                                     size=(height, width), mode='nearest')
        else:
            size = self.resized_size(image.height, image.width)
            if size != (image.height, image.width):
                resized = F.interpolate(TF.to_tensor(image)[None], size=size, mode='bilinear', align_corners=False)
                image = TF.to_pil_image(resized[0])
                instance = F.interpolate(instance, size=size, mode='nearest')
        image = self.appearance(self.color_jitter(image))
        if random.random() < self.flip_probability:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            instance = instance.flip(-1)
        if frame is None:
            top = random.randint(0, image.height - self.crop_size)
            left = random.randint(0, image.width - self.crop_size)
            image = image.crop((left, top, left + self.crop_size, top + self.crop_size))
            instance = instance[..., top:top + self.crop_size, left:left + self.crop_size]
        instance = instance[0, 0].long()
        instance = torch.where(instance == np.iinfo(np.uint16).max, -3, instance - 1)
        return {**self.empty_sample(image.height, image.width), 'image': TF.to_tensor(image), 'instance': instance,
                'instance_supervised': bool((instance >= 0).any())}
