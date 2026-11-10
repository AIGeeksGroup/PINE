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

"""Paired image/point-map transforms and the point-map scale statistic.

Used by the training dataloader, ``dataloader/img_depth_intrinsics.py``.
"""

import random

import torch


# --- Custom Paired Transformation (Augmentation) ---

class RandomCropPaired:
    """
    Applies the exact same random crop to both the RGB image and the Point Cloud array.
    Used for the training set (augmentation).
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img_pil, pc_np):
        width, height = img_pil.size
        target_h, target_w = self.size

        if width < target_w or height < target_h:
            print(f"Warning: Requested crop size {self.size} is larger than image size ({height}, {width}). No cropping applied.")
            return img_pil, pc_np

        i = random.randint(0, height - target_h)
        j = random.randint(0, width - target_w)

        cropped_img = img_pil.crop((j, i, j + target_w, i + target_h))
        cropped_pc = pc_np[i:i + target_h, j:j + target_w, :]

        return cropped_img, cropped_pc

# --- Paired Transformation (Evaluation) ---

class CenterCropPaired:
    """
    Applies a deterministic center crop to both the RGB image and the Point Cloud array.
    Used for the test/validation set.
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img_pil, pc_np):
        width, height = img_pil.size
        target_h, target_w = self.size

        if width < target_w or height < target_h:
            print(f"Warning: Requested crop size {self.size} is larger than image size ({height}, {width}). No cropping applied.")
            return img_pil, pc_np

        i = (height - target_h) // 2
        j = (width - target_w) // 2

        cropped_img = img_pil.crop((j, i, j + target_w, i + target_h))
        cropped_pc = pc_np[i:i + target_h, j:j + target_w, :]

        return cropped_img, cropped_pc


class RandomCropPairedWithOffset:
    """
    Applies the exact same random crop to both the RGB image and the Point Cloud array.
    Returns the crop offset (x, y) for accurate intrinsics adjustment.
    Used for high-res finetuning mode where we need to adjust cx, cy based on crop position.
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img_pil, pc_np):
        width, height = img_pil.size
        target_h, target_w = self.size

        if width < target_w or height < target_h:
            print(f"Warning: Requested crop size {self.size} is larger than image size ({height}, {width}). No cropping applied.")
            return img_pil, pc_np, (0, 0)

        i = random.randint(0, height - target_h)  # y offset
        j = random.randint(0, width - target_w)   # x offset

        cropped_img = img_pil.crop((j, i, j + target_w, i + target_h))
        cropped_pc = pc_np[i:i + target_h, j:j + target_w, :]

        return cropped_img, cropped_pc, (j, i)  # Return (x_offset, y_offset)


def random_frame_crop(height, width, frame_height, frame_width, side):
    """(top, left, crop height, crop width) of a crop of the aspect ratio of the frame at a random place in a
    height x width image, side times the largest such crop (dataloader/aspect_batches.py)."""
    scale = side * min(height / frame_height, width / frame_width)
    crop_height, crop_width = round(frame_height * scale), round(frame_width * scale)
    return random.randint(0, height - crop_height), random.randint(0, width - crop_width), crop_height, crop_width


class CenterCropPairedWithOffset:
    """
    Applies a deterministic center crop to both the RGB image and the Point Cloud array.
    Returns the crop offset (x, y) for accurate intrinsics adjustment.
    Used for high-res finetuning mode validation/test sets.
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img_pil, pc_np):
        width, height = img_pil.size
        target_h, target_w = self.size

        if width < target_w or height < target_h:
            print(f"Warning: Requested crop size {self.size} is larger than image size ({height}, {width}). No cropping applied.")
            return img_pil, pc_np, (0, 0)

        i = (height - target_h) // 2  # y offset
        j = (width - target_w) // 2   # x offset

        cropped_img = img_pil.crop((j, i, j + target_w, i + target_h))
        cropped_pc = pc_np[i:i + target_h, j:j + target_w, :]

        return cropped_img, cropped_pc, (j, i)  # Return (x_offset, y_offset)


def compute_distance_statistic(
    point_cloud: torch.Tensor,
    valid_mask: torch.Tensor = None,
    use_mean: bool = True,
    use_std: bool = False,
    use_percentile: bool = False,
) -> torch.Tensor:
    """
    Computes the median (or mean) Euclidean distance of valid points.

    Args:
        point_cloud: Tensor of shape (3, H, W).
        valid_mask: Optional boolean Tensor of shape (1, H, W) or (H, W).
                    True indicates a valid point to include.
        use_mean: If True, computes mean instead of median.
        use_percentile: If True, computes 99th percentile instead of mean/median.

    Returns:
        A scalar tensor.
    """
    # 1. Flatten the point cloud -> (3, N)
    #    points_flat shape: [3, H*W]
    points_flat = point_cloud.reshape(3, -1)

    # 2. Compute distances for all points -> (N,)
    distances = torch.linalg.norm(points_flat, ord=2, dim=0)

    # 3. Apply the mask if provided
    if valid_mask is not None:
        # Flatten mask to match distances shape: (N,)
        mask_flat = valid_mask.reshape(-1)
        
        # Filter distances using boolean indexing
        distances = distances[mask_flat]

    # 4. Handle Edge Case: Empty mask (no valid points)
    if distances.numel() == 0:
        return torch.tensor(0.0, device=point_cloud.device)

    # 5. Compute Statistic
    if use_percentile:
        scale_factor = torch.quantile(distances, 0.98)
    elif use_mean:
        scale_factor = torch.mean(distances)
    else:
        scale_factor = torch.median(distances)

    if use_std:
        # FIX: Check if we have enough points to compute std
        if distances.numel() > 1:
            std = torch.std(distances)
        else:
            # If N=1, variance/std is 0
            std = torch.tensor(0.0, device=point_cloud.device)
            
        scale_factor = scale_factor + 2 * std

    return scale_factor
    
