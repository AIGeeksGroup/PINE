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
MixedImageDepthIntrinsicsDataset - Multi-dataset wrapper for mixed training

Supports combining multiple datasets (SceneNet, TartanAir, etc.) with:
- Per-dataset configurations (max_depth, sky handling, etc.)
- Weighted sampling to control dataset mixing ratios
- YAML-based configuration files
- Unified preprocessing (resize, crop) while preserving dataset-specific features

Usage:
    from dataloader.mixed_dataset import MixedImageDepthIntrinsicsDataset

    dataset = MixedImageDepthIntrinsicsDataset(
        config_path='dataloader/configs/pretrain_scenenet_tartanair.yaml',
        split='train',
        crop_size=256,
        # ... other shared args
    )
"""

import bisect
import os
import yaml
from torch.utils.data import ConcatDataset, WeightedRandomSampler
from dataloader.img_depth_intrinsics import ImageDepthIntrinsicsDataset
from dataloader.img_instance_segmentation import ImageInstanceSegmentationDataset


def load_dataset_config(config_path):
    """
    Load dataset configuration from YAML file.

    Args:
        config_path (str): Path to YAML config file

    Returns:
        dict: Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Dataset config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if 'datasets' not in config or not config['datasets']:
        raise ValueError(f"Config file must contain 'datasets' list: {config_path}")

    return config


class MixedImageDepthIntrinsicsDataset(ConcatDataset):
    """
    Multi-dataset wrapper that combines multiple ImageDepthIntrinsicsDataset instances.

    Features:
    - Loads config from YAML file
    - Creates separate dataset instance for each dataset with its own config
    - Supports weighted sampling via WeightedRandomSampler
    - Preserves per-dataset settings (max_depth, sky handling, etc.)

    The config file should have structure:
        datasets:
          - name: scenenet
            data_path: path/to/scenenet
            max_depth: 10.0
            weight: 0.5
            handle_sky: false
            ...
          - name: tartanair
            data_path: path/to/tartanair
            max_depth: 100.0
            weight: 0.5
            handle_sky: true
            ...
        img_size: 256
        resize_height: 256
        samples_per_epoch: 160000   # optional; the default is the sum of the dataset sizes

    samples_per_epoch fixes the epoch length. Set it when a dataset holds a different part of its data on each node
    (each node packs its own share into its /tmp), so that every process draws the same number of batches.
    """

    def __init__(
        self,
        config_path,
        split='train',
        crop_size=None,
        # Shared args (will be overridden by per-dataset config if specified)
        center_shift=False,
        normalize_by_mean=False,
        num_overfit_samples=None,
        num_dataset_duplicates=None,
        subsample_scenes=None,
        center_shift_z_only=False,
        more_img_aug=False,
        stronger_img_aug=False,
        debug_timing=False,
        use_gpu_conversion=False,
        # Test scenes handling
        num_test_scenes=2,
        test_scenes_file=None,
        samples_per_scene=None,
        # Weight overrides
        weight_overrides=None,
    ):
        """
        Args:
            config_path (str): Path to YAML config file
            split (str): 'train', 'test', 'val', or 'all'
            crop_size (int): Final crop size (overridden by config's img_size if present)
            weight_overrides (list or None): List of weights to override config weights (e.g., [0.7, 0.3])
            ... (other args are shared defaults, can be overridden per-dataset in config)
        """
        # Load config
        self.config = load_dataset_config(config_path)
        self.config_path = config_path

        # Validate and apply weight overrides if provided
        if weight_overrides is not None:
            if len(weight_overrides) != len(self.config['datasets']):
                raise ValueError(
                    f"Number of weight overrides ({len(weight_overrides)}) must match "
                    f"number of datasets ({len(self.config['datasets'])}) in config"
                )

            # Normalize weights to sum to 1.0
            total_weight = sum(weight_overrides)
            if abs(total_weight - 1.0) > 1e-6:
                print(f"[MixedDataset] Normalizing weights from sum={total_weight:.4f} to 1.0")
                weight_overrides = [w / total_weight for w in weight_overrides]

            # Override config weights
            for i, ds_config in enumerate(self.config['datasets']):
                original_weight = ds_config.get('weight', 1.0)
                ds_config['weight'] = weight_overrides[i]
                print(f"[MixedDataset] Overriding {ds_config['name']} weight: {original_weight} -> {weight_overrides[i]:.3f}")

        self.weight_overrides = weight_overrides

        # Shared settings from the config; the YAML's img_size wins over the passed crop_size.
        shared_img_size = self.config.get('img_size', crop_size)
        if shared_img_size is not None:
            crop_size = shared_img_size
        shared_resize_height = self.config.get('resize_height', None)
        shared_resize_width = self.config.get('resize_width', None)

        # High-res finetuning mode settings (from config)
        finetune_highres_mode = self.config.get('finetune_highres_mode', False)
        finetune_target_height = self.config.get('finetune_target_height', 512)
        finetune_target_width = self.config.get('finetune_target_width', None)  # None = height-only mode
        finetune_target_crop = self.config.get('finetune_target_crop', 512)

        if finetune_highres_mode:
            print(f"[MixedDataset] High-res finetuning mode enabled: target_height={finetune_target_height}, target_width={finetune_target_width}, crop={finetune_target_crop}")

        # Create individual datasets
        datasets = []
        dataset_weights = []
        dataset_info = []

        for ds_config in self.config['datasets']:
            dataset_name = ds_config['name']
            data_path = ds_config['data_path']
            weight = ds_config.get('weight', 1.0)

            # Per-dataset overrides (with shared defaults)
            max_depth = ds_config.get('max_depth', 10.0)
            handle_sky = ds_config.get('handle_sky', False)
            use_sky_dome = ds_config.get('use_sky_dome', False)
            sky_loss_weight = ds_config.get('sky_loss_weight', 0.0)
            sky_far_plane_value = ds_config.get('sky_far_plane_value', 3.0)
            sky_normal_dome = ds_config.get('sky_normal_dome', False)
            sky_instance_background = ds_config.get('sky_instance_background', False)
            compute_scale_factor_only_valid = ds_config.get('compute_scale_factor_only_valid', False)
            compute_scale_factor_use_std = ds_config.get('compute_scale_factor_use_std', False)
            compute_scale_factor_use_percentile = ds_config.get('compute_scale_factor_use_percentile', False)
            clamp_max_depth = ds_config.get('clamp_max_depth', None)
            remove_outliers = ds_config.get('remove_outliers', False)
            outlier_threshold = ds_config.get('outlier_threshold', 5.0)
            scale_factor_augment = ds_config.get('scale_factor_augment', False)
            scale_factor_augment_range = ds_config.get('scale_factor_augment_range', (0.8, 1.2))
            load_instance = ds_config.get('load_instance', False)
            instance_counts = ds_config.get('instance_counts', None)
            instance_min_scene_objects = ds_config.get('instance_min_scene_objects', 0)
            instance_loss = ds_config.get('instance_loss', True)

            # Per-dataset minimum image size (for filtering undersized samples)
            min_height = ds_config.get('min_height', None)
            min_width = ds_config.get('min_width', None)

            # Per-dataset resize height (defaults to shared setting)
            dataset_resize_height = ds_config.get('resize_height', shared_resize_height)

            # Per-dataset resize width (defaults to shared setting)
            dataset_resize_width = ds_config.get('resize_width', shared_resize_width)

            # Per-dataset num_test_scenes (defaults to shared setting)
            dataset_num_test_scenes = ds_config.get('num_test_scenes', num_test_scenes)

            # No dataset ships a curated holdout list. Pass through whatever the caller
            # supplied (None by default), in which case each loader falls back to its
            # `num_test_scenes` auto-split. A path given here is resolved relative to
            # dataloader/ by ImageDepthIntrinsicsDataset. Note that a list whose names do
            # not match any scene still counts as a "manual" split and so disables the
            # auto-split entirely, so only pass one that matches the dataset.
            dataset_test_scenes_file = test_scenes_file

            if ds_config.get('task') == 'instance_segmentation':
                dataset = ImageInstanceSegmentationDataset(
                    root_dir=data_path, dataset_name=dataset_name, target_height=finetune_target_height,
                    crop_size=finetune_target_crop,
                    instance_background_supervised=ds_config['instance_background_supervised'],
                    **ds_config['augmentation'])
            else:
                dataset = ImageDepthIntrinsicsDataset(
                    root_dir=data_path,
                    dataset_name=dataset_name,
                    split=split,
                    num_test_scenes=dataset_num_test_scenes,
                    test_scenes_file=dataset_test_scenes_file,
                    samples_per_scene=samples_per_scene,
                    crop_size=crop_size,
                    resize_height=dataset_resize_height,
                    resize_width=dataset_resize_width,
                    center_shift=center_shift,
                    normalize_by_mean=normalize_by_mean,
                    num_overfit_samples=num_overfit_samples,
                    num_dataset_duplicates=num_dataset_duplicates,
                    max_depth=max_depth,
                    subsample_scenes=subsample_scenes,
                    center_shift_z_only=center_shift_z_only,
                    more_img_aug=more_img_aug,
                    stronger_img_aug=stronger_img_aug,
                    compute_scale_factor_only_valid=compute_scale_factor_only_valid,
                    compute_scale_factor_use_std=compute_scale_factor_use_std,
                    compute_scale_factor_use_percentile=compute_scale_factor_use_percentile,
                    clamp_max_depth=clamp_max_depth,
                    handle_sky=handle_sky,
                    use_sky_dome=use_sky_dome,
                    sky_far_plane_value=sky_far_plane_value,
                    sky_normal_dome=sky_normal_dome,
                    sky_instance_background=sky_instance_background,
                    debug_timing=debug_timing,
                    use_gpu_conversion=use_gpu_conversion,
                    remove_outliers=remove_outliers,
                    outlier_threshold=outlier_threshold,
                    min_height=min_height,
                    min_width=min_width,
                    scale_factor_augment=scale_factor_augment,
                    scale_factor_augment_range=scale_factor_augment_range,
                    # High-res finetuning mode
                    finetune_highres_mode=finetune_highres_mode,
                    finetune_target_height=finetune_target_height,
                    finetune_target_width=finetune_target_width,
                    finetune_target_crop=finetune_target_crop,
                    load_instance=load_instance,
                    instance_counts=instance_counts,
                    instance_min_scene_objects=instance_min_scene_objects,
                    instance_loss=instance_loss,
                )

            datasets.append(dataset)
            dataset_weights.append(weight)
            dataset_info.append({
                'name': dataset_name,
                'size': len(dataset),
                'weight': weight,
                'max_depth': max_depth,
            })

            print(f"[MixedDataset] Loaded {dataset_name}: {len(dataset)} samples, weight={weight}, max_depth={max_depth}")

        # Initialize ConcatDataset
        super().__init__(datasets)

        # Store metadata
        self.datasets_info = dataset_info
        self.dataset_weights = dataset_weights
        self.num_datasets = len(datasets)

        # Print summary
        total_samples = len(self)

        # Check for empty datasets
        if total_samples == 0:
            raise ValueError(
                f"Total samples is 0. Check dataset paths and configurations.\n"
                f"Config: {config_path}\n"
                f"Dataset info: {dataset_info}"
            )

        print(f"\n[MixedDataset] Total samples: {total_samples:,}")
        print(f"[MixedDataset] Config: {config_path}")
        print(f"[MixedDataset] Split: {split}")
        print(f"[MixedDataset] Crop size: {crop_size}")
        print(f"[MixedDataset] Resize height: {shared_resize_height}")
        print(f"\nDataset composition:")
        for info in dataset_info:
            size_percentage = (info['size'] / total_samples) * 100
            sampling_percentage = info['weight'] * 100
            print(f"  - {info['name']}: {info['size']:,} samples ({size_percentage:.1f}% of total)")
            print(f"      → Sampling weight: {info['weight']:.3f} ({sampling_percentage:.1f}% of training samples per epoch)")

    def __getitem__(self, idx):
        """idx: an index, or (index, height, width, side) from dataloader/aspect_batches.AspectBatchSampler."""
        if not isinstance(idx, tuple):
            return super().__getitem__(idx)
        dataset_index = bisect.bisect_right(self.cumulative_sizes, idx[0])
        start = self.cumulative_sizes[dataset_index - 1] if dataset_index else 0
        return self.datasets[dataset_index][(idx[0] - start, *idx[1:])]

    def create_weighted_sampler(self, num_samples=None):
        """
        Create a WeightedRandomSampler for this mixed dataset.

        This sampler ensures that samples are drawn according to the specified
        dataset weights, rather than proportional to dataset sizes.

        Args:
            num_samples (int, optional): Total number of samples to draw per epoch.
                If None, defaults to samples_per_epoch of the config, else to the total dataset size.

        Returns:
            torch.utils.data.WeightedRandomSampler
        """
        # Compute per-sample weights based on dataset membership
        sample_weights = []

        for dataset, weight in zip(self.datasets, self.dataset_weights):
            dataset_size = len(dataset)
            if dataset_size == 0:
                print(f"[MixedDataset] WARNING: skipping empty dataset (weight={weight}) in weighted sampler")
                continue
            # Each sample in this dataset gets the same weight (normalized by dataset size)
            # This ensures the dataset is sampled proportional to its weight, not its size
            per_sample_weight = weight / dataset_size
            sample_weights.extend([per_sample_weight] * dataset_size)

        # Normalize weights to sum to 1
        total_weight = sum(sample_weights)
        sample_weights = [w / total_weight for w in sample_weights]

        if num_samples is None:
            num_samples = self.config.get('samples_per_epoch', len(self))

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples,
            replacement=True,
        )

        print(f"[MixedDataset] Created WeightedRandomSampler with {num_samples} samples")

        return sampler

    def __repr__(self):
        info_strs = [f"{d['name']}({d['size']} samples, weight={d['weight']})"
                     for d in self.datasets_info]
        return f"MixedImageDepthIntrinsicsDataset({', '.join(info_strs)})"
