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

# The evaluation metric suite (affine-invariant point-map and depth metrics).
# Adapted from MoGe (https://github.com/microsoft/MoGe), MIT License.
from typing import *
from numbers import Number

import torch


from metrics.alignment import (
    align_points_scale_xyz_shift,
    align_points_xyz_shift,
    align_affine_lstsq,
    align_depth_scale,
    align_depth_affine,
    align_points_scale,
)


from metrics.masked_resize import masked_nearest_resize

from metrics.metrics import (
    rel_depth,
    delta1_depth,
    rel_point,
    delta1_point,
    rel_point_local,
    delta1_point_local,
    boundary_f1,
)

from metrics.boundary_metrics import SI_boundary_F1


DEPTH_RANGE_BINS = {'near': (0.0, 1/3), 'medium': (1/3, 2/3), 'far': (2/3, 1.0)}


def compute_depth_range_metrics(pred_depth, gt_depth, pred_points, gt_points, mask):
    """Compute metrics stratified by GT depth percentile ranges (near/mid/far)."""
    gt_depth_valid = gt_depth[mask]
    if gt_depth_valid.numel() < 30:
        return {}

    sorted_depths = gt_depth_valid.sort().values
    n = sorted_depths.shape[0]

    range_metrics = {}
    for bin_name, (lo_pct, hi_pct) in DEPTH_RANGE_BINS.items():
        lo_val = sorted_depths[int(lo_pct * (n - 1))]
        hi_val = sorted_depths[min(int(hi_pct * (n - 1)), n - 1)]
        if hi_pct >= 1.0:
            hi_val = sorted_depths[-1] + 1e-6

        bin_mask = mask & (gt_depth >= lo_val) & (gt_depth < hi_val)
        if bin_mask.sum() < 10:
            continue

        bin_metrics = {
            'rel_depth': rel_depth(pred_depth[bin_mask], gt_depth[bin_mask]),
            'delta1_depth': delta1_depth(pred_depth[bin_mask], gt_depth[bin_mask]),
        }
        if pred_points is not None and gt_points is not None:
            bin_metrics['rel_point'] = rel_point(pred_points[bin_mask], gt_points[bin_mask])
            bin_metrics['delta1_point'] = delta1_point(pred_points[bin_mask], gt_points[bin_mask])
        range_metrics[bin_name] = bin_metrics

    return range_metrics


def compute_metrics(
    pred: Dict[str, torch.Tensor],
    gt: Dict[str, torch.Tensor],
    vis: bool = False,
    compute_si_boundary: bool = False,
    compute_depth_range: bool = False,
) -> Tuple[Dict[str, Dict[str, Number]], Dict[str, torch.Tensor]]:
    """
    A unified function to compute metrics for different types of predictions and ground truths.
    
    #### Supported keys in pred:
        - `disparity_affine_invariant`: disparity map predicted by a depth estimator with scale and shift invariant. 
        - `depth_scale_invariant`: depth map predicted by a depth estimator with scale invariant. 
        - `depth_affine_invariant`: depth map predicted by a depth estimator with scale and shift invariant. 
        - `depth_metric`: depth map predicted by a depth estimator with no scale or shift. 
        - `points_scale_invariant`: point map predicted by a point estimator with scale invariant. 
        - `points_affine_invariant`: point map predicted by a point estimator with scale and xyz shift invariant. 
        - `points_metric`: point map predicted by a point estimator with no scale or shift. 
        - `intrinsics`: normalized camera intrinsics matrix.

    #### Required keys in gt:
        - `depth`: depth map ground truth (in metric units if `depth_metric` is used)
        - `points`: point map ground truth in camera coordinates.
        - `mask`: mask indicating valid pixels in the ground truth.
        - `intrinsics`: normalized ground-truth camera intrinsics matrix.
        - `is_metric`: whether the depth is in metric units.
    """
    metrics = {}
    misc = {}
    
    mask = gt['depth_mask']
    gt_depth = gt['depth']
    gt_points = gt['points']

    height, width = mask.shape[-2:]
    lr_mask, lr_index = masked_nearest_resize(mask=mask, size=(64, 64), return_index=True)

    only_depth = not any('point' in k for k in pred)
    pred_depth_aligned, pred_points_aligned = None, None
    affine_points_scale, affine_points_shift = None, None

    # Metric depth
    if 'depth_metric' in pred and 'is_metric' in gt and gt['is_metric']:
        pred_depth, gt_depth = pred['depth_metric'], gt['depth']
        metrics['depth_metric'] = {
            'rel': rel_depth(pred_depth[mask], gt_depth[mask]),
            'delta1': delta1_depth(pred_depth[mask], gt_depth[mask])
        }

        if pred_depth_aligned is None:
            pred_depth_aligned = pred_depth

    # Scale-invariant depth
    if 'depth_scale_invariant' in pred:
        pred_depth_scale_invariant = pred['depth_scale_invariant']
    elif 'depth_metric' in pred:
        pred_depth_scale_invariant = pred['depth_metric']
    else:
        pred_depth_scale_invariant = None

    if pred_depth_scale_invariant is not None:
        pred_depth = pred_depth_scale_invariant

        pred_depth_lr_masked, gt_depth_lr_masked = pred_depth[lr_index][lr_mask], gt_depth[lr_index][lr_mask]
        scale = align_depth_scale(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
        pred_depth = pred_depth * scale
    
        metrics['depth_scale_invariant'] = {
            'rel': rel_depth(pred_depth[mask], gt_depth[mask]),
            'delta1': delta1_depth(pred_depth[mask], gt_depth[mask])
        }

        if pred_depth_aligned is None:
            pred_depth_aligned = pred_depth

    # Affine-invariant depth
    if 'depth_affine_invariant' in pred:
        pred_depth_affine_invariant = pred['depth_affine_invariant']
    elif 'depth_scale_invariant' in pred:
        pred_depth_affine_invariant = pred['depth_scale_invariant']
    elif 'depth_metric' in pred:
        pred_depth_affine_invariant = pred['depth_metric']
    else:
        pred_depth_affine_invariant = None

    if pred_depth_affine_invariant is not None:
        pred_depth = pred_depth_affine_invariant

        pred_depth_lr_masked, gt_depth_lr_masked = pred_depth[lr_index][lr_mask], gt_depth[lr_index][lr_mask]
        scale, shift = align_depth_affine(pred_depth_lr_masked, gt_depth_lr_masked, 1 / gt_depth_lr_masked)
        pred_depth = pred_depth * scale + shift

        metrics['depth_affine_invariant'] = {
            'rel': rel_depth(pred_depth[mask], gt_depth[mask]),
            'delta1': delta1_depth(pred_depth[mask], gt_depth[mask])
        }

        if pred_depth_aligned is None:
            pred_depth_aligned = pred_depth

    # Affine-invariant LOG-depth (e.g. PPD, whose raw output is normalized log(d+1)).
    # Align in log space then exponentiate, matching the model's native parameterization
    # and PPD's official recover_metric_depth_ransac. Metrics are still computed/reported
    # in linear depth under the standard 'depth_affine_invariant' key.
    if 'depth_logaffine_invariant' in pred:
        pred_log = pred['depth_logaffine_invariant']
        gt_log = torch.log(gt_depth + 1.0)

        pred_log_lr_masked, gt_log_lr_masked = pred_log[lr_index][lr_mask], gt_log[lr_index][lr_mask]
        scale, shift = align_depth_affine(pred_log_lr_masked, gt_log_lr_masked, 1 / gt_depth[lr_index][lr_mask])
        pred_depth = torch.exp(pred_log * scale + shift) - 1.0

        metrics['depth_affine_invariant'] = {
            'rel': rel_depth(pred_depth[mask], gt_depth[mask]),
            'delta1': delta1_depth(pred_depth[mask], gt_depth[mask])
        }

        if pred_depth_aligned is None:
            pred_depth_aligned = pred_depth

    # Affine-invariant disparity
    if 'disparity_affine_invariant' in pred:
        pred_disparity_affine_invariant = pred['disparity_affine_invariant']
    elif 'depth_scale_invariant' in pred:
        pred_disparity_affine_invariant = 1 / pred['depth_scale_invariant']
    elif 'depth_metric' in pred:
        pred_disparity_affine_invariant = 1 / pred['depth_metric']
    else:
        pred_disparity_affine_invariant = None
        
    if pred_disparity_affine_invariant is not None:
        pred_disp = pred_disparity_affine_invariant

        scale, shift = align_affine_lstsq(pred_disp[mask], 1 / gt_depth[mask])
        pred_disp = pred_disp * scale + shift

        # NOTE: The alignment is done on the disparity map could introduce extreme outliers at disparities close to 0.
        #       Therefore we clamp the disparities by minimum ground truth disparity.
        pred_depth = 1 / pred_disp.clamp_min(1 / gt_depth[mask].max().item())

        metrics['disparity_affine_invariant'] = {
            'rel': rel_depth(pred_depth[mask], gt_depth[mask]),
            'delta1': delta1_depth(pred_depth[mask], gt_depth[mask])
        }

        if pred_depth_aligned is None:
            pred_depth_aligned = 1 / pred_disp.clamp_min(1e-6)

    # Metric points
    if 'points_metric' in pred and gt['is_metric']:
        pred_points = pred['points_metric']

        pred_points_lr_masked, gt_points_lr_masked = pred_points[lr_index][lr_mask], gt_points[lr_index][lr_mask]
        shift = align_points_xyz_shift(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
        pred_points = pred_points + shift

        metrics['points_metric'] = {
            'rel': rel_point(pred_points[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points[mask], gt_points[mask])
        }

        if pred_points_aligned is None:
            pred_points_aligned = pred['points_metric']

    # Scale-invariant points (in camera space)
    if 'points_scale_invariant' in pred:
        pred_points_scale_invariant = pred['points_scale_invariant']
    elif 'points_metric' in pred:
        pred_points_scale_invariant = pred['points_metric']
    else:
        pred_points_scale_invariant = None
        
    if pred_points_scale_invariant is not None:
        pred_points = pred_points_scale_invariant

        pred_points_lr_masked, gt_points_lr_masked = pred_points_scale_invariant[lr_index][lr_mask], gt_points[lr_index][lr_mask]
        scale = align_points_scale(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
        pred_points = pred_points * scale

        metrics['points_scale_invariant'] = {
            'rel': rel_point(pred_points[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points[mask], gt_points[mask])
        }

        if vis and pred_points_aligned is None:
            pred_points_aligned = pred['points_scale_invariant'] * scale
    
    # Affine-invariant points
    if 'points_affine_invariant' in pred:
        pred_points_affine_invariant = pred['points_affine_invariant']
    elif 'points_scale_invariant' in pred:
        pred_points_affine_invariant = pred['points_scale_invariant']
    elif 'points_metric' in pred:
        pred_points_affine_invariant = pred['points_metric']
    else:
        pred_points_affine_invariant = None

    if pred_points_affine_invariant is not None:
        pred_points = pred_points_affine_invariant

        pred_points_lr_masked, gt_points_lr_masked = pred_points[lr_index][lr_mask], gt_points[lr_index][lr_mask]
        scale, shift = align_points_scale_xyz_shift(pred_points_lr_masked, gt_points_lr_masked, 1 / gt_points_lr_masked.norm(dim=-1))
        pred_points = pred_points * scale + shift

        metrics['points_affine_invariant'] = {
            'rel': rel_point(pred_points[mask], gt_points[mask]),
            'delta1': delta1_point(pred_points[mask], gt_points[mask])
        }

        if vis and pred_points_aligned is None:
            pred_points_aligned = pred['points_affine_invariant'] * scale + shift
            affine_points_scale = scale
            affine_points_shift = shift

    # Local points
    if 'segmentation_mask' in gt and 'points' in gt and any('points' in k for k in pred.keys()):
        pred_points = next(pred[k] for k in pred.keys() if 'points' in k)
        gt_points = gt['points']
        segmentation_mask = gt['segmentation_mask']
        segmentation_labels = gt['segmentation_labels']
        segmentation_mask_lr =  segmentation_mask[lr_index]
        local_points_metrics = []
        for _, seg_id in segmentation_labels.items():
            valid_mask = (segmentation_mask == seg_id) & mask
            
            pred_points_masked = pred_points[valid_mask]
            gt_points_masked = gt_points[valid_mask]

            valid_mask_lr = (segmentation_mask_lr == seg_id) & lr_mask
            if valid_mask_lr.sum().item() < 10:
                continue
            pred_points_masked_lr = pred_points[lr_index][valid_mask_lr]
            gt_points_masked_lr = gt_points[lr_index][valid_mask_lr]
            diameter = (gt_points_masked.max(dim=0).values - gt_points_masked.min(dim=0).values).max()
            scale, shift = align_points_scale_xyz_shift(pred_points_masked_lr, gt_points_masked_lr, 1 / diameter.expand(gt_points_masked_lr.shape[0]))
            pred_points_masked = pred_points_masked * scale + shift

            local_points_metrics.append({
                'rel': rel_point_local(pred_points_masked, gt_points_masked, diameter),
                'delta1': delta1_point_local(pred_points_masked, gt_points_masked, diameter),
            })
        
        metrics['local_points'] = key_average(local_points_metrics)

    # FOV. NOTE: If there is no random augmentation applied to the input images, all GT FOV are generallly the same. 
    #            Fair evaluation of FOV requires random augmentation.
    if 'intrinsics' in pred and 'intrinsics' in gt:
        pred_intrinsics = pred['intrinsics']
        gt_intrinsics = gt['intrinsics']
        pred_fov_x, pred_fov_y = intrinsics_to_fov(pred_intrinsics)
        gt_fov_x, gt_fov_y = intrinsics_to_fov(gt_intrinsics)
        metrics['fov_x'] = {
            'mae': torch.rad2deg(pred_fov_x - gt_fov_x).abs().mean().item(),
            'deviation': torch.rad2deg(pred_fov_x - gt_fov_x).item(),
        }

    # Depth-range stratified metrics
    if compute_depth_range and pred_depth_aligned is not None:
        range_metrics = compute_depth_range_metrics(
            pred_depth_aligned, gt_depth,
            pred_points_aligned, gt_points,
            mask,
        )
        if range_metrics:
            metrics['depth_range'] = range_metrics

    # Boundary F1
    if pred_depth_aligned is not None and 'has_sharp_boundary' in gt and gt['has_sharp_boundary']:
        metrics['boundary'] = {
            'radius1_f1': boundary_f1(pred_depth_aligned, gt_depth, mask, radius=1),
            'radius2_f1': boundary_f1(pred_depth_aligned, gt_depth, mask, radius=2),
            'radius3_f1': boundary_f1(pred_depth_aligned, gt_depth, mask, radius=3),
        }

    # SI Boundary F1 (computed for specified datasets via CLI)
    if compute_si_boundary and pred_depth_aligned is not None:
        pred_np = pred_depth_aligned.squeeze().cpu().numpy()
        gt_np = gt_depth.squeeze().cpu().numpy()
        metrics['si_boundary_f1'] = SI_boundary_F1(pred_np, gt_np)

        # Add boundary visualization
        if vis:
            # Imported lazily: util.viz_boundary imports
            # metrics.boundary_metrics, so a module-level import here
            # would be a circular import.
            from util.viz_boundary import viz_boundary_comparison
            misc['boundary_viz'] = viz_boundary_comparison(pred_np, gt_np, threshold=1.10)

    if vis:
        if pred_points_aligned is not None:
            misc['pred_points'] = pred_points_aligned
        if pred_depth_aligned is not None:
            misc['pred_depth'] = pred_depth_aligned
        if affine_points_scale is not None:
            misc['affine_points_scale'] = affine_points_scale
        if affine_points_shift is not None:
            misc['affine_points_shift'] = affine_points_shift

    return metrics, misc


def _test():
    pred = {
        'depth_affine_invariant': torch.rand(1, 256, 256),
        'points_affine_invariant': torch.rand(1, 256, 256, 3),
    }
    gt = {
        'depth': torch.rand(1, 256, 256) * 10 + 0.1,
        'points': torch.rand(1, 256, 256, 3) * 10,
        'depth_mask': torch.rand(1, 256, 256) > 0.2,
    }
    metrics, misc = compute_metrics(pred, gt, vis=False)

    print(metrics)


if __name__ == "__main__":
    _test()



