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

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _sinkhorn_log(log_K, n_iter):
    """Sinkhorn-Knopp normalization in log domain on a square cost matrix."""
    log_M = log_K
    for _ in range(int(n_iter)):
        log_M = log_M - torch.logsumexp(log_M, dim=-1, keepdim=True)
        log_M = log_M - torch.logsumexp(log_M, dim=-2, keepdim=True)
    return log_M


def window_matched_l1_loss(pred, gt, valid, window=5, iters=5, tau=0.1, masked_cost=1.0e6):
    """Marigold V2 WindowMatchedL1Loss (marigoldv2/loss/loss.py:29): L1 under a soft Sinkhorn matching of the
    pixels inside every non-overlapping window x window block, averaged over blocks with valid pairs."""
    pred = pred.float()
    gt = gt.float()
    mask = valid[:, :1].float()
    K = window
    bsz, _, height, width = pred.shape
    pad_h = (K - height % K) % K
    pad_w = (K - width % K) % K
    if pad_h or pad_w:
        pred = F.pad(pred, (0, pad_w, 0, pad_h))
        gt = F.pad(gt, (0, pad_w, 0, pad_h))
        mask = F.pad(mask, (0, pad_w, 0, pad_h))
    padded_h, padded_w = pred.shape[-2:]
    h_blocks, w_blocks = padded_h // K, padded_w // K
    n_block = K * K
    n_blocks = h_blocks * w_blocks

    def to_blocks(x):
        return (
            x.reshape(bsz, x.shape[1], h_blocks, K, w_blocks, K)
            .permute(0, 2, 4, 3, 5, 1)
            .reshape(bsz, n_blocks, n_block, x.shape[1])
        )

    pred_blocks = to_blocks(pred)
    gt_blocks = to_blocks(gt)
    mask_blocks = to_blocks(mask)

    pred_i = pred_blocks.unsqueeze(3)
    gt_j = gt_blocks.unsqueeze(2)
    real_cost = (gt_j - pred_i).abs().mean(dim=-1)

    mask_i = mask_blocks[..., 0].unsqueeze(-1)
    mask_j = mask_blocks[..., 0].unsqueeze(-2)
    valid_pair = mask_i * mask_j
    cost_for_sinkhorn = real_cost * valid_pair + masked_cost * (1.0 - valid_pair)

    log_M = _sinkhorn_log(-cost_for_sinkhorn / tau, n_iter=iters)
    M = log_M.exp()

    block_loss = (M * real_cost * valid_pair).sum(dim=(-2, -1))
    block_M_valid = (M * valid_pair).sum(dim=(-2, -1))
    block_valid = (block_M_valid > 1.0e-3).float()
    denom = block_valid.sum().clamp_min(1.0)
    block_avg_cost = block_loss / block_M_valid.clamp_min(1.0e-3)
    return (block_avg_cost * block_valid).sum() / denom


def rel_point_loss(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Relative point loss - normalizes error by distance from origin.
    Directly optimizes rel_point_metric.

    Args:
        pred: Predicted points [B, 3, H, W] or [B, H, W, 3]
        gt: Ground truth points [B, 3, H, W] or [B, H, W, 3]
        eps: Small constant for numerical stability

    Returns:
        Loss tensor [B, H, W]
    """
    # Handle [B, 3, H, W] -> [B, H, W, 3]
    if pred.shape[1] == 3 and pred.dim() == 4:
        pred = pred.permute(0, 2, 3, 1)
        gt = gt.permute(0, 2, 3, 1)

    dist_gt = torch.norm(gt, dim=-1)  # [B, H, W]
    dist_err = (pred - gt).abs().sum(dim=-1)  # L1 norm
    return dist_err / (dist_gt + eps)


def log_depth_gradient_matching_loss(pred_points, gt_points, valid, depth_offset=None, scales=4, eps=1e-3):
    """Multi-scale gradient matching (MiDaS, Ranftl et al. 2020, K=4 scales) on the log-depth error.

    MiDaS applies it to the scale-and-shift-aligned disparity error; here the error is log(z_pred) - log(z_gt) with
    its per-image median removed, which aligns the scale.  Depths are taken in the un-centered normalized frame by
    adding back `depth_offset` ([B, 1, 1, 1]: the loader's z center divided by its scale factor).  Per scale the mean
    absolute gradient over valid pixel pairs is summed.
    """
    offset = depth_offset.float() if depth_offset is not None else 0.0
    z_pred = (pred_points[:, 2:3].float() + offset).clamp_min(eps)
    z_gt_raw = gt_points[:, 2:3].float() + offset
    valid = valid[:, :1] & (z_gt_raw > eps)
    diff = torch.log(z_pred) - torch.log(z_gt_raw.clamp_min(eps))
    medians = torch.stack([d[m].median() if m.any() else d.new_zeros(()) for d, m in zip(diff, valid)])
    diff = torch.where(valid, diff - medians.view(-1, 1, 1, 1), torch.zeros_like(diff))
    mask = valid.float()
    total = diff.new_zeros(())
    for scale in range(scales):
        if scale > 0:
            pooled_mask = torch.nn.functional.avg_pool2d(mask, 2)
            diff = torch.nn.functional.avg_pool2d(diff * mask, 2) / pooled_mask.clamp_min(1e-6)
            mask = (pooled_mask > 0).float()
        grad_x = (diff[..., :, 1:] - diff[..., :, :-1]).abs()
        mask_x = mask[..., :, 1:] * mask[..., :, :-1]
        grad_y = (diff[..., 1:, :] - diff[..., :-1, :]).abs()
        mask_y = mask[..., 1:, :] * mask[..., :-1, :]
        total = total + (grad_x * mask_x).sum() / mask_x.sum().clamp_min(1.0) + (grad_y * mask_y).sum() / mask_y.sum().clamp_min(1.0)
    return total


# The instance stream follows gen2seg: the loss groups pixels by their target 3-vector and treats (0, 0, 0) as
# background. Instance ids (-1 = background) are encoded as (id + 1, 0, 0), which gives every instance its own vector
# and the background zero; -2 marks background that is pulled to black even when no_bg leaves the rest of the
# background out (the sky), encoded as (0, -1, 0); -3 marks pixels left out of the loss, encoded as (0, 0, -1).
# gen2seg scales the prediction to 0..255 before its loss with the whole-batch shift and span of its training/train.py,
# and before prompting with the per-image shift and span of gen2seg_sd_pipeline.decode_prediction.
def instance_target(instance_ids):
    """[B, H, W] int ids with -1 as background, -2 as background pulled to black whatever no_bg says and -3 as pixels
    outside the loss -> [B, 3, H, W] float target for InstanceSegmentationLoss."""
    target = torch.zeros(instance_ids.shape[0], 3, *instance_ids.shape[1:], device=instance_ids.device)
    target[:, 0] = (instance_ids + 1).clamp_min(0).float()
    target[:, 1] = -(instance_ids == -2).float()
    target[:, 2] = -(instance_ids == -3).float()
    return target


def gen2seg_training_scale(prediction):
    """training/train.py: (x + |min|) / (|max| + |min| + 1e-5) * 255 over the whole batch."""
    min_value = prediction.min().abs()
    max_value = prediction.max().abs()
    return (prediction + min_value) / (max_value + min_value + 1e-5) * 255.0


def gen2seg_inference_scale(prediction):
    """gen2seg_sd_pipeline.decode_prediction on one image: shift by |min|, divide by |max| + |min| of the shifted map."""
    prediction = prediction + prediction.min().abs()
    return prediction / (prediction.max().abs() + prediction.min().abs()) * 255.0


class InstanceSegmentationLoss(nn.Module):
    """gen2seg instance-segmentation loss (De Brabandere discriminative clustering) on a 3-channel map: per GT
    instance, pull its predicted pixels to their mean (intra variance) and push every other pixel away from that mean
    (inter separation), plus a mean-level repulsion between instances; background is pulled to black unless no_bg.
    Background marked (0, -1, 0) (instance_target, the sky) is pulled to black whatever no_bg says. Pixels marked
    (0, 0, -1) leave the loss: no pull, no push and no share in any mean or count. Prediction and target are in
    [0, value_scale] (gen2seg tunes at 255)."""
    def __init__(
        self,
        separation_strength=300.0,
        mean_strength=300.0,
        separation_weight=10.0,
        huber_delta=1.0,
        mean_epsilon=1.0,
        separation_chunk_size=8,
    ):
        super().__init__()
        self.separation_strength = separation_strength
        self.mean_strength = mean_strength
        self.separation_weight = separation_weight
        self.huber_delta = huber_delta
        self.mean_epsilon = mean_epsilon
        self.separation_chunk_size = separation_chunk_size

    def forward(self, prediction, target, no_bg):
        losses = [self._image_loss(prediction[i], target[i], no_bg[i]) for i in range(prediction.shape[0])]
        return torch.stack(losses).mean()

    def _chunk_separation(self, pixels, chunk_means, inverse_indices, chunk_indices, instance_counts):
        pixel_count = pixels.shape[0]
        differences = pixels[None] - chunk_means[:, None]
        squared_distances = differences.square().sum(dim=2)
        own_pixels = inverse_indices[None] == chunk_indices[:, None]
        separation_values = self.separation_strength / (1.0 + squared_distances)
        separation_values = separation_values.masked_fill(own_pixels, 0.0)
        other_counts = pixel_count - instance_counts[chunk_indices]
        separation_means = separation_values.sum(dim=1) / other_counts
        separation_weights = self.separation_weight / instance_counts[chunk_indices].sqrt()
        return (separation_weights * separation_means).sum()

    def _image_loss(self, prediction, target, no_bg):
        pixels = prediction.float().permute(1, 2, 0).reshape(-1, 3)
        instance_ids = target.float().permute(1, 2, 0).reshape(-1, 3)
        kept = instance_ids[:, 2].ge(0)
        pixels, instance_ids = pixels[kept], instance_ids[kept]
        instances, inverse_indices, instance_counts = torch.unique(
            instance_ids,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )
        instance_sums = torch.zeros_like(instances)
        instance_sums.index_add_(0, inverse_indices, pixels)
        instance_means = instance_sums / instance_counts[:, None]
        plain_background = instances.eq(0).all(dim=1)
        background_instances = plain_background | instances[:, 1].lt(0)
        included_instances = torch.ones_like(background_instances)
        if no_bg:
            included_instances = ~plain_background
        included_count = int(included_instances.sum())
        loss = pixels.mean() * 0.0
        if included_count == 0:
            return loss

        zero_means = torch.zeros_like(instance_means)
        clustering_means = torch.where(
            background_instances[:, None],
            zero_means,
            instance_means,
        )
        pixel_means = clustering_means[inverse_indices]
        pull_per_channel = F.huber_loss(
            pixels,
            pixel_means,
            reduction="none",
            delta=self.huber_delta,
        )
        pull_per_pixel = pull_per_channel.mean(dim=1)
        pull_sums = torch.zeros_like(instance_counts, dtype=pull_per_pixel.dtype)
        pull_sums.index_add_(0, inverse_indices, pull_per_pixel)
        pull_means = pull_sums / instance_counts
        loss = loss + pull_means[included_instances].sum()

        pixel_count = pixels.shape[0]
        foreground_instances = ~background_instances
        separable_instances = foreground_instances & (instance_counts < pixel_count)
        separation_indices = separable_instances.nonzero().flatten()
        for start in range(0, separation_indices.numel(), self.separation_chunk_size):
            chunk_indices = separation_indices[start:start + self.separation_chunk_size]
            chunk_means = instance_means[chunk_indices]
            # recomputed in backward: the [chunk, pixels] distance maps are never stored, so memory no longer grows
            # with the number of instances in the batch (the loss and its gradients are unchanged)
            separation = checkpoint(
                self._chunk_separation, pixels, chunk_means, inverse_indices, chunk_indices, instance_counts,
                use_reentrant=False,
            )
            loss = loss + separation

        included_means = clustering_means[included_instances]
        if included_count > 1:
            mean_differences = included_means[:, None] - included_means[None]
            pairwise_squared_distances = mean_differences.square().sum(dim=2)
            row_indices, column_indices = torch.triu_indices(
                included_count,
                included_count,
                offset=1,
                device=pixels.device,
            )
            pairwise_squared_distances = pairwise_squared_distances[
                row_indices,
                column_indices,
            ]
            mean_repulsion = self.mean_strength / (
                pairwise_squared_distances + self.mean_epsilon
            )
            loss = loss + mean_repulsion.mean()
        loss = loss / included_count
        return loss
