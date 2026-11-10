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

# Simple unaligned depth/point metrics, kept for reference.
# The reported numbers come from metrics/compute_metrics.py.
import torch


def rel_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    rel = (torch.abs(pred - gt) / (gt + eps)).mean()
    return rel.item()


def delta1_depth(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    delta1 = (torch.maximum(gt / pred, pred / gt) < 1.25).float().mean()
    return delta1.item()


def rel_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / (dist_gt + eps)).mean()
    return rel.item()


def delta1_point(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-6):
    dist_pred = torch.norm(pred, dim=-1)
    dist_gt = torch.norm(gt, dim=-1)
    dist_err = torch.norm(pred - gt, dim=-1)

    delta1 = (dist_err < 0.25 * torch.minimum(dist_gt, dist_pred)).float().mean()
    return delta1.item()


def rel_point_local(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
    dist_err = torch.norm(pred - gt, dim=-1)
    rel = (dist_err / diameter).mean()
    return rel.item()


def delta1_point_local(pred: torch.Tensor, gt: torch.Tensor, diameter: torch.Tensor):
    dist_err = torch.norm(pred - gt, dim=-1)
    delta1 = (dist_err < 0.25 * diameter).float().mean()
    return delta1.item()


def boundary_f1(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor, radius: int = 1):
    neighbor_x, neight_y = torch.meshgrid(
        torch.linspace(-radius, radius, 2 * radius + 1, device=pred.device),
        torch.linspace(-radius, radius, 2 * radius + 1, device=pred.device),
        indexing='xy'
    )
    neighbor_mask = (neighbor_x ** 2 + neight_y ** 2) <= radius ** 2 + 1e-5

    pred_window = utils3d.pt.sliding_window_2d(pred, window_size=2 * radius + 1, stride=1, dim=(-2, -1))                 # [H, W, 2*R+1, 2*R+1]
    gt_window = utils3d.pt.sliding_window_2d(gt, window_size=2 * radius + 1, stride=1, dim=(-2, -1))                     # [H, W, 2*R+1, 2*R+1]
    mask_window = neighbor_mask & utils3d.pt.sliding_window_2d(mask, window_size=2 * radius + 1, stride=1, dim=(-2, -1)) # [H, W, 2*R+1, 2*R+1]

    pred_rel = pred_window / pred[radius:-radius, radius:-radius, None, None]
    gt_rel = gt_window / gt[radius:-radius, radius:-radius, None, None]
    valid = mask[radius:-radius, radius:-radius, None, None] & mask_window
    
    f1_list = []
    w_list = t_list = torch.linspace(0.05, 0.25, 10).tolist()

    for t in t_list:
        pred_label = pred_rel > 1 + t
        gt_label = gt_rel > 1 + t
        TP = (pred_label & gt_label & valid).float().sum()
        precision = TP / (gt_label & valid).float().sum().clamp_min(1e-12)
        recall = TP / (pred_label & valid).float().sum().clamp_min(1e-12)
        f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
        f1_list.append(f1.item())
    
    f1_avg = sum(w * f1 for w, f1 in zip(w_list, f1_list)) / sum(w_list)
    return f1_avg


