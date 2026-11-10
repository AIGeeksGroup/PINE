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

import numpy as np

from metrics.boundary_metrics import fgbg_depth_thinned, invert_depth


def viz_boundary_comparison(
    pred_depth: np.ndarray,
    gt_depth: np.ndarray,
    threshold: float = 1.10,
) -> np.ndarray:
    """Visualize predicted vs GT boundary edges.

    Args:
        pred_depth: Predicted depth map [H, W]
        gt_depth: Ground truth depth map [H, W]
        threshold: Depth ratio threshold for boundary detection (default 1.10)

    Returns:
        RGB image [H, W, 3] with:
        - Red: predicted boundaries only (false positives)
        - Green: GT boundaries only (false negatives)
        - Yellow: matched boundaries (true positives)
    """
    h, w = pred_depth.shape

    # Get boundary masks (4 directions)
    # left/right have shape (H, W-1), top/bottom have shape (H-1, W)
    pred_l, pred_t, pred_r, pred_b = fgbg_depth_thinned(invert_depth(pred_depth), threshold)
    gt_l, gt_t, gt_r, gt_b = fgbg_depth_thinned(invert_depth(gt_depth), threshold)

    # Pad each direction to full size (H, W)
    # Left/right: pad 1 column on right
    # Top/bottom: pad 1 row on bottom
    pred_l = np.pad(pred_l, ((0, 0), (0, 1)), mode='constant')
    pred_r = np.pad(pred_r, ((0, 0), (0, 1)), mode='constant')
    pred_t = np.pad(pred_t, ((0, 1), (0, 0)), mode='constant')
    pred_b = np.pad(pred_b, ((0, 1), (0, 0)), mode='constant')

    gt_l = np.pad(gt_l, ((0, 0), (0, 1)), mode='constant')
    gt_r = np.pad(gt_r, ((0, 0), (0, 1)), mode='constant')
    gt_t = np.pad(gt_t, ((0, 1), (0, 0)), mode='constant')
    gt_b = np.pad(gt_b, ((0, 1), (0, 0)), mode='constant')

    # Combine all directions
    pred_boundary = pred_l | pred_t | pred_r | pred_b
    gt_boundary = gt_l | gt_t | gt_r | gt_b

    # Create RGB visualization
    viz = np.zeros((h, w, 3), dtype=np.uint8)

    matched = pred_boundary & gt_boundary      # True positives
    pred_only = pred_boundary & ~gt_boundary   # False positives
    gt_only = gt_boundary & ~pred_boundary     # False negatives

    viz[matched] = [255, 255, 0]    # Yellow - correct
    viz[pred_only] = [255, 0, 0]    # Red - false positive
    viz[gt_only] = [0, 255, 0]      # Green - false negative

    return viz
