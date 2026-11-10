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
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm


def viz_depth_tensor(disp, return_numpy=True, colormap="plasma", viz_inverse_depth=True, shifted_depth=False):
    # Convert input to a NumPy array
    if isinstance(disp, torch.Tensor):
        disp_np = disp.detach().cpu().numpy()
    elif isinstance(disp, np.ndarray):
        disp_np = disp
    else:
        raise TypeError(
            "Input must be a torch.Tensor or a numpy.ndarray, "
            f"but got {type(disp)}"
        )

    # Ensure the input is 2D
    if disp_np.ndim == 3:
        disp_np = disp_np.squeeze()
    if disp_np.ndim != 2:
        raise ValueError("Depth/Disparity array must be 2-dimensional.")

    # in the dataloader, the point is shift to have zero center
    # normalize when visualizing
    if shifted_depth:
        # For zero-centered model output (unknown scale/shift):
        #
        # Step 1 – shift raw depth to positive by subtracting p_low, preserving scale ratios.
        # We do NOT normalize to [0,1] here because that destroys scale: inverting values
        # already compressed into [1e-3, 1.001] produces a heavily skewed distribution
        # where 75%+ of pixels collapse into the bottom 2% of the colormap.
        finite_vals = disp_np[np.isfinite(disp_np)]
        if finite_vals.size == 0:
            H, W = disp_np.shape
            black = np.zeros((H, W, 3), dtype=np.uint8)
            if return_numpy:
                return black
            return torch.from_numpy(black).permute(2, 0, 1)
        p_low  = np.percentile(finite_vals, 0.1)
        p_high = np.percentile(finite_vals, 99.9)
        # Shift so the 2nd-percentile depth becomes 1e-3 (all values positive, scale preserved)
        disp_shifted = np.clip(disp_np, p_low, p_high) - p_low + 1e-3

        # Step 2 – take inverse depth (near=small depth → large inverse → bright)
        inv_disp = 1.0 / disp_shifted  # all positive since disp_shifted >= 1e-3

        # Step 3 – normalize by percentile, same as the non-shifted path.
        # Two degenerate cases have to be handled or this writes NaNs into the
        # PNG: np.clip above preserves NaN (hence the nan-aware statistics), and
        # a constant prediction gives vmin == vmax. Note that nudging by a fixed
        # 1e-6 is not enough to fix the latter -- float32 spacing near the
        # typical inverse-depth magnitude of 1e3 is ~6e-5, so the nudge rounds
        # away and leaves a 0/0. Compute the span in float64 instead.
        vmin = float(np.nanmin(inv_disp))
        vmax = float(np.nanpercentile(inv_disp, 95))
        if not np.isfinite(vmin):
            vmin = 0.0
        span = vmax - vmin
        if not np.isfinite(span) or span <= 1e-12:
            span = 1.0  # constant depth -> flat image rather than NaNs
        normalized = np.nan_to_num(
            (inv_disp.astype(np.float64) - vmin) / span, nan=0.0).clip(0, 1)

        # Step 4 – apply colormap and return early
        cmap_fn = mpl.colormaps[colormap]
        colormapped_im = (cmap_fn(normalized)[:, :, :3] * 255).astype(np.uint8)
        if return_numpy:
            return colormapped_im
        return torch.from_numpy(colormapped_im).permute(2, 0, 1)

    # --- KEY CHANGE: INVERSE DEPTH VISUALIZATION ---
    if viz_inverse_depth:
        # Create a mask for valid (positive) depth values
        valid_mask = disp_np > 0

        # Calculate inverse depth only for valid pixels
        disp_viz = np.zeros_like(disp_np, dtype=np.float32)
        disp_viz[valid_mask] = 1.0 / disp_np[valid_mask]
    else:
        disp_viz = disp_np
    # -----------------------------------------------

    # Handle NaNs and Infs for normalization (now on the inverse depth data)
    final_valid_mask = np.isfinite(disp_viz) & (disp_viz > 0)
    
    if not np.any(final_valid_mask):
         H, W = disp_np.shape[:2]
         black = np.zeros((H, W, 3), dtype=np.uint8)
         if return_numpy:
             return black
         return torch.from_numpy(black).permute(2, 0, 1)

    disp_valid = disp_viz[final_valid_mask]
    
    vmin = disp_valid.min()
    vmax = np.percentile(disp_valid, 95)

    if vmax <= vmin + 1e-6:
        vmax = vmin + 1e-6 

    # Normalization and Colormapping
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=colormap)
    
    # Map values to RGB [H, W, 3]
    # Use the disp_viz array (inverse depth) for colormapping
    colormapped_im = (mapper.to_rgba(disp_viz)[:, :, :3] * 255).astype(np.uint8)

    if return_numpy:
        return colormapped_im
    
    return torch.from_numpy(colormapped_im).permute(2, 0, 1)
