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

# Mask-aware tensor resizing.
# Adapted from utils3d (https://github.com/EasternJournalist/utils3d), MIT License.
import torch
from torch import Tensor


from typing import *
from typing_extensions import Unpack
import math

from metrics.utils import sliding_window



def masked_nearest_resize(
    *image: Tensor,
    mask: Tensor, 
    size: Tuple[int, int], 
    return_index: bool = False
) -> Tuple[Unpack[Tuple[Tensor, ...]], Tensor, Tuple[Tensor, ...]]:
    """
    Resize image(s) by nearest sampling with mask awareness. Suitable for sparse maps. ![masked_nearest_resize.png](doc/masked_nearest_resize.png)
    - Downsampling: Assign the nearest valid pixel within the target pixel's receptive field.
    - Upsampling: Assign the valid pixel to only the nearest pixel in the resized map.


    ### Parameters
    - `*image`: Itorchut image(s) of shape `(..., H, W, C)` or `(... , H, W)` 
        - You can pass multiple images to be resized at the same time for efficiency.
    - `mask`: itorchut mask of shape `(..., H, W)`, dtype=bool
    - `size`: target size `(H', W')`
    - `return_index`: whether to return the nearest neighbor indices in the original map for each pixel in the resized map.
        Defaults to False.

    ### Returns
    - `*resized_image`: resized image(s) of shape `(..., H', W', C)`. or `(..., H', W')`
    - `resized_mask`: mask of the resized map of shape `(..., H', W')`
    - `nearest_indices`: tuple of shape `(..., H', W')`. The nearest neighbor indices of the resized map of each dimension.
    """
    device = mask.device
    height, width = mask.shape[-2:]
    target_height, target_width = size
    filter_h_f, filter_w_f = height / target_height, width / target_width
    filter_h_i, filter_w_i = math.ceil(filter_h_f), math.ceil(filter_w_f)
    filter_size = filter_h_i * filter_w_i
    filter_shape = (filter_h_i, filter_w_i)
    padding_h, padding_w = filter_h_i // 2 + 1, filter_w_i // 2 + 1
    padding_shape = ((padding_h, padding_h), (padding_w, padding_w))
    
    # Window the original mask and uv
    pixels = pixel_coord_map(height, width, convention='integer-corner', dtype=torch.float32, device=device)
    indices = torch.arange(height * width, dtype=torch.long, device=device).reshape(height, width)
    window_pixels = sliding_window(pixels, window_size=filter_shape, pad_size=padding_shape, dim=(0, 1))
    window_indices = sliding_window(indices, window_size=filter_shape, pad_size=padding_shape, dim=(0, 1))
    window_mask = sliding_window(mask, window_size=filter_shape, pad_size=padding_shape, dim=(-2, -1))

    # Gather the target pixels's local window
    target_centers = uv_map(target_height, target_width, dtype=torch.float32, device=device) * torch.tensor([width, height], dtype=torch.float32, device=device)
    target_lefttop = target_centers - torch.tensor((filter_w_f / 2, filter_h_f / 2), dtype=torch.float32, device=device)
    target_window = torch.round(target_lefttop).to(torch.long) + torch.tensor((padding_w, padding_h), dtype=torch.long, device=device)

    target_window_pixels = window_pixels[target_window[..., 1], target_window[..., 0], :, :, :].reshape(target_height, target_width, 2, filter_size)                  # (target_height, tgt_width, 2, filter_size)
    target_window_mask = window_mask[..., target_window[..., 1], target_window[..., 0], :, :].reshape(*mask.shape[:-2], target_height, target_width, filter_size)     # (..., target_height, tgt_width, filter_size)
    target_window_indices = window_indices[target_window[..., 1], target_window[..., 0], :, :].reshape(target_height, target_width, filter_size)                      # (target_height, tgt_width, filter_size)

    # Compute nearest neighbor in the local window for each pixel 
    delta = target_window_pixels - target_centers[..., None]
    eps = torch.finfo(torch.float32).eps * max(height, width) # Shift a small epsilon to avoid numerical issues when the pixel is exactly on the border
    target_window_mask &= (-filter_w_f / 2 + eps < delta[..., 0, :]) & (delta[..., 0, :] <= filter_w_f / 2 + eps) & (-filter_h_f / 2 + eps < delta[..., 1, :]) & (delta[..., 1, :] <= filter_h_f / 2 + eps)
    dist = torch.where(target_window_mask, torch.square(delta[..., 0, :]) + torch.square(delta[..., 1, :]), torch.inf)      # (..., target_height, tgt_width, filter_size)
    nearest_in_window = torch.argmin(dist, dim=-1, keepdim=True)                                                 # (..., target_height, tgt_width, 1)
    nearest_idx = torch.gather(
        target_window_indices.expand(dist.shape),
        dim=-1,
        index=nearest_in_window,
    ).squeeze(-1)     # (..., target_height, tgt_width)
    nearest_i, nearest_j = nearest_idx // width, nearest_idx % width
    target_mask = torch.any(target_window_mask, dim=-1)
    batch_indices = [torch.arange(n, device=device).reshape([1] * i + [n] + [1] * (mask.ndim - i - 1)) for i, n in enumerate(mask.shape[:-2])]

    nearest_indices = (*batch_indices, nearest_i, nearest_j)
    outputs = tuple(x[nearest_indices] for x in image)

    if return_index:
        ret = (*outputs, target_mask, nearest_indices)
    else:
        ret = (*outputs, target_mask)

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def pixel_coord_map(
    *size: Union[int, Tuple[int, int]],
    top: int = 0,
    left: int = 0,
    convention: Literal['integer-center', 'integer-corner'] = 'integer-center',
    dtype: torch.dtype = torch.float32,
    device: torch.device = None
) -> Tensor:
    """
    Get image pixel coordinates map. Support two conventions: `'integer-center'` and `'integer-corner'`.

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `int`, optional top boundary of the pixel coord map. Defaults to 0.
    - `left`: `int`, optional left boundary of the pixel coord map. Defaults to 0.
    - `convention`: `str`, optional `'integer-center'` or `'integer-corner'`, whether integer coordinates correspond to pixel centers or corners. Defaults to 'integer-center'.
        - `'integer-center'`: `pixel[i][j]` has integer coordinates `(j, i)` as its center, and occupies square area `[j - 0.5, j + 0.5) × [i - 0.5, i + 0.5)`. 
            The top-left corner of the top-left pixel is `(-0.5, -0.5)`, and the bottom-right corner of the bottom-right pixel is `(width - 0.5, height - 0.5)`.
        - `'integer-corner'`: `pixel[i][j]` has coordinates `(j + 0.5, i + 0.5)` as its center, and occupies square area `[j, j + 1) × [i, i + 1)`.
            The top-left corner of the top-left pixel is `(0, 0)`, and the bottom-right corner of the bottom-right pixel is `(width, height)`.
    - `dtype`: `torch.dtype`, optional data type of the output pixel coord map. Defaults to torch.float32.

    ## Returns
        Tensor: shape (height, width, 2)
    
    >>> pixel_coord_map(10, 10, convention='integer-center', dtype=torch.long):
    [[[0, 0], [1, 0], ..., [9, 0]],
     [[0, 1], [1, 1], ..., [9, 1]],
        ...      ...         ...
     [[0, 9], [1, 9], ..., [9, 9]]]

    >>> pixel_coord_map(10, 10, convention='integer-corner', dtype=torch.float32):
    [[[0.5, 0.5], [1.5, 0.5], ..., [9.5, 0.5]],
     [[0.5, 1.5], [1.5, 1.5], ..., [9.5, 1.5]],
      ...             ...                  ...
    [[0.5, 9.5], [1.5, 9.5], ..., [9.5, 9.5]]]
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = torch.arange(left, left + width, dtype=dtype, device=device)
    v = torch.arange(top, top + height, dtype=dtype, device=device)
    if convention == 'integer-corner':
        assert torch.is_floating_point(u), "dtype should be a floating point type when convention is 'integer-corner'"
        u = u + 0.5
        v = v + 0.5
    u, v = torch.meshgrid(u, v, indexing='xy')
    return torch.stack([u, v], dim=2)


def uv_map(
    *size: Union[int, Tuple[int, int]],
    top: float = 0.,
    left: float = 0.,
    bottom: float = 1.,
    right: float = 1.,
    dtype: torch.dtype = torch.float32,
    device: torch.device = None
) -> Tensor:
    """
    Get image UV coordinate map. By default, (0., 0.) is the top-left corner of the image, and (1., 1.) is the bottom-right corner of the image.

    ## Parameters
    - `*size`: `Tuple[int, int]` or two integers of map size `(height, width)`
    - `top`: `float` defaults to 0.
    - `left`: `float` defaults to 0.
    - `bottom`: `float` defaults to 1.
    - `right`: `float` defaults to 1.
    - `dtype`: `torch.dtype` data type of the output uv map. Defaults to torch.float32.
    - `device`: `torch.device`, device of the output uv map. Defaults to None.

    ## Returns
    - `uv (Tensor)`: shape `(height, width, 2)`

    ## Example Usage

    >>> uv_map(10, 10):
    [[[0.05, 0.05], [0.15, 0.05], ..., [0.95, 0.05]],
     [[0.05, 0.15], [0.15, 0.15], ..., [0.95, 0.15]],
      ...             ...                  ...
     [[0.05, 0.95], [0.15, 0.95], ..., [0.95, 0.95]]]
    """
    if len(size) == 1 and isinstance(size[0], tuple):
        height, width = size[0]
    else:
        height, width = size
    u = torch.linspace(left + 0.5 / width * (right - left), right - 0.5 / width * (right - left), width, dtype=dtype, device=device)
    v = torch.linspace(top + 0.5 / height * (bottom - top), bottom - 0.5 / height * (bottom - top), height, dtype=dtype, device=device)
    uv = torch.stack(torch.meshgrid(u, v, indexing='xy'), dim=-1)
    return uv



def _test():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.rand(1, 256, 256).to(device)
    mask = x > 0.5
    height, width = mask.shape[-2:]
    lr_mask, lr_index = masked_nearest_resize(mask=mask, size=(64, 64), return_index=True)

    print(lr_mask.shape)


if __name__ == "__main__":
    _test()

