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

# Tensor helpers used by the metric suite.
# Adapted from utils3d (https://github.com/EasternJournalist/utils3d), MIT License.
from typing import *
from numbers import Number
from itertools import chain
from numbers import Integral

import torch
from torch import Tensor
import torch.nn.functional as F



def sliding_window(
    x: Tensor, 
    window_size: Union[int, Tuple[int, ...]], 
    stride: Optional[Union[int, Tuple[int, ...]]] = None, 
    dilation: Optional[Union[int, Tuple[int, ...]]] = None,
    pad_size: Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]] = None, 
    pad_mode: str = 'constant',
    pad_value: Number = 0,
    dim: Tuple[int, ...] = None
) -> Tensor:
    """
    Get a sliding window of the input array.
    This function is a wrapper of `torch.nn.functional.unfold` with additional support for padding and stride.

    ## Parameters
    - `x` (Tensor): Input tensor.
    - `window_size` (int or Tuple[int,...]): Size of the sliding window. If int
        is provided, the same size is used for all specified axes.
    - `stride` (Optional[Tuple[int,...]]): Stride between the sliding windows. If None,
        no stride is applied. If int is provided, the same stride is used for all specified axes.
    - `dilation` (Optional[Tuple[int,...]]): Dilation in each sliding window. If None,
        no dilation is applied. If int is provided, the same dilation is used for all specified axes.
    - `pad_size` (Optional[Union[int, Tuple[int, int], Tuple[Tuple[int, int]]]]): Size of padding to apply before sliding window.
        Corresponding to `axis`.
        - General format is `((before_1, after_1), (before_2, after_2), ...)`.
        - Shortcut formats: 
            - `int` -> same padding before and after for all axes;
            - `(int, int)` -> same padding before and after for each axis;
            - `((int,), (int,) ...)` -> specify padding for each axis, same before and after.
    - `pad_mode` (str): Padding mode to use. Refer to `numpy.pad` for more details.
    - `pad_value` (Union[int, float]): Value to use for constant padding. Only used
        when `pad_mode` is 'constant'.
    - `axis` (Optional[Tuple[int,...]]): Axes to apply the sliding window. If None, all axes are used.

    ## Returns
    - (Tensor): Sliding window of the input array. 
        - If no padding, the output is a view of the input array with zero copy.
        - Otherwise, the output is no longer a view but a copy of the padded array.
    """
    if dim is None:
        dim = tuple(range(x.ndim))
    if isinstance(dim, Integral):
        dim = (dim,)
    dim = [dim[i] % x.ndim for i in range(len(dim))]
    if isinstance(window_size, Integral):
        window_size = (window_size,) * len(dim)
    if stride is None:
        stride = (1,) * len(dim)
    elif isinstance(stride, Integral):
        stride = (stride,) * len(dim)
    if dilation is None:
        dilation = (1,) * len(dim)
    elif isinstance(dilation, Integral):
        dilation = (dilation,) * len(dim)
    assert len(window_size) == len(stride) == len(dim)

    # Pad the input array if needed
    if pad_size is not None:
        if isinstance(pad_size, Integral):
            pad_size = ((pad_size, pad_size),) * len(dim)
        elif isinstance(pad_size, tuple) and len(pad_size) == 2 and all(isinstance(p, Integral) for p in pad_size):
            pad_size = (pad_size,) * len(dim)
        elif isinstance(pad_size, tuple) and all(isinstance(p, tuple) and 1 <= len(p) <= 2 for p in pad_size):
            if len(pad_size) == 1:
                pad_size = pad_size * len(dim)
            else:
                assert len(pad_size) == len(dim), f"pad_size {pad_size} must match the number of axes {len(dim)}"
            pad_size = tuple(p * 2 if len(p) == 1 else p for p in pad_size)
        else:
            raise ValueError(f"Invalid pad_size {pad_size}")
        full_pad = [(0, 0) if i not in dim else pad_size[dim.index(i)] for i in range(x.ndim)]
        full_pad = tuple(chain(*reversed(full_pad)))
        x = F.pad(x, full_pad, mode=pad_mode, value=pad_value)
    
    for i in range(len(window_size)):
        x = x.unfold(dim[i], (window_size[i] - 1) * dilation[i] + 1, stride[i])[..., ::dilation[i]]
    return x
