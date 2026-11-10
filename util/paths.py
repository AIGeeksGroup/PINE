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

"""Paths that resolve against the repository, not the current directory.

Constants the code hardcodes -- ``pretrained/dinov3``, ``generation`` -- go
through :func:`repo_path` instead of being used as bare relative strings, so a
command launched from somewhere other than the repository root still finds the
``pretrained`` and ``datasets`` symlinks that live next to this package.
"""

import os

# The repository root: this file is <repo>/util/paths.py, so go up two levels.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def repo_path(*parts):
    """Join `parts` onto the repository root and return an absolute path."""
    return os.path.join(REPO_ROOT, *parts)


# sha256 prefixes in the file names of the released DINOv3 LVD-1689M weights
dinov3_weight_hashes = {
    'vits16': '08c60483',
    'vits16plus': '4057cbaa',
    'vitb16': '73cec8be',
    'vitl16': '8aa4cbdd',
    'vith16plus': '7c1da9a5',
    'vit7b16': 'a955f4ea',
}


def dinov3_weights_path(feature_embedding_type):
    """Path of the DINOv3 weights of an encoder such as dinov3_vitl16, in $DINOV3_WEIGHTS_DIR (default
    pretrained/dinov3 in the repository)."""
    vit_type = feature_embedding_type.split('_')[-1]
    weights_dir = os.environ.get('DINOV3_WEIGHTS_DIR', repo_path('pretrained/dinov3'))
    return os.path.join(weights_dir, f'dinov3_{vit_type}_pretrain_lvd1689m-{dinov3_weight_hashes[vit_type]}.pth')
