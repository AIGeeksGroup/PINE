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
import torch.nn.functional as F
import math


def resize_vit_pos_embed(checkpoint_model, new_num_patches=64, key='net.pos_embed'):
    """
    Resizes the positional embedding in a checkpoint dictionary.

    Args:
        checkpoint_model: The dict from torch.load
        new_num_patches: Target number of patches (e.g., 64 for 256x256 image with 32x32 patches)
        key: The key in the dict pointing to the pos_embed
    """
    if key not in checkpoint_model:
        print(f"Key {key} not found in checkpoint.")
        return checkpoint_model

    pos_embed_checkpoint = checkpoint_model[key]  # Shape: [1, Old_N, Dim]
    embedding_size = pos_embed_checkpoint.shape[-1]

    # Calculate grid sizes
    # We check if there's an extra token (CLS token)
    # Standard ViT: num_patches = grid_size * grid_size
    # If Old_N is not a perfect square, we assume 1 extra token (CLS)
    old_num_tokens = pos_embed_checkpoint.shape[1]
    # TODO: here we assume square images and patches
    new_grid_size = int(math.sqrt(new_num_patches))

    assert new_grid_size * new_grid_size == new_num_patches, "new_num_patches must be a perfect square."

    # Determine if there is a CLS token by checking if old_num_tokens is a perfect square
    old_grid_size = int(math.sqrt(old_num_tokens))
    assert old_grid_size * old_grid_size == old_num_tokens, "Old number of tokens must be a perfect square or have one extra token."
    if old_grid_size * old_grid_size == old_num_tokens:
        num_extra_tokens = 0
    else:
        num_extra_tokens = old_num_tokens - (int(math.sqrt(old_num_tokens - 1))**2)
        old_grid_size = int(math.sqrt(old_num_tokens - num_extra_tokens))

    print(f"Resizing {key}: {old_grid_size}x{old_grid_size} -> {new_grid_size}x{new_grid_size} "
            f"({num_extra_tokens} extra tokens)")

    # 1. Separate extra tokens and patch tokens
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    patch_tokens = pos_embed_checkpoint[:, num_extra_tokens:]

    # 2. Reshape to [1, C, H, W] for interpolation
    patch_tokens = patch_tokens.reshape(1, old_grid_size, old_grid_size, embedding_size).permute(0, 3, 1, 2)

    # 3. Interpolate
    patch_tokens = F.interpolate(
        patch_tokens, 
        size=(new_grid_size, new_grid_size), 
        mode='bicubic', 
        align_corners=False
    )

    # 4. Reshape back to [1, New_N, C]
    patch_tokens = patch_tokens.permute(0, 2, 3, 1).flatten(1, 2)

    # 5. Concatenate and update
    checkpoint_model[key] = torch.cat((extra_tokens, patch_tokens), dim=1)

    return checkpoint_model


def resize_patch_embed_and_final_layer(checkpoint_model, old_patch_size, new_patch_size, out_channels=3):
    """
    Resize patch embedding kernel and final layer when changing patch sizes.

    Args:
        checkpoint_model: checkpoint state dict
        old_patch_size: original patch size (e.g., 16)
        new_patch_size: target patch size (e.g., 32)
        out_channels: number of output channels (default 3 for RGB)
    """
    # Resize x_embedder.proj1.weight: [C_out, C_in, old_ps, old_ps] -> [C_out, C_in, new_ps, new_ps]
    key = 'net.x_embedder.proj1.weight'
    if key in checkpoint_model:
        old_weight = checkpoint_model[key]
        new_weight = F.interpolate(old_weight, size=(new_patch_size, new_patch_size),
                                   mode='bicubic', align_corners=False)
        checkpoint_model[key] = new_weight
        print(f"Resized {key}: {old_weight.shape} -> {new_weight.shape}")

    # Resize final_layer.linear.weight: [old_ps*old_ps*C, hidden] -> [new_ps*new_ps*C, hidden]
    key = 'net.final_layer.linear.weight'
    if key in checkpoint_model:
        old_weight = checkpoint_model[key]
        hidden_size = old_weight.shape[1]
        old_weight_4d = old_weight.view(old_patch_size, old_patch_size, out_channels, hidden_size)
        old_weight_4d = old_weight_4d.permute(2, 3, 0, 1)
        new_weight = F.interpolate(old_weight_4d, size=(new_patch_size, new_patch_size),
                                   mode='bicubic', align_corners=False)
        new_weight = new_weight.permute(2, 3, 0, 1).reshape(-1, hidden_size)
        checkpoint_model[key] = new_weight
        print(f"Resized {key}: {old_weight.shape} -> {new_weight.shape}")

    # Resize final_layer.linear.bias: [old_ps*old_ps*C] -> [new_ps*new_ps*C]
    key = 'net.final_layer.linear.bias'
    if key in checkpoint_model:
        old_bias = checkpoint_model[key]
        old_bias_4d = old_bias.view(old_patch_size, old_patch_size, out_channels).permute(2, 0, 1).unsqueeze(0)
        new_bias = F.interpolate(old_bias_4d, size=(new_patch_size, new_patch_size),
                                 mode='bicubic', align_corners=False)
        new_bias = new_bias.squeeze(0).permute(1, 2, 0).reshape(-1)
        checkpoint_model[key] = new_bias
        print(f"Resized {key}: {old_bias.shape} -> {new_bias.shape}")

    return checkpoint_model

