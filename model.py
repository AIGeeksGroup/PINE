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

# --------------------------------------------------------
# References:
# SiT: https://github.com/willisma/SiT
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------
import copy
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from util.model_util import get_2d_sincos_pos_embed, RMSNorm, rotate_half
from util.paths import dinov3_weights_path, repo_path

POINT_MAP_CHANNELS = 3
# every stream denoises one 3-channel map; the instance stream is a colour map whose colours tell the instances apart
STREAMS = ('pointmap', 'normal', 'instance')
GEOMETRY_STREAMS = STREAMS[:2]
# state-dict prefix of the per-stream modules (the point stream keeps the PointDiT names)
STREAM_PREFIXES = ('', 'normal_', 'instance_')
STREAM_MODULES = ('x_embedder', 'concat_proj', 'final_layer', 't_embedder')
# which streams exchange tokens in attention; the paper uses isolated_instance
STREAM_ATTENTIONS = ('isolated_instance', 'bidirectional', 'instance_reads_geometry',
                     'normal_to_point_isolated_instance')
# DINOv3 grounding conditions the normal and the instance stream
GROUNDED_STREAMS = (STREAMS.index('normal'), STREAMS.index('instance'))

try:
    from flash_attn_interface import flash_attn_func
    FA3_AVAILABLE = True
except ImportError:
    FA3_AVAILABLE = False



def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class DynamicRoPE:
    """Applies precomputed 2-D rotary position embeddings to q/k."""
    def __init__(self, cos, sin):
        self.cos = cos
        self.sin = sin

    def __call__(self, t):
        return t * self.cos + rotate_half(t) * self.sin


CAMERA_PE_SITES = ('image_features', 'geometry_tokens', 'rope')


class CameraPositionalEncoding(nn.Module):
    """Pinhole-normalized positions with an explicit unknown-camera condition. Positions are centred on the square
    training grid, whatever the input grid, so a wider or larger input keeps its optical axis where training put it.
    The coordinates feed three sites: the sin-cos position added to the DINOv3 image features ('image_features'), the
    sin-cos position added to the geometry stream tokens ('geometry_tokens') and the attention 2D RoPE ('rope').  A site
    left out of `sites` always gets the reference camera's coordinates, so it keeps positions but never sees K."""
    def __init__(self, reference_focal, patch_size, hidden_size, num_heads, true_prob=0.8, *, training_grid,
                 sites=CAMERA_PE_SITES):
        super().__init__()
        if not set(sites) <= set(CAMERA_PE_SITES):
            raise ValueError(f'Unknown camera positional encoding sites: {sites}')
        self.sites = tuple(sites)
        self.training_grid = training_grid
        self.register_buffer('reference_focal', torch.tensor(reference_focal, dtype=torch.float32))
        self.unknown_camera = nn.Parameter(torch.zeros(hidden_size))
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.axis_rotary_dim = hidden_size // num_heads // 2
        self.true_prob = true_prob

    def select_intrinsics(self, intrinsics, batch_size, height, width):
        # The unknown camera assumes no focal length: its short edge spans the training grid at any resolution, which
        # is the reference focal on the square training input and scales with the short edge elsewhere.
        focal = self.reference_focal * min(height, width) / (self.training_grid * self.patch_size)
        center = focal.new_tensor(((width - 1) / 2, (height - 1) / 2))
        reference = torch.cat((focal.expand(2), center)).expand(batch_size, -1)
        if intrinsics is None:
            unknown = torch.ones(batch_size, dtype=torch.bool, device=focal.device)
            return reference, unknown
        if not self.training:
            unknown = torch.zeros(batch_size, dtype=torch.bool, device=focal.device)
            return intrinsics.float(), unknown
        # NaN intrinsics come from datasets without a camera (dataloader/img_instance_segmentation.py)
        unknown = (torch.rand(batch_size, device=focal.device) >= self.true_prob) | intrinsics.isnan().any(1)
        selected = torch.where(unknown[:, None], reference, intrinsics.float())
        return selected, unknown

    def coordinates(self, intrinsics, grid_size):
        height, width = grid_size
        device = intrinsics.device
        columns = torch.arange(width, device=device, dtype=torch.float32)
        rows = torch.arange(height, device=device, dtype=torch.float32)
        vertical, horizontal = torch.meshgrid(rows, columns, indexing='ij')
        centers = torch.stack((horizontal, vertical), dim=-1).reshape(-1, 2)
        centers = (centers + 0.5) * self.patch_size - 0.5
        scale = self.reference_focal / intrinsics[:, :2] / self.patch_size
        origin = (self.training_grid - 1) / 2
        return (centers[None] - intrinsics[:, None, 2:]) * scale[:, None] + origin

    @staticmethod
    def position_embedding(coordinates, embedding_dim):
        frequency = torch.arange(embedding_dim // 4, device=coordinates.device).float()
        frequency = 1.0 / 10000 ** (frequency / (embedding_dim // 4))
        phase = coordinates[..., None] * frequency
        embedding = torch.cat((phase.sin(), phase.cos()), dim=-1)
        return embedding.flatten(-2)

    def forward(self, intrinsics, batch_size, grid_size, image_size, image_feature_dim):
        height, width = image_size
        intrinsics, unknown = self.select_intrinsics(intrinsics, batch_size, height, width)
        coordinates = self.coordinates(intrinsics, grid_size)
        fixed = coordinates
        if len(self.sites) < len(CAMERA_PE_SITES):
            reference, _ = self.select_intrinsics(None, batch_size, height, width)
            fixed = self.coordinates(reference, grid_size)
        site = lambda name: coordinates if name in self.sites else fixed
        frequency = torch.arange(0, self.axis_rotary_dim, 2, device=coordinates.device).float()
        frequency = 1.0 / 10000 ** (frequency / self.axis_rotary_dim)
        phase = site('rope').flip(-1)[..., None] * frequency
        phase = phase.repeat_interleave(2, dim=-1).flatten(-2)[:, None]
        position = self.position_embedding(site('geometry_tokens'), self.hidden_size)
        image_position = self.position_embedding(site('image_features'), image_feature_dim)
        camera_condition = unknown[:, None].float() * self.unknown_camera[None]
        rotary = DynamicRoPE(phase.cos(), phase.sin())
        return position, image_position, rotary, camera_condition


class BottleneckPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, pca_dim=768, embed_dim=768, bias=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj1 = nn.Conv2d(in_chans, pca_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.proj2 = nn.Conv2d(pca_dim, embed_dim, kernel_size=1, stride=1, bias=bias)

    def forward(self, x):
        B, C, H, W = x.shape
        # Allow any size divisible by patch_size (support rectangular inputs)
        assert H % self.patch_size[0] == 0 and W % self.patch_size[1] == 0, \
            f"Input size ({H}x{W}) must be divisible by patch size ({self.patch_size[0]}x{self.patch_size[1]})"
        x = self.proj2(self.proj1(x)).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


def scaled_dot_product_attention_flash3(query, key, value, dropout_p=0.0) -> torch.Tensor:
    """
    Computes scaled dot product attention using the flash_attn library (FA3).
    
    Args:
        query: Shape (Batch, NumHeads, SeqLen_Q, HeadDim)
        key:   Shape (Batch, NumHeads, SeqLen_K, HeadDim)
        value: Shape (Batch, NumHeads, SeqLen_K, HeadDim)
    
    Returns:
        output: Shape (Batch, NumHeads, SeqLen_Q, HeadDim)
    """
    assert FA3_AVAILABLE, "flash_attn_func is not available. Please install flash attention library."

    # 1. Permute dimensions from (B, H, S, D) to (B, S, H, D)
    # Flash Attention requires the 'Heads' dimension to be 3rd, not 2nd.
    # Mannually cast to bfloat16 for better performance and compatibility.
    q = query.transpose(1, 2).to(torch.bfloat16)
    k = key.transpose(1, 2).to(torch.bfloat16)
    v = value.transpose(1, 2).to(torch.bfloat16)

    # 2. Run Flash Attention
    # Note: softmax_scale defaults to 1/sqrt(d) if not provided.
    assert dropout_p == 0.0, "Flash Attention 3 currently does not support dropout."
    # https://github.com/Dao-AILab/flash-attention/issues/1377
    output = flash_attn_func(
        q, k, v,
        # dropout_p=dropout_p,
    )[0]

    # 3. Permute back to (B, H, S, D)
    return output.transpose(1, 2)


class Attention(nn.Module):
    """Attention weights of one block; StreamBlock runs them over the streams."""
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True, attn_drop=0., proj_drop=0.,
                 attention_type='torch'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.attention_type = attention_type

        self.q_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(head_dim) if qk_norm else nn.Identity()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def attend(self, query, key, value):
        dropout = self.attn_drop.p if self.training else 0.
        if self.attention_type == 'torch':
            return F.scaled_dot_product_attention(query, key, value, dropout_p=dropout)
        if self.attention_type == 'flash3':
            return scaled_dot_product_attention_flash3(query, key, value, dropout_p=dropout)
        raise NotImplementedError(f'Attention type {self.attention_type} not implemented')


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        drop=0.0,
        bias=True
    ) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)
        self.ffn_dropout = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.ffn_dropout(hidden)
        return self.w3(hidden)


class FinalLayer(nn.Module):
    """Linear readout of one stream: every token gives the patch_size x patch_size pixels of its patch."""
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """c: [B, C] for the whole image, or [B, N, C] for one condition per token."""
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale) if c.dim() == 2 else self.norm_final(x) * (1 + scale) + shift
        return self.linear(x)


def modulate_geometry(hidden, shift, scale):
    streams = hidden.unflatten(1, (shift.shape[1], -1))
    streams = streams * (1 + scale.unsqueeze(2)) + shift.unsqueeze(2)
    return streams.flatten(1, 2)


def scale_geometry(hidden, scale):
    streams = hidden.unflatten(1, (scale.shape[1], -1))
    return (streams * scale.unsqueeze(2)).flatten(1, 2)


def modulate_streams(hidden, shifts, scales):
    """Per-stream modulation; the shift and scale of a stream are [B, 1, C] (broadcast) or [B, N, C] (per token)."""
    streams = hidden.chunk(len(shifts), dim=1)
    return torch.cat([stream * (1 + scale) + shift for stream, shift, scale in zip(streams, shifts, scales)], dim=1)


def scale_streams(hidden, scales):
    streams = hidden.chunk(len(scales), dim=1)
    return torch.cat([stream * scale for stream, scale in zip(streams, scales)], dim=1)


class StreamBlock(nn.Module):
    """Transformer block over the point, normal and instance tokens of an image, concatenated along the sequence. The
    streams share the weights; each has its own time condition, and stream_attention says which streams exchange
    tokens."""
    def __init__(self, hidden_size, num_heads, stream_attention, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0,
                 attention_type='torch'):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, qk_norm=True,
                              attn_drop=attn_drop, proj_drop=proj_drop,
                              attention_type=attention_type)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim, drop=proj_drop)

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        self.attend_streams = getattr(self, stream_attention + '_attention')

    def attention(self, hidden, rope, affinity=None):
        batch_size, token_count, hidden_size = hidden.shape
        head_count = self.attn.num_heads
        qkv = self.attn.qkv(hidden)
        qkv = qkv.reshape(batch_size, token_count, 3, head_count, hidden_size // head_count)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        queries = [rope(stream) for stream in self.attn.q_norm(query).chunk(len(STREAMS), dim=2)]
        keys = [rope(stream) for stream in self.attn.k_norm(key).chunk(len(STREAMS), dim=2)]
        values = value.chunk(len(STREAMS), dim=2)
        if affinity is not None:
            queries, keys, values = self.grounded_attention_inputs(queries, keys, values, affinity)
        output = self.attend_streams(queries, keys, values)[..., :hidden_size // head_count]
        output = output.transpose(1, 2).reshape(batch_size, token_count, hidden_size)
        return self.attn.proj_drop(self.attn.proj(output))

    def grounded_attention_inputs(self, queries, keys, values, affinity):
        """Extra per-head dims on the grounded queries and keys, zeros elsewhere; logits gain (U g_j).(V g_k)/sqrt(d)."""
        query_dims, key_dims, grounded = affinity
        query_dims, key_dims = (dims.unflatten(-1, (self.attn.num_heads, -1)).transpose(1, 2)
                                for dims in (query_dims, key_dims))
        zeros = query_dims.new_zeros(query_dims.shape)
        # the scaled queries keep the softmax at 1 / sqrt(d) of the original head dim d
        scale = math.sqrt(query_dims.shape[-1] / queries[0].shape[-1] + 1)
        queries = [torch.cat((query, (query_dims if index in grounded else zeros).to(query.dtype)), dim=-1) * scale
                   for index, query in enumerate(queries)]
        keys = [torch.cat((key, (key_dims if index in grounded else zeros).to(key.dtype)), dim=-1)
                for index, key in enumerate(keys)]
        values = [torch.cat((value, zeros.to(value.dtype)), dim=-1) for value in values]
        return queries, keys, values

    def bidirectional_attention(self, queries, keys, values):
        return self.attn.attend(torch.cat(queries, dim=2), torch.cat(keys, dim=2), torch.cat(values, dim=2))

    def isolated_instance_attention(self, queries, keys, values):
        """Point and normal attend to each other both ways; the instance stream attends only to itself and is unseen."""
        geometry = self.attn.attend(torch.cat(queries[:2], dim=2), torch.cat(keys[:2], dim=2), torch.cat(values[:2], dim=2))
        instance = self.attn.attend(queries[2], keys[2], values[2])
        return torch.cat((geometry, instance), dim=2)

    def normal_to_point_isolated_instance_attention(self, queries, keys, values):
        """Point reads point and normal, normal reads only itself (normal_to_point), instance reads only itself."""
        point = self.attn.attend(queries[0], torch.cat(keys[:2], dim=2), torch.cat(values[:2], dim=2))
        normal = self.attn.attend(queries[1], keys[1], values[1])
        instance = self.attn.attend(queries[2], keys[2], values[2])
        return torch.cat((point, normal, instance), dim=2)

    def instance_reads_geometry_attention(self, queries, keys, values):
        """Point and normal attend to each other both ways and never to the instance stream, which reads all three."""
        geometry = self.attn.attend(torch.cat(queries[:2], dim=2), torch.cat(keys[:2], dim=2), torch.cat(values[:2], dim=2))
        instance = self.attn.attend(queries[2], torch.cat(keys, dim=2), torch.cat(values, dim=2))
        return torch.cat((geometry, instance), dim=2)

    def forward(self, hidden, condition, rope, token_conditions=None, affinity=None):
        """token_conditions {stream index: [B, N, C]}: streams modulated per token; affinity: grounded dims."""
        modulation = self.adaLN_modulation(condition)
        if token_conditions is None:
            shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = modulation.chunk(6, dim=-1)
            attention_input = modulate_geometry(self.norm1(hidden), shift_attn, scale_attn)
            hidden = hidden + scale_geometry(self.attention(attention_input, rope, affinity), gate_attn)
            feedforward_input = modulate_geometry(self.norm2(hidden), shift_ffn, scale_ffn)
            return hidden + scale_geometry(self.mlp(feedforward_input), gate_ffn)
        modulations = [self.adaLN_modulation(token_conditions[index]) if index in token_conditions
                       else modulation[:, index, None] for index in range(modulation.shape[1])]
        shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn = zip(
            *(stream.chunk(6, dim=-1) for stream in modulations))
        attention_input = modulate_streams(self.norm1(hidden), shift_attn, scale_attn)
        hidden = hidden + scale_streams(self.attention(attention_input, rope, affinity), gate_attn)
        feedforward_input = modulate_streams(self.norm2(hidden), shift_ffn, scale_ffn)
        return hidden + scale_streams(self.mlp(feedforward_input), gate_ffn)


class PINE(nn.Module):
    """
    Pixel-space diffusion transformer that denoises a point map, a normal map and an instance map of one image
    together, conditioned on frozen DINOv3 features of the image. Each map is a stream with its own input and output
    layers and its own timestep; the streams share the transformer blocks.
    """
    def __init__(
        self,
        input_size=256,
        patch_size=16,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        attention_type='torch',
        feature_embedding_type='dinov3_vitb16',
        dinov3_use_intermediate_layers=True,
        dinov3_num_intermediate_layers=4,
        finetune_feature_embedding=False,
        stream_attention='isolated_instance',
        camera_positional_encoding=False,
        reference_focal=None,
        camera_true_prob=0.8,
        camera_pe_sites=CAMERA_PE_SITES,
        dino_grounding='none',
        dino_grounding_rank=256,
        dino_grounding_head_dim=32,
    ):
        super().__init__()
        self.in_channels = self.out_channels = len(STREAMS) * POINT_MAP_CHANNELS
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.attention_type = attention_type
        self.feature_embedding_type = feature_embedding_type
        self.dinov3_use_intermediate_layers = dinov3_use_intermediate_layers
        self.dinov3_num_intermediate_layers = dinov3_num_intermediate_layers
        self.finetune_feature_embedding = finetune_feature_embedding

        # time embed
        self.t_embedder = TimestepEmbedder(hidden_size)

        # frozen pre-trained image encoder producing the conditioning tokens
        if self.feature_embedding_type.startswith('dinov3'):
            vit_type = self.feature_embedding_type.split('_')[-1]
            # Configurable so the model can be built on any machine. Defaults assume the
            # DINOv3 checkout in third_party/ and the gated weights in pretrained/dinov3/,
            # both resolved inside the repository rather than against the cwd.
            dinov3_repo = os.environ.get(
                'DINOV3_REPO', repo_path('third_party/dinov3'))
            local_weights_path = dinov3_weights_path(self.feature_embedding_type)
            if not os.path.isdir(dinov3_repo):
                raise FileNotFoundError(
                    f'DINOv3 repository not found at "{dinov3_repo}". Clone it with\n'
                    f'    git clone https://github.com/facebookresearch/dinov3.git third_party/dinov3\n'
                    f'or set DINOV3_REPO to an existing checkout.')
            # NOTE: need to load the weights in the main function due to the random init of the __init__
            if os.path.isfile(local_weights_path):
                self.y_embedder = torch.hub.load(
                    dinov3_repo, f'dinov3_{vit_type}', source='local', weights=local_weights_path)
            else:
                self.y_embedder = torch.hub.load(
                    dinov3_repo, f'dinov3_{vit_type}', source='local', pretrained=False)
            # freeze DINOv3 parameters
            self.y_embedder.requires_grad_(self.finetune_feature_embedding)

            self.y_embedder.num_patches = self.y_embedder.patch_embed.num_patches

            # compute equally-spaced layer indices for intermediate feature extraction
            if self.dinov3_use_intermediate_layers:
                dinov3_depth = len(self.y_embedder.blocks)
                n = self.dinov3_num_intermediate_layers
                # n equally-spaced layers
                self.dinov3_intermediate_layer_indices = [
                    int((i + 1) * dinov3_depth / n) - 1 for i in range(n)
                ]
                self.y_embedder.num_features = self.y_embedder.num_features * n

        else:
            raise NotImplementedError(f'Feature embedding type {self.feature_embedding_type} not implemented')

        # linear embed
        self.x_embedder = BottleneckPatchEmbed(input_size, patch_size, POINT_MAP_CHANNELS, bottleneck_dim, hidden_size, bias=True)

        # use fixed sin-cos embedding
        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        # position embedding for the conditioning image tokens (frozen sin-cos, added in
        # forward); only needed when the encoder's feature dim differs from the model dim.
        y_feat_dim = self.y_embedder.num_features
        if y_feat_dim != hidden_size:
            self.pos_embed_y = nn.Parameter(torch.zeros(1, num_patches, y_feat_dim), requires_grad=False)
        else:
            self.pos_embed_y = None

        # Image and stream tokens are concatenated along the channel dim, then projected
        # back to the model dim (this is PointDiT's conditioning mechanism).
        self.concat_proj = nn.Linear(y_feat_dim + hidden_size, hidden_size, bias=True)

        # transformer
        self.blocks = nn.ModuleList([
            StreamBlock(hidden_size, num_heads, stream_attention, mlp_ratio=mlp_ratio,
                        attn_drop=attn_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                        proj_drop=proj_drop if (depth // 4 * 3 > i >= depth // 4) else 0.0,
                        attention_type=attention_type,
                        )
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, POINT_MAP_CHANNELS)

        self.initialize_weights()

        # The normal and the instance stream start as copies of the point stream's input, output and time layers.
        for name in STREAM_MODULES:
            for prefix in STREAM_PREFIXES[1:]:
                setattr(self, prefix + name, copy.deepcopy(getattr(self, name)))

        self.dino_grounding = dino_grounding
        if dino_grounding == 'adaln':
            # PixelDiT2 spatial adaLN: g_j joins the condition of each grounded token; zero last layer, so g = 0
            self.grounding_proj = nn.Sequential(
                nn.Linear(y_feat_dim, hidden_size), nn.SiLU(), nn.Linear(hidden_size, hidden_size))
            nn.init.zeros_(self.grounding_proj[-1].weight)
            nn.init.zeros_(self.grounding_proj[-1].bias)
        elif dino_grounding == 'attention':
            # per block: extra query and key dims per head from g_j; zero query layers, so no logit changes
            self.grounding_proj = nn.Sequential(
                nn.LayerNorm(y_feat_dim), nn.Linear(y_feat_dim, dino_grounding_rank))
            width = num_heads * dino_grounding_head_dim
            self.grounding_queries = nn.ModuleList([nn.Linear(dino_grounding_rank, width) for _ in range(depth)])
            self.grounding_keys = nn.ModuleList([nn.Linear(dino_grounding_rank, width) for _ in range(depth)])
            for query in self.grounding_queries:
                nn.init.zeros_(query.weight)
                nn.init.zeros_(query.bias)

        self.camera_positional_encoding = None
        if camera_positional_encoding:
            self.camera_positional_encoding = CameraPositionalEncoding(
                reference_focal, patch_size, hidden_size, num_heads, camera_true_prob,
                training_grid=input_size // patch_size, sites=camera_pe_sites,
            )
        self.y_embedder.eval()

    def train(self, mode=True):
        super().train(mode)
        self.y_embedder.eval()
        return self

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if self.pos_embed_y is not None:
            # Initialize (and freeze) pos_embed_y by sin-cos embedding:
            pos_embed_y = get_2d_sincos_pos_embed(self.pos_embed_y.shape[-1], int(self.x_embedder.num_patches ** 0.5))
            self.pos_embed_y.data.copy_(torch.from_numpy(pos_embed_y).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w1 = self.x_embedder.proj1.weight.data
        nn.init.xavier_uniform_(w1.view([w1.shape[0], -1]))
        w2 = self.x_embedder.proj2.weight.data
        nn.init.xavier_uniform_(w2.view([w2.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj2.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-initialize the output head.
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    def interpolate_pos_encoding(self, h, w):
        """
        Interpolate position embeddings to handle variable input sizes.
        h, w: number of patches in height and width
        Returns: pos_embed [1, h*w, C]
        """
        N = h * w
        if N == self.pos_embed.shape[1]:
            # Check if it's the same square size
            orig_size = int(self.pos_embed.shape[1] ** 0.5)
            if h == orig_size and w == orig_size:
                return self.pos_embed

        # Check cache
        cache_key = (h, w)
        if not hasattr(self, '_pos_embed_cache'):
            self._pos_embed_cache = {}
        if cache_key in self._pos_embed_cache:
            return self._pos_embed_cache[cache_key]

        # Get original grid size (square)
        orig_size = int(self.pos_embed.shape[1] ** 0.5)

        # Reshape to 2D grid
        pos_embed = self.pos_embed.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)

        # Interpolate to new size
        pos_embed = F.interpolate(pos_embed, size=(h, w), mode='bicubic', align_corners=False)

        # Reshape back
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)

        if len(self._pos_embed_cache) > 64:
            self._pos_embed_cache.clear()
        self._pos_embed_cache[cache_key] = pos_embed
        return pos_embed

    def interpolate_pos_encoding_y(self, h, w):
        """
        Interpolate y position embeddings (for image features) to handle variable input sizes.
        h, w: number of patches in height and width
        Returns: pos_embed_y [1, h*w, C]
        """
        if self.pos_embed_y is None:
            return self.interpolate_pos_encoding(h, w)

        N = h * w
        if N == self.pos_embed_y.shape[1]:
            orig_size = int(self.pos_embed_y.shape[1] ** 0.5)
            if h == orig_size and w == orig_size:
                return self.pos_embed_y

        # Check cache
        cache_key = (h, w)
        if not hasattr(self, '_pos_embed_y_cache'):
            self._pos_embed_y_cache = {}
        if cache_key in self._pos_embed_y_cache:
            return self._pos_embed_y_cache[cache_key]

        orig_size = int(self.pos_embed_y.shape[1] ** 0.5)
        pos_embed = self.pos_embed_y.reshape(1, orig_size, orig_size, -1).permute(0, 3, 1, 2)
        pos_embed = F.interpolate(pos_embed, size=(h, w), mode='bicubic', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, -1)

        if len(self._pos_embed_y_cache) > 64:
            self._pos_embed_y_cache.clear()
        self._pos_embed_y_cache[cache_key] = pos_embed
        return pos_embed

    def get_rope_for_size(self, h, w, device):
        """
        Generate rotary position embeddings for given grid size (cached).
        h, w: number of patches in height and width
        Returns: (cos, sin) each of shape [h*w, D]
        """
        cache_key = (h, w, device)
        if not hasattr(self, '_rope_cache'):
            self._rope_cache = {}
        if cache_key in self._rope_cache:
            return self._rope_cache[cache_key]

        half_head_dim = self.hidden_size // self.num_heads // 2
        theta = 10000
        dim = half_head_dim
        freqs = 1. / (theta ** (torch.arange(0, dim, 2, device=device)[:(dim // 2)].float() / dim))

        # Create position indices for h and w
        t_h = torch.arange(h, device=device).float()
        t_w = torch.arange(w, device=device).float()

        # Compute frequencies for each dimension
        freqs_h = torch.einsum('i,j->ij', t_h, freqs)  # [h, D/4]
        freqs_w = torch.einsum('i,j->ij', t_w, freqs)  # [w, D/4]

        # Repeat for complex representation (interleave)
        freqs_h = freqs_h.repeat_interleave(2, dim=-1)  # [h, D/2]
        freqs_w = freqs_w.repeat_interleave(2, dim=-1)  # [w, D/2]

        # Create 2D grid: [h, w, D]
        freqs_grid = torch.cat([
            freqs_h[:, None, :].expand(-1, w, -1),
            freqs_w[None, :, :].expand(h, -1, -1)
        ], dim=-1)

        # Flatten to [h*w, D]
        freqs_flat = freqs_grid.reshape(h * w, -1)

        result = (freqs_flat.cos(), freqs_flat.sin())
        if len(self._rope_cache) > 64:
            self._rope_cache.clear()
        self._rope_cache[cache_key] = result
        return result

    def unpatchify(self, x, h, w):
        """
        x: (N, h * w, patch_size**2 * C)
        h, w: number of patches
        imgs: (N, C, h * patch_size, w * patch_size)
        """
        p = self.patch_size
        c = x.shape[-1] // (p * p)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        return x.reshape(shape=(x.shape[0], c, h * p, w * p))

    def forward(self, x, t, y, intrinsics=None, cached_y_emb=None):
        """
        x: (N, 9, H, W) states of the point, normal and instance streams
        t: (N, 3) one time per stream
        y: (N, 3, H, W) image in [-1, 1]
        intrinsics: (N, 4) pinhole parameters (fx, fy, cx, cy) of the input; None for an unknown camera
        cached_y_emb: Optional pre-computed y embedding [N, L, C] for sampling efficiency
        Returns the (N, 9, H, W) clean prediction of the three streams.
        """
        B, C, H, W = x.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        if self.camera_positional_encoding is not None:
            pos_embed, image_position, feat_rope, camera_condition = self.camera_positional_encoding(
                intrinsics, B, (h_patches, w_patches), (H, W), self.y_embedder.num_features,
            )
        else:
            pos_embed = self.interpolate_pos_encoding(h_patches, w_patches)
            rope_cos, rope_sin = self.get_rope_for_size(h_patches, w_patches, x.device)
            feat_rope = DynamicRoPE(rope_cos, rope_sin)

        # one time embedding per stream
        c = torch.stack([getattr(self, prefix + 't_embedder')(t[:, index])
                         for index, prefix in enumerate(STREAM_PREFIXES)], dim=1)

        # Reuse the image embedding throughout the sampling trajectory.
        y_emb = cached_y_emb if cached_y_emb is not None else self.extract_y_embedding(y, h_patches, w_patches)
        if self.camera_positional_encoding is not None:
            y_emb = y_emb + image_position
            c = c + camera_condition.to(c.dtype)[:, None, :]

        # concat image and stream tokens along the channel dim, then project to the model dim
        tokens = []
        for prefix, state in zip(STREAM_PREFIXES, x.chunk(len(STREAMS), dim=1)):
            embedded = getattr(self, prefix + 'x_embedder')(state) + pos_embed
            tokens.append(getattr(self, prefix + 'concat_proj')(torch.cat((y_emb, embedded), dim=-1)))
        hidden = torch.cat(tokens, dim=1)

        stream_conditions = list(c.unbind(1))
        token_conditions = None
        if self.dino_grounding != 'none':
            grounding = self.grounding_proj(y_emb)
        if self.dino_grounding == 'adaln':
            token_conditions = {index: c[:, index, None] + grounding for index in GROUNDED_STREAMS}
            stream_conditions = [token_conditions.get(index, stream) for index, stream in enumerate(stream_conditions)]
        for depth, block in enumerate(self.blocks):
            affinity = None
            if self.dino_grounding == 'attention':
                affinity = (self.grounding_queries[depth](grounding), self.grounding_keys[depth](grounding),
                            GROUNDED_STREAMS)
            hidden = block(hidden, c, feat_rope, token_conditions, affinity)

        outputs = []
        for index, (prefix, stream) in enumerate(zip(STREAM_PREFIXES, hidden.chunk(len(STREAMS), dim=1))):
            patches = getattr(self, prefix + 'final_layer')(stream, stream_conditions[index])
            outputs.append(self.unpatchify(patches, h_patches, w_patches))
        return torch.cat(outputs, dim=1)

    def extract_y_embedding(self, y, h_patches, w_patches):
        """
        Extract DINOv3 feature embeddings from conditioning image.
        Used for caching during sampling to avoid redundant feature extraction.

        Args:
            y: Conditioning image tensor [B, 3, H, W] in range [-1, 1]
            h_patches: Number of patches in height (for position embedding)
            w_patches: Number of patches in width (for position embedding)

        Returns:
            y_emb: Feature embeddings [B, num_patches, feature_dim]
        """
        # DINOv3 image normalization
        y = (y + 1) / 2.  # to [0, 1]
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(y.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(y.device)
        y = (y - mean) / std

        if self.patch_size != 16:
            downsample_factor = self.patch_size // 16
            y = F.interpolate(y, scale_factor=1.0 / downsample_factor, mode='bilinear', align_corners=False)

        if self.dinov3_use_intermediate_layers:
            intermediate_features = self.y_embedder.get_intermediate_layers(
                y, n=self.dinov3_intermediate_layer_indices, norm=True,
            )
            y_emb = torch.cat(intermediate_features, dim=-1)
        else:
            y_emb = self.y_embedder.forward_features(y)['x_norm_patchtokens']

        # The camera positional encoding adds its own position to the image features in forward.
        if self.camera_positional_encoding is None:
            y_emb = y_emb + self.interpolate_pos_encoding_y(h_patches, w_patches)
        return y_emb


def PINE_B_16(**kwargs):
    return PINE(depth=12, hidden_size=768, num_heads=12, bottleneck_dim=128, patch_size=16, **kwargs)


def PINE_L_16(**kwargs):
    return PINE(depth=24, hidden_size=1024, num_heads=16, bottleneck_dim=128, patch_size=16, **kwargs)


def PINE_H_16(**kwargs):
    return PINE(depth=32, hidden_size=1280, num_heads=16, bottleneck_dim=256, patch_size=16, **kwargs)


# The /16 suffix is the patch size; main.py parses it back out of --model.
PINE_models = {
    'PINE-B/16': PINE_B_16,
    'PINE-L/16': PINE_L_16,
    'PINE-H/16': PINE_H_16,
}
