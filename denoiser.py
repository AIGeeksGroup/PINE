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

# Derived from JiT (https://github.com/LTH14/JiT), MIT License,
# Copyright (c) 2025 Tianhong Li. See THIRD_PARTY_NOTICES.
import torch
import torch.nn as nn
from model import GEOMETRY_STREAMS, PINE_models, POINT_MAP_CHANNELS, STREAMS


def masked_mean(values, mask):
    """Mean of the selected entries; zero when nothing is selected, so an all-sky or sky-free batch has no such term."""
    if mask.any():
        return values[mask].mean()
    return values.new_zeros(())


class Denoiser(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        self.net = PINE_models[args.model](
            input_size=args.img_size,
            attn_drop=args.attn_dropout,
            proj_drop=args.proj_dropout,
            attention_type=args.attention_type,
            feature_embedding_type=args.feature_embedding_type,
            dinov3_use_intermediate_layers=args.dinov3_use_intermediate_layers,
            dinov3_num_intermediate_layers=args.dinov3_num_intermediate_layers,
            finetune_feature_embedding=args.feature_embedding_lr_scale > 0,
            stream_attention=args.stream_attention,
            camera_positional_encoding=args.camera_positional_encoding,
            reference_focal=args.reference_focal,
            camera_true_prob=args.camera_true_prob,
            camera_pe_sites=tuple(args.camera_pe_sites),
            dino_grounding=args.dino_grounding,
            dino_grounding_rank=args.dino_grounding_rank,
            dino_grounding_head_dim=args.dino_grounding_head_dim,
        )
        self.P_mean = args.P_mean
        self.P_std = args.P_std
        self.t_eps = args.t_eps
        self.noise_scale = args.noise_scale

        # ema
        self.ema_decay1 = args.ema_decay1
        self.ema_params1 = None

        # generation hyper params
        self.steps = args.num_sampling_steps

        self.args = args

    def sample_t(self, n: int, device=None):

        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def image_embedding(self, image):
        """DINOv3 embedding of the conditioning image, computed once per image and reused over the sampling steps."""
        height, width = image.shape[-2:]
        return self.net.extract_y_embedding(image, height // self.net.patch_size, width // self.net.patch_size)

    def forward(self, x, labels, intrinsics=None,
                return_model_input_output=False,
                force_zero_t=False,
                args=None,
                target_valid=None,
                target_sky=None,
                ):
        """x: (N, 9, H, W) point map, normal map and three unused instance channels; labels: the image in [-1, 1];
        target_valid, target_sky: per-channel masks of x. Returns the velocity loss of the point and normal streams."""
        batch_size = x.size(0)

        # logit-normal timestep sampling: t = sigmoid(P_mean + P_std * eps), independent for the point and the normal
        # stream of every sample
        t = self.sample_t(batch_size * len(GEOMETRY_STREAMS), device=x.device).view(-1, *([1] * (x.ndim - 1)))

        if force_zero_t:
            # Force a fraction of the timesteps to be exactly 0.0
            if args.force_zero_t_ratio is not None:
                force_zero_t_ratio = args.force_zero_t_ratio
            else:
                force_zero_t_ratio = 0.1

            mask_zero = torch.rand(t.shape, device=t.device) < force_zero_t_ratio
            t[mask_zero] = 0.0

        # The instance stream has no target colours (its loss is permutation invariant): it is always run at t = 0
        # from a zero state, like the gen2seg one-step regression, and trained only by the instance loss.
        stream_timesteps = t.reshape(batch_size, -1)
        stream_timesteps = torch.cat((stream_timesteps, stream_timesteps.new_zeros(batch_size, 1)), dim=1)
        t = stream_timesteps.repeat_interleave(POINT_MAP_CHANNELS, dim=1)[:, :, None, None]

        e = torch.randn_like(x) * self.noise_scale

        if args.noise_fill_invalid:
            x_filled = torch.where(target_valid, x, e)
            z = t * x_filled + (1 - t) * e
        else:
            z = t * x + (1 - t) * e
        z = z.clone()
        z[:, len(GEOMETRY_STREAMS) * POINT_MAP_CHANNELS:] = 0

        x_pred = self.net(z, stream_timesteps, labels, intrinsics=intrinsics,
                          cached_y_emb=self.image_embedding(labels))
        denominator = (1 - t).clamp_min(self.t_eps)
        v = (x.float() - z.float()) / denominator
        v_pred = (x_pred.float() - z.float()) / denominator

        # velocity (v-) loss, averaged separately over the valid non-sky pixels and over the sky pixels of each map
        loss = (v - v_pred) ** 2
        point, normal = slice(0, POINT_MAP_CHANNELS), slice(POINT_MAP_CHANNELS, 2 * POINT_MAP_CHANNELS)
        point_term = masked_mean(loss[:, point], target_valid[:, point] & ~target_sky[:, point])
        point_term = point_term + args.sky_loss_weight * masked_mean(loss[:, point], target_sky[:, point])
        normal_term = masked_mean(loss[:, normal], target_valid[:, normal] & ~target_sky[:, normal])
        normal_term = normal_term + args.sky_loss_weight * masked_mean(loss[:, normal], target_sky[:, normal])
        loss = point_term + args.normal_loss_weight * normal_term

        if return_model_input_output:
            model_in_out = {
                't': t,
                'input': z,
                'output': x_pred,
                'denoise_loss': loss,
            }
            return loss, model_in_out

        return loss

    @torch.no_grad()
    def generate(self, labels, intrinsics=None, return_intermediate_steps=False, task=None, *, pointmap=None,
                 normal=None):
        """Predict the maps of the image labels (N, 3, H, W) in [-1, 1].

        task None returns the (N, 9, H, W) point map, normal map and instance map; task 'pointmap', 'normal' or
        'instance' returns that (N, 3, H, W) map. A given pointmap or normal conditions the other map and is returned
        unchanged. With return_intermediate_steps, also returns {step: prediction} of the point and normal maps (-1 is
        the initial state)."""
        if pointmap is not None and normal is not None:
            raise ValueError('Supply one geometry condition and predict the other modality')
        batch_size, _, height, width = labels.shape
        geometry_channels = len(GEOMETRY_STREAMS) * POINT_MAP_CHANNELS
        state = labels.new_zeros(batch_size, len(STREAMS) * POINT_MAP_CHANNELS, height, width)
        if self.args.generate_noise_scale != 0:
            state[:, :geometry_channels].normal_(std=self.args.generate_noise_scale)
        timestep = labels.new_zeros(batch_size, len(STREAMS))
        if pointmap is not None:
            state[:, :POINT_MAP_CHANNELS] = pointmap
            timestep[:, 0] = 1
        if normal is not None:
            state[:, POINT_MAP_CHANNELS:geometry_channels] = normal
            timestep[:, 1] = 1
        # Streams at time zero are generated with x-prediction Euler steps; a given stream stays clean at time one.
        # The last step lands on t = 1, where the Euler update equals the prediction, so the prediction is returned.
        generated = (timestep == 0).repeat_interleave(POINT_MAP_CHANNELS, dim=1)[:, :, None, None]
        times = torch.linspace(0.0, 1.0, self.steps + 1, device=labels.device)
        initial_state = state[:, :geometry_channels]
        image_embedding = self.image_embedding(labels)
        step_predictions = []
        instance_prediction = None
        for time, next_time in zip(times[:-1], times[1:]):
            step_timestep = torch.where(timestep == 0, time, timestep)
            prediction = self.net(
                state, step_timestep, labels, intrinsics=intrinsics, cached_y_emb=image_embedding,
            ).float()
            step_predictions.append(prediction[:, :geometry_channels])
            if instance_prediction is None:
                instance_prediction = prediction[:, geometry_channels:]  # the first step: t = 0, zero state
            if next_time < 1:
                velocity = (prediction - state) / (1 - time).clamp_min(self.args.sample_t_eps)
                state = torch.where(generated, state + (next_time - time) * velocity, state).to(state.dtype)
        if task == 'instance':
            # the instance stream, run from its zero state at t = 0 like in training (one step)
            return instance_prediction
        prediction = step_predictions[-1]
        state = initial_state
        if pointmap is not None:
            prediction = torch.cat((state[:, :POINT_MAP_CHANNELS], prediction[:, POINT_MAP_CHANNELS:]), dim=1)
        if normal is not None:
            prediction = torch.cat((prediction[:, :POINT_MAP_CHANNELS], state[:, POINT_MAP_CHANNELS:]), dim=1)
        if task is not None:
            index = GEOMETRY_STREAMS.index(task)
            prediction = prediction.chunk(len(GEOMETRY_STREAMS), dim=1)[index]
            state = state.chunk(len(GEOMETRY_STREAMS), dim=1)[index]
            step_predictions = [step.chunk(len(GEOMETRY_STREAMS), dim=1)[index] for step in step_predictions]
        intermediate = {-1: state, **dict(enumerate(step_predictions[:-1])), len(step_predictions) - 1: prediction}
        if task is None:
            prediction = torch.cat((prediction, instance_prediction), dim=1)
        if return_intermediate_steps:
            return prediction, intermediate
        return prediction

    @torch.no_grad()
    def update_ema(self):
        for targ, src in zip(self.ema_params1, self.parameters()):
            if src.requires_grad:
                targ.detach().mul_(self.ema_decay1).add_(src, alpha=1 - self.ema_decay1)
