# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Load a PINE training checkpoint for evaluation and inference."""
import torch

from denoiser import Denoiser
from main import get_args_parser
from util.paths import dinov3_weights_path

# the training arguments that define the network and its sampler
model_arguments = ('model', 'img_size', 'feature_embedding_type', 'dinov3_use_intermediate_layers',
                   'dinov3_num_intermediate_layers', 'attention_type', 'camera_positional_encoding', 'reference_focal',
                   'camera_true_prob', 'camera_pe_sites', 'dino_grounding', 'dino_grounding_rank',
                   'dino_grounding_head_dim', 'noise_scale', 'generate_noise_scale', 't_eps', 'sample_t_eps')


def load_model(checkpoint_path, weights='ema', num_sampling_steps=1, device='cuda'):
    """Denoiser in eval mode with the 'ema' (model_ema1) or 'raw' (model) weights of the checkpoint, built from the
    arguments the checkpoint was trained with. The frozen DINOv3 encoder is not part of a checkpoint; its weights come
    from $DINOV3_WEIGHTS_DIR."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved = vars(checkpoint['args'])
    args = get_args_parser().parse_args([])
    for name in model_arguments:
        setattr(args, name, saved[name])
    # checkpoints trained before the release named the model PointDiT and the stream mask joint_geometry_attention
    args.model = args.model.replace('PointDiT', 'PINE')
    args.stream_attention = saved.get('stream_attention', saved.get('joint_geometry_attention'))
    args.num_sampling_steps = num_sampling_steps
    model = Denoiser(args)
    encoder = torch.load(dinov3_weights_path(args.feature_embedding_type), map_location='cpu')
    model.net.y_embedder.load_state_dict(encoder, strict=True)
    state = checkpoint['model_ema1' if weights == 'ema' else 'model']
    incompatible = model.load_state_dict(state, strict=False)
    assert all(key.startswith('net.y_embedder.') for key in incompatible.missing_keys), incompatible.missing_keys
    assert not incompatible.unexpected_keys, incompatible.unexpected_keys
    return model.to(device).eval()
