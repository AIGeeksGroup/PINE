# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Evaluate the normals of a PINE checkpoint with the full-frame DSINE protocol on NYUv2, ScanNet, iBims-1 and Sintel
(engine.evaluate_normals), with the dataset intrinsics (known) and without a camera (unknown). The benchmark files and
split lists come from $DSINE_DATA_ROOT and $DSINE_SPLIT_ROOT. Writes <output-dir>/normal_benchmarks.json."""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from engine import evaluate_normals  # noqa: E402
from util.checkpoint import load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', default='ema', choices=['ema', 'raw'])
    parser.add_argument('--num-sampling-steps', type=int, default=1)
    parser.add_argument('--benchmarks', default='nyuv2,scannet,ibims,sintel')
    parser.add_argument('--max-samples', type=int, default=None)
    parser.add_argument('--camera-modes', nargs='+', choices=['known', 'unknown'], default=['known', 'unknown'])
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    model = load_model(args.checkpoint, args.weights, args.num_sampling_steps)
    model.args.eval_normal_benchmarks = args.benchmarks
    model.args.eval_normal_max_samples = args.max_samples
    model.args.seed = args.seed
    results = {}
    for camera_mode in args.camera_modes:
        torch.manual_seed(args.seed)
        results[camera_mode] = evaluate_normals(model, model.args, 0, use_camera=camera_mode == 'known')
    with open(os.path.join(args.output_dir, 'normal_benchmarks.json'), 'w') as f:
        json.dump({'checkpoint': args.checkpoint, 'weights': args.weights,
                   'num_sampling_steps': model.steps, 'generate_noise_scale': model.args.generate_noise_scale,
                   'resize_short_edge': model.args.img_size, 'seed': args.seed, 'results': results}, f, indent=2)


if __name__ == '__main__':
    main()
