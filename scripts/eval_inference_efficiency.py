# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Inference cost of a PINE checkpoint: parameters, latency and peak GPU memory for one 512 x 512 image at batch size 1.
A timed run is one network pass (point map, normals and instance map) and the mask of one prompt point.
Writes <output-dir>/efficiency.json."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader.eval_depth import EvalDepthDataset  # noqa: E402
from loss import gen2seg_inference_scale  # noqa: E402
from metrics.instance_prompt import PROMPT, instance_map_uint8, prompt_masks  # noqa: E402
from util.checkpoint import load_model  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='datasets/eval')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image = EvalDepthDataset(args.data_root, 'DIODE', crop_size=512, resize_height=512)[0]['image']
    source = image[None].cuda() * 2 - 1
    centre = torch.tensor([[256, 256]], device='cuda')
    model = load_model(args.checkpoint, 'ema', num_sampling_steps=1)
    weights = torch.cuda.memory_allocated() / 1e9

    @torch.no_grad()
    def run():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            instance = model.generate(source, None)[:, 6:].float()
        prompt_masks(instance_map_uint8(gen2seg_inference_scale(instance[0]).permute(1, 2, 0)), centre, **PROMPT)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    record = dict(checkpoint=args.checkpoint, gpu=torch.cuda.get_device_name(),
                  parameters=sum(parameter.numel() for parameter in model.parameters()),
                  latency_ms_median=statistics.median(times), latency_ms_mean=statistics.mean(times),
                  latency_ms_std=statistics.stdev(times), weights_gb=weights,
                  peak_gb=torch.cuda.max_memory_allocated() / 1e9, repeats=args.repeats)
    print(json.dumps(record), flush=True)
    (args.output_dir / 'efficiency.json').write_text(json.dumps(record, indent=1))


if __name__ == '__main__':
    main()
