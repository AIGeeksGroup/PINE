# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Field of view that a PINE checkpoint recovers from one image, on the 512 crops of the point map protocol.

The model gets no intrinsics. util/self_calibration.estimate_intrinsics fits the focal length of its point map, with
the principal point at the image centre. The ground truth is the fx of the crop intrinsics. The scores per dataset are
the mean and median absolute error of the horizontal field of view in degrees and the median |log(f / f_GT)|.
Writes metrics.json and per_sample.csv to --output-dir.
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader.eval_depth import EvalDepthDataset  # noqa: E402
from util.checkpoint import load_model  # noqa: E402
from util.self_calibration import estimate_intrinsics  # noqa: E402

eval_sets = ('DIODE', 'KITTI', 'NYUv2', 'ETH3D', 'HAMMER', 'iBims-1', 'Booster')


def fov(focal, width):
    return math.degrees(2 * math.atan(width / (2 * focal)))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-root', default='datasets/eval')
    parser.add_argument('--datasets', nargs='+', choices=eval_sets, default=list(eval_sets))
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, 'ema', num_sampling_steps=1)
    results, all_errors = {}, []
    with (args.output_dir / 'per_sample.csv').open('w') as handle:
        writer = csv.writer(handle)
        writer.writerow(['dataset', 'sample_id', 'focal', 'focal_gt', 'fov', 'fov_gt'])
        for name in args.datasets:
            dataset = EvalDepthDataset(args.data_root, name, crop_size=512, resize_height=512)
            rows = []
            for index in range(len(dataset)):
                sample = dataset[index]
                image = sample['image'].float()
                width = image.shape[-1]
                target = float(sample['intrinsics'][0, 0])
                torch.manual_seed(args.seed + index)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    points = model.generate(image[None].cuda() * 2 - 1, None, task='pointmap')
                predicted = float(estimate_intrinsics(points)[0, 0])
                rows.append((predicted, target, fov(predicted, width), fov(target, width)))
                writer.writerow([name, sample['sample_id'], *rows[-1]])
            rows = np.array(rows)
            fov_error = np.abs(rows[:, 2] - rows[:, 3])
            all_errors.extend(fov_error)
            results[name] = dict(fov_error_mean=float(fov_error.mean()), fov_error_median=float(np.median(fov_error)),
                                 log_focal_error_median=float(np.median(np.abs(np.log(rows[:, 0] / rows[:, 1])))),
                                 focal_ratio_median=float(np.median(rows[:, 0] / rows[:, 1])),
                                 fov_gt_median=float(np.median(rows[:, 3])), count=len(rows))
            print(name, json.dumps({k: round(v, 3) for k, v in results[name].items()}), flush=True)
    errors = np.array(all_errors)
    results['all'] = dict(fov_error_mean=float(errors.mean()), fov_error_median=float(np.median(errors)),
                          count=len(errors))
    print('all', json.dumps({k: round(v, 3) for k, v in results['all'].items()}), flush=True)
    record = dict(checkpoint=args.checkpoint, results=results)
    (args.output_dir / 'metrics.json').write_text(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
