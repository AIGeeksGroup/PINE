# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Evaluate the point map and the depth of a PINE checkpoint with the PointDiT protocol.

Every image of DIODE, KITTI, NYUv2, ETH3D, HAMMER, iBims-1 and Booster (3,444 in total) is resized to a short side of
512 and centre-cropped to 512 x 512. The prediction is aligned to the ground truth by a scale and a 3D translation
(metrics/compute_metrics.py). Camera mode known gives the model the dataset intrinsics; unknown gives it no camera.
Writes <output-dir>/metrics.json (per dataset and the sample-weighted average: Rel^p, delta_1^p, Rel^d, delta_1^d, and
the scale-invariant boundary F1 on --boundary-datasets) and <output-dir>/<mode>_per_sample.csv.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader.eval_depth import EvalDepthDataset  # noqa: E402
from engine import evaluation_intrinsics  # noqa: E402
from metrics.compute_metrics import compute_metrics  # noqa: E402
from util.checkpoint import load_model  # noqa: E402

eval_sets = ('DIODE', 'KITTI', 'NYUv2', 'ETH3D', 'HAMMER', 'iBims-1', 'Booster')


def parameter_counts(model):
    components = {name: sum(parameter.numel() for parameter in module.parameters())
                  for name, module in model.net.named_children()}
    total = sum(parameter.numel() for parameter in model.parameters())
    encoder = components['y_embedder']
    return dict(total=total, dinov3=encoder, generator=total - encoder, components=components)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--weights', choices=['ema', 'raw'], default='ema')
    parser.add_argument('--num-sampling-steps', type=int, default=1)
    parser.add_argument('--data-root', default='datasets/eval')
    parser.add_argument('--datasets', nargs='+', choices=eval_sets, default=list(eval_sets))
    parser.add_argument('--boundary-datasets', nargs='+', choices=eval_sets, default=['HAMMER', 'iBims-1', 'Booster'])
    parser.add_argument('--camera-modes', nargs='+', default=['known', 'unknown'], choices=['known', 'unknown'])
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, args.weights, args.num_sampling_steps)
    counts = parameter_counts(model)
    print(f'Parameters: {json.dumps(counts)}', flush=True)
    record = dict(checkpoint=str(Path(args.checkpoint).resolve()), weights=args.weights,
                  num_sampling_steps=model.steps, noise_scale=model.args.generate_noise_scale,
                  resolution=[512, 512], seed=args.seed, parameters=counts,
                  boundary_datasets=args.boundary_datasets, metrics={})
    fields = ['dataset', 'sample_id', 'n_valid', 'rel_p', 'd1_p', 'rel_d', 'd1_d', 'si_boundary_f1']
    for mode in args.camera_modes:
        datasets = {}
        with (args.output_dir / f'{mode}_per_sample.csv').open('w') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for name in args.datasets:
                compute_boundary = name in args.boundary_datasets
                dataset = EvalDepthDataset(args.data_root, name, crop_size=512, resize_height=512)
                loader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers)
                totals = dict.fromkeys(['rel_p', 'd1_p', 'rel_d', 'd1_d'], 0.0)
                boundary_total = 0.0
                count = 0
                for index, batch in enumerate(loader):
                    sample = {key: value[0] for key, value in batch.items()}
                    image = sample['image'][None].cuda() * 2 - 1
                    intrinsics = None
                    if mode == 'known':
                        size = image.shape[-2:]
                        intrinsics = evaluation_intrinsics(sample['intrinsics'][None].cuda(), size, size)
                    torch.manual_seed(args.seed + index)
                    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                        prediction = model.generate(image, intrinsics, task='pointmap').float()
                    target = sample['pointcloud'][None].cuda()
                    valid = sample['valid_mask'][None].cuda()
                    n_valid = int(valid.sum())
                    row = dict(dataset=name, sample_id=sample['sample_id'], n_valid=n_valid)
                    if n_valid >= 100:
                        pred = dict(points_affine_invariant=prediction.permute(0, 2, 3, 1),
                                    depth_affine_invariant=prediction[:, 2])
                        gt = dict(points=target.permute(0, 2, 3, 1), depth=target[:, 2], depth_mask=valid)
                        scores, _ = compute_metrics(pred, gt, compute_si_boundary=compute_boundary)
                        for suffix, task in [('p', 'points_affine_invariant'), ('d', 'depth_affine_invariant')]:
                            row[f'rel_{suffix}'] = scores[task]['rel'] * 100
                            row[f'd1_{suffix}'] = scores[task]['delta1'] * 100
                        for metric in totals:
                            totals[metric] += row[metric]
                        if compute_boundary:
                            row['si_boundary_f1'] = scores['si_boundary_f1'] * 100
                            boundary_total += row['si_boundary_f1']
                        count += 1
                    writer.writerow(row)
                    if (index + 1) % 100 == 0:
                        handle.flush()
                        print(f'[{mode}/{name}] {index + 1}/{len(dataset)}', flush=True)
                datasets[name] = {key: value / count for key, value in totals.items()}
                datasets[name].update(count=count, total_samples=len(dataset))
                if compute_boundary:
                    datasets[name]['si_boundary_f1'] = boundary_total / count
                print(f'[{mode}/{name}] {datasets[name]}', flush=True)
                count_all = sum(item['count'] for item in datasets.values())
                average = {metric: sum(item[metric] * item['count'] for item in datasets.values()) / count_all
                           for metric in totals}
                average['count'] = count_all
                record['metrics'][mode] = dict(datasets=datasets, average=average)
                with (args.output_dir / 'metrics.json').open('w') as output:
                    json.dump(record, output, indent=2)


if __name__ == '__main__':
    main()
