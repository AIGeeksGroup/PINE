# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Planarity and orientation error of a PINE checkpoint on iBims-1 with the official evaluation code.

Needs the official ibims1_core_mat files (--mat-dir), their image list (--image-list) and the folder of the official
evaluate_ibims_error_metrics.py (--official-code). The model runs without a camera on the full frame at a short side of
512. The depth is aligned to the ground truth by a scale and a shift over the valid pixels; then the official
compute_planarity_error runs unchanged. Writes planarity.json and per_plane.csv to --output-dir."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import io

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from util.checkpoint import load_model  # noqa: E402


def read_sample(path):
    data = io.loadmat(path)['data']
    return {key: data[key][0][0] for key in data.dtype.names}


@torch.no_grad()
def predict_depth(model, rgb, index):
    """[480, 640] depth of an [480, 640, 3] uint8 image."""
    image = torch.from_numpy(rgb).permute(2, 0, 1)[None].float().cuda() / 255
    scale = 512 / min(image.shape[-2:])
    size = [round(side * scale / 16) * 16 for side in image.shape[-2:]]
    resized = F.interpolate(image, size=size, mode='bilinear', align_corners=False)
    torch.manual_seed(index)
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        depth = model.generate(resized * 2 - 1, None, task='pointmap')[:, 2:3].float()
    return F.interpolate(depth, size=image.shape[-2:], mode='bilinear', align_corners=False)[0, 0].cpu().numpy()


def align(prediction, target, valid):
    """The prediction mapped onto the target by least squares (scale and shift) over the valid pixels."""
    source = prediction[valid]
    scale, shift = np.linalg.lstsq(np.stack((source, np.ones_like(source)), 1), target[valid], rcond=None)[0]
    return prediction * scale + shift


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mat-dir', type=Path, required=True)
    parser.add_argument('--image-list', type=Path, required=True)
    parser.add_argument('--official-code', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.official_code))
    from evaluate_ibims_error_metrics import compute_planarity_error
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(args.checkpoint, 'ema', num_sampling_steps=1)
    rows = []
    errors = {kind: ([], []) for kind in ('wall', 'table', 'floor')}
    for index, name in enumerate(args.image_list.read_text().split()):
        sample = read_sample(args.mat_dir / f'{name}.mat')
        gt = sample['depth'].astype(np.float64)
        valid = (sample['mask_transp'] * sample['mask_invalid'] * (gt != 0)).astype(bool)
        depth = align(predict_depth(model, sample['rgb'], index).astype(np.float64), gt, valid)
        mask_valid = valid.astype(np.float64)
        for kind, (flat, orientation) in errors.items():
            paras = sample[f'mask_{kind}_paras']
            if paras.size == 0:
                continue
            pe_flat, pe_orientation = compute_planarity_error((gt * mask_valid).copy(), (depth * mask_valid).copy(),
                                                              paras, sample[f'mask_{kind}'] * mask_valid,
                                                              sample['calib'])
            flat.extend(pe_flat)
            orientation.extend(pe_orientation)
            rows += [[name, kind, plane, f'{a:.4f}', f'{b:.4f}']
                     for plane, (a, b) in enumerate(zip(pe_flat, pe_orientation))]
    every = [np.concatenate([np.array(errors[kind][part]) for kind in errors]) for part in (0, 1)]
    summary = dict(checkpoint=args.checkpoint, planarity_cm=float(np.nanmean(every[0])),
                   orientation_deg=float(np.nanmean(every[1])), planes=int(np.isfinite(every[0]).sum()),
                   per_type={kind: dict(planarity_cm=float(np.nanmean(flat)),
                                        orientation_deg=float(np.nanmean(orientation)),
                                        planes=int(np.isfinite(flat).sum()))
                             for kind, (flat, orientation) in errors.items()})
    print(json.dumps(summary), flush=True)
    (args.output_dir / 'planarity.json').write_text(json.dumps(summary, indent=1))
    with (args.output_dir / 'per_plane.csv').open('w', newline='') as handle:
        csv.writer(handle).writerows([['image', 'type', 'plane', 'planarity_cm', 'orientation_deg'], *rows])


if __name__ == '__main__':
    main()
