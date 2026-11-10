# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Surface-normal angular-error metrics as reported by DSINE, accumulated as reducible statistics.

Per-pixel angles are folded into [count, sum, sum of squares, histogram]; mean, RMSE and the threshold shares come
out exactly, the median at the histogram resolution."""
import torch
import torch.nn.functional as F

HISTOGRAM_BIN_DEGREES = 0.05
HISTOGRAM_BINS = int(180.0 / HISTOGRAM_BIN_DEGREES)
THRESHOLDS_DEGREES = (5.0, 7.5, 11.25, 22.5, 30.0)
METRIC_NAMES = ('mean', 'median', 'rmse') + tuple(f'below_{threshold:g}' for threshold in THRESHOLDS_DEGREES)


def resize_benchmark_normals(prediction, size, camera_facing=True):
    """Resize a prediction to the benchmark grid, convert its frame, and normalize its vectors."""
    prediction = F.interpolate(prediction.float(), size=size, mode='bilinear', align_corners=False)
    if camera_facing:
        prediction = -prediction
    return F.normalize(prediction, dim=1)


def full_frame_angular_errors(prediction, target, valid, camera_facing=True):
    """Score native benchmark GT using DSINE cosine similarity after prediction resizing and frame conversion."""
    prediction = resize_benchmark_normals(prediction, target.shape[-2:], camera_facing)
    cosine = F.cosine_similarity(prediction, target.float(), dim=1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cosine))[valid]


def error_statistics(errors):
    """[3 + HISTOGRAM_BINS] float64: count, sum, sum of squares, then the histogram of the angles."""
    errors = errors.double()
    histogram = torch.histc(errors, bins=HISTOGRAM_BINS, min=0.0, max=180.0)
    return torch.cat((torch.stack((errors.numel() * torch.ones_like(errors[:1].sum()), errors.sum(), errors.square().sum())), histogram))


def normal_metrics(statistics):
    count, total, total_squared = statistics[:3].tolist()
    cumulative = statistics[3:].cumsum(0)
    median_bin = int(torch.searchsorted(cumulative, torch.tensor(count / 2.0, dtype=cumulative.dtype, device=cumulative.device)))
    metrics = {'n_pixels': int(count), 'mean': total / count, 'rmse': (total_squared / count) ** 0.5,
               'median': (median_bin + 0.5) * HISTOGRAM_BIN_DEGREES}
    for threshold in THRESHOLDS_DEGREES:
        metrics[f'below_{threshold:g}'] = 100.0 * float(cumulative[round(threshold / HISTOGRAM_BIN_DEGREES) - 1]) / count
    return metrics


def format_metrics(name, metrics):
    thresholds = '  '.join(f"<{threshold:g} {metrics[f'below_{threshold:g}']:5.1f}" for threshold in THRESHOLDS_DEGREES)
    return f"[{name}] normal  mean {metrics['mean']:6.2f}  median {metrics['median']:6.2f}  rmse {metrics['rmse']:6.2f}  {thresholds}"
