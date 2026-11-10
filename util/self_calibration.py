# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Pinhole focal length implied by an affine-invariant point map.

The point map is defined up to a scale and a 3D shift; for a pinhole camera with principal point (c_x, c_y),
(x + s_x) / (z + s_z) = (u - c_x) / f and (y + s_y) / (z + s_z) = (v - c_y) / f.  For a fixed s_z this is a linear
least-squares problem in (1/f, s_x, s_y); s_z is found by a bounded 1-D search."""
import torch
from scipy.optimize import minimize_scalar


def implied_focal(points, valid, cx, cy):
    """(f, s_z, mean pixel residual) for a [3, H, W] point map; valid is a [H, W] bool mask."""
    points = points.double()
    height, width = valid.shape
    v, u = torch.meshgrid(torch.arange(height, dtype=torch.float64, device=points.device),
                          torch.arange(width, dtype=torch.float64, device=points.device), indexing='ij')
    du, dv = (u - cx)[valid], (v - cy)[valid]
    x, y, z = points[0][valid], points[1][valid], points[2][valid]
    ones = torch.ones_like(x)
    z_min = float(z.min())

    def solve(s_z):
        zs = z + s_z
        # x = du * zs * (1/f) - s_x ;  y = dv * zs * (1/f) - s_y
        design = torch.cat((torch.stack((du * zs, -ones, torch.zeros_like(ones)), dim=1),
                            torch.stack((dv * zs, torch.zeros_like(ones), -ones), dim=1)), dim=0)
        target = torch.cat((x, y))
        solution = torch.linalg.lstsq(design, target[:, None]).solution[:, 0]
        residual = design @ solution - target
        # residual in pixels: divide by zs / f
        inv_f = solution[0]
        pixel_residual = residual / (torch.cat((zs, zs)) * inv_f)
        return float(pixel_residual.abs().mean()), float(1.0 / inv_f)

    result = minimize_scalar(lambda s: solve(s)[0], bounds=(-z_min + 1e-3, -z_min + 20.0), method='bounded')
    residual, focal = solve(result.x)
    return focal, float(result.x), residual


FOCAL_PIXEL_STRIDE = 4
FOCAL_DEPTH_QUANTILE = 0.9


def estimate_intrinsics(points):
    """[B, 4] (f, f, c_x, c_y) from [B, 3, H, W] predicted point maps: principal point at the image centre, one focal
    fitted on every 4th pixel after dropping the farthest 10% of predicted depths (sky and far outliers)."""
    height, width = points.shape[-2:]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    grid = torch.zeros(height, width, dtype=torch.bool, device=points.device)
    grid[::FOCAL_PIXEL_STRIDE, ::FOCAL_PIXEL_STRIDE] = True
    rows = []
    for point_map in points.float():
        depth = point_map[2]
        mask = grid & (depth <= torch.quantile(depth[grid], FOCAL_DEPTH_QUANTILE))
        focal = implied_focal(point_map, mask, cx, cy)[0]
        rows.append([focal, focal, cx, cy])
    return torch.tensor(rows, dtype=torch.float32, device=points.device)
