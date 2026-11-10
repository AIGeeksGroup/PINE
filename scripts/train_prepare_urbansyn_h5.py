# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Pack UrbanSyn (scripts/train_prepare_urbansyn_download.py) into h5 shards for the urbansyn_h5 loader
(dataloader/h5_shards.py), at --scale of the native 2048 x 1024, and write the camera sidecar <--camera-dir>/urbansyn.h5.

Per sample: rgb (JPEG), depth (float32 EXR, metres, planar z, resized without mixing depths across occlusion edges),
normal (from the stored depth by one-sided differences, lossless WebP) and instance_segmentation (uint16 PNG; UrbanSyn
labels only the dynamic classes). The camera is one pinhole with fx = fy = 4.9 / 5.8 * 2048 px.
--worker k of --workers n packs shards k, k + n, ...; --write-sidecar writes the sidecar from the finished shards."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('OPENCV_IO_ENABLE_OPENEXR', '1')
import cv2  # noqa: E402
import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

FOCAL = 4.9 / 5.8 * 2048
METERS_PER_UNIT = 1e5
NATIVE_HEIGHT_WIDTH = (1024, 2048)
SAMPLE_COUNT = 7539
SKY_CLASS = 10
SKY_RULE_METRES = 100.0


def one_sided(points, depth, valid, dim):
    """Difference along dim (1 = x, 0 = y) with the smaller depth change, and its relative depth jump."""
    forward = torch.full_like(points, float('nan'))
    backward = torch.full_like(points, float('nan'))
    step = points.diff(dim=dim)
    pair = valid.narrow(dim, 1, valid.shape[dim] - 1) & valid.narrow(dim, 0, valid.shape[dim] - 1)
    step = torch.where(pair[..., None], step, float('nan'))
    forward.narrow(dim, 0, points.shape[dim] - 1).copy_(step)
    backward.narrow(dim, 1, points.shape[dim] - 1).copy_(step)
    jump_forward = (forward[..., 2].abs() / depth).nan_to_num(float('inf'))
    jump_backward = (backward[..., 2].abs() / depth).nan_to_num(float('inf'))
    take_forward = jump_forward <= jump_backward
    return torch.where(take_forward[..., None], forward, backward), torch.minimum(jump_forward, jump_backward)


def normals_from_depth(depth, K, jump):
    """([H, W, 3] unit normals, [H, W] defined) from a [H, W] metric z-depth on its device."""
    height, width = depth.shape
    v, u = torch.meshgrid(torch.arange(height, device=depth.device, dtype=depth.dtype),
                          torch.arange(width, device=depth.device, dtype=depth.dtype), indexing='ij')
    points = torch.stack(((u - K[0, 2]) / K[0, 0] * depth, (v - K[1, 2]) / K[1, 1] * depth, depth), -1)
    valid = torch.isfinite(depth) & (depth > 0)
    dx, jump_x = one_sided(points, depth, valid, 1)
    dy, jump_y = one_sided(points, depth, valid, 0)
    normal = F.normalize(torch.cross(dx, dy, dim=-1).nan_to_num(0.0), dim=-1)
    normal = torch.where((normal * points).sum(-1, keepdim=True) > 0, -normal, normal)
    defined = valid & (jump_x < jump) & (jump_y < jump) & (normal.norm(dim=-1) > 0.5)
    return torch.where(defined[..., None], normal, 0.0), defined


def intrinsics(scale):
    """[3, 3] K at --scale of the native resolution (zero-based pixel centres)."""
    height, width = NATIVE_HEIGHT_WIDTH
    cx, cy = (width - 1) / 2, (height - 1) / 2
    return np.array([[FOCAL * scale, 0, (cx + 0.5) * scale - 0.5], [0, FOCAL * scale, (cy + 0.5) * scale - 0.5], [0, 0, 1]])


def resize_depth(depth, size, jump):
    """[H, W] metric depth at the pixel centres of size: bilinear where the 2 x 2 source depths agree within jump of the
    nearest one, else nearest-exact."""
    depth = depth[None, None]
    bilinear = F.interpolate(depth, size=size, mode='bilinear', align_corners=False)
    nearest = F.interpolate(depth, size=size, mode='nearest-exact')
    high = F.interpolate(F.max_pool2d(depth, 2, 1), size=size, mode='bilinear', align_corners=False)
    low = -F.interpolate(F.max_pool2d(-depth, 2, 1), size=size, mode='bilinear', align_corners=False)
    smooth = (high - low < jump * low.clamp_min(1e-6)) & torch.isfinite(bilinear) & (low > 0)
    return torch.where(smooth, bilinear, nearest)[0, 0]


def resize(tensor, size, mode):
    """[H, W, C] tensor resized to size; bilinear uses antialiasing."""
    tensor = tensor.permute(2, 0, 1)[None].float()
    if mode == 'bilinear':
        tensor = F.interpolate(tensor, size=size, mode='bilinear', antialias=True, align_corners=False)
    else:
        tensor = F.interpolate(tensor, size=size, mode='nearest-exact')
    return tensor[0].permute(1, 2, 0)


def encode(extension, array, parameters=()):
    ok, data = cv2.imencode(extension, array, list(parameters))
    assert ok, extension
    return np.frombuffer(data.tobytes(), dtype=np.uint8)


def pack_sample(root, index, scale, jump, stats):
    rgb = cv2.imread(str(root / 'rgb' / f'rgb_{index:04d}.png'), cv2.IMREAD_COLOR)
    exr = cv2.imread(str(root / 'depth' / f'depth_{index:04d}.exr'), cv2.IMREAD_UNCHANGED)
    panoptic = cv2.imread(str(root / 'panoptic' / f'panoptic_{index:04d}.png'), cv2.IMREAD_COLOR).astype(np.int64)
    semantic = cv2.imread(str(root / 'ss' / f'ss_{index:04d}.png'), cv2.IMREAD_UNCHANGED)
    semantic = semantic[..., 0] if semantic.ndim == 3 else semantic
    depth = torch.from_numpy((exr[..., 0] if exr.ndim == 3 else exr).astype(np.float32)).cuda() * METERS_PER_UNIT
    instance = torch.from_numpy(panoptic[..., 2] * 256 + panoptic[..., 1]).cuda()  # BGR: R * 256 + G
    size = (round(NATIVE_HEIGHT_WIDTH[0] * scale), round(NATIVE_HEIGHT_WIDTH[1] * scale))
    rgb_small = resize(torch.from_numpy(rgb).cuda(), size, 'bilinear').round().clamp(0, 255).byte().cpu().numpy()
    depth_small = resize_depth(depth, size, jump)
    depth_small = torch.where(torch.isfinite(depth_small) & (depth_small > 0), depth_small, 0.0)
    normal_small, defined = normals_from_depth(depth_small, torch.from_numpy(intrinsics(scale)).cuda().float(), jump)
    instance_small = resize(instance[..., None], size, 'nearest')[..., 0].round().long()
    depth_small_native = resize(depth[..., None], size, 'nearest')[..., 0]
    normal_png = torch.where(defined[..., None], ((normal_small + 1) * 127.5).round().clamp(0, 255), 0)
    normal_png = normal_png.byte().cpu().numpy()[..., ::-1]  # RGB stored through cv2 (BGR)
    sky = torch.from_numpy(semantic == SKY_CLASS).cuda()
    stats['sky_pixels'] += int(sky.sum())
    stats['sky_at_most_100m'] += int((sky & (depth <= SKY_RULE_METRES)).sum())
    stats['non_sky_pixels'] += int((~sky).sum())
    stats['non_sky_beyond_100m'] += int((~sky & (depth > SKY_RULE_METRES)).sum())
    stats['non_sky_within_100m'] += int((~sky & (depth <= SKY_RULE_METRES)).sum())
    sky_small = resize(sky[..., None], size, 'nearest')[..., 0] > 0.5
    stats['normal_defined_within_100m'] += int((~sky_small & (depth_small_native <= SKY_RULE_METRES) & defined).sum())
    stats['non_sky_within_100m_stored'] += int((~sky_small & (depth_small_native <= SKY_RULE_METRES)).sum())
    assert int(instance_small.max()) < 65536
    return {'rgb': encode('.jpg', rgb_small, (cv2.IMWRITE_JPEG_QUALITY, 95)),
            'depth': encode('.exr', depth_small.cpu().numpy(), (cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_FLOAT,
                                                                cv2.IMWRITE_EXR_COMPRESSION, cv2.IMWRITE_EXR_COMPRESSION_PIZ)),
            'normal': encode('.webp', np.ascontiguousarray(normal_png), (cv2.IMWRITE_WEBP_QUALITY, 101)),
            'instance_segmentation': encode('.png', instance_small.cpu().numpy().astype(np.uint16))}


def write_shard(path, references, samples, scale):
    with h5py.File(path.with_suffix('.tmp'), 'w') as shard:
        shard['canonical_reference'] = np.array(references, dtype='S')
        for name in ('rgb', 'depth', 'normal', 'instance_segmentation'):
            chunks = [sample[name] for sample in samples]
            shard[f'{name}_offsets'] = np.concatenate(([0], np.cumsum([len(c) for c in chunks]))).astype(np.uint64)
            shard[f'{name}_bytes'] = np.concatenate(chunks)
        shard.attrs.update({
            'source_dataset': 'UrbanSyn', 'source_resolution': '1024x2048', 'scale': scale, 'sample_count': len(samples),
            'instance_background_supervised': False,
            'rgb_encoding': 'uint8 JPEG quality 95, bilinear antialiased resize',
            'depth_encoding': 'float32 EXR (PIZ) in meters, edge-aware bilinear resize, 0 = no depth',
            'normal_encoding': 'lossless WebP of uint8 RGB (n + 1) * 127.5 from the stored depth, OpenCV camera frame, all zero = no normal',
            'instance_segmentation_encoding': 'uint16 PNG of panoptic R * 256 + G, 0 = no instance',
            'preparation_script': 'train_prepare_urbansyn_h5.py'})
    path.with_suffix('.tmp').rename(path)


def write_sidecar(output_dir, camera_dir, scale):
    references = []
    for shard_path in sorted(output_dir.glob('*.h5')):
        with h5py.File(shard_path, 'r') as shard:
            references += [r.decode() for r in shard['canonical_reference'][:]]
    K = intrinsics(scale)
    size = (round(NATIVE_HEIGHT_WIDTH[0] * scale), round(NATIVE_HEIGHT_WIDTH[1] * scale))
    with h5py.File(camera_dir / 'urbansyn.h5', 'w') as sidecar:
        sidecar['sample_key'] = np.array(references, dtype='S')
        sidecar['camera_rays_from_pixel'] = np.repeat(np.linalg.inv(K)[None], len(references), 0)
        sidecar['camera_rays_valid'] = np.ones(len(references), bool)
        sidecar['image_height_width'] = np.repeat(np.array([size], np.int32), len(references), 0)
        sidecar.attrs.update({'camera_coordinate_system': 'x right, y down, z forward', 'dataset_name': 'urbansyn',
                              'pixel_coordinate_convention': 'zero-based pixel centers', 'sample_count': len(references),
                              'sample_key_convention': 'dataset-local canonical reference', 'schema_version': 1})
    print('sidecar', camera_dir / 'urbansyn.h5', len(references), 'samples, K', K.tolist(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--camera-dir', type=Path, required=True)
    parser.add_argument('--stats-dir', type=Path, required=True)
    parser.add_argument('--scale', type=float, default=0.75)
    parser.add_argument('--jump', type=float, default=0.05)
    parser.add_argument('--shard-size', type=int, default=500)
    parser.add_argument('--worker', type=int, default=0)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--limit', type=int, help='pack only the first samples of the first shard (size check)')
    parser.add_argument('--write-sidecar', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    if args.write_sidecar:
        write_sidecar(args.output_dir, args.camera_dir, args.scale)
        return
    indices = list(range(1, SAMPLE_COUNT + 1))
    shards = [indices[i:i + args.shard_size] for i in range(0, len(indices), args.shard_size)]
    stats = {key: 0 for key in ('sky_pixels', 'sky_at_most_100m', 'non_sky_pixels', 'non_sky_beyond_100m',
                                'non_sky_within_100m', 'non_sky_within_100m_stored', 'normal_defined_within_100m', 'samples', 'bytes')}
    for number in range(args.worker, len(shards), args.workers):
        path = args.output_dir / f'urbansyn_{number:03d}.h5'
        if path.exists() and not args.limit:
            continue
        chosen = shards[number][:args.limit] if args.limit else shards[number]
        samples = [pack_sample(args.root, index, args.scale, args.jump, stats) for index in chosen]
        if args.limit:
            path = args.stats_dir / f'size_check_{args.limit}.h5'
        write_shard(path, [f'urbansyn/rgb_{index:04d}' for index in chosen], samples, args.scale)
        stats['samples'] += len(samples)
        stats['bytes'] += path.stat().st_size
        print(path.name, len(samples), 'samples', f'{path.stat().st_size / len(samples) / 1e6:.2f} MB per sample', flush=True)
        if args.limit:
            break
    (args.stats_dir / f'worker{args.worker}{"_limit" if args.limit else ""}.json').write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    main()
