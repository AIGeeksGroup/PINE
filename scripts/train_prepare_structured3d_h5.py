# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Pack the furnished perspective views of Structured3D (scripts/train_prepare_structured3d_download.sh) into h5 shards
for the structured3d_h5 loader (dataloader/h5_shards.py), one shard per zip, and write the camera sidecar
<--camera-dir>/structured3d.h5 (--write-sidecar).

Depth is planar z in millimetres. Normals are converted to the OpenCV camera frame. Instance ids come from
Structured3D_bbox.zip; walls, floor and ceiling are background (0). Scenes with ids >= --scene-limit and scenes without
an object instance are skipped. --worker k of --workers n packs zips k, k + n, ..."""
import argparse
import json
import re
import zipfile
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F


def read_image(archive, name, flags):
    return cv2.imdecode(np.frombuffer(archive.read(name), np.uint8), flags)


def encode(extension, array, parameters=()):
    return np.frombuffer(cv2.imencode(extension, array, list(parameters))[1], dtype=np.uint8)


def intrinsics(archive, view, height, width):
    """(fx, fy, cx, cy) from camera_pose.txt: position, forward, up, half fields of view in radians."""
    pose = np.array(archive.read(f'{view}/camera_pose.txt').split(), dtype=np.float64)
    return width / 2 / np.tan(pose[9]), height / 2 / np.tan(pose[10]), (width - 1) / 2, (height - 1) / 2


def opencv_normal_png(stored_bgr):
    """RGB PNG of (n + 1) * 127.5 in the OpenCV frame from the stored 128 * (1 + n), zero where undefined."""
    stored = torch.from_numpy(cv2.cvtColor(stored_bgr, cv2.COLOR_BGR2RGB)).cuda().float() / 128 - 1
    defined = stored.norm(dim=-1, keepdim=True) > 0.5
    normal = F.normalize(stored * torch.tensor([1.0, -1.0, 1.0], device='cuda'), dim=-1)
    rgb = torch.where(defined, ((normal + 1) * 127.5).round().clamp(0, 255), 0.0).byte().cpu().numpy()
    return encode('.png', np.ascontiguousarray(rgb[..., ::-1]))


def stored_instance_ids(ids):
    """Ids plus one, with the walls, floor and ceiling id (uint16 maximum) as 0."""
    ids = torch.from_numpy(ids.astype(np.int32)).cuda()
    stored = torch.where(ids == np.iinfo(np.uint16).max, 0, ids + 1)
    return stored.cpu().numpy().astype(np.uint16)


def write_shard(path, references, intrinsic_rows, samples):
    with h5py.File(path.with_suffix('.tmp'), 'w') as shard:
        shard['canonical_reference'] = np.array([reference.encode() for reference in references])
        shard['intrinsics'] = np.array(intrinsic_rows)
        for name in ('rgb', 'depth', 'normal', 'instance_segmentation'):
            lengths = [len(sample[name]) for sample in samples]
            shard[f'{name}_offsets'] = np.concatenate(([0], np.cumsum(lengths))).astype(np.uint64)
            shard[f'{name}_bytes'] = np.concatenate([sample[name] for sample in samples])
        shard.attrs.update({'source_dataset': 'Structured3D perspective full', 'sample_count': len(references),
                            'instance_background_supervised': True,
                            'rgb_encoding': 'uint8 JPEG quality 95 of rgb_rawlight.png',
                            'depth_encoding': 'released uint16 PNG in millimeters, 0 = no depth',
                            'normal_encoding': 'uint8 RGB PNG of (n + 1) * 127.5, OpenCV camera frame, 0 = no normal',
                            'instance_segmentation_encoding': 'uint16 PNG, released id + 1, 0 = walls, floor, ceiling',
                            'preparation_script': 'train_prepare_structured3d_h5.py'})
    path.with_suffix('.tmp').rename(path)


def pack_zip(archive_path, instance_archive, scene_limit, output_dir, stats):
    archive = zipfile.ZipFile(archive_path)
    views = sorted({name.rsplit('/', 1)[0] for name in archive.namelist()
                    if re.search(r'/perspective/full/[^/]+/depth\.png$', name)})
    by_scene = {}
    for view in views:
        by_scene.setdefault(int(re.search(r'scene_(\d+)', view).group(1)), []).append(view)
    references, intrinsic_rows, samples = [], [], []
    for scene, scene_views in sorted(by_scene.items()):
        if scene >= scene_limit:
            continue
        ids = [read_image(instance_archive, f'{view}/instance.png', cv2.IMREAD_UNCHANGED) for view in scene_views]
        objects = [int((np.unique(view_ids) != np.iinfo(np.uint16).max).sum()) for view_ids in ids]
        if max(objects) == 0:
            stats['unlabeled_scenes'].append(scene)
            continue
        stats['scenes'] += 1
        stats['objects_per_view'] += objects
        for view, view_ids in zip(scene_views, ids):
            rgb = read_image(archive, f'{view}/rgb_rawlight.png', cv2.IMREAD_COLOR)
            samples.append({'rgb': encode('.jpg', rgb, (cv2.IMWRITE_JPEG_QUALITY, 95)),
                            'depth': np.frombuffer(archive.read(f'{view}/depth.png'), np.uint8),
                            'normal': opencv_normal_png(read_image(archive, f'{view}/normal.png', cv2.IMREAD_COLOR)),
                            'instance_segmentation': encode('.png', stored_instance_ids(view_ids))})
            intrinsic_rows.append(intrinsics(archive, view, *rgb.shape[:2]))
            references.append('/'.join(view.split('/')[1:2] + view.split('/')[3:4] + view.split('/')[-1:]))
    path = output_dir / f'structured3d_{archive_path.stem[-2:]}.h5'
    write_shard(path, references, intrinsic_rows, samples)
    print(path.name, len(references), 'views', f'{path.stat().st_size / 1e9:.2f} GB', flush=True)


def write_sidecar(output_dir, camera_dir):
    references, rows = [], []
    for shard_path in sorted(output_dir.glob('*.h5')):
        with h5py.File(shard_path, 'r') as shard:
            references += [reference.decode() for reference in shard['canonical_reference'][:]]
            rows.append(shard['intrinsics'][:])
    fx, fy, cx, cy = np.concatenate(rows).T
    K = np.zeros((len(references), 3, 3))
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = fx, fy, cx, cy, 1
    with h5py.File(camera_dir / 'structured3d.h5', 'w') as sidecar:
        sidecar['sample_key'] = np.array([reference.encode() for reference in references])
        sidecar['camera_rays_from_pixel'] = np.linalg.inv(K)
        sidecar['camera_rays_valid'] = np.ones(len(references), bool)
        sidecar['image_height_width'] = np.stack((2 * cy + 1, 2 * cx + 1), 1).round().astype(np.int32)
        sidecar.attrs.update({'camera_coordinate_system': 'x right, y down, z forward', 'dataset_name': 'structured3d',
                              'pixel_coordinate_convention': 'zero-based pixel centers',
                              'sample_count': len(references),
                              'sample_key_convention': 'dataset-local canonical reference', 'schema_version': 1})
    print('sidecar', camera_dir / 'structured3d.h5', len(references), 'views', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--scene-limit', type=int, required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--camera-dir', type=Path, required=True)
    parser.add_argument('--stats-dir', type=Path, required=True)
    parser.add_argument('--write-sidecar', action='store_true')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.stats_dir.mkdir(parents=True, exist_ok=True)
    if args.write_sidecar:
        write_sidecar(args.output_dir, args.camera_dir)
        return
    instance_archive = zipfile.ZipFile(args.root / 'Structured3D_bbox.zip')
    archives = sorted(args.root.glob('Structured3D_perspective_full_*.zip'))
    stats = {'scenes': 0, 'unlabeled_scenes': [], 'objects_per_view': []}
    for archive_path in archives[args.worker::args.workers]:
        pack_zip(archive_path, instance_archive, args.scene_limit, args.output_dir, stats)
    objects = np.array(stats['objects_per_view'])
    summary = {'scenes': stats['scenes'], 'unlabeled_scenes': stats['unlabeled_scenes'], 'views': len(objects),
               'views_without_objects': int((objects == 0).sum()), 'median_objects_per_view': float(np.median(objects))}
    (args.stats_dir / f'worker{args.worker}.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
