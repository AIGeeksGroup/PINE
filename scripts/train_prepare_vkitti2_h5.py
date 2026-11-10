# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Pack the training subset of Virtual KITTI 2 into h5 shards for the vkitti_h5 loader (dataloader/h5_shards.py), one
shard per scene, condition and camera, and write the camera sidecar <--camera-dir>/vkitti.h5.

--root holds Scene<NN>/<condition>/intrinsic.txt and
Scene<NN>/<condition>/frames/{rgb,depth,normal,instanceSegmentation}/Camera_{0,1}/. Virtual KITTI 2 has no normals:
make frames/normal from the depth with utils/depth2normal.sh of Lotus (https://github.com/EnVision-Research/Lotus).
The subset is scenes 02, 06, 18 and 20 in the conditions 15-deg-left, 15-deg-right, clone, morning, overcast and
sunset: 20,148 frames. A shard holds the released file bytes. The loader crops every frame to the KITTI benchmark size
(the bottom 352 rows, the centre 1216 columns), and the sidecar holds the camera of that crop."""
import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataloader.h5_shards import VKITTI_BENCHMARK_HEIGHT_WIDTH  # noqa: E402

scenes = ('Scene02', 'Scene06', 'Scene18', 'Scene20')
conditions = ('15-deg-left', '15-deg-right', 'clone', 'morning', 'overcast', 'sunset')
cameras = ('Camera_0', 'Camera_1')
native_height_width = (375, 1242)
modalities = {'rgb': ('rgb', 'rgb_', '.jpg'), 'depth': ('depth', 'depth_', '.png'),
              'normal': ('normal', 'normal_', '.png'),
              'instance_segmentation': ('instanceSegmentation', 'instancegt_', '.png')}


def pack_shard(root, scene, condition, camera, output_dir):
    """Write one shard and return its (canonical references, [N, 3, 3] camera_rays_from_pixel of the cropped frames)."""
    condition_dir = root / scene / condition
    depth_dir = condition_dir / 'frames/depth' / camera
    frames = sorted(path.stem.removeprefix('depth_') for path in depth_dir.glob('*.png'))
    references = [f'vkitti2/{scene}/{condition}/frames/depth/{camera}/depth_{frame}.png' for frame in frames]
    with h5py.File(output_dir / f'{scene}_{condition}_{camera}.h5'.lower(), 'w') as shard:
        shard['canonical_reference'] = np.array(references, dtype='S')
        for name, (folder, prefix, suffix) in modalities.items():
            files = [(condition_dir / 'frames' / folder / camera / f'{prefix}{frame}{suffix}').read_bytes()
                     for frame in frames]
            shard[f'{name}_offsets'] = np.cumsum([0] + [len(data) for data in files]).astype(np.uint64)
            shard[f'{name}_bytes'] = np.frombuffer(b''.join(files), dtype=np.uint8)
        shard.attrs.update({'source_dataset': 'Virtual KITTI 2', 'scene': scene, 'condition': condition,
                            'camera': camera, 'sample_count': len(frames),
                            'preparation_script': 'train_prepare_vkitti2_h5.py'})
    # intrinsic.txt rows: frame cameraID K[0,0] K[1,1] K[0,2] K[1,2]
    rows = np.atleast_2d(np.loadtxt(condition_dir / 'intrinsic.txt', skiprows=1))
    intrinsics = {(int(row[0]), int(row[1])): row[2:6] for row in rows}
    top = native_height_width[0] - VKITTI_BENCHMARK_HEIGHT_WIDTH[0]
    left = (native_height_width[1] - VKITTI_BENCHMARK_HEIGHT_WIDTH[1]) // 2
    native_from_crop = np.array([[1.0, 0.0, left], [0.0, 1.0, top], [0.0, 0.0, 1.0]])
    rays = []
    for frame in frames:
        fx, fy, cx, cy = intrinsics[(int(frame), int(camera.removeprefix('Camera_')))]
        rays.append(np.linalg.inv(np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])) @ native_from_crop)
    return references, np.stack(rays)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--camera-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.camera_dir.mkdir(parents=True, exist_ok=True)
    references, rays = [], []
    for scene in scenes:
        for condition in conditions:
            for camera in cameras:
                shard_references, shard_rays = pack_shard(args.root, scene, condition, camera, args.output_dir)
                references += shard_references
                rays.append(shard_rays)
                print(scene, condition, camera, len(shard_references), 'frames', flush=True)
    with h5py.File(args.camera_dir / 'vkitti.h5', 'w') as sidecar:
        sidecar['sample_key'] = np.array(references, dtype='S')
        sidecar['camera_rays_from_pixel'] = np.concatenate(rays)
        sidecar['camera_rays_valid'] = np.ones(len(references), bool)
        sidecar['image_height_width'] = np.tile(np.array(VKITTI_BENCHMARK_HEIGHT_WIDTH, dtype=np.int32),
                                                (len(references), 1))
        sidecar.attrs.update({'camera_coordinate_system': 'x right, y down, z forward', 'dataset_name': 'vkitti2',
                              'pixel_coordinate_convention': 'zero-based pixel centers',
                              'sample_count': len(references),
                              'sample_key_convention': 'dataset-local canonical reference', 'schema_version': 1})
    print('sidecar', args.camera_dir / 'vkitti.h5', len(references), 'frames', flush=True)


if __name__ == '__main__':
    main()
