# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Pack the Hypersim training frames into h5 shards for the hypersim_h5 loader (dataloader/h5_shards.py), one shard per
scene camera, and write the camera sidecar <--camera-dir>/hypersim.h5.

--root holds the released scenes: <scene>/images/scene_cam_XX_final_preview/frame.NNNN.tonemap.jpg and
<scene>/images/scene_cam_XX_geometry_hdf5/frame.NNNN.{depth_meters,normal_cam,semantic_instance}.hdf5. A shard holds,
per frame, image (the JPEG bytes), ray_depth (metres along the ray), normal and instance, copied as the released
compressed chunks. dataloader/configs/hypersim_train_frames.txt lists the 39,241 frames (scene, camera, frame ids): the
official train partition without the frames whose depth holds a NaN (as in Lotus), the black images, the frames
without labels and the scene ai_013_001 (invalid depth). The sidecar holds the camera of each scene from the official
metadata_camera_parameters.csv (--camera-parameters). A rerun skips the finished shards."""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np

array_suffixes = {'ray_depth': '.depth_meters.hdf5', 'normal': '.normal_cam.hdf5',
                  'instance': '.semantic_instance.hdf5'}
train_frames = Path(__file__).resolve().parents[1] / 'dataloader/configs/hypersim_train_frames.txt'


def copy_chunks(source_path, destination, index):
    """Copy the compressed chunks of one released array to sample index of destination."""
    with h5py.File(source_path, 'r') as source_file:
        source = source_file['dataset']
        for chunk in source.iter_chunks():
            offset = tuple(part.start for part in chunk)
            filter_mask, data = source.id.read_direct_chunk(offset)
            destination.id.write_direct_chunk((index, *offset), data, filter_mask=filter_mask)


def pack_camera(task):
    root, output_dir, scene, camera, frames = task
    output_path = output_dir / f'{scene}_scene_{camera}.h5'
    if output_path.exists():
        return
    images_dir = root / scene / 'images'
    geometry = [images_dir / f'scene_{camera}_geometry_hdf5' / f'frame.{frame}' for frame in frames]
    with h5py.File(f'{output_path}.partial', 'w') as shard:
        folder = f'hypersim/{scene}/images/scene_{camera}_geometry_hdf5'
        shard['canonical_reference'] = np.array([f'{folder}/frame.{frame}.depth_meters.hdf5' for frame in frames],
                                                dtype='S')
        for name, suffix in array_suffixes.items():
            with h5py.File(f'{geometry[0]}{suffix}', 'r') as first:
                source = first['dataset']
                array = shard.create_dataset(
                    name, shape=(len(frames), *source.shape), dtype=source.dtype, chunks=(1, *source.chunks),
                    compression=source.compression, compression_opts=source.compression_opts, shuffle=source.shuffle,
                    fletcher32=source.fletcher32)
            for index, stem in enumerate(geometry):
                copy_chunks(f'{stem}{suffix}', array, index)
        images = [(images_dir / f'scene_{camera}_final_preview' / f'frame.{frame}.tonemap.jpg').read_bytes()
                  for frame in frames]
        shard['image_offsets'] = np.cumsum([0] + [len(image) for image in images]).astype(np.uint64)
        shard['image_bytes'] = np.frombuffer(b''.join(images), dtype=np.uint8)
        shard.attrs.update({'source_dataset': 'Hypersim', 'sample_count': len(frames),
                            'image_encoding': 'original JPEG bytes', 'ray_depth_unit': 'meters',
                            'preparation_script': 'train_prepare_hypersim_h5.py'})
    os.replace(f'{output_path}.partial', output_path)


def camera_rays_from_pixel(row):
    """[3, 3] map from a pixel centre (x, y, 1) to its camera ray (x right, y down, z forward) for one scene. Hypersim
    cameras are not all centred pinholes, so the matrix comes from the released M_cam_from_uv."""
    height, width = int(float(row['settings_output_img_height'])), int(float(row['settings_output_img_width']))
    camera_from_uv = np.array([float(row[f'M_cam_from_uv_{i}{j}']) for i in range(3) for j in range(3)]).reshape(3, 3)
    uv_from_pixel = np.array([[2.0 / width, 0.0, 1.0 / width - 1.0], [0.0, -2.0 / height, 1.0 - 1.0 / height],
                              [0.0, 0.0, 1.0]])
    return np.diag((1.0, -1.0, -1.0)) @ camera_from_uv @ uv_from_pixel, (height, width)


def write_sidecar(output_dir, camera_parameters, camera_dir):
    with open(camera_parameters, newline='') as handle:
        cameras = {row['scene_name']: camera_rays_from_pixel(row) for row in csv.DictReader(handle)}
    references = []
    for shard_path in sorted(output_dir.glob('*.h5')):
        with h5py.File(shard_path, 'r') as shard:
            references += [reference.decode() for reference in shard['canonical_reference'][:]]
    scenes = [reference.split('/')[1] for reference in references]
    camera_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(camera_dir / 'hypersim.h5', 'w') as sidecar:
        sidecar['sample_key'] = np.array(references, dtype='S')
        sidecar['camera_rays_from_pixel'] = np.stack([cameras[scene][0] for scene in scenes])
        sidecar['camera_rays_valid'] = np.ones(len(references), bool)
        sidecar['image_height_width'] = np.array([cameras[scene][1] for scene in scenes], dtype=np.int32)
        sidecar.attrs.update({'camera_coordinate_system': 'x right, y down, z forward', 'dataset_name': 'hypersim',
                              'pixel_coordinate_convention': 'zero-based pixel centers',
                              'sample_count': len(references),
                              'sample_key_convention': 'dataset-local canonical reference', 'schema_version': 1})
    print('sidecar', camera_dir / 'hypersim.h5', len(references), 'frames', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True, help='the folder of the released ai_VVV_NNN scenes')
    parser.add_argument('--camera-parameters', type=Path, required=True,
                        help='metadata_camera_parameters.csv of github.com/apple/ml-hypersim (contrib/mikeroberts3000)')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--camera-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = []
    for line in train_frames.read_text().splitlines():
        scene, camera, *frames = line.split()
        tasks.append((args.root, args.output_dir, scene, camera, frames))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(pack_camera, tasks))
    print('packed', sum(len(task[4]) for task in tasks), 'frames of', len(tasks), 'scene cameras', flush=True)
    write_sidecar(args.output_dir, args.camera_parameters, args.camera_dir)


if __name__ == '__main__':
    main()
