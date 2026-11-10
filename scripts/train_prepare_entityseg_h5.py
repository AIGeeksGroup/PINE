# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Pack the low-resolution EntitySeg training split (scripts/train_prepare_entityseg_download.py) into h5 shards for
dataloader/img_instance_segmentation.py: the released JPEG bytes and a uint16 PNG of entity ids (1..N, 0 = none).

Images from --exclude-sources (coco_val: the COCOexc benchmark images) and images without masks are skipped.
--things-only keeps the fully categorised images and packs their thing entities and the --as-things stuff categories;
the other stuff is background. --worker k of --workers n packs shards k, k + n, ..."""
import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from pycocotools import mask as mask_utils


def entity_ids(annotations):
    """[H, W] uint16 ids of non-overlapping RLE masks, numbered in annotation order."""
    masks = torch.from_numpy(mask_utils.decode([annotation['segmentation'] for annotation in annotations])).cuda()
    numbers = torch.arange(1, masks.shape[-1] + 1, device='cuda', dtype=torch.int32)
    return (masks.int() * numbers).amax(-1).cpu().numpy().astype(np.uint16)


def write_shard(path, references, rgb, instance, encoding):
    with h5py.File(path.with_suffix('.tmp'), 'w') as shard:
        shard['canonical_reference'] = np.array([reference.encode() for reference in references])
        for name, chunks in (('rgb', rgb), ('instance_segmentation', instance)):
            lengths = [len(chunk) for chunk in chunks]
            shard[f'{name}_offsets'] = np.concatenate(([0], np.cumsum(lengths))).astype(np.uint64)
            shard[f'{name}_bytes'] = np.concatenate(chunks)
        shard.attrs.update({'source_dataset': 'EntitySeg (low resolution)', 'sample_count': len(references),
                            'rgb_encoding': 'released JPEG bytes',
                            'instance_segmentation_encoding': encoding,
                            'preparation_script': 'train_prepare_entityseg_h5.py'})
    path.with_suffix('.tmp').rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--annotations', default='entityseg_train_lr.json')
    parser.add_argument('--exclude-sources', nargs='+', required=True)
    parser.add_argument('--shard-size', type=int, required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--things-only', action='store_true')
    parser.add_argument('--as-things', nargs='*', default=[], help='stuff categories packed as things')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = json.loads((args.root / args.annotations).read_text())
    sources = {entry['img_name']: entry['img_source'] for entry in json.loads((args.root / 'url.json').read_text())}
    by_image = {}
    for annotation in data['annotations']:
        by_image.setdefault(annotation['image_id'], []).append(annotation)
    encoding = 'uint16 PNG of entity ids 1..N, 0 = no entity'
    if args.things_only:
        categories = {category['id']: category for category in data['categories']}
        thing = {i for i, category in categories.items()
                 if category['type'] == 'thing' or category['name'] in args.as_things}
        known = {i: annotations for i, annotations in by_image.items()
                 if all(categories[a['category_id']]['name'] != 'unknown' for a in annotations)}
        by_image = {i: [a for a in annotations if a['category_id'] in thing] for i, annotations in known.items()}
        by_image = {i: annotations for i, annotations in by_image.items() if annotations}
        encoding = f'uint16 PNG of thing ids 1..N, 0 = stuff or no entity; stuff packed as things: {args.as_things}'
    images = [image for image in sorted(data['images'], key=lambda image: image['file_name'])
              if sources[image['file_name']] not in args.exclude_sources and image['id'] in by_image]
    print(f'{len(images)} of {len(data["images"])} images kept', flush=True)
    shards = [images[start:start + args.shard_size] for start in range(0, len(images), args.shard_size)]
    for number in range(args.worker, len(shards), args.workers):
        path = args.output_dir / f'entityseg_{number:03d}.h5'
        rgb = [np.fromfile(args.root / 'images_lr' / image['file_name'], dtype=np.uint8) for image in shards[number]]
        instance = [np.frombuffer(cv2.imencode('.png', entity_ids(by_image[image['id']]))[1], dtype=np.uint8)
                    for image in shards[number]]
        write_shard(path, [image['file_name'] for image in shards[number]], rgb, instance, encoding)
        print(path.name, len(shards[number]), 'samples', f'{path.stat().st_size / 1e9:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
