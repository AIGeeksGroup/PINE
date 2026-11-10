# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Pack the ADE20K training split (scripts/train_prepare_ade20k_download.sh) into h5 shards for
dataloader/img_instance_segmentation.py: the released JPEG bytes and a uint16 PNG of instance ids.

The ids are the object instances of the 100 thing classes (Places Challenge 2017). 0 is background: the stuff classes
and the unlabelled pixels. Pixels of the --ignore classes without an instance get 65535 and stay out of the instance
loss. Images without an instance are skipped. --worker k of --workers n packs shards k, k + n, ..."""
import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from PIL import Image


def read_ids(root, name, ignored):
    """[H, W] uint16 ids: the instance index, 0 for background and the uint16 maximum for ignored pixels."""
    green = np.asarray(Image.open(root / 'annotations_instance' / 'training' / f'{name}.png'))[..., 1]
    semantic = np.asarray(Image.open(root / 'ADEChallengeData2016' / 'annotations' / 'training' / f'{name}.png'))
    return np.where(np.isin(semantic, ignored) & (green == 0), np.iinfo(np.uint16).max, green).astype(np.uint16)


def write_shard(path, references, rgb, instance, encoding):
    with h5py.File(path.with_suffix('.tmp'), 'w') as shard:
        shard['canonical_reference'] = np.array([reference.encode() for reference in references])
        for name, chunks in (('rgb', rgb), ('instance_segmentation', instance)):
            lengths = [len(chunk) for chunk in chunks]
            shard[f'{name}_offsets'] = np.concatenate(([0], np.cumsum(lengths))).astype(np.uint64)
            shard[f'{name}_bytes'] = np.concatenate(chunks)
        shard.attrs.update({'source_dataset': 'ADE20K (SceneParsing training, Places Challenge 2017 instances)',
                            'sample_count': len(references), 'rgb_encoding': 'released JPEG bytes',
                            'instance_segmentation_encoding': encoding,
                            'preparation_script': 'train_prepare_ade20k_h5.py'})
    path.with_suffix('.tmp').rename(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--ignore', nargs='+', required=True, help='ADE20K class names, as in objectInfo150.txt')
    parser.add_argument('--shard-size', type=int, required=True)
    parser.add_argument('--worker', type=int, required=True)
    parser.add_argument('--workers', type=int, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ['unlabeled'] + [line.split('\t')[-1].split(',')[0].strip()
                             for line in (args.root / 'ADEChallengeData2016' / 'objectInfo150.txt').read_text()
                             .splitlines()[1:]]
    ignored = [names.index(name) for name in args.ignore]
    encoding = (f'uint16 PNG of instance ids (green channel of the instance annotation), 0 = background, '
                f'{np.iinfo(np.uint16).max} = ignored: pixels of {args.ignore} without an instance')
    images = sorted(path.stem for path in (args.root / 'ADEChallengeData2016' / 'images' / 'training').glob('*.jpg'))
    shards = [images[start:start + args.shard_size] for start in range(0, len(images), args.shard_size)]
    for number in range(args.worker, len(shards), args.workers):
        path = args.output_dir / f'ade20k_{number:03d}.h5'
        references, rgb, instance = [], [], []
        for name in shards[number]:
            ids = read_ids(args.root, name, ignored)
            if ((ids > 0) & (ids < np.iinfo(np.uint16).max)).any():
                references.append(name)
                rgb.append(np.fromfile(args.root / 'ADEChallengeData2016' / 'images' / 'training' / f'{name}.jpg',
                                       dtype=np.uint8))
                instance.append(np.frombuffer(cv2.imencode('.png', ids)[1], dtype=np.uint8))
        write_shard(path, references, rgb, instance, encoding)
        print(path.name, len(references), 'of', len(shards[number]), 'images kept',
              f'{path.stat().st_size / 1e9:.2f} GB', flush=True)


if __name__ == '__main__':
    main()
