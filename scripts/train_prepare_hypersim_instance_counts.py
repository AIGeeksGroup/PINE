# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Count the instances of the Hypersim training shards for the gen2seg data curation of the instance stream: frames
without an instance and scenes with fewer than 10 labelled objects get no instance loss.

Writes {'frames': {reference: instances in the frame}, 'scenes': {scene: distinct instances over its frames}}, which
the loader reads through the dataset option instance_counts."""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np


def shard_counts(path):
    frames, scenes = {}, {}
    with h5py.File(path, 'r') as shard:
        references = [reference.decode() for reference in shard['canonical_reference'][:]]
        for index, reference in enumerate(references):
            ids = np.unique(shard['instance'][index])
            ids = ids[ids >= 0]
            frames[reference] = int(ids.size)
            scenes.setdefault(reference.split('/')[1], set()).update(ids.tolist())
    return frames, scenes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('datasets/train/hypersim'))
    parser.add_argument('--workers', type=int, default=16)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frames, scenes = {}, {}
    with Pool(args.workers) as pool:
        for index, (shard_frames, shard_scenes) in enumerate(pool.imap_unordered(shard_counts, sorted(args.root.glob('*.h5')))):
            frames.update(shard_frames)
            for scene, ids in shard_scenes.items():
                scenes.setdefault(scene, set()).update(ids)
            if index % 50 == 0:
                print(f'{index} shards, {len(frames)} frames', flush=True)
    scene_counts = {scene: len(ids) for scene, ids in sorted(scenes.items())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'frames': frames, 'scenes': scene_counts}))
    counts = np.array(list(frames.values()))
    print(f'{len(frames)} frames, {len(scene_counts)} scenes; frames without instances {np.mean(counts == 0):.1%}; '
          f'scenes with < 10 objects {np.mean(np.array(list(scene_counts.values())) < 10):.1%}', flush=True)


if __name__ == '__main__':
    main()
