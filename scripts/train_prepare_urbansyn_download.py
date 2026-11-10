# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Download UrbanSyn (Hugging Face UrbanSyn/UrbanSyn, CC BY-SA 4.0, 86 GB) into --output-dir: 7,539 driving scenes at
2048 x 1024 with rgb, depth, semantics, panoptic instances and 2D boxes."""
import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=16)
    args = parser.parse_args()
    path = snapshot_download(repo_id='UrbanSyn/UrbanSyn', repo_type='dataset', local_dir=args.output_dir,
                             max_workers=args.workers)
    print('downloaded to', path, flush=True)


if __name__ == '__main__':
    main()
