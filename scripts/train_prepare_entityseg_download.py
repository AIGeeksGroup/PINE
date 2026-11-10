# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Download EntitySeg (Adobe Research License, non-commercial research only) into --output-dir: the low-resolution
images of the gated Hugging Face dataset qqlu1992/Adobe_EntitySeg (accept its terms first), and the annotations and
url.json of the GitHub release v1.0 of adobe-research/EntitySeg-Dataset."""
import argparse
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

RELEASE = 'https://github.com/adobe-research/EntitySeg-Dataset/releases/download/v1.0/'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image-zips', nargs='+', default=['images_lr/entity_03_10049.zip'])
    parser.add_argument('--annotations', nargs='+', default=['entityseg_train_lr.json', 'entityseg_val_lr.json', 'url.json'])
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.annotations:
        if not (args.output_dir / name).exists():
            urllib.request.urlretrieve(RELEASE + name, args.output_dir / name)
        print('downloaded', name, flush=True)
    for name in args.image_zips:
        path = Path(hf_hub_download('qqlu1992/Adobe_EntitySeg', name, repo_type='dataset', local_dir=args.output_dir))
        print('downloaded', name, flush=True)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(args.output_dir / Path(name).parent)
        path.unlink()
        print('extracted', name, flush=True)


if __name__ == '__main__':
    main()
