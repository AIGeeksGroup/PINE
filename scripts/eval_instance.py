# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Point-prompted instance segmentation of a PINE checkpoint under the gen2seg protocol (metrics/instance_prompt.py).

The instance map is generated in one step at a short side of img_size and resized back. One prompt at the deepest
interior point of every ground-truth instance selects a mask; the score is the mIoU with (refined) and without (raw)
the joint bilateral filter. Benchmarks in --eval-root: coco_exc (small, medium, large), dram, egohos, ishape and
pidray; the last three report every subset and the instance-weighted mean 'all'. Writes <output-dir>/instance.json."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loss import gen2seg_inference_scale  # noqa: E402
from metrics.instance_prompt import (benchmark_noise_seed, deepest_interior_points, instance_map_uint8,  # noqa: E402
                                     load_coco_exc, load_label_image, load_rgb_image, prompt_ious)
from util.checkpoint import load_model  # noqa: E402

buckets = ('small', 'medium', 'large')
coco_area = 640 * 480
egohos_subsets = ('indomain', 'outdomain')
ishape_subsets = ('antenna', 'branch', 'fence', 'hanger', 'log', 'wire')


class InstanceMap:
    def __init__(self, checkpoint, num_sampling_steps):
        self.seed = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)['args'].seed
        self.model = load_model(checkpoint, 'ema', num_sampling_steps)
        self.size, self.patch = self.model.args.img_size, self.model.net.patch_size

    @torch.no_grad()
    def __call__(self, rgb, noise_seed):
        """[H, W, 3] CUDA instance map in 0..255 of an [H, W, 3] image in 0..1."""
        image = torch.from_numpy(rgb).permute(2, 0, 1)[None].cuda()
        height, width = image.shape[-2:]
        scale = self.size / min(height, width)
        size = (round(height * scale / self.patch) * self.patch, round(width * scale / self.patch) * self.patch)
        model_input = F.interpolate(image, size=size, mode='bilinear', align_corners=False, antialias=True)
        torch.manual_seed(noise_seed)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            prediction = self.model.generate(model_input * 2 - 1, None, task='instance').float()
        prediction = F.interpolate(prediction, size=rgb.shape[:2], mode='bilinear', align_corners=False)
        return instance_map_uint8(gen2seg_inference_scale(prediction[0]).permute(1, 2, 0))


def empty_scores():
    return {kind: {name: [] for name in buckets} for kind in ('refined', 'raw')}


def score(feature, targets, target_buckets, scores):
    """Append the refined and raw IoU of every [N, H, W] CUDA mask to scores[kind][bucket]."""
    if not len(targets):
        return
    ious = prompt_ious(feature, targets, deepest_interior_points(targets))
    for kind, iou in zip(('refined', 'raw'), ious):
        for name, value in zip(target_buckets, iou.tolist()):
            scores[kind][name].append(value)


def bucket(area):
    return 'small' if area < 32 ** 2 else 'medium' if area < 96 ** 2 else 'large'


def summarize(scores):
    result = {}
    for kind, by_bucket in scores.items():
        every = sum(by_bucket.values(), [])
        result[kind] = {'miou': 100 * float(np.mean(every)), 'instances': len(every)}
        result[kind].update({name: 100 * float(np.mean(v)) if v else None for name, v in by_bucket.items()})
    return result


def summarize_subsets(parts):
    """{subset: scores} -> {'all': summary over every instance, subset: summary}."""
    merged = empty_scores()
    for part in parts.values():
        for kind, by_bucket in part.items():
            for name, values in by_bucket.items():
                merged[kind][name] += values
    return {'all': summarize(merged), **{name: summarize(part) for name, part in parts.items()}}


def evaluate_coco(model, root):
    coco, annotations = load_coco_exc(str(root / 'coco' / 'annotations' / 'instances_val2017.json'))
    scores = empty_scores()
    for image_id in sorted(annotations):
        info = coco.loadImgs(image_id)[0]
        feature = model(load_rgb_image(root / 'coco' / 'val2017' / info['file_name']), model.seed + image_id)
        targets = torch.from_numpy(np.stack([coco.annToMask(a) for a in annotations[image_id]])).cuda() > 0
        score(feature, targets, [bucket(a['area']) for a in annotations[image_id]], scores)
    return summarize(scores)


def evaluate_dram(model, root):
    scores = empty_scores()
    for name in (root / 'dram' / 'test' / 'dram.txt').read_text().splitlines():
        feature = model(load_rgb_image(root / 'dram' / 'test' / f'{name}.jpg'),
                        benchmark_noise_seed(model.seed, 'dram', name))
        label = torch.from_numpy(load_label_image(root / 'dram' / 'labels' / f'{name}.png').astype(np.int32)).cuda()
        values = torch.unique(label)
        values = values[(values != 0) & (values != 255)]  # label values 0 and 255 are not instances
        score(feature, label[None] == values[:, None, None], ['large'] * len(values), scores)  # one mean over all
    return summarize(scores)


def score_label_maps(model, benchmark, samples):
    """Scores over (name, image path, label path) samples; every label value other than 0 is one target."""
    scores = empty_scores()
    for name, image_path, label_path in samples:
        label = torch.from_numpy(load_label_image(label_path).astype(np.int32)).cuda()
        values, areas = torch.unique(label, return_counts=True)
        values, areas = values[values != 0], areas[values != 0]
        feature = model(load_rgb_image(image_path), benchmark_noise_seed(model.seed, benchmark, name))
        target_buckets = [bucket(a * coco_area / label.numel()) for a in areas.tolist()]
        score(feature, label[None] == values[:, None, None], target_buckets, scores)
    return scores


def evaluate_egohos(model, root):
    parts = {}
    for subset in egohos_subsets:
        directory = root / 'egohos' / f'test_{subset}'
        images = sorted((directory / 'image').glob('*.jpg'))
        samples = [(path.stem, path, directory / 'label' / f'{path.stem}.png') for path in images]
        parts[subset] = score_label_maps(model, f'egohos_{subset}', samples)
    return summarize_subsets(parts)


def evaluate_ishape(model, root):
    parts = {}
    for subset in ishape_subsets:
        directory = root / 'ishape' / subset / 'val'
        images = sorted((directory / 'image').glob('*.jpg'))
        samples = [(path.stem, path, directory / 'instance_map' / f'{path.stem}.png') for path in images]
        parts[subset] = score_label_maps(model, subset, samples)
    return summarize_subsets(parts)


def evaluate_pidray(model, root):
    coco = COCO(str(root / 'pidray' / 'annotations' / 'test_easy.json'))
    annotations = {}
    for annotation in coco.loadAnns(coco.getAnnIds(iscrowd=0)):
        annotations.setdefault(annotation['image_id'], []).append(annotation)
    scores = empty_scores()
    for image_id in sorted(annotations):
        info = coco.loadImgs(image_id)[0]
        feature = model(load_rgb_image(root / 'pidray' / 'test' / info['file_name']),
                        benchmark_noise_seed(model.seed, 'pidray_easy', image_id))
        targets = torch.from_numpy(np.stack([coco.annToMask(a) for a in annotations[image_id]])).cuda() > 0
        score(feature, targets, [bucket(a['area']) for a in annotations[image_id]], scores)
    return summarize_subsets({'easy': scores})


evaluators = {'coco_exc': evaluate_coco, 'dram': evaluate_dram, 'egohos': evaluate_egohos, 'ishape': evaluate_ishape,
              'pidray': evaluate_pidray}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--num-sampling-steps', type=int, default=1)
    parser.add_argument('--benchmarks', nargs='+', choices=list(evaluators), default=list(evaluators))
    parser.add_argument('--eval-root', type=Path, default=Path('datasets/eval/instance_segmentation'))
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = InstanceMap(args.checkpoint, args.num_sampling_steps)
    result = {'checkpoint': args.checkpoint}
    for name in args.benchmarks:
        start = time.time()
        result[name] = evaluators[name](model, args.eval_root)
        print(name, f'{time.time() - start:.0f}s', json.dumps(result[name]), flush=True)
        (args.output_dir / 'instance.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
