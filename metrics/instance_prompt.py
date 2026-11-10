# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Point-prompted instance segmentation under the gen2seg protocol, on the GPU.

A prompt point selects a mask from an instance colour map: Gaussian-weighted query colour at the point, colour
similarity, joint bilateral filter guided by the map, threshold. The filter and the prompt point (deepest interior
point of the target) reproduce cv2.ximgproc.jointBilateralFilter and cv2.distanceTransform."""
import hashlib
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO


# cv2.distanceTransform(DIST_L2, 5) (distransform.cpp): chamfer steps 1, 1.4 and 2.1969 in 16-bit fixed point
CHAMFER_STEPS = tuple((weight, dy, dx) for weight, offsets in (
    (65536, ((0, 1), (1, 0), (0, -1), (-1, 0))),
    (91750, ((1, 1), (1, -1), (-1, 1), (-1, -1))),
    (143976, ((1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)))) for dy, dx in offsets)
FAR = 1 << 30


def deepest_interior_points(masks, chunk=64):
    """[N, 2] (x, y) of the pixel deepest inside each of [N, H, W] boolean masks.

    The chamfer distance to the nearest pixel outside the mask is relaxed in place until it stops to change, chunk masks
    at a time. Pixels outside the image do not count as outside the mask, as in cv2. The fixed-point distances become
    float32 as in cv2, and argmax takes the first maximum in row-major order, as numpy argmax does."""
    height, width = masks.shape[-2:]
    indices = []
    for part in masks.split(chunk):
        distance = torch.where(part, FAR, 0).int()
        while True:
            previous = distance.clone()
            for weight, dy, dx in CHAMFER_STEPS:
                padded = F.pad(distance, (2, 2, 2, 2), value=FAR)
                distance = torch.minimum(distance, padded[:, 2 + dy:2 + dy + height, 2 + dx:2 + dx + width] + weight)
            if torch.equal(distance, previous):
                break
        indices.append((distance.float() * (1.0 / 65536)).flatten(1).argmax(1))
    index = torch.cat(indices)
    return torch.stack((index % width, index // width), 1)


def bilateral_weights(guidance, diameter, sigma_color, sigma_space):
    """[K, H, W] float32 weights of the K disc offsets for a [H, W, 3] uint8 guidance, and the offsets."""
    radius = max(diameter // 2, 1)
    padded = F.pad(guidance.permute(2, 0, 1)[None].float(), (radius,) * 4, mode='reflect')[0].to(torch.int32)
    centre = guidance.permute(2, 0, 1).to(torch.int32)
    height, width = guidance.shape[:2]
    lookup = torch.exp(torch.arange(256 * 3, device=guidance.device, dtype=torch.float64) ** 2
                       * (-0.5 / sigma_color ** 2)).float()
    weights, offsets = [], []
    for i in range(-radius, radius + 1):
        for j in range(-radius, radius + 1):
            if i * i + j * j > radius * radius:
                continue
            space = torch.tensor((i * i + j * j) * (-0.5 / sigma_space ** 2), dtype=torch.float64).exp().float()
            neighbour = padded[:, radius + i:radius + i + height, radius + j:radius + j + width]
            weights.append(space.to(guidance.device) * lookup[(neighbour - centre).abs().sum(0)])
            offsets.append((i, j))
    return torch.stack(weights), offsets, radius


def joint_bilateral_filter(source, weights, offsets, radius):
    """[N, H, W] uint8 sources filtered with precomputed weights, rounded to uint8 as cv2 saturate_cast."""
    height, width = source.shape[-2:]
    padded = F.pad(source[:, None].float(), (radius,) * 4, mode='reflect')[:, 0]
    total = torch.zeros(source.shape, device=source.device, dtype=torch.float32)
    weight_sum = torch.zeros((height, width), device=source.device, dtype=torch.float32)
    for weight, (i, j) in zip(weights, offsets):
        total += weight * padded[:, radius + i:radius + i + height, radius + j:radius + j + width]
        weight_sum += weight
    return torch.round(total / weight_sum).clamp(0, 255).to(torch.uint8)


def prompt_masks(feature, points, gaussian_sigma, mask_threshold, similarity_epsilon, jbf_diameter, jbf_sigma_color,
                 jbf_sigma_space, chunk=16):
    """(refined, raw) [N, H, W] boolean masks for N (x, y) points on a [H, W, 3] float feature map (0-255) on the GPU."""
    height, width = feature.shape[:2]
    guidance = feature.clamp(0, 255).to(torch.uint8)
    weights, offsets, radius = bilateral_weights(guidance, jbf_diameter, jbf_sigma_color, jbf_sigma_space)
    ys = torch.arange(height, device=feature.device, dtype=torch.float32)[:, None]
    xs = torch.arange(width, device=feature.device, dtype=torch.float32)[None]
    refined, raw = [], []
    for start in range(0, len(points), chunk):
        point = points[start:start + chunk].float()
        squared = ((xs - point[:, 0, None, None]) / (width - 1)) ** 2 + ((ys - point[:, 1, None, None]) / (height - 1)) ** 2
        gaussian = torch.exp(-squared / (2.0 * gaussian_sigma ** 2))
        query = torch.einsum('nhw,hwc->nc', gaussian, feature) / (gaussian.sum((1, 2))[:, None] + similarity_epsilon)
        similarity = 1.0 / (torch.linalg.vector_norm(feature[None] - query[:, None, None], dim=-1) + similarity_epsilon)
        similarity = similarity - similarity.amin((1, 2), keepdim=True)
        similarity = similarity / (similarity.amax((1, 2), keepdim=True) + similarity_epsilon)
        similarity_uint8 = (similarity * 255.0).clamp(0, 255).to(torch.uint8)
        refined.append(joint_bilateral_filter(similarity_uint8, weights, offsets, radius).float() > mask_threshold)
        raw.append(similarity * 255.0 > mask_threshold)
    return torch.cat(refined), torch.cat(raw)


PROMPT = dict(gaussian_sigma=0.01, mask_threshold=3.0, similarity_epsilon=1e-6, jbf_diameter=9, jbf_sigma_color=75.0,
              jbf_sigma_space=75.0)


def percentile(values, q):
    """np.percentile (linear interpolation) of a CUDA tensor of any size."""
    ordered = values.flatten().sort().values
    position = q / 100 * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def instance_map_uint8(feature):
    """[H, W, 3] float instance map in 0..255 divided by its 99th percentile, clipped to 255 and cast to uint8 values
    (kept as float), the map that the prompts read."""
    return (feature / percentile(feature, 99) * 255.0).clamp(0, 255).to(torch.uint8).float()


def prompt_ious(feature, targets, points, chunk=64):
    """(refined, raw) IoU of the protocol prompt at points ([N, 2] x, y) for [N, H, W] boolean targets, chunk targets at
    a time, since a boolean sum makes an int64 copy of the masks."""
    ious = []
    for part, point in zip(targets.split(chunk), points.split(chunk)):
        refined, raw = prompt_masks(feature, point, **PROMPT)
        ious.append([(mask & part).sum((1, 2)) / (mask | part).sum((1, 2)).clamp_min(1) for mask in (refined, raw)])
    return tuple(torch.cat(kind) for kind in zip(*ious))


# COCOexc (gen2seg Appendix D): the COCO val2017 categories absent from the Hypersim and Virtual KITTI 2 training
# sets of gen2seg, used for zero-shot instance segmentation.
coco_exc_categories = [
    "person", "traffic light", "handbag", "bird", "boat", "truck", "umbrella", "cow", "banana",
    "motorcycle", "backpack", "carrot", "sheep", "donut", "kite", "bicycle", "broccoli", "cake",
    "suitcase", "orange", "bus", "pizza", "horse", "surfboard", "zebra", "sports ball", "elephant",
    "tie", "skis", "giraffe", "tennis racket", "dog", "cat", "train", "skateboard", "sandwich",
    "baseball glove", "baseball bat", "airplane", "hot dog", "frisbee", "fire hydrant", "stop sign",
    "bear", "snowboard", "parking meter",
]


def load_coco_exc(annotation_path):
    """(COCO api, {image id: non-crowd annotations of the COCOexc categories})."""
    coco = COCO(annotation_path)
    exc_category_ids = coco.getCatIds(catNms=coco_exc_categories)
    annotations_by_image = defaultdict(list)
    for annotation in coco.loadAnns(coco.getAnnIds(catIds=exc_category_ids, iscrowd=0)):
        annotations_by_image[annotation["image_id"]].append(annotation)
    return coco, annotations_by_image


def load_rgb_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def load_label_image(label_path):
    return cv2.imread(str(label_path), cv2.IMREAD_UNCHANGED)


def benchmark_noise_seed(base_seed, benchmark_name, sample_identifier):
    """Noise seed of one benchmark sample: the base seed plus a hash of the benchmark and the sample."""
    sample_key = f"{benchmark_name}:{sample_identifier}".encode()
    digest = hashlib.blake2b(sample_key, digest_size=8).digest()
    sample_offset = int.from_bytes(digest, byteorder="little")
    maximum_seed = torch.iinfo(torch.int64).max
    return (base_seed + sample_offset) % maximum_seed
