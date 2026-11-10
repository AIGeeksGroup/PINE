# Copyright 2026 Zeyu Ren. Licensed under the Apache License, Version 2.0.
"""Training batches of random aspect ratio and field of view at a fixed token budget.

AspectBatchSampler gives every batch one frame size (height, width): multiples of the patch, about --tokens patches,
width / height log-uniform in aspect_range. Every item of the batch is (index, height, width, side): the dataset takes
a crop of that aspect ratio at a random place in the frame, side (uniform in side_range) times the largest such crop,
and resizes it to the frame size (dataloader/point_transforms.random_frame_crop). A smaller side narrows the field
of view, so the batches vary both the aspect ratio and the field of view.
"""
import math
import random


def frame_size(aspect, tokens, patch):
    """(height, width), multiples of patch, of about tokens patches with width / height = aspect."""
    return max(round(math.sqrt(tokens / aspect)), 1) * patch, max(round(math.sqrt(tokens * aspect)), 1) * patch


class AspectBatchSampler:
    """Batches of the indices of sampler as (index, height, width, side) items, one frame size per batch."""

    drop_last = True

    def __init__(self, sampler, batch_size, aspect_range, side_range, tokens, patch):
        self.sampler = sampler
        self.batch_size = batch_size
        self.log_aspect_range = tuple(math.log(aspect) for aspect in aspect_range)
        self.side_range = side_range
        self.tokens = tokens
        self.patch = patch

    def __iter__(self):
        batch = []
        for index in self.sampler:
            if not batch:
                height, width = frame_size(math.exp(random.uniform(*self.log_aspect_range)), self.tokens, self.patch)
            batch.append((int(index), height, width, random.uniform(*self.side_range)))
            if len(batch) == self.batch_size:
                yield batch
                batch = []

    def __len__(self):
        return len(self.sampler) // self.batch_size
