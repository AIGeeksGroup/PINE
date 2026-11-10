# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Derived from JiT (https://github.com/LTH14/JiT), MIT License,
# Copyright (c) 2025 Tianhong Li. MetricLogger/SmoothedValue trace further
# back to DeiT and Detectron2 (Apache-2.0).
# See THIRD_PARTY_NOTICES.
import builtins
import datetime
import os
import time
from collections import defaultdict, deque
from pathlib import Path
import copy

import torch
import torch.distributed as dist


def print_main_process(*args, **kwargs):
    if is_main_process():
        builtins.print(*args, **kwargs)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", print_func=print_main_process):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.print = print_func

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    self.print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    self.print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0



def add_weight_decay(model, weight_decay=0, skip_list=(), feature_embedding_lr_scale=0.0):
    """AdamW parameter groups: no weight decay on biases and 1-D parameters; the DINOv3 encoder, when it is trained,
    gets its own groups with lr_scale = feature_embedding_lr_scale (util/lr_sched.py applies it)."""
    scales = {'default': 1.0, 'feature_embedding': feature_embedding_lr_scale}
    parameters_by_group = {name: ([], []) for name in scales}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        group_name = 'feature_embedding' if 'y_embedder' in name.split('.') else 'default'
        no_decay, decay = parameters_by_group[group_name]
        skip_decay = param.ndim == 1 or name.endswith('.bias')
        skip_decay = skip_decay or name in skip_list or 'diffloss' in name
        if skip_decay:
            no_decay.append(param)
        else:
            decay.append(param)
    groups = []
    for group_name, (no_decay, decay) in parameters_by_group.items():
        for parameters, group_decay in ((no_decay, 0.0), (decay, weight_decay)):
            if not parameters and group_name != 'default':
                continue
            group = {'params': parameters, 'weight_decay': group_decay}
            if group_name != 'default':
                group['lr_scale'] = scales[group_name]
            groups.append(group)
    return groups


FROZEN_ENCODER_PREFIX = 'net.y_embedder.'


def save_model(args, model_without_ddp, optimizer, epoch, epoch_name=None, global_step=None,
               resume_epoch=None, checkpoint_aliases=()):
    if not is_main_process():
        return
    if epoch_name is None:
        epoch_name = str(epoch)
    output_dir = Path(args.output_dir)
    checkpoint_path = output_dir / ('checkpoint-%s.pth' % epoch_name)

    # Omit the frozen encoder.
    strip_encoder = not getattr(model_without_ddp.net, 'finetune_feature_embedding', False)
    trainable_state = {
        name: value for name, value in model_without_ddp.state_dict().items()
        if not (strip_encoder and name.startswith(FROZEN_ENCODER_PREFIX))
    }
    ema_state_dict1 = dict(trainable_state)
    for i, (name, _value) in enumerate(model_without_ddp.named_parameters()):
        if name in ema_state_dict1:
            ema_state_dict1[name] = model_without_ddp.ema_params1[i]
    to_save = {
        'model': trainable_state,
        'model_ema1': ema_state_dict1,
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'global_step': global_step,
        # Sub-epoch saves resume the current epoch; end-of-epoch saves resume the next one.
        'resume_epoch': resume_epoch,
        'args': args,
    }
    if strip_encoder:
        to_save['dinov3_stripped'] = {'prefix': FROZEN_ENCODER_PREFIX}

    # Publish once, then share the snapshot through atomic hard-link aliases.
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + '.tmp')
    torch.save(to_save, tmp_path)
    os.replace(tmp_path, checkpoint_path)
    for name in checkpoint_aliases:
        alias_path = output_dir / f'checkpoint-{name}.pth'
        alias_tmp_path = alias_path.with_suffix(alias_path.suffix + '.tmp')
        os.link(checkpoint_path, alias_tmp_path)
        os.replace(alias_tmp_path, alias_path)


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        # NCCL needs the tensor on the GPU; gloo (CPU-only training) needs it on
        # the host, so follow whatever this process actually has.
        x_reduce = torch.tensor(
            x, device='cuda' if torch.cuda.is_available() else 'cpu')
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x
