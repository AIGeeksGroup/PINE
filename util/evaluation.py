"""Standalone evaluation artifacts belong to their experiment; online metrics stay in training logs."""
import json
from pathlib import Path


def evaluation_label(weights, use_camera):
    return weights if use_camera else f'{weights}_nocam'


def evaluation_directory(args, task, weights):
    run_name = Path(args.pretrained).stem if args.pretrained else 'untrained'
    return Path(args.output_dir) / 'eval' / run_name / weights / task


def save_evaluation_results(args, task, weights, results):
    if not args.evaluate_gen:
        return
    output_dir = evaluation_directory(args, task, weights)
    output_dir.mkdir(parents=True, exist_ok=True)
    record = {
        'epoch': None,
        'checkpoint': args.pretrained,
        'weights': weights,
        'num_sampling_steps': args.num_sampling_steps,
        'resize_height': args.eval_depth_resize_height,
        'metrics': results,
    }
    with (output_dir / 'metrics.jsonl').open('a') as handle:
        handle.write(json.dumps(record) + '\n')
