"""Evaluation script for COCO-format datasets

Usage example:
  python tools/eval.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml -r /path/to/checkpoint.pth --device cuda:0
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import torch

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import det_engine


class _DummyCriterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, *args, **kwargs):
        return {}


def load_checkpoint(path: str):
    checkpoint = torch.load(path, map_location='cpu')
    # support different checkpoint layouts
    if isinstance(checkpoint, dict):
        if 'ema' in checkpoint:
            state = checkpoint['ema'].get('module', checkpoint['ema'])
            return state
        if 'model' in checkpoint:
            return checkpoint['model']
        # fallback: assume the dict is the state_dict
        return checkpoint
    return checkpoint


def _normalize_state_dict_keys(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if not any(k.startswith('module.') for k in state_dict.keys()):
        return state_dict
    return {k[7:] if k.startswith('module.') else k: v for k, v in state_dict.items()}


def safe_load_state_dict(model, state_dict, model_name='model'):
    state_dict = _normalize_state_dict_keys(state_dict)
    model_state = model.state_dict()

    compatible = {}
    skipped_shape = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            skipped_shape.append((key, tuple(value.shape), tuple(model_state[key].shape)))
            continue
        compatible[key] = value

    missing_keys, unexpected_keys = model.load_state_dict(compatible, strict=False)

    print(f'Loaded {len(compatible)}/{len(model_state)} tensors into {model_name}')
    if skipped_shape:
        print(f'Warning: skipped {len(skipped_shape)} tensors due to shape mismatch.')
    if missing_keys:
        print(f'Warning: {len(missing_keys)} missing model keys after load.')
    if unexpected_keys:
        print(f'Warning: {len(unexpected_keys)} unexpected checkpoint keys ignored.')


def main(args):
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update_dict = yaml_utils.parse_cli(args.update) if args.update is not None else {}
    update_dict.update({k: v for k, v in args.__dict__.items() if k not in ['update', ] and v is not None})

    cfg = YAMLConfig(args.config, **update_dict)
    print('cfg: ', cfg.__dict__)

    device = torch.device(args.device if args.device is not None else 'cpu')

    model = cfg.model

    # load checkpoint if provided
    if args.resume:
        state = load_checkpoint(args.resume)
        safe_load_state_dict(model, state, model_name='RTDETR')
        print(f'Loaded model state from {args.resume}')

    model = model.to(device)

    # criterion (may not be used during eval but det_engine expects a module)
    try:
        criterion = cfg.criterion
    except Exception:
        criterion = _DummyCriterion()

    postprocessor = cfg.postprocessor

    # prepare dataloader and evaluator
    val_loader = cfg.val_dataloader
    try:
        evaluator = cfg.evaluator
    except Exception as e:
        print(f'Warning: could not create evaluator: {e}')
        evaluator = None

    # run evaluation
    stats, coco_evaluator = det_engine.evaluate(model, criterion, postprocessor, val_loader, evaluator, device)

    print('Evaluation stats:')
    for k, v in stats.items():
        print(f"{k}: {v}")

    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='checkpoint file')
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env / distributed
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')
    parser.add_argument('--local-rank', type=int, help='local rank id')

    args = parser.parse_args()
    main(args)

