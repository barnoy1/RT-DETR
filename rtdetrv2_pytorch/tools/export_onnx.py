"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import torch.nn as nn

from src.core import YAMLConfig, yaml_utils


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


def main(args, ):
    """main
    """
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({k: v for k, v in args.__dict__.items() \
                        if k not in ['update', ] and v is not None})
    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        if 'ema' in checkpoint:
            state = checkpoint['ema']['module']
        else:
            state = checkpoint['model']

        # NOTE load train mode state -> convert to deploy mode
        safe_load_state_dict(cfg.model, state, model_name='RTDETR')

    else:
        # raise AttributeError('Only support resume to load model.state_dict by now.')
        print('not load model.state_dict, use default init state dict...')

    class Model(nn.Module):
        def __init__(self, ) -> None:
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    model = Model()
    model.eval()

    data = torch.rand(1, 3, args.input_size, args.input_size)
    size = torch.tensor([[args.input_size, args.input_size]])
    _ = model(data, size)

    dynamic_axes = {
        'images': {0: 'N', },
        'orig_target_sizes': {0: 'N'}
    }

    torch.onnx.export(
        model,
        (data, size),
        args.output_file,
        input_names=['images', 'orig_target_sizes'],
        output_names=['labels', 'boxes', 'scores'],
        dynamic_axes=dynamic_axes,
        opset_version=18,
        verbose=False,
        do_constant_folding=True,
    )

    if args.check:
        import onnx
        onnx_model = onnx.load(args.output_file)
        onnx.checker.check_model(onnx_model)
        print('Check export onnx model done...')

    if args.simplify:
        import onnx 
        import onnxsim
        overwrite_input_shapes = {
            'images': list(data.shape),
            'orig_target_sizes': list(size.shape),
        }
        onnx_model_simplify, check = onnxsim.simplify(
            args.output_file,
            overwrite_input_shapes=overwrite_input_shapes,
        )
        onnx.save(onnx_model_simplify, args.output_file)
        print(f'Simplify onnx model {check}...')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--output_file', '-o', type=str, default='model.onnx')
    parser.add_argument('--input_size', '-s', type=int, default=640)
    parser.add_argument('--check', action='store_true', default=False)
    parser.add_argument('--simplify', action='store_true', default=False)
    parser.add_argument('--update', '-u', nargs='+', help='update yaml config')

    args = parser.parse_args()

    main(args)