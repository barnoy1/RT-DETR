"""Run inference on a folder of images and save visualization outputs.

Usage example:
  python tools/inference.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml \
      -r /path/to/checkpoint.pth -i /path/to/images -o /path/to/out --device cuda:0 --batch-size 8
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import argparse
import json
import traceback
from typing import List
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw
import numpy as np
try:
    from tqdm import tqdm
except Exception:
    # fallback: identity iterator when tqdm is not installed
    def tqdm(x, **kwargs):
        return x

# Third-party imports for ONNX and TensorRT
import onnxruntime as ort
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa
except ImportError:
    trt = None
    cuda = None

from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils


IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


def list_images(folder: str) -> List[str]:
    files = []
    for fn in sorted(os.listdir(folder)):
        _, ext = os.path.splitext(fn.lower())
        if ext in IMG_EXTS:
            files.append(os.path.join(folder, fn))
    return files


def load_checkpoint(path: str):
    checkpoint = torch.load(path, map_location='cpu')
    if isinstance(checkpoint, dict):
        if 'ema' in checkpoint:
            state = checkpoint['ema'].get('module', checkpoint['ema'])
            return state
        if 'model' in checkpoint:
            return checkpoint['model']
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
        print(f'Warning: skipped {len(skipped_shape)} tensors due to shape mismatch (checkpoint vs model).')
        preview = skipped_shape[:8]
        for key, src_shape, dst_shape in preview:
            print(f'  - {key}: {src_shape} -> {dst_shape}')
        if len(skipped_shape) > len(preview):
            print(f'  ... and {len(skipped_shape) - len(preview)} more')

    if missing_keys:
        print(f'Warning: {len(missing_keys)} missing model keys after load (expected for head mismatch).')
    if unexpected_keys:
        print(f'Warning: {len(unexpected_keys)} unexpected checkpoint keys ignored.')


def build_preproc_from_cfg(cfg, default_size=640):
    """Build a torchvision transforms pipeline from cfg.yaml_cfg val_dataloader.dataset.transforms.ops
    Supports Resize and ConvertPILImage and Normalize keys.
    Falls back to Resize(default_size) + ToTensor.
    """
    try:
        ops = cfg.yaml_cfg.get('val_dataloader', {}).get('dataset', {}).get('transforms', {}).get('ops', None)
    except Exception:
        ops = None

    transforms_list = []
    if ops is None:
        transforms_list.append(T.Resize((default_size, default_size)))
        transforms_list.append(T.ToTensor())
        return T.Compose(transforms_list)

    for op in ops:
        if not isinstance(op, dict):
            continue
        typ = op.get('type')
        if typ == 'Resize':
            size = op.get('size')
            if isinstance(size, list) and len(size) == 2:
                transforms_list.append(T.Resize((size[0], size[1])))
            elif isinstance(size, int):
                transforms_list.append(T.Resize(size))
        elif typ == 'ConvertPILImage':
            # ConvertPILImage(dtype='float32', scale=True) -> ToTensor
            transforms_list.append(T.ToTensor())
        elif typ == 'Normalize':
            mean = op.get('mean', [0.485, 0.456, 0.406])
            std = op.get('std', [0.229, 0.224, 0.225])
            transforms_list.append(T.Normalize(mean=mean, std=std))
        else:
            # unsupported op -> skip
            pass

    if len(transforms_list) == 0:
        transforms_list.append(T.Resize((default_size, default_size)))
        transforms_list.append(T.ToTensor())

    return T.Compose(transforms_list)


def draw_boxes(im_pil: Image.Image, labels, boxes, scores, score_thr=0.3):
    draw = ImageDraw.Draw(im_pil)
    for j, b in enumerate(boxes):
        s = float(scores[j]) if scores is not None else 1.0
        if s < score_thr:
            continue
        box = list(b)
        draw.rectangle(box, outline='red')
        lab = int(labels[j]) if labels is not None else -1
        draw.text((box[0], box[1]), text=f"{lab} {round(s, 3)}", fill='blue')
    return im_pil


def warn_exception(context: str, exc: Exception, show_trace: bool = False):
    msg = f"Warning: {context}: {exc}"
    print(msg)
    if show_trace:
        print(traceback.format_exc())


class InferenceModel:
    def __init__(self, model_type, model_path, cfg_model=None, cfg_postprocessor=None, device='cpu'):
        self.model_type = model_type
        self.device = device
        self.model = None
        self.postprocessor = None

        # Map ONNXRuntime type strings to NumPy dtypes
        self._onnx_type_to_np_type = {
            "tensor(float)": np.float32,
            "tensor(float16)": np.float16,
            "tensor(int64)": np.int64,
            "tensor(int32)": np.int32,
            # Add other types as needed
        }

        if self.model_type == 'pytorch':
            self.model = cfg_model
            self.postprocessor = cfg_postprocessor
            if self.postprocessor:
                try:
                    if hasattr(self.postprocessor, 'deploy'):
                        self.postprocessor = self.postprocessor.deploy()
                except Exception as e:
                    warn_exception('PyTorch postprocessor.deploy failed', e, show_trace=False)
            
            self.model.to(device)
            self.model.eval()

        elif self.model_type == 'onnx':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']
            self.session = ort.InferenceSession(model_path, providers=providers)
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
        elif self.model_type == 'trt':
            if trt is None or cuda is None:
                raise ImportError("TensorRT and PyCUDA must be installed to use TRT engine.")
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            self.context = self.engine.create_execution_context()
            self.inputs = []
            self.outputs = []
            self.bindings = []
            for binding in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.get_binding_dtype(binding).itemsize
                host_mem = cuda.mem_alloc(size)
                self.bindings.append(int(host_mem))
                if self.engine.binding_is_input(binding):
                    self.inputs.append(host_mem)
                else:
                    self.outputs.append(host_mem)
            
            self.stream = cuda.Stream()
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    def forward(self, batch_images, orig_target_sizes):
        if self.model_type == 'pytorch':
            with torch.no_grad():
                outputs = self.model(batch_images, orig_target_sizes)
                if isinstance(outputs, dict) and self.postprocessor is not None:
                    outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs
        
        elif self.model_type == 'onnx':
            # ONNX model is expected to output labels, boxes, scores directly
            # Convert torch tensors to numpy
            batch_images_np = batch_images.cpu().numpy()
            # Get expected numpy dtype for orig_target_sizes from ONNX session
            orig_target_sizes_onnx_type_str = self.session.get_inputs()[1].type
            orig_target_sizes_np_dtype = self._onnx_type_to_np_type.get(orig_target_sizes_onnx_type_str, np.int64) # Default to int64

            orig_target_sizes_np = orig_target_sizes.cpu().numpy().astype(orig_target_sizes_np_dtype)

            ort_inputs = {
                self.input_name: batch_images_np, 
                "orig_target_sizes": orig_target_sizes_np
            }
            ort_outputs = self.session.run(self.output_names, ort_inputs)
            
            labels = torch.from_numpy(ort_outputs[0]).to(self.device)
            boxes = torch.from_numpy(ort_outputs[1]).to(self.device)
            scores = torch.from_numpy(ort_outputs[2]).to(self.device)
            return labels, boxes, scores

        elif self.model_type == 'trt':
            # Assume TRT model also outputs labels, boxes, scores directly
            # Convert torch tensors to numpy, then to cuda memory
            batch_images_np = batch_images.cpu().numpy()
            orig_target_sizes_np = orig_target_sizes.cpu().numpy()

            # Allocate device memory for inputs
            cuda.memcpy_htod(self.inputs[0], batch_images_np)
            cuda.memcpy_htod(self.inputs[1], orig_target_sizes_np)

            # Execute inference
            self.context.execute_async_v2(
                bindings=self.bindings,
                stream_handle=self.stream.handle
            )

            # Fetch outputs from device
            outputs_np = [np.empty(self.engine.get_binding_shape(self.engine[i + len(self.inputs)]), dtype=np.float32) for i in range(len(self.outputs))]
            for i, out in enumerate(self.outputs):
                cuda.memcpy_dtoh_async(outputs_np[i], out, self.stream)
            self.stream.synchronize()

            labels = torch.from_numpy(outputs_np[0]).to(self.device)
            boxes = torch.from_numpy(outputs_np[1]).to(self.device)
            scores = torch.from_numpy(outputs_np[2]).to(self.device)
            return labels, boxes, scores
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

def _get_inference_backend(args):
    model_count = sum([bool(args.resume), bool(args.onnx_model), bool(args.trt_engine)])
    if model_count > 1:
        raise ValueError("Please specify only one model source (--resume, --onnx_model, or --trt_engine).")
    
    if args.onnx_model:
        return 'onnx', args.onnx_model
    elif args.trt_engine:
        return 'trt', args.trt_engine
    elif args.resume:
        return 'pytorch', args.resume
    else:
        raise ValueError("No model source specified. Please use --resume, --onnx_model, or --trt_engine.")


def main(args):
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    update_dict = yaml_utils.parse_cli(args.update) if args.update is not None else {}
    update_dict.update({k: v for k, v in args.__dict__.items() if k not in ['update', ] and v is not None})

    # Resolve config path: allow relative paths from project root (repo root)
    config_path = args.config
    if config_path is None:
        raise FileNotFoundError('Missing --config argument')

    if not os.path.isabs(config_path) and not os.path.exists(config_path):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        alt = os.path.join(project_root, config_path)
        if os.path.exists(alt):
            config_path = alt
        else:
            raise FileNotFoundError(f"Config file not found: {args.config} (tried {alt})")

    cfg = YAMLConfig(config_path, **update_dict)
    print('cfg: ', cfg.__dict__)

    device_str = args.device if args.device is not None else 'cpu'
    # For ONNX and TRT, the device needs to be handled by their respective runtimes.
    # For PyTorch, we explicitly set it.
    device = torch.device(device_str)

    model_type, model_path = _get_inference_backend(args)

    if model_type == 'pytorch':
        model_instance = cfg.model
        postprocessor_instance = cfg.postprocessor
        # load checkpoint
        state = load_checkpoint(model_path)
        safe_load_state_dict(model_instance, state, model_name='RTDETR')
        print(f'Loaded PyTorch model state from {model_path}')

        # try to switch to deploy mode if available
        try:
            if hasattr(model_instance, 'deploy'):
                model_instance = model_instance.deploy()
        except Exception as e:
            warn_exception('PyTorch model.deploy failed', e, args.debug)
        
        inference_model = InferenceModel(model_type, None, cfg_model=model_instance, cfg_postprocessor=postprocessor_instance, device=device_str)

    elif model_type == 'onnx':
        inference_model = InferenceModel(model_type, model_path, device=device_str)
        # For ONNX, postprocessing is assumed to be embedded in the ONNX model
        # So we don't need cfg.postprocessor here.
        
    elif model_type == 'trt':
        inference_model = InferenceModel(model_type, model_path, device=device_str)
        # For TRT, postprocessing is assumed to be embedded in the TRT engine
        # So we don't need cfg.postprocessor here.
        
    else:
        raise ValueError("Invalid model type or path configuration.")


    # build preprocessing from config to match val pipeline
    transforms = build_preproc_from_cfg(cfg, default_size=args.input_size or 640)

    # prepare label->name mapping if possible
    label2name = None
    try:
        val_loader = cfg.val_dataloader
        dataset = val_loader.dataset
        # dataset may provide label2category and category2name
        if hasattr(dataset, 'label2category') and hasattr(dataset, 'category2name'):
            l2c = dataset.label2category
            c2n = dataset.category2name
            label2name = {int(l): c2n[int(c)] for l, c in l2c.items()}
    except Exception:
        label2name = None

    # resolve input and output directories against project root when given as relative paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_dir = args.input_dir
    if input_dir is None:
        raise FileNotFoundError('Missing --input-dir argument')

    if not os.path.isabs(input_dir) and not os.path.exists(input_dir):
        alt_in = os.path.join(project_root, input_dir)
        if os.path.exists(alt_in):
            input_dir = alt_in
        else:
            raise FileNotFoundError(f"Input directory not found: {args.input_dir} (tried {alt_in}). Create it or pass absolute path.")

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(project_root, 'tools', 'inference_out')
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(project_root, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    image_paths = list_images(input_dir)
    if len(image_paths) == 0:
        print('No images found in', input_dir)
        return

    results_json = []

    # threaded image loading and preprocessing (avoids multiprocessing pickling issues)
    num_workers = args.num_workers if args.num_workers is not None else 4
    debug = args.debug

    def _load_and_preprocess_with_transform(path, transforms):
        try:
            im = Image.open(path).convert('RGB')
            w, h = im.size
            t = transforms(im)
            return (path, t, (w, h))
        except Exception as e:
            if debug:
                warn_exception(f'preprocess failed for {path}', e, debug)
            return (path, None, None)

    worker = partial(_load_and_preprocess_with_transform, transforms=transforms)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        processed = list(tqdm(executor.map(worker, image_paths), total=len(image_paths), desc='Preprocessing'))

    # filter out failed
    processed = [x for x in processed if x[1] is not None]

    batch_size = max(1, args.batch_size)
    for i in range(0, len(processed), batch_size):
        batch_items = processed[i:i+batch_size]
        batch_paths = [x[0] for x in batch_items]
        tensors = [x[1] for x in batch_items]
        orig_sizes = [x[2] for x in batch_items]

        batch = torch.stack(tensors, dim=0).to(device)
        orig_sizes_t = torch.tensor(orig_sizes, dtype=torch.float32, device=device)

        # unified inference call
        labels_b, boxes_b, scores_b = inference_model.forward(batch, orig_sizes_t)

        # normalize to list of per-image outputs
        per_image_outputs = []
        for bi in range(len(batch_paths)):
            labels_i = labels_b[bi]
            boxes_i = boxes_b[bi]
            scores_i = scores_b[bi]
            per_image_outputs.append((labels_i, boxes_i, scores_i))

        # load corresponding PIL images again for drawing (small overhead)
        im_pils = [Image.open(p).convert('RGB') for p in batch_paths]

        for p, im_pil, out in zip(batch_paths, im_pils, per_image_outputs):
            # expected out: (labels, boxes, scores)
            try:
                labels, boxes, scores = out
                # convert tensors to CPU lists
                if isinstance(boxes, torch.Tensor):
                    boxes_np = boxes.detach().cpu().numpy().tolist()
                else:
                    boxes_np = [list(map(float, b)) for b in boxes]
                if isinstance(scores, torch.Tensor):
                    scores_np = scores.detach().cpu().numpy().tolist()
                else:
                    scores_np = [float(s) for s in scores]
                if isinstance(labels, torch.Tensor):
                    labels_np = labels.detach().cpu().numpy().tolist()
                else:
                    labels_np = [int(l) for l in labels]
            except Exception as e:
                warn_exception('failed to parse model outputs for an image', e, args.debug)
                # unexpected format
                labels_np, boxes_np, scores_np = [], [], []

            # map labels to names if possible
            if label2name is not None:
                names = [label2name.get(int(l), str(int(l))) for l in labels_np]
            else:
                names = [str(int(l)) for l in labels_np]

            # draw and save image
            im_out = im_pil.copy()
            # draw text with class names
            draw = ImageDraw.Draw(im_out)
            for j, b in enumerate(boxes_np):
                s = float(scores_np[j]) if scores_np is not None and j < len(scores_np) else 1.0
                if s < args.score_thr:
                    continue
                draw.rectangle(list(b), outline='red')
                draw.text((b[0], b[1]), text=f"{names[j]} {round(s, 3)}", fill='blue')

            out_name = os.path.basename(p)
            im_out.save(os.path.join(output_dir, out_name))

            # accumulate simple result
            results_json.append({
                'file_name': out_name,
                'boxes': boxes_np,
                'labels': labels_np,
                'scores': scores_np,
            })

        print(f'Processed {min(i+batch_size, len(image_paths))}/{len(image_paths)}')

    if args.save_json:
        json_path = os.path.join(output_dir, 'detections.json')
        with open(json_path, 'w') as f:
            json.dump(results_json, f)
        print('Wrote detections json to', json_path)

    dist_utils.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, help='Path to PyTorch checkpoint file.')
    parser.add_argument('-i', '--input-dir', type=str, required=True)
    parser.add_argument('-o', '--output-dir', type=str, required=True)
    parser.add_argument('-d', '--device', type=str, default='cpu')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--input-size', type=int, default=640, help='size to resize shorter/longer side (square)')
    parser.add_argument('--score-thr', type=float, default=0.3)
    parser.add_argument('--save-json', action='store_true')
    parser.add_argument('--seed', type=int, help='exp reproducibility')
    parser.add_argument('-u', '--update', nargs='+', help='update yaml config')

    # env / distributed
    parser.add_argument('--print-method', type=str, default='builtin', help='print method')
    parser.add_argument('--print-rank', type=int, default=0, help='print rank id')
    parser.add_argument('--local-rank', type=int, help='local rank id')
    parser.add_argument('--num-workers', type=int, help='number of workers for data loading')
    parser.add_argument('--debug', action='store_true', help='print exception tracebacks for fallbacks')

    parser.add_argument('--onnx_model', type=str, help='Path to an ONNX model file (e.g., /path/to/model.onnx)')
    parser.add_argument('--trt_engine', type=str, help='Path to a TensorRT engine file')

    args = parser.parse_args()
    main(args)
