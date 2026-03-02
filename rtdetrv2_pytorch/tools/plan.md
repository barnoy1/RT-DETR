# Plan for tools/inference.py and tools/eval.py

Purpose
- inference.py: run model inference on a folder of images and save visualized detections (and optional json results).
- eval.py: run COCO evaluation using the project's config and checkpoint; reuses the project's evaluator and det_engine.evaluate.

High-level steps for inference.py
1. Parse CLI arguments: --config, --resume, --input-dir, --output-dir, --device, --batch-size, --input-size.
2. Load project config via YAMLConfig and build model + postprocessor.
3. Load checkpoint (support 'ema' or 'model' keys) into model state_dict.
4. Create an inference wrapper that runs model and postprocessor; move to device.
5. Enumerate images in input-dir, run in batches, gather outputs (labels, boxes, scores).
6. Save results: draw boxes on images and write to output-dir; optionally save a JSON file of detections.
7. Provide examples and notes about image resizing, deploy vs. train mode, and GPU usage.

High-level steps for eval.py
1. Parse CLI arguments: --config, --resume, --device, optional distributed args.
2. Load YAMLConfig and build val dataloader and evaluator via cfg.evaluator.
3. Load checkpoint and set model state_dict.
4. Ensure criterion is available (use dummy no-op criterion if absent).
5. Call src.solver.det_engine.evaluate(model, criterion, postprocessor, val_dataloader, evaluator, device).
6. Print summarized COCO metrics and exit.

Dependencies and notes
- Uses existing YAMLConfig, cfg.model, cfg.postprocessor, cfg.val_dataloader, and cfg.evaluator.
- Checkpoint loading supports 'ema' and 'model' keys. Map location defaults to 'cpu'.
- For inference visual output, PIL and torchvision transforms are used.
- For distributed evaluation (eval.py), small helper calls to dist_utils are used to setup/cleanup; single-process evaluation works without explicit distributed setup.

Usage examples
- Inference (single-machine):
  python tools/inference.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml -r /path/to/checkpoint.pth -i /path/to/images -o /path/to/out --device cuda:0 --batch-size 8

- Evaluation on COCO val set:
  python tools/eval.py -c configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml -r /path/to/checkpoint.pth --device cuda:0

Security and performance
- For faster inference use batch-size > 1 and a GPU device.
- The postprocessor expects orig sizes; keep images unmodified for accurate boxes if possible, or rely on cfg.preproc behavior.

Troubleshooting
- If model state keys mismatch, check whether the checkpoint stores keys under 'model' or 'ema'.
- If cfg.evaluator is not configured, eval.py will attempt to create a CocoEvaluator from the val dataset automatically.

