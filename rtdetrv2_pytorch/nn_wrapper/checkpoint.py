from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

PACKAGE_PARENT = Path(__file__).resolve().parents[5]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from nn_framework.model.base import CheckpointAdapter


class RTDETRCheckpointAdapter(CheckpointAdapter):
    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root

    def resolve_checkpoint_path(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        if candidate.exists():
            return candidate.resolve()

        candidates = []
        if not candidate.is_absolute():
            candidates.append((self.repo_root / candidate).resolve())

        candidates.append((self.repo_root / "weights" / candidate.name).resolve())
        candidates.append((self.repo_root.parent / "weights" / candidate.name).resolve())
        candidates.append((PACKAGE_PARENT / "nn_framework" / "weights" / candidate.name).resolve())

        for fallback in candidates:
            if fallback.exists():
                print(f"Warning: checkpoint not found at {candidate}, using {fallback}")
                return fallback

        checked = "\n  - ".join(str(p) for p in [candidate, *candidates])
        raise FileNotFoundError(f"Checkpoint file not found. Checked:\n  - {checked}")

    def load_checkpoint_state(self, path: str) -> Dict[str, torch.Tensor]:
        checkpoint_path = self.resolve_checkpoint_path(path)
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(checkpoint, dict):
            if "ema" in checkpoint:
                return checkpoint["ema"].get("module", checkpoint["ema"])
            if "model" in checkpoint:
                return checkpoint["model"]
            return checkpoint
        return checkpoint

    @staticmethod
    def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if not any(key.startswith("module.") for key in state_dict.keys()):
            return state_dict
        return {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}

    def get_checkpoint_num_classes(self, state_dict: Dict[str, torch.Tensor]) -> Optional[int]:
        state_dict = self._normalize_state_dict_keys(state_dict)
        weight = state_dict.get("decoder.enc_score_head.weight")
        if weight is None or weight.ndim == 0:
            return None
        return int(weight.shape[0])

    @staticmethod
    def get_model_num_classes(model: nn.Module) -> Optional[int]:
        weight = model.state_dict().get("decoder.enc_score_head.weight")
        if weight is None or weight.ndim == 0:
            return None
        return int(weight.shape[0])

    def validate_checkpoint_class_compatibility(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        allow_mismatch: bool = False,
    ) -> None:
        checkpoint_classes = self.get_checkpoint_num_classes(state_dict)
        model_classes = self.get_model_num_classes(model)

        if checkpoint_classes is None or model_classes is None:
            return

        if checkpoint_classes == model_classes:
            return

        message = (
            "Checkpoint/model class mismatch detected: "
            f"checkpoint classes={checkpoint_classes}, model classes={model_classes}. "
            "This leaves detection score heads uninitialized and usually produces very low-confidence detections. "
            "Use a checkpoint trained with the same num_classes as inference config, "
            "or set --allow-class-mismatch to bypass this safety check."
        )
        if allow_mismatch:
            print(f"Warning: {message}")
            return
        raise RuntimeError(message)

    def safe_load_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
        state_dict = self._normalize_state_dict_keys(state_dict)
        model_state = model.state_dict()

        compatible: Dict[str, torch.Tensor] = {}
        skipped_shape = 0
        for key, value in state_dict.items():
            if key not in model_state:
                continue
            if model_state[key].shape != value.shape:
                skipped_shape += 1
                continue
            compatible[key] = value

        missing_keys, _ = model.load_state_dict(compatible, strict=False)
        return len(compatible), skipped_shape, len(missing_keys)


def create_checkpoint_adapter(repo_root: Path) -> CheckpointAdapter:
    return RTDETRCheckpointAdapter(repo_root=repo_root)
