from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn

PACKAGE_PARENT = Path(__file__).resolve().parents[5]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
WRAPPER_ROOT = Path(__file__).resolve().parent
if str(WRAPPER_ROOT) not in sys.path:
    sys.path.insert(0, str(WRAPPER_ROOT))

from nn_framework.config import AppConfig
from nn_framework.model.base import BuiltComponents, ModelWrapperAdapter

from builder import RTDETRv2ModelBuilder, configure_fixed_dn_num_group
from checkpoint import RTDETRCheckpointAdapter


class RTDETRv2WrapperAdapter(ModelWrapperAdapter):
    def __init__(self, app_config: AppConfig, repo_root: Path) -> None:
        self.app_config = app_config
        self.repo_root = repo_root
        self._builder = RTDETRv2ModelBuilder(app_config=app_config, repo_root=repo_root)
        self._checkpoint = RTDETRCheckpointAdapter(repo_root=repo_root)

    def build_components(self) -> BuiltComponents:
        return self._builder.build()

    def configure_fixed_dn_num_group(self, model: nn.Module, targets: List[Dict], dn_num_group: int) -> None:
        configure_fixed_dn_num_group(model=model, targets=targets, dn_num_group=dn_num_group)

    def load_checkpoint_state(self, path: str) -> Dict[str, torch.Tensor]:
        return self._checkpoint.load_checkpoint_state(path)

    def validate_checkpoint_class_compatibility(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
        allow_mismatch: bool = False,
    ) -> None:
        self._checkpoint.validate_checkpoint_class_compatibility(
            model=model,
            state_dict=state_dict,
            allow_mismatch=allow_mismatch,
        )

    def safe_load_state_dict(self, model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
        return self._checkpoint.safe_load_state_dict(model=model, state_dict=state_dict)


def create_model_wrapper(app_config: AppConfig, repo_root: Path) -> ModelWrapperAdapter:
    return RTDETRv2WrapperAdapter(app_config=app_config, repo_root=repo_root)
