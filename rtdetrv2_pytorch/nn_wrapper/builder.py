from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
from torch import nn

PACKAGE_PARENT = Path(__file__).resolve().parents[5]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from nn_framework.config import AppConfig
from nn_framework.model.base import BuiltComponents, ModelBuilder
from nn_framework.model.ema import EMAModel


class RTDETRv2ModelBuilder(ModelBuilder):
    def __init__(self, app_config: AppConfig, repo_root: Path) -> None:
        self.app_config = app_config
        self.repo_root = repo_root
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def _load_official_config(self):
        from src.core import YAMLConfig

        config_path = self.repo_root / self.app_config.model.official_config_path
        if not config_path.exists():
            raise FileNotFoundError(f"Official config not found: {config_path}")
        return YAMLConfig(str(config_path))

    def _split_backbone_params(self, model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
        backbone_params: List[nn.Parameter] = []
        non_backbone_params: List[nn.Parameter] = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if "backbone" in name:
                backbone_params.append(parameter)
            else:
                non_backbone_params.append(parameter)
        return backbone_params, non_backbone_params

    def _build_optimizer_and_scheduler(self, model: nn.Module):
        backbone_params, non_backbone_params = self._split_backbone_params(model)
        lr = self.app_config.train.lr
        backbone_lr = lr * self.app_config.train.backbone_lr_multiplier

        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": backbone_lr},
                {"params": non_backbone_params, "lr": lr},
            ],
            lr=lr,
            weight_decay=self.app_config.train.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.app_config.train.epochs,
            eta_min=lr * 0.01,
        )
        return optimizer, scheduler

    def build(self) -> BuiltComponents:
        yaml_cfg = self._load_official_config()
        model = yaml_cfg.model
        criterion = yaml_cfg.criterion
        postprocessor = yaml_cfg.postprocessor

        if self.app_config.model.sync_bn and torch.cuda.device_count() > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        optimizer, scheduler = self._build_optimizer_and_scheduler(model)

        ema_model = None
        if self.app_config.train.use_ema:
            ema_model = EMAModel(model, decay=self.app_config.train.ema_decay)

        return BuiltComponents(
            model=model,
            criterion=criterion,
            postprocessor=postprocessor,
            optimizer=optimizer,
            scheduler=scheduler,
            ema_model=ema_model,
        )


def create_model_builder(app_config: AppConfig, repo_root: Path) -> ModelBuilder:
    return RTDETRv2ModelBuilder(app_config=app_config, repo_root=repo_root)


def configure_fixed_dn_num_group(model: nn.Module, targets: List[dict], dn_num_group: int) -> None:
    max_gt = max((len(t["labels"]) for t in targets), default=0)
    if max_gt <= 0:
        return
    num_denoising = max_gt * dn_num_group
    decoder = getattr(model, "decoder", None)
    if decoder is not None and hasattr(decoder, "num_denoising"):
        decoder.num_denoising = int(num_denoising)
