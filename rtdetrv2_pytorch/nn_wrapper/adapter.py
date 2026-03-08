from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

PACKAGE_PARENT = Path(__file__).resolve().parents[5]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
WRAPPER_ROOT = Path(__file__).resolve().parent
if str(WRAPPER_ROOT) not in sys.path:
    sys.path.insert(0, str(WRAPPER_ROOT))

from infra.config import AppConfig
from infra.engine.model.ema import EMAModel
from infra.engine.model.losses import (
    CompositeCriterion,
    ConcreteCriterionAdapter,
    DualCriterionSpecResolver,
    prepare_base_criterion_for_agnostic_flow,
)
from infra.engine.model.wrappers import BuiltComponents, ModelBuilder, WrapperComponents
from infra.engine.model.wrappers.common import BackboneGroupedAdamWFactory


class RTDETRv2ModelBuilder(ModelBuilder):
    def __init__(self, app_config: AppConfig, repo_root: Path) -> None:
        self.app_config = app_config
        self.repo_root = repo_root
        self._optimizer_factory = BackboneGroupedAdamWFactory(
            lr=app_config.train.lr,
            weight_decay=app_config.train.weight_decay,
            epochs=app_config.train.epochs,
            backbone_lr_multiplier=app_config.train.backbone_lr_multiplier,
        )
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

    def _build_composite_criterion(self, base_criterion):
        resolver = DualCriterionSpecResolver.from_app_config(self.app_config)
        prepare_base_criterion_for_agnostic_flow(base_criterion, resolver)
        dfl_provider = RTDETRDFLProvider()
        return CompositeCriterion(
            base_criterion=base_criterion,
            adapters=[ConcreteCriterionAdapter()],
            resolver=resolver,
            dfl_provider=dfl_provider,
        )

    def _load_model_config(self):
        from src.core import YAMLConfig

        config_rel_path = self.app_config.model.model_config_path or self.app_config.model.model_config_path
        config_path = self.repo_root / str(config_rel_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Official config not found: {config_path}")
        return YAMLConfig(str(config_path))

    def build(self) -> BuiltComponents:
        yaml_cfg = self._load_model_config()
        model = yaml_cfg.model
        base_criterion = yaml_cfg.criterion
        criterion = self._build_composite_criterion(base_criterion)
        postprocessor = yaml_cfg.postprocessor
        class_id_to_name = self.app_config.data.class_id_to_name or self.app_config.data.label2classid

        if self.app_config.model.sync_bn and torch.cuda.device_count() > 1:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        optimizer, scheduler = self._optimizer_factory.build(model)

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
            class_id_to_name=class_id_to_name,
        )

    def apply_architecture_specifics(self, model: nn.Module, targets: List[dict], *, dn_num_group: int) -> None:
        max_gt = max((len(target["labels"]) for target in targets), default=0)
        if max_gt <= 0:
            return
        num_denoising = max_gt * dn_num_group
        decoder = getattr(model, "decoder", None)
        if decoder is not None and hasattr(decoder, "num_denoising"):
            decoder.num_denoising = int(num_denoising)


def create_model_builder(app_config: AppConfig, repo_root: Path) -> ModelBuilder:
    return RTDETRv2ModelBuilder(app_config=app_config, repo_root=repo_root)


class RTDETRDFLProvider:
    _precomputed_loss_keys = ("loss_dfl", "dfl_loss", "loss_distribution_focal")
    _logits_keys = ("dfl_logits", "pred_distri", "pred_dist", "pred_distributions", "pred_ltrb_dist")
    _target_keys = ("dfl_targets", "target_dfl", "dfl_target")

    def supports(self, *, outputs, targets=None) -> bool:
        if not isinstance(outputs, dict):
            return False
        if any(key in outputs for key in self._precomputed_loss_keys):
            return True
        has_logits = any(key in outputs for key in self._logits_keys)
        has_targets = any(key in outputs for key in self._target_keys)
        return has_logits and has_targets

    def __call__(self, *, outputs, targets, indices, num_boxes: float):
        if not isinstance(outputs, dict):
            return {}

        for key in self._precomputed_loss_keys:
            value = outputs.get(key)
            if torch.is_tensor(value):
                return {"loss_dfl": value}

        logits = None
        for key in self._logits_keys:
            candidate = outputs.get(key)
            if torch.is_tensor(candidate):
                logits = candidate
                break

        target_bins = None
        for key in self._target_keys:
            candidate = outputs.get(key)
            if torch.is_tensor(candidate):
                target_bins = candidate
                break

        if logits is None or target_bins is None:
            return {}
        if logits.ndim < 2 or target_bins.ndim < 1:
            return {}

        num_bins = int(logits.shape[-1])
        flat_logits = logits.reshape(-1, num_bins)
        flat_targets = target_bins.reshape(-1).long()
        if flat_logits.shape[0] != flat_targets.shape[0]:
            return {}

        valid = (flat_targets >= 0) & (flat_targets < num_bins)
        if int(valid.sum().item()) == 0:
            return {}

        loss = F.cross_entropy(flat_logits[valid], flat_targets[valid], reduction="mean")
        return {"loss_dfl": loss}


def create_wrapper_components(app_config: AppConfig, repo_root: Path) -> WrapperComponents:
    return WrapperComponents(
        model_builder=RTDETRv2ModelBuilder(app_config=app_config, repo_root=repo_root),
    )
