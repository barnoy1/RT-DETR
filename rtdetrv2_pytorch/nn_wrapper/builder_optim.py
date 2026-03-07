from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn


class RTDETROptimizerFactory:
    def __init__(self, app_config) -> None:
        self.app_config = app_config

    @staticmethod
    def split_backbone_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
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

    def build(self, model: nn.Module):
        backbone_params, non_backbone_params = self.split_backbone_params(model)
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
