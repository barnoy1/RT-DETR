from __future__ import annotations

from typing import Dict, Optional

from infra.common.loss_aliases import canonical_loss_alias


class RTDETRLossController:
    def __init__(self, app_config) -> None:
        self.app_config = app_config

    def collect_configured_loss_specs(self) -> list[tuple[str, Optional[float]]]:
        pairs = self.app_config.model.losses.criterion_pairs
        grouped = [pairs.iter_adapter_common(), pairs.iter_concrete_model()]
        specs: list[tuple[str, Optional[float]]] = []
        for entries in grouped:
            for item in entries:
                specs.append((canonical_loss_alias(item.loss), item.coef))
        return specs

    def configure_criterion_loss_ops(self, criterion) -> None:
        losses = getattr(criterion, "losses", None)
        if not isinstance(losses, list):
            return

        weight_dict = getattr(criterion, "weight_dict", None)
        if not isinstance(weight_dict, dict):
            weight_dict = {}

        configured_specs = self.collect_configured_loss_specs()
        active_patterns = [
            pattern
            for pattern, coef in configured_specs
            if coef is None or float(coef) > 0.0
        ]

        wants_boxes = any(any(token in pattern for token in ("loss_bbox", "loss_giou")) for pattern in active_patterns)
        wants_vfl = any("loss_vfl" in pattern for pattern in active_patterns)
        wants_focal = any("loss_focal" in pattern for pattern in active_patterns)

        if wants_boxes and "boxes" not in losses:
            losses.append("boxes")
        if wants_vfl and "vfl" not in losses:
            losses.append("vfl")
        if wants_focal and "focal" not in losses:
            losses.append("focal")

        if wants_focal and "loss_focal" not in weight_dict:
            weight_dict["loss_focal"] = 1.0
        if wants_vfl and "loss_vfl" not in weight_dict:
            weight_dict["loss_vfl"] = 1.0
        if wants_boxes:
            if "loss_bbox" not in weight_dict:
                weight_dict["loss_bbox"] = 1.0
            if "loss_giou" not in weight_dict:
                weight_dict["loss_giou"] = 1.0

        criterion.losses = losses
        criterion.weight_dict = weight_dict

    @staticmethod
    def resolve_configured_coef(
        weight_key: str,
        configured_specs: list[tuple[str, Optional[float]]],
    ) -> Optional[float]:
        lowered_key = str(weight_key).strip().lower()
        matched_coef: Optional[float] = None
        matched_len = -1

        for pattern, coef in configured_specs:
            if not pattern:
                continue
            if pattern.endswith("_"):
                is_match = lowered_key.startswith(pattern)
            else:
                is_match = lowered_key == pattern or lowered_key.startswith(f"{pattern}_")
            if not is_match:
                continue
            if len(pattern) > matched_len:
                matched_len = len(pattern)
                matched_coef = coef
        return matched_coef

    @staticmethod
    def base_loss_key(loss_key: str) -> str:
        lowered = str(loss_key).strip().lower()
        for marker in ("_aux_", "_dn_", "_enc_"):
            marker_index = lowered.find(marker)
            if marker_index > 0:
                return lowered[:marker_index]
        return lowered

    @classmethod
    def has_variant_specific_override(
        cls,
        base_key: str,
        configured_specs: list[tuple[str, Optional[float]]],
    ) -> bool:
        prefix = f"{str(base_key).strip().lower()}_"
        for pattern, coef in configured_specs:
            if coef is None or float(coef) <= 0.0:
                continue
            candidate = str(pattern).strip().lower()
            if candidate.startswith(prefix):
                return True
        return False

    def attach_output_loss_rescaler(
        self,
        criterion,
        configured_specs: list[tuple[str, Optional[float]]],
        applied_weight_dict: Dict[str, float],
    ) -> None:
        if getattr(criterion, "_nnf_output_rescaler_attached", False):
            return

        original_forward = criterion.forward

        def _forward_with_output_rescaling(outputs, targets, **kwargs):
            losses = original_forward(outputs, targets, **kwargs)
            if not isinstance(losses, dict) or not losses:
                return losses

            for key, value in list(losses.items()):
                loss_key = str(key).strip().lower()
                target_coef = self.resolve_configured_coef(loss_key, configured_specs)
                target_coef = 0.0 if target_coef is None else float(target_coef)

                base_key = self.base_loss_key(loss_key)
                base_coef = float(applied_weight_dict.get(base_key, 0.0))
                if base_coef <= 0.0:
                    scale = 0.0
                else:
                    scale = target_coef / base_coef

                losses[key] = value * scale

            return losses

        criterion.forward = _forward_with_output_rescaling
        criterion._nnf_output_rescaler_attached = True

    def apply_to_rtdetr(self, criterion) -> None:
        weight_dict = getattr(criterion, "weight_dict", None)
        if not isinstance(weight_dict, dict) or not weight_dict:
            return

        configured_specs = self.collect_configured_loss_specs()
        if not configured_specs:
            return

        updated_weight_dict: Dict[str, float] = {}
        for key, existing in weight_dict.items():
            resolved_coef = self.resolve_configured_coef(key, configured_specs)
            if resolved_coef is None:
                if self.has_variant_specific_override(key, configured_specs):
                    updated_weight_dict[key] = float(existing) if float(existing) > 0.0 else 1.0
                else:
                    updated_weight_dict[key] = 0.0
            else:
                updated_weight_dict[key] = float(resolved_coef)
            if resolved_coef is None and float(existing) == 0.0:
                updated_weight_dict[key] = float(existing)

        criterion.weight_dict = updated_weight_dict

        configured_losses = getattr(criterion, "losses", None)
        if isinstance(configured_losses, list) and configured_losses:
            enabled_ops: set[str] = set()
            for pattern, coef in configured_specs:
                if coef is not None and float(coef) <= 0.0:
                    continue
                if "vfl" in pattern:
                    enabled_ops.add("vfl")
                if "focal" in pattern:
                    enabled_ops.add("focal")
                if "bbox" in pattern or "giou" in pattern or "boxes" in pattern:
                    enabled_ops.add("boxes")

            filtered_losses = [loss_name for loss_name in configured_losses if str(loss_name).lower() in enabled_ops]
            if filtered_losses:
                criterion.losses = filtered_losses

        self.attach_output_loss_rescaler(
            criterion=criterion,
            configured_specs=configured_specs,
            applied_weight_dict=updated_weight_dict,
        )
