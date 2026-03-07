from __future__ import annotations

from typing import Any


class RTDETRConfigOverrides:
    def __init__(self, app_config) -> None:
        self.app_config = app_config

    def runtime_overrides(self) -> dict:
        export_payload = self.app_config.runtime.export
        if hasattr(export_payload, "model_dump"):
            export_payload = export_payload.model_dump()
        elif not isinstance(export_payload, dict):
            export_payload = {
                "post_process": True,
                "nms": True,
                "benchmark": False,
                "fuse_conv_bn": False,
            }

        return {
            "use_gpu": self.app_config.runtime.use_gpu,
            "use_xpu": self.app_config.runtime.use_xpu,
            "use_mlu": self.app_config.runtime.use_mlu,
            "use_npu": self.app_config.runtime.use_npu,
            "log_iter": self.app_config.runtime.log_iter,
            "output_dir": self.app_config.runtime.output_dir,
            "epoches": self.app_config.runtime.epoches if self.app_config.runtime.epoches is not None else self.app_config.train.epochs,
            "export": export_payload,
        }

    def model_overrides(self) -> dict:
        return {
            "num_classes": self.app_config.model.num_classes,
            "RTDETRTransformerv2": {
                "num_queries": self.app_config.model.num_queries,
                "hidden_dim": self.app_config.model.hidden_dim,
            },
            "HybridEncoder": {
                "hidden_dim": self.app_config.model.hidden_dim,
            },
        }

    def dataset_overrides(self) -> dict:
        configured_remap_mscoco = bool(self.app_config.data.remap_mscoco_category)
        has_custom_label_mapping = bool(self.app_config.data.mapping)
        effective_remap_mscoco = configured_remap_mscoco and not has_custom_label_mapping

        train_sets = [
            {"img_folder": dataset_pair.img_dir, "ann_file": dataset_pair.ann_file}
            for dataset_pair in self.app_config.data.train_sets
        ]
        val_sets = [
            {"img_folder": dataset_pair.img_dir, "ann_file": dataset_pair.ann_file}
            for dataset_pair in self.app_config.data.val_sets
        ]

        default_train_loader = {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "datasets": train_sets,
                "return_masks": "segm" in self.app_config.data.iou_types,
                "transforms": {"type": "Compose", "ops": None},
            },
            "shuffle": True,
            "num_workers": self.app_config.train.num_workers,
            "drop_last": True,
            "collate_fn": {"type": "BatchImageCollateFunction"},
        }
        default_val_loader = {
            "type": "DataLoader",
            "dataset": {
                "type": "CocoDetection",
                "datasets": val_sets,
                "return_masks": "segm" in self.app_config.data.iou_types,
                "transforms": {"type": "Compose", "ops": None},
            },
            "shuffle": False,
            "num_workers": self.app_config.train.num_workers,
            "drop_last": False,
            "collate_fn": {"type": "BatchImageCollateFunction"},
        }

        train_loader = self.app_config.data.train_dataloader or default_train_loader
        val_loader = self.app_config.data.val_dataloader or default_val_loader
        label2classid = self.app_config.data.label2classid or self.app_config.data.class_id_to_name
        class_id_to_name = self.app_config.data.class_id_to_name or self.app_config.data.label2classid

        return {
            "task": self.app_config.data.task,
            "evaluator": self.app_config.data.evaluator,
            "num_classes": self.app_config.data.num_classes,
            "remap_mscoco_category": effective_remap_mscoco,
            "label2classid": label2classid,
            "class_id_to_name": class_id_to_name,
            "train_dataloader": train_loader,
            "val_dataloader": val_loader,
        }

    def yaml_overrides(self) -> dict:
        payload = {}
        payload.update(self.dataset_overrides())
        payload.update(self.runtime_overrides())
        payload.update(self.model_overrides())
        return payload

    @staticmethod
    def extract_class_id_to_name(yaml_cfg: Any) -> dict[int, str]:
        payload = getattr(yaml_cfg, "yaml_cfg", None)
        if not isinstance(payload, dict):
            return {}

        mapping = payload.get("class_id_to_name") or payload.get("label2classid") or {}
        if not isinstance(mapping, dict):
            return {}

        class_id_to_name: dict[int, str] = {}

        for key, value in mapping.items():
            if isinstance(value, str):
                try:
                    class_id = int(key)
                except (TypeError, ValueError):
                    continue
                class_id_to_name[class_id] = value

        if class_id_to_name:
            return dict(sorted(class_id_to_name.items(), key=lambda item: item[0]))

        for label, class_id in mapping.items():
            if not isinstance(label, str):
                continue
            try:
                class_id_to_name[int(class_id)] = label
            except (TypeError, ValueError):
                continue

        return dict(sorted(class_id_to_name.items(), key=lambda item: item[0]))
