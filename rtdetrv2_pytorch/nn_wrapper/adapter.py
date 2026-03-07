from __future__ import annotations

import sys
import warnings
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[5]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
WRAPPER_ROOT = Path(__file__).resolve().parent
if str(WRAPPER_ROOT) not in sys.path:
    sys.path.insert(0, str(WRAPPER_ROOT))

from infra.config import AppConfig
from infra.engine.model.adapter import FrameworkModelAdapter
from infra.engine.model.base import ModelWrapperAdapter, WrapperComponents

from builder import RTDETRv2ModelBuilder, configure_fixed_dn_num_group
from checkpoint import RTDETRCheckpointAdapter


def create_wrapper_components(app_config: AppConfig, repo_root: Path) -> WrapperComponents:
    return WrapperComponents(
        model_builder=RTDETRv2ModelBuilder(app_config=app_config, repo_root=repo_root),
        checkpoint_adapter=RTDETRCheckpointAdapter(repo_root=repo_root),
        dn_group_configurer=configure_fixed_dn_num_group,
    )


def create_model_wrapper(app_config: AppConfig, repo_root: Path) -> ModelWrapperAdapter:
    warnings.warn(
        "create_model_wrapper(...) is deprecated for concrete wrappers; "
        "framework now prefers create_wrapper_components(...)",
        DeprecationWarning,
        stacklevel=2,
    )
    components = create_wrapper_components(app_config=app_config, repo_root=repo_root)
    return FrameworkModelAdapter(
        model_builder=components.model_builder,
        checkpoint_adapter=components.checkpoint_adapter,
        dn_group_configurer=components.dn_group_configurer,
    )
