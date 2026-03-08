from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_PARENT = Path(__file__).resolve().parents[5]
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))
WRAPPER_ROOT = Path(__file__).resolve().parent
if str(WRAPPER_ROOT) not in sys.path:
    sys.path.insert(0, str(WRAPPER_ROOT))

from infra.config import AppConfig
from infra.engine.model.wrappers import WrapperComponents
from infra.engine.model.wrappers.common import AgnosticModelBuilderBase


class RTDETRv2ModelBuilder(AgnosticModelBuilderBase):
    pass


def create_wrapper_components(app_config: AppConfig, repo_root: Path) -> WrapperComponents:
    return WrapperComponents(
        model_builder=RTDETRv2ModelBuilder(app_config=app_config, repo_root=repo_root),
    )
