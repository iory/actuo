"""actuo: one Python layer for Dynamixel, Kondo, Feetech and CAN actuators.

The public API is what this package exports (``__all__``) plus the modules
:mod:`actuo.core` (contracts and the plugin registry), :mod:`actuo.safety` and
:mod:`actuo.testing`. Everything under ``actuo.drivers`` is the implementation
and may move between releases; reach it through a robot config and these names.

The names here are imported on first use, so ``import actuo`` loads neither
scikit-robot nor any vendor library.
"""

from __future__ import annotations

import importlib
from typing import Dict, Tuple

__version__ = "0.0.1"

#: Exported name -> (module, attribute).
_EXPORTS: Dict[str, Tuple[str, str]] = {
    "ActuoRobotInterface": ("actuo.robot_interface", "ActuoRobotInterface"),
    "Catalog": ("actuo.modular", "Catalog"),
    "ModularRobot": ("actuo.modular", "ModularRobot"),
    "build_bus": ("actuo.drivers.config", "build_bus"),
    "load_document": ("actuo.drivers.config", "load_document"),
}

__all__ = sorted(_EXPORTS) + ["__version__"]


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError("module 'actuo' has no attribute {!r}".format(name))
    module, attribute = _EXPORTS[name]
    value = getattr(importlib.import_module(module), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))
