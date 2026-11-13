"""Bus orchestration implementations (batch write + scheduled reads).

The classes are imported on first access, so ``from actuo.drivers.buses import
DynamixelBus`` does not import rcb4 or python-can.
"""

import importlib

_MODULES = {
    "CanBus": "can_bus",
    "StaleFeedbackError": "can_bus",
    "DynamixelBus": "dynamixel_bus",
    "KondoH7Bus": "kondo_h7_bus",
    "KondoRcb4Bus": "kondo_rcb4_bus",
}

__all__ = sorted(_MODULES)


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError("module {!r} has no attribute {!r}".format(__name__, name))
    module = importlib.import_module("{}.{}".format(__name__, _MODULES[name]))
    return getattr(module, name)
