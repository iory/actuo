"""Plugin registry for transports and actuator models.

New actuators or buses are added as plugins registered here -- never by editing
a per-protocol ``switch`` in shared code, so adding a protocol never edits the
core. Driver packages register
their classes at import time; the core stays closed for modification.

The plugins shipped with actuo are listed by name and module below and
imported on first use, so naming a Kondo servo never imports python-can and a
CAN robot does not need rcb4. ``registered_*()`` return the plugins imported
so far.
"""

from __future__ import annotations

import importlib
from typing import Callable, Dict, Tuple, Type

from actuo.core.contracts import Actuator, Sensor, Transport

_TRANSPORTS: Dict[str, Type[Transport]] = {}
_ACTUATORS: Dict[str, Type[Actuator]] = {}
_SENSORS: Dict[str, Type[Sensor]] = {}

#: The plugins this package ships: name -> (module that registers it, the
#: extra its vendor library comes from, or None). A module is imported the
#: first time its name is asked for, so a robot of one vendor never imports
#: another vendor's library.
_BUILTIN_TRANSPORTS: Dict[str, Tuple[str, "str | None"]] = {
    "kondo_h7": ("actuo.drivers.transports.kondo_h7", "kondo"),
    "kondo_rcb4": ("actuo.drivers.transports.kondo_rcb4", "kondo"),
    "kondo_rcb4_udp": ("actuo.drivers.transports.kondo_rcb4_udp", "kondo"),
    "python_can": ("actuo.drivers.transports.python_can", "can"),
    "serial": ("actuo.drivers.transports.serial_transport", None),
}
_BUILTIN_ACTUATORS: Dict[str, Tuple[str, "str | None"]] = {
    "cubemars": ("actuo.drivers.actuators.cubemars", None),
    "cubemars_mit": ("actuo.drivers.actuators.cubemars_mit", None),
    "cubemars_v3": ("actuo.drivers.actuators.cubemars_v3", None),
    "dynamixel": ("actuo.drivers.actuators.dynamixel", None),
    "feetech_scs0009": ("actuo.drivers.actuators.feetech_scs", None),
    "kondo_ics": ("actuo.drivers.actuators.kondo_ics", "kondo"),
    "kondo_ics_wheel": ("actuo.drivers.actuators.kondo_ics_wheel", "kondo"),
}
_BUILTIN_SENSORS: Dict[str, Tuple[str, "str | None"]] = {
    "kjs03": ("actuo.drivers.sensors.kjs03", None),
}


def _load_builtin(kind: str, name: str,
                  builtins: Dict[str, Tuple[str, "str | None"]]) -> None:
    """Import the module that registers built-in plugin ``name``.

    Raises
    ------
    ImportError
        The plugin's vendor library is not installed; the message names the
        extra that installs it.
    """
    module, extra = builtins[name]
    try:
        importlib.import_module(module)
    except ModuleNotFoundError as error:
        if extra is None or (error.name or "").split(".")[0] == "actuo":
            raise
        raise ImportError(
            "{} {!r} needs {} (not installed); install it with "
            "pip install 'actuo[{}]'".format(kind, name, error.name, extra),
            name=error.name) from error


def register_transport(name: str) -> Callable[[Type[Transport]], Type[Transport]]:
    """Class decorator registering a Transport implementation under ``name``."""

    def _decorator(cls: Type[Transport]) -> Type[Transport]:
        if name in _TRANSPORTS:
            raise ValueError("transport {!r} already registered".format(name))
        _TRANSPORTS[name] = cls
        return cls

    return _decorator


def register_actuator(name: str) -> Callable[[Type[Actuator]], Type[Actuator]]:
    """Class decorator registering an Actuator model under ``name``."""

    def _decorator(cls: Type[Actuator]) -> Type[Actuator]:
        if name in _ACTUATORS:
            raise ValueError("actuator {!r} already registered".format(name))
        _ACTUATORS[name] = cls
        return cls

    return _decorator


def register_sensor(name: str) -> Callable[[Type[Sensor]], Type[Sensor]]:
    """Class decorator registering a Sensor model under ``name``.

    This is how custom telemetry hardware (pressure, IMU, user boards) plugs in
    without touching actuo.core.
    """

    def _decorator(cls: Type[Sensor]) -> Type[Sensor]:
        if name in _SENSORS:
            raise ValueError("sensor {!r} already registered".format(name))
        _SENSORS[name] = cls
        return cls

    return _decorator


def get_transport(name: str) -> Type[Transport]:
    if name not in _TRANSPORTS and name in _BUILTIN_TRANSPORTS:
        _load_builtin("transport", name, _BUILTIN_TRANSPORTS)
    if name not in _TRANSPORTS:
        raise KeyError(
            "unknown transport {!r}; known: {}".format(
                name, sorted(set(_TRANSPORTS) | set(_BUILTIN_TRANSPORTS))
            )
        )
    return _TRANSPORTS[name]


def get_actuator(name: str) -> Type[Actuator]:
    if name not in _ACTUATORS and name in _BUILTIN_ACTUATORS:
        _load_builtin("actuator", name, _BUILTIN_ACTUATORS)
    if name not in _ACTUATORS:
        raise KeyError(
            "unknown actuator {!r}; known: {}".format(
                name, sorted(set(_ACTUATORS) | set(_BUILTIN_ACTUATORS))
            )
        )
    return _ACTUATORS[name]


def get_sensor(name: str) -> Type[Sensor]:
    if name not in _SENSORS and name in _BUILTIN_SENSORS:
        _load_builtin("sensor", name, _BUILTIN_SENSORS)
    if name not in _SENSORS:
        raise KeyError(
            "unknown sensor {!r}; known: {}".format(
                name, sorted(set(_SENSORS) | set(_BUILTIN_SENSORS)))
        )
    return _SENSORS[name]


def registered_transports() -> Dict[str, Type[Transport]]:
    return dict(_TRANSPORTS)


def registered_actuators() -> Dict[str, Type[Actuator]]:
    return dict(_ACTUATORS)


def registered_sensors() -> Dict[str, Type[Sensor]]:
    return dict(_SENSORS)
