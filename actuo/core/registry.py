"""Plugin registry for transports, actuator models, sensors and buses.

New hardware is added as a plugin registered here -- never by editing a
per-protocol ``switch`` in shared code, so adding a protocol never edits the
core. A plugin is a class registered under the name a robot config uses for it
(``plugin: <name>``), in one of three ways:

* shipped with actuo: listed by name and module below and imported on first
  use, so naming a Kondo servo never imports python-can and a CAN robot does
  not need rcb4;
* a class decorator (``@register_actuator("name")``), for code that is
  imported anyway;
* an entry point in another installed package, under the group of its kind
  (:data:`ENTRY_POINT_GROUPS`), pointing at the class::

      [project.entry-points."actuo.actuators"]
      my_servo = "my_package.servo:MyServo"

  It is loaded the first time a config names it. A shipped plugin's name
  cannot be taken by an entry point: the shipped one is used.

How a robot config becomes these objects is
:func:`actuo.drivers.config.build_bus`. A plugin from another package brings
its own config reader as a classmethod:

* actuator / sensor: ``from_config(spec, where) -> instance``, where ``spec``
  is the joint's ``actuator:`` (or ``sensor:``) block without ``plugin`` and
  ``where`` names it for error messages;
* bus: ``from_config(transport, actuators, joint_names, spec, path) -> Bus``,
  where ``spec`` is the ``bus:`` block and ``path`` the config file.

A transport is built from its block's keys as keyword arguments. Which bus
drives a transport, when the config does not say (``bus.plugin``), is the
transport class's ``default_bus``, unless every actuator on it names the same
``preferred_bus``.

``registered_*()`` return the plugins loaded so far.
"""

from __future__ import annotations

import importlib
import inspect
from importlib import metadata
from typing import Callable, Dict, List, Tuple, Type, TypeVar

from actuo.core.contracts import Actuator, Bus, Sensor, Transport

#: The entry-point group another package registers each kind of plugin under.
ENTRY_POINT_GROUPS: Dict[str, str] = {
    "transport": "actuo.transports",
    "actuator": "actuo.actuators",
    "sensor": "actuo.sensors",
    "bus": "actuo.buses",
}

_TRANSPORTS: Dict[str, Type[Transport]] = {}
_ACTUATORS: Dict[str, Type[Actuator]] = {}
_SENSORS: Dict[str, Type[Sensor]] = {}
_BUSES: Dict[str, Type[Bus]] = {}

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
    "feetech_sts": ("actuo.drivers.actuators.feetech_sts", "feetech"),
    "kondo_ics": ("actuo.drivers.actuators.kondo_ics", "kondo"),
    "kondo_ics_wheel": ("actuo.drivers.actuators.kondo_ics_wheel", "kondo"),
}
_BUILTIN_SENSORS: Dict[str, Tuple[str, "str | None"]] = {
    "air_relay_board": ("actuo.drivers.sensors.air_relay_board", None),
}
_BUILTIN_BUSES: Dict[str, Tuple[str, "str | None"]] = {
    "can_bus": ("actuo.drivers.buses.can_bus", None),
    "dynamixel_bus": ("actuo.drivers.buses.dynamixel_bus", None),
    "feetech_bus": ("actuo.drivers.buses.feetech_bus", "feetech"),
    "kondo_h7_bus": ("actuo.drivers.buses.kondo_h7_bus", "kondo"),
    "kondo_rcb4_bus": ("actuo.drivers.buses.kondo_rcb4_bus", "kondo"),
    "serial_bus": ("actuo.drivers.buses.serial_bus", None),
}

#: kind -> (loaded plugins, shipped plugins, base class).
_KINDS = {
    "transport": (_TRANSPORTS, _BUILTIN_TRANSPORTS, Transport),
    "actuator": (_ACTUATORS, _BUILTIN_ACTUATORS, Actuator),
    "sensor": (_SENSORS, _BUILTIN_SENSORS, Sensor),
    "bus": (_BUSES, _BUILTIN_BUSES, Bus),
}

T = TypeVar("T")


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


def _entry_points(kind: str) -> List[metadata.EntryPoint]:
    return list(metadata.entry_points(group=ENTRY_POINT_GROUPS[kind]))


def _load_entry_point(kind: str, name: str) -> None:
    """Load the entry point registering plugin ``name``, if a package has one.

    The entry point names the class; a module whose import registers the class
    with the decorator is accepted too.

    Raises
    ------
    TypeError
        The entry point is not a subclass of the kind's contract.
    """
    loaded, _, base = _KINDS[kind]
    for entry_point in _entry_points(kind):
        if entry_point.name != name:
            continue
        target = entry_point.load()
        if name in loaded:              # its module registered it on import
            return
        if not (inspect.isclass(target) and issubclass(target, base)):
            raise TypeError(
                "entry point {!r} = {!r} in group {!r} is not a {} class".format(
                    name, entry_point.value, entry_point.group, base.__name__))
        loaded[name] = target
        return


def known(kind: str) -> List[str]:
    """Every plugin name of ``kind`` a config can use: loaded, shipped or
    offered by an installed package's entry point."""
    loaded, builtins, _ = _KINDS[kind]
    names = set(loaded) | set(builtins)
    names.update(entry_point.name for entry_point in _entry_points(kind))
    return sorted(names)


def _get(kind: str, name: str):
    loaded, builtins, _ = _KINDS[kind]
    if name not in loaded and name in builtins:
        _load_builtin(kind, name, builtins)
    if name not in loaded:
        _load_entry_point(kind, name)
    if name not in loaded:
        raise KeyError("unknown {} {!r}; known: {}".format(kind, name, known(kind)))
    return loaded[name]


def _register(kind: str, name: str) -> Callable[[Type[T]], Type[T]]:
    loaded = _KINDS[kind][0]

    def _decorator(cls: Type[T]) -> Type[T]:
        if name in loaded:
            raise ValueError("{} {!r} already registered".format(kind, name))
        loaded[name] = cls
        return cls

    return _decorator


def register_transport(name: str) -> Callable[[Type[Transport]], Type[Transport]]:
    """Class decorator registering a Transport implementation under ``name``."""
    return _register("transport", name)


def register_actuator(name: str) -> Callable[[Type[Actuator]], Type[Actuator]]:
    """Class decorator registering an Actuator model under ``name``."""
    return _register("actuator", name)


def register_sensor(name: str) -> Callable[[Type[Sensor]], Type[Sensor]]:
    """Class decorator registering a Sensor model under ``name``.

    This is how custom telemetry hardware (pressure, IMU, user boards) plugs in
    without touching actuo.core.
    """
    return _register("sensor", name)


def register_bus(name: str) -> Callable[[Type[Bus]], Type[Bus]]:
    """Class decorator registering a Bus implementation under ``name``."""
    return _register("bus", name)


def get_transport(name: str) -> Type[Transport]:
    return _get("transport", name)


def get_actuator(name: str) -> Type[Actuator]:
    return _get("actuator", name)


def get_sensor(name: str) -> Type[Sensor]:
    return _get("sensor", name)


def get_bus(name: str) -> Type[Bus]:
    return _get("bus", name)


def registered_transports() -> Dict[str, Type[Transport]]:
    return dict(_TRANSPORTS)


def registered_actuators() -> Dict[str, Type[Actuator]]:
    return dict(_ACTUATORS)


def registered_sensors() -> Dict[str, Type[Sensor]]:
    return dict(_SENSORS)


def registered_buses() -> Dict[str, Type[Bus]]:
    return dict(_BUSES)
