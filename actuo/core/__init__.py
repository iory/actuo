"""actuo.core -- ROS-free actuator-abstraction contracts and plugin registry.

The design principles this package enforces are listed at the top of
:mod:`actuo.core.contracts`.
"""

from actuo.core.contracts import (
    Actuator,
    ActuatorCommand,
    ActuatorId,
    ActuatorState,
    Bus,
    Calibration,
    Capabilities,
    CapabilityError,
    ChannelKind,
    ChannelSpec,
    Device,
    DeviceId,
    Frame,
    OperatingMode,
    Sensor,
    Telemetry,
    Transport,
)
from actuo.core.diagnostics import DiagItem, DiagLevel, fold_diag_level
from actuo.core.registry import (
    get_actuator,
    get_sensor,
    get_transport,
    register_actuator,
    register_sensor,
    register_transport,
    registered_actuators,
    registered_sensors,
    registered_transports,
)

__all__ = [
    "Actuator",
    "ActuatorCommand",
    "ActuatorId",
    "ActuatorState",
    "Bus",
    "Calibration",
    "Capabilities",
    "CapabilityError",
    "ChannelKind",
    "ChannelSpec",
    "Device",
    "DeviceId",
    "DiagItem",
    "DiagLevel",
    "Frame",
    "OperatingMode",
    "Sensor",
    "Telemetry",
    "Transport",
    "fold_diag_level",
    "get_actuator",
    "get_sensor",
    "get_transport",
    "register_actuator",
    "register_sensor",
    "register_transport",
    "registered_actuators",
    "registered_sensors",
    "registered_transports",
]

__version__ = "0.0.1"
