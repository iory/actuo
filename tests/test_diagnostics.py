"""Tests for ROS-free diagnostics primitives and bus hook defaults."""

from __future__ import annotations

import os
import sys


from actuo.core.contracts import (  # noqa: E402
    Actuator,
    ActuatorCommand,
    ActuatorState,
    Bus,
    Calibration,
    Capabilities,
    Frame,
    OperatingMode,
    Transport,
)
from actuo.core.diagnostics import (  # noqa: E402
    DiagItem,
    DiagLevel,
    fold_diag_level,
)


class _NullTransport(Transport):
    def open(self):
        pass

    def close(self):
        pass

    @property
    def is_open(self):
        return True

    def write(self, frames):
        pass

    def recv(self, timeout):
        return None


class _NullActuator(Actuator):
    @property
    def capabilities(self):
        return Capabilities(modes=(OperatingMode.POSITION,))

    def encode_command(self, command):
        return Frame(payload=b"", address=self.id)

    def matches(self, frame):
        return frame.address == self.id

    def decode_state(self, frame):
        return ActuatorState()

    def encode_mode(self, mode):
        self.capabilities.require(mode)
        return []


class _NullBus(Bus):
    def set_mode(self, mode):
        pass

    def write_batch(self, commands):
        pass

    def read_batch(self, timeout):
        return [ActuatorState() for _ in self.actuators]

    def last_state(self, device_id):
        return None


def test_fold_diag_level_uses_worst_case():
    items = [
        DiagItem("a", DiagLevel.OK, "ok", {}),
        DiagItem("b", DiagLevel.WARN, "warn", {}),
        DiagItem("c", DiagLevel.ERROR, "error", {}),
    ]
    assert fold_diag_level(items) == DiagLevel.ERROR


def test_fold_diag_level_handles_empty_list():
    assert fold_diag_level([]) == DiagLevel.OK
    assert fold_diag_level([], default=DiagLevel.STALE) == DiagLevel.STALE


def test_bus_diagnose_default_is_empty():
    bus = _NullBus(
        _NullTransport(), [_NullActuator(1, Calibration(counts_per_rev=1.0))]
    )
    assert bus.diagnose() == []


def test_actuator_fault_name_default_is_honest_unknown():
    actuator = _NullActuator(1, Calibration(counts_per_rev=1.0))
    assert actuator.fault_name(7) == "unknown fault code 7"


def test_actuator_fault_name_does_not_call_a_healthy_device_faulty():
    """0 is the absence of a fault, not a fault this family failed to name.

    A healthy Kondo arm published `error_name: unknown fault code 0` on every
    joint, which reads as "something is wrong and we do not know what".
    """
    actuator = _NullActuator(1, Calibration(counts_per_rev=1.0))
    assert actuator.fault_name(0) == "none"
