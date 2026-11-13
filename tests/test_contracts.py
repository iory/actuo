"""Standalone unit tests for actuo.core -- run without ROS.

    pytest tests/test_contracts.py -q

These exercise the pure abstraction: calibration round-trips, capability
checking (fail-loud), and a fake Transport/Actuator/Bus wired through the
registry -- proving the layers compose without any real hardware.
"""

from __future__ import annotations

import os
import sys

import numpy as np


from actuo.core.contracts import (  # noqa: E402
    Actuator,
    ActuatorCommand,
    ActuatorState,
    Bus,
    Calibration,
    Capabilities,
    CapabilityError,
    ChannelKind,
    ChannelSpec,
    Frame,
    OperatingMode,
    Sensor,
    Transport,
)
from actuo.core.registry import (  # noqa: E402
    get_actuator,
    get_sensor,
    get_transport,
    register_actuator,
    register_sensor,
    register_transport,
)


def test_calibration_roundtrip():
    calib = Calibration(counts_per_rev=4096.0, zero_offset_counts=2048.0, direction=1)
    for rad in np.linspace(-np.pi, np.pi, 11):
        counts = calib.rad_to_counts(rad)
        assert np.isclose(calib.counts_to_rad(counts), rad, atol=1e-9)


def test_calibration_direction_and_gear():
    calib = Calibration(counts_per_rev=1000.0, direction=-1, gear_ratio=6.0)
    # direction flips sign, gear scales
    assert np.isclose(calib.counts_to_rad(calib.rad_to_counts(0.5)), 0.5, atol=1e-9)


def test_capability_fail_loud():
    caps = Capabilities(modes=(OperatingMode.POSITION,))
    caps.require(OperatingMode.POSITION)  # ok
    try:
        caps.require(OperatingMode.CURRENT)
    except CapabilityError:
        pass
    else:
        raise AssertionError("expected CapabilityError for unsupported mode")


# --- fake plugins exercising the full contract through the registry ---


@register_transport("fake")
class FakeTransport(Transport):
    """A request/response transport: it answers each request with a status frame.

    Contrast with a broadcast transport (see FakeStreamingTransport), which has
    no pairing at all -- the contract has to serve both.
    """

    def __init__(self):
        self._open = False
        self.written = []

    def open(self):
        self._open = True

    def close(self):
        self._open = False

    @property
    def is_open(self):
        return self._open

    def write(self, frames):
        self.written.extend(frames)

    def recv(self, timeout):
        return None  # nothing arrives unsolicited on this transport

    def transact(self, frames, timeout):
        # echo a status frame per request, preserving the address it came from
        return [
            Frame(payload=b"\x01" + f.payload, address=f.address) for f in frames
        ]


@register_actuator("fake")
class FakeActuator(Actuator):
    @property
    def capabilities(self):
        return Capabilities(
            modes=(OperatingMode.POSITION, OperatingMode.VELOCITY),
            has_velocity_sense=True,
        )

    def encode_command(self, command):
        counts = int(round(self.calibration.rad_to_counts(command.position or 0.0)))
        return Frame(
            payload=counts.to_bytes(4, "little", signed=True), address=self.id
        )

    def matches(self, frame):
        return frame.address == self.id

    def decode_state(self, frame):
        counts = int.from_bytes(frame.payload[1:5], "little", signed=True)
        return ActuatorState(position=self.calibration.counts_to_rad(counts))

    def encode_mode(self, mode):
        self.capabilities.require(mode)
        return [
            Frame(
                payload=bytes([0xA0, list(OperatingMode).index(mode)]),
                address=self.id,
            )
        ]


class FakeBus(Bus):
    def set_mode(self, mode):
        for act in self.actuators:
            for frame in act.encode_mode(mode):
                self.transport.write([frame])

    def write_batch(self, commands):
        frames = [a.encode_command(c) for a, c in zip(self.actuators, commands)]
        self.transport.write(frames)  # one batched write

    def read_batch(self, timeout):
        req = [a.encode_command(ActuatorCommand(mode=OperatingMode.POSITION,
                                                position=0.0))
               for a in self.actuators]
        replies = self.transport.transact(req, timeout)
        self._states = {a.id: a.decode_state(r)
                        for a, r in zip(self.actuators, replies)}
        return list(self._states.values())

    def last_state(self, device_id):
        return getattr(self, "_states", {}).get(device_id)


def test_registry_and_bus_roundtrip():
    transport_cls = get_transport("fake")
    actuator_cls = get_actuator("fake")
    transport = transport_cls()
    transport.open()
    calib = Calibration(counts_per_rev=4096.0, zero_offset_counts=2048.0)
    actuators = [actuator_cls(i, calib) for i in (1, 2, 3)]
    bus = FakeBus(transport, actuators)
    assert bus.size == 3

    bus.set_mode(OperatingMode.POSITION)
    bus.write_batch([
        ActuatorCommand(mode=OperatingMode.POSITION, position=0.0),
        ActuatorCommand(mode=OperatingMode.POSITION, position=np.pi / 2),
        ActuatorCommand(mode=OperatingMode.POSITION, position=-np.pi / 2),
    ])
    # one batched write of 3 command frames (+ 3 mode frames from set_mode)
    assert len(transport.written) == 6

    states = bus.read_batch(timeout=0.01)
    assert len(states) == 3
    assert all(s.position is not None for s in states)


# --- broadcast/streaming transports: no request/response pairing at all ---
# CAN servo drivers upload status on their own schedule and never "reply", so
# the contract must serve them without forcing a fake transact(). This pins that.


class FakeStreamingTransport(Transport):
    """Frames arrive unsolicited; nothing answers a request."""

    def __init__(self, inbox=()):
        self._open = False
        self.written = []
        self._inbox = list(inbox)

    def open(self):
        self._open = True

    def close(self):
        self._open = False

    @property
    def is_open(self):
        return self._open

    def write(self, frames):
        self.written.extend(frames)

    def recv(self, timeout):
        return self._inbox.pop(0) if self._inbox else None


def test_streaming_transport_refuses_transact_instead_of_faking_it():
    transport = FakeStreamingTransport()
    transport.open()
    # It must NOT invent a request/response pairing the wire protocol lacks.
    try:
        transport.transact([Frame(payload=b"\x01", address=1)], timeout=0.01)
    except CapabilityError:
        pass
    else:
        raise AssertionError("streaming transport should refuse transact()")


def test_frame_carries_address_so_can_style_ids_survive():
    # A CAN frame's identity is its arbitration id, which for CubeMars also
    # encodes the control mode. Frame must carry it without smuggling it into
    # the payload, or narrowing it to a uint8 servo id -- the narrowing this
    # design exists to prevent.
    frame = Frame(payload=b"\x00\x01", address=(6 << 8) | 104, meta={"extended": True})
    assert frame.address == 0x668
    assert frame.meta["extended"] is True
    assert frame.payload == b"\x00\x01"


# --- custom telemetry extensibility: a user-defined pressure sensor plugin ---
# This is the whole point of the Device/Sensor/channel design: adding a custom
# barometric board like rcb4's air board requires NO edits to actuo.core, only a
# registered plugin conforming to the contract.


@register_sensor("air_board")
class AirPressureSensor(Sensor):
    """A custom board on the same bus reporting barometric pressure + temp.
    Nothing about this class lives in actuo.core."""

    @property
    def channels(self):
        return (
            ChannelSpec("air_board/pressure", "kPa", ChannelKind.STATE),
            ChannelSpec("air_board/temperature", "degC", ChannelKind.STATE),
        )

    def encode_query(self):
        return [Frame(payload=bytes([0x51]), address=self.id)]  # request-reading frame

    def decode_state(self, frame):
        raw_p = int.from_bytes(frame.payload[0:2], "little")
        raw_t = frame.payload[2]
        return {
            "air_board/pressure": raw_p / 10.0,     # kPa
            "air_board/temperature": float(raw_t),  # degC
        }


def test_custom_sensor_plugin_zero_core_edits():
    sensor_cls = get_sensor("air_board")           # resolved purely via registry
    sensor = sensor_cls(0x20)
    names = {c.name for c in sensor.channels}
    assert names == {"air_board/pressure", "air_board/temperature"}
    # simulate a reading frame: pressure=1013 (=>101.3 kPa), temp=25
    frame = Frame(payload=(1013).to_bytes(2, "little") + bytes([25]), address=0x20)
    telemetry = sensor.decode_state(frame)
    assert np.isclose(telemetry["air_board/pressure"], 101.3)
    assert np.isclose(telemetry["air_board/temperature"], 25.0)


def test_actuator_aux_channels():
    """An actuator can also carry arbitrary extra channels via ActuatorState.aux
    without any fixed-struct change."""
    state = ActuatorState(position=0.5, aux={"strain_gauge/force": 12.3})
    assert state.velocity is None            # unavailable stays None, never faked
    assert state.aux["strain_gauge/force"] == 12.3


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("\nall {} tests passed".format(len(fns)))


if __name__ == "__main__":
    _run_all()
