"""Hardware from another package: a robot config builds it without editing actuo.

A package adds a transport, an actuator model and a bus by entry points (or
decorators), each actuator and bus carrying its own ``from_config``; the
config names them like any shipped plugin.
"""

import subprocess
import sys
import textwrap

import pytest

import actuo
from actuo.core import registry
from actuo.core.contracts import (
    Actuator, ActuatorState, Bus, Calibration, Capabilities, Frame, OperatingMode,
    Transport)
from actuo.drivers.config import _default_bus, build_bus


class VendorTransport(Transport):
    default_bus = "vendor_bus"

    def __init__(self, channel):
        self.channel = channel
        self._open = False

    def open(self):
        self._open = True

    def close(self):
        self._open = False

    @property
    def is_open(self):
        return self._open

    def write(self, frames):
        pass

    def recv(self, timeout):
        return None


class VendorActuator(Actuator):
    @classmethod
    def from_config(cls, spec, where):
        if "id" not in spec:
            raise KeyError("{}: missing required key 'id'".format(where))
        return cls(int(spec["id"]), Calibration(counts_per_rev=4096.0, direction=int(spec.get("direction", 1))))

    @property
    def capabilities(self):
        return Capabilities(modes=(OperatingMode.POSITION,))

    def encode_command(self, command):
        return Frame(payload=b"", address=self.id)

    def matches(self, frame):
        return frame.address == self.id

    def decode_state(self, frame):
        return ActuatorState(position=0.0)

    def encode_mode(self, mode):
        return []


class VendorBus(Bus):
    @classmethod
    def from_config(cls, transport, actuators, joint_names, spec, path):
        bus = cls(transport, actuators)
        bus.joint_names = list(joint_names)
        bus.rate_hz = float(spec.get("rate_hz", 100.0))
        bus.config_path = path
        return bus

    def set_mode(self, mode):
        pass

    def write_batch(self, commands):
        pass

    def read_batch(self, timeout):
        return []

    def last_state(self, device_id):
        return None


class NoReaderActuator(VendorActuator):
    from_config = None


class FakeEntryPoint:
    def __init__(self, name, group, target):
        self.name = name
        self.group = group
        self.value = "vendor_package:{}".format(getattr(target, "__name__", target))
        self._target = target

    def load(self):
        return self._target


@pytest.fixture
def vendor_package(monkeypatch):
    """An installed package offering vendor_* plugins by entry point."""
    offered = {
        "transport": {"vendor_serial": VendorTransport},
        "actuator": {"vendor_servo": VendorActuator,
                     "vendor_no_reader": NoReaderActuator,
                     "vendor_not_a_class": "just a string"},
        "bus": {"vendor_bus": VendorBus},
        "sensor": {},
    }

    def entry_points(kind):
        group = registry.ENTRY_POINT_GROUPS[kind]
        return [FakeEntryPoint(name, group, target)
                for name, target in offered[kind].items()]

    monkeypatch.setattr(registry, "_entry_points", entry_points)
    for kind, names in offered.items():
        loaded = registry._KINDS[kind][0]
        for name in names:
            monkeypatch.delitem(loaded, name, raising=False)
    yield
    for kind, names in offered.items():
        loaded = registry._KINDS[kind][0]
        for name in names:
            loaded.pop(name, None)


def write_config(tmp_path, actuator_plugin="vendor_servo", bus_block=""):
    path = tmp_path / "robot.yaml"
    path.write_text(textwrap.dedent("""\
        bus:
          transport:
            plugin: vendor_serial
            channel: /dev/null
        {bus_block}
        joints:
          - name: shoulder
            actuator: {{plugin: {plugin}, id: 3}}
          - name: elbow
            actuator: {{plugin: {plugin}, id: 4, direction: -1}}
        """).format(plugin=actuator_plugin, bus_block=bus_block))
    return str(path)


def test_a_config_builds_plugins_offered_by_entry_points(vendor_package, tmp_path):
    path = write_config(tmp_path, bus_block="  rate_hz: 50")

    bus, names = build_bus(path)

    assert type(bus) is VendorBus             # the transport's default_bus
    assert isinstance(bus.transport, VendorTransport)
    assert bus.transport.channel == "/dev/null"
    assert names == bus.joint_names == ["shoulder", "elbow"]
    assert [a.id for a in bus.actuators] == [3, 4]
    assert bus.actuators[1].calibration.direction == -1
    assert bus.rate_hz == 50.0
    assert bus.config_path == path


def test_entry_point_plugins_are_listed_with_the_shipped_ones(vendor_package):
    assert {"vendor_servo", "kondo_ics", "dynamixel"} <= set(registry.known("actuator"))
    assert {"vendor_bus", "kondo_rcb4_bus"} <= set(registry.known("bus"))


def test_an_actuator_without_a_config_reader_is_refused_by_name(vendor_package, tmp_path):
    with pytest.raises(TypeError, match="vendor_no_reader.*from_config"):
        build_bus(write_config(tmp_path, actuator_plugin="vendor_no_reader"))


def test_an_entry_point_that_is_not_a_plugin_class_is_refused(vendor_package):
    with pytest.raises(TypeError, match="vendor_not_a_class.*not a Actuator class"):
        registry.get_actuator("vendor_not_a_class")


def test_an_unknown_plugin_lists_what_is_known(vendor_package):
    with pytest.raises(KeyError, match="vendor_servo"):
        registry.get_actuator("no_such_servo")


def test_a_shipped_name_is_not_taken_by_an_entry_point(monkeypatch):
    impostor = FakeEntryPoint("serial", registry.ENTRY_POINT_GROUPS["transport"],
                              VendorTransport)
    monkeypatch.setattr(registry, "_entry_points", lambda kind: [impostor])

    assert registry.get_transport("serial") is not VendorTransport


def test_the_rcb4_over_udp_transport_is_driven_by_the_rcb4_bus():
    # Swapping the cable for the WiFi relay (--transport kondo_rcb4_udp) on a
    # config without bus.plugin used to fall through to the CAN bus.
    pytest.importorskip("rcb4")
    assert _default_bus({"plugin": "kondo_rcb4_udp"}, [], "robot.yaml") == "kondo_rcb4_bus"
    assert _default_bus({"plugin": "kondo_rcb4"}, [], "robot.yaml") == "kondo_rcb4_bus"
    assert _default_bus({"plugin": "kondo_h7"}, [], "robot.yaml") == "kondo_h7_bus"


def test_servos_on_the_plain_serial_transport_pick_their_own_bus():
    from actuo.drivers.actuators.dynamixel import DynamixelActuator

    servo = DynamixelActuator.__new__(DynamixelActuator)
    assert _default_bus({"plugin": "serial"}, [servo], "robot.yaml") == "dynamixel_bus"
    assert _default_bus({"plugin": "serial"}, [], "robot.yaml") == "serial_bus"


def test_import_actuo_loads_no_robot_model_or_vendor_library():
    code = ("import sys, actuo; "
            "heavy = [m for m in ('skrobot', 'rcb4', 'can', 'feetech_cli') if m in sys.modules]; "
            "print(heavy)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         check=True).stdout.strip()
    assert out == "[]"


def test_the_exported_names_resolve():
    for name in actuo.__all__:
        assert getattr(actuo, name) is not None
    assert actuo.build_bus is build_bus
