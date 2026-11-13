"""Config -> live objects for a DYNAMIXEL robot, with no hardware.

The point of these is that a robot is described by one file and nothing is
restated: the protocol version, the control table, the resolution and the mode
list all follow from ``model:``. Every assertion below is really "this cannot be
said twice, so it cannot be said inconsistently" -- two sources of truth for
one fact drift apart, and the drift only shows on hardware.
"""

import math
import textwrap

import pytest

from actuo.core.contracts import OperatingMode

from actuo.drivers.actuators.dynamixel import DynamixelActuator
from actuo.drivers.buses.dynamixel_bus import DynamixelBus
from actuo.drivers.config import build_actuators, build_bus, load_document


def write(tmp_path, text):
    path = tmp_path / "robot.yaml"
    path.write_text(textwrap.dedent(text))
    return str(path)


AX_ROBOT = """
    bus:
      transport:
        plugin: serial
        channel: /dev/null
        bitrate: 1000000

    joints:
      - name: head_joint0
        actuator: {plugin: dynamixel, id: 15, model: AX-12A,
                   position_limit_deg: [-100.0, 100.0]}
      - name: hinge_joint1
        actuator: {plugin: dynamixel, id: 14, model: AX-12A, direction: -1}
      - name: pick_joint
        actuator: {plugin: dynamixel, id: 8, model: AX-12A, offset_deg: 5.0}
"""


def test_a_robot_file_builds_actuators_in_document_order(tmp_path):
    """That order is the contract with ros2_control and the shm array."""
    document = load_document(write(tmp_path, AX_ROBOT))
    actuators, names = build_actuators(document["joints"])
    assert names == ["head_joint0", "hinge_joint1", "pick_joint"]
    assert [a.id for a in actuators] == [15, 14, 8]
    assert all(isinstance(a, DynamixelActuator) for a in actuators)


def test_the_model_supplies_resolution_and_protocol(tmp_path):
    document = load_document(write(tmp_path, AX_ROBOT))
    actuators, _ = build_actuators(document["joints"])
    head = actuators[0]
    assert head.protocol == 1
    assert head.model.counts_per_rev == pytest.approx(1228.8)
    assert head.capabilities.modes == (OperatingMode.POSITION,)


def test_limits_and_direction_come_from_the_robot_file(tmp_path):
    document = load_document(write(tmp_path, AX_ROBOT))
    actuators, _ = build_actuators(document["joints"])
    head, hinge, pick = actuators

    low, high = head.calibration.position_limit
    assert (math.degrees(low), math.degrees(high)) == pytest.approx((-100, 100))
    assert hinge.calibration.direction == -1
    assert hinge.calibration.position_limit is None      # not taught => no limit
    # offset_deg shifts the count that reads as 0 rad, the same convention the
    # kondo_ics and feetech plugins use: commanding 0 rad now targets
    # centre + 5 deg, so the mechanical centre reads as -5 deg.
    assert pick.position_counts(0.0) == 529
    assert pick.position_counts(math.radians(-5.0)) == 512


def test_bus_plugin_and_framing_are_inferred_from_the_servos(tmp_path):
    """Neither `bus.plugin` nor `transport.protocol` appears in AX_ROBOT."""
    bus, names = build_bus(write(tmp_path, AX_ROBOT))
    assert isinstance(bus, DynamixelBus)
    assert bus.protocol == 1
    assert bus.transport.protocol == 1        # serial framing follows the model
    assert len(names) == 3


def test_a_protocol_stated_twice_and_inconsistently_is_refused(tmp_path):
    path = write(tmp_path, """
        bus:
          transport:
            plugin: serial
            channel: /dev/null
            protocol: 2
        joints:
          - name: j
            actuator: {plugin: dynamixel, id: 1, model: AX-12A}
    """)
    with pytest.raises(ValueError, match="protocol"):
        build_bus(path)


def test_an_unknown_model_name_is_refused_at_config_time(tmp_path):
    """Better to fail while loading a file than to write to a guessed address."""
    path = write(tmp_path, """
        bus:
          transport: {plugin: serial, channel: /dev/null}
        joints:
          - name: j
            actuator: {plugin: dynamixel, id: 1, model: XM430-W350-T}
    """)
    with pytest.raises(LookupError, match="XM430"):
        build_bus(path)


def test_model_is_required(tmp_path):
    path = write(tmp_path, """
        bus:
          transport: {plugin: serial, channel: /dev/null}
        joints:
          - name: j
            actuator: {plugin: dynamixel, id: 1}
    """)
    with pytest.raises(KeyError, match="model"):
        build_bus(path)


def test_duplicate_ids_are_refused(tmp_path):
    path = write(tmp_path, """
        bus:
          transport: {plugin: serial, channel: /dev/null}
        joints:
          - name: a
            actuator: {plugin: dynamixel, id: 1, model: AX-12A}
          - name: b
            actuator: {plugin: dynamixel, id: 1, model: AX-12A}
    """)
    with pytest.raises(ValueError, match="duplicate actuator ids"):
        build_bus(path)
