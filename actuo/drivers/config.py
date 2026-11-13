"""Build a live Bus from a per-robot YAML file.

The joint->actuator map, node ids, gearing, direction, offsets and limits live
in config, never hard-coded. This module is the only place
that turns that document into objects, and it resolves plugins purely through
``actuo.core.registry`` -- so a new transport or actuator family is usable from
config with zero edits here.

Schema
------
.. code-block:: yaml

    bus:
      stale_after: 1.0            # s without a status frame => fault
      transport:
        plugin: python_can
        interface: socketcan      # slcan / gs_usb on macOS + a USB adapter
        channel: can0             # /dev/tty.usbmodem... for slcan
        bitrate: 1000000          # ignored by socketcan (kernel owns it)

    joints:
      - name: joint1              # must match the URDF joint name
        actuator:
          plugin: cubemars
          id: 104                 # CAN node id
          model: AK60-39          # documentation only
          gear_ratio: 39.0
          direction: 1            # +1 / -1
          zero_offset_deg: 0.0    # output-shaft deg reported at 0 rad
          pole_pairs: 0           # 0 => leave rad/s conversion unavailable
          torque_constant: null   # N*m/A => enables CURRENT mode
          position_limit_deg: [-45.0, 45.0]   # optional, output shaft
          limits:                 # SET_POS_SPD trapezoid, wire-native ERPM
            max_velocity_erpm: 220.0
            max_acceleration_erpm_s: 220.0

A 3.0-generation CubeMars motor driven by impedance instead reads:

.. code-block:: yaml

      - name: right_hip_yaw
        actuator:
          plugin: cubemars_v3     # force control, NOT the 2.0 `cubemars_mit`
          id: 5
          model: AK70-9
          gear_ratio: 9.0
          direction: -1
          zero_offset_deg: 15.4   # same datum as servo mode: the status frame
          pole_pairs: 21
          position_limit_deg: [-180.0, 180.0]
          force_control:          # wire ranges for THIS variant. No defaults.
            p_max: 12.56          # rad
            v_max: 30.0           # rad/s
            t_max: 32.0           # N*m
          impedance:
            kp: 300               # 0..500, N*m/rad at the output shaft
            kd: 4.5               # 0..5 ONLY. A bigger damping gain cannot be
                                  # put on this wire at all; it is a 12-bit
                                  # field over 0..5 on every AK model, so it
                                  # would arrive clamped with no error
                                  # anywhere. A policy trained with kd = 20
                                  # needs its PD run on the host against a
                                  # torque command, not this field.
            torque_limit_nm: 29.2
          # no `limits`: that bounds the SET_POS_SPD trapezoid, and force
          # control has no trapezoid.

A Kondo ARM-H7 robot instead reads:

.. code-block:: yaml

    bus:
      plugin: kondo_h7_bus
      symbols: kondoh7_symbols.yaml   # resolved relative to THIS file
      state_slots: [current_angle]    # one board round trip each
      move_time_ms: 20.0              # board-side interpolation time per cycle
      transport:
        plugin: kondo_h7
        channel: /dev/ttyUSB1
        bitrate: 1000000

    joints:
      - name: head_neck_y
        actuator:
          plugin: kondo_ics
          id: 0                       # flat servo id = index in servo_vector
          ics_channel: J1             # documentation only; not an address
          direction: 1
          offset_deg: 0.0             # output angle at the servo's neutral pulse
          position_limit_deg: [-60.0, 60.0]

A DYNAMIXEL robot (e.g. servos on a DXHUB, which is a plain USB-serial adapter
with no gateway MCU) reads:

.. code-block:: yaml

    bus:
      plugin: dynamixel_bus       # inferred when every actuator is a dynamixel
      read_timeout: 0.05
      transport:
        plugin: serial
        channel: /dev/ttyUSB0     # FTDI; /dev/tty.usbserial-* on macOS
        bitrate: 1000000
        # `protocol` is deliberately omitted: it follows from the servo model,
        # and stating it twice is a chance to state it inconsistently.

    joints:
      - name: head_joint0
        actuator:
          plugin: dynamixel
          id: 15                  # plain bus id -- one line, no port field
          model: AX-12A           # REQUIRED. e-Manual product name; decides the
                                  # control table. actuo-dxl-probe reads it off
                                  # the hardware so it never has to be guessed.
          direction: 1
          offset_deg: 0.0         # 0 rad targets centre + this many degrees
          position_limit_deg: [-134.0, 134.0]
          # homing_offset_counts: 512   # only if this servo's own EEPROM
          #                              # Homing Offset register was written;
          #                              # 0 (omit) for every untouched servo

A robot brought up from a URDF can leave most of ``joints`` out: with
``modules``, ``urdf`` and ``joint_defaults`` the list is built from the
model (see :mod:`actuo.drivers.robot_document`).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import yaml

from actuo.core.contracts import Actuator, Bus, Transport
from actuo.core.registry import get_actuator, get_transport

# Each actuator family and bus is imported by the builder that uses it, not
# here: a vendor's modules need that vendor's extra (``actuo[kondo]`` for
# rcb4, ``actuo[can]`` for python-can), and a config naming only Dynamixel
# servos must load without either. The plugin names resolve through
# actuo.core.registry, which imports a plugin's module when it is first named.
from actuo.drivers.robot_document import (  # noqa: F401  (resolve_data_path re-exported)
    expand_document, resolve_data_path)


def _require(mapping: Dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise KeyError("{}: missing required key {!r}".format(where, key))
    return mapping[key]


def load_document(path: str) -> Dict[str, Any]:
    """The config at ``path``, its ``joints`` expanded (``modules``, ``urdf``,
    ``joint_defaults``; see :mod:`actuo.drivers.robot_document`)."""
    with open(path, "r") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError("{}: expected a YAML mapping at the top level".format(path))
    return expand_document(document, path)


def build_transport(spec: Dict[str, Any]) -> Transport:
    """Instantiate the named transport plugin from its config block."""
    options = dict(spec)
    plugin = options.pop("plugin", None)
    if plugin is None:
        raise KeyError("bus.transport: missing required key 'plugin'")
    return get_transport(plugin)(**options)


def _build_cubemars(spec: Dict[str, Any], where: str) -> Actuator:
    """Translate the config block for one CubeMars motor into an Actuator."""
    from actuo.drivers.actuators.cubemars import CubeMarsCalibration, CubeMarsLimits

    limits_spec = _require(spec, "limits", where)
    position_limit = spec.get("position_limit_deg")

    calibration = CubeMarsCalibration(
        gear_ratio=float(_require(spec, "gear_ratio", where)),
        direction=int(spec.get("direction", 1)),
        # config speaks output-shaft degrees; "counts" are deg*10 on this wire
        zero_offset_counts=float(spec.get("zero_offset_deg", 0.0)) * 10.0,
        pole_pairs=int(spec.get("pole_pairs", 0)),
        torque_constant=spec.get("torque_constant"),
        position_limit=(
            (math.radians(position_limit[0]), math.radians(position_limit[1]))
            if position_limit
            else None
        ),
    )
    limits = CubeMarsLimits(
        max_velocity_erpm=float(_require(limits_spec, "max_velocity_erpm", where)),
        max_acceleration_erpm_s=float(
            _require(limits_spec, "max_acceleration_erpm_s", where)
        ),
    )
    actuator_cls = get_actuator("cubemars")
    actuator = actuator_cls(
        int(_require(spec, "id", where)),
        calibration,
        limits=limits,
        use_pos_spd=bool(spec.get("use_pos_spd", True)),
    )
    actuator.startup_reference_rad = _startup_reference(spec)
    return actuator


def _startup_reference(spec: Dict[str, Any]) -> Optional[float]:
    """``startup_reference_deg`` -> radians, or None when absent.

    Where this joint is expected to be at power-on, in output-shaft degrees.
    Optional, and its ONLY job is to pick which gear window the encoder is in
    (see CubeMarsActuator.snap_to_window): the encoder supplies every bit of
    precision, this supplies one integer. So it needs to be accurate only to a
    quarter of a window -- +-15 deg on an AK80-6 -- which an eyeballed startup
    pose reaches easily. Absent means no window recovery, i.e. the previous
    behaviour of trusting whichever window the driver powered up in.

    One function for every CubeMars model whose feedback is the servo status
    frame. It used to be read by the servo builder alone, so a config switched
    from `cubemars` to `cubemars_v3` -- the same motor, the same frame, the
    same windows -- silently LOST its window recovery: every reader that asks
    ``getattr(actuator, "startup_reference_rad", None)`` took the missing
    attribute for "no recovery wanted" and showed joints a whole window out as
    ordinary poses. Found on a CubeMars humanoid's upper body, 2026-09-29.
    """
    reference = spec.get("startup_reference_deg")
    return None if reference is None else math.radians(float(reference))


def _build_feetech_scs(spec: Dict[str, Any], where: str) -> Actuator:
    """Translate the config block for one SCS0009 servo into an Actuator.

    ``middle_offset_deg`` shifts the servo's electrical centre (the demo's
    per-servo ``MiddlePos`` calibration): it moves the count that reads as 0 rad
    off the nominal 511, so a finger's neutral pose is truly neutral.
    """
    from actuo.drivers.actuators.feetech_scs import ScsCalibration, SCS_CENTER_COUNTS

    position_limit = spec.get("position_limit_deg")
    # counts per output degree = resolution / range = 1024 / 300.
    counts_per_deg = 1024.0 / 300.0
    zero_offset = float(SCS_CENTER_COUNTS) + \
        float(spec.get("middle_offset_deg", 0.0)) * counts_per_deg
    calibration = ScsCalibration(
        direction=int(spec.get("direction", 1)),
        zero_offset_counts=zero_offset,
        position_limit=(
            (math.radians(position_limit[0]), math.radians(position_limit[1]))
            if position_limit else None),
    )
    return get_actuator("feetech_scs0009")(int(_require(spec, "id", where)),
                                            calibration)


def _build_dynamixel(spec: Dict[str, Any], where: str) -> Actuator:
    """Translate the config block for one DYNAMIXEL servo into an Actuator.

    ``model`` is required and is the e-Manual product name (e.g. ``AX-12A``).
    It is not optional and has no default: the control-table layout, resolution
    and available modes all follow from it, and a wrong guess writes a goal
    position into whatever register happens to live at that address on the real
    servo. ``actuo-dxl-probe`` reads the model number off the hardware, so this
    field can always be filled from the robot rather than from memory.

    ``offset_deg`` shifts the count that reads as 0 rad -- the same convention
    the ``kondo_ics`` and ``feetech_scs0009`` plugins use. Commanding 0 rad
    targets ``centre + offset_deg``, so the servo's mechanical centre reports
    ``-offset_deg``. It is the joint's zero expressed relative to the servo's
    centre, and it belongs in the robot file because it is a property of how the
    horn was bolted on, not of the servo model.

    ``homing_offset_counts`` (default 0.0) is a SEPARATE, HARDWARE fact: the
    value actually written to this servo's own EEPROM Homing Offset register,
    if that register was ever touched. It is not another way to spell
    ``offset_deg`` -- offset_deg relabels which rad a raw count is called;
    homing_offset_counts tells `position_refusal` where THIS servo's valid
    goal-count window actually starts, because a Homing Offset write shifts
    that window on the real hardware whether or not the software knows about
    it. Leave it at 0 unless `actuo-dxl-probe` (or the write itself) confirms
    a nonzero value is on this specific servo.
    """
    from actuo.drivers.actuators.dynamixel import DynamixelCalibration
    from actuo.drivers.dynamixel_models import load_model_table

    table = load_model_table(spec.get("model_table"))
    model = table.by_name(_require(spec, "model", where))
    counts_per_deg = model.counts_per_rev / 360.0
    position_limit = spec.get("position_limit_deg")
    calibration = DynamixelCalibration(
        counts_per_rev=model.counts_per_rev,
        zero_offset_counts=(float(spec.get("center_counts", model.center_counts))
                            + float(spec.get("offset_deg", 0.0)) * counts_per_deg),
        homing_offset_counts=float(spec.get("homing_offset_counts", 0.0)),
        direction=int(spec.get("direction", 1)),
        gear_ratio=float(spec.get("gear_ratio", 1.0)),
        torque_constant=spec.get("torque_constant"),
        position_limit=(
            (math.radians(position_limit[0]), math.radians(position_limit[1]))
            if position_limit else None),
    )
    return get_actuator("dynamixel")(
        int(_require(spec, "id", where)), calibration, model,
        profile_velocity=spec.get("profile_velocity"))


def _build_kondo_ics(spec: Dict[str, Any], where: str) -> Actuator:
    """Translate the config block for one Kondo ICS servo into an Actuator.

    ``offset_deg`` shifts which output angle the servo's neutral pulse reports,
    i.e. it is the joint's zero relative to the servo's mechanical centre. The
    servo id doubles as its index in the board's servo vector, so it is the one
    field that must match the hardware exactly.

    ``stretch`` (1..127) and ``hold_relax`` (``error_deg``, ``seconds``,
    ``squeeze_deg``, all required) are optional; only the RCB-4mini bus acts
    on them (see ``KondoIcsActuator``).
    """
    from actuo.drivers.actuators.kondo_ics import (
        CENTER_PULSE, DEFAULT_VELOCITY, HoldRelax, KondoIcsCalibration, PULSE_MAX,
        PULSE_MIN, PULSE_PER_DEG)

    hold_relax = spec.get("hold_relax")
    if hold_relax is not None:
        relax_where = where + ".hold_relax"
        hold_relax = HoldRelax(
            error_rad=math.radians(float(_require(hold_relax, "error_deg", relax_where))),
            seconds=float(_require(hold_relax, "seconds", relax_where)),
            squeeze_rad=math.radians(float(_require(hold_relax, "squeeze_deg",
                                                    relax_where))))
    position_limit = spec.get("position_limit_deg")
    pulse_limit = spec.get("pulse_limit", [PULSE_MIN, PULSE_MAX])
    calibration = KondoIcsCalibration(
        direction=int(spec.get("direction", 1)),
        zero_offset_counts=(float(CENTER_PULSE)
                            + float(spec.get("offset_deg", 0.0)) * PULSE_PER_DEG),
        gear_ratio=float(spec.get("gear_ratio", 1.0)),
        torque_constant=spec.get("torque_constant"),
        position_limit=(
            (math.radians(position_limit[0]), math.radians(position_limit[1]))
            if position_limit else None),
        pulse_limit=(int(pulse_limit[0]), int(pulse_limit[1])),
    )
    return get_actuator("kondo_ics")(
        int(_require(spec, "id", where)),
        calibration,
        velocity=int(spec.get("velocity", DEFAULT_VELOCITY)),
        ics_channel=spec.get("ics_channel"),
        stretch=spec.get("stretch"),
        hold_relax=hold_relax,
    )


#: Per-plugin config translators. A new actuator family adds an entry here (or
#: ships its own builder) instead of growing a branch inside the loader.
def _build_kondo_ics_wheel(spec: Dict[str, Any], where: str) -> Actuator:
    """One continuous-rotation ICS servo, from its MEASURED speed map.

    The three numbers come from ``actuo-wheel-calibrate`` and have no defaults on
    purpose. They differ by up to 2x between wheels of the same model on the
    same robot, so a shared guess does not make a slower base, it makes one that
    drives off at an angle. The tool emits this block; nobody should be typing
    it.
    """
    from actuo.drivers.actuators.kondo_ics_wheel import KondoIcsWheelCalibration

    calibration = KondoIcsWheelCalibration(
        counts_per_rev=1.0,          # unused: this model speaks rad/s, not counts
        direction=int(spec.get("direction", 1)),
        deadband_deg=float(_require(spec, "deadband_deg", where)),
        saturate_deg=float(_require(spec, "saturate_deg", where)),
        rev_s_per_deg=float(_require(spec, "rev_s_per_command_deg", where)),
    )
    rotation_checked = spec.get("rotation_checked")
    return get_actuator("kondo_ics_wheel")(
        int(_require(spec, "id", where)), calibration,
        rotation_checked=(None if rotation_checked is None
                          else str(rotation_checked)))


def _build_cubemars_mit(spec: Dict[str, Any], where: str) -> Actuator:
    """Translate the config block for one CubeMars motor in MIT mode.

    A different plugin from ``cubemars`` rather than a flag on it, because a
    motor is in one control mode or the other for the life of a power cycle and
    the two protocols share nothing on the wire. The manual warns that driving
    with the wrong one can burn the board, so which one is in use should be a
    visible choice in the config, not a default someone inherits.

    Required: ``force_control.{p_max,v_max,t_max}`` (or the older ``mit.*``)
    and the impedance gains. None of them are defaulted; see the actuator module for why a guessed
    range is worse than a missing one.
    """
    from actuo.drivers.actuators.cubemars_mit import (
        MitCalibration as CubeMarsMitCalibration)

    # `force_control` is the name `cubemars_v3` uses for the same three
    # numbers, and they mean the same thing in both generations. Accepting it
    # here means a joint that changes generation changes only its `plugin`
    # line. `mit` stays valid for configs written before that.
    mit = spec.get("force_control") or _require(spec, "mit", where)
    impedance = _require(spec, "impedance", where)
    position_limit = spec.get("position_limit_deg")
    calibration = CubeMarsMitCalibration(
        gear_ratio=float(_require(spec, "gear_ratio", where)),
        direction=int(spec.get("direction", 1)),
        zero_offset_rad=math.radians(float(spec.get("zero_offset_deg", 0.0))),
        p_max=float(_require(mit, "p_max", where + ".mit")),
        v_max=float(_require(mit, "v_max", where + ".mit")),
        t_max=float(_require(mit, "t_max", where + ".mit")),
        kp=float(_require(impedance, "kp", where + ".impedance")),
        kd=float(_require(impedance, "kd", where + ".impedance")),
        torque_limit_nm=impedance.get("torque_limit_nm"),
        position_limit=(
            (math.radians(position_limit[0]), math.radians(position_limit[1]))
            if position_limit
            else None
        ),
    )
    actuator_cls = get_actuator("cubemars_mit")
    actuator = actuator_cls(int(_require(spec, "id", where)), calibration)
    # The same gear-window recovery as servo mode and 3.0: opt in with
    # startup_reference_deg.
    actuator.startup_reference_rad = _startup_reference(spec)
    return actuator


def _build_cubemars_v3(spec: Dict[str, Any], where: str) -> Actuator:
    """Translate the config block for one 3.0-generation CubeMars motor.

    A different plugin from ``cubemars`` even though the same firmware answers
    both: force control and the servo position loop are two controllers, and
    pointing both at one motor is a fight, not a fallback. Which one drives a
    joint belongs in the config where it can be read.

    Shares every servo-mode field -- ``gear_ratio``, ``direction``,
    ``zero_offset_deg``, ``pole_pairs``, ``torque_constant``,
    ``position_limit_deg`` -- because the feedback path is the servo status
    frame unchanged. Adds ``force_control.{p_max,v_max,t_max}`` and the
    impedance gains, none of them defaulted; see the actuator module for why a
    guessed range is worse than a missing one.

    ``limits`` is deliberately NOT read: it bounds the ``SET_POS_SPD``
    trapezoid, and force control has no trapezoid. Accepting it here would
    imply an envelope that nothing applies.
    """
    from actuo.drivers.actuators.cubemars_v3 import CubeMarsV3Calibration

    force = _require(spec, "force_control", where)
    impedance = _require(spec, "impedance", where)
    position_limit = spec.get("position_limit_deg")

    calibration = CubeMarsV3Calibration(
        gear_ratio=float(_require(spec, "gear_ratio", where)),
        direction=int(spec.get("direction", 1)),
        # config speaks output-shaft degrees; "counts" are deg*10 on this wire
        zero_offset_counts=float(spec.get("zero_offset_deg", 0.0)) * 10.0,
        pole_pairs=int(spec.get("pole_pairs", 0)),
        torque_constant=spec.get("torque_constant"),
        position_limit=(
            (math.radians(position_limit[0]), math.radians(position_limit[1]))
            if position_limit
            else None
        ),
        p_max=float(_require(force, "p_max", where + ".force_control")),
        v_max=float(_require(force, "v_max", where + ".force_control")),
        t_max=float(_require(force, "t_max", where + ".force_control")),
        kp=float(_require(impedance, "kp", where + ".impedance")),
        kd=float(_require(impedance, "kd", where + ".impedance")),
        torque_limit_nm=impedance.get("torque_limit_nm"),
    )
    actuator_cls = get_actuator("cubemars_v3")
    actuator = actuator_cls(int(_require(spec, "id", where)), calibration)
    # Same feedback as servo mode, so the same gear-window recovery.
    actuator.startup_reference_rad = _startup_reference(spec)
    return actuator


_ACTUATOR_BUILDERS = {
    "cubemars": _build_cubemars,
    "cubemars_mit": _build_cubemars_mit,
    "cubemars_v3": _build_cubemars_v3,
    "dynamixel": _build_dynamixel,
    "feetech_scs0009": _build_feetech_scs,
    "kondo_ics": _build_kondo_ics,
    "kondo_ics_wheel": _build_kondo_ics_wheel,
}


def build_actuators(joints: List[Dict[str, Any]]) -> Tuple[List[Actuator], List[str]]:
    """Return (actuators, joint_names) in the document's order.

    That order is the contract with ros2_control: it must match the URDF joint
    order the C++ side writes into shm.
    """
    actuators: List[Actuator] = []
    names: List[str] = []
    for index, joint in enumerate(joints):
        where = "joints[{}]".format(index)
        name = str(_require(joint, "name", where))
        spec = dict(_require(joint, "actuator", where))
        plugin = spec.pop("plugin", None)
        if plugin is None:
            raise KeyError("{}.actuator: missing required key 'plugin'".format(where))
        if plugin not in _ACTUATOR_BUILDERS:
            raise KeyError(
                "no config builder for actuator plugin {!r}; known: {}".format(
                    plugin, sorted(_ACTUATOR_BUILDERS)
                )
            )
        actuators.append(_ACTUATOR_BUILDERS[plugin](spec, "{}.actuator".format(where)))
        names.append(name)

    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError("duplicate joint names in config: {}".format(sorted(duplicates)))
    ids = [a.id for a in actuators]
    duplicate_ids = {i for i in ids if ids.count(i) > 1}
    if duplicate_ids:
        raise ValueError(
            "duplicate actuator ids in config: {}".format(sorted(duplicate_ids))
        )
    return actuators, names


def _build_kjs03(spec: Dict[str, Any], where: str):
    """Build a KJS-03 / AIRRELAY sensor board from its config block."""
    from actuo.drivers.sensors.kjs03 import Kjs03Sensor
    device_id = int(_require(spec, "id", where))
    return Kjs03Sensor(device_id, board_rev=spec.get("board_rev"))


#: Sensor plugins the config can name. A new sensor family adds one entry, the
#: same open/closed contract the actuators use -- no core edit, no protocol
#: ``switch`` in shared code.
_SENSOR_BUILDERS = {
    "kjs03": _build_kjs03,
}


def build_sensors(sensors: "List[Dict[str, Any]] | None"):
    """Return ``[(name, Sensor)]`` from an optional top-level ``sensors:`` list.

    Sensors ride the same bus as the joints but are not joints: they carry named
    telemetry channels (pressure, IMU, ...) and, for a gripper board, air
    commands -- never a commanded position. A robot with
    no sensor board simply omits the section.
    """
    out = []
    names: List[str] = []
    for index, entry in enumerate(sensors or []):
        where = "sensors[{}]".format(index)
        name = str(_require(entry, "name", where))
        spec = dict(_require(entry, "sensor", where))
        plugin = spec.pop("plugin", None)
        if plugin is None:
            raise KeyError("{}.sensor: missing required key 'plugin'".format(where))
        if plugin not in _SENSOR_BUILDERS:
            raise KeyError(
                "no config builder for sensor plugin {!r}; known: {}".format(
                    plugin, sorted(_SENSOR_BUILDERS)))
        if name in names:
            raise ValueError("duplicate sensor name in config: {!r}".format(name))
        out.append((name, _SENSOR_BUILDERS[plugin](spec, "{}.sensor".format(where))))
        names.append(name)
    return out


def load_sensors(path: str):
    """Convenience: parse just the ``sensors:`` section of a robot config."""
    document = load_document(path)
    return build_sensors(document.get("sensors"))


def sensor_frames(sensors: "List[Dict[str, Any]] | None") -> Dict[str, str]:
    """Return ``{sensor name: frame_id}`` for the sensors that declare one.

    ``frame_id`` sits beside ``name`` rather than inside the ``sensor:`` block
    because it is not a property of the device -- the same board bolted to a
    different link reports the same numbers in a different frame. It is the
    robot's statement about where that board is, and it must match the link the
    URDF puts there (for example ``imu_link``, whose transform should be
    measured on the robot, not read off a drawing).

    A sensor with no ``frame_id`` is not defaulted. An IMU published in a
    guessed frame is silently rotated for everything downstream that transforms
    it, which is exactly the class of plausible-but-wrong output this package
    refuses to produce.
    """
    out: Dict[str, str] = {}
    for index, entry in enumerate(sensors or []):
        where = "sensors[{}]".format(index)
        name = str(_require(entry, "name", where))
        frame = entry.get("frame_id")
        if frame is not None:
            out[name] = str(frame)
    return out


def load_sensor_frames(path: str) -> Dict[str, str]:
    """Convenience: read ``{sensor name: frame_id}`` from a robot config."""
    return sensor_frames(load_document(path).get("sensors"))


def load_max_velocities(path: str) -> Dict[str, float]:
    """{joint: rad/s} for joints whose entry sets ``max_velocity``.

    A joint-level key, not an actuator one: it is a property of the joint as
    the servicer drives it (how fast its command may move), whatever actuator
    sits behind it. See ServoSupervisor ``max_velocity``.
    """
    out = {}
    for joint in load_document(path).get("joints", []):
        if "max_velocity" in joint:
            value = float(joint["max_velocity"])
            if not value > 0:
                raise ValueError("{}: joint {} max_velocity must be > 0".format(
                    path, joint.get("name")))
            out[str(joint["name"])] = value
    return out


def load_joint_datums(path: str) -> Dict[str, Dict[str, float]]:
    """{joint: {"zero_offset_deg", "direction"}} for joints configured by them.

    Read from the config rather than asked of the actuator: the numbers are
    the config's, and what window_state needs is WHICH datum a remembered
    position was recorded under, so it can convert when the config changes.
    Only joints whose actuator block carries ``zero_offset_deg`` are listed --
    the CubeMars plugins, which read it as ``model = direction * (raw -
    zero_offset_deg)``. Others are left out and their memory is never
    converted.
    """
    out = {}
    for joint in load_document(path).get("joints", []):
        spec = joint.get("actuator", {})
        if "zero_offset_deg" not in spec:
            continue
        out[str(joint["name"])] = {
            "zero_offset_deg": float(spec["zero_offset_deg"]),
            "direction": float(spec.get("direction", 1)),
        }
    return out


#: Actuator plugins a ``servo_map`` fills in, and the rcb4 ``type`` each one
#: must have there. ``continuous`` is a wheel in rcb4's map.
_SERVO_MAP_PLUGINS = {"kondo_ics": None, "kondo_ics_wheel": "continuous"}

#: Actuator keys a ``servo_map`` owns. A joint that also states one has two
#: sources for it, and the config refuses rather than pick one.
_SERVO_MAP_KEYS = ("id", "direction", "offset_deg")


def joints_with_servo_map(document: Dict[str, Any], path: str
                          ) -> List[Dict[str, Any]]:
    """The ``joints`` list, with ids filled in from ``servo_map`` if it has one.

    A Kondo robot driven by rcb4 already has its servo map, a
    ``joint_name_to_servo_id`` file kept in the robot's description package.
    Copying its ids into this config makes two maps of the same servos, and
    the second one goes stale the first time an id is changed. So a config may
    name that file instead::

        servo_map:
          format: rcb4
          file: package://my_robot_description/config/servo_config.yaml

    and every ``kondo_ics`` / ``kondo_ics_wheel`` joint then takes ``id``,
    ``direction`` and ``offset_deg`` (rcb4's ``offset``, in the same degrees;
    references.md) from it by joint name. The joint list and its order still
    come from this config, because that order is the contract with the URDF's
    ros2_control block.

    Raises
    ------
    ValueError
        A joint states one of the keys the map owns, uses another actuator
        plugin, or disagrees with the map about being a wheel.
    KeyError
        A joint is not in the map.
    """
    joints = _require(document, "joints", path)
    spec = document.get("servo_map")
    if spec is None:
        return joints
    if str(spec.get("format", "")) != "rcb4":
        raise ValueError(
            "{}: servo_map.format must be 'rcb4' (rcb4's joint_name_to_servo_id "
            "map), got {!r}".format(path, spec.get("format")))
    map_path = resolve_data_path(str(_require(spec, "file", "servo_map")),
                                 path, "servo_map.file")
    from actuo.drivers.kondo_import_rcb4 import load_rcb4_entries
    entries = {entry.name: entry for entry in load_rcb4_entries(map_path)}

    out = []
    for index, joint in enumerate(joints):
        where = "{}: joints[{}]".format(path, index)
        name = str(_require(joint, "name", where))
        actuator = dict(_require(joint, "actuator", where))
        plugin = actuator.get("plugin")
        if plugin not in _SERVO_MAP_PLUGINS:
            raise ValueError(
                "{} ({}): actuator plugin {!r} is not one servo_map fills in "
                "({})".format(where, name, plugin, sorted(_SERVO_MAP_PLUGINS)))
        stated = [key for key in _SERVO_MAP_KEYS if key in actuator]
        if stated:
            raise ValueError(
                "{} ({}): {} given here AND by servo_map {}; remove them here, "
                "the map is the one source".format(where, name, stated, map_path))
        if name not in entries:
            raise KeyError(
                "{} ({}): not in servo_map {}".format(where, name, map_path))
        entry = entries[name]
        wanted_type = _SERVO_MAP_PLUGINS[plugin]
        if (entry.servo_type or None) != wanted_type:
            raise ValueError(
                "{} ({}): actuator plugin {!r} needs type {!r} in servo_map, "
                "which says {!r} (servo id {}). A wheel driven as a joint, or a "
                "joint driven as a wheel, does something else entirely.".format(
                    where, name, plugin, wanted_type, entry.servo_type,
                    entry.servo_id))
        actuator.update(id=entry.servo_id, direction=entry.direction,
                        offset_deg=entry.offset)
        filled = dict(joint)
        filled["actuator"] = actuator
        out.append(filled)
    return out


def build_bus(
    path: str, transport_overrides: "Dict[str, Any] | None" = None
) -> Tuple[Bus, List[str]]:
    """Build the bus and its joint order from a robot config file.

    Parameters
    ----------
    path : str
        Robot config YAML.
    transport_overrides : dict, optional
        Keys replacing entries of the ``bus.transport`` block, e.g.
        ``{"interface": "gs_usb", "channel": 0}``. This exists so the *same*
        joint map serves every machine: the identical adapter is ``socketcan``/
        ``can0`` on a Linux robot PC and ``gs_usb``/index ``0`` on a mac, and
        duplicating the joint map per platform would create two sources of
        truth that drift apart. Only the
        transport can be overridden -- calibration must come from the file.
    """
    document = load_document(path)
    bus_spec = dict(_require(document, "bus", path))
    transport_spec = dict(_require(bus_spec, "transport", "bus"))
    transport_spec.update(transport_overrides or {})

    # Actuators are built BEFORE the transport so a DYNAMIXEL bus can take its
    # framing version from the servo model rather than from a second, hand-typed
    # field. Two places stating the protocol is two places to disagree, and the
    # symptom of disagreement is a silent bus.
    actuators, names = build_actuators(joints_with_servo_map(document, path))

    # Which orchestration layer drives this transport. Defaults from the
    # transport family (serial => Feetech request/response bus, otherwise the
    # CAN streaming bus), overridable via ``bus.plugin`` for an explicit choice.
    bus_plugin = bus_spec.get("plugin")
    if bus_plugin is None:
        transport_plugin = transport_spec.get("plugin")
        bus_plugin = {"serial": "serial_bus",
                      "kondo_h7": "kondo_h7_bus",
                      "kondo_rcb4": "kondo_rcb4_bus"}.get(transport_plugin, "can_bus")
        from actuo.drivers.actuators.dynamixel import DynamixelActuator
        if bus_plugin == "serial_bus" and all(
                isinstance(a, DynamixelActuator) for a in actuators):
            bus_plugin = "dynamixel_bus"

    if bus_plugin == "dynamixel_bus" and transport_spec.get("plugin") == "serial":
        protocols = {a.protocol for a in actuators}
        if len(protocols) > 1:
            raise ValueError(
                "{}: this bus mixes DYNAMIXEL protocol versions {}; one serial "
                "line can only be framed as one of them".format(
                    path, sorted(protocols)))
        model_protocol = protocols.pop()
        declared = transport_spec.get("protocol")
        if declared is None:
            transport_spec["protocol"] = model_protocol
        elif int(declared) != model_protocol:
            raise ValueError(
                "{}: bus.transport.protocol is {} but the configured servo "
                "models speak DYNAMIXEL protocol {}. Remove the field and let "
                "the model decide -- a framing that disagrees with the servos "
                "produces a bus that is silent rather than wrong, but it is "
                "still a config that cannot work.".format(
                    path, declared, model_protocol))

    transport = build_transport(transport_spec)
    if bus_plugin == "kondo_h7_bus":
        from actuo.drivers.buses.kondo_h7_bus import DEFAULT_STATE_SLOTS, KondoH7Bus
        from actuo.drivers.kondo_h7_protocol import load_symbol_map
        symbols_path = resolve_data_path(
            str(_require(bus_spec, "symbols", "bus")), path, "bus.symbols")
        bus: Bus = KondoH7Bus(
            transport, actuators, load_symbol_map(symbols_path),
            state_slots=tuple(bus_spec.get("state_slots", DEFAULT_STATE_SLOTS)),
            move_time_ms=bus_spec.get("move_time_ms"),
            read_timeout=float(bus_spec.get("read_timeout", 0.1)),
            joint_names=names,
            battery_voltage_warn_below=bus_spec.get("battery_voltage_warn_below"))
    elif bus_plugin == "kondo_rcb4_bus":
        from actuo.drivers.buses.kondo_rcb4_bus import (
            DEFAULT_ABSENT_TIMEOUT_S, DEFAULT_CONNECT_BITRATES, DEFAULT_NAK_RETRIES,
            DEFAULT_PRESENCE_PERIOD,
            DEFAULT_READING_CONFIRM as RCB4_DEFAULT_READING_CONFIRM,
            DEFAULT_READING_JUMP_RAD as RCB4_DEFAULT_READING_JUMP_RAD,
            DEFAULT_STATE_SLOTS as RCB4_DEFAULT_STATE_SLOTS, KondoRcb4Bus)
        bus = KondoRcb4Bus(
            transport, actuators,
            state_slots=tuple(bus_spec.get("state_slots", RCB4_DEFAULT_STATE_SLOTS)),
            move_time_ms=bus_spec.get("move_time_ms"),
            read_timeout=float(bus_spec.get("read_timeout", 0.1)),
            joint_names=names,
            ics_bitrate=int(bus_spec.get("ics_bitrate", 1250000)),
            connect_bitrates=tuple(
                bus_spec.get("connect_bitrates", DEFAULT_CONNECT_BITRATES)),
            nak_retries=int(bus_spec.get("nak_retries", DEFAULT_NAK_RETRIES)),
            battery_voltage_warn_below=bus_spec.get("battery_voltage_warn_below"),
            hot_plug=bool(bus_spec.get("hot_plug", False)),
            presence_period=int(bus_spec.get("presence_period",
                                             DEFAULT_PRESENCE_PERIOD)),
            absent_timeout_s=float(bus_spec.get("absent_timeout_s",
                                                DEFAULT_ABSENT_TIMEOUT_S)),
            port_detect=bool(bus_spec.get("port_detect", False)),
            reading_jump_rad=bus_spec.get("reading_jump_rad",
                                          RCB4_DEFAULT_READING_JUMP_RAD),
            reading_confirm=int(bus_spec.get("reading_confirm",
                                             RCB4_DEFAULT_READING_CONFIRM)))
    elif bus_plugin == "serial_bus":
        from actuo.drivers.buses.serial_bus import SerialBus
        bus = SerialBus(
            transport, actuators,
            read_timeout=float(bus_spec.get("read_timeout", 0.05)))
    elif bus_plugin == "dynamixel_bus":
        from actuo.drivers.buses.dynamixel_bus import DynamixelBus
        bus = DynamixelBus(
            transport, actuators,
            read_timeout=float(bus_spec.get("read_timeout", 0.05)),
            use_sync_read=bool(bus_spec.get("use_sync_read", True)),
            silence_error_misses=int(bus_spec.get("silence_error_misses", 10)),
            silence_error_age_s=float(bus_spec.get("silence_error_age_s", 1.0)))
    elif bus_plugin == "can_bus":
        from actuo.drivers.buses.can_bus import CanBus
        bus = CanBus(
            transport, actuators,
            stale_after=float(bus_spec.get("stale_after", 1.0)))
    else:
        raise KeyError(
            "unknown bus plugin {!r}; known: can_bus, dynamixel_bus, "
            "kondo_h7_bus, kondo_rcb4_bus, serial_bus"
            .format(bus_plugin))
    return bus, names
