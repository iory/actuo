"""KJS-03 / AIRRELAY sensor board -- a DYNAMIXEL device that is not a motor.

This board sits on the same bus as the servos and impersonates a DYNAMIXEL
device, but it has no shaft: it carries an IMU, four proximity sensors, five
ADC channels, a magnetic encoder, and on the AIRRELAY variant two solenoid
valves driving a vacuum gripper. It is exactly the case the device/plugin design
exists for -- custom hardware joining the stack as a plugin, with zero core
edits, publishing named channels rather than being forced into a fixed struct.

Why this is a ``Sensor`` and not an ``Actuator``
-----------------------------------------------
``Actuator`` promises a position, a set of operating modes and a calibration.
This board has none of those, and pretending otherwise means a joint appears in
the URDF that cannot be commanded and reports a fabricated angle. So it
registers as a sensor for the telemetry path, and the valves -- which are
genuinely commands -- are exposed as COMMAND channels plus
:meth:`encode_valve`, rather than smuggled in as a fake joint.

The rcb4 heritage matters here: the same board on the old stack was reached
through ``ri.control_air_board(38, 'open_relay_valve')``, an ad-hoc method on a
god-object. Here it is a named channel on a plugin, which is the whole point of
the redesign.

Provenance: ``KJS-03/docs/CONTROL_TABLE.md`` in the firmware repository, which
is generated from the firmware's own ``include/control_table.h``. Addresses
below are firmware v2.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from actuo.core.contracts import (
    CapabilityError, ChannelKind, ChannelSpec, Frame, Sensor, Telemetry)
from actuo.core.registry import register_sensor

from actuo.drivers import dynamixel_protocol as dxl

#: Model number the board reports at control-table address 0.
KJS03_MODEL_NUMBER = 0x0A03

#: Model Information (addresses 2..5) carries this magic. Model numbers are
#: ROBOTIS's to assign and this board is not a ROBOTIS product, so matching on
#: the number alone would load a completely wrong control table the day it
#: collides with a real one. Both must agree.
KJS03_MODEL_INFO_MAGIC = 0x3053414B

# --- control table (firmware v2) --------------------------------------------
REG_MODEL_NUMBER = 0
REG_MODEL_INFO = 2
REG_FIRMWARE_VERSION = 6
REG_GYRO_RANGE_DPS = 12
REG_ACCEL_RANGE_G = 14

REG_GYRO_X = 64
REG_ADC_0 = 76
REG_PROXIMITY_0 = 86
REG_MAG_ANGLE = 94
REG_GPIO_STATE = 96
REG_VALVE_STATE = 97
REG_SW_STATE = 98
REG_BOARD_REV = 99
REG_IMU_SELECT = 100
REG_IMU_ENABLED = 101
REG_HW_ERROR_STATUS = 102
REG_PRESSURE_CKPA = 104
REG_SAMPLE_COUNTER = 106

#: One READ covers 64..107: IMU, ADC, proximity, angle, status, pressure and
#: the sample counter. The firmware packs them contiguously precisely so this
#: is one transaction.
SENSOR_BLOCK_ADDR = REG_GYRO_X
SENSOR_BLOCK_LEN = REG_SAMPLE_COUNTER + 2 - REG_GYRO_X  # 44

BOARD_REV_AIRRELAY = 3

#: Valve states the firmware defines. Named rather than numeric at the call
#: site: `1` does not say "this is the one that grips".
VALVE_CLOSED = 0
VALVE_SUCTION = 1
VALVE_RELEASE = 2
VALVE_STATES = {"closed": VALVE_CLOSED, "suction": VALVE_SUCTION,
                "release": VALVE_RELEASE}

#: Verified air control via GPIO_STATE (reg 96), NOT VALVE_STATE (reg 97).
#:
#: Hardware truth, measured on 2026-07-24 on a KJS-03 on a DXHUB bus. reg 96
#: drives the two solenoid/relay outputs directly:
#:   bit 1 (0x02) = PA15 = PUMP  (the vacuum source; 1 = running)
#:   bit 0 (0x01) = PB5  = RELAY (connects the pump/vacuum to the cup)
#: Pressure observed at reg 104 for each combination:
#:   0x03 pump+relay -> -45 kPa   draws vacuum            (GRIP)
#:   0x01 relay only -> holds ~-42 kPa with the pump off  (HOLD; a check valve
#:                      or the stopped pump blocks backflow)
#:   0x00 both off    ->   0 kPa   vents to atmosphere     (RELEASE)
#:   0x02 pump only   ->   0 kPa   running, cup unconnected (PUMP_ONLY)
#:
#: Why this and NOT reg 97 VALVE_STATE: the firmware's VALVE_STATE_RELEASE
#: leaves the relay (PB5) energised, which on this plumbing HOLDS the vacuum
#: (measured: stayed at -42 kPa) instead of venting. Driving GPIO_STATE is the
#: only path that actually releases, so it is the one exposed for real use -- a
#: "release" that does not release is exactly the silent breakage that must
#: never be hidden behind a call that reports success.
GPIO_PUMP_BIT = 0x02
GPIO_RELAY_BIT = 0x01
AIR_OPS = {
    "grip": GPIO_PUMP_BIT | GPIO_RELAY_BIT,     # 0x03 draw vacuum
    "suction": GPIO_PUMP_BIT | GPIO_RELAY_BIT,  # alias of grip
    "hold": GPIO_RELAY_BIT,                     # 0x01 pump off, vacuum held
    "release": 0x00,                            # vent to atmosphere
    "off": 0x00,                                # alias of release
    "pump_only": GPIO_PUMP_BIT,                 # 0x02 pump on, cup open
}

#: Hardware error bits (control table 102).
HW_ERR_BITS = {
    0x01: "imu",
    0x02: "proximity",
    0x04: "magnetic_encoder",
    0x08: "adc",
    0x10: "rx_overrun",
}

#: Error bits that mean something is actually WRONG on an AIRRELAY board.
#:
#: The firmware raises `proximity` and `magnetic_encoder` on every AIRRELAY,
#: because it probes for parts this variant is not fitted with. Measured
#: 2026-07-24: a healthy board reports 0x06 continuously. Reporting that
#: as a fault trains an operator to ignore the fault line, which is worse than
#: not having one -- so the bits are still published as channels (nothing is
#: hidden), and only these are escalated.
AIRRELAY_REAL_ERROR_BITS = 0x01 | 0x08 | 0x10  # imu, adc, rx_overrun

_G = 9.80665


@dataclass(frozen=True)
class Kjs03Ranges:
    """IMU full scale, read from the board rather than assumed.

    The board ships with one of three ICM parts and they do not share a range.
    A host that hardcodes one is silently wrong on the others -- by a factor,
    not by noise -- so the firmware publishes the ranges and this reads them.
    """

    gyro_dps: float
    accel_g: float


@register_sensor("kjs03")
class Kjs03Sensor(Sensor):
    """One KJS-03 (or AIRRELAY) board on a DYNAMIXEL bus.

    Parameters
    ----------
    device_id : int
        DYNAMIXEL bus id.
    ranges : Kjs03Ranges, optional
        IMU full scale. Left ``None`` until :meth:`decode_identity` has read it
        off the board, and until then the IMU channels are reported in raw
        counts only -- never scaled with a guessed range.
    board_rev : int, optional
        Filled in from the board. Decides whether the valve channels exist.
    """

    def __init__(self, device_id: int,
                 ranges: Optional[Kjs03Ranges] = None,
                 board_rev: Optional[int] = None) -> None:
        super().__init__(device_id, calibration=None)
        self.ranges = ranges
        self.board_rev = board_rev

    # --- identity ------------------------------------------------------------
    @property
    def has_valves(self) -> bool:
        return self.board_rev == BOARD_REV_AIRRELAY

    def real_faults(self, error: int) -> List[str]:
        """Names of the error bits that are genuine faults on this board.

        The raw bitmask stays available as the ``hardware_error`` channel; this
        is the shorter question a monitor actually wants answered -- "is this
        board unhealthy?" -- and the answer depends on the variant, which is
        board knowledge and so lives here rather than in whatever publishes it.

        See :data:`AIRRELAY_REAL_ERROR_BITS`: an AIRRELAY permanently flags the
        proximity and magnetic-encoder parts it is not fitted with.
        """
        mask = AIRRELAY_REAL_ERROR_BITS if self.has_valves else 0xFF
        return [name for bit, name in sorted(HW_ERR_BITS.items())
                if error & bit & mask]

    def encode_identity_query(self) -> List[Frame]:
        """READ the EEPROM block that identifies the board and its IMU ranges."""
        payload = dxl.build_read(dxl.PROTOCOL_2, self.id, REG_MODEL_NUMBER, 16)
        return [Frame(payload=payload, address=self.id)]

    def decode_identity(self, frame: Frame) -> Dict[str, object]:
        """Parse the identity block, refusing anything that is not this board.

        The magic check is the point. Without it a model-number collision means
        every address below is read against a different device's control table,
        and every value that comes back is plausible.
        """
        parsed = dxl.parse_status(dxl.PROTOCOL_2, frame.payload)
        if parsed is None or parsed[0] != self.id:
            raise ValueError("id {}: not this device's status".format(self.id))
        data = parsed[2]
        if len(data) < 16:
            raise ValueError("id {}: short identity block".format(self.id))

        model = dxl.from_le(data[0:2])
        magic = dxl.from_le(data[2:6])
        if model != KJS03_MODEL_NUMBER or magic != KJS03_MODEL_INFO_MAGIC:
            raise CapabilityError(
                "id {} reports model {} / info {:#010x}; expected KJS-03 "
                "({} / {:#010x}). Refusing to read a control table that may "
                "belong to a different device.".format(
                    self.id, model, magic, KJS03_MODEL_NUMBER,
                    KJS03_MODEL_INFO_MAGIC))

        self.ranges = Kjs03Ranges(gyro_dps=float(dxl.from_le(data[12:14])),
                                  accel_g=float(dxl.from_le(data[14:16])))
        return {"model_number": model,
                "firmware_version": data[6],
                "gyro_range_dps": self.ranges.gyro_dps,
                "accel_range_g": self.ranges.accel_g}

    # --- channels ------------------------------------------------------------
    @property
    def channels(self) -> Tuple[ChannelSpec, ...]:
        specs = [
            ChannelSpec("imu/angular_velocity.x", "rad/s"),
            ChannelSpec("imu/angular_velocity.y", "rad/s"),
            ChannelSpec("imu/angular_velocity.z", "rad/s"),
            ChannelSpec("imu/linear_acceleration.x", "m/s^2"),
            ChannelSpec("imu/linear_acceleration.y", "m/s^2"),
            ChannelSpec("imu/linear_acceleration.z", "m/s^2"),
            ChannelSpec("magnetic_encoder/angle", "rad"),
            ChannelSpec("switch", "bool"),
            ChannelSpec("sample_counter", "count"),
            ChannelSpec("hardware_error", "bitmask"),
        ]
        specs.extend(ChannelSpec("adc/{}".format(i), "count")
                     for i in range(5))
        specs.extend(ChannelSpec("proximity/{}".format(i), "count")
                     for i in range(4))
        if self.has_valves:
            specs.append(ChannelSpec("air/pressure", "kPa"))
            specs.append(ChannelSpec("air/gpio_state", "bitmask"))
            specs.append(ChannelSpec("air/valve_state", "enum"))
            # The command the driver actually drives the gripper with -- the
            # verified GPIO_STATE path (grip|hold|release|pump_only), NOT the
            # firmware VALVE_STATE whose release does not vent (see AIR_OPS).
            specs.append(ChannelSpec(
                "air/command", "enum", ChannelKind.COMMAND,
                description="grip|hold|release|pump_only (writes GPIO_STATE)"))
        return tuple(specs)

    # --- reads ---------------------------------------------------------------
    def encode_query(self) -> List[Frame]:
        """One READ for every channel this board reports."""
        payload = dxl.build_read(dxl.PROTOCOL_2, self.id,
                                 SENSOR_BLOCK_ADDR, SENSOR_BLOCK_LEN)
        return [Frame(payload=payload, address=self.id)]

    def matches(self, frame: Frame) -> bool:
        parsed = dxl.parse_status(dxl.PROTOCOL_2, frame.payload)
        return parsed is not None and parsed[0] == self.id

    def decode_state(self, frame: Frame) -> Telemetry:
        parsed = dxl.parse_status(dxl.PROTOCOL_2, frame.payload)
        if parsed is None or parsed[0] != self.id:
            raise ValueError("id {}: not this device's status".format(self.id))
        return self.decode_block(parsed[2])

    def decode_block(self, data: bytes) -> Telemetry:
        """Decode the 44-byte sensor block into named channels."""
        if len(data) < SENSOR_BLOCK_LEN:
            raise ValueError("id {}: short sensor block ({} of {} bytes)".format(
                self.id, len(data), SENSOR_BLOCK_LEN))

        def at(register: int) -> int:
            return register - SENSOR_BLOCK_ADDR

        def s16(register: int) -> int:
            return dxl.from_le(data[at(register):at(register) + 2], signed=True)

        def u16(register: int) -> int:
            return dxl.from_le(data[at(register):at(register) + 2])

        out: Telemetry = {}

        gyro_raw = [s16(REG_GYRO_X + 2 * i) for i in range(3)]
        acc_raw = [s16(REG_GYRO_X + 6 + 2 * i) for i in range(3)]
        for i, axis in enumerate("xyz"):
            out["imu/angular_velocity_raw.{}".format(axis)] = float(gyro_raw[i])
            out["imu/linear_acceleration_raw.{}".format(axis)] = float(acc_raw[i])

        if self.ranges is not None:
            # int16 full scale maps to +/- the published range.
            gyro_scale = math.radians(self.ranges.gyro_dps) / 32768.0
            acc_scale = self.ranges.accel_g * _G / 32768.0
            for i, axis in enumerate("xyz"):
                out["imu/angular_velocity.{}".format(axis)] = gyro_raw[i] * gyro_scale
                out["imu/linear_acceleration.{}".format(axis)] = acc_raw[i] * acc_scale
        # else: raw only. Scaling with a guessed range would be wrong by a
        # factor of up to 8 between the supported IMUs, and would look fine.

        for i in range(5):
            out["adc/{}".format(i)] = float(u16(REG_ADC_0 + 2 * i))
        for i in range(4):
            out["proximity/{}".format(i)] = float(u16(REG_PROXIMITY_0 + 2 * i))

        # 14-bit absolute encoder over one turn.
        out["magnetic_encoder/angle"] = u16(REG_MAG_ANGLE) * 2.0 * math.pi / 16384.0

        out["switch"] = float(data[at(REG_SW_STATE)])
        # Learn the board revision from the board itself, so has_valves (and the
        # air channels / encode_air) reflect what is actually plugged in rather
        # than what the constructor was told. Board rev lives at reg 99, inside
        # the sensor block but not the 16-byte identity block, so this is the
        # first read that can know it.
        self.board_rev = data[at(REG_BOARD_REV)]
        out["board_revision"] = float(data[at(REG_BOARD_REV)])
        out["imu/select"] = float(data[at(REG_IMU_SELECT)])
        out["imu/enabled"] = float(data[at(REG_IMU_ENABLED)])
        out["sample_counter"] = float(u16(REG_SAMPLE_COUNTER))

        error = data[at(REG_HW_ERROR_STATUS)]
        out["hardware_error"] = float(error)
        for bit, name in HW_ERR_BITS.items():
            out["hardware_error/{}".format(name)] = 1.0 if error & bit else 0.0

        if data[at(REG_BOARD_REV)] == BOARD_REV_AIRRELAY:
            out["air/pressure"] = s16(REG_PRESSURE_CKPA) / 100.0  # centi-kPa
            out["air/valve_state"] = float(data[at(REG_VALVE_STATE)])
            out["air/gpio_state"] = float(data[at(REG_GPIO_STATE)])

        return out

    # --- air / gripper command -----------------------------------------------
    def encode_air(self, op) -> Frame:
        """Frame that drives the vacuum gripper -- the verified control path.

        ``op`` is one of ``grip``/``suction`` / ``hold`` / ``release``/``off``
        / ``pump_only`` (see :data:`AIR_OPS`), or a raw ``GPIO_STATE`` byte.

        This writes ``GPIO_STATE`` (reg 96) directly, which is what was checked
        on hardware: ``grip`` reaches -45 kPa, ``release`` returns to 0 kPa.
        Prefer it over :meth:`encode_valve` -- the firmware ``VALVE_STATE``
        ``release`` keeps the relay energised and holds the vacuum on this
        plumbing instead of venting, so it cannot actually let a workpiece go.
        """
        if not self.has_valves:
            raise CapabilityError(
                "id {}: this board is a KJS-03 (rev {}), not an AIRRELAY; it "
                "has no gripper".format(self.id, self.board_rev))
        if isinstance(op, str):
            if op not in AIR_OPS:
                raise ValueError("unknown air op {!r}; known: {}".format(
                    op, sorted(AIR_OPS)))
            value = AIR_OPS[op]
        else:
            value = int(op) & 0xFF
        payload = dxl.build_write(dxl.PROTOCOL_2, self.id, REG_GPIO_STATE,
                                  bytes([value]))
        return Frame(payload=payload, address=self.id)

    # --- valve command (firmware VALVE_STATE; see caveat) --------------------
    def encode_valve(self, state) -> Frame:
        """Frame that sets the firmware ``VALVE_STATE`` (reg 97).

        Accepts the firmware's numeric state or one of ``closed`` / ``suction``
        / ``release``.

        CAVEAT, verified on hardware: ``release`` here does NOT vent the cup
        with the pump/relay/check-valve plumbing described at module level --
        the firmware leaves the relay energised and the vacuum is held at
        ~-42 kPa. Use :meth:`encode_air` (``release``) to
        actually let go. This method is kept only for the firmware-native
        states; ``suction`` and ``closed`` behave as named.
        """
        if not self.has_valves:
            raise CapabilityError(
                "id {}: this board is a KJS-03 (rev {}), not an AIRRELAY; it "
                "has no valves".format(self.id, self.board_rev))
        if isinstance(state, str):
            if state not in VALVE_STATES:
                raise ValueError("unknown valve state {!r}; known: {}".format(
                    state, sorted(VALVE_STATES)))
            value = VALVE_STATES[state]
        else:
            value = int(state)
            if value not in VALVE_STATES.values():
                raise ValueError(
                    "valve state {} is not one of {}".format(
                        value, sorted(VALVE_STATES.values())))
        payload = dxl.build_write(dxl.PROTOCOL_2, self.id, REG_VALVE_STATE,
                                  bytes([value]))
        return Frame(payload=payload, address=self.id)
