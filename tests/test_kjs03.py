"""The KJS-03 sensor board plugin.

The board is not a motor, so what is worth pinning is different from an
actuator: that it refuses to decode a device it has not positively identified,
that it does not scale the IMU with a range it has not read, and that the
valve command -- the one output here that can drop a suspended workpiece --
cannot be issued to a board that has no valves.
"""

import math

import pytest

from actuo.core.contracts import CapabilityError, ChannelKind, Frame
from actuo.core.registry import get_sensor

from actuo.drivers import dynamixel_protocol as dxl
from actuo.drivers.sensors.kjs03 import (
    AIR_OPS, BOARD_REV_AIRRELAY, KJS03_MODEL_INFO_MAGIC, KJS03_MODEL_NUMBER,
    Kjs03Ranges, Kjs03Sensor, SENSOR_BLOCK_LEN)

DEVICE_ID = 19


def status(device_id, params):
    return Frame(payload=dxl.build_packet(dxl.PROTOCOL_2, device_id, dxl.STATUS,
                                          b"\x00" + params),
                 address=device_id)


def identity_block(model=KJS03_MODEL_NUMBER, magic=KJS03_MODEL_INFO_MAGIC):
    return (dxl.to_le(model, 2) + dxl.to_le(magic, 4) + bytes([2, DEVICE_ID, 3, 0])
            + bytes([2, 0]) + dxl.to_le(2000, 2) + dxl.to_le(16, 2))


def sensor_block(gyro=(0, 0, 0), acc=(0, 0, 0), pressure_ckpa=0,
                 board_rev=BOARD_REV_AIRRELAY, error=0, counter=0,
                 valve=0, mag=0):
    block = b""
    for value in gyro:
        block += dxl.to_le(value, 2)
    for value in acc:
        block += dxl.to_le(value, 2)
    block += b"".join(dxl.to_le(1000 + i, 2) for i in range(5))   # adc
    block += b"".join(dxl.to_le(2000 + i, 2) for i in range(4))   # proximity
    block += dxl.to_le(mag, 2)                                    # mag angle
    block += bytes([0, valve, 1, board_rev, 0, 1, error, 0])      # 96..103
    block += dxl.to_le(pressure_ckpa, 2)                          # 104
    block += dxl.to_le(counter, 2)                                # 106
    assert len(block) == SENSOR_BLOCK_LEN
    return block


def test_the_plugin_is_registered():
    assert get_sensor("kjs03") is Kjs03Sensor


# --- identity ----------------------------------------------------------------
def test_identity_reads_the_imu_ranges_off_the_board():
    board = Kjs03Sensor(DEVICE_ID)
    info = board.decode_identity(status(DEVICE_ID, identity_block()))
    assert info["model_number"] == KJS03_MODEL_NUMBER
    assert board.ranges == Kjs03Ranges(gyro_dps=2000.0, accel_g=16.0)


def test_a_model_number_collision_is_refused():
    """The magic exists because model numbers are ROBOTIS's to assign. Matching
    on the number alone would read every later address against some other
    device's control table, and every value would look plausible."""
    board = Kjs03Sensor(DEVICE_ID)
    with pytest.raises(CapabilityError, match="Refusing"):
        board.decode_identity(status(DEVICE_ID, identity_block(magic=0xDEADBEEF)))


# --- IMU scaling -------------------------------------------------------------
def test_imu_is_not_scaled_until_the_range_is_known():
    """This board carries one of three IMUs whose ranges differ by up to 8x.
    Guessing one would be wrong by a factor and would look entirely normal."""
    board = Kjs03Sensor(DEVICE_ID)
    out = board.decode_block(sensor_block(gyro=(16384, 0, 0)))
    assert out["imu/angular_velocity_raw.x"] == 16384.0
    assert "imu/angular_velocity.x" not in out


def test_imu_scales_with_the_published_range():
    board = Kjs03Sensor(DEVICE_ID, ranges=Kjs03Ranges(2000.0, 16.0))
    out = board.decode_block(sensor_block(gyro=(16384, 0, 0), acc=(0, 0, 16384)))
    assert out["imu/angular_velocity.x"] == pytest.approx(math.radians(1000.0), rel=1e-3)
    assert out["imu/linear_acceleration.z"] == pytest.approx(8.0 * 9.80665, rel=1e-3)


# --- channels ----------------------------------------------------------------
def test_pressure_arrives_in_kpa_already_converted():
    """Converted on the board: the conversion needs the amplifier's offset and
    gain, which a host can only know by copying the schematic."""
    board = Kjs03Sensor(DEVICE_ID, board_rev=BOARD_REV_AIRRELAY)
    out = board.decode_block(sensor_block(pressure_ckpa=-8055))
    assert out["air/pressure"] == pytest.approx(-80.55)


def test_hardware_error_bits_are_named():
    board = Kjs03Sensor(DEVICE_ID)
    out = board.decode_block(sensor_block(error=0x01 | 0x10))
    assert out["hardware_error/imu"] == 1.0
    assert out["hardware_error/rx_overrun"] == 1.0
    assert out["hardware_error/adc"] == 0.0


def test_a_kjs03_without_valves_has_no_air_channels():
    board = Kjs03Sensor(DEVICE_ID, board_rev=2)
    names = {(c.name, c.kind) for c in board.channels}
    assert ("air/valve_state", ChannelKind.COMMAND) not in names


def test_an_airrelay_declares_the_air_command_channel():
    board = Kjs03Sensor(DEVICE_ID, board_rev=BOARD_REV_AIRRELAY)
    names = {(c.name, c.kind) for c in board.channels}
    # The command channel drives the verified GPIO_STATE path (grip/release),
    # not the firmware VALVE_STATE whose release does not vent.
    assert ("air/command", ChannelKind.COMMAND) in names
    assert ("air/pressure", ChannelKind.STATE) in names
    assert ("air/valve_state", ChannelKind.STATE) in names


# --- valve command -----------------------------------------------------------
def test_valve_command_targets_the_control_table_register():
    board = Kjs03Sensor(DEVICE_ID, board_rev=BOARD_REV_AIRRELAY)
    frame = board.encode_valve("suction")
    body = dxl.remove_stuffing(frame.payload[7:-2])
    assert body[0] == dxl.WRITE
    assert dxl.from_le(body[1:3]) == 97      # CTRL_VALVE_STATE
    assert body[3] == 1                       # VALVE_SUCTION


def test_valve_command_is_refused_on_a_board_without_valves():
    board = Kjs03Sensor(DEVICE_ID, board_rev=2)
    with pytest.raises(CapabilityError, match="no valves"):
        board.encode_valve("suction")


def test_an_unknown_valve_state_is_refused():
    board = Kjs03Sensor(DEVICE_ID, board_rev=BOARD_REV_AIRRELAY)
    with pytest.raises(ValueError):
        board.encode_valve("open_everything")
    with pytest.raises(ValueError):
        board.encode_valve(7)


# --- air / gripper command (verified GPIO_STATE path) ------------------------
def _air_write(board, op):
    """Decode an encode_air frame into (register, value)."""
    body = dxl.remove_stuffing(board.encode_air(op).payload[7:-2])
    assert body[0] == dxl.WRITE
    return dxl.from_le(body[1:3]), body[3]


def test_air_command_writes_gpio_state_not_valve_state():
    board = Kjs03Sensor(DEVICE_ID, board_rev=BOARD_REV_AIRRELAY)
    reg, value = _air_write(board, "grip")
    assert reg == 96                      # CTRL_GPIO_STATE, NOT 97 VALVE_STATE
    assert value == 0x03                  # pump (0x02) + relay (0x01)


def test_air_release_drops_the_relay_so_it_actually_vents():
    # The whole reason air control uses GPIO_STATE: release must clear the relay
    # bit. The firmware VALVE_STATE "release" leaves it set and holds vacuum.
    board = Kjs03Sensor(DEVICE_ID, board_rev=BOARD_REV_AIRRELAY)
    _reg, value = _air_write(board, "release")
    assert value == 0x00                  # both off -> vent
    assert AIR_OPS["hold"] == 0x01        # relay only -> vacuum held, pump off
    assert AIR_OPS["pump_only"] == 0x02   # pump only -> running, cup open


def test_air_command_is_refused_on_a_board_without_a_gripper():
    board = Kjs03Sensor(DEVICE_ID, board_rev=2)
    with pytest.raises(CapabilityError, match="no gripper"):
        board.encode_air("grip")


def test_an_unknown_air_op_is_refused():
    board = Kjs03Sensor(DEVICE_ID, board_rev=BOARD_REV_AIRRELAY)
    with pytest.raises(ValueError):
        board.encode_air("suck_hard")


# --- framing -----------------------------------------------------------------
def test_query_is_one_read_covering_every_channel():
    board = Kjs03Sensor(DEVICE_ID)
    frames = board.encode_query()
    assert len(frames) == 1
    body = dxl.remove_stuffing(frames[0].payload[7:-2])
    assert body[0] == dxl.READ
    assert dxl.from_le(body[1:3]) == 64
    assert dxl.from_le(body[3:5]) == SENSOR_BLOCK_LEN


def test_another_devices_status_is_not_decoded():
    board = Kjs03Sensor(DEVICE_ID)
    assert not board.matches(status(DEVICE_ID + 1, sensor_block()))
    with pytest.raises(ValueError):
        board.decode_state(status(DEVICE_ID + 1, sensor_block()))


def test_a_short_block_raises_rather_than_decoding_garbage():
    board = Kjs03Sensor(DEVICE_ID)
    with pytest.raises(ValueError):
        board.decode_block(bytes(10))
