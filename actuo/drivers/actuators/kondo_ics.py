"""Kondo ICS servo (KRS series) driven through an ARM-H7 gateway board.

This model owns everything unit-shaped about an ICS servo -- pulse<->radian,
the current byte's sign convention, the temperature table, its motion envelope
-- and nothing about how bytes reach it. It emits packet *bodies* built by
``actuo.drivers.kondo_h7_protocol``; the transport frames them, and the bus
merges them into one batch.

What an ICS servo can and cannot do
-----------------------------------
The wire has exactly one control quantity: a target pulse. There is no velocity
mode, no current mode, no PWM mode. Accordingly:

* ``capabilities.modes`` is ``(POSITION,)`` and nothing else. Asking for
  VELOCITY fails loudly instead of being emulated.
* Velocity is **not sensed**. ``ActuatorState.velocity`` stays ``None`` -- it is
  never differentiated behind the caller's back: an estimate reported as a
  sensed value is a fabrication the caller cannot detect.
* Current and temperature *are* sensed, but only while the board's read flags
  are on (the bus owns those flags); an unpolled byte decodes to ``None``, not 0.

Servo on/off has no packet of its own: the board reads two sentinel values out
of the pulse field (:data:`SERVO_ON_PULSE` / :data:`SERVO_OFF_PULSE`). That is
why :meth:`KondoIcsActuator.encode_release` is a normal move packet carrying a
sentinel rather than a distinct command.

Provenance: ``actuo/drivers/references.md``, section "Kondo ARM-H7".
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from actuo.core.contracts import (
    Actuator, ActuatorCommand, ActuatorState, Calibration, Capabilities,
    CapabilityError, ChannelKind, ChannelSpec, Frame, OperatingMode)
from actuo.core.registry import register_actuator

from actuo.drivers.actuators.kondo_ics_temperature import setting_value_to_celsius
from actuo.drivers.kondo_h7_protocol import build_multi_servo

#: Pulse counts per degree of output shaft: the servo spans 4000 counts over
#: 135 degrees either side of neutral.
PULSE_PER_DEG = 4000.0 / 135.0
#: Pulse counts per full output revolution -- the form ``Calibration`` wants.
PULSE_PER_REV = 360.0 * PULSE_PER_DEG
#: Pulse at the servo's mechanical neutral.
CENTER_PULSE = 7500
#: Mechanical envelope of a standard KRS servo: neutral +/- 135 degrees.
PULSE_MIN = 3500
PULSE_MAX = 11500

#: Sentinels the firmware reads out of the pulse field instead of a target.
SERVO_ON_PULSE = 32767
SERVO_OFF_PULSE = 32768

#: Default move duration byte. The board interpolates to the target over
#: ``velocity * FRAME_SCALE`` ms, so this is a *duration*, not a speed, and each
#: new packet restarts the interpolation.
DEFAULT_VELOCITY = 127

#: Raw servo slots this model can decode, in the order the bus packs them.
DECODABLE_SLOTS = ("current_angle", "current", "temperature", "error_angle",
                   "ref_angle", "move_state", "flag")


@dataclass
class KondoIcsCalibration(Calibration):
    """Per-servo calibration in ICS pulse units.

    ``counts_per_rev`` and ``zero_offset_counts`` come from
    :class:`~actuo.core.contracts.Calibration`; the defaults here are the KRS
    servo's, and every field stays per-servo (each motor carries its own model;
    no global constants) -- there is no global 7500 anywhere else in this stack.

    Attributes
    ----------
    pulse_limit : tuple of int
        Hard mechanical envelope in raw pulses. A command outside it is refused,
        not clamped: clamping would quietly deliver a different pose than the
        controller asked for.
    """

    counts_per_rev: float = PULSE_PER_REV
    zero_offset_counts: float = float(CENTER_PULSE)
    pulse_limit: Tuple[int, int] = (PULSE_MIN, PULSE_MAX)


class KondoBatchServo(Actuator):
    """What :class:`~actuo.drivers.buses.kondo_h7_bus.KondoH7Bus` needs of any
    servo it puts in a packet.

    One board packet carries a bitmap of servo ids, one 16-bit field each, and
    one move-duration byte (or one per servo). What that 16-bit field MEANS is
    the servo's business: for a position servo it is a target pulse, and for one
    with ``ServoStruct.rotation`` set it is a speed. The bus must not know the
    difference -- an arm joint and a wheel hang off the same board and belong in
    the SAME packet, and a bus that asked "is this a wheel?" would be exactly
    the per-type ``switch`` in shared code that this design rules out.

    So the bus asks each servo for its own two numbers and packs them.
    """

    def batch_pulse(self, command: ActuatorCommand) -> int:
        """The 16-bit wire value for ``command``, in whatever this servo's
        field means. Raises rather than clamps."""
        raise NotImplementedError

    def move_byte(self, bus_default: Optional[int]) -> int:
        """Move-duration byte for this servo.

        ``bus_default`` is what the bus's ``move_time_ms`` works out to, or
        ``None`` when the bus has none and the servo's own default applies.
        """
        raise NotImplementedError

    #: What ``ServoStruct.rotation`` must read for this model to be the right
    #: one. Checked at startup, because the two models write the same 16-bit
    #: field and disagree about what it means -- and getting it backwards is not
    #: a wrong angle, it is a servo that spins up and never stops.
    expects_rotation = False


@dataclass(frozen=True)
class HoldRelax:
    """When to hold a servo lighter: it is held away from its command.

    A servo that cannot reach its command -- a gripper closed on an object --
    pushes with all its stretch for as long as the command stands, and heats.
    When the command has not changed for ``seconds`` and the servo is still
    more than ``error_rad`` from it, the bus sends instead the measured angle
    plus ``squeeze_rad`` toward the command, so it keeps pressing, lightly. A
    new command ends it.

    Parameters
    ----------
    error_rad : float
        Distance from the command that counts as not getting there.
    seconds : float
        How long the command must stand, with the servo that far from it.
    squeeze_rad : float
        How far past the measured angle, toward the command, it is then held.
    """

    error_rad: float
    seconds: float
    squeeze_rad: float

    def __post_init__(self) -> None:
        if not (self.error_rad > 0 and self.seconds > 0 and self.squeeze_rad >= 0):
            raise ValueError(
                "hold_relax needs error > 0, seconds > 0 and squeeze >= 0, "
                "got {}".format(self))
        if self.squeeze_rad >= self.error_rad:
            raise ValueError(
                "hold_relax squeeze ({:.4f} rad) must be less than its error "
                "({:.4f} rad), or it holds harder than the command".format(
                    self.squeeze_rad, self.error_rad))


@register_actuator("kondo_ics")
class KondoIcsActuator(KondoBatchServo):
    """One ICS servo addressed by its flat id on the H7 board's servo vector.

    Parameters
    ----------
    actuator_id : int
        The servo's id, which is also its index in the board's ``servo_vector``.
    calibration : KondoIcsCalibration
        Pulse units, direction, offset and limits for this servo.
    velocity : int
        Default move-duration byte used when a command carries no timing.
    ics_channel : str, optional
        Which ICS UART (``"J1"``..``"J6"``) the servo hangs off. Documentation
        only: the board addresses servos by flat id, so this never reaches the
        wire. Recorded because it is what a human needs when a servo goes quiet.
    stretch : int, optional
        Holding gain (1..127) to set on the servo at connect. ``None`` leaves
        the servo's own. Only the RCB-4mini bus sets it.
    hold_relax : HoldRelax, optional
        Hold the servo lighter when it is held away from its command. Only the
        RCB-4mini bus does it.

    Notes
    -----
    ``calibration.torque_constant`` (N*m per amp) is optional. Without it,
    ``ActuatorState.current`` stays ``None`` and the measured amps ride in
    ``aux['current_a']`` -- a torque figure is not invented from a guessed
    constant.
    """

    def __init__(self, actuator_id: int, calibration: KondoIcsCalibration,
                 velocity: int = DEFAULT_VELOCITY,
                 ics_channel: Optional[str] = None,
                 stretch: Optional[int] = None,
                 hold_relax: Optional[HoldRelax] = None) -> None:
        super().__init__(int(actuator_id), calibration)
        self.velocity = int(velocity)
        self.ics_channel = ics_channel
        if stretch is not None and not 1 <= int(stretch) <= 127:
            raise ValueError("stretch {} outside 1..127".format(stretch))
        self.stretch = None if stretch is None else int(stretch)
        self.hold_relax = hold_relax

    # -- declaration -------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            modes=(OperatingMode.POSITION,),
            has_position_sense=True,
            has_velocity_sense=False,
            has_current_sense=True,
            has_temperature_sense=True,
            multiturn=False,
        )

    @property
    def channels(self) -> Tuple[ChannelSpec, ...]:
        return (
            ChannelSpec("position", "rad", ChannelKind.COMMAND),
            ChannelSpec("position", "rad", ChannelKind.STATE),
            ChannelSpec("current", "A", ChannelKind.STATE,
                        "servo current; signed by direction of load"),
            ChannelSpec("temperature", "degC", ChannelKind.STATE),
            ChannelSpec("error_angle", "rad", ChannelKind.STATE,
                        "commanded minus measured, as the board computes it"),
        )

    # -- unit conversion ---------------------------------------------------

    def pulse_to_rad(self, pulse: float) -> float:
        return self.calibration.counts_to_rad(pulse)

    def rad_to_pulse(self, rad: float) -> float:
        return self.calibration.rad_to_counts(rad)

    def position_envelope(self) -> Tuple[float, float]:
        """The angles :meth:`goal_pulse` accepts, rad: the mechanical pulse
        envelope, narrowed by the taught limit when there is one.

        Exposed so the torque layer judges "outside" against the same range
        the wire encoder refuses, not only the taught one -- a KRS servo has
        no taught limit by default, and one resting a few pulses past 3500 was
        stranded: refused torque because its pose could not be commanded, and
        no way to command it back (2026-10-17, j0 at pulse 3496).
        """
        ends = sorted(self.pulse_to_rad(p) for p in self.calibration.pulse_limit)
        limit = self.calibration.position_limit
        if limit is not None:
            ends = [max(ends[0], limit[0]), min(ends[1], limit[1])]
        return ends[0], ends[1]

    def position_refusal(self, position: float) -> Optional[str]:
        """Why this angle cannot be commanded, or None. See the base contract.

        Two envelopes, and the caller gets a reason rather than an exception so
        the torque layer can REFUSE instead of crashing. Until this existed,
        enabling a joint parked outside its taught limit passed every guard --
        the jump check compares commanded against measured, and they agree
        exactly when the controller has adopted the measured pose. One cycle
        later ``goal_pulse`` raised inside ``service()`` and took the whole
        servicer down, telemetry for every other joint with it. Every other
        guard on this bus declines with a reason; this one aborted the process.
        """
        limit = self.calibration.position_limit
        if limit is not None and not limit[0] <= position <= limit[1]:
            return ("servo {}: commanded {:.4f} rad outside its taught limit "
                    "[{:.4f}, {:.4f}] rad".format(
                        self.id, position, limit[0], limit[1]))
        pulse = int(round(self.rad_to_pulse(position)))
        low, high = self.calibration.pulse_limit
        if not low <= pulse <= high:
            return ("servo {}: commanded {:.4f} rad = pulse {} outside the "
                    "servo's mechanical envelope [{}, {}]".format(
                        self.id, position, pulse, low, high))
        return None

    def move_byte(self, bus_default: Optional[int]) -> int:
        """The bus's tuned interpolation time, or this servo's own default."""
        return self.velocity if bus_default is None else bus_default

    def batch_pulse(self, command: ActuatorCommand) -> int:
        """Target pulse for a position command; see :class:`KondoBatchServo`."""
        self.capabilities.require(command.mode)
        if command.position is None:
            raise ValueError("POSITION mode needs command.position (rad)")
        return self.goal_pulse(command.position)

    def goal_pulse(self, position: float) -> int:
        """Convert a commanded angle to a pulse, refusing anything unsafe.

        Both envelopes are checked, by :meth:`position_refusal` -- one
        implementation behind two entry points, so what the torque layer refuses
        before enabling and what the wire encoder rejects mid-cycle can never
        drift apart. Neither is clamped: a clamped command silently becomes a
        different command, a silent degradation this stack never allows.
        """
        refusal = self.position_refusal(position)
        if refusal is not None:
            raise ValueError(refusal)
        return int(round(self.rad_to_pulse(position)))

    # -- encoding ----------------------------------------------------------

    def encode_command(self, command: ActuatorCommand) -> Frame:
        """Build a one-servo move packet.

        A batch is normally assembled by :class:`KondoH7Bus` -- this exists for
        single-joint tools (jogging, probing). It uses the same multi-servo
        opcode with a one-servo bitmap, because that is the form validated
        against the board; the board's single-servo opcode is untested here and
        so is not used.
        """
        self.capabilities.require(command.mode)
        if command.position is None:
            raise ValueError("POSITION mode needs command.position (rad)")
        pulse = self.goal_pulse(command.position)
        return Frame(payload=build_multi_servo([self.id], [pulse], self.velocity),
                     address=self.id)

    def encode_release(self) -> Frame:
        """Servo OFF: hand the board the off sentinel so the joint goes limp.

        Note what this does *not* do: stop sending commands. The board holds its
        last target forever, so silence is not release -- the sentinel is.
        """
        return Frame(
            payload=build_multi_servo([self.id], [SERVO_OFF_PULSE], self.velocity),
            address=self.id)

    def encode_hold(self) -> Frame:
        """Servo ON at wherever the joint currently is.

        The board reads the measured pulse into its reference when it sees this
        sentinel, so enabling cannot itself produce motion. Deciding *whether*
        enabling is safe is the servicer's guarded transition, not this layer's.
        """
        return Frame(
            payload=build_multi_servo([self.id], [SERVO_ON_PULSE], self.velocity),
            address=self.id)

    def encode_mode(self, mode: OperatingMode) -> List[Frame]:
        """No wire mode switch exists -- the servo is position-only.

        The capability check still runs, so requesting VELOCITY raises rather
        than quietly doing nothing and leaving the caller in position mode.
        """
        self.capabilities.require(mode)
        return []

    # -- decoding ----------------------------------------------------------

    def matches(self, frame: Frame) -> bool:
        """Whether ``frame`` is this servo's slice of a board state read.

        The board's replies are not addressed -- one reply carries every servo.
        The bus slices that reply and stamps each piece with a servo id and a
        ``kind``. Both are checked: an id alone is not identity (on CAN, a CubeMars
        driver once decoded its own echoed command frame, which carries the
        same id, as a status reporting a fault), so a frame that is not a state
        slice is refused even if its address matches.
        """
        return (frame.meta.get("kind") == "servo_state"
                and frame.address == self.id)

    def decode_state(self, frame: Frame) -> ActuatorState:
        """Decode one servo's state slice into engineering units.

        The slice is the raw slot values the bus read this cycle, packed as
        little-endian int32 in the order named by ``frame.meta['slots']``. The
        bus owns the board's struct layout and signedness; this method owns what
        the numbers *mean*. Slots the bus did not read this cycle are simply
        absent, and the corresponding state fields stay ``None``.
        """
        slots = frame.meta.get("slots")
        if not slots:
            raise ValueError(
                "servo state frame carries no 'slots' metadata; the bus must "
                "name the slots it packed")
        values = self._unpack(frame.payload, len(slots))
        raw: Dict[str, int] = dict(zip(slots, values))

        unknown = [name for name in raw if name not in DECODABLE_SLOTS]
        if unknown:
            raise ValueError(
                "servo {}: cannot decode slot(s) {}; known: {}".format(
                    self.id, unknown, list(DECODABLE_SLOTS)))

        state = ActuatorState()
        aux: Dict[str, float] = {}

        if "current_angle" in raw:
            pulse = raw["current_angle"]
            aux["pulse"] = float(pulse)
            # 0 means the firmware has not heard from this servo yet; the
            # neutral-looking angle it would decode to is not a measurement.
            if pulse != 0:
                state.position = self.pulse_to_rad(pulse)

        if "ref_angle" in raw:
            reference = raw["ref_angle"]
            aux["ref_pulse"] = float(reference)
            # The reference field doubles as the servo on/off state.
            aux["servo_on"] = float(reference != SERVO_OFF_PULSE)

        if "current" in raw:
            amps = self._decode_current(raw["current"])
            if amps is not None:
                aux["current_a"] = amps
                if self.calibration.torque_constant is not None:
                    state.current = amps * self.calibration.torque_constant

        if "temperature" in raw:
            state.temperature = setting_value_to_celsius(raw["temperature"])

        if "error_angle" in raw:
            # A pulse difference, so it converts with the scale but not the
            # offset -- running it through counts_to_rad would subtract neutral.
            aux["error_angle_rad"] = (
                self.calibration.direction * raw["error_angle"]
                * (2.0 * math.pi / self.calibration.counts_per_rev)
                / self.calibration.gear_ratio)

        if "move_state" in raw:
            aux["moving"] = float(raw["move_state"])
        if "flag" in raw:
            aux["present"] = float(raw["flag"])

        state.aux = aux
        return state

    @staticmethod
    def _decode_current(raw: int) -> Optional[float]:
        """Servo current byte -> amps, or ``None`` when it was not polled.

        The byte encodes magnitude in 0.1 A steps with the sign in bit 6:
        ``0..63`` is one direction of load, ``64..127`` the other.
        """
        if raw == 0:
            return None
        if raw >= 64:
            return -(raw - 64) / 10.0
        return raw / 10.0

    @staticmethod
    def _unpack(payload: bytes, count: int) -> Sequence[int]:
        if len(payload) != 4 * count:
            raise ValueError(
                "servo state slice: expected {} bytes for {} slots, got {}"
                .format(4 * count, count, len(payload)))
        return struct.unpack("<" + "i" * count, payload)
