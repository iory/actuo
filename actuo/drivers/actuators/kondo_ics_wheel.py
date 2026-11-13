"""Kondo ICS servo in CONTINUOUS-ROTATION mode: a wheel, commanded by speed.

Same servo, same board, same wire format as :mod:`kondo_ics` -- and a different
meaning. With ``ServoStruct.rotation == 1`` the firmware reads the pulse field
as a SPEED: 7500 stands still, either side turns. So nothing new goes on the
wire; what changes is what the number means, and that is why this is a separate
actuator model rather than a mode flag on the position one. A model whose units
depend on a byte set somewhere else is a model you cannot read.

The map is measured, never assumed
----------------------------------
Nothing on the wire says how fast a given offset turns a wheel, and the
relationship is not a plain proportionality. Measured on the four KRS wheel
servos of a mecanum-base humanoid (``actuo-wheel-calibrate``, wheel speed taken from the time between the reported
angle's sawtooth wraps -- one wrap is one revolution, so no pot scale enters):

    servo  deadband  saturates   rev/s per deg   top rev/s
       20     6 deg     20 deg          0.0528       1.093
       21     4 deg     15 deg          0.0720       1.088
       22     8 deg     20 deg          0.0532       1.098
       23     8 deg     20 deg          0.0529       1.092

Three properties follow, and all three are reasons a single gain would be
wrong:

* **A deadband.** Below it the wheel does not turn at all, and the deadbands
  differ by 2x across four servos of the same model. A controller asking for a
  slow crawl gets nothing from one wheel and motion from another -- on a mecanum
  base that is not a slower robot, it is a robot going somewhere else.
* **Saturation.** Past roughly 20 deg more command buys no more speed. Every
  wheel tops out near 1.09 rev/s, so anything above that cannot be honoured by
  any of them.
* **Per-servo scale.** 0.0528 vs 0.0720 rev/s per deg is a 36 % spread.

So the three numbers are REQUIRED from the robot's config, with no defaults, for
the same reason CubeMars refuses to guess ``pole_pairs``: a wrong one is not a
worse robot, it is a robot doing something else confidently.

What this model refuses to do
-----------------------------
* It does not creep. A speed below what the deadband allows is commanded as a
  STOP, and :attr:`last_refusal` says so -- rather than sending an offset that
  produces either nothing or a lurch depending on the servo.
* It does not pretend to saturate gracefully. A request above the wheel's top
  speed is clamped AND reported, because on a mecanum base the ratio between
  wheels is the direction, and a silently clamped wheel steers the robot.
* It does not report a velocity it has not measured. ``ActuatorState.velocity``
  stays ``None`` and the raw pulse rides in ``aux``.

  What the reading actually is, measured on the same robot 2026-08-01 at 100 Hz -- these
  are the numbers an odometry layer has to be built on:

  * The pot tracks the wheel **1:1 at the ordinary ICS scale**, 29.63 pulses
    per degree, so a full revolution is 360 x 29.63 = 10667 pulses.
  * The board reports it **clamped to [3000, 12000]**, a 9000-pulse window.
    The 1667 pulses outside are 56.3 deg, **15.6 % of every revolution**, and
    in that sector the reading sits at exactly 3000 or exactly 12000 -- an
    exact rail, not noise, which is what says clamp rather than a pot dead
    zone. Measured time at the rails: 15.1 / 14.9 / 14.5 / 14.0 % on the four
    wheels. The arithmetic and the measurement agree to half a percent.
  * Within the window the resolution is 5 pulses, and at 1 rev/s consecutive
    100 Hz samples differ by ~112 pulses -- 22 quanta. A real signal, not the
    "differentiated staircase" an earlier draft of this docstring asserted.

  So odometry IS buildable, and the error it carries is bounded rather than
  accumulating: a wrap is exactly one revolution, so distance per revolution is
  exact and only the position WITHIN the clamped sector is unknown -- at most
  1667 pulses = 56 deg = 29 mm of travel on a 30 mm wheel. An estimator
  unwraps with 10667 pulses per revolution and interpolates across the rails.

  Two things this file deliberately does not do with that. It does not build the
  estimator -- that needs unwrapping, a revolution counter and a policy for the
  clamped sector, and an estimate belongs in a sibling layer above the driver,
  where it stays visibly separate from the measurement it estimates. And it does not report the
  clamp as a per-servo defect: all four wheels show it, it follows exactly from
  the window and the scale, and it is the firmware reporting a continuous
  rotation through a field sized for a position servo's range.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

from actuo.core.contracts import (
    Actuator, ActuatorCommand, ActuatorState, Calibration, Capabilities,
    ChannelKind, ChannelSpec, DeviceId, Frame, OperatingMode)
from actuo.core.registry import register_actuator

from actuo.drivers.actuators.kondo_ics import (
    CENTER_PULSE, KondoBatchServo, PULSE_PER_DEG, SERVO_OFF_PULSE)
from actuo.drivers.kondo_h7_protocol import build_multi_servo

#: Board move-duration byte for a wheel. rcb4 keeps a separate
#: ``wheel_frame_count`` at 1 for exactly this reason: the pulse is a speed, so
#: there is nothing to ramp to and a long duration only delays it.
WHEEL_MOVE_BYTE = 1

TWO_PI = 2.0 * math.pi


@dataclass
class KondoIcsWheelCalibration(Calibration):
    """The measured speed map for ONE wheel servo.

    Parameters
    ----------
    deadband_deg : float
        Command offset below which this wheel does not turn.
    saturate_deg : float
        Offset at which it reaches top speed; beyond it, more buys nothing.
    rev_s_per_deg : float
        Wheel revolutions per second per command degree, fitted between those
        two. Positive; ``direction`` carries the sign.
    """

    # A wheel has no encoder counts and no zero: the base's field is
    # redeclared with a default only so nobody has to invent a number for it.
    counts_per_rev: float = 1.0
    deadband_deg: float = 0.0
    saturate_deg: float = 0.0
    rev_s_per_deg: float = 0.0

    def __post_init__(self) -> None:
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 or -1, got {!r}".format(
                self.direction))
        for name in ("deadband_deg", "saturate_deg", "rev_s_per_deg"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(
                    "{} must be measured and positive, got {!r}. Run "
                    "actuo-wheel-calibrate on THIS servo: these differ by up to "
                    "2x between wheels of the same model, and a shared guess "
                    "steers a mecanum base.".format(name, getattr(self, name)))
        if self.saturate_deg <= self.deadband_deg:
            raise ValueError(
                "saturate_deg ({}) must exceed deadband_deg ({}); between them "
                "is the only range this wheel is controllable in".format(
                    self.saturate_deg, self.deadband_deg))

    @property
    def min_rad_s(self) -> float:
        """Slowest speed this wheel can actually hold, rad/s."""
        return TWO_PI * self.rev_s_per_deg * self.deadband_deg

    @property
    def max_rad_s(self) -> float:
        """Top speed, rad/s."""
        return TWO_PI * self.rev_s_per_deg * self.saturate_deg


@register_actuator("kondo_ics_wheel")
class KondoIcsWheelActuator(KondoBatchServo):
    """One continuous-rotation ICS servo, commanded in rad/s."""

    expects_rotation = True

    def __init__(self, actuator_id: DeviceId,
                 calibration: KondoIcsWheelCalibration,
                 rotation_checked: Optional[str] = None) -> None:
        super().__init__(int(actuator_id), calibration)
        #: How and when this servo was seen to be in rotation mode, for a board
        #: that cannot read it (the RCB-4mini). The ARM-H7 bus reads it from
        #: the board instead. Identity is the servo id only: a servo swapped in
        #: under the same id is not caught.
        self.rotation_checked = rotation_checked
        #: Why the last command could not be honoured, or ``None``. Read it;
        #: the alternative is a wheel quietly doing something else.
        self.last_refusal: Optional[str] = None

    @property
    def capabilities(self) -> Capabilities:
        return Capabilities(
            modes=(OperatingMode.VELOCITY,),
            has_position_sense=False,   # the reading is a pot, and it wraps
            has_velocity_sense=False,   # measurable, but only over a revolution
            has_current_sense=False,
            has_temperature_sense=False,
        )

    @property
    def channels(self) -> Tuple[ChannelSpec, ...]:
        return (
            ChannelSpec("pulse", "count", ChannelKind.STATE,
                        "raw reported angle; a sawtooth, one wrap per wheel "
                        "revolution, with a blind sector"),
            ChannelSpec("pot_rad", "rad", ChannelKind.STATE,
                        "the same reading in radians, using the ICS pulse "
                        "scale. NOT a joint position: it wraps every "
                        "revolution and its zero is wherever the pot happens "
                        "to sit. The SCALE is right -- the pot tracks the "
                        "wheel 1:1 at 29.63 pulses/deg -- but the reading is "
                        "clamped to [3000, 12000], so 15.6 % of every "
                        "revolution reads as an exact rail instead of an "
                        "angle. Good for SEEING which wheel turns which way "
                        "(in a live pose viewer); anything that needs the number "
                        "must handle the clamped sector itself."),
        )

    # -- commanding --------------------------------------------------------

    def offset_for(self, velocity_rad_s: float) -> Tuple[float, Optional[str]]:
        """Command offset in degrees for a wanted wheel speed, and any refusal.

        Returns ``(offset_deg, refusal)``. The offset is 0 when the speed is
        below what the deadband permits: commanding into the deadband is not a
        slow wheel, it is an unpredictable one.
        """
        calibration = self.calibration
        wanted = float(velocity_rad_s) * calibration.direction
        magnitude = abs(wanted)
        if magnitude < 1e-9:
            return 0.0, None
        if magnitude < calibration.min_rad_s:
            return 0.0, (
                "servo {}: {:.3f} rad/s is below this wheel's slowest holdable "
                "speed {:.3f} rad/s (its {:.1f} deg deadband); commanded stop "
                "rather than a crawl it cannot keep".format(
                    self.id, abs(velocity_rad_s), calibration.min_rad_s,
                    calibration.deadband_deg))

        offset = magnitude / (TWO_PI * calibration.rev_s_per_deg)
        refusal = None
        if offset > calibration.saturate_deg:
            refusal = (
                "servo {}: {:.3f} rad/s needs {:.1f} deg but this wheel "
                "saturates at {:.1f} deg ({:.3f} rad/s); clamped. On a mecanum "
                "base the ratio between wheels is the direction, so a clamp is "
                "a turn.".format(
                    self.id, abs(velocity_rad_s), offset,
                    calibration.saturate_deg, calibration.max_rad_s))
            offset = calibration.saturate_deg
        return math.copysign(offset, wanted), refusal

    def pulse_for(self, velocity_rad_s: float) -> int:
        """The wire value for a wanted speed. 7500 is stop."""
        offset, self.last_refusal = self.offset_for(velocity_rad_s)
        return int(round(CENTER_PULSE + offset * PULSE_PER_DEG))

    def batch_pulse(self, command: ActuatorCommand) -> int:
        """Speed value for a velocity command; see :class:`KondoBatchServo`."""
        self.capabilities.require(command.mode)
        if command.velocity is None:
            raise ValueError(
                "VELOCITY mode needs command.velocity (rad/s at the wheel)")
        return self.pulse_for(command.velocity)

    def move_byte(self, bus_default: Optional[int]) -> int:
        """Always 1, whatever the bus uses for the arm joints.

        The field is a speed, so there is nothing to interpolate towards; a
        longer duration would only delay the board acting on it. rcb4 keeps a
        separate ``wheel_frame_count`` at 1 for the same reason.
        """
        return WHEEL_MOVE_BYTE

    def encode_command(self, command: ActuatorCommand) -> Frame:
        self.capabilities.require(command.mode)
        if command.velocity is None:
            raise ValueError(
                "VELOCITY mode needs command.velocity (rad/s at the wheel)")
        return Frame(
            payload=build_multi_servo([self.id], [self.pulse_for(command.velocity)],
                                      WHEEL_MOVE_BYTE),
            address=self.id)

    def encode_release(self) -> Frame:
        """Servo OFF. Note that STOPPING and RELEASING are different here: a
        stop is speed zero and still holds against a push, the off sentinel
        lets the wheel roll."""
        return Frame(
            payload=build_multi_servo([self.id], [SERVO_OFF_PULSE],
                                      WHEEL_MOVE_BYTE),
            address=self.id)

    def encode_mode(self, mode: OperatingMode) -> List[Frame]:
        """No wire mode switch exists. Whether this servo is in rotation mode is
        a property of the SERVO, read from ``ServoStruct.rotation``; commanding
        cannot change it, and pretending otherwise would let a config turn a
        position servo into a wheel by wishing."""
        self.capabilities.require(mode)
        return []

    def pot_angle(self, pulse: float) -> float:
        """Reported pulse as radians of wheel rotation, direction applied.

        Uses the same ICS pulse scale as every other servo on this board, so
        nothing new is assumed; what it cannot fix is that the reading wraps and
        has a blind sector. Read the ``pot_rad`` channel's description before
        using this for anything but looking at.
        """
        return (self.calibration.direction
                * (float(pulse) - CENTER_PULSE) / PULSE_PER_DEG
                * math.pi / 180.0)

    def position_refusal(self, position: float) -> Optional[str]:
        """A wheel has no position to refuse; it has no envelope to leave."""
        return None

    # -- decoding ----------------------------------------------------------

    def matches(self, frame: Frame) -> bool:
        """An id alone is not identity: a frame with the right id can still be
        something other than this servo's state (see
        :meth:`~actuo.drivers.actuators.kondo_ics.KondoIcsActuator.matches`). The bus stamps every state slice
        with a ``kind``, and both must agree."""
        return (frame.meta.get("kind") == "servo_state"
                and frame.address == self.id)

    def decode_state(self, frame: Frame) -> ActuatorState:
        """Report the raw reading and NOTHING derived from it.

        The reported angle is a potentiometer that wraps once per revolution and
        does not report roughly a sixth of each turn. A position from it would
        be meaningless and a velocity differentiated from it at the control rate
        would be a differentiated staircase, so both stay absent rather than
        reported as values they are not.
        """
        slots = frame.meta.get("slots")
        if not slots:
            raise ValueError(
                "servo state frame carries no 'slots' metadata; the bus must "
                "name the slots it packed")
        if len(frame.payload) != 4 * len(slots):
            raise ValueError(
                "servo {} state slice: expected {} bytes for {} slots, got {}"
                .format(self.id, 4 * len(slots), len(slots),
                        len(frame.payload)))
        values = struct.unpack("<" + "i" * len(slots), frame.payload)
        raw = dict(zip(slots, values))

        state = ActuatorState()
        if "current_angle" in raw:
            # 0 means the firmware has not heard from this servo yet.
            if raw["current_angle"] != 0:
                state.aux["pulse"] = float(raw["current_angle"])
                state.aux["pot_rad"] = self.pot_angle(raw["current_angle"])
        if "ref_angle" in raw:
            reference = raw["ref_angle"]
            state.aux["ref_pulse"] = float(reference)
            state.aux["servo_on"] = float(reference != SERVO_OFF_PULSE)
            if reference not in (SERVO_OFF_PULSE, 0):
                # What speed the board is currently driving at, which for a
                # wheel is the only honest "velocity" available -- it is the
                # COMMAND, not a measurement, so it is named as one.
                state.aux["commanded_offset_deg"] = (
                    reference - CENTER_PULSE) / PULSE_PER_DEG
        return state
