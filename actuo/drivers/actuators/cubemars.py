"""CubeMars AK-series actuator model (servo mode).

Owns CubeMars' units, calibration, capabilities and frame (de)serialisation. It
knows nothing about how the bytes travel -- the same model runs over socketcan
on a Linux robot PC or over slcan on a mac (transport and actuator semantics
are separate layers).

Servo mode (not MIT mode)
-------------------------
In servo mode the driver runs its own position/speed/current loops and streams
a status frame periodically at 1-500 Hz; nothing ever "replies" to a command.
That is why this model pairs with ``Transport.write`` + ``Transport.recv`` and
never ``transact``.

Units -- the asymmetry that matters
-----------------------------------
The driver's *position* command and feedback are already **output-shaft
degrees**: the firmware applies the gearbox internally. Its *speed* is
**electrical RPM** of the rotor, which is upstream of the gearbox. So:

* position  -- gear ratio must **not** be applied (doing so would command a
  39:1 motor to 39x the intended angle),
* speed     -- gear ratio **must** be applied
  (``ERPM = rad/s * pole_pairs * gear_ratio * 60 / 2pi``),
* torque    -- gear ratio **must** be applied (output torque = kt * I * ratio).

This is cross-checked three ways: the vendor manual (below), an independent
ros2_control implementation [3], and ERPM values hand-tuned on a CubeMars arm
-- whose 220/50/35 ERPM for 39:1/9:1/6:1 reduction at one common output speed
reproduce ``ERPM ~ gear_ratio`` (50 * 39/9 = 217 ~ 220,
50 * 6/9 = 33 ~ 35).

References
----------
[1] CubeMars AK Series Module Driver User Manual V1.0.14, section 5.2 "Servo
    Mode CAN Message Protocol" -- packet id enum, ``comm_can_set_pos`` /
    ``comm_can_set_pos_spd`` scaling, and section 5.2.1 upload frame layout.
    https://www.cubemars.com/images/file/20240523/1716429938564781.pdf
[2] CubeMars AK Series Module Product Manual v3.0.0.
    https://img.cubemars.com/products/cubemars-product-parameter/AK-Series-Module-Product-Manual-v3.0.0-Download.pdf
[3] OpenFieldAutomation-OFA/cubemars_hardware -- independent ros2_control
    interface; corroborates the deg*10000 position scaling and the
    ``pole_pairs * gear_ratio * 60 / 2pi`` ERPM conversion.
    https://github.com/OpenFieldAutomation-OFA/cubemars_hardware
"""

from __future__ import annotations

import enum
import math
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple

from actuo.core.contracts import (
    Actuator,
    ActuatorCommand,
    ActuatorState,
    Calibration,
    Capabilities,
    ChannelKind,
    ChannelSpec,
    DeviceId,
    Frame,
    OperatingMode,
)
from actuo.core.registry import register_actuator


class ServoPacket(enum.IntEnum):
    """Servo-mode control packet ids; ref [1] section 5.2.

    The CAN arbitration id is ``motor_id | (packet << 8)`` sent as an extended
    (29-bit) id -- so the *control mode* lives in the arbitration id itself.
    That is exactly why ``Frame`` carries an address: this cannot be expressed
    by a payload alone.
    """

    SET_DUTY = 0
    SET_CURRENT = 1
    SET_CURRENT_BRAKE = 2
    SET_RPM = 3
    SET_POS = 4
    SET_ORIGIN_HERE = 5
    SET_POS_SPD = 6


#: Packet ids the vendor defines as commands, i.e. things a *host* sends
#: (ref [1] section 5.2). A frame carrying one of these is never a status
#: upload, so it must never be decoded as motor state -- see
#: :meth:`CubeMarsActuator.matches`. The driver's periodic status upload uses a
#: different, undocumented id (observed: 41 / 0x29 on AK60-39, AK70-9, AK80-6).
_COMMAND_PACKETS = frozenset(int(p) for p in ServoPacket)


def _is_command_packet(packet_id: int) -> bool:
    return packet_id in _COMMAND_PACKETS


#: Error codes from the status frame, ref [1] section 5.2.1. Checking only
#: ``!= 0`` would lose the reason; naming them turns "ABORT: error 4" into "under-voltage".
ERROR_NAMES = {
    0: "none",
    1: "motor over-temperature",
    2: "over-current",
    3: "over-voltage",
    4: "under-voltage",
    5: "encoder fault",
    6: "MOSFET over-temperature",
    7: "motor stall",
}

# Wire scaling constants, ref [1] section 5.2. Kept here in the model that owns
# them -- never in a shared layer, which has no global constants.
#
# "counts" throughout this model means output-shaft degrees*10, which is exactly
# the status frame's native int16 unit -- so decoding is counts_to_rad(pos_raw)
# with no extra factor. The *command* uses a finer degrees*10000, hence the
# counts -> command factor of 1000 below.
_CMD_PER_COUNT = 1000.0      # (degrees*10000) / (degrees*10)
_ERPM_CMD_SCALE = 10.0       # SET_POS_SPD speed/accel fields: ERPM / 10, int16
_CURRENT_CMD_SCALE = 1000.0  # SET_CURRENT: amps * 1000, int32
_SPD_FB_SCALE = 10.0         # status: int16 -> electrical RPM
_CUR_FB_SCALE = 0.01         # status: int16 -> amps
_STATUS_DLC = 8


@dataclass
class CubeMarsCalibration(Calibration):
    """Per-motor parameters for a CubeMars AK actuator.

    Extends the shared :class:`Calibration` and **overrides its counts helpers**,
    because for this device "counts" are output-shaft degrees*10 that the
    firmware has already de-geared. The base implementation divides by
    ``gear_ratio``; doing that here would apply the reduction twice.

    Attributes
    ----------
    pole_pairs : int
        Rotor pole pairs, needed only to convert rad/s <-> electrical RPM. Read
        it from the motor's datasheet -- there is no safe default, so leaving it
        at 0 raises rather than guessing.
    gear_ratio : float
        Reduction ratio, e.g. 39.0 for an AK60-39. Applied to speed and torque
        only; see the module docstring.
    torque_constant : float, optional
        Motor kt in N*m per amp, before the gearbox. Required for CURRENT mode.
    zero_offset_counts : float
        Output-shaft degrees*10 reported when the joint is at 0 rad.
    """

    counts_per_rev: float = 3600.0  # output-shaft degrees * 10 per revolution
    pole_pairs: int = 0
    #: Whole gear windows to add to every reported position. See
    #: :meth:`gear_window_deg` and :meth:`CubeMarsActuator.snap_to_window`.
    #: An integer, because a fractional window is not a thing this can be.
    window_offset: int = 0

    def __post_init__(self) -> None:
        if self.gear_ratio <= 0:
            raise ValueError("gear_ratio must be positive, got {!r}".format(self.gear_ratio))
        if self.direction not in (1, -1):
            raise ValueError("direction must be +1 or -1, got {!r}".format(self.direction))

    @property
    def gear_window_deg(self) -> float:
        """Output-shaft degrees per ONE rotor revolution: ``360 / gear_ratio``.

        This is the quantum of this actuator's origin ambiguity. The rotor's
        absolute encoder is single-turn, so the firmware knows exactly where the
        rotor is but can lose count of *which* rotor turn the output shaft is on.
        Losing one turn moves the reported output angle by exactly this much, and
        by nothing else.

        MEASURED on a 4-joint CubeMars arm, 2026-07-30, joint3 (AK80-6,
        gear_ratio 6.0): with the arm untouched across a power cycle, the reported position went
        160.40 deg -> 100.40 deg. That is **-60.00 deg, to 0.01 deg**, against a
        predicted window of 360/6 = 60.0 -- which also confirms the reduction is
        exactly 6.0 and not some nearby planetary ratio. The other three joints
        moved <= 0.3 deg (mechanical settling) in the same power cycle.

        The three earlier corrections in this joint's calibration history fit the same
        rule once hand-positioning error is allowed for: errors of 106.1, 11.2
        and 172.3 deg sit 13.9, 11.2 and 7.7 deg from 120, 0 and 180 -- all
        multiples of 60, all residuals the size of placing a wrist by eye. This
        power-cycle test is what removed that ambiguity, because nothing was
        touched and the residual came out 0.00.
        """
        return 360.0 / self.gear_ratio

    def counts_to_rad(self, counts: float) -> float:
        """Output-shaft degrees*10 -> radians. No gear division: see class doc."""
        deg = (counts - self.zero_offset_counts) * 0.1
        deg += self.window_offset * self.gear_window_deg
        return self.direction * math.radians(deg)

    def rad_to_counts(self, rad: float) -> float:
        """Radians -> output-shaft degrees*10. No gear multiply: see class doc.

        Subtracts the window offset that :meth:`counts_to_rad` adds. This MUST
        stay the exact inverse: reporting a position one window away from the one
        we command would make the driver chase a target 60 deg from where the
        controller thinks it is, which is a whole-window lurch -- the precise
        failure the servo design exists to prevent. Pinned by
        ``test_window_offset_round_trips``.
        """
        deg = math.degrees(self.direction * rad)
        deg -= self.window_offset * self.gear_window_deg
        return deg * 10.0 + self.zero_offset_counts

    def require_pole_pairs(self) -> int:
        if self.pole_pairs <= 0:
            raise ValueError(
                "pole_pairs is required to convert rad/s <-> electrical RPM but "
                "is unset; take it from the motor datasheet"
            )
        return self.pole_pairs

    def rad_s_to_erpm(self, rad_s: float) -> float:
        """rad/s at the output shaft -> electrical RPM; ref [1]/[3]."""
        pole_pairs = self.require_pole_pairs()
        return rad_s * pole_pairs * self.gear_ratio * 60.0 / (2.0 * math.pi)

    def erpm_to_rad_s(self, erpm: float) -> float:
        """Electrical RPM -> rad/s at the output shaft."""
        pole_pairs = self.require_pole_pairs()
        return erpm * 2.0 * math.pi / (60.0 * pole_pairs * self.gear_ratio)

    def require_kt(self) -> float:
        if not self.torque_constant:
            raise ValueError(
                "torque_constant (kt, N*m/A) is required for CURRENT mode but is unset"
            )
        return self.torque_constant

    def amps_to_torque(self, amps: float) -> float:
        """Motor amps -> output-shaft N*m (kt is pre-gearbox, so scale by ratio)."""
        return amps * self.require_kt() * self.gear_ratio

    def torque_to_amps(self, torque: float) -> float:
        """Output-shaft N*m -> motor amps."""
        return torque / (self.require_kt() * self.gear_ratio)


@dataclass(frozen=True)
class CubeMarsLimits:
    """The ``SET_POS_SPD`` trapezoid envelope, held in the wire's own unit.

    ``SET_POS_SPD`` is a trapezoidal move: the driver ramps to the speed limit
    at the acceleration limit and stops at the target. These are that
    trapezoid's bounds, not a servo gain.

    Stored as electrical RPM rather than rad/s on purpose. ERPM is what the
    protocol actually carries, and converting rad/s -> ERPM needs ``pole_pairs``
    -- a datasheet number that, if wrong, silently scales how fast real hardware
    moves. Keeping the native unit available means a value already validated
    against the motors (e.g. 220/50/35 ERPM hand-tuned for 39:1/9:1/6:1
    motors) can be carried over verbatim with nothing inferred.
    Use :meth:`from_rad_s` when you would rather express limits in SI and have
    a trustworthy ``pole_pairs``.
    """

    max_velocity_erpm: float
    max_acceleration_erpm_s: float

    @classmethod
    def from_rad_s(
        cls,
        calibration: "CubeMarsCalibration",
        max_velocity: float,
        max_acceleration: float,
    ) -> "CubeMarsLimits":
        """Build the envelope from output-shaft rad/s and rad/s^2.

        Requires ``calibration.pole_pairs``; raises if it is unset rather than
        assuming a value.
        """
        return cls(
            max_velocity_erpm=calibration.rad_s_to_erpm(max_velocity),
            max_acceleration_erpm_s=calibration.rad_s_to_erpm(max_acceleration),
        )


@register_actuator("cubemars")
class CubeMarsActuator(Actuator):
    """One CubeMars AK actuator in servo mode.

    Parameters
    ----------
    actuator_id : int
        The driver's CAN node id (1-255), e.g. 104. This is the low byte of
        every arbitration id it sends and receives.
    calibration : CubeMarsCalibration
        Per-motor units/gearing. Never shared between motors.
    limits : CubeMarsLimits, optional
        Trapezoid bounds for POSITION mode. Required to use ``SET_POS_SPD``.
    use_pos_spd : bool
        ``True`` (default) sends ``SET_POS_SPD`` (mode 6): position with an
        explicit speed/accel envelope -- the safe choice for hardware
        bring-up. ``False`` sends ``SET_POS`` (mode 4),
        which moves at the driver's configured default speed.
    """

    def __init__(
        self,
        actuator_id: DeviceId,
        calibration: CubeMarsCalibration,
        limits: "CubeMarsLimits | None" = None,
        use_pos_spd: bool = True,
    ) -> None:
        node_id = int(actuator_id)
        if not 1 <= node_id <= 0xFF:
            raise ValueError(
                "CubeMars CAN node id must fit the arbitration id low byte "
                "(1-255), got {!r}".format(actuator_id)
            )
        super().__init__(node_id, calibration)
        self.node_id = node_id
        self.calibration: CubeMarsCalibration = calibration
        self.limits = limits
        self.use_pos_spd = use_pos_spd

    @property
    def capabilities(self) -> Capabilities:
        """What this *particular* motor can do, given how it was calibrated.

        Capability follows configuration rather than the model name: the driver
        supports every loop, but this stack can only speak rad/s once it knows
        ``pole_pairs`` and N*m once it knows ``kt``. Declaring a mode we could
        only serve by inventing a constant would be a silent degradation that
        looks like a working mode -- so the mode is absent and the request fails
        with a CapabilityError naming what is missing.
        """
        has_erpm_conversion = self.calibration.pole_pairs > 0
        caps_modes = [OperatingMode.POSITION, OperatingMode.PWM]
        if has_erpm_conversion:
            caps_modes.append(OperatingMode.VELOCITY)
        if self.calibration.torque_constant:
            caps_modes.append(OperatingMode.CURRENT)
        return Capabilities(
            modes=tuple(caps_modes),
            has_position_sense=True,
            has_velocity_sense=has_erpm_conversion,
            has_current_sense=True,
            has_temperature_sense=True,
            multiturn=True,  # status position spans +/-3200 deg, ref [1] 5.2.1
        )

    @property
    def channels(self) -> Tuple[ChannelSpec, ...]:
        return super().channels + (
            ChannelSpec(
                "error",
                "code",
                ChannelKind.STATE,
                "0 = healthy; see ERROR_NAMES (ref [1] section 5.2.1)",
            ),
        )

    def _arbitration_id(self, packet: ServoPacket) -> int:
        """``motor_id | (packet << 8)``, ref [1] section 5.2."""
        return self.node_id | (int(packet) << 8)

    def fault_name(self, code: int) -> str:
        return ERROR_NAMES.get(int(code), "unknown fault code {}".format(int(code)))

    def _frame(self, packet: ServoPacket, payload: bytes) -> Frame:
        return Frame(
            payload=payload,
            address=self._arbitration_id(packet),
            meta={"extended": True},
        )

    def matches(self, frame: Frame) -> bool:
        """Whether ``frame`` is this motor's *status* frame.

        The node id is the arbitration id's low byte whatever packet id the
        firmware stamps on its periodic upload (observed: 41), so the low byte
        is what identifies the motor.

        But the low byte alone is NOT enough, and assuming it was caused a real
        misdiagnosis. A **command** frame carries the same node id in the same
        low byte and is also 8 bytes long, so it satisfies every other check --
        including our own ``SET_POS_SPD``, whose payload happens to decode as
        ``error=5``, "encoder fault". When a backend echoes transmitted frames
        back (python-can's gs_usb does), this actuator diagnosed a healthy motor
        as faulted, from a frame it had sent itself. So: a frame carrying a
        packet id the vendor defines as a *command* is by definition not status,
        no matter who put it on the wire.

        Error and remote frames are rejected for the same reason: a bus read at
        the wrong bitrate produces error frames whose garbled ids can collide
        with a real node id, and decoding one would fabricate a joint angle.
        """
        if frame.address is None:
            return False
        if frame.meta.get("error_frame") or frame.meta.get("remote_frame"):
            return False
        address = int(frame.address)
        if (address & 0xFF) != self.node_id:
            return False
        return not _is_command_packet(address >> 8)

    def encode_command(self, command: ActuatorCommand) -> Frame:
        self.capabilities.require(command.mode)

        if command.mode is OperatingMode.POSITION:
            return self._encode_position(command)
        if command.mode is OperatingMode.VELOCITY:
            return self._encode_velocity(command)
        if command.mode is OperatingMode.CURRENT:
            return self._encode_current(command)
        return self._encode_duty(command)

    def _encode_position(self, command: ActuatorCommand) -> Frame:
        if command.position is None:
            raise ValueError("POSITION mode needs command.position (rad)")
        self._check_position_limit(command.position)

        counts = self.calibration.rad_to_counts(command.position)
        pos_raw = int(round(counts * _CMD_PER_COUNT))
        # deg*10000 must fit int32; ref [1] uses (int32_t)(pos * 10000.0).
        if not -(2 ** 31) <= pos_raw < 2 ** 31:
            raise ValueError(
                "position {:.3f} rad is outside the protocol's int32 "
                "degrees*10000 range".format(command.position)
            )

        if not self.use_pos_spd:
            return self._frame(ServoPacket.SET_POS, struct.pack(">i", pos_raw))

        if self.limits is None:
            raise ValueError(
                "SET_POS_SPD needs limits (max_velocity/max_acceleration); pass "
                "limits= or set use_pos_spd=False to use the driver's defaults"
            )

        speed_raw = int(round(self.limits.max_velocity_erpm / _ERPM_CMD_SCALE))
        accel_raw = int(round(self.limits.max_acceleration_erpm_s / _ERPM_CMD_SCALE))
        # A zero/negative envelope means "never arrives" or a firmware-defined
        # surprise; refuse loudly instead of shipping it.
        if not 1 <= speed_raw <= 32767:
            raise ValueError(
                "max_velocity {:.1f} ERPM maps to {} (ERPM/10), outside the "
                "int16 range 1..32767".format(self.limits.max_velocity_erpm, speed_raw)
            )
        if not 1 <= accel_raw <= 32767:
            raise ValueError(
                "max_acceleration {:.1f} ERPM/s maps to {} (ERPM/10), outside "
                "the int16 range 1..32767".format(
                    self.limits.max_acceleration_erpm_s, accel_raw
                )
            )

        return self._frame(
            ServoPacket.SET_POS_SPD,
            struct.pack(">ihh", pos_raw, speed_raw, accel_raw),
        )

    def _encode_velocity(self, command: ActuatorCommand) -> Frame:
        if command.velocity is None:
            raise ValueError("VELOCITY mode needs command.velocity (rad/s)")
        erpm = int(round(self.calibration.rad_s_to_erpm(command.velocity)))
        if not -(2 ** 31) <= erpm < 2 ** 31:
            raise ValueError("velocity maps outside the int32 ERPM range")
        return self._frame(ServoPacket.SET_RPM, struct.pack(">i", erpm))

    def _encode_current(self, command: ActuatorCommand) -> Frame:
        if command.current is None:
            raise ValueError("CURRENT mode needs command.current (N*m)")
        amps = self.calibration.torque_to_amps(command.current)
        raw = int(round(amps * _CURRENT_CMD_SCALE))
        if not -(2 ** 31) <= raw < 2 ** 31:
            raise ValueError("current maps outside the int32 milliamp range")
        return self._frame(ServoPacket.SET_CURRENT, struct.pack(">i", raw))

    def _encode_duty(self, command: ActuatorCommand) -> Frame:
        if command.pwm is None:
            raise ValueError("PWM mode needs command.pwm (duty, -1..1)")
        if not -1.0 <= command.pwm <= 1.0:
            raise ValueError("duty must be within -1..1, got {!r}".format(command.pwm))
        # ref [1]: buffer_append_int32(buffer, (int32_t)(duty * 100000.0))
        return self._frame(
            ServoPacket.SET_DUTY, struct.pack(">i", int(round(command.pwm * 100000.0)))
        )

    def position_refusal(self, position: float) -> Optional[str]:
        """Why this angle cannot be commanded, or None. See the base contract.

        The taught envelope, reported rather than raised, so the torque layer
        can decline before applying torque. A CubeMars arm hit the alternative twice
        in one day: a joint parked just outside its taught limit satisfied every
        guard (the controller adopts the measured pose, so the jump is zero),
        and one cycle later ``encode_command`` raised inside ``service()`` and
        killed the servicer -- `bus write failed on cycle N`. A guard that
        crashes the process is worse than the ones that say no.
        """
        limit = self.calibration.position_limit
        if limit is None:
            return None
        low, high = limit
        if not low <= position <= high:
            return ("position {:.4f} rad is outside this joint's limit "
                    "[{:.4f}, {:.4f}]".format(position, low, high))
        return None

    def _check_position_limit(self, position: float) -> None:
        """Same check, raising, for the encode path. One implementation."""
        refusal = self.position_refusal(position)
        if refusal is not None:
            raise ValueError(refusal)

    def encode_mode(self, mode: OperatingMode) -> List[Frame]:
        """No mode-switch frames exist: servo mode selects the loop per command.

        The control mode rides in each command's arbitration id, so there is
        nothing to pre-arm. Capability is still checked so an unsupported mode
        fails here rather than silently sending nothing.
        """
        self.capabilities.require(mode)
        return []

    def encode_release(self) -> Frame:
        """Servo OFF: command zero current, so the joint goes limp.

        Servo mode has **no disable/release packet** -- ref [1] section 5.2
        defines exactly seven, all of which are "drive like this". So "off" has
        to be synthesised, and zero current is the honest way: no torque.

        Simply *stopping* commands is NOT off. The driver holds its last
        target. The manual mentions a configurable
        "sudden CAN communication interruption" behaviour, but that is set out
        of band via R-LINK and invisible to us -- relying on it would be
        assuming a setting we cannot read.

        Unlike CURRENT mode this needs no ``torque_constant``: zero torque is
        zero amps in any calibration, so nothing is inferred and this works on a
        motor whose kt we were never told.
        """
        return self._frame(ServoPacket.SET_CURRENT, struct.pack(">i", 0))

    def snap_to_window(
        self,
        measured_rad: float,
        coarse_rad: float,
        ambiguity_margin: float = 0.5,
    ) -> int:
        """Pick the gear window that puts ``measured_rad`` nearest ``coarse_rad``.

        The fix for this actuator family's origin ambiguity, and a much better
        one than a hand-written offset. The rotor's absolute encoder is
        single-turn, so what a power cycle can lose is only *which rotor turn*
        the output shaft is on -- an integer number of
        :attr:`CubeMarsCalibration.gear_window_deg`. So a coarse reference
        accurate to only +-half a window is enough to recover the exact angle:
        the encoder supplies all the precision, the reference supplies one
        integer.

        That means +-30 deg on an AK80-6 (6:1) and +-20 deg on an AK10-9 or
        AK70-9 (9:1) -- an eyeballed startup pose reaches that easily. An
        AK60-39 (39:1) is the tight one at +-4.6 deg.

        Call this ONCE, at startup, while the robot is at the reference pose.
        Never per cycle: once the arm moves away from the reference, re-deriving
        the window against it would snap the joint to a different one mid-motion.
        The window is an integer chosen once; after that the firmware's own
        continuous multi-turn count carries the joint across window boundaries
        with no discontinuity, which is exactly why this needs no modulo.

        Parameters
        ----------
        measured_rad : float
            What the actuator reports right now, decoded with the CURRENT window
            offset (so calling this twice is idempotent).
        coarse_rad : float
            Roughly where the joint actually is. Must be within half a window.
        ambiguity_margin : float
            Fraction of a HALF window the reference may sit away from the chosen
            window before this refuses. The default 0.5 states a simple rule:
            **the reference must land within a quarter of a window, and the
            remaining quarter is the margin before the boundary** -- so the
            reference has to be twice as bad as accepted before it picks the
            wrong window.

            Not 1.0 on purpose: at a full half-window the choice is a coin flip,
            and a wrong window is a position off by exactly one window that looks
            entirely normal downstream -- the one output this stack must never
            emit: a wrong value that looks right.

            Not tighter than 0.5 either, and that was measured rather than
            guessed: hand-positioning errors of 7.7-13.9 deg were recorded on an
            AK80-6 joint, so a 0.35 margin (10.5 deg on a 60 deg window) refused
            the ordinary case. 0.5 gives an AK80-6 +-15 deg, which covers every
            error yet observed. It gives an AK60-39 (9.2 deg window) only
            +-2.3 deg, which correctly says a hand-placed pose cannot
            disambiguate that joint -- an honest refusal, not a gap.

        Returns
        -------
        int
            The window offset now in force (absolute, not a delta).

        Raises
        ------
        ValueError
            If the reference cannot pick a window confidently. Refusing is the
            point: the alternative is silently reporting a pose 60 deg from the
            truth, which reads as completely normal downstream.
        """
        period = math.radians(self.calibration.gear_window_deg)
        error = coarse_rad - measured_rad
        steps = int(round(error / period))
        residual = error - steps * period

        limit = ambiguity_margin * (period / 2.0)
        if abs(residual) > limit:
            raise ValueError(
                "cannot pick a gear window for actuator {}: the reference pose "
                "{:.2f} deg is {:.2f} deg from the nearest window of the "
                "measured {:.2f} deg, which is more than {:.0f}% of the way to "
                "the {:.2f} deg boundary. Either the robot is not at the "
                "reference pose, or the reference is not accurate enough for a "
                "{:.1f} deg window (this joint needs +-{:.1f} deg). Refusing "
                "rather than reporting a position that may be one whole window "
                "out.".format(
                    self.id, math.degrees(coarse_rad), math.degrees(residual),
                    math.degrees(measured_rad), ambiguity_margin * 100.0,
                    math.degrees(period / 2.0),
                    self.calibration.gear_window_deg,
                    self.calibration.gear_window_deg / 2.0,
                )
            )

        # `steps` is relative to the CURRENT decode, so accumulate rather than
        # assign: that makes a second call with the same reference a no-op
        # instead of throwing the first correction away.
        self.calibration.window_offset += self.calibration.direction * steps
        return self.calibration.window_offset

    def encode_set_origin(self, permanent: bool = False) -> Frame:
        """Zero the encoder at the present position; ref [1] ``comm_can_set_origin``.

        Parameters
        ----------
        permanent : bool
            ``False`` sets a temporary origin (lost on power cycle), ``True``
            writes it to the driver's non-volatile memory.
        """
        return self._frame(
            ServoPacket.SET_ORIGIN_HERE, struct.pack(">B", 1 if permanent else 0)
        )

    def decode_state(self, frame: Frame) -> ActuatorState:
        """Parse a periodic status frame; ref [1] section 5.2.1.

        Layout: pos int16 (deg*10) | speed int16 (ERPM/10) | current int16
        (A*100) | temperature int8 | error uint8.

        **The temperature is the DRIVER BOARD's, not the motor's**, and the
        manual's own field name says otherwise. Verbatim, ref [1] section 5.2.1:

            "Motor Temperature: int8, range -20~127 represents the driver board
             temperature -20 C~127 C."

        So the field is *named* Motor Temperature and *defined* as the driver
        board temperature. Anyone reasoning about thermal headroom from this
        number is reasoning about the FET side. Consequences worth knowing:

        * The relevant limit is the DRIVER's: **100 C, over-temperature
          protection** (ref [1], "Maximum Allowable Temperature of the Driver";
          some smaller driver variants are rated 85 C). Ambient is rated
          -20..65 C.
        * The matching fault is ``6`` (MOSFET over-temperature). Fault ``1``
          (motor over-temperature) refers to a winding sensor whose value this
          protocol **never uploads** -- so a joint can fault on a temperature
          nothing here can show climbing.
        * The winding temperature IS obtainable, but only over the motor's 3-pin
          UART port, never over CAN: the servo-mode CAN protocol defines seven
          packets (0-6) and all seven command motion, with no telemetry request.
          Measured on the real bus for confirmation -- only packet 4 (our
          commands) and packet 41 (this frame) ever appear.
          Over UART, ``02 01 04 40 84 03`` returns MOS temperature and motor
          temperature as consecutive int16/10 (0.1 C). Nothing in the manual
          forwards those over CAN.
        * If that is ever implemented, **do not trust the selective-read bit
          numbering to ask for one of them.** ``02 05 32 <4-byte mask>`` picks
          parameters by bit, and the manual's bit table says bit 1 = MOS,
          bit 2 = Motor -- while its own worked example sets mask ``0x00000001``
          and captions it "Get motor temperature command". Both sensors return a
          plausible temperature, so choosing the wrong bit is silent. Ask for
          both (mask ``0x00000003``) and identify them by ORDER: MOS first,
          Motor second, which the get-all layout and both conversion-formula
          lists agree on.
        """
        if len(frame.payload) != _STATUS_DLC:
            raise ValueError(
                "CubeMars status frame must be {} bytes, got {}".format(
                    _STATUS_DLC, len(frame.payload)
                )
            )

        pos_raw, spd_raw, cur_raw, temp_c, error = struct.unpack(">hhhbB", frame.payload)

        # pos_raw is already degrees*10 == this model's "counts".
        position = self.calibration.counts_to_rad(pos_raw)
        erpm = spd_raw * _SPD_FB_SCALE
        amps = cur_raw * _CUR_FB_SCALE

        state = ActuatorState(
            position=position,
            temperature=float(temp_c),
            error=int(error),
            # The raw wire quantities always survive, even when we lack the
            # datasheet constants to turn them into SI. Better a named channel
            # in its own unit than a plausible-looking wrong number.
            aux={"current_a": amps, "erpm": erpm},
        )
        # rad/s needs pole_pairs; N*m needs kt. Without them the channel stays
        # None -- ActuatorState's contract is that absent means absent, never 0.
        if self.calibration.pole_pairs > 0:
            state.velocity = self.calibration.direction * self.calibration.erpm_to_rad_s(
                erpm
            )
        if self.calibration.torque_constant:
            state.current = self.calibration.amps_to_torque(amps)
        return state
