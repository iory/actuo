"""CubeMars AK-series actuator model, MIT mode.

A SEPARATE model from :mod:`cubemars`, not a mode inside it, and the manual is
the reason: "multiple optional control modes cannot be switched while the
driver board is running, and the communication protocols between different
control modes are different ... using the wrong protocol control may result in
the burning of the driver board" [1]. A motor is in one mode or the other for
the life of a power cycle. Two models that cannot both be right about one motor
should not share a class where a stray flag can pick the wrong one.

The difference on the wire is total. Servo mode sends EXTENDED ids carrying the
packet type in the upper bits and streams status unprompted; MIT mode sends
STANDARD ids equal to the motor id and replies only when addressed.

WHAT THIS MODE IS FOR
---------------------
Impedance control. One frame carries a position target, a velocity target, kp,
kd and a feedforward torque, and the motor computes

    tau = kp * (p_des - p) + kd * (v_des - v) + t_ff

in its own loop, far faster than a bus round trip. That is the control law an
RL locomotion policy is trained against, and servo mode cannot express it: its
SET_POS runs the motor's own position loop with the motor's own gains.

THE FIELD RANGES ARE PER MOTOR AND ARE NOT OPTIONAL
---------------------------------------------------
Every field is a fixed-point number scaled into its own range, so a wrong range
is a silently wrong command rather than an error: send 1.0 rad with p_max set
to 50 when the motor uses 12.5 and the motor moves a quarter as far as asked.
The ranges are therefore required config, never defaulted. CubeMars publishes
them per model [1]:

    model      position rad   speed rad/s   torque N*m
    AK10-9     +/- 12.5       +/- 50        +/- 65
    AK60-6     +/- 12.5       +/- 45        +/- 15
    AK70-10    +/- 12.5       +/- 50        +/- 25
    AK80-6     +/- 12.5       +/- 76        +/- 12
    AK80-9     +/- 12.5       +/- 50        +/- 18
    AK80-80    +/- 12.5       +/-  8        +/- 144
    AK80-8     +/- 12.5       +/- 37.5      +/- 32

kp is 0 to 500 and kd is 0 to 5 on every model. Those two are the same
everywhere, so they are the one thing this file does fix.

Note that AK70-9 and AK80-64 are NOT in that table, although both run in MIT
mode. Their ranges have to come from the manual for those parts or from the
vendor's upper-computer software or firmware package; guessing from a neighbouring model is how the failure above happens.

[1] CubeMars, AK Series Module Driver User Manual V1.0.14, sections 5.3 and 6.
    https://www.cubemars.com/images/file/20240523/1716429938564781.pdf
"""

from __future__ import annotations

import dataclasses
import math
import struct
from typing import Optional

from actuo.core.contracts import (
    Actuator,
    ActuatorCommand,
    ActuatorState,
    Capabilities,
    Frame,
    OperatingMode,
)
from actuo.core.registry import register_actuator

#: kp and kd ranges. Identical on every AK model, unlike the three above.
KP_MIN, KP_MAX = 0.0, 500.0
KD_MIN, KD_MAX = 0.0, 5.0

#: The three stateless commands, sent as the whole 8-byte payload.
ENTER_MOTOR_MODE = bytes([0xFF] * 7 + [0xFC])
EXIT_MOTOR_MODE = bytes([0xFF] * 7 + [0xFD])
SET_ZERO_POSITION = bytes([0xFF] * 7 + [0xFE])

#: The reply is 8 bytes: id, position (2), speed and current sharing a byte
#: (3), temperature, error flag. Ref [1] "MIT Mode Driver Board Send Data
#: Definition", which states DLC 8 explicitly.
#:
#: Read it off the table rather than inferring it from the fields the decoder
#: uses: a count of 7, inferred that way, made `matches` reject every real
#: reply from an AK80-64 -- a motor that answers correctly looking exactly like
#: a motor that is not there.
_REPLY_LEN = 8

#: Temperature is reported with this offset, so the wire's 0 means -40 C.
_TEMP_OFFSET_C = 40.0


def float_to_uint(value: float, low: float, high: float, bits: int) -> int:
    """Scale a float in [low, high] onto an unsigned integer of `bits` bits.

    Parameters
    ----------
    value : float
        Quantity in engineering units. Clamped into range before scaling.
    low, high : float
        The field's range for this motor model.
    bits : int
        Field width.

    Returns
    -------
    int
        The fixed-point value to put on the wire.

    Notes
    -----
    Uses ``(1 << bits) - 1`` as the span, matching the manual's own
    ``uint_to_float``. The manual's sample ``float_to_uint`` writes
    ``(1 << bits)`` instead, which is inconsistent with its inverse by one
    part in 2**bits; taking the encoder's word for it would make a round trip
    drift. The difference is far below a count, but only one of the two can be
    the inverse of the other.
    """
    span = high - low
    clamped = min(max(value, low), high)
    return int((clamped - low) * ((1 << bits) - 1) / span)


def uint_to_float(raw: int, low: float, high: float, bits: int) -> float:
    """Inverse of :func:`float_to_uint`."""
    span = high - low
    return raw * span / ((1 << bits) - 1) + low


@dataclasses.dataclass
class MitCalibration:
    """Per-joint wire ranges, gains and datum for one motor in MIT mode.

    Attributes
    ----------
    gear_ratio : float
        Recorded for the caller's benefit only. MIT mode's position and speed
        are already at the OUTPUT shaft, exactly as servo mode's are, so this
        model never multiplies or divides by it. It is here so a config that
        states it is not silently dropped.
    direction : int
        +1 or -1. ``model = direction * (wire - zero_offset)``.
    zero_offset_rad : float
        Where the wire reads when the joint is at the model's zero.
    p_max, v_max, t_max : float
        Field ranges for THIS motor model, symmetric about zero. Required.
    kp, kd : float
        Impedance gains, in N*m/rad and N*m*s/rad at the output shaft. These
        go on the wire every frame; they are constants of the joint, not a
        channel anything upstream modulates.
    torque_limit_nm : float, optional
        Clamp applied to the feedforward torque before scaling. Not the
        motor's own limit, which it enforces regardless.
    position_limit : tuple of float, optional
        ``(low, high)`` in MODEL radians. Enforced at encode time, so an
        out-of-range target is refused rather than clamped.

        On a biped the MIT-mode joints can be the hip pitch, hip roll and
        knee -- the heaviest on the robot -- so this model needs the same
        fence the servo model has, read from the same ``position_limit_deg``
        config key.
    """

    gear_ratio: float
    direction: int = 1
    zero_offset_rad: float = 0.0
    p_max: float = 0.0
    v_max: float = 0.0
    t_max: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    torque_limit_nm: Optional[float] = None
    position_limit: Optional[tuple] = None
    #: Whole gear windows added to every position, as on the servo/3.0 models
    #: (:attr:`CubeMarsCalibration.window_offset`). A MIT drive keeps its
    #: written origin across a power cycle but, its rotor encoder being single
    #: turn, can come back a whole rotor turn -- 360 / gear_ratio at the output
    #: -- from where it was: a biped's right knee (AK80-64) read 5.35,
    #: 5.48 and 5.61 deg from home after three restarts on 2026-09-29/30,
    #: against a 5.625 deg window.
    window_offset: int = 0

    def __post_init__(self):
        for name in ("p_max", "v_max", "t_max"):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    "{} is required and must be positive: MIT mode scales every "
                    "field into its model's own range, so an absent range is a "
                    "silently wrong command rather than an error. See this "
                    "module's docstring for the published table.".format(name))
        if not KP_MIN <= self.kp <= KP_MAX:
            raise ValueError(
                "kp {} is outside the wire's range [{}, {}]".format(
                    self.kp, KP_MIN, KP_MAX))
        if not KD_MIN <= self.kd <= KD_MAX:
            raise ValueError(
                "kd {} is outside the wire's range [{}, {}]. The field is 12 "
                "bits over 0..5 on every AK model, so a larger gain cannot be "
                "sent -- it would arrive clamped to 5 with no error anywhere. "
                "Either the gain is in units this wire does not speak, or the "
                "PD has to run on the host against a current command.".format(
                    self.kd, KD_MIN, KD_MAX))

    @property
    def gear_window_deg(self) -> float:
        """Output degrees per rotor turn, the quantum of the origin ambiguity."""
        return 360.0 / self.gear_ratio

    def to_wire(self, model_rad: float) -> float:
        """Model angle -> the angle to put on the wire.

        The exact inverse of :meth:`to_model`, window included: reporting one
        window and commanding another would be a whole-window lurch.
        """
        return (self.direction * model_rad + self.zero_offset_rad
                - self.window_offset * math.radians(self.gear_window_deg))

    def to_model(self, wire_rad: float) -> float:
        """Wire angle -> the model's angle."""
        return self.direction * (
            wire_rad - self.zero_offset_rad
            + self.window_offset * math.radians(self.gear_window_deg))


@register_actuator("cubemars_mit")
class CubeMarsMitActuator(Actuator):
    """One AK-series motor driven in MIT mode."""

    #: This mode answers only when addressed, unlike every other CubeMars model
    #: in this package. Stated so a bus that merely listens can know to ask.
    answers_only_when_addressed = True

    def __init__(self, actuator_id, calibration: MitCalibration) -> None:
        node_id = int(actuator_id)
        if not 1 <= node_id <= 0x7FF:
            raise ValueError(
                "MIT mode addresses the motor by a STANDARD 11-bit id, so the "
                "node id must fit in 1..2047; got {}".format(node_id))
        super().__init__(actuator_id, calibration)
        self.node_id = node_id
        self.calibration = calibration

    @property
    def capabilities(self) -> Capabilities:
        """POSITION only, and that is the whole point of the mode.

        The frame does carry a feedforward torque, but a caller that wanted
        torque control would be asking for kp = kd = 0, which is a different
        control law wearing this one's clothes. Declaring CURRENT here would
        invite exactly that.

        Velocity sense needs no pole_pairs here, unlike servo mode: the reply
        carries rad/s at the output shaft directly rather than electrical RPM.
        """
        return Capabilities(
            modes=(OperatingMode.POSITION,),
            has_position_sense=True,
            has_velocity_sense=True,
            has_current_sense=True,
            has_temperature_sense=True,
            multiturn=False,  # position is bounded by p_max, not a turn count
        )

    def _frame(self, payload: bytes) -> Frame:
        # Standard (11-bit) id equal to the motor id. Servo mode's extended id
        # carrying a packet type in the upper bits is the other protocol.
        return Frame(payload=payload,
                     address=self.node_id,
                     meta={"extended": False})

    def enter_motor_mode(self) -> Frame:
        """The motor ignores control frames until this has been sent."""
        return self._frame(ENTER_MOTOR_MODE)

    def exit_motor_mode(self) -> Frame:
        """Leaves the motor un-driven. Send this before dropping the bus."""
        return self._frame(EXIT_MOTOR_MODE)

    def set_zero_position(self) -> Frame:
        """Makes the present position the wire's zero. Persists in the motor."""
        return self._frame(SET_ZERO_POSITION)

    def matches(self, frame: Frame) -> bool:
        """Whether `frame` is this motor's reply.

        The arbitration id cannot decide it. In MIT mode a reply comes back on
        `0x00 + driver id`, which is the same standard id the master addressed
        the motor on, so replies from a bus of twelve are told apart only by
        the first payload byte, which the motor fills with its own id.

        A command is also eight bytes on that same id, so length cannot
        separate the two either -- this used to think it could, by believing a
        reply was seven. What separates them is direction: a backend that
        echoes what it transmits (python-can's gs_usb does) marks those frames
        `is_rx` False, and the transport drops them before this is ever asked.
        Frames that arrive here still carrying that mark are rejected rather
        than trusted, because decoding a command as state would read byte 0 --
        the position high byte -- as a motor id.

        **Extended frames are rejected, and that is not a formality.** On a
        biped's leg bus one motor ran MIT mode while eleven still ran servo
        mode, and a servo status frame is also eight bytes and also carries an
        arbitrary byte 0. A hip roll motor's status frame began 0x01, so a MIT
        actuator with node id 1 decoded its neighbour's
        position as its own -- reporting 655 deg of movement and a saturated
        144 N*m from a motor that had not moved at all. The seven-byte length
        check had been hiding this by accident. The manual is explicit that a
        reply is a STANDARD frame ("Identifier: 0X00+Driver ID, Frame Type:
        Standard Frame") while every servo frame is extended, so that is the
        discriminator with a reason behind it.
        """
        if frame.meta.get("is_rx") is False:
            return False
        if frame.meta.get("extended"):
            return False
        payload = frame.payload
        return len(payload) == _REPLY_LEN and payload[0] == self.node_id

    def encode_mode(self, mode: OperatingMode) -> list:
        """Arm the motor. Unlike servo mode, MIT mode must be entered.

        The manual is explicit: "it is necessary to enter Motor Control Mode
        before controlling the motor using CAN communication". Until this
        frame lands the motor ignores commands, which looks exactly like a
        wiring fault.
        """
        self.capabilities.require(mode)
        return [self.enter_motor_mode()]

    def snap_to_window(self, measured_rad: float, coarse_rad: float,
                       ambiguity_margin: float = 0.5) -> int:
        """Pick the gear window nearest a coarse reference; see
        :meth:`CubeMarsActuator.snap_to_window`, which this shares.

        The window here is 5.625 deg on a 64:1 drive, so the default margin
        needs the reference within +-1.4 deg. A hand-placed pose does not reach
        that; the position this servicer last recorded with the joint limp,
        on a robot nobody moved while the power was off, does.
        """
        from actuo.drivers.actuators.cubemars import CubeMarsActuator
        return CubeMarsActuator.snap_to_window(self, measured_rad, coarse_rad,
                                               ambiguity_margin)

    def encode_release(self) -> Frame:
        """Leave motor mode, which is a real disable rather than a synthesised one.

        Servo mode has no off switch and has to fake one by commanding zero
        current; MIT mode has EXIT_MOTOR_MODE and the motor stops driving. Note
        that simply ceasing to send is NOT off in either mode.
        """
        return self.exit_motor_mode()

    def encode_limp(self) -> Frame:
        """A command with every gain and the torque at zero.

        Sent before EXIT on a release, so the command the drive keeps through
        EXIT -- and runs again the instant the next ENTER lands -- asks for
        nothing. EXIT alone left the last walking target stored at kp 500: on
        2026-10-13, the cycle between ENTER and the first new command put
        105.7 N*m on a biped's AK80-64 knee and moved it 5 deg toward a target
        from an earlier run that had ended without a release.
        """
        cal = self.calibration
        p_int = float_to_uint(0.0, -cal.p_max, cal.p_max, 16)
        v_int = float_to_uint(0.0, -cal.v_max, cal.v_max, 12)
        t_int = float_to_uint(0.0, -cal.t_max, cal.t_max, 12)
        payload = bytes([
            (p_int >> 8) & 0xFF,
            p_int & 0xFF,
            (v_int >> 4) & 0xFF,
            (v_int & 0x0F) << 4,                          # kp high nibble 0
            0x00,                                         # kp low 0
            0x00,                                         # kd high 0
            (t_int >> 8) & 0x0F,                          # kd low nibble 0
            t_int & 0xFF,
        ])
        return self._frame(payload)

    def position_refusal(self, position: float) -> Optional[str]:
        """Why this angle cannot be commanded, or None. See the base contract.

        Reported rather than raised, so a torque layer can decline before
        applying torque. Raising inside a servicer's cycle kills the servicer,
        which is worse than the guard it implements.
        """
        limit = self.calibration.position_limit
        if limit is None:
            return None
        low, high = limit
        if not low <= position <= high:
            return ("position {:.4f} rad is outside this joint's limit "
                    "[{:.4f}, {:.4f}]".format(position, low, high))
        return None

    def encode_command(self, command: ActuatorCommand) -> Frame:
        """Pack one impedance command.

        The velocity target is 0 unless a feedforward is given, and that is
        deliberate: the damping term is meant to pull the MEASURED velocity to
        zero. Feeding the derivative of the position target instead turns the
        law into something the policy was not trained against.
        """
        self.capabilities.require(command.mode)
        cal = self.calibration
        if command.position is None:
            raise ValueError("POSITION mode needs a position")
        refusal = self.position_refusal(command.position)
        if refusal is not None:
            raise ValueError(refusal)

        p_des = cal.to_wire(command.position)
        v_des = cal.direction * (command.velocity_ff or 0.0)
        t_ff = command.current_ff or 0.0
        if cal.torque_limit_nm is not None:
            t_ff = min(max(t_ff, -cal.torque_limit_nm), cal.torque_limit_nm)
        t_ff *= cal.direction

        p_int = float_to_uint(p_des, -cal.p_max, cal.p_max, 16)
        v_int = float_to_uint(v_des, -cal.v_max, cal.v_max, 12)
        kp_int = float_to_uint(cal.kp, KP_MIN, KP_MAX, 12)
        kd_int = float_to_uint(cal.kd, KD_MIN, KD_MAX, 12)
        t_int = float_to_uint(t_ff, -cal.t_max, cal.t_max, 12)

        payload = bytes([
            (p_int >> 8) & 0xFF,
            p_int & 0xFF,
            (v_int >> 4) & 0xFF,
            ((v_int & 0x0F) << 4) | ((kp_int >> 8) & 0x0F),
            kp_int & 0xFF,
            (kd_int >> 4) & 0xFF,
            ((kd_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F),
            t_int & 0xFF,
        ])
        return self._frame(payload)

    def decode_command(self, frame: Frame) -> dict:
        """What a command frame actually carries, back in model terms.

        The inverse of :meth:`encode_command`, quantisation included, so a log
        built from this records the gains and target that reached the wire
        rather than the ones that were asked for. The two differ by clamping
        into each field's range and by a count of rounding.

        Returns
        -------
        dict
            ``position`` and ``velocity`` (rad, rad/s) and ``torque`` (N*m) in
            the model's sign convention, and ``kp``, ``kd`` as sent.
        """
        payload = frame.payload
        cal = self.calibration
        p_int = (payload[0] << 8) | payload[1]
        v_int = (payload[2] << 4) | (payload[3] >> 4)
        kp_int = ((payload[3] & 0x0F) << 8) | payload[4]
        kd_int = (payload[5] << 4) | (payload[6] >> 4)
        t_int = ((payload[6] & 0x0F) << 8) | payload[7]
        return {
            "position": cal.to_model(
                uint_to_float(p_int, -cal.p_max, cal.p_max, 16)),
            "velocity": cal.direction * uint_to_float(
                v_int, -cal.v_max, cal.v_max, 12),
            "kp": uint_to_float(kp_int, KP_MIN, KP_MAX, 12),
            "kd": uint_to_float(kd_int, KD_MIN, KD_MAX, 12),
            "torque": cal.direction * uint_to_float(
                t_int, -cal.t_max, cal.t_max, 12),
        }

    def decode_state(self, frame: Frame) -> ActuatorState:
        """Unpack one reply.

        Eight bytes, and its first byte is the motor's own id rather than the
        arbitration id: in MIT mode replies come back on `0x00 + driver id`,
        so that byte is how a bus of motors is told apart. Ref [1], "MIT Mode
        Driver Board Send Data Definition"::

            [0] driver id   [1..2] position   [3] speed high 8
            [4] speed low 4 | current high 4  [5] current low 8
            [6] motor temperature             [7] motor error flag

        Confirmed on hardware 2026-09-12, the first AK80-64 switched into MIT
        mode: `01 80 80 7f d7 ff 42 00` at rest, decoding to +2.9 deg, -0.01
        rad/s, -0.07 N*m, 26 C, no fault.
        """
        payload = frame.payload
        if len(payload) < _REPLY_LEN:
            raise ValueError(
                "MIT reply is {} bytes, got {}".format(_REPLY_LEN, len(payload)))
        reported_id = payload[0]
        if reported_id != self.node_id:
            raise ValueError(
                "MIT reply is from motor {}, not {}".format(
                    reported_id, self.node_id))

        cal = self.calibration
        p_int = (payload[1] << 8) | payload[2]
        v_int = (payload[3] << 4) | (payload[4] >> 4)
        i_int = ((payload[4] & 0x0F) << 8) | payload[5]

        wire_rad = uint_to_float(p_int, -cal.p_max, cal.p_max, 16)
        return ActuatorState(
            position=cal.to_model(wire_rad),
            velocity=cal.direction * uint_to_float(
                v_int, -cal.v_max, cal.v_max, 12),
            current=cal.direction * uint_to_float(
                i_int, -cal.t_max, cal.t_max, 12),
            temperature=float(payload[6]) - _TEMP_OFFSET_C,
            # Byte 7, which this dropped while it believed the reply was seven
            # bytes long. Same codes as servo mode: 0 healthy, 1 motor
            # over-temperature, 2 over-current, 3 over-voltage, 4
            # under-voltage, 5 encoder, 6 MOSFET over-temperature, 7 stall.
            error=int(payload[7]),
        )
