"""CubeMars AK-series actuator, 3.0-generation force control (impedance).

THREE PROTOCOLS, NOT TWO
------------------------
:mod:`cubemars` speaks the servo protocol. :mod:`cubemars_mit` speaks the
2.0-generation MIT protocol, which is a separate firmware reached only through
the Windows upper computer's Mode Switch page. This module speaks the third:
the 3.0 generation folds impedance control into the servo protocol as one more
control mode, addressed by an extended arbitration id exactly like the servo
packets [2]_ section 4.2. There is no mode to switch and no firmware to flash::

    "the upper computer CubeMarsTool has been fully upgraded, and there is no
     need to switch between servo mode and force control mode"     -- [2]_ p.3

Which generation a motor is depends on the motor, and a bus can hold both.

WHAT MAKES THIS A SUBCLASS RATHER THAN A THIRD SEPARATE MODEL
-------------------------------------------------------------
2.0 MIT is a different model from servo because the two cannot both be right
about one motor: different firmware, standard ids instead of extended, a
7-byte reply instead of a periodic 8-byte status, and the manual warns that
using the wrong one can burn the driver board.

None of that applies here. 3.0 force control is the *same firmware* as 3.0
servo, and its feedback is the identical periodic status frame [2]_ section
4.3.1 -- position as output-shaft degrees*10, speed as ERPM/10, current as
A*100, driver-board temperature, error code. So every part of
:class:`~actuo.drivers.actuators.cubemars.CubeMarsActuator` that reads the
motor is correct here unchanged, gear-window recovery included, and
duplicating it would mean maintaining the same decode twice.

Only the command differs, so only the command is overridden.

MEASURED ON A BIPED'S CUBEMARS LEGS, 2026-09-12
-----------------------------------------------
The twelve leg motors were probed with this packet at ``kp = 0``, ``t_ff = 0``
and a rising velocity target, watching the status frame's current field.
Current, not motion, is the detector: a driver that implements the packet
answers a stalled velocity target with torque whether or not the joint turns.

===========  =====  ================================  =========
model        count  joints                            responded
===========  =====  ================================  =========
AK10-9           2  ankle pitch                       yes
AK70-9           4  hip yaw, ankle roll               yes
AK80-64          6  hip pitch, hip roll, knee         **no**
===========  =====  ================================  =========

Two controls, both negative: the same eight bytes sent as packet 7 did
nothing, and packet 8 with ``kd = 0`` did nothing. So what acted was packet
8's kd term, not drift and not the bytes alone.

The AK80-64 is a 2.0-generation motor -- CubeMars ships it separate MIT and
servo binaries -- and belongs to :mod:`cubemars_mit`, not here.

THE BYTE ORDER IS NOT THE 2.0 MIT ORDER
---------------------------------------
2.0 MIT leads with position; 3.0 leads with kp. The same eight bytes mean
different things to the two generations, so :mod:`cubemars_mit`'s packing is
not reusable here even though the five fields are the same five fields::

    byte 0  kp high 8
    byte 1  kp low 4 | kd high 4
    byte 2  kd low 8
    byte 3  position high 8
    byte 4  position low 8
    byte 5  speed high 8
    byte 6  speed low 4 | torque high 4
    byte 7  torque low 8

THE FIELD RANGES ARE PER MOTOR AND ARE NOT OPTIONAL
---------------------------------------------------
Every field is a fixed-point number scaled into its own range, so a wrong
range is a silently wrong command rather than an error: send 1.0 rad with
``p_max`` set to 50 when the motor uses 12.56 and the motor moves a quarter as
far as asked. They are therefore required config, never defaulted.

Published for the 3.0 line in [2]_ section 4.2. Position is 12.56 on every
model; only speed and torque vary::

    model     Kv    position rad   speed rad/s   torque N*m
    AK10-9     60     +/- 12.56     +/- 28        +/- 54
    AK60-6     80     +/- 12.56     +/- 60        +/- 12
    AK70-9     60     +/- 12.56     +/- 30        +/- 32
    AK80-9    100     +/- 12.56     +/- 65        +/- 18
    AKE60-8    80     +/- 12.56     +/- 40        +/- 15
    AKE80-8    30     +/- 12.56     +/- 20        +/- 35

Note the Kv column. The table is per *variant*, not per model name: the
AK10-9 also ships as a Kv100, and the 2.0 Kv100 firmware carries +/- 50 rad/s
where this Kv60 row says +/- 28. Read the motor's own label before trusting a
row by model name alone.

``kp`` is 0..500 and ``kd`` is 0..5 on every model and both generations.

.. [1] CubeMars, AK Series Driver and Control Manual v1.0.14 (2.0 generation).
   https://img.cubemars.com/products/cubemars-product-parameter/AK-Series-Driver-and-Control-Manual-v1.0.14.pdf
.. [2] CubeMars, AK Series Module Product Manual V3, 2024-12-11, sections 4.2
   (force control protocol and ranges) and 4.3.1 (CAN upload message).
   https://www.cubemars.com/images/file/20241217/1734428582786471.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

from actuo.core.contracts import (
    ActuatorCommand,
    Capabilities,
    Frame,
    OperatingMode,
)
from actuo.core.registry import register_actuator

from actuo.drivers.actuators.cubemars import CubeMarsActuator, CubeMarsCalibration
from actuo.drivers.actuators.cubemars_mit import (
    KD_MAX,
    KD_MIN,
    KP_MAX,
    KP_MIN,
    float_to_uint,
    uint_to_float,
)

#: The force-control mode's arbitration id is ``motor_id | (8 << 8)``, extended,
#: alongside the servo modes 0..6 ([2]_ section 4.2). It is the only value in
#: that field the 2.0 generation does not answer, which is what makes probing
#: it a generation test.
FORCE_CONTROL_PACKET = 8

#: Field widths, [2]_ section 4.2. Position gets 16 bits, everything else 12.
_POSITION_BITS = 16
_SPEED_BITS = 12
_GAIN_BITS = 12
_TORQUE_BITS = 12


@dataclass
class CubeMarsV3Calibration(CubeMarsCalibration):
    """A :class:`CubeMarsCalibration` plus the force-control wire ranges.

    Inherits the whole servo-mode calibration because the *feedback* is
    unchanged: ``zero_offset_counts`` is still output-shaft degrees*10 read at
    the model's zero, ``window_offset`` still names which gear window the
    encoder woke up in, and ``gear_window_deg`` is still the quantum of this
    actuator's origin ambiguity.

    Attributes
    ----------
    p_max, v_max, t_max : float
        Force-control field ranges for THIS motor variant, symmetric about
        zero, in rad, rad/s and N*m at the output shaft. Required; see the
        module docstring for the published table and for why a guessed range
        is worse than a missing one.
    kp, kd : float
        Impedance gains in N*m/rad and N*m*s/rad at the output shaft. They go
        on the wire every frame, because they are constants of the joint
        rather than a channel anything upstream modulates.
    torque_limit_nm : float, optional
        Clamp applied to the feedforward torque before scaling. Not the
        motor's own limit, which it enforces regardless of what we ask.

    Notes
    -----
    **The force-control position field shares the status frame's datum and
    scale.** Nothing in [2]_ says so: one firmware and one encoder make it
    likely, but the two fields are expressed differently (+-12.56 rad against
    +-3200 deg) and the manual never relates them. So it was measured, on a
    biped's right hip yaw (AK70-9), 2026-09-12:

    * **Datum.** Commanded its own measured position at kp up to 10. It moved
      0.10 deg and drew under 0.2 A, where a datum that disagreed would have
      produced torque proportional to the disagreement.
    * **Scale.** Driven to +5, +10, +15, -10 and -15 deg at kp 100, it reached
      ratios of 1.020, 0.990, 0.993, 0.950 and 0.980 -- slope **0.9830**
      through the origin. A ``p_max`` that disagreed with the motor's own
      would show as a constant ratio away from 1, not as this.

    What is left is a 0.18 deg mean shortfall, constant in angle rather than
    proportional to the step, which is what friction looks like: 0.31 N*m at
    kp 100. Holding the datum alone does not prove the scale, which is why
    both halves are there -- an early run with steps too small, and with the
    position read after the frames stopped rather than while they were still
    going, reported a slope of 0.75 that was entirely settling and stiction.
    """

    p_max: float = 0.0
    v_max: float = 0.0
    t_max: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    torque_limit_nm: Optional[float] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        for name in ("p_max", "v_max", "t_max"):
            if getattr(self, name) <= 0.0:
                raise ValueError(
                    "{} is required and must be positive: force control scales "
                    "every field into this motor's own range, so an absent "
                    "range is a silently wrong command rather than an error. "
                    "See this module's docstring for the published table."
                    .format(name))
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

    def rad_to_wire_rad(self, model_rad: float) -> float:
        """Model radians -> the radians the position field carries.

        Routed through :meth:`rad_to_counts` rather than reimplemented, so the
        datum, the direction flip and the gear-window offset are applied in
        exactly one place. Counts are output-shaft degrees*10.
        """
        return math.radians(self.rad_to_counts(model_rad) * 0.1)

    def wire_rad_to_rad(self, wire_rad: float) -> float:
        """Inverse of :meth:`rad_to_wire_rad`."""
        return self.counts_to_rad(math.degrees(wire_rad) * 10.0)


@register_actuator("cubemars_v3")
class CubeMarsV3Actuator(CubeMarsActuator):
    """One 3.0-generation AK motor driven by impedance commands.

    The motor computes, in its own loop and far faster than a bus round trip::

        tau = kp * (p_des - p) + kd * (v_des - v) + t_ff

    which is the control law an RL locomotion policy is trained against, and
    the one servo mode cannot express: its ``SET_POS`` runs the motor's own
    position loop with the motor's own gains.

    Registered as a separate plugin from ``cubemars`` even though the same
    firmware answers both, because a motor should be driven by one loop at a
    time. Sending servo position commands and force-control commands to one
    motor concurrently would have two controllers fighting over it, and the
    config is where that choice should be visible.
    """

    def __init__(self, actuator_id, calibration: CubeMarsV3Calibration) -> None:
        super().__init__(actuator_id, calibration)
        self.calibration: CubeMarsV3Calibration = calibration

    @property
    def capabilities(self) -> Capabilities:
        """POSITION only, and that is the whole point of the mode.

        The frame does carry a feedforward torque, but a caller that wanted
        torque control would be asking for kp = kd = 0, which is a different
        control law wearing this one's clothes. Declaring CURRENT here would
        invite exactly that.

        Sense is delegated to the servo-mode status frame, so ``velocity``
        still needs ``pole_pairs`` and ``current`` still needs
        ``torque_constant`` -- the force-control command changes what we send,
        not what the motor reports back.
        """
        base = super().capabilities
        return Capabilities(
            modes=(OperatingMode.POSITION,),
            has_position_sense=base.has_position_sense,
            has_velocity_sense=base.has_velocity_sense,
            has_current_sense=base.has_current_sense,
            has_temperature_sense=base.has_temperature_sense,
            multiturn=base.multiturn,
        )

    def matches(self, frame: Frame) -> bool:
        """Whether ``frame`` is this motor's status frame.

        Everything the servo model checks, plus one thing it cannot know:
        packet 8 is a COMMAND, and the servo model's command list stops at 6.

        This matters because a force-control frame carries this motor's node
        id in the same low byte and is also eight bytes long, so it satisfies
        every other test. On a backend that echoes what it transmits
        (python-can's gs_usb does) the actuator would be handed its own
        command back and decode it as state -- byte 0, the kp high byte, read
        as the position high byte.
        """
        if not super().matches(frame):
            return False
        return (int(frame.address) >> 8) != FORCE_CONTROL_PACKET

    def encode_mode(self, mode: OperatingMode) -> List[Frame]:
        """Nothing to arm: the control mode rides in each command's id.

        Unlike 2.0 MIT, which ignores commands until ``ENTER_MOTOR_MODE`` has
        landed and so looks exactly like a wiring fault until it does.
        """
        self.capabilities.require(mode)
        return []

    def encode_release(self) -> Frame:
        """Zero torque, by zeroing every term of the law rather than the target.

        There is no exit packet here -- 2.0 MIT has ``EXIT_MOTOR_MODE``, this
        generation does not -- so release has to be synthesised. With kp, kd
        and t_ff all zero the motor's own expression evaluates to zero torque
        whatever the position and velocity targets say, which makes this a
        real release rather than a command that merely asks for no motion.

        Confirmed 2026-09-12 on a biped's AK70-9 hip yaw: it drew 0.25 A while
        driven and returned to 0.00 A on these frames.

        Simply *stopping* commands is not release. Nothing in [2]_ documents a
        force-control watchdog, and assuming one is assuming a behaviour we
        cannot read back.
        """
        return self._impedance_frame(
            p_des_wire=self.calibration.rad_to_wire_rad(0.0),
            v_des=0.0, kp=0.0, kd=0.0, t_ff=0.0)

    def encode_command(self, command: ActuatorCommand) -> Frame:
        """Pack one impedance command.

        The velocity target stays 0 unless a feedforward is given, and that is
        deliberate: the damping term is meant to pull the MEASURED velocity to
        zero. Feeding the derivative of the position target instead turns the
        law into something the policy was not trained against.
        """
        self.capabilities.require(command.mode)
        if command.position is None:
            raise ValueError("POSITION mode needs a position")
        self._check_position_limit(command.position)

        cal = self.calibration
        t_ff = command.current_ff or 0.0
        if cal.torque_limit_nm is not None:
            t_ff = min(max(t_ff, -cal.torque_limit_nm), cal.torque_limit_nm)

        return self._impedance_frame(
            p_des_wire=cal.rad_to_wire_rad(command.position),
            # Direction flips a rate and a torque, but not an offset, so these
            # two take the sign alone where the position took the whole datum.
            v_des=cal.direction * (command.velocity_ff or 0.0),
            kp=cal.kp,
            kd=cal.kd,
            t_ff=cal.direction * t_ff,
        )

    def decode_command(self, frame: Frame) -> dict:
        """What a packet-8 frame actually carries, back in model terms.

        The inverse of :meth:`_impedance_frame`, quantisation included. Same
        return shape as the 2.0 MIT actuator's, so one log serves both.
        """
        payload = frame.payload
        cal = self.calibration
        kp_int = (payload[0] << 4) | (payload[1] >> 4)
        kd_int = ((payload[1] & 0x0F) << 8) | payload[2]
        p_int = (payload[3] << 8) | payload[4]
        v_int = (payload[5] << 4) | (payload[6] >> 4)
        t_int = ((payload[6] & 0x0F) << 8) | payload[7]
        return {
            "position": cal.wire_rad_to_rad(uint_to_float(
                p_int, -cal.p_max, cal.p_max, _POSITION_BITS)),
            "velocity": cal.direction * uint_to_float(
                v_int, -cal.v_max, cal.v_max, _SPEED_BITS),
            "kp": uint_to_float(kp_int, KP_MIN, KP_MAX, _GAIN_BITS),
            "kd": uint_to_float(kd_int, KD_MIN, KD_MAX, _GAIN_BITS),
            "torque": cal.direction * uint_to_float(
                t_int, -cal.t_max, cal.t_max, _TORQUE_BITS),
        }

    def _impedance_frame(self, p_des_wire: float, v_des: float, kp: float,
                         kd: float, t_ff: float) -> Frame:
        """Build one packet-8 frame from quantities already in wire terms."""
        cal = self.calibration
        kp_int = float_to_uint(kp, KP_MIN, KP_MAX, _GAIN_BITS)
        kd_int = float_to_uint(kd, KD_MIN, KD_MAX, _GAIN_BITS)
        p_int = float_to_uint(p_des_wire, -cal.p_max, cal.p_max, _POSITION_BITS)
        v_int = float_to_uint(v_des, -cal.v_max, cal.v_max, _SPEED_BITS)
        t_int = float_to_uint(t_ff, -cal.t_max, cal.t_max, _TORQUE_BITS)

        payload = bytes([
            (kp_int >> 4) & 0xFF,
            ((kp_int & 0x0F) << 4) | ((kd_int >> 8) & 0x0F),
            kd_int & 0xFF,
            (p_int >> 8) & 0xFF,
            p_int & 0xFF,
            (v_int >> 4) & 0xFF,
            ((v_int & 0x0F) << 4) | ((t_int >> 8) & 0x0F),
            t_int & 0xFF,
        ])
        return Frame(
            payload=payload,
            address=self.node_id | (FORCE_CONTROL_PACKET << 8),
            meta={"extended": True},
        )
