"""Free drive: a joint pushed by hand, its weight carried by the model.

A joint in free drive leaves its position loop. Every cycle it is sent the
torque the gravity model says holds it, plus:

* a soft hold: a spring towards an anchor, pulling with at most ``hold`` N m,
  which a hand drags along once it pushes the joint ``hold_band`` rad away.
  A hand meets that and the gear friction, not a stiff position loop; let go,
  and the spring keeps the joint against what the model gets wrong (a load it
  does not know, an error in the masses);
* ``friction`` N m along the motion once the joint moves faster than
  ``friction_velocity``, and ``back_emf`` N m per rad/s of joint velocity: at
  low duty a servo's driver brakes the motor a hand is turning, and this
  cancels the voltage the motor generates. Neither ever pushes a joint further
  out past its envelope;
* at rest, ``dither`` N m of alternating sign every cycle, which keeps the
  gears out of static friction so a push starts the joint sooner.

The velocity is estimated from the angle with an alpha-beta filter: a servo's
reported velocity can lag the angle and read non-zero at rest (an STS3215
reads up to ~0.2 rad/s standing still). A joint moving faster than
``max_velocity`` -- a compensation pushing harder than the arm needs -- is
reported as running away, and the caller takes it out of free drive.

Units are N m throughout, so the same parameters mean the same thing on a
joint driven by current (the torque goes out as is) and on one driven by PWM
duty (divided by its ``stall_torque``, the torque at 100 % duty). Values
identified in duty on a FEETECH arm convert with that joint's stall torque.

Ported from attachable_arm's feetech_drivers (free drive in PWM mode, tuned
by hand on an STS3215 arm, 2026-10-16).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


class AlphaBetaFilter:
    """Angle and velocity of several joints, updated once per cycle.

    The steady-state Kalman filter of a constant-velocity model.

    Parameters
    ----------
    alpha : float
        Gain on the angle residual, 0..1. Higher follows the encoder faster
        and passes more of its one-count steps.
    beta : float
        Gain on the velocity, 0..alpha. ``alpha**2 / (2 - alpha)`` is the
        critically damped choice.
    """

    def __init__(self, alpha: float, beta: float):
        if not 0 < beta <= alpha <= 1:
            raise ValueError("need 0 < beta <= alpha <= 1, got {} {}".format(alpha, beta))
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.angle: Optional[np.ndarray] = None
        self.velocity: Optional[np.ndarray] = None

    def reset(self, angles: Sequence[float], mask=None) -> None:
        """Restart the estimate at ``angles``, at rest (all joints, or ``mask``)."""
        angles = np.asarray(angles, dtype=np.float64)
        if self.angle is None or mask is None:
            self.angle = angles.copy()
            self.velocity = np.zeros_like(angles)
            return
        self.angle[mask] = angles[mask]
        self.velocity[mask] = 0.0

    def update(self, angles: Sequence[float], dt: float) -> np.ndarray:
        """Fold in one reading taken ``dt`` seconds after the previous one.

        Returns
        -------
        numpy.ndarray
            Velocity estimate [rad/s].
        """
        angles = np.asarray(angles, dtype=np.float64)
        if self.angle is None:
            self.reset(angles)
            return self.velocity.copy()
        predicted = self.angle + self.velocity * dt
        residual = angles - predicted
        self.angle = predicted + self.alpha * residual
        self.velocity = self.velocity + self.beta * residual / dt
        return self.velocity.copy()


@dataclass(frozen=True)
class FreeDriveJoint:
    """Per-joint free-drive parameters, all in N m.

    Attributes
    ----------
    friction : float
        Added along the motion once the joint moves [N m].
    back_emf : float
        Per rad/s of joint velocity [N m s/rad].
    hold : float
        Soft-hold pull at full stretch (``hold_band``) [N m].
    stall_torque : float or None
        Torque at 100 % PWM duty [N m]; needed only on a joint driven by PWM.
    gravity_scale : float
        Factor on the model's holding torque. 1 trusts the URDF; a measured
        ratio corrects a model (or a stall torque) that is off for this joint
        -- e.g. the holding duty in position mode, approached from both sides,
        over the model's.
    """

    friction: float = 0.0
    back_emf: float = 0.0
    hold: float = 0.0
    stall_torque: Optional[float] = None
    gravity_scale: float = 1.0

    def __post_init__(self):
        for name in ("friction", "back_emf", "hold"):
            if getattr(self, name) < 0:
                raise ValueError("free_drive {} must be >= 0, got {}".format(
                    name, getattr(self, name)))
        if not self.gravity_scale >= 0:
            raise ValueError("free_drive gravity_scale must be >= 0, got {}".format(
                self.gravity_scale))
        if self.stall_torque is not None and not self.stall_torque > 0:
            raise ValueError("free_drive stall_torque must be > 0, got {}".format(
                self.stall_torque))


@dataclass(frozen=True)
class FreeDriveConfig:
    """Free-drive parameters shared by the joints of one bus.

    Attributes
    ----------
    joints : dict of str to FreeDriveJoint
        The joints that may enter free drive.
    hold_band : float
        Stretch past which a push drags the soft hold's anchor [rad].
    friction_velocity : float
        Speed above which a joint counts as moving [rad/s].
    velocity_filter : tuple of float
        Alpha-beta gains of the velocity estimate.
    dither : float
        Torque alternated every cycle at rest [N m]; 0 = none.
    max_velocity : float
        Speed above which a joint counts as running away [rad/s].
    duty_limit : float
        Cap on the PWM duty of a joint driven by PWM, 0..1.
    """

    joints: Dict[str, FreeDriveJoint]
    hold_band: float = 0.02
    friction_velocity: float = 0.05
    velocity_filter: Tuple[float, float] = (0.35, 0.06)
    dither: float = 0.0
    max_velocity: float = 2.0
    duty_limit: float = 0.3

    def __post_init__(self):
        if not self.hold_band > 0:
            raise ValueError("free_drive hold_band must be > 0")
        if not self.friction_velocity > 0 or not self.max_velocity > 0:
            raise ValueError("free_drive friction_velocity and max_velocity must be > 0")
        if self.dither < 0:
            raise ValueError("free_drive dither must be >= 0")
        if not 0 < self.duty_limit <= 1:
            raise ValueError("free_drive duty_limit must be in (0, 1]")


class FreeDrive:
    """The free-drive torques of the joints of one bus.

    Parameters
    ----------
    config : FreeDriveConfig
    joint_names : sequence of str
        The bus's joints, in bus order.
    envelopes : sequence of (float, float) or None
        Each joint's envelope [rad], or None; friction and back-EMF help never
        push a joint further out past it.
    """

    def __init__(self, config: FreeDriveConfig, joint_names: Sequence[str],
                 envelopes: Sequence[Optional[Tuple[float, float]]]):
        unknown = set(config.joints) - set(joint_names)
        if unknown:
            raise KeyError("free_drive names unknown joints: {}".format(sorted(unknown)))
        self.config = config
        self.joint_names = list(joint_names)
        self.envelopes = list(envelopes)
        n = len(self.joint_names)
        self.filter = AlphaBetaFilter(*config.velocity_filter)
        self.anchor = np.zeros(n)
        self._dither_sign = 1.0

    def joint(self, index: int) -> Optional[FreeDriveJoint]:
        """The parameters of joint ``index``, or None if it has none."""
        return self.config.joints.get(self.joint_names[index])

    def start(self, index: int, positions: Sequence[float]) -> None:
        """Joint ``index`` enters free drive at ``positions[index]``."""
        positions = np.asarray(positions, dtype=np.float64)
        self.anchor[index] = positions[index]
        mask = np.zeros(len(self.joint_names), dtype=bool)
        mask[index] = True
        self.filter.reset(positions, mask)

    def torques(self, active: Sequence[int], positions: Sequence[float],
                hold_torques: Sequence[float], dt: float
                ) -> Tuple[Dict[int, float], List[Tuple[int, float]]]:
        """One cycle of free drive.

        Parameters
        ----------
        active : sequence of int
            Joints in free drive.
        positions : sequence of float
            Every joint's angle this cycle [rad].
        hold_torques : sequence of float
            Every joint's gravity-holding torque [N m] (what it must exert).
        dt : float
            Seconds since the previous call.

        Returns
        -------
        torques : dict of int to float
            The torque each active joint is to exert [N m].
        runaway : list of (int, float)
            Active joints moving faster than ``max_velocity``, with their
            speed; they get no torque here and must leave free drive.
        """
        cfg = self.config
        positions = np.asarray(positions, dtype=np.float64)
        velocities = self.filter.update(positions, max(dt, 1e-3))
        self._dither_sign = -self._dither_sign
        out: Dict[int, float] = {}
        runaway: List[Tuple[int, float]] = []
        for k in active:
            joint = self.joint(k)
            position, velocity = float(positions[k]), float(velocities[k])
            if abs(velocity) > cfg.max_velocity:
                runaway.append((k, velocity))
                continue
            torque = joint.gravity_scale * float(hold_torques[k])
            band = cfg.hold_band
            self.anchor[k] = min(max(self.anchor[k], position - band), position + band)
            torque += joint.hold * (self.anchor[k] - position) / band
            envelope = self.envelopes[k]
            inside = envelope is None or (
                (velocity > 0 and position < envelope[1])
                or (velocity < 0 and position > envelope[0]))
            if abs(velocity) > cfg.friction_velocity and inside:
                torque += math.copysign(joint.friction, velocity)
                torque += joint.back_emf * velocity
            else:
                torque += self._dither_sign * cfg.dither
            out[k] = torque
        return out, runaway

    def duty(self, index: int, torque: float) -> float:
        """PWM duty for ``torque`` on joint ``index``, capped at ``duty_limit``."""
        stall = self.joint(index).stall_torque
        if stall is None:
            raise ValueError("{} is driven by PWM but its free_drive has no "
                             "stall_torque".format(self.joint_names[index]))
        limit = self.config.duty_limit
        return min(max(torque / stall, -limit), limit)


def free_drive_config(document: dict, path: str) -> Optional[FreeDriveConfig]:
    """The free-drive configuration of a robot config, or None if it has none.

    The bus-wide parameters are the top-level ``free_drive`` block; a joint
    takes part when its entry (or ``joint_defaults``) has a ``free_drive``
    block, which gives its own parameters in N m.

    Parameters
    ----------
    document : dict
        The expanded robot config (``actuo.drivers.config.load_document``).
    path : str
        Its file, for error messages.
    """
    joints = {}
    for entry in document.get("joints", []):
        spec = entry.get("free_drive")
        if spec is None:
            continue
        unknown = set(spec) - {"friction", "back_emf", "hold", "stall_torque",
                               "gravity_scale"}
        if unknown:
            raise ValueError("{}: joint {} free_drive has unknown keys {}".format(
                path, entry.get("name"), sorted(unknown)))
        joints[str(entry["name"])] = FreeDriveJoint(
            friction=float(spec.get("friction", 0.0)),
            back_emf=float(spec.get("back_emf", 0.0)),
            hold=float(spec.get("hold", 0.0)),
            gravity_scale=float(spec.get("gravity_scale", 1.0)),
            stall_torque=(float(spec["stall_torque"]) if "stall_torque" in spec
                          else None))
    block = document.get("free_drive")
    if block is None:
        if joints:
            raise ValueError("{}: joints have free_drive parameters but there is no "
                             "top-level free_drive block (gravity, ...)".format(path))
        return None
    if not joints:
        raise ValueError("{}: a free_drive block but no joint has free_drive "
                         "parameters".format(path))
    known = {"gravity", "unmounted_links", "backend", "hold_band",
             "friction_velocity", "velocity_filter", "dither", "max_velocity",
             "duty_limit"}
    unknown = set(block) - known
    if unknown:
        raise ValueError("{}: free_drive has unknown keys {}".format(path, sorted(unknown)))
    kwargs = {k: float(block[k]) for k in ("hold_band", "friction_velocity", "dither",
                                           "max_velocity", "duty_limit") if k in block}
    if "velocity_filter" in block:
        kwargs["velocity_filter"] = tuple(float(v) for v in block["velocity_filter"])
    return FreeDriveConfig(joints=joints, **kwargs)
