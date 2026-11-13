"""A scikit-robot robot interface (``ri``) that drives an actuo bus, without ROS.

Built on :class:`skrobot.interfaces.base.RobotInterfaceBase`, so it has the
same calls as scikit-robot's other interfaces -- ``angle_vector`` (vector or
``{joint: angle}``), ``angle_vector_sequence``, ``wait_interpolation``,
``potentio_vector`` (unclamped), ``outside_limits``, ``servo_on`` /
``servo_off``::

    from skrobot.model import RobotModel
    from actuo.robot_interface import ActuoRobotInterface

    robot = RobotModel.from_urdf("humanoid.urdf")
    ri = ActuoRobotInterface(robot, "humanoid_rcb4.yaml")
    ri.servo_on()                       # holds where it is; never moves it
    ri.angle_vector({"rarm_joint0": 0.5}, 1.0)
    ri.wait_interpolation()

How a command reaches the wire
------------------------------
The base turns a goal into a timed trajectory and hands it to
:meth:`_send_trajectory`, which only stores it as a cubic Hermite trajectory
(:func:`skrobot.interpolation.hermite_trajectory`).
A thread in this process services the bus: each cycle it samples the segments
into a command block and gives that to
:class:`actuo.drivers.servo.ServoSupervisor` -- the same object the ROS 2
servicer uses, unchanged -- which is the only writer of motion frames. So all
of its guards hold here: a limp joint gets no position, enabling is refused if
the command is far from the measured pose, every command is fenced into the
joint's envelope and rate-limited.

Servos that come and go
-----------------------
On a bus that reports presence and allows servos to be missing (``hot_plug``
on the RCB-4mini bus), the config may list more joints than are plugged in. A
joint is **attached** once its servo has answered for
:data:`ATTACH_CYCLES` cycles in a row; only then is its command seeded from
the measured pose and may it be moved or switched on. The controller group
holds the attached joints only, so a vector goal moves those and a
``{joint: angle}`` goal naming a detached joint is an error. A joint whose
servo stops answering is detached at once: switched off, its motion dropped,
its command cleared and its measurement forgotten, so that when it is plugged
back in it comes back limp and starts again from where it then is.
:meth:`ActuoRobotInterface.attached_joints` says which are attached now.

On any other bus every configured joint must answer at start, as before, and
none is ever detached.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from skrobot.interfaces.base import RobotInterfaceBase
from skrobot.interpolation import hermite_trajectory

from actuo.core.contracts import OperatingMode
from actuo.drivers.config import build_bus, load_max_velocities
from actuo.drivers.servo import ServoSupervisor

#: Command slots per joint: [position, velocity, effort], as the supervisor
#: reads them (it was written against the ros2_control shared-memory block).
COMMAND_STRIDE = 3

#: Consecutive cycles a servo must answer before its joint is attached. The
#: first readings of a servo just plugged in can still be the bus's last
#: reading from before it was pulled (a bus may hold back a reading that jumps,
#: until a few agree), and a command seeded from those would be stale.
ATTACH_CYCLES = 5

_LOG = logging.getLogger(__name__)

#: A sent trajectory as seen by one joint: (f, column, end time). ``f(t)`` is
#: the whole trajectory (every joint it moves) on the monotonic clock.
JointTrajectory = Tuple[Callable[[float], np.ndarray], int, float]


class CommandBlock:
    """In-process stand-in for the shared-memory command block.

    :class:`ServoSupervisor` asks it for one thing, ``read_command()``. NaN
    means "no controller owns this slot", exactly as in shared memory.
    """

    def __init__(self, n_joints: int):
        self._command = np.full(n_joints * COMMAND_STRIDE, math.nan)
        self._lock = threading.Lock()

    def read_command(self) -> np.ndarray:
        with self._lock:
            return self._command.copy()

    def set_positions(self, values: Dict[int, float]) -> None:
        with self._lock:
            for index, value in values.items():
                self._command[index * COMMAND_STRIDE] = value

    def position(self, index: int) -> float:
        with self._lock:
            return float(self._command[index * COMMAND_STRIDE])

    def clear(self, index: int) -> None:
        """Make a slot unowned again (every value NaN)."""
        with self._lock:
            start = index * COMMAND_STRIDE
            self._command[start:start + COMMAND_STRIDE] = math.nan


class ActuoRobotInterface(RobotInterfaceBase):
    """Drive a robot from a skrobot model and an actuo bus config, without ROS.

    Parameters
    ----------
    robot : skrobot.model.RobotModel
        The model. Every joint of the config must be in it.
    config : str
        actuo robot config (bus, transport, joints).
    rate : float
        Service rate [Hz]: how often the trajectory is sampled and the bus is
        read and written.
    jump_threshold : float
        [rad] ``servo_on`` refuses a joint whose command is farther than this
        from where it is. ``servo_on`` re-syncs the command first, so this
        only bites if something moved in between.
    transport_overrides : dict, optional
        Replace transport settings of the config (e.g. ``{"channel": ...}``).
    startup_timeout : float
        [s] How long to wait for the joints that must answer at start (every
        joint, or ``required_joints`` on a hot-plug bus) before refusing.
    required_joints : sequence of str, optional
        On a hot-plug bus, the joints that must be attached before the
        constructor returns; the others are attached whenever they answer.
        Default: none. Only a hot-plug bus accepts it: on any other bus every
        joint is required.
    """

    def __init__(self, robot, config: str, rate: float = 50.0,
                 jump_threshold: float = 0.05,
                 transport_overrides: Optional[dict] = None,
                 startup_timeout: float = 5.0,
                 required_joints: Optional[Sequence[str]] = None):
        self.period = 1.0 / rate
        self.bus, self.bus_joints = build_bus(config, transport_overrides)
        set_names = getattr(self.bus, "set_joint_names", None)
        if callable(set_names):
            set_names(self.bus_joints)
        self._index = {name: i for i, name in enumerate(self.bus_joints)}
        model_names = [j.name for j in robot.joint_list]
        missing = [n for n in self.bus_joints if n not in model_names]
        if missing:
            raise ValueError("joints in the config but not in the model: {}"
                             .format(", ".join(missing)))
        #: Whether servos of this bus may be missing and come and go.
        self.hot_plug = bool(getattr(self.bus, "hot_plug", False))
        if required_joints is not None and not self.hot_plug:
            raise ValueError(
                "required_joints needs a hot-plug bus (bus.hot_plug: true); "
                "on this bus every joint is required")
        unknown = [n for n in (required_joints or ()) if n not in self._index]
        if unknown:
            raise KeyError("required_joints not in the config: {}".format(
                ", ".join(unknown)))

        self.supervisor = ServoSupervisor(
            self.bus, self.bus_joints, jump_threshold=jump_threshold,
            command_stride=COMMAND_STRIDE, mode=OperatingMode.POSITION,
            max_velocity=load_max_velocities(config))
        # Position-controlled joints are the ones angle_vector may move; a
        # wheel (velocity mode) is not part of a pose.
        self._position_joints = [
            n for n, mode in zip(self.bus_joints, self.supervisor.modes)
            if mode is OperatingMode.POSITION]
        #: Model joints with no actuator here (mimic fingers, joints on another bus).
        self.unactuated = [n for n in model_names if n not in self._index]

        self.block = CommandBlock(len(self.bus_joints))
        self._bus_measured = [math.nan] * len(self.bus_joints)
        self._trajectories: Dict[int, JointTrajectory] = {}
        self._lock = threading.Lock()
        self._error: Optional[BaseException] = None
        self._running = False
        # Bus indices of the attached joints, and how many cycles in a row
        # each servo has answered.
        self._attached: Set[int] = set()
        self._answering: Dict[int, int] = {}

        super().__init__(robot,
                         controllers={"default_controller": self._position_joints},
                         poll_period=self.period)

        self.bus.transport.open()
        verify = getattr(self.bus, "verify_servos", None)
        if callable(verify):
            verify()
        if self.hot_plug:
            required = list(required_joints or ())
        else:
            required = list(self.bus_joints)
        self._wait_until_attached(
            [self._index[n] for n in required
             if n in self._position_joints], startup_timeout)
        self._running = True
        self._thread = threading.Thread(target=self._run, name="actuo-ri",
                                        daemon=True)
        self._thread.start()
        self.update_robot_state()

    # ------------------------------------------------------------ service loop

    def _answers(self, index: int, state) -> bool:
        """Whether the servo of bus joint ``index`` is answering now."""
        present = getattr(self.bus, "present", None)
        if self.hot_plug and callable(present):
            return bool(present(self.bus.actuators[index].id))
        return state is not None and state.position is not None

    def _read(self, timeout: float) -> None:
        """Read the bus, then attach and detach joints by who answers."""
        states = self.bus.read_batch(timeout=timeout)
        attached, detached = [], []
        with self._lock:
            for i, state in enumerate(states):
                if not self.hot_plug:
                    # Every joint was required at start and none is taken
                    # away: a missed read keeps the last measurement.
                    if state is not None and state.position is not None:
                        self._bus_measured[i] = float(state.position)
                        if i not in self._attached:
                            self._attach(i)
                    continue
                if not self._answers(i, state):
                    # A bus keeps an absent servo's last reading; that is not
                    # where the servo is now.
                    self._bus_measured[i] = math.nan
                    self._answering[i] = 0
                    if i in self._attached:
                        self._detach(i)
                        detached.append(self.bus_joints[i])
                    continue
                if state is not None and state.position is not None:
                    self._bus_measured[i] = float(state.position)
                self._answering[i] = self._answering.get(i, 0) + 1
                if (i not in self._attached
                        and self._answering[i] >= ATTACH_CYCLES
                        and math.isfinite(self._bus_measured[i])):
                    self._attach(i)
                    attached.append(self.bus_joints[i])
        if detached:
            _LOG.warning("detached (not answering): %s", ", ".join(detached))
        if attached and self._running:
            _LOG.info("attached: %s", ", ".join(attached))

    def _attach(self, index: int) -> None:
        """Start commanding a joint from where it is. Holds ``self._lock``."""
        # Adopt the measured pose as the command, so servo_on has nothing to
        # jump to (the ros2_control path does the same on_activate).
        if self.bus_joints[index] in self._position_joints:
            self.block.set_positions({index: self._bus_measured[index]})
        self._attached.add(index)

    def _detach(self, index: int) -> None:
        """Forget a joint whose servo went away. Holds ``self._lock``."""
        self._attached.discard(index)
        self._trajectories.pop(index, None)
        self.block.clear(index)
        name = self.bus_joints[index]
        if name in self.supervisor.enabled_joints:
            ok, message = self.supervisor.disable([name])
            if not ok:
                _LOG.error("could not switch off %s: %s", name, message)

    def _wait_until_attached(self, indices: Sequence[int], timeout: float) -> None:
        deadline = time.monotonic() + timeout
        while True:
            self._read(self.period)
            with self._lock:
                silent = [self.bus_joints[i] for i in indices
                          if i not in self._attached]
            if not silent:
                return
            if time.monotonic() > deadline:
                raise RuntimeError(
                    "no position from {} within {:.1f} s; refusing to start "
                    "(a command seeded from a guessed pose is how a robot "
                    "lurches on servo_on){}".format(
                        ", ".join(silent), timeout, self._why_absent(silent)))

    def _why_absent(self, names: Sequence[str]) -> str:
        reason = getattr(self.bus, "absence_reason", None)
        if not callable(reason):
            return ""
        reasons = ["{}: {}".format(n, reason(self.bus.actuators[self._index[n]].id))
                   for n in names]
        return " -- " + "; ".join(r for r in reasons if not r.endswith(": "))

    # ------------------------------------------------------------ attachment

    def attached_joints(self) -> List[str]:
        """Joints whose servos answer and may be commanded, in config order."""
        with self._lock:
            return [n for i, n in enumerate(self.bus_joints) if i in self._attached]

    def detached_joints(self) -> List[str]:
        """Configured joints that are not attached now, in config order."""
        with self._lock:
            return [n for i, n in enumerate(self.bus_joints)
                    if i not in self._attached]

    @property
    def _controllers(self):
        # Only attached joints are part of the group, so a vector goal moves
        # those and servo_on() switches those on.
        with self._lock:
            attached = set(self._attached)
        return {"default_controller": [n for n in self._position_joints
                                       if self._index[n] in attached]}

    def controllers(self):
        return {k: list(v) for k, v in self._controllers.items()}

    def _resolve(self, av, controller_type):
        if isinstance(av, dict):
            self._refuse_detached(list(av))
        return super()._resolve(av, controller_type)

    def _refuse_detached(self, names: Sequence[str]) -> None:
        with self._lock:
            gone = [n for n in names
                    if n in self._index and self._index[n] not in self._attached]
        if gone:
            raise ValueError("not attached (not answering): {}{}".format(
                ", ".join(gone), self._why_absent(gone)))

    def update_robot_state(self):
        state = self._read_joint_state()
        # A detached joint has no measurement, not its last one.
        for name in self._index:
            if name not in state:
                self._measured.pop(name, None)
        for name, position in state.items():
            self._measured[name] = float(position)
        return dict(self._measured)

    def _sample(self, now: float) -> Dict[int, float]:
        out = {}
        evaluated = {}
        with self._lock:
            for index, (f, column, t_end) in list(self._trajectories.items()):
                if id(f) not in evaluated:     # one evaluation per trajectory
                    evaluated[id(f)] = f(now)
                out[index] = float(evaluated[id(f)][column])
                if now >= t_end:               # finished: the last point holds
                    del self._trajectories[index]
        return out

    def _run(self) -> None:
        next_cycle = time.monotonic()
        try:
            while self._running:
                self.block.set_positions(self._sample(time.monotonic()))
                self.supervisor.service(self.block)
                self._read(self.period)
                next_cycle += self.period
                delay = next_cycle - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_cycle = time.monotonic()
        except BaseException as error:   # noqa: BLE001 -- re-raised to the caller
            self._error = error
            self._running = False

    def _check(self) -> None:
        if self._error is not None:
            raise RuntimeError("the bus service thread stopped: {!r}"
                               .format(self._error)) from self._error

    # ------------------------------------------------- RobotInterfaceBase hooks

    def _send_trajectory(self, controller_type, joint_names, positions,
                         velocities, times, start_time):
        self._check()
        self._refuse_detached(joint_names)
        t_start = time.monotonic() + start_time
        indices = [self._index[name] for name in joint_names]
        with self._lock:
            gone = [n for n, i in zip(joint_names, indices) if i not in self._attached]
            if gone:     # detached since the check above
                raise ValueError("not attached (not answering): {}".format(
                    ", ".join(gone)))
            # Start from the COMMAND, not the measurement: the command is what
            # the servo is tracking, so the path has no step in it.
            start = [self.block.position(i) for i in indices]
            knots = np.concatenate(([t_start], t_start + np.asarray(times, dtype=float)))
            f = hermite_trajectory(
                knots,
                np.vstack((start, positions)),
                np.vstack((np.zeros(len(indices)), velocities)))
            for column, index in enumerate(indices):
                self._trajectories[index] = (f, column, float(knots[-1]))

    def _read_joint_state(self):
        self._check()
        with self._lock:
            return {name: self._bus_measured[i] for name, i in self._index.items()
                    if math.isfinite(self._bus_measured[i])}

    def _is_interpolating(self, controller_type):
        self._check()
        with self._lock:
            return bool(self._trajectories)

    def _cancel(self, controller_type):
        with self._lock:
            self._trajectories.clear()

    def _set_servo(self, joint_names, on):
        self._check()
        names = list(self.bus_joints if joint_names is None else joint_names)
        unknown = [n for n in names if n not in self._index]
        if unknown:
            raise KeyError("no actuator for: {}".format(", ".join(unknown)))
        if on:
            self._refuse_detached(names)
        if not on:
            ok, message = self.supervisor.disable(names)
        else:
            # Re-sync first, and drop any motion of those joints: turning
            # torque on never moves the robot.
            with self._lock:
                for name in names:
                    index = self._index[name]
                    self._trajectories.pop(index, None)
                    if math.isfinite(self._bus_measured[index]):
                        self.block.set_positions({index: self._bus_measured[index]})
            ok, message = self.supervisor.enable(self.block, names)
        if not ok:
            raise RuntimeError(message)

    # ----------------------------------------------------------- lifecycle

    def close(self) -> None:
        """Stop servicing and close the port. Torque is left as it is."""
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self.bus.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
