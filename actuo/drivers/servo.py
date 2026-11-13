"""Servo (torque) supervision: the thing that decides whether the robot holds.

ROS-free on purpose, so the safety rule is unit-testable without a middleware,
a bus, or a robot. The ROS node is a thin shell around this.

The hazard this exists for
--------------------------
Torque OFF, operator hand-moves the arm, torque ON -> the joint slams to
wherever the controller was still commanding. JTC *holds* its command; it does
not track measurement. So the stale command and the real pose diverge silently
while the robot is limp, and enabling closes that gap at full authority.

"Be careful" cannot fix that. Three structural properties do:

1. **Fully OFF cannot emit a position.** :meth:`ServoSupervisor.service` reads
   the shared-memory command *only* when at least one joint is enabled. With
   every joint off there is no reachable path from a controller's setpoint to a
   frame -- not a flag that is checked, an absence of code.
2. **Enabling refuses by default.** It is allowed only if the controller is
   already commanding where the robot actually is. Never clamps, never ramps
   silently, never "just this once".
3. **The fix points at the controller, not the robot.** The way to satisfy the
   guard is to re-sync the CONTROLLER to the robot (deactivate/reactivate the
   trajectory controller, which adopts the measured pose). The robot never
   leaps to meet a stale command. Direction is the whole safety argument.

Per-joint torque, and what it costs
-----------------------------------
Torque is owned per joint, so an operator can free one joint and leave the rest
holding. Be honest about what that does to property 1: with SOME joint enabled
the command block *is* read, and a disabled joint is kept limp by a checked
condition (its slot becomes ``None`` -> release) rather than by the absence of
any code that could command it. The "absence of code" guarantee survives only
for the all-off case; per-joint is inherently a decision, and a decision is a
flag. That weakening is inherent to the feature, not an oversight -- so it is
written down here rather than glossed. Property 2 is unaffected: the jump guard
was always per joint and stays so.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from actuo.core.contracts import ActuatorCommand, OperatingMode


def resolve_modes(actuators, joint_names, mode):
    """Which mode each joint is actually in, given the bus-wide default.

    A joint whose actuator cannot do ``mode`` -- a continuous-rotation wheel
    servo has no position at all -- takes the one mode it does support. Same
    board, same packet, different quantities.

    Module-level rather than a method because two callers need the SAME answer:
    :class:`ServoSupervisor`, which reads a joint's shm slot by it, and the
    servicer, which builds the pose filter over the joints this says are in
    POSITION mode. Two copies of this rule that disagree would hand the filter a
    vector mis-aligned by one joint, silently.

    Raises
    ------
    ValueError
        If an actuator supports several modes but not ``mode``. Then the answer
        would have to be guessed, and guessing which quantity a controller is
        commanding is not a recoverable kind of wrong.
    """
    out = []
    for name, actuator in zip(joint_names, actuators):
        supported = actuator.capabilities.modes
        if mode in supported:
            out.append(mode)
        elif len(supported) == 1:
            out.append(supported[0])
        else:
            raise ValueError(
                "{}: its actuator does not support {} mode and supports more "
                "than one alternative ({}), so which quantity the controller is "
                "commanding cannot be inferred. Say it in config.".format(
                    name, mode.name, ", ".join(m.name for m in supported)))
    return out


#: Seconds between two announcements of the envelope clamp for one joint.
RECOVER_REPEAT_S = 5.0


@dataclass(frozen=True)
class JointServo:
    """What a supervisor knows about one joint right now."""

    name: str
    enabled: bool
    measured: float           # rad; NaN if this actuator has no position sense
    commanded: float          # what the controller asks for, in `mode`'s units
    error: int
    error_name: str

    #: Which quantity ``commanded`` is. Per JOINT, not per robot: a Kondo
    #: humanoid on wheels has continuous-rotation wheel servos taking a SPEED and its arm joints take a
    #: pose, and both hang off the same board and ride in the same packet. A
    #: single robot-wide mode would have to lie about one of them.
    mode: OperatingMode = OperatingMode.POSITION

    # Load and thermals. NaN means the driver did not report it -- never 0.
    # A missing temperature carried as 0.0 reads as a real and reassuring
    # 0 degC; a missing current as a real 0 A. Both are silent degradation,
    # which this package never allows.
    temperature: float = math.nan       # degC
    velocity: float = math.nan          # rad/s at the output shaft
    current_a: float = math.nan         # motor amps, as the wire reports them
    effort: float = math.nan            # N*m at the output shaft
    #: Device-specific extras by name, straight from ActuatorState.aux, so a
    #: new channel needs no change here: telemetry is named channels, never a
    #: struct that has to be widened.
    aux: Dict[str, float] = field(default_factory=dict)
    #: Whether the actuator answers (a hot-plug bus can say no), and why not.
    present: bool = True
    absence_reason: str = ""

    @property
    def mismatch(self) -> float:
        """rad the joint would travel the instant torque is applied.

        Meaningful in POSITION mode only. Elsewhere it subtracts a speed from an
        angle, so it is NaN rather than a number that invites comparison -- and
        it is NaN too for an actuator with no position sense, because there is
        nothing to be far from.
        """
        if self.mode is not OperatingMode.POSITION:
            return math.nan
        return self.commanded - self.measured


class ServoSupervisor:
    """Owns torque state per joint for one bus. Every joint starts OFF.

    Starting OFF is deliberate: a supervisor that came up enabled would apply
    torque before any operator asked, using whatever the controller happened to
    be commanding. The safe state is the default state.
    """

    #: Which shm command slot each mode reads. The block is stride-3
    #: ``[position, velocity, effort]`` per joint, matching the C++ hardware
    #: component's kCmdStride (see :mod:`actuo.drivers.shm`), and the C++ side fills only the slots the URDF declares.
    _SLOT = {OperatingMode.POSITION: 0,
             OperatingMode.VELOCITY: 1,
             OperatingMode.CURRENT: 2}

    def __init__(self, bus, joint_names: Sequence[str], jump_threshold: float,
                 command_stride: int, error_names: Optional[Dict[int, str]] = None,
                 mode: OperatingMode = OperatingMode.POSITION,
                 velocity_threshold: float = 0.05,
                 command_filter=None, release_period: float = 0.0,
                 max_velocity: Optional[Dict[str, float]] = None):
        #: Optional last look at the command before it reaches the wire (an
        #: ``actuo.core.contracts.CommandFilter``). POSITION only, and that is
        #: not a limitation to lift later: such a filter judges a POSE, and in
        #: VELOCITY or CURRENT mode the command vector is speeds or torques.
        #: Handing those to something that asks "would this pose collide" is
        #: comparing units, which is how a guard starts reporting confident
        #: nonsense.
        self.command_filter = command_filter
        #: Cycles in which the filter changed the command; published so a robot
        #: that is quietly being held back does not look like one that is
        #: tracking badly.
        self.filtered_cycles = 0
        self._filter_gated = False
        #: One-shot description of the last gate/ungate transition, for the
        #: servicer to print. A filter that silently held the robot back would
        #: look exactly like a robot tracking badly -- and when tuning by trial
        #: and error that is the difference between "why is it not moving" and
        #: "ah, it stopped there".
        self._filter_event = None
        #: The last vector the filter allowed. This, not the controller's
        #: command, is what a release adopts as safe -- see :meth:`release_filter`.
        self._last_applied = None
        self.bus = bus
        self.joint_names = list(joint_names)
        self.jump_threshold = jump_threshold
        self.command_stride = command_stride
        self._error_names = dict(error_names or {})
        if mode not in self._SLOT:
            raise ValueError(
                "unsupported servo mode {!r}; this supervisor knows {}".format(
                    mode, ", ".join(m.name for m in self._SLOT)))
        #: Which quantity the controller is commanding. Everything below that
        #: reads a number out of shm depends on this, and so does what "enabling
        #: would surprise you" means -- see :meth:`would_jump`.
        self.mode = mode
        #: The mode of EACH joint, which is not always ``mode``. A joint whose
        #: actuator cannot do the default -- a continuous-rotation wheel servo
        #: has no position at all -- takes the one mode it does support. Same
        #: board, same packet, different quantities; see :class:`JointServo`.
        self.modes: List[OperatingMode] = resolve_modes(
            self.bus.actuators, self.joint_names, mode)
        #: Indices of the joints a POSE filter may look at. A collision filter
        #: asks "would this pose collide"; handing it a wheel's speed would be
        #: comparing units, which is how a guard starts reporting confident
        #: nonsense. So wheels are simply not shown to it -- and they cannot
        #: collide with anything by turning.
        self._filter_indices = [i for i, m in enumerate(self.modes)
                                if m is OperatingMode.POSITION]
        if command_filter is not None and not self._filter_indices:
            raise ValueError(
                "a command filter judges a POSE, but no joint on this bus is "
                "in POSITION mode. Refusing rather than filtering the wrong "
                "quantity.")
        #: rad/s. In VELOCITY mode the enable hazard is not a distant position,
        #: it is a NON-ZERO commanded speed: torque comes on and the joint simply
        #: starts moving. So this is the velocity-mode counterpart of
        #: ``jump_threshold``, and it is much easier to satisfy -- a controller
        #: that has not been asked to move commands 0.
        self.velocity_threshold = velocity_threshold
        #: Control cycles serviced since start. Published, so an endurance run
        #: can tell "the servicer restarted" apart from "the robot went quiet":
        #: a counter that jumps back to 0 mid-bag is the former, and the two
        #: look identical in the joint angles alone.
        self.cycles = 0
        #: While EVERY joint is off, re-send the release at most this often,
        #: seconds (0 = every cycle). The release is latched in the motors
        #: (MIT stays out of motor mode, a zero-gain 3.0 frame holds until the
        #: next command), so repeating it is only insurance against a lost
        #: frame and, on ask-only motors, the poll that keeps their state
        #: fresh. At 200 Hz every cycle was an EXIT to each 2.0 MIT motor,
        #: which answers on the SAME id: two bursts bunched by scheduling
        #: jitter put the next EXIT on the wire during the last answer, and a
        #: bus of MIT leg drives logged ~0.8 error frames a second while idle
        #: and none while held (2026-09-29). The first cycle after the last joint goes
        #: off always sends.
        self.release_period = float(release_period)
        self._last_release: Optional[float] = None
        # Joints that come up DRIVEN, because for them the hazard this class
        # exists for cannot happen. The whole point of starting released is
        # that a joint holding a stale position command gets thrown to it the
        # instant torque arrives -- and a continuous-rotation wheel has no
        # position at all, so there is no stale pose, nothing to be far from,
        # and nothing to be thrown to. Its command is a SPEED, and a speed of
        # zero is a wheel standing still.
        #
        # This also restores how these servos have always worked: on the Kondo
        # board a wheel takes its speed in the same packet as everything else
        # and was never gated behind a torque-enable step. Requiring one made a
        # base that would not move until an operator pressed a button whose
        # entire purpose does not apply to it.
        #
        # Releasing a wheel is still available and still means something --
        # limp is free-rolling, which is what you want to push the robot by
        # hand -- it is simply no longer the state it starts in.
        self._default_on: Set[str] = {
            name for name, actuator in zip(self.joint_names, self.bus.actuators)
            if not actuator.capabilities.has_position_sense}
        # Names of joints holding torque. Deliberately NOT a bool: there is no
        # `enabled` property, because a caller that reads one would have to guess
        # whether it meant "all" or "any" -- and both callers guessing
        # differently is how a UI ends up showing a limp robot as held.
        self._on: Set[str] = set(self._default_on)
        # Joints currently being walked back inside their envelope, so entering
        # and leaving that state is said once rather than every cycle.
        self._recovering: Set[int] = set()
        # Joint index -> when the clamp was last announced.
        self._recover_said: Dict[int, float] = {}
        #: rad/s a POSITION joint's command may move, by joint name. The last
        #: line against a command that leaps: a trajectory controller that
        #: aborts holds "the current position", and if that was one bad reading
        #: it is a pose the robot is not at -- on 2026-10-17 a servo followed
        #: such a leap 1.4 rad. Whatever the cause, the command reaching the
        #: wire never moves faster than this; when it would, it is slowed and
        #: that is announced.
        unknown = set(max_velocity or {}) - set(self.joint_names)
        if unknown:
            raise KeyError("max_velocity names unknown joints: {}".format(sorted(unknown)))
        self._max_velocity: Dict[int, float] = {
            self.joint_names.index(n): float(v) for n, v in (max_velocity or {}).items()}
        # Joint index -> when its last command went out.
        self._sent_at: Dict[int, float] = {}
        self._rate_said: Dict[int, float] = {}
        # (name, measured, commanded) captured at the last enable; the
        # forensic record of what the robot was told when torque went on.
        self._last_enable: List[Tuple[str, float, float]] = []
        # Joint index -> the position a held joint keeps while its slot is
        # unowned (see _unowned). Forgotten when the joint is released.
        self._hold: Dict[int, float] = {}

    @property
    def default_enabled(self) -> List[str]:
        """Joints that come up driven and return to it after a reconnect.

        The ones with no position sense; see :meth:`__init__`. Exposed so a
        caller can see WHICH rather than having to know the rule.
        """
        return [n for n in self.joint_names if n in self._default_on]

    def reset_to_default(self) -> None:
        """Back to the startup state: the hazardless joints driven, the rest
        released.

        Used after a link recovery. Coming back with the arms released is the
        whole point of that path -- the robot may have moved while it was gone
        -- but leaving the WHEELS released as well would mean a base that stops
        answering after a cable hiccup and stays that way until someone notices
        and presses a button. There is no hazard in a wheel commanded zero.
        """
        self._on = set(self._default_on)

    @property
    def enabled_joints(self) -> List[str]:
        """Names holding torque, in joint order (not set order)."""
        return [n for n in self.joint_names if n in self._on]

    @property
    def all_enabled(self) -> bool:
        return len(self._on) == len(self.joint_names)

    @property
    def any_enabled(self) -> bool:
        return bool(self._on)

    def _resolve(self, joints: Optional[Sequence[str]]) -> List[str]:
        """Empty/None selects every joint; an unknown name is an error.

        Never silently drops a name it does not recognise: a request to release
        "elbow" on a robot whose joint is called "joint2" must fail loudly, not
        quietly do nothing while reporting success.
        """
        if not joints:
            return list(self.joint_names)
        unknown = [j for j in joints if j not in self.joint_names]
        if unknown:
            raise KeyError(
                "no such joint: {}. This bus has: {}".format(
                    ", ".join(repr(j) for j in unknown),
                    ", ".join(self.joint_names)))
        # Preserve joint order and drop duplicates, so a caller passing
        # ["joint2", "joint2"] cannot make a reason string say it twice.
        return [n for n in self.joint_names if n in set(joints)]

    def _fault_name(self, actuator, error: int) -> str:
        error = int(error)
        if error in self._error_names:
            return self._error_names[error]
        name_fn = getattr(actuator, "fault_name", None)
        if callable(name_fn):
            return str(name_fn(error))
        return "unknown fault code {}".format(error)

    # --- observation -------------------------------------------------------

    def inspect(self, shm) -> List[JointServo]:
        """Current per-joint picture, including the jump enabling would cause.

        Reads the command block even while OFF -- but only to *describe* it.
        Nothing here can reach the wire, which is what makes it safe to look.
        """
        command = shm.read_command()
        out: List[JointServo] = []
        hot_plug = bool(getattr(self.bus, "hot_plug", False))
        bus_present = getattr(self.bus, "present", None)
        bus_reason = getattr(self.bus, "absence_reason", None)
        for i, (name, actuator) in enumerate(zip(self.joint_names, self.bus.actuators)):
            state = self.bus.last_state(actuator.id)
            present = (bool(bus_present(actuator.id)) if callable(bus_present)
                       else state is not None)
            reason = ("" if present or not callable(bus_reason)
                      else str(bus_reason(actuator.id)))
            if state is None and hot_plug:
                # Absent since start on a hot-plug bus: described, not
                # invented -- every quantity is NaN.
                out.append(JointServo(
                    name=name, enabled=name in self._on, measured=math.nan,
                    mode=self.modes[i],
                    commanded=float(command[i * self.command_stride
                                            + self._SLOT[self.modes[i]]]),
                    error=0, error_name="not answering",
                    present=present, absence_reason=reason))
                continue
            if state is None:
                raise RuntimeError("{} has never reported".format(name))
            error = int(state.error or 0)
            out.append(JointServo(
                name=name,
                enabled=name in self._on,
                # A wheel has no position sense at all -- its reading is a pot
                # that wraps. NaN, never 0.0, which would read as a real pose.
                measured=(state.position if state.position is not None
                          else math.nan),
                mode=self.modes[i],
                commanded=float(
                    command[i * self.command_stride
                            + self._SLOT[self.modes[i]]]),
                error=error,
                error_name=self._fault_name(actuator, error),
                # `if not None else nan`, not `else 0.0`: absent must stay
                # absent all the way to the topic.
                temperature=(state.temperature if state.temperature is not None
                             else math.nan),
                velocity=(state.velocity if state.velocity is not None
                          else math.nan),
                effort=(state.current if state.current is not None
                        else math.nan),
                # ActuatorState puts the wire's own quantities here whether or
                # not the datasheet constants to convert them are known -- for
                # a CubeMars that is amps and electrical RPM. Amps need no
                # constant, so they are pulled out as a typed field; the rest
                # travels by name.
                current_a=float(state.aux.get("current_a", math.nan))
                if state.aux else math.nan,
                aux={k: float(v) for k, v in (state.aux or {}).items()
                     if isinstance(v, (int, float))},
                present=present, absence_reason=reason,
            ))
        return out

    def would_jump(self, joint: JointServo) -> bool:
        """Would applying torque move this joint suddenly?

        The question is the same in every mode; the quantity that answers it is
        not, so this branches rather than pretending one threshold fits both.

        * POSITION -- the hazard is a controller commanding somewhere far from
          where the robot is: torque comes on and the joint is thrown there.
          Judged on ``mismatch`` against ``jump_threshold`` (rad).
        * VELOCITY -- there IS no commanded position to be far from, so the
          position hazard does not exist at all; velocity mode has no start-up
          jump by construction. The hazard is instead a NON-ZERO commanded
          speed, which starts the joint moving the instant torque arrives.
          Judged on the commanded velocity against ``velocity_threshold``
          (rad/s).
        * CURRENT -- same shape again: a non-zero commanded torque at enable.
          Reuses ``velocity_threshold`` as a bare "is it about zero" test,
          which is honest about being a magnitude check and not a physical
          limit.

        In VELOCITY and CURRENT modes ``JointServo.commanded`` and ``mismatch``
        carry the MODE'S units, not radians. That is why ``ServoState`` publishes
        the mode: a consumer must never have to guess which quantity it is
        reading.
        """
        if joint.mode is OperatingMode.POSITION:
            return bool(abs(joint.mismatch) > self.jump_threshold)
        return bool(abs(joint.commanded) > self.velocity_threshold)

    # --- the one place frames are emitted ----------------------------------

    def service(self, shm) -> None:
        """Drive one control cycle. The ONLY writer of motion frames.

        Note the asymmetry: the branch with any joint enabled reads ``shm``; the
        fully-off branch does not touch it. That is property 1 above, expressed
        as structure rather than as a promise.

        Held and limp joints go out in ONE batch (``None`` = release). Splitting
        them into two bursts would leave the joints in different states for the
        gap between, which is exactly when a freed limb starts to fall.
        """
        self.cycles += 1
        if not self._on:
            now = time.monotonic()
            if (self._last_release is not None
                    and now - self._last_release < self.release_period):
                return
            self._last_release = now
            self.bus.release_all()
            return
        self._last_release = None

        command = shm.read_command()
        values = [float(command[i * self.command_stride
                                + self._SLOT[self.modes[i]]])
                  for i in range(len(self.joint_names))]
        # NaN is "no controller owns this slot" (actuo.drivers.shm): its module's
        # hardware component is inactive. A held joint keeps holding rather
        # than going limp under a controller that merely stopped.
        values = [self._unowned(i, v) if name in self._on else v
                  for i, (name, v) in enumerate(zip(self.joint_names, values))]
        if self.command_filter is not None:
            values = self._filtered(values)
        sent = [self._fence(self.bus.actuators[i], values[i], i)
                if name in self._on and math.isfinite(values[i]) else None
                for i, name in enumerate(self.joint_names)]
        now = time.monotonic()
        sent = [self._rate_limit(i, v, now) if v is not None else None
                for i, v in enumerate(sent)]
        # Remember what actually went out, after the fence, as the pose an
        # unowned joint holds. The controller's raw value can be one the
        # fence moved (a joint coming back inside its envelope); holding THAT
        # once the controller let go handed the actuator a pose it refuses and
        # took the servicer down (2026-10-17, j0 at -2.3586 on a restack).
        for i, value in enumerate(sent):
            if value is not None and self.modes[i] is OperatingMode.POSITION:
                self._hold[i] = value
        self.bus.write_batch([None if value is None else self._command(i, value)
                              for i, value in enumerate(sent)])

    def _rate_limit(self, index: int, value: float, now: float) -> float:
        """Hold a POSITION command to ``max_velocity`` from the last one sent."""
        limit = self._max_velocity.get(index)
        if (limit is None or self.modes[index] is not OperatingMode.POSITION
                or index not in self._hold or index not in self._sent_at):
            self._sent_at[index] = now
            return value
        # Capped, so a pause in servicing (a slow cycle) does not bank a
        # large allowance that one leap can then spend.
        dt = min(now - self._sent_at[index], 0.1)
        self._sent_at[index] = now
        last = self._hold[index]
        step = limit * dt
        if abs(value - last) <= step:
            return value
        limited = last + math.copysign(step, value - last)
        if now - self._rate_said.get(index, -math.inf) >= RECOVER_REPEAT_S:
            self._rate_said[index] = now
            self._filter_event = (
                "{}: the command jumped {:+.4f} rad in one cycle ({:.1f} rad/s); "
                "moving it at the {:.2f} rad/s max_velocity instead".format(
                    self.joint_names[index], value - last,
                    abs(value - last) / max(dt, 1e-6), limit))
        return limited

    def _unowned(self, index: int, value: float) -> float:
        """The value to drive a held joint with when its slot is unowned.

        POSITION: hold where it was last sent, or, if it never was, where it
        is now. VELOCITY/CURRENT: zero -- holding the last speed would keep a
        wheel driving after its controller stopped. NaN only when a position
        joint has neither, which releases it for this cycle.
        """
        if math.isfinite(value):
            return value
        if self.modes[index] is not OperatingMode.POSITION:
            return 0.0
        if index in self._hold:
            return self._hold[index]
        state = self.bus.last_state(self.bus.actuators[index].id)
        if state is not None and state.position is not None:
            self._hold[index] = float(state.position)
            return self._hold[index]
        return math.nan

    def _filtered(self, values):
        """Run the command through the filter and report what it did.

        What the filter is shown is the pose the robot would END UP in, which is
        not the command vector: a limp joint is not going where the controller
        says, it is staying where it sagged to. So held joints contribute their
        command and released joints their MEASURED angle. Showing a filter the
        command for a limp joint would have it judge a pose the robot is not
        going to be in -- in both directions, stopping motion over an imaginary
        contact and missing a real one.

        A joint that has never reported falls back to its command, because the
        alternative is inventing a number, and this is not the layer that gets
        to decide the robot is at zero.

        Only the POSITION joints are shown, in their own order
        (``_filter_indices``). A wheel's command is a speed; putting it in a
        vector labelled "pose" would have the filter compare rad/s against link
        geometry. Wheels are left exactly as the controller asked -- turning a
        wheel cannot collide the arm with the body.
        """
        import numpy as np

        pose = []
        for index in self._filter_indices:
            name = self.joint_names[index]
            value = values[index]
            if name not in self._on:
                state = self.bus.last_state(self.bus.actuators[index].id)
                if state is not None and state.position is not None:
                    value = float(state.position)
            pose.append(value)

        result = self.command_filter.filter(np.asarray(pose, dtype=float))
        applied = list(values)
        for slot, index in enumerate(self._filter_indices):
            applied[index] = float(result.applied[slot])
        self._last_applied = list(applied)
        if result.gated:
            self.filtered_cycles += 1
            if not self._filter_gated:
                self._filter_gated = True
                self._filter_event = "command filter STOPPED the robot: {}".format(
                    result.reason)
        elif self._filter_gated:
            self._filter_gated = False
            self._filter_event = ("command filter released; the command is "
                                  "acceptable again")
        # Only held joints are commanded at all, so a filtered value for a limp
        # joint is discarded by the caller -- deliberately, not incidentally.
        return applied

    def pop_filter_event(self):
        """The last gate/ungate transition, once. ``None`` if nothing changed.

        Consumed rather than latched so a caller can print it in a loop without
        repeating it every cycle -- the alternative, printing per gated cycle,
        is 100 lines a second on a robot that is simply being held.
        """
        event, self._filter_event = self._filter_event, None
        return event

    def release_filter(self):
        """Clear a latched command filter, and REPORT what happened next.

        A latched stop is a judgement withheld: the filter refuses to decide on
        its own that the danger has passed, because the path back to a pose that
        is itself safe may sweep through the one that is not. Releasing is
        therefore an operator act, and this is where it lands.

        Two things it does not do, both deliberate:

        * It does not adopt the CONTROLLER's command as safe. It adopts the pose
          the filter is holding, which is collision-free by construction. Adopt
          the command instead and you have told the filter that the pose it
          stopped for is fine.
        * It does not predict whether the release will hold. It releases, runs
          one cycle, and says what the filter did with the controller's current
          command. If that command still collides the filter latches again
          immediately and the operator is told so -- which is the honest answer
          and needs no model of what the controller is about to do.

        Returns ``(ok, message)`` like every other operator-facing decision here.
        """
        if self.command_filter is None:
            return False, "no command filter is configured; nothing to release."
        if self._last_applied is None:
            return False, ("the filter has not run yet, so there is no held "
                           "pose to adopt as safe. Enable torque first.")

        import numpy as np

        # The filter's own vector -- POSITION joints only, in its order. See
        # _filtered: it never saw the wheels and must not be handed them here.
        held = np.asarray([self._last_applied[i] for i in self._filter_indices],
                          dtype=float)
        self.command_filter.reset(held)
        self._filter_gated = False

        result = self.command_filter.filter(held)
        if result.gated:
            # Cannot happen with a correct filter -- the held pose was allowed a
            # moment ago -- so say so plainly rather than pretending it released.
            return False, (
                "released, but the filter immediately objected to the pose it "
                "was itself holding: {}. Treat this as a filter bug, not as a "
                "robot that needs freeing.".format(result.reason))
        return True, (
            "released, holding {}. The filter judges the CONTROLLER's next "
            "command afresh: if it still asks for the pose that stopped this, "
            "it will stop again immediately. Move the controller somewhere "
            "reachable first.".format(
                ", ".join(
                    "{}={:+.4f}".format(self.joint_names[i], v)
                    for i, v in zip(self._filter_indices, held))))

    @staticmethod
    def _envelope(actuator):
        """(low, high) rad an actuator accepts position commands in, or None.

        The actuator's own ``position_envelope`` when it has one -- for a KRS
        servo that is its mechanical pulse range, which a taught limit only
        narrows -- otherwise the taught ``calibration.position_limit``.
        """
        envelope = getattr(actuator, "position_envelope", None)
        if callable(envelope):
            return envelope()
        return getattr(getattr(actuator, "calibration", None),
                       "position_limit", None)

    def _fence(self, actuator, value: float, index: int) -> float:
        """Zero a velocity/torque that would drive a joint FURTHER out of bounds.

        Restores, at this layer, a guarantee that quietly disappeared when the
        arm gained a velocity command interface.

        In POSITION mode the actuator model checks every commanded angle against
        the taught envelope (``CubeMarsActuator._check_position_limit``) and
        refuses one outside it. That check lives in the position encoder, and in
        VELOCITY or CURRENT mode there is no commanded position to check -- so
        the actuator-level fence, which is meant to be the last line,
        silently stopped existing. Observed 2026-07-30: joint2 sat 1.2 deg past
        its envelope and neither the actuator nor the servicer said anything,
        where position mode would have refused outright.

        Only motion that goes FURTHER out is blocked. A joint already outside
        must still be able to come back, or it is stranded there with no way to
        recover except cutting power -- and being outside the envelope is exactly
        the state that already deactivates arm_controller and looks like a dead
        robot.

        A hard zero rather than a taper, deliberately: a taper needs a distance
        over which to decelerate, which is another number nobody has measured on
        the hardware, and inventing one here would be the kind of plausible constant
        this project keeps paying for. Position mode's answer to the same
        question is an outright refusal, so this is no more abrupt than what it
        replaces.
        """
        if self.modes[index] is OperatingMode.POSITION:
            return self._recover_into_limit(actuator, value, index,
                                            self._envelope(actuator))
        limit = getattr(getattr(actuator, "calibration", None),
                        "position_limit", None)
        if limit is None:
            return value          # no taught envelope: nothing to enforce
        state = self.bus.last_state(actuator.id)
        if state is None or state.position is None:
            # Position unknown. Refusing to move is the only honest answer: this
            # cannot judge, and guessing "probably fine" is how a joint gets
            # driven further into a stop. Staleness itself is caught upstream by
            # the bus freshness check.
            return 0.0
        low, high = limit
        if state.position >= high and value > 0.0:
            return 0.0
        if state.position <= low and value < 0.0:
            return 0.0
        return value

    def _recover_into_limit(self, actuator, value: float, index: int,
                            limit) -> float:
        """Keep every position command inside the joint's envelope.

        The actuator checks each commanded angle and refuses one outside its
        envelope -- by raising, inside :meth:`service`, which takes the whole
        servicer and every other joint's torque down with it. Two ordinary
        situations reached that:

        * a joint RESTING outside (a KRS servo that sagged a few pulses past
          3500 when limp) can never be commanded to hold where it is, so it
          was stranded until pushed by hand;
        * a controller commanding outside -- a trajectory back to a start pose
          recorded out there, or a pose the controller adopted from one --
          killed the servicer mid-motion (2026-10-17, twice).

        Velocity mode already settles the same question (see :meth:`_fence`:
        motion further out is blocked, motion back is not). This is that rule
        for position mode: the command goes out clamped to the nearest edge,
        so a joint is only ever sent inward or along the edge, never further
        out. It is announced when it starts, because a command that becomes a
        different command silently is a fallback that hides breakage -- the ban
        is on the silence. Torque-on still refuses a pose the actuator cannot
        express (see :meth:`enable`); this is what happens once it is on.
        """
        if limit is None:
            return value
        low, high = limit
        clamped = min(max(value, low), high)
        state = self.bus.last_state(actuator.id)
        resting_out = (state is not None and state.position is not None
                       and not low <= state.position <= high)
        if clamped == value and not resting_out:
            self._recovering.discard(index)
            return value
        now = time.monotonic()
        if (index not in self._recovering
                and now - self._recover_said.get(index, -math.inf)
                >= RECOVER_REPEAT_S):
            # Once per episode, and at most every RECOVER_REPEAT_S: a joint
            # parked on the edge flips in and out by a pulse every few cycles
            # (j2 on the rig, -2.3568 vs -2.3562) and said this ~70 times.
            self._recover_said[index] = now
            if resting_out:
                self._filter_event = (
                    "{} is OUTSIDE its envelope ({:+.4f} rad, allowed "
                    "[{:+.4f}, {:+.4f}]). Commands are being held at the nearest "
                    "edge so it can come back in; it cannot be driven further "
                    "out from here.".format(
                        self.joint_names[index], state.position, low, high))
            else:
                self._filter_event = (
                    "{}: the controller commands {:+.4f} rad, outside the "
                    "envelope [{:+.4f}, {:+.4f}]; holding {:+.4f}, the nearest "
                    "edge, until it commands inside again.".format(
                        self.joint_names[index], value, low, high, clamped))
        self._recovering.add(index)
        return clamped

    def _command(self, index: int, value: float) -> ActuatorCommand:
        """Wrap one shm number as a command in whichever mode is in force.

        The field the value lands in is not cosmetic: an actuator raises if the
        field its mode needs is absent, so a mis-wired mode fails loudly at the
        first cycle rather than commanding a plausible zero.
        """
        mode = self.modes[index]
        if mode is OperatingMode.VELOCITY:
            return ActuatorCommand(mode=mode, velocity=value)
        if mode is OperatingMode.CURRENT:
            return ActuatorCommand(mode=mode, effort=value)
        return ActuatorCommand(mode=mode, position=value)

    # --- transitions -------------------------------------------------------

    def disable(self, joints: Optional[Sequence[str]] = None) -> Tuple[bool, str]:
        """Servo off, for ``joints`` (default: all). Always allowed.

        Removing torque cannot cause a lurch, so there is nothing to refuse.
        What it CAN do is drop whatever the freed joints were holding up -- the
        reason says which joints went limp so that is never a surprise.
        """
        try:
            target = self._resolve(joints)
        except KeyError as err:
            return False, "refusing: {}".format(err.args[0])

        went_limp = [n for n in target if n in self._on]
        self._on.difference_update(target)
        for name in target:
            self._hold.pop(self.joint_names.index(name), None)
            self._sent_at.pop(self.joint_names.index(name), None)

        if not went_limp:
            return True, "already off: {}".format(", ".join(target))
        return True, "torque removed from {}; {} limp and will fall under gravity{}".format(
            ", ".join(went_limp),
            "they are" if len(went_limp) > 1 else "it is",
            "" if self.all_off else " (still holding: {})".format(
                ", ".join(self.enabled_joints)))

    @property
    def all_off(self) -> bool:
        return not self._on

    def enable(self, shm, joints: Optional[Sequence[str]] = None,
               force: bool = False) -> Tuple[bool, str]:
        """Servo on for ``joints`` (default: all), refused unless the controller
        already matches the robot.

        Judged per joint, over the REQUESTED joints only: enabling joint1 must
        not be refused because joint2 -- which the operator deliberately left
        limp -- is far from its command. Refusing on an untouched joint would
        teach operators to reach for ``force``, which is the opposite of the
        point.

        ``force`` skips the check. It exists so an override is explicit, logged
        and greppable -- not so that someone quietly widens the threshold.
        """
        try:
            target = self._resolve(joints)
        except KeyError as err:
            return False, "refusing: {}".format(err.args[0])

        pending = [n for n in target if n not in self._on]
        if not pending:
            return True, "already on: {}".format(", ".join(target))

        present = getattr(self.bus, "present", None)
        if callable(present):
            gone = [n for n in pending
                    if not present(self.bus.actuators[self.joint_names.index(n)].id)]
            if gone:
                return False, "refusing: {} not answering".format(", ".join(gone))

        wanted = [j for j in self.inspect(shm) if j.name in set(pending)]

        faulted = [j for j in wanted if j.error]
        if faulted:
            return False, "refusing: {}".format("; ".join(
                "{} reports {}".format(j.name, j.error_name) for j in faulted))

        # A position joint no controller commands has nothing to hold: NaN is
        # an unowned slot (actuo.drivers.shm), and every guard below would read
        # it as "no mismatch". Its module's hardware component is inactive.
        unowned = [j for j in wanted if j.mode is OperatingMode.POSITION
                   and not math.isfinite(j.commanded)]
        if unowned:
            return False, (
                "refusing: no controller commands {} -- its hardware component "
                "is not active (activate it, which adopts the measured pose, "
                "then enable)".format(", ".join(j.name for j in unowned)))

        # Ask each actuator whether the value about to be written is one it can
        # actually express. `force` does NOT skip this: force is an override of
        # the JUMP judgement -- "yes, I accept the robot moving" -- and this is
        # not a judgement, it is the wire saying the number does not fit.
        # Overriding it does not get the requested motion, it gets a different
        # one. Without this check (before 2026-07-26), enabling a joint whose
        # measured angle was unrepresentable slammed it ~340 deg into an
        # overload fault while every radian-space guard read perfectly clean.
        # POSITION only. In VELOCITY/CURRENT mode `commanded` is a speed or a
        # torque, and handing that to a POSITION reachability check would ask
        # "is 0.3 rad/s a reachable angle" -- a question about the wrong
        # quantity, whose answer means nothing. Silently comparing units is how
        # a guard starts reporting confident nonsense.
        impossible = []
        # Joints that will travel back into their envelope the moment torque
        # arrives. Collected so enabling can SAY it, since the jump guard below
        # cannot see this one: after a re-sync the controller commands the
        # measured pose, so `mismatch` is zero while the joint is still about to
        # move by however far outside it is.
        returning = []
        for joint in wanted:
            if joint.mode is not OperatingMode.POSITION:
                continue
            index = self.joint_names.index(joint.name)
            actuator = self.bus.actuators[index]
            limit = self._envelope(actuator)
            if limit is not None and not limit[0] <= joint.measured <= limit[1]:
                # Outside the envelope. Refusing here is what strands it: it
                # cannot be commanded to hold where it is, so it can never be
                # driven back. _recover_into_limit walks it to the nearest edge
                # and no further; what is owed the operator is knowing it will
                # move, and by how much, BEFORE they press the button.
                edge = limit[1] if joint.measured > limit[1] else limit[0]
                if abs(joint.measured - edge) > self.jump_threshold and not force:
                    impossible.append(
                        "{} is {:+.4f} rad, {:.4f} rad OUTSIDE its envelope "
                        "[{:+.4f}, {:+.4f}]; torque would move it back to "
                        "{:+.4f}, more than the {:.4f} rad jump threshold. Move "
                        "it in by hand, or enable with force to accept the "
                        "move".format(joint.name, joint.measured,
                                      abs(joint.measured - edge), limit[0],
                                      limit[1], edge, self.jump_threshold))
                    continue
                returning.append(
                    "{} is {:+.4f} rad OUTSIDE its envelope [{:+.4f}, {:+.4f}] "
                    "and will move {:.4f} rad back to {:+.4f} as soon as "
                    "torque is applied".format(
                        joint.name, joint.measured, limit[0], limit[1],
                        abs(joint.measured - edge), edge))
                continue
            reason = actuator.position_refusal(joint.commanded)
            if reason is not None:
                impossible.append("{}: {}".format(joint.name, reason))
        if impossible:
            return False, (
                "refusing: the pose to hold cannot be commanded. {}".format(
                    "; ".join(impossible)))

        jumpy = [j for j in wanted if self.would_jump(j)]
        if jumpy and not force:
            # Both refusals can be live at once on a robot whose wheels and arms
            # share a board, and they have DIFFERENT remedies -- one is "re-sync
            # the controller", the other is "stop asking for motion". Reporting
            # only the first would send the operator to fix the wrong end.
            far = [j for j in jumpy if j.mode is OperatingMode.POSITION]
            moving = [j for j in jumpy if j.mode is not OperatingMode.POSITION]
            parts = []
            if far:
                parts.append(
                    "enabling would throw the robot: {}. Threshold is {:.4f} "
                    "rad. Re-sync the CONTROLLER to the robot -- deactivate and "
                    "reactivate the trajectory controller so it adopts the "
                    "measured pose -- rather than letting the robot jump to the "
                    "controller.".format(
                        "; ".join(
                            "{} is at {:.4f} but the controller commands "
                            "{:.4f} ({:+.4f} rad away)".format(
                                j.name, j.measured, j.commanded, j.mismatch)
                            for j in far),
                        self.jump_threshold))
            if moving:
                # Nothing is "far away" here, so the remedy is not a re-sync --
                # the controller is actively asking for motion and must be asked
                # to stop first.
                parts.append(
                    "the controller is asking for motion, so torque would start "
                    "the joint moving immediately: {}. Threshold is {:.4f}. Let "
                    "the controller settle to zero first (end or cancel the "
                    "trajectory); there is nothing to re-sync there, because "
                    "there is no commanded position to be far from.".format(
                        "; ".join(
                            "{} is commanded {:+.4f} {}".format(
                                j.name, j.commanded,
                                "rad/s" if j.mode is OperatingMode.VELOCITY
                                else "N*m")
                            for j in moving),
                        self.velocity_threshold))
            return False, "refusing: " + " AND ".join(parts)

        self._on.update(pending)
        if returning:
            self._filter_event = "; ".join(returning)

        # Record what the robot is about to be told, per joint, at the instant
        # torque is applied. Enabling is the one moment a joint can be sent
        # somewhere far away, and when one of them did exactly that -- joint3
        # spun ~188 deg against a commanded 35 deg, at ~100x its ERPM envelope,
        # 2026-07-17 -- the logs said what was *asked* and what was *measured*,
        # and nothing about what was *commanded*. Three theories, no evidence.
        # This is the line that would have settled it.
        self._last_enable = [
            (j.name, j.measured, j.commanded) for j in wanted]

        # A velocity/effort joint's measured is NaN when its actuator senses
        # no position (a Kondo wheel: the pot wraps and has a blind sector).
        # Printing "at +nan" beside "within 0.05 rad" read as a fault; say
        # what is held instead, and keep the threshold to the joints it bounds.
        detail = "; ".join(
            "{} at {:+.4f} -> commanding {:+.4f}".format(
                j.name, j.measured, j.commanded)
            if j.mode is OperatingMode.POSITION else
            "{} (no position sensed) -> commanding {:+.4f} {}".format(
                j.name, j.commanded,
                "rad/s" if j.mode is OperatingMode.VELOCITY else "N*m")
            for j in wanted)
        if jumpy:
            return True, "FORCED {} on despite a {:.4f} rad mismatch [{}]".format(
                ", ".join(j.name for j in jumpy),
                max(abs(j.mismatch) for j in jumpy), detail)
        if all(j.mode is not OperatingMode.POSITION for j in wanted):
            return True, ("torque on for {}; every one commanded within {:.4f} "
                          "of standstill [{}]").format(
                ", ".join(pending), self.velocity_threshold, detail)
        return True, ("torque on for {}; every one within {:.4f} rad of its "
                      "command [{}]").format(
            ", ".join(pending), self.jump_threshold, detail)
