#!/usr/bin/env python3
"""actuo bus-servicer: owns one end of the shared-memory bridge and drives the
plant. In ``--sim`` it steps a MuJoCo model (the joint servo loop lives in the
model's onboard position actuator, mirroring a real smart servo); in ``--real``
it services a hardware bus built from a robot config file.

Runs OFF the controller's RT critical path: ros2_control's C++
``actuo_hardware/ActuoSystem`` does the deterministic read/write against the
shm seqlock; this process services the bus/plant at its own rate.

    ros2 run actuo_ros actuo_bus_node --sim --model one_dof.xml \
        --shm-name sim_1dof
    ros2 run actuo_ros actuo_bus_node --real --config <robot>.yaml \
        --shm-name arm

Sim and real are the same spine: identical shm layout, identical joint order,
identical controllers. Only what sits behind this process differs.
"""

from __future__ import annotations

import argparse
from collections import deque
import fcntl
import math
import os
import sys
import time

import numpy as np

from actuo.core.diagnostics import DiagItem, DiagLevel
from actuo.drivers import window_state
from actuo.drivers.shm import (
    CMD_STRIDE, STATE_PRESENT, STATE_STRIDE, ShmBridge, lock_path,
    wait_for_commands)

#: Module of the python-can transport. Imported only by the registry, when a
#: config names that transport, because importing it needs python-can.
_PYTHON_CAN_MODULE = "actuo.drivers.transports.python_can"


class _NeverRaised(Exception):
    """Stands in for an exception type whose module is not loaded."""


def transmit_queue_full_type():
    """The exception a full CAN transmit queue raises on this process's bus.

    Only the python-can transport raises ``TransmitQueueFull``, and it is
    imported (by :func:`actuo.drivers.config.build_bus`, through the registry)
    exactly when the config names it. When it was not imported, no transport in
    use can raise it, so a private type that nothing raises is returned and the
    ``except`` clause for it is inert -- importing the module here instead
    would make every non-CAN robot depend on python-can.

    Returns
    -------
    type
        ``TransmitQueueFull``, or :class:`_NeverRaised`.
    """
    module = sys.modules.get(_PYTHON_CAN_MODULE)
    if module is None:
        return _NeverRaised
    return module.TransmitQueueFull


def _create_shm(name, n_joints):
    """Create this servicer's segment (the servicer owns it).

    Re-created in place on every start, so whatever a previous run left
    in the file -- state, commands, presence -- is gone before any hardware
    component can attach. Components attach only while this process holds
    the servicer lock (see ShmBridge::servicer_alive), so they never read a
    leftover file either.
    """
    return ShmBridge(name, n_joints=n_joints, cmd_stride=CMD_STRIDE,
                     state_stride=STATE_STRIDE, create=True)


def parse_window_references(items):
    """``["joint3=-23.5"]`` -> ``{"joint3": -0.410...}`` (radians).

    An operator assertion about where a joint physically is, used to pick a gear
    window when nothing better is available. Parsed strictly and separately from
    everything else so a typo is a startup error rather than a joint placed in
    the wrong window -- ``joint3=-23,5`` must not silently become ``-23``.
    """
    out = {}
    for item in items or []:
        joint, sep, value = str(item).partition("=")
        joint = joint.strip()
        if not sep or not joint:
            raise ValueError(
                "--window-reference wants JOINT=DEGREES, got {!r}".format(item))
        try:
            degrees = float(value)
        except ValueError:
            raise ValueError(
                "--window-reference {!r}: {!r} is not a number of "
                "degrees".format(item, value))
        if degrees != degrees or degrees in (float("inf"), float("-inf")):
            raise ValueError(
                "--window-reference {!r}: not a finite angle".format(item))
        out[joint] = math.radians(degrees)
    return out


class TxStallPolicy:
    """Decide what a full transmit queue means for a periodic setpoint stream.

    Separated from ``run_real``'s loop so it can be tested at all: the loop
    only runs against hardware, so a branch written inline in it is first
    executed on a live robot. The policy is the part with the interesting
    behaviour, so the policy is the part that gets a test.

    The rule
    --------
    Dropping ONE cycle's command is safe and correct: the next cycle is 20 ms
    away and carries a fresher setpoint, so a dropped command is strictly better
    than a queued stale one. Dying instead drops the arm -- the servicer exits,
    ``on_exit=Shutdown()`` takes the stack down, and torque goes off wherever
    the arm happened to be.

    But a queue that NEVER drains means commands are not reaching the motors at
    all while torque is on, and that is fatal rather than transient. So drops are
    counted consecutively and only an unbroken streak aborts (never mask
    breakage -- bound how long it may be tolerated).
    """

    def __init__(self, max_consecutive: int) -> None:
        self.max_consecutive = max_consecutive
        self.streak = 0
        self.total = 0

    def dropped(self) -> bool:
        """Record a dropped cycle; return True if this must now be fatal."""
        self.streak += 1
        self.total += 1
        return self.streak >= self.max_consecutive

    def succeeded(self) -> int:
        """Record a clean write; return the streak it just ended (0 if none).

        Only a clean write clears the streak, so a drop can never reset its own
        counter and make an unbroken stall look intermittent.
        """
        ended = self.streak
        self.streak = 0
        return ended


def _rolling_mean(values):
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _rolling_p95(values):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), 95.0))


class ServicerDiagnostics:
    """Backend-agnostic bus-loop metrics, measured in the servicer once.

    This is ROS-free on purpose: it owns arithmetic and level judgement, and the
    ROS diagnostics layer only translates the resulting `DiagItem`.
    """

    def __init__(self, target_rate_hz: float, window_cycles: int,
                 loop_warn_ratio: float, loop_error_ratio: float,
                 latency_warn_ratio: float, latency_error_ratio: float,
                 drop_rate_warn: float, drop_rate_error: float,
                 consecutive_miss_warn: int, consecutive_miss_error: int,
                 stale_warn_s: float, stale_error_s: float,
                 cmd_stall_warn_s: float, cmd_stall_error_s: float,
                 loop_spike_error_ratio: float = 3.0) -> None:
        self.target_rate_hz = float(target_rate_hz)
        self.target_period_s = 1.0 / self.target_rate_hz
        self.window_cycles = int(max(1, window_cycles))

        self.loop_warn_ratio = float(loop_warn_ratio)
        self.loop_error_ratio = float(loop_error_ratio)
        self.loop_spike_error_ratio = float(loop_spike_error_ratio)
        self.latency_warn_ratio = float(latency_warn_ratio)
        self.latency_error_ratio = float(latency_error_ratio)
        self.drop_rate_warn = float(drop_rate_warn)
        self.drop_rate_error = float(drop_rate_error)
        self.consecutive_miss_warn = int(consecutive_miss_warn)
        self.consecutive_miss_error = int(consecutive_miss_error)
        self.stale_warn_s = float(stale_warn_s)
        self.stale_error_s = float(stale_error_s)
        self.cmd_stall_warn_s = float(cmd_stall_warn_s)
        self.cmd_stall_error_s = float(cmd_stall_error_s)

        self._periods = deque(maxlen=self.window_cycles)
        self._latencies = deque(maxlen=self.window_cycles)
        self._drops = deque(maxlen=self.window_cycles)
        self._cycles = 0
        self._drop_cycles = 0
        self._consecutive_misses = 0
        self._last_hardware_read_mono = None

        self._last_cmd_seq = None
        self._last_cmd_seq_change_mono = None
        self._last_state_seq = None
        self._last_state_seq_change_mono = None
        self._latest_cmd_seq = 0
        self._latest_state_seq = 0

    def observe_cycle_period(self, period_s: float) -> None:
        if period_s > 0.0:
            self._periods.append(float(period_s))

    def observe_read_latency(self, latency_s: float) -> None:
        if latency_s >= 0.0:
            self._latencies.append(float(latency_s))

    def observe_read_result(self, had_miss: bool, had_fresh_state: bool,
                            now_mono: float) -> None:
        self._cycles += 1
        if had_miss:
            self._drop_cycles += 1
            self._consecutive_misses += 1
        else:
            self._consecutive_misses = 0
        self._drops.append(1 if had_miss else 0)
        if had_fresh_state:
            self._last_hardware_read_mono = float(now_mono)

    def observe_shm(self, cmd_seq: int, state_seq: int, now_mono: float) -> None:
        cmd_seq = int(cmd_seq)
        state_seq = int(state_seq)
        self._latest_cmd_seq = cmd_seq
        self._latest_state_seq = state_seq
        if self._last_cmd_seq != cmd_seq:
            self._last_cmd_seq = cmd_seq
            self._last_cmd_seq_change_mono = float(now_mono)
        if self._last_state_seq != state_seq:
            self._last_state_seq = state_seq
            self._last_state_seq_change_mono = float(now_mono)

    @staticmethod
    def _worse(level: DiagLevel, other: DiagLevel) -> DiagLevel:
        if other > level:
            return other
        return level

    def snapshot_item(self, now_mono: float) -> DiagItem:
        period_mean = _rolling_mean(self._periods)
        period_p95 = _rolling_p95(self._periods)
        period_max = max(self._periods) if self._periods else 0.0
        jitters = [abs(p - self.target_period_s) for p in self._periods]
        jitter_p95 = _rolling_p95(jitters)

        latency_mean = _rolling_mean(self._latencies)
        latency_p95 = _rolling_p95(self._latencies)
        latency_max = max(self._latencies) if self._latencies else 0.0

        drop_rate = (0.0 if not self._drops
                     else float(sum(self._drops)) / float(len(self._drops)))
        staleness_s = (float("inf") if self._last_hardware_read_mono is None
                       else max(0.0, float(now_mono) - self._last_hardware_read_mono))
        cmd_stall_s = (float("inf") if self._last_cmd_seq_change_mono is None
                       else max(0.0, float(now_mono) - self._last_cmd_seq_change_mono))
        state_stall_s = (float("inf") if self._last_state_seq_change_mono is None
                         else max(0.0, float(now_mono) - self._last_state_seq_change_mono))

        level = DiagLevel.OK
        reasons = []

        # A loop that cannot keep up and a loop that hiccupped once are
        # different faults, and folding them into one verdict makes the healthy
        # case unreachable. A healthy Kondo bus at a 10 ms target measures
        # p95 10.06 ms, mean 10.03 ms -- and the occasional ONE cycle at
        # 13.08 ms, which a single `p95 > err or max > err` rule would paint as
        # permanently ERROR with "loop period slower than error threshold". An
        # operator reading that goes looking for a slow loop that does not
        # exist, and having been sent there once, ignores the next one.
        #
        # A software loop also cannot average FASTER than its target period
        # (sleeping to a deadline yields period >= target), so any rule that
        # judges the mean or the max against a barely-above-1.0 ratio is a
        # check a healthy system cannot pass.
        loop_warn = self.target_period_s * self.loop_warn_ratio
        loop_error = self.target_period_s * self.loop_error_ratio
        loop_spike_error = self.target_period_s * self.loop_spike_error_ratio
        if period_p95 > loop_error:
            level = self._worse(level, DiagLevel.ERROR)
            reasons.append(
                "loop period p95 {:.1f} ms exceeds {:.1f} ms -- the loop is not "
                "keeping up".format(period_p95 * 1e3, loop_error * 1e3))
        elif period_max > loop_spike_error:
            level = self._worse(level, DiagLevel.ERROR)
            reasons.append(
                "one cycle took {:.1f} ms against a {:.1f} ms target ({:.0f}x) "
                "-- a stall, not jitter".format(
                    period_max * 1e3, self.target_period_s * 1e3,
                    period_max / self.target_period_s))
        elif period_p95 > loop_warn:
            level = self._worse(level, DiagLevel.WARN)
            reasons.append(
                "loop period p95 {:.1f} ms exceeds {:.1f} ms".format(
                    period_p95 * 1e3, loop_warn * 1e3))
        elif period_max > loop_error:
            level = self._worse(level, DiagLevel.WARN)
            reasons.append(
                "worst cycle {:.1f} ms exceeds {:.1f} ms, but p95 is {:.1f} ms "
                "-- one spike, not a slow loop".format(
                    period_max * 1e3, loop_error * 1e3, period_p95 * 1e3))

        latency_warn = self.target_period_s * self.latency_warn_ratio
        latency_error = self.target_period_s * self.latency_error_ratio
        if latency_p95 > latency_error:
            level = self._worse(level, DiagLevel.ERROR)
            reasons.append("read latency above error threshold")
        elif latency_p95 > latency_warn:
            level = self._worse(level, DiagLevel.WARN)
            reasons.append("read latency above warn threshold")

        if drop_rate >= self.drop_rate_error:
            level = self._worse(level, DiagLevel.ERROR)
            reasons.append("drop rate above error threshold")
        elif drop_rate >= self.drop_rate_warn:
            level = self._worse(level, DiagLevel.WARN)
            reasons.append("drop rate above warn threshold")

        if self._consecutive_misses >= self.consecutive_miss_error:
            level = self._worse(level, DiagLevel.ERROR)
            reasons.append("consecutive misses above error threshold")
        elif self._consecutive_misses >= self.consecutive_miss_warn:
            level = self._worse(level, DiagLevel.WARN)
            reasons.append("consecutive misses above warn threshold")

        if staleness_s >= self.stale_error_s:
            level = self._worse(level, DiagLevel.ERROR)
            reasons.append("state staleness above error threshold")
        elif staleness_s >= self.stale_warn_s:
            level = self._worse(level, DiagLevel.WARN)
            reasons.append("state staleness above warn threshold")

        if cmd_stall_s >= self.cmd_stall_error_s:
            level = self._worse(level, DiagLevel.ERROR)
            reasons.append("cmd_seq not advancing")
        elif cmd_stall_s >= self.cmd_stall_warn_s:
            level = self._worse(level, DiagLevel.WARN)
            reasons.append("cmd_seq advancing slowly")

        values = {
            "window_cycles": str(int(self.window_cycles)),
            "target_rate_hz": "{:.3f}".format(float(self.target_rate_hz)),
            "target_period_s": "{:.6f}".format(float(self.target_period_s)),
            "loop_period_mean_s": "{:.6f}".format(float(period_mean)),
            "loop_period_p95_s": "{:.6f}".format(float(period_p95)),
            "loop_period_max_s": "{:.6f}".format(float(period_max)),
            "loop_jitter_p95_s": "{:.6f}".format(float(jitter_p95)),
            "read_latency_mean_s": "{:.6f}".format(float(latency_mean)),
            "read_latency_p95_s": "{:.6f}".format(float(latency_p95)),
            "read_latency_max_s": "{:.6f}".format(float(latency_max)),
            "drop_cycles_total": str(int(self._drop_cycles)),
            "drop_rate": "{:.6f}".format(float(drop_rate)),
            "consecutive_misses": str(int(self._consecutive_misses)),
            "state_staleness_s": (
                "inf" if math.isinf(staleness_s)
                else "{:.6f}".format(float(staleness_s))
            ),
            "cmd_seq": str(int(self._latest_cmd_seq)),
            "cmd_seq_stall_s": (
                "inf" if math.isinf(cmd_stall_s)
                else "{:.6f}".format(float(cmd_stall_s))
            ),
            "state_seq": str(int(self._latest_state_seq)),
            "state_seq_stall_s": (
                "inf" if math.isinf(state_stall_s)
                else "{:.6f}".format(float(state_stall_s))
            ),
        }
        message = ("loop/read metrics within thresholds"
                   if not reasons else "; ".join(reasons))
        return DiagItem(
            name="bus/generic_metrics",
            level=level,
            message=message,
            values=values,
        )


def _diagnose_stall(before, after, channel="can0"):
    """Say whether the BUS was erroring while the transmit queue was stalled.

    This is the measurement that separates two faults which present identically
    -- "the queue filled and the servicer gave up" -- and which have opposite
    fixes. Without it, the cause has to be guessed from stats lines a second
    apart.

    Returns a sentence, never raises: a diagnostic that can fail is worse than
    none, because it fails exactly when something else already has.
    """
    if not before or not after:
        return ("  [no link counters: not socketcan, or `ip` unavailable, so "
                "whether the bus was erroring is UNKNOWN]")
    delta = {k: after.get(k, 0) - before.get(k, 0) for k in before}
    interesting = {k: v for k, v in delta.items() if v}
    errs = sum(delta.get(k, 0) for k in
               ("error-warn", "error-pass", "bus-off", "bus-errors"))
    detail = ", ".join("{} {:+d}".format(k, v)
                       for k, v in sorted(interesting.items())) or "nothing moved"
    if errs > 0:
        return ("  BUS ERRORS DURING THE STALL ({}). This is a PHYSICAL fault -- "
                "termination, CAN_H/CAN_L, a missing common-mode ground, or "
                "noise from motor current. No queue depth or retry budget fixes "
                "it; measure 60 ohm across CAN_H-CAN_L with the power off and "
                "check the wiring that moves with the arm.".format(detail))
    return ("  NO bus errors during the stall ({}). The CAN layer was healthy, "
            "so the stall is BELOW it -- the USB adapter or its driver. Queue "
            "depth (`ip link set {} txqueuelen`) and --max-tx-stall-cycles are "
            "the levers here; wiring is not.".format(detail, channel))


def _expose_move_time_parameter(servo_node, bus):
    """Let the board's move duration be tuned WHILE the robot is moving.

    Kondo's move byte (rcb4 calls it `frame_count`; the board interpolates to
    each target over `byte * 10 ms`) decides how smooth a trajectory looks, and
    the only way to judge it is to watch the joints move. rcb4 put it on
    dynamic_reconfigure for exactly that reason. Restarting the servicer per
    value would also drop torque per value, which makes a sweep both slow and
    unsafe.

    Declared only when the bus HAS the knob. A parameter that exists on every
    bus and silently does nothing on most of them is worse than its absence:
    the operator turns it and watches for a change that was never possible.
    """
    if not hasattr(bus, "move_time_ms"):
        return
    from rcl_interfaces.msg import SetParametersResult

    node = servo_node.node
    current = bus.move_time_ms
    node.declare_parameter(
        "move_time_ms", -1.0 if current is None else float(current))

    def on_set(params):
        for param in params:
            if param.name != "move_time_ms":
                continue
            value = float(param.value)
            # <= 0 means "use the actuator's own default", which is what an
            # unset config means. Represented as a number because a ROS
            # parameter cannot be None, and named here so it is not folklore.
            previous = bus.move_time_ms
            bus.move_time_ms = None if value <= 0 else value
            try:
                # A bus that cannot convert the value (the RCB-4mini, whose
                # move-byte unit is unmeasured) raises here. Refuse the
                # parameter instead of keeping a value that would make the
                # next write raise and stop the servicer.
                bus._bus_move_byte()
            except ValueError as err:
                bus.move_time_ms = previous
                return SetParametersResult(successful=False, reason=str(err))
            node.get_logger().warn(
                "move_time_ms -> {} (board move byte {}); this changes how "
                "smooth every subsequent trajectory is".format(
                    "actuator default" if bus.move_time_ms is None
                    else "{:.0f} ms".format(bus.move_time_ms),
                    # The bus converts with its own board's frame time.
                    "default" if bus.move_time_ms is None
                    else bus._bus_move_byte()))
        return SetParametersResult(successful=True)

    node.add_on_set_parameters_callback(on_set)
    print("[actuo_bus_node] tunable: ros2 param set {} move_time_ms <ms>  "
          "(<=0 = the actuator's default)".format(node.get_name()), flush=True)


def build_command_filter(spec, joint_names):
    """Import and build a ``CommandFilter`` from a ``module:factory`` string.

    actuo does not ship one and cannot: a filter that judges self-collision
    needs link geometry, which does not belong in the bus layer -- mixing robot
    geometry into the driver couples every driver to every robot. So the
    robot's own package supplies it and names it here -- e.g.
    ``--command-filter my_robot_safety.collision_filter:build``.

    The factory is called with the bus's joint names, in bus order, because a
    filter that has to be told the order separately is a filter that will one
    day be told the wrong one.

    Every failure here is fatal and loud. A servicer that started with a
    misspelled filter, logged a warning and ran unguarded would be worse than
    one that refused: the operator asked for a guard and would believe they had
    it.
    """
    if not spec:
        return None
    if ":" not in spec:
        raise ValueError(
            "--command-filter wants 'module:factory', got {!r}".format(spec))
    module_name, _, factory_name = spec.partition(":")
    import importlib
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise RuntimeError(
            "--command-filter {!r}: cannot import {!r} ({}). It must be "
            "importable by THIS interpreter -- the servicer's venv, not the "
            "one the robot package was built with.".format(
                spec, module_name, error))
    try:
        factory = getattr(module, factory_name)
    except AttributeError:
        raise RuntimeError(
            "--command-filter {!r}: {} has no {!r}".format(
                spec, module_name, factory_name))
    command_filter = factory(list(joint_names))
    if not hasattr(command_filter, "filter"):
        raise RuntimeError(
            "--command-filter {!r} returned {!r}, which has no filter(); see "
            "actuo.core.contracts.CommandFilter".format(
                spec, type(command_filter).__name__))
    print("[actuo_bus_node] command filter: {} ({} joints). Every command "
          "passes through it before the wire.".format(
              spec, len(joint_names)), flush=True)
    return command_filter


def _needs_position(actuator) -> bool:
    """Must this actuator report a pose before the bus may start?

    Almost always yes -- starting from an unknown pose is how a lurch happens.
    A continuous-rotation wheel servo is the exception and not a special case:
    it declares ``has_position_sense=False``, so there is no pose to be unknown
    ABOUT and nothing that could jump. Asked of the capability rather than of
    the plugin name, so the next such device needs no edit here.
    """
    return bool(actuator.capabilities.has_position_sense)


def _is_present(bus, actuator) -> bool:
    """Whether an actuator answers, as published in the shm presence flag.

    A bus that tracks presence (a hot-plug bus) is asked. Any other bus never
    takes a joint away once it has reported -- its start-up refuses to run
    without every joint -- so there it is "has reported".
    """
    present = getattr(bus, "present", None)
    if callable(present):
        return bool(present(actuator.id))
    return bus.last_state(actuator.id) is not None


def _where(bus, actuator) -> str:
    """`` on <where>`` for a bus that knows where a servo is plugged in (e.g.
    which RCB-4mini port), else ''."""
    locate = getattr(bus, "where", None)
    if not callable(locate):
        return ""
    where = locate(actuator.id)
    return " on {}".format(where) if where else ""


def _absence_reason(bus, actuator) -> str:
    """`` (<why>)`` for a bus that can say why a servo reads absent."""
    why = getattr(bus, "absence_reason", None)
    if not callable(why):
        return ""
    reason = why(actuator.id)
    return " ({})".format(reason) if reason else ""


def _fault_name(actuator, error: int) -> str:
    """Human-readable fault naming delegated to the actuator model."""
    name_fn = getattr(actuator, "fault_name", None)
    if callable(name_fn):
        return str(name_fn(int(error)))
    return "unknown fault code {}".format(int(error))


def run_sim(args):
    import mujoco

    model_path = args.model
    print("[actuo_bus_node] sim plant:", model_path, flush=True)
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)

    joint_names = args.joints.split(",")
    n = len(joint_names)
    qpos_adr, dof_adr, act_adr = [], [], []
    for jn in joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise ValueError("joint {!r} not in model".format(jn))
        qpos_adr.append(model.jnt_qposadr[jid])
        dof_adr.append(model.jnt_dofadr[jid])
        aid = jid  # position actuator per joint, same order in this model
        act_adr.append(aid)

    # The lock is what hardware components check before believing the segment.
    _lock = acquire_servicer_lock(args.shm_name)  # noqa: F841 -- held to exit
    shm = _create_shm(args.shm_name, n)
    print("[actuo_bus_node] created shm '{}' ({} joints); stepping at {} Hz"
          .format(args.shm_name, n, args.rate), flush=True)

    # Optional interactive physics viewer -- the Gazebo-like window. On macOS this
    # process MUST be launched with `mjpython` (Cocoa main-thread requirement);
    # under regular python launch_passive raises with that instruction.
    viewer = None
    if args.view:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(model, data)
        print("[actuo_bus_node] interactive MuJoCo viewer open", flush=True)

    dt = 1.0 / args.rate
    substeps = max(1, int(round(dt / model.opt.timestep)))
    state = np.zeros((n, STATE_STRIDE), dtype=np.float64)
    # Every simulated joint is always there.
    state[:, STATE_PRESENT] = 1.0
    next_t = time.time()
    try:
        while True:
            # stride 3 per joint: [position, velocity, effort]; this plant's
            # actuators are position-driven, so take that slot. NaN is an
            # unowned slot: that actuator keeps its previous target.
            cmd = shm.read_command()
            target = cmd[0::CMD_STRIDE][:n]
            owned = np.isfinite(target)
            data.ctrl[:n][owned] = target[owned]
            for _ in range(substeps):
                mujoco.mj_step(model, data)
            for i in range(n):
                state[i, 0] = data.qpos[qpos_adr[i]]
                state[i, 1] = data.qvel[dof_adr[i]]
            shm.write_state(state)
            if viewer is not None:
                if not viewer.is_running():
                    break
                viewer.sync()
            next_t += dt
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()
    except KeyboardInterrupt:
        pass
    finally:
        if viewer is not None:
            viewer.close()
        shm.close()
        print("[actuo_bus_node] stopped", flush=True)


#: How often the servicer asks the models that answer only when addressed.
#: The staleness watchdog allows a second, so 50 Hz is ample and the loop rate
#: is not the right number: at 200 Hz the same solicit is six frames every 5 ms
#: on an interface whose queue holds ten.
SOLICIT_PERIOD_S = 1.0 / 50.0


def servicer_lock_path(shm_name):
    """The lock file guarding one shm name. NOT the shm file itself.

    Separate on purpose: the shm file is deleted and recreated as part of a
    normal bring-up, and a lock held on a deleted inode guards nothing. The
    path is :func:`actuo.drivers.shm.lock_path`, which tools that write to the
    bus themselves check (``lock_holder``) to refuse running beside a servicer.
    """
    return lock_path(shm_name)


def acquire_servicer_lock(shm_name):
    """Take an exclusive lock for this shm name, or fail saying who has it.

    Two servicers on one bus is not a configuration anyone chooses; it happens
    by relaunching without noticing the old one, and it does NOT announce
    itself. Both write commands to the same motors and the same shm block, two
    controller_managers appear under one node name, and the spawner then gets a
    mixed answer -- "controller already loaded" from one and "no controller with
    this name exists" from the other -- so arm_controller ends up loaded but
    never activated. Its action server still accepts every goal and reports
    success, and the arm simply never moves -- a fault that sends everyone to
    look at the arm rather than at the process list.

    flock, not a pidfile: the kernel drops it when the process dies however it
    dies, so a crashed servicer never leaves a lock that blocks the next one.
    """
    path = servicer_lock_path(shm_name)
    handle = open(path, "w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError(
            "another bus servicer already holds shm {!r} (lock: {}). Two "
            "servicers on one bus fight over the same motors and the same "
            "shared memory, and the damage shows up somewhere else entirely. "
            "Stop the other one first (and the launch that started it):\n"
            "    pkill -9 -f ros2_control_node; pkill -9 -f actuo_bus_node"
            .format(shm_name, path))
    handle.write("{}\n".format(os.getpid()))
    handle.flush()
    # Returned so the caller keeps a reference: closing the file releases the
    # lock, and letting it be garbage-collected would release it silently.
    return handle


def run_real(args):
    """Service a hardware bus: shm command block -> wire, wire -> shm state block.

    The command block is stride-3 ``[position, velocity, effort]`` per joint; in
    POSITION mode only the first slot is live, matching the sim path.
    """
    # Before the bus is opened, so a second servicer refuses without ever
    # putting a frame on the wire.
    _lock = acquire_servicer_lock(args.shm_name)  # noqa: F841 -- held to exit
    from actuo.core.contracts import CapabilityError, OperatingMode
    from actuo.drivers.buses.can_bus import OverspeedError, StaleFeedbackError
    from actuo.drivers.link import BoardSilent, is_link_error
    from actuo.drivers.config import (
        build_bus, load_max_velocities, load_sensor_frames, load_sensors)
    from actuo.drivers.servo import ServoSupervisor
    from actuo.drivers.stall_guard import StallError, StallGuard

    if args.ros:
        import rclpy

    # Transport-only overrides, so one robot config serves every machine: the
    # same adapter is socketcan/can0 on the Linux robot PC and gs_usb/index 0 on
    # a mac. Calibration always comes from the file.
    overrides = {}
    if args.interface:
        overrides["interface"] = args.interface
    if args.channel:
        try:
            overrides["channel"] = int(args.channel)
        except ValueError:
            overrides["channel"] = args.channel
    if args.bitrate:
        overrides["bitrate"] = args.bitrate
    if args.transport:
        overrides["plugin"] = args.transport
    import yaml
    for item in args.transport_param:
        key, sep, value = item.partition("=")
        if not sep:
            raise SystemExit("--transport-param takes KEY=VALUE, got {!r}"
                             .format(item))
        overrides[key] = yaml.safe_load(value)

    servo_node = None
    bus, joint_names = build_bus(args.config, transport_overrides=overrides or None)
    # After build_bus: it imports the python-can transport if, and only if,
    # this config uses it.
    TransmitQueueFull = transmit_queue_full_type()
    if args.move_time_ms is not None and hasattr(bus, "move_time_ms"):
        bus.move_time_ms = (None if args.move_time_ms <= 0
                            else float(args.move_time_ms))
        # Fail at startup, not on the first write with torque on.
        bus._bus_move_byte()
    set_names = getattr(bus, "set_joint_names", None)
    if callable(set_names):
        set_names(joint_names)
    n = len(joint_names)
    mode = OperatingMode[args.mode.upper()]

    # Non-joint devices on the same bus (KJS-03 air/IMU board, ...). They share
    # the transport with the servos, so they are serviced from this same loop --
    # a second process on the same serial port is not a thing.
    sensors = [] if args.no_sensors else load_sensors(args.config)
    sensor_frames = load_sensor_frames(args.config)

    print("[actuo_bus_node] real bus from {}".format(args.config), flush=True)
    print("[actuo_bus_node] joints: {}".format(", ".join(joint_names)), flush=True)
    if sensors:
        print("[actuo_bus_node] sensors: {}".format(
            ", ".join(name for name, _ in sensors)), flush=True)

    # Capability-check every motor for the requested mode BEFORE opening the bus,
    # so an unsupported mode fails here rather than half way through a move.
    #
    # An actuator that supports exactly ONE mode is exempt, and that is not a
    # loophole -- it is what a mecanum base needs. A continuous-rotation wheel servo
    # declares VELOCITY and nothing else, so there is no ambiguity to resolve
    # and no chance of it silently doing the wrong thing: the supervisor gives
    # it the only mode it has (ServoSupervisor.modes), which is exactly what an
    # actuator declaring one capability means. An actuator that supports several
    # but not the requested one is still an error, because then the answer would
    # have to be guessed.
    from actuo.drivers.servo import resolve_modes
    try:
        joint_modes = resolve_modes(bus.actuators, joint_names, mode)
    except ValueError as err:
        raise RuntimeError(str(err))
    for name, actuator, joint_mode in zip(joint_names, bus.actuators,
                                          joint_modes):
        try:
            actuator.capabilities.require(joint_mode)
        except Exception as err:
            raise RuntimeError("{}: {}".format(name, err))
    for name, joint_mode in zip(joint_names, joint_modes):
        if joint_mode is not mode:
            print("[actuo_bus_node] {} runs in {} mode, not {}: its actuator "
                  "supports nothing else".format(
                      name, joint_mode.name, mode.name), flush=True)

    bus.transport.open()
    try:
        verify_servos = getattr(bus, "verify_servos", None)
        if callable(verify_servos):
            startup_items = verify_servos()
            for item in startup_items:
                stream = sys.stderr if item.level >= DiagLevel.WARN else sys.stdout
                print(
                    "[actuo_bus_node] startup {}: {} ({})".format(
                        item.name,
                        item.message,
                        ", ".join(
                            "{}={}".format(key, value)
                            for key, value in sorted(item.values.items())
                        )
                        if item.values else "no details",
                    ),
                    file=stream,
                    flush=True,
                )

        # A shallow transmit queue is a latent abort, not a tuning preference:
        # if overflowing it killed this process outright, on_exit=Shutdown()
        # would drop the arm. The transport can only report it --
        # raising it needs root -- so say so now, while somebody is watching the
        # terminal, rather than in the traceback afterwards.
        advice = getattr(bus.transport, "tx_queue_advice", None)
        if advice is not None:
            message = advice()
            if message:
                print("[actuo_bus_node] WARNING: {}".format(message),
                      file=sys.stderr, flush=True)

        # Identify every sensor board BEFORE the control loop starts. This is
        # what teaches a KJS-03 its IMU full scale; without it the board reports
        # raw counts only and the telemetry publisher deliberately emits no Imu
        # message rather than one that looks like m/s^2 and is not.
        live_sensors = []
        for name, sensor in sensors:
            try:
                reply = bus.transport.transact(
                    sensor.encode_identity_query(), args.sensor_timeout)[0]
                if reply is None:
                    raise TimeoutError("no reply")
                identity = sensor.decode_identity(reply)
            except Exception as err:                              # noqa: BLE001
                # A missing sensor must not stop the arm: the joints are the
                # safety-relevant path and they answered. Say so loudly and
                # carry on WITHOUT it, rather than publishing an empty topic
                # that reads as "the board is there and quiet".
                print("[actuo_bus_node] sensor {!r} did not identify ({}); its "
                      "telemetry will NOT be published".format(name, err),
                      file=sys.stderr, flush=True)
                continue
            live_sensors.append((name, sensor))
            print("[actuo_bus_node]   sensor {:<12s} {}".format(
                name, " ".join("{}={}".format(k, v)
                               for k, v in sorted(identity.items()))), flush=True)

        # Refuse to start until every motor is actually talking. Seeding the
        # controller from a guessed state is how you get a lurch on enable.
        #
        # Retry rather than one shot: a SYNC_READ on an FTDI bus regularly drops
        # ONE servo's reply (often the last in the group), and a single dropped
        # reply must not fail start-up when the servo is plainly there -- the
        # very next read gets it. read_batch caches what did answer, so this
        # accumulates the stragglers across attempts until every joint has
        # reported or the whole feedback_timeout budget is spent.
        # The retry must survive read_batch RAISING, not just returning gaps.
        # CanBus.read_batch throws StaleFeedbackError the moment one actuator
        # has never reported, so the previous form -- one 0.1 s call, then a
        # loop -- threw on that first call and the loop below was unreachable.
        # The effective start-up budget was 0.1 s, not feedback_timeout, and a
        # bus that went quiet for one tenth of a second killed the servicer and
        # then ros2_control with it (its on_activate aborts the process after
        # 10 s without state) -- with every motor plainly alive on the bus
        # seconds later.
        deadline = time.monotonic() + args.feedback_timeout
        needs_position = [_needs_position(a) for a in bus.actuators]
        velocity_sensed = [bool(a.capabilities.has_velocity_sense)
                           for a in bus.actuators]
        states = None
        stale = None
        ask = getattr(bus, "solicit", None)
        while time.monotonic() < deadline:
            try:
                # Before the read, not after. read_batch only ever receives, so
                # a model that answers when addressed and nowhere else -- 2.0
                # MIT -- is silent here forever and the servicer aborts naming
                # its ids, on a robot whose motors are all awake: "no status
                # frame ever received from actuator(s) [1,7,4,2,8,3]".
                # Costs nothing where nothing needs it; solicit returns 0.
                if callable(ask):
                    ask()
                states = bus.read_batch(timeout=0.1)
            except StaleFeedbackError as err:
                # Some actuator has still never spoken. Keep the reason: if the
                # budget runs out it is the message worth showing, since it
                # names the silent ids.
                stale = err
                states = None
                continue
            if not any(s is None or (s.position is None and needs_position[i])
                       for i, s in enumerate(states)):
                break
            if getattr(bus, "hot_plug", False) and all(
                    bus.present(a.id) == (s is not None)
                    for a, s in zip(bus.actuators, states)):
                # Every servo the bus calls present has reported; the rest
                # are absent, and waiting out the budget would not change it.
                break
        # A hot-plug bus starts with whatever is plugged in: a configured
        # servo that never reported is ABSENT -- published as such, and its
        # module's hardware component refuses to activate -- not a failure.
        hot_plug = bool(getattr(bus, "hot_plug", False))
        if states is None and hot_plug:
            states = [None] * n
        if states is None:
            raise stale if stale is not None else RuntimeError(
                "no actuator reported within {:.1f}s".format(
                    args.feedback_timeout))
        for name, actuator, state in zip(joint_names, bus.actuators, states):
            if hot_plug and state is None:
                print("[actuo_bus_node]   {:<52s} ABSENT (hot-plug: joins when "
                      "it answers)".format(name), flush=True)
                continue
            if state is None or (state.position is None
                                 and _needs_position(actuator)):
                raise RuntimeError(
                    "{} (id {}) did not report a position within {:.2f}s -- the "
                    "bus cannot start from an unknown pose".format(
                        name, actuator.id, args.feedback_timeout))
            if state.position is None:
                # A wheel. Not a failure and not a pose: it has no position
                # sense at all, so there is nothing to start from and nothing
                # that could jump. Skipped rather than defaulted.
                continue
            if state.error:
                raise RuntimeError(
                    "{} (id {}) reports error code {} before we start".format(
                        name, actuator.id, state.error
                    )
                )
            # Temperature is optional: an actuator that does not read it each
            # cycle (e.g. an XL430 whose cyclic block stops before the temp
            # register) reports None, and the startup summary must not assume it.
            temp = ("{:.0f}C".format(state.temperature)
                    if state.temperature is not None else "T?")
            print("[actuo_bus_node]   {:<52s} start={:+.4f} rad  {}{}".format(
                name, state.position, temp, _where(bus, actuator)), flush=True)

        # Recover each joint's gear window BEFORE anything downstream sees a
        # position. The AK series' rotor encoder is single-turn absolute, so a
        # power cycle can lose which rotor turn the output shaft is on -- an
        # integer number of 360/gear_ratio output degrees -- e.g. a joint left
        # untouched across a power cycle reading 160.40 deg before and
        # 100.40 deg after, exactly one 60 deg window.
        #
        # This must happen before _create_shm and before the measured state is
        # published, because the C++ side seeds its command from that state. A
        # correction applied afterwards would leave the controller commanding a
        # pose one whole window from the truth, which is the lurch the whole
        # servo design exists to prevent.
        #
        # The reference is REMEMBERED, not predicted. A constant in the config is
        # trying to guess where a limp arm settles, which depends on where it was
        # released and what it is leaning on: a limp joint can drift ~12 deg,
        # and after an abort mid-motion fall ~24 deg from a configured pose --
        # far enough that the snap refuses and the stack will not start. The
        # position this servicer itself recorded last run is only
        # wrong by however far the arm sagged with the power off. The config
        # constant remains the first-run fallback.
        window_overrides = parse_window_references(args.window_reference)
        from actuo.drivers.config import load_joint_datums
        # Each remembered position is converted from the datum it was
        # recorded under to the config's current one: a re-zero between two
        # runs must not leave the memory in the old frame.
        datums = load_joint_datums(args.config)
        remembered = window_state.load_positions(args.shm_name, datums)
        if remembered:
            age = window_state.load_age(args.shm_name)
            print("[actuo_bus_node] gear-window reference: remembered from {} "
                  "({}), covering {}".format(
                      window_state.state_path(args.shm_name),
                      "{:.0f} s ago".format(age) if age is not None
                      else "age unknown",
                      ", ".join(sorted(remembered))), flush=True)
        snapped = False
        #: (joint name, actuator) for joints under window recovery. These are
        #: the ones whose position must be remembered for the next start-up.
        window_joints = []
        for name, actuator in zip(joint_names, bus.actuators):
            reference = getattr(actuator, "startup_reference_rad", None)
            if reference is None:
                continue
            window_joints.append((name, actuator))
            # The config field stays the opt-in signal: a joint that never asked
            # for window recovery does not get it just because a file mentions
            # it. What the file changes is only WHICH reference is used.
            source = "config"
            if name in remembered:
                reference = remembered[name]
                source = "remembered"
            # An explicit operator assertion outranks both. Deliberately not
            # persisted as itself -- what gets remembered is the resulting
            # measured position, so a one-off override cannot quietly become a
            # permanent constant nobody re-examines.
            if name in window_overrides:
                reference = window_overrides[name]
                source = "--window-reference"
                print("[actuo_bus_node]   {} gear-window reference OVERRIDDEN "
                      "to {:+.2f} deg by --window-reference".format(
                          name, math.degrees(reference)), flush=True)
            state = bus.last_state(actuator.id)
            if state is None or state.position is None:
                continue
            before = state.position
            try:
                offset = actuator.snap_to_window(before, reference)
            except (ValueError, AttributeError) as err:
                # REFUSE to start rather than run on a position that may be one
                # window out. A wrong window looks like a perfectly ordinary
                # pose, so there is nothing downstream that could catch it.
                raise RuntimeError(
                    "{}: cannot establish the gear window at startup. Either "
                    "the joint was moved by hand while the power was off, or "
                    "the reference is stale -- delete {} to fall back to the "
                    "config's startup_reference_deg, put the robot in that "
                    "pose, or remove that joint's startup_reference_deg to "
                    "disable window recovery entirely. {}".format(
                        name, window_state.state_path(args.shm_name), err))
            if offset:
                snapped = True
                # No "-> after" computed here: whether a window moves the
                # model angle up or down depends on the joint's direction, and
                # a hand-derived value printed -159.90 for a joint that was
                # correctly at +0.10. The re-read below prints what the
                # actuator actually decodes now.
                print("[actuo_bus_node]   {} gear window {:+d} ({:+.2f} deg, "
                      "wire) [{}]: reported {:+.2f} deg, reference {:+.2f} deg"
                      .format(name, offset,
                              offset * actuator.calibration.gear_window_deg,
                              source, math.degrees(before),
                              math.degrees(reference)), flush=True)
        if snapped:
            # Re-read so every published position is decoded with the new
            # offsets. Cheap: most of these drivers stream status unsolicited,
            # and the ones that do not are asked first.
            ask = getattr(bus, "solicit", None)
            if callable(ask):
                ask()
            states = bus.read_batch(timeout=args.feedback_timeout)
            for name, state in zip(joint_names, states):
                if state is None or state.position is None:
                    continue
                print("[actuo_bus_node]   {:<52s} windowed={:+.4f} rad "
                      "({:+.2f} deg)".format(name, state.position,
                                             math.degrees(state.position)),
                      flush=True)

        shm = _create_shm(args.shm_name, n)
        print("[actuo_bus_node] created shm '{}' ({} joints); servicing at {} Hz"
              .format(args.shm_name, n, args.rate), flush=True)

        # Publish the MEASURED state before any hardware component can read
        # it: a component seeds its command from exactly this. Every command
        # slot starts unowned (NaN), so there is nothing to wait for -- torque
        # is off, and enabling refuses a joint no controller commands.
        state_block = np.zeros((n, STATE_STRIDE), dtype=np.float64)
        for i, st in enumerate(states):
            if st is None:
                # Absent (hot-plug): no pose to claim. The presence flag says
                # so and keeps its module's component inactive.
                state_block[i, 0] = math.nan
                state_block[i, 1] = math.nan
                continue
            # NaN, never 0.0, for an actuator with no position sense -- and
            # never `None`, which numpy refuses outright. A wheel reports no
            # pose at all, and 0.0 here would be the C++ side's seed for its
            # command block, i.e. the robot's opening claim about where it is.
            state_block[i, 0] = (st.position if st.position is not None
                                 else math.nan)
            if st.velocity is not None:
                state_block[i, 1] = st.velocity
            elif not bus.actuators[i].capabilities.has_velocity_sense:
                state_block[i, 1] = math.nan
        for i, actuator in enumerate(bus.actuators):
            state_block[i, STATE_PRESENT] = 1.0 if _is_present(
                bus, actuator) else 0.0
        shm.write_state(state_block)

        if not args.ros:
            # --no-ros forces torque on below, with nobody to refuse it: wait
            # for the controller to command every present position joint, and
            # check that the first command does not jump the robot.
            commanded = [i for i, st in enumerate(states)
                         if st is not None and st.position is not None]
            if not wait_for_commands(shm, commanded, args.timeout):
                raise RuntimeError(
                    "the controller never commanded every joint; is ros2_control "
                    "running with the actuo_hardware/ActuoSystem plugin and "
                    "shm_name='{}'?".format(
                        args.shm_name))
            seeded = shm.read_command()
            jumps = []
            for i in commanded:
                have = states[i].position
                want = seeded[i * CMD_STRIDE + 0]
                if abs(want - have) > args.max_start_jump:
                    jumps.append((joint_names[i], have, want))
            if jumps:
                detail = "; ".join(
                    "{} is at {:+.4f} rad but the controller would command "
                    "{:+.4f}".format(name, have, want)
                    for name, have, want in jumps)
                raise RuntimeError(
                    "refusing to start: the first command would jump the robot. "
                    "{}. Raise --max-start-jump if this is expected.".format(detail))

        # A control rate the adapter cannot answer at is not a slow robot, it is
        # a frozen one: every read times out, the last state is republished
        # forever, and every counter says the loop is keeping up. Refuse, and
        # name both remedies -- this is arithmetic (a reply cannot arrive before
        # the adapter is willing to send it), not a guess.
        latency_ms = getattr(bus.transport, "latency_timer_ms", None)
        read_budget_ms = 1000.0 * min(1.0 / args.rate,
                                      getattr(bus, "read_timeout", 1.0))
        if latency_ms is not None and read_budget_ms < latency_ms:
            port = os.path.basename(os.path.realpath(bus.transport.channel))
            raise RuntimeError(
                "this adapter holds replies for up to {:.0f} ms (its USB "
                "latency timer), but one cycle at --rate {:g} leaves only "
                "{:.1f} ms to read one. No reply can arrive, so the robot "
                "would publish a FROZEN pose while every counter looked "
                "healthy.\n"
                "  Either lower the timer (takes effect immediately, survives "
                "until replug):\n"
                "    echo 1 | sudo tee /sys/bus/usb-serial/devices/{}/latency_timer\n"
                "  and to make it permanent:\n"
                "    echo 'ACTION==\"add\", SUBSYSTEM==\"usb-serial\", "
                "DRIVER==\"ftdi_sio\", ATTR{{latency_timer}}=\"1\"' | "
                "sudo tee /etc/udev/rules.d/99-actuo-ftdi-latency.rules\n"
                "  or run slower: --rate {:g} or below."
                .format(latency_ms, args.rate, read_budget_ms, port,
                        1000.0 / latency_ms))

        stall_guard = StallGuard(math.radians(args.stall_error_deg),
                                 args.stall_effort_nm, args.stall_seconds)

        # Torque starts OFF. The supervisor owns that decision and is the only
        # thing allowed to put a motion frame on the wire.
        supervisor = ServoSupervisor(
            bus, joint_names, jump_threshold=args.max_start_jump,
            command_stride=CMD_STRIDE,
            # The mode decides which shm slot is read and what the enable guard
            # means. It is the same --mode the capability check above already
            # validated every actuator against, so an actuator that cannot do it
            # has failed before we get here.
            mode=mode, velocity_threshold=args.max_start_velocity,
            # The filter is built over the POSE joints only, and gets them in
            # the same order the supervisor will hand them over. A collision
            # filter asks "would this pose collide"; a wheel's command is a
            # speed, and a filter built over all thirteen joints would be handed
            # nine numbers and silently mis-align every one of them.
            command_filter=build_command_filter(
                args.command_filter,
                [n for n, m in zip(joint_names, joint_modes)
                 if m is OperatingMode.POSITION]),
            # Per joint `max_velocity` in the config: the fastest a position
            # command may move on the wire. See ServoSupervisor.
            max_velocity=load_max_velocities(args.config))

        recovery = None
        if args.reconnect:
            from actuo.drivers.link import LinkRecovery

            def _recovery_log(message, is_error):
                print("[actuo_bus_node] " + message,
                      file=sys.stderr if is_error else sys.stdout, flush=True)

            usb_reset = None
            if args.usb_reset:
                from actuo.drivers.usb_reset import reset_usb_device
                usb_reset = reset_usb_device
            recovery = LinkRecovery(bus, supervisor, args.reconnect_interval,
                                    log=_recovery_log,
                                    usb_reset_after=args.usb_reset_after,
                                    usb_reset=usb_reset)

        servo_node = None
        telemetry = None
        air_commands = None
        if args.ros:
            from actuo_ros.servo_node import ServoNode
            rclpy.init(args=args.ros_args)
            servo_node = ServoNode(supervisor, lambda: shm)
            _expose_move_time_parameter(servo_node, bus)
            print("[actuo_bus_node] servo action: <node>/servo/set, state: "
                  "<node>/servo/state -- torque is OFF until asked", flush=True)
            if live_sensors:
                from actuo_ros.telemetry_node import (
                    AirCommandReceiver, SensorTelemetryPublisher)
                telemetry = SensorTelemetryPublisher(
                    servo_node.node, live_sensors, sensor_frames)
                if telemetry.topics:
                    print("[actuo_bus_node] telemetry: {}".format(
                        ", ".join(telemetry.topics)), flush=True)
                else:
                    print("[actuo_bus_node] no telemetry topics: no sensor "
                          "declares both a known IMU range and a frame_id",
                          file=sys.stderr, flush=True)
                # Receives commands only; the actual bus write happens below,
                # in the main loop, on the thread that owns the transport --
                # see AirCommandReceiver's docstring for why that split exists.
                #
                # Subscribed for every live sensor regardless of has_valves:
                # that flag is only known after the FIRST state read (it comes
                # from the sensor block, not identity), which has not happened
                # yet at this point in startup. A command sent to a sensor that
                # turns out to have no valves is refused loudly where it is
                # attempted, not silently pre-filtered here on a flag that
                # would still be an untrustworthy guess.
                air_commands = AirCommandReceiver(servo_node.node, live_sensors)
                print("[actuo_bus_node] air commands: {}".format(
                    ", ".join("~/{}/air/command".format(name)
                             for name, _ in live_sensors)), flush=True)
        else:
            supervisor.enable(shm, force=True)
            print("[actuo_bus_node] --no-ros: torque FORCED on with no action "
                  "server to turn it off", flush=True)

        dt = 1.0 / args.rate
        stale_warn_s = (
            float(args.diag_stale_warn_s) if args.diag_stale_warn_s > 0.0
            else dt * float(args.diag_stale_warn_periods)
        )
        stale_error_s = (
            float(args.diag_stale_error_s) if args.diag_stale_error_s > 0.0
            else dt * float(args.diag_stale_error_periods)
        )
        cmd_warn_s = (
            float(args.diag_cmd_stall_warn_s) if args.diag_cmd_stall_warn_s > 0.0
            else dt * float(args.diag_cmd_stall_warn_periods)
        )
        cmd_error_s = (
            float(args.diag_cmd_stall_error_s) if args.diag_cmd_stall_error_s > 0.0
            else dt * float(args.diag_cmd_stall_error_periods)
        )
        loop_metrics = ServicerDiagnostics(
            target_rate_hz=args.rate,
            window_cycles=args.diag_window_cycles,
            loop_warn_ratio=args.diag_loop_warn_ratio,
            loop_error_ratio=args.diag_loop_error_ratio,
            loop_spike_error_ratio=args.diag_loop_spike_error_ratio,
            latency_warn_ratio=args.diag_latency_warn_ratio,
            latency_error_ratio=args.diag_latency_error_ratio,
            drop_rate_warn=args.diag_drop_rate_warn,
            drop_rate_error=args.diag_drop_rate_error,
            consecutive_miss_warn=args.diag_consecutive_miss_warn,
            consecutive_miss_error=args.diag_consecutive_miss_error,
            stale_warn_s=stale_warn_s,
            stale_error_s=stale_error_s,
            cmd_stall_warn_s=cmd_warn_s,
            cmd_stall_error_s=cmd_error_s,
        )

        diagnostics = None
        diag_dt = 0.0
        next_diag = float("inf")
        if args.ros and args.diagnostics:
            from actuo_ros.diagnostics_node import (
                BusDiagnosticsPublisher, hardware_id_for_bus)

            diagnostics = BusDiagnosticsPublisher(
                servo_node.node, bus,
                lambda: [loop_metrics.snapshot_item(time.monotonic())],
                hardware_id=hardware_id_for_bus(bus))
            diag_dt = 1.0 / float(args.diagnostics_rate)
            next_diag = time.time() + diag_dt
            print(
                "[actuo_bus_node] diagnostics: {}/diagnostics at {:.2f} Hz "
                "(backend diagnose() kept off the per-cycle control path)"
                .format(servo_node.node.get_namespace(), args.diagnostics_rate),
                flush=True,
            )

        next_t = time.time()
        # Instrumentation, not decoration: when a write fails we must know
        # whether it failed on cycle 1 or cycle 10000, and whether frames were
        # coming back at all; without counters that can only be guessed.
        cycles = 0
        frames_written = 0
        frames_read = 0
        sensor_reads = 0
        sensor_misses = 0
        sensor_dt = 1.0 / args.sensor_rate if args.sensor_rate > 0 else 0.0
        next_sensor = time.time()
        next_stats = time.time() + 1.0
        # {joint name: last error int seen}, on a request/response bus. Report
        # (and release) only on a CHANGE, the same shape as
        # telemetry_node.SensorTelemetryPublisher._warn_on_new_error -- a fault
        # logged every cycle at 50 Hz is a fault nobody reads, and a fault that
        # clears silently is one nobody knows cleared.
        last_fault = {}
        # {joint: was it above the temperature warning last cycle}. Same
        # change-detection shape as last_fault: report on crossing, both ways.
        last_hot = {}
        # Same change-detection shape, for the transport's retry counter: warn
        # when it MOVES, not once per cycle it happens to be non-zero.
        last_tx_retries = 0
        # Drop-or-die policy for a full transmit queue. The streak decides
        # fatality; the total is what tells an operator afterwards that it
        # happened at all.
        tx_stalls = TxStallPolicy(args.max_tx_stall_cycles)
        # The link's own error counters, sampled each stats interval. Kept so a
        # transmit stall can be diagnosed by the DELTA across it rather than by
        # comparing stats lines a second apart, which is easily misread. See
        # PythonCanTransport.link_counters.
        link_before = getattr(bus.transport, "link_counters", lambda: None)()
        stall_link_snapshot = None
        # How this bus delivers feedback. A streaming bus (CAN servo mode)
        # pushes status frames the driver drains with poll(); a request/response
        # bus (DYNAMIXEL, Feetech serial, Kondo H7) only answers when asked, so
        # its state comes from read_batch(). Detecting it once here keeps the
        # per-cycle path branch-free and lets one servicer drive either family.
        streaming = hasattr(bus, "poll")
        if streaming:
            # Idle releases at the solicit pace, not every cycle: see
            # ServoSupervisor.release_period. The streaming branch below still
            # counts every cycle as addressed, so solicit() adds nothing.
            supervisor.release_period = SOLICIT_PERIOD_S
        # Skip the per-cycle write when nothing about the command changed. On a
        # request/response bus the write and the read are serialised, so a
        # redundant write (re-sending a goal the servo is already holding, or
        # re-releasing an already-limp joint) doubles the cycle time for no
        # effect -- on an FTDI adapter, ~22 ms of pure waste per cycle, which is
        # what made a static display update at half the rate the reads allow.
        # A position servo holds its last goal in firmware, so not re-sending it
        # changes nothing on the wire. Streaming buses keep writing every cycle
        # (their timing model is different), so this only guards the else-branch.
        #
        # A REPEATED command does not rescue a joint that fell short; that is
        # measured, not assumed. A Kondo servo given a 30 deg move in 0.1 s can
        # end about 3 deg short and STAY there. Sending a fresh trajectory to
        # the same target clears it, which looks like proof that the skip is
        # the bug. It is not: jazzy's JTC starts a new trajectory from the
        # MEASURED state, so that "same target" is a moving reference, not a
        # repeat. Forcing the write every cycle while a joint sits off-target
        # leaves the error exactly where it was. The servo needs a target that
        # MOVES; repeating one it is already holding does nothing, so the skip
        # stays and the residual is the servo's, not the loop's.
        #
        # A bus can still need a write with the command unchanged: the
        # RCB-4mini's hold_relax replaces a target the servo has been held
        # short of for a while, which happens with nothing new in shm. It
        # says so through needs_write().
        bus_needs_write = getattr(bus, "needs_write", None)
        last_written_cmd = None
        addressed_everything = False
        last_solicit = 0.0
        last_on_key = None
        previous_cycle_start = None
        # When the board last said anything. Not "when we last read" -- a read
        # that times out every cycle looks identical to a healthy loop from the
        # outside, which is exactly how a servicer ends up commanding a robot
        # that is switched off.
        silent_since = None
        while True:
            cycle_start = time.monotonic()
            if previous_cycle_start is not None:
                loop_metrics.observe_cycle_period(cycle_start - previous_cycle_start)
            previous_cycle_start = cycle_start
            try:
                # The supervisor is the ONLY writer of motion frames. While
                # torque is off it emits releases and never even reads `shm`,
                # so a stale controller setpoint has no path to the wire.
                if streaming:
                    # Writes a frame to every actuator, command or release,
                    # every cycle: nothing is left to solicit.
                    supervisor.service(shm)
                    addressed_everything = True
                else:
                    # Request/response: write only when the command or the
                    # torque set actually changed. The comparison reads the same
                    # shm the supervisor would, so nothing is missed -- a new
                    # setpoint or an enable/disable still writes on the very
                    # cycle it appears.
                    on_key = tuple(supervisor.enabled_joints)
                    cmd_now = shm.read_command()
                    # equal_nan: the command block's velocity/effort slots are
                    # NaN in position mode, and NaN != NaN would make every
                    # comparison read "changed" -- which is exactly why the
                    # first cut of this skip never fired and the bus kept
                    # writing at full rate.
                    changed = (
                        last_written_cmd is None
                        or on_key != last_on_key
                        or not np.array_equal(cmd_now, last_written_cmd,
                                              equal_nan=True)
                        or (bus_needs_write is not None and bus_needs_write()))
                    if changed:
                        supervisor.service(shm)
                        last_written_cmd = np.array(cmd_now, copy=True)
                        last_on_key = on_key
                        addressed_everything = True
                # Printed on the transition only. A filter that silently held
                # the robot back looks exactly like a robot tracking badly.
                event = supervisor.pop_filter_event()
                if event:
                    print("[actuo_bus_node] " + event, flush=True)
            except TransmitQueueFull as err:
                # A full transmit queue is NOT fatal on a periodic setpoint
                # stream, and treating it as fatal drops the arm: the servicer
                # dies, on_exit=Shutdown() takes the stack with it, and torque
                # goes off wherever the arm happened to be.
                #
                # The right response is to abandon THIS cycle's command. The next
                # one is 20 ms away and carries a fresher setpoint, so a dropped
                # command is strictly better than a queued stale one -- which is
                # the same reasoning as PythonCanTransport.tx_queue_advice.
                #
                # But it must not mask a genuinely dead bus: if the queue
                # stays full for --max-tx-stall-cycles in a row,
                # commands are not reaching the motors at all while torque is on,
                # and that IS fatal. Dropping is for the burst; giving up is for
                # the bus.
                if stall_link_snapshot is None:
                    # First dropped cycle of this streak: freeze the counters so
                    # the abort can report what the BUS did while we stalled.
                    stall_link_snapshot = link_before
                if tx_stalls.dropped():
                    verdict = _diagnose_stall(
                        stall_link_snapshot,
                        getattr(bus.transport, "link_counters",
                                lambda: None)(),
                        getattr(bus.transport, "channel", "can0"))
                    raise RuntimeError(
                        "bus write failed on {} consecutive cycles (cycle {}, "
                        "{} frames written, {} read): commands are not reaching "
                        "the motors.\n{}\n{}".format(
                            tx_stalls.streak, cycles, frames_written,
                            frames_read, verdict, err))
                if tx_stalls.streak == 1:
                    print("[actuo_bus_node] WARNING: transmit queue full on "
                          "cycle {}; DROPPING this cycle's command rather than "
                          "queueing a stale one. Giving up after {} consecutive "
                          "cycles. {}".format(
                              cycles, tx_stalls.max_consecutive, err),
                          file=sys.stderr, flush=True)
            except Exception as err:
                # The transport dying is not the robot misbehaving, and the two
                # need opposite answers: a servo fault must surface, a vanished
                # USB device must be waited for. is_link_error() draws that line
                # narrowly -- an OSError with an errno that does not mean "gone"
                # still aborts, because retrying forever through a real bug is
                # worse than stopping.
                if recovery is not None and is_link_error(err):
                    recovery.recover(err)
                    last_written_cmd = None    # nothing was written; do not skip
                    last_on_key = None
                    continue
                raise RuntimeError(
                    "bus write failed on cycle {} ({} frames written, {} read, "
                    "{:.1f}s in): {}".format(
                        cycles, frames_written, frames_read,
                        time.time() - (next_stats - 1.0), err))
            else:
                # Only a clean write clears the streak. Putting this in `else`
                # rather than after the try means a drop cannot reset its own
                # counter and make an unbroken stall look intermittent.
                ended = tx_stalls.succeeded()
                if ended:
                    print("[actuo_bus_node] transmit queue recovered after {} "
                          "dropped cycle(s) on cycle {}".format(ended, cycles),
                          flush=True)
                # Counted here, not after the try: a dropped cycle put NOTHING on
                # the wire, and a tx count that includes frames the kernel
                # refused would overstate exactly the number an operator uses to
                # judge whether the bus is keeping up.
                frames_written += n
            # The cycle itself did happen, dropped command or not, so `cycles`
            # keeps counting: it is the loop's clock, and tx/cycles diverging is
            # precisely the signal that commands are being dropped.
            cycles += 1

            read_started = time.monotonic()
            had_miss = False
            had_fresh_state = False
            # Ask the models that only answer when addressed, on any cycle
            # that did not already address them.
            #
            # A broadcast-upload bus is the assumption baked into the branch
            # below: poll, and trust the motors to be talking. That is true of
            # every CubeMars servo and 3.0 driver and false of 2.0 MIT, which
            # says nothing until spoken to. A robot with both -- say six of
            # each -- would have half its joints go stale here and
            # check_fresh() would abort the servicer over hardware that is
            # healthy and waiting to be asked.
            #
            # Skipped on a cycle that wrote commands, because those frames
            # elicit the same replies and sending twice only fills the queue.
            # bus.solicit() sends a release, which is zero torque, and returns
            # 0 on a bus where nothing needs it.
            # Paced, and well below the loop. What this has to beat is the
            # staleness watchdog, which allows a second; at the servicer's 200
            # Hz an unpaced solicit is six frames every 5 ms, which fills the
            # transmit queue and is the shape that precedes an ENOBUFS abort.
            now_mono = time.monotonic()
            if (not addressed_everything
                    and now_mono - last_solicit >= SOLICIT_PERIOD_S):
                ask = getattr(bus, "solicit", None)
                if callable(ask):
                    sent = ask()
                    frames_written += sent
                    if sent:
                        last_solicit = now_mono
            addressed_everything = False
            if streaming:
                poll_counts = getattr(bus, "poll_counts", None)
                if callable(poll_counts):
                    polled, matched = poll_counts(timeout=0.0)
                else:
                    polled = bus.poll(timeout=0.0)
                    matched = polled
                frames_read += polled
                bus.check_fresh()          # raises if a motor went quiet
                # ...and if one is moving far faster than we told it to. A
                # driver can run at 113x its envelope into a stall while every
                # frame we sent is correct; it reports its own speed, so we can
                # catch that without knowing why it happens.
                bus.check_speed(tolerance=args.overspeed_tolerance)
                # Broadcast upload buses and request/response buses do NOT
                # share a "miss" definition. On request/response every cycle
                # asks for a reply, so no reply this cycle is a miss. On a
                # broadcast upload bus there is no per-cycle request at all --
                # zero frames in one cycle can be normal when upload rate is
                # lower than loop rate -- and stale_after is the mechanism that
                # knows whether a servo actually went silent.
                had_miss = False
                had_fresh_state = matched > 0
            else:
                # Request/response: one read_batch per cycle refreshes the
                # last_state cache the state-publish loop below reads. The
                # freshness and overspeed guards are CAN-specific (they watch an
                # unsolicited stream); here a servo that stops answering surfaces
                # as last_state() staying stale, handled per-joint below.
                #
                # Timeout is the per-CYCLE budget, NOT the startup
                # feedback_timeout (which is 5 s -- long, so a robot powering up
                # is waited for). Passing that 5 s here made one missed servo
                # stall the whole servicer for up to five seconds, dropping it
                # from 50 Hz to ~2 Hz and making the displayed pose lag badly.
                # `None` lets read_batch use the bus's own read_timeout (0.05 s),
                # capped to the cycle period so a miss costs at most one cycle.
                read_to = min(dt, getattr(bus, "read_timeout", dt))
                missed_before = getattr(bus, "missed", None)
                updates_before = getattr(bus, "state_updates", None)
                try:
                    states = bus.read_batch(timeout=read_to)
                except Exception as err:                       # noqa: BLE001
                    # Same rule as the write above. A read is usually where a
                    # power cut is noticed first, because it happens every cycle
                    # whether or not there is anything to command.
                    if recovery is None or not is_link_error(err):
                        raise
                    recovery.recover(err)
                    last_written_cmd = None
                    last_on_key = None
                    continue
                frames_read += sum(1 for s in states if s is not None)
                missed_after = getattr(bus, "missed", None)
                if missed_before is not None and missed_after is not None:
                    misses_this_cycle = max(0, int(missed_after) - int(missed_before))
                else:
                    misses_this_cycle = (
                        1 if any(state is None for state in states) else 0
                    )
                had_miss = misses_this_cycle > 0
                updates_after = getattr(bus, "state_updates", None)
                if updates_before is not None and updates_after is not None:
                    # The bus's own answer to "was anything refreshed this
                    # cycle". Asking it is not pedantry: subtracting misses from
                    # the actuator count gets this WRONG on a bus that reads one
                    # slot for every servo, because a cycle where nothing
                    # answered scores one miss and a dozen apparently-fresh
                    # actuators. That is how a robot published an unchanging
                    # pose for minutes while the loop looked healthy.
                    had_fresh_state = updates_after > updates_before
                else:
                    fresh_count = max(
                        0, len(bus.actuators) - int(misses_this_cycle))
                    had_fresh_state = fresh_count > 0

            # Silence is the trigger that fires when someone uses the power
            # switch. Nothing raises: the CH340 is powered by the PC, so the
            # port stays open and the reads simply stop being answered. Without
            # this the servicer keeps "servicing" a robot that is not there and
            # commands the stale pose the moment it comes back -- into servos
            # that went limp and drooped in the meantime.
            if had_fresh_state:
                silent_since = None
            elif recovery is not None and args.max_silence > 0:
                now_mono = time.monotonic()
                if silent_since is None:
                    silent_since = now_mono
                elif now_mono - silent_since >= args.max_silence:
                    recovery.recover(BoardSilent(
                        "no joint has answered for {:.1f}s -- the robot's power "
                        "switch, most likely (the USB bridge is powered by this "
                        "PC, so nothing raised)".format(now_mono - silent_since)))
                    silent_since = None
                    last_written_cmd = None
                    last_on_key = None
                    continue

            # A held joint pushing hard without following: on the bus that
            # reads it, on both the streaming and the request/response path.
            if stall_guard.enabled and supervisor.any_enabled:
                stall_guard.check(supervisor.inspect(shm), time.monotonic())

            read_latency = time.monotonic() - read_started
            loop_metrics.observe_read_latency(read_latency)
            loop_metrics.observe_read_result(
                had_miss=had_miss,
                had_fresh_state=had_fresh_state,
                now_mono=time.monotonic(),
            )

            # Sensor boards share the servos' transport, so they are read from
            # this loop -- but at their own, slower rate. They are telemetry,
            # not the control path, and every sensor transaction is time the
            # joints are not being read; at the FTDI's default 16 ms latency
            # timer, reading a board every cycle would visibly halve the rate
            # the displayed pose updates at.
            if live_sensors and time.time() >= next_sensor:
                next_sensor = time.time() + sensor_dt
                for name, sensor in live_sensors:
                    try:
                        reply = bus.transport.transact(
                            sensor.encode_query(), args.sensor_timeout)[0]
                        if reply is None:
                            sensor_misses += 1
                            continue
                        reading = sensor.decode_state(reply)
                    except Exception as err:                      # noqa: BLE001
                        # A garbled or missed sensor reply must not take the
                        # arm down: the joints are unaffected. Count it, keep
                        # servicing, and let the miss counter say it is happening.
                        sensor_misses += 1
                        if sensor_misses in (1, 100):
                            print("[actuo_bus_node] sensor {!r} read failed "
                                  "({}); {} misses so far".format(
                                      name, err, sensor_misses),
                                  file=sys.stderr, flush=True)
                        continue
                    sensor_reads += 1
                    if telemetry is not None:
                        telemetry.publish(name, reading)

            if servo_node is not None:
                servo_node.spin_once()
                servo_node.publish()

            # Air/valve commands (pump grip/release, ...): drained here, on
            # THIS loop's own thread, which is the one thing allowed to touch
            # the transport (one process, one loop, owns the bus). A ROS
            # callback firing on a different thread queued the request;
            # nothing about that queueing
            # ever writes a byte -- see AirCommandReceiver.
            if air_commands is not None:
                for name, op in air_commands.pending().items():
                    sensor = next((s for n, s in live_sensors if n == name), None)
                    if sensor is None:
                        continue    # a stale command for a sensor no longer live
                    try:
                        ack = bus.transport.transact(
                            [sensor.encode_air(op)], args.sensor_timeout)[0]
                    except (ValueError, CapabilityError) as err:
                        # Bad op name, or this board turns out to have no
                        # valves -- a request that cannot be honoured, refused
                        # loudly rather than silently dropped.
                        print("[actuo_bus_node] air {!r} on {!r} REFUSED: {}"
                              .format(op, name, err), file=sys.stderr, flush=True)
                        continue
                    if ack is None:
                        print("[actuo_bus_node] air {!r} on {!r}: no reply"
                              .format(op, name), file=sys.stderr, flush=True)
                    else:
                        print("[actuo_bus_node] air {!r} on {!r}: ok"
                              .format(op, name), flush=True)

            now_s = time.time()
            if now_s >= next_stats:
                if supervisor.all_enabled:
                    torque = "ON"
                elif supervisor.all_off:
                    torque = "off"
                else:
                    # Name them: "partial" would leave the operator guessing
                    # which limb is the one hanging limp.
                    torque = "ON({})".format(",".join(supervisor.enabled_joints))
                # err_frames/unmatched are CAN counters; a request/response bus
                # reports missed reads instead. Use whichever the bus exposes so
                # the line means something on either family rather than crashing.
                err_frames = getattr(bus, "error_frames", 0)
                unmatched = getattr(bus, "unmatched_frames",
                                    getattr(bus, "missed", 0))
                # Sensor counters only appear when there are sensors: a line
                # reading "sensors=0/0" on a robot that has none is noise.
                sensor_note = ""
                if live_sensors:
                    sensor_note = "  sensors={}/{}".format(
                        sensor_reads, sensor_reads + sensor_misses)
                # A retried transmit is not a healthy one. It only appears once
                # it has happened, so a clean run's line is unchanged, but the
                # approach to an ENOBUFS death is now visible in the stats
                # instead of arriving as the death itself.
                retries = getattr(bus.transport, "tx_retries", 0)
                retry_note = ""
                if tx_stalls.total:
                    # Dropped cycles first: a command that never reached the
                    # motors matters more than a retry that eventually did.
                    retry_note = "  tx_dropped={}cyc".format(tx_stalls.total)
                if retries:
                    retry_note += "  tx_retries={}({} stalls)".format(
                        retries, getattr(bus.transport, "tx_stalls", 0))
                    if retries != last_tx_retries:
                        print("[actuo_bus_node] WARNING: the transmit queue "
                              "filled up {} time(s) and was retried. Frames are "
                              "being offered faster than the bus carries them "
                              "away. This is what precedes an ENOBUFS abort."
                              .format(getattr(bus.transport, "tx_stalls", 0)),
                              flush=True)
                        last_tx_retries = retries
                # A robot that reconnects every few minutes has a loose cable,
                # and with recovery working that reads as "it works" unless
                # someone is counting. Appears only once it has happened.
                if recovery is not None and recovery.recoveries:
                    retry_note += "  reconnects={}".format(recovery.recoveries)
                print("[actuo_bus_node] {:6d} cycles  tx={:7d}  rx={:7d}  "
                      "err_frames={}  unmatched={}{}{}  torque={}".format(
                          cycles, frames_written, frames_read,
                          err_frames, unmatched, sensor_note, retry_note,
                          torque), flush=True)
                next_stats = now_s + 1.0

                # Record where the windowed joints are, for the NEXT start-up.
                # Written here rather than only on exit because the failure this
                # protects against is an abrupt death -- an abort, a crash, a
                # kill -- and none of those run a cleanup path. A second of
                # staleness costs a second of arm sag; missing the write costs
                # a start-up.
                link_before = getattr(bus.transport, "link_counters",
                                      lambda: None)()
                #
                # Only LIMP joints are updated: a held joint falls when this
                # process dies, so its held pose is not where the next start-up
                # will find it. See window_state.update_positions.
                if window_joints:
                    held = set(supervisor.enabled_joints)
                    positions = {}
                    for jn, actuator in window_joints:
                        if jn in held:
                            continue
                        st = bus.last_state(actuator.id)
                        if st is not None and st.position is not None:
                            positions[jn] = float(st.position)
                    window_state.update_positions(args.shm_name, positions,
                                                  held, datums)

            for i, actuator in enumerate(bus.actuators):
                # Presence first: it is published even for a joint with no
                # state, and a joint that disappears while held is released
                # here, so its servo comes back limp when it is plugged in
                # again rather than snapping to the old target.
                present = _is_present(bus, actuator)
                if present != bool(state_block[i, STATE_PRESENT]):
                    state_block[i, STATE_PRESENT] = 1.0 if present else 0.0
                    if present:
                        print("[actuo_bus_node] {} (id {}) answers again{}. It is "
                              "limp; activating its module's hardware component "
                              "adopts its pose, then enable it.".format(
                                  joint_names[i], actuator.id,
                                  _where(bus, actuator)), flush=True)
                    else:
                        held = joint_names[i] in supervisor.enabled_joints
                        supervisor.disable([joint_names[i]])
                        print("[actuo_bus_node] {} (id {}) STOPPED ANSWERING{}{}; "
                              "its module's hardware component deactivates, "
                              "every other joint keeps running.".format(
                                  joint_names[i], actuator.id,
                                  _absence_reason(bus, actuator),
                                  " and was released" if held else ""),
                              file=sys.stderr, flush=True)
                st = bus.last_state(actuator.id)
                if st is None:
                    # A request/response servo that missed this cycle's read
                    # keeps whatever it last published rather than lurching the
                    # displayed pose to 0 -- a joint shown at 0 that is really
                    # elsewhere is the one output this stack must never emit.
                    continue
                error = int(st.error or 0)
                if error != last_fault.get(joint_names[i], 0):
                    # A fault must not take the WHOLE servicer down -- every
                    # other joint's telemetry with it, with only a stack trace
                    # on stderr a moment before the process disappears.
                    # actuo_msgs/ServoState has an error/error_name field for
                    # exactly this (the DYNAMIXEL name comes from the Hardware
                    # Error Status register, not just the opaque Alert bit --
                    # see DynamixelBus._diagnose_hardware_faults), so the fault
                    # is RELEASED and REPORTED instead. Torque removal is
                    # "always allowed" and cannot itself cause a lurch
                    # (ServoSupervisor.disable), so this is strictly safer than
                    # aborting, not merely quieter.
                    last_fault[joint_names[i]] = error
                    if error:
                        ok, reason = supervisor.disable([joint_names[i]])
                        # NAME the fault. "error 1" sends an operator to a
                        # datasheet mid-incident; "motor over-temperature" says
                        # what to do about it. Fault naming is actuator-owned:
                        # the same integer means different faults on different
                        # actuator families.
                        print("[actuo_bus_node] FAULT {} (id {}): {} (error {})"
                              " -- {}".format(
                                  joint_names[i], actuator.id,
                                  _fault_name(actuator, error),
                                  error, reason if ok
                                  else "release failed: " + reason),
                              file=sys.stderr, flush=True)
                        print("[actuo_bus_node]   this joint is now LIMP; the "
                              "others keep running. It will not re-enable until "
                              "the fault clears AND you servo_on again.",
                              file=sys.stderr, flush=True)
                    else:
                        print("[actuo_bus_node] {} (id {}) fault CLEARED -- but "
                              "it is still limp. servo_on to take it back."
                              .format(joint_names[i], actuator.id), flush=True)

                # Early thermal warning. A fault is the motor protecting itself
                # AFTER the fact; by then a run is already lost. The status frame
                # carries temperature every cycle, so the approach is visible for
                # free -- reported on CROSSING, not per cycle, and on the way
                # back down too, so "it cooled" is as visible as "it is hot".
                if st.temperature is not None and args.temp_warn > 0:
                    hot = st.temperature >= args.temp_warn
                    if hot != last_hot.get(joint_names[i], False):
                        last_hot[joint_names[i]] = hot
                        if hot:
                            print("[actuo_bus_node] HOT {} (id {}): {:.0f}C, over "
                                  "the {:.0f}C warning. Not a fault yet -- the "
                                  "motor faults on its own and goes limp. Ease "
                                  "off or pause.".format(
                                      joint_names[i], actuator.id,
                                      st.temperature, args.temp_warn),
                                  file=sys.stderr, flush=True)
                        else:
                            print("[actuo_bus_node] {} (id {}) cooled to {:.0f}C"
                                  .format(joint_names[i], actuator.id,
                                          st.temperature), flush=True)
                # NaN, not 0.0, when the actuator has no position sense: a
                # wheel's reported pot angle is not a joint pose, and 0.0 would
                # reach /joint_states as a real (and stationary) wheel. The
                # controller side never claims a position interface for such a
                # joint, so nothing reads this -- it is written honestly anyway,
                # because "nothing reads it" is a property of today's URDF.
                state_block[i, 0] = (st.position if st.position is not None
                                     else math.nan)
                # velocity is None until pole_pairs is configured; shm is a
                # flat float array with no way to say "unknown", so publish the
                # state we have and leave rate at 0 rather than inventing one.
                #
                # Except where the actuator says it CANNOT sense velocity at
                # all -- a wheel. There 0.0 is not a stale reading, it is the
                # claim "this wheel is stationary", published every cycle while
                # it spins. NaN is the honest value and the consumer that
                # matters is built for it: mecanum_drive_controller skips its
                # odometry update on a NaN wheel velocity rather than
                # integrating one (mecanum_drive_controller.cpp, the isnan guard
                # before odometry_.update).
                if st.velocity is not None:
                    state_block[i, 1] = st.velocity
                elif velocity_sensed[i]:
                    state_block[i, 1] = 0.0
                else:
                    state_block[i, 1] = math.nan
            shm.write_state(state_block)
            loop_metrics.observe_shm(
                cmd_seq=shm.cmd_seq(),
                state_seq=shm.state_seq(),
                now_mono=time.monotonic(),
            )

            now_wall = time.time()
            if diagnostics is not None and now_wall >= next_diag:
                diagnostics.publish()
                next_diag = now_wall + diag_dt

            next_t += dt
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()
    except KeyboardInterrupt:
        # The launch's shutdown arrives here as SIGINT, including when ANOTHER
        # bus of the same robot aborts mid-motion. Without a release, this bus
        # would leave its joints on their last command at full gains.
        print("\n[actuo_bus_node] interrupted; releasing", flush=True)
        try:
            bus.release_all()
        except Exception as release_err:                      # noqa: BLE001
            print("[actuo_bus_node] could not release: {}".format(release_err),
                  file=sys.stderr)
    except StaleFeedbackError as err:
        # Release the rest before returning. A 2.0 MIT drive has no command
        # timeout: left alone it holds its last target at its gains for as
        # long as it has power, so one drive whose connector drops out must not
        # leave every other drive on whatever it last received. The silent
        # drive may not take the release; the others do.
        print("[actuo_bus_node] ABORT: {}".format(err), file=sys.stderr)
        try:
            bus.release_all()
        except Exception as release_err:                      # noqa: BLE001
            print("[actuo_bus_node] could not release: {}".format(release_err),
                  file=sys.stderr)
        return 1
    except StallError as err:
        # Release, do not hold: the joint is pressing into something, and
        # holding its target keeps it pressing.
        print("[actuo_bus_node] ABORT (stall): {}".format(err), file=sys.stderr)
        try:
            bus.release_all()
        except Exception as release_err:                      # noqa: BLE001
            print("[actuo_bus_node] could not release: {}".format(release_err),
                  file=sys.stderr)
        return 1
    except OverspeedError as err:
        # Release before returning: this fires when the driver is ALREADY doing
        # something we did not ask for, and leaving it holding its last target
        # would let it finish the move that triggered the abort.
        print("[actuo_bus_node] ABORT (overspeed): {}".format(err),
              file=sys.stderr)
        try:
            bus.release_all()
        except Exception as release_err:                      # noqa: BLE001
            print("[actuo_bus_node] could not release: {}".format(release_err),
                  file=sys.stderr)
        return 1
    except Exception as err:
        # Every other way out with torque possibly on -- a command refused for
        # its limit, a write that failed. Same reason as above: a 2.0 MIT
        # drive holds its last target for as long as it has power, so a goal
        # refused for its limit must not leave the other drives holding.
        # Release, then let the error out with its traceback.
        print("[actuo_bus_node] ABORT: {}".format(err), file=sys.stderr)
        try:
            bus.release_all()
        except Exception as release_err:                      # noqa: BLE001
            print("[actuo_bus_node] could not release: {}".format(release_err),
                  file=sys.stderr)
        raise
    finally:
        if servo_node is not None:
            servo_node.destroy()
            # The launch's SIGINT has usually shut the context down already;
            # raising here would skip closing the bus.
            if rclpy.ok():
                rclpy.shutdown()
        bus.transport.close()
        print("[actuo_bus_node] bus closed", flush=True)
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(description="actuo bus-servicer")
    p.add_argument("--shm-name", default="actuo")
    p.add_argument("--joints", default="joint1", help="comma-separated joint order")
    p.add_argument("--rate", type=float, default=500.0)
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--sim", action="store_true",
                   help="drive a MuJoCo plant (requires --model)")
    p.add_argument("--real", action="store_true",
                   help="drive real hardware from --config")
    p.add_argument("--config", default="",
                   help="robot bus YAML (required with --real)")
    p.add_argument("--mode", default="position",
                   choices=["position", "velocity", "current"],
                   help="operating mode for the real bus (default: position)")
    p.add_argument("--overspeed-tolerance", type=float, default=4.0,
                   help="abort if a motor reports more than this multiple of "
                        "its configured ERPM envelope (default 4x). Set high "
                        "on purpose: the runaway this catches was 113x, and a "
                        "guard that cries wolf gets widened until it is "
                        "useless.")
    p.add_argument("--stall-error-deg", type=float, default=8.0,
                   help="stall guard: a held joint this far from its command "
                        "(deg) ... (default 8)")
    p.add_argument("--stall-effort-nm", type=float, default=80.0,
                   help="... while reporting more than this effort (N*m) ... "
                        "(default 80)")
    p.add_argument("--stall-seconds", type=float, default=0.12,
                   help="... for this long without a break releases the bus "
                        "and exits (default 0.12; 0 disables). See "
                        "actuo.drivers.stall_guard for where the numbers "
                        "come from.")
    p.add_argument("--window-reference", action="append", default=[],
                   metavar="JOINT=DEG",
                   help="override the gear-window reference for one joint, in "
                        "degrees, for THIS start only (repeatable). Use when "
                        "there is no remembered position yet, or the arm was "
                        "moved by hand while off, and you can SEE roughly "
                        "where the joint is. It only picks which window -- it "
                        "needs to be right to a quarter of a window, not "
                        "exactly. From the next start the remembered position "
                        "takes over again.")
    p.add_argument("--max-start-velocity", type=float, default=0.05,
                   help="rad/s. VELOCITY/CURRENT mode counterpart of "
                        "--max-start-jump: refuse to apply torque while the "
                        "controller is commanding motion, because torque would "
                        "start the joint moving at once. There is no position "
                        "jump to guard in those modes.")
    p.add_argument("--temp-warn", type=float, default=65.0,
                   help="warn once when a joint crosses this temperature in C, "
                        "and again when it cools back under it. 0 disables. "
                        "This is an EARLY warning, well under any motor's own "
                        "thermal limit: the motor's fault is protection after "
                        "the fact, by which point the run is already lost.")
    p.add_argument("--reconnect", dest="reconnect", action="store_true",
                   default=True,
                   help="survive the robot's power being switched off: on a "
                        "transport failure, release every joint, wait for the "
                        "board to come back, re-run its startup checks, and "
                        "carry on with torque OFF. On by default. It comes "
                        "back released rather than holding on purpose -- the "
                        "robot was limp and gravity moved it while the "
                        "controller kept commanding the old pose.")
    p.add_argument("--no-reconnect", dest="reconnect", action="store_false",
                   help="die on a transport failure instead. For debugging a "
                        "link problem, where a servicer that quietly reopens "
                        "hides how often it is happening.")
    p.add_argument("--reconnect-interval", type=float, default=1.0,
                   help="seconds between reopen attempts (default 1.0)")
    p.add_argument("--usb-reset", dest="usb_reset", action="store_true",
                   default=True,
                   help="escalate to a USB-level device reset when reopening "
                        "the port does not bring the board back. This is what "
                        "rcb4 does, and it is needed because a wedged adapter "
                        "does not answer a reopened port -- measured, "
                        "including across a physical unplug and replug. Needs "
                        "write access to /dev/bus/usb; the error names the "
                        "udev rule if it does not have it.")
    p.add_argument("--no-usb-reset", dest="usb_reset", action="store_false",
                   help="never reset the USB device, only reopen the port")
    p.add_argument("--usb-reset-after", type=int, default=3,
                   help="failed reopen attempts before the first USB reset "
                        "(default 3). Not the first attempt: most outages are "
                        "the board being briefly busy or unpowered, and a "
                        "device reset is a heavier hammer than that needs.")
    p.add_argument("--max-silence", type=float, default=1.0,
                   help="seconds with NO joint answering before the board is "
                        "declared gone and torque released (default 1.0). This "
                        "is the trigger that matters on a Kondo board: its USB "
                        "bridge is powered by the PC, so switching the ROBOT "
                        "off raises nothing at all -- reads just stop coming "
                        "back. 0 disables the silence trigger and leaves only "
                        "the raises-an-exception one.")
    p.add_argument("--max-tx-stall-cycles", type=int, default=10,
                   help="consecutive cycles whose command may be DROPPED "
                        "because the CAN transmit queue was full, before "
                        "aborting (default 10 = 0.2 s at 50 Hz). Dropping a "
                        "stale setpoint beats queueing it -- the next cycle "
                        "carries a fresher one -- but a queue that never drains "
                        "means commands are not reaching the motors at all, "
                        "which is fatal rather than transient.")
    p.add_argument("--sensor-rate", type=float, default=20.0,
                   help="Hz to poll the config's non-joint sensor boards "
                        "(default 20). They share the servos' transport, so "
                        "every sensor read is time the joints are not read.")
    p.add_argument("--sensor-timeout", type=float, default=0.05,
                   help="seconds to wait for one sensor board's reply")
    p.add_argument("--no-sensors", action="store_true",
                   help="ignore the config's `sensors:` section entirely")
    p.add_argument("--feedback-timeout", type=float, default=5.0,
                   help="seconds to wait for every motor to report before starting")
    p.add_argument("--move-time-ms", type=float, default=None,
                   help="override bus.move_time_ms from the config. On a Kondo "
                        "bus this is the board's move duration per command "
                        "(rcb4's frame_count * 10 ms) and it decides how smooth "
                        "a trajectory looks. It is also a live ROS parameter, "
                        "which is the useful way to tune it.")
    p.add_argument("--command-filter", default=None, metavar="MODULE:FACTORY",
                   help="a last look at the command before the wire, supplied "
                        "by the ROBOT's package: e.g. "
                        "my_robot_safety.collision_filter:build. actuo ships "
                        "none and cannot -- self-collision needs link geometry, "
                        "which does not belong in this layer. POSITION mode "
                        "only. Absent = the code path this had before.")
    p.add_argument("--max-start-jump", type=float, default=0.05,
                   help="rad; refuse to start if the controller's seeded command "
                        "differs from the measured pose by more than this "
                        "(default 0.05 rad ~ 2.9 deg)")
    p.add_argument("--ros", action="store_true", default=True,
                   help="serve the SetServo action and publish ServoState "
                        "(default). Torque stays OFF until asked.")
    p.add_argument("--no-ros", dest="ros", action="store_false",
                   help="no ROS: torque is forced ON with no way to "
                        "release it. Bring-up only.")
    p.add_argument("--diagnostics", action="store_true", default=True,
                   help="publish bus diagnostics via diagnostic_updater "
                        "(default when --ros is enabled)")
    p.add_argument("--no-diagnostics", dest="diagnostics", action="store_false",
                   help="disable /diagnostics publication")
    p.add_argument("--diagnostics-rate", type=float, default=1.0,
                   help="Hz for backend diagnose() + generic metrics publish "
                        "(default 1.0). Kept low because Kondo diagnose() costs "
                        "multiple board round trips.")
    p.add_argument("--diag-window-cycles", type=int, default=200,
                   help="rolling window length, in control cycles, for period/"
                        "latency stats (default 200)")
    p.add_argument("--diag-loop-warn-ratio", type=float, default=1.10,
                   help="WARN when loop period p95 exceeds this multiple of the "
                        "target period (default 1.10 = 10%% slack for scheduler "
                        "jitter; heuristic)")
    p.add_argument("--diag-loop-error-ratio", type=float, default=1.30,
                   help="ERROR when loop period p95/max exceeds this multiple "
                        "of target period (default 1.30; heuristic)")
    p.add_argument("--diag-loop-spike-error-ratio", type=float, default=3.0,
                   help="ERROR when the single WORST cycle in the window "
                        "exceeds this multiple of the target period (default "
                        "3.0). Separate from --diag-loop-error-ratio, which "
                        "judges p95: one late cycle is a stall, a high p95 is "
                        "a loop that cannot keep up, and treating them alike "
                        "made a measured-healthy bus (p95 10.06 ms at a 10 ms "
                        "target) report ERROR forever. Heuristic, but this one "
                        "is at least clear of a real measurement.")
    p.add_argument("--diag-latency-warn-ratio", type=float, default=0.50,
                   help="WARN when read-latency p95 exceeds this fraction of "
                        "the target period (default 0.50; heuristic)")
    p.add_argument("--diag-latency-error-ratio", type=float, default=0.90,
                   help="ERROR when read-latency p95 exceeds this fraction of "
                        "the target period (default 0.90; heuristic)")
    p.add_argument("--diag-drop-rate-warn", type=float, default=0.02,
                   help="WARN when read-drop rate exceeds this fraction of "
                        "cycles (default 0.02; heuristic)")
    p.add_argument("--diag-drop-rate-error", type=float, default=0.10,
                   help="ERROR when read-drop rate exceeds this fraction of "
                        "cycles (default 0.10; heuristic)")
    p.add_argument("--diag-consecutive-miss-warn", type=int, default=2,
                   help="WARN at this many consecutive read misses "
                        "(default 2; heuristic)")
    p.add_argument("--diag-consecutive-miss-error", type=int, default=5,
                   help="ERROR at this many consecutive read misses "
                        "(default 5; heuristic)")
    p.add_argument("--diag-stale-warn-s", type=float, default=0.0,
                   help="WARN staleness threshold in seconds. <=0 means "
                        "computed from --rate (default 2x target period).")
    p.add_argument("--diag-stale-error-s", type=float, default=0.0,
                   help="ERROR staleness threshold in seconds. <=0 means "
                        "computed from --rate (default 5x target period).")
    p.add_argument("--diag-stale-warn-periods", type=float, default=2.0,
                   help="default WARN staleness multiplier when "
                        "--diag-stale-warn-s <= 0 (default 2x target period)")
    p.add_argument("--diag-stale-error-periods", type=float, default=5.0,
                   help="default ERROR staleness multiplier when "
                        "--diag-stale-error-s <= 0 (default 5x target period)")
    p.add_argument("--diag-cmd-stall-warn-s", type=float, default=0.0,
                   help="WARN threshold for cmd_seq not advancing, seconds. "
                        "<=0 means computed from --rate (default 2x period).")
    p.add_argument("--diag-cmd-stall-error-s", type=float, default=0.0,
                   help="ERROR threshold for cmd_seq not advancing, seconds. "
                        "<=0 means computed from --rate (default 5x period).")
    p.add_argument("--diag-cmd-stall-warn-periods", type=float, default=2.0,
                   help="default WARN multiplier for cmd_seq stall when "
                        "--diag-cmd-stall-warn-s <= 0 (default 2x period)")
    p.add_argument("--diag-cmd-stall-error-periods", type=float, default=5.0,
                   help="default ERROR multiplier for cmd_seq stall when "
                        "--diag-cmd-stall-error-s <= 0 (default 5x period)")
    p.add_argument("--interface", default="",
                   help="override bus.transport.interface (e.g. gs_usb on macOS)")
    p.add_argument("--channel", default="",
                   help="override bus.transport.channel (gs_usb takes an int index)")
    p.add_argument("--bitrate", type=int, default=0,
                   help="override bus.transport.bitrate (ignored by socketcan)")
    p.add_argument("--transport", default="",
                   help="override bus.transport.plugin, e.g. kondo_rcb4_udp "
                        "to reach a Kondo board through a WiFi relay")
    p.add_argument("--transport-param", action="append", default=[],
                   metavar="KEY=VALUE",
                   help="override one more bus.transport entry (repeatable; "
                        "VALUE is read as YAML)")
    p.add_argument("--view", action="store_true",
                   help="open the interactive MuJoCo viewer (run under mjpython on macOS)")
    p.add_argument("--model", default="",
                   help="MJCF model to step with --sim. Its first actuators "
                        "must be position actuators for --joints, in order.")
    # parse_known_args, not parse_args: launch_ros.actions.Node always appends
    # --ros-args (plus any remaps) to every node it starts, and a strict parse
    # rejects that with "unrecognized arguments" before the servicer ever
    # opens the port.
    args, ros_args = p.parse_known_args(argv)
    args.ros_args = ros_args
    if args.diagnostics_rate <= 0.0:
        print("[actuo_bus_node] --diagnostics-rate must be > 0",
              file=sys.stderr)
        return 2
    if args.diag_window_cycles < 1:
        print("[actuo_bus_node] --diag-window-cycles must be >= 1",
              file=sys.stderr)
        return 2
    if args.diag_loop_warn_ratio <= 0.0 or args.diag_loop_error_ratio <= 0.0:
        print("[actuo_bus_node] loop period ratios must be > 0", file=sys.stderr)
        return 2
    if args.diag_loop_warn_ratio > args.diag_loop_error_ratio:
        print("[actuo_bus_node] loop WARN ratio must be <= ERROR ratio",
              file=sys.stderr)
        return 2
    if args.diag_latency_warn_ratio <= 0.0 or args.diag_latency_error_ratio <= 0.0:
        print("[actuo_bus_node] latency ratios must be > 0", file=sys.stderr)
        return 2
    if args.diag_latency_warn_ratio > args.diag_latency_error_ratio:
        print("[actuo_bus_node] latency WARN ratio must be <= ERROR ratio",
              file=sys.stderr)
        return 2
    if args.diag_drop_rate_warn < 0.0 or args.diag_drop_rate_error < 0.0:
        print("[actuo_bus_node] drop-rate thresholds must be >= 0",
              file=sys.stderr)
        return 2
    if args.diag_drop_rate_warn > args.diag_drop_rate_error:
        print("[actuo_bus_node] drop-rate WARN must be <= ERROR",
              file=sys.stderr)
        return 2
    if args.diag_consecutive_miss_warn < 0 or args.diag_consecutive_miss_error < 0:
        print("[actuo_bus_node] consecutive-miss thresholds must be >= 0",
              file=sys.stderr)
        return 2
    if args.diag_consecutive_miss_warn > args.diag_consecutive_miss_error:
        print("[actuo_bus_node] consecutive-miss WARN must be <= ERROR",
              file=sys.stderr)
        return 2

    if args.sim and args.real:
        print("[actuo_bus_node] --sim and --real are mutually exclusive",
              file=sys.stderr)
        return 2
    if args.sim:
        if not args.model:
            print("[actuo_bus_node] --sim requires --model <plant>.xml",
                  file=sys.stderr)
            return 2
        run_sim(args)
        return 0
    if args.real:
        if not args.config:
            print("[actuo_bus_node] --real requires --config <robot>.yaml",
                  file=sys.stderr)
            return 2
        return run_real(args)

    print("[actuo_bus_node] pick a backend: --sim or --real", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
