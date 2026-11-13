"""DYNAMIXEL bus: one SYNC_WRITE per register group, SYNC_READ where it exists.

Orchestration only. The bus decides how many transactions a control cycle costs
and in what order; it holds no register addresses and no unit scaling -- it asks
each actuator for ``(register, bytes)`` and groups by the register name. That is
what lets one bus drive a mix of models: two servos with different control-table
layouts simply land in different groups, and each group is still one transaction
rather than a loop of singles.

The 1.0 / 2.0 asymmetry is real and is surfaced, not hidden
------------------------------------------------------------
Protocol 2.0 has SYNC_READ: one instruction, N status packets, one round trip's
worth of turnaround. Protocol 1.0 has no batch read at all -- N servos cost N
request/response round trips, and on a 1 Mbit/s line that is the dominant term
in the cycle time. This bus does the N reads (there is no alternative) but
reports the cost through :attr:`reads_per_cycle` so a slow loop is explained by
a number rather than by guesswork.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from actuo.core.contracts import (
    ActuatorCommand, ActuatorState, Bus, Frame, OperatingMode)
from actuo.core.diagnostics import DiagItem, DiagLevel

from actuo.drivers import dynamixel_protocol as dxl


class DynamixelBus(Bus):
    """Drives a set of :class:`DynamixelActuator` on one serial line.

    Parameters
    ----------
    transport : Transport
        A serial transport framing this bus's protocol version.
    actuators : sequence of DynamixelActuator
        In URDF joint order -- that order is the contract with ros2_control.
    read_timeout : float
        Seconds to wait for a reply (per servo on 1.0, for the whole reply
        burst on 2.0).
    use_sync_read : bool
        Protocol 2.0 only. Set false to fall back to per-servo READs, which is
        slower but isolates which servo is misbehaving when a sync read comes
        back short.
    silence_error_misses : int
        Consecutive read misses after which an already-seen servo is reported
        as ERROR rather than WARN. Default 10.
    silence_error_age_s : float
        Seconds since the last reply after which an already-seen servo is
        reported as ERROR rather than WARN. Default 1.0 s.

    Notes
    -----
    The silence escalation defaults are operational guardrails, not
    datasheet-derived limits: 10 misses and 1.0 s were chosen to separate a
    dropped packet from sustained communication loss.
    """

    def __init__(self, transport, actuators, read_timeout: float = 0.05,
                 use_sync_read: bool = True,
                 silence_error_misses: int = 10,
                 silence_error_age_s: float = 1.0) -> None:
        super().__init__(transport, actuators)
        self.read_timeout = float(read_timeout)
        self.use_sync_read = bool(use_sync_read)
        self.silence_error_misses = int(max(1, silence_error_misses))
        self.silence_error_age_s = float(max(0.0, silence_error_age_s))
        self._states: Dict[int, ActuatorState] = {}
        self._missed = 0
        self._mode = OperatingMode.POSITION
        self._comm_failures: Dict[int, int] = {}
        self._consecutive_misses: Dict[int, int] = {}
        self._last_rx: Dict[int, float] = {}
        self._joint_name_by_id: Dict[int, str] = {}
        #: Servo ids believed to have torque_enable = 1. Used only to decide
        #: when profile velocity needs (re)writing -- the enable itself is
        #: re-asserted every batch, so a stale set here cannot leave a joint
        #: limp, only cost one redundant profile write.
        self._torque_on: set = set()

        protocols = {a.protocol for a in self.actuators}
        if len(protocols) > 1:
            raise ValueError(
                "one serial line cannot be framed as two protocol versions at "
                "once; this bus has {}. Split the servos across two buses (and "
                "two adapters), or reflash them to a common protocol.".format(
                    sorted(protocols)))
        self.protocol = protocols.pop() if protocols else dxl.PROTOCOL_2

    # --- instrumentation -----------------------------------------------------
    @property
    def missed(self) -> int:
        """Reads that produced no fresh state. Never silently absorbed."""
        return self._missed

    @property
    def reads_per_cycle(self) -> int:
        """Round trips one ``read_batch`` costs. 1 on protocol 2.0 (SYNC_READ),
        N on protocol 1.0 -- the number to look at when a loop is slow."""
        if self.protocol == dxl.PROTOCOL_2 and self.use_sync_read:
            return 1
        return len(self.actuators)

    def last_state(self, actuator_id: int) -> Optional[ActuatorState]:
        return self._states.get(actuator_id)

    def set_joint_names(self, joint_names: Sequence[str]) -> None:
        """Name each configured id for operator-facing diagnostics."""
        if len(joint_names) != len(self.actuators):
            raise ValueError(
                "joint_names has {} entries for {} actuators".format(
                    len(joint_names), len(self.actuators)
                )
            )
        self._joint_name_by_id = {
            int(actuator.id): str(name)
            for actuator, name in zip(self.actuators, joint_names)
        }

    def _joint_name(self, actuator_id: int) -> str:
        return self._joint_name_by_id.get(int(actuator_id), "id_{}".format(actuator_id))

    # --- mode / torque -------------------------------------------------------
    def set_mode(self, mode: OperatingMode) -> None:
        """Capability-check every servo, then switch them all.

        Checked *before* anything is written, so a bus with one servo that
        cannot do the requested mode leaves every servo in its previous mode
        rather than half-switched.
        """
        for actuator in self.actuators:
            actuator.require_mode(mode)
        frames: List[Frame] = []
        for actuator in self.actuators:
            frames.extend(actuator.encode_mode(mode))
        if frames:
            self.transport.write(frames)
        self._mode = mode

    def enable_all(self) -> None:
        """Torque on for every servo, in one SYNC_WRITE per register group.

        Profile velocity (if the config set one) is written first: a servo that
        is enabled before its speed limit is applied will honour whatever limit
        it happened to hold, which on a fresh servo is "as fast as possible".
        """
        profile: List[Frame] = []
        for actuator in self.actuators:
            profile.extend(actuator.encode_profile_velocity())
        if profile:
            self.transport.write(profile)
        self._write_torque(True)

    def release_all(self) -> None:
        """Torque off for every servo, so the arm is back-drivable."""
        self._write_torque(False)

    def _write_torque(self, enabled: bool) -> None:
        self._write_torque_for(self.actuators, enabled)
        if enabled:
            self._torque_on.update(a.id for a in self.actuators)
        else:
            self._torque_on.clear()

    def _write_torque_for(self, actuators, enabled: bool) -> None:
        groups: Dict[Tuple[int, int], List[Tuple[int, bytes]]] = {}
        for actuator in actuators:
            entry = actuator.model.table["torque_enable"]
            groups.setdefault((entry.address, entry.size), []).append(
                (actuator.id, bytes([1 if enabled else 0])))
        self._write_groups(groups)

    def _write_groups(self, groups: Dict[Tuple[int, int], List[Tuple[int, bytes]]]
                      ) -> None:
        frames = [
            Frame(payload=dxl.build_sync_write(self.protocol, address, size, items),
                  address=None)
            for (address, size), items in sorted(groups.items())
        ]
        if frames:
            self.transport.write(frames)

    # --- write ---------------------------------------------------------------
    def write_batch(self, commands: Sequence[Optional[ActuatorCommand]]) -> None:
        """One SYNC_WRITE per register group; ``None`` releases that servo.

        A released servo cannot ride in the goal sync-write -- that would
        command it a pose -- so it is torqued off in the same burst, and the
        released and held servos never straddle two cycles.

        **A commanded servo is also torque-ENABLED here, and that is not
        optional.** A DYNAMIXEL with ``torque_enable = 0`` accepts a goal
        position and ignores it: the write succeeds, the servo answers, its
        Present Position keeps reporting the truth, and the arm stays limp.
        Everything upstream looks perfectly healthy. This bus used to write
        only the goal, so the servicer could report ``torque=ON``, the
        trajectory controller could report ``Motion completed successfully``,
        and the robot would not move at all -- observed on a DYNAMIXEL arm's
        first real playback, which is the only way a fault like this ever
        shows up.

        It is the mirror of the rule that "a release is a command, not the
        absence of one". So is an enable. The supervisor above
        is actuator-agnostic and was written against CubeMars servo mode, where
        there IS no enable register (off is a synthesised ``SET_CURRENT 0``, on
        is simply resuming position frames). Making "enabled" mean "we are
        sending goals" is right there and wrong here.

        Order is load-bearing: goal first, THEN enable. A servo enabled before
        its goal is written drives to whatever goal it still held -- which is
        exactly the start-up slam the rest of this stack is built to prevent.
        Profile velocity goes before both, for the same reason it does in
        :meth:`enable_all`.

        The enable is re-asserted on every batch rather than only on the
        transition, so a servo that lost torque on its own -- an overload
        shutdown, a power blip, anything that clears the register behind our
        back -- comes back rather than staying silently limp for the rest of
        the run. It costs one extra SYNC_WRITE on cycles that write at all.
        """
        if len(commands) != len(self.actuators):
            raise ValueError("write_batch expects {} commands, got {}".format(
                len(self.actuators), len(commands)))

        groups: Dict[Tuple[int, int], List[Tuple[int, bytes]]] = {}
        releases: List[Frame] = []
        held: List = []
        for actuator, command in zip(self.actuators, commands):
            if command is None:
                releases.append(actuator.encode_release())
                self._torque_on.discard(actuator.id)
                continue
            register, data = actuator.goal_payload(command)
            entry = actuator.model.table[register]
            groups.setdefault((entry.address, entry.size), []).append(
                (actuator.id, data))
            held.append(actuator)

        if releases:
            self.transport.write(releases)

        fresh = [a for a in held if a.id not in self._torque_on]
        if fresh:
            profile: List[Frame] = []
            for actuator in fresh:
                profile.extend(actuator.encode_profile_velocity())
            if profile:
                self.transport.write(profile)

        self._write_groups(groups)

        if held:
            self._write_torque_for(held, True)
            self._torque_on.update(a.id for a in held)

    # --- read ----------------------------------------------------------------
    def read_batch(self, timeout: Optional[float] = None) -> List[ActuatorState]:
        """One state per actuator, batched where the protocol allows.

        A servo that does not answer keeps its last known state (``None`` until
        it has ever answered) and increments :attr:`missed`. It is never given a
        faked zero -- a joint reported at 0 rad that is actually at 1.5 rad is
        the single most dangerous value this driver could emit.
        """
        wait = self.read_timeout if timeout is None else float(timeout)
        if self.protocol == dxl.PROTOCOL_2 and self.use_sync_read:
            fresh = self._sync_read(wait)
        else:
            fresh = self._individual_reads(wait)

        self._diagnose_hardware_faults(fresh, wait)

        states: List[ActuatorState] = []
        for actuator in self.actuators:
            state = fresh.get(actuator.id)
            if state is None:
                self._missed += 1
                self._comm_failures[actuator.id] = (
                    int(self._comm_failures.get(actuator.id, 0)) + 1
                )
                self._consecutive_misses[actuator.id] = (
                    int(self._consecutive_misses.get(actuator.id, 0)) + 1
                )
            else:
                self._states[actuator.id] = state
                self._last_rx[actuator.id] = time.monotonic()
                self._consecutive_misses[actuator.id] = 0
            states.append(self._states.get(actuator.id))
        return states

    def diagnose(self) -> List[DiagItem]:
        """Backend-specific DYNAMIXEL health from cached read-loop observations."""
        items: List[DiagItem] = []
        items.append(DiagItem(
            name="bus/reads",
            level=DiagLevel.OK if self._missed == 0 else DiagLevel.WARN,
            message=(
                "No read misses recorded."
                if self._missed == 0
                else "Read misses were recorded; check bus load and wiring."
            ),
            values={
                "protocol": str(int(self.protocol)),
                "reads_per_cycle": str(int(self.reads_per_cycle)),
                "missed_total": str(int(self._missed)),
            },
        ))

        now = time.monotonic()
        for actuator in self.actuators:
            actuator_id = int(actuator.id)
            state = self._states.get(actuator_id)
            failures = int(self._comm_failures.get(actuator_id, 0))
            streak = int(self._consecutive_misses.get(actuator_id, 0))
            age = (None if actuator_id not in self._last_rx
                   else now - self._last_rx[actuator_id])
            values = {
                "actuator_id": str(actuator_id),
                "comm_failures_total": str(failures),
                "consecutive_misses": str(streak),
                "last_reply_age_s": (
                    "never" if age is None else "{:.3f}".format(float(age))
                ),
                "silence_error_misses": str(int(self.silence_error_misses)),
                "silence_error_age_s": "{:.3f}".format(float(self.silence_error_age_s)),
            }

            level = DiagLevel.OK
            message_parts: List[str] = []
            if state is None:
                level = DiagLevel.STALE
                message_parts.append(
                    "No status packet has been received from this servo yet."
                )
            elif (streak >= self.silence_error_misses or
                  (age is not None and age >= self.silence_error_age_s)):
                level = DiagLevel.ERROR
                message_parts.append(
                    "Communication loss is sustained silence ({} consecutive "
                    "misses, {:.3f}s since the last reply).".format(
                        streak, 0.0 if age is None else float(age))
                )
            elif streak > 0:
                level = DiagLevel.WARN
                message_parts.append(
                    "Most recent read cycle missed this servo ({} consecutive misses)."
                    .format(streak)
                )
            else:
                message_parts.append("Communication healthy this sample.")

            if state is not None and state.error is not None:
                code = int(state.error)
                values["hardware_error_code"] = str(code)
                names = actuator.fault_name(code)
                values["hardware_error_name"] = names
                if "packet_error" in state.aux:
                    values["packet_error"] = str(int(state.aux["packet_error"]))
                if code != 0:
                    level = DiagLevel.ERROR
                    message_parts.append(
                        "Hardware error bits set: {} (code {}).".format(
                            names, code
                        )
                    )

            if level == DiagLevel.OK and not message_parts:
                message_parts.append("No communication or hardware faults.")

            items.append(DiagItem(
                name="servo/{}".format(self._joint_name(actuator_id)),
                level=level,
                message=" ".join(message_parts),
                values=values,
            ))

        return items

    def _diagnose_hardware_faults(self, fresh: Dict[int, ActuatorState],
                                  timeout: float) -> None:
        """Name a fault the moment its cheap signal (the Alert bit) appears.

        Every status reply already carries a packet-level error byte, and bit
        7 of it (``dxl.ERROR_ALERT_BIT``) means "Hardware Error Status is
        nonzero" -- but not WHICH bit, so the driver used to report a real
        Overload Error as the meaningless "error code 128". That register (one
        byte, address 70 on the X-series) is not worth reading every cycle --
        it is 0x00 essentially always -- so it is read explicitly, and only for
        the id(s) that just showed the Alert bit.

        On success, ``state.error`` is OVERWRITTEN with the decoded byte (e.g.
        0x20 for Overload alone): that is the more specific, more actionable
        fact, and reporting the vaguer Alert bit alongside it would just leave
        a caller to guess which one to trust. The original packet byte survives
        in ``aux['packet_error']`` for anyone who wants the wire-level fact.
        Failure to diagnose -- a transport hiccup, or a model with no
        hardware_error_status row -- is exactly that, a failure to diagnose,
        not a failure to see the fault: ``state.error`` is left as the Alert
        bit, still nonzero, still enough to refuse an ``enable``.
        """
        for actuator in self.actuators:
            state = fresh.get(actuator.id)
            if (state is None or not state.error
                    or not state.error & dxl.ERROR_ALERT_BIT
                    or not actuator.has_hardware_error_status):
                continue
            try:
                reply = self.transport.transact(
                    actuator.encode_hardware_error_query(), timeout)[0]
                if reply is None:
                    continue
                byte = actuator.decode_hardware_error_status(reply)
            except Exception:                                  # noqa: BLE001
                # The diagnosis is a nicety on top of a fault already visible
                # via the Alert bit; a problem asking the follow-up question
                # must not hide the fault itself or take the read cycle down.
                continue
            state.aux["packet_error"] = float(state.error)
            state.error = byte

    def _individual_reads(self, timeout: float) -> Dict[int, ActuatorState]:
        queries: List[Frame] = []
        for actuator in self.actuators:
            queries.extend(actuator.encode_query())
        replies = self.transport.transact(queries, timeout)

        fresh: Dict[int, ActuatorState] = {}
        for actuator, reply in zip(self.actuators, replies):
            if reply is not None and actuator.matches(reply):
                fresh[actuator.id] = actuator.decode_state(reply)
        return fresh

    def _sync_read(self, timeout: float) -> Dict[int, ActuatorState]:
        """One SYNC_READ per block group, then collect the status burst.

        The replies arrive as N separate status packets in id order, but this
        does not assume the order: each is matched to its actuator by id, so one
        silent servo costs one missing state rather than mis-assigning every
        state after it -- which would move the wrong joints.
        """
        groups: Dict[Tuple[int, int], List] = {}
        for actuator in self.actuators:
            table = actuator.model.table
            groups.setdefault((table.block_address, table.block_length),
                              []).append(actuator)

        fresh: Dict[int, ActuatorState] = {}
        for (address, length), members in sorted(groups.items()):
            by_id = {a.id: a for a in members}
            reset = getattr(self.transport, "reset_input", None)
            if reset is not None:
                reset()
            self.transport.write([Frame(
                payload=dxl.build_sync_read(self.protocol, address, length,
                                            [a.id for a in members]),
                address=None)])

            deadline = time.monotonic() + timeout
            remaining = set(by_id)
            while remaining:
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                reply = self.transport.recv(left)
                if reply is None:
                    break
                parsed = dxl.parse_status(self.protocol, reply.payload)
                if parsed is None or parsed[0] not in remaining:
                    continue
                actuator = by_id[parsed[0]]
                # A status packet addressed to this id but carrying fewer bytes
                # than the block is not this cycle's read -- a leftover write-ack
                # or a truncated reply on a busy FTDI bus. Skip it and keep
                # waiting rather than decoding a short block into a crash; the id
                # simply stays in `remaining` and misses this cycle.
                if len(parsed[2]) < length:
                    continue
                fresh[actuator.id] = actuator.decode_block(parsed[2],
                                                           error=parsed[1])
                remaining.discard(actuator.id)
        return fresh
