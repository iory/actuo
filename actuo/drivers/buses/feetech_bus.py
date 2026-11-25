"""FEETECH STS/SMS/HLS bus: one SYNC_READ and one SYNC_WRITE per register group.

Orchestration only, like :class:`DynamixelBus`: the actuators hand over
``(address, bytes)`` and the bus groups them by address, so STS servos (goal at
42) and HLS servos (a 7-byte goal from 41) on one line each cost one
transaction per cycle rather than one per servo. Reads are one SYNC_READ of the
status block; FEETECH answers it with one status packet per id, in id order,
and each is matched to its servo by id rather than by position, so one silent
servo costs one missing state rather than shifting every state after it.

Start-up (:meth:`verify_servos`) reads each servo's model number, homing offset
and operating mode once. A servo that does not answer, or is not the model the
config names, stops the start-up; a servo left out of position mode (by a
process that died in PWM free drive) is put back into it, torque off.

Servos that come and go (``hot_plug``)
--------------------------------------
For a robot whose modules are plugged together at run time, servos may be
missing at start and may leave and come back. A servo is **present** once it
has answered and passed the start-up check above; it becomes **absent** after
``absent_after_misses`` SYNC_READs in a row without its reply (one lost packet
is not a missing module). :meth:`FeetechBus.present` and
:meth:`FeetechBus.absence_reason` say which and why, the same interface the
RCB-4mini bus offers, so :class:`~actuo.robot_interface.ActuoRobotInterface`
and the servo supervisor attach and detach joints from it.

An absent servo is left out of the cycle's SYNC_READ: kept in, every cycle
would wait the whole read timeout for its reply, and a robot with modules
missing would run slower the more were missing. Instead one absent servo, in
turn, gets its own READ every ``presence_period`` reads, with the short
``probe_timeout``. A servo that answers there is checked like at start-up
(model, homing offset, position mode, torque limit) before it counts as
present -- a servo that lost power is back at its EEPROM settings, and a
module of another model plugged into the same id is refused (it stays absent,
and :meth:`absence_reason` names the model) rather than driven.

An absent servo is never sent a goal or an enable, only torque off, so one
that comes back arrives limp; whoever drives the bus decides when it holds.

Checked on 2026-11-24 with two STS3215 (ids 1, 2) and two HLS3950M (ids 3, 4)
at 1 Mbit/s, each on its own port of a servo-fault-injector board
(iory/servo-fault-injector) that switches a port's power or signal: through
:class:`~actuo.robot_interface.ActuoRobotInterface`, a servo whose power or
signal was cut was detached within 0.10-0.15 s, and one switched back on was
attached again within 0.95-1.10 s (probe turn, start-up check, then the
interface's own attach cycles).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from actuo.core.contracts import (
    ActuatorCommand, ActuatorState, Bus, CapabilityError, Frame, OperatingMode)
from actuo.core.registry import register_bus
from actuo.core.diagnostics import DiagItem, DiagLevel
from actuo.drivers import feetech_protocol as fp
from actuo.drivers.buses.presence import (
    ABSENT, DEFAULT_ABSENT_AFTER_MISSES, DEFAULT_PRESENCE_PERIOD, PresenceTracker)
from actuo.drivers.actuators.feetech_sts import (
    STATE_BLOCK_ADDR, STATE_BLOCK_LEN, TORQUE_LIMIT_FULL_SCALE,
    WIRE_MODE_POSITION, FeetechStsActuator)
from feetech_cli.registers import CONTROL_TABLE

_TORQUE_ENABLE = CONTROL_TABLE["torque_enable"].address

#: Counts an HLS origin may land away from the pose asked for (set_origin).
HLS_ORIGIN_SLACK = 16


def hls_homing_offset(raw: int, target: int, max_offset: int,
                      resolution: int) -> Tuple[int, int]:
    """``(homing_offset, landed count)`` that makes ``raw`` read near ``target``
    on an HLS servo.

    An HLS takes a negative homing offset into its register and ignores it: an
    HLS3950M given -2046 read it back as -2046 and went on reading the same
    position (2026-11-24); 2047 took effect at once. ``present = (raw - offset)
    mod resolution`` with the offset in ``0..max_offset``, so a pose that needs
    a negative offset gets the largest one, and lands that many counts away.
    """
    new = raw - target
    if new >= 0:
        return new, target
    return max_offset, (raw - max_offset) % resolution


#: Seconds an absent servo's probe waits: one reply at 1 Mbit/s plus USB latency.
DEFAULT_PROBE_TIMEOUT = 0.01


@register_bus("feetech_bus")
class FeetechBus(Bus):
    """Drives a set of :class:`FeetechStsActuator` on one serial line.

    Parameters
    ----------
    transport : Transport
        A serial transport with protocol-1 framing.
    actuators : sequence of FeetechStsActuator
        In URDF joint order.
    read_timeout : float
        Seconds to wait for a whole SYNC_READ burst (or one servo's reply).
    use_sync_read : bool
        False reads each servo with its own READ: slower, but it isolates a
        servo that breaks the burst.
    reset_torque_limit : bool
        At start-up, set every servo's torque_limit (SRAM 48) back to full: a
        process that died in free drive leaves it lowered.
    hot_plug : bool
        Servos may be absent at start and may come and go (see the module
        docstring). Without it a servo that does not answer at start-up is an
        error, and one that stops answering keeps its last state.
    absent_after_misses : int
        With ``hot_plug``, SYNC_READs in a row without a servo's reply that
        make it absent.
    presence_period : int
        With ``hot_plug``, ``read_batch`` calls between two probes of an
        absent servo (one servo per probe, in turn).
    probe_timeout : float
        Seconds to wait for an absent servo's reply to its probe.
    watch_ids : sequence of int
        Ids that are not this bus's servos but whose presence is wanted -- the
        servos of another configuration of a modular robot
        (:mod:`actuo.modular`). Each gets a PING in the same rotation as the
        absent servos (every ``presence_period`` reads, one id per turn), and
        :meth:`answering_ids` reports them. They are never written.
    """

    supports_hot_plug = True
    supports_watch = True

    def __init__(self, transport, actuators, read_timeout: float = 0.05,
                 use_sync_read: bool = True,
                 reset_torque_limit: bool = True,
                 hot_plug: bool = False,
                 absent_after_misses: int = DEFAULT_ABSENT_AFTER_MISSES,
                 presence_period: int = DEFAULT_PRESENCE_PERIOD,
                 probe_timeout: float = DEFAULT_PROBE_TIMEOUT,
                 watch_ids: Sequence[int] = ()) -> None:
        super().__init__(transport, actuators)
        for actuator in self.actuators:
            if not isinstance(actuator, FeetechStsActuator):
                raise TypeError("FeetechBus drives feetech_sts actuators only, "
                                "got {}".format(type(actuator).__name__))
        self.read_timeout = float(read_timeout)
        self.use_sync_read = bool(use_sync_read)
        self.reset_torque_limit = bool(reset_torque_limit)
        self.hot_plug = bool(hot_plug)
        self.probe_timeout = float(probe_timeout)
        self.presence = PresenceTracker(
            [int(a.id) for a in self.actuators], hot_plug=self.hot_plug,
            absent_after_misses=absent_after_misses, presence_period=presence_period,
            watch_ids=[int(i) for i in watch_ids])
        self._states: Dict[int, ActuatorState] = {}
        self._missed = 0
        self._last_rx: Dict[int, float] = {}
        self._joint_name_by_id: Dict[int, str] = {}

    # --- bookkeeping ---------------------------------------------------------
    @property
    def missed(self) -> int:
        return self._missed

    @property
    def reads_per_cycle(self) -> int:
        return 1 if self.use_sync_read else len(self.actuators)

    def last_state(self, actuator_id: int) -> Optional[ActuatorState]:
        return self._states.get(actuator_id)

    @property
    def absent_after_misses(self) -> int:
        return self.presence.absent_after_misses

    @property
    def presence_period(self) -> int:
        return self.presence.presence_period

    @property
    def watch_ids(self) -> Tuple[int, ...]:
        return self.presence.watch_ids

    def watch(self, ids: Sequence[int]) -> None:
        """PING ``ids`` in the probe rotation from now on (see ``watch_ids``)."""
        self.presence.set_watch_ids(int(i) for i in ids)

    def present(self, actuator_id: int) -> bool:
        """Whether a servo is answering.

        With ``hot_plug``: it passed the start-up check and has not missed
        ``absent_after_misses`` reads in a row since. Without it a servo is
        never taken away, so this is whether it has ever reported.
        """
        return self.presence.present(int(actuator_id))

    def absence_reason(self, actuator_id: int) -> str:
        """Why a servo is not present; empty when it is."""
        return self.presence.absence_reason(int(actuator_id))

    def _mark_absent(self, actuator_id: int, why: str) -> None:
        self.presence.mark_absent(actuator_id, why)

    def set_joint_names(self, joint_names: Sequence[str]) -> None:
        if len(joint_names) != len(self.actuators):
            raise ValueError("joint_names has {} entries for {} actuators".format(
                len(joint_names), len(self.actuators)))
        self._joint_name_by_id = {int(a.id): str(n)
                                  for a, n in zip(self.actuators, joint_names)}

    def _joint_name(self, actuator_id: int) -> str:
        return self._joint_name_by_id.get(int(actuator_id),
                                          "id_{}".format(actuator_id))

    # --- acknowledged single-servo transactions ------------------------------
    def _ack(self, frames: Sequence[Frame], what: str) -> List[Frame]:
        """Send frames that each get a status reply; raise on a silent one."""
        replies = self.transport.transact(list(frames), self.read_timeout)
        for frame, reply in zip(frames, replies):
            if reply is None:
                raise RuntimeError("id {}: no reply to {}".format(
                    frame.address, what))
        return replies

    def _set_wire_mode(self, actuator: FeetechStsActuator,
                       mode: OperatingMode) -> None:
        """Torque off and switch ``operating_mode``, then read it back."""
        self._ack(actuator.encode_mode(mode), "the operating mode change")
        wanted = actuator.wire_mode_of(mode)
        reply = self._ack([actuator.encode_read("operating_mode")],
                          "the operating mode read-back")[0]
        got = actuator.decode_register("operating_mode", reply)
        actuator.facts.wire_mode = got
        if got != wanted:
            raise RuntimeError("id {}: operating_mode reads {} after writing {}"
                               .format(actuator.id, got, wanted))

    # --- start-up ------------------------------------------------------------
    def verify_servos(self, allow_missing: bool = False) -> List[DiagItem]:
        """Read each servo's start-up facts; put it in position mode.

        Parameters
        ----------
        allow_missing : bool
            Report a servo that does not answer as a WARN item instead of
            raising. Always so with ``hot_plug``, where such a servo is absent
            until it answers.

        Raises
        ------
        RuntimeError
            A configured servo does not answer (and missing ones are not allowed).
        CapabilityError
            A servo is not the model its config names (without ``hot_plug``).
        """
        allow_missing = allow_missing or self.hot_plug
        items: List[DiagItem] = []
        for actuator in self.actuators:
            reply = self.transport.transact(actuator.encode_facts_query(),
                                            self.read_timeout)[0]
            if reply is None:
                if not allow_missing:
                    raise RuntimeError(
                        "{} (id {}) did not answer on the bus".format(
                            self._joint_name(actuator.id), actuator.id))
                self._mark_absent(actuator.id, "did not answer at start-up")
                items.append(DiagItem(
                    name="servo/{}".format(self._joint_name(actuator.id)),
                    level=DiagLevel.WARN, message="not answering",
                    values={"id": str(actuator.id)}))
                continue
            try:
                items.append(self._check_servo(actuator, reply))
            except CapabilityError as error:
                if not self.hot_plug:
                    raise
                self._mark_absent(actuator.id, str(error))
                items.append(DiagItem(
                    name="servo/{}".format(self._joint_name(actuator.id)),
                    level=DiagLevel.ERROR, message=str(error),
                    values={"id": str(actuator.id)}))
                continue
            self.presence.mark_present(actuator.id)
        return items

    def _check_servo(self, actuator: FeetechStsActuator,
                     reply: Frame) -> DiagItem:
        """Start-up check of one servo from its facts reply.

        Raises
        ------
        CapabilityError
            The servo is not the model its config names.
        """
        facts = actuator.decode_facts(reply)
        notes = []
        if facts.wire_mode != WIRE_MODE_POSITION:
            notes.append("operating_mode {} -> position (torque off)".format(
                facts.wire_mode))
            self._set_wire_mode(actuator, OperatingMode.POSITION)
        if self.reset_torque_limit:
            self._ack([actuator.encode_write("torque_limit",
                                             TORQUE_LIMIT_FULL_SCALE)],
                      "the torque_limit reset")
        values = {"id": str(actuator.id),
                  "model": actuator.model.name,
                  "homing_offset": str(facts.homing_offset)}
        if actuator.is_hls:
            values["goal_current"] = str(facts.protection_current)
        if actuator.current_scale is None:
            notes.append("current scale unknown for this model")
        return DiagItem(
            name="servo/{}".format(self._joint_name(actuator.id)),
            level=DiagLevel.WARN if notes else DiagLevel.OK,
            message="; ".join(notes) or "verified",
            values=values)

    # --- mode / torque -------------------------------------------------------
    def set_mode(self, mode: OperatingMode) -> None:
        """Switch every servo's operating mode (an absent hot-plug servo is
        skipped: it comes back in position mode, see :meth:`_probe_absent`)."""
        for actuator in self.actuators:
            actuator.require_mode(mode)
        for actuator in self.actuators:
            if self.hot_plug and not self.present(actuator.id):
                continue
            self._set_wire_mode(actuator, mode)

    def _write_torque(self, actuators, enabled: bool) -> None:
        if not actuators:
            return
        self.transport.write([Frame(payload=fp.build_sync_write(
            _TORQUE_ENABLE, 1, [(a.id, bytes([1 if enabled else 0]))
                                for a in actuators]), address=None)])

    def enable_all(self) -> None:
        self._write_torque(self.actuators, True)

    def release_all(self) -> None:
        self._write_torque(self.actuators, False)

    # --- write ---------------------------------------------------------------
    def write_batch(self, commands: Sequence[Optional[ActuatorCommand]]) -> None:
        """One SYNC_WRITE per goal address; ``None`` releases that servo.

        Goals go out before torque_enable, so a servo is never enabled onto a
        stale goal. The enable is re-sent every batch, so a servo whose torque
        went off behind the driver's back comes back rather than staying
        silently limp (the same rule as :class:`DynamixelBus`). A command in
        another mode than the servo is in switches it first (torque off,
        acknowledged writes): that is how a joint enters and leaves PWM.
        """
        if len(commands) != len(self.actuators):
            raise ValueError("write_batch expects {} commands, got {}".format(
                len(self.actuators), len(commands)))
        released = []
        held = []
        groups: Dict[Tuple[int, int], List[Tuple[int, bytes]]] = {}
        for actuator, command in zip(self.actuators, commands):
            if command is None or (self.hot_plug
                                   and not self.present(actuator.id)):
                # An absent servo gets torque off only, so it comes back limp.
                released.append(actuator)
                continue
            if actuator.facts.wire_mode != actuator.wire_mode_of(command.mode):
                self._set_wire_mode(actuator, command.mode)
            address, data = actuator.goal_payload(command)
            groups.setdefault((address, len(data)), []).append(
                (actuator.id, data))
            held.append(actuator)
        self._write_torque(released, False)
        frames = [Frame(payload=fp.build_sync_write(address, size, items),
                        address=None)
                  for (address, size), items in sorted(groups.items())]
        if frames:
            self.transport.write(frames)
        self._write_torque(held, True)

    # --- read ----------------------------------------------------------------
    def read_batch(self, timeout: Optional[float] = None) -> List[ActuatorState]:
        """One state per servo. A servo that does not answer keeps its last
        state (``None`` until it has answered once) and counts as a miss --
        never a faked zero."""
        wait = self.read_timeout if timeout is None else float(timeout)
        asked = set(self.presence.polled(a.id for a in self.actuators))
        polled = [a for a in self.actuators if a.id in asked]
        fresh = (self._sync_read(polled, wait) if self.use_sync_read
                 else self._individual_reads(polled, wait))
        now = time.monotonic()
        states: List[ActuatorState] = []
        polled_ids = {a.id for a in polled}
        for actuator in self.actuators:
            state = fresh.get(actuator.id)
            if state is None and actuator.id in polled_ids:
                self._missed += 1
                self.presence.missed(actuator.id)
            elif state is not None:
                self._states[actuator.id] = state
                self._last_rx[actuator.id] = now
                self.presence.answered(actuator.id)
            states.append(self._states.get(actuator.id))
        probe = self.presence.next_probe()
        if probe is not None:
            kind, servo_id = probe
            if kind == ABSENT:
                self._probe_absent(self._by_id(servo_id))
            else:
                self._probe_watched(servo_id)
        return states

    def _by_id(self, servo_id: int) -> FeetechStsActuator:
        return next(a for a in self.actuators if a.id == servo_id)

    def _probe_watched(self, servo_id: int) -> None:
        """PING a watched id and record whether it answered."""
        reply = self.transport.transact(
            [Frame(payload=fp.build_ping(servo_id), address=servo_id)],
            self.probe_timeout)[0]
        self.presence.watched_answered(servo_id, reply is not None)

    def watched(self, servo_id: int) -> bool:
        """Whether a watched id answered its last PING (False before the first)."""
        return self.presence.watched(int(servo_id))

    def answering_ids(self) -> List[int]:
        """Every id known to answer now: present servos and watched ids."""
        return self.presence.answering_ids()

    def _probe_absent(self, actuator: FeetechStsActuator) -> None:
        """Ask one absent servo whether it is back; check it if so.

        A servo that answers is checked like at start-up before it counts as
        present; one of another model stays absent with the reason recorded.
        """
        reply = self.transport.transact(actuator.encode_facts_query(),
                                        self.probe_timeout)[0]
        if reply is None:
            return
        try:
            self._check_servo(actuator, reply)
        except CapabilityError as error:
            self._mark_absent(actuator.id, str(error))
            return
        except RuntimeError as error:
            # It answered the probe but not the settings writes after it:
            # not back yet (a connector still being pushed in).
            self._mark_absent(actuator.id, "answered, then {}".format(error))
            return
        self.presence.mark_present(actuator.id)

    def _individual_reads(self, actuators, timeout: float) -> Dict[int, ActuatorState]:
        queries: List[Frame] = []
        for actuator in actuators:
            queries.extend(actuator.encode_query())
        if not queries:
            return {}
        replies = self.transport.transact(queries, timeout)
        fresh: Dict[int, ActuatorState] = {}
        for actuator, reply in zip(actuators, replies):
            if reply is not None and actuator.matches(reply):
                fresh[actuator.id] = actuator.decode_state(reply)
        return fresh

    def _sync_read(self, actuators, timeout: float) -> Dict[int, ActuatorState]:
        by_id = {a.id: a for a in actuators}
        if not by_id:
            return {}
        reset = getattr(self.transport, "reset_input", None)
        if reset is not None:
            reset()
        self.transport.write([Frame(payload=fp.build_sync_read(
            STATE_BLOCK_ADDR, STATE_BLOCK_LEN, list(by_id)), address=None)])
        fresh: Dict[int, ActuatorState] = {}
        deadline = time.monotonic() + timeout
        remaining = set(by_id)
        while remaining:
            left = deadline - time.monotonic()
            if left <= 0:
                break
            reply = self.transport.recv(left)
            if reply is None:
                break
            parsed = fp.parse_status(reply.payload)
            # The adapter's echo of the SYNC_READ (id 0xFE) and anything short
            # of a full block is not this cycle's answer: skip it, keep waiting.
            if (parsed is None or parsed[0] not in remaining
                    or len(parsed[2]) < STATE_BLOCK_LEN):
                continue
            fresh[parsed[0]] = by_id[parsed[0]].decode_block(parsed[2],
                                                             error=parsed[1])
            remaining.discard(parsed[0])
        return fresh

    # --- calibration ---------------------------------------------------------
    def set_origin(self, index: int, position: float, torque: bool) -> Dict[str, float]:
        """Make the pose servo ``index`` is in right now read as ``position``.

        Writes the servo's EEPROM ``homing_offset`` (unlocking and locking the
        EEPROM around it, as feetech-cli does), so the calibration lives in
        the servo and survives a power cycle; the config's ``offset_deg``
        stays what it was. The relationship, measured on an STS3215, is
        ``present = (raw - homing_offset) mod 4096`` in position mode.

        The servo does not move: torque goes off for the write, the goal is
        re-pointed at the new reading before it may come back on, and only
        then is torque restored to ``torque``. A servo left in PWM mode is
        put back in position mode first, since outside it the reading carries
        no offset to work from.

        A write is confirmed by its effect: the position is read back and must
        be within 2 counts of ``position``, else this raises rather than
        report a calibration that did not happen.

        An HLS ignores a negative homing offset (see :func:`hls_homing_offset`):
        a pose that needs one is declared up to ``HLS_ORIGIN_SLACK`` counts away
        from ``position``, and ``reported`` says where.

        Parameters
        ----------
        index : int
            The servo, in bus order.
        position : float
            The angle the present pose is declared to be [rad].
        torque : bool
            Torque on afterwards (holding the pose), or off.

        Returns
        -------
        dict
            ``old_offset``, ``new_offset`` and ``reported`` (counts).

        Raises
        ------
        ValueError
            The angle is outside one turn, or the offset it needs is outside
            what the register holds.
        RuntimeError
            A servo did not answer, or the read-back does not match.
        """
        from feetech_cli.registers import (EEPROM_LOCKED, EEPROM_UNLOCKED,
                                           MAX_HOMING_OFFSET, RESOLUTION)

        actuator = self.actuators[index]
        target = int(round(actuator.calibration.rad_to_counts(position)))
        if not 0 <= target < RESOLUTION:
            raise ValueError("id {}: {:+.4f} rad is count {}, outside one turn"
                             .format(actuator.id, position, target))
        if actuator.facts.wire_mode != WIRE_MODE_POSITION:
            self._set_wire_mode(actuator, OperatingMode.POSITION)

        def read(register: str) -> int:
            reply = self._ack([actuator.encode_read(register)],
                              "the {} read".format(register))[0]
            return actuator.decode_register(register, reply)

        old = read("homing_offset")
        present = read("present_position")
        raw = (present + old) % RESOLUTION
        new = raw - target
        if actuator.is_hls and new < 0:
            new, landed = hls_homing_offset(raw, target, MAX_HOMING_OFFSET, RESOLUTION)
            if abs(landed - target) > HLS_ORIGIN_SLACK:
                raise ValueError(
                    "id {}: an HLS takes no negative homing offset, and the nearest "
                    "it can do reads count {} instead of {}; turn the horn so the "
                    "pose is on the other side of the middle".format(
                        actuator.id, landed, target))
            target = landed
        if abs(new) > MAX_HOMING_OFFSET:
            raise ValueError(
                "id {}: declaring this pose {:+.4f} rad needs a homing offset of "
                "{}, and the register holds +/-{}; the joint is too far from "
                "that angle".format(actuator.id, position, new, MAX_HOMING_OFFSET))
        self._ack([actuator.encode_write("torque_enable", 0)], "the torque off")
        self._ack([actuator.encode_write("lock", EEPROM_UNLOCKED)], "the EEPROM unlock")
        try:
            self._ack([actuator.encode_write("homing_offset", new)],
                      "the homing_offset write")
        finally:
            self._ack([actuator.encode_write("lock", EEPROM_LOCKED)],
                      "the EEPROM lock")
        time.sleep(0.05)
        reported = read("present_position")
        actuator.facts.homing_offset = read("homing_offset")
        if actuator.facts.homing_offset != new or abs(reported - target) > 2:
            raise RuntimeError(
                "id {}: wrote homing_offset {} (was {}) to read {}, but the servo "
                "now has offset {} and reads {}".format(
                    actuator.id, new, old, target, actuator.facts.homing_offset,
                    reported))
        # The goal in the new frame before torque may return: the goal write
        # alone energises an STS (feetech-cli set_zero), hence the restore.
        self._ack([Frame(payload=fp.build_write(
            actuator.id, *actuator.position_goal(reported)), address=actuator.id)],
            "the goal re-point")
        self._ack([actuator.encode_write("torque_enable", 1 if torque else 0)],
                  "the torque restore")
        return {"old_offset": float(old), "new_offset": float(new),
                "reported": float(reported)}

    # --- diagnostics ---------------------------------------------------------
    def diagnose(self) -> List[DiagItem]:
        items = [DiagItem(
            name="bus/reads",
            level=DiagLevel.OK if self._missed == 0 else DiagLevel.WARN,
            message=("No read misses recorded." if self._missed == 0
                     else "Read misses were recorded; check bus load and wiring."),
            values={"reads_per_cycle": str(self.reads_per_cycle),
                    "missed_total": str(self._missed)})]
        now = time.monotonic()
        for actuator in self.actuators:
            state = self._states.get(actuator.id)
            streak = self.presence.consecutive_misses(actuator.id)
            age = (None if actuator.id not in self._last_rx
                   else now - self._last_rx[actuator.id])
            values = {"actuator_id": str(actuator.id),
                      "consecutive_misses": str(streak),
                      "last_reply_age_s": ("never" if age is None
                                           else "{:.3f}".format(age))}
            if self.hot_plug and not self.present(actuator.id):
                level = DiagLevel.WARN
                message = "Not present: {}.".format(
                    self.absence_reason(actuator.id))
            elif state is None:
                level, message = DiagLevel.STALE, "No status received yet."
            elif streak > 0:
                level = DiagLevel.WARN
                message = "Missed the last {} read(s).".format(streak)
            else:
                level, message = DiagLevel.OK, "Communication healthy."
            if state is not None and state.error:
                level = DiagLevel.ERROR
                values["hardware_error_code"] = str(int(state.error))
                message = "Error bits set: {}.".format(
                    actuator.fault_name(state.error))
            if state is not None:
                values["status"] = str(int(state.aux.get("status", 0)))
            items.append(DiagItem(
                name="servo/{}".format(self._joint_name(actuator.id)),
                level=level, message=message, values=values))
        return items
