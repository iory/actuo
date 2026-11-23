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
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from actuo.core.contracts import (
    ActuatorCommand, ActuatorState, Bus, Frame, OperatingMode)
from actuo.core.diagnostics import DiagItem, DiagLevel
from actuo.drivers import feetech_protocol as fp
from actuo.drivers.actuators.feetech_sts import (
    STATE_BLOCK_ADDR, STATE_BLOCK_LEN, TORQUE_LIMIT_FULL_SCALE,
    WIRE_MODE_POSITION, FeetechStsActuator)
from feetech_cli.registers import CONTROL_TABLE

_TORQUE_ENABLE = CONTROL_TABLE["torque_enable"].address


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
    """

    def __init__(self, transport, actuators, read_timeout: float = 0.05,
                 use_sync_read: bool = True,
                 reset_torque_limit: bool = True) -> None:
        super().__init__(transport, actuators)
        for actuator in self.actuators:
            if not isinstance(actuator, FeetechStsActuator):
                raise TypeError("FeetechBus drives feetech_sts actuators only, "
                                "got {}".format(type(actuator).__name__))
        self.read_timeout = float(read_timeout)
        self.use_sync_read = bool(use_sync_read)
        self.reset_torque_limit = bool(reset_torque_limit)
        self._states: Dict[int, ActuatorState] = {}
        self._missed = 0
        self._consecutive_misses: Dict[int, int] = {}
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
    def verify_servos(self) -> List[DiagItem]:
        """Read each servo's start-up facts; put it in position mode.

        Raises
        ------
        RuntimeError
            A configured servo does not answer.
        CapabilityError
            A servo is not the model its config names.
        """
        items: List[DiagItem] = []
        for actuator in self.actuators:
            reply = self.transport.transact(actuator.encode_facts_query(),
                                            self.read_timeout)[0]
            if reply is None:
                raise RuntimeError(
                    "{} (id {}) did not answer on the bus".format(
                        self._joint_name(actuator.id), actuator.id))
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
            items.append(DiagItem(
                name="servo/{}".format(self._joint_name(actuator.id)),
                level=DiagLevel.WARN if notes else DiagLevel.OK,
                message="; ".join(notes) or "verified",
                values=values))
        return items

    # --- mode / torque -------------------------------------------------------
    def set_mode(self, mode: OperatingMode) -> None:
        for actuator in self.actuators:
            actuator.require_mode(mode)
        for actuator in self.actuators:
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
            if command is None:
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
        fresh = (self._sync_read(wait) if self.use_sync_read
                 else self._individual_reads(wait))
        now = time.monotonic()
        states: List[ActuatorState] = []
        for actuator in self.actuators:
            state = fresh.get(actuator.id)
            if state is None:
                self._missed += 1
                self._consecutive_misses[actuator.id] = (
                    self._consecutive_misses.get(actuator.id, 0) + 1)
            else:
                self._states[actuator.id] = state
                self._last_rx[actuator.id] = now
                self._consecutive_misses[actuator.id] = 0
            states.append(self._states.get(actuator.id))
        return states

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
        by_id = {a.id: a for a in self.actuators}
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
            streak = self._consecutive_misses.get(actuator.id, 0)
            age = (None if actuator.id not in self._last_rx
                   else now - self._last_rx[actuator.id])
            values = {"actuator_id": str(actuator.id),
                      "consecutive_misses": str(streak),
                      "last_reply_age_s": ("never" if age is None
                                           else "{:.3f}".format(age))}
            if state is None:
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
