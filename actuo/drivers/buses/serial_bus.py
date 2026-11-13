"""Feetech serial bus: one SYNC_WRITE per cycle, request/response reads.

Groups the SCS0009 servos sharing one serial transport and drives them as a
batch. ``write_batch`` builds a single SYNC_WRITE (all goal positions in one
transaction) rather than a loop of individual writes: batch is the primary
operation, per-servo writes the special case.
Reads are request/response -- there is no unsolicited status stream to poll, so
``read_batch`` sends one READ per servo through the transport's ``transact`` and
decodes the replies.

This bus knows the Feetech SYNC_WRITE convention (register + broadcast id) the
way ``CanBus`` knows CAN framing -- that is a transport-family orchestration
detail, not actuo.core. Per-servo value encoding still lives in the actuator
(``goal_position_payload``), so the bus holds no register scaling.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from actuo.core.contracts import (
    ActuatorCommand, ActuatorState, Bus, OperatingMode)
from actuo.drivers.actuators.feetech_scs import (
    REG_GOAL_POSITION, build_sync_write)


class SerialBus(Bus):
    """Orchestrates a set of Feetech servos on one half-duplex serial line."""

    def __init__(self, transport, actuators, read_timeout: float = 0.05) -> None:
        super().__init__(transport, actuators)
        self.read_timeout = float(read_timeout)
        self._states: Dict[int, ActuatorState] = {}
        self._missed = 0

    def set_mode(self, mode: OperatingMode) -> None:
        """Capability-check every servo. SCS0009 is POSITION-only, so there is
        no wire mode switch -- but the check still fails loudly on a bad mode."""
        for actuator in self.actuators:
            actuator.capabilities.require(mode)

    def enable_all(self) -> None:
        """Torque on for every servo (they hold position)."""
        self.transport.write([a.encode_torque_enable(True) for a in self.actuators])

    def release_all(self) -> None:
        """Torque off for every servo, so the hand is back-drivable."""
        self.transport.write([a.encode_release() for a in self.actuators])

    def write_batch(self, commands: Sequence[Optional[ActuatorCommand]]) -> None:
        """One SYNC_WRITE of goal positions; ``None`` releases that servo.

        A released servo cannot ride in the position SYNC_WRITE (that would
        command it a pose), so it is torqued off in the same burst -- the
        released and held servos never straddle two transactions.
        """
        if len(commands) != len(self.actuators):
            raise ValueError("write_batch expects {} commands, got {}".format(
                len(self.actuators), len(commands)))

        items = []
        releases = []
        for actuator, command in zip(self.actuators, commands):
            if command is None:
                releases.append(actuator.encode_release())
                continue
            actuator.capabilities.require(command.mode)
            if command.position is None:
                raise ValueError("POSITION mode needs command.position (rad)")
            items.append((actuator.id,
                          actuator.goal_position_payload(command.position)))

        frames = list(releases)
        if items:
            from actuo.core.contracts import Frame
            frames.append(Frame(payload=build_sync_write(REG_GOAL_POSITION, 2, items),
                                address=None))
        if frames:
            self.transport.write(frames)

    def read_batch(self, timeout: Optional[float] = None) -> List[ActuatorState]:
        """READ every servo's present-state block and decode it.

        A servo that does not answer keeps its last known state (or None on the
        first miss) and increments ``missed`` -- never a faked zero.
        """
        wait = self.read_timeout if timeout is None else float(timeout)
        queries = []
        for actuator in self.actuators:
            queries.extend(actuator.encode_query())
        replies = self.transport.transact(queries, wait)

        states: List[ActuatorState] = []
        for actuator, reply in zip(self.actuators, replies):
            if reply is not None and actuator.matches(reply):
                state = actuator.decode_state(reply)
                self._states[actuator.id] = state
            else:
                self._missed += 1
            states.append(self._states.get(actuator.id))
        return states

    def last_state(self, actuator_id: int) -> Optional[ActuatorState]:
        return self._states.get(actuator_id)

    @property
    def missed(self) -> int:
        return self._missed
