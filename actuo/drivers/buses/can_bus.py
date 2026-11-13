"""Bus orchestration for broadcast CAN actuators.

Protocol-agnostic on purpose: it drives whatever ``Actuator`` models it is
handed via ``encode_command`` / ``matches`` / ``decode_state``, so adding
another CAN servo family (RMD, Robstride, ODrive...) needs no edit here: a new
actuator is a new plugin, never a per-protocol ``switch`` in shared code.

Why there is no ``transact``
---------------------------
CAN is broadcast and these drivers upload status on their own schedule, so a
control cycle is "burst N command frames, then drain whatever has arrived".
Commands and state are decoupled; there is no request/response to pair up.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple

from actuo.core.contracts import (
    Actuator,
    ActuatorCommand,
    ActuatorState,
    Bus,
    DeviceId,
    Frame,
    OperatingMode,
    Transport,
)
from actuo.core.diagnostics import DiagItem, DiagLevel

#: How often an actuator that must be armed is re-armed while commanded. At a
#: 100 Hz control loop, every tenth cycle; proven on a biped's MIT leg drives.
REARM_PERIOD_S = 0.1

#: Longest a cycle waits for a motor to answer the first of two frames before
#: sending the second. A frame is about 0.13 ms on a 1 Mbit/s bus, so six
#: requests and six answers fit with room to spare; a bound, not a measured
#: latency. Spent only on cycles that arm, or release, a commanded motor.
REPLY_WAIT_S = 0.005


class OverspeedError(RuntimeError):
    """Raised when an actuator reports moving far faster than it was told to.

    The sibling of :class:`StaleFeedbackError`, for the same reason: something
    the driver is doing contradicts what we asked, and continuing to command it
    is the dangerous option.

    **Only meaningful while a POSITION is being commanded**, because
    ``max_velocity_erpm`` is the ``SET_POS_SPD`` trapezoid's own speed limit and
    nothing else declares one. Fired spuriously the first time it met CURRENT
    mode -- 14 deg/s of perfectly correct pushing, reported as "9x" against an
    envelope that had never been sent. See :meth:`CanBus.check_speed`.

    Written for a suspected joint runaway on a CubeMars arm (2026-07-17) whose
    headline number (113x) was later withdrawn: the distance was real, the
    interval was assumed.
    What survives is narrower and still worth guarding -- a driver in POSITION
    mode reporting a speed far above the envelope in that very packet is not
    doing what it was told, and the driver reports its own speed honestly (it
    does so even while unpowered, which is what let ``pole_pairs`` be verified
    by back-driving). If it ever happens, this stops it early and logs the real
    number, which is what the guesswork could not.
    """


class StaleFeedbackError(RuntimeError):
    """Raised when an actuator stops reporting.

    A position loop driving hardware from stale state is dangerous, so this is
    an error rather than a warning -- the caller is expected to stop. Named
    distinctly so a supervisor can catch exactly this.
    """


class CanBus(Bus):
    """Group CubeMars-style CAN actuators behind one transport.

    Parameters
    ----------
    transport : Transport
        An open-able CAN transport. The bus never assumes which backend.
    actuators : Sequence[Actuator]
        Ordered to match the robot's joint order; ``read_batch`` and
        ``write_batch`` keep that order.
    stale_after : float
        Seconds without a status frame before an actuator counts as stale.
        Defaults to 1.0 s, a threshold proven workable with a 20 Hz
        keepalive.
    """

    def __init__(
        self,
        transport: Transport,
        actuators: Sequence[Actuator],
        stale_after: float = 1.0,
    ) -> None:
        super().__init__(transport, actuators)
        self.stale_after = stale_after
        self._states: Dict[DeviceId, ActuatorState] = {}
        # Last loop each actuator was commanded in; check_speed needs it.
        self._last_mode: Dict[DeviceId, Optional[OperatingMode]] = {}
        #: When each actuator that must be ARMED was last armed. See
        #: write_batch; only models whose encode_mode() emits frames appear.
        self._armed_at: Dict[DeviceId, float] = {}
        #: Re-arm a commanded actuator this often, seconds.
        self.rearm_period = REARM_PERIOD_S
        self._last_rx: Dict[DeviceId, float] = {}
        self._unmatched_frames = 0
        self._unmatched_addresses: Dict[int, int] = {}
        self._error_frames = 0
        self._joint_name_by_id: Dict[DeviceId, str] = {}
        self._diag_last_error_frames: Optional[int] = None
        self._diag_last_unmatched_frames: Optional[int] = None
        self._diag_last_unmatched_addresses: Dict[int, int] = {}

    def set_joint_names(self, joint_names: Sequence[str]) -> None:
        """Name each configured id for operator-facing diagnostics."""
        if len(joint_names) != len(self.actuators):
            raise ValueError(
                "joint_names has {} entries for {} actuators".format(
                    len(joint_names), len(self.actuators)
                )
            )
        self._joint_name_by_id = {
            actuator.id: str(name)
            for actuator, name in zip(self.actuators, joint_names)
        }

    def _joint_name(self, device_id: DeviceId) -> str:
        return self._joint_name_by_id.get(device_id, "id_{}".format(device_id))

    @property
    def unmatched_frames(self) -> int:
        """Frames seen on the bus that no configured actuator claimed.

        Non-zero is not automatically wrong (other nodes may share the bus) but
        a steadily climbing count with no states usually means a mis-configured
        node id. Exposed rather than hidden: bus health is measured and
        reported, not assumed.
        """
        return self._unmatched_frames

    @property
    def unmatched_addresses(self) -> Dict[int, int]:
        """Arbitration id -> frame count, for *data* traffic nothing claimed.

        "I received 1701 frames and matched none" is only half an answer; the
        other half is *which* ids were on the wire, which turns a node-id
        mismatch from a guessing game into a lookup.

        Error frames are excluded on purpose. Their "arbitration id" is an error
        bitmask, not an address: counting them here made this driver report
        "motors are talking on node id 12" for what was really
        ``CAN_ERR_PROT|CAN_ERR_CRTL`` -- a fabricated answer that sent a real
        debugging session after the wrong fault. See :attr:`error_frames`.
        """
        return dict(self._unmatched_addresses)

    @property
    def error_frames(self) -> int:
        """Bus error frames seen. Non-zero means the wire is failing, full stop.

        Kept separate from :attr:`unmatched_frames` because the two demand
        opposite responses: unmatched data = fix your config; errors = fix the
        bus (or, as measured here, the adapter's sample point).
        """
        return self._error_frames

    def set_mode(self, mode: OperatingMode) -> None:
        """Capability-check every actuator and emit any mode-arming frames."""
        frames: List[Frame] = []
        for actuator in self.actuators:
            actuator.capabilities.require(mode)
            frames.extend(actuator.encode_mode(mode))
        if frames:
            self.transport.write(frames)

    def write_batch(self, commands: Sequence[Optional[ActuatorCommand]]) -> None:
        """Encode and send one command per actuator as a single burst.

        ``None`` in a slot means **release that actuator** (zero torque) rather
        than command it. That is what makes per-joint servo possible: "hold
        joint1, let joint2 go limp" is ONE bus transaction, not a command burst
        plus a separate release burst.

        Mixing them in one burst is not a micro-optimisation -- two bursts would
        put the joints in different states for the gap between them, and at 50 Hz
        that gap is long enough for a limb to start falling before the frame
        holding its neighbour lands (batch is the primitive, per-actuator ops
        are the special case).
        """
        if len(commands) != len(self.actuators):
            raise ValueError(
                "expected {} commands to match the actuators, got {}".format(
                    len(self.actuators), len(commands)
                )
            )
        now = time.monotonic()
        frames: List[Frame] = []
        # Second frames for motors that need two this cycle, sent once each
        # has answered its first: see _await_replies.
        follow: Dict[DeviceId, Frame] = {}
        for actuator, command in zip(self.actuators, commands):
            if command is None:
                # A release is an EXIT on a model that has one: whatever arming
                # there was is gone, and the next command must arm again. A
                # 2.0 MIT drive keeps its last command through EXIT and runs it
                # again on the next ENTER, so a drive that was commanded gets a
                # zero-gain command first and the EXIT after it.
                limp = getattr(actuator, "encode_limp", None)
                if limp is not None and actuator.id in self._armed_at:
                    frames.append(limp())
                    follow[actuator.id] = actuator.encode_release()
                else:
                    frames.append(actuator.encode_release())
                self._armed_at.pop(actuator.id, None)
                continue
            # Arm the models that ignore commands until armed (2.0 MIT:
            # ENTER_MOTOR_MODE) on the first command after a release, and
            # again every rearm_period while commanding. Without this, six MIT
            # leg drives (2026-09-29) reported "enabled" while limp and ignored
            # every target: knees asked for 10 deg moved 0.0. Re-arming
            # periodically means a dropped arming frame costs one period, not
            # the whole run. Models with nothing to arm (servo, 3.0) return no
            # frames and are never tracked.
            #
            # ONE frame per actuator per burst, so an arming cycle carries the
            # arming frame INSTEAD of the command, which follows next cycle.
            # 2.0 MIT answers on the same standard id it is addressed on, so
            # ENTER followed straight away by a command to the same motor puts
            # the host's command on the wire while the motor is answering the
            # ENTER: same id, different data, and arbitration cannot separate
            # them -- a bit error ("unable to send recessive bit"). That showed
            # as error frames only while MIT drives held torque (943 in three
            # minutes, against 0 on a bus with no MIT motor). A MIT motor holds
            # its last command across the skipped cycle.
            #
            # Except on the FIRST arm after a release. There the command it
            # holds is whatever it kept through EXIT -- after an abort that
            # never released, the last walking target at kp 500 -- and a whole
            # cycle of it kicked an AK80-64 knee with 105.7 N*m at torque on.
            # So the new command follows in the same cycle, as soon as the
            # motor has answered the ENTER; that keeps both the stored
            # command's window to about a millisecond and the wire free of the
            # collision above.
            last = self._armed_at.get(actuator.id)
            if last is None or now - last >= self.rearm_period:
                arming = actuator.encode_mode(command.mode)
                if arming:
                    frames.extend(arming)
                    self._armed_at[actuator.id] = now
                    if last is None:
                        follow[actuator.id] = actuator.encode_command(command)
                    continue
            frames.append(actuator.encode_command(command))
        # Remember what loop each actuator was last put in. check_speed() needs
        # it: the ERPM envelope belongs to SET_POS_SPD's trapezoid, so it only
        # means anything for a joint that is being commanded a POSITION.
        for actuator, command in zip(self.actuators, commands):
            self._last_mode[actuator.id] = None if command is None else command.mode
        self.transport.write(frames)
        if follow:
            self._await_replies(set(follow))
            self.transport.write(list(follow.values()))

    def _await_replies(self, ids: set) -> None:
        """Take in frames until each of ``ids`` has answered, or the wait ends.

        The second frame to a motor that answers on the id it is addressed on
        must not go out while that answer is on the wire. Running out of time
        sends it anyway: the cost is a retransmitted frame, and holding back a
        release or a fresh command is the worse failure. A motor that never
        answers is the stale-feedback guard's to report, not this wait's.
        """
        pending = set(ids)
        deadline = time.monotonic() + REPLY_WAIT_S
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return
            frame = self.transport.recv(timeout=remaining)
            if frame is None:
                return
            pending.discard(self._ingest(frame))

    def release_all(self) -> None:
        """Command zero torque to every actuator, in one burst.

        The counterpart of :meth:`write_batch`, and the only frame a fully-OFF
        supervisor is allowed to emit. Kept here so "off" is a bus-level
        operation with no actuator-specific knowledge leaking into the caller.

        Delegates rather than building its own frames: one code path means the
        all-off case cannot drift from the per-joint case.
        """
        self.write_batch([None] * len(self.actuators))

    def poll(self, timeout: float = 0.0) -> int:
        """Drain currently available status frames into the state cache.

        Returns the number of frames consumed. Does not block beyond ``timeout``
        for the first frame.
        """
        consumed, _matched = self.poll_counts(timeout=timeout)
        return consumed

    def solicit(self) -> int:
        """Ask the actuators that answer only when addressed, and say how many.

        Returns 0 on a bus where every model uploads on its own, which is the
        common case and costs one generator pass.

        The frame sent is ``encode_release()``, which is zero torque on every
        model that needs this. That is deliberate: a reader has no business
        commanding, and the one frame guaranteed safe to repeat at the loop
        rate is the one that asks for nothing. A cycle that IS commanding does
        not need this at all -- the command elicits the same reply -- so the
        caller should skip it then rather than send twice.
        """
        # Never an actuator that is being commanded: its release is an EXIT,
        # and sent between two commands it disarms the joint every period. On
        # a streaming bus this ran every 20 ms under the legs' commands, so a
        # MIT joint held torque for at most one cycle in two. Its commands
        # already elicit the replies this exists to get.
        frames = [actuator.encode_release() for actuator in self.actuators
                  if getattr(actuator, "answers_only_when_addressed", False)
                  and actuator.id not in self._armed_at]
        if frames:
            self.transport.write(frames)
        return len(frames)

    def poll_counts(self, timeout: float = 0.0) -> Tuple[int, int]:
        """Drain currently available frames and return ``(consumed, matched)``.

        ``consumed`` counts every drained CAN frame (including unmatched/error
        frames). ``matched`` counts only frames that decoded into a configured
        actuator state.
        """
        consumed = 0
        matched = 0
        while True:
            frame = self.transport.recv(timeout=timeout if consumed == 0 else 0.0)
            if frame is None:
                return consumed, matched
            device_id = self._ingest(frame)
            if device_id is not None:
                matched += 1
            consumed += 1

    def _ingest(self, frame: Frame) -> Optional[DeviceId]:
        """Route one frame to its actuator; return that actuator's id, or None."""
        # An error frame carries an error bitmask where an address would be, so
        # it is neither this actuator's state nor evidence about node ids.
        if frame.meta.get("error_frame"):
            self._error_frames += 1
            return None
        for actuator in self.actuators:
            if not actuator.matches(frame):
                continue
            # A malformed frame from a matched node is a real fault; let the
            # decoder's error surface rather than dropping it silently.
            self._states[actuator.id] = actuator.decode_state(frame)
            self._last_rx[actuator.id] = time.monotonic()
            return actuator.id
        self._unmatched_frames += 1
        if frame.address is not None:
            key = int(frame.address)
            self._unmatched_addresses[key] = self._unmatched_addresses.get(key, 0) + 1
        return None

    def read_batch(self, timeout: float) -> List[ActuatorState]:
        """Return the freshest state per actuator, in actuator order.

        Waits up to ``timeout`` for every actuator to report at least once
        during this call, then returns the cache. Raises rather than inventing
        a state for a silent motor.
        """
        deadline = time.monotonic() + timeout
        seen: set = set()

        while len(seen) < len(self.actuators):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            frame = self.transport.recv(timeout=remaining)
            if frame is None:
                break
            device_id = self._ingest(frame)
            if device_id is not None:
                seen.add(device_id)

        missing = [a.id for a in self.actuators if a.id not in self._states]
        if missing:
            raise StaleFeedbackError(
                "no status frame ever received from actuator(s) {}; check node "
                "ids, bus wiring and bitrate".format(missing)
            )
        return [self._states[a.id] for a in self.actuators]

    def check_fresh(self) -> None:
        """Raise if any actuator's last status frame is older than ``stale_after``.

        A driver that stops talking while a position loop keeps commanding is
        the dangerous case.
        """
        now = time.monotonic()
        stale = [
            (a.id, now - self._last_rx[a.id])
            for a in self.actuators
            if a.id in self._last_rx and now - self._last_rx[a.id] > self.stale_after
        ]
        never = [a.id for a in self.actuators if a.id not in self._last_rx]
        if never:
            raise StaleFeedbackError(
                "actuator(s) {} have never reported".format(never)
            )
        if stale:
            detail = ", ".join(
                "{} silent for {:.2f}s".format(device_id, age) for device_id, age in stale
            )
            raise StaleFeedbackError(detail)

    def check_speed(self, tolerance: float = 4.0) -> None:
        """Raise if an actuator reports a speed its envelope does not allow.

        **Only for joints being commanded a POSITION via SET_POS_SPD.**
        ``max_velocity_erpm`` is that packet's speed limit; it is a parameter
        of that packet and nothing else, so a joint configured
        ``use_pos_spd: false`` is skipped too -- SET_POS carries no envelope
        and the motor runs at its own maximum by design. In CURRENT mode no speed is commanded at all --
        the motor turns as fast as the torque and the load agree on, and that is
        the whole point of the mode. Checking a current-mode joint against a
        position-mode envelope compares a measurement to a limit that was never
        sent, and the first thing it did was abort a perfectly healthy 14 deg/s
        push during a limit search, calling it "9x". A guard that fires on
        correct behaviour is worse than no guard: it gets switched off.

        Compared in ERPM, the wire's own unit, on purpose: it is what the
        envelope was configured in and what the driver reports, so the check
        holds even for a motor whose ``pole_pairs`` is unknown. Converting to
        rad/s to compare would make this guard need a datasheet number and
        vanish exactly where the config is least complete.

        Parameters
        ----------
        tolerance : float
            Multiple of the configured envelope treated as still plausible.
            Default 4x, which is deliberately loose: SET_POS_SPD is a trapezoid,
            the reported speed is a noisy estimate, and this must never fire on
            a healthy overshoot. The event it exists for was 113x. A guard that
            cries wolf gets its threshold widened until it is useless, so it is
            set to catch runaways and nothing else.
        """
        offenders = []
        for actuator in self.actuators:
            if self._last_mode.get(actuator.id) is not OperatingMode.POSITION:
                continue          # no speed was commanded; nothing to check
            if not getattr(actuator, "use_pos_spd", True):
                # POSITION, but sent as SET_POS (mode 4), which carries no
                # envelope: the motor runs to the target at ITS maximum. So
                # max_velocity_erpm was never transmitted, and comparing a
                # measurement against it is the same mistake this guard
                # already documents for CURRENT mode -- a limit that was never
                # sent. It fired on 2026-07-29 at "1720 ERPM against 410 (4x)"
                # on a joint doing exactly what ros2_control had asked, and
                # took the servicer down mid-motion. A guard that fires on
                # correct behaviour is worse than no guard.
                #
                # Nothing replaces it here, and that is honest rather than
                # convenient: with SET_POS the speed fence lives entirely in
                # ros2_control's enforce_command_limits, above this layer, and
                # this class has no access to it. Pretending otherwise would
                # mean inventing a limit.
                continue
            limit = getattr(getattr(actuator, "limits", None),
                            "max_velocity_erpm", None)
            if not limit:
                continue
            state = self._states.get(actuator.id)
            if state is None or not state.aux:
                continue
            erpm = state.aux.get("erpm")
            if erpm is None:
                continue
            if abs(erpm) > limit * tolerance:
                offenders.append((actuator.id, erpm, limit))
        if offenders:
            raise OverspeedError(
                "; ".join(
                    "actuator {} reports {:.0f} ERPM but was given an envelope "
                    "of {:.0f} ({:.0f}x)".format(
                        device_id, erpm, limit, abs(erpm) / limit)
                    for device_id, erpm, limit in offenders)
                + ". The driver is not honouring the speed it was commanded.")

    def last_state(self, device_id: DeviceId) -> Optional[ActuatorState]:
        return self._states.get(device_id)

    def diagnose(self) -> List[DiagItem]:
        """Backend-specific CAN health: bus trends plus per-node silence/faults."""
        items: List[DiagItem] = []

        error_delta: Optional[int] = None
        if self._diag_last_error_frames is not None:
            error_delta = self._error_frames - self._diag_last_error_frames
        self._diag_last_error_frames = int(self._error_frames)

        unmatched_delta: Optional[int] = None
        if self._diag_last_unmatched_frames is not None:
            unmatched_delta = self._unmatched_frames - self._diag_last_unmatched_frames
        self._diag_last_unmatched_frames = int(self._unmatched_frames)

        can_level = DiagLevel.OK
        can_message = "No CAN error frames observed."
        if error_delta is not None and error_delta > 0:
            can_level = DiagLevel.WARN
            can_message = (
                "CAN error frames increased by {} since last diagnostics sample; "
                "rising error-frame trends can precede a node going silent."
                .format(error_delta)
            )
        elif self._error_frames > 0:
            can_level = DiagLevel.WARN
            can_message = (
                "CAN error-frame total is nonzero; not rising this sample, but "
                "the wire has seen physical-layer errors."
            )

        values = {
            "error_frames_total": str(int(self._error_frames)),
            "error_frames_delta": (
                "unavailable" if error_delta is None else str(int(error_delta))
            ),
            "unmatched_frames_total": str(int(self._unmatched_frames)),
            "unmatched_frames_delta": (
                "unavailable" if unmatched_delta is None else str(int(unmatched_delta))
            ),
        }
        items.append(DiagItem(
            name="bus/can_error_frames",
            level=can_level,
            message=can_message,
            values=values,
        ))

        address_values: Dict[str, str] = {}
        unmatched_level = DiagLevel.OK
        unmatched_message = "No unmatched CAN data-frame addresses observed."
        rising = False
        for address, total in sorted(self._unmatched_addresses.items()):
            previous = int(self._diag_last_unmatched_addresses.get(address, 0))
            delta = int(total) - previous
            address_values["0x{:03X}_total".format(int(address))] = str(int(total))
            address_values["0x{:03X}_delta".format(int(address))] = str(int(delta))
            if delta > 0:
                rising = True
        if self._unmatched_addresses:
            unmatched_level = DiagLevel.WARN
            if rising:
                unmatched_message = (
                    "Unmatched CAN data frames increased this sample; check node-id "
                    "mapping and whether another protocol family is sharing the bus."
                )
            else:
                unmatched_message = (
                    "Unmatched CAN data frames exist but did not increase this sample."
                )
        items.append(DiagItem(
            name="bus/unmatched_addresses",
            level=unmatched_level,
            message=unmatched_message,
            values=address_values,
        ))
        self._diag_last_unmatched_addresses = dict(self._unmatched_addresses)

        now = time.monotonic()
        for actuator in self.actuators:
            device_id = actuator.id
            state = self._states.get(device_id)
            age = (None if device_id not in self._last_rx
                   else now - self._last_rx[device_id])
            values = {
                "device_id": str(device_id),
                "stale_after_s": "{:.3f}".format(float(self.stale_after)),
                "silence_age_s": (
                    "never" if age is None else "{:.3f}".format(float(age))
                ),
            }

            level = DiagLevel.OK
            message_parts: List[str] = []
            if age is None:
                level = DiagLevel.STALE
                message_parts.append(
                    "No status frame has ever been received from this node."
                )
            elif age > self.stale_after:
                level = DiagLevel.ERROR
                message_parts.append(
                    "Node is silent for {:.3f}s, over stale_after {:.3f}s."
                    .format(float(age), float(self.stale_after))
                )
            else:
                message_parts.append(
                    "Last status frame age {:.3f}s.".format(float(age))
                )

            if state is not None and state.error is not None:
                code = int(state.error)
                values["fault_code"] = str(code)
                fault_name = actuator.fault_name(code)
                values["fault_name"] = fault_name
                if code != 0:
                    level = DiagLevel.ERROR
                    message_parts.append(
                        "Fault code {} ({}).".format(code, fault_name)
                    )

            items.append(DiagItem(
                name="servo/{}".format(self._joint_name(device_id)),
                level=level,
                message=" ".join(message_parts),
                values=values,
            ))

        return items
