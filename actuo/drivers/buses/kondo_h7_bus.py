"""Kondo ARM-H7 bus: one servo packet per cycle, one strided read per slot.

Orchestrates the ICS servos hanging off an H7 gateway board. This is where the
board's memory layout meets the actuator models, and it is deliberately the
*only* place that knows both:

* ``write_batch`` builds a **single** packet carrying every servo's target
  (batch is the primary operation), not a loop of singles. The board is request/response,
  so one cycle is one round trip regardless of joint count.
* ``read_batch`` reads each requested state slot with **one** strided
  ``MREADV`` -- e.g. all 36 servos' measured pulses in a single reply -- then
  hands each actuator its own slice to decode. The struct layout stays here;
  the units stay in the actuator.

Why the state slice is a synthesised frame
------------------------------------------
The board answers a read with one reply covering every servo, so there is no
per-actuator frame arriving off the wire to hand to ``decode_state``. Rather
than teach the actuator the whole vector's layout (which would put board
knowledge in a servo model), the bus slices the reply and synthesises a
one-servo ``Frame`` stamped with the servo id and the slot names it contains.
``actuo.core`` needed no change to allow this -- a second protocol family
fits without editing the shared contracts.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from actuo.core.contracts import (
    ActuatorCommand, ActuatorState, Bus, Frame, OperatingMode)
from actuo.core.diagnostics import DiagItem, DiagLevel

from actuo.drivers.actuators.kondo_ics import (
    KondoIcsActuator, SERVO_OFF_PULSE, SERVO_ON_PULSE)
from actuo.drivers.actuators.kondo_ics_temperature import (
    setting_value_to_celsius)
from actuo.drivers.kondo_h7_protocol import (
    CONTROL_SLOTS, SYSTEM_SLOTS, ProtocolError, SERVO_VECTOR_LENGTH,
    SymbolMap,
    build_multi_servo, build_version, control_slot_read, decode_servo_slot,
    servo_slot_read, servo_slot_write, system_slot_read,
    system_slot_write)

#: Milliseconds per unit of the board's move-duration byte. Set by the firmware
#: (``FRAME_SCALE``), so a move byte of 127 means ~1.27 s to the target.
FRAME_SCALE_MS = 10.0

#: State slots read every cycle by default: just the measured pulse, which is
#: one round trip. Current and temperature cost an extra round trip *each* and
#: only report anything while the board's read flags are on, so they are opt-in.
DEFAULT_STATE_SLOTS = ("current_angle",)


@dataclass(frozen=True)
class KondoDiscovery:
    """Snapshot of board discovery, with verified and inferred sources separated.

    ``present_ids`` and ``ports`` come from ``ServoStruct.flag`` and
    ``ServoStruct.port`` and are the authoritative discovery facts.

    ``per_port`` and ``timed_out_ids`` come from ``Control.device_port``, whose
    index-to-servo-id packing was measured against ``ServoStruct.port`` on one
    board (see :meth:`KondoH7Bus._decode_device_port`). One board is not every
    board, so it stays the second source and the cross-check stays worth running.
    ``timed_out_ids`` is the only fact it contributes that the authoritative
    source does not carry at all.

    ``ics_channel_scan_error`` explains why ``per_port``/``timed_out_ids`` are
    unavailable (``None``), so an unavailable scan never looks like an empty one.
    """

    present_ids: Tuple[int, ...]
    ports: Mapping[int, int]
    per_port: Optional[Mapping[int, Tuple[int, ...]]]
    timed_out_ids: Optional[Tuple[int, ...]]
    firmware_version: str
    ics_channel_scan_error: Optional[str] = None


def compare_discovery(discovery: KondoDiscovery,
                      expected_ids: Sequence[int]) -> Tuple[Set[int], Set[int], Optional[Set[int]]]:
    """Compare board discovery against the ids the robot config expects."""
    expected = {int(servo_id) for servo_id in expected_ids}
    found = set(discovery.present_ids)
    missing = expected - found
    unexpected = found - expected
    timed_out = (None if discovery.timed_out_ids is None
                 else set(discovery.timed_out_ids))
    return missing, unexpected, timed_out


class KondoH7Bus(Bus):
    """Drives a set of :class:`KondoIcsActuator` through one H7 board.

    Parameters
    ----------
    transport : Transport
        A :class:`KondoH7Transport` (or anything with the same contract).
    actuators : sequence of Actuator
        In URDF joint order -- that order is the contract with ros2_control.
    symbols : SymbolMap
        Firmware addresses, loaded from the committed symbol map.
    state_slots : sequence of str
        Which ``ServoStruct`` members to read each cycle. Each costs one round
        trip, so this is an explicit cost the robot config owns.
    move_time_ms : float, optional
        How long the board should take to reach each target. Converted to the
        firmware's move-duration byte. Leave ``None`` to use each actuator's own
        default.

        Set this from the control period, not from a habit: the packet is a
        *move*, and every packet restarts the interpolation. A move time far
        shorter than the cycle makes the servo lurch and stop between packets;
        far longer makes it never arrive.
    read_timeout : float
        Seconds to wait for one reply.
    """

    def __init__(self, transport, actuators, symbols: SymbolMap,
                 state_slots: Sequence[str] = DEFAULT_STATE_SLOTS,
                 move_time_ms: Optional[float] = None,
                 read_timeout: float = 0.1,
                 joint_names: Optional[Sequence[str]] = None,
                 battery_voltage_warn_below: Optional[float] = None) -> None:
        super().__init__(transport, actuators)
        unsupported = [a.id for a in actuators
                       if getattr(a, "stretch", None) is not None
                       or getattr(a, "hold_relax", None) is not None]
        if unsupported:
            # Ignoring them would leave a gripper pushing at full stretch
            # that its config says is held lighter.
            raise ValueError(
                "servo ids {} set `stretch` or `hold_relax`, which only the "
                "RCB-4mini bus implements".format(unsupported))
        self.symbols = symbols
        self.state_slots = tuple(state_slots)
        if not self.state_slots:
            raise ValueError(
                "a bus that reads no state slot reports no state; name at "
                "least 'current_angle'")
        self.read_timeout = float(read_timeout)
        self.move_time_ms = move_time_ms
        self._servo_base = symbols.address("servo_vector")
        self._system_base = symbols.address("SysB")
        self._control_base = symbols.address("Control") if symbols.has("Control") else None
        self._states: Dict[int, ActuatorState] = {}
        self._missed = 0
        # Cycles in which at least one slot answered, i.e. in which any
        # actuator's state was actually refreshed. Counted because `missed`
        # cannot answer that question: misses are per SLOT and this bus reads
        # one slot for all 13 servos, so a cycle where NOTHING answered scores
        # one miss and twelve apparently-fresh actuators. That arithmetic let a
        # completely frozen robot look healthy (2026-08-01, FTDI: every read
        # timed out and joint_states published an unchanging pose for minutes).
        self._state_updates = 0
        self._servo_on: Dict[int, bool] = {a.id: False for a in actuators}
        self._joint_name_by_id: Dict[int, str] = {}
        self._startup_discovery: Optional[KondoDiscovery] = None
        self._startup_present_ids: Optional[Set[int]] = None
        self._startup_missing_ids: Set[int] = set()
        self._startup_allow_missing = False
        self._diag_last_timeout_count: Dict[int, int] = {}
        self._diag_last_rxerror_count: Dict[int, int] = {}
        self._battery_voltage_warn_below = (
            None if battery_voltage_warn_below is None
            else float(battery_voltage_warn_below))

        for actuator in actuators:
            if not 0 <= int(actuator.id) < SERVO_VECTOR_LENGTH:
                raise ValueError(
                    "servo id {} is outside the board's servo vector (0..{})"
                    .format(actuator.id, SERVO_VECTOR_LENGTH - 1))

        if joint_names is not None:
            self.set_joint_names(joint_names)

    def set_joint_names(self, joint_names: Sequence[str]) -> None:
        """Name each configured servo id for operator-facing diagnostics."""
        if len(joint_names) != len(self.actuators):
            raise ValueError(
                "joint_names has {} entries for {} actuators".format(
                    len(joint_names), len(self.actuators)))
        self._joint_name_by_id = {
            int(actuator.id): str(name)
            for actuator, name in zip(self.actuators, joint_names)
        }

    def set_battery_voltage_threshold(self, warn_below: Optional[float]) -> None:
        """Set/clear the optional low-voltage warning threshold in volts."""
        self._battery_voltage_warn_below = (
            None if warn_below is None else float(warn_below))

    # -- housekeeping ------------------------------------------------------

    @property
    def state_updates(self) -> int:
        """Cycles in which at least one state slot answered. See __init__."""
        return self._state_updates

    @property
    def missed(self) -> int:
        """Reads that did not answer. Never turned into a fabricated state."""
        return self._missed

    def firmware_version(self) -> str:
        """Ask the board what firmware it runs (a plain string)."""
        reply = self.transport.transact(
            [Frame(payload=build_version())], self.read_timeout)[0]
        if reply is None:
            raise ProtocolError("board did not answer the version request")
        return reply.payload.decode("utf-8", errors="replace").strip("\x00").strip()

    def verify_firmware(self) -> str:
        """Refuse to continue unless the board matches the loaded symbol map.

        Called at connect time. Every address this bus uses comes from that map,
        so a mismatch would read the right number of bytes from the wrong place
        and report a confident, wrong angle.
        """
        version = self.firmware_version()
        self.symbols.require_version(version)
        return version

    def _bus_move_byte(self) -> Optional[int]:
        """The move-duration byte this bus wants, or ``None`` for "servo's own".

        ``move_time_ms`` is a board-side interpolation time tuned for the robot
        as a whole (20 ms on the humanoid it was tuned on), so when it is set it applies to every servo
        -- but each servo still has the last word through
        :meth:`KondoBatchServo.move_byte`, because a servo in rotation mode is
        reading that field as nothing of the kind.
        """
        if self.move_time_ms is None:
            return None
        return max(1, min(255, int(round(self.move_time_ms / FRAME_SCALE_MS))))

    def _velocity_for(self, actuator) -> int:
        return actuator.move_byte(self._bus_move_byte())

    def _command(self, payload: bytes) -> Optional[Frame]:
        """Send one packet and consume its reply (the board answers everything)."""
        return self.transport.transact(
            [Frame(payload=payload)], self.read_timeout)[0]

    def _read_servo_slot_vector(self, slot_name: str) -> List[int]:
        """Read one ``ServoStruct`` slot across every servo id."""
        reply = self.transport.transact(
            [Frame(payload=servo_slot_read(self._servo_base, slot_name))],
            self.read_timeout)[0]
        if reply is None:
            raise ProtocolError(
                "board answered the link but not ServoStruct.{}; cannot scan"
                .format(slot_name))
        return decode_servo_slot(slot_name, reply.payload)

    def _require_control_base(self) -> int:
        """Address of ``Control`` or a loud error naming the missing symbol."""
        if self._control_base is None:
            raise ProtocolError(
                "symbol map for firmware {!r} is missing required symbol "
                "'Control' for ICS discovery; regenerate it with "
                "actuo-kondo-symbols and retry".format(self.symbols.version))
        return self._control_base

    def _read_system_slot(self, slot_name: str):
        """Read one ``SystemStruct`` slot and decode it to Python values."""
        reply = self.transport.transact(
            [Frame(payload=system_slot_read(self._system_base, slot_name))],
            self.read_timeout)[0]
        if reply is None:
            raise ProtocolError(
                "board did not answer SystemStruct.{} read".format(slot_name))
        return SYSTEM_SLOTS[slot_name].decode(reply.payload)

    @staticmethod
    def _try_parse_ics_port(ics_channel) -> Optional[int]:
        """``J1``..``J6`` (or ``1``..``6``) to an integer port index."""
        if ics_channel is None:
            return None
        text = str(ics_channel).strip().upper()
        if text.startswith("J"):
            text = text[1:]
        try:
            port = int(text)
        except ValueError:
            return None
        if 1 <= port <= 6:
            return port
        return None

    def _configured_port_for(self, actuator) -> Optional[int]:
        return self._try_parse_ics_port(getattr(actuator, "ics_channel", None))

    @staticmethod
    def _port_label(port: Optional[int]) -> str:
        return "J{}".format(port) if port is not None else "unknown"

    def _joint_name(self, servo_id: int) -> str:
        return self._joint_name_by_id.get(int(servo_id), "id_{}".format(servo_id))

    @staticmethod
    def _format_ids(ids: Sequence[int]) -> str:
        return "[" + ", ".join(str(i) for i in ids) + "]"

    def _read_servo_slot_vector_safe(
            self, slot_name: str) -> Tuple[Optional[List[int]], Optional[str]]:
        try:
            return self._read_servo_slot_vector(slot_name), None
        except ProtocolError as error:
            return None, str(error)

    def _read_system_slot_safe(
            self, slot_name: str) -> Tuple[Optional[Tuple], Optional[str]]:
        try:
            return tuple(self._read_system_slot(slot_name)), None
        except ProtocolError as error:
            return None, str(error)

    def _decode_device_port(
            self, values: Sequence[int]) -> Tuple[Dict[int, List[int]], Dict[int, int], Set[int]]:
        """Decode ``Control.device_port`` into per-port ids and timeout flags.

        Per byte: bit 7 = present, bit 6 = timed out, the low bits are a port
        bitmask equal to ``2 ** (port - 1)``. That part is rcb4's and is
        unchanged.

        The index, however, IS the servo id
        -----------------------------------
        rcb4 reads this table as ``servo_id = index // 2`` (``max_SIO_num == 2``,
        ``rcb4/struct_header.py:3``). On real hardware that is wrong, and the
        table itself says so: measured on a humanoid's ARM-H7 board (firmware v0.6.5,
        2026-07-31, `references.md`), ``index == servo_id`` reproduces the
        authoritative ``ServoStruct.port`` for **13 of 13** present servos while
        ``index // 2`` manages 3, and only by accident. The robot that settled it
        alternates arms across two UARTs (ids 0/2/4/6 on J4, 1/3/5 on J3), so the
        ``// 2`` reading folds two servos on *different* ports onto one id and
        contradicts itself -- which is how this was noticed rather than believed.

        Consequently there is no id >= 32 blind spot: ids 33 and 35 (that robot's head)
        appear at indices 33 and 35, and were read there. The table's real bound
        is 64 ids; this driver still refuses anything outside the servo vector.

        ``ServoStruct.flag`` / ``ServoStruct.port`` remain authoritative --
        keeping them so is what let a second source be *checked* instead of
        trusted. Cross-check with
        :meth:`compare_servo_ports_with_inferred_ics_ports` on any new board.
        """
        if len(values) != CONTROL_SLOTS["device_port"].length:
            raise ProtocolError(
                "Control.device_port expected {} entries, got {}"
                .format(CONTROL_SLOTS["device_port"].length, len(values)))

        per_port_sets: Dict[int, Set[int]] = {port: set() for port in range(1, 7)}
        servo_ports: Dict[int, int] = {}
        timed_out: Set[int] = set()
        valid_masks = {1, 2, 4, 8, 16, 32}

        for index, raw_state in enumerate(values):
            state = int(raw_state)
            if state < 128:
                continue

            state -= 128
            timeout = False
            if state >= 64:
                state -= 64
                timeout = True

            bitmask = state
            if bitmask not in valid_masks:
                raise ProtocolError(
                    "Control.device_port[{}] has implausible port bitmask {}; "
                    "raw value {} decodes to no valid ICS UART"
                    .format(index, bitmask, int(raw_state)))

            servo_id = index
            if not 0 <= servo_id < SERVO_VECTOR_LENGTH:
                raise ProtocolError(
                    "Control.device_port index {} decodes to servo id {} "
                    "outside 0..{}".format(
                        index, servo_id, SERVO_VECTOR_LENGTH - 1))

            port = bitmask.bit_length()
            servo_ports[servo_id] = port
            per_port_sets[port].add(servo_id)
            if timeout:
                timed_out.add(servo_id)

        per_port = {port: sorted(ids) for port, ids in per_port_sets.items()}
        return per_port, servo_ports, timed_out

    def _read_device_port_snapshot(
            self) -> Tuple[Dict[int, List[int]], Dict[int, int], Set[int]]:
        """Read and decode ``Control.device_port`` once."""
        reply = self.transport.transact(
            [Frame(payload=control_slot_read(self._require_control_base(),
                                             "device_port"))],
            self.read_timeout)[0]
        if reply is None:
            raise ProtocolError(
                "board did not answer Control.device_port; cannot scan ICS "
                "channel discovery")
        values = CONTROL_SLOTS["device_port"].decode(reply.payload)
        return self._decode_device_port(values)

    # -- Bus contract ------------------------------------------------------

    def set_mode(self, mode: OperatingMode) -> None:
        """Capability-check every servo. ICS is position-only, so there is no
        wire mode switch -- but asking for VELOCITY still fails loudly."""
        for actuator in self.actuators:
            actuator.capabilities.require(mode)

    def write_batch(self, commands: Sequence[Optional[ActuatorCommand]]) -> None:
        """Push every servo's target in one packet.

        A ``None`` command releases that servo. Released and held servos ride in
        the *same* packet -- the off sentinel simply takes the place of a pulse
        -- so they can never straddle two transactions and leave the robot half
        commanded.
        """
        if len(commands) != len(self.actuators):
            raise ValueError("write_batch expects {} commands, got {}".format(
                len(self.actuators), len(commands)))

        ids: List[int] = []
        pulses: List[int] = []
        velocities: List[int] = []
        for actuator, command in zip(self.actuators, commands):
            ids.append(actuator.id)
            velocities.append(self._velocity_for(actuator))
            if command is None:
                pulses.append(SERVO_OFF_PULSE)
                self._servo_on[actuator.id] = False
                continue
            # What the 16-bit field means -- a target pulse, or a speed for a
            # servo in rotation mode -- belongs to the servo, not here. See
            # KondoBatchServo: an arm joint and a wheel share this packet, and
            # asking which is which would be a protocol switch in shared code.
            pulses.append(actuator.batch_pulse(command))
            self._servo_on[actuator.id] = True

        if not ids:
            return
        uniform = len(set(velocities)) == 1
        self._command(build_multi_servo(
            ids, pulses, velocities[0] if uniform else velocities))

    def read_batch(self, timeout: Optional[float] = None) -> List[ActuatorState]:
        """Read the configured state slots and decode one state per actuator.

        One strided read per slot, each covering every servo. A slot that does
        not answer is dropped from this cycle rather than filled in; if nothing
        answered, every actuator keeps its previous state (or ``None`` on the
        first cycle) and ``missed`` counts up.
        """
        wait = self.read_timeout if timeout is None else float(timeout)
        raw: Dict[str, List[int]] = {}
        for slot in self.state_slots:
            reply = self.transport.transact(
                [Frame(payload=servo_slot_read(self._servo_base, slot))], wait)[0]
            if reply is None:
                self._missed += 1
                continue
            try:
                raw[slot] = decode_servo_slot(slot, reply.payload)
            except ProtocolError:
                # A short or over-long reply means the stream is out of step;
                # decoding it anyway would mis-attribute one servo's angle to
                # another. Drop the cycle and let the counter show it.
                self._missed += 1

        if raw:
            self._state_updates += 1

        states: List[ActuatorState] = []
        for actuator in self.actuators:
            if raw:
                frame = self._state_frame(actuator.id, raw)
                if actuator.matches(frame):
                    self._states[actuator.id] = actuator.decode_state(frame)
            states.append(self._states.get(actuator.id))
        return states

    def check_rotation_modes(self) -> None:
        """Refuse to run if a servo's mode disagrees with the config's model.

        ``ServoStruct.rotation`` says whether the board treats this servo's
        16-bit command field as a target pulse or as a SPEED. The two actuator
        models write that same field and mean opposite things by it, so a
        mismatch is not a scaling error:

        * a wheel driven by the position model gets a pose command read as a
          speed -- it spins up and keeps spinning, since nothing in a position
          controller ever sends the value that means stop;
        * a joint driven by the wheel model gets a speed read as a pose and
          steps to it.

        The board knows which it is, so this is checked rather than trusted, and
        it raises: a warning here would be read after the wheels were already
        turning.
        """
        values = self._read_servo_slot_vector("rotation")
        wrong = []
        for actuator in self.actuators:
            servo_id = int(actuator.id)
            if servo_id >= len(values):
                continue
            actual = bool(values[servo_id])
            if actual != bool(actuator.expects_rotation):
                wrong.append(
                    "{} (id {}) is configured as {!r}, which needs "
                    "ServoStruct.rotation={}, but the board has {}".format(
                        self._joint_name(servo_id), servo_id,
                        type(actuator).__name__,
                        int(actuator.expects_rotation), int(actual)))
        if wrong:
            raise RuntimeError(
                "servo rotation mode disagrees with the config: {}. Set the "
                "servo's mode on the board (or fix the config's plugin), then "
                "restart -- commanding it as configured would drive it as the "
                "other kind.".format("; ".join(wrong)))

    def clear_feedback_flags(self) -> List[int]:
        """Clear ``ServoStruct.feedback`` where it is set, and say which.

        A servo whose ``feedback`` byte is non-zero is NOT DRIVEN by this board.
        Commands still land -- ``ref_angle`` updates, the wire frame is
        byte-for-byte what rcb4 sends -- and the servo does not move. Measured
        on a wheeled humanoid from a cold power-up: the four wheels came up with
        ``feedback = 1`` and every arm and head servo with 0, which is why this
        was invisible until something continuous-rotation was tried.

        rcb4 clears it inside ``search_servo_ids`` without comment, so it reads
        as bookkeeping rather than the thing that makes a wheel turn. It is the
        latter: clearing the byte and repeating the identical command turned the
        wheel immediately.

        Returns the ids it changed, so a caller can report it. This modifies the
        board, so it is a named call rather than a side effect of a scan.
        """
        values = self._read_servo_slot_vector("feedback")
        cleared = []
        for actuator in self.actuators:
            servo_id = int(actuator.id)
            if servo_id < len(values) and values[servo_id]:
                self._command(servo_slot_write(
                    self._servo_base, "feedback", servo_id, 0))
                cleared.append(servo_id)
        return cleared

    def last_state(self, device_id) -> Optional[ActuatorState]:
        """Most recent decoded state for one servo; ``None`` before its first.

        The board answers every read, so a servo with no entry here has not been
        read yet -- it is not a servo sitting at zero.
        """
        return self._states.get(device_id)

    def _state_frame(self, servo_id: int, raw: Dict[str, List[int]]) -> Frame:
        """Slice the board-wide read into one servo's frame.

        Values are packed as little-endian int32 regardless of their width on
        the board. The struct's own types (uint16 pulse, uint8 current, int16
        error) were already applied by :func:`decode_servo_slot`, so the widening
        is lossless and the actuator never has to know the board's signedness.
        """
        slots = tuple(raw)
        values = [raw[slot][servo_id] for slot in slots]
        payload = struct.pack("<" + "i" * len(values), *values)
        return Frame(payload=payload, address=servo_id,
                     meta={"kind": "servo_state", "slots": slots})

    # -- discovery ---------------------------------------------------------

    def scan_servo_flags(self) -> Dict[int, bool]:
        """Board-side presence scan from ``ServoStruct.flag``."""
        values = self._read_servo_slot_vector("flag")
        return {servo_id: bool(values[servo_id]) for servo_id in range(len(values))}

    def scan_servo_ports(self) -> Dict[int, int]:
        """Board-side UART assignment from ``ServoStruct.port``."""
        values = self._read_servo_slot_vector("port")
        return {servo_id: int(values[servo_id]) for servo_id in range(len(values))}

    def scan_ics_channels_per_port(self) -> Dict[int, List[int]]:
        """Decode inferred ``Control.device_port`` into ``{port: [ics ids...]}``.

        This read is optional for discovery. Its packing is measured, not
        inferred (see :meth:`_decode_device_port`), but on one board only -- use
        ``scan_servo_flags`` / ``scan_servo_ports`` as the authoritative presence
        source and this as the cross-check.
        """
        per_port, _, _ = self._read_device_port_snapshot()
        return per_port

    def compare_servo_ports_with_inferred_ics_ports(
            self,
            servo_ports: Mapping[int, int],
            inferred_ics_ports: Mapping[int, int]) -> Dict[int, Tuple[int, int]]:
        """Return UART disagreements between verified and inferred sources.

        Parameters
        ----------
        servo_ports : mapping of int to int
            Verified UART assignment from ``ServoStruct.port``.
        inferred_ics_ports : mapping of int to int
            Inferred UART assignment decoded from ``Control.device_port``.

        Returns
        -------
        dict
            ``{servo_id: (servo_struct_port, inferred_device_port)}`` for ids
            present in both mappings where the UART differs.
        """
        disagreements: Dict[int, Tuple[int, int]] = {}
        for servo_id, inferred_port in sorted(inferred_ics_ports.items()):
            if servo_id not in servo_ports:
                continue
            verified_port = int(servo_ports[servo_id])
            inferred_port_int = int(inferred_port)
            if verified_port != inferred_port_int:
                disagreements[int(servo_id)] = (verified_port, inferred_port_int)
        return disagreements

    def discover(self) -> KondoDiscovery:
        """Read startup discovery, always preferring verified ``ServoStruct`` data.

        ``Control.device_port`` enriches the result when it is available and
        decodes cleanly. If that inferred scan is unavailable or implausible,
        discovery still returns verified ``flag``/``port`` data and records the
        reason in ``ics_channel_scan_error``.
        """
        firmware_version = self.firmware_version()
        flags = self.scan_servo_flags()
        ports = self.scan_servo_ports()
        per_port: Optional[Mapping[int, Tuple[int, ...]]] = None
        timed_out_ids: Optional[Tuple[int, ...]] = None
        ics_channel_scan_error = None
        try:
            inferred_per_port, _, timed_out = self._read_device_port_snapshot()
        except ProtocolError as error:
            ics_channel_scan_error = str(error)
        else:
            per_port = {
                port: tuple(ids) for port, ids in inferred_per_port.items()
            }
            timed_out_ids = tuple(sorted(timed_out))
        present_ids = tuple(sorted(
            servo_id for servo_id, is_present in flags.items() if is_present))
        present_ports = {servo_id: int(ports[servo_id]) for servo_id in present_ids}
        return KondoDiscovery(
            present_ids=present_ids,
            ports=present_ports,
            per_port=per_port,
            timed_out_ids=timed_out_ids,
            firmware_version=firmware_version,
            ics_channel_scan_error=ics_channel_scan_error)

    def verify_servos(self, allow_missing: bool = False) -> List[DiagItem]:
        """Run startup servo verification and pin a baseline for runtime checks.

        Presence and absence are judged from ``ServoStruct.flag`` only.
        ``Control.device_port`` never decides whether a servo exists.

        The firmware check runs **first**, and that ordering is load-bearing.
        Every address used below comes from the symbol map, so against the wrong
        firmware this method reads the right number of bytes from the wrong
        place and concludes "these servos were not recognised" -- a confident,
        wrong answer that sends the operator to check id dials and wiring that
        are fine. A version mismatch must be reported as a version mismatch.

        Parameters
        ----------
        allow_missing : bool
            If False (default), configured ids missing at startup raise.
            If True, missing ids are returned as WARN diagnostics instead.
        """
        self.verify_firmware()
        discovery = self.discover()
        expected_ids = [int(actuator.id) for actuator in self.actuators]
        missing, unexpected, _ = compare_discovery(discovery, expected_ids)

        self._startup_discovery = discovery
        self._startup_present_ids = set(discovery.present_ids)
        self._startup_missing_ids = set(missing)
        self._startup_allow_missing = bool(allow_missing)

        items: List[DiagItem] = []

        self.check_rotation_modes()

        cleared = self.clear_feedback_flags()
        if cleared:
            items.append(DiagItem(
                "bus/feedback_flag", DiagLevel.WARN,
                "Cleared ServoStruct.feedback on servo(s) {}: while that byte "
                "is set the board accepts commands and does NOT drive the "
                "servo. Reported rather than done silently, because this "
                "CHANGED the board.".format(
                    ", ".join(str(i) for i in cleared)),
                {"cleared_ids": ", ".join(str(i) for i in cleared)}))

        if missing:
            present_by_port: Dict[int, List[int]] = {port: [] for port in range(1, 7)}
            for present_id, port in discovery.ports.items():
                if 1 <= int(port) <= 6:
                    present_by_port[int(port)].append(int(present_id))

            per_joint = []
            for actuator in self.actuators:
                servo_id = int(actuator.id)
                if servo_id not in missing:
                    continue
                joint_name = self._joint_name(servo_id)
                expected_port = self._configured_port_for(actuator)
                found = sorted(present_by_port.get(expected_port, []))
                per_joint.append(
                    "{} (id {}, config {}, board found {})".format(
                        joint_name, servo_id, self._port_label(expected_port),
                        self._format_ids(found)))

            message = (
                "Configured servos were not recognised at startup: {}. "
                "Check servo id dials, wiring, and the configured ICS UART."
                .format("; ".join(per_joint)))
            values = {
                "missing_ids": self._format_ids(sorted(missing)),
                "found_ids": self._format_ids(sorted(discovery.present_ids)),
            }
            if allow_missing:
                items.append(DiagItem(
                    name="bus/startup_missing",
                    level=DiagLevel.WARN,
                    message=message,
                    values=values))
            else:
                raise RuntimeError(message)

        if unexpected:
            extras = []
            for servo_id in sorted(unexpected):
                port = discovery.ports.get(servo_id)
                extras.append(
                    "id {} on {}".format(servo_id, self._port_label(port)))
            items.append(DiagItem(
                name="bus/startup_unexpected",
                level=DiagLevel.WARN,
                message=(
                    "Board discovered servo ids not listed in config: {}."
                    .format("; ".join(extras))),
                values={"unexpected_ids": self._format_ids(sorted(unexpected))}))

        return items

    def diagnose(self) -> List[DiagItem]:
        """Backend-specific Kondo board and servo health.

        This is designed for a low-rate diagnostics loop, not for the control
        hot path. Each call performs fresh reads and does not reuse prior values
        as if they were measured now.

        The loaded symbol map is the *only* thing that makes mapped addresses
        meaningful. If firmware version cannot be read, or reads but does not
        match the map, this method returns only ``bus/firmware`` and performs
        no mapped struct reads. Returning any mapped-address conclusion in that
        state would be a confident wrong answer.
        """
        items: List[DiagItem] = []

        # bus/firmware -----------------------------------------------------
        firmware_values = {"symbol_map_version": str(self.symbols.version)}
        try:
            reported_version = self.firmware_version()
        except ProtocolError as error:
            firmware_values["error"] = str(error)
            items.append(DiagItem(
                name="bus/firmware",
                level=DiagLevel.ERROR,
                message=(
                    "Could not read board firmware version; check transport "
                    "link and power."),
                values=firmware_values))
            return items

        firmware_values["reported_version"] = reported_version
        try:
            self.symbols.require_version(reported_version)
        except ProtocolError as error:
            items.append(DiagItem(
                name="bus/firmware",
                level=DiagLevel.ERROR,
                message=str(error),
                values=firmware_values))
            return items
        items.append(DiagItem(
            name="bus/firmware",
            level=DiagLevel.OK,
            message="Firmware version matches the loaded symbol map.",
            values=firmware_values))

        # bus/discovery (authoritative presence from ServoStruct.flag) ------
        expected_ids = {int(actuator.id) for actuator in self.actuators}
        flags, flags_error = self._read_servo_slot_vector_safe("flag")
        ports, ports_error = self._read_servo_slot_vector_safe("port")
        present_ids_now: Set[int] = set()
        disappeared: Set[int] = set()
        appeared: Set[int] = set()
        if flags is None:
            discovery_values = {"error": str(flags_error)}
            items.append(DiagItem(
                name="bus/discovery",
                level=DiagLevel.STALE,
                message=(
                    "Could not re-scan ServoStruct.flag this cycle, so "
                    "disappearance cannot be judged."),
                values=discovery_values))
        else:
            present_ids_now = {
                servo_id for servo_id, present in enumerate(flags) if bool(present)
            }
            discovery_values = {
                "present_ids": self._format_ids(sorted(present_ids_now)),
            }
            if self._startup_present_ids is None:
                level = DiagLevel.WARN
                message = (
                    "Startup baseline is not recorded; call verify_servos() "
                    "once so runtime disappearance can be detected.")
            else:
                disappeared = self._startup_present_ids - present_ids_now
                appeared = present_ids_now - self._startup_present_ids
                discovery_values["startup_present_ids"] = self._format_ids(
                    sorted(self._startup_present_ids))
                if disappeared:
                    level = DiagLevel.ERROR
                    missing_joints = [
                        "{}(id {})".format(self._joint_name(servo_id), servo_id)
                        for servo_id in sorted(disappeared)
                    ]
                    message = (
                        "Servos disappeared after startup: {}. Check wiring, "
                        "power, and ICS UART."
                        .format(", ".join(missing_joints)))
                elif appeared:
                    level = DiagLevel.WARN
                    appeared_desc = [
                        ("id {}".format(servo_id)
                         if servo_id not in expected_ids
                         else "{}(id {})".format(self._joint_name(servo_id), servo_id))
                        for servo_id in sorted(appeared)
                    ]
                    message = (
                        "New servo ids appeared after startup: {}. Check "
                        "whether config and hardware ids match."
                        .format(", ".join(appeared_desc)))
                else:
                    level = DiagLevel.OK
                    message = "Servo presence matches the startup baseline."
            if ports_error is not None:
                discovery_values["port_scan_error"] = str(ports_error)
            items.append(DiagItem(
                name="bus/discovery",
                level=level,
                message=message,
                values=discovery_values))

        # servo slots read fresh each call ---------------------------------
        timeout_counts, timeout_error = self._read_servo_slot_vector_safe(
            "timeout_count")
        rxerror_counts, rxerror_error = self._read_servo_slot_vector_safe(
            "rxerror_count")
        pulses, pulse_error = self._read_servo_slot_vector_safe("current_angle")

        current_flag, current_flag_error = self._read_system_slot_safe(
            "servo_current_read_flag")
        temperature_flag, temperature_flag_error = self._read_system_slot_safe(
            "servo_temperature_read_flag")
        current_enabled = (
            None if current_flag is None else bool(int(current_flag[0])))
        temperature_enabled = (
            None if temperature_flag is None else bool(int(temperature_flag[0])))

        current_values: Optional[List[int]] = None
        current_error: Optional[str] = None
        if current_enabled:
            current_values, current_error = self._read_servo_slot_vector_safe(
                "current")
        temperature_values: Optional[List[int]] = None
        temperature_error: Optional[str] = None
        if temperature_enabled:
            temperature_values, temperature_error = self._read_servo_slot_vector_safe(
                "temperature")

        current_missing_expected = expected_ids - present_ids_now if flags is not None else set()
        startup_missing_ids = set(self._startup_missing_ids)

        for actuator in self.actuators:
            servo_id = int(actuator.id)
            joint_name = self._joint_name(servo_id)
            values = {
                "servo_id": str(servo_id),
                "port": (
                    str(int(ports[servo_id])) if ports is not None else "unavailable"),
                "timeout_count": (
                    str(int(timeout_counts[servo_id]))
                    if timeout_counts is not None else "unavailable"),
                "rxerror_count": (
                    str(int(rxerror_counts[servo_id]))
                    if rxerror_counts is not None else "unavailable"),
                "measured_pulse": (
                    str(int(pulses[servo_id])) if pulses is not None else "unavailable"),
            }
            level = DiagLevel.OK
            message_parts = []

            if flags is None:
                level = DiagLevel.STALE
                message_parts.append(
                    "Presence unknown this cycle because ServoStruct.flag did not answer.")
            elif servo_id in current_missing_expected:
                if self._startup_present_ids is None:
                    level = DiagLevel.ERROR
                    message_parts.append(
                        "Servo is absent in the latest authoritative flag scan; "
                        "startup baseline is unavailable, so startup-vs-runtime "
                        "classification is unknown.")
                elif servo_id in disappeared:
                    level = DiagLevel.ERROR
                    message_parts.append(
                        "Servo disappeared after startup (was recognised, now absent).")
                elif servo_id in startup_missing_ids:
                    level = (DiagLevel.WARN if self._startup_allow_missing
                             else DiagLevel.ERROR)
                    message_parts.append(
                        "Configured servo was never recognised at startup.")
                else:
                    level = DiagLevel.ERROR
                    message_parts.append(
                        "Servo is absent in the latest authoritative flag scan.")
            else:
                timeout_increase = 0
                rxerror_increase = 0
                if timeout_counts is not None:
                    timeout_now = int(timeout_counts[servo_id])
                    timeout_prev = self._diag_last_timeout_count.get(servo_id)
                    if timeout_prev is not None and timeout_now > timeout_prev:
                        timeout_increase = timeout_now - timeout_prev
                    self._diag_last_timeout_count[servo_id] = timeout_now
                if rxerror_counts is not None:
                    rxerror_now = int(rxerror_counts[servo_id])
                    rxerror_prev = self._diag_last_rxerror_count.get(servo_id)
                    if rxerror_prev is not None and rxerror_now > rxerror_prev:
                        rxerror_increase = rxerror_now - rxerror_prev
                    self._diag_last_rxerror_count[servo_id] = rxerror_now

                if timeout_increase > 0 or rxerror_increase > 0:
                    level = DiagLevel.WARN
                    if timeout_increase > 0:
                        message_parts.append(
                            "timeout_count increased by {} this cycle."
                            .format(timeout_increase))
                    if rxerror_increase > 0:
                        message_parts.append(
                            "rxerror_count increased by {} this cycle."
                            .format(rxerror_increase))
                else:
                    if timeout_counts is not None and rxerror_counts is not None:
                        timeout_now = int(timeout_counts[servo_id])
                        rxerror_now = int(rxerror_counts[servo_id])
                        if timeout_now > 0 or rxerror_now > 0:
                            message_parts.append(
                                "Counters are nonzero but unchanged since the last check.")
                        else:
                            message_parts.append("Counters stable.")
                    else:
                        message_parts.append(
                            "Counter trend unavailable this cycle due to read error.")

            if pulse_error is not None:
                level = max(level, DiagLevel.STALE)
                message_parts.append(
                    "Measured pulse unavailable this cycle ({})".format(pulse_error))
            if timeout_error is not None:
                level = max(level, DiagLevel.STALE)
                message_parts.append(
                    "timeout_count unavailable this cycle ({})".format(timeout_error))
            if rxerror_error is not None:
                level = max(level, DiagLevel.STALE)
                message_parts.append(
                    "rxerror_count unavailable this cycle ({})".format(rxerror_error))
            if ports_error is not None:
                level = max(level, DiagLevel.STALE)
                message_parts.append(
                    "Port read unavailable this cycle ({})".format(ports_error))

            if current_enabled is False:
                message_parts.append(
                    "Current telemetry is stale by construction (servo_current_read_flag=0).")
            elif current_enabled is True:
                if current_error is not None or current_values is None:
                    message_parts.append(
                        "Current read failed this cycle ({}).".format(current_error))
                else:
                    amps = KondoIcsActuator._decode_current(
                        int(current_values[servo_id]))
                    if amps is None:
                        message_parts.append(
                            "Current value unavailable this cycle (raw current byte is 0).")
                    else:
                        values["current_a"] = "{:.3f}".format(float(amps))
            else:
                message_parts.append(
                    "Current read flag unavailable this cycle ({})."
                    .format(current_flag_error))

            if temperature_enabled is False:
                message_parts.append(
                    "Temperature telemetry is stale by construction "
                    "(servo_temperature_read_flag=0).")
            elif temperature_enabled is True:
                if temperature_error is not None or temperature_values is None:
                    message_parts.append(
                        "Temperature read failed this cycle ({})."
                        .format(temperature_error))
                else:
                    values["temperature_c"] = "{:.1f}".format(
                        float(setting_value_to_celsius(
                            int(temperature_values[servo_id]))))
            else:
                message_parts.append(
                    "Temperature read flag unavailable this cycle ({})."
                    .format(temperature_flag_error))

            items.append(DiagItem(
                name="servo/{}".format(joint_name),
                level=level,
                message=" ".join(message_parts),
                values=values))

        # bus/power --------------------------------------------------------
        battery, battery_error = self._read_system_slot_safe("battery_voltage")
        if battery is None:
            items.append(DiagItem(
                name="bus/power",
                level=DiagLevel.STALE,
                message=(
                    "Battery voltage could not be read this cycle; no voltage "
                    "claim is made."),
                values={"error": str(battery_error)}))
        else:
            voltage_v = float(battery[0])
            values = {"battery_voltage_v": "{:.3f}".format(voltage_v)}
            if self._battery_voltage_warn_below is None:
                items.append(DiagItem(
                    name="bus/power",
                    level=DiagLevel.OK,
                    message=(
                        "Battery voltage measured; no low-voltage threshold is "
                        "configured, so no low-voltage level is claimed."),
                    values=values))
            else:
                threshold = float(self._battery_voltage_warn_below)
                values["warn_below_v"] = "{:.3f}".format(threshold)
                if voltage_v < threshold:
                    items.append(DiagItem(
                        name="bus/power",
                        level=DiagLevel.WARN,
                        message=(
                            "Battery voltage is below the configured warning "
                            "threshold; check supply and load."),
                        values=values))
                else:
                    items.append(DiagItem(
                        name="bus/power",
                        level=DiagLevel.OK,
                        message="Battery voltage is above the configured warning threshold.",
                        values=values))

        # bus/ics_comm_stop -----------------------------------------------
        ics_comm_stop, ics_comm_error = self._read_system_slot_safe("ics_comm_stop")
        if ics_comm_stop is None:
            items.append(DiagItem(
                name="bus/ics_comm_stop",
                level=DiagLevel.STALE,
                message=(
                    "Could not read ics_comm_stop this cycle; UART stop state "
                    "is unknown."),
                values={"error": str(ics_comm_error)}))
        else:
            expected_by_port: Dict[int, List[str]] = {port: [] for port in range(1, 7)}
            for actuator in self.actuators:
                expected_port = self._configured_port_for(actuator)
                if expected_port is None:
                    continue
                expected_by_port[expected_port].append(
                    "{}(id {})".format(self._joint_name(int(actuator.id)),
                                       int(actuator.id)))

            stopped_expected_ports = []
            stopped_unexpected_ports = []
            values = {}
            for port in range(1, 7):
                stop_value = int(ics_comm_stop[port - 1])
                values["J{}_stop".format(port)] = str(stop_value)
                if stop_value == 0:
                    continue
                if expected_by_port[port]:
                    stopped_expected_ports.append(
                        "{} for {}".format(
                            self._port_label(port),
                            ", ".join(expected_by_port[port])))
                else:
                    stopped_unexpected_ports.append(self._port_label(port))

            if stopped_expected_ports:
                level = DiagLevel.ERROR
                message = (
                    "ICS communication is stopped on expected UARTs: {}."
                    .format("; ".join(stopped_expected_ports)))
            elif stopped_unexpected_ports:
                level = DiagLevel.WARN
                message = (
                    "ICS communication is stopped on UARTs with no configured "
                    "joints: {}.".format(", ".join(stopped_unexpected_ports)))
            else:
                level = DiagLevel.OK
                message = "ICS communication is enabled on all UARTs."
            items.append(DiagItem(
                name="bus/ics_comm_stop",
                level=level,
                message=message,
                values=values))

        return items

    # -- servo on / off ----------------------------------------------------

    def release_all(self) -> None:
        """Servo OFF for every joint, in one packet."""
        ids = [a.id for a in self.actuators]
        self._command(build_multi_servo(
            ids, [SERVO_OFF_PULSE] * len(ids), self._velocity_for(self.actuators[0])))
        for servo_id in ids:
            self._servo_on[servo_id] = False

    def hold_all(self) -> None:
        """Servo ON for every joint, each holding wherever it currently is."""
        ids = [a.id for a in self.actuators]
        self._command(build_multi_servo(
            ids, [SERVO_ON_PULSE] * len(ids), self._velocity_for(self.actuators[0])))
        for servo_id in ids:
            self._servo_on[servo_id] = True

    def servo_on_states(self) -> Dict[int, bool]:
        """Last servo on/off state this bus commanded, per servo id.

        This is what the driver *asked for*. What the board actually holds is in
        the ``ref_angle`` slot -- add it to ``state_slots`` to observe it rather
        than assume it.
        """
        return dict(self._servo_on)

    # -- optional telemetry switches ---------------------------------------

    def set_current_reading(self, enable: bool) -> None:
        """Turn the board's servo-current polling on or off.

        Off by default in firmware because polling costs ICS bus bandwidth that
        would otherwise go to motion. While it is off, the current byte stays 0
        and decodes to ``None`` -- not to 0 A.
        """
        self._command(system_slot_write(
            self._system_base, "servo_current_read_flag", [1 if enable else 0]))

    def set_temperature_reading(self, enable: bool) -> None:
        """Turn the board's servo-temperature polling on or off."""
        self._command(system_slot_write(
            self._system_base, "servo_temperature_read_flag",
            [1 if enable else 0]))

    def set_ics_comm(self, enable: bool) -> None:
        """Start or stop the board's ICS communication on all six UARTs.

        Stopping it is what lets a maintenance tool talk to a servo directly
        without the board's loop contending for the wire. It also means the
        servos stop being driven -- so it is not a substitute for servo off.
        """
        value = 0 if enable else 1
        self._command(system_slot_write(
            self._system_base, "ics_comm_stop", [value] * 6))
