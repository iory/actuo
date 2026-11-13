"""An RCB-4mini behind a pseudo-terminal, for running the real stack without one.

The real bus opens a serial port, so to exercise the whole real path --
transport framing, :class:`~actuo.drivers.buses.kondo_rcb4_bus.KondoRcb4Bus`
cold start, and whatever runs above it (a bus-servicing process, shared
memory, a ros2_control hardware component, the controllers) -- with no board on the desk, something has to answer on a serial port the way
the board does. This is that: a pty whose far end speaks the RCB-4mini
protocol, with servos that move toward their targets.

What it models, and from where
------------------------------
The board RAM and the replies are those of ``FakeRcb4`` in
``tests/test_kondo_rcb4_bus.py``, which encodes what an RCB-4mini was seen to do
(references.md, "Kondo RCB-4mini"):

* a RAM read (``0x00``/``0x20``) answers the bytes, a RAM write (``0x00``/``0x02``)
  stores them and acknowledges;
* a servo's pulse is written into its record only while its ICS number is
  assigned to that record and the board's ICS rate matches the servos';
* the batch move (``0x10``) is answered with an acknowledge.

What it adds, because the tests never needed it: the servos MOVE. A target in
a batch move is reached linearly over the packet's move byte times the
output cycle (15 ms flat), as the board interpolates; ``SERVO_OFF_PULSE``
leaves the servo where it is and limp; ``SERVO_ON_PULSE`` holds it there.

What it does NOT model, so a pass here says nothing about these
---------------------------------------------------------------
* **Bit rates.** A pty has none. The emulator answers at whatever rate the
  host opened the port, so the bus's connect-at-115200-then-switch sequence
  runs but cannot fail here.
* **Electrical timing**, the FTDI latency timer, a wedged adapter.
* **Servo dynamics**: no load, no gravity, no stall, perfect tracking.
* **Continuous-rotation servos**: a wheel's speed command is taken as a
  position like any other, so wheel odometry from this is meaningless.

Servos can be pulled out and plugged back in while it runs
(:meth:`Rcb4Emulator.unplug`, :meth:`Rcb4Emulator.plug`): an unplugged servo
stops writing its record, so a pulse the host clears stays zero, as on the
board.
"""

from __future__ import annotations

import os
import pty
import select
import struct
import threading
import time
import tty
from typing import Dict, Iterable, Optional

from rcb4.protocol import rcb4mini as protocol
from actuo.drivers.actuators.kondo_ics import (
    CENTER_PULSE, SERVO_OFF_PULSE, SERVO_ON_PULSE)
from actuo.drivers.kondo_h7_protocol import (
    OP_MULTI_SERVO_SINGLE_VELOCITY, SERVO_ID_BITMAP_BYTES, checksum)

#: Config word of an RCB-4mini without a HeartToHeart project (references.md).
FACTORY_CONFIG = 0x8313
#: Battery ADC counts reported: 7.4 V, a charged 2-cell pack.
BATTERY_COUNTS = int(round(7.4 / protocol.BATTERY_VOLTS_PER_COUNT))
#: Output cycle of a flat board; one move-byte unit [s].
FRAME_S = 0.015


class Rcb4Emulator:
    """Board RAM, servos and the reply to each packet body.

    Parameters
    ----------
    servos : iterable of int
        ICS numbers that have a servo wired to them.
    servo_ics_bitrate : int
        The servos' ICS rate; they answer only when the board's matches.
    start_pulse : int
        Where every servo is at power-up.
    clock : callable
        Monotonic seconds; replaceable for tests.
    trace : iterable of int
        ICS numbers whose commanded pulse is reported through ``log`` every
        time it changes (e.g. the wheels, to see a failsafe stop them).
    log : callable
    """

    def __init__(self, servos: Iterable[int], servo_ics_bitrate: int = 1250000,
                 start_pulse: int = CENTER_PULSE, clock=time.monotonic,
                 trace: Iterable[int] = (), log=print) -> None:
        self._trace = {int(s) for s in trace}
        self._commanded: Dict[int, int] = {}
        self._log = log
        self.ram = bytearray(0x400)
        self._put(protocol.CONFIG_ADDRESS, struct.pack("<H", FACTORY_CONFIG))
        self._put(protocol.BATTERY_ADC_ADDRESS,
                  struct.pack("<H", BATTERY_COUNTS))
        for slot in range(protocol.ICS_SLOT_COUNT):
            self._put(protocol.ICS_ASSIGN_ADDRESS + 2 * slot,
                      struct.pack("<H", protocol.ICS_UNASSIGNED))
        self.servo_ics_code = protocol.ICS_BITRATE_CODES[servo_ics_bitrate]
        self._clock = clock
        now = clock()
        #: ICS number -> (from pulse, to pulse, start time, duration).
        self._motion: Dict[int, tuple] = {
            int(s): (start_pulse, start_pulse, now, 0.0) for s in servos}
        #: ICS number -> torque on.
        self.enabled: Dict[int, bool] = {s: False for s in self._motion}
        # plug/unplug come from another thread than the one answering.
        self._lock = threading.Lock()

    # -- wiring ----------------------------------------------------------------

    def unplug(self, slot: int) -> None:
        """Pull the servo at ICS number ``slot`` out."""
        with self._lock:
            del self._motion[int(slot)]
            del self.enabled[int(slot)]

    def plug(self, slot: int, pulse: int = CENTER_PULSE) -> None:
        """Plug a servo, limp at ``pulse``, into ICS number ``slot``."""
        with self._lock:
            if int(slot) in self._motion:
                raise ValueError("ICS {} already has a servo".format(slot))
            self._motion[int(slot)] = (int(pulse), int(pulse), self._clock(), 0.0)
            self.enabled[int(slot)] = False

    # -- RAM -------------------------------------------------------------------

    def _put(self, address: int, data: bytes) -> None:
        self.ram[address:address + len(data)] = data

    def _word(self, address: int) -> int:
        return struct.unpack("<H", self.ram[address:address + 2])[0]

    # -- servos ----------------------------------------------------------------

    def pulse(self, slot: int) -> int:
        """Where servo ``slot`` is now."""
        start, end, t0, duration = self._motion[slot]
        if duration <= 0.0:
            return end
        alpha = min(1.0, (self._clock() - t0) / duration)
        return int(round(start + (end - start) * alpha))

    def _command(self, slot: int, pulse: int, move_byte: int) -> None:
        if slot not in self._motion:
            return
        # Every change of what the host COMMANDS, whether or not this
        # emulator acts on it: a wheel is switched on by a RAM write it does
        # not model, so its target never moves here.
        if slot in self._trace and pulse != self._commanded.get(slot):
            self._commanded[slot] = pulse
            self._log("trace {:.3f} ICS {} commanded {}{}".format(
                time.monotonic(), slot, pulse,
                "" if self.enabled[slot] else " (servo off here)"))
        now_pulse = self.pulse(slot)
        if pulse == SERVO_OFF_PULSE:
            self.enabled[slot] = False
            self._motion[slot] = (now_pulse, now_pulse, self._clock(), 0.0)
        elif pulse == SERVO_ON_PULSE:
            self.enabled[slot] = True
            self._motion[slot] = (now_pulse, now_pulse, self._clock(), 0.0)
        elif self.enabled[slot]:
            self._motion[slot] = (now_pulse, int(pulse), self._clock(),
                                  move_byte * FRAME_S)

    def _ics_cycle(self) -> None:
        """What the board's servo loop leaves in RAM between host packets."""
        ics_code = ((self._word(protocol.CONFIG_ADDRESS)
                     & protocol.ICS_BITRATE_MASK) >> protocol.ICS_BITRATE_SHIFT)
        for slot in self._motion:
            record = protocol.servo_record_address(slot)
            assigned = (self._word(protocol.ICS_ASSIGN_ADDRESS + 2 * slot)
                        == record)
            if assigned and ics_code == self.servo_ics_code:
                self._put(record + 4, struct.pack("<H", self.pulse(slot)))
                self._put(record + 6, struct.pack(
                    "<H", self._motion[slot][1]))

    # -- protocol --------------------------------------------------------------

    def reply(self, body: bytes) -> Optional[bytes]:
        """The reply body to one request body, or None for no answer."""
        with self._lock:
            return self._reply(body)

    def _reply(self, body: bytes) -> Optional[bytes]:
        self._ics_cycle()
        if body == protocol.build_ack():
            return bytes([0xFE, protocol.ACK_BYTE])
        if body == protocol.build_version():
            return protocol.FIRMWARE_VERSION_REPLY
        if body[:2] == bytes([protocol.OP_MOVE, protocol.SUB_RAM_TO_COM]):
            address = body[5] | body[6] << 8
            size = body[7]
            return bytes([0x00]) + bytes(self.ram[address:address + size])
        if body[:2] == bytes([protocol.OP_MOVE, protocol.SUB_COM_TO_RAM]):
            address = body[2] | body[3] << 8
            self._put(address, body[5:])
            return bytes([0x00, protocol.ACK_BYTE])
        if body[0] == OP_MULTI_SERVO_SINGLE_VELOCITY:
            bitmap = int.from_bytes(body[1:1 + SERVO_ID_BITMAP_BYTES], "little")
            move_byte = body[1 + SERVO_ID_BITMAP_BYTES]
            pulses = body[2 + SERVO_ID_BITMAP_BYTES:]
            slots = [i for i in range(8 * SERVO_ID_BITMAP_BYTES)
                     if bitmap >> i & 1]
            for k, slot in enumerate(slots):
                pulse = struct.unpack_from("<H", pulses, 2 * k)[0]
                self._command(slot, pulse, move_byte)
            return bytes([body[0], protocol.ACK_BYTE])
        return bytes([body[0], protocol.ACK_BYTE])


def _frame(body: bytes) -> bytes:
    head = bytes([len(body) + 2]) + bytes(body)
    return head + bytes([checksum(head)])


def serve(emulator: Rcb4Emulator, link: str, log=print) -> None:
    """Answer on a pty until interrupted; ``link`` is a symlink to its device.

    Parameters
    ----------
    emulator : Rcb4Emulator
    link : str
        Path the host opens, e.g. ``/tmp/rcb4_emulator``. Replaced if present.
    log : callable
        Where to say what is happening.
    """
    master, slave = pty.openpty()
    tty.setraw(master)
    name = os.ttyname(slave)
    if os.path.lexists(link):
        os.unlink(link)
    os.symlink(name, link)
    log("RCB-4mini emulator on {} -> {} (servos at ICS numbers {})".format(
        link, name, sorted(emulator._motion)))
    buffer = bytearray()
    packets = 0
    try:
        while True:
            ready, _, _ = select.select([master], [], [], 1.0)
            if not ready:
                continue
            try:
                buffer += os.read(master, 4096)
            except OSError:
                # The host closed the port (it reopens to switch rates).
                time.sleep(0.01)
                continue
            while buffer and len(buffer) >= buffer[0]:
                length = buffer[0]
                if length < 2:
                    buffer.clear()
                    break
                packet = bytes(buffer[:length])
                del buffer[:length]
                answer = emulator.reply(packet[1:-1])
                packets += 1
                if answer is not None:
                    os.write(master, _frame(answer))
    finally:
        log("RCB-4mini emulator: {} packets answered".format(packets))
        if os.path.islink(link):
            os.unlink(link)
        os.close(master)
        os.close(slave)
