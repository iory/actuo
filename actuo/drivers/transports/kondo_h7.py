"""Serial transport to the Kondo ARM-H7 control board.

Moves opaque packet bodies to and from the board and knows nothing about
servos, angles, memory addresses, or opcodes -- the bus and the actuator own
those (see ``actuo/drivers/kondo_h7_protocol.py``). All this layer does is add
the length byte and checksum on the way out, and strip them on the way in.

The board is strictly request/response: every packet, including a servo move
command, is answered. That makes :meth:`transact` the natural primitive, and it
also makes *consuming the reply mandatory* -- skipping it leaves the answer in
the input buffer where it will be mistaken for the next reply, and every read
after that returns the previous cycle's data. So ``write`` here still reads and
discards the reply rather than pretending the board is fire-and-forget.

Provenance: ``actuo/drivers/references.md``, section "Kondo ARM-H7".
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence

import serial

from actuo.core.contracts import Frame, Transport
from actuo.core.registry import register_transport

from actuo.drivers.kondo_h7_protocol import ChecksumError, ProtocolError, frame, unframe
from actuo.drivers.transports.exclusive_serial import open_exclusive

#: The H7 firmware's host-side UART runs at 1 Mbit/s.
DEFAULT_BITRATE = 1_000_000


@register_transport("kondo_h7")
class KondoH7Transport(Transport):
    """Length-prefixed packet transport over a USB serial link to the board.

    Parameters
    ----------
    channel : str
        Serial device path, e.g. ``/dev/ttyUSB1`` or ``/dev/tty.usbserial-*``.
        Named ``channel`` to match the CAN and Feetech transports so config
        overrides stay uniform across buses.
    bitrate : int
        Baud rate. Defaults to the firmware's 1 Mbit/s.
    timeout : float
        Default per-packet read timeout in seconds.
    """

    def __init__(self, channel: str, bitrate: int = DEFAULT_BITRATE,
                 timeout: float = 0.1) -> None:
        self.channel = channel
        self.bitrate = int(bitrate)
        self.timeout = float(timeout)
        self._serial: Optional[serial.Serial] = None
        self._checksum_errors = 0
        self._framing_errors = 0

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        if self._serial is not None:
            return
        self._serial = open_exclusive(
            self.channel, self.bitrate, timeout=self.timeout)
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None

    @property
    def is_open(self) -> bool:
        return self._serial is not None and self._serial.is_open

    @property
    def latency_timer_ms(self) -> Optional[float]:
        """The adapter's USB latency timer in ms, or ``None`` if not knowable.

        An FTDI chip holds a short reply until this many milliseconds of line
        idle have passed, and ships with **16 ms**. That is a floor on the
        round trip, and a control loop whose per-cycle read budget is below it
        gets no reply, ever -- not a slow robot but a frozen one, publishing an
        unchanging pose while every counter says the loop is keeping up.
        Measured here 2026-08-01 at 100 Hz: 665 cycles, 665 timeouts, and nine
        joint angles that did not move by a single bit in six seconds.

        Linux exposes it per port under sysfs. Absent (a CH340, a non-Linux
        host) means "no floor known", which is reported as ``None`` rather than
        as zero -- an unknown floor is not the same claim as no floor.
        """
        name = os.path.basename(os.path.realpath(self.channel))
        path = "/sys/bus/usb-serial/devices/{}/latency_timer".format(name)
        try:
            with open(path) as handle:
                return float(handle.read().strip())
        except (OSError, ValueError):
            return None

    def _require_open(self) -> serial.Serial:
        if self._serial is None:
            # A recognisable type, not a bare RuntimeError: this is what a
            # cycle hits after a reopen attempt failed, and recovery has to be
            # able to tell it apart from a genuine bug (actuo.drivers.link).
            from actuo.drivers.link import TransportNotOpen
            raise TransportNotOpen("kondo_h7 transport is not open")
        return self._serial

    # -- counters ----------------------------------------------------------

    @property
    def checksum_errors(self) -> int:
        """Replies refused because their checksum did not match.

        The board checksums its replies (measured on an RCB-4mini, see
        ``rcb4/protocol/references.md``), so ``unframe`` raises on a mismatch
        and the reply is dropped like a lost one. A count that grows means a
        noisy line, never data taken from a corrupted reply.
        """
        return self._checksum_errors

    @property
    def framing_errors(self) -> int:
        """Replies whose length byte disagreed with the bytes received."""
        return self._framing_errors

    # -- Transport contract ------------------------------------------------

    def write(self, frames: Sequence[Frame]) -> None:
        """Send each packet and drain its reply.

        The reply is discarded, not ignored: the board answers everything, and
        an unread answer desynchronises every subsequent read.
        """
        self.transact(frames, self.timeout)

    def recv(self, timeout: float) -> Optional[Frame]:
        """Read one complete reply packet, or ``None`` if none arrived in time."""
        port = self._require_open()
        deadline = time.monotonic() + timeout
        buffer = bytearray()
        while time.monotonic() < deadline:
            waiting = port.in_waiting
            chunk = port.read(waiting if waiting > 0 else 1)
            if chunk:
                buffer.extend(chunk)
            # The first byte is the packet length, so the packet is complete
            # exactly when as many bytes have arrived as it declared.
            if buffer and len(buffer) >= buffer[0]:
                packet = bytes(buffer[:buffer[0]])
                del buffer[:buffer[0]]
                try:
                    body = unframe(packet)
                except ChecksumError:
                    self._checksum_errors += 1
                    return None
                except ProtocolError:
                    self._framing_errors += 1
                    return None
                return Frame(payload=body)
        return None

    def transact(self, frames: Sequence[Frame],
                 timeout: float) -> List[Optional[Frame]]:
        """Send each packet and collect its reply, strictly one at a time.

        A packet that does not answer within ``timeout`` yields ``None`` in that
        slot. The input buffer is flushed before each request so a late reply
        from a previous cycle cannot be mistaken for this one's.
        """
        port = self._require_open()
        replies: List[Optional[Frame]] = []
        for item in frames:
            port.reset_input_buffer()
            port.write(frame(item.payload))
            port.flush()
            replies.append(self.recv(timeout))
        return replies
