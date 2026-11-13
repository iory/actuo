"""Serial (TTL half-duplex) transport for Feetech- and DYNAMIXEL-style buses.

Moves opaque packet bytes over a serial port and knows nothing about registers
or angles -- the actuator owns those. Unlike CAN servo mode (which streams status
unsolicited), these buses are strictly request/response: you send a READ and the
addressed servo answers, one at a time on a shared half-duplex wire. So the
request/response pairing lives in :meth:`transact`; ``recv`` frames a single
inbound status packet for callers that already know one is coming.

Two framings are supported, selected by the ``protocol`` parameter: ``FF FF ID
LEN ..`` (Feetech SCS, DYNAMIXEL 1.0) and ``FF FF FD 00 ID LEN_L LEN_H ..``
(DYNAMIXEL 2.0). The transport needs the header and length field *only* to know
where a packet ends; it passes the whole packet up as an opaque
``Frame.payload`` and never parses the register data -- that is the actuator's
job. Adding a framing is not the same as adding protocol knowledge: no address,
unit or model constant lives in this file.

The same physical adapter serves both -- a DXHUB (FT232H + isolated TTL/RS-485
drivers) is just a serial port here, because it contains no gateway MCU. Its
one wire-level quirk worth knowing is the FTDI latency timer: the default 16 ms
turnaround caps a request/response bus far below its baud rate, and it must be
measured rather than assumed.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

import serial

from actuo.core.contracts import CapabilityError, Frame, Transport
from actuo.core.registry import register_transport
from actuo.drivers.transports.exclusive_serial import open_exclusive

_HEADER = b"\xff\xff"          # Feetech / DYNAMIXEL 1.0
_HEADER_2 = b"\xff\xff\xfd\x00"  # DYNAMIXEL 2.0 (the reserved 00 is part of it)


@register_transport("serial")
class SerialTransport(Transport):
    """A half-duplex serial bus (e.g. a Waveshare USB-TTL adapter).

    Parameters
    ----------
    channel : str
        Serial device path, e.g. ``/dev/ttyACM0`` (Linux) or a COM port. Named
        ``channel`` to match the CAN transport, so config overrides are uniform.
    bitrate : int
        Baud rate, e.g. ``1000000`` for the Amazing Hand's SCS0009 bus.
    timeout : float
        Default per-read timeout in seconds; ``transact`` overrides per call.
    protocol : int
        Which framing to use when finding a packet boundary: ``1`` for the
        ``FF FF ID LEN`` form (Feetech SCS, DYNAMIXEL 1.0) or ``2`` for
        DYNAMIXEL 2.0's ``FF FF FD 00 ID LEN_L LEN_H``.

        Framing genuinely belongs here: a transport cannot hand up "one frame"
        without knowing where a frame ends. What stays out is what the bytes
        *mean* -- no register, unit or model knowledge crosses this boundary.
    discard_echo : bool
        Drop an inbound packet that is byte-identical to the one just sent.

        Some half-duplex adapters loop the transmitter back into the receiver.
        On a protocol 2.0 bus the echo is rejected anyway (a status packet is
        marked by instruction ``0x55``), but on a 1.0 bus an instruction packet
        and a status packet have the same shape, so an echoed command decodes as
        a plausible status. This stack has already been bitten by exactly that
        on CAN, where a driver diagnosed an "encoder fault" from a frame it had
        transmitted itself. Default off, because a correct adapter does not echo
        and silently swallowing traffic would hide a real wiring fault;
        ``actuo-dxl-probe`` reports whether the adapter needs it.
    """

    def __init__(self, channel: str, bitrate: int = 1_000_000,
                 timeout: float = 0.05, protocol: int = 1,
                 discard_echo: bool = False) -> None:
        self.channel = channel
        self.bitrate = int(bitrate)
        self.timeout = float(timeout)
        self.protocol = int(protocol)
        if self.protocol not in (1, 2):
            raise ValueError(
                "serial framing protocol must be 1 or 2, got {!r}".format(protocol))
        self.discard_echo = bool(discard_echo)
        self._last_sent: Optional[bytes] = None
        # Bytes read but not yet consumed, kept ACROSS calls. A batch read is
        # one write followed by N receives, and the N status packets arrive
        # back to back -- the first read pulls several of them out of the OS
        # buffer at once. A per-call buffer would return the first and drop the
        # rest, so the bus would see one servo answer and N-1 "time out" at
        # exactly the moment batching started working.
        self._rx = bytearray()
        self._serial: Optional[serial.Serial] = None

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

    def _require_open(self) -> serial.Serial:
        if self._serial is None:
            raise RuntimeError("serial transport is not open")
        return self._serial

    def reset_input(self) -> None:
        """Drop bytes already buffered from before the next request.

        Not part of the ``Transport`` contract -- it is a serial-line detail --
        but a batch read needs it: SYNC_READ writes once and then collects N
        replies, so a stale packet left over from the previous cycle would be
        matched to this cycle's request and reported as fresh state.
        """
        self._rx.clear()
        self._require_open().reset_input_buffer()

    def write(self, frames: Sequence[Frame]) -> None:
        port = self._require_open()
        for frame in frames:
            port.write(frame.payload)
        port.flush()
        if frames:
            self._last_sent = frames[-1].payload

    @property
    def _header(self) -> bytes:
        return _HEADER if self.protocol == 1 else _HEADER_2

    def _packet_total(self, buffer: bytes) -> Optional[int]:
        """Total packet length once enough header bytes have arrived."""
        if self.protocol == 1:
            return buffer[3] + 4 if len(buffer) >= 4 else None
        if len(buffer) < 7:
            return None
        return (buffer[5] | (buffer[6] << 8)) + 7

    def _read_packet(self, timeout: float) -> Optional[bytes]:
        """Frame and return one packet, or None within ``timeout``.

        Scans the byte stream for the protocol's header, reads the declared
        length, then the remainder. Re-syncs on garbage rather than raising: a
        serial line drops bytes, and a lost packet is a missed read, not a fault.
        """
        port = self._require_open()
        deadline = time.monotonic() + timeout
        buffer = self._rx
        header = self._header
        while True:
            # Drain what is already buffered before waiting on the port again:
            # a burst of status packets arrives faster than this loop iterates,
            # so several may already be sitting here.
            while True:
                start = buffer.find(header)
                if start < 0:
                    # Keep a partial header that might complete on the next read.
                    keep = len(header) - 1
                    del buffer[:max(0, len(buffer) - keep)]
                    break
                if start > 0:
                    del buffer[:start]
                total = self._packet_total(bytes(buffer))
                if total is None or len(buffer) < total:
                    break
                packet = bytes(buffer[:total])
                del buffer[:total]
                if self.discard_echo and packet == self._last_sent:
                    continue  # the adapter looped our own transmission back
                return packet

            if time.monotonic() >= deadline:
                return None
            waiting = port.in_waiting
            chunk = port.read(waiting if waiting > 0 else 1)
            if chunk:
                buffer.extend(chunk)

    def recv(self, timeout: float) -> Optional[Frame]:
        packet = self._read_packet(timeout)
        if packet is None:
            return None
        # The id sits after the header in both framings.
        return Frame(payload=packet, address=packet[len(self._header)])

    def transact(self, frames: Sequence[Frame],
                 timeout: float) -> List[Optional[Frame]]:
        """Send each request and collect its reply, one at a time.

        The wire is half-duplex and single-drop-per-reply, so requests are
        strictly serialized: write one, read its answer, then the next. A request
        that does not answer in ``timeout`` yields ``None`` in that slot rather
        than stalling the rest.
        """
        port = self._require_open()
        replies: List[Optional[Frame]] = []
        for frame in frames:
            self._rx.clear()
            port.reset_input_buffer()      # drop any stale bytes before asking
            port.write(frame.payload)
            port.flush()
            self._last_sent = frame.payload   # so discard_echo can recognise it
            replies.append(self.recv(timeout))
        return replies
