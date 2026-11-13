"""The datagrams between a host and a UART relay (a WiFi microcontroller).

A relay puts a serial device on a network: the host sends it a request packet
in a UDP datagram, the relay writes it to its UART, reads the device's answer
and sends that back in one datagram. The relay knows ONE thing about the bytes
-- where a packet ends: the device's packets are length-prefixed (first byte =
the whole packet's length, Kondo RCB-4mini / ARM-H7 framing), which is a
property of the framing, not of what the packets mean. It knows nothing of
servos, addresses or opcodes; those stay in the bus on the host, so that
moving bytes and knowing what they mean remain separate layers.

Implemented by the Arduino relay sketch (Arduino UNO R4 WiFi,
``firmware/actuo_uart_relay/``) and, for testing without one, by :mod:`actuo.drivers.uart_relay`
(``actuo-uart-relay-emulator``). Keep the three in step.

Every datagram starts with ``type (u8), seq (u16 LE)``. Host to relay:

=====  ===============================================  =====================
type   rest                                             answer
=====  ===============================================  =====================
``F``  timeout_ms (u16), one framed packet              ``R``: the reply
``B``  baud (u32): reopen the UART (8E1) at this rate   ``A``
``S``  silence_ms (u16), framed packets back to back    ``A``
``P``  (nothing)                                        ``A``
=====  ===============================================  =====================

Relay to host:

* ``R``: status (u8), failsafes (u16), the reply packet (absent unless status
  is OK).
* ``A``: status (u8), failsafes (u16).

``failsafes`` counts the times the relay has sent the failsafe packets since
it booted: every datagram reports it, so a host that comes back learns the
relay acted while it was gone.

The failsafe (``S``): when no datagram has come from the host for
``silence_ms``, the relay writes the packets to the device, one by one,
reading and dropping each answer, once per silence. ``silence_ms`` 0 or no
packets disarms it.
"""

from __future__ import annotations

import struct
from typing import List, Sequence, Tuple

FORWARD = ord("F")
BAUD = ord("B")
FAILSAFE = ord("S")
PING = ord("P")
REPLY = ord("R")
ACK = ord("A")

STATUS_OK = 0
#: The device did not complete a packet within the request's timeout.
STATUS_TIMEOUT = 1
#: The relay could not parse the request.
STATUS_BAD_REQUEST = 2

STATUS_NAMES = {STATUS_OK: "ok", STATUS_TIMEOUT: "device timeout",
                STATUS_BAD_REQUEST: "bad request"}

#: Bytes before a reply packet: type, seq, status, failsafes.
REPLY_HEADER = struct.Struct("<BHBH")
#: Bytes before a request's payload: type, seq.
REQUEST_HEADER = struct.Struct("<BH")

#: The relay's buffer for one packet; the length byte caps a packet at 255.
MAX_PACKET = 255
#: Failsafe packets the relay holds at most (``S`` past this is refused).
MAX_FAILSAFE_BYTES = 512


def forward(seq: int, timeout_s: float, packet: bytes) -> bytes:
    """An ``F`` datagram."""
    return (REQUEST_HEADER.pack(FORWARD, seq & 0xFFFF)
            + struct.pack("<H", _ms(timeout_s)) + bytes(packet))


def baud(seq: int, rate: int) -> bytes:
    """A ``B`` datagram."""
    return REQUEST_HEADER.pack(BAUD, seq & 0xFFFF) + struct.pack("<I", int(rate))


def failsafe(seq: int, silence_s: float, packets: Sequence[bytes]) -> bytes:
    """An ``S`` datagram.

    Raises
    ------
    ValueError
        If the packets do not fit the relay's buffer.
    """
    body = b"".join(bytes(p) for p in packets)
    if len(body) > MAX_FAILSAFE_BYTES:
        raise ValueError("failsafe packets are {} bytes; the relay holds {}"
                         .format(len(body), MAX_FAILSAFE_BYTES))
    return (REQUEST_HEADER.pack(FAILSAFE, seq & 0xFFFF)
            + struct.pack("<H", _ms(silence_s)) + body)


def ping(seq: int) -> bytes:
    """A ``P`` datagram."""
    return REQUEST_HEADER.pack(PING, seq & 0xFFFF)


def parse_answer(datagram: bytes) -> Tuple[int, int, int, int, bytes]:
    """An ``R`` or ``A`` datagram.

    Returns
    -------
    kind : int
        :data:`REPLY` or :data:`ACK`.
    seq, status, failsafes : int
    packet : bytes
        The device's reply (``R`` with status OK), else empty.

    Raises
    ------
    ValueError
        For a datagram that is neither.
    """
    if len(datagram) < REPLY_HEADER.size:
        raise ValueError("short datagram ({} bytes)".format(len(datagram)))
    kind, seq, status, failsafes = REPLY_HEADER.unpack_from(datagram)
    if kind not in (REPLY, ACK):
        raise ValueError("not an answer: type {:#04x}".format(kind))
    return kind, seq, status, failsafes, bytes(datagram[REPLY_HEADER.size:])


def split_packets(body: bytes) -> List[bytes]:
    """Length-prefixed packets back to back, one by one.

    Raises
    ------
    ValueError
        If a length byte is 0 or runs past the end.
    """
    packets = []
    i = 0
    while i < len(body):
        n = body[i]
        if n == 0 or i + n > len(body):
            raise ValueError("bad packet length {} at byte {}".format(n, i))
        packets.append(bytes(body[i:i + n]))
        i += n
    return packets


def _ms(seconds: float) -> int:
    return max(0, min(0xFFFF, int(round(seconds * 1000.0))))
