"""The UART relay, in Python, on the host: the Arduino's job without one.

The relay itself runs on an Arduino UNO R4 WiFi (the sketch is
``firmware/actuo_uart_relay/``). This is the same logic, step for step (same datagrams,
:mod:`actuo.drivers.uart_relay_protocol`; same failsafe rule), with the UART
being a serial device the host opens -- the RCB-4mini emulator's pty -- so
the WiFi path runs end to end on a machine with no board and no Arduino:

    actuo-kondo-rcb4-emulator --link /tmp/rcb4 ...     # the board
    actuo-uart-relay-emulator --serial /tmp/rcb4       # the Arduino
    # then open a KondoRcb4Bus on a kondo_rcb4_udp transport pointed at it

And it can be a bad network on purpose, which a real one is only sometimes:
``loss`` drops each datagram (either way) with that probability, ``delay``
holds each answer back, and while ``cut_file`` exists every datagram is
dropped -- the WiFi gone -- so the failsafe fires.

What it does NOT tell you about the Arduino: its timing (a 48 MHz M4 and the
ESP32-S3 bridge between it and the radio), its RAM, whether its UART keeps up
with the board. Those can only be checked on real hardware and are not
verified by this emulator.
"""

from __future__ import annotations

import os
import random
import socket
import struct
import time
from typing import List, Optional

from actuo.drivers import uart_relay_protocol as relay


class UartRelay:
    """One relay: a UDP port in front of one UART.

    Parameters
    ----------
    uart : serial.Serial-like
        ``write``, ``read``, ``in_waiting``, ``reset_input_buffer``,
        ``close``, ``open``, ``baudrate``.
    port : int
    bind : str
    loss : float
        Probability of dropping each datagram, in and out.
    delay : float
        Seconds each answer is held back.
    cut_file : str or None
        While this path exists, nothing gets through.
    rng : random.Random or None
    log : callable
    """

    def __init__(self, uart, port: int, bind: str = "0.0.0.0",
                 loss: float = 0.0, delay: float = 0.0,
                 cut_file: Optional[str] = None, rng=None, log=print) -> None:
        self.uart = uart
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((bind, port))
        self.loss = float(loss)
        self.delay = float(delay)
        self.cut_file = cut_file
        self.rng = rng or random.Random(0)
        self.log = log
        self.failsafe_packets: List[bytes] = []
        self.silence = 0.0
        self.failsafes = 0
        self._fired = False
        self._last_heard = time.monotonic()
        self.stats = {"in": 0, "dropped_in": 0, "out": 0, "dropped_out": 0,
                      "device_timeouts": 0}

    def _cut(self) -> bool:
        return self.cut_file is not None and os.path.exists(self.cut_file)

    def _read_packet(self, timeout: float) -> Optional[bytes]:
        """One length-prefixed packet from the UART, or None by ``timeout``."""
        deadline = time.monotonic() + timeout
        buffer = bytearray()
        while time.monotonic() < deadline:
            chunk = self.uart.read(max(1, self.uart.in_waiting))
            if chunk:
                buffer += chunk
                if buffer[0] == 0:
                    return None
                if len(buffer) >= buffer[0]:
                    return bytes(buffer[:buffer[0]])
        return None

    def _transact(self, packet: bytes, timeout: float) -> Optional[bytes]:
        self.uart.reset_input_buffer()
        self.uart.write(packet)
        return self._read_packet(timeout)

    def _answer(self, address, kind: int, seq: int, status: int,
                packet: bytes = b"") -> None:
        if self.delay > 0.0:
            time.sleep(self.delay)
        if self._cut() or self.rng.random() < self.loss:
            self.stats["dropped_out"] += 1
            return
        self.sock.sendto(relay.REPLY_HEADER.pack(
            kind, seq, status, self.failsafes & 0xFFFF) + packet, address)
        self.stats["out"] += 1

    def handle(self, datagram: bytes, address) -> None:
        """One datagram from the host."""
        if len(datagram) < relay.REQUEST_HEADER.size:
            return
        kind, seq = relay.REQUEST_HEADER.unpack_from(datagram)
        body = datagram[relay.REQUEST_HEADER.size:]
        self._last_heard = time.monotonic()
        self._fired = False
        if kind == relay.FORWARD and len(body) >= 3:
            timeout_ms = struct.unpack_from("<H", body)[0]
            reply = self._transact(body[2:], timeout_ms / 1000.0)
            if reply is None:
                self.stats["device_timeouts"] += 1
                self._answer(address, relay.REPLY, seq, relay.STATUS_TIMEOUT)
            else:
                self._answer(address, relay.REPLY, seq, relay.STATUS_OK, reply)
        elif kind == relay.BAUD and len(body) == 4:
            rate = struct.unpack("<I", body)[0]
            if rate != self.uart.baudrate:
                # Closed first, as the sketch does (Serial1.end / begin).
                try:
                    self.uart.close()
                    self.uart.baudrate = rate
                    self.uart.open()
                except Exception as error:  # noqa: BLE001 -- answered, logged
                    # A pty refuses a second open with parity set (EINVAL):
                    # it has no bit rate to change. Said, not pretended.
                    self.log("relay: could NOT open the UART at {}: {}: {}"
                             .format(rate, type(error).__name__, error))
                    self._answer(address, relay.ACK, seq,
                                 relay.STATUS_BAD_REQUEST)
                    return
                self.log("relay: UART at {} bit/s, 8E1".format(rate))
            self._answer(address, relay.ACK, seq, relay.STATUS_OK)
        elif kind == relay.FAILSAFE and len(body) >= 2:
            silence_ms = struct.unpack_from("<H", body)[0]
            try:
                packets = relay.split_packets(body[2:])
            except ValueError:
                self._answer(address, relay.ACK, seq, relay.STATUS_BAD_REQUEST)
                return
            self.failsafe_packets = packets
            self.silence = silence_ms / 1000.0
            self.log("relay: failsafe {} after {} ms of silence".format(
                "{} packet(s)".format(len(packets)) if packets and silence_ms
                else "disarmed", silence_ms))
            self._answer(address, relay.ACK, seq, relay.STATUS_OK)
        elif kind == relay.PING:
            self._answer(address, relay.ACK, seq, relay.STATUS_OK)
        else:
            self._answer(address, relay.ACK, seq, relay.STATUS_BAD_REQUEST)

    def watchdog(self) -> None:
        """Send the failsafe once per silence longer than ``silence``."""
        if (self._fired or not self.failsafe_packets or self.silence <= 0.0
                or time.monotonic() - self._last_heard < self.silence):
            return
        self._fired = True
        self.failsafes += 1
        for packet in self.failsafe_packets:
            self._transact(packet, 0.05)
        self.log("relay: FAILSAFE sent ({} packet(s)) after {:.3f} s without "
                 "the host".format(len(self.failsafe_packets),
                                   time.monotonic() - self._last_heard))

    def spin_once(self, timeout: float = 0.01) -> None:
        self.sock.settimeout(timeout)
        try:
            datagram, address = self.sock.recvfrom(2048)
        except socket.timeout:
            datagram = None
        if datagram is not None:
            if self._cut() or self.rng.random() < self.loss:
                self.stats["dropped_in"] += 1
            else:
                self.stats["in"] += 1
                self.handle(datagram, address)
        self.watchdog()
