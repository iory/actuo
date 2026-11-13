"""The RCB-4mini over WiFi: its packets in UDP datagrams, through a relay.

The relay (an Arduino UNO R4 WiFi wired to the board's COM port, running
the UART relay sketch, ``firmware/actuo_uart_relay/``) forwards each packet to the board's UART and
sends back the answer; the datagram format is
:mod:`actuo.drivers.uart_relay_protocol`. Everything above the bytes --
the bus, the actuator models, ros2_control -- is the cable's, unchanged:
``kondo_rcb4_bus`` takes this transport where it took ``kondo_rcb4``.

What the network adds, and how it is handled:

* **Loss and delay.** A request whose answer does not come back within the
  board's timeout plus ``network_timeout`` is sent again (``retries``), and
  then yields ``None`` in that slot, as a silent board does on the cable; the bus counts it as a miss and its silence
  watchdog (``actuo.drivers.link``) takes over if it goes on. Every request
  carries a sequence number, so an answer that arrives late -- after the host
  gave up on it -- is dropped, never taken as the answer to the next request
  (counted as ``stale_replies``).
* **The host going away.** On a cable a dead host leaves the board executing
  its last command; through a relay too, so the bus hands the relay a
  failsafe (:meth:`set_failsafe`: the wheels stopped) and the relay sends it
  when the host has been silent for ``failsafe_silence`` seconds. The relay
  reports how often it did, and that count is logged when it changes.

Bit rate: the relay's UART is reopened at :meth:`reopen`'s rate with even
parity (the board's). Whether the UNO R4's UART reaches the board's 1.25
Mbit/s is NOT verified; the WiFi config runs the board at 115200.
"""

from __future__ import annotations

import logging
import socket
import time
from typing import List, Optional, Sequence

from actuo.core.contracts import Frame, Transport
from actuo.core.registry import register_transport

from actuo.drivers import uart_relay_protocol as relay
from actuo.drivers.kondo_h7_protocol import ChecksumError, ProtocolError, frame, unframe

LOG = logging.getLogger(__name__)

#: The relay's port in the sketch.
DEFAULT_PORT = 9750


@register_transport("kondo_rcb4_udp")
class KondoRcb4UdpTransport(Transport):
    """Length-prefixed packets to an RCB-4mini through a UDP relay.

    Parameters
    ----------
    channel : str
        ``host`` or ``host:port`` of the relay. Named ``channel`` like the
        serial transports so config overrides stay uniform.
    bitrate : int
        The board's COM rate, which the relay's UART is opened at.
    timeout : float
        Default per-packet read timeout of the BOARD [s].
    network_timeout : float
        Added to every wait for the round trip over the network [s].
    failsafe_silence : float
        Host silence after which the relay sends the failsafe [s]; 0 disarms.
    retries : int
        Times a request with no answer is sent again before it counts as a
        miss. Safe because what the RCB-4mini bus sends is idempotent (RAM
        reads and writes, a batch move to absolute targets): if the request
        got through and only its answer was lost, doing it twice changes
        nothing. Without it one lost datagram during the bus's startup -- a
        sequence that assumes a cable -- aborts the servicer (measured: 5 %
        loss, "board did not answer a 20-byte read at 0x016c").
    """

    def __init__(self, channel: str, bitrate: int = 115200,
                 timeout: float = 0.1, network_timeout: float = 0.05,
                 failsafe_silence: float = 0.5, retries: int = 2) -> None:
        host, _, port = str(channel).partition(":")
        self.channel = str(channel)
        self.address = (host, int(port) if port else DEFAULT_PORT)
        self.bitrate = int(bitrate)
        #: The rate the relay's UART is at; the bus compares it to ``bitrate``.
        self.current_bitrate = self.bitrate
        self.timeout = float(timeout)
        self.network_timeout = float(network_timeout)
        self.failsafe_silence = float(failsafe_silence)
        self.retries = int(retries)
        self._socket: Optional[socket.socket] = None
        self._seq = 0
        self._failsafes: Optional[int] = None
        self._checksum_errors = 0
        self._framing_errors = 0
        self._stale_replies = 0
        self._lost = 0
        self._device_timeouts = 0
        self._retries_sent = 0

    # -- lifecycle ---------------------------------------------------------

    def open(self) -> None:
        if self._socket is not None:
            return
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.connect(self.address)
        try:
            self._set_baud(self.current_bitrate)
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None

    @property
    def is_open(self) -> bool:
        return self._socket is not None

    def reopen(self, bitrate: int) -> None:
        """Have the relay reopen its UART at ``bitrate`` (8E1)."""
        self.open()
        self._set_baud(int(bitrate))
        self.current_bitrate = int(bitrate)

    @property
    def latency_timer_ms(self) -> Optional[float]:
        """No USB adapter on this link, so no latency timer to report."""
        return None

    # -- counters ----------------------------------------------------------

    @property
    def checksum_errors(self) -> int:
        return self._checksum_errors

    @property
    def framing_errors(self) -> int:
        return self._framing_errors

    @property
    def stale_replies(self) -> int:
        """Answers that came after their request had been given up on."""
        return self._stale_replies

    @property
    def lost(self) -> int:
        """Requests with no answer from the relay (the network lost one)."""
        return self._lost

    @property
    def retries_sent(self) -> int:
        """Requests sent again because the first had no answer."""
        return self._retries_sent

    @property
    def device_timeouts(self) -> int:
        """Requests the relay delivered and the board did not answer."""
        return self._device_timeouts

    @property
    def failsafes(self) -> Optional[int]:
        """How often the relay has sent the failsafe; ``None`` before it said."""
        return self._failsafes

    # -- the exchange --------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFFFF
        return self._seq

    def _require_open(self) -> socket.socket:
        if self._socket is None:
            from actuo.drivers.link import TransportNotOpen
            raise TransportNotOpen("kondo_rcb4_udp transport is not open")
        return self._socket

    def _note_failsafes(self, count: int) -> None:
        if self._failsafes is not None and count != self._failsafes:
            LOG.warning(
                "the relay at %s sent the failsafe (stop the wheels) %d "
                "time(s) while this host was silent for more than %.2f s",
                self.channel, count - self._failsafes, self.failsafe_silence)
        self._failsafes = count

    def _exchange(self, datagram: bytes, seq: int, wait: float):
        """Send ``datagram``; the answer with ``seq``, or ``None`` by ``wait``."""
        sock = self._require_open()
        sock.send(datagram)
        deadline = time.monotonic() + wait
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self._lost += 1
                return None
            sock.settimeout(left)
            try:
                data = sock.recv(2048)
            except socket.timeout:
                self._lost += 1
                return None
            except ConnectionRefusedError:
                # ICMP port unreachable: nothing listening at the relay's
                # address. The same as no answer, which is what it is.
                continue
            try:
                kind, got, status, failsafes, packet = relay.parse_answer(data)
            except ValueError:
                self._framing_errors += 1
                continue
            self._note_failsafes(failsafes)
            if got != seq:
                self._stale_replies += 1
                continue
            return kind, status, packet

    def _set_baud(self, rate: int) -> None:
        seq = self._next_seq()
        answer = self._exchange(relay.baud(seq, rate), seq,
                                self.network_timeout + self.timeout)
        if answer is None:
            # A link failure (actuo.drivers.link retries it), like a USB
            # adapter that is not plugged in.
            from actuo.drivers.link import TransportNotOpen
            raise TransportNotOpen(
                "the relay at {} did not answer (asked to open its UART at {} "
                "bit/s). Is it powered, on this network, and running "
                "the UART relay sketch?".format(self.channel, rate))
        if answer[1] != relay.STATUS_OK:
            raise OSError("the relay at {} refused {} bit/s: {}".format(
                self.channel, rate, relay.STATUS_NAMES.get(answer[1])))

    def set_failsafe(self, frames: Sequence[Frame]) -> bool:
        """Give the relay the packets to send if this host goes silent.

        Returns
        -------
        bool
            Always ``True``: there is a relay on this link. Raises if it did
            not take them -- a failsafe that silently did not arm is worse
            than none.
        """
        seq = self._next_seq()
        packets = [frame(f.payload) for f in frames]
        answer = self._exchange(
            relay.failsafe(seq, self.failsafe_silence if packets else 0.0,
                           packets),
            seq, self.network_timeout + self.timeout)
        if answer is None or answer[1] != relay.STATUS_OK:
            raise OSError("the relay at {} did not take the failsafe ({})"
                          .format(self.channel, "no answer" if answer is None
                                  else relay.STATUS_NAMES.get(answer[1])))
        return True

    # -- Transport contract ------------------------------------------------

    def write(self, frames: Sequence[Frame]) -> None:
        """Send each packet and drop its reply (the board answers everything)."""
        self.transact(frames, self.timeout)

    def recv(self, timeout: float) -> Optional[Frame]:
        """Nothing comes from the board unasked; replies come via transact."""
        return None

    def transact(self, frames: Sequence[Frame],
                 timeout: float) -> List[Optional[Frame]]:
        """One request at a time, each answered or ``None``."""
        replies: List[Optional[Frame]] = []
        for item in frames:
            answer = None
            for _ in range(1 + self.retries):
                seq = self._next_seq()
                answer = self._exchange(
                    relay.forward(seq, timeout, frame(item.payload)), seq,
                    timeout + self.network_timeout)
                if answer is not None:
                    break
                self._retries_sent += 1
            if answer is None:
                replies.append(None)
                continue
            _, status, packet = answer
            if status != relay.STATUS_OK:
                self._device_timeouts += 1
                replies.append(None)
                continue
            try:
                body = unframe(packet)
            except ChecksumError:
                self._checksum_errors += 1
                replies.append(None)
                continue
            except ProtocolError:
                self._framing_errors += 1
                replies.append(None)
                continue
            replies.append(Frame(payload=body))
        return replies
