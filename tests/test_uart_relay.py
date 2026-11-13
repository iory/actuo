"""The WiFi path: kondo_rcb4_udp -> UDP -> the relay -> the emulated board.

The relay is actuo.drivers.uart_relay (the Arduino sketch's logic) on a
localhost port, with the RCB-4mini emulator behind a fake UART.
"""

import threading
import time

from actuo.core.contracts import Frame

from actuo.drivers.actuators.kondo_ics import CENTER_PULSE
from actuo.drivers.kondo_h7_protocol import build_multi_servo
from actuo.drivers.kondo_rcb4_emulator import Rcb4Emulator, _frame
from rcb4.protocol.rcb4mini import build_ack
from actuo.drivers.transports.kondo_rcb4_udp import KondoRcb4UdpTransport
from actuo.drivers.uart_relay import UartRelay


class FakeUart:
    """The emulator behind a UART: a written packet's answer becomes readable."""

    def __init__(self, emulator):
        self.emulator = emulator
        self.buffer = bytearray()
        self.baudrate = 115200
        self.lock = threading.Lock()

    def close(self):
        pass

    def open(self):
        pass

    @property
    def in_waiting(self):
        return len(self.buffer)

    def reset_input_buffer(self):
        with self.lock:
            self.buffer.clear()

    def write(self, packet):
        answer = self.emulator.reply(bytes(packet[1:-1]))
        if answer is not None:
            with self.lock:
                self.buffer += _frame(answer)

    def read(self, n):
        with self.lock:
            out = bytes(self.buffer[:n])
            del self.buffer[:n]
        if not out:
            time.sleep(0.001)
        return out


class Running:
    """A relay spinning in a thread on a free localhost port."""

    def __init__(self, **kwargs):
        self.emulator = Rcb4Emulator([20, 21])
        self.relay = UartRelay(FakeUart(self.emulator), 0, "127.0.0.1",
                               log=lambda m: None, **kwargs)
        self.port = self.relay.sock.getsockname()[1]
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()

    def _spin(self):
        while not self.stop.is_set():
            self.relay.spin_once(0.005)

    def transport(self, **kwargs):
        t = KondoRcb4UdpTransport("127.0.0.1:{}".format(self.port), **kwargs)
        t.open()
        return t

    def close(self):
        self.stop.set()
        self.thread.join(1.0)
        self.relay.sock.close()


def test_a_packet_goes_through_and_its_answer_comes_back():
    r = Running()
    try:
        t = r.transport()
        reply = t.transact([Frame(payload=build_ack())], 0.1)[0]
        assert reply is not None and reply.payload == bytes([0xFE, 0x06])
        assert t.lost == 0 and t.failsafes == 0
    finally:
        r.close()


def test_the_uart_rate_can_be_changed():
    r = Running()
    try:
        t = r.transport()
        t.reopen(1250000)
        assert r.relay.uart.baudrate == 1250000
        assert t.transact([Frame(payload=build_ack())], 0.1)[0] is not None
    finally:
        r.close()


def test_a_lost_datagram_is_a_missed_reply_not_a_wrong_one():
    r = Running()
    try:
        t = r.transport(network_timeout=0.02, retries=0)
        r.relay.loss = 1.0
        assert t.transact([Frame(payload=build_ack())], 0.02) == [None]
        assert t.lost == 1
        r.relay.loss = 0.0
        assert t.transact([Frame(payload=build_ack())], 0.1)[0] is not None
    finally:
        r.close()


def test_a_lost_datagram_is_sent_again():
    r = Running(rng=__import__("random").Random(3))
    try:
        t = r.transport(network_timeout=0.02, retries=5)
        r.relay.loss = 0.3
        replies = t.transact([Frame(payload=build_ack())] * 40, 0.02)
        assert all(reply is not None for reply in replies)
        assert t.retries_sent > 0
    finally:
        r.close()


def test_a_late_answer_is_never_taken_for_the_next_request():
    r = Running()
    try:
        t = r.transport(network_timeout=0.01, retries=0)
        r.relay.delay = 0.08          # longer than the host waits
        assert t.transact([Frame(payload=build_ack())], 0.01) == [None]
        r.relay.delay = 0.0
        time.sleep(0.15)              # the late answer is now in the socket
        reply = t.transact([Frame(payload=build_ack())], 0.1)[0]
        assert reply is not None
        assert t.stale_replies >= 1
    finally:
        r.close()


def test_the_relay_stops_the_wheels_when_the_host_goes_silent():
    r = Running()
    try:
        t = r.transport(failsafe_silence=0.1)
        # Wheels on and turning: their commanded pulse away from the stop.
        t.transact([Frame(payload=build_multi_servo([20, 21], [32767, 32767],
                                                    1))], 0.1)
        t.transact([Frame(payload=build_multi_servo([20, 21], [9000, 6000],
                                                    1))], 0.1)
        assert r.emulator._motion[20][1] == 9000
        stop = Frame(payload=build_multi_servo([20, 21], [CENTER_PULSE] * 2, 1))
        assert t.set_failsafe([stop]) is True
        time.sleep(0.3)               # the host is gone
        assert r.relay.failsafes == 1
        assert r.emulator._motion[20][1] == CENTER_PULSE
        assert r.emulator._motion[21][1] == CENTER_PULSE
        time.sleep(0.2)               # still silent: sent once, not repeatedly
        assert r.relay.failsafes == 1
        # The host is back; it learns the relay acted.
        t.transact([Frame(payload=build_ack())], 0.1)
        assert t.failsafes == 1
    finally:
        r.close()
