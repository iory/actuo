"""End-to-end test over a REAL python-can bus, using its in-memory 'virtual' backend.

This exercises the actual PythonCanTransport (not a fake): frames are encoded by
the CubeMars model, handed to python-can, carried over a virtual bus, received,
and decoded back through the Bus. It runs on macOS and Linux alike, which is the
whole point of keeping the backend in config -- the only thing a real robot
changes is ``interface: socketcan`` / ``channel: can0``.

    pytest tests/test_python_can_transport.py -q
"""

from __future__ import annotations

import errno
import math
import os
import struct
import sys
import time

import pytest

pytest.importorskip("can", reason="python-can not installed")

_HERE = os.path.dirname(__file__)

import can  # noqa: E402

from actuo.core.contracts import (  # noqa: E402
    ActuatorCommand,
    Frame,
    OperatingMode,
)
from actuo.drivers.actuators.cubemars import (  # noqa: E402
    CubeMarsActuator,
    CubeMarsCalibration,
    CubeMarsLimits,
    ServoPacket,
)
from actuo.drivers.buses.can_bus import CanBus  # noqa: E402
from actuo.drivers.transports.python_can import (  # noqa: E402
    PythonCanTransport,
    TransmitQueueFull,
    cia_sample_point,
)

CHANNEL = "actuo-test"


def make_transport(channel=CHANNEL):
    return PythonCanTransport(interface="virtual", channel=channel)


def make_bus(transport):
    actuators = []
    for node_id, ratio, erpm in ((104, 39.0, 220.0), (105, 9.0, 50.0), (106, 6.0, 35.0)):
        actuators.append(
            CubeMarsActuator(
                node_id,
                CubeMarsCalibration(gear_ratio=ratio),
                limits=CubeMarsLimits(
                    max_velocity_erpm=erpm, max_acceleration_erpm_s=erpm
                ),
            )
        )
    return CanBus(transport, actuators, stale_after=1.0)


def test_command_reaches_the_wire_with_the_right_id_and_bytes():
    """A second listener on the same virtual bus must see exactly our frame."""
    transport = make_transport()
    transport.open()
    listener = can.Bus(interface="virtual", channel=CHANNEL, receive_own_messages=True)
    try:
        bus = make_bus(transport)
        bus.write_batch(
            [
                ActuatorCommand(mode=OperatingMode.POSITION, position=math.radians(d))
                for d in (10.0, 20.0, 5.0)
            ]
        )

        seen = [listener.recv(timeout=0.5) for _ in range(3)]
        assert all(m is not None for m in seen)
        assert [m.arbitration_id for m in seen] == [
            (int(ServoPacket.SET_POS_SPD) << 8) | n for n in (104, 105, 106)
        ]
        assert all(m.is_extended_id for m in seen)
        # motor 2 (id 105) -> 20 deg, 50 ERPM envelope
        pos, spd, acc = struct.unpack(">ihh", seen[1].data)
        assert pos == 200000
        assert spd == 5
        assert acc == 5
    finally:
        listener.shutdown()
        transport.close()


def test_status_frames_flow_back_through_the_bus():
    transport = make_transport("actuo-test-rx")
    transport.open()
    injector = can.Bus(interface="virtual", channel="actuo-test-rx")
    try:
        bus = make_bus(transport)
        for node_id, deg in ((104, 11.5), (105, -3.2), (106, 90.0)):
            injector.send(
                can.Message(
                    arbitration_id=(9 << 8) | node_id,
                    is_extended_id=True,
                    data=struct.pack(">hhhbB", int(deg * 10), 0, 0, 33, 0),
                )
            )

        states = bus.read_batch(timeout=1.0)
        assert states[0].position == pytest.approx(math.radians(11.5))
        assert states[1].position == pytest.approx(math.radians(-3.2))
        assert states[2].position == pytest.approx(math.radians(90.0))
        assert all(s.error == 0 for s in states)
    finally:
        injector.shutdown()
        transport.close()


def test_socketcan_bitrate_is_not_smuggled_into_the_kernel_link():
    """socketcan's bitrate belongs to `ip link`, so we must not pass one."""
    transport = PythonCanTransport(
        interface="socketcan", channel="can0", bitrate=1000000
    )
    # Not opened (no can0 here); just check what open() would build.
    assert transport.bitrate == 1000000
    assert transport.interface == "socketcan"


def test_cia_sample_point_matches_the_linux_kernel_rule():
    """Mirrors can_calc_sample_point_nrz in drivers/net/can/dev/calc_bittiming.c.

    Not a style preference: measured on real CubeMars motors, python-can's
    hardcoded 87.5% at 1 Mbit/s produced 1959 error frames and zero data, while
    75% read every motor cleanly on the same adapter and bus.
    """
    assert cia_sample_point(1_000_000) == 75.0
    assert cia_sample_point(800_001) == 75.0
    assert cia_sample_point(800_000) == 80.0
    assert cia_sample_point(500_001) == 80.0
    assert cia_sample_point(500_000) == 87.5
    assert cia_sample_point(125_000) == 87.5


def test_one_megabit_timing_reproduces_the_kernel_timing():
    """The corrected timing must equal what socketcan programs for this adapter.

    The InnoMaker USB2CAN V3.3 reports a 48 MHz clock, and the kernel drove it
    at 1 Mbit/s with prop_seg=5 phase_seg1=6 phase_seg2=4 brp=3 -- i.e.
    tseg1=11, tseg2=4, 16 tq, 75%. Reaching the same numbers from
    from_sample_point is what makes the macOS path equivalent to the Linux one.
    """
    timing = can.BitTiming.from_sample_point(
        f_clock=48_000_000, bitrate=1_000_000,
        sample_point=cia_sample_point(1_000_000),
    )
    assert timing.brp == 3
    assert timing.tseg1 == 11        # kernel: prop_seg 5 + phase_seg1 6
    assert timing.tseg2 == 4
    assert timing.nbt == 16
    assert timing.bitrate == 1_000_000
    assert timing.sample_point == 75.0


def test_sample_point_patch_is_scoped_and_always_restored():
    """The monkeypatch must not leak: it is global while it is installed."""
    before = can.BitTiming.__dict__["from_sample_point"]

    transport = PythonCanTransport(interface="gs_usb", channel=0, bitrate=1_000_000)
    with transport._corrected_sample_point():
        forced = can.BitTiming.from_sample_point(
            f_clock=48_000_000, bitrate=1_000_000, sample_point=87.5
        )
        # inside the scope, 87.5 is overridden to the CiA value
        assert forced.sample_point == 75.0
        assert forced.tseg1 == 11 and forced.tseg2 == 4

    assert can.BitTiming.__dict__["from_sample_point"] is before
    # outside, the caller's argument is honoured again
    assert can.BitTiming.from_sample_point(
        f_clock=48_000_000, bitrate=1_000_000, sample_point=87.5
    ).sample_point == 87.5


def test_sample_point_patch_restored_even_when_construction_raises():
    before = can.BitTiming.__dict__["from_sample_point"]
    transport = PythonCanTransport(interface="gs_usb", channel=0, bitrate=1_000_000)
    try:
        with transport._corrected_sample_point():
            raise RuntimeError("adapter exploded")
    except RuntimeError:
        pass
    assert can.BitTiming.__dict__["from_sample_point"] is before


def test_non_gs_usb_backends_are_left_alone():
    """socketcan gets its timing from the kernel; we must not touch anything."""
    before = can.BitTiming.__dict__["from_sample_point"]
    transport = PythonCanTransport(interface="socketcan", channel="can0")
    with transport._corrected_sample_point():
        assert can.BitTiming.__dict__["from_sample_point"] is before


def test_gs_usb_without_bitrate_fails_loudly():
    transport = PythonCanTransport(interface="gs_usb", channel=0, bitrate=None)
    with pytest.raises(ValueError, match="bitrate"):
        with transport._corrected_sample_point():
            pass


def test_explicit_sample_point_overrides_the_cia_default():
    transport = PythonCanTransport(
        interface="gs_usb", channel=0, bitrate=1_000_000, sample_point=62.5
    )
    with transport._corrected_sample_point():
        assert can.BitTiming.from_sample_point(
            f_clock=48_000_000, bitrate=1_000_000, sample_point=87.5
        ).sample_point == 62.5


def test_gs_usb_close_does_not_reset_or_reprogram_the_adapter():
    """Regression: every run needed a physical replug before the next one worked.

    python-can's GsUsbBus.shutdown() runs a "workaround" that calls
    set_bitrate(87.5%) -- undoing our timing -- and start(), which issues a USB
    device reset that wedges the adapter on macOS. close() must do neither.
    """
    # The gs_usb backend imports pyusb AND gs_usb at module scope, and they
    # are separate PyPI packages. On a socketcan-only machine neither needs to
    # be installed -- this test is about a gs_usb quirk, so skip rather than
    # fail for a dependency the platform has no use for.
    #
    # Guarding on "usb" alone was not enough: a machine can have pyusb from the
    # distro and no gs_usb, which is what the system python3 outside this
    # package's venv looks like here. The guard passed and the test then died
    # on `No module named 'gs_usb'`, reading as a real failure in whatever was
    # being changed at the time.
    pytest.importorskip("usb", reason="pyusb absent; gs_usb backend unused here")
    pytest.importorskip("gs_usb", reason="gs_usb absent; backend unused here")
    from can.interfaces.gs_usb import GsUsbBus

    calls = []

    class FakeGsUsb:
        def stop(self):
            calls.append("stop")

        def start(self, *args, **kwargs):
            calls.append("start")      # would USB-reset the device

        def set_bitrate(self, *args, **kwargs):
            calls.append("set_bitrate")

    class StubBus(GsUsbBus):
        def __init__(self):
            self._is_shutdown = False
            self.gs_usb = FakeGsUsb()
            self._index = 0
            self._bitrate = 1_000_000
            self._periodic_tasks = []
            self._filters = None

    transport = PythonCanTransport(interface="gs_usb", channel=0, bitrate=1_000_000)
    transport._bus = StubBus()
    bus = transport._bus
    transport.close()

    assert calls == ["stop"]
    assert transport._bus is None

    # python-can's own __del__ may call shutdown() later; it must stay inert.
    calls.clear()
    bus.shutdown()
    assert calls == []


def test_zero_timeout_recv_still_reads_buffered_frames():
    """Regression: timeout=0.0 must mean "don't block", not "don't read".

    Returning None without reading silently disabled every zero-timeout drain
    (`bus.poll(timeout=0.0)` in the servicer). The adapter's queues then backed
    up until a USB *write* timed out -- a symptom that points nowhere near the
    receive path.
    """
    transport = make_transport("actuo-test-zero")
    transport.open()
    injector = can.Bus(interface="virtual", channel="actuo-test-zero")
    try:
        injector.send(can.Message(arbitration_id=0x123, is_extended_id=True,
                                  data=bytes(8)))
        frame = transport.recv(timeout=0.0)
        assert frame is not None, "zero-timeout recv dropped a buffered frame"
        assert frame.address == 0x123
        # ...and with nothing buffered it still returns promptly.
        assert transport.recv(timeout=0.0) is None
    finally:
        injector.shutdown()
        transport.close()


def test_zero_timeout_poll_drains_the_whole_backlog():
    """What the servicer actually relies on, end to end through CanBus."""
    from actuo.drivers.buses.can_bus import CanBus

    transport = make_transport("actuo-test-drain")
    transport.open()
    injector = can.Bus(interface="virtual", channel="actuo-test-drain")
    try:
        bus = make_bus(transport)
        for _ in range(4):
            for node in (104, 105, 106):
                injector.send(can.Message(
                    arbitration_id=(41 << 8) | node, is_extended_id=True,
                    data=struct.pack(">hhhbB", 100, 0, 0, 30, 0)))
        assert bus.poll(timeout=0.0) == 12
    finally:
        injector.shutdown()
        transport.close()


class EchoingBus:
    """Stands in for gs_usb: hands our own sent frames back as is_rx=False.

    The virtual backend cannot reproduce this, which is why the real bug only
    showed up on hardware.
    """

    def __init__(self, messages):
        self._queue = list(messages)

    def recv(self, timeout=None):
        return self._queue.pop(0) if self._queue else None

    def shutdown(self):
        pass


def _echo(node):
    m = can.Message(arbitration_id=(6 << 8) | node, is_extended_id=True,
                    data=bytes(8))
    m.is_rx = False          # gs_usb marks our own transmitted frames like this
    return m


def _status(node):
    m = can.Message(arbitration_id=(41 << 8) | node, is_extended_id=True,
                    data=struct.pack(">hhhbB", 100, 0, 0, 30, 0))
    m.is_rx = True
    return m


def test_echoes_at_the_head_of_the_queue_do_not_stop_a_zero_timeout_drain():
    """Regression: this throttled the drain to ~1 frame per cycle.

    Right after write_batch the adapter's queue starts with our own echoes.
    Treating a skipped echo as a timeout ended the drain there, so the servicer
    read 49 frames per second while the motors sent ~390 -- the queues backed
    up and the FIRST symptom was a USB write timing out, 63 cycles in.
    """
    transport = PythonCanTransport(interface="gs_usb", channel=0, bitrate=1_000_000)
    transport._bus = EchoingBus(
        [_echo(104), _echo(105), _echo(106)] + [_status(n) for n in (104, 105, 106)]
    )
    drained = []
    while True:
        frame = transport.recv(timeout=0.0)
        if frame is None:
            break
        drained.append(frame)

    assert len(drained) == 3, "echoes stopped the drain: got {}".format(len(drained))
    assert all(f.meta["is_rx"] for f in drained)
    assert sorted(f.address & 0xFF for f in drained) == [104, 105, 106]


def test_echo_only_queue_drains_to_none_without_hanging():
    transport = PythonCanTransport(interface="gs_usb", channel=0, bitrate=1_000_000)
    transport._bus = EchoingBus([_echo(104), _echo(105)])
    assert transport.recv(timeout=0.0) is None


def test_recv_returns_none_on_timeout_rather_than_blocking_forever():
    transport = make_transport("actuo-test-idle")
    transport.open()
    try:
        assert transport.recv(timeout=0.01) is None
    finally:
        transport.close()


def test_write_before_open_fails_loudly():
    transport = make_transport("actuo-test-closed")
    with pytest.raises(RuntimeError, match="not open"):
        transport.write([])


# --------------------------------------------------------------------------
# ENOBUFS: a full transmit queue used to kill the servicer outright.
#
# python-can's send(timeout=...) does NOT cover this. Its socketcan loop waits
# for select() writability and retries only a PARTIAL write; an ENOBUFS from
# socket.send propagates straight out. So the 50 ms timeout bought no tolerance
# at all, and one momentary backlog took the process down -- and with
# on_exit=Shutdown(), the arm with it.
# --------------------------------------------------------------------------

def _enobufs():
    return can.CanOperationError(
        "Failed to transmit: No buffer space available", errno.ENOBUFS
    )


class QueueFullBus:
    """Refuses the first ``fail_times`` sends with ENOBUFS, then accepts."""

    def __init__(self, fail_times, error=None):
        self.fail_times = fail_times
        self.error = error or _enobufs()
        self.sent = []
        self.attempts = 0

    def send(self, msg, timeout=None):
        self.attempts += 1
        if self.fail_times > 0:
            self.fail_times -= 1
            raise self.error
        self.sent.append(msg)

    def shutdown(self):
        pass


def _frames(*ids):
    return [Frame(payload=bytes(8), address=i, meta={"extended": True})
            for i in ids]


def test_a_momentary_full_queue_is_retried_instead_of_fatal():
    transport = PythonCanTransport(interface="socketcan", channel="can0")
    transport._bus = QueueFullBus(fail_times=2)

    transport.write(_frames(0x100))

    assert [m.arbitration_id for m in transport._bus.sent] == [0x100]
    assert transport.tx_retries == 2
    assert transport.tx_stalls == 1, "one frame stalled, whatever the attempts"


def test_a_retry_resumes_at_the_failed_frame_and_never_re_sends_earlier_ones():
    """Restarting the burst would duplicate frames already on the wire."""
    transport = PythonCanTransport(interface="socketcan", channel="can0")

    class FailOnThird(QueueFullBus):
        def send(self, msg, timeout=None):
            self.attempts += 1
            # third frame fails once, everything else goes through
            if msg.arbitration_id == 0x300 and self.fail_times > 0:
                self.fail_times -= 1
                raise _enobufs()
            self.sent.append(msg)

    transport._bus = FailOnThird(fail_times=1)
    transport.write(_frames(0x100, 0x200, 0x300, 0x400))

    assert [m.arbitration_id for m in transport._bus.sent] == [
        0x100, 0x200, 0x300, 0x400
    ], "a frame was duplicated or dropped"
    assert transport.tx_retries == 1


def test_an_unrelenting_full_queue_fails_with_a_diagnosable_error():
    """It must still fail -- just not as a bare ENOBUFS traceback."""
    transport = PythonCanTransport(
        interface="socketcan", channel="can0", tx_retry_window=0.005
    )
    transport._bus = QueueFullBus(fail_times=10**6)

    with pytest.raises(TransmitQueueFull) as caught:
        transport.write(_frames(0x2968))

    text = str(caught.value)
    assert "frame 1 of 1" in text
    assert "0x2968" in text
    # The message must send the operator to the DECIDING measurement first --
    # the link's error counters, which separate a physical fault from simply
    # offering frames too fast. Naming causes without saying how to tell them
    # apart is what made the first version of this message useless.
    assert "ip -s -details link show can0" in text
    assert "error-pass" in text
    assert "termination" in text
    # ...and it must warn about the backlog rather than recommend a deep queue.
    assert "STALE" in text
    assert transport.tx_retries >= 1


def test_python_cans_own_transmit_buffer_full_counts_as_queue_full():
    """Same event, other code path: select() never reported writable.

    It carries no errno, so it is matched by message. Without this it stayed
    fatal even though it is exactly as transient as ENOBUFS.
    """
    transport = PythonCanTransport(interface="socketcan", channel="can0")
    transport._bus = QueueFullBus(
        fail_times=1, error=can.CanOperationError("Transmit buffer full")
    )
    transport.write(_frames(0x100))
    assert transport.tx_retries == 1


def test_a_down_link_is_NOT_retried():
    """ENETDOWN does not become true again by waiting.

    Retrying it would turn "your CAN interface is down" into a delayed version
    of the same message, having burned the retry window first.
    """
    transport = PythonCanTransport(
        interface="socketcan", channel="can0", tx_retry_window=1.0
    )
    down = can.CanOperationError(
        "Failed to transmit: Network is down", errno.ENETDOWN
    )
    transport._bus = QueueFullBus(fail_times=10**6, error=down)

    started = time.monotonic()
    with pytest.raises(can.CanOperationError, match="Network is down"):
        transport.write(_frames(0x100))

    assert transport._bus.attempts == 1, "a non-transient error was retried"
    assert time.monotonic() - started < 0.5, "burned the retry window on ENETDOWN"
    assert transport.tx_retries == 0


def test_the_inner_send_timeout_cannot_eat_the_whole_retry_budget():
    """Regression from a real crash on a 4-joint CubeMars arm, 2026-07-30.

    python-can's socketcan send() spends its ``timeout`` blocking in select()
    BEFORE raising "Transmit buffer full". With send_timeout=0.05 against a 2 ms
    window, the deadline was already 48 ms in the past on the first failure, so
    the loop reported "after 1 attempt(s)" and never retried at all -- the retry
    feature was dead code on the exact path it was written for.
    """
    seen_timeouts = []

    class RecordingBus(QueueFullBus):
        def send(self, msg, timeout=None):
            seen_timeouts.append(timeout)
            return super().send(msg, timeout=timeout)

    transport = PythonCanTransport(
        interface="socketcan", channel="can0",
        send_timeout=0.05, tx_retry_window=0.01,
    )
    transport._bus = RecordingBus(fail_times=10**6)
    with pytest.raises(TransmitQueueFull):
        transport.write(_frames(0x100))

    assert all(t <= 0.01 for t in seen_timeouts), (
        "send_timeout {} exceeded the retry window".format(seen_timeouts))
    assert transport.tx_retries > 1, (
        "only {} attempt(s): the budget was consumed by the inner timeout"
        .format(transport.tx_retries))


def test_a_short_send_timeout_is_not_stretched_by_the_retry_window():
    """The cap is a minimum of the two, never a replacement for send_timeout."""
    seen = []

    class RecordingBus(QueueFullBus):
        def send(self, msg, timeout=None):
            seen.append(timeout)
            return super().send(msg, timeout=timeout)

    transport = PythonCanTransport(
        interface="socketcan", channel="can0",
        send_timeout=0.001, tx_retry_window=1.0,
    )
    transport._bus = RecordingBus(fail_times=2)
    transport.write(_frames(0x100))
    assert all(t <= 0.001 for t in seen), seen


def test_the_retry_budget_is_per_burst_not_per_frame():
    """Worst-case stall must not scale with the number of joints.

    The retry window is time stolen from the control cycle that services the
    bus. Per-frame, a 4-joint burst could stall 4x the window -- 8 ms at the
    2 ms default, which blows a 200 Hz cycle. So the whole burst shares one
    deadline and the cost of retrying is independent of DOF count.
    """
    transport = PythonCanTransport(
        interface="socketcan", channel="can0", tx_retry_window=0.02
    )
    transport._bus = QueueFullBus(fail_times=10**6)

    started = time.monotonic()
    with pytest.raises(TransmitQueueFull):
        transport.write(_frames(0x100, 0x200, 0x300, 0x400))
    elapsed = time.monotonic() - started

    # One window, not four. Generous upper bound so this cannot flake on a
    # loaded machine, but 4x20 ms = 80 ms would fail it.
    assert elapsed < 0.06, "elapsed {:.3f}s suggests a per-frame budget".format(
        elapsed)


def test_retrying_can_be_switched_off_entirely():
    transport = PythonCanTransport(
        interface="socketcan", channel="can0", tx_retry_window=0.0
    )
    transport._bus = QueueFullBus(fail_times=1)
    with pytest.raises(TransmitQueueFull):
        transport.write(_frames(0x100))
    assert transport._bus.attempts == 1


def test_a_clean_write_never_touches_the_retry_counters():
    """A healthy run's telemetry must be unchanged by any of this."""
    transport = PythonCanTransport(interface="socketcan", channel="can0")
    transport._bus = QueueFullBus(fail_times=0)
    transport.write(_frames(0x100, 0x200))
    assert transport.tx_retries == 0
    assert transport.tx_stalls == 0
    assert transport._bus.attempts == 2


def test_tx_queue_advice_only_speaks_for_socketcan():
    """The queue belongs to a kernel netdev; other backends have no such thing."""
    assert PythonCanTransport(
        interface="virtual", channel="whatever").tx_queue_len is None
    assert PythonCanTransport(
        interface="virtual", channel="whatever").tx_queue_advice() is None
    # A socketcan channel that does not exist is unknowable, not zero.
    assert PythonCanTransport(
        interface="socketcan", channel="can-nope").tx_queue_len is None
    assert PythonCanTransport(
        interface="socketcan", channel="can-nope").tx_queue_advice() is None


def test_tx_queue_advice_warns_about_a_DEEP_queue_not_a_shallow_one(monkeypatch):
    """The obvious advice is backwards here, and hardware said so.

    Raising txqueuelen to 1000 on a 4-joint CubeMars arm gave the link 5 s of
    setpoints.
    This queue carries periodic position commands, so a backlog is a delay line:
    once the bus recovers the motors execute every stale command in order. A
    shallow queue that drops is the safe configuration, so a deep one is what
    deserves the warning.
    """
    transport = PythonCanTransport(interface="socketcan", channel="can0")

    monkeypatch.setattr(
        PythonCanTransport, "tx_queue_len", property(lambda self: 1000))
    advice = transport.tx_queue_advice()
    assert advice is not None
    assert "txqueuelen=1000" in advice
    assert "5.0 s of setpoints" in advice
    assert "delay line" in advice
    assert "txqueuelen 20" in advice, "must recommend shrinking, not growing"

    # The kernel default is only 0.05 s of backlog: nothing to warn about.
    monkeypatch.setattr(
        PythonCanTransport, "tx_queue_len", property(lambda self: 10))
    assert transport.tx_queue_advice() is None


def test_queue_staleness_is_measured_against_the_ACTUAL_frame_rate(monkeypatch):
    """A depth only becomes seconds once you know the offered rate."""
    transport = PythonCanTransport(interface="socketcan", channel="can0")
    monkeypatch.setattr(
        PythonCanTransport, "tx_queue_len", property(lambda self: 100))

    # 100 frames at 200/s = 0.5 s: worth warning about.
    assert "0.5 s of setpoints" in transport.tx_queue_advice(200.0)
    # The same depth on a 1-joint robot at 50 Hz is 2 s: worse, not the same.
    assert "2.0 s of setpoints" in transport.tx_queue_advice(50.0)
    # ...and a nonsensical rate must not divide by zero.
    assert transport.tx_queue_advice(0.0) is None
