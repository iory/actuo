"""CAN transport backed by python-can.

This transport moves opaque CAN frames and knows nothing about motors, angles,
or control modes -- the arbitration id it routes on is chosen by whichever
actuator model emitted the frame. Transport and actuator semantics are
separate layers, each replaceable without touching the other.

Portability
-----------
python-can is a pure-Python library available on both macOS and Linux; what
differs per platform is only the *backend* it drives. That difference is
expressed as configuration (``interface``/``channel``), never as a branch on
``sys.platform`` here:

===============  ==========================  ==========================
interface        channel example             where it works
===============  ==========================  ==========================
``socketcan``    ``can0``                    Linux only (kernel CAN)
``slcan``        ``/dev/tty.usbmodem14201``  macOS + Linux (serial CAN)
``gs_usb``       ``0`` (device index)        macOS + Linux (candleLight)
``pcan``         ``PCAN_USBBUS1``            macOS + Linux (PEAK)
``virtual``      any name                    everywhere (no hardware)
===============  ==========================  ==========================

So the same robot config runs on the Linux robot PC via ``socketcan`` and on a
mac with a USB-CAN adapter via ``slcan``/``gs_usb`` by changing one string.
Nothing is silently substituted: asking for ``socketcan`` on macOS raises
python-can's own error rather than quietly degrading.

References
----------
python-can bus configuration and supported interfaces:
https://python-can.readthedocs.io/en/stable/configuration.html
https://python-can.readthedocs.io/en/stable/interfaces.html
"""

from __future__ import annotations

import contextlib
import errno
import time
from typing import Any, Dict, List, Optional, Sequence

import can

from actuo.core.contracts import Frame, Transport
from actuo.core.registry import register_transport

#: Pause between retries of a frame the kernel had no queue room for. One
#: 8-byte frame at 1 Mbit/s occupies the wire for roughly 130 us, so this is
#: about the time it takes for one queue slot to free on a healthy bus. Shorter
#: would spin; much longer would waste the retry window on sleeping.
_TX_RETRY_PAUSE = 0.0002


def _is_queue_full(err: BaseException) -> bool:
    """Is this transmit failure the transient "no room right now" one?

    Deliberately narrow. Two conditions qualify, and they are the same event
    seen from two places in python-can's socketcan backend:

    * ``errno.ENOBUFS`` -- ``socket.send`` refused because the qdisc queue
      (``txqueuelen``) is full. This is what actually killed a run.
    * python-can's own ``"Transmit buffer full"`` -- raised when its
      ``select()`` loop never saw the socket become writable within the send
      timeout. It carries no errno, hence the string match.

    Everything else is passed through untouched. ``ENETDOWN`` means the link is
    down and ``EBADF`` means the socket is closed; neither becomes true again by
    waiting, and retrying them would replace a correct diagnosis with a delayed
    one.
    """
    if getattr(err, "error_code", None) == errno.ENOBUFS:
        return True
    return "transmit buffer full" in str(err).lower()


class TransmitQueueFull(RuntimeError):
    """The kernel/adapter had no room for a frame and never got room.

    Distinct from every other transmit failure on purpose. A full transmit
    queue means frames are being *offered* faster than the bus carries them
    away, which has exactly two causes -- we are writing too fast, or the bus
    has stopped acknowledging -- and both are diagnosable. A bare
    ``CanOperationError: No buffer space available`` is neither.

    Raised only after the retry window is exhausted; see
    :meth:`PythonCanTransport.write`.
    """



def cia_sample_point(bitrate: int) -> float:
    """CiA-recommended sample point, in percent, for ``bitrate``.

    Mirrors the Linux kernel's ``can_calc_sample_point_nrz``
    (``drivers/net/can/dev/calc_bittiming.c``), which is the reference
    behaviour every socketcan link already gets::

        if (bt->bitrate > 800 * KILO) return 750;
        if (bt->bitrate > 500 * KILO) return 800;
        return 875;

    The point moves down as the bitrate rises because propagation delay eats
    the margin: at 1 Mbit/s a late sample lands in the next bit's transition.
    """
    if bitrate > 800_000:
        return 75.0
    if bitrate > 500_000:
        return 80.0
    return 87.5


@register_transport("python_can")
class PythonCanTransport(Transport):
    """Send/receive CAN frames through any python-can backend.

    Parameters
    ----------
    interface : str
        python-can interface name, e.g. ``"socketcan"`` (Linux), ``"slcan"`` or
        ``"gs_usb"`` (macOS/Linux USB adapters), ``"virtual"`` (no hardware).
    channel : str
        Backend-specific channel: ``"can0"`` for socketcan, a serial device path
        for slcan, a device index for gs_usb.
    bitrate : int, optional
        Bus bitrate in bit/s. Required by adapter backends (slcan, gs_usb);
        ignored by ``socketcan``, where the bitrate belongs to the kernel link
        and is set out of band (``ip link set can0 type can bitrate 1000000``).
    receive_own_messages : bool
        Whether frames this process sends are echoed back on ``recv``.
    sample_point : float, optional
        Where in each bit the controller samples, in percent. ``None`` (default)
        uses :func:`cia_sample_point`, matching what the Linux kernel would pick.
        Only applied to backends that can be told; see ``_corrected_sample_point``.
    settle : float
        Seconds spent draining the bus after opening a gs_usb adapter, before any
        transmit is allowed. See :meth:`_settle_gs_usb`. 0 disables it.
    send_timeout : float
        Seconds python-can may wait for the socket to report writable, per frame.
        Note this does NOT cover a full transmit queue; see ``tx_retry_window``.
    tx_retry_window : float
        Seconds :meth:`write` keeps re-offering a frame the transmit queue had no
        room for. 0 disables retrying. Default 2 ms, which is roughly the time a
        full ``txqueuelen=10`` queue takes to drain at 1 Mbit/s and a tenth of a
        50 Hz control cycle -- long enough to ride out a momentary backlog, short
        enough that a genuinely dead bus is reported this cycle rather than next.
    extra : dict, optional
        Additional keyword arguments passed straight to ``can.Bus`` for backend
        specific options. Kept opaque so a new adapter needs no code change.
    """

    def __init__(
        self,
        interface: str,
        channel: str,
        bitrate: Optional[int] = None,
        receive_own_messages: bool = False,
        sample_point: Optional[float] = None,
        settle: float = 0.5,
        send_timeout: float = 0.05,
        tx_retry_window: float = 0.002,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.interface = interface
        self.channel = channel
        self.bitrate = bitrate
        self.receive_own_messages = receive_own_messages
        self.sample_point = sample_point
        self.settle = settle
        self.send_timeout = send_timeout
        self.tx_retry_window = tx_retry_window
        self.extra = dict(extra or {})
        self._bus: Optional[can.BusABC] = None
        self._tx_retries = 0
        self._tx_stalls = 0

    @property
    def tx_retries(self) -> int:
        """Times a frame had to be re-offered because the queue was full.

        Exposed rather than swallowed: bus health is measured and reported,
        not assumed. A retry that
        succeeds is not "fine" -- it is the leading indicator of the failure that
        killed a run on 2026-07-30, where four clean minutes were followed by an
        error burst, then ENOBUFS. This counter is what makes the approach to
        that cliff visible instead of only its edge.
        """
        return self._tx_retries

    @property
    def tx_stalls(self) -> int:
        """Frames that needed at least one retry. Counts events, not attempts."""
        return self._tx_stalls

    def open(self) -> None:
        if self._bus is not None:
            return

        kwargs: Dict[str, Any] = dict(self.extra)
        kwargs["interface"] = self.interface
        kwargs["channel"] = self.channel
        kwargs["receive_own_messages"] = self.receive_own_messages
        # socketcan takes its bitrate from the kernel link, not from us. Passing
        # one there would imply this process configured the bus when it did not.
        if self.bitrate is not None and self.interface != "socketcan":
            kwargs["bitrate"] = self.bitrate

        with self._corrected_sample_point():
            self._bus = can.Bus(**kwargs)
        if self.interface == "gs_usb" and self.settle > 0.0:
            self._settle_gs_usb()

    def _settle_gs_usb(self) -> None:
        """Let the adapter come up before anyone transmits.

        ``GsUsb.start()`` issues a USB device ``reset()`` and then immediately
        sends MODE_START. Writing right after that returns from libusb as
        ``[Errno 60] Operation timed out`` on the bulk OUT endpoint -- the first
        frame, every time -- while reads work perfectly.

        Measured, which is the only reason this exists: the jog tool writes
        successfully because a human answers its "type MOVE" prompt first, i.e.
        seconds pass; the bus servicer wrote ~milliseconds after open and failed
        on cycle 0 with 0 frames written. Same transport, same code path -- the
        only variable was the delay.

        So: wait, and spend the wait draining rather than sleeping blind, which
        also clears whatever the drivers streamed while we were opening.
        """
        deadline = time.monotonic() + self.settle
        while time.monotonic() < deadline:
            if self.recv(timeout=0.02) is None:
                continue

    @contextlib.contextmanager
    def _corrected_sample_point(self):
        """Make python-can compute a usable sample point for gs_usb adapters.

        python-can (checked through 4.6.1) hardcodes ``sample_point=87.5`` inside
        ``GsUsbBus.__init__`` and offers no override. 87.5% is the CiA value for
        <=500 kbit/s only; faster buses must sample earlier because propagation
        delay eats the margin. The consequence is not subtle -- measured on an
        InnoMaker USB2CAN V3.3 against CubeMars motors at their fixed 1 Mbit/s:
        87.5% gave 1959 error frames and zero data in 5 s, while 75% (what the
        Linux kernel programs for the very same adapter) read every motor
        cleanly.

        Fixing it *after* construction is not possible: ``GsUsb.start()`` issues
        a USB ``reset()``, which on macOS re-enumerates the device and
        invalidates the handle (``[Errno 19] No such device``). So the value has
        to be right while python-can builds the bus, and the only seam for that
        is the ``BitTiming.from_sample_point`` it calls. We swap it for the
        duration of that one call and always restore it.

        Yes, this is a monkeypatch. It is scoped to a single constructor call,
        confined to the one backend that needs it, restored in a ``finally``,
        and it uses only public API. The alternative -- letting a robot stream
        position commands over a bus that cannot carry a single frame -- is not
        a trade worth making. Delete this once python-can
        accepts a sample point or a ``BitTiming`` for gs_usb; ``cia_sample_point``
        remains useful either way.
        """
        if self.interface != "gs_usb":
            yield
            return
        if self.bitrate is None:
            raise ValueError(
                "gs_usb needs an explicit bitrate to compute its bit timing"
            )

        wanted = (
            self.sample_point if self.sample_point is not None
            else cia_sample_point(self.bitrate)
        )
        original = can.BitTiming.__dict__["from_sample_point"]
        underlying = original.__func__

        def forced(cls, f_clock, bitrate, sample_point=87.5):
            return underlying(
                cls, f_clock=f_clock, bitrate=bitrate, sample_point=wanted
            )

        can.BitTiming.from_sample_point = classmethod(forced)
        try:
            yield
        finally:
            can.BitTiming.from_sample_point = original

    def close(self) -> None:
        if self._bus is None:
            return
        try:
            if self.interface == "gs_usb":
                self._close_gs_usb()
            else:
                self._bus.shutdown()
        finally:
            self._bus = None

    def _close_gs_usb(self) -> None:
        """Stop a gs_usb adapter without python-can's destructive shutdown.

        ``GsUsbBus.shutdown()`` does not merely stop the device. When it was
        opened by index (ours is), it then runs a "workaround" that re-scans and::

            gs_usb.set_bitrate(self._bitrate)   # hardcoded 87.5% sample point
            gs_usb.start()                      # -> gs_usb.reset(): a USB reset
            gs_usb.stop()

        Both halves hurt us. ``set_bitrate`` reprograms the adapter to the very
        87.5% sample point that cannot carry 1 Mbit/s, undoing the timing we
        installed. Worse, ``start()`` issues a **USB device reset**, and on macOS
        that re-enumerates the adapter and leaves it wedged -- every run then
        needed a physical replug before the next one could see any traffic. (We
        independently hit the same reset trap earlier trying to re-time an open
        device.)

        So we stop the CAN controller and skip the workaround: mark the bus shut
        down first, which makes ``GsUsbBus.shutdown()`` a no-op if anything --
        including python-can's own ``__del__`` -- calls it later, then do the
        base-class bookkeeping and the one operation that is actually wanted.
        """
        gs_usb = getattr(self._bus, "gs_usb", None)
        # Must come first: it is what neutralises the workaround, including on
        # the __del__ path we do not control.
        self._bus._is_shutdown = True
        try:
            can.BusABC.shutdown(self._bus)
        finally:
            if gs_usb is not None:
                gs_usb.stop()

    @property
    def is_open(self) -> bool:
        return self._bus is not None

    def _require_bus(self) -> can.BusABC:
        if self._bus is None:
            raise RuntimeError(
                "CAN transport is not open: call open() before {!r}/{!r}".format(
                    "write", "recv"
                )
            )
        return self._bus

    def write(self, frames: Sequence[Frame]) -> None:
        """Send frames back to back -- one bus transaction per control cycle.

        CAN is inherently broadcast: every node sees every frame, so a batch of
        per-actuator commands is simply a burst of frames with no round trip in
        between. Nothing here waits for a reply.

        A full transmit queue is retried, briefly
        -----------------------------------------
        python-can's ``send(timeout=...)`` does **not** cover this case. Its
        socketcan loop waits for ``select()`` writability and then retries only a
        *partial* write; an ``ENOBUFS`` raised by the underlying ``socket.send``
        propagates straight out of the loop, so the timeout buys no tolerance at
        all. That made one momentary backlog fatal to the whole servicer -- and
        because the servicer carries ``on_exit=Shutdown()``, fatal to the arm,
        which then fell.

        So ENOBUFS specifically is re-offered for up to ``tx_retry_window``.
        Nothing else is: ``ENETDOWN`` (link down) and ``EBADF`` (socket closed)
        are not transient, and retrying them would only delay a correct
        diagnosis. Every retry is counted (:attr:`tx_retries`) so riding out a
        backlog can never look the same as a healthy bus.

        Retrying resumes at the frame that failed. Restarting the burst would
        re-send the frames already on the wire, and while a CubeMars position
        setpoint happens to be idempotent, "the frames before the failure were
        already delivered" is a property of CAN, not of one vendor -- a transport
        must not duplicate traffic on a layer that cannot know whether that is
        safe.
        """
        bus = self._require_bus()
        messages: List[can.Message] = []
        for frame in frames:
            if frame.address is None:
                raise ValueError(
                    "CAN frames need an arbitration id in Frame.address"
                )
            messages.append(
                can.Message(
                    arbitration_id=int(frame.address),
                    is_extended_id=bool(frame.meta.get("extended", False)),
                    data=frame.payload,
                )
            )

        # ONE deadline for the whole burst, not one per frame. The retry window
        # is time stolen from a control cycle, and this loop is what services the
        # bus -- a per-frame window would let an N-joint burst stall for N times
        # the window (8 ms for this 4-DOF arm at the 2 ms default), which blows a
        # 200 Hz cycle outright. Bounding it per burst makes the worst-case cost
        # of retrying independent of how many joints the robot has.
        deadline = time.monotonic() + max(0.0, self.tx_retry_window)
        for index, message in enumerate(messages):
            self._send_one(bus, message, index, len(messages), deadline)

    def _send_one(
        self,
        bus: can.BusABC,
        message: can.Message,
        index: int,
        total: int,
        deadline: float,
    ) -> None:
        """Offer one frame, retrying only while the transmit queue is full."""
        attempts = 0
        while True:
            # The inner timeout must fit INSIDE the remaining budget, or it eats
            # it whole. python-can's socketcan send() spends its timeout blocking
            # in select() before it raises "Transmit buffer full", so a 50 ms
            # send_timeout against a 2 ms window meant the deadline was already
            # 48 ms in the past on the first failure: the retry loop logged
            # "after 1 attempt(s)" and never actually retried. Observed on the
            # 4-DOF arm on 2026-07-30, which is what exposed it.
            remaining = deadline - time.monotonic()
            timeout = self.send_timeout
            if self.tx_retry_window > 0.0:
                timeout = min(timeout, max(_TX_RETRY_PAUSE, remaining))
            try:
                bus.send(message, timeout=timeout)
                return
            except can.CanOperationError as err:
                if not _is_queue_full(err):
                    raise
                attempts += 1
                if attempts == 1:
                    self._tx_stalls += 1
                self._tx_retries += 1
                if time.monotonic() >= deadline:
                    raise TransmitQueueFull(
                        "transmit queue still full after {} attempt(s) within "
                        "this burst's {:.1f} ms retry budget, on frame {} of {} "
                        "(id 0x{:X}): {}. The queue only backs up when the "
                        "controller cannot get frames onto the wire, so check "
                        "the link's error counters first: `ip -s -details link "
                        "show {}`. A climbing error-warn/error-pass means a "
                        "PHYSICAL fault (termination, CAN_H/CAN_L, a missing "
                        "common-mode ground, or noise from motor current) and no "
                        "amount of queue fixes it. Clean counters instead mean "
                        "frames really are being offered faster than the bus "
                        "carries them, i.e. lower the rate.\n"
                        "NOTE `{}` holds {} frames, which for a periodic "
                        "setpoint stream is a depth, not a safety margin: at 4 "
                        "frames per cycle and 50 Hz that is {:.1f} s of STALE "
                        "commands that would be executed in order once the bus "
                        "recovers. A shallow queue that drops is safer than a "
                        "deep one that replays history.".format(
                            attempts,
                            self.tx_retry_window * 1e3,
                            index + 1,
                            total,
                            message.arbitration_id,
                            err,
                            self.channel,
                            self.channel,
                            self.tx_queue_len,
                            (self.tx_queue_len or 0) / 200.0,
                        )
                    ) from err
                time.sleep(_TX_RETRY_PAUSE)

    @property
    def tx_queue_len(self) -> Optional[int]:
        """The kernel's queue depth for this link, or ``None`` if unknowable.

        Only meaningful for ``socketcan``, where the queue belongs to the
        network device rather than to us. Read from sysfs, which needs no
        privileges -- unlike raising it.
        """
        if self.interface != "socketcan":
            return None
        try:
            with open("/sys/class/net/{}/tx_queue_len".format(self.channel)) as handle:
                return int(handle.read().strip())
        except (OSError, ValueError):
            return None

    def link_counters(self) -> Optional[Dict[str, int]]:
        """The kernel's own CAN error counters for this link, or ``None``.

        Exists to settle a question that stats printed a second apart cannot:
        **when a transmit stall happens, is the bus erroring or not?** Two
        different faults present identically as "the transmit queue filled and
        the servicer gave up", and they have opposite fixes:

        * ``error-warn``/``error-pass`` climbing across the stall -> the bus is
          failing electrically (termination, noise from motor current, a
          connector). No amount of queue or retry helps.
        * those counters FLAT across the stall -> nothing is wrong at the CAN
          layer, and the stall is below it (the USB adapter). Queue depth and
          retry budget are then the right lever.

        Both have been seen on a CubeMars arm, and each time the cause had to be
        inferred from stats lines a second apart rather than read. Sampling this
        cheaply every cycle-group and diffing it across the stall replaces the
        inference.

        socketcan only: these counters belong to a kernel netdev, and `ip` is
        their only interface (they come over netlink as CAN xstats, not sysfs).
        Returns ``None`` on any failure -- an absent diagnostic must never be
        able to break a running robot.
        """
        if self.interface != "socketcan":
            return None
        try:
            import subprocess
            out = subprocess.run(
                ["ip", "-s", "-details", "link", "show", str(self.channel)],
                capture_output=True, text=True, timeout=2.0, check=False).stdout
        except (OSError, ValueError, Exception):     # noqa: BLE001
            return None
        lines = out.splitlines()
        counters: Dict[str, int] = {}
        for i, line in enumerate(lines):
            if "re-started" in line and "bus-errors" in line:
                names = line.split()
                if i + 1 < len(lines):
                    values = lines[i + 1].split()
                    for name, value in zip(names, values):
                        try:
                            counters[name] = int(value)
                        except ValueError:
                            break
                break
        # The generic netdev counters live in sysfs and cost nothing.
        for key in ("tx_dropped", "tx_errors", "rx_errors", "rx_dropped"):
            try:
                with open("/sys/class/net/{}/statistics/{}".format(
                        self.channel, key)) as handle:
                    counters[key] = int(handle.read().strip())
            except (OSError, ValueError):
                pass
        return counters or None

    def tx_queue_advice(self, frames_per_second: float = 200.0) -> Optional[str]:
        """Warn if this link's transmit queue can hold a dangerous backlog.

        Deliberately the OPPOSITE of the obvious advice, and this method said the
        obvious thing for one day before hardware corrected it. The reflex on a
        full transmit queue is to raise ``txqueuelen``, and for bulk traffic that
        is right -- but this queue carries *periodic position setpoints*, where a
        backlog is not a buffer, it is a delay line. Raised to 1000 on the 4-DOF
        arm it held five seconds of commands, which the motors would execute **in
        order** once the bus recovered: a bus glitch turns into the arm replaying
        five seconds of history. Dropping a stale setpoint is always better than
        queueing it, because the next cycle carries a fresher one 20 ms later.

        Returns text rather than printing it: which stream an operator warning
        belongs on is the caller's business, not a transport's.

        Parameters
        ----------
        frames_per_second : float
            Offered frame rate, used only to turn a queue depth into the seconds
            of staleness it represents. Default 200 = 4 joints at 50 Hz.
        """
        depth = self.tx_queue_len
        if depth is None or frames_per_second <= 0.0:
            return None
        seconds = depth / frames_per_second
        if seconds < 0.2:
            return None
        return (
            "{} has txqueuelen={}, which at {:.0f} frames/s is {:.1f} s of "
            "setpoints. This queue carries periodic position commands, so a "
            "backlog is a delay line, not a buffer: after a bus stall the motors "
            "would execute every stale command in order. Prefer a shallow queue "
            "that drops (sudo ip link set {} txqueuelen 20) over a deep one that "
            "replays history.".format(
                self.channel, depth, frames_per_second, seconds, self.channel
            )
        )

    def recv(self, timeout: float) -> Optional[Frame]:
        """Return the next frame seen on the bus, or ``None`` on timeout.

        The caller decides which frames are interesting; this layer does not
        filter by id, because which ids mean what is actuator knowledge.

        Error frames are passed up rather than dropped, flagged in ``meta``.
        A controller mis-reading a bus at the wrong bitrate sees a storm of
        error frames with nonsense ids; silently presenting those as data would
        turn "your bitrate is wrong" into "your motor is at node 12", which is
        exactly the kind of masking a driver must never do: failures surface
        loudly instead of being disguised as plausible data.
        """
        bus = self._require_bus()
        deadline = time.monotonic() + timeout
        while True:
            # The deadline bounds how long we may BLOCK, nothing else. Always
            # attempt a read: timeout=0.0 means "take what is already buffered",
            # never "return without reading". And skipping an echo must not end
            # the call either -- echoes sit at the head of the queue right after
            # a write, so treating one as a timeout stopped every zero-timeout
            # drain dead. Measured: 51 cycles drained 49 frames while the motors
            # were sending ~390/s, the adapter's queues backed up, and the
            # failure surfaced as a USB *write* timeout on cycle 63.
            msg = bus.recv(timeout=max(0.0, deadline - time.monotonic()))
            if msg is None:
                return None

            # Some backends hand our OWN transmitted frames back rather than
            # honouring receive_own_messages=False -- python-can's gs_usb marks
            # them is_rx=False and delivers them anyway. Silently letting those
            # through is not cosmetic: a CubeMars command frame carries the node
            # id in its arbitration id and 8 bytes of payload, so it looks
            # exactly like a status frame and decodes into a fictional motor
            # state. Honour the flag here rather than let every caller re-learn
            # this.
            if not self.receive_own_messages and not getattr(msg, "is_rx", True):
                continue  # our own echo; keep draining, do not report a timeout

            return Frame(
                payload=bytes(msg.data),
                address=msg.arbitration_id,
                meta={
                    "extended": msg.is_extended_id,
                    "timestamp": msg.timestamp,
                    "error_frame": bool(msg.is_error_frame),
                    "remote_frame": bool(msg.is_remote_frame),
                    "is_rx": bool(getattr(msg, "is_rx", True)),
                    "dlc": msg.dlc,
                },
            )
