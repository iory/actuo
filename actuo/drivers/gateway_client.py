"""Talks to a servo gateway -- a flashed board, or the firmware's simulator.

The gateway firmware lives in a separate repository and is not included in
this package. The point of this module is that a board and the simulator are
the *same object* to everything above it. The firmware's native
``gateway_sim`` build speaks the protocol on
a pipe; a Kondo ARM-H7 speaks it on a UART. Driving both from one client is what
makes a differential test possible: the same command script produces two
transcripts, and any divergence is either the fake bus lying about the hardware
or a firmware bug the fake cannot see.

That matters because hardware time is the scarce resource. The firmware's 131
native tests settle everything the fake bus can decide; what is left for a real
board is line settings, turnaround latency, and whether an ICS servo actually
behaves the way the fake assumes. A diff finds those, and nothing else needs to
run on hardware at all.
"""

from __future__ import annotations

import subprocess
import time
from typing import List, Optional, Tuple

from actuo.drivers import gateway_protocol as gw


class GatewayError(RuntimeError):
    """The gateway answered with NACK, or did not answer at all."""


class NackError(GatewayError):
    def __init__(self, command: int, nack: gw.Nack) -> None:
        super().__init__("command {:#04x} rejected: {} (detail {})".format(
            command, nack.code.name, nack.detail))
        self.command = command
        self.nack = nack


class Link:
    """A byte pipe to a gateway. Two implementations; nothing above cares."""

    def write(self, data: bytes) -> None:
        raise NotImplementedError

    def read(self, timeout: float) -> bytes:
        """Returns whatever bytes have arrived, or b'' on timeout."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class SerialLink(Link):
    """A real board over a UART. On the Kondo ARM-H7 this is UART8 at 1 Mbit/s."""

    def __init__(self, port: str, baudrate: int = 1_000_000,
                 timeout: float = 0.05) -> None:
        import serial  # imported here so the simulator path needs no pyserial
        self._serial = serial.Serial(port, baudrate, timeout=timeout)
        self._timeout = timeout
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

    def write(self, data: bytes) -> None:
        self._serial.write(data)
        self._serial.flush()

    def read(self, timeout: float) -> bytes:
        # Assign the timeout only when it actually changes.
        #
        # pyserial's `timeout` setter re-runs `_reconfigure_port()`, and on
        # macOS a non-standard rate like this bus's 1 Mbit/s is applied with an
        # `IOSSIOSPEED` ioctl -- so re-setting it discards whatever the driver
        # had already buffered. This method is the poll loop, called every
        # 50 ms, and the board answers in ~55 ms: every reply was being wiped
        # out by the next poll's reconfigure, one poll before it could be read.
        #
        # The board looked dead and was not. Measured on a Kondo ARM-H7 running
        # the gateway firmware: raw pyserial answered 5/5, this method 0/5,
        # interleaved in one process; restoring the assignment reproduces the
        # failure exactly. Linux does not show it, which is precisely why it
        # survived this long -- and since one configuration must serve both
        # macOS and Linux, that makes it a bug rather than a quirk.
        if timeout != self._timeout:
            self._serial.timeout = timeout
            self._timeout = timeout
        waiting = self._serial.in_waiting
        return self._serial.read(waiting if waiting > 0 else 1)

    def close(self) -> None:
        self._serial.close()


class SimulatorLink(Link):
    """The firmware itself, running as a process on the host.

    Not a mock of the gateway -- it *is* the gateway, compiled for the host,
    driving an emulated ICS bus. The only thing that differs from a flashed
    board is the physical layer, which is precisely the part a diff is meant to
    put under test.
    """

    def __init__(self, binary: str, servos: str = "0",
                 fail_after: Optional[str] = None,
                 cycles_per_frame: Optional[int] = None) -> None:
        argv = [binary, "--servos", servos]
        if fail_after is not None:
            argv += ["--fail-after", fail_after]
        if cycles_per_frame is not None:
            argv += ["--cycles-per-frame", str(cycles_per_frame)]
        self._process = subprocess.Popen(
            argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

    def write(self, data: bytes) -> None:
        assert self._process.stdin is not None
        self._process.stdin.write(data)
        self._process.stdin.flush()

    def read(self, timeout: float) -> bytes:
        import select
        assert self._process.stdout is not None
        ready, _, _ = select.select([self._process.stdout], [], [], timeout)
        if not ready:
            return b""
        # `bufsize=0` gives a raw FileIO, whose read() returns whatever is
        # available rather than blocking for the full count.
        return self._process.stdout.read(4096) or b""

    def close(self) -> None:
        if self._process.stdin is not None:
            self._process.stdin.close()
        self._process.wait(timeout=2)


class GatewayClient:
    """Request/response over a :class:`Link`, with the deferred replies handled.

    Three commands are answered later rather than inline (`ENUMERATE`,
    `GET_PARAM`, `SET_PARAM`), because they need the servo bus and the bus
    belongs to the port task. The protocol echoes the request's sequence number,
    so pairing is by `seq` -- which is also why this client never reuses one
    while a reply is outstanding.
    """

    def __init__(self, link: Link, timeout: float = 2.0) -> None:
        self.link = link
        self.timeout = timeout
        self._reader = gw.FrameReader()
        self._seq = 0
        self._pending: List[gw.Frame] = []

    def close(self) -> None:
        self.link.close()

    def _next_seq(self) -> int:
        self._seq = (self._seq + 1) & 0xFF
        return self._seq

    def request(self, command: gw.Command, payload: bytes = b"",
                timeout: Optional[float] = None) -> gw.Frame:
        """Sends a command and waits for the frame carrying its sequence number.

        A `NACK` raises: a refusal is an outcome the caller must not be able to
        mistake for success by forgetting to inspect a field.
        """
        deadline = time.monotonic() + (self.timeout if timeout is None else timeout)
        seq = self._next_seq()
        self.link.write(gw.encode_frame(int(command), seq, payload))

        while True:
            for frame in list(self._pending):
                if frame.seq == seq:
                    self._pending.remove(frame)
                    return self._checked(int(command), frame)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GatewayError(
                    "no reply to {} (seq {}) within {:.2f} s".format(
                        command.name, seq, self.timeout))
            data = self.link.read(min(remaining, 0.05))
            if data:
                self._pending.extend(self._reader.feed(data))

    @staticmethod
    def _checked(command: int, frame: gw.Frame) -> gw.Frame:
        if frame.command == int(gw.Command.NACK):
            raise NackError(command, gw.decode_nack(frame.payload))
        return frame

    # -- typed commands ------------------------------------------------------

    def hello(self) -> gw.Hello:
        return gw.decode_hello(self.request(gw.Command.HELLO).payload)

    def ping(self, echo: bytes = b"") -> Tuple[int, bytes]:
        payload = self.request(gw.Command.PING, echo).payload
        return int.from_bytes(payload[:4], "little"), payload[4:]

    def enumerate(self, port_mask: int = 0x3F,
                  timeout: float = 10.0) -> gw.Topology:
        """Rescans the selected ports. Slow by nature -- one probe per control
        cycle -- so it gets its own, longer timeout."""
        frame = self.request(gw.Command.ENUMERATE,
                             gw.encode_port_mask(port_mask), timeout=timeout)
        return gw.decode_topology(frame.payload)

    def topology(self, port_mask: int = 0x3F) -> gw.Topology:
        frame = self.request(gw.Command.GET_TOPOLOGY,
                             gw.encode_port_mask(port_mask))
        return gw.decode_topology(frame.payload)

    def write_targets(
            self, targets: List[Tuple[int, int, int, bool]]) -> gw.StateReply:
        frame = self.request(gw.Command.WRITE_TARGETS, gw.encode_targets(targets))
        return gw.decode_state(frame.payload)

    def read_state(self, port_mask: int = 0x3F) -> gw.StateReply:
        frame = self.request(gw.Command.READ_STATE, gw.encode_port_mask(port_mask))
        return gw.decode_state(frame.payload)

    def set_torque(self, entries: List[Tuple[int, int, bool]]) -> None:
        self.request(gw.Command.SET_TORQUE, gw.encode_torque(entries))

    def get_param(self, port: int, servo_id: int, param: int) -> int:
        frame = self.request(gw.Command.GET_PARAM,
                             gw.encode_param_request(port, servo_id, param))
        return gw.decode_param(frame.payload).value

    def set_param(self, port: int, servo_id: int, param: int,
                  value: int) -> None:
        self.request(gw.Command.SET_PARAM,
                     gw.encode_param(port, servo_id, param, value))
