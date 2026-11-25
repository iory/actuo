"""Hardware that makes servos vanish on purpose, behind one interface.

A robot whose modules are plugged together at run time (``hot_plug`` buses,
:mod:`actuo.drivers.robot_modes`) is only tested once modules really come and
go. A fault injector sits in the servo line and does that from software:

``unplug(target)``
    The servo stops answering and receives nothing, as if its cable were out.
``plug(target)``
    Undo ``unplug``.
``drop_replies(target, servo_id)``
    Commands still reach the servo (it moves), but its replies never reach the
    controller: a servo that is driven but reads as gone.
``restore()``
    Every servo plugged and answering, the line a plain cable again.

What a *target* is depends on the hardware: a port of a
:class:`ServoFaultInjector`, a servo id for an :class:`IcsFaultRelay`.
:class:`actuo.testing.rig.ServoRig` maps joint names to targets so tests speak
in joints.

Both injectors keep their USB serial port open: opening the Nano Every's port
resets it to pass-through, and holding one port per object keeps two processes
from driving one injector.
"""

from __future__ import annotations

import abc
import time
from typing import FrozenSet, List, Optional, Sequence, Tuple

import serial
from serial.tools import list_ports

#: Capabilities an injector may offer.
UNPLUG = "unplug"                  # make one servo vanish
UNPLUG_MANY = "unplug_many"        # several at once
DROP_REPLIES = "drop_replies"      # commands in, replies cut
POWER = "power"                    # the servo really loses power


class FaultInjectorError(RuntimeError):
    """The injector refused, did not answer, or cannot do what was asked."""


class FaultInjector(abc.ABC):
    """Something in a servo line that can make one servo vanish."""

    #: What this injector can do (see the module constants).
    capabilities: FrozenSet[str] = frozenset()

    @abc.abstractmethod
    def unplug(self, target: int) -> None:
        """Make the servo at ``target`` stop answering and receive nothing."""

    @abc.abstractmethod
    def plug(self, target: int) -> None:
        """Undo :meth:`unplug` for ``target``."""

    @abc.abstractmethod
    def drop_replies(self, target: int, servo_id: int) -> None:
        """Let commands reach servo ``servo_id`` at ``target``, cut its replies."""

    @abc.abstractmethod
    def restore(self) -> None:
        """Every servo plugged and answering; nothing dropped."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the injector's port (the line is left as it is)."""

    def require(self, capability: str) -> None:
        """Raise unless this injector offers ``capability``."""
        if capability not in self.capabilities:
            raise FaultInjectorError("{} cannot {}".format(type(self).__name__,
                                                           capability))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        try:
            self.restore()
        finally:
            self.close()


def _number(text: str):
    """``"12"`` -> 12, ``"11.99"`` -> 11.99, anything else as it is."""
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def _only_port(vid_pid: Tuple[int, int], what: str) -> str:
    found = [p.device for p in list_ports.comports() if (p.vid, p.pid) == vid_pid]
    if len(found) != 1:
        raise FaultInjectorError(
            "expected one {} ({:04x}:{:04x}), found {}; pass the port".format(
                what, vid_pid[0], vid_pid[1], found or "none"))
    return found[0]


class ServoFaultInjector(FaultInjector):
    """iory/servo-fault-injector: a Pico 2 board with four servo ports.

    Each port has its own signal switch (TMUX1102) and power switch. A target
    is a port number, 1-4. The firmware speaks one command per line over USB
    serial and ends every reply with one ``ok ...`` or ``error ...`` line.

    Parameters
    ----------
    port : str, optional
        USB serial device of the Pico; found by VID 2e8a when there is exactly
        one.
    unplug_by : {"power", "signal"}
        How :meth:`unplug` makes a servo vanish. ``power`` switches the port
        off (the firmware also isolates its signal), so the servo really
        reboots when plugged back, as a module pushed in again would.
        ``signal`` only opens the switch: the servo keeps power and state.
        On the first v0.1 board PORT 4's switch did not open (2026-11-24), so
        ``power`` is the default.
    timeout : float
        Seconds to wait for a reply line.
    settle : float
        Seconds :meth:`plug` waits after powering a port, for the servo to boot.
    signal_after_s : float
        With ``unplug_by: power``, seconds a port's signal stays cut after its
        power comes on (:meth:`plug`, :meth:`restore`): a connector whose power
        pins mate first. 0 (default) connects both at once, as a plain plug
        does. A KRS servo powered with its signal on an RCB-4mini's live ICS
        line sometimes starts in a state that blocks the whole line until it
        is powered off (19 of 30 power-ons of three servos at once, against
        30 of 30 with the signal 1 s later, 2026-11-25). The firmware connects
        the signal with the power and this cuts it again a command later, so
        the servo sees the line for about a millisecond.
    """

    VID = 0x2E8A
    PORTS = (1, 2, 3, 4)
    capabilities = frozenset({UNPLUG, UNPLUG_MANY, DROP_REPLIES, POWER})

    def __init__(self, port: Optional[str] = None, unplug_by: str = "power",
                 timeout: float = 1.0, settle: float = 0.0,
                 signal_after_s: float = 0.0) -> None:
        if unplug_by not in ("power", "signal"):
            raise ValueError("unplug_by must be 'power' or 'signal', got {!r}"
                             .format(unplug_by))
        self.unplug_by = unplug_by
        self.timeout = float(timeout)
        self.settle = float(settle)
        self.signal_after_s = float(signal_after_s)
        if self.signal_after_s < 0:
            raise ValueError("signal_after_s must be >= 0")
        if self.signal_after_s and unplug_by != "power":
            raise ValueError("signal_after_s needs unplug_by: power")
        if port is None:
            found = [p.device for p in list_ports.comports() if p.vid == self.VID]
            if len(found) != 1:
                raise FaultInjectorError(
                    "expected one servo-fault-injector (VID 2e8a), found {}; "
                    "pass the port".format(found or "none"))
            port = found[0]
        self._serial = serial.Serial(port, 115200, timeout=self.timeout)

    def command(self, line: str) -> List[str]:
        """Send one command; return the reply lines (the last starts with ok).

        Raises
        ------
        FaultInjectorError
            The board answered ``error ...`` or did not finish its reply.
        """
        self._serial.reset_input_buffer()
        self._serial.write((line + "\n").encode())
        lines: List[str] = []
        deadline = time.monotonic() + self.timeout
        while time.monotonic() < deadline:
            text = self._serial.readline().decode(errors="replace").strip()
            if not text:
                continue
            lines.append(text)
            if text.startswith("error"):
                raise FaultInjectorError("board refused {!r}: {}".format(line, text))
            if text.startswith("ok"):
                return lines
        raise FaultInjectorError("no reply to {!r} (got {})".format(line, lines))

    def _check(self, target: int) -> int:
        if int(target) not in self.PORTS:
            raise FaultInjectorError("port must be one of {}, got {}".format(
                self.PORTS, target))
        return int(target)

    def unplug(self, target: int) -> None:
        port = self._check(target)
        if self.unplug_by == "power":
            self.command("power {} off".format(port))
        else:
            self.command("isolate {}".format(port))

    def plug(self, target: int) -> None:
        port = self._check(target)
        if self.unplug_by == "power":
            self._power_on([port])
            if self.settle > 0:
                time.sleep(self.settle)
        else:
            self.command("unisolate {}".format(port))

    def drop_replies(self, target: int, servo_id: int) -> None:
        self.command("drop {} {}".format(self._check(target), int(servo_id)))

    def _power_on(self, ports: Sequence[int]) -> None:
        """Power ``ports`` on; with ``signal_after_s`` their signal follows."""
        for port in ports:
            self.command("power {} on".format(port))
            if self.signal_after_s:
                self.command("isolate {}".format(port))
        if self.signal_after_s and ports:
            time.sleep(self.signal_after_s)
            for port in ports:
                self.command("unisolate {}".format(port))

    def restore(self) -> None:
        if self.signal_after_s:
            self._power_on([int(p["n"]) for p in self.status() if not p.get("power")])
        else:
            self.command("power all on")
        self.command("pass")

    def status(self) -> List[dict]:
        """Per port: ``n``, ``power``, ``isolated``, ``volt``, ``ma``."""
        out = []
        for line in self.command("status")[:-1]:
            kind, _, rest = line.partition(" ")
            if kind != "port":
                continue
            fields = dict(f.split("=", 1) for f in rest.split() if "=" in f)
            out.append({k: _number(v) for k, v in fields.items()})
        return out

    def close(self) -> None:
        self._serial.close()


class IcsFaultRelay(FaultInjector):
    """iory/ics-fault-relay: an Arduino Nano Every in a Kondo ICS line.

    A target is the servo id (0-31). The relay holds at most one ``isolate``
    and one ``drop`` at a time, so unplugging a second servo is refused here
    rather than silently replacing the first. 115200 bit/s ICS only.
    Opening the port resets the relay to pass-through.

    Parameters
    ----------
    port : str, optional
        USB serial device of the Nano Every; found by VID:PID 2341:0058 when
        there is exactly one.
    timeout : float
        Seconds to wait for each reply; a command without one is sent again.
    attempts : int
        Times a command is sent before giving up.
    """

    VID_PID = (0x2341, 0x0058)
    MAX_ID = 31
    capabilities = frozenset({UNPLUG, DROP_REPLIES})

    def __init__(self, port: Optional[str] = None, timeout: float = 0.3,
                 attempts: int = 10) -> None:
        self.timeout = float(timeout)
        self.attempts = int(attempts)
        self._serial = serial.Serial(port or _only_port(self.VID_PID, "ICS fault relay"),
                                     115200, timeout=self.timeout)
        self._isolated: Optional[int] = None

    def command(self, line: str) -> str:
        """Send ``!line``; return the relay's one-line reply."""
        self._serial.reset_input_buffer()
        for _ in range(self.attempts):
            self._serial.write("!{}\n".format(line).encode())
            deadline = time.monotonic() + self.timeout
            while time.monotonic() < deadline:
                reply = self._serial.readline().decode(errors="replace").strip()
                if reply.startswith("error"):
                    raise FaultInjectorError("relay refused {!r}: {}".format(line, reply))
                if reply.startswith("ok"):
                    return reply
        raise FaultInjectorError("relay did not answer {!r}".format(line))

    def _check(self, servo_id: int) -> int:
        if not 0 <= int(servo_id) <= self.MAX_ID:
            raise FaultInjectorError("ICS id must be 0-{}, got {}".format(
                self.MAX_ID, servo_id))
        return int(servo_id)

    def unplug(self, target: int) -> None:
        servo_id = self._check(target)
        if self._isolated is not None and self._isolated != servo_id:
            raise FaultInjectorError(
                "the ICS relay isolates one servo at a time; id {} is already "
                "unplugged".format(self._isolated))
        self.command("relay")
        self.command("isolate {}".format(servo_id))
        self._isolated = servo_id

    def plug(self, target: int) -> None:
        servo_id = self._check(target)
        if self._isolated == servo_id:
            # `nodrop` clears isolate and drop; `relay` only sets the mode.
            self.command("nodrop")
            self._isolated = None

    def drop_replies(self, target: int, servo_id: int) -> None:
        self.command("drop {}".format(self._check(servo_id)))

    def restore(self) -> None:
        self.command("relay")
        self.command("nodrop")
        self._isolated = None

    def close(self) -> None:
        self._serial.close()


def make_injector(spec: dict) -> FaultInjector:
    """Build an injector from a rig file's ``injector`` block.

    ``plugin`` is ``servo_fault_injector`` or ``ics_fault_relay``; the other
    keys are the constructor's parameters.
    """
    spec = dict(spec)
    plugin = spec.pop("plugin", None)
    makers = {"servo_fault_injector": ServoFaultInjector,
              "ics_fault_relay": IcsFaultRelay}
    if plugin not in makers:
        raise ValueError("injector.plugin must be one of {}, got {!r}".format(
            sorted(makers), plugin))
    return makers[plugin](**spec)


__all__: Sequence[str] = (
    "DROP_REPLIES", "POWER", "UNPLUG", "UNPLUG_MANY", "FaultInjector",
    "FaultInjectorError", "IcsFaultRelay", "ServoFaultInjector", "make_injector")
