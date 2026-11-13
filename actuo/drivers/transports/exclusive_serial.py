"""Open a serial port that no other process can write to while it is open.

A request/response board answers whoever spoke last. A second process on the
same port -- a terminal left open, a probe tool, another bus servicer -- makes
replies go to the wrong reader and requests overlap on the line, and nothing on
either side reports it as such: it looks like a board that stops answering.

Two locks, because each misses what the other catches:

* ``exclusive=True`` (pyserial): an advisory ``flock``. It stops only other
  openers that ask for it too, but it is portable across POSIX systems.
* ``TIOCEXCL`` (Linux, macOS): the kernel refuses any further ``open`` of the
  device from a process without CAP_SYS_ADMIN, whatever that process asks
  for. Released when the port is closed.

Both raise at open time, naming the port, instead of corrupting a running bus.
"""

from __future__ import annotations

import serial

try:
    import fcntl
    import termios
except ImportError:  # Windows: no flock / TIOCEXCL; a COM port is exclusive
    fcntl = None
    termios = None

#: ``TIOCEXCL`` where the platform has it (Linux, macOS); ``None`` elsewhere.
_TIOCEXCL = getattr(termios, "TIOCEXCL", None)


def open_exclusive(channel: str, bitrate: int, **kwargs) -> serial.Serial:
    """Open ``channel`` for this process only.

    Parameters
    ----------
    channel : str
        Serial device path.
    bitrate : int
        Baud rate.
    **kwargs
        Passed to :class:`serial.Serial` (``timeout``, ``parity``, ...).

    Returns
    -------
    serial.Serial
        The open port.

    Raises
    ------
    serial.SerialException
        The port is already held by another process (or cannot be opened).
    """
    try:
        port = serial.Serial(channel, bitrate,
                             exclusive=True if fcntl is not None else None,
                             **kwargs)
    except serial.SerialException as exc:
        raise serial.SerialException(
            "{}: cannot open for exclusive use ({}). Another process may "
            "have it open: `fuser -v {}` lists it.".format(
                channel, exc, channel)) from exc
    if _TIOCEXCL is not None:
        try:
            fcntl.ioctl(port.fileno(), _TIOCEXCL)
        except OSError:
            # Not a tty (a socket-backed test double, say): the flock above
            # still holds.
            pass
    return port
