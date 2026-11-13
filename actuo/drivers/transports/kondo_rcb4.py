"""Serial transport to the Kondo RCB-4mini control board.

The packet framing and the request/response discipline are the ARM-H7's (see
``transports/kondo_h7.py``), so this is that transport with two differences
the RCB-4mini needs:

* the COM port runs **even parity**; with none, the board does not answer;
* the host bit rate is a board RAM setting that the bus may change, so the
  port can be reopened at another rate (:meth:`reopen`).

Provenance: ``actuo/drivers/references.md``, section "Kondo RCB-4mini".
"""

from __future__ import annotations

import serial

from actuo.core.registry import register_transport

from actuo.drivers.transports.kondo_h7 import KondoH7Transport

#: The rate the bus switches the board to. The board itself powers up at the
#: rate in its ROM config (115200 on a board without a HeartToHeart project);
#: ``KondoRcb4Bus`` connects at that and switches.
DEFAULT_BITRATE = 1_250_000


@register_transport("kondo_rcb4")
class KondoRcb4Transport(KondoH7Transport):
    """Length-prefixed packet transport over the RCB-4mini's COM port.

    Parameters
    ----------
    channel : str
        Serial device path, e.g. ``/dev/ttyUSB0``.
    bitrate : int
        Rate to run at once connected.
    timeout : float
        Default per-packet read timeout in seconds.
    """

    def __init__(self, channel: str, bitrate: int = DEFAULT_BITRATE,
                 timeout: float = 0.1) -> None:
        super().__init__(channel, bitrate=bitrate, timeout=timeout)
        #: Rate the port is open at right now; differs from ``bitrate`` while
        #: the bus is talking to a board that has not been switched yet.
        self.current_bitrate = self.bitrate

    def open(self) -> None:
        if self._serial is not None:
            return
        self._serial = serial.Serial(
            self.channel, self.current_bitrate, parity=serial.PARITY_EVEN,
            timeout=self.timeout)
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

    def reopen(self, bitrate: int) -> None:
        """Close the port and open it again at ``bitrate``."""
        self.close()
        self.current_bitrate = int(bitrate)
        self.open()
