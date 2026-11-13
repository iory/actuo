"""Decoding SocketCAN error frames, and not blaming the robot for our own frame.

    pytest tests/test_can_errors.py -q
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(__file__)

from actuo.drivers.can_errors import decode, explain  # noqa: E402

# CAN_ERR_CRTL | CAN_ERR_ACK -- what a real unACKed transmit looks like on
# this adapter, captured live on 2026-07-30.
_CTRL_AND_NO_ACK = 0x00000004 | 0x00000020


def _frame(tx_errors, rx_errors, controller=0x20):
    """An error frame with the given counters. controller default = TX passive."""
    data = bytearray(8)
    data[1] = controller
    data[6] = tx_errors
    data[7] = rx_errors
    return bytes(data)


def test_the_error_counters_come_from_the_documented_bytes():
    decoded = decode(_CTRL_AND_NO_ACK, _frame(tx_errors=128, rx_errors=0))
    assert decoded["tx_errors"] == 128
    assert decoded["rx_errors"] == 0
    assert "no ACK on transmission" in decoded["classes"]
    assert "TX error passive" in decoded["controller"]


def test_a_short_error_frame_reports_unknown_counters_rather_than_zero():
    """`None` and `0` mean opposite things to the hint below, so never guess."""
    decoded = decode(_CTRL_AND_NO_ACK, bytes(4))
    assert decoded["tx_errors"] is None
    assert decoded["rx_errors"] is None
    # ...and the rx_err=0 hint must not fire on an absent counter.
    assert not any("rx_err=0" in h for h in explain(decoded))


def test_our_own_stale_transmission_is_not_reported_as_a_wiring_fault():
    """Regression for a real misdiagnosis, 2026-07-30.

    tx_err>0 with rx_err==0 means every error is our own unACKed transmit and
    nothing corrupted has been received. Because a CAN controller retransmits
    an unACKed frame forever, ONE stale frame produced 4600 error frames/s with
    the host sending nothing -- read at the time as a live bus fault, and it was
    a leftover from a flood test. The hint must say so, including that a link
    down/up does not clear it.
    """
    hints = explain(decode(_CTRL_AND_NO_ACK, _frame(tx_errors=128, rx_errors=0)))
    text = " ".join(hints)
    assert "rx_err=0" in text
    assert "our OWN transmission" in text
    assert "replug" in text
    # It must not stop at the older hint, which reads as being about the robot.
    assert "will NOT stop on its own" in text


def test_a_receiving_fault_is_still_reported_as_a_bus_fault():
    """rx_err>0 means the wire really is corrupting incoming bits."""
    hints = explain(decode(_CTRL_AND_NO_ACK, _frame(tx_errors=8, rx_errors=96)))
    text = " ".join(hints)
    assert "not hearing us" in text
    # Our-own-frame caveat must NOT appear: something else is transmitting.
    assert "our OWN transmission" not in text


def test_bit_level_corruption_points_at_termination():
    data = bytearray(8)
    data[2] = 0x04                       # bit stuffing error
    decoded = decode(0x00000008, bytes(data))
    text = " ".join(explain(decoded))
    assert "termination" in text
    assert "CAN_H/CAN_L not swapped" in text


def test_an_unclassified_error_still_says_something_actionable():
    hints = explain(decode(0x00000000, b""))
    assert len(hints) == 1
    assert "bitrate" in hints[0]
