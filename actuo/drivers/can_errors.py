"""Decode SocketCAN-style error frames into a human answer.

python-can surfaces bus faults as messages with ``is_error_frame=True``, whose
arbitration id is a bitmask of error *classes* and whose 8 data bytes carry the
detail. Left raw, that looks like "a motor at node 12" -- a wrong and dangerous
reading. Decoded, it says which layer is broken.

Reference: Linux kernel ``include/uapi/linux/can/error.h``
https://github.com/torvalds/linux/blob/master/include/uapi/linux/can/error.h
"""

from __future__ import annotations

from typing import Dict, List

# Error classes, carried in the arbitration id (error.h CAN_ERR_*).
ERROR_CLASS = {
    0x00000001: "TX timeout",
    0x00000002: "lost arbitration",
    0x00000004: "controller problem",
    0x00000008: "protocol violation",
    0x00000010: "transceiver status",
    0x00000020: "no ACK on transmission",
    0x00000040: "bus off",
    0x00000080: "bus error",
    0x00000100: "controller restarted",
}

# data[1] -- controller status (CAN_ERR_CRTL_*)
CTRL_STATUS = {
    0x01: "RX buffer overflow",
    0x02: "TX buffer overflow",
    0x04: "RX error warning level",
    0x08: "TX error warning level",
    0x10: "RX error passive",
    0x20: "TX error passive",
    0x40: "back to error active",
}

# data[2] -- protocol error type (CAN_ERR_PROT_*)
PROT_TYPE = {
    0x01: "single bit error",
    0x02: "frame format error",
    0x04: "bit stuffing error",
    0x08: "unable to send dominant bit",
    0x10: "unable to send recessive bit",
    0x20: "bus overload",
    0x40: "active error announcement",
    0x80: "error occurred on transmission",
}

# data[3] -- where in the frame (CAN_ERR_PROT_LOC_*)
PROT_LOC = {
    0x00: "unspecified",
    0x03: "start of frame",
    0x02: "ID bits 28-21",
    0x06: "ID bits 20-18",
    0x04: "substitute RTR",
    0x05: "identifier extension",
    0x07: "ID bits 17-13",
    0x0F: "ID bits 12-5",
    0x0E: "ID bits 4-0",
    0x0C: "RTR",
    0x0D: "reserved bit 1",
    0x09: "reserved bit 0",
    0x0B: "data length code",
    0x0A: "data section",
    0x08: "CRC sequence",
    0x18: "CRC delimiter",
    0x19: "ACK slot",
    0x1B: "ACK delimiter",
    0x1A: "end of frame",
    0x12: "intermission",
}


def _flags(value: int, table: Dict[int, str]) -> List[str]:
    return [name for bit, name in table.items() if value & bit]


def decode(arbitration_id: int, data: bytes) -> Dict[str, object]:
    """Decode one error frame into classes, detail and error counters."""
    out: Dict[str, object] = {
        "classes": _flags(arbitration_id, ERROR_CLASS),
        "controller": [],
        "protocol": [],
        "location": None,
        "tx_errors": None,
        "rx_errors": None,
    }
    if len(data) >= 2:
        out["controller"] = _flags(data[1], CTRL_STATUS)
    if len(data) >= 3:
        out["protocol"] = _flags(data[2], PROT_TYPE)
    if len(data) >= 4:
        out["location"] = PROT_LOC.get(data[3], "0x{:02X}".format(data[3]))
    if len(data) >= 8:
        # error.h: data[6] = TX error counter, data[7] = RX error counter
        out["tx_errors"] = data[6]
        out["rx_errors"] = data[7]
    return out


def explain(decoded: Dict[str, object]) -> List[str]:
    """Turn a decoded error into concrete things to check, most likely first.

    Deliberately opinionated: an error storm at the correct bitrate is nearly
    always physical, and the frame tells us which physical fault.
    """
    hints: List[str] = []
    protocol = decoded.get("protocol") or []
    classes = decoded.get("classes") or []

    bit_level = {"bit stuffing error", "single bit error", "frame format error"}
    if bit_level.intersection(protocol):
        hints.append(
            "Bits are being corrupted on the wire. At the correct bitrate this "
            "is physical: check 120 ohm termination at BOTH ends of the bus "
            "(CubeMars AK motors have none built in), CAN_H/CAN_L not swapped, "
            "a shared ground between adapter and motors, and stub length."
        )
    if "no ACK on transmission" in classes:
        hints.append(
            "Nobody acknowledged a transmission -- the other nodes are not "
            "hearing us (wiring, or they are on a different bitrate)."
        )
        # tx_errors>0 with rx_errors==0 means every error is OUR OWN unACKed
        # transmit and we have not received a single corrupted bit. That is
        # evidence about our frame, not about the robot -- and it is a trap,
        # because a CAN controller retransmits an unACKed frame FOREVER. One
        # stale frame therefore generates an unbounded error storm that hides
        # whatever the motors are or are not doing, and looks exactly like a
        # live wiring fault. Measured 2026-07-30: 4600 error frames/s with the
        # host sending nothing, error-pass past 5.2 million, which read as
        # "the bus is failing" and was one queued frame from a flood test.
        if decoded.get("rx_errors") == 0 and (decoded.get("tx_errors") or 0) > 0:
            hints.append(
                "NOTE rx_err=0: every error here is our OWN transmission being "
                "retried, and nothing corrupted has been received. If frames "
                "were sent before this, the storm may be one stale unACKed "
                "frame retransmitting forever rather than a live fault -- it "
                "will NOT stop on its own, and `ip link set can0 down/up` does "
                "not clear it on gs_usb. Physically replug the adapter, then "
                "listen again before concluding anything about the wiring."
            )
    if "bus off" in classes:
        hints.append(
            "The controller took itself off the bus after too many errors. "
            "Power-cycle the adapter after fixing the wiring."
        )
    if {"RX error passive", "TX error passive"}.intersection(
        decoded.get("controller") or []
    ):
        hints.append(
            "The controller has gone error-passive: it is seeing errors "
            "continuously, not occasionally."
        )
    if not hints:
        hints.append(
            "Unclassified bus error; check bitrate against the motors' "
            "configured value, then wiring."
        )
    return hints
