"""Serial framing: finding where a packet ends, for both wire formats.

Framing is the one thing a transport must know, and it is easy to get subtly
wrong in a way that only shows up under load: a header found one byte late, a
length convention off by the size of its own field, a re-sync that throws away
the start of the next packet. Each produces an intermittently silent bus, which
is far more expensive to debug than a bus that never works.

A fake serial port stands in for the wire so this runs with no hardware.
"""

import io

import pytest

from actuo.drivers import dynamixel_protocol as dxl
from actuo.drivers.transports import serial_transport
from actuo.drivers.transports.serial_transport import SerialTransport


class FakePort:
    """Just enough of pyserial to frame packets out of a byte stream."""

    def __init__(self, stream=b""):
        self.stream = bytearray(stream)
        self.sent = bytearray()
        self.is_open = True
        self.timeout = 0.0

    @property
    def in_waiting(self):
        return len(self.stream)

    def read(self, count):
        chunk = bytes(self.stream[:count])
        del self.stream[:count]
        return chunk

    def write(self, data):
        self.sent.extend(data)

    def flush(self):
        pass

    def reset_input_buffer(self):
        self.stream.clear()

    def reset_output_buffer(self):
        pass

    def close(self):
        self.is_open = False

    def fileno(self):
        raise io.UnsupportedOperation("no file descriptor")


@pytest.fixture
def fake_serial(monkeypatch):
    created = {}

    def factory(channel, bitrate, timeout=0.0, exclusive=None):
        port = FakePort()
        created["port"] = port
        return port

    monkeypatch.setattr(serial_transport.serial, "Serial", factory)
    return created


def open_transport(fake_serial, protocol, stream=b"", **kwargs):
    transport = SerialTransport("/dev/fake", 1000000, timeout=0.01,
                                protocol=protocol, **kwargs)
    transport.open()
    fake_serial["port"].stream = bytearray(stream)
    return transport, fake_serial["port"]


def test_protocol2_packet_is_framed_by_its_two_byte_length(fake_serial):
    packet = dxl.build_packet(dxl.PROTOCOL_2, 7, dxl.STATUS, b"\x00\x2a")
    transport, _ = open_transport(fake_serial, 2, packet)

    frame = transport.recv(0.05)
    assert frame is not None
    assert frame.payload == packet
    assert frame.address == 7          # the id sits after the 4-byte header


def test_leading_garbage_is_resynchronised_past(fake_serial):
    packet = dxl.build_packet(dxl.PROTOCOL_2, 3, dxl.STATUS, b"\x00\x01")
    transport, _ = open_transport(fake_serial, 2, b"\x11\xff\xfd\x00" + packet)
    assert transport.recv(0.05).payload == packet


def test_two_back_to_back_packets_are_both_recovered(fake_serial):
    """A re-sync that over-trims eats the packet that follows it."""
    first = dxl.build_packet(dxl.PROTOCOL_2, 1, dxl.STATUS, b"\x00\x01")
    second = dxl.build_packet(dxl.PROTOCOL_2, 2, dxl.STATUS, b"\x00\x02")
    transport, _ = open_transport(fake_serial, 2, first + second)
    assert transport.recv(0.05).payload == first
    assert transport.recv(0.05).payload == second


def test_a_stuffed_payload_does_not_break_framing(fake_serial):
    """The length field counts stuffed bytes, so the packet still ends where it
    says it does even when its data contains the header pattern."""
    packet = dxl.build_packet(dxl.PROTOCOL_2, 4, dxl.STATUS,
                              b"\x00\xff\xff\xfd\x09")
    transport, _ = open_transport(fake_serial, 2, packet)
    frame = transport.recv(0.05)
    assert frame.payload == packet
    assert dxl.parse_status(dxl.PROTOCOL_2, frame.payload)[2] == b"\xff\xff\xfd\x09"


def test_protocol1_framing_still_works(fake_serial):
    packet = dxl.build_packet(dxl.PROTOCOL_1, 15, 0x00, bytes(8))
    transport, _ = open_transport(fake_serial, 1, packet)
    frame = transport.recv(0.05)
    assert frame.payload == packet
    assert frame.address == 15


def test_echo_is_only_discarded_when_asked(fake_serial):
    """Default off: silently swallowing traffic would hide a wiring fault. But
    on an adapter that loops back, a protocol 1.0 instruction packet decodes as
    a plausible status packet, so the option has to exist."""
    from actuo.core.contracts import Frame

    sent = dxl.build_write(dxl.PROTOCOL_1, 1, 30, dxl.to_le(512, 2))
    reply = dxl.build_packet(dxl.PROTOCOL_1, 1, 0x00, bytes(8))

    transport, port = open_transport(fake_serial, 1)
    transport.write([Frame(payload=sent, address=1)])
    port.stream = bytearray(sent + reply)
    assert transport.recv(0.05).payload == sent      # the echo is handed up

    transport, port = open_transport(fake_serial, 1, discard_echo=True)
    transport.write([Frame(payload=sent, address=1)])
    port.stream = bytearray(sent + reply)
    assert transport.recv(0.05).payload == reply     # the echo is skipped


def test_an_invalid_framing_version_is_refused():
    with pytest.raises(ValueError):
        SerialTransport("/dev/fake", 1000000, protocol=3)
