"""Resetting the RIGHT USB device.

The reset itself cannot be tested without hardware. What can, and what matters
on a bench with several identical robots on it, is WHICH device gets picked:
resetting the wrong CH340 knocks out a healthy robot to fix a sick one, and the
symptom is the second robot appearing to develop the same fault.
"""

import os
import sys
import types

import pytest

_HERE = os.path.dirname(__file__)

from actuo.drivers import usb_reset as ur  # noqa: E402


class FakePort:
    def __init__(self, device, vid=0x1A86, pid=0x55D3, serial_number="5AF6",
                 location="3-1:1.0"):
        self.device = device
        self.vid = vid
        self.pid = pid
        self.serial_number = serial_number
        self.location = location


class FakeUsbDevice:
    def __init__(self, vid=0x1A86, pid=0x55D3, bus=3, ports=(1,),
                 serial_number="5AF6"):
        self._vid, self._pid, self._bus = vid, pid, bus
        self._ports, self._serial = list(ports), serial_number
        self.opened = False
        self.reset = False

    def getVendorID(self):
        return self._vid

    def getProductID(self):
        return self._pid

    def getBusNumber(self):
        return self._bus

    def getPortNumberList(self):
        return list(self._ports)

    def getSerialNumber(self):
        return self._serial

    def open(self):
        self.opened = True
        return self

    def resetDevice(self):
        self.reset = True

    def close(self):
        pass


def install_fakes(monkeypatch, ports, devices):
    monkeypatch.setattr(ur, "_comports", lambda: ports)
    monkeypatch.setattr(ur, "_real_device", lambda channel: channel)

    class FakeContext:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def getDeviceList(self, skip_on_error=True):
            return devices

    fake_usb1 = types.SimpleNamespace(USBContext=FakeContext,
                                      USBError=RuntimeError)
    monkeypatch.setitem(sys.modules, "usb1", fake_usb1)


# -- identity ---------------------------------------------------------------


def test_a_non_usb_port_has_nothing_to_reset(monkeypatch):
    """A built-in /dev/ttyS0 has no USB device, and saying so beats resetting
    something else."""
    install_fakes(monkeypatch, [FakePort("/dev/ttyS0", vid=None, pid=None)], [])
    with pytest.raises(ur.UsbResetError, match="not a USB serial port"):
        ur.usb_identity("/dev/ttyS0")


def test_an_unplugged_port_says_so(monkeypatch):
    install_fakes(monkeypatch, [], [])
    with pytest.raises(ur.UsbResetError, match="unplugged"):
        ur.usb_identity("/dev/ttyACM0")


# -- picking the right one --------------------------------------------------


def test_the_matching_device_is_reset(monkeypatch):
    device = FakeUsbDevice()
    install_fakes(monkeypatch, [FakePort("/dev/ttyACM0")], [device])
    described = ur.reset_usb_device("/dev/ttyACM0")
    assert device.reset is True
    assert "1a86:55d3" in described and "bus 3" in described


def test_a_SECOND_identical_adapter_is_not_touched(monkeypatch):
    """The case this project has to get right: several identical robots on one
    machine. rcb4 opens by vendor/product id alone and would reset whichever
    came first."""
    sick = FakeUsbDevice(bus=3, ports=(1,), serial_number="5AF6")
    healthy = FakeUsbDevice(bus=3, ports=(4,), serial_number="OTHER")
    install_fakes(monkeypatch, [FakePort("/dev/ttyACM0", location="3-1:1.0")],
                  [healthy, sick])
    ur.reset_usb_device("/dev/ttyACM0")
    assert sick.reset is True
    assert healthy.reset is False


def test_an_ambiguous_match_is_REFUSED_not_guessed(monkeypatch):
    """Resetting the wrong adapter looks exactly like the other robot
    developing the same fault, so a guess here is expensive to diagnose."""
    twins = [FakeUsbDevice(bus=3, ports=(1,), serial_number=None),
             FakeUsbDevice(bus=3, ports=(1,), serial_number=None)]
    install_fakes(monkeypatch,
                  [FakePort("/dev/ttyACM0", serial_number=None)], twins)
    with pytest.raises(ur.UsbResetError, match="Refusing to guess"):
        ur.reset_usb_device("/dev/ttyACM0")
    assert not any(device.reset for device in twins)


def test_a_device_that_vanished_between_listing_and_reset(monkeypatch):
    install_fakes(monkeypatch, [FakePort("/dev/ttyACM0")], [])
    with pytest.raises(ur.UsbResetError, match="no USB device matched"):
        ur.reset_usb_device("/dev/ttyACM0")


# -- the address parsing the choice rests on --------------------------------


def test_location_is_matched_on_bus_and_port_path_not_the_interface():
    """pyserial reports "3-1.4:1.0": bus 3, ports 1 then 4, interface 1.0. The
    interface identifies an interface OF the device, not the device."""
    assert ur._location_matches(FakeUsbDevice(bus=3, ports=(1, 4)), "3-1.4:1.0")
    assert not ur._location_matches(FakeUsbDevice(bus=3, ports=(1,)), "3-1.4:1.0")
    assert not ur._location_matches(FakeUsbDevice(bus=4, ports=(1,)), "3-1:1.0")


def test_an_unparseable_location_does_not_exclude_anything():
    """Falling back to vendor/product id is right here: refusing on a location
    string we failed to read would make the reset unavailable on a platform
    whose format differs, which is worse than rcb4's behaviour rather than
    better."""
    assert ur._location_matches(FakeUsbDevice(), None)
    assert ur._location_matches(FakeUsbDevice(), "COM3")
