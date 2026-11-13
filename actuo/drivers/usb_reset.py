"""Reset the USB device behind a serial port, when the port itself is wedged.

The failure this exists for
---------------------------
A USB-serial board can stop answering while everything below it looks perfectly
healthy: the device is still enumerated, ``/dev/ttyACM0`` still opens, writes
still succeed, and nothing raises. Reopening the port does not help, and neither
does unplugging and replugging it (measured on a Kondo board behind a CH340 adapter, 2026-08-01: after a replug
the device re-enumerated -- new inode, new bus address -- and the board stayed
silent through 30 version requests).

rcb4 hit this too and its ``ARMH7Interface.from_port`` answers it the same way:
from the second connection attempt onward it performs a USB-level
``resetDevice()``, waits two seconds, reopens, and then retries the ACK up to
ten times because "after powering on, the ACK value becomes unstable for some
reason". That recipe is reproduced here rather than reinvented -- it is the one
that has been run on this hardware for years.

Why not just vendor and product id
----------------------------------
rcb4 opens the device by vid/pid, which resets *a* CH340 rather than *the*
CH340. This project must support several identical robots connected to one
host computer, and resetting the wrong one would knock out a healthy
robot to fix a sick one -- a failure that looks exactly like a second robot
developing the same fault. So the port's own USB address (bus and port path,
which pyserial reports as ``location``) is matched as well, and its serial
number when the adapter has one. Ambiguity is refused, never resolved by
picking the first match.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple


class UsbResetError(RuntimeError):
    """The device could not be identified, found, or reset."""


def _real_device(channel: str) -> str:
    """The ``/dev/ttyACM*`` a possibly-symlinked channel points at.

    Robot configs name the port through ``/dev/serial/by-id/...`` because the
    number is not stable -- and a USB reset is precisely an event that can
    change it. pyserial's port list reports real device nodes, so the symlink
    has to be followed to match them.
    """
    return os.path.realpath(channel)


def _comports():
    """The system's serial ports. A seam, so the selection logic below can be
    tested without plugging in two identical adapters."""
    import serial.tools.list_ports

    return serial.tools.list_ports.comports()


def usb_identity(channel: str) -> Tuple[int, int, Optional[str], Optional[str]]:
    """``(vendor_id, product_id, serial_number, location)`` for a serial port.

    Raises
    ------
    UsbResetError
        If the port is not in the system's port list, or is not a USB device --
        a built-in ``/dev/ttyS0`` has nothing to reset, and saying so beats
        resetting something else.
    """
    device = _real_device(channel)
    for port in _comports():
        if port.device != device:
            continue
        if port.vid is None or port.pid is None:
            raise UsbResetError(
                "{} is not a USB serial port (no vendor/product id), so there "
                "is no USB device to reset".format(channel))
        return port.vid, port.pid, port.serial_number, port.location
    raise UsbResetError(
        "{} (which is {}) is not in the system's serial port list; it may have "
        "been unplugged".format(channel, device))


def _location_matches(device, location: Optional[str]) -> bool:
    """Is this libusb device at the bus/port path pyserial reported?

    pyserial's ``location`` reads like ``"3-1.4:1.0"``: bus 3, then the port
    numbers walked down from the root hub, then the interface. The interface
    part is dropped -- it identifies an interface of the device, not the device.
    """
    if not location:
        return True                     # nothing to check against; vid/pid only
    path = location.split(":")[0]
    bus_text, _, ports_text = path.partition("-")
    try:
        bus = int(bus_text)
        ports = [int(p) for p in ports_text.split(".") if p != ""]
    except ValueError:
        return True
    if device.getBusNumber() != bus:
        return False
    return list(device.getPortNumberList()) == ports


def reset_usb_device(channel: str) -> str:
    """Issue a USB reset to the device behind ``channel``.

    Returns a one-line description of what was reset, for logging: an operator
    watching a robot recover needs to know WHICH device was reset, especially
    with more than one plugged in.

    Raises
    ------
    UsbResetError
        No match, more than one match, or the reset itself failed. Permission
        errors are re-raised with the udev rule that fixes them, because
        "LIBUSB_ERROR_ACCESS" on its own sends people to the wrong place.
    """
    vendor_id, product_id, serial_number, location = usb_identity(channel)

    try:
        import usb1
    except ImportError as error:
        raise UsbResetError(
            "usb1 (libusb1) is not installed, so a wedged adapter cannot be "
            "reset from here: {}".format(error))

    with usb1.USBContext() as context:
        candidates = []
        for device in context.getDeviceList(skip_on_error=True):
            if (device.getVendorID(), device.getProductID()) != (vendor_id,
                                                                 product_id):
                continue
            if not _location_matches(device, location):
                continue
            if serial_number:
                try:
                    if device.getSerialNumber() != serial_number:
                        continue
                except usb1.USBError:
                    # Reading the serial number needs the device open, which
                    # can fail on a wedged adapter. Fall through on bus/port
                    # path alone rather than skipping the device we came for.
                    pass
            candidates.append(device)

        if not candidates:
            raise UsbResetError(
                "no USB device matched {} ({:04x}:{:04x} at {}); it may have "
                "been unplugged".format(channel, vendor_id, product_id,
                                        location or "an unknown address"))
        if len(candidates) > 1:
            raise UsbResetError(
                "{} matched {} USB devices ({:04x}:{:04x}). Refusing to guess: "
                "resetting the wrong adapter would knock out a healthy robot "
                "to fix a sick one.".format(
                    channel, len(candidates), vendor_id, product_id))

        device = candidates[0]
        described = "{:04x}:{:04x} on bus {} port {}".format(
            vendor_id, product_id, device.getBusNumber(),
            ".".join(str(p) for p in device.getPortNumberList()))
        try:
            handle = device.open()
        except usb1.USBError as error:
            raise UsbResetError(_permission_hint(error, vendor_id, product_id))
        try:
            handle.resetDevice()
        except usb1.USBError as error:
            raise UsbResetError(
                "reset of {} failed: {}".format(described, error))
        finally:
            handle.close()
    return described


def _permission_hint(error, vendor_id: int, product_id: int) -> str:
    text = str(error)
    if "ACCESS" not in text.upper():
        return "could not open the USB device: {}".format(text)
    return (
        "no permission to reset the USB device ({}). Reading the serial port "
        "only needs the `dialout` group; resetting the USB device needs write "
        "access to /dev/bus/usb. Add a rule and replug:\n"
        "  echo 'SUBSYSTEM==\"usb\", ATTR{{idVendor}}==\"{:04x}\", "
        "ATTR{{idProduct}}==\"{:04x}\", MODE=\"0666\"' | "
        "sudo tee /etc/udev/rules.d/99-actuo-usb.rules\n"
        "  sudo udevadm control --reload-rules && sudo udevadm trigger"
        .format(text, vendor_id, product_id))
