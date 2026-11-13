#!/usr/bin/env bash
# Install actuo's udev rules, then apply them without needing a replug.
#
#   sudo $(ros2 pkg prefix actuo_ros)/share/actuo_ros/udev/install.sh
#
#   99-actuo-kondo-symlink.rules  /dev/actuo_kondo_rcb4mini and
#                                 /dev/actuo_kondo_armh7 for the Kondo servo
#                                 bus adapters, so a robot config names a board
#                                 type instead of one chip's serial number.
#   99-actuo-ftdi-latency.rules   FTDI reply latency 16 ms -> 1 ms.
#   99-actuo-usb-reset.rules      let plugdev reset a wedged adapter.
#
# Re-running is safe; it overwrites and re-applies.
set -euo pipefail

if [ "$(id -u)" -ne 0 ]; then
    echo "needs root to write /etc/udev/rules.d. Re-run with sudo." >&2
    exit 1
fi

rules="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for file in "$rules"/*.rules; do
    install -m 0644 "$file" /etc/udev/rules.d/
    echo "installed $(basename "$file")"
done

udevadm control --reload-rules
# `trigger` re-runs the rules against devices that are ALREADY plugged in.
# `tty` is needed as well: SYMLINK+= lands on the tty device, not on its usb or
# usb-serial parent, so without it the names appear only after a replug.
udevadm trigger --subsystem-match=usb --action=add
udevadm trigger --subsystem-match=usb-serial --action=add
udevadm trigger --subsystem-match=tty --action=add
udevadm settle || true

echo
echo "in effect now:"
shopt -s nullglob
for port in /sys/bus/usb-serial/devices/*/latency_timer; do
    printf '  %-12s latency_timer = %s ms\n' \
        "$(basename "$(dirname "$port")")" "$(cat "$port")"
done
found=0
for name in actuo_kondo_rcb4mini actuo_kondo_armh7; do
    if [ -e "/dev/$name" ]; then
        printf '  %-22s -> %s\n' "$name" "$(readlink -f "/dev/$name")"
        found=1
    else
        printf '  %-22s absent (no matching adapter plugged in)\n' "$name"
    fi
done
# Every usb-serial port, so a name that went to the wrong device (a power
# supply with the same bridge chip, say) is visible here and not only as a
# board that never answers.
ports=(/dev/serial/by-id/*)
if [ "${#ports[@]}" -gt 0 ]; then
    echo "  usb-serial adapters attached:"
    for entry in "${ports[@]}"; do
        printf '    %s -> %s\n' "$(basename "$entry")" "$(readlink -f "$entry")"
    done
fi
if [ "$found" -eq 0 ]; then
    echo "  Check with: lsusb; ls -l /dev/serial/by-id/"
fi

user="${SUDO_USER:-$USER}"
if ! id -nG "$user" | grep -qw plugdev; then
    echo
    echo "NOTE: $user is not in the plugdev group, so the reset rule grants it"
    echo "nothing. Fix with:  sudo usermod -aG plugdev $user   (then log in again)"
fi
