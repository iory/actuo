#!/usr/bin/env python3
"""Extract the Kondo ARM-H7 firmware's symbol addresses into a committed file.

Run this **once per firmware build**, offline, and commit the result next to the
robot config. The driver then reads that file; it never opens an ELF and never
needs a toolchain at run time.

Why not just read the ELF at run time, as ``rcb4`` does?
-------------------------------------------------------
Because a host library lock-stepped to the firmware ELF's symbol layout is one
of the failures of earlier Kondo drivers that this project exists to not repeat. The
lock-step itself is unavoidable while the firmware exposes its state only as
memory. What is avoidable is doing it *implicitly*: resolving addresses at
connect time, from whichever ELF happens to be on the machine, with pyelftools
in the driver's dependency tree. Making it an artefact turns an invisible
coupling into a reviewable file with a version stamp -- and the driver refuses
to run against a board whose firmware does not match that stamp.

Usage
-----
::

    # ask the board which firmware it is running (no symbol map needed)
    actuo-kondo-probe --channel /dev/ttyUSB1 --version-only

    # then pin that string to the matching ELF
    actuo-kondo-symbols v0.6.5.elf \\
        --firmware-version v0.6.5 -o kondoh7_v0.6.5_symbols.yaml

The version must be supplied by the operator rather than guessed from the ELF:
the string the board reports is produced by the firmware at run time, and
inferring it from a filename or a build path would defeat the whole check.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterator, Mapping, Tuple


from actuo.drivers.kondo_h7_protocol import (  # noqa: E402
    CONTROL_SLOTS,
    CONTROL_STRUCT_SIZE,
    REQUIRED_SYMBOLS,
    SERVO_SLOTS,
    SERVO_STRUCT_SIZE,
    SYSTEM_SLOTS,
    SYSTEM_STRUCT_SIZE,
    Slot,
    dump_symbol_map,
)

#: Symbols worth recording beyond the required ones: they are what the
#: maintenance tools (IMU, air board, worm modules, flash) will need, and
#: extracting them now costs nothing.
OPTIONAL_SYMBOLS = (
    "Control",
    "Sensor_vector",
    "Worm_vector",
    "GPIO_vector",
    "imu_data_",
    "Mfilter",
    "data_address",
    "servo_idmode_scan",
    "buzzer_init_sound",
    "pump_switch",
    "valve_switch",
    "gpio_cmd",
    "rom_to_flash",
    "flash_to_rom",
    "dataram_to_dataflash",
    "dataflash_to_dataram",
)


def read_symbols(elf_path, wanted):
    """Return ``{name: address}`` for the wanted symbols present in the ELF."""
    try:
        from elftools.elf.elffile import ELFFile
        from elftools.elf.sections import SymbolTableSection
    except ImportError:
        raise SystemExit(
            "pyelftools is required to read the firmware ELF.\n"
            "It is an offline tool dependency only -- the driver itself never "
            "opens an ELF.\n"
            "    pip install pyelftools")

    found = {}
    with open(elf_path, "rb") as stream:
        elf = ELFFile(stream)
        for section in elf.iter_sections():
            if not isinstance(section, SymbolTableSection):
                continue
            for symbol in section.iter_symbols():
                if symbol.name in wanted:
                    found[symbol.name] = symbol.entry["st_value"]
    return found


def _die_name(die) -> str:
    attr = die.attributes.get("DW_AT_name")
    if attr is None:
        return ""
    value = attr.value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _iter_die_tree(die) -> Iterator:
    yield die
    for child in die.iter_children():
        yield from _iter_die_tree(child)


def _member_offset(member_die, dwarf_info) -> int:
    attr = member_die.attributes.get("DW_AT_data_member_location")
    if attr is None:
        raise RuntimeError(
            "member {} has no DW_AT_data_member_location".format(_die_name(member_die)))
    from elftools.dwarf.descriptions import describe_form_class
    form_class = describe_form_class(attr.form)
    if form_class == "constant":
        return int(attr.value)
    if form_class == "exprloc":
        from elftools.dwarf.dwarf_expr import DWARFExprParser
        parser = DWARFExprParser(dwarf_info.structs)
        ops = parser.parse_expr(attr.value)
        if len(ops) == 1 and ops[0].op_name == "DW_OP_plus_uconst":
            return int(ops[0].args[0])
        raise RuntimeError(
            "member {} uses unsupported location expression {}".format(
                _die_name(member_die), [op.op_name for op in ops]))
    raise RuntimeError(
        "member {} has unsupported location form class {}".format(
            _die_name(member_die), form_class))


def _type_size(type_die) -> int:
    if type_die is None:
        raise RuntimeError("missing type DIE while computing member size")
    if "DW_AT_byte_size" in type_die.attributes:
        return int(type_die.attributes["DW_AT_byte_size"].value)
    if type_die.tag in (
            "DW_TAG_typedef", "DW_TAG_const_type", "DW_TAG_volatile_type",
            "DW_TAG_restrict_type"):
        return _type_size(type_die.get_DIE_from_attribute("DW_AT_type"))
    if type_die.tag == "DW_TAG_array_type":
        base_size = _type_size(type_die.get_DIE_from_attribute("DW_AT_type"))
        count = 1
        for child in type_die.iter_children():
            if child.tag != "DW_TAG_subrange_type":
                continue
            if "DW_AT_count" in child.attributes:
                count *= int(child.attributes["DW_AT_count"].value)
            elif "DW_AT_upper_bound" in child.attributes:
                count *= int(child.attributes["DW_AT_upper_bound"].value) + 1
            else:
                raise RuntimeError(
                    "array type has subrange without count or upper bound")
        return base_size * count
    raise RuntimeError(
        "cannot compute size for DIE tag {} (name={!r})"
        .format(type_die.tag, _die_name(type_die)))


def _find_struct_die(dwarf_info, struct_name):
    for cu in dwarf_info.iter_CUs():
        root = cu.get_top_DIE()
        for die in _iter_die_tree(root):
            if die.tag != "DW_TAG_structure_type":
                continue
            if _die_name(die) == struct_name:
                return die
    return None


def _collect_struct_members(struct_die, dwarf_info) -> Dict[str, Tuple[int, int]]:
    members = {}
    for child in struct_die.iter_children():
        if child.tag != "DW_TAG_member":
            continue
        name = _die_name(child)
        if not name:
            continue
        offset = _member_offset(child, dwarf_info)
        size = _type_size(child.get_DIE_from_attribute("DW_AT_type"))
        members[name] = (offset, size)
    return members


def _verify_one_struct(elf_path: str, dwarf_info, struct_name: str,
                       expected_size: int, expected_slots: Mapping[str, Slot]) -> int:
    struct_die = _find_struct_die(dwarf_info, struct_name)
    if struct_die is None:
        print("{}: {} missing in DWARF".format(elf_path, struct_name))
        return 1

    mismatches = 0
    byte_size_attr = struct_die.attributes.get("DW_AT_byte_size")
    if byte_size_attr is None:
        print("{}: {} has no DW_AT_byte_size".format(elf_path, struct_name))
        return 1
    dwarf_size = int(byte_size_attr.value)
    status = "OK" if dwarf_size == expected_size else "MISMATCH"
    if status != "OK":
        mismatches += 1
    print("\n{} size  driver={}  dwarf={}  {}".format(
        struct_name, expected_size, dwarf_size, status))
    print("  {:<24} {:>8} {:>8} {:>8} {:>8}  status".format(
        "member", "drv_off", "dw_off", "drv_sz", "dw_sz"))

    members = _collect_struct_members(struct_die, dwarf_info)
    for name, slot in expected_slots.items():
        if name not in members:
            mismatches += 1
            print("  {:<24} {:>8} {:>8} {:>8} {:>8}  MISSING".format(
                name, slot.offset, "-", slot.size, "-"))
            continue
        dwarf_offset, dwarf_size = members[name]
        ok = (slot.offset == dwarf_offset and slot.size == dwarf_size)
        if not ok:
            mismatches += 1
        print("  {:<24} {:>8} {:>8} {:>8} {:>8}  {}".format(
            name, slot.offset, dwarf_offset, slot.size, dwarf_size,
            "OK" if ok else "MISMATCH"))
    return mismatches


def verify_layout(elf_path: str) -> int:
    """Verify driver struct constants against DWARF; return mismatch count."""
    try:
        from elftools.elf.elffile import ELFFile
    except ImportError:
        raise SystemExit(
            "pyelftools is required to read the firmware ELF.\n"
            "It is an offline tool dependency only -- the driver itself never "
            "opens an ELF.\n"
            "    pip install pyelftools")

    with open(elf_path, "rb") as stream:
        elf = ELFFile(stream)
        if not elf.has_dwarf_info():
            raise SystemExit(
                "{}: DWARF debug info is required for --verify-layout, but this "
                "ELF has no .debug_info".format(elf_path))
        dwarf_info = elf.get_dwarf_info()
        total = 0
        total += _verify_one_struct(
            elf_path, dwarf_info, "ServoStruct", SERVO_STRUCT_SIZE, SERVO_SLOTS)
        total += _verify_one_struct(
            elf_path, dwarf_info, "SystemStruct", SYSTEM_STRUCT_SIZE, SYSTEM_SLOTS)
        total += _verify_one_struct(
            elf_path, dwarf_info, "ControlStruct", CONTROL_STRUCT_SIZE, CONTROL_SLOTS)
        return total


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract Kondo ARM-H7 firmware symbol addresses.")
    parser.add_argument("elf", help="path to the firmware ELF")
    parser.add_argument(
        "--firmware-version", required=True,
        help="the version string the BOARD reports (actuo-kondo-probe "
             "--version-only). The driver refuses to run unless the board "
             "still reports this.")
    parser.add_argument("-o", "--output", default="-",
                        help="output YAML path, or '-' for stdout")
    parser.add_argument(
        "--source",
        help="what to record as the map's origin. Defaults to the ELF path, "
             "which is only meaningful on the machine that ran this; pass a "
             "release URL when committing the map so it stays retrievable.")
    parser.add_argument(
        "--verify-layout", action="store_true",
        help="verify ServoStruct/SystemStruct/ControlStruct sizes and member "
             "offsets against DWARF in the ELF; exits non-zero on mismatch.")
    args = parser.parse_args(argv)

    wanted = set(REQUIRED_SYMBOLS) | set(OPTIONAL_SYMBOLS)
    found = read_symbols(args.elf, wanted)

    missing = [name for name in REQUIRED_SYMBOLS if name not in found]
    if missing:
        raise SystemExit(
            "{}: does not define the symbols the driver needs: {}\n"
            "Is this the kondoh7 firmware ELF, and was it built with symbols?"
            .format(args.elf, missing))

    if args.verify_layout:
        mismatch_count = verify_layout(args.elf)
        if mismatch_count:
            raise SystemExit(
                "--verify-layout failed: {} mismatch(es) found".format(
                    mismatch_count))
        print("\n--verify-layout: all checked struct sizes/slots match DWARF")

    document = dump_symbol_map(args.firmware_version, found,
                               source=args.source or args.elf)
    if args.output == "-":
        sys.stdout.write(document)
    else:
        with open(args.output, "w") as handle:
            handle.write(document)
        skipped = sorted(wanted - set(found))
        print("wrote {} ({} symbols for firmware {!r})".format(
            args.output, len(found), args.firmware_version))
        if skipped:
            # Not an error -- the optional symbols only matter to tools that
            # use them -- but silently dropping them would look like coverage.
            print("not present in this ELF: {}".format(", ".join(skipped)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
