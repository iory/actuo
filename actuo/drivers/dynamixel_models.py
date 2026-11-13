"""Loads the DYNAMIXEL model table (``data/dynamixel_models.yaml``).

The table is data; this module turns it into objects and, crucially, decides
what happens when a servo answers with a model number the table does not know.

Why an unknown model must be an error
-------------------------------------
The tempting shortcut is ``TABLES.get(model, X_SERIES_DEFAULT)``. It would make
the driver "support every DYNAMIXEL" in the sense that it never crashes -- and
it would be the single most dangerous line in this package. Control-table
addresses are not conventions, they are per-family layouts: on a protocol 2.0
X-series servo address 116 is Goal Position, on a protocol 1.0 AX it is nothing
at all, and on a P-series it is something else again. Writing a position to the
wrong address either errors, or lands in a limit register and quietly changes
what the servo will do later.

So an unknown model raises :class:`UnknownModelError`, and the message contains
a ready-to-paste YAML skeleton plus the e-Manual URL pattern to fill it from.
The user is never blocked for long, and is never handed a guessed register
address that only looks like it works.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import yaml

from actuo.core.contracts import OperatingMode

#: Shipped table. Lives inside the python package so the existing install rule
#: (``install(DIRECTORY actuo.drivers ...)``) carries it -- no CMake edit, and no
#: chance of a driver being installed without the data it cannot run without.
DEFAULT_TABLE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "dynamixel_models.yaml")

#: Sign conventions a control table can use for velocity/load.
DIRECTION_BIT_10 = "direction_bit_10"
TWOS_COMPLEMENT = "twos_complement"


class UnknownModelError(LookupError):
    """A servo reported a model number (or a config named a model) with no row."""


@dataclass(frozen=True)
class Register:
    """One control-table entry: where it is and how wide."""

    address: int
    size: int


@dataclass(frozen=True)
class ControlTable:
    """A control-table layout shared by a family of models."""

    name: str
    protocol: int
    registers: Dict[str, Register]
    present_block: Tuple[str, ...]
    signed_convention: str
    source: str

    def __getitem__(self, name: str) -> Register:
        try:
            return self.registers[name]
        except KeyError:
            raise KeyError(
                "control table {!r} has no register {!r}; known: {}".format(
                    self.name, name, sorted(self.registers))) from None

    def has(self, name: str) -> bool:
        return name in self.registers

    @property
    def block_address(self) -> int:
        """First address of the cyclic read block."""
        return min(self[n].address for n in self.present_block)

    @property
    def block_length(self) -> int:
        """Byte count of the cyclic read block.

        Computed as the span from the lowest address to the end of the highest
        register in the block, so an intentional gap (a register the driver
        does not decode) is read and ignored rather than costing a second
        transaction.
        """
        start = self.block_address
        end = max(self[n].address + self[n].size for n in self.present_block)
        return end - start

    def block_offset(self, name: str) -> int:
        """Where ``name`` sits inside a block read, in bytes."""
        if name not in self.present_block:
            raise KeyError(
                "{!r} is not in the cyclic read block of table {!r}".format(
                    name, self.name))
        return self[name].address - self.block_address


@dataclass(frozen=True)
class DynamixelModel:
    """Everything model-specific the actuator needs, resolved from the table."""

    number: int
    name: str
    table: ControlTable
    resolution: int
    range_deg: float
    center_counts: int
    modes: Dict[OperatingMode, Optional[int]]
    velocity_unit_rpm: Optional[float]
    voltage_divisor: Optional[float]
    load_unit_percent: Optional[float]
    current_unit_ma: Optional[float]
    torque_constant: Optional[float]
    has_temperature_sense: bool
    source: str
    baudrates: Tuple[int, ...] = ()

    @property
    def protocol(self) -> int:
        return self.table.protocol

    @property
    def counts_per_rev(self) -> float:
        """Counts per full 360 deg turn.

        An AX spans 1024 counts over only 300 deg of travel, so its counts per
        *revolution* is 1228.8, not 1024. Feeding the raw resolution into the
        generic radians conversion would scale every angle by 300/360 -- a 20%
        error that still looks like a plausible robot pose.
        """
        return self.resolution * 360.0 / self.range_deg

    def supports(self, mode: OperatingMode) -> bool:
        return mode in self.modes


def _parse_table(name: str, spec: Dict) -> ControlTable:
    registers = {
        key: Register(int(value["address"]), int(value["size"]))
        for key, value in spec["registers"].items()
    }
    block = tuple(spec["present_block"])
    missing = [n for n in block if n not in registers]
    if missing:
        raise ValueError(
            "control table {!r}: present_block names unknown registers {}".format(
                name, missing))
    return ControlTable(
        name=name,
        protocol=int(spec["protocol"]),
        registers=registers,
        present_block=block,
        signed_convention=str(spec.get("signed_convention", TWOS_COMPLEMENT)),
        source=str(spec["source"]),
    )


def _parse_model(number: int, spec: Dict, tables: Dict[str, ControlTable]
                 ) -> DynamixelModel:
    table_name = str(spec["table"])
    if table_name not in tables:
        raise ValueError(
            "model {}: unknown control table {!r}; known: {}".format(
                number, table_name, sorted(tables)))
    table = tables[table_name]

    modes: Dict[OperatingMode, Optional[int]] = {}
    for mode_name, value in (spec.get("modes") or {}).items():
        try:
            mode = OperatingMode(mode_name)
        except ValueError:
            raise ValueError(
                "model {}: {!r} is not an actuo OperatingMode; use one of {}"
                .format(number, mode_name,
                        [m.value for m in OperatingMode])) from None
        modes[mode] = None if value is None else int(value)
    if not modes:
        raise ValueError("model {}: declares no operating mode".format(number))

    return DynamixelModel(
        number=number,
        name=str(spec["name"]),
        table=table,
        resolution=int(spec["resolution"]),
        range_deg=float(spec["range_deg"]),
        center_counts=int(spec["center_counts"]),
        modes=modes,
        velocity_unit_rpm=_optional_float(spec.get("velocity_unit_rpm")),
        voltage_divisor=_optional_float(spec.get("voltage_divisor")),
        load_unit_percent=_optional_float(spec.get("load_unit_percent")),
        current_unit_ma=_optional_float(spec.get("current_unit_ma")),
        torque_constant=_optional_float(spec.get("torque_constant")),
        has_temperature_sense=bool(spec.get("has_temperature_sense", False)),
        source=str(spec["source"]),
        baudrates=tuple(int(b) for b in (spec.get("baudrates") or ())),
    )


def _optional_float(value) -> Optional[float]:
    return None if value is None else float(value)


class ModelTable:
    """The loaded model table, queryable by model number or by name."""

    def __init__(self, models: Dict[int, DynamixelModel], path: str) -> None:
        self._by_number = models
        self._by_name = {}
        for model in models.values():
            if model.name in self._by_name:
                raise ValueError(
                    "{}: duplicate model name {!r}".format(path, model.name))
            self._by_name[model.name] = model
        self.path = path

    def __len__(self) -> int:
        return len(self._by_number)

    @property
    def numbers(self):
        return sorted(self._by_number)

    @property
    def names(self):
        return sorted(self._by_name)

    def by_number(self, number: int) -> DynamixelModel:
        try:
            return self._by_number[int(number)]
        except KeyError:
            raise UnknownModelError(_unknown_message(number, self)) from None

    def by_name(self, name: str) -> DynamixelModel:
        try:
            return self._by_name[str(name)]
        except KeyError:
            raise UnknownModelError(
                "no model named {!r} in {}; known: {}. Model names are the "
                "e-Manual's exact product names (e.g. 'AX-12A').".format(
                    name, self.path, self.names)) from None

    def get(self, key) -> DynamixelModel:
        """Look up by model number (int) or product name (str)."""
        if isinstance(key, int):
            return self.by_number(key)
        return self.by_name(key)


def _unknown_message(number: int, table: "ModelTable") -> str:
    """The error a user actually needs: what to add, where, and from which URL."""
    return (
        "DYNAMIXEL model number {number} is not in {path}.\n"
        "\n"
        "This is deliberate: guessing a control-table layout would command\n"
        "registers at addresses that mean something else on this servo. Add a\n"
        "row instead -- it is data, no code changes.\n"
        "\n"
        "1. Find the model at https://emanual.robotis.com/docs/en/dxl/ and open\n"
        "   its page (the Control Table section lists every address).\n"
        "2. Append to the `models:` block of {path}:\n"
        "\n"
        "  {number}:\n"
        "    name: <exact product name, e.g. XM430-W350-T>\n"
        "    table: <one of: {tables}>   # or add a new table block\n"
        "    source: <the e-Manual URL you just opened>\n"
        "    resolution: <counts over the full travel>\n"
        "    range_deg: <degrees of that travel>\n"
        "    center_counts: <count reported at the mechanical centre>\n"
        "    modes:\n"
        "      position: <operating_mode register value, or null if none>\n"
        "    velocity_unit_rpm: <rev/min per LSB, or null if not documented>\n"
        "    voltage_divisor: null\n"
        "    load_unit_percent: null\n"
        "    current_unit_ma: <mA per LSB, or null if no current sensing>\n"
        "    torque_constant: <N*m/A, or null -- null disables CURRENT mode>\n"
        "    has_temperature_sense: <true only if temperature is in the block>\n"
        "\n"
        "Known model numbers: {known}"
    ).format(
        number=number,
        path=table.path,
        tables=", ".join(sorted({m.table.name for m in table._by_number.values()})),
        known=table.numbers,
    )


_CACHE: Dict[str, ModelTable] = {}


def load_model_table(path: Optional[str] = None) -> ModelTable:
    """Load (and memoise) the model table.

    Parameters
    ----------
    path : str, optional
        Table file. Defaults to the one shipped with this package. A robot with
        an unusual servo can point at its own file rather than patching the
        installed one.
    """
    resolved = os.path.abspath(path or DEFAULT_TABLE_PATH)
    if resolved in _CACHE:
        return _CACHE[resolved]

    with open(resolved, "r") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError("{}: expected a YAML mapping".format(resolved))

    tables = {name: _parse_table(name, spec)
              for name, spec in (document.get("tables") or {}).items()}
    if not tables:
        raise ValueError("{}: no control tables defined".format(resolved))

    models = {int(number): _parse_model(int(number), spec, tables)
              for number, spec in (document.get("models") or {}).items()}
    if not models:
        raise ValueError("{}: no models defined".format(resolved))

    table = ModelTable(models, resolved)
    _CACHE[resolved] = table
    return table
