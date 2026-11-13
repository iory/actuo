"""Import an rcb4 Kondo servo map into actuo's robot-config schema.

The importer exists to avoid hand-transcribing a field-proven map into a new
format. It reads the legacy ``joint_name_to_servo_id`` map and emits an actuo
config that ``actuo.drivers.config.build_bus`` can parse directly.

Safety rules this importer enforces loudly
------------------------------------------
1. ``type: continuous`` is never emitted as a position actuator.
2. Shared servo ids are reported and emitted once.
3. With a URDF, servo/URDF joint-set disagreement is reported both ways.
4. With a URDF, emitted order follows ``<ros2_control>`` joint order.
5. With ``--present-ids``, a servo the board did not report is commented out
   rather than emitted. An active entry for an absent servo is not a harmless
   spare: the bus would read its all-zero ``ServoStruct`` slot and publish
   -253 deg as that joint's measured position.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import yaml


@dataclass(frozen=True)
class ServoEntry:
    """One legacy servo-map entry.

    Parameters
    ----------
    name : str
        Joint name in the legacy map.
    servo_id : int
        Flat Kondo servo id.
    direction : int
        Legacy sign convention for this joint.
    offset : float
        Legacy offset value from the map.
    servo_type : str, optional
        Optional legacy type marker, e.g. ``continuous``.
    """

    name: str
    servo_id: int
    direction: int
    offset: float
    servo_type: Optional[str] = None


@dataclass(frozen=True)
class JointLimits:
    """Joint limits in radians from URDF."""

    lower_rad: float
    upper_rad: float


@dataclass(frozen=True)
class ActiveJoint:
    """A joint that will be emitted in the actuo config."""

    entry: ServoEntry
    limit_deg: Optional[Tuple[float, float]]


@dataclass(frozen=True)
class SharedIdConflict:
    """One shared-servo-id conflict group."""

    servo_id: int
    kept_joint: str
    skipped_joints: Tuple[str, ...]


@dataclass
class ImportReport:
    """Structured report for stderr summary and tests."""

    written_joint_names: List[str] = field(default_factory=list)
    continuous_skipped: List[ServoEntry] = field(default_factory=list)
    shared_id_conflicts: List[SharedIdConflict] = field(default_factory=list)
    shared_id_skipped_joints: List[str] = field(default_factory=list)
    absent_from_board: List[ServoEntry] = field(default_factory=list)
    servo_not_in_urdf: List[ServoEntry] = field(default_factory=list)
    urdf_not_in_servo: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    order_verified: bool = False


@dataclass(frozen=True)
class ImportResult:
    """Emission-ready data."""

    generated_at: str
    source_config_path: str
    source_urdf_path: Optional[str]
    symbols_file: str
    channel_placeholder: str
    active_joints: Tuple[ActiveJoint, ...]
    report: ImportReport
    present_ids: Optional[Tuple[int, ...]] = None


def load_rcb4_entries(path: str) -> List[ServoEntry]:
    """Parse ``joint_name_to_servo_id`` from an rcb4 YAML config."""
    with open(path, "r") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError("{}: expected a YAML mapping".format(path))
    mapping = document.get("joint_name_to_servo_id")
    if not isinstance(mapping, dict):
        raise ValueError(
            "{}: expected 'joint_name_to_servo_id' mapping".format(path)
        )

    out: List[ServoEntry] = []
    for joint_name, raw in mapping.items():
        name = str(joint_name)
        if isinstance(raw, int):
            out.append(
                ServoEntry(
                    name=name,
                    servo_id=int(raw),
                    direction=1,
                    offset=0.0,
                    servo_type=None,
                )
            )
            continue
        if not isinstance(raw, dict):
            raise ValueError(
                "{}: joint {!r} must be int or mapping, got {}".format(
                    path, name, type(raw).__name__
                )
            )
        if "id" not in raw:
            raise ValueError(
                "{}: joint {!r} mapping is missing required key 'id'".format(
                    path, name
                )
            )
        out.append(
            ServoEntry(
                name=name,
                servo_id=int(raw["id"]),
                direction=int(raw.get("direction", 1)),
                offset=float(raw.get("offset", 0.0)),
                servo_type=(
                    None
                    if raw.get("type") is None
                    else str(raw.get("type"))
                ),
            )
        )
    return out


def parse_urdf_ros2_control(path: str) -> Tuple[List[str], Dict[str, JointLimits]]:
    """Return ros2_control joint order and kinematic joint limits from URDF."""
    root = ET.parse(path).getroot()

    ros2_control = root.find("ros2_control")
    if ros2_control is None:
        raise ValueError(
            "{}: URDF has no <ros2_control> block".format(path)
        )

    joint_order: List[str] = []
    for joint in ros2_control.findall("joint"):
        name = joint.attrib.get("name")
        if name:
            joint_order.append(name)
    if not joint_order:
        raise ValueError(
            "{}: <ros2_control> has no <joint> entries".format(path)
        )

    limits: Dict[str, JointLimits] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        if not name:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        lower = limit.attrib.get("lower")
        upper = limit.attrib.get("upper")
        if lower is None or upper is None:
            continue
        limits[name] = JointLimits(lower_rad=float(lower), upper_rad=float(upper))

    return joint_order, limits


def _resolve_shared_id_conflicts(
    entries: Sequence[ServoEntry], preferred_order: Sequence[str]
) -> Tuple[Mapping[int, str], List[SharedIdConflict]]:
    grouped: Dict[int, List[str]] = defaultdict(list)
    for entry in entries:
        grouped[entry.servo_id].append(entry.name)

    preferred_pos = {name: idx for idx, name in enumerate(preferred_order)}
    winners: Dict[int, str] = {}
    conflicts: List[SharedIdConflict] = []

    for servo_id, names in grouped.items():
        if len(names) == 1:
            winners[servo_id] = names[0]
            continue

        ranked = sorted(names, key=lambda name: preferred_pos.get(name, len(preferred_pos)))
        kept = ranked[0]
        skipped = tuple(name for name in names if name != kept)
        winners[servo_id] = kept
        conflicts.append(
            SharedIdConflict(
                servo_id=servo_id,
                kept_joint=kept,
                skipped_joints=skipped,
            )
        )

    conflicts.sort(key=lambda conflict: conflict.servo_id)
    return winners, conflicts


def build_import(
    entries: Sequence[ServoEntry],
    source_config_path: str,
    urdf_path: Optional[str],
    symbols_file: str,
    channel_placeholder: str,
    present_ids: Optional[Sequence[int]] = None,
    urdf_label: Optional[str] = None,
) -> ImportResult:
    """Build an import result from parsed entries and optional URDF context.

    Parameters
    ----------
    present_ids : sequence of int, optional
        Servo ids the board actually reported (``actuo-kondo-probe --scan``,
        ``present ids:``). Entries whose id is absent from this list are
        commented out instead of emitted. Omit it to port the map as written.
    """
    by_name = {entry.name: entry for entry in entries}
    if len(by_name) != len(entries):
        raise ValueError("servo config contains duplicate joint names")

    report = ImportReport()
    urdf_order: List[str] = []
    urdf_limits: Dict[str, JointLimits] = {}

    if urdf_path is not None:
        urdf_order, urdf_limits = parse_urdf_ros2_control(urdf_path)
        report.order_verified = True
        urdf_set = set(urdf_order)
        report.servo_not_in_urdf = [
            entry for entry in entries if entry.name not in urdf_set
        ]
        report.urdf_not_in_servo = [
            name for name in urdf_order if name not in by_name
        ]
        if report.servo_not_in_urdf:
            names = [entry.name for entry in report.servo_not_in_urdf]
            report.warnings.append(
                "servo config joints missing from URDF <ros2_control>: {}".format(
                    ", ".join(names)
                )
            )
        if report.urdf_not_in_servo:
            report.warnings.append(
                "URDF <ros2_control> joints missing from servo config: {}".format(
                    ", ".join(report.urdf_not_in_servo)
                )
            )
    else:
        report.order_verified = False
        report.warnings.append(
            "no URDF supplied: emitted order follows input map and is unverified"
        )

    preferred_order = urdf_order if urdf_order else [entry.name for entry in entries]
    id_winner, conflicts = _resolve_shared_id_conflicts(entries, preferred_order)
    report.shared_id_conflicts = conflicts
    for conflict in conflicts:
        report.warnings.append(
            "shared servo id {}: kept {}, skipped {}".format(
                conflict.servo_id,
                conflict.kept_joint,
                ", ".join(conflict.skipped_joints),
            )
        )

    present_set = None if present_ids is None else {int(v) for v in present_ids}
    absent_by_name: Dict[str, ServoEntry] = {}
    if present_set is not None:
        for entry in entries:
            if (entry.servo_type or "").lower() == "continuous":
                continue
            if entry.servo_id not in present_set:
                absent_by_name[entry.name] = entry
        report.absent_from_board = [
            entry for entry in entries if entry.name in absent_by_name
        ]
        # A joint kept out of <ros2_control> *because* the board cannot see it
        # would otherwise be listed twice, under two different reasons. The
        # board's answer is the one that explains the other.
        report.servo_not_in_urdf = [
            entry for entry in report.servo_not_in_urdf
            if entry.name not in absent_by_name
        ]
        if report.absent_from_board:
            report.warnings.append(
                "board did not report servo id(s) {}; those joints are "
                "commented out, not emitted".format(
                    ", ".join(
                        "{} (id {})".format(entry.name, entry.servo_id)
                        for entry in report.absent_from_board
                    )
                )
            )

    continuous_by_name = {
        entry.name: entry
        for entry in entries
        if (entry.servo_type or "").lower() == "continuous"
    }
    report.continuous_skipped = [
        entry
        for entry in entries
        if entry.name in continuous_by_name
    ]

    active: List[ActiveJoint] = []
    chosen_names: set = set()

    if urdf_order:
        candidate_names = urdf_order
    else:
        candidate_names = [entry.name for entry in entries]

    for name in candidate_names:
        if name not in by_name:
            continue
        entry = by_name[name]

        if entry.name in continuous_by_name:
            continue

        if entry.name in absent_by_name:
            continue

        if id_winner[entry.servo_id] != entry.name:
            report.shared_id_skipped_joints.append(entry.name)
            continue

        limit_deg: Optional[Tuple[float, float]] = None
        if urdf_order:
            if name not in urdf_limits:
                raise ValueError(
                    "URDF joint {!r} is in <ros2_control> but has no "
                    "<limit lower= upper=> for position_limit_deg".format(name)
                )
            limit = urdf_limits[name]
            limit_deg = (
                math.degrees(limit.lower_rad),
                math.degrees(limit.upper_rad),
            )

        active.append(ActiveJoint(entry=entry, limit_deg=limit_deg))
        chosen_names.add(name)

    report.written_joint_names = [joint.entry.name for joint in active]

    # If there are shared-id conflicts among joints that are outside candidate
    # order, still record skipped names so they are visible in emitted comments.
    shared_skipped = set(report.shared_id_skipped_joints)
    for conflict in conflicts:
        for name in conflict.skipped_joints:
            if name not in chosen_names:
                shared_skipped.add(name)
    report.shared_id_skipped_joints = sorted(shared_skipped)

    return ImportResult(
        generated_at=_datetime.datetime.now().isoformat(timespec="seconds"),
        source_config_path=os.path.abspath(source_config_path),
        source_urdf_path=(
            urdf_label if urdf_label is not None
            else (None if urdf_path is None else os.path.abspath(urdf_path))
        ),
        symbols_file=symbols_file,
        channel_placeholder=channel_placeholder,
        active_joints=tuple(active),
        report=report,
        present_ids=(
            None if present_set is None else tuple(sorted(present_set))
        ),
    )


def _fmt_number(value: float, decimals: int = 6) -> str:
    text = "{:.{}f}".format(value, decimals)
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def _render_commented_entry(entry: ServoEntry, reason: str) -> List[str]:
    return [
        "# - name: {}".format(entry.name),
        "#   actuator:",
        "#     plugin: kondo_ics",
        "#     id: {}".format(entry.servo_id),
        "#     direction: {}".format(entry.direction),
        "#     offset_deg: {}".format(_fmt_number(entry.offset)),
        "#   skipped: {}".format(reason),
    ]


def render_yaml(result: ImportResult) -> str:
    """Render actuo robot config YAML text with explicit skip comments."""
    report = result.report
    lines: List[str] = []

    lines.append(
        "# Generated by actuo.cli.kondo_import_rcb4 from {}".format(
            result.source_config_path
        )
    )
    if result.source_urdf_path:
        lines.append("# URDF source: {}".format(result.source_urdf_path))
    lines.append("# Generated at: {}".format(result.generated_at))
    lines.append(
        "# offset_deg values come from rcb4 'offset' (degrees in that stack)."
    )
    if result.present_ids is not None:
        lines.append(
            "# Board reported these ids present at import time: {}".format(
                ", ".join(str(value) for value in result.present_ids)
            )
        )
    if report.order_verified:
        lines.append(
            "# Joint order below follows URDF <ros2_control> joint order; this "
            "order is the ros2_control/shm contract."
        )
        lines.append(
            "# position_limit_deg comes from URDF <joint><limit lower/upper> "
            "(radians->degrees), not from servo mechanical stops."
        )
    else:
        lines.append(
            "# No URDF was provided; joint order follows input file order and is "
            "unverified."
        )
    lines.append("")

    lines.extend(
        [
            "bus:",
            "  plugin: kondo_h7_bus",
            "  symbols: {}".format(result.symbols_file),
            "  # move_time_ms: board-side interpolation time per cycle. Left",
            "  # unset here because this importer has no measurement to derive",
            "  # it from; the bus then uses KondoIcsActuator's own default move",
            "  # byte (127 = 1270 ms), which is the gentlest setting and the",
            "  # right one to meet hardware with. Shorten it toward one control",
            "  # period once the loop rate is chosen -- each packet RESTARTS the",
            "  # board's interpolation, so a duration far above the cycle time",
            "  # means the joint never reaches a setpoint before the next.",
            "  transport:",
            "    plugin: kondo_h7",
            "    channel: {}  # operator must set the real serial device path".format(
                result.channel_placeholder
            ),
            "    bitrate: 1000000",
            "",
            "joints:",
        ]
    )

    if not result.active_joints:
        lines.append("  []")
    else:
        for joint in result.active_joints:
            entry = joint.entry
            lines.extend(
                [
                    "  - name: {}".format(entry.name),
                    "    actuator:",
                    "      plugin: kondo_ics",
                    "      id: {}".format(entry.servo_id),
                    "      direction: {}".format(entry.direction),
                    "      offset_deg: {}".format(_fmt_number(entry.offset)),
                ]
            )
            if joint.limit_deg is not None:
                lines.append(
                    "      position_limit_deg: [{}, {}]".format(
                        _fmt_number(joint.limit_deg[0]),
                        _fmt_number(joint.limit_deg[1]),
                    )
                )

    if report.continuous_skipped:
        lines.append("")
        lines.append(
            "# Skipped type: continuous joints (kondo_ics is POSITION-only in "
            "actuo/drivers/actuators/kondo_ics.py)."
        )
        for entry in report.continuous_skipped:
            lines.extend(
                _render_commented_entry(
                    entry,
                    "legacy entry is continuous-rotation; not a position actuator",
                )
            )

    if report.absent_from_board:
        lines.append("")
        lines.append(
            "# The board did not report these servo ids present at import time"
        )
        lines.append(
            "# (ServoStruct.flag, via 'actuo-kondo-probe --scan'). They are the"
        )
        lines.append(
            "# robot's servo map as written -- ported, not deleted -- but left"
        )
        lines.append(
            "# inactive, because an entry for a servo the board cannot see does"
        )
        lines.append(
            "# not simply idle: the bus reads its all-zero ServoStruct slot and"
        )
        lines.append(
            "# publishes pulse 0 = -253 deg as that joint's MEASURED position."
        )
        lines.append(
            "# Connect the servo, re-run the scan, and uncomment (the joint"
        )
        lines.append(
            "# order here is the ros2_control/shm contract -- restore the entry"
        )
        lines.append(
            "# in place, and add the joint back to the URDF <ros2_control>)."
        )
        for entry in report.absent_from_board:
            lines.extend(
                _render_commented_entry(
                    entry,
                    "board did not report servo id {} present at import time"
                    .format(entry.servo_id),
                )
            )

    if report.servo_not_in_urdf:
        non_continuous = [
            entry
            for entry in report.servo_not_in_urdf
            if (entry.servo_type or "").lower() != "continuous"
        ]
        if non_continuous:
            lines.append("")
            lines.append(
                "# Present in servo config but absent from URDF <ros2_control>."
            )
            lines.append(
                "# Extend the same URDF <ros2_control> block first to keep "
                "sim/real parity (one URDF for both; only the hardware plugin "
                "differs)."
            )
            for entry in non_continuous:
                lines.extend(
                    _render_commented_entry(
                        entry,
                        "joint is not in URDF <ros2_control>",
                    )
                )

    if report.shared_id_conflicts:
        lines.append("")
        lines.append(
            "# Shared-servo-id conflicts (one servo cannot take two independent "
            "joint commands)."
        )
        for conflict in report.shared_id_conflicts:
            lines.append(
                "# - id {}: kept {}, skipped {}".format(
                    conflict.servo_id,
                    conflict.kept_joint,
                    ", ".join(conflict.skipped_joints),
                )
            )

    if report.urdf_not_in_servo:
        lines.append("")
        lines.append(
            "# URDF <ros2_control> joints missing from servo config (not emitted): "
            "{}".format(
                ", ".join(report.urdf_not_in_servo)
            )
        )

    lines.append("")
    return "\n".join(lines)


def format_summary(result: ImportResult) -> str:
    """Human-readable stderr summary."""
    report = result.report
    lines = [
        "wrote {} joints".format(len(result.active_joints)),
        "skipped {} continuous joint(s)".format(len(report.continuous_skipped)),
    ]

    if result.present_ids is None:
        lines.append("board presence check: not run (no --present-ids)")
    else:
        lines.append(
            "skipped {} joint(s) the board did not report".format(
                len(report.absent_from_board)
            )
        )

    if report.shared_id_conflicts:
        count = sum(len(conflict.skipped_joints) for conflict in report.shared_id_conflicts)
        lines.append("skipped {} joint(s) due to shared servo ids".format(count))
    else:
        lines.append("shared servo id conflicts: none")

    if result.source_urdf_path:
        lines.append(
            "servo-only joints vs URDF: {}".format(len(report.servo_not_in_urdf))
        )
        lines.append(
            "URDF-only joints vs servo config: {}".format(len(report.urdf_not_in_servo))
        )
    else:
        lines.append("URDF comparison: skipped (no URDF provided)")

    if report.warnings:
        lines.append("warnings:")
        for warning in report.warnings:
            lines.append("  - {}".format(warning))
    else:
        lines.append("warnings: none")

    return "\n".join(lines)


def parse_present_ids(text: Optional[str]) -> Optional[List[int]]:
    """Parse ``--present-ids`` as written by ``actuo-kondo-probe --scan``.

    That line is ``present ids: 0, 1, 2, ...``, so commas and spaces both
    separate. An empty string is rejected rather than read as "no servos": a
    board with nothing on it is a finding, and passing the flag with nothing
    after it is a shell accident.
    """
    if text is None:
        return None
    values = [chunk for chunk in text.replace(",", " ").split() if chunk]
    if not values:
        raise ValueError(
            "--present-ids was given no ids. Omit the flag to skip the board "
            "presence check; do not pass it empty."
        )
    return [int(value) for value in values]


def run_import(
    servo_config_path: str,
    output_path: str,
    urdf_path: Optional[str],
    symbols_file: str,
    channel_placeholder: str,
    present_ids: Optional[Sequence[int]] = None,
    urdf_label: Optional[str] = None,
) -> ImportResult:
    """Run import end-to-end and write the output YAML."""
    entries = load_rcb4_entries(servo_config_path)
    result = build_import(
        entries=entries,
        source_config_path=servo_config_path,
        urdf_path=urdf_path,
        symbols_file=symbols_file,
        channel_placeholder=channel_placeholder,
        present_ids=present_ids,
        urdf_label=urdf_label,
    )
    rendered = render_yaml(result)

    if output_path == "-":
        sys.stdout.write(rendered)
    else:
        with open(output_path, "w") as handle:
            handle.write(rendered)

    return result


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Import an rcb4 Kondo servo map into actuo's YAML schema."
    )
    parser.add_argument(
        "servo_config",
        help="path to rcb4 servo config YAML (joint_name_to_servo_id)",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output actuo robot config YAML path",
    )
    parser.add_argument(
        "--urdf",
        help="optional URDF path for ros2_control order + limits",
    )
    parser.add_argument(
        "--urdf-label",
        help="what to record as the URDF source in the output header. Use it "
             "when --urdf points at a temporary xacro expansion, so the header "
             "names the file a human can re-expand rather than a dead path.",
    )
    parser.add_argument(
        "--symbols",
        required=True,
        help="what to write into bus.symbols: the firmware symbol map. A "
             "package://<pkg>/<path> reference is resolved through "
             "ACTUO_PACKAGE_PATH or the ament index, which suits a map shared "
             "by several robots' packages; a bare filename is resolved next to "
             "the config.",
    )
    parser.add_argument(
        "--channel-placeholder",
        default='"__SET_REAL_DEVICE_PATH__"',
        help="placeholder scalar written into bus.transport.channel",
    )
    parser.add_argument(
        "--present-ids",
        help="servo ids the board reported, e.g. '0, 1, 2, 20' -- paste the "
             "'present ids:' line from 'actuo-kondo-probe --scan'. Joints whose "
             "id is not listed are commented out instead of emitted. Omit to "
             "port the map without consulting a board.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entrypoint."""
    args = parse_args(argv)
    result = run_import(
        servo_config_path=args.servo_config,
        output_path=args.output,
        urdf_path=args.urdf,
        symbols_file=args.symbols,
        channel_placeholder=args.channel_placeholder,
        present_ids=parse_present_ids(args.present_ids),
        urdf_label=args.urdf_label,
    )
    sys.stderr.write(format_summary(result) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
