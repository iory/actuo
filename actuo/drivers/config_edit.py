"""Edit calibration values in place, without destroying the file around them.

Why not just load and dump the YAML
-----------------------------------
Because the comments ARE the calibration. This project's robot configs carry
paragraphs explaining why a number is what it is -- which power-up an offset was
measured on, why an envelope has no inward margin, which policy a pose belongs
to. ``yaml.safe_load`` + ``yaml.dump`` silently deletes every one of them, and a
number with its reasoning removed is exactly the thing that gets re-derived
wrongly later.

So this edits the TEXT: it finds the one line holding a key inside one joint's
block and rewrites that line, leaving every other byte -- comments, ordering,
anchors, spacing -- untouched. Surgical rather than structural.

The same applies to the URDF, where a joint's limits appear twice (``<limit>``
and the ros2_control ``command_interface`` bounds) and both must move together
with the YAML.
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple


class ConfigEditError(RuntimeError):
    """A value could not be placed. Raised rather than writing a guess."""


def _joint_block(lines: List[str], joint: str) -> Tuple[int, int, str]:
    """Return (start, end, key_indent) of ``joint``'s entry in a robot config.

    ``start`` is the ``- name: <joint>`` line; ``end`` is one past the last line
    belonging to it. ``key_indent`` is the indentation used by the keys inside
    its ``actuator:`` block, taken from the file rather than assumed.
    """
    name_re = re.compile(r"^(\s*)-\s+name:\s*(\S+)\s*$")
    start = None
    indent = ""
    for i, line in enumerate(lines):
        match = name_re.match(line)
        if match and match.group(2).strip("'\"") == joint:
            start = i
            indent = match.group(1)
            break
    if start is None:
        raise ConfigEditError(
            "no joint named {!r} in this config".format(joint))

    end = len(lines)
    for i in range(start + 1, len(lines)):
        match = name_re.match(lines[i])
        if match and len(match.group(1)) <= len(indent):
            end = i
            break

    # Take the key indent from a key that is actually there, so this works
    # whatever the file's style is.
    key_indent = None
    for line in lines[start:end]:
        found = re.match(r"^(\s*)(id|plugin|gear_ratio):\s", line)
        if found:
            key_indent = found.group(1)
            break
    if key_indent is None:
        raise ConfigEditError(
            "joint {!r} has no actuator keys to match indentation "
            "against".format(joint))
    return start, end, key_indent


def set_joint_value(text: str, joint: str, key: str, value: str) -> str:
    """Set ``key: value`` inside ``joint``'s actuator block.

    Rewrites the existing line if the key is present -- preserving any trailing
    inline comment -- and inserts it after ``id:`` otherwise. Never touches a
    same-named key belonging to a different joint.
    """
    lines = text.splitlines(keepends=True)
    start, end, key_indent = _joint_block(lines, joint)

    key_re = re.compile(r"^(\s*)" + re.escape(key) + r":\s*(.*?)(\s*#.*)?$")
    for i in range(start, end):
        match = key_re.match(lines[i].rstrip("\n"))
        if not match:
            continue
        trailing = match.group(3) or ""
        newline = "\n" if lines[i].endswith("\n") else ""
        lines[i] = "{}{}: {}{}{}".format(
            match.group(1), key, value, trailing, newline)
        return "".join(lines)

    # Absent: put it straight after `id:`, which every actuator block has.
    for i in range(start, end):
        if re.match(r"^\s*id:\s", lines[i]):
            lines.insert(i + 1, "{}{}: {}\n".format(key_indent, key, value))
            return "".join(lines)
    raise ConfigEditError(
        "joint {!r} has no `id:` line to insert {!r} after".format(joint, key))


def set_urdf_joint_limits(
    text: str, joint: str, lower: float, upper: float, decimals: int = 6
) -> str:
    """Set BOTH copies of a joint's limits in a URDF/xacro.

    The ``<limit>`` element (what skrobot clamps to, silently) and the
    ros2_control ``command_interface`` ``min``/``max`` (what bounds a
    controller's command). Editing one and not the other is the drift this
    exists to prevent, so both are required to be present and both are written.
    """
    low = "{:.{}f}".format(lower, decimals)
    high = "{:.{}f}".format(upper, decimals)

    # 1) <joint name="X" type="revolute"> ... <limit lower= upper= .../>
    joint_open = re.search(
        r'<joint\s+name="' + re.escape(joint) + r'"\s+type="[^"]+"\s*>', text)
    if joint_open is None:
        raise ConfigEditError(
            "no <joint name=\"{}\" type=...> in this URDF".format(joint))
    close = text.find("</joint>", joint_open.end())
    if close < 0:
        raise ConfigEditError("unterminated <joint> for {!r}".format(joint))
    body = text[joint_open.end():close]
    new_body, count = re.subn(
        r'(<limit\s+[^>]*?lower=")[^"]*(")([^>]*?\bupper=")[^"]*(")',
        lambda m: m.group(1) + low + m.group(2) + m.group(3) + high + m.group(4),
        body, count=1)
    if count != 1:
        raise ConfigEditError(
            "joint {!r} has no <limit lower= upper=> to update".format(joint))
    text = text[:joint_open.end()] + new_body + text[close:]

    # 2) the ros2_control block's command_interface bounds
    control = re.search(
        r'<joint\s+name="' + re.escape(joint) + r'"\s*>', text)
    if control is None:
        raise ConfigEditError(
            "joint {!r} has no ros2_control <joint> block; its command bounds "
            "would drift from the <limit> above".format(joint))
    close = text.find("</joint>", control.end())
    body = text[control.end():close]
    body, low_n = re.subn(
        r'(<param\s+name="min"\s*>)[^<]*(</param>)',
        lambda m: m.group(1) + low + m.group(2), body, count=1)
    body, high_n = re.subn(
        r'(<param\s+name="max"\s*>)[^<]*(</param>)',
        lambda m: m.group(1) + high + m.group(2), body, count=1)
    if low_n != 1 or high_n != 1:
        raise ConfigEditError(
            "joint {!r}'s ros2_control block has no min/max params to "
            "update".format(joint))
    return text[:control.end()] + body + text[close:]


def find_joint_value(text: str, joint: str, key: str) -> Optional[str]:
    """Read one key back out, for verifying a write by its effect."""
    lines = text.splitlines()
    start, end, _ = _joint_block(lines, joint)
    key_re = re.compile(r"^\s*" + re.escape(key) + r":\s*(.*?)(\s*#.*)?$")
    for i in range(start, end):
        match = key_re.match(lines[i])
        if match:
            return match.group(1).strip()
    return None
