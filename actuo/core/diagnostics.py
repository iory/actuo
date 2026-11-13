"""ROS-free diagnostics primitives shared by backend implementations.

The ROS layer should only translate these items to ROS message types. It should
not decide levels or rewrite backend conclusions.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Iterable, Mapping


class DiagLevel(enum.IntEnum):
    """Diagnostic severity.

    Integer values intentionally match ``diagnostic_msgs/DiagnosticStatus``:
    ``OK=0``, ``WARN=1``, ``ERROR=2``, ``STALE=3``. This deliberate parity
    keeps the ROS layer a translation step, not a policy decision point.
    """

    OK = 0
    WARN = 1
    ERROR = 2
    STALE = 3


@dataclass(frozen=True)
class DiagItem:
    """One operator-facing diagnostic line.

    Parameters
    ----------
    name : str
        Path-like item id, e.g. ``"servo/rarm_joint0"``.
    level : DiagLevel
        Severity.
    message : str
        One-line operator message describing what is wrong and what to check.
    values : mapping of str to str
        Free-form detail fields (all strings to keep rendering simple).
    """

    name: str
    level: DiagLevel
    message: str
    values: Mapping[str, str]


def fold_diag_level(items: Iterable[DiagItem],
                    default: DiagLevel = DiagLevel.OK) -> DiagLevel:
    """Return the worst level among ``items``.

    Severity order follows ``DiagLevel``'s integer ordering, i.e.
    ``OK < WARN < ERROR < STALE``.
    """
    worst = default
    for item in items:
        if item.level > worst:
            worst = item.level
    return worst
