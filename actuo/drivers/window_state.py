"""Remember where each joint was, so the next start-up can pick its gear window.

Why a file
----------
:meth:`CubeMarsActuator.snap_to_window` needs a coarse reference -- roughly where
a joint is -- to choose which gear window its single-turn rotor encoder powered up
in. The obvious source is a constant in the robot config: "this arm rests here".

That constant turned out to be a poor predictor. On a CubeMars arm, joint3's
resting angle was measured drifting 11.9 deg in one sitting while limp, the config
records two rest values 9.4 deg apart, and after the servicer aborted mid-motion
the arm fell to a pose 23.5 deg from the configured one -- enough for the snap to
refuse and the stack to fail to start. A fixed constant is trying to *predict*
where a limp arm settles, which depends on where it was released, what it is
holding and what it is leaning on.

Remembering is strictly easier than predicting. The servicer knows each joint's
correct angle the whole time it runs, so the only error in a remembered reference
is however far the arm sagged while the power was off -- a few degrees -- rather
than the full spread of resting poses. The config constant stays as the first-run
fallback, for a robot this has never run on.

What is stored
--------------
The **window-corrected** position, in radians, per joint name. That is the frame
``snap_to_window`` compares against, so nothing has to be re-derived on load.

And the DATUM each position was recorded under -- the joint's configured
``zero_offset_deg`` and ``direction``, where the joint's plugin reads them as
``model = direction * (raw - zero_offset_deg)``. A position is a number in one
frame; change the zero (a panel re-zero, a hand-edited config) and the same pose
reads differently, and a remembered value in the OLD frame is then wrong by
exactly the change. On a 64:1 joint a 2 deg re-zero already exceeds the snap's
ambiguity margin: the next start-up would refuse, or on a 9:1 joint quietly pick
the neighbouring window. With the datum stored, :func:`load_positions` converts
through the raw angle, which is the one thing a re-zero does not change:

    raw = direction_old * old + zero_old
    new = direction_new * (raw - zero_new)

A file with no datum for a joint (written before this, or a plugin configured
some other way) is read as it was: there is nothing to convert with, and
guessing would be worse.

Failure behaviour
-----------------
Every read failure -- missing, unreadable, truncated, not JSON, wrong shape --
returns "no reference", never an exception and never a partial answer. A file
this layer cannot vouch for must not be able to place a joint in the wrong
window, and falling back to the config constant is always available.

Writes are atomic (temp file + ``os.replace``) because the process this serves is
one that dies abruptly: a crash halfway through a write must leave the previous
reference intact rather than a truncated file that breaks the next start-up.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
import math
from typing import Dict, Optional

#: {joint: {"zero_offset_deg": float, "direction": float}}
Datums = Dict[str, Dict[str, float]]

#: Environment variable naming the state directory outright.
STATE_DIR_ENV = "ACTUO_STATE_DIR"
#: Subdirectory of ``$XDG_STATE_HOME`` used when ``ACTUO_STATE_DIR`` is unset.
STATE_SUBDIR = "actuo"


def state_dir() -> str:
    """Directory holding the per-robot reference files.

    ``$ACTUO_STATE_DIR`` when set; otherwise ``$XDG_STATE_HOME/actuo``, with
    ``XDG_STATE_HOME`` defaulting to ``~/.local/state``. A process that shares
    these files with another program (for example a ROS 2 layer writing them
    somewhere else) sets ``ACTUO_STATE_DIR`` to that program's directory.

    Follows XDG so it survives a reboot, which matters: the arm's physical pose
    survives a reboot too, so a reference in ``/tmp`` would be thrown away
    exactly when it is still valid.
    """
    explicit = os.environ.get(STATE_DIR_ENV)
    if explicit:
        return explicit
    base = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return os.path.join(base, STATE_SUBDIR)


def state_path(name: str) -> str:
    """Path of the reference file for the robot called ``name``.

    ``name`` is the servicer's shm name, which already identifies one robot on
    one machine -- the same key that keeps two arms from sharing a segment.
    """
    return os.path.join(state_dir(), "window_{}.json".format(name))


def _datum(value) -> Optional[Dict[str, float]]:
    """A datum as stored, validated, or None."""
    if not isinstance(value, dict):
        return None
    try:
        zero = float(value["zero_offset_deg"])
        direction = float(value["direction"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(zero) or direction not in (1.0, -1.0):
        return None
    return {"zero_offset_deg": zero, "direction": direction}


def convert(position: float, recorded: Dict[str, float],
            current: Dict[str, float]) -> float:
    """``position`` (rad) under datum ``recorded``, re-expressed under
    ``current``. See the module docstring."""
    raw = recorded["direction"] * math.degrees(position) \
        + recorded["zero_offset_deg"]
    return math.radians(current["direction"]
                        * (raw - current["zero_offset_deg"]))


def load_positions(name: str, datums: Optional[Datums] = None
                   ) -> Dict[str, float]:
    """Return {joint: window-corrected radians} last recorded, or ``{}``.

    With ``datums`` (each joint's CURRENT zero and direction), a position
    recorded under a different datum is converted to the current one.

    Never raises. See the module docstring: a reference this layer cannot vouch
    for must not be able to put a joint in the wrong window.
    """
    try:
        with open(state_path(name)) as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(document, dict):
        return {}
    positions = document.get("positions")
    if not isinstance(positions, dict):
        return {}
    recorded = document.get("datums")
    if not isinstance(recorded, dict):
        recorded = {}
    out = {}
    for joint, value in positions.items():
        # A NaN would sail through float() and then through the snap's
        # arithmetic, producing a window of int(round(nan)) -- an exception far
        # from here, or worse, silence. Reject anything not a real number.
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if number != number or number in (float("inf"), float("-inf")):
            continue
        was = _datum(recorded.get(joint))
        now = _datum((datums or {}).get(joint))
        if was is not None and now is not None:
            number = convert(number, was, now)
        out[str(joint)] = number
    return out


def load_age(name: str) -> Optional[float]:
    """Seconds since the reference was written, or ``None`` if unknown.

    Age does NOT invalidate a reference -- what matters is whether the arm moved,
    not how long ago it was recorded. It is exposed so the operator can see it
    and judge, which is a different thing from the code deciding.
    """
    try:
        with open(state_path(name)) as handle:
            document = json.load(handle)
        stamp = float(document["stamp"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return max(0.0, time.time() - stamp)


def save_positions(name: str, positions: Dict[str, float],
                   datums: Optional[Datums] = None) -> bool:
    """Record the current window-corrected positions. Returns success.

    Never raises: failing to save a reference must not take down a running
    robot. The cost of a failed save is one start-up falling back to the config
    constant, which is the behaviour that existed before this file.
    """
    if not positions:
        return False
    document = {
        "version": 1,
        "stamp": time.time(),
        "positions": {str(k): float(v) for k, v in positions.items()},
    }
    kept = {str(k): _datum(v) for k, v in (datums or {}).items()
            if k in positions and _datum(v) is not None}
    if kept:
        document["datums"] = kept
    path = state_path(name)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Atomic: the reader either sees the old file or the new one, never a
        # half-written one. This process is expected to die abruptly.
        handle = tempfile.NamedTemporaryFile(
            "w", dir=os.path.dirname(path), prefix=".window_", suffix=".tmp",
            delete=False)
        try:
            json.dump(document, handle)
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        os.replace(handle.name, path)
        return True
    except OSError:
        try:
            os.unlink(handle.name)
        except (OSError, NameError, UnboundLocalError):
            pass
        return False


def update_positions(name: str, limp: Dict[str, float],
                     held: Optional[set] = None,
                     datums: Optional[Datums] = None) -> bool:
    """Record the LIMP joints' positions and keep the held joints' last ones.

    A joint holding torque is not where it will be after an abrupt death: the
    torque goes with the process and the joint falls. Recording it held made
    the next start-up refuse, twice on 2026-09-29/30: a biped's arms were
    held 30 deg forward when the legs' bus aborted, the file said -29 deg, the
    arm that had dropped to -8 was 19-21 deg from it, and the upper body would
    not start until the file was moved aside by hand. Where a held joint lands
    is best predicted by where it last was limp -- the rest pose it was picked
    up from -- so that value is kept until the joint goes limp again.

    Parameters
    ----------
    name : str
        The shm name the file is keyed by.
    limp : dict
        Joint -> window-corrected position (rad) for joints with no torque.
    held : set, optional
        Joints holding torque. Their previously recorded value is kept; a held
        joint with no previous record is simply not recorded, which falls back
        to the config's startup reference next time -- the arm's rest pose.
    datums : dict, optional
        Each joint's current zero and direction. Kept values are converted to
        them before they are written back, so the whole file is in one frame.

    Returns
    -------
    bool
        Whether a file was written.
    """
    merged = {k: v for k, v in load_positions(name, datums).items()
              if k in (held or set())}
    merged.update(limp)
    return save_positions(name, merged, datums)
