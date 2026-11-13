"""File-backed shared-memory seqlock bridging the C++ ros2_control hardware
interface and the Python bus-servicer.

This is the seam between C++ and Python: a C++ ros2_control hardware component
(a plugin that lives outside this package; its name is passed to
``robot_document.hardware_urdf`` as ``hardware_plugin``) runs in the
controller's RT read/write loop and exchanges only flat float64 command/state
vectors with this Python process, the servicer, which owns the bus (or the sim
plant). No protocol knowledge crosses this boundary -- there is nothing to
duplicate.

Contract with the C++ reader. Two values are not chosen by this file alone;
the C++ side must use exactly the same ones or it will not find, or will
refuse, the segment:

* the path prefix: the file is ``/tmp/<prefix><name>.shm``
  (:func:`default_path`), and the servicer's lock file is
  ``/tmp/<prefix><name>.lock`` (:func:`lock_path`). The prefix is
  :data:`DEFAULT_PREFIX` (``"actuo_"``) unless the environment variable
  ``ACTUO_SHM_PREFIX`` is set, and every function here also takes an explicit
  ``prefix`` argument that wins over both;
* the magic: :data:`MAGIC` = ``0x32544341``, which is the four bytes
  ``b"ACT2"`` read as a little-endian uint32. :class:`ShmBridge` takes a
  ``magic`` argument to speak to a reader built with another value.

Layout (little-endian, one mmap'd file ``/tmp/<prefix><name>.shm``)::

    off  type      field
    0    uint32    magic  = MAGIC (default 0x32544341, bytes 'ACT2')
    4    uint32    n_joints       slots, not "joints the controller uses"
    8    uint32    cmd_stride     floats per slot in command block (3)
    12   uint32    state_stride   floats per slot in state block (3)
    16   uint64    cmd_seq        seqlock counter, writers = C++ components
    24   uint64    state_seq      seqlock counter, writer = Python
    32   uint64    epoch          changes every time the servicer creates it
    40   ...       reserved (zero) up to 64
    64   f64[n*cmd_stride]        command block  [position, velocity, effort]
    ...  f64[n*state_stride]      state block    [position, velocity, present]

Ownership (layout version 2). The SERVICER creates the segment, once, at its
start, and is the only side that ever zeroes it. Every ros2_control hardware
component attaches to it and touches only its own slots. That is what lets one
module be deactivated and re-activated while the others keep running: in
version 1 the C++ side recreated (and zeroed) the whole segment on every
activation, which with one component per module would wipe every other
module's command.

Two conventions carry what version 1 could not say:

* A command POSITION of NaN means "no controller owns this slot". The servicer
  creates the block all-NaN and a component writes NaN back over its slots when
  it deactivates. The supervisor holds a position joint where it was and stops
  a velocity/current joint (see ``ServoSupervisor.service``).
* ``present`` is 1.0 while the slot's actuator answers and 0.0 when it does
  not. A component whose slot reads 0.0 returns DEACTIVATE from write(), which
  makes controller_manager deactivate exactly the controllers commanding it.

``epoch`` lets an attached component notice that the servicer restarted and
re-created the segment underneath it: it must not keep commanding through a
segment whose state it adopted from a previous servicer.

Seqlock: the writer bumps its counter to odd, writes, bumps to even; a reader
retries while the counter is odd or changed across the read. Aligned 8-byte
loads/stores are atomic on x86-64/arm64. Memory ordering is best-effort (no
fence primitive from Python).
"""

from __future__ import annotations

import errno
import math
import mmap
import os
import struct
import time
from typing import Optional, Tuple

import numpy as np

#: Default magic at offset 0: the bytes ``b"ACT2"`` as a little-endian uint32.
#: A C++ reader must check for the same value; :class:`ShmBridge` takes
#: ``magic=`` to match a reader built with another one.
MAGIC = 0x32544341
#: Default file-name prefix of the segment and lock files under ``/tmp``.
DEFAULT_PREFIX = "actuo_"
#: Environment variable that overrides :data:`DEFAULT_PREFIX`.
PREFIX_ENV = "ACTUO_SHM_PREFIX"
#: Directory of the segment and lock files. Fixed rather than TMPDIR so the
#: C++ and Python sides compute the identical path (macOS gives each process a
#: different ``/var/folders`` TMPDIR).
SHM_DIR = "/tmp"
_HEADER = struct.Struct("<IIII")   # magic, n, cmd_stride, state_stride
_SEQ = struct.Struct("<Q")
_HEADER_SIZE = 64
_CMD_SEQ_OFF = 16
_STATE_SEQ_OFF = 24
_EPOCH_OFF = 32
_MAX_SPIN = 1000

#: Floats per slot in each block. Fixed by the format rather than chosen per
#: segment: both sides index slots by these, and a mismatch is refused.
CMD_STRIDE = 3
STATE_STRIDE = 3
#: Index of the presence flag inside one slot of the state block.
STATE_PRESENT = 2


def shm_prefix(prefix: Optional[str] = None) -> str:
    """The file-name prefix in effect.

    Parameters
    ----------
    prefix : str or None
        Explicit prefix; wins when not None.

    Returns
    -------
    str
        ``prefix`` if given, else ``$ACTUO_SHM_PREFIX`` if set, else
        :data:`DEFAULT_PREFIX`.
    """
    if prefix is not None:
        return prefix
    return os.environ.get(PREFIX_ENV, DEFAULT_PREFIX)


def default_path(name: str, prefix: Optional[str] = None) -> str:
    """Runtime path ``/tmp/<prefix><name>.shm`` of a named segment.

    The prefix is part of the contract with the C++ side, which opens exactly
    this path: both must be given the same one (see :func:`shm_prefix`).
    """
    return os.path.join(SHM_DIR, "{}{}.shm".format(shm_prefix(prefix), name))


def lock_path(name: str, prefix: Optional[str] = None) -> str:
    """Path ``/tmp/<prefix><name>.lock`` of the lock a servicer for the
    segment ``name`` holds while it runs.

    Tools that write to the bus themselves check it to refuse running beside a
    servicer, which would make two writers on one bus.
    """
    return os.path.join(SHM_DIR, "{}{}.lock".format(shm_prefix(prefix), name))


def lock_holder(name: str, prefix: Optional[str] = None) -> Optional[str]:
    """Whether a live servicer holds the lock of segment ``name``.

    Checked by trying the lock rather than by scanning a process list: the lock
    is what actually decides, and the kernel releases it when the holder dies
    however it died.

    Parameters
    ----------
    name : str
        Segment name, as the servicer was given it.
    prefix : str or None
        File-name prefix (see :func:`shm_prefix`).

    Returns
    -------
    str or None
        None when no lock file exists or nobody holds it; otherwise the lock
        file's content (the holder's pid, when it wrote one), possibly empty.
    """
    import fcntl   # POSIX only, like the /tmp segment itself

    path = lock_path(name, prefix)
    if not os.path.exists(path):
        return None
    try:
        handle = open(path, "r")
    except OSError:
        return None
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(handle, fcntl.LOCK_UN)
    except OSError as err:
        if err.errno in (errno.EACCES, errno.EAGAIN):
            return handle.read().strip()
        raise
    finally:
        handle.close()
    return None


def _total_size(n: int, cmd_stride: int, state_stride: int) -> int:
    return _HEADER_SIZE + n * cmd_stride * 8 + n * state_stride * 8


class ShmBridge:
    """One end of the shared-memory exchange.

    Parameters
    ----------
    name : str
        Segment name; the file is ``default_path(name)`` unless ``path`` is set.
    n_joints : int
        Number of slots.
    cmd_stride, state_stride : int
        Floats per slot. Anything but ``CMD_STRIDE``/``STATE_STRIDE`` is only
        for tests of the layout itself.
    create : bool
        True for the servicer: create (or re-create in place) the file, zero
        it, give it a new epoch, and mark every command slot unowned (NaN) and
        every state slot absent. False to attach and validate the header.
    path : str
        Override of the file path.
    prefix : str or None
        File-name prefix for ``default_path`` (see :func:`shm_prefix`);
        ignored when ``path`` is set.
    magic : int
        Magic written at creation and required on attach. Must equal the C++
        reader's.
    """

    def __init__(
        self,
        name: str,
        n_joints: int,
        cmd_stride: int = CMD_STRIDE,
        state_stride: int = STATE_STRIDE,
        create: bool = False,
        path: str = "",
        prefix: Optional[str] = None,
        magic: int = MAGIC,
    ) -> None:
        self.name = name
        self.n = n_joints
        self.cmd_stride = cmd_stride
        self.state_stride = state_stride
        self.magic = magic
        self.path = path or default_path(name, prefix)
        size = _total_size(n_joints, cmd_stride, state_stride)

        if create:
            fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o600)
            os.ftruncate(fd, size)
        else:
            fd = os.open(self.path, os.O_RDWR)
            actual = os.fstat(fd).st_size
            if actual < size:
                os.close(fd)
                raise ValueError("shm {!r}: {} bytes, layout needs {}".format(
                    self.path, actual, size))
        self._fd = fd
        self._mm = mmap.mmap(fd, size)

        if not create:
            magic, n, cs, ss = _HEADER.unpack(self._mm[0:_HEADER.size])
            if magic != self.magic:
                raise ValueError(
                    "shm {!r}: bad magic 0x{:08X} (want 0x{:08X})".format(
                        self.path, magic, self.magic))
            if (n, cs, ss) != (n_joints, cmd_stride, state_stride):
                raise ValueError(
                    "shm {!r}: layout mismatch got {} want {}".format(
                        self.path, (n, cs, ss),
                        (n_joints, cmd_stride, state_stride)))

        self._cmd_off = _HEADER_SIZE
        self._state_off = _HEADER_SIZE + n_joints * cmd_stride * 8
        # numpy views over the mmap buffer (zero-copy)
        buf = np.frombuffer(self._mm, dtype=np.float64)
        base = _HEADER_SIZE // 8
        self._cmd = buf[base:base + n_joints * cmd_stride]
        self._state = buf[base + n_joints * cmd_stride:
                          base + n_joints * cmd_stride + n_joints * state_stride]

        if create:
            # The magic goes in LAST. A component polling for the file can
            # find it the moment it exists; it must not accept a half-written
            # header or a payload from the previous session as this one's.
            self._mm[0:size] = b"\x00" * size
            self._cmd[:] = math.nan
            state = np.full((n_joints, state_stride), math.nan)
            if state_stride > STATE_PRESENT:
                state[:, STATE_PRESENT] = 0.0
            self._state[:] = state.ravel()
            self._mm[_EPOCH_OFF:_EPOCH_OFF + 8] = _SEQ.pack(
                time.time_ns() & 0xFFFFFFFFFFFFFFFF)
            self._mm[4:_HEADER.size] = _HEADER.pack(
                0, n_joints, cmd_stride, state_stride)[4:]
            self._mm[0:4] = struct.pack("<I", self.magic)

    # --- seqlock helpers ---
    def _load_seq(self, off: int) -> int:
        return _SEQ.unpack(self._mm[off:off + 8])[0]

    def _store_seq(self, off: int, value: int) -> None:
        self._mm[off:off + 8] = _SEQ.pack(value & 0xFFFFFFFFFFFFFFFF)

    def _write_block(self, off: int, view: np.ndarray, data: np.ndarray) -> None:
        s = self._load_seq(off)
        self._store_seq(off, s + 1)   # odd: write in progress
        view[:] = data
        self._store_seq(off, s + 2)   # even: done

    def _read_block(self, off: int, view: np.ndarray) -> np.ndarray:
        for _ in range(_MAX_SPIN):
            s1 = self._load_seq(off)
            if s1 & 1:
                continue
            out = np.array(view, dtype=np.float64)  # copy under the lock
            s2 = self._load_seq(off)
            if s1 == s2:
                return out
        # torn read after spinning: surface it rather than return garbage
        raise RuntimeError("shm {!r}: seqlock read did not converge".format(self.name))

    # --- header ---
    def cmd_seq(self) -> int:
        """Sequence counter of the command block; 0 means never written."""
        return self._load_seq(_CMD_SEQ_OFF)

    def state_seq(self) -> int:
        """Sequence counter of the state block; 0 means never written."""
        return self._load_seq(_STATE_SEQ_OFF)

    def epoch(self) -> int:
        """Changes every time a servicer creates the segment."""
        return self._load_seq(_EPOCH_OFF)

    # --- Python (state writer / command reader) API ---
    def write_state(self, state: np.ndarray) -> None:
        self._write_block(_STATE_SEQ_OFF, self._state, np.asarray(state, np.float64).ravel())

    def read_command(self) -> np.ndarray:
        return self._read_block(_CMD_SEQ_OFF, self._cmd)

    # --- command writer API (used by tests standing in for the C++ side) ---
    def write_command(self, command: np.ndarray) -> None:
        self._write_block(_CMD_SEQ_OFF, self._cmd, np.asarray(command, np.float64).ravel())

    def write_command_slots(self, slots, values) -> None:
        """Write only ``slots`` (each a full ``cmd_stride`` row), as one
        component does. ``values`` has shape ``(len(slots), cmd_stride)``."""
        values = np.asarray(values, np.float64).reshape(len(slots), self.cmd_stride)
        s = self._load_seq(_CMD_SEQ_OFF)
        self._store_seq(_CMD_SEQ_OFF, s + 1)
        for slot, row in zip(slots, values):
            self._cmd[slot * self.cmd_stride:(slot + 1) * self.cmd_stride] = row
        self._store_seq(_CMD_SEQ_OFF, s + 2)

    def read_state(self) -> np.ndarray:
        return self._read_block(_STATE_SEQ_OFF, self._state)

    def close(self, unlink: bool = False) -> None:
        self._cmd = None
        self._state = None
        self._mm.close()
        os.close(self._fd)
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


def layout(name: str, n: int, cmd_stride: int = CMD_STRIDE,
           state_stride: int = STATE_STRIDE,
           prefix: Optional[str] = None) -> Tuple[str, int]:
    """Return (path, total_size) for the C++ side / diagnostics."""
    return (default_path(name, prefix),
            _total_size(n, cmd_stride, state_stride))


def wait_for_commands(shm: ShmBridge, slots, timeout: float,
                      poll: float = 0.02) -> bool:
    """Wait until every one of ``slots`` has a finite commanded position.

    Only for a servicer that must enable torque by itself (``--no-ros``):
    everywhere else torque starts off and enabling refuses an unowned slot, so
    nothing needs to wait.

    Parameters
    ----------
    shm : ShmBridge
        The created segment.
    slots : sequence of int
        Slots that must be commanded.
    timeout : float
        Seconds to wait.
    poll : float
        Seconds between checks.

    Returns
    -------
    bool
        True once all are commanded, False at ``timeout``.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        command = shm.read_command()
        if all(math.isfinite(command[s * shm.cmd_stride]) for s in slots):
            return True
        time.sleep(poll)
    return False
