"""Teach-and-playback recordings: joint angles over time, kept as YAML files.

ROS-free: the recorder, the file format and the playback timing. The node that
records ``/joint_states`` and plays a recording back through the trajectory
controllers is ``actuo_ros``'s ``actuo_teach_node``.

Ported from attachable_arm's feetech_drivers.
"""

import datetime
import os
import re

import numpy as np
import yaml

_NAME = re.compile(r'^[A-Za-z0-9_-]+$')


def check_name(name):
    """Raise ValueError unless ``name`` is usable as a file name."""
    if not _NAME.match(name or ''):
        raise ValueError("recording name must be letters, digits, '-' or '_', got {!r}"
                         .format(name))
    return name


class Recorder:
    """Collects joint angles at most every ``1 / rate`` seconds.

    Parameters
    ----------
    joint_names : sequence of str
        Joints to keep, in this order.
    rate : float
        Highest sampling rate [Hz]; faster input is thinned out.
    """

    def __init__(self, joint_names, rate):
        self.joint_names = list(joint_names)
        self.min_interval = 1.0 / rate
        self.times = []
        self.positions = []

    def add(self, stamp, names, positions):
        """Fold in one joint state taken at ``stamp`` [s]; other joints are ignored.

        Returns
        -------
        bool
            Whether the sample was kept.
        """
        # 1 us of slack so samples exactly one interval apart are not dropped.
        if self.times and stamp - self.times[-1] < self.min_interval - 1e-6:
            return False
        lookup = dict(zip(names, positions))
        if any(name not in lookup for name in self.joint_names):
            return False
        self.times.append(float(stamp))
        self.positions.append([float(lookup[name]) for name in self.joint_names])
        return True

    @property
    def duration(self):
        return self.times[-1] - self.times[0] if len(self.times) > 1 else 0.0

    def to_dict(self, name):
        if len(self.times) < 2:
            raise ValueError('nothing recorded (need at least 2 samples, got {})'
                             .format(len(self.times)))
        t0 = self.times[0]
        return {
            'name': name,
            'recorded': datetime.datetime.now().isoformat(timespec='seconds'),
            'joint_names': self.joint_names,
            'times': [round(t - t0, 4) for t in self.times],
            'positions': [[round(p, 5) for p in row] for row in self.positions],
        }


def save(directory, recording):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, check_name(recording['name']) + '.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(recording, f, sort_keys=False, default_flow_style=None)
    return path


def load(directory, name):
    path = os.path.join(directory, check_name(name) + '.yaml')
    if not os.path.exists(path):
        raise ValueError('no recording named {!r} in {}'.format(name, directory))
    with open(path) as f:
        recording = yaml.safe_load(f)
    times = np.asarray(recording['times'], dtype=np.float64)
    positions = np.asarray(recording['positions'], dtype=np.float64)
    if positions.shape != (len(times), len(recording['joint_names'])) or len(times) < 2:
        raise ValueError('{} is malformed'.format(path))
    if np.any(np.diff(times) <= 0):
        raise ValueError('{}: times must increase'.format(path))
    return recording['joint_names'], times, positions


def listing(directory):
    """(name, duration [s]) of every recording, sorted by name."""
    if not os.path.isdir(directory):
        return []
    out = []
    for entry in sorted(os.listdir(directory)):
        name, ext = os.path.splitext(entry)
        if ext != '.yaml' or not _NAME.match(name):
            continue
        try:
            _, times, _ = load(directory, name)
        except (ValueError, KeyError, yaml.YAMLError):
            continue
        out.append((name, float(times[-1] - times[0])))
    return out


def approach_duration(current, start, speed):
    """Time [s] to move from ``current`` to ``start`` at ``speed`` [rad/s], >= 1 s."""
    distance = np.max(np.abs(np.asarray(start) - np.asarray(current)))
    return max(1.0, float(distance) / speed)


def playback_velocities(times, positions):
    """Joint velocities at each point, for a trajectory that does not stop at
    every sample: central differences inside, zero at both ends.

    Parameters
    ----------
    times : sequence of float
        time_from_start of each point [s], increasing.
    positions : array_like, shape (n_points, n_joints)

    Returns
    -------
    numpy.ndarray, shape (n_points, n_joints)
    """
    times = np.asarray(times, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    velocities = np.zeros_like(positions)
    if len(times) > 2:
        velocities[1:-1] = ((positions[2:] - positions[:-2])
                            / (times[2:] - times[:-2])[:, None])
    return velocities


def playback_times(times, speed, lead=0.05):
    """time_from_start of each point when played at ``speed`` x.

    The first point comes ``lead`` seconds in, since the arm already stands
    there after the approach move.
    """
    if speed <= 0:
        raise ValueError('speed must be > 0, got {}'.format(speed))
    times = np.asarray(times, dtype=np.float64)
    return (times - times[0]) / speed + lead
