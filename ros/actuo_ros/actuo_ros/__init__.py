"""ROS 2 layer of actuo: the bus servicer's nodes and a scikit-robot interface.

Importing this package imports nothing from ROS. :func:`connect`,
:class:`RobotInterface`, :class:`ServoRefused` and :class:`PoseRefused` are
loaded from :mod:`actuo_ros.robot_interface` on first access.
"""

_LAZY = ('connect', 'fetch_robot', 'PoseRefused', 'RobotInterface',
         'ServoRefused')

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        from actuo_ros import robot_interface
        return getattr(robot_interface, name)
    raise AttributeError(
        'module {!r} has no attribute {!r}'.format(__name__, name))
