"""ROS diagnostics shell for one bus.

Deliberately thin: policy lives in the ROS-free layer (`Bus.diagnose()` and the
servicer's generic metrics calculator). This module only translates `DiagItem`
to `diagnostic_msgs/DiagnosticStatus` and publishes.

`bus.diagnose()` is intentionally called by an explicit low-rate scheduler in
`actuo_bus_node.py` (default ~1 Hz), not from the control loop's per-cycle read/
write path.
"""

from __future__ import annotations

from typing import Callable, List, Sequence

from diagnostic_msgs.msg import DiagnosticArray
from diagnostic_updater import DiagnosticStatusWrapper, Updater

from actuo.core.diagnostics import DiagItem, DiagLevel


def hardware_id_for_bus(bus) -> str:
    """Identify the physical transport channel for operator diagnostics."""
    transport = getattr(bus, "transport", None)
    if transport is None:
        return "{}:transport=unknown".format(type(bus).__name__)
    parts = [
        "bus={}".format(type(bus).__name__),
        "transport={}".format(type(transport).__name__),
    ]
    for key in ("interface", "channel", "bitrate"):
        if hasattr(transport, key):
            parts.append("{}={}".format(key, getattr(transport, key)))
    return ",".join(parts)


class BusDiagnosticsPublisher:
    """Publish generic + backend diagnostics for one bus on an existing node.

    Parameters
    ----------
    node : rclpy.node.Node
        Existing node used by the servicer.
    bus : Bus
        Backend bus implementation exposing `diagnose()`.
    generic_items : callable
        Returns one or more generic `DiagItem` values computed in the servicer.
    hardware_id : str
        Physical bus identity (channel/interface) for multi-robot disambiguation.
    topic : str
        Topic name. Defaults to the **absolute** ``/diagnostics``, which is a
        ROS-wide convention rather than a per-robot topic: ``rqt_robot_monitor``
        and ``diagnostic_aggregator`` subscribe to it by that exact name. A
        relative name here would be namespaced to e.g. ``/my_robot/diagnostics``,
        where none of that tooling would find it. Measured, not assumed --
        ``diagnostic_updater``'s own ``Updater`` publishes to ``/diagnostics``.

    Notes
    -----
    ``diagnostic_updater.Updater`` declares ``diagnostic_updater.period`` on
    the node. Constructing a second ``BusDiagnosticsPublisher`` on the same node
    raises ``ParameterAlreadyDeclaredException``.
    """

    def __init__(self, node, bus,
                 generic_items: Callable[[], Sequence[DiagItem]],
                 hardware_id: str, topic: str = "/diagnostics") -> None:
        self.node = node
        self.bus = bus
        self._generic_items = generic_items
        self._hardware_id = str(hardware_id)

        # Use diagnostic_updater's publication path, but schedule explicitly
        # from the servicer loop rather than via Updater's timer callback.
        self._updater = Updater(node, period=3600.0, starting_up_status=None)
        self._updater.setHardwareID(self._hardware_id)
        self._updater.publisher = node.create_publisher(DiagnosticArray, topic, 1)
        if hasattr(self._updater, "timer") and hasattr(self._updater.timer, "cancel"):
            self._updater.timer.cancel()

    @staticmethod
    def diag_item_to_status(item: DiagItem,
                            hardware_id: str) -> DiagnosticStatusWrapper:
        """Translate one ROS-free diagnostic into ROS message fields.

        ``DiagnosticStatus.level`` is a ROS ``byte`` field, which rclpy
        represents as a one-byte ``bytes`` -- its own constants are ``b'\\x00'``
        .. ``b'\\x03'``. ``DiagnosticStatusWrapper.summary`` assigns whatever it
        is given straight through with no coercion, so handing it a plain
        ``int`` stores an ``int`` that **serialises to 0xff**: every status then
        arrives at the operator with an invalid level while looking correct on
        the publishing side. Confirmed over a real DDS round trip, and invisible
        to any test that fakes the node, which is why one of the tests here
        publishes for real.
        """
        status = DiagnosticStatusWrapper()
        status.name = str(item.name)
        status.hardware_id = str(hardware_id)
        status.summary(bytes([int(item.level)]), str(item.message))
        for key in sorted(item.values):
            status.add(str(key), str(item.values[key]))
        return status

    def publish(self) -> List[DiagItem]:
        """Publish one diagnostics sample.

        Returns the items that were published (useful for tests).
        """
        items = list(self._generic_items() or [])
        try:
            items.extend(self.bus.diagnose())
        except Exception as error:  # noqa: BLE001
            items.append(DiagItem(
                name="bus/diagnostics_runtime",
                level=DiagLevel.ERROR,
                message="bus.diagnose() failed: {}".format(error),
                values={
                    "exception_type": type(error).__name__,
                },
            ))
        statuses = [
            self.diag_item_to_status(item, self._hardware_id)
            for item in items
        ]
        self._updater.publish(statuses)
        return items
