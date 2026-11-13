"""ROS shell around a bus's non-joint devices: publish named sensor channels.

Deliberately thin, and deliberately separate from the joint path. Joint
command/state is real-time and goes shm -> C++ -> ros2_control; a sensor board's
telemetry is not real-time and goes out as plain topics from the Python driver.

What this translates, and what it refuses to translate
------------------------------------------------------
A :class:`~actuo.core.contracts.Sensor` reports a ``Telemetry`` dict of named
channels. This maps the ones that have a standard ROS type onto that type and
publishes nothing for the rest -- the rest are still readable with
``actuo-air read`` and can grow a publisher when a consumer
appears. Adding a channel never widens a shared struct.

Two channels are handled here:

``imu/*``
    Published as ``sensor_msgs/Imu`` -- but ONLY once the board has told us its
    full scale. Until :meth:`Kjs03Sensor.decode_identity` has run, the driver
    reports raw counts and no scaled values, and an ``Imu`` message filled with
    counts would look exactly like one filled with m/s^2. So a sensor with no
    known range gets no Imu topic and says so, rather than publishing a number
    that is wrong by a factor of up to 8.

``air/pressure``
    Published as ``std_msgs/Float64`` in **kPa gauge**, on a topic whose name
    says so. NOT ``sensor_msgs/FluidPressure``: that field is documented as
    absolute pressure in Pa, and this board reports gauge (0 kPa at atmosphere,
    -45 kPa under vacuum). Converting would mean inventing an ambient pressure.

The orientation field of the Imu message is left at identity with
``orientation_covariance[0] = -1``, the ``sensor_msgs/Imu`` way of saying "this
sensor does not estimate orientation". This board has an accelerometer and a
gyro and no fusion, so it does not.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64, String

#: Channels that together make up a ``sensor_msgs/Imu``. All six must be present
#: and scaled, or no Imu is published for that sensor.
_IMU_CHANNELS = tuple(
    "imu/{}.{}".format(kind, axis)
    for kind in ("angular_velocity", "linear_acceleration")
    for axis in "xyz"
)

#: Best-effort with a depth of 1: telemetry is a stream of the latest reading,
#: and a late subscriber wants the newest sample, not a replay of a queue.
_TELEMETRY_QOS = QoSProfile(depth=1,
                            history=HistoryPolicy.KEEP_LAST,
                            reliability=ReliabilityPolicy.BEST_EFFORT)

#: RELIABLE, unlike telemetry: a grip/release button press is a discrete,
#: operator-visible event, not a resamplable stream -- losing one is a real
#: "I pressed it and nothing happened", not a stale reading nobody will miss.
#: Depth 5 rather than 1 so a quick double-press cannot overwrite itself before
#: the servicer's next drain.
_COMMAND_QOS = QoSProfile(depth=5,
                          history=HistoryPolicy.KEEP_LAST,
                          reliability=ReliabilityPolicy.RELIABLE)


class SensorTelemetryPublisher:
    """Publish one bus's sensor channels on an existing ROS node.

    Parameters
    ----------
    node : rclpy.node.Node
        The node to publish from. Passed in rather than created so the
        servicer's telemetry lands in the same namespace as its servo topics,
        and so this class owns no lifecycle.
    sensors : sequence of (str, Sensor)
        Named sensors, exactly as :func:`actuo.drivers.config.build_sensors`
        returns them.
    frame_ids : dict, optional
        Sensor name -> the URDF frame its IMU readings are expressed in. A
        sensor with no entry gets no Imu publisher: an IMU published in an
        unknown frame is unusable by anything that transforms it, and guessing
        ``base_link`` would silently rotate every reading.
    """

    def __init__(self, node: Node,
                 sensors: Sequence[Tuple[str, object]],
                 frame_ids: Optional[Dict[str, str]] = None) -> None:
        self.node = node
        self._frame_ids = dict(frame_ids or {})
        self._imu_pubs: Dict[str, object] = {}
        self._pressure_pubs: Dict[str, object] = {}
        self._last_error: Dict[str, int] = {}
        self._sensors: Dict[str, object] = {name: s for name, s in sensors}

        for name, sensor in sensors:
            channels = {spec.name for spec in sensor.channels}
            if all(ch in channels for ch in _IMU_CHANNELS):
                frame = self._frame_ids.get(name)
                if frame:
                    self._imu_pubs[name] = node.create_publisher(
                        Imu, "~/{}/imu".format(name), _TELEMETRY_QOS)
            # The pressure publisher is created lazily in publish(), not here.
            # A KJS-03 only knows it is the AIRRELAY variant -- the one with the
            # pump -- after its first sensor block, because the board revision
            # register sits outside the identity block. At construction time the
            # air channels do not exist yet, so deciding here would silently
            # drop the pressure topic on the very board that has one.

    @property
    def topics(self) -> List[str]:
        """Resolved topic names, for the servicer's start-up banner."""
        out = []
        for name, pub in sorted(self._imu_pubs.items()):
            out.append("{} ({})".format(pub.topic_name, self._frame_ids[name]))
        for _, pub in sorted(self._pressure_pubs.items()):
            out.append(pub.topic_name)
        return out

    def publish(self, name: str, telemetry: dict) -> None:
        """Publish one sensor's latest reading.

        Parameters
        ----------
        name : str
            Sensor name, as given to the constructor.
        telemetry : dict
            Channel name -> value, straight from ``Sensor.decode_state``.
        """
        stamp = self.node.get_clock().now().to_msg()

        pub = self._imu_pubs.get(name)
        if pub is not None:
            # Absent when the board's ranges were never read. Publishing raw
            # counts here would be indistinguishable from m/s^2 downstream.
            if all(ch in telemetry for ch in _IMU_CHANNELS):
                msg = Imu()
                msg.header.stamp = stamp
                msg.header.frame_id = self._frame_ids[name]
                msg.angular_velocity.x = telemetry["imu/angular_velocity.x"]
                msg.angular_velocity.y = telemetry["imu/angular_velocity.y"]
                msg.angular_velocity.z = telemetry["imu/angular_velocity.z"]
                msg.linear_acceleration.x = telemetry["imu/linear_acceleration.x"]
                msg.linear_acceleration.y = telemetry["imu/linear_acceleration.y"]
                msg.linear_acceleration.z = telemetry["imu/linear_acceleration.z"]
                # "no orientation estimate", per sensor_msgs/Imu. The rest stay
                # 0 = "covariance unknown", which is the honest answer for a
                # board whose noise has not been characterised.
                msg.orientation_covariance[0] = -1.0
                pub.publish(msg)

        if "air/pressure" in telemetry:
            pub = self._pressure_pubs.get(name)
            if pub is None:
                pub = self.node.create_publisher(
                    Float64, "~/{}/pressure_kpa".format(name), _TELEMETRY_QOS)
                self._pressure_pubs[name] = pub
                self.node.get_logger().info(
                    "{}: publishing gauge pressure on {}".format(
                        name, pub.topic_name))
            pub.publish(Float64(data=float(telemetry["air/pressure"])))

        self._warn_on_new_error(name, telemetry)

    def _warn_on_new_error(self, name: str, telemetry: dict) -> None:
        """Log the board's hardware error bits, once per change.

        Once per change rather than every cycle: a fault logged 20x a second is
        a fault nobody reads, and a fault that clears silently is one nobody
        knows cleared.

        Which bits count as a fault is asked of the sensor, not decided here --
        a board that permanently flags parts it is not fitted with (the KJS-03
        AIRRELAY does) would otherwise fire a warning on every run and teach
        the operator to ignore the one that matters.
        """
        error = int(telemetry.get("hardware_error", 0))
        if error == self._last_error.get(name, 0):
            return
        previous = self._last_error.get(name, 0)
        self._last_error[name] = error

        sensor = self._sensors.get(name)
        if hasattr(sensor, "real_faults"):
            faults = sensor.real_faults(error)
            was_faulted = bool(sensor.real_faults(previous))
        else:
            faults = [key.split("/")[-1]
                      for key, value in sorted(telemetry.items())
                      if key.startswith("hardware_error/") and value]
            was_faulted = bool(previous)

        if faults:
            self.node.get_logger().warning(
                "{}: hardware error {:#04x} ({})".format(
                    name, error, ", ".join(faults)))
        elif was_faulted:
            self.node.get_logger().info("{}: hardware error cleared".format(name))


class AirCommandReceiver:
    """ROS shell that RECEIVES pump/valve commands. Never writes the bus itself.

    Subscribes ``~/{name}/air/command`` (``std_msgs/String``) for every sensor
    passed in. The transport is owned by the servicer's single control loop
    (one process services the bus at its own rate), so a
    ROS subscription callback -- which can fire on a different thread than the
    loop that reads/writes the wire -- must never touch it directly. This class
    only remembers the latest requested op per sensor; the caller (the main
    loop) drains it with :meth:`pending` at a point in its own cycle where
    writing is safe, the same shape joint commands already take through shm.

    Last-value-wins, not a queue: a second button press before the first is
    drained REPLACES the pending op. A "grip" queued behind an operator's very
    next "release" must never fire after the release -- a queue would do
    exactly that.
    """

    def __init__(self, node: Node, sensors: Sequence[Tuple[str, object]]) -> None:
        self.node = node
        self._pending: Dict[str, str] = {}
        self._subs = [
            node.create_subscription(
                String, "~/{}/air/command".format(name),
                (lambda msg, _name=name: self._on_command(_name, msg)),
                _COMMAND_QOS)
            for name, _sensor in sensors
        ]

    def _on_command(self, name: str, msg) -> None:
        self._pending[name] = msg.data

    def pending(self) -> Dict[str, str]:
        """Return and clear the ops requested since the last call."""
        out, self._pending = self._pending, {}
        return out
