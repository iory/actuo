"""Layer contracts for the actuo actuator-abstraction stack.

This module is deliberately ROS-free and dependency-light (numpy only) so it can
be unit-tested standalone. It encodes the package's non-negotiable design
principles:

* Transport, Actuator, and Bus are three independent layers.
* A ``Transport`` moves opaque frames and knows nothing about angles or
  registers, and must not assume a fixed-length synchronous reply.
* An ``Actuator`` owns its own units, calibration, capabilities, and operating
  modes; it knows nothing about serial vs CAN.
* A ``Bus`` groups actuators sharing a transport and builds one batch
  (sync-write) transaction per control cycle -- batch is the default, not a
  loop of singles.
* Identity is opaque (``ActuatorId``) and never narrowed to ``uint8`` at a
  shared layer.
* Unsupported requests fail loudly (see ``CapabilityError``); nothing silently
  degrades.
"""

from __future__ import annotations

import abc
import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from actuo.core.diagnostics import DiagItem

# A device identity is intentionally opaque: a serial int id, a CAN arbitration
# id (int), a MAC/string endpoint, or a tuple for composite addressing. It is
# never narrowed at a shared layer. (``ActuatorId`` kept as an alias.)
DeviceId = Union[int, str, Tuple[Union[int, str], ...]]
ActuatorId = DeviceId


@dataclass(frozen=True)
class Frame:
    """One unit of transport traffic: an opaque payload plus its addressing.

    This is deliberately a struct rather than bare ``bytes``. A CAN frame is
    (arbitration id, payload, extended flag); a UDP datagram is (endpoint,
    payload); a serial frame is a bare payload. Collapsing all of that into
    ``bytes`` forces an addressed protocol to smuggle its address inside the
    payload -- which is precisely the narrowing that made CAN arbitration ids
    unrepresentable in an earlier design that collapsed every id to ``uint8``.
    Identity here is rich and is never narrowed at a shared layer.

    The transport routes on ``address``/``meta`` but still never interprets
    ``payload`` -- the actuator model owns that. ``meta`` carries transport-
    family details the address alone cannot express (for CAN, ``extended``);
    it is written by the actuator that emits the frame and read only by the
    matching transport, so adding a family needs no core edit.

    Attributes
    ----------
    payload : bytes
        Opaque protocol bytes. Never inspected by a transport.
    address : DeviceId, optional
        Where the frame goes to / came from. For CAN this is the arbitration id
        (which for CubeMars-style protocols also encodes the control mode);
        for serial buses a device id; for UDP an endpoint. ``None`` means the
        transport's single implicit endpoint.
    meta : Mapping[str, Any]
        Transport-family metadata, e.g. ``{"extended": True}`` for CAN.
    """

    payload: bytes
    address: Optional[DeviceId] = None
    meta: Mapping[str, Any] = field(default_factory=dict)

# Telemetry is a set of named, typed channels -- never a fixed struct. This is
# the deliberate fix for the fixed-struct trap (a single
# BulkStatusRaw{position,velocity,load,voltage,temperature} that every device
# had to squeeze into) and rcb4's ad-hoc pressure/IMU/pneumatic fields: a
# custom board (e.g. a barometric pressure sensor) declares its own channels and
# flows through the stack with ZERO core edits. Values are keyed by channel name.
Telemetry = Dict[str, float]


class OperatingMode(enum.Enum):
    """First-class operating modes. Not every actuator supports every mode;
    an actuator declares what it supports via :class:`Capabilities`."""

    POSITION = "position"
    VELOCITY = "velocity"
    CURRENT = "current"  # torque, in engineering units (N*m via calibration)
    PWM = "pwm"


class ChannelKind(enum.Enum):
    STATE = "state"      # read from the device (sensor value, measured state)
    COMMAND = "command"  # written to the device (setpoint, actuator input)


@dataclass(frozen=True)
class ChannelSpec:
    """Declares one named, typed data channel a device exposes.

    A channel name is an arbitrary string (e.g. ``"position"``,
    ``"air_board/pressure"``, ``"imu/linear_acceleration.x"``). ``ros_interface``
    is how it surfaces to ros2_control when routed through the controller graph
    (a joint interface name, or a ``<sensor>``/``<gpio>`` interface). Custom
    channels are first-class -- adding one is a plugin concern, never a core edit.
    """

    name: str
    unit: str
    kind: ChannelKind = ChannelKind.STATE
    description: str = ""
    ros_interface: Optional[str] = None  # defaults to ``name`` when unset


class CapabilityError(RuntimeError):
    """Raised when a caller requests something an actuator cannot do.

    Unsupported operations fail loudly rather than silently returning zeros or
    a faked value: a fallback that hides breakage is never acceptable.
    """


@dataclass(frozen=True)
class Capabilities:
    """What an actuator model can actually do. Checked, never assumed."""

    modes: Tuple[OperatingMode, ...]
    has_position_sense: bool = True
    has_velocity_sense: bool = False
    has_current_sense: bool = False
    has_temperature_sense: bool = False
    multiturn: bool = False

    def require(self, mode: OperatingMode) -> None:
        if mode not in self.modes:
            raise CapabilityError(
                "actuator does not support mode {!r}; supported: {}".format(
                    mode.value, [m.value for m in self.modes]
                )
            )


@dataclass
class ActuatorCommand:
    """A command in engineering units. Fields not relevant to the active mode
    are ``None``. Feedforward terms enable torque/velocity+FF control needed for
    dynamic tasks such as walking; operating modes are first-class, and a mode
    an actuator does not declare is refused."""

    mode: OperatingMode
    position: Optional[float] = None       # rad
    velocity: Optional[float] = None       # rad/s
    current: Optional[float] = None        # N*m (via calibration)
    pwm: Optional[float] = None            # duty [-1, 1]
    velocity_ff: Optional[float] = None    # rad/s feedforward
    current_ff: Optional[float] = None     # N*m feedforward


@dataclass
class ActuatorState:
    """Measured state in engineering units. Unavailable channels stay ``None``
    -- they are never faked to 0."""

    position: Optional[float] = None       # rad
    velocity: Optional[float] = None       # rad/s
    current: Optional[float] = None        # N*m
    # deg C -- but WHICH temperature is actuator-specific, and the difference
    # matters the moment anyone reasons about thermal headroom from it. A
    # CubeMars AK reports its DRIVER BOARD (limit 100 C), not its windings, and
    # the vendor's own field name says "Motor Temperature" -- see
    # CubeMarsActuator.decode_state. So an actuator model must document which
    # sensor this is; nothing here can infer it, and "degC" alone invites
    # comparing two robots' numbers as if they measured the same thing.
    temperature: Optional[float] = None
    voltage: Optional[float] = None        # V
    error: Optional[int] = None            # actuator-specific error/status word
    # Arbitrary extra named channels this actuator also reports (e.g. a custom
    # board bolted onto the servo). Standard fields stay typed; anything beyond
    # them lives here as named channels -- no fixed-struct trap.
    aux: Telemetry = field(default_factory=dict)


class Transport(abc.ABC):
    """Moves opaque frames. Knows nothing about actuators, angles, or registers.

    Concrete transports: serial(USB), can(socketcan), udp(gateway), ethercat...
    Implementations must support async/broadcast and must NOT assume a
    fixed-length synchronous reply.
    """

    @abc.abstractmethod
    def open(self) -> None: ...

    @abc.abstractmethod
    def close(self) -> None: ...

    @property
    @abc.abstractmethod
    def is_open(self) -> bool: ...

    @abc.abstractmethod
    def write(self, frames: Sequence[Frame]) -> None:
        """Send frames without waiting for a reply (fire-and-forget / broadcast)."""

    @abc.abstractmethod
    def recv(self, timeout: float) -> Optional[Frame]:
        """Receive the next inbound frame, or ``None`` if none arrived in time.

        This is the primitive for protocols that stream state unsolicited rather
        than answering requests -- e.g. CubeMars servo mode uploads its status
        frame periodically at 1-500 Hz, and nothing ever "replies" to a command.
        ``transact`` cannot express that, so a receive primitive is part of the
        contract rather than an afterthought.
        """

    def transact(
        self, frames: Sequence[Frame], timeout: float
    ) -> List[Optional[Frame]]:
        """Send request frames and collect their replies within ``timeout``.

        Returns one entry per request; ``None`` marks a request that did not
        answer in time. Implementations decide framing (length-prefixed, delimiter,
        protocol-specific) -- the contract does not mandate fixed-length replies.

        Only meaningful for transports whose protocol pairs a reply to a request.
        Broadcast/streaming transports (CAN periodic upload, UDP fan-out) do not
        override this and raise :class:`CapabilityError`; their callers use
        ``write`` + ``recv``. Failing loudly here beats faking a pairing that the
        wire protocol does not have.
        """
        raise CapabilityError(
            "{} has no request/response pairing; use write() + recv()".format(
                type(self).__name__
            )
        )

    def set_failsafe(self, frames: Sequence[Frame]) -> bool:
        """Leave ``frames`` with whatever sits between this host and the device.

        A link with a relay in it (a microcontroller forwarding a WiFi link onto
        the device's UART) can lose the host while the device keeps executing
        the last command -- a wheel servo in speed mode keeps turning. The bus
        knows what "stop" is on its device; the relay knows when the host went
        silent. So the bus hands the relay opaque frames, and the relay sends
        them by itself when the host has been silent for longer than the
        transport's configured time. The relay still interprets nothing.

        Returns
        -------
        bool
            ``False``: nothing on this link can hold them (a cable straight to
            the device), which is the default. ``True``: the relay holds them.
        """
        return False


class Device(abc.ABC):
    """Anything addressable on a transport: an actuator, a sensor, or a custom
    board. Owns its identity, its (optional) calibration, and the declaration of
    the named channels it exposes. This is the extension point that lets custom
    hardware (a barometric pressure sensor, an IMU, a pneumatic board) join the
    stack as a plugin without editing the core.
    """

    def __init__(
        self, device_id: DeviceId, calibration: "Optional[Calibration]" = None
    ) -> None:
        self.id = device_id
        self.calibration = calibration

    @property
    @abc.abstractmethod
    def channels(self) -> Tuple[ChannelSpec, ...]:
        """The named, typed channels this device reads and/or writes."""


class Actuator(Device):
    """A device that owns units, calibration, capabilities, operating modes, and
    frame (de)serialization for a single actuator. Knows nothing about the
    transport medium.

    Concrete models: dynamixel, feetech, kondo_ics, rmd, robstride, odrive...
    """

    #: Whether this model says nothing until a frame is addressed to it.
    #:
    #: False for anything that uploads state on its own -- CubeMars servo and
    #: 3.0 force control both stream at a configured rate, and a bus that only
    #: listens sees them fine. True for request/response models sharing a bus
    #: with those, which is the case that needs stating: a poll-only reader
    #: watches them go stale and reports a fault about hardware that is healthy
    #: and waiting to be asked. See CanBus.solicit.
    answers_only_when_addressed: bool = False

    def __init__(self, actuator_id: DeviceId, calibration: "Calibration") -> None:
        super().__init__(actuator_id, calibration)

    @property
    @abc.abstractmethod
    def capabilities(self) -> Capabilities: ...

    @property
    def channels(self) -> Tuple[ChannelSpec, ...]:
        """Standard motor channels derived from capabilities. Actuators that also
        expose custom channels override this and add their own ChannelSpecs."""
        caps = self.capabilities
        specs = [ChannelSpec("position", "rad", ChannelKind.COMMAND)]
        specs.append(ChannelSpec("position", "rad", ChannelKind.STATE))
        if OperatingMode.VELOCITY in caps.modes or caps.has_velocity_sense:
            specs.append(ChannelSpec("velocity", "rad/s", ChannelKind.STATE))
        if caps.has_current_sense:
            specs.append(ChannelSpec("current", "N*m", ChannelKind.STATE))
        if caps.has_temperature_sense:
            specs.append(ChannelSpec("temperature", "degC", ChannelKind.STATE))
        return tuple(specs)

    @abc.abstractmethod
    def encode_command(self, command: ActuatorCommand) -> Frame:
        """Build the wire frame for a single-actuator command (batch is built by
        the Bus, which may delegate per-actuator encoding here)."""

    def position_refusal(self, position: float) -> Optional[str]:
        """Why this actuator could not be commanded to ``position``, or None.

        Asked BEFORE torque is applied, so an impossible target is refused with
        a reason instead of being discovered mid-cycle -- or, worse, not
        discovered at all. Default: no objection, for actuators that have no
        way to know.

        This exists because "the controller commands where the robot already
        is" is NOT sufficient to make enabling safe. On 2026-07-26 the head
        joint of a Dynamixel arm sat at -6.7058 rad, the controller
        commanded exactly that (mismatch 0.0000, every guard satisfied), and
        enabling torque threw it about 340 degrees into an overload fault: on
        an XL430 that angle converts to raw count 4392, outside the 0..4095 a
        Position-Control goal can express, and the servo silently took it
        modulo its range. The jump guard compares radians and cannot see that;
        only the actuator knows what its wire format can say.

        Implementations should check whatever they know -- taught envelope,
        representable wire range, mode-specific limits -- and return a message
        naming the joint's value and the bound it broke.
        """
        return None

    def fault_name(self, code: int) -> str:
        """Human-readable name for a device-specific fault ``code``.

        Fault-code semantics belong to the actuator model, not a shared bus.
        The default is intentionally honest: this family did not name this code.

        Zero is the one code this layer may name, because it is not a code at
        all -- ``ActuatorState.error`` starts at 0 and ``None`` folds into 0
        upstream, so 0 means "nothing was reported" for every family here.
        Calling that "unknown fault code 0" told an operator watching a healthy
        Kondo arm that its servos had an unrecognised fault, which is both
        untrue and precisely the direction a fault display must never err in.
        """
        code = int(code)
        return ("none" if code == 0
                else "unknown fault code {}".format(code))

    @abc.abstractmethod
    def matches(self, frame: Frame) -> bool:
        """Whether ``frame`` is this actuator's status frame.

        Routing an inbound frame is actuator knowledge, not bus knowledge: a
        device's status frames need not share the address its commands are sent
        to (CubeMars stamps a different packet id into the arbitration id, and
        carries the node id only in its low byte). Asking the actuator keeps the
        Bus free of a per-protocol ``switch``: shared code never branches on
        the protocol.
        """

    @abc.abstractmethod
    def decode_state(self, frame: Frame) -> ActuatorState:
        """Parse a status frame into engineering-unit state."""

    @abc.abstractmethod
    def encode_mode(self, mode: OperatingMode) -> List[Frame]:
        """Frames that switch the actuator into ``mode`` (may be empty)."""

    def encode_release(self) -> Frame:
        """The frame that makes this actuator produce NO torque (servo off).

        Deliberately part of the contract rather than a per-driver extra: "stop
        holding" is the one command a supervisor must be able to issue to
        anything on the bus, and it is what lets an OFF state be expressed
        without any actuator-specific knowledge leaking upward.

        Not every actuator can do it -- some protocols have no release at all --
        so the default refuses. Silently returning "no frame" would be far worse:
        the caller would believe the joint was limp while the driver kept holding
        its last target.
        """
        raise CapabilityError(
            "{} cannot be released; it has no zero-torque command".format(
                type(self).__name__
            )
        )


class Sensor(Device):
    """A read-only device exposing one or more named state channels: IMU, force/
    torque, battery, a custom barometric pressure board, or anything a user adds.

    A concrete sensor is a plugin registered via ``register_sensor`` -- adding one
    requires ZERO edits to actuo.core. Its telemetry flows out either as plain ROS
    topics from the driver (default, non-RT) or, if a controller needs it, as
    ros2_control ``<sensor>``/``<gpio>`` interfaces named by ``ChannelSpec``.
    """

    @abc.abstractmethod
    def encode_query(self) -> List[Frame]:
        """Frames that request a reading. May be empty for streaming sensors."""

    @abc.abstractmethod
    def decode_state(self, frame: Frame) -> Telemetry:
        """Parse a reading into named channel values (keys match ``channels``)."""


@dataclass
class FilterResult:
    """What a :class:`CommandFilter` decided about one cycle's command.

    Parameters
    ----------
    applied : np.ndarray
        The command to actually send. A filter that objects returns something
        the robot can hold, NOT a clamped version of what was asked: silently
        turning a command into a different command is what section 9 bans.
    gated : bool
        Whether ``applied`` differs from what the controller asked for. Drives
        logging and reporting; it is not itself a safety decision.
    reason : str
        Why, in words an operator can act on. Empty when nothing was gated.
    """

    applied: "np.ndarray"
    gated: bool = False
    reason: str = ""


class CommandFilter(abc.ABC):
    """A last look at the command vector before it reaches the wire.

    **actuo supplies the seam and nothing else.** This interface deliberately
    cannot express what is being filtered FOR: it takes a flat command vector
    in the bus's joint order and returns one. Self-collision, workspace bounds,
    a hardware interlock -- all of them fit here, and none of them puts link
    geometry, URDFs or scikit-robot inside the driver layer. Safety and
    self-collision are a separate layer: a driver that knows link geometry has
    mixed two layers that must stay apart.

    So the implementation lives in the layer ABOVE (a safety package, or a
    robot's own package), and the servicer is handed one by name at startup. A robot
    with no filter configured has exactly the code path it had before.

    Timing is the caller's problem to respect, not this interface's to solve:
    the servicer calls this every cycle, inside its control period. A filter
    that cannot answer in that budget will show up as loop-period diagnostics,
    which is why one self-collision monitor grew a fast boolean path (0.42 ms
    against a 10 ms cycle) rather than the 114 ms named check it started with.
    """

    @abc.abstractmethod
    def filter(self, command: "np.ndarray") -> FilterResult:
        """Return the command to send, given the one that was asked for."""

    def reset(self, command: "np.ndarray") -> None:
        """Forget any latched objection, adopting ``command`` as safe.

        Default is a no-op for stateless filters. A latching filter must not
        clear itself: the operator asked for the robot to stop, and deciding on
        its behalf that the danger has passed is exactly the judgement the
        latch exists to withhold.
        """


class Bus(abc.ABC):
    """Groups actuators sharing one transport and drives them as a batch.

    The Bus owns actuator ordering (matching the URDF joint order) and exposes
    flat, ordered command/state vectors. ``write_batch`` MUST build a single
    sync-write transaction where the protocol allows -- not a loop of singles.
    """

    def __init__(self, transport: Transport, actuators: Sequence[Actuator]) -> None:
        self.transport = transport
        self.actuators = list(actuators)

    @property
    def size(self) -> int:
        return len(self.actuators)

    @abc.abstractmethod
    def set_mode(self, mode: OperatingMode) -> None:
        """Switch every actuator on the bus into ``mode`` (capability-checked)."""

    @abc.abstractmethod
    def write_batch(self, commands: Sequence[ActuatorCommand]) -> None:
        """Push one command per actuator in a single bus transaction."""

    @abc.abstractmethod
    def read_batch(self, timeout: float) -> List[ActuatorState]:
        """Read one state per actuator, batched where the protocol allows."""

    @abc.abstractmethod
    def last_state(self, device_id: DeviceId) -> Optional[ActuatorState]:
        """Most recent state of one actuator, WITHOUT touching the bus.

        Required, not optional, because the torque layer above is bus-agnostic
        and cannot do its job without it: deciding whether enabling a joint
        would throw it means comparing the controller's command against the
        last *measurement*, and that decision must not itself provoke a bus
        transaction (it is taken while torque is off, possibly mid-cycle).

        Returns ``None`` for an actuator that has not reported yet -- which the
        caller must treat as "unknown", never as zero.

        This was learned the hard way: three buses grew this method
        independently and the fourth did not, so the first robot to run the
        servo layer over that bus died with an ``AttributeError`` after the
        board was already talking. A shared requirement that lives in three
        implementations and no contract is not a contract.
        """

    def diagnose(self) -> List[DiagItem]:
        """Backend-specific health details.

        Default is empty: a backend that exposes no extra detail is still a
        valid bus. Generic loop-rate/latency/drop metrics are intentionally
        outside this hook; they are orchestration-layer concerns and collecting
        them per backend would duplicate and drift.
        """
        return []


@dataclass
class Calibration:
    """Per-actuator model parameters. No global magic constants. Every actuator carries its own resolution, direction, offset,
    gearing, and limits; unit conversion lives in the actuator model that owns
    this calibration, never in a shared layer."""

    counts_per_rev: float                       # encoder counts per output rev
    zero_offset_counts: float = 0.0             # counts at 0 rad (neutral)
    direction: int = 1                          # +1 or -1
    gear_ratio: float = 1.0                     # output = motor / gear_ratio
    position_limit: Optional[Tuple[float, float]] = None  # (min, max) rad
    torque_constant: Optional[float] = None     # N*m per current unit
    extra: Dict[str, float] = field(default_factory=dict)

    def counts_to_rad(self, counts: float) -> float:
        rad = (counts - self.zero_offset_counts) * (2.0 * np.pi / self.counts_per_rev)
        return self.direction * rad / self.gear_ratio

    def rad_to_counts(self, rad: float) -> float:
        motor_rad = self.direction * rad * self.gear_ratio
        return motor_rad * (self.counts_per_rev / (2.0 * np.pi)) + self.zero_offset_counts
