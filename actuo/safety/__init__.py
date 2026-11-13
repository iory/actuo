"""actuo.safety -- self-collision safety layer (hrpsys CollisionDetector analog).

Sits above the driver layer, beside or inside the controllers, and uses
scikit-robot for geometry. ``actuo.core`` never imports this package: link
geometry and collision checking are a separate layer from bus I/O.
"""

from actuo.safety.gate import CollisionGate, GateResult
from actuo.safety.monitor import SelfCollisionMonitor

__all__ = ["SelfCollisionMonitor", "CollisionGate", "GateResult"]
