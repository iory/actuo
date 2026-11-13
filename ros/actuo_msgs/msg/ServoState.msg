# Whole-robot servo state, published continuously with transient_local durability
# so a late subscriber (an RViz panel, a bag replay) immediately learns the state
# instead of waiting for a change.
#
# This topic is the audit trail. An action/service records what was REQUESTED --
# including requests that were refused. Only this says what the robot WAS.

builtin_interfaces/Time stamp

bool all_enabled          # true only if every joint has torque
bool any_would_jump       # true if enabling is currently refused for any joint
float64 jump_threshold    # rad; the limit `would_jump` was tested against
                          # (POSITION mode). In VELOCITY/CURRENT mode the limit
                          # is a magnitude on the commanded value instead -- see
                          # `mode`, and ServoSupervisor.would_jump.
string mode               # "position" | "velocity" | "current": WHICH quantity
                          # the controller is commanding, and therefore the units
                          # of every joint's `commanded` and `mismatch` below.
                          # Published because those two fields are documented in
                          # rad and are NOT rad outside position mode; a consumer
                          # must never have to guess which quantity it is reading.

# --- bus health, for a run nobody is watching --------------------------------
# Counters the servicer keeps, published so that a bag of an unattended run can
# answer "when did the errors start" afterwards.
#
# error_frames and unmatched_frames are kept apart on purpose because they
# demand opposite responses: unmatched data means the config is wrong, error
# frames mean the wiring or the bus is.
uint64 cycles             # control cycles since the servicer started
uint64 error_frames       # CAN error frames seen. Should stay 0. A count that
                          # only climbs under load is the early warning.
uint64 unmatched_frames   # frames from an id no configured actuator claims

JointServoState[] joints
