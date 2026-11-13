# Live servo state for ONE joint. Published continuously, never on request.
#
# `commanded` and `measured` are both here on purpose: their difference is the
# jump that enabling torque would cause. A UI can show the danger BEFORE anyone
# presses a button, and a bag records what the robot was actually doing rather
# than only what someone asked for.

string name

string mode               # "position" | "velocity" | "current": WHICH quantity
                          # `commanded` (and `mismatch`) is in, for THIS joint.
                          # Per joint, not per robot: a wheel driven by a
                          # continuous-rotation servo takes a SPEED while the
                          # arm joints on the same board take a pose.
                          # ServoState.mode is the bus default; this is what the
                          # joint actually is.

bool enabled              # true = torque applied. false = limp (zero current).

bool present              # false = the actuator is not answering: unplugged,
                          # unpowered, or not there yet on a hot-plug bus.
                          # `measured` is then the last pose it reported (NaN
                          # if it never did), not where it is.
string absence_reason     # why `present` is false, as the bus knows it; empty
                          # when present or when the bus cannot say.

float64 measured          # rad, from the driver's encoder. NaN when the
                          # actuator has no position sense at all -- a wheel's
                          # reading is a pot that wraps once a revolution, and
                          # publishing 0.0 would read as a real pose.
float64 commanded         # what the controller is currently asking for, in the
                          # units of `mode`: rad in position mode, rad/s in
                          # velocity mode, N*m in current mode. Read `mode`
                          # before comparing this with `measured`.
float64 mismatch          # commanded - measured, in the units of `mode`.
                          # NaN outside POSITION mode, where it would subtract a
                          # speed from an angle. In POSITION mode this is THE
                          # number that matters: enabling torque makes the joint
                          # travel this far.
bool would_jump           # true = enabling is currently REFUSED. Precomputed
                          # here so a client never has to re-derive the safety
                          # rule -- and it differs by mode: POSITION tests
                          # |mismatch| against jump_threshold, VELOCITY/CURRENT
                          # test |commanded| against a magnitude limit, because
                          # outside position mode there is no commanded position
                          # to be far from.

uint8 error               # driver fault code; 0 = healthy
string error_name         # human-readable form of `error`

# --- load and thermals -------------------------------------------------------
# EVERY field below is NaN when the driver cannot report it, never 0. A missing
# temperature published as 0.0 reads as a real (and reassuring) 0 degC, and a
# missing current as a real 0 A. Check with math.isnan(), not `== 0`.
float64 temperature       # degC -- WHICH sensor is actuator-specific. A
                          # CubeMars AK reports its DRIVER BOARD temperature
                          # (protection at 100 degC), NOT its windings, even
                          # though the vendor labels the wire field "Motor
                          # Temperature". Do not compare this across actuator
                          # families as if it were the same measurement.
float64 velocity          # rad/s at the output shaft. NaN unless the actuator
                          # model has what it needs to convert (a CubeMars needs
                          # pole_pairs; without it the raw rate is in aux).
float64 current_a         # AMPS at the motor, as the wire reports it. It needs
                          # no calibration constant, so it is never NaN on a
                          # driver that reports current at all.
float64 effort            # N*m at the OUTPUT shaft. NaN unless the actuator's
                          # torque_constant is known.

# --- whatever else this particular device decoded ----------------------------
# Parallel arrays, by name, so that a channel for the next actuator does not
# widen this message again. The typed fields above are the ones every smart
# servo has; device-specific quantities live here -- e.g. a CubeMars publishes
# `erpm`, its native electrical RPM, which is meaningful even when pole_pairs is
# unknown and `velocity` therefore is not.
string[] aux_names
float64[] aux_values
