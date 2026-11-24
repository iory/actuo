# Which configuration of a modular robot is running (actuo.modular), published
# latched by the bus servicer whenever it changes.

builtin_interfaces/Time stamp
string name                # the configuration running now
string select              # "detect" (the servos that answer decide) | "declare"
string[] problems          # why detection does not settle, or a declaration's missing servos
int32[] answering_ids      # every servo id that answers, in this configuration or not
string[] held_joints       # joints that kept their torque across the last switch
float64 switch_seconds     # duration of the last switch; NaN before the first
