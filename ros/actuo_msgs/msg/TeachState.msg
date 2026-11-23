# What actuo_teach_node is doing: "idle", "recording", "moving_to_start" or
# "playing".
string state
# Recording being played, empty otherwise.
string name
# [s] since recording or playback started, and the playback length.
float64 elapsed
float64 duration
# Outcome of the last action (an error, or what was saved / played).
string message
