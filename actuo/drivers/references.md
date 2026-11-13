# Protocol provenance

Rule: no register address, frame layout, or scaling constant is encoded without
a retrievable source. Each entry below backs constants that appear in
`actuo/drivers/actuators/` or `actuo/drivers/transports/`.

## CubeMars AK series — servo mode CAN

Implemented in `actuo/drivers/actuators/cubemars.py`.

| Fact used in code | Source |
| --- | --- |
| Packet id enum `SET_DUTY=0, SET_CURRENT=1, SET_CURRENT_BRAKE=2, SET_RPM=3, SET_POS=4, SET_ORIGIN_HERE=5, SET_POS_SPD=6` | [1] §5.2 |
| Arbitration id = `controller_id \| (packet << 8)`, extended (29-bit) frame | [1] §5.2, [4] |
| `SET_POS`: int32 `degrees * 10000` | [1] §5.2 `comm_can_set_pos` |
| `SET_POS_SPD`: int32 `degrees * 10000`, int16 `ERPM / 10`, int16 `accel / 10` | [1] §5.2 `comm_can_set_pos_spd` |
| `SET_RPM`: int32 electrical RPM | [1] §5.2 `comm_can_set_rpm` |
| `SET_CURRENT`: int32 `amps * 1000` | [1] §5.2 `comm_can_set_current` |
| `SET_DUTY`: int32 `duty * 100000` | [1] §5.2 `comm_can_set_duty` |
| Status frame (8 bytes): pos int16 `deg*10`, speed int16 `ERPM/10`, current int16 `A*100`, temp int8, error uint8 | [1] §5.2.1 |
| Status position range ±32000 ⇒ ±3200° (hence `multiturn=True`) | [1] §5.2.1 |
| **The status frame's temperature is the DRIVER BOARD's, not the motor's**, despite being labelled "Motor Temperature": *"Motor Temperature: int8, range -20~127 represents the driver board temperature -20°C~127°C."* | [1] §5.2.1, verbatim |
| Driver over-temperature protection at **100 °C** (85 °C on some smaller driver variants); rated ambient −20…65 °C | [1] "Maximum Allowable Temperature of the Driver" |
| The winding temperature exists (hence fault code 1) but is **never uploaded over CAN**. Servo mode defines seven packets (0–6) and all seven command motion — there is no telemetry-request packet | [1] §5.2, confirmed on the real bus: only packet 4 and packet 41 ever appear |
| Both temperatures ARE readable over the motor's 3-pin **UART** port: `02 01 04 40 84 03` returns `MOS temperature` then `Motor temperature` as consecutive `int16/10` (0.1 °C). No CAN forwarding of these is documented | [1] §5.2 UART `COMM_PACKET_ID` |
| ⚠ **Do NOT trust the selective-read bit numbering to pick one temperature.** `02 05 32 <4-byte mask>` selects parameters by bit, and the manual's bit table says bit 1 = MOS, bit 2 = Motor — but its own worked example sets mask `0x00000001` (bit 1) and captions it *"Get motor temperature command"*. The manual contradicts itself, and both sensors return a plausible temperature, so getting it wrong is silent. Request **both** (mask `0x00000003`) or use the get-all command, and identify them by ORDER: MOS first, Motor second, which the get-all layout and both conversion-formula lists state consistently | [1] §5.2, bit table vs its example |
| Error codes 0–7 (`ERROR_NAMES`): 0 none, 1 motor over-temperature, 2 over-current, 3 over-voltage, 4 under-voltage, 5 encoder fault, 6 MOSFET over-temperature, 7 motor stall | [1] §5.2.1 |
| Status is uploaded **periodically at 1–500 Hz**, never as a reply — why the transport contract needs `recv` and refuses `transact` | [1] §5.2.1 |
| `ERPM = rad/s × pole_pairs × gear_ratio × 60 / 2π` | [3] `erpm_conversion` |
| Position command/feedback are **output-shaft** degrees; gear ratio is **not** applied to position | [3] (`position` conversion omits gear_ratio; `enc_off` only), corroborated by [5] |

### The gear-ratio asymmetry

Position is de-geared by the firmware; speed (ERPM) is not. Getting this
backwards would command an AK60-39 to 39× the intended angle, so it is pinned by
`test_position_is_gear_ratio_independent`.

Cross-check, independent of any datasheet: ERPM limits hand-tuned on real
motors [5] were 220/50/35 for 39:1/9:1/6:1 motors at one common output speed.
That is `ERPM ∝ gear_ratio` (50 × 39/9 = 217 ≈ 220; 50 × 6/9 = 33 ≈ 35), which is exactly the formula from
[3]. Pinned by `test_erpm_scales_with_gear_ratio_like_reference_script`.

### Not sourced — deliberately required from config

**`pole_pairs`** is a per-motor datasheet number and is **not** guessed
anywhere. It is only needed to convert rad/s ⇄ ERPM, so:

- POSITION mode never needs it (the `SET_POS_SPD` envelope is carried in the
  wire's native ERPM),
- VELOCITY mode is not advertised as a capability until it is supplied,
- `velocity` in feedback stays `None` (with raw `erpm` in `aux`) rather than
  being reported as a wrong or zero number.

Same for **`kt`** (`torque_constant`, N·m/A) and CURRENT mode.

## CubeMars AK series — 3.0-generation force control (MIT) CAN

Implemented in `actuo/drivers/actuators/cubemars_v3.py`.

| Fact used in code | Source |
| --- | --- |
| Force control is a servo control mode, not a separate firmware: `CAN id = motor_id \| (8 << 8)`, extended frame, DLC 8 | [24] §4.2 |
| *"the upper computer CubeMarsTool has been fully upgraded, and there is no need to switch between servo mode and force control mode"* | [24] p.3, verbatim |
| Byte layout `kp(12) kd(12) pos(16) vel(12) torque(12)`: `b0 = kp>>4`, `b1 = (kp&0xF)<<4 \| kd>>8`, `b2 = kd&0xFF`, `b3 = p>>8`, `b4 = p&0xFF`, `b5 = v>>4`, `b6 = (v&0xF)<<4 \| t>>8`, `b7 = t&0xFF` | [24] §4.2 `pack_cmd` |
| **This is NOT the 2.0 MIT layout**, which leads with position rather than kp | [1] §5.3 vs [24] §4.2 |
| Feedback is the servo status frame unchanged — pos int16 `deg*10`, speed int16 `ERPM/10`, current int16 `A*100`, temp int8, error uint8, timed upload 1–500 Hz | [24] §4.3.1 |
| No enter/exit motor mode: `FF..FC/FD/FE` appear nowhere in [24] | [24], zero occurrences |
| Ranges AK10-9 ±12.56/±28/±54, AK60-6 ±12.56/±60/±12, AK70-9 ±12.56/±30/±32, AK80-9 ±12.56/±65/±18, AKE60-8 ±12.56/±40/±15, AKE80-8 ±12.56/±20/±35 (rad, rad/s, N·m). The table is per **KV variant**, not per model name | [24] §4.2, which prints the KV row |
| kp 0–500, kd 0–5 on every model and both generations | [1] §5.3, [24] §4.2 |
| ⚠ [24]'s own `pack_cmd` scales with `(1 << bits) / span`, which maps the top of the range to `1 << bits` — 4096 in a 12-bit field, one count past what fits. `(1 << bits) - 1` is used instead, as in `cubemars_mit.py` | [24] §4.2, self-inconsistent |
| ⚠ [24]'s `pack_cmd` also clamps the literal `0` rather than `p_des` and `v_des` (`p_des = fminf(fmaxf(P_MIN, 0), P_MAX)`). A literal port always commands position 0 | [24] §4.2, verbatim |

### Which motors are 3.0, measured rather than assumed

Sending packet 8 to all twelve leg motors on 2026-09-12 and watching the
status frame's **current** field — a driver that implements the packet answers
a stalled velocity target with torque whether or not the joint turns:

| model | count | responded |
| --- | --- | --- |
| AK10-9 | 2 | yes |
| AK70-9 | 4 | yes |
| AK80-64 | 6 | **no**, 0.00 A at every step |

Controls, both negative: the same eight bytes as packet 7 did nothing, and
packet 8 with `kd = 0` did nothing. The AK80-64 is a 2.0-generation motor —
CubeMars ships separate MIT and servo firmware for it [26] — and belongs to
`cubemars_mit`.

### The AK80-64 ranges

The AK80-64 is in no published table, in either manual. The values used are
those of the motor's MIT-mode firmware (AK80-64 KV80, version 2024-01-18) [26]:

| Value | |
| --- | --- |
| position | ±12.56 rad |
| speed | ±8.0 rad/s |
| torque | ±144 N·m |
| kp, kd | max 500, max 5 |

`motor_poles` 14, i.e. **7 pole pairs**, comes from the motor parameter file in
the same package (V2.1, 2023-09-03).

Cross-check against motors whose ranges ARE published, read from their own
MIT-mode firmware in the same way: the **AK80-6** gives 12.56 / 76 / 12 against a published 12.5 / 76 / 12; the
**AK10-9 KV100** 12.56 / 50 / 54, whose speed matches [1]'s table and whose
torque matches [24]'s, consistent with the two tables covering different KV
variants; the **AK70-10** 12.56 / 50 / 40 where [1] says 25 N·m, so the
firmware permits more than the table advertises.

The speed and torque ranges agree with physics and with the AK80-80 in [1]'s
table (±8 and ±144, the same motor with a different gearbox): a 64:1 output
shaft cannot turn at 144 rad/s, and an AK80-64 does not stop at 8 N·m.

### The datum, measured

The force-control position field and the status frame share a zero and a
scale. [24] never says so, so it was measured on a biped's AK70-9 hip yaw
(a 3.0-generation motor), 2026-09-12: commanding its own measured position at kp up to 10 moved it 0.10 deg at under 0.2 A, and steps
of ±5/10/15 deg at kp 100 came back with a slope of **0.9830** through the
origin, the 0.18 deg shortfall being 0.31 N·m of friction.

[24] CubeMars, *AK Series Module Product Manual* V3.0.1, 2025-03-14 (formal
version change to 3.0.0 on 2024-12-11), sections 4.2 and 4.3.1.
<https://www.cubemars.com/images/file/20241217/1734428582786471.pdf>

[25] CubeMars technical support and software download index — the two upper
computers (V1.32 for AK 2.0, V3.23 for AK 3.0), the CH340 driver, and the
per-motor firmware packages.
<https://www.cubemars.com/technical-support-and-software-download.html>

[26] CubeMars, *AK80-64 KV80 V2.1 firmware and parameters*.
<https://www.cubemars.com/data/cms/202602/ak80-64-kv80-v2-1-firmware-and-parameters.rar>

## python-can

Implemented in `actuo/drivers/transports/python_can.py`.

- Bus construction and the `interface`/`channel`/`bitrate` parameters: [6]
- Supported interfaces per platform (socketcan is Linux-only; slcan/gs_usb work
  on macOS and Linux): [7]

### ⚠️ python-can's `gs_usb` backend cannot do 1 Mbit/s (measured)

CubeMars runs at 1 Mbit/s and says so ([1] spec table: *"CAN Bus Bit Rate:
1Mbps (Not recommended to change)"*). On the **same adapter** (InnoMaker USB2CAN
V3.3, `1d50:606f`) and the **same physical bus**, measured 2026-07-17:

| Path | tq | BRP | Sample point | Result over 5 s |
| --- | --- | --- | --- | --- |
| Linux kernel `gs_usb` + socketcan | 16 | 3 | **75.0 %** | 751 data frames, **0 errors** |
| macOS python-can `gs_usb` 4.6.1 | 16 | 3 | **87.5 %** | 0 data frames, **1959 errors** (1474 form, 483 stuff), `rx_err` pinned at 255 |

Identical bitrate, identical tq count, identical prescaler — the *only* variable
is where in the bit the controller samples. CiA recommends **75 % at 1 Mbit/s**
(87.5 % applies at ≤500 kbit/s): at 1 Mbit/s propagation delay eats the margin,
so sampling at 87.5 % lands in the next bit's transition and every frame fails
its form/stuff check.

`GsUsbBus.__init__` hardcodes it and exposes no override [8]:

```python
bit_timing = can.BitTiming.from_sample_point(
    f_clock=self.gs_usb.device_capability.fclk_can,
    bitrate=bitrate,
    sample_point=87.5,          # <- not configurable; wrong for 1 Mbit/s
)
```

**Proven by controlled experiment**, same mac / same adapter / same bus / same
code, changing only the sample point: 87.5 % → error storm; 75 % → all three
motors read at 98.50°/76.50°/−87.50°, **identical to what Linux socketcan
reported for the same stationary robot**. Two independent paths agreeing on the
same physical state also confirms the decode.

**Worked around in `PythonCanTransport._corrected_sample_point`.** The fix has
to land *while* python-can constructs the bus: fixing it afterwards is
impossible because `GsUsb.start()` issues a USB `reset()`, which on macOS
re-enumerates the device and invalidates the handle (`[Errno 19] No such
device`) — and a half-applied stop/start leaves the adapter wedged until it is
physically replugged (observed). So the transport temporarily swaps
`BitTiming.from_sample_point` for the duration of that one constructor call and
restores it in a `finally`. Linux/socketcan is untouched: the kernel already
computes 75 % itself.

**The symptom masquerades perfectly as a broken bus**, which is worth
remembering: an error storm whose raw ids (`0x0C`, `0x04`) look like plausible
node ids until decoded as `CAN_ERR_PROT|CAN_ERR_CRTL`. This driver reported
"motors are talking on node id 12" twice before the error frames were being
classified. Hence: `python_can.py` flags `error_frame` in `Frame.meta`,
`CubeMarsActuator.matches` refuses to decode error frames as state, `CanBus`
counts them separately from unmatched *data* (`error_frames` vs
`unmatched_addresses`), and `can_errors.py` names the cause.

Upstream fix worth making: teach python-can's gs_usb backend the same
bitrate-dependent rule the kernel uses (or accept a `timing`/`sample_point`
argument like other backends). Every gs_usb user above 500 kbit/s — CANable,
candleLight, InnoMaker — hits this today.

## Sources


1. CubeMars, *AK Series Module Driver User Manual* V1.0.14, §5.2 "Servo Mode CAN
   Message Protocol" and §5.2.1 "Servo Mode CAN Upload Message Protocol".
   <https://www.cubemars.com/images/file/20240523/1716429938564781.pdf>
2. CubeMars, *AK Series Module Product Manual* v3.0.0.
   <https://img.cubemars.com/products/cubemars-product-parameter/AK-Series-Module-Product-Manual-v3.0.0-Download.pdf>
3. OpenFieldAutomation-OFA/cubemars_hardware — independent ros2_control hardware
   interface for the AK series; source of the `erpm_conversion` formula and
   corroboration that gear ratio is absent from the position path.
   <https://github.com/OpenFieldAutomation-OFA/cubemars_hardware>
4. Announcement thread confirming CubeMars requires extended frame format, with
   mode in bits 28–8 and motor id in bits 7–0.
   <https://discourse.openrobotics.org/t/cubemars-ak-series-ros2-control-hardware-interface/39835>
5. An earlier CubeMars test script whose speed limits were hand-tuned on real
   AK-series motors. Unpublished; no URL.
6. python-can, bus configuration.
   <https://python-can.readthedocs.io/en/stable/configuration.html>
7. python-can, supported interfaces.
   <https://python-can.readthedocs.io/en/stable/interfaces.html>
8. python-can `gs_usb` backend source (v4.6.1), `GsUsbBus.__init__` — the
   hardcoded `sample_point=87.5`.
   <https://github.com/hardbyte/python-can/blob/main/can/interfaces/gs_usb.py>
9. CiA / Linux kernel bit-timing practice: 75 % sample point at 1 Mbit/s,
   87.5 % at lower rates. Kernel calculation in `drivers/net/can/dev/calc_bittiming.c`.
   <https://github.com/torvalds/linux/blob/master/drivers/net/can/dev/calc_bittiming.c>
10. Linux SocketCAN error-frame format (`can_frame` error class + data bytes),
    decoded by `actuo/drivers/can_errors.py`.
    <https://github.com/torvalds/linux/blob/master/include/uapi/linux/can/error.h>

## Feetech SCS0009 — serial (TTL half-duplex) servo

Implemented in `actuo/drivers/actuators/feetech_scs.py` +
`actuo/drivers/transports/serial_transport.py`. Used by the Amazing Hand
(8x SCS0009, 2 per finger; <https://github.com/pollen-robotics/AmazingHand>,
paths below are relative to that repository).

Packet layer is the Feetech / Dynamixel-1.0 form:
`0xFF 0xFF ID LEN INSTR PARAM.. CHECKSUM`, `LEN = nparams + 2`,
`CHECKSUM = (~(ID+LEN+INSTR+sum(params))) & 0xFF`. Instructions
`PING=0x01 READ=0x02 WRITE=0x03 SYNC_WRITE=0x83`; sync-write broadcasts to
`ID=0xFE`. Status reply `0xFF 0xFF ID LEN ERROR DATA.. CHECKSUM`.

| Fact used in code | Source |
| --- | --- |
| Register map (addr, size): `model r@3 u16`, `id rw@5 u8`, `baudrate rw@6 u8`, `min_angle_limit rw@9 i16`, `max_angle_limit rw@11 i16`, `max_torque_limit rw@16 u16`, `torque_enable rw@40 u8`, `goal_position rw@42 i16`, `goal_time rw@44 u16`, `goal_speed rw@46 u16`, `lock rw@48 u8`, `present_position r@56 i16`, `present_speed r@58 u16`, `present_load r@60 u16`, `present_voltage r@62 u8`, `present_temperature r@63 u8`, `status r@65 u8`, `moving r@66 u8` | [11] |
| 2-byte register values are **big-endian** (high byte first on the wire) — SCS series, unlike STS/SMS which are little-endian | [11] (`.to_be()` in every 2-byte conversion) |
| Position: 10-bit raw `0..1023`, centre `511`, full scale `1024` counts = `300 deg`; `angle_rad = radians(300) * ((raw & 0x3ff) - 511) / 1024` | [11] `AnglePosition` |
| Speed: `radians(300)/1024` per count, sign bit at `1<<15`, magnitude `& 0x3ff` | [11] `Velocity` |
| torque_enable: `1 = on, 0 = off` (demo uses 1/2/3 = on/off/free) | [11], AmazingHand `PythonExample/AmazingHand_Demo.py` |
| Servo IDs: right hand `1..8`, left hand `11..18` (+10 offset) | AmazingHand `PythonExample/AmazingHand_Demo.py` `sid()` |
| Bus: serial TTL, **1 Mbit/s**, Waveshare USB driver (`/dev/ttyACM*`, CH340 `1a86:55d3`) | AmazingHand `PythonExample/*.py` (`baudrate=1000000`) |

[11] rustypot (Pollen Robotics), `src/servo/feetech/scs0009.rs` — the library the
     upstream Amazing Hand demo drives the servos with.
     <https://github.com/pollen-robotics/rustypot/blob/main/src/servo/feetech/scs0009.rs>

## Kondo ARM-H7 control board + ICS servos — serial gateway

Implemented in `actuo/drivers/kondo_h7_protocol.py` (packets + board memory
layout), `actuo/drivers/transports/kondo_h7.py` (framing),
`actuo/drivers/actuators/kondo_ics.py` (+ `kondo_ics_temperature.py`, servo
units), and `actuo/drivers/buses/kondo_h7_bus.py` (batch orchestration).

**The board is a gateway, not a servo.** The host speaks this packet protocol to
an ARM-H7 board; the board runs the ICS servo loop on six UARTs (J1..J6). So the
host never emits an ICS frame — it writes pulses into the board's memory-mapped
servo vector, or hands it a multi-servo move command.

Packet form, both directions: `[len] [body ...] [checksum]`, where `len` counts
the whole packet and `checksum = (len + sum(body)) & 0xFF`.
A reply whose checksum does not match is refused (`rcb4.protocol.framing.unframe`
raises `ChecksumError`; the transport counts it in `checksum_errors` and drops
the reply). Replies were measured to carry the checksum on an RCB-4mini
(rcb4's own `rcb4/protocol/references.md`, its entry [22]); **not yet measured on an ARM-H7**.

| Fact used in code | Source |
| --- | --- |
| Opcodes `MULTI_SERVO_SINGLE_VELOCITY=0x10`, `MULTI_SERVO_MULTI_VELOCITIES=0x11`, `MREADV=0xFB`, `MWRITEV=0xFC`, `CALL=0xFA`, `VERSION=0xFD`, `ACK=0xFE` | [12] `rcb4/rcb4interface.py` `CommandTypes`, `rcb4/armh7interface.py` |
| Servo ids are a **5-byte little-endian bitmap** (ids 0..39); pulses follow in ascending id order, little-endian `uint16` | [12] `rcb4/asm.py` `encode_servo_ids_to_5bytes_bin`, `encode_servo_positions_to_bytes` |
| `MREADV` args: `addr u32 LE`, `count u8`, `element_size u8`, `stride u16 LE`; reply capped at 250 bytes | [12] `rcb4/armh7interface.py` `memory_read_aux`, `read_cstruct_slot_vector` |
| Batch state read = one strided `MREADV` over `servo_vector` (count 36, stride 120) | [12] `read_cstruct_slot_vector` |
| Pulse units: neutral **7500**, `4000/135 = 29.6296` pulses per degree, envelope 3500..11500 | [12] `rcb4/rcb4interface.py` `deg_to_servovector`, ICS servo spec |
| Servo ON / OFF are **sentinel pulses 32767 / 32768**, not a distinct packet | [12] `rcb4/rcb4interface.py` `ServoOnOffValues` |
| Move byte is a **duration**: `ms = velocity * FRAME_SCALE`, `FRAME_SCALE = 10`, clamped 1..255 | [13] `Core/Inc/main.h`, cited by [12] `servo_angle_vector` |
| `ServoStruct` layout (size **120**, 36 entries): `flag@0 u8`, `mode@1`, `rotation@2`, `port@3`, `ref_angle@4 u16`, `current_angle@6 u16`, `trim@8 i16`, `stretch@81`, `speed@82`, `current_limit@83`, `temperature_limit@84`, `timeout_count@86 u8`, `rxerror_count@87 u8`, `init_state@96 u8`, `init_count@97 u8`, `error_angle@98 i16`, `feedback@116`, `move_state@117`, `current@118 u8`, `temperature@119 u8` | [23] firmware DWARF (**measured**), agreeing with [12] `rcb4/struct_header.py` |
| `SystemStruct` layout (size **76**): `battery_voltage@28 float`, `servo_current_read_flag@49 u8`, `servo_temperature_read_flag@50 u8`, `ics_comm_stop@64 u8[6]` | [23] firmware DWARF (**measured**), agreeing with [12] |
| `ControlStruct` layout (size **424**): `device_port@356 u8[64]`, `device_timeout_th@420 u8` | [23] firmware DWARF (**measured**) |
| The robot config's per-joint `offset` in `rcb4` is in **degrees** of joint angle, and transfers to actuo's `offset_deg` with no scale conversion | [12] `rcb4_ros_bridge.py:466` feeds `direction * offset` to `trim_vector`; `armh7interface.py:780` does `angle_vector_to_servo_angle_vector(av) - 7500`; `armh7interface.py:1593` builds that transform as `pulse = deg * deg_to_servovector + 7500` |
| Servo current byte: `0..63` one direction, `64..127` the other, 0.1 A per step; `0` = not polled | [12] `read_servo_current`, `rcb4/ics.py` `get_current` |
| Servo temperature is a 128-entry non-linear **table**, not a formula; index 0 = not polled | [12] `rcb4/temperature.py` |
| Current/temperature are only updated while the board's read flags are set | [12] `switch_reading_servo_current`, `switch_reading_servo_temperature` |
| ICS UART map `J1..J6` → `huart2, huart7, huart5, huart4, huart3, huart1` | [12] `ICS_CHANNELS` (from euslisp), [13] |

[12] `rcb4` — the hardware-validated host library this port replaces, at commit
     `dc1ccf559ea90e0334a652711f3d2fec9aaab6a3`.
     <https://github.com/iory/rcb4>
     Every packet layout above is pinned **byte-for-byte** against vectors
     generated from this implementation by `test_kondo_h7_protocol.py`.
[13] `kondoh7` firmware (jsk-ros-pkg/control_boards) — the C source the struct
     declarations and `FRAME_SCALE` come from.
     <https://github.com/jsk-ros-pkg/control_boards/tree/master/projects/kondoh7>
[23] The **firmware binary's own DWARF debug info**, from the v0.6.5 release
     (md5 `2ec2d0cde4c3d5c5a9da2f705d3c6c50`, verified on download).
     <https://github.com/iory/rcb4/releases/download/v0.6.5.elf/v0.6.5.elf>

### The struct layout is MEASURED from the firmware, not inferred from a header

The released ELF carries full DWARF. So every offset above is checkable against
the binary that actually runs, rather than against a C declaration transcribed
into `rcb4`:

    curl -L -o v0.6.5.elf \
      https://github.com/iory/rcb4/releases/download/v0.6.5.elf/v0.6.5.elf
    actuo-kondo-symbols v0.6.5.elf \
      --firmware-version v0.6.5 --verify-layout -o -

Run 2026-07-31: **every declared slot and all three struct sizes match**
(120 / 76 / 424). Rerun this for any new firmware version before trusting the
map — it prints a table and exits non-zero on any mismatch.

The check was itself falsified before being relied on: perturbing one offset,
one array length and one struct size each made it exit non-zero. A layout check
that cannot fail is worse than none.

### `Control.device_port` — the index IS the servo id (MEASURED 2026-07-31)

`Control` (address in the symbol map, size 424) holds a 64-byte `device_port`
table describing what the board found on its ICS UARTs. Per byte, bit 7 =
present, bit 6 = timeout, the low bits are a port bitmask equal to
`2 ** (port - 1)`. That much is [12] `rcb4/armh7interface.py:1433`
`scan_ics_channels_per_port` and is unchanged.

**The index, however, is the servo id — not `index // 2`.** `rcb4` reads it as
`servo_id = index // 2` (`max_SIO_num == 2`, `rcb4/struct_header.py:3`), and one
hardware run refuted that:

| packing | agrees with `ServoStruct.port` |
| --- | --- |
| `servo_id = index` | **13 / 13** present servos |
| `servo_id = index // 2` (rcb4) | 3 / 13, and those by coincidence |

Measured on an ARM-H7 board in a small Kondo humanoid (two arms, four wheels, a
two-servo head; firmware v0.6.5, `/dev/ttyACM0`, 2026-07-31). The raw
table read back was

    88 84 88 84 88 84 88 00 ... 82 81 82 81 ... 00 90 00 90 ...
     ^index 0..6                 ^index 20..23      ^index 33, 35

i.e. exactly the ids `ServoStruct.flag` reported present (0..6, 20..23, and the
head), each at its **own** index. Under `index // 2`, indices 20..23 would name
servos 10 and 11 — which that robot does not have — and indices 0 and 1 would be
one servo on two different UARTs at once.

**This humanoid's wiring is what exposed it, and the reason is worth knowing.** Its arms
alternate across two UARTs: right arm ids 0/2/4/6 on J4, left arm ids 1/3/5 on
J3 (see the id rule below). So adjacent indices hold servos on *different*
ports, and the `// 2` reading folds them together into a self-contradiction —
which is how the driver noticed rather than silently reporting a wrong port. A
robot with all its servos on one UART would have hidden this indefinitely.

**Consequently there is no ids >= 32 blind spot.** The head at ids 32 and 34
appears at indices 32 and 34 and was read there. Under the `// 2` packing such
servos could never appear in this table; that was a consequence of the wrong
packing, not of the firmware.

`ServoStruct.flag` / `.port` nevertheless stays **authoritative**: its offsets
are DWARF-measured, `rcb4`'s own ROS bridge uses it (`search_servo_ids`), and it
is what made the packing checkable in the first place. `device_port` remains the
second source, and `timed_out_ids` is the one fact it contributes that the
authoritative source does not carry at all. `KondoDiscovery` keeps them apart:
`per_port` / `timed_out_ids` are `Optional` and become `None` with the reason in
`ics_channel_scan_error`, so an unavailable scan never reads as an empty one, and
a `device_port` problem can never suppress the verified scan.

**Rerun the cross-check on any new board.** `actuo-kondo-probe --scan` prints
`ServoStruct.port` against the `device_port` bitmask; one board is not every
board, and the packing is now a measurement of exactly one firmware build.

### The ICS servo id is wiring, not a dial (same humanoid, v0.6.5)

The six ICS UARTs split into two groups of three, and a servo's id is decided by
**which group its port is in and where it sits in that port's chain**: one group
yields `2 * n`, the other `2 * n + 1`.

Measured on the same run — even group **J2 / J4 / J6**, odd group **J1 / J3 /
J5**:

| port | ids found | what it is |
| --- | --- | --- |
| J4 | 0, 2, 4, 6 | right arm, 4 joints |
| J3 | 1, 3, 5 | left arm, 3 joints |
| J2 | 20, 22 | wheels |
| J1 | 21, 23 | wheels |
| J6 | 32, 34 | head |

Confirmed by moving one connector: with the head on **J5**
the same two servos enumerated as **33 / 35**; moved to **J6** they became
**32 / 34**, matching the robot's existing servo map with nothing else changed.

So "servo 32 is missing" can mean the connector is in the wrong group, and
re-teaching ids or editing the joint map would be the wrong repair. It also
means a robot config's ids are only valid for a particular cabling.

### Firmware addresses are data, not a run-time lookup

`servo_vector` and `SysB` are firmware globals; their addresses change with
every build. `rcb4` resolves them at connect time by opening the firmware ELF
with pyelftools, which lock-steps the host library to the firmware ELF's symbol
layout: every host needs the exact ELF of the board it talks to. Here they are
extracted **once, offline** (`actuo-kondo-symbols`), committed as YAML next to
the robot config,
and checked against the board's reported version string at connect time
(`KondoH7Bus.verify_firmware`). A mismatch refuses to run, because reading a
struct at the wrong address returns plausible-looking numbers.

That refusal has to come **first**, and both entry points now do. A version
check placed after the struct reads produces the worst possible output: the scan
reads the wrong memory, finds nothing, and reports "these servos were not
recognised" — sending the operator to check id dials and wiring that are fine.
`verify_servos()` calls `verify_firmware()` before interpreting any struct, and
`diagnose()` returns the firmware item **alone**, emitting nothing that depends
on a mapped address. There is no such thing as a partially trustworthy read from
the wrong map.

### Not sourced — deliberately absent

- **No velocity, current, or PWM mode.** The ICS wire carries exactly one
  control quantity: a target pulse. `capabilities.modes` is `(POSITION,)` and
  requesting anything else raises rather than being emulated.
- **No sensed velocity.** `ActuatorState.velocity` stays `None`; it is never
  differentiated from position behind the caller's back.
- **No torque without `torque_constant`.** Amps are reported in
  `aux['current_a']`; `state.current` stays `None` until a config supplies N·m/A.

## Kondo RCB-4mini control board + ICS servos — serial gateway

Implemented in `rcb4.protocol.rcb4mini` (packets + RAM layout; a copy of this
section sits beside it in rcb4 as `rcb4/protocol/references.md`),
`actuo/drivers/transports/kondo_rcb4.py` (even parity, reopen at another rate)
and `actuo/drivers/buses/kondo_rcb4_bus.py` (batch orchestration, cold start).
The servos are the same ICS servos as on the ARM-H7 and use the same actuator
model, `actuators/kondo_ics.py`.

A different board from the ARM-H7: the H7 firmware grew out of the RCB-4, so
the framing, ack, version and batch move are shared, and the memory is not.

| Fact used in code | Source |
| --- | --- |
| Framing, `ACK=0xFE`, `VERSION=0xFD`, batch move `0x10`/`0x11` are byte-for-byte the H7's | [17], pinned by rcb4 `tests/rcb4_tests/test_protocol_rcb4mini.py` |
| **Never `0x11`** (a move byte per servo). The bus sends only `0x10` and refuses servos with different move bytes | **measured** [19]: a `0x11` packet with a real target (servo 5 at its own position, every move byte 1) was answered with a lone `00` and the board then answered nothing until power-cycled; streaming `0x11` (servo 5 at move byte 2, the rest 1) reset the board (it answered again at 115200). The same targets as `0x10` never failed, incl. a new packet every 20 ms during a 2-frame interpolation with the bus's reads (300 packets). The byte order sent was kxreus's `:m-s-mv` [20] `rcb4asm.l`; rcb4 [17] defines the opcode and never sends it |
| From the OFF reference (`0x8000`) the board does not interpolate toward a target; servo ON (`0x7FFF`, which copies the measured pulse into the reference) must come first, so `write_batch` sends it in its own packet for a servo going from released to commanded (the servo supervisor enables torque by sending targets, not `hold_all`) | **measured** [19]: the record's interpolation width read `target - 0x8000` and the reference stayed `0x8000` |
| One move-byte unit is one frame; with config frame bits 4-5 = `01` that is 15 ms | **measured** [19], servo 5, +10 deg: move byte 20 ramped linearly for ~300 ms, 127 reached 90 % at ~1.8 s; frame-field meaning from [20] `:sysreg` (`00` 10 ms, `01` 15 ms, `10` 20 ms, `11` 25 ms) |
| Move byte 1 streamed at 50 Hz tracks a +-10 deg, 0.5 Hz sine with a 42 ms lag and 0.20 deg rms error after the lag; the measured path is within 0.25 deg rms (0.63 max) of a smooth sine | **measured** [19], 200 cycles, 19.8..20.2 ms period, no missed read, no refusal |
| RAM read: `00 20 00 00 00 addr_lo addr_hi size`; reply `[00, data...]` | [17] `move_ram_to_com_command`; reply **recorded** [19] |
| RAM write: `00 02 addr_lo addr_hi 00 data...`; reply `[00, 06]` | [17] `move_com_to_ram_command`; reply **recorded** [19] |
| A RAM read can carry up to 252 bytes (the length-byte limit), but `plan_record_reads` plans reads of at most **126** (`READ_CHUNK_BYTES`) | **measured** [19]: `$01FC`+242 repeated alone was refused exactly every second time (30 of 60), and `$00BC`+162 alternating with it had the 242 refused every time; every board hang whose last request was logged ended on a read of 162 bytes or more, after which the board answered nothing at any rate until power-cycled. Reads of up to 126 (180 in a row, incl. the same span in 126-byte pieces) drew no refusal and no hang. [17] `_angle_vector` and [20] `:read-servos-three` read 126 at a time |
| Config bit 2 makes the board send a command out of COM when a servo's interpolation ends; bits 4-5 are the output cycle (`00` 10 ms, `01` 15 ms, `10` 20 ms, `11` 25 ms). The bus clears bit 2 and sets the cycle to `01` at connect (`flat_config`), so a project's settings do not change the protocol or the move-byte unit | [21] firmware reference p.4-5 (システムレジスタ, system registers); `01` is what a board without a project powered up with (`0x8313`), **measured** [19]; a second RCB-4mini, in a 35-servo humanoid with a project in ROM, read `0xc291` (bit 2 clear, cycle `01`) on 2026-10-20 |
| Record `+8..+12` is the interpolation state (speed, end position, steps, width); with `+8` (speed) non-zero the board rewrites `+6` every cycle, so the bus writes `+8` = 0 with every OFF it writes into a record | [21] firmware reference p.16-17 (補間動作, interpolation) |
| Record `+14`/`+17` (uint16) name each of two mixings' source data (RAM address, or ICS number + offset), `+16`/`+19` its ratio byte (sign, factor); the board drives `+6` + mixing 1 + mixing 2 every cycle, each source x ratio. The bus sets the ratio of any mixing with a source to 0 at connect and switches its servo off (`_flatten_records`) | [21] firmware reference p.17-18 (ミキシング, mixing); offsets also [21] programming manual (`Mixing1AddressOffset` 14 ... `Mixing2RatioAddressOffset` 19) and [20] `:servo-state`. Source `$FFFF` taken as "none": every one of the 35 records on that humanoid's board read `$FFFF`, ratio `01` (**recorded** 2026-10-20), with servos driven exactly to their reference; as a source it is no RAM address (RAM ends at `$048F`) and no ICS number (bits 8-13 = 63), **inferred** from [21] |
| Config bit 1 runs the ROM program, bit 3 enables vector jumps; the bus clears both at connect | bit names: [17] `ConfigData`, [20] `*rcb4-sysreg-bits*`; [20] `:send-servo-vector-com` clears ROM-run around every servo command, [17] `suspend_motion` clears both. A board without a project powered up with ROM-run set (`0x8313`), once with vector jumps too (`0x831b`), **measured** [19]. Not shown to cause the refusals: they persisted with both cleared |
| The board sometimes **refuses** a request with `[x, 0x15]` (x seen: 0x00, 0x01, 0x02, 0x0b), and answers the same request moments later | **measured** [19]; [18] retries reads for the same reason |
| Firmware reply to `VERSION`: `fd "CB-4 V1.0      090715          " c7` | [17] `check_firmware_version`; **recorded** [19] |
| COM port: **even parity** | [17] `RCB4Interface.open` |
| Config word at `$0000`; COM rate in bits 6-7: `0` = 115200, `2` = 1.25 Mbit/s; the board acks a config write at the old rate, then switches | **measured** [19] (switched to 2, answered at 1.25 M only) |
| ICS rate in config bits 13-14: KRS servos set to 1.25 Mbit/s answer with `2`, not `0` or `1` (`3` also answered, meaning unknown, unused) | **measured** [19] |
| ICS rate code `0` is 115200 bit/s on the ICS line | **measured** 2026-10-16 with an oscilloscope on the ICS line of the same rig as [19] (unpublished measurement, no URL); KRS servos set to 115200 (IDs 0, 2, 8) answered with it here, through rcb4 `assign_ics_servos` (config `0x8391`) |
| ICS assignment table at `$0044`, one `uint16` per ICS number, `$FFFF` = unassigned, else the record address; loaded from the ROM project at power-up. A project may point a number at any 20 bytes of user RAM; the bus maps every such number back to its own record (`$0090` + 20 x number), servo off | [18] `servo_bus_rcb4.cpp` (cites the RCB-4 firmware reference); [21] firmware reference p.15 (records placed anywhere in user RAM `$0090..$048F`); **recorded** [19] |
| Servo records at `$0090`, 20 bytes each; `+4` measured pulse, `+6` reference pulse (`0x8000` = off) | [17] `_angle_vector`, [18]; **recorded** [19] |
| The board sends a record's reference to the servo the moment its number is assigned, so it is set to off first | [18] `try_assign` |
| A record keeps its last pulse after its servo goes quiet, so it is cleared before assigning | **measured** [19]: a stale pulse read as a servo until cleared |
| A board without a project powers up at COM 115200 with every ICS number unassigned (config `0x8313`) | **measured** [19] |
| A record's `+1` byte is the servo's ICS ID, and ICS number = 2 x ID + port (port 0 = SIO1-4, 1 = SIO5-8) | **recorded** [19]: ICS 2, 5, 8, 10, 18, 30 held 1, 2, 4, 5, 9, 15; port numbering from [21] `Rcb4.xml` `CommunicateDevice` |
| A MOV with an ICS source (`00 21 00 00 00 offset ics size`) reads the board's RAM record through the assignment table, not the servo | **measured** [19]: ICS 5, offset 0, 20 bytes returned the 20 bytes of record `$00F4` exactly; encoding from [20] `rcb4asm.l` `rcb4-src-bytes` and [21] command reference p.3 (MOV) |
| `0x12` sets a servo parameter of the ICS numbers whose bit is set in a 5-byte mask (bit n = ICS number n), then a sub-command (`01` stretch 1..127, `02` speed) and one value per set bit in ICS order; answered `SIZE CMD 06 SUM`. The reference's table prints the command code as `11h`, the batch move it copies its layout from. The bus sets a configured `stretch` with it at connect and when a servo answers again (`build_stretch`) | [21] command reference p.20-21 (サーボのストレッチ・スピードの変更, changing servo stretch and speed); code `0x12` and the bit order from [17] `rcb4` `set_servo_parameters_command` (the vectors in rcb4 `tests/rcb4_tests/test_protocol_rcb4mini.py`, generated at rcb4 `e5cd18d`). rcb4's ROS bridge sends it only to an ARM-H7 ([17] `ros/kxr_controller/node_scripts/rcb4_ros_bridge.py`, `if not self.use_rcb4`). **Not yet measured** on an RCB-4mini |
| **Never `0x0E`** (ICS relay: `[09, 0E, port, count, tx_lo, tx_hi, rx_lo, rx_hi, sum]`, which the reference says sends RAM bytes to the ICS bus and stores the reply in RAM). Nothing in actuo sends it | format from [21] command reference p.14 (ICS通信, ICS communication); **measured** [19]: relaying `a2 04` (temperature read, ICS 5 = ID 2) on port 1 drew no reply in 1 s and wrote nothing to the reply area; the same on port 0 left the board answering nothing at any rate until power-cycled |
| The user-variable area (`$0462`) is not cleared by a power cycle | **measured** [19]: bytes written there were still present after the board was power-cycled |
| The board rewrites the `+4` measured pulse of every assigned number whose servo answers, once per frame, limp or held; one with no servo stays at whatever was last written. Zeroing `+4` and reading it back is the presence test (`hot_plug`) | **measured** [19]: ICS 18 unplugged read 0 within 50 ms of a clear and came back after replugging without reassigning; assigned unwired ICS 0, 1, 3 stayed 0 |
| A record's `+2` (int16) is the board's per-servo **trim**: the board reports `servo pulse - trim` at `+4` and drives `reference + trim`. A project in the board's ROM sets it at power-up, so the bus switches any servo with a trim off and zeroes the trim at every connect (`_flatten_records`) | [21] firmware reference p.15-16 (`基準値`, reference value: drive = mixing 1 + mixing 2 + `+6` + `+2`; `+4` = servo − `+2`). **measured** 2026-10-20 on the humanoid's RCB-4mini (project in ROM): ICS 5 and 7 held trims -2700 and +2000 (the bus's first connect after that cleared trims on ICS 4, 6, 8, 9 and 12-17 as well); with both servos released, writing 0 to `+2` changed their `+4` by exactly -2700 and +2000 (7497 -> 4797, 7500 -> 9500). rcb4 [17] `_angle_vector` reads the same word as its "error" slot |
| Writing a record's `+6` reference in RAM drives the servo to it, as a batch move does | **measured** 2026-10-20: a position servo (ICS 21..23, before they were set to rotation mode) went to 7500 when its `+6` was written to 7500. `write_batch` turns a wheel on this way (`_stop_wheel`), because `SERVO_ON_PULSE` would copy the pot into a wheel's speed |
| A servo's port is found by assigning both 2 x ID and 2 x ID + 1 and seeing which `+4` the board rewrites (`port_detect`), never by the `0x0E` relay | follows from the two rows above; **not yet measured** with a servo moved between SIO1-4 and SIO5-8, and whether polling the extra, empty numbers lengthens the board's frame is not measured either |

[17] `rcb4` — the host library, at commit
     `dfa222528e0f21a9ecd1495b7398ae7eb12e25bc`, which added the ICS assignment
     and the 1.25 M switch to `RCB4Interface.open`.
     <https://github.com/iory/rcb4>
[18] `build-your-own-arduino-rtos`, `code/13_quadruped/src/servo_bus_rcb4.cpp`
     — an RCB-4mini driven from an Arduino, which assigns ICS numbers the same
     way. <https://github.com/iory/build-your-own-arduino-rtos>
[19] Measured on an RCB-4mini (no HeartToHeart project, six KRS servos at ICS
     numbers 2, 5, 8, 10, 18, 30), 2026-10-12/13, through a Kondo Dual USB
     Adapter HS (165c:0008).
[20] `kxreus` — EusLisp interface to RCB-4 boards, at commit
     `76bb12718f09bad3c309b62993059a747130588b`: `rcb4asm.l`
     `*rcb4-sysreg-bits*` (config bit names), `rcb4interface.l`
     `:send-servo-vector-com` (ROM-run off around servo commands) and
     `:read-servos-three` (126-byte record reads).
     <https://github.com/inabajsk/kxreus>
[21] Kondo Kagaku, RCB-4 reference set Ver.2.0 (`RCB4RefSetV200R20120420.zip`):
     `RCB-4コマンドリファレンス20100805.pdf` (command reference; p.3 MOV,
     p.14 ICS通信 = ICS communication),
     `RCB-4ファームウェアリファレンス20100805.pdf` (firmware reference) and
     `RCB-4プログラミングマニュアル 20120420.pdf` (programming manual; RAM maps), and the `Rcb4.dll`
     documentation `Rcb4.xml`.
     <https://kondo-robot.com/faq/rcb-4-reference-ver-2_0>,
     <https://kondo-robot.com/w/wp-content/uploads/RCB4RefSetV200R20120420.zip>

### Not sourced — deliberately absent

- **625000 bit/s** on COM, and every ICS rate but 115200 and 1.25 M: their
  config values were not tried, so asking for one raises.
- **Why large reads are refused or hang the board.** Not the address (a
  16-byte scan of `$0090..$0400` and `$01FC`+242 alone both answered), not
  the request spacing (2-byte reads at 0.5 ms gaps never refused), not the
  ROM program (cleared, refusals persisted). Unknown; the bus avoids it.
- **What `x` in a refusal means**, and whether a refused move was applied. The
  bus only resends, which is safe either way because every request is
  idempotent.
- **`FRAME_SCALE_MS`.** One unit is one frame (15 ms with the frame field this
  board powered up with, measured above), but the frame length is a config
  field a project can change and only `01` was measured, so `move_time_ms`
  is still refused rather than converted with a constant that may not hold.
- **Servo EEPROM, rotation mode, current and temperature.** The ICS-source
  MOV reads only RAM and the `0x0E` relay does not work on this firmware
  (both measured above), so none of them is reachable through the board; the
  bus reports position only. They can be read with the servo on the adapter
  in ICS mode ([17] `rcb4/ics.py`).

## DYNAMIXEL (protocol 1.0 and 2.0)

The host speaks these protocols directly -- there is no gateway board in this
path. A **DXHUB (BTE068C, Best Technology)** is an FT232H USB-serial converter
plus isolated TTL and RS-485 drivers and a power distribution header; it
contains no MCU, so it is an ordinary serial port to this stack and appears
only in the transport's `channel`.
<https://www.besttechnology.co.jp/modules/onlineshop/index.php?fct=photo&p=255>

[14] DYNAMIXEL protocol 1.0 — packet layout (`FF FF ID LEN INSTR .. CHK`),
     instruction set, and the inverted-sum checksum.
     <https://emanual.robotis.com/docs/en/dxl/protocol1/>
[15] DYNAMIXEL protocol 2.0 — packet layout (`FF FF FD 00 ID LEN_L LEN_H ..`),
     byte stuffing, instruction set including SYNC_READ/SYNC_WRITE, and the
     status-packet error field. **The example packets printed on this page are
     frozen verbatim as the fixtures in `tests/test_dynamixel_protocol.py`** --
     an encoder checked only against a spec this repo also wrote proves nothing,
     so the vendor's own bytes are the test.
     <https://emanual.robotis.com/docs/en/dxl/protocol2/>
[16] CRC-16 for protocol 2.0 — polynomial `0x8005`, init 0, computed over every
     byte of the packet except the CRC itself.
     <https://emanual.robotis.com/docs/en/dxl/crc/>

Per-model control tables, resolutions and units are **not** in this file or in
any Python source. They live in `actuo/drivers/data/dynamixel_models.yaml`, one
row per model, each carrying its own e-Manual URL — so adding a servo model is
data with a citation, never code. Currently transcribed:

- **AX-12A**, model number 12, protocol 1.0 only.
  <https://emanual.robotis.com/docs/en/dxl/ax/ax-12a/>
- **XL430-W250-T**, model number 1060, protocol 2.0.
  <https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/>

### Not sourced — deliberately absent

- **No PWM mode.** The `goal_pwm` register exists, but the full-scale value it
  is expressed in was not transcribed, so a duty cycle cannot be scaled to
  counts. PWM is withheld from `capabilities` and requesting it names the
  missing field, rather than being emulated with a guessed scale.
- **No current/torque on either shipped model.** The AX has no current sensing
  at all; the XL430 has neither current control nor current sensing. Neither has
  a published `torque_constant`, so `state.current` stays `None` and CURRENT is
  absent from `capabilities` — the same rule CubeMars applies to `kt`.
- **No temperature on the XL430.** The register exists at 146, but reaching it
  in the cyclic block would mean reading through addresses 136..143, which were
  not transcribed. `has_temperature_sense` is false because the driver genuinely
  does not deliver it; extend `present_block` once those addresses are sourced.
- **No wheel mode on the AX.** It is entered by writing
  `cw_angle_limit = ccw_angle_limit = 0` — a mode switch disguised as a limit
  write, which also silently changes what `goal_position` means. Unimplemented,
  so VELOCITY raises rather than moving the joint unexpectedly.
- **No batch read on protocol 1.0.** There is no SYNC_READ instruction, so N
  joints cost N round trips. `build_sync_read` raises on 1.0 rather than
  emulating a batch with a loop of singles, and `DynamixelBus.reads_per_cycle`
  reports the real number.

### Two traps that produce plausible wrong numbers

- **Travel is not a revolution.** An AX-12A spans 1024 counts over 300°, so its
  counts-per-*revolution* is 1228.8. Using 1024 scales every angle by 0.83 and
  still looks like a robot pose. Pinned by `test_travel_is_not_a_revolution`.
- **Protocol 1.0 velocity and load are not two's complement.** Bit 10 is a
  direction flag and bits 9..0 are magnitude. Decoding as signed puts the sign
  change in the wrong place. Pinned by
  `test_protocol1_direction_bit_is_not_twos_complement`.

### FTDI latency timer

An FT232H defaults to a 16 ms latency timer, which caps a request/response bus
far below its baud rate regardless of how fast the servos are. DynamixelSDK
works around this internally; this stack opens the port with pyserial and does
not, so the floor must be **measured** — `actuo-dxl-probe` prints min/max round
trip and calls out a >10 ms floor explicitly: bus latency is measured and
exposed, never assumed.

### KJS-03 / AIRRELAY sensor board

A custom board that impersonates a DYNAMIXEL device on the servo bus: IMU, four
proximity sensors, five ADC channels, a magnetic encoder, and on the AIRRELAY
variant two solenoid valves for a vacuum gripper. It replaces the rcb4-era "air
board", which was reached through `ri.control_air_board(38, 'open_relay_valve')`
— an ad-hoc method on a god-object. Here it is `actuo/drivers/sensors/kjs03.py`,
registered via `register_sensor`, publishing named `ChannelSpec` channels
(telemetry is named channels, never a widened shared struct) with **zero
`actuo.core` edits**.

- Control table: `KJS-03/docs/CONTROL_TABLE.md` (firmware v2), generated from
  the firmware's own `include/control_table.h`.
  <https://github.com/iory/KJS-03>

Two identification rules, both load-bearing:

- **Model number 0x0A03 alone is not enough.** Model numbers are assigned by
  ROBOTIS and this board is not a ROBOTIS product, so the number could one day
  collide with a real product. `Model Information` (addresses 2..5) carries the
  magic `0x3053414B`, and the plugin refuses to decode unless BOTH match — a
  collision would otherwise mean reading every address against a different
  device's control table, with every value coming back plausible.
- **The IMU full-scale range is read from the board, never assumed.** The board
  ships with one of three ICM parts whose ranges differ by up to 8x. Until
  `decode_identity()` has read `Gyro Range DPS` / `Accel Range G`, the plugin
  reports raw counts only and omits the rad/s and m/s² channels entirely.

### Not sourced — deliberately absent

- **No fake joint.** The board has no shaft, so it is a `Sensor`, not an
  `Actuator`. Registering it as an actuator would put a joint in the URDF that
  cannot be commanded and that reports an invented angle.
- **Pressure is converted on the board**, not here: the conversion needs the
  amplifier's `V_OFFSET` and `GAIN`, which are properties of that PCB. A host
  that hardcodes them is holding a copy of the schematic, and it goes stale
  silently the day the board is revised.

## RCB-4mini over WiFi — Arduino UNO R4 WiFi as a UART relay (2026-11-07)

Code: `transports/kondo_rcb4_udp.py` (host), `uart_relay_protocol.py` (datagrams),
an Arduino sketch (the relay, `firmware/actuo_uart_relay/` in this repository), `uart_relay.py` +
`actuo-uart-relay-emulator` (the relay's logic on the host, for tests).

**Decision: the UNO R4 WiFi is a protocol-agnostic UDP <-> UART relay, and the
host keeps the whole RCB-4mini stack** (`kondo_rcb4_bus`, ros2_control). Considered:

| Option | Why not / why |
| --- | --- |
| micro-ROS on the UNO R4 WiFi, the board's protocol on the Arduino | The RCB-4mini protocol would be written a second time, in C++ on the MCU, beside actuo's -- two sources of truth for one protocol, which drift apart. ros2_control and the servo safety (jump guard, silence watchdog) would not be in the loop, so the wireless robot would not be the wired robot with only the link changed. micro-ROS does list the board (community-contributed, USB and WiFi transports, release with #1863 [28][29]), but on 32 kB of SRAM [27] a robot-sized message set is the "painful" case its own users describe [30]. |
| **Relay (chosen)** | Transport only: the relay knows that the first byte is a packet's length and nothing else (transport and protocol semantics stay separate layers). Same bus, same config, same controllers: only the transport plugin changes. Fits easily: 61.6 kB flash (23 %), 10.3 kB RAM (31 %), measured by `arduino-cli compile --fqbn arduino:renesas_uno:unor4wifi` with renesas_uno 1.5.3. |
| ESP32-S3 module alone | Would need to reflash the UNO R4's bridge firmware and give up the stock Arduino API on the RA4M1. |

Facts the design rests on:
- RA4M1: 32 kB SRAM, 256 kB flash [27].
- `Serial1` is the RA4M1's hardware UART on D0 (RX) / D1 (TX), separate from
  the USB `Serial` [31]. Even parity (`SERIAL_8E1`, which the RCB-4mini needs) is
  handled by the installed core: `cores/arduino/Serial.cpp` line 249 maps it to
  `UART_PARITY_EVEN` (renesas_uno 1.5.3, read from the installed core).
- `WiFiS3` provides `WiFiUDP` [32].

**Network failure handling** (each tested in `tests/test_uart_relay.py`):
- Lost datagram: the request yields `None`, a miss like a silent board; the
  bus node's silence watchdog (`--max-silence`) and link recovery take over.
- Late answer: every datagram carries a sequence number; an answer to a request
  the host gave up on is dropped (`stale_replies`), never taken for the next.
- Host gone: the bus hands the relay a failsafe packet (wheels to the stop
  pulse) through the new `Transport.set_failsafe` contract method; the relay
  sends it after `failsafe_silence` (default 0.5 s; 0.3 s tripped on the host bus loop's own 0.3-0.35 s pauses) without a datagram, once per
  silence, counts it, and reports the count in every answer (logged by the
  host). Arm joints are deliberately NOT in it: released, an arm falls.

NOT verified on hardware (no board was available): whether the RA4M1's SCI
reaches 1.25 Mbit/s 8E1 (the WiFi config runs the board at 115200), the
RCB-4mini COM port's voltage levels against the UNO R4's 5 V I/O, round-trip
time over WiFi, and the relay's behaviour on the board
itself.

[27] https://docs.arduino.cc/hardware/uno-r4-wifi/
[28] https://github.com/micro-ROS/micro_ros_arduino (README, "Community contributed boards")
[29] https://github.com/micro-ROS/micro_ros_arduino/releases ("add Arduino UNO R4 WiFi (USB and WiFi)", #1863)
[30] https://github.com/Pana1v/ultimate-robotics-handbook/blob/main/ros-2/micro-ros.md ("Practical RAM minimum")
[31] https://docs.arduino.cc/tutorials/uno-r4-wifi/cheat-sheet
[32] https://docs.arduino.cc/tutorials/uno-r4-wifi/wifi-examples
