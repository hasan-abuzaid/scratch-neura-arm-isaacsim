# WAVE block: what changed (23 Nov 2026)

Replace these files in `...\exts\com.example.scratch_blocks\com\example\scratch_blocks\`, then restart Isaac Sim.
`robot_motion.py` must be the version with `wave_joint(..., sharpness=...)` (included here, unchanged).
Commit your current state to git BEFORE replacing anything.

## The block
`WAVE  [joint ▾]  low [55]  high [85]  Hz [1.5]  x [3]`  (in the MOTION library as "Wave (preset)")
- low / high: swing range in degrees. The wave starts at **high**.
- Hz: full swings per second (simulation only).
- x: number of full swings: 0.5, 1, 1.5 ... (whole number = ends at high, .5 = ends at low).

## Per file
- **model.py**: new block type `wave`, `validate_wave_parameters()`, default values, loading from saved projects, joint retargeting (lara8 ↔ MAiRA) also for WAVE.
- **extension.py**: library entry, wider block (520 px), joint dropdown (same as SINGLE JOINT), four number fields (Enter to save, invalid input is restored), Compile checks (joint chosen, joint exists, **peak speed vs. LARA 8 datasheet limit** with the max allowed Hz in the message), cleanup on delete/clear/open.
- **robot_control.py**: `wave_joint()` for the simulation; `WAVE_SHARPNESS = 0.93` (one place to change the turn shape).
- **runner.py**: WAVE in the simulation run and in the real run. (Still has the `[timing]` lines.)
- **neurapy_backend.py**: `wave_single_joint()` = ONE `move_joint` call through high, low, high … with `enable_blending=True`, after `check_joint_feasibility`. Other joints stay where they are.
- **real_robot.py**: `wave` allowed on the real arm; LARA 8 datasheet speed table + `max_joint_speed_deg_s()` (first piece of the "real limits in the sim" to-do); fixed the misleading override comment. **Speed/accel/override values NOT changed** — your uploaded file has `REAL_ACCEL_PERCENT = 10.0`; set it to the 80 you were happy with if that's what you want.

## Sim vs real (important)
- Simulation: continuous curve at the chosen Hz, quick smooth turns, refuses speeds above the joint's datasheet limit.
- Real arm: Hz is ignored (like SET SPEED); speed/acceleration come from `real_robot.py`. **Not lab-tested yet**: first real run slow, area clear, emergency stop in hand. LED green the whole time = blending works.

## Tested here (without Isaac Sim / robot)
All files parse; block create/save/load, validation errors, joint retargeting, datasheet lookup, and the real command (points 85, 55, 85, 55 …, blending on, speed/accel passed) checked with a fake robot. The UI and the real motion still need your test.
