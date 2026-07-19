# kinova-tele

Quest passthrough teleoperation for a full Kinova Gen3 arm using the Kortex
Python API. The server combines XR or keyboard translation, trigger-clutched
yaw, orientation hold, gripper commands, a centered planar workspace,
auto-home, a motion watchdog, and Cartesian contact compliance.

## Cartesian impedance behavior

`impedance_tuning/impedance_controller.py` implements a translational virtual
mass-spring-damper:

```text
M * x_ddot + D * x_dot + K * x = F_external
```

Kortex `BaseCyclic.RefreshFeedback()` supplies the computed external tool
wrench. The controller tares the resting force, filters it, applies a deadband,
and generates a limited compliance velocity. `main.py` adds that velocity to
the normal XR or keyboard command before the existing workspace limiter and
the final Kortex velocity caps.

This is an admittance outer loop that provides Cartesian impedance behavior
through high-level twist commands. It is not a 1 kHz joint-torque impedance
controller. A true joint-torque controller belongs in Kortex low-level UDP
control and should not run from this Python web server.

Only translation is compliant. Startup pitch and roll remain locked, and yaw
remains controlled by the right thumbstick. This avoids contact torque changing
the orientation constraints already used by the teleoperation system.

## Integrated safety behavior

- The external wrench tares only while the tool is stationary, auto-home is
  inactive, the gripper has settled, and the movement trigger or desktop
  motion keys are released.
- Operator motion and auto-home remain interlocked until the unloaded tare is
  complete.
- Auto-home suspends compliance and cancels if the configured external-force
  limit is reached.
- The WebSocket watchdog clears compliance motion and stops the arm when input
  becomes stale.
- Compliance and operator motion pass through the same 4 ft by 4 ft X/Y
  workspace limiter.
- External force above the configured limit suppresses operator motion. The
  limit stays latched until force falls below its release threshold and the
  operator releases the trigger or desktop motion keys.
- Quest haptics combine workspace proximity and contact-force intensity.
- Missing or invalid wrench fields disable the compliance layer without
  disabling the existing teleoperation controls. The debug overlay reports
  `wrench_unavailable` in this case.

The application workspace and force limit are not safety-rated. Configure
matching Kortex protection zones, speed limits, payload, tool transform, mass,
and center of mass before testing. Keep an emergency stop available.

The full Gen3 firmware must provide changing `tool_external_wrench_*` values for
this controller to respond. Kinova users have reported static wrench fields in
low-level mode and imperfect end-effector wrench accuracy. This application
keeps the arm in `SINGLE_LEVEL_SERVOING`, but the raw values in the debug overlay
still need to be checked on the actual arm before contact testing. See Kinova's
[low-level wrench feedback report](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L/issues/52)
and [wrench accuracy report](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L/issues/145).

## Impedance settings

All settings are optional environment variables. Vector settings accept one
value for all axes or three comma-separated values in base-frame X,Y,Z order.

| Variable | Default | Meaning |
| --- | --- | --- |
| `KINOVA_IMPEDANCE_ENABLED` | `true` | Enables the compliance layer |
| `KINOVA_IMPEDANCE_WRENCH_FRAME` | `base` | Set to `tool` to rotate tool-frame wrench feedback into the base frame |
| `KINOVA_IMPEDANCE_MASS_KG` | `4,4,4` | Virtual mass per axis |
| `KINOVA_IMPEDANCE_STIFFNESS_N_M` | `180,180,220` | Virtual stiffness per axis |
| `KINOVA_IMPEDANCE_DAMPING_NS_M` | `55,55,60` | Virtual damping per axis |
| `KINOVA_IMPEDANCE_FORCE_DEADBAND_N` | `1.5,1.5,1.5` | Force ignored around the tared zero |
| `KINOVA_IMPEDANCE_FORCE_SIGN` | `1,1,1` | Per-axis force direction, each value must be `1` or `-1` |
| `KINOVA_IMPEDANCE_MAX_DISPLACEMENT_M` | `0.05,0.05,0.05` | Maximum virtual spring displacement |
| `KINOVA_IMPEDANCE_MAX_VELOCITY_M_S` | `0.05` | Maximum compliance speed per axis |
| `KINOVA_IMPEDANCE_FILTER_CUTOFF_HZ` | `8` | External-force low-pass cutoff |
| `KINOVA_IMPEDANCE_FORCE_LIMIT_N` | `35` | Force norm that latches operator-motion suppression |
| `KINOVA_IMPEDANCE_FORCE_RELEASE_N` | `24` | Force norm required before the latch may clear |
| `KINOVA_IMPEDANCE_HAPTIC_START_N` | `2` | Contact force where Quest haptics begin |
| `KINOVA_IMPEDANCE_HAPTIC_FULL_SCALE_N` | `20` | Contact force for full haptic intensity |
| `KINOVA_IMPEDANCE_TARE_SAMPLES` | `30` | Stationary samples used to tare the wrench |
| `KINOVA_IMPEDANCE_TARE_MAX_FORCE_N` | `10` | Maximum raw force norm accepted as an unloaded tare sample |
| `KINOVA_IMPEDANCE_TARE_MAX_TOOL_SPEED_M_S` | `0.01` | Maximum measured tool speed allowed during tare |
| `KINOVA_IMPEDANCE_TARE_MAX_TOOL_ANGULAR_SPEED_DEG_S` | `1` | Maximum measured angular speed allowed during tare |
| `KINOVA_IMPEDANCE_TARE_AFTER_GRIPPER_DELAY_S` | `0.25` | Settling delay after a gripper command before tare samples resume |

## Impedance tuning and MuJoCo simulation

All impedance-specific tooling now lives in
[`impedance_tuning/`](impedance_tuning/README.md). That directory contains the
shared controller and profile, the Kortex hardware tuner, the physics-based
MuJoCo Gen3 tuner, a macOS Conda environment, tests, and complete operating
instructions.

The default generated profile is
`impedance_tuning/impedance_profile.json`; `main.py` loads it automatically.
The simulator never imports Kortex or connects to a robot.

## First hardware test

1. Start with the arm clear of people and obstacles and use low Kortex speed
   limits.
2. Leave the tool unloaded and motionless with the trigger released until the
   overlay changes from `calibrating` to `active`. A
   `tare_force_too_high` state means the tool is loaded, touching something, or
   its Kortex payload configuration needs correction.
3. Apply a light force along one axis at a time. The robot must yield in the
   same direction as the applied external force.
4. If an axis moves into the applied force, stop immediately and invert only
   that entry in `KINOVA_IMPEDANCE_FORCE_SIGN`.
5. Repeat the direction check after changing yaw. If the reported force axes
   rotate with the tool, set `KINOVA_IMPEDANCE_WRENCH_FRAME=tool`, restart, and
   repeat the unloaded tare and direction checks.
6. Confirm that workspace boundaries block both operator and compliant motion.
7. Confirm that exceeding the force limit suppresses operator motion and that
   releasing the trigger clears the latch only after force drops below the
   release threshold.

## Running

Set the Kortex credentials and provide the TLS certificate files expected by
`main.py`:

```bash
export KINOVA_USERNAME="your-user"
export KINOVA_PASSWORD="your-password"
python main.py
```

The current robot address is configured as `192.168.1.10`, and the Quest URL is
printed at startup.

Run the controller tests without robot hardware:

```bash
python -m unittest discover -s impedance_tuning/tests -v
```

Kortex references:

- [BaseCyclic feedback fields](https://docs.kinovarobotics.com/ref/autogen/Messages/BaseCyclic.html)
- [Kinova Gen3 user guide](https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf)
