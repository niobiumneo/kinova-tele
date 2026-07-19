# Kinova Gen3 impedance tuning

This directory contains the Cartesian impedance controller shared with
`main.py`, its JSON profile schema, the real-robot tuning program, and a
physics-based MuJoCo tuning program. Neither tuner starts the Quest server.

## Directory contents

- `impedance_controller.py`: translational virtual mass-spring-damper
- `impedance_profile.py`: validated JSON profile loading and saving
- `impedance_profile.example.json`: conservative checked-in example
- `tune_impedance.py`: terminal tuner for a real full Gen3 through Kortex
- `simulate_impedance.py`: terminal tuner for a full Gen3 in MuJoCo
- `environment.yml`: macOS-compatible Conda environment for simulation

## Conda setup on macOS

From the `kinova-tele` repository root:

```bash
conda env create -f impedance_tuning/environment.yml
conda activate kinova-impedance-sim
```

The environment uses conda-forge packages only. It supports both Intel and
Apple Silicon Macs. It does not need Kortex, FastAPI, Uvicorn, robot
credentials, or a Quest headset.

Fetch the official MuJoCo Menagerie model once:

```bash
git clone --depth 1 \
  https://github.com/google-deepmind/mujoco_menagerie.git \
  third_party/mujoco_menagerie
```

## Create and edit the shared profile

The default local profile lives in this directory and is ignored by Git:

```bash
python -m impedance_tuning.tune_impedance init
python -m impedance_tuning.tune_impedance validate
python -m impedance_tuning.tune_impedance wizard
```

The wizard does not connect to a robot or start MuJoCo. It edits the same
mass, stiffness, damping, deadband, force sign, filter, displacement, velocity,
and safety values consumed by `main.py`.

## MuJoCo tuning on a Mac

Use `mjpython` because the passive MuJoCo viewer requires it on macOS:

```bash
mjpython -m impedance_tuning.simulate_impedance --mode hold
mjpython -m impedance_tuning.simulate_impedance --mode sine --axis x
mjpython -m impedance_tuning.simulate_impedance --mode nullspace
```

There is no test timer. Close the viewer or press `Ctrl+C` to stop and save
the run.

Keyboard controls in the MuJoCo window:

| Key | Action |
| --- | --- |
| `1` / `2` | Apply continuous `-X` / `+X` tool force |
| `3` / `4` | Apply continuous `-Y` / `+Y` tool force |
| `5` / `6` | Apply continuous `-Z` / `+Z` tool force |
| `0` | Release the applied tool force |
| `[` / `]` | Decrease / increase the applied force |
| `J` / `L` | Negative / positive projected null-space motion |
| `K` | Stop projected null-space motion |
| `H` | Print the controls |

In `hold`, apply a force and press `0`; the virtual spring should return the
TCP toward its captured pose. `sine` adds the configured smooth single-axis
motion while the same compliance controller remains active. `nullspace`
projects the selected joint command through `I - J#J` while correcting TCP
position and orientation drift. Use `--nullspace-joint 1..7` to select the
seed joint. The null-space pose correction is independently adjustable with
`--nullspace-position-gain` and `--orientation-gain`.

The simulator reads MuJoCo's rigid-body external wrench, runs the exact
`CartesianImpedanceController`, converts its Cartesian velocity through a
damped full-pose Gen3 Jacobian, and integrates the seven position-actuator
targets. It is not a scripted force-response calculation.

The Menagerie Gen3 currently has an open report of aggressive default actuator
dynamics. The runner therefore applies conservative **simulation-only** gains,
joint damping, and armature by default. Use `--official-model-dynamics` to leave
the downloaded model untouched. Never copy these simulator actuator values to
Kortex or the impedance JSON profile.

Simulation is useful for checking controller state transitions, force signs,
limits, Cartesian return behavior, plotting, and qualitative gain changes. It
cannot validate the real arm's wrench estimate, payload model, friction,
latency, Kortex inner servo, protection zones, or native null-space controller.
Retune conservatively on the physical robot.

## Real-robot tuning

Use the machine and Python/Conda environment where Kinova's Kortex API is
installed. Set credentials first:

```bash
export KINOVA_USERNAME="your-user"
export KINOVA_PASSWORD="your-password"
```

Read-only monitoring sends no nonzero motion command:

```bash
python -m impedance_tuning.tune_impedance live --mode monitor
```

Motion modes require `--enable-motion` and an interactive `MOVE GEN3`
confirmation:

```bash
python -m impedance_tuning.tune_impedance live --mode hold --enable-motion
python -m impedance_tuning.tune_impedance live --mode sine --axis x --enable-motion
python -m impedance_tuning.tune_impedance live --mode nullspace --enable-motion
```

All modes run until `Ctrl+C` or a safety check stops them. The short timeout on
each Kortex twist is a stale-command watchdog, not a test timer.

## Artifacts and production loading

Real runs are written under `impedance_tuning/tuning_runs/`; simulated runs
are written under `impedance_tuning/simulation_runs/`. Each folder contains
the exact profile, telemetry CSV, summary JSON, and a six-panel PNG.

`main.py` automatically loads
`impedance_tuning/impedance_profile.json`. A different profile can be selected
with:

```bash
export KINOVA_IMPEDANCE_PROFILE="/absolute/path/to/profile.json"
python main.py
```

Environment variables named `KINOVA_IMPEDANCE_*` still override corresponding
JSON values.

## Tests

From the repository root, without a robot or MuJoCo model:

```bash
python -m unittest discover -s impedance_tuning/tests -v
```

References:

- [MuJoCo Python and macOS viewer documentation](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo Menagerie Kinova Gen3 model](https://github.com/google-deepmind/mujoco_menagerie/tree/main/kinova_gen3)
- [Open Gen3 model actuator-instability report](https://github.com/google-deepmind/mujoco_menagerie/issues/232)
- [Kinova Gen3 user guide](https://www.kinovarobotics.com/uploads/User-Guide-Gen3-R07.pdf)
