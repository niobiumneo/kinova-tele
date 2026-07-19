#!/usr/bin/env python3
"""MuJoCo hardware-style tuning harness for the full seven-axis Kinova Gen3."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from math import acos, asin, atan2, cos, degrees, pi, sin
from pathlib import Path
import sys
from threading import Lock
from time import monotonic, sleep
from typing import Any, Callable, Optional, Sequence

import numpy as np

from .impedance_controller import (
    CartesianImpedanceController,
    CartesianImpedanceOutput,
)
from .impedance_profile import DEFAULT_PROFILE_FILENAME, ImpedanceProfile
from .tune_impedance import (
    APP_DIR,
    AXIS_INDEX,
    LivePlot,
    RobotSnapshot,
    ZERO_VECTOR,
    build_sample,
    cap_vector_per_axis,
    load_profile_for_command,
    make_run_directory,
    resolve_profile_path,
    save_run_artifacts,
    smooth_sine_target,
    summarize_samples,
    vector_norm,
)


REPO_ROOT = APP_DIR.parent
DEFAULT_MODEL_PATH = (
    REPO_ROOT
    / "third_party"
    / "mujoco_menagerie"
    / "kinova_gen3"
    / "scene.xml"
)
DEFAULT_OUTPUT_DIR = APP_DIR / "simulation_runs"
JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
TOOL_SITE_NAME = "pinch_site"
TOOL_BODY_NAME = "bracelet_link"


def _load_mujoco() -> tuple[Any, Any]:
    try:
        import mujoco
        import mujoco.viewer
    except ImportError as exc:
        raise RuntimeError(
            "MuJoCo is not installed. Create the Conda environment with: "
            "conda env create -f impedance_tuning/environment.yml"
        ) from exc
    return mujoco, mujoco.viewer


def resolve_model_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def rotation_matrix_to_euler_xyz_degrees(matrix: np.ndarray) -> tuple[float, float, float]:
    """Convert a world rotation matrix to the XYZ convention used by Kortex."""
    matrix = np.asarray(matrix, dtype=float).reshape(3, 3)
    theta_y = asin(float(np.clip(-matrix[2, 0], -1.0, 1.0)))
    if abs(cos(theta_y)) > 1e-8:
        theta_x = atan2(matrix[2, 1], matrix[2, 2])
        theta_z = atan2(matrix[1, 0], matrix[0, 0])
    else:
        theta_x = atan2(-matrix[1, 2], matrix[1, 1])
        theta_z = 0.0
    return degrees(theta_x), degrees(theta_y), degrees(theta_z)


def rotation_error_vector(
    target_world: np.ndarray,
    current_world: np.ndarray,
) -> np.ndarray:
    """Return a world-frame rotation vector that turns current into target."""
    error = np.asarray(target_world) @ np.asarray(current_world).T
    cosine = float(np.clip((np.trace(error) - 1.0) / 2.0, -1.0, 1.0))
    angle = acos(cosine)
    skew = np.array(
        [
            error[2, 1] - error[1, 2],
            error[0, 2] - error[2, 0],
            error[1, 0] - error[0, 1],
        ],
        dtype=float,
    )
    if angle < 1e-8:
        return 0.5 * skew
    sine = sin(angle)
    if abs(sine) < 1e-8:
        diagonal_axis = np.sqrt(np.maximum(0.0, (np.diag(error) + 1.0) / 2.0))
        if np.linalg.norm(diagonal_axis) < 1e-8:
            return np.zeros(3)
        return angle * diagonal_axis / np.linalg.norm(diagonal_axis)
    return angle * skew / (2.0 * sine)


def damped_least_squares(
    jacobian: np.ndarray,
    damping: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a damped pseudoinverse and its joint-space null projector."""
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.ndim != 2:
        raise ValueError("jacobian must be a matrix")
    damping = float(damping)
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be a positive finite number")
    task_regularizer = (damping * damping) * np.eye(jacobian.shape[0])
    pseudoinverse = jacobian.T @ np.linalg.solve(
        jacobian @ jacobian.T + task_regularizer,
        np.eye(jacobian.shape[0]),
    )
    null_projector = np.eye(jacobian.shape[1]) - pseudoinverse @ jacobian
    return pseudoinverse, null_projector


def wrapped_joint_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    difference = np.asarray(target) - np.asarray(current)
    return np.arctan2(np.sin(difference), np.cos(difference))


class SimulationInputs:
    """Thread-safe keyboard state shared with MuJoCo's viewer callback."""

    FORCE_DIRECTIONS = {
        ord("1"): (-1.0, 0.0, 0.0),
        ord("2"): (1.0, 0.0, 0.0),
        ord("3"): (0.0, -1.0, 0.0),
        ord("4"): (0.0, 1.0, 0.0),
        ord("5"): (0.0, 0.0, -1.0),
        ord("6"): (0.0, 0.0, 1.0),
    }

    def __init__(self, force_magnitude_n: float) -> None:
        self._lock = Lock()
        self._force_magnitude_n = float(force_magnitude_n)
        self._force_direction = np.zeros(3)
        self._nullspace_direction = 0.0

    def key_callback(self, keycode: int) -> None:
        with self._lock:
            if keycode in self.FORCE_DIRECTIONS:
                self._force_direction = np.asarray(
                    self.FORCE_DIRECTIONS[keycode],
                    dtype=float,
                )
                print(
                    f"\nApplied force: {self.force_vector_unlocked()} N",
                    flush=True,
                )
            elif keycode == ord("0"):
                self._force_direction[:] = 0.0
                print("\nApplied force cleared.", flush=True)
            elif keycode == ord("["):
                self._force_magnitude_n = max(0.25, self._force_magnitude_n - 0.5)
                print(f"\nForce magnitude: {self._force_magnitude_n:.2f} N", flush=True)
            elif keycode == ord("]"):
                self._force_magnitude_n += 0.5
                print(f"\nForce magnitude: {self._force_magnitude_n:.2f} N", flush=True)
            elif keycode == ord("J"):
                self._nullspace_direction = -1.0
                print("\nNull-space command: negative", flush=True)
            elif keycode == ord("K"):
                self._nullspace_direction = 0.0
                print("\nNull-space command cleared.", flush=True)
            elif keycode == ord("L"):
                self._nullspace_direction = 1.0
                print("\nNull-space command: positive", flush=True)
            elif keycode == ord("H"):
                print_simulation_controls()

    def force_vector_unlocked(self) -> np.ndarray:
        return self._force_direction * self._force_magnitude_n

    def snapshot(self) -> tuple[np.ndarray, float, float]:
        with self._lock:
            return (
                self.force_vector_unlocked().copy(),
                self._nullspace_direction,
                self._force_magnitude_n,
            )


def print_simulation_controls() -> None:
    print(
        "\nSimulation controls:\n"
        "  1/2  apply -X/+X tool force\n"
        "  3/4  apply -Y/+Y tool force\n"
        "  5/6  apply -Z/+Z tool force\n"
        "  0    clear the applied force\n"
        "  [/]  decrease/increase force magnitude\n"
        "  J/L  command negative/positive projected null-space motion\n"
        "  K    clear null-space motion\n"
        "  H    print these controls\n"
        "  Close the viewer or press Ctrl+C to finish and save plots.\n"
    )


@dataclass(frozen=True)
class ModelBindings:
    tool_site_id: int
    tool_body_id: int
    joint_ids: np.ndarray
    actuator_ids: np.ndarray
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    home_key_id: int


def _required_id(mujoco: Any, model: Any, object_type: Any, name: str) -> int:
    object_id = int(mujoco.mj_name2id(model, object_type, name))
    if object_id < 0:
        raise RuntimeError(f"MuJoCo model is missing required object {name!r}")
    return object_id


def bind_model(mujoco: Any, model: Any) -> ModelBindings:
    joint_ids = np.asarray(
        [
            _required_id(mujoco, model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ],
        dtype=int,
    )
    actuator_ids = np.asarray(
        [
            _required_id(mujoco, model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in JOINT_NAMES
        ],
        dtype=int,
    )
    home_key_id = int(
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    )
    return ModelBindings(
        tool_site_id=_required_id(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_SITE,
            TOOL_SITE_NAME,
        ),
        tool_body_id=_required_id(
            mujoco,
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            TOOL_BODY_NAME,
        ),
        joint_ids=joint_ids,
        actuator_ids=actuator_ids,
        qpos_addresses=np.asarray(model.jnt_qposadr[joint_ids], dtype=int),
        dof_addresses=np.asarray(model.jnt_dofadr[joint_ids], dtype=int),
        home_key_id=home_key_id,
    )


def apply_conservative_sim_dynamics(model: Any, bindings: ModelBindings) -> None:
    """Apply documented simulation-only stabilization to the Menagerie model."""
    for joint_index, (actuator_id, dof_address) in enumerate(
        zip(bindings.actuator_ids, bindings.dof_addresses)
    ):
        kp, kv = ((500.0, 20.0) if joint_index < 4 else (100.0, 2.0))
        model.actuator_gainprm[actuator_id, 0] = kp
        model.actuator_biasprm[actuator_id, 1] = -kp
        model.actuator_biasprm[actuator_id, 2] = -kv
        model.dof_armature[dof_address] = max(
            float(model.dof_armature[dof_address]),
            0.1,
        )
        model.dof_damping[dof_address] = max(
            float(model.dof_damping[dof_address]),
            1.0,
        )


def reset_home(
    mujoco: Any,
    model: Any,
    data: Any,
    bindings: ModelBindings,
) -> np.ndarray:
    if bindings.home_key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, bindings.home_key_id)
    else:
        mujoco.mj_resetData(model, data)
    targets = np.asarray(data.qpos[bindings.qpos_addresses], dtype=float).copy()
    data.ctrl[bindings.actuator_ids] = targets
    mujoco.mj_forward(model, data)
    return targets


def read_snapshot(
    mujoco: Any,
    model: Any,
    data: Any,
    bindings: ModelBindings,
    profile: ImpedanceProfile,
) -> tuple[RobotSnapshot, Optional[Callable[[tuple[float, float, float]], Sequence[float]]], np.ndarray]:
    mujoco.mj_rnePostConstraint(model, data)
    site_rotation = np.asarray(
        data.site_xmat[bindings.tool_site_id],
        dtype=float,
    ).reshape(3, 3).copy()
    site_velocity = np.zeros(6)
    mujoco.mj_objectVelocity(
        model,
        data,
        mujoco.mjtObj.mjOBJ_SITE,
        bindings.tool_site_id,
        site_velocity,
        0,
    )
    world_wrench = np.asarray(data.cfrc_ext[bindings.tool_body_id], dtype=float)
    world_torque = world_wrench[:3]
    world_force = world_wrench[3:]
    force_transform = None
    if profile.runtime.wrench_frame == "tool":
        raw_force = site_rotation.T @ world_force
        raw_torque = site_rotation.T @ world_torque

        def force_transform(force: tuple[float, float, float]) -> Sequence[float]:
            return site_rotation @ np.asarray(force, dtype=float)

    else:
        raw_force = world_force
        raw_torque = world_torque

    joint_positions = np.asarray(data.qpos[bindings.qpos_addresses], dtype=float)
    joint_velocities = np.asarray(data.qvel[bindings.dof_addresses], dtype=float)
    joint_torques = np.asarray(data.qfrc_actuator[bindings.dof_addresses], dtype=float)
    snapshot = RobotSnapshot(
        position_m=tuple(
            float(value) for value in data.site_xpos[bindings.tool_site_id]
        ),
        orientation_deg=rotation_matrix_to_euler_xyz_degrees(site_rotation),
        linear_velocity_m_s=tuple(float(value) for value in site_velocity[3:]),
        angular_velocity_deg_s=tuple(
            degrees(float(value)) for value in site_velocity[:3]
        ),
        raw_force_n=tuple(float(value) for value in raw_force),
        raw_torque_nm=tuple(float(value) for value in raw_torque),
        wrench_available=bool(
            np.all(np.isfinite(raw_force)) and np.all(np.isfinite(raw_torque))
        ),
        joint_positions_deg=tuple(degrees(float(value)) for value in joint_positions),
        joint_velocities_deg_s=tuple(
            degrees(float(value)) for value in joint_velocities
        ),
        joint_torques_nm=tuple(float(value) for value in joint_torques),
    )
    return snapshot, force_transform, site_rotation


def compute_joint_velocity(
    *,
    jacobian: np.ndarray,
    task_twist: np.ndarray,
    posture_seed: np.ndarray,
    damping: float,
    max_joint_speed_rad_s: float,
) -> np.ndarray:
    pseudoinverse, null_projector = damped_least_squares(jacobian, damping)
    joint_velocity = pseudoinverse @ task_twist + null_projector @ posture_seed
    return np.clip(
        joint_velocity,
        -max_joint_speed_rad_s,
        max_joint_speed_rad_s,
    )


def clamp_control_targets(
    model: Any,
    bindings: ModelBindings,
    targets: np.ndarray,
) -> np.ndarray:
    targets = np.asarray(targets, dtype=float).copy()
    for index, actuator_id in enumerate(bindings.actuator_ids):
        if bool(model.actuator_ctrllimited[actuator_id]):
            lower, upper = model.actuator_ctrlrange[actuator_id]
            targets[index] = np.clip(targets[index], lower, upper)
    return targets


def _validate_args(args: argparse.Namespace) -> None:
    positive_fields = (
        "dls_damping",
        "orientation_gain",
        "nullspace_position_gain",
        "posture_gain",
        "max_joint_speed_deg_s",
        "nullspace_speed_deg_s",
        "manual_force_n",
        "realtime_factor",
    )
    for name in positive_fields:
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name.replace('_', '-')} must be positive and finite")
    if not 1 <= args.nullspace_joint <= 7:
        raise ValueError("nullspace-joint must be from 1 through 7")


def run_simulation(args: argparse.Namespace, profile_path: Path) -> int:
    _validate_args(args)
    profile = load_profile_for_command(profile_path)
    model_path = resolve_model_path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Gen3 MuJoCo scene not found: {model_path}\n"
            "Fetch it with:\n"
            "  git clone --depth 1 https://github.com/google-deepmind/"
            "mujoco_menagerie.git third_party/mujoco_menagerie"
        )

    mujoco, viewer_module = _load_mujoco()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    bindings = bind_model(mujoco, model)
    if not args.official_model_dynamics:
        apply_conservative_sim_dynamics(model, bindings)
    control_targets = reset_home(mujoco, model, data, bindings)

    initial_snapshot, _, initial_rotation = read_snapshot(
        mujoco,
        model,
        data,
        bindings,
        profile,
    )
    initial_position = np.asarray(initial_snapshot.position_m, dtype=float)
    initial_joints = np.asarray(data.qpos[bindings.qpos_addresses], dtype=float).copy()
    controller = CartesianImpedanceController(profile.impedance)
    controls = SimulationInputs(args.manual_force_n)
    run_mode = f"simulation_{args.mode}"
    run_directory = make_run_directory(args.output_dir, run_mode)
    plot = LivePlot(
        enabled=not args.no_live_plot,
        window_s=profile.tuning.plot_window_s,
        title=f"MuJoCo Gen3: {args.mode}",
    )
    samples: list[dict[str, Any]] = []
    stop_reason = "stopped before the simulation loop started"
    completed = False
    tare_announced = False
    motion_started_at: Optional[float] = None
    control_period_s = 1.0 / profile.tuning.loop_hz
    previous_control_time = float(data.time)
    next_control_time = float(data.time)
    next_view_sync_time = float(data.time)
    wall_anchor = monotonic()
    simulation_anchor = float(data.time)
    last_output: Optional[CartesianImpedanceOutput] = None

    print(f"Loaded MuJoCo Gen3: {model_path}")
    print(f"Profile: {profile_path}")
    print(f"Mode: {args.mode}; no Kortex connection will be made.")
    if not args.official_model_dynamics:
        print("Using conservative simulation-only actuator stabilization.")
    print_simulation_controls()

    try:
        with viewer_module.launch_passive(
            model,
            data,
            key_callback=controls.key_callback,
        ) as viewer:
            while viewer.is_running():
                simulation_time = float(data.time)
                if simulation_time >= next_view_sync_time:
                    viewer.sync()
                    next_view_sync_time = simulation_time + 1.0 / 60.0

                manual_force, nullspace_direction, _ = controls.snapshot()
                data.xfrc_applied[bindings.tool_body_id, :3] += manual_force

                if simulation_time + 1e-12 >= next_control_time:
                    dt_s = max(
                        profile.impedance.min_dt_s,
                        simulation_time - previous_control_time,
                    )
                    previous_control_time = simulation_time
                    next_control_time += control_period_s
                    if next_control_time < simulation_time:
                        next_control_time = simulation_time + control_period_s

                    snapshot, force_transform, current_rotation = read_snapshot(
                        mujoco,
                        model,
                        data,
                        bindings,
                        profile,
                    )
                    tare_allowed = (
                        vector_norm(snapshot.linear_velocity_m_s)
                        <= profile.runtime.tare_max_tool_speed_m_s
                        and vector_norm(snapshot.angular_velocity_deg_s)
                        <= profile.runtime.tare_max_tool_angular_speed_deg_s
                    )
                    output = controller.update(
                        snapshot.raw_force_n,
                        dt_s,
                        wrench_available=snapshot.wrench_available,
                        allow_tare=tare_allowed,
                        allow_motion=args.mode in ("hold", "sine"),
                        allow_force_limit_release=True,
                        force_transform=force_transform,
                    )
                    last_output = output
                    if not output.wrench_available:
                        stop_reason = "simulated wrench became non-finite"
                        break
                    if output.force_limit_active:
                        stop_reason = (
                            f"force limit reached ({output.force_norm_n:.2f} N)"
                        )
                        break

                    current_position = np.asarray(snapshot.position_m, dtype=float)
                    tcp_offset = current_position - initial_position
                    tcp_deviation = float(np.linalg.norm(tcp_offset))
                    orientation_error = rotation_error_vector(
                        initial_rotation,
                        current_rotation,
                    )
                    orientation_drift_deg = degrees(
                        float(np.linalg.norm(orientation_error))
                    )
                    tcp_limit = (
                        profile.tuning.nullspace_max_tcp_drift_m
                        if args.mode == "nullspace"
                        else profile.tuning.max_tcp_deviation_m
                    )
                    orientation_limit = (
                        profile.tuning.nullspace_max_orientation_drift_deg
                        if args.mode == "nullspace"
                        else profile.tuning.max_orientation_deviation_deg
                    )
                    if tcp_deviation > tcp_limit:
                        stop_reason = (
                            f"TCP deviation {tcp_deviation:.4f}m exceeded "
                            f"{tcp_limit:.4f}m"
                        )
                        break
                    if orientation_drift_deg > orientation_limit:
                        stop_reason = (
                            f"orientation drift {orientation_drift_deg:.2f}deg "
                            f"exceeded {orientation_limit:.2f}deg"
                        )
                        break

                    current_joints = np.asarray(
                        data.qpos[bindings.qpos_addresses],
                        dtype=float,
                    )
                    joint_delta_deg = np.abs(
                        np.degrees(wrapped_joint_error(current_joints, initial_joints))
                    )
                    if (
                        args.mode == "nullspace"
                        and float(np.max(joint_delta_deg))
                        > profile.tuning.nullspace_max_joint_delta_deg
                    ):
                        stop_reason = "null-space joint displacement bound reached"
                        break

                    tare_complete = (
                        output.tare_samples_collected
                        >= output.tare_samples_required
                    )
                    if tare_complete and not tare_announced:
                        tare_announced = True
                        motion_started_at = simulation_time
                        print(f"Tare complete. {args.mode} simulation is active.")

                    command_velocity = np.zeros(3)
                    nominal_velocity = np.zeros(3)
                    desired_offset = np.zeros(3)
                    if args.mode == "hold":
                        command_velocity = np.asarray(
                            cap_vector_per_axis(
                                output.compliance_velocity_m_s,
                                min(
                                    profile.tuning.max_test_velocity_m_s,
                                    profile.impedance.max_velocity_m_s,
                                ),
                            )
                        )
                    elif args.mode == "sine":
                        axis = AXIS_INDEX[args.axis]
                        active_time = (
                            simulation_time - motion_started_at
                            if motion_started_at is not None
                            else 0.0
                        )
                        desired_position, feedforward_velocity = smooth_sine_target(
                            active_time,
                            profile.tuning.excitation_amplitude_m,
                            profile.tuning.excitation_frequency_hz,
                        )
                        desired_offset[axis] = desired_position
                        nominal_velocity[axis] = (
                            feedforward_velocity
                            + profile.tuning.excitation_position_gain_s_inv
                            * (desired_position - tcp_offset[axis])
                        )
                        nominal_velocity = np.asarray(
                            cap_vector_per_axis(
                                nominal_velocity,
                                profile.tuning.max_test_velocity_m_s,
                            )
                        )
                        command_velocity = np.asarray(
                            cap_vector_per_axis(
                                nominal_velocity
                                + np.asarray(output.compliance_velocity_m_s),
                                profile.tuning.max_test_velocity_m_s,
                            )
                        )
                    elif args.mode == "nullspace":
                        command_velocity = np.asarray(
                            cap_vector_per_axis(
                                -args.nullspace_position_gain * tcp_offset,
                                profile.tuning.max_test_velocity_m_s,
                            )
                        )

                    jacobian_position = np.zeros((3, model.nv))
                    jacobian_rotation = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(
                        model,
                        data,
                        jacobian_position,
                        jacobian_rotation,
                        bindings.tool_site_id,
                    )
                    jacobian = np.vstack(
                        (
                            jacobian_position[:, bindings.dof_addresses],
                            jacobian_rotation[:, bindings.dof_addresses],
                        )
                    )
                    angular_velocity = args.orientation_gain * orientation_error
                    task_twist = np.concatenate((command_velocity, angular_velocity))
                    posture_seed = args.posture_gain * wrapped_joint_error(
                        initial_joints,
                        current_joints,
                    )
                    if args.mode == "nullspace":
                        posture_seed[:] = 0.0
                        posture_seed[args.nullspace_joint - 1] = (
                            nullspace_direction
                            * args.nullspace_speed_deg_s
                            * pi
                            / 180.0
                        )
                    if not tare_complete:
                        task_twist[:] = 0.0
                        posture_seed[:] = 0.0

                    joint_velocity = compute_joint_velocity(
                        jacobian=jacobian,
                        task_twist=task_twist,
                        posture_seed=posture_seed,
                        damping=args.dls_damping,
                        max_joint_speed_rad_s=(
                            args.max_joint_speed_deg_s * pi / 180.0
                        ),
                    )
                    control_targets += joint_velocity * dt_s
                    control_targets = clamp_control_targets(
                        model,
                        bindings,
                        control_targets,
                    )
                    data.ctrl[bindings.actuator_ids] = control_targets

                    sample = build_sample(
                        elapsed_s=simulation_time,
                        dt_s=dt_s,
                        mode=run_mode,
                        output=output,
                        snapshot=snapshot,
                        initial_snapshot=initial_snapshot,
                        command_velocity_m_s=tuple(command_velocity),
                        nominal_velocity_m_s=tuple(nominal_velocity),
                        desired_offset_m=tuple(desired_offset),
                    )
                    samples.append(sample)
                    plot.update(samples, monotonic())
                    if len(samples) % max(1, int(profile.tuning.loop_hz)) == 0:
                        print(
                            f"t={simulation_time:7.2f}s "
                            f"state={output.state:18s} "
                            f"F={output.force_norm_n:6.2f}N "
                            f"TCP={tcp_deviation * 1000.0:6.1f}mm",
                            end="\r",
                            flush=True,
                        )

                mujoco.mj_step(model, data)
                data.xfrc_applied[bindings.tool_body_id, :3] -= manual_force

                wall_target = wall_anchor + (
                    (float(data.time) - simulation_anchor) / args.realtime_factor
                )
                remaining = wall_target - monotonic()
                if remaining > 0.0:
                    sleep(min(remaining, 0.01))
                elif remaining < -0.25:
                    wall_anchor = monotonic()
                    simulation_anchor = float(data.time)

            if stop_reason == "stopped before the simulation loop started":
                completed = True
                stop_reason = "operator closed the MuJoCo viewer"
    except KeyboardInterrupt:
        completed = True
        stop_reason = "operator pressed Ctrl+C"
    except Exception as exc:
        stop_reason = f"error: {exc}"
        print(stop_reason, file=sys.stderr)
    finally:
        plot.close()
        summary = summarize_samples(
            samples,
            mode=run_mode,
            stop_reason=stop_reason,
            completed=completed,
        )
        summary.update(
            {
                "simulator": "MuJoCo",
                "model_path": str(model_path),
                "profile_path": str(profile_path),
                "official_model_dynamics": args.official_model_dynamics,
                "dls_damping": args.dls_damping,
                "orientation_gain_s_inv": args.orientation_gain,
                "nullspace_position_gain_s_inv": args.nullspace_position_gain,
                "posture_gain_s_inv": args.posture_gain,
                "maximum_joint_speed_deg_s": args.max_joint_speed_deg_s,
                "nullspace_joint": args.nullspace_joint,
                "nullspace_speed_deg_s": args.nullspace_speed_deg_s,
                "last_controller_state": (
                    last_output.state if last_output is not None else None
                ),
            }
        )
        save_run_artifacts(run_directory, profile, samples, summary)
        print("\n" + json.dumps(summary, indent=2))
        print(f"Artifacts: {run_directory}")
    return 0 if completed else 2


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Cartesian impedance controller against a physics-based "
            "MuJoCo model of the full Kinova Gen3."
        )
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_FILENAME,
        help=(
            "profile path relative to impedance_tuning/ "
            f"(default: {DEFAULT_PROFILE_FILENAME})"
        ),
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="path to MuJoCo Menagerie kinova_gen3/scene.xml",
    )
    parser.add_argument(
        "--mode",
        choices=("hold", "sine", "nullspace"),
        required=True,
    )
    parser.add_argument("--axis", choices=tuple(AXIS_INDEX), default="x")
    parser.add_argument("--manual-force-n", type=float, default=5.0)
    parser.add_argument("--dls-damping", type=float, default=0.05)
    parser.add_argument("--orientation-gain", type=float, default=2.0)
    parser.add_argument("--nullspace-position-gain", type=float, default=2.0)
    parser.add_argument("--posture-gain", type=float, default=0.25)
    parser.add_argument("--max-joint-speed-deg-s", type=float, default=15.0)
    parser.add_argument("--nullspace-joint", type=int, default=3)
    parser.add_argument("--nullspace-speed-deg-s", type=float, default=8.0)
    parser.add_argument("--realtime-factor", type=float, default=1.0)
    parser.add_argument(
        "--official-model-dynamics",
        action="store_true",
        help=(
            "do not apply conservative simulation-only actuator stabilization"
        ),
    )
    parser.add_argument(
        "--no-live-plot",
        action="store_true",
        help="show only MuJoCo while still saving the final plot and telemetry",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    profile_path = resolve_profile_path(args.profile)
    try:
        return run_simulation(args, profile_path)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
