#!/usr/bin/env python3
"""Standalone Cartesian impedance tuner for the full Kinova Gen3.

This program is intentionally separate from the Quest teleoperation server.
It can edit/validate a JSON profile, monitor a connected arm without commanding
motion, and run safety-bounded live tests with explicit operator confirmation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import json
from math import (
    acos,
    cos,
    degrees,
    isfinite,
    pi,
    radians,
    sin,
    sqrt,
)
import os
from pathlib import Path
import multiprocessing
import queue
import statistics
import sys
import tempfile
from time import monotonic, sleep
from typing import Any, Callable, Optional, Sequence, Tuple

from .impedance_controller import (
    CartesianImpedanceController,
    CartesianImpedanceOutput,
)
from .impedance_profile import (
    DEFAULT_PROFILE_FILENAME,
    ImpedanceProfile,
    conservative_tuning_profile,
    critical_damping,
    damping_ratios,
    load_impedance_profile,
    production_default_profile,
    save_impedance_profile,
)


Vector3 = Tuple[float, float, float]
ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)
AXIS_INDEX = {"x": 0, "y": 1, "z": 2}
MOTION_CONFIRMATION = "MOVE GEN3"
APP_DIR = Path(__file__).resolve().parent


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def vector_norm(values: Sequence[float]) -> float:
    return sqrt(sum(float(value) ** 2 for value in values))


def vector_subtract(left: Sequence[float], right: Sequence[float]) -> Vector3:
    return tuple(float(left[index]) - float(right[index]) for index in range(3))  # type: ignore[return-value]


def wrapped_angle_difference_degrees(current: float, initial: float) -> float:
    return (float(current) - float(initial) + 180.0) % 360.0 - 180.0


def quaternion_multiply(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def euler_xyz_degrees_to_quaternion(
    theta_x: float,
    theta_y: float,
    theta_z: float,
) -> tuple[float, float, float, float]:
    half_x = radians(theta_x) / 2.0
    half_y = radians(theta_y) / 2.0
    half_z = radians(theta_z) / 2.0
    qx = (cos(half_x), sin(half_x), 0.0, 0.0)
    qy = (cos(half_y), 0.0, sin(half_y), 0.0)
    qz = (cos(half_z), 0.0, 0.0, sin(half_z))
    return quaternion_multiply(qz, quaternion_multiply(qy, qx))


def quaternion_rotate_vector(
    quaternion: Sequence[float],
    vector: Sequence[float],
) -> Vector3:
    conjugate = (
        quaternion[0],
        -quaternion[1],
        -quaternion[2],
        -quaternion[3],
    )
    rotated = quaternion_multiply(
        quaternion_multiply(quaternion, (0.0, *vector)),
        conjugate,
    )
    return rotated[1], rotated[2], rotated[3]


def orientation_distance_degrees(
    left_deg: Sequence[float],
    right_deg: Sequence[float],
) -> float:
    left = euler_xyz_degrees_to_quaternion(*left_deg)
    right = euler_xyz_degrees_to_quaternion(*right_deg)
    dot = abs(sum(left[index] * right[index] for index in range(4)))
    return degrees(2.0 * acos(clamp(dot, -1.0, 1.0)))


def rotate_tool_force_to_base(
    tool_angles_deg: Sequence[float],
) -> Callable[[Vector3], Vector3]:
    quaternion = euler_xyz_degrees_to_quaternion(*tool_angles_deg)
    return lambda force: quaternion_rotate_vector(quaternion, force)


def _float_attribute(value: Any, name: str, *, default: float = 0.0) -> float:
    try:
        result = float(getattr(value, name))
    except (AttributeError, TypeError, ValueError):
        return default
    return result if isfinite(result) else default


@dataclass(frozen=True)
class RobotSnapshot:
    position_m: Vector3
    orientation_deg: Vector3
    linear_velocity_m_s: Vector3
    angular_velocity_deg_s: Vector3
    raw_force_n: Vector3
    raw_torque_nm: Vector3
    wrench_available: bool
    joint_positions_deg: tuple[float, ...]
    joint_velocities_deg_s: tuple[float, ...]
    joint_torques_nm: tuple[float, ...]


class TuningRobot:
    """Small Kortex connection used only by the standalone tuning program."""

    def __init__(self, ip: str, port: int, username: str, password: str) -> None:
        self.ip = ip
        self.port = int(port)
        self.username = username
        self.password = password
        self.transport = None
        self.router = None
        self.session_manager = None
        self.base = None
        self.base_cyclic = None
        self.Base_pb2 = None
        self._native_admittance_active = False

    def connect(self) -> None:
        try:
            from kortex_api.RouterClient import RouterClient
            from kortex_api.SessionManager import SessionManager
            from kortex_api.TCPTransport import TCPTransport
            from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
            from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import (
                BaseCyclicClient,
            )
            from kortex_api.autogen.messages import Base_pb2, Session_pb2
        except ImportError as exc:
            raise RuntimeError(
                "The Kortex Python API is required only for live modes. "
                "Install the Kinova wheel in this Python environment."
            ) from exc

        self.Base_pb2 = Base_pb2
        self.transport = TCPTransport()
        self.router = RouterClient(
            self.transport,
            lambda error: print(f"Kortex router error: {error}", file=sys.stderr),
        )
        self.transport.connect(self.ip, self.port)

        session_info = Session_pb2.CreateSessionInfo()
        session_info.username = self.username
        session_info.password = self.password
        session_info.session_inactivity_timeout = 60000
        session_info.connection_inactivity_timeout = 2000

        self.session_manager = SessionManager(self.router)
        self.session_manager.CreateSession(session_info)
        self.base = BaseClient(self.router)
        self.base_cyclic = BaseCyclicClient(self.router)

        servo_mode = Base_pb2.ServoingModeInformation()
        servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
        self.base.SetServoingMode(servo_mode)

    def read_snapshot(self) -> RobotSnapshot:
        if self.base_cyclic is None:
            raise RuntimeError("robot is not connected")
        feedback = self.base_cyclic.RefreshFeedback()
        base = feedback.base

        raw_force = (
            _float_attribute(base, "tool_external_wrench_force_x", default=float("nan")),
            _float_attribute(base, "tool_external_wrench_force_y", default=float("nan")),
            _float_attribute(base, "tool_external_wrench_force_z", default=float("nan")),
        )
        raw_torque = (
            _float_attribute(base, "tool_external_wrench_torque_x", default=float("nan")),
            _float_attribute(base, "tool_external_wrench_torque_y", default=float("nan")),
            _float_attribute(base, "tool_external_wrench_torque_z", default=float("nan")),
        )
        wrench_available = all(isfinite(value) for value in raw_force + raw_torque)
        if not wrench_available:
            raw_force = ZERO_VECTOR
            raw_torque = ZERO_VECTOR

        positions = tuple(float(actuator.position) for actuator in feedback.actuators)
        velocities = tuple(
            _float_attribute(actuator, "velocity") for actuator in feedback.actuators
        )
        torques = tuple(
            _float_attribute(actuator, "torque") for actuator in feedback.actuators
        )
        return RobotSnapshot(
            position_m=(
                _float_attribute(base, "tool_pose_x"),
                _float_attribute(base, "tool_pose_y"),
                _float_attribute(base, "tool_pose_z"),
            ),
            orientation_deg=(
                _float_attribute(base, "tool_pose_theta_x"),
                _float_attribute(base, "tool_pose_theta_y"),
                _float_attribute(base, "tool_pose_theta_z"),
            ),
            linear_velocity_m_s=(
                _float_attribute(base, "tool_twist_linear_x"),
                _float_attribute(base, "tool_twist_linear_y"),
                _float_attribute(base, "tool_twist_linear_z"),
            ),
            angular_velocity_deg_s=(
                _float_attribute(base, "tool_twist_angular_x"),
                _float_attribute(base, "tool_twist_angular_y"),
                _float_attribute(base, "tool_twist_angular_z"),
            ),
            raw_force_n=raw_force,
            raw_torque_nm=raw_torque,
            wrench_available=wrench_available,
            joint_positions_deg=positions,
            joint_velocities_deg_s=velocities,
            joint_torques_nm=torques,
        )

    def send_twist(
        self,
        velocity_m_s: Sequence[float],
        *,
        timeout_s: float = 0.1,
    ) -> None:
        if self.base is None or self.Base_pb2 is None:
            raise RuntimeError("robot is not connected")
        command = self.Base_pb2.TwistCommand()
        command.reference_frame = self.Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE
        command.twist.linear_x = float(velocity_m_s[0])
        command.twist.linear_y = float(velocity_m_s[1])
        command.twist.linear_z = float(velocity_m_s[2])
        command.twist.angular_x = 0.0
        command.twist.angular_y = 0.0
        command.twist.angular_z = 0.0
        # Current Kortex APIs express this timeout in seconds. Older generated
        # messages used an integer millisecond field, so retain compatibility.
        if hasattr(command, "duration"):
            try:
                command.duration = float(timeout_s)
            except (TypeError, ValueError):
                command.duration = max(1, int(round(timeout_s * 1000.0)))
        self.base.SendTwistCommand(command)

    def set_native_nullspace_admittance(self, enabled: bool) -> None:
        if self.base is None or self.Base_pb2 is None:
            raise RuntimeError("robot is not connected")
        admittance = self.Base_pb2.Admittance()
        mode_name = "NULL_SPACE" if enabled else "DISABLED"
        try:
            admittance.admittance_mode = getattr(self.Base_pb2, mode_name)
        except AttributeError as exc:
            raise RuntimeError(
                f"this Kortex API does not expose {mode_name} admittance"
            ) from exc
        self.base.SetAdmittance(admittance)
        self._native_admittance_active = enabled

    def stop(self) -> None:
        if self.base is None:
            return
        if self._native_admittance_active:
            try:
                self.set_native_nullspace_admittance(False)
            except Exception as exc:
                print(f"Could not disable native admittance: {exc}", file=sys.stderr)
        try:
            self.send_twist(ZERO_VECTOR)
        except Exception:
            pass
        try:
            self.base.Stop()
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        if self.session_manager is not None:
            try:
                self.session_manager.CloseSession()
            except Exception:
                pass
        if self.transport is not None:
            try:
                self.transport.disconnect()
            except Exception:
                pass
        self.base = None
        self.base_cyclic = None


def resolve_profile_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = APP_DIR / path
    return path.resolve()


def load_profile_for_command(path: Path) -> ImpedanceProfile:
    if not path.exists():
        raise FileNotFoundError(
            f"profile does not exist: {path}\n"
            "Create it with: python -m impedance_tuning.tune_impedance "
            f"--profile {path} init"
        )
    return load_impedance_profile(path)


def print_profile_summary(profile: ImpedanceProfile, path: Optional[Path] = None) -> None:
    config = profile.impedance
    critical = critical_damping(config)
    ratios = damping_ratios(config)
    if path is not None:
        print(f"Profile: {path}")
    print(f"Name: {profile.name}")
    if profile.description:
        print(f"Description: {profile.description}")
    print(f"M [kg]: {config.mass_kg}")
    print(f"K [N/m]: {config.stiffness_n_m}")
    print(f"D [N*s/m]: {config.damping_n_s_m}")
    print("Critical D: " + ", ".join(f"{value:.2f}" for value in critical))
    print("Damping ratio: " + ", ".join(f"{value:.2f}" for value in ratios))
    print(f"Deadband [N]: {config.force_deadband_n}")
    print(f"Force sign: {config.force_axis_sign}")
    print(f"Wrench frame: {profile.runtime.wrench_frame}")
    print(f"Max displacement [m]: {config.max_displacement_m}")
    print(f"Max compliance speed [m/s]: {config.max_velocity_m_s}")
    print(
        f"Force latch/release [N]: {config.force_limit_n} / "
        f"{config.force_release_n}"
    )


def _prompt_text(label: str, current: str) -> str:
    value = input(f"{label} [{current}]: ").strip()
    return value if value else current


def _prompt_float(label: str, current: float) -> float:
    while True:
        value = input(f"{label} [{current:g}]: ").strip()
        if not value:
            return current
        try:
            result = float(value)
        except ValueError:
            print("Enter one finite number.")
            continue
        if isfinite(result):
            return result
        print("Enter one finite number.")


def _prompt_vector(label: str, current: Sequence[float]) -> Vector3:
    current_text = ",".join(f"{value:g}" for value in current)
    while True:
        value = input(f"{label} X,Y,Z [{current_text}]: ").strip()
        if not value:
            return tuple(float(item) for item in current)  # type: ignore[return-value]
        try:
            result = tuple(float(item.strip()) for item in value.split(","))
        except ValueError:
            print("Enter three comma-separated finite numbers.")
            continue
        if len(result) == 3 and all(isfinite(item) for item in result):
            return result  # type: ignore[return-value]
        print("Enter three comma-separated finite numbers.")


def command_init(args: argparse.Namespace, profile_path: Path) -> int:
    profile = (
        conservative_tuning_profile()
        if args.preset == "conservative"
        else production_default_profile()
    )
    save_impedance_profile(profile_path, profile, overwrite=args.force)
    print(f"Wrote {args.preset} profile: {profile_path}")
    print_profile_summary(profile)
    return 0


def command_wizard(_args: argparse.Namespace, profile_path: Path) -> int:
    if not sys.stdin.isatty():
        raise RuntimeError("wizard requires an interactive terminal")
    profile = (
        load_impedance_profile(profile_path)
        if profile_path.exists()
        else conservative_tuning_profile()
    )
    config = profile.impedance
    runtime = profile.runtime
    tuning = profile.tuning

    print("\nCartesian impedance profile wizard")
    print("Press Enter to keep the displayed value. No robot connection is made.\n")
    name = _prompt_text("Profile name", profile.name)
    description = _prompt_text("Description", profile.description)
    mass = _prompt_vector("Virtual mass [kg]", config.mass_kg)
    stiffness = _prompt_vector("Stiffness [N/m]", config.stiffness_n_m)
    suggested_damping = tuple(
        2.0 * sqrt(mass[index] * stiffness[index]) for index in range(3)
    )
    print(
        "Suggested critical damping: "
        + ",".join(f"{value:.2f}" for value in suggested_damping)
    )
    damping = _prompt_vector("Damping [N*s/m]", config.damping_n_s_m)
    deadband = _prompt_vector("Force deadband [N]", config.force_deadband_n)
    sign = _prompt_vector("Force sign (+1 or -1)", config.force_axis_sign)
    displacement = _prompt_vector(
        "Maximum displacement [m]",
        config.max_displacement_m,
    )
    max_velocity = _prompt_float(
        "Maximum compliance speed [m/s]",
        config.max_velocity_m_s,
    )
    cutoff = _prompt_float("Force filter cutoff [Hz]", config.filter_cutoff_hz)
    force_limit = _prompt_float("Force limit [N]", config.force_limit_n)
    force_release = _prompt_float("Force release [N]", config.force_release_n)
    tare_max = _prompt_float("Maximum raw tare force [N]", config.tare_max_force_n)
    frame = _prompt_text("Wrench frame (base/tool)", runtime.wrench_frame).lower()
    print("\nStandalone live-test envelope")
    max_test_velocity = _prompt_float(
        "Maximum live-test speed [m/s]",
        tuning.max_test_velocity_m_s,
    )
    max_tcp_deviation = _prompt_float(
        "Maximum TCP deviation from test start [m]",
        tuning.max_tcp_deviation_m,
    )
    excitation_amplitude = _prompt_float(
        "Sine test amplitude [m]",
        tuning.excitation_amplitude_m,
    )
    excitation_frequency = _prompt_float(
        "Sine test frequency [Hz]",
        tuning.excitation_frequency_hz,
    )

    updated_config = replace(
        config,
        mass_kg=mass,
        stiffness_n_m=stiffness,
        damping_n_s_m=damping,
        force_deadband_n=deadband,
        force_axis_sign=sign,
        max_displacement_m=displacement,
        max_velocity_m_s=max_velocity,
        filter_cutoff_hz=cutoff,
        force_limit_n=force_limit,
        force_release_n=force_release,
        tare_max_force_n=tare_max,
    )
    updated = replace(
        profile,
        name=name,
        description=description,
        impedance=updated_config,
        runtime=replace(runtime, wrench_frame=frame),
        tuning=replace(
            tuning,
            max_test_velocity_m_s=max_test_velocity,
            max_tcp_deviation_m=max_tcp_deviation,
            excitation_amplitude_m=excitation_amplitude,
            excitation_frequency_hz=excitation_frequency,
        ),
    )
    print("\nValidated profile:\n")
    print_profile_summary(updated)
    answer = input(f"\nSave to {profile_path}? [y/N]: ").strip().lower()
    if answer != "y":
        print("Not saved.")
        return 1
    save_impedance_profile(profile_path, updated)
    print(f"Saved: {profile_path}")
    return 0


def command_validate(_args: argparse.Namespace, profile_path: Path) -> int:
    profile = load_profile_for_command(profile_path)
    print_profile_summary(profile, profile_path)
    print("Profile is valid.")
    return 0


def make_run_directory(root: Path, mode: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate = root / f"{stamp}_{mode}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{mode}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def _empty_output(state: str, config_profile: ImpedanceProfile) -> CartesianImpedanceOutput:
    controller = CartesianImpedanceController(config_profile.impedance, enabled=False)
    output = controller.update(ZERO_VECTOR, 1.0 / config_profile.tuning.loop_hz)
    return replace(output, state=state)


def build_sample(
    *,
    elapsed_s: float,
    dt_s: float,
    mode: str,
    output: CartesianImpedanceOutput,
    snapshot: RobotSnapshot,
    initial_snapshot: RobotSnapshot,
    command_velocity_m_s: Vector3 = ZERO_VECTOR,
    nominal_velocity_m_s: Vector3 = ZERO_VECTOR,
    desired_offset_m: Vector3 = ZERO_VECTOR,
) -> dict[str, Any]:
    tcp_offset = vector_subtract(snapshot.position_m, initial_snapshot.position_m)
    record: dict[str, Any] = {
        "time_s": elapsed_s,
        "dt_s": dt_s,
        "mode": mode,
        "state": output.state,
        "wrench_available": output.wrench_available,
        "force_limit_active": output.force_limit_active,
        "force_norm_n": output.force_norm_n,
        "tare_samples_collected": output.tare_samples_collected,
        "tare_samples_required": output.tare_samples_required,
        "orientation_drift_deg": orientation_distance_degrees(
            snapshot.orientation_deg,
            initial_snapshot.orientation_deg,
        ),
    }
    vector_fields = {
        "raw_force_n": output.raw_force_n,
        "force_bias_n": output.force_bias_n,
        "filtered_force_n": output.filtered_force_n,
        "effective_force_n": output.effective_force_n,
        "compliance_velocity_m_s": output.compliance_velocity_m_s,
        "compliance_displacement_m": output.compliance_displacement_m,
        "command_velocity_m_s": command_velocity_m_s,
        "nominal_velocity_m_s": nominal_velocity_m_s,
        "desired_offset_m": desired_offset_m,
        "tcp_position_m": snapshot.position_m,
        "tcp_offset_m": tcp_offset,
        "tool_velocity_m_s": snapshot.linear_velocity_m_s,
        "raw_torque_nm": snapshot.raw_torque_nm,
    }
    for name, values in vector_fields.items():
        for axis, value in zip("xyz", values):
            record[f"{name}_{axis}"] = float(value)
    for index in range(7):
        position = (
            snapshot.joint_positions_deg[index]
            if index < len(snapshot.joint_positions_deg)
            else ""
        )
        initial = (
            initial_snapshot.joint_positions_deg[index]
            if index < len(initial_snapshot.joint_positions_deg)
            else None
        )
        record[f"joint_{index + 1}_position_deg"] = position
        record[f"joint_{index + 1}_delta_deg"] = (
            wrapped_angle_difference_degrees(float(position), float(initial))
            if position != "" and initial is not None
            else ""
        )
        record[f"joint_{index + 1}_velocity_deg_s"] = (
            snapshot.joint_velocities_deg_s[index]
            if index < len(snapshot.joint_velocities_deg_s)
            else ""
        )
        record[f"joint_{index + 1}_torque_nm"] = (
            snapshot.joint_torques_nm[index]
            if index < len(snapshot.joint_torques_nm)
            else ""
        )
    return record


def write_csv(path: Path, samples: Sequence[dict[str, Any]]) -> None:
    if not samples:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)


def percentile(values: Sequence[float], probability: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = clamp(probability, 0.0, 1.0) * (len(ordered) - 1)
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def summarize_samples(
    samples: Sequence[dict[str, Any]],
    *,
    mode: str,
    stop_reason: str,
    completed: bool,
) -> dict[str, Any]:
    if not samples:
        return {
            "mode": mode,
            "completed": completed,
            "stop_reason": stop_reason,
            "sample_count": 0,
        }
    positive_dt = [float(sample["dt_s"]) for sample in samples if sample["dt_s"] > 0]
    max_force = max(float(sample["force_norm_n"]) for sample in samples)
    max_tcp = max(
        vector_norm(
            (
                sample["tcp_offset_m_x"],
                sample["tcp_offset_m_y"],
                sample["tcp_offset_m_z"],
            )
        )
        for sample in samples
    )
    max_orientation = max(float(sample["orientation_drift_deg"]) for sample in samples)
    max_compliance_speed = max(
        vector_norm(
            (
                sample["compliance_velocity_m_s_x"],
                sample["compliance_velocity_m_s_y"],
                sample["compliance_velocity_m_s_z"],
            )
        )
        for sample in samples
    )
    max_compliance_displacement = max(
        vector_norm(
            (
                sample["compliance_displacement_m_x"],
                sample["compliance_displacement_m_y"],
                sample["compliance_displacement_m_z"],
            )
        )
        for sample in samples
    )
    final = samples[-1]
    final_offset = vector_norm(
        (
            final["tcp_offset_m_x"],
            final["tcp_offset_m_y"],
            final["tcp_offset_m_z"],
        )
    )
    active_samples = [
        sample
        for sample in samples
        if sample["state"] not in ("calibrating", "tare_force_too_high")
    ]
    recommended_deadband = None
    if mode == "monitor":
        recommended_deadband = []
        for axis in "xyz":
            residuals = [
                abs(
                    float(sample[f"raw_force_n_{axis}"])
                    - float(sample[f"force_bias_n_{axis}"])
                )
                for sample in active_samples
            ]
            recommended_deadband.append(
                max(0.25, 1.25 * percentile(residuals, 0.99))
            )

    max_joint_delta = []
    max_joint_torque = []
    for index in range(7):
        values = [
            abs(float(sample[f"joint_{index + 1}_delta_deg"]))
            for sample in samples
            if sample[f"joint_{index + 1}_delta_deg"] != ""
        ]
        max_joint_delta.append(max(values, default=0.0))
        torques = [
            abs(float(sample[f"joint_{index + 1}_torque_nm"]))
            for sample in samples
            if sample[f"joint_{index + 1}_torque_nm"] != ""
        ]
        max_joint_torque.append(max(torques, default=0.0))

    return {
        "mode": mode,
        "completed": completed,
        "stop_reason": stop_reason,
        "sample_count": len(samples),
        "elapsed_s": float(samples[-1]["time_s"]),
        "tare_completed": (
            int(samples[-1]["tare_samples_collected"])
            >= int(samples[-1]["tare_samples_required"])
        ),
        "median_loop_hz": (
            1.0 / statistics.median(positive_dt) if positive_dt else 0.0
        ),
        "maximum_loop_gap_s": max(positive_dt, default=0.0),
        "maximum_force_norm_n": max_force,
        "maximum_tcp_offset_m": max_tcp,
        "final_tcp_offset_m": final_offset,
        "maximum_orientation_drift_deg": max_orientation,
        "maximum_compliance_speed_m_s": max_compliance_speed,
        "maximum_virtual_displacement_m": max_compliance_displacement,
        "recommended_deadband_from_this_run_n": recommended_deadband,
        "maximum_joint_delta_deg": max_joint_delta,
        "maximum_measured_joint_torque_nm": max_joint_torque,
    }


def _matplotlib_pyplot():
    cache_dir = Path(tempfile.gettempdir()) / "kinova-tele-matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plotting requires matplotlib from impedance_tuning/environment.yml."
        ) from exc
    return plt


def render_plot(
    axes: Sequence[Any],
    samples: Sequence[dict[str, Any]],
    *,
    title: str,
    window_s: Optional[float] = None,
) -> None:
    if not samples:
        return
    displayed = samples
    if window_s is not None:
        threshold = float(samples[-1]["time_s"]) - window_s
        displayed = [sample for sample in samples if float(sample["time_s"]) >= threshold]
    times = [float(sample["time_s"]) for sample in displayed]
    for axis in axes:
        axis.clear()

    colors = {"x": "tab:red", "y": "tab:green", "z": "tab:blue"}
    for coordinate in "xyz":
        axes[0].plot(
            times,
            [sample[f"raw_force_n_{coordinate}"] for sample in displayed],
            color=colors[coordinate],
            alpha=0.35,
            label=f"raw {coordinate}",
        )
        axes[0].plot(
            times,
            [sample[f"effective_force_n_{coordinate}"] for sample in displayed],
            color=colors[coordinate],
            label=f"effective {coordinate}",
        )
        axes[1].plot(
            times,
            [sample[f"compliance_velocity_m_s_{coordinate}"] for sample in displayed],
            color=colors[coordinate],
            label=coordinate,
        )
        axes[2].plot(
            times,
            [sample[f"compliance_displacement_m_{coordinate}"] for sample in displayed],
            color=colors[coordinate],
            label=coordinate,
        )
        axes[3].plot(
            times,
            [sample[f"tcp_offset_m_{coordinate}"] for sample in displayed],
            color=colors[coordinate],
            label=coordinate,
        )
        axes[4].plot(
            times,
            [sample[f"command_velocity_m_s_{coordinate}"] for sample in displayed],
            color=colors[coordinate],
            label=coordinate,
        )
    for index in range(7):
        values = [sample[f"joint_{index + 1}_delta_deg"] for sample in displayed]
        if any(value != "" for value in values):
            axes[5].plot(
                times,
                [float(value) if value != "" else float("nan") for value in values],
                label=f"J{index + 1}",
            )

    panel_settings = (
        ("External force", "N"),
        ("Compliance velocity", "m/s"),
        ("Virtual displacement", "m"),
        ("Measured TCP offset", "m"),
        ("Commanded Cartesian velocity", "m/s"),
        ("Joint displacement", "deg"),
    )
    for axis, (panel_title, units) in zip(axes, panel_settings):
        axis.set_title(panel_title)
        axis.set_ylabel(units)
        axis.grid(True, alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(
                handles,
                labels,
                loc="upper right",
                fontsize="x-small",
                ncol=2,
            )
    axes[-1].set_xlabel("time [s]")
    axes[-2].set_xlabel("time [s]")
    axes[0].figure.suptitle(title)
    axes[0].figure.tight_layout(rect=(0, 0, 1, 0.97))


def _live_plot_process(
    sample_queue: Any,
    window_s: float,
    title: str,
) -> None:
    """Own the GUI event loop away from the robot-control process."""
    try:
        plt = _matplotlib_pyplot()
        backend = plt.get_backend().lower()
        if "agg" in backend:
            return
        plt.ion()
        figure, grid = plt.subplots(3, 2, figsize=(13, 9))
        axes = list(grid.flat)
        plt.show(block=False)
        samples: list[dict[str, Any]] = []
        last_draw = 0.0
        running = True
        while running and plt.fignum_exists(figure.number):
            try:
                item = sample_queue.get(timeout=0.05)
                if item is None:
                    break
                samples.append(item)
                while True:
                    item = sample_queue.get_nowait()
                    if item is None:
                        running = False
                        break
                    samples.append(item)
            except queue.Empty:
                pass

            if samples:
                threshold = float(samples[-1]["time_s"]) - window_s
                samples = [
                    sample
                    for sample in samples
                    if float(sample["time_s"]) >= threshold
                ]
            now = monotonic()
            if samples and now - last_draw >= 0.5:
                last_draw = now
                render_plot(axes, samples, title=title, window_s=window_s)
                figure.canvas.draw_idle()
                figure.canvas.flush_events()
            plt.pause(0.001)
        plt.close(figure)
    except Exception as exc:
        print(f"Live plotting stopped: {exc}", file=sys.stderr)


class LivePlot:
    """Non-blocking publisher for a dedicated Matplotlib process."""

    def __init__(self, enabled: bool, window_s: float, title: str) -> None:
        self.enabled = False
        self.process = None
        self.queue = None
        if not enabled:
            return
        try:
            context = multiprocessing.get_context("spawn")
            self.queue = context.Queue(maxsize=256)
            self.process = context.Process(
                target=_live_plot_process,
                args=(self.queue, window_s, title),
                name="kinova-impedance-live-plot",
                daemon=True,
            )
            self.process.start()
            self.enabled = True
        except Exception as exc:
            print(f"Live plotting unavailable: {exc}", file=sys.stderr)

    def update(self, samples: Sequence[dict[str, Any]], _now: float) -> None:
        if not self.enabled or not samples:
            return
        try:
            self.queue.put_nowait(samples[-1])
        except queue.Full:
            # Plotting is diagnostic. Dropping a graph sample is safer than
            # delaying the robot command loop.
            pass

    def close(self) -> None:
        if not self.enabled:
            return
        try:
            self.queue.put(None, timeout=0.5)
        except queue.Full:
            pass
        self.process.join(timeout=3.0)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=1.0)
        try:
            self.queue.close()
            self.queue.join_thread()
        except Exception:
            pass
        self.enabled = False


def save_static_plot(
    path: Path,
    samples: Sequence[dict[str, Any]],
    title: str,
) -> None:
    if not samples:
        return
    plt = _matplotlib_pyplot()
    figure, grid = plt.subplots(3, 2, figsize=(13, 9))
    render_plot(list(grid.flat), samples, title=title)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_run_artifacts(
    run_directory: Path,
    profile: ImpedanceProfile,
    samples: Sequence[dict[str, Any]],
    summary: dict[str, Any],
) -> None:
    save_impedance_profile(run_directory / "profile_used.json", profile)
    write_csv(run_directory / "telemetry.csv", samples)
    with (run_directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    try:
        save_static_plot(
            run_directory / "plot.png",
            samples,
            f"Kinova Gen3 impedance tuning: {summary['mode']}",
        )
    except Exception as exc:
        print(f"Could not save plot: {exc}", file=sys.stderr)


def require_motion_confirmation(args: argparse.Namespace, profile: ImpedanceProfile) -> None:
    if not args.enable_motion:
        raise RuntimeError(
            f"live mode {args.mode!r} can move the robot; add --enable-motion "
            "after reviewing the profile and safety envelope"
        )
    if not sys.stdin.isatty():
        raise RuntimeError("motion confirmation requires an interactive terminal")
    print("\nLIVE ROBOT MOTION REQUESTED")
    print(f"Mode: {args.mode}")
    print(f"Robot: {args.robot_ip}:{args.robot_port}")
    print(f"Profile: {args.profile}")
    print(f"Max test speed: {profile.tuning.max_test_velocity_m_s:.4f} m/s")
    print(f"Max TCP deviation: {profile.tuning.max_tcp_deviation_m:.4f} m")
    print(f"Force stop threshold: {profile.impedance.force_limit_n:.1f} N")
    print(
        "Secure the base, clear the workspace, configure payload/protection "
        "zones, and keep the emergency stop within reach."
    )
    answer = input(f"Type {MOTION_CONFIRMATION!r} to continue: ").strip()
    if answer != MOTION_CONFIRMATION:
        raise RuntimeError("motion confirmation did not match; nothing was started")


def smooth_sine_target(
    elapsed_s: float,
    amplitude_m: float,
    frequency_hz: float,
) -> tuple[float, float]:
    """Return a startup-ramped sinusoidal position and analytic velocity."""
    ramp_time_s = max(1.0, 1.0 / frequency_hz)
    if elapsed_s < ramp_time_s:
        ramp = 0.5 * (1.0 - cos(pi * elapsed_s / ramp_time_s))
        ramp_rate = 0.5 * pi / ramp_time_s * sin(pi * elapsed_s / ramp_time_s)
    else:
        ramp = 1.0
        ramp_rate = 0.0
    phase = 2.0 * pi * frequency_hz * elapsed_s
    position = ramp * amplitude_m * sin(phase)
    velocity = amplitude_m * (
        ramp_rate * sin(phase)
        + ramp * 2.0 * pi * frequency_hz * cos(phase)
    )
    return position, velocity


def cap_vector_per_axis(values: Sequence[float], limit: float) -> Vector3:
    return tuple(clamp(float(value), -limit, limit) for value in values)  # type: ignore[return-value]


def command_live(args: argparse.Namespace, profile_path: Path) -> int:
    profile = load_profile_for_command(profile_path)
    if not 1 <= args.robot_port <= 65535:
        raise ValueError("robot-port must be between 1 and 65535")
    if args.mode != "monitor":
        require_motion_confirmation(args, profile)

    username = os.environ.get("KINOVA_USERNAME")
    password = os.environ.get("KINOVA_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Set KINOVA_USERNAME and KINOVA_PASSWORD before a live test"
        )

    loop_interval_s = 1.0 / profile.tuning.loop_hz
    run_directory = make_run_directory(args.output_dir, args.mode)
    plot = LivePlot(
        enabled=not args.no_live_plot,
        window_s=profile.tuning.plot_window_s,
        title=f"Gen3 tuning: {args.mode}",
    )
    robot = TuningRobot(
        args.robot_ip,
        args.robot_port,
        username,
        password,
    )
    controller = CartesianImpedanceController(profile.impedance)
    samples: list[dict[str, Any]] = []
    stop_reason = "stopped before the live loop started"
    completed = False
    native_nullspace_enabled = False
    controller_motion_started_at: Optional[float] = None
    next_tick = monotonic()

    print(f"Connecting to full Gen3 at {args.robot_ip}:{args.robot_port}...")
    try:
        robot.connect()
        initial_snapshot = robot.read_snapshot()
        if not initial_snapshot.joint_positions_deg:
            raise RuntimeError("Kortex returned no actuator feedback")
        if args.mode == "nullspace" and len(initial_snapshot.joint_positions_deg) != 7:
            raise RuntimeError("native null-space admittance requires a 7-DoF Gen3")

        print(
            f"Connected. Captured TCP at {initial_snapshot.position_m}; "
            f"{len(initial_snapshot.joint_positions_deg)} actuators detected."
        )
        if args.mode == "monitor":
            print("MONITOR MODE: no motion commands will be sent.")
        else:
            print("Keep the robot unloaded and stationary while the wrench is tared.")
        print("Press Ctrl+C to stop.\n")

        start = monotonic()
        next_tick = start
        previous = start
        while True:
            tick = monotonic()
            elapsed = tick - start
            dt_s = tick - previous
            previous = tick
            if samples and dt_s > profile.tuning.max_loop_gap_s:
                stop_reason = (
                    f"loop gap {dt_s:.3f}s exceeded "
                    f"{profile.tuning.max_loop_gap_s:.3f}s"
                )
                break

            snapshot = robot.read_snapshot()
            transform = (
                rotate_tool_force_to_base(snapshot.orientation_deg)
                if profile.runtime.wrench_frame == "tool"
                else None
            )
            tare_allowed = (
                not native_nullspace_enabled
                and vector_norm(snapshot.linear_velocity_m_s)
                <= profile.runtime.tare_max_tool_speed_m_s
                and vector_norm(snapshot.angular_velocity_deg_s)
                <= profile.runtime.tare_max_tool_angular_speed_deg_s
            )
            allow_controller_motion = args.mode in ("hold", "sine")
            output = controller.update(
                snapshot.raw_force_n,
                dt_s,
                wrench_available=snapshot.wrench_available,
                allow_tare=tare_allowed,
                allow_motion=allow_controller_motion,
                allow_force_limit_release=True,
                force_transform=transform,
            )

            if not output.wrench_available and args.mode != "monitor":
                stop_reason = "external wrench feedback became unavailable"
                break
            if output.force_limit_active:
                stop_reason = (
                    f"force limit reached ({output.force_norm_n:.2f} N)"
                )
                break

            tcp_offset = vector_subtract(snapshot.position_m, initial_snapshot.position_m)
            tcp_deviation = vector_norm(tcp_offset)
            orientation_drift = orientation_distance_degrees(
                snapshot.orientation_deg,
                initial_snapshot.orientation_deg,
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
                    f"measured TCP deviation {tcp_deviation:.4f}m exceeded "
                    f"{tcp_limit:.4f}m"
                )
                break
            if orientation_drift > orientation_limit:
                stop_reason = (
                    f"orientation drift {orientation_drift:.2f}deg exceeded "
                    f"{orientation_limit:.2f}deg"
                )
                break

            joint_deltas = tuple(
                abs(
                    wrapped_angle_difference_degrees(
                        snapshot.joint_positions_deg[index],
                        initial_snapshot.joint_positions_deg[index],
                    )
                )
                for index in range(len(snapshot.joint_positions_deg))
            )
            if (
                args.mode == "nullspace"
                and max(joint_deltas, default=0.0)
                > profile.tuning.nullspace_max_joint_delta_deg
            ):
                stop_reason = "null-space joint displacement safety bound reached"
                break

            command_velocity = ZERO_VECTOR
            nominal_velocity = ZERO_VECTOR
            desired_offset = ZERO_VECTOR

            tare_complete = (
                output.tare_samples_collected >= output.tare_samples_required
            )
            if (
                tare_complete
                and controller_motion_started_at is None
                and args.mode in ("hold", "sine")
            ):
                controller_motion_started_at = tick
                print(f"Tare complete. {args.mode} test is active.")
            command_timeout_s = max(2.0 * loop_interval_s, 0.05)
            if args.mode == "hold":
                command_velocity = cap_vector_per_axis(
                    output.compliance_velocity_m_s,
                    min(
                        profile.tuning.max_test_velocity_m_s,
                        profile.impedance.max_velocity_m_s,
                    ),
                )
                robot.send_twist(
                    command_velocity,
                    timeout_s=command_timeout_s,
                )
            elif args.mode == "sine":
                axis_index = AXIS_INDEX[args.axis]
                test_elapsed = (
                    tick - controller_motion_started_at
                    if controller_motion_started_at is not None
                    else 0.0
                )
                desired_position, feedforward_velocity = smooth_sine_target(
                    test_elapsed,
                    profile.tuning.excitation_amplitude_m,
                    profile.tuning.excitation_frequency_hz,
                )
                desired_values = [0.0, 0.0, 0.0]
                nominal_values = [0.0, 0.0, 0.0]
                desired_values[axis_index] = desired_position
                position_error = desired_position - tcp_offset[axis_index]
                nominal_values[axis_index] = (
                    feedforward_velocity
                    + profile.tuning.excitation_position_gain_s_inv * position_error
                )
                desired_offset = tuple(desired_values)  # type: ignore[assignment]
                nominal_velocity = cap_vector_per_axis(
                    nominal_values,
                    profile.tuning.max_test_velocity_m_s,
                )
                combined = tuple(
                    nominal_velocity[index]
                    + output.compliance_velocity_m_s[index]
                    for index in range(3)
                )
                command_velocity = cap_vector_per_axis(
                    combined,
                    profile.tuning.max_test_velocity_m_s,
                )
                if not tare_complete:
                    command_velocity = ZERO_VECTOR
                robot.send_twist(
                    command_velocity,
                    timeout_s=command_timeout_s,
                )
            elif args.mode == "nullspace":
                if tare_complete and not native_nullspace_enabled:
                    print(
                        "Tare complete. Enabling Kinova native NULL_SPACE "
                        "admittance. Manipulate the arm links, not the tool."
                    )
                    robot.set_native_nullspace_admittance(True)
                    native_nullspace_enabled = True

            samples.append(
                build_sample(
                    elapsed_s=elapsed,
                    dt_s=dt_s,
                    mode=args.mode,
                    output=output,
                    snapshot=snapshot,
                    initial_snapshot=initial_snapshot,
                    command_velocity_m_s=command_velocity,
                    nominal_velocity_m_s=nominal_velocity,
                    desired_offset_m=desired_offset,
                )
            )
            plot.update(samples, tick)

            if len(samples) % max(1, int(profile.tuning.loop_hz)) == 0:
                print(
                    f"t={elapsed:6.2f}s state={output.state:20s} "
                    f"F={output.force_norm_n:6.2f}N "
                    f"TCP={tcp_deviation * 1000:6.1f}mm",
                    end="\r",
                    flush=True,
                )

            next_tick += loop_interval_s
            remaining = next_tick - monotonic()
            if remaining > 0.0:
                sleep(remaining)
            elif remaining < -profile.tuning.max_loop_gap_s:
                next_tick = monotonic()
    except KeyboardInterrupt:
        stop_reason = "operator pressed Ctrl+C"
        completed = True
    except Exception as exc:
        stop_reason = f"error: {exc}"
        print(stop_reason, file=sys.stderr)
    finally:
        print("\nStopping robot and disabling test modes...")
        robot.close()
        plot.close()
        summary = summarize_samples(
            samples,
            mode=args.mode,
            stop_reason=stop_reason,
            completed=completed,
        )
        summary["profile_path"] = str(profile_path)
        summary["robot_ip"] = args.robot_ip
        summary["utc_finished"] = datetime.now(timezone.utc).isoformat()
        save_run_artifacts(run_directory, profile, samples, summary)
        print(json.dumps(summary, indent=2))
        print(f"Artifacts: {run_directory}")
    return 0 if completed else 2


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune and perform safety-bounded live tests of the Kinova Gen3 "
            "Cartesian impedance profile. No VR headset is used."
        )
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_FILENAME,
        help=f"JSON profile path (default: {DEFAULT_PROFILE_FILENAME})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="write a starter JSON profile")
    init_parser.add_argument(
        "--preset",
        choices=("conservative", "production"),
        default="conservative",
    )
    init_parser.add_argument("--force", action="store_true", help="overwrite an existing profile")

    subparsers.add_parser("wizard", help="interactively edit and validate a profile")
    subparsers.add_parser("validate", help="validate and display a profile")

    live_parser = subparsers.add_parser(
        "live",
        help="run a monitored hardware test; motion modes require confirmation",
    )
    live_parser.add_argument(
        "--mode",
        choices=("monitor", "hold", "sine", "nullspace"),
        required=True,
    )
    live_parser.add_argument("--axis", choices=tuple(AXIS_INDEX), default="x")
    live_parser.add_argument("--robot-ip", default="192.168.1.10")
    live_parser.add_argument("--robot-port", type=int, default=10000)
    live_parser.add_argument(
        "--enable-motion",
        action="store_true",
        help="required for hold, sine, and nullspace modes",
    )
    live_parser.add_argument(
        "--no-live-plot",
        action="store_true",
        help="log normally and save a final PNG without opening a live window",
    )
    live_parser.add_argument(
        "--output-dir",
        type=Path,
        default=APP_DIR / "tuning_runs",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = create_parser()
    args = parser.parse_args(argv)
    profile_path = resolve_profile_path(args.profile)
    try:
        if args.command == "init":
            return command_init(args, profile_path)
        if args.command == "wizard":
            return command_wizard(args, profile_path)
        if args.command == "validate":
            return command_validate(args, profile_path)
        if args.command == "live":
            return command_live(args, profile_path)
        parser.error(f"unknown command: {args.command}")
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
