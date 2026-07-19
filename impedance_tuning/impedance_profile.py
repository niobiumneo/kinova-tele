"""Validated JSON profiles shared by the tuner and teleoperation server."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from math import isfinite, sqrt
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, Optional, Tuple

from .impedance_controller import CartesianImpedanceConfig


SCHEMA_VERSION = 1
DEFAULT_PROFILE_FILENAME = "impedance_profile.json"
Vector3 = Tuple[float, float, float]


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, not a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be true or false")
    return value


def _vector3(value: Any, name: str) -> Vector3:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a JSON array with three numbers")
    try:
        result = tuple(_finite_float(item, name) for item in value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a JSON array with three numbers") from exc
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return result  # type: ignore[return-value]


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    name: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"unknown {name} field(s): {', '.join(unknown)}")


@dataclass(frozen=True)
class ImpedanceRuntimeConfig:
    """Non-dynamic settings used around the Cartesian controller."""

    enabled: bool = True
    wrench_frame: str = "base"
    tare_max_tool_speed_m_s: float = 0.01
    tare_max_tool_angular_speed_deg_s: float = 1.0
    tare_after_gripper_delay_s: float = 0.25

    def __post_init__(self) -> None:
        object.__setattr__(self, "enabled", _strict_bool(self.enabled, "enabled"))
        frame = str(self.wrench_frame).strip().lower()
        if frame not in ("base", "tool"):
            raise ValueError("wrench_frame must be 'base' or 'tool'")
        object.__setattr__(self, "wrench_frame", frame)

        for name in (
            "tare_max_tool_speed_m_s",
            "tare_max_tool_angular_speed_deg_s",
            "tare_after_gripper_delay_s",
        ):
            value = _finite_float(getattr(self, name), name)
            if value < 0.0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class TuningTestConfig:
    """Safety envelope and excitation defaults used only by the tuner."""

    loop_hz: float = 40.0
    plot_window_s: float = 15.0
    max_loop_gap_s: float = 0.15
    max_tcp_deviation_m: float = 0.04
    max_orientation_deviation_deg: float = 8.0
    max_test_velocity_m_s: float = 0.015
    excitation_amplitude_m: float = 0.01
    excitation_frequency_hz: float = 0.20
    excitation_position_gain_s_inv: float = 1.5
    nullspace_max_tcp_drift_m: float = 0.02
    nullspace_max_orientation_drift_deg: float = 5.0
    nullspace_max_joint_delta_deg: float = 30.0

    def __post_init__(self) -> None:
        positive = (
            "loop_hz",
            "plot_window_s",
            "max_loop_gap_s",
            "max_tcp_deviation_m",
            "max_orientation_deviation_deg",
            "max_test_velocity_m_s",
            "excitation_amplitude_m",
            "excitation_frequency_hz",
            "excitation_position_gain_s_inv",
            "nullspace_max_tcp_drift_m",
            "nullspace_max_orientation_drift_deg",
            "nullspace_max_joint_delta_deg",
        )
        for name in positive:
            value = _finite_float(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)

        if self.loop_hz > 100.0:
            raise ValueError("loop_hz cannot exceed 100 Hz for this TCP tuner")
        if self.max_loop_gap_s < 1.0 / self.loop_hz:
            raise ValueError("max_loop_gap_s must be at least one loop interval")
        if self.excitation_amplitude_m >= self.max_tcp_deviation_m:
            raise ValueError(
                "excitation_amplitude_m must be below max_tcp_deviation_m"
            )
        peak_speed = (
            2.0
            * 3.141592653589793
            * self.excitation_frequency_hz
            * self.excitation_amplitude_m
        )
        if peak_speed > self.max_test_velocity_m_s:
            raise ValueError(
                "sine excitation peak speed exceeds max_test_velocity_m_s"
            )


@dataclass(frozen=True)
class ImpedanceProfile:
    """Complete, versioned tuning profile."""

    schema_version: int = SCHEMA_VERSION
    name: str = "default"
    description: str = ""
    impedance: CartesianImpedanceConfig = CartesianImpedanceConfig()
    runtime: ImpedanceRuntimeConfig = ImpedanceRuntimeConfig()
    tuning: TuningTestConfig = TuningTestConfig()

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != SCHEMA_VERSION
        ):
            raise ValueError(
                f"unsupported impedance profile schema_version {self.schema_version!r}; "
                f"expected {SCHEMA_VERSION}"
            )
        object.__setattr__(self, "schema_version", SCHEMA_VERSION)
        name = str(self.name).strip()
        if not name:
            raise ValueError("profile name cannot be empty")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "description", str(self.description).strip())
        if not isinstance(self.impedance, CartesianImpedanceConfig):
            raise ValueError("impedance must be a CartesianImpedanceConfig")
        if not isinstance(self.runtime, ImpedanceRuntimeConfig):
            raise ValueError("runtime must be an ImpedanceRuntimeConfig")
        if not isinstance(self.tuning, TuningTestConfig):
            raise ValueError("tuning must be a TuningTestConfig")


IMPEDANCE_FIELDS = set(CartesianImpedanceConfig.__dataclass_fields__)
RUNTIME_FIELDS = set(ImpedanceRuntimeConfig.__dataclass_fields__)
TUNING_FIELDS = set(TuningTestConfig.__dataclass_fields__)
PROFILE_FIELDS = {
    "schema_version",
    "name",
    "description",
    "impedance",
    "runtime",
    "tuning",
}
VECTOR_IMPEDANCE_FIELDS = {
    "mass_kg",
    "stiffness_n_m",
    "damping_n_s_m",
    "force_deadband_n",
    "force_axis_sign",
    "max_displacement_m",
}
FLOAT_IMPEDANCE_FIELDS = IMPEDANCE_FIELDS - VECTOR_IMPEDANCE_FIELDS - {
    "tare_samples"
}


def production_default_profile() -> ImpedanceProfile:
    """Return the exact defaults historically used by main.py."""
    return ImpedanceProfile(
        name="production-default",
        description="Built-in teleoperation defaults used when no profile exists.",
    )


def conservative_tuning_profile() -> ImpedanceProfile:
    """Return a deliberately motion-limited profile for first hardware tests."""
    return ImpedanceProfile(
        name="gen3-conservative",
        description=(
            "Low-speed, one-centimeter Cartesian compliance profile for initial "
            "full Gen3 commissioning."
        ),
        impedance=CartesianImpedanceConfig(
            mass_kg=(8.0, 8.0, 8.0),
            stiffness_n_m=(300.0, 300.0, 350.0),
            damping_n_s_m=(100.0, 100.0, 110.0),
            force_deadband_n=(2.0, 2.0, 2.0),
            force_axis_sign=(1.0, 1.0, 1.0),
            max_displacement_m=(0.01, 0.01, 0.01),
            max_velocity_m_s=0.01,
            filter_cutoff_hz=8.0,
            force_limit_n=15.0,
            force_release_n=8.0,
            contact_haptic_start_n=2.0,
            contact_haptic_full_scale_n=12.0,
            tare_samples=30,
            tare_max_force_n=6.0,
            min_dt_s=0.001,
            max_dt_s=0.05,
        ),
    )


def _dataclass_values(instance: Any) -> dict[str, Any]:
    values = asdict(instance)
    for key, value in tuple(values.items()):
        if isinstance(value, tuple):
            values[key] = list(value)
    return values


def profile_to_dict(profile: ImpedanceProfile) -> dict[str, Any]:
    return {
        "schema_version": profile.schema_version,
        "name": profile.name,
        "description": profile.description,
        "impedance": _dataclass_values(profile.impedance),
        "runtime": _dataclass_values(profile.runtime),
        "tuning": _dataclass_values(profile.tuning),
    }


def profile_from_dict(
    data: Mapping[str, Any],
    *,
    defaults: Optional[ImpedanceProfile] = None,
) -> ImpedanceProfile:
    """Validate a decoded JSON object and fill omitted values from defaults."""
    data = _mapping(data, "profile")
    _reject_unknown_keys(data, PROFILE_FIELDS, "profile")
    base = defaults or production_default_profile()

    schema_version = data.get("schema_version", base.schema_version)
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("schema_version must be an integer")

    impedance_data = _mapping(data.get("impedance", {}), "impedance")
    _reject_unknown_keys(impedance_data, IMPEDANCE_FIELDS, "impedance")
    impedance_values = _dataclass_values(base.impedance)
    for key, value in impedance_data.items():
        if key in VECTOR_IMPEDANCE_FIELDS:
            impedance_values[key] = _vector3(value, f"impedance.{key}")
        elif key in FLOAT_IMPEDANCE_FIELDS:
            impedance_values[key] = _finite_float(value, f"impedance.{key}")
        else:
            impedance_values[key] = value
    if "tare_samples" in impedance_values:
        tare_samples = impedance_values["tare_samples"]
        if isinstance(tare_samples, bool) or not isinstance(tare_samples, int):
            raise ValueError("impedance.tare_samples must be an integer")
        impedance_values["tare_samples"] = tare_samples
    impedance = CartesianImpedanceConfig(**impedance_values)

    runtime_data = _mapping(data.get("runtime", {}), "runtime")
    _reject_unknown_keys(runtime_data, RUNTIME_FIELDS, "runtime")
    runtime_values = _dataclass_values(base.runtime)
    runtime_values.update(runtime_data)
    runtime = ImpedanceRuntimeConfig(**runtime_values)

    tuning_data = _mapping(data.get("tuning", {}), "tuning")
    _reject_unknown_keys(tuning_data, TUNING_FIELDS, "tuning")
    tuning_values = _dataclass_values(base.tuning)
    tuning_values.update(tuning_data)
    tuning = TuningTestConfig(**tuning_values)

    return ImpedanceProfile(
        schema_version=schema_version,
        name=data.get("name", base.name),
        description=data.get("description", base.description),
        impedance=impedance,
        runtime=runtime,
        tuning=tuning,
    )


def load_impedance_profile(path: os.PathLike[str] | str) -> ImpedanceProfile:
    profile_path = Path(path).expanduser()
    try:
        with profile_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"invalid JSON in impedance profile {profile_path}: {exc}"
        ) from exc
    return profile_from_dict(_mapping(data, "profile"))


def load_optional_impedance_profile(
    path: os.PathLike[str] | str,
) -> tuple[ImpedanceProfile, bool]:
    profile_path = Path(path).expanduser()
    if not profile_path.exists():
        return production_default_profile(), False
    return load_impedance_profile(profile_path), True


def save_impedance_profile(
    path: os.PathLike[str] | str,
    profile: ImpedanceProfile,
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically save a profile so main.py never reads a partial file."""
    profile_path = Path(path).expanduser().resolve()
    if profile_path.exists() and not overwrite:
        raise FileExistsError(f"profile already exists: {profile_path}")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(profile_to_dict(profile), indent=2, sort_keys=False) + "\n"

    temporary_name: Optional[str] = None
    try:
        with NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=profile_path.parent,
            prefix=f".{profile_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, profile_path)
    finally:
        if temporary_name and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return profile_path


def with_impedance_updates(
    profile: ImpedanceProfile,
    **updates: Any,
) -> ImpedanceProfile:
    """Return a validated copy with selected impedance fields replaced."""
    unknown = sorted(set(updates) - IMPEDANCE_FIELDS)
    if unknown:
        raise ValueError(f"unknown impedance field(s): {', '.join(unknown)}")
    for key in VECTOR_IMPEDANCE_FIELDS & set(updates):
        updates[key] = _vector3(updates[key], key)
    return replace(profile, impedance=replace(profile.impedance, **updates))


def critical_damping(config: CartesianImpedanceConfig) -> Vector3:
    return tuple(
        2.0 * sqrt(config.mass_kg[axis] * config.stiffness_n_m[axis])
        for axis in range(3)
    )  # type: ignore[return-value]


def damping_ratios(config: CartesianImpedanceConfig) -> Vector3:
    critical = critical_damping(config)
    return tuple(
        config.damping_n_s_m[axis] / critical[axis]
        if critical[axis] > 0.0
        else 0.0
        for axis in range(3)
    )  # type: ignore[return-value]
