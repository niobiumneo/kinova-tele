"""Cartesian impedance behavior shared by Kortex and MuJoCo pipelines.

The Kortex Python application sends high-level Cartesian velocity commands, so
this module realizes a virtual mass-spring-damper as an admittance outer loop:

    M * x_ddot + D * x_dot + K * x = F_external

The resulting compliance velocity is added to the nominal teleoperation
velocity. This is intentionally separate from a 1 kHz joint-torque impedance
controller, which should be implemented with Kortex low-level control in C++.
"""

from dataclasses import dataclass
from math import exp, isfinite, pi, sqrt
from typing import Callable, Iterable, Optional, Tuple


Vector3 = Tuple[float, float, float]
ZERO_VECTOR: Vector3 = (0.0, 0.0, 0.0)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _vector(values: Iterable[float], name: str) -> Vector3:
    result = tuple(float(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    if not all(isfinite(value) for value in result):
        raise ValueError(f"{name} values must be finite")
    return result  # type: ignore[return-value]


def _norm(values: Vector3) -> float:
    return sqrt(sum(value * value for value in values))


@dataclass(frozen=True)
class CartesianImpedanceConfig:
    """Tuning and safety limits for translational Cartesian compliance."""

    mass_kg: Vector3 = (4.0, 4.0, 4.0)
    stiffness_n_m: Vector3 = (180.0, 180.0, 220.0)
    damping_n_s_m: Vector3 = (55.0, 55.0, 60.0)
    force_deadband_n: Vector3 = (1.5, 1.5, 1.5)
    force_axis_sign: Vector3 = (1.0, 1.0, 1.0)
    max_displacement_m: Vector3 = (0.05, 0.05, 0.05)
    max_velocity_m_s: float = 0.05
    filter_cutoff_hz: float = 8.0
    force_limit_n: float = 35.0
    force_release_n: float = 24.0
    contact_haptic_start_n: float = 2.0
    contact_haptic_full_scale_n: float = 20.0
    tare_samples: int = 30
    tare_max_force_n: float = 10.0
    min_dt_s: float = 0.001
    max_dt_s: float = 0.05

    def __post_init__(self) -> None:
        vector_fields = (
            "mass_kg",
            "stiffness_n_m",
            "damping_n_s_m",
            "force_deadband_n",
            "force_axis_sign",
            "max_displacement_m",
        )
        for field_name in vector_fields:
            object.__setattr__(
                self,
                field_name,
                _vector(getattr(self, field_name), field_name),
            )

        if any(value <= 0.0 for value in self.mass_kg):
            raise ValueError("mass_kg values must be positive")
        if any(value < 0.0 for value in self.stiffness_n_m):
            raise ValueError("stiffness_n_m values cannot be negative")
        if any(value < 0.0 for value in self.damping_n_s_m):
            raise ValueError("damping_n_s_m values cannot be negative")
        if any(value < 0.0 for value in self.force_deadband_n):
            raise ValueError("force_deadband_n values cannot be negative")
        if any(abs(value) != 1.0 for value in self.force_axis_sign):
            raise ValueError("force_axis_sign values must be either -1 or 1")
        if any(value <= 0.0 for value in self.max_displacement_m):
            raise ValueError("max_displacement_m values must be positive")
        scalar_values = (
            self.max_velocity_m_s,
            self.filter_cutoff_hz,
            self.force_limit_n,
            self.force_release_n,
            self.contact_haptic_start_n,
            self.contact_haptic_full_scale_n,
            self.tare_max_force_n,
            self.min_dt_s,
            self.max_dt_s,
        )
        if not all(isfinite(value) for value in scalar_values):
            raise ValueError("controller scalar values must be finite")
        if self.max_velocity_m_s <= 0.0:
            raise ValueError("max_velocity_m_s must be positive")
        if self.filter_cutoff_hz <= 0.0:
            raise ValueError("filter_cutoff_hz must be positive")
        if self.force_limit_n <= 0.0:
            raise ValueError("force_limit_n must be positive")
        if not 0.0 <= self.force_release_n < self.force_limit_n:
            raise ValueError("force_release_n must be below force_limit_n")
        if self.contact_haptic_start_n < 0.0:
            raise ValueError("contact_haptic_start_n cannot be negative")
        if self.contact_haptic_full_scale_n <= self.contact_haptic_start_n:
            raise ValueError(
                "contact_haptic_full_scale_n must exceed contact_haptic_start_n"
            )
        if self.tare_max_force_n <= 0.0:
            raise ValueError("tare_max_force_n must be positive")
        if self.tare_max_force_n >= self.force_limit_n:
            raise ValueError("tare_max_force_n must be below force_limit_n")
        if (
            isinstance(self.tare_samples, bool)
            or not isinstance(self.tare_samples, int)
            or self.tare_samples < 0
        ):
            raise ValueError("tare_samples must be a non-negative integer")
        if self.min_dt_s <= 0.0 or self.max_dt_s < self.min_dt_s:
            raise ValueError("invalid controller integration interval limits")


@dataclass(frozen=True)
class CartesianImpedanceOutput:
    """One controller update, including data used by UI and safety logic."""

    enabled: bool
    state: str
    wrench_available: bool
    force_limit_active: bool
    raw_force_n: Vector3
    force_bias_n: Vector3
    filtered_force_n: Vector3
    effective_force_n: Vector3
    force_norm_n: float
    compliance_velocity_m_s: Vector3
    compliance_displacement_m: Vector3
    contact_haptic_intensity: float
    tare_samples_collected: int
    tare_samples_required: int

    def feedback(self) -> dict:
        """Return JSON-safe fields for the teleoperation feedback packet."""
        return {
            "impedance_enabled": self.enabled,
            "impedance_state": self.state,
            "impedance_wrench_available": self.wrench_available,
            "impedance_force_limit_active": self.force_limit_active,
            "impedance_force_norm_n": self.force_norm_n,
            "impedance_external_force_x_n": self.effective_force_n[0],
            "impedance_external_force_y_n": self.effective_force_n[1],
            "impedance_external_force_z_n": self.effective_force_n[2],
            "impedance_raw_force_x_n": self.raw_force_n[0],
            "impedance_raw_force_y_n": self.raw_force_n[1],
            "impedance_raw_force_z_n": self.raw_force_n[2],
            "impedance_force_bias_x_n": self.force_bias_n[0],
            "impedance_force_bias_y_n": self.force_bias_n[1],
            "impedance_force_bias_z_n": self.force_bias_n[2],
            "impedance_velocity_x_m_s": self.compliance_velocity_m_s[0],
            "impedance_velocity_y_m_s": self.compliance_velocity_m_s[1],
            "impedance_velocity_z_m_s": self.compliance_velocity_m_s[2],
            "impedance_displacement_x_m": self.compliance_displacement_m[0],
            "impedance_displacement_y_m": self.compliance_displacement_m[1],
            "impedance_displacement_z_m": self.compliance_displacement_m[2],
            "impedance_contact_haptic_intensity": self.contact_haptic_intensity,
            "impedance_tare_samples_collected": self.tare_samples_collected,
            "impedance_tare_samples_required": self.tare_samples_required,
        }


class CartesianImpedanceController:
    """Virtual translational mass-spring-damper driven by external force."""

    def __init__(
        self,
        config: Optional[CartesianImpedanceConfig] = None,
        *,
        enabled: bool = True,
    ) -> None:
        self.config = config or CartesianImpedanceConfig()
        self.enabled = bool(enabled)
        self._force_bias_n = [0.0, 0.0, 0.0]
        self._tare_accumulator_n = [0.0, 0.0, 0.0]
        self._tare_samples_collected = 0
        self._tare_complete = self.config.tare_samples == 0
        self._filtered_force_n = [0.0, 0.0, 0.0]
        self._velocity_m_s = [0.0, 0.0, 0.0]
        self._displacement_m = [0.0, 0.0, 0.0]
        self._force_limit_active = False

    @property
    def force_bias_n(self) -> Vector3:
        return tuple(self._force_bias_n)  # type: ignore[return-value]

    def reset_dynamics(self) -> None:
        """Clear compliance motion while retaining the calibrated force bias."""
        self._filtered_force_n[:] = ZERO_VECTOR
        self._reset_motion_state()

    def _reset_motion_state(self) -> None:
        """Clear virtual motion without discarding the filtered force sample."""
        self._velocity_m_s[:] = ZERO_VECTOR
        self._displacement_m[:] = ZERO_VECTOR

    def begin_tare(self) -> None:
        """Restart zero-force calibration and clear all dynamic state."""
        self.reset_dynamics()
        self._force_bias_n[:] = ZERO_VECTOR
        self._tare_accumulator_n[:] = ZERO_VECTOR
        self._tare_samples_collected = 0
        self._tare_complete = self.config.tare_samples == 0
        self._force_limit_active = False

    def _output(
        self,
        *,
        state: str,
        wrench_available: bool,
        raw_force_n: Vector3,
        effective_force_n: Vector3 = ZERO_VECTOR,
        force_norm_n: float = 0.0,
        contact_haptic_intensity: float = 0.0,
    ) -> CartesianImpedanceOutput:
        return CartesianImpedanceOutput(
            enabled=self.enabled,
            state=state,
            wrench_available=wrench_available,
            force_limit_active=self._force_limit_active,
            raw_force_n=raw_force_n,
            force_bias_n=tuple(self._force_bias_n),  # type: ignore[arg-type]
            filtered_force_n=tuple(self._filtered_force_n),  # type: ignore[arg-type]
            effective_force_n=effective_force_n,
            force_norm_n=force_norm_n,
            compliance_velocity_m_s=tuple(self._velocity_m_s),  # type: ignore[arg-type]
            compliance_displacement_m=tuple(self._displacement_m),  # type: ignore[arg-type]
            contact_haptic_intensity=contact_haptic_intensity,
            tare_samples_collected=self._tare_samples_collected,
            tare_samples_required=self.config.tare_samples,
        )

    @staticmethod
    def _safe_force(
        external_force_n: Iterable[float],
        wrench_available: bool,
    ) -> Tuple[Vector3, bool]:
        try:
            force = tuple(float(value) for value in external_force_n)
        except (TypeError, ValueError):
            return ZERO_VECTOR, False

        if len(force) != 3 or not all(isfinite(value) for value in force):
            return ZERO_VECTOR, False
        return force, bool(wrench_available)  # type: ignore[return-value]

    def update(
        self,
        external_force_n: Iterable[float],
        dt_s: float,
        *,
        wrench_available: bool = True,
        allow_tare: bool = True,
        allow_motion: bool = True,
        allow_force_limit_release: bool = True,
        force_transform: Optional[Callable[[Vector3], Iterable[float]]] = None,
        velocity_limiter: Optional[Callable[[Vector3], Iterable[float]]] = None,
    ) -> CartesianImpedanceOutput:
        """Advance the compliance model and return its base-frame velocity."""
        raw_force_n, wrench_available = self._safe_force(
            external_force_n,
            wrench_available,
        )

        if not self.enabled:
            self.reset_dynamics()
            self._force_limit_active = False
            return self._output(
                state="disabled",
                wrench_available=wrench_available,
                raw_force_n=raw_force_n,
            )

        if not wrench_available:
            self.reset_dynamics()
            self._force_limit_active = False
            return self._output(
                state="wrench_unavailable",
                wrench_available=False,
                raw_force_n=raw_force_n,
            )

        try:
            dt_s = float(dt_s)
        except (TypeError, ValueError):
            dt_s = self.config.min_dt_s
        if not isfinite(dt_s):
            dt_s = self.config.min_dt_s
        dt_s = _clamp(dt_s, self.config.min_dt_s, self.config.max_dt_s)

        if not self._tare_complete:
            self.reset_dynamics()
            tare_force_norm_n = _norm(raw_force_n)
            tare_force_is_safe = tare_force_norm_n <= self.config.tare_max_force_n
            if tare_force_norm_n >= self.config.force_limit_n:
                self._force_limit_active = True
            elif (
                self._force_limit_active
                and tare_force_norm_n <= self.config.force_release_n
                and allow_force_limit_release
            ):
                self._force_limit_active = False
            if allow_tare and tare_force_is_safe:
                for axis in range(3):
                    self._tare_accumulator_n[axis] += raw_force_n[axis]
                self._tare_samples_collected += 1

                if self._tare_samples_collected >= self.config.tare_samples:
                    sample_count = float(self._tare_samples_collected)
                    for axis in range(3):
                        self._force_bias_n[axis] = (
                            self._tare_accumulator_n[axis] / sample_count
                        )
                    self._tare_complete = True

            state = "active" if self._tare_complete and allow_motion else "calibrating"
            if self._force_limit_active:
                state = "force_limited"
            elif self._tare_complete and not allow_motion:
                state = "suspended"
            elif allow_tare and not tare_force_is_safe:
                state = "tare_force_too_high"
            reported_force_norm_n = 0.0 if self._tare_complete else tare_force_norm_n
            return self._output(
                state=state,
                wrench_available=True,
                raw_force_n=raw_force_n,
                force_norm_n=reported_force_norm_n,
                contact_haptic_intensity=(
                    1.0 if self._force_limit_active else 0.0
                ),
            )

        bias_corrected_force_n = tuple(
            raw_force_n[axis] - self._force_bias_n[axis]
            for axis in range(3)
        )
        if force_transform is not None:
            try:
                bias_corrected_force_n = _vector(
                    force_transform(bias_corrected_force_n),
                    "transformed external force",
                )
            except (TypeError, ValueError):
                self.reset_dynamics()
                self._force_limit_active = False
                return self._output(
                    state="wrench_unavailable",
                    wrench_available=False,
                    raw_force_n=raw_force_n,
                )
        corrected_force_n = tuple(
            bias_corrected_force_n[axis] * self.config.force_axis_sign[axis]
            for axis in range(3)
        )
        filter_alpha = 1.0 - exp(-2.0 * pi * self.config.filter_cutoff_hz * dt_s)
        for axis in range(3):
            self._filtered_force_n[axis] += filter_alpha * (
                corrected_force_n[axis] - self._filtered_force_n[axis]
            )

        effective_force_n = []
        for axis in range(3):
            filtered = self._filtered_force_n[axis]
            deadband = self.config.force_deadband_n[axis]
            magnitude = max(0.0, abs(filtered) - deadband)
            effective_force_n.append(magnitude if filtered >= 0.0 else -magnitude)
        effective_force = tuple(effective_force_n)  # type: ignore[assignment]
        force_norm_n = _norm(effective_force)

        if force_norm_n >= self.config.force_limit_n:
            self._force_limit_active = True
        elif (
            self._force_limit_active
            and force_norm_n <= self.config.force_release_n
            and allow_force_limit_release
        ):
            self._force_limit_active = False

        contact_haptic_intensity = _clamp(
            (force_norm_n - self.config.contact_haptic_start_n)
            / (
                self.config.contact_haptic_full_scale_n
                - self.config.contact_haptic_start_n
            ),
            0.0,
            1.0,
        )
        if self._force_limit_active:
            contact_haptic_intensity = 1.0

        if not allow_motion:
            self._reset_motion_state()
            return self._output(
                state="force_limited" if self._force_limit_active else "suspended",
                wrench_available=True,
                raw_force_n=raw_force_n,
                effective_force_n=effective_force,
                force_norm_n=force_norm_n,
                contact_haptic_intensity=contact_haptic_intensity,
            )

        force_for_dynamics = effective_force
        if force_norm_n > self.config.force_limit_n:
            scale = self.config.force_limit_n / force_norm_n
            force_for_dynamics = tuple(value * scale for value in effective_force)

        for axis in range(3):
            acceleration_m_s2 = (
                force_for_dynamics[axis]
                - self.config.damping_n_s_m[axis] * self._velocity_m_s[axis]
                - self.config.stiffness_n_m[axis]
                * self._displacement_m[axis]
            ) / self.config.mass_kg[axis]
            self._velocity_m_s[axis] = _clamp(
                self._velocity_m_s[axis] + acceleration_m_s2 * dt_s,
                -self.config.max_velocity_m_s,
                self.config.max_velocity_m_s,
            )

        if velocity_limiter is not None:
            limited_velocity = _vector(
                velocity_limiter(tuple(self._velocity_m_s)),
                "limited compliance velocity",
            )
            self._velocity_m_s[:] = limited_velocity

        for axis in range(3):
            next_displacement = (
                self._displacement_m[axis] + self._velocity_m_s[axis] * dt_s
            )
            max_displacement = self.config.max_displacement_m[axis]
            if abs(next_displacement) >= max_displacement:
                boundary = max_displacement if next_displacement >= 0.0 else -max_displacement
                moving_outward = self._velocity_m_s[axis] * boundary > 0.0
                self._displacement_m[axis] = boundary
                if moving_outward:
                    self._velocity_m_s[axis] = 0.0
            else:
                self._displacement_m[axis] = next_displacement

        return self._output(
            state="force_limited" if self._force_limit_active else "active",
            wrench_available=True,
            raw_force_n=raw_force_n,
            effective_force_n=effective_force,
            force_norm_n=force_norm_n,
            contact_haptic_intensity=contact_haptic_intensity,
        )
