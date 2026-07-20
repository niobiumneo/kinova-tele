"""Pure helpers for planning and gating Kinova joint auto-home actions."""

from math import isfinite


def wrapped_joint_error_degrees(current_angle, target_angle):
    """Return the shortest signed angular error in degrees."""
    current = float(current_angle)
    target = float(target_angle)
    if not isfinite(current) or not isfinite(target):
        raise ValueError("joint angles must be finite")
    return (target - current + 180.0) % 360.0 - 180.0


def max_joint_error_degrees(current_angles, target_angles):
    """Return the largest shortest-path actuator error in degrees."""
    current = tuple(current_angles)
    target = tuple(target_angles)
    if len(current) != len(target):
        raise ValueError(
            "current and target joint configurations have different sizes"
        )
    return max(
        (
            abs(wrapped_joint_error_degrees(current_value, target_value))
            for current_value, target_value in zip(current, target)
        ),
        default=0.0,
    )


def plan_home_duration_seconds(
    current_angles,
    target_angles,
    max_joint_velocity_deg_s,
    *,
    minimum_duration_s=2.0,
    duration_margin=2.0,
    maximum_duration_s=45.0,
):
    """Plan a conservative waypoint duration from the largest joint move.

    The two-times margin keeps the average joint speed at no more than half of
    the configured maximum, leaving room for acceleration and deceleration.
    """
    max_velocity = float(max_joint_velocity_deg_s)
    minimum_duration = float(minimum_duration_s)
    margin = float(duration_margin)
    maximum_duration = float(maximum_duration_s)
    values = (max_velocity, minimum_duration, margin, maximum_duration)
    if not all(isfinite(value) for value in values):
        raise ValueError("home duration settings must be finite")
    if max_velocity <= 0.0:
        raise ValueError("max_joint_velocity_deg_s must be positive")
    if minimum_duration <= 0.0 or maximum_duration < minimum_duration:
        raise ValueError("invalid home duration bounds")
    if margin < 1.0:
        raise ValueError("duration_margin must be at least 1")

    max_error = max_joint_error_degrees(current_angles, target_angles)
    requested_duration = margin * max_error / max_velocity
    duration = max(minimum_duration, requested_duration)
    return min(duration, maximum_duration)


def joint_velocities_are_settled(
    joint_velocities_deg_s,
    expected_actuator_count,
    threshold_deg_s,
):
    """Return true only for a complete, finite, near-zero velocity sample."""
    velocities = tuple(float(value) for value in joint_velocities_deg_s)
    expected_count = int(expected_actuator_count)
    threshold = float(threshold_deg_s)
    if expected_count <= 0 or threshold < 0.0 or not isfinite(threshold):
        return False
    if len(velocities) != expected_count:
        return False
    if not all(isfinite(value) for value in velocities):
        return False
    return all(abs(value) <= threshold for value in velocities)
