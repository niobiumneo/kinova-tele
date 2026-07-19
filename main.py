import json
import os
from contextlib import asynccontextmanager
from copy import deepcopy
from math import atan2, cos, degrees, isfinite, radians, sin, sqrt
from pathlib import Path
from threading import Lock
from time import time
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from impedance_tuning.impedance_controller import (
    CartesianImpedanceConfig,
    CartesianImpedanceController,
)
from impedance_tuning.impedance_profile import (
    DEFAULT_PROFILE_FILENAME,
    load_impedance_profile,
    load_optional_impedance_profile,
)

# Pure Python Kinova Kortex API Imports
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2


def environment_bool(name, default):
    """Read a strict boolean environment setting."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false")


def environment_float(name, default):
    """Read a finite floating-point environment setting."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return float(default)
    value = float(raw_value)
    if not isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def environment_int(name, default):
    """Read an integer environment setting."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return int(default)
    return int(raw_value)


def environment_vector3(name, default):
    """Read one scalar or a comma-separated X,Y,Z environment setting."""
    raw_value = os.environ.get(name)
    if raw_value is None:
        return tuple(float(value) for value in default)

    values = tuple(float(value.strip()) for value in raw_value.split(","))
    if len(values) == 1:
        values *= 3
    if len(values) != 3 or not all(isfinite(value) for value in values):
        raise ValueError(f"{name} must contain one value or three finite values")
    return values

# --- CONFIGURATION ---
APP_DIR = Path(__file__).resolve().parent
IMPEDANCE_TUNING_DIR = APP_DIR / "impedance_tuning"
ROBOT_IP = "192.168.1.10"
ROBOT_PORT = 10000
USERNAME_ENV_VAR = "KINOVA_USERNAME"
PASSWORD_ENV_VAR = "KINOVA_PASSWORD"

# The standalone tuner writes this JSON file. An explicit environment path is
# required to exist; the default local filename is optional so existing setups
# retain the historical built-in defaults until a profile is exported.
IMPEDANCE_PROFILE_ENV_VAR = "KINOVA_IMPEDANCE_PROFILE"
_profile_setting = os.environ.get(IMPEDANCE_PROFILE_ENV_VAR)
if _profile_setting:
    IMPEDANCE_PROFILE_PATH = Path(_profile_setting).expanduser()
    if not IMPEDANCE_PROFILE_PATH.is_absolute():
        IMPEDANCE_PROFILE_PATH = APP_DIR / IMPEDANCE_PROFILE_PATH
    if not IMPEDANCE_PROFILE_PATH.exists():
        raise FileNotFoundError(
            f"{IMPEDANCE_PROFILE_ENV_VAR} does not exist: "
            f"{IMPEDANCE_PROFILE_PATH}"
        )
    IMPEDANCE_PROFILE = load_impedance_profile(IMPEDANCE_PROFILE_PATH)
    IMPEDANCE_PROFILE_LOADED = True
else:
    IMPEDANCE_PROFILE_PATH = IMPEDANCE_TUNING_DIR / DEFAULT_PROFILE_FILENAME
    IMPEDANCE_PROFILE, IMPEDANCE_PROFILE_LOADED = (
        load_optional_impedance_profile(IMPEDANCE_PROFILE_PATH)
    )

# --- SAFETY SETTINGS ---
VELOCITY_SCALE = 1.0
MAX_VELOCITY = 0.15  # m/s, per axis. Keep this low until you've verified directions.
# Keep twist commands in the base frame so they match the workspace coordinates.
REFERENCE_FRAME = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE

# The robot base X/Y plane is assumed to be parallel to the work surface. At
# startup, the current end-effector X/Y position becomes the center of a 4 ft
# by 4 ft rectangle. This keeps the boundary tied to the physical starting
# point instead of the base-frame origin.
# This application-level limiter is not a safety-rated substitute for matching
# Kortex protection zones or an external safety system.
FEET_TO_METERS = 0.3048
WORKSPACE_SIZE_X_M = 4.0 * FEET_TO_METERS
WORKSPACE_SIZE_Y_M = 4.0 * FEET_TO_METERS
WORKSPACE_SLOWDOWN_DISTANCE_M = 0.10
WORKSPACE_HAPTIC_THRESHOLD_M = 0.10

# Manual Cartesian motion holds startup pitch/roll while the right joystick
# adjusts base-frame yaw. Linear X/Y/Z translation remains enabled. Angular
# velocity is in degrees per second because Kortex TwistCommand expects it.
ORIENTATION_HOLD_GAIN = 2.0
ORIENTATION_HOLD_DEADBAND_DEG = 0.5
MAX_ANGULAR_VELOCITY_DEG_S = 20.0
JOYSTICK_YAW_RATE_DEG_S = 10.0
MAX_YAW_INTEGRATION_INTERVAL_S = 0.05

# Kortex gripper positions are normalized: 0.0 is open and 1.0 is closed.
GRIPPER_OPEN_POSITION = 0.0
GRIPPER_CLOSED_POSITION = 1.0
GRIPPER_COMMAND_INTERVAL_S = 0.05
GRIPPER_POSITION_DEADBAND = 0.02

# If the WebSocket goes quiet (headset takes off, wifi hiccup, browser tab suspends)
# for longer than this, we force-stop the robot even if we never got an explicit
# zero command. The short queue-drain timeout keeps only the newest motion command
# and is separate from the motion watchdog timeout.
QUEUE_DRAIN_TIMEOUT_S = 0.001
MOTION_WATCHDOG_TIMEOUT_S = 0.1

POSITION_GAIN = 2.0
AUTO_HOME_MAX_JOINT_VELOCITY_DEG_S = 10.0
AUTO_HOME_TIMEOUT_S = 30.0

# The Python controller uses Kortex's computed external tool wrench in a
# translational mass-spring-damper outer loop. It emits an additional
# base-frame velocity that is merged with the nominal XR or keyboard command.
# Pitch, roll, and yaw remain owned by the existing orientation controller.
IMPEDANCE_ENABLED = environment_bool(
    "KINOVA_IMPEDANCE_ENABLED",
    IMPEDANCE_PROFILE.runtime.enabled,
)
IMPEDANCE_WRENCH_FRAME = os.environ.get(
    "KINOVA_IMPEDANCE_WRENCH_FRAME",
    IMPEDANCE_PROFILE.runtime.wrench_frame,
).strip().lower()
if IMPEDANCE_WRENCH_FRAME not in ("base", "tool"):
    raise ValueError("KINOVA_IMPEDANCE_WRENCH_FRAME must be base or tool")
IMPEDANCE_TARE_MAX_TOOL_SPEED_M_S = environment_float(
    "KINOVA_IMPEDANCE_TARE_MAX_TOOL_SPEED_M_S",
    IMPEDANCE_PROFILE.runtime.tare_max_tool_speed_m_s,
)
IMPEDANCE_TARE_MAX_TOOL_ANGULAR_SPEED_DEG_S = environment_float(
    "KINOVA_IMPEDANCE_TARE_MAX_TOOL_ANGULAR_SPEED_DEG_S",
    IMPEDANCE_PROFILE.runtime.tare_max_tool_angular_speed_deg_s,
)
IMPEDANCE_TARE_AFTER_GRIPPER_DELAY_S = environment_float(
    "KINOVA_IMPEDANCE_TARE_AFTER_GRIPPER_DELAY_S",
    IMPEDANCE_PROFILE.runtime.tare_after_gripper_delay_s,
)
if IMPEDANCE_TARE_MAX_TOOL_SPEED_M_S < 0.0:
    raise ValueError("KINOVA_IMPEDANCE_TARE_MAX_TOOL_SPEED_M_S cannot be negative")
if IMPEDANCE_TARE_MAX_TOOL_ANGULAR_SPEED_DEG_S < 0.0:
    raise ValueError(
        "KINOVA_IMPEDANCE_TARE_MAX_TOOL_ANGULAR_SPEED_DEG_S cannot be negative"
    )
if IMPEDANCE_TARE_AFTER_GRIPPER_DELAY_S < 0.0:
    raise ValueError("KINOVA_IMPEDANCE_TARE_AFTER_GRIPPER_DELAY_S cannot be negative")
IMPEDANCE_CONFIG = CartesianImpedanceConfig(
    mass_kg=environment_vector3(
        "KINOVA_IMPEDANCE_MASS_KG",
        IMPEDANCE_PROFILE.impedance.mass_kg,
    ),
    stiffness_n_m=environment_vector3(
        "KINOVA_IMPEDANCE_STIFFNESS_N_M",
        IMPEDANCE_PROFILE.impedance.stiffness_n_m,
    ),
    damping_n_s_m=environment_vector3(
        "KINOVA_IMPEDANCE_DAMPING_NS_M",
        IMPEDANCE_PROFILE.impedance.damping_n_s_m,
    ),
    force_deadband_n=environment_vector3(
        "KINOVA_IMPEDANCE_FORCE_DEADBAND_N",
        IMPEDANCE_PROFILE.impedance.force_deadband_n,
    ),
    force_axis_sign=environment_vector3(
        "KINOVA_IMPEDANCE_FORCE_SIGN",
        IMPEDANCE_PROFILE.impedance.force_axis_sign,
    ),
    max_displacement_m=environment_vector3(
        "KINOVA_IMPEDANCE_MAX_DISPLACEMENT_M",
        IMPEDANCE_PROFILE.impedance.max_displacement_m,
    ),
    max_velocity_m_s=environment_float(
        "KINOVA_IMPEDANCE_MAX_VELOCITY_M_S",
        IMPEDANCE_PROFILE.impedance.max_velocity_m_s,
    ),
    filter_cutoff_hz=environment_float(
        "KINOVA_IMPEDANCE_FILTER_CUTOFF_HZ",
        IMPEDANCE_PROFILE.impedance.filter_cutoff_hz,
    ),
    force_limit_n=environment_float(
        "KINOVA_IMPEDANCE_FORCE_LIMIT_N",
        IMPEDANCE_PROFILE.impedance.force_limit_n,
    ),
    force_release_n=environment_float(
        "KINOVA_IMPEDANCE_FORCE_RELEASE_N",
        IMPEDANCE_PROFILE.impedance.force_release_n,
    ),
    contact_haptic_start_n=environment_float(
        "KINOVA_IMPEDANCE_HAPTIC_START_N",
        IMPEDANCE_PROFILE.impedance.contact_haptic_start_n,
    ),
    contact_haptic_full_scale_n=environment_float(
        "KINOVA_IMPEDANCE_HAPTIC_FULL_SCALE_N",
        IMPEDANCE_PROFILE.impedance.contact_haptic_full_scale_n,
    ),
    tare_samples=environment_int(
        "KINOVA_IMPEDANCE_TARE_SAMPLES",
        IMPEDANCE_PROFILE.impedance.tare_samples,
    ),
    tare_max_force_n=environment_float(
        "KINOVA_IMPEDANCE_TARE_MAX_FORCE_N",
        IMPEDANCE_PROFILE.impedance.tare_max_force_n,
    ),
    min_dt_s=IMPEDANCE_PROFILE.impedance.min_dt_s,
    max_dt_s=IMPEDANCE_PROFILE.impedance.max_dt_s,
)


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def wrap_degrees(angle):
    """Wrap an angle to [-180, 180) degrees."""
    return (angle + 180.0) % 360.0 - 180.0


def finite_command_value(payload, key):
    """Read an untrusted WebSocket number, rejecting bools, NaN, and infinity."""
    value = payload.get(key, 0.0)
    if isinstance(value, bool):
        return 0.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if isfinite(value) else 0.0


def external_wrench_from_feedback(base_feedback):
    """Read Kortex's computed tool wrench, rejecting missing or invalid fields."""
    force_fields = (
        "tool_external_wrench_force_x",
        "tool_external_wrench_force_y",
        "tool_external_wrench_force_z",
    )
    torque_fields = (
        "tool_external_wrench_torque_x",
        "tool_external_wrench_torque_y",
        "tool_external_wrench_torque_z",
    )
    try:
        force = tuple(float(getattr(base_feedback, name)) for name in force_fields)
        torque = tuple(float(getattr(base_feedback, name)) for name in torque_fields)
    except (AttributeError, TypeError, ValueError):
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False

    if not all(isfinite(value) for value in force + torque):
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False

    return force, torque, True


def measured_tool_linear_speed(base_feedback):
    """Return the measured Cartesian speed used to gate wrench taring."""
    try:
        velocity = (
            float(base_feedback.tool_twist_linear_x),
            float(base_feedback.tool_twist_linear_y),
            float(base_feedback.tool_twist_linear_z),
        )
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if not all(isfinite(value) for value in velocity):
        return 0.0
    return sqrt(sum(value * value for value in velocity))


def measured_tool_angular_speed(base_feedback):
    """Return measured angular speed used to reject moving tare samples."""
    try:
        velocity = (
            float(base_feedback.tool_twist_angular_x),
            float(base_feedback.tool_twist_angular_y),
            float(base_feedback.tool_twist_angular_z),
        )
    except (AttributeError, TypeError, ValueError):
        return 0.0
    if not all(isfinite(value) for value in velocity):
        return 0.0
    return sqrt(sum(value * value for value in velocity))


def add_linear_twists(nominal_twist, compliance_twist):
    """Merge the operator and impedance velocities before safety limiting."""
    return tuple(
        nominal + compliance
        for nominal, compliance in zip(nominal_twist, compliance_twist)
    )


def max_joint_error_degrees(current_angles, target_angles):
    """Return the largest wrapped actuator error in degrees."""
    if len(current_angles) != len(target_angles):
        raise ValueError(
            "current and target joint configurations have different sizes"
        )

    return max(
        (
            abs((target - current + 180.0) % 360.0 - 180.0)
            for current, target in zip(current_angles, target_angles)
        ),
        default=0.0,
    )


def quaternion_multiply(left, right):
    """Multiply two quaternions stored as (w, x, y, z)."""
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def quaternion_rotate_vector(quaternion, vector):
    """Rotate a tool-frame vector into the base frame."""
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


def euler_xyz_degrees_to_quaternion(theta_x, theta_y, theta_z):
    """Convert Kortex extrinsic X-Y-Z Euler angles to a quaternion."""
    half_x = radians(theta_x) / 2.0
    half_y = radians(theta_y) / 2.0
    half_z = radians(theta_z) / 2.0
    qx = (cos(half_x), sin(half_x), 0.0, 0.0)
    qy = (cos(half_y), 0.0, sin(half_y), 0.0)
    qz = (cos(half_z), 0.0, 0.0, sin(half_z))
    return quaternion_multiply(qz, quaternion_multiply(qy, qx))


def orientation_hold_velocity(pose, target_orientation_deg):
    """Return a capped base-frame angular correction toward the target orientation."""
    current = euler_xyz_degrees_to_quaternion(
        pose.tool_pose_theta_x,
        pose.tool_pose_theta_y,
        pose.tool_pose_theta_z,
    )
    target = euler_xyz_degrees_to_quaternion(*target_orientation_deg)
    current_conjugate = (current[0], -current[1], -current[2], -current[3])
    error = quaternion_multiply(target, current_conjugate)

    # q and -q describe the same orientation. Choose the shortest rotation.
    if error[0] < 0.0:
        error = tuple(-component for component in error)

    vector_norm = sqrt(sum(component**2 for component in error[1:]))
    if vector_norm < 1e-9:
        return 0.0, 0.0, 0.0, 0.0

    angle_deg = degrees(2.0 * atan2(vector_norm, clamp(error[0], -1.0, 1.0)))
    if angle_deg <= ORIENTATION_HOLD_DEADBAND_DEG:
        return 0.0, 0.0, 0.0, angle_deg

    requested_speed = ORIENTATION_HOLD_GAIN * angle_deg
    angular_speed = min(requested_speed, MAX_ANGULAR_VELOCITY_DEG_S)
    scale = angular_speed / vector_norm
    return (
        error[1] * scale,
        error[2] * scale,
        error[3] * scale,
        angle_deg,
    )


class PlanarWorkspace:
    """Server-side rectangular X/Y workspace constraint."""

    def __init__(self, startup_pose):
        half_x = WORKSPACE_SIZE_X_M / 2.0
        half_y = WORKSPACE_SIZE_Y_M / 2.0

        self.center_x = float(startup_pose.tool_pose_x)
        self.center_y = float(startup_pose.tool_pose_y)
        self.x_min = self.center_x - half_x
        self.x_max = self.center_x + half_x
        self.y_min = self.center_y - half_y
        self.y_max = self.center_y + half_y

    def contains(self, pose):
        return (
            self.x_min <= pose.tool_pose_x <= self.x_max
            and self.y_min <= pose.tool_pose_y <= self.y_max
        )

    def clamp_target(self, target_x, target_y):
        """Clamp a requested planar target to the configured rectangle."""
        return (
            clamp(target_x, self.x_min, self.x_max),
            clamp(target_y, self.y_min, self.y_max),
        )

    @staticmethod
    def _slow_outward_velocity(position, velocity, lower, upper):
        """Slow motion toward a wall and reject motion through or beyond it."""
        if velocity > 0.0:
            clearance = upper - position
        elif velocity < 0.0:
            clearance = position - lower
        else:
            return 0.0

        if clearance <= 0.0:
            return 0.0
        if clearance < WORKSPACE_SLOWDOWN_DISTANCE_M:
            velocity *= clearance / WORKSPACE_SLOWDOWN_DISTANCE_M
        return velocity

    def constrain_motion(self, pose, requested_vx, requested_vy, requested_vz):
        """Limit X/Y motion at the boundary while leaving Z unrestricted."""
        vx = self._slow_outward_velocity(
            pose.tool_pose_x,
            requested_vx,
            self.x_min,
            self.x_max,
        )
        vy = self._slow_outward_velocity(
            pose.tool_pose_y,
            requested_vy,
            self.y_min,
            self.y_max,
        )
        return vx, vy, requested_vz

    def feedback(self, pose):
        distances = {
            "x_min": pose.tool_pose_x - self.x_min,
            "x_max": self.x_max - pose.tool_pose_x,
            "y_min": pose.tool_pose_y - self.y_min,
            "y_max": self.y_max - pose.tool_pose_y,
        }
        nearest_boundary = min(distances, key=distances.get)
        signed_distance = distances[nearest_boundary]
        distance = max(0.0, signed_distance)
        haptic_intensity = clamp(
            1.0 - distance / WORKSPACE_HAPTIC_THRESHOLD_M,
            0.0,
            1.0,
        )

        return {
            "type": "workspace",
            "distance_to_boundary_m": distance,
            "nearest_boundary": nearest_boundary,
            "at_limit": signed_distance <= 0.002,
            "haptic_intensity": haptic_intensity,
            "tool_x_m": pose.tool_pose_x,
            "tool_y_m": pose.tool_pose_y,
            "workspace_center_x_m": self.center_x,
            "workspace_center_y_m": self.center_y,
            "workspace_x_min_m": self.x_min,
            "workspace_x_max_m": self.x_max,
            "workspace_y_min_m": self.y_min,
            "workspace_y_max_m": self.y_max,
        }

    def describe(self):
        return (
            f"startup center=({self.center_x:.3f}, {self.center_y:.3f}) m, "
            f"X=[{self.x_min:.3f}, {self.x_max:.3f}] m, "
            f"Y=[{self.y_min:.3f}, {self.y_max:.3f}] m"
        )


class KinovaController:
    def __init__(self):
        self.transport = None
        self.router = None
        self.base = None
        self.baseCyclic = None
        self.session_manager = None
        self._home_action_lock = Lock()
        self._home_action_state = "idle"
        self._home_cancel_state = None
        self._home_notification_handle = None

    def connect(self):
        try:
            username = os.environ.get(USERNAME_ENV_VAR)
            password = os.environ.get(PASSWORD_ENV_VAR)
            if not username or not password:
                raise RuntimeError(
                    "Kinova credentials are not configured. Set "
                    f"{USERNAME_ENV_VAR} and {PASSWORD_ENV_VAR} before startup."
                )

            print(f"Connecting directly to Kinova Gen3 at {ROBOT_IP}...")
            self.transport = TCPTransport()
            error_callback = lambda kException: print(f"API Error: {kException}")
            self.router = RouterClient(self.transport, error_callback)
            self.transport.connect(ROBOT_IP, ROBOT_PORT)

            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = username
            session_info.password = password
            session_info.session_inactivity_timeout = 60000
            session_info.connection_inactivity_timeout = 2000

            self.session_manager = SessionManager(self.router)
            self.session_manager.CreateSession(session_info)
            self.base = BaseClient(self.router)
            self.baseCyclic = BaseCyclicClient(self.router)

            try:
                self.base.ClearFaults()
            except Exception:
                pass

            try:
                servo_mode = Base_pb2.ServoingModeInformation()
                servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
                self.base.SetServoingMode(servo_mode)
                print("Set to SINGLE_LEVEL_SERVOING.")
            except Exception as e:
                print(f"SetServoingMode failed: {e}")

            print("Successfully linked to Kortex API. Robot Ready.")
        except Exception as e:
            print(f"Hardware connection failed: {e}")

    def send_cartesian_velocity(
        self,
        vx,
        vy,
        vz,
        angular_x=0.0,
        angular_y=0.0,
        angular_z=0.0,
    ):
        """Send a capped base-frame Cartesian linear and angular twist."""
        if not self.base:
            return

        safe_vx = max(-MAX_VELOCITY, min(MAX_VELOCITY, vx * VELOCITY_SCALE))
        safe_vy = max(-MAX_VELOCITY, min(MAX_VELOCITY, vy * VELOCITY_SCALE))
        safe_vz = max(-MAX_VELOCITY, min(MAX_VELOCITY, vz * VELOCITY_SCALE))
        safe_angular_x = clamp(
            angular_x,
            -MAX_ANGULAR_VELOCITY_DEG_S,
            MAX_ANGULAR_VELOCITY_DEG_S,
        )
        safe_angular_y = clamp(
            angular_y,
            -MAX_ANGULAR_VELOCITY_DEG_S,
            MAX_ANGULAR_VELOCITY_DEG_S,
        )
        safe_angular_z = clamp(
            angular_z,
            -MAX_ANGULAR_VELOCITY_DEG_S,
            MAX_ANGULAR_VELOCITY_DEG_S,
        )

        # print(
        #     f"Commanding -> X: {safe_vx:.4f} m/s | Y: {safe_vy:.4f} m/s | Z: {safe_vz:.4f} m/s",
        #     end="\r",
        # )

        command = Base_pb2.TwistCommand()
        command.reference_frame = REFERENCE_FRAME
        command.twist.linear_x = safe_vx
        command.twist.linear_y = safe_vy
        command.twist.linear_z = safe_vz
        command.twist.angular_x = safe_angular_x
        command.twist.angular_y = safe_angular_y
        command.twist.angular_z = safe_angular_z

        try:
            self.base.SendTwistCommand(command)
        except Exception as e:
            print(f"\nSendTwistCommand failed: {e}")

    def send_gripper_position(self, position):
        """Move the configured end-effector gripper to a normalized position."""
        if not self.base:
            return False

        target = max(0.0, min(1.0, float(position)))
        command = Base_pb2.GripperCommand()
        command.mode = Base_pb2.GRIPPER_POSITION

        finger = command.gripper.finger.add()
        finger.finger_identifier = 1
        finger.value = target

        try:
            self.base.SendGripperCommand(command)
            print(f"\nGripper target: {target:.2f}")
            return True
        except Exception as e:
            print(f"\nSendGripperCommand failed: {e}")
            return False

    def get_current_robot_state(self):
        """Return one synchronized sample of tool pose and all joint angles."""
        if self.baseCyclic is None:
            raise RuntimeError("Kinova cyclic feedback is unavailable")

        feedback = self.baseCyclic.RefreshFeedback()
        joint_angles = tuple(
            float(actuator.position) for actuator in feedback.actuators
        )
        if not joint_angles:
            raise RuntimeError("Kinova returned no actuator feedback")
        return feedback.base, joint_angles

    def get_current_robot_pose(self):
        pose, _joint_angles = self.get_current_robot_state()
        return pose

    def _set_home_action_state(self, state):
        with self._home_action_lock:
            self._home_action_state = state

    def get_home_action_state(self):
        with self._home_action_lock:
            return self._home_action_state

    def _on_home_action_notification(self, notification):
        """Record Kortex completion on its notification callback thread."""
        if notification.action_event == Base_pb2.ACTION_END:
            self._set_home_action_state("complete")
        elif notification.action_event == Base_pb2.ACTION_ABORT:
            with self._home_action_lock:
                self._home_action_state = (
                    self._home_cancel_state or "aborted"
                )

    def finish_joint_home(self):
        """Release the Kortex action subscription outside its callback."""
        with self._home_action_lock:
            notification_handle = self._home_notification_handle
            self._home_notification_handle = None

        if self.base and notification_handle is not None:
            try:
                self.base.Unsubscribe(notification_handle)
            except Exception:
                pass

    def start_joint_home(self, target_joint_angles):
        """Start a speed-limited waypoint to the captured startup joints."""
        if not self.base or not target_joint_angles:
            self._set_home_action_state("aborted")
            return False

        self.finish_joint_home()

        try:
            waypoint_list = Base_pb2.WaypointList()
            waypoint_list.use_optimal_blending = False
            waypoint = waypoint_list.waypoints.add()
            waypoint.name = "Startup joint configuration"
            waypoint.angular_waypoint.angles.extend(target_joint_angles)
            waypoint.angular_waypoint.maximum_velocities.extend(
                [AUTO_HOME_MAX_JOINT_VELOCITY_DEG_S]
                * len(target_joint_angles)
            )

            validation = self.base.ValidateWaypointList(waypoint_list)
            validation_errors = (
                validation.trajectory_error_report.trajectory_error_elements
            )
            if validation_errors:
                print(
                    "\nAuto-home waypoint rejected by Kortex: "
                    f"{len(validation_errors)} validation error(s)"
                )
                self._set_home_action_state("aborted")
                return False

            notification_handle = self.base.OnNotificationActionTopic(
                self._on_home_action_notification,
                Base_pb2.NotificationOptions(),
            )
            with self._home_action_lock:
                self._home_notification_handle = notification_handle
                self._home_cancel_state = None
                self._home_action_state = "moving"

            self.base.ExecuteWaypointTrajectory(waypoint_list)
            print(
                "\nAuto-home: returning all arm joints to the startup "
                "configuration."
            )
            return True
        except Exception as e:
            print(f"\nFailed to start joint auto-home: {e}")
            self._set_home_action_state("aborted")
            self.finish_joint_home()
            return False

    def cancel_joint_home(self, state="cancelled"):
        """Stop an active joint-space home action and record why it stopped."""
        with self._home_action_lock:
            was_active = self._home_action_state == "moving"
            self._home_cancel_state = state

        if was_active and self.base:
            try:
                self.base.StopAction()
            except Exception:
                try:
                    self.base.Stop()
                except Exception:
                    pass

        if was_active:
            self._set_home_action_state(state)
        self.finish_joint_home()

    def stop(self):
        if self.base:
            self.cancel_joint_home()
            try:
                self.base.Stop()
            except Exception:
                pass


# Initialize Robot Controller
robot = KinovaController()
workspace = None
home_joint_angles = None
startup_orientation_deg = None


@asynccontextmanager
async def lifespan(_app):
    global home_joint_angles, startup_orientation_deg, workspace
    robot.connect()
    try:
        try:
            initial_pose, initial_joint_angles = robot.get_current_robot_state()
            initial_pose = deepcopy(initial_pose)
            workspace = PlanarWorkspace(initial_pose)
            if not workspace.contains(initial_pose):
                raise RuntimeError(
                    "tool is outside the configured X/Y workspace; "
                    f"tool=({initial_pose.tool_pose_x:.3f}, "
                    f"{initial_pose.tool_pose_y:.3f}) m, "
                    f"workspace={workspace.describe()}"
                )
            home_joint_angles = tuple(initial_joint_angles)
            startup_orientation_deg = (
                float(initial_pose.tool_pose_theta_x),
                float(initial_pose.tool_pose_theta_y),
                float(initial_pose.tool_pose_theta_z),
            )
            print(f"Planar workspace active: {workspace.describe()}")
            print(
                "Manual Cartesian orientation lock: "
                f"theta=({startup_orientation_deg[0]:.2f}, "
                f"{startup_orientation_deg[1]:.2f}, "
                f"{startup_orientation_deg[2]:.2f}) deg; "
                "right thumbstick yaw enabled"
            )
            print(
                "Auto-home startup joint configuration: "
                + ", ".join(
                    f"J{index + 1}={angle:.2f} deg"
                    for index, angle in enumerate(home_joint_angles)
                )
            )
            if IMPEDANCE_PROFILE_LOADED:
                print(
                    "Impedance profile loaded: "
                    f"{IMPEDANCE_PROFILE.name!r} from {IMPEDANCE_PROFILE_PATH}"
                )
            else:
                print(
                    "No impedance_tuning/impedance_profile.json found; using built-in "
                    "impedance defaults."
                )
            if IMPEDANCE_ENABLED:
                print(
                    "Cartesian impedance outer loop enabled: "
                    f"M={IMPEDANCE_CONFIG.mass_kg} kg, "
                    f"K={IMPEDANCE_CONFIG.stiffness_n_m} N/m, "
                    f"D={IMPEDANCE_CONFIG.damping_n_s_m} N*s/m, "
                    f"force limit={IMPEDANCE_CONFIG.force_limit_n:.1f} N, "
                    f"wrench frame={IMPEDANCE_WRENCH_FRAME}"
                )
                print(
                    "Keep the tool unloaded and stationary with the motion "
                    "clutch released while the external wrench is tared."
                )
            else:
                print("Cartesian impedance outer loop disabled by environment.")
        except Exception as e:
            workspace = None
            home_joint_angles = None
            startup_orientation_deg = None
            print(f"Failed to initialize workspace: {e}")

        yield
    finally:
        robot.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def get_index():
    return FileResponse(APP_DIR / "index.html")


@app.get("/static/three.module.js", include_in_schema=False)
async def get_three_module():
    return FileResponse(
        APP_DIR / "three.module.js",
        media_type="text/javascript",
    )


@app.get("/static/VRButton.js", include_in_schema=False)
async def get_vr_button():
    return FileResponse(
        APP_DIR / "VRButton.js",
        media_type="text/javascript",
    )


@app.get("/static/ARButton.js", include_in_schema=False)
async def get_ar_button():
    return FileResponse(
        APP_DIR / "ARButton.js",
        media_type="text/javascript",
    )


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n[WS] Quest client connected successfully!")
    if (
        workspace is None
        or home_joint_angles is None
        or startup_orientation_deg is None
    ):
        await websocket.close(
            code=1011,
            reason="Robot workspace or startup pose is unavailable",
        )
        return

    ref_robot_pose = deepcopy(robot.get_current_robot_pose())
    target_orientation_deg = list(startup_orientation_deg)
    last_yaw_update_time = time()
    last_gripper_position = None
    last_gripper_command_time = 0.0
    impedance_controller = CartesianImpedanceController(
        IMPEDANCE_CONFIG,
        enabled=IMPEDANCE_ENABLED,
    )
    last_impedance_update_time = time()
    try:
        last_motion_message_time = time()
        watchdog_stopped_motion = False
        auto_home_active = False
        auto_home_state = "idle"
        auto_home_started_at = 0.0
        while True:
            data = None
            while True:
                try:
                    check = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=QUEUE_DRAIN_TIMEOUT_S,
                    )
                    data = check
                except asyncio.TimeoutError:
                    break

            if data is None:
                if (
                    not watchdog_stopped_motion
                    and time() - last_motion_message_time > MOTION_WATCHDOG_TIMEOUT_S
                ):
                    if auto_home_active:
                        robot.cancel_joint_home("cancelled")
                        auto_home_active = False
                        auto_home_state = "cancelled"
                    else:
                        robot.send_cartesian_velocity(0, 0, 0)
                    impedance_controller.reset_dynamics()
                    watchdog_stopped_motion = True
                continue
            
            payload = json.loads(data)
            msg_type = payload.get("msg", None)

            gripper_position = payload.get("gripper_position")
            if isinstance(gripper_position, (int, float)) and not isinstance(
                gripper_position,
                bool,
            ) and isfinite(gripper_position):
                target = max(
                    GRIPPER_OPEN_POSITION,
                    min(GRIPPER_CLOSED_POSITION, float(gripper_position)),
                )

                # Snap the ends of the trigger range to fully open/closed.
                if target <= GRIPPER_POSITION_DEADBAND / 2:
                    target = GRIPPER_OPEN_POSITION
                elif target >= GRIPPER_CLOSED_POSITION - GRIPPER_POSITION_DEADBAND / 2:
                    target = GRIPPER_CLOSED_POSITION

                now = time()
                position_changed = (
                    last_gripper_position is None
                    or abs(target - last_gripper_position)
                    >= GRIPPER_POSITION_DEADBAND
                )
                command_interval_elapsed = (
                    now - last_gripper_command_time >= GRIPPER_COMMAND_INTERVAL_S
                )

                if position_changed and command_interval_elapsed:
                    last_gripper_command_time = now
                    if robot.send_gripper_position(target):
                        last_gripper_position = target

            if msg_type is not None:
                vx = finite_command_value(payload, "vx")
                vy = finite_command_value(payload, "vy")
                vz = finite_command_value(payload, "vz")
                yaw_input = clamp(
                    finite_command_value(payload, "yaw_input"),
                    -1.0,
                    1.0,
                )
                grip_pressed = payload.get("grip_pressed") is True
                home_requested = payload.get("home_request") is True
                xr_presenting = payload.get("xr_presenting") is True
                controller_present = payload.get("controller_present") is True
                pose, current_joint_angles = robot.get_current_robot_state()
                pose = deepcopy(pose)
                home_joint_error_deg = max_joint_error_degrees(
                    current_joint_angles,
                    home_joint_angles,
                )
                keyboard_motion_requested = (
                    msg_type == "keyboard"
                    and (
                        abs(vx) >= 0.001
                        or abs(vy) >= 0.001
                        or abs(vz) >= 0.001
                    )
                )
                external_force_n, external_torque_nm, wrench_available = (
                    external_wrench_from_feedback(pose)
                )
                force_transform = None
                if wrench_available and IMPEDANCE_WRENCH_FRAME == "tool":
                    try:
                        tool_angles_deg = (
                            float(pose.tool_pose_theta_x),
                            float(pose.tool_pose_theta_y),
                            float(pose.tool_pose_theta_z),
                        )
                        if not all(isfinite(value) for value in tool_angles_deg):
                            raise ValueError("invalid tool orientation feedback")
                        tool_orientation = euler_xyz_degrees_to_quaternion(
                            *tool_angles_deg
                        )
                        external_torque_nm = quaternion_rotate_vector(
                            tool_orientation,
                            external_torque_nm,
                        )
                        if not all(
                            isfinite(value) for value in external_torque_nm
                        ):
                            raise ValueError("invalid transformed wrench feedback")
                        force_transform = lambda force: quaternion_rotate_vector(
                            tool_orientation,
                            force,
                        )
                    except (AttributeError, TypeError, ValueError):
                        external_force_n = (0.0, 0.0, 0.0)
                        external_torque_nm = (0.0, 0.0, 0.0)
                        wrench_available = False

                now = time()
                yaw_update_interval = clamp(
                    now - last_yaw_update_time,
                    0.0,
                    MAX_YAW_INTEGRATION_INTERVAL_S,
                )
                last_yaw_update_time = now
                impedance_dt_s = now - last_impedance_update_time
                last_impedance_update_time = now
                tare_allowed = (
                    not grip_pressed
                    and not keyboard_motion_requested
                    and not auto_home_active
                    and not home_requested
                    and now - last_gripper_command_time
                    >= IMPEDANCE_TARE_AFTER_GRIPPER_DELAY_S
                    and measured_tool_linear_speed(pose)
                    <= IMPEDANCE_TARE_MAX_TOOL_SPEED_M_S
                    and measured_tool_angular_speed(pose)
                    <= IMPEDANCE_TARE_MAX_TOOL_ANGULAR_SPEED_DEG_S
                )
                impedance_output = impedance_controller.update(
                    external_force_n,
                    impedance_dt_s,
                    wrench_available=wrench_available,
                    allow_tare=tare_allowed,
                    allow_motion=not auto_home_active and not home_requested,
                    allow_force_limit_release=(
                        not grip_pressed and not keyboard_motion_requested
                    ),
                    force_transform=force_transform,
                    velocity_limiter=lambda velocity: workspace.constrain_motion(
                        pose,
                        *velocity,
                    ),
                )
                impedance_tare_pending = IMPEDANCE_ENABLED and (
                    impedance_output.state
                    in ("calibrating", "tare_force_too_high")
                )
                operator_motion_interlocked = (
                    impedance_output.force_limit_active
                    or impedance_tare_pending
                )
                if (
                    auto_home_state
                    in ("force_limited", "impedance_calibrating")
                    and not auto_home_active
                    and not operator_motion_interlocked
                ):
                    auto_home_state = "idle"

                if home_requested and not operator_motion_interlocked:
                    # Joint auto-home returns to the complete startup pose.
                    target_orientation_deg[:] = startup_orientation_deg
                elif (
                    not auto_home_active
                    and not operator_motion_interlocked
                    and msg_type == "XR"
                    and grip_pressed
                ):
                    # Change only base-frame yaw. Startup pitch and roll remain
                    # fixed so a correctly initialized tool stays parallel to
                    # the working surface.
                    target_orientation_deg[2] = wrap_degrees(
                        target_orientation_deg[2]
                        + yaw_input
                        * JOYSTICK_YAW_RATE_DEG_S
                        * yaw_update_interval
                    )

                (
                    orientation_angular_x,
                    orientation_angular_y,
                    orientation_angular_z,
                    orientation_error_deg,
                ) = orientation_hold_velocity(pose, target_orientation_deg)
                orientation_twist = (
                    orientation_angular_x,
                    orientation_angular_y,
                    orientation_angular_z,
                )
                if impedance_output.force_limit_active:
                    # Do not keep driving toward an orientation target while
                    # the contact-force interlock is active.
                    orientation_twist = (0.0, 0.0, 0.0)

                if (
                    home_requested
                    and not auto_home_active
                    and not operator_motion_interlocked
                ):
                    # Clear the last manual twist before handing control to the
                    # high-level joint trajectory.
                    robot.send_cartesian_velocity(0, 0, 0)
                    impedance_controller.reset_dynamics()
                    auto_home_active = robot.start_joint_home(home_joint_angles)
                    auto_home_state = robot.get_home_action_state()
                    if auto_home_active:
                        auto_home_started_at = time()
                    ref_robot_pose = deepcopy(pose)
                elif (
                    home_requested
                    and not auto_home_active
                    and impedance_output.force_limit_active
                ):
                    auto_home_state = "force_limited"
                elif (
                    home_requested
                    and not auto_home_active
                    and impedance_tare_pending
                ):
                    auto_home_state = "impedance_calibrating"

                cancel_auto_home = (
                    auto_home_active
                    and (
                        impedance_output.force_limit_active
                        or (
                            not home_requested
                            and (
                                grip_pressed
                                or keyboard_motion_requested
                                or not xr_presenting
                                or not controller_present
                            )
                        )
                    )
                )
                auto_home_timed_out = (
                    auto_home_active
                    and time() - auto_home_started_at > AUTO_HOME_TIMEOUT_S
                )
                auto_home_left_workspace = (
                    auto_home_active and not workspace.contains(pose)
                )

                if (
                    cancel_auto_home
                    or auto_home_timed_out
                    or auto_home_left_workspace
                ):
                    auto_home_active = False
                    if impedance_output.force_limit_active:
                        auto_home_state = "force_limited"
                    elif auto_home_timed_out:
                        auto_home_state = "timeout"
                    elif auto_home_left_workspace:
                        auto_home_state = "workspace_blocked"
                    else:
                        auto_home_state = "cancelled"
                    robot.cancel_joint_home(auto_home_state)
                    impedance_controller.reset_dynamics()
                    target_orientation_deg[:] = (
                        startup_orientation_deg[0],
                        startup_orientation_deg[1],
                        wrap_degrees(float(pose.tool_pose_theta_z)),
                    )
                    ref_robot_pose = deepcopy(pose)
                elif auto_home_active:
                    action_state = robot.get_home_action_state()
                    if action_state in ("complete", "aborted"):
                        auto_home_active = False
                        auto_home_state = action_state
                        robot.finish_joint_home()
                        impedance_controller.reset_dynamics()
                        if action_state == "complete":
                            target_orientation_deg[:] = startup_orientation_deg
                        else:
                            target_orientation_deg[:] = (
                                startup_orientation_deg[0],
                                startup_orientation_deg[1],
                                wrap_degrees(float(pose.tool_pose_theta_z)),
                            )
                        ref_robot_pose = deepcopy(pose)
                    else:
                        auto_home_state = "moving"

                elif msg_type == "keyboard":
                    if operator_motion_interlocked:
                        nominal_twist = (0.0, 0.0, 0.0)
                    else:
                        nominal_twist = (
                            clamp(vx, -1.0, 1.0) * MAX_VELOCITY,
                            clamp(vy, -1.0, 1.0) * MAX_VELOCITY,
                            clamp(vz, -1.0, 1.0) * MAX_VELOCITY,
                        )
                    requested_twist = add_linear_twists(
                        nominal_twist,
                        impedance_output.compliance_velocity_m_s,
                    )
                    limited_twist = workspace.constrain_motion(
                        pose,
                        *requested_twist,
                    )
                    robot.send_cartesian_velocity(
                        *limited_twist,
                        *orientation_twist,
                    )
                    if keyboard_motion_requested:
                        auto_home_state = "idle"
                elif msg_type == "XR":
                    if operator_motion_interlocked:
                        # Drop the old hand-relative target. The force limit
                        # or tare interlock requires the operator to release
                        # the movement clutch before motion resumes.
                        ref_robot_pose = deepcopy(pose)
                        nominal_twist = (0.0, 0.0, 0.0)
                    elif (
                        abs(vx) < 0.001
                        and abs(vy) < 0.001
                        and abs(vz) < 0.001
                    ):
                        ref_robot_pose = deepcopy(pose)
                        nominal_twist = (0.0, 0.0, 0.0)
                    else:
                        target_x, target_y = workspace.clamp_target(
                            ref_robot_pose.tool_pose_x + vz,
                            ref_robot_pose.tool_pose_y - vx,
                        )
                        target_z = ref_robot_pose.tool_pose_z + vy

                        x_speed = POSITION_GAIN * (target_x - pose.tool_pose_x)
                        y_speed = POSITION_GAIN * (target_y - pose.tool_pose_y)
                        z_speed = POSITION_GAIN * (target_z - pose.tool_pose_z)
                        nominal_twist = (x_speed, y_speed, z_speed)
                        auto_home_state = "idle"

                    requested_twist = add_linear_twists(
                        nominal_twist,
                        impedance_output.compliance_velocity_m_s,
                    )
                    limited_twist = workspace.constrain_motion(
                        pose,
                        *requested_twist,
                    )
                    robot.send_cartesian_velocity(
                        *limited_twist,
                        *orientation_twist,
                    )
                else:
                    requested_twist = add_linear_twists(
                        (0.0, 0.0, 0.0),
                        impedance_output.compliance_velocity_m_s,
                    )
                    limited_twist = workspace.constrain_motion(
                        pose,
                        *requested_twist,
                    )
                    robot.send_cartesian_velocity(
                        *limited_twist,
                        *orientation_twist,
                    )

                feedback = workspace.feedback(pose)
                feedback["auto_home_state"] = auto_home_state
                feedback["auto_home_joint_error_deg"] = home_joint_error_deg
                feedback["orientation_error_deg"] = orientation_error_deg
                feedback["orientation_target_yaw_deg"] = target_orientation_deg[2]
                feedback["tool_yaw_deg"] = float(pose.tool_pose_theta_z)
                feedback["external_torque_x_nm"] = external_torque_nm[0]
                feedback["external_torque_y_nm"] = external_torque_nm[1]
                feedback["external_torque_z_nm"] = external_torque_nm[2]
                feedback["impedance_wrench_frame"] = IMPEDANCE_WRENCH_FRAME
                feedback.update(impedance_output.feedback())
                await websocket.send_json(feedback)
                last_motion_message_time = time()
                watchdog_stopped_motion = False

            
    except WebSocketDisconnect:
        print("\n[WS] Client disconnected. Stopping robot.")
        robot.stop()
    except Exception as e:
        print(f"\n[WS] Handler error: {e}")
        robot.stop()


def start_server():
    """Run the FastAPI teleoperation server."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="error",
        ssl_keyfile=str(APP_DIR / "key.pem"),
        ssl_certfile=str(APP_DIR / "cert.pem"),
    )


if __name__ == "__main__":
    local_ip = "192.168.1.70"
    print("\n" + "=" * 50)
    print(" SERVER RUNNING (HTTPS/WSS REQUIRED FOR WEBXR) ")
    print(f" Quest Browser URL: https://{local_ip}:8000")
    print(" On first visit, accept the self-signed cert warning")
    print(" (Advanced -> Proceed) BEFORE trying to enter AR.")
    print("=" * 50 + "\n")

    start_server()
