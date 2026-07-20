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

from autohome import (
    joint_velocities_are_settled,
    max_joint_error_degrees,
    plan_home_duration_seconds,
)

# Pure Python Kinova Kortex API Imports
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

# --- CONFIGURATION ---
APP_DIR = Path(__file__).resolve().parent
ROBOT_IP = "192.168.1.10"
ROBOT_PORT = 10000
USERNAME_ENV_VAR = "KINOVA_USERNAME"
PASSWORD_ENV_VAR = "KINOVA_PASSWORD"

# --- SAFETY SETTINGS ---
VELOCITY_SCALE = 1.0
MAX_VELOCITY = 0.15  # m/s, per axis. Keep this low until you've verified directions.
# Maximum hand-relative target displacement accepted from an XR client for one
# clutch engagement. Translation gain changes distance, not this velocity cap.
MAX_XR_TARGET_OFFSET_M = 0.15
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
# without conflating it with the actual motion watchdog duration.
QUEUE_DRAIN_TIMEOUT_S = 0.001
MOTION_WATCHDOG_TIMEOUT_S = 0.1

POSITION_GAIN = 2.0
AUTO_HOME_MAX_JOINT_VELOCITY_DEG_S = 10.0
AUTO_HOME_ALREADY_THERE_TOLERANCE_DEG = 0.5
AUTO_HOME_SETTLE_VELOCITY_DEG_S = 0.5
AUTO_HOME_SETTLE_SAMPLES = 3
AUTO_HOME_SETTLE_TIMEOUT_S = 3.0
AUTO_HOME_ACTION_TIMEOUT_MARGIN_S = 10.0
AUTO_HOME_WATCHDOG_TIMEOUT_S = 0.5
AUTO_HOME_WORKSPACE_TOLERANCE_M = 0.005


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

    def contains(self, pose, tolerance_m=0.0):
        tolerance = max(0.0, float(tolerance_m))
        return (
            self.x_min - tolerance
            <= pose.tool_pose_x
            <= self.x_max + tolerance
            and self.y_min - tolerance
            <= pose.tool_pose_y
            <= self.y_max + tolerance
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
        self._home_action_error = ""
        self._home_planned_duration_s = 0.0
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
        """Return one synchronized tool pose, joint position, and velocity sample."""
        if self.baseCyclic is None:
            raise RuntimeError("Kinova cyclic feedback is unavailable")

        feedback = self.baseCyclic.RefreshFeedback()
        joint_angles = tuple(
            float(actuator.position) for actuator in feedback.actuators
        )
        joint_velocities = tuple(
            float(getattr(actuator, "velocity", float("nan")))
            for actuator in feedback.actuators
        )
        if not joint_angles:
            raise RuntimeError("Kinova returned no actuator feedback")
        return feedback.base, joint_angles, joint_velocities

    def get_current_robot_pose(self):
        pose, _joint_angles, _joint_velocities = self.get_current_robot_state()
        return pose

    def _set_home_action_state(self, state, error=""):
        with self._home_action_lock:
            self._home_action_state = state
            self._home_action_error = error

    def get_home_action_state(self):
        with self._home_action_lock:
            return self._home_action_state

    def get_home_action_status(self):
        with self._home_action_lock:
            return (
                self._home_action_state,
                self._home_action_error,
                self._home_planned_duration_s,
            )

    @staticmethod
    def _enum_name(enum_wrapper, value, fallback):
        try:
            return enum_wrapper.Name(value)
        except (AttributeError, TypeError, ValueError):
            return fallback

    @classmethod
    def _format_validation_error(cls, error):
        error_type = int(getattr(error, "error_type", 0))
        type_name = cls._enum_name(
            getattr(Base_pb2, "TrajectoryErrorType", None),
            error_type,
            f"trajectory_error_{error_type}",
        )
        message = str(getattr(error, "message", "")).strip()
        waypoint_index = int(getattr(error, "waypoint_index", 0))
        actuator_index = int(getattr(error, "index", 0))
        error_value = float(getattr(error, "error_value", 0.0))
        limits = (
            float(getattr(error, "min_value", 0.0)),
            float(getattr(error, "max_value", 0.0)),
        )
        detail = (
            f"{type_name} at waypoint {waypoint_index}, actuator "
            f"{actuator_index}: value={error_value:g}, "
            f"allowed=[{limits[0]:g}, {limits[1]:g}]"
        )
        return f"{detail} ({message})" if message else detail

    @classmethod
    def _format_kortex_exception(cls, error):
        details = [str(error)]
        get_error_code = getattr(error, "get_error_code", None)
        get_sub_error_code = getattr(error, "get_error_sub_code", None)
        if callable(get_error_code):
            try:
                details.append(f"code={get_error_code()}")
            except Exception:
                pass
        if callable(get_sub_error_code):
            try:
                sub_code = int(get_sub_error_code())
                sub_name = cls._enum_name(
                    getattr(Base_pb2, "SubErrorCodes", None),
                    sub_code,
                    f"sub_error_{sub_code}",
                )
                details.append(f"subcode={sub_name} ({sub_code})")
            except Exception:
                pass
        return "; ".join(detail for detail in details if detail)

    def _on_home_action_notification(self, notification):
        """Record Kortex completion on its notification callback thread."""
        action_type = int(
            getattr(getattr(notification, "handle", None), "action_type", -1)
        )
        waypoint_action_type = int(getattr(Base_pb2, "EXECUTE_WAYPOINT_LIST", 41))
        if action_type != waypoint_action_type:
            return

        if notification.action_event == Base_pb2.ACTION_END:
            self._set_home_action_state("complete", "")
        elif notification.action_event in (
            Base_pb2.ACTION_ABORT,
            getattr(Base_pb2, "ACTION_PREPROCESS_ABORT", -1),
        ):
            abort_code = int(getattr(notification, "abort_details", 0))
            abort_name = self._enum_name(
                getattr(Base_pb2, "SubErrorCodes", None),
                abort_code,
                f"abort_{abort_code}",
            )
            with self._home_action_lock:
                cancelled_state = self._home_cancel_state
                self._home_action_state = cancelled_state or "aborted"
                if not cancelled_state:
                    self._home_action_error = (
                        f"Kortex action aborted: {abort_name} ({abort_code})"
                    )
                    print(f"\nAuto-home {self._home_action_error}")

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

    def prepare_joint_home(self):
        """End live Cartesian control before waiting for the arm to settle."""
        if not self.base:
            self._set_home_action_state("aborted", "Kortex base client unavailable")
            return False

        self.finish_joint_home()
        try:
            self.base.Stop()
        except Exception as error:
            detail = self._format_kortex_exception(error)
            message = f"failed to stop Cartesian control: {detail}"
            print(f"\nAuto-home preparation failed: {message}")
            self._set_home_action_state("aborted", message)
            return False

        with self._home_action_lock:
            self._home_action_state = "settling"
            self._home_action_error = ""
            self._home_planned_duration_s = 0.0
            self._home_cancel_state = None
        print("\nAuto-home: stopping Cartesian control and waiting for the arm to settle.")
        return True

    def start_joint_home(self, target_joint_angles, current_joint_angles):
        """Start a conservative timed waypoint to the captured startup joints."""
        if not self.base or not target_joint_angles:
            self._set_home_action_state("aborted", "Kortex base client unavailable")
            return False

        try:
            max_error = max_joint_error_degrees(
                current_joint_angles,
                target_joint_angles,
            )
        except (TypeError, ValueError) as error:
            message = f"invalid joint feedback for auto-home: {error}"
            self._set_home_action_state("aborted", message)
            return False
        if max_error <= AUTO_HOME_ALREADY_THERE_TOLERANCE_DEG:
            self._set_home_action_state("complete", "")
            print("\nAuto-home: robot is already at the startup joint configuration.")
            return False

        planned_duration_s = plan_home_duration_seconds(
            current_joint_angles,
            target_joint_angles,
            AUTO_HOME_MAX_JOINT_VELOCITY_DEG_S,
        )

        self.finish_joint_home()

        try:
            waypoint_list = Base_pb2.WaypointList()
            waypoint_list.duration = 0.0
            waypoint_list.use_optimal_blending = False
            waypoint = waypoint_list.waypoints.add()
            waypoint.name = "Startup joint configuration"
            waypoint.angular_waypoint.angles.extend(target_joint_angles)
            waypoint.angular_waypoint.duration = planned_duration_s

            validation = self.base.ValidateWaypointList(waypoint_list)
            validation_errors = list(
                validation.trajectory_error_report.trajectory_error_elements
            )
            if validation_errors:
                details = [
                    self._format_validation_error(error)
                    for error in validation_errors
                ]
                message = "; ".join(details)
                print(
                    "\nAuto-home waypoint rejected by Kortex:\n  - "
                    + "\n  - ".join(details)
                )
                self._set_home_action_state("aborted", message)
                return False

            notification_handle = self.base.OnNotificationActionTopic(
                self._on_home_action_notification,
                Base_pb2.NotificationOptions(),
            )
            with self._home_action_lock:
                self._home_notification_handle = notification_handle
                self._home_cancel_state = None
                self._home_action_state = "moving"
                self._home_action_error = ""
                self._home_planned_duration_s = planned_duration_s

            self.base.ExecuteWaypointTrajectory(waypoint_list)
            print(
                "\nAuto-home: returning all arm joints to the startup "
                f"configuration over {planned_duration_s:.1f} s."
            )
            return True
        except Exception as error:
            detail = self._format_kortex_exception(error)
            print(f"\nFailed to start joint auto-home: {detail}")
            self._set_home_action_state("aborted", detail)
            self.finish_joint_home()
            return False

    def cancel_joint_home(self, state="cancelled", error=""):
        """Stop an active joint-space home action and record why it stopped."""
        with self._home_action_lock:
            previous_state = self._home_action_state
            was_active = previous_state in ("settling", "moving")
            self._home_cancel_state = state

        if previous_state == "moving" and self.base:
            try:
                self.base.StopAction()
            except Exception:
                try:
                    self.base.Stop()
                except Exception:
                    pass

        if was_active:
            self._set_home_action_state(state, error)
            if error:
                print(f"\nAuto-home stopped: {error}")
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
            (
                initial_pose,
                initial_joint_angles,
                _initial_joint_velocities,
            ) = robot.get_current_robot_state()
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
    try:
        last_motion_message_time = time()
        watchdog_stopped_motion = False
        auto_home_active = False
        auto_home_state = "idle"
        auto_home_settle_started_at = 0.0
        auto_home_settle_samples = 0
        auto_home_action_started_at = 0.0
        auto_home_action_timeout_s = 0.0
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
                watchdog_timeout_s = (
                    AUTO_HOME_WATCHDOG_TIMEOUT_S
                    if auto_home_active
                    else MOTION_WATCHDOG_TIMEOUT_S
                )
                if (
                    not watchdog_stopped_motion
                    and time() - last_motion_message_time > watchdog_timeout_s
                ):
                    if auto_home_active:
                        robot.cancel_joint_home(
                            "cancelled",
                            "Quest command watchdog expired during auto-home",
                        )
                        auto_home_active = False
                        auto_home_state = "cancelled"
                    else:
                        robot.send_cartesian_velocity(0, 0, 0)
                    watchdog_stopped_motion = True
                continue
            
            payload = json.loads(data)
            msg_type = payload.get("msg", None)

            gripper_position = payload.get("gripper_position")
            if (
                not auto_home_active
                and isinstance(gripper_position, (int, float))
                and not isinstance(gripper_position, bool)
                and isfinite(gripper_position)
            ):
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
                if msg_type == "XR":
                    vx = clamp(vx, -MAX_XR_TARGET_OFFSET_M, MAX_XR_TARGET_OFFSET_M)
                    vy = clamp(vy, -MAX_XR_TARGET_OFFSET_M, MAX_XR_TARGET_OFFSET_M)
                    vz = clamp(vz, -MAX_XR_TARGET_OFFSET_M, MAX_XR_TARGET_OFFSET_M)
                yaw_input = clamp(
                    finite_command_value(payload, "yaw_input"),
                    -1.0,
                    1.0,
                )
                grip_pressed = payload.get("grip_pressed") is True
                home_requested = payload.get("home_request") is True
                xr_presenting = payload.get("xr_presenting") is True
                controller_present = payload.get("controller_present") is True
                (
                    pose,
                    current_joint_angles,
                    current_joint_velocities,
                ) = robot.get_current_robot_state()
                pose = deepcopy(pose)
                home_joint_error_deg = max_joint_error_degrees(
                    current_joint_angles,
                    home_joint_angles,
                )

                now = time()
                yaw_update_interval = clamp(
                    now - last_yaw_update_time,
                    0.0,
                    MAX_YAW_INTEGRATION_INTERVAL_S,
                )
                last_yaw_update_time = now

                if home_requested:
                    # Joint auto-home returns to the complete startup pose.
                    target_orientation_deg[:] = startup_orientation_deg
                elif (
                    not auto_home_active
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

                if home_requested and not auto_home_active:
                    # End live twist control first. The action starts only after
                    # cyclic feedback confirms several near-zero velocity samples.
                    robot.send_cartesian_velocity(0, 0, 0)
                    auto_home_active = robot.prepare_joint_home()
                    auto_home_state = robot.get_home_action_state()
                    if auto_home_active:
                        auto_home_settle_started_at = now
                        auto_home_settle_samples = 0
                        auto_home_action_started_at = 0.0
                        auto_home_action_timeout_s = 0.0
                    ref_robot_pose = deepcopy(pose)

                keyboard_motion_requested = (
                    msg_type == "keyboard"
                    and (
                        abs(vx) >= 0.001
                        or abs(vy) >= 0.001
                        or abs(vz) >= 0.001
                    )
                )
                cancel_auto_home = (
                    auto_home_active
                    and not home_requested
                    and (
                        grip_pressed
                        or keyboard_motion_requested
                        or not xr_presenting
                        or not controller_present
                    )
                )
                controller_home_state = robot.get_home_action_state()
                auto_home_settle_timed_out = (
                    auto_home_active
                    and controller_home_state == "settling"
                    and now - auto_home_settle_started_at
                    > AUTO_HOME_SETTLE_TIMEOUT_S
                )
                auto_home_action_timed_out = (
                    auto_home_active
                    and controller_home_state == "moving"
                    and auto_home_action_started_at > 0.0
                    and now - auto_home_action_started_at
                    > auto_home_action_timeout_s
                )
                auto_home_left_workspace = (
                    auto_home_active
                    and not workspace.contains(
                        pose,
                        tolerance_m=AUTO_HOME_WORKSPACE_TOLERANCE_M,
                    )
                )

                if (
                    cancel_auto_home
                    or auto_home_settle_timed_out
                    or auto_home_action_timed_out
                    or auto_home_left_workspace
                ):
                    auto_home_active = False
                    auto_home_error = ""
                    if auto_home_settle_timed_out:
                        auto_home_state = "settle_timeout"
                        auto_home_error = (
                            "joint velocities did not settle below "
                            f"{AUTO_HOME_SETTLE_VELOCITY_DEG_S:.2f} deg/s "
                            f"within {AUTO_HOME_SETTLE_TIMEOUT_S:.1f} s"
                        )
                    elif auto_home_action_timed_out:
                        auto_home_state = "timeout"
                        auto_home_error = (
                            "Kortex joint action exceeded its planned timeout"
                        )
                    elif auto_home_left_workspace:
                        auto_home_state = "workspace_blocked"
                        auto_home_error = (
                            "TCP left the application X/Y workspace during auto-home"
                        )
                    else:
                        auto_home_state = "cancelled"
                    robot.cancel_joint_home(auto_home_state, auto_home_error)
                    target_orientation_deg[:] = (
                        startup_orientation_deg[0],
                        startup_orientation_deg[1],
                        wrap_degrees(float(pose.tool_pose_theta_z)),
                    )
                    ref_robot_pose = deepcopy(pose)
                elif auto_home_active:
                    action_state = controller_home_state
                    if action_state == "settling":
                        if joint_velocities_are_settled(
                            current_joint_velocities,
                            len(home_joint_angles),
                            AUTO_HOME_SETTLE_VELOCITY_DEG_S,
                        ):
                            auto_home_settle_samples += 1
                        else:
                            auto_home_settle_samples = 0

                        if auto_home_settle_samples >= AUTO_HOME_SETTLE_SAMPLES:
                            started = robot.start_joint_home(
                                home_joint_angles,
                                current_joint_angles,
                            )
                            (
                                action_state,
                                _home_error,
                                planned_duration_s,
                            ) = robot.get_home_action_status()
                            if started and action_state == "moving":
                                auto_home_action_started_at = now
                                auto_home_action_timeout_s = (
                                    planned_duration_s
                                    + AUTO_HOME_ACTION_TIMEOUT_MARGIN_S
                                )

                    if action_state in ("complete", "aborted"):
                        auto_home_active = False
                        auto_home_state = action_state
                        robot.finish_joint_home()
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
                        auto_home_state = action_state

                elif msg_type == "keyboard":
                    keyboard_vx = clamp(vx, -1.0, 1.0) * MAX_VELOCITY
                    keyboard_vy = clamp(vy, -1.0, 1.0) * MAX_VELOCITY
                    keyboard_vz = clamp(vz, -1.0, 1.0) * MAX_VELOCITY
                    limited_twist = workspace.constrain_motion(
                        pose,
                        keyboard_vx,
                        keyboard_vy,
                        keyboard_vz,
                    )
                    robot.send_cartesian_velocity(
                        *limited_twist,
                        *orientation_twist,
                    )
                    if keyboard_motion_requested:
                        auto_home_state = "idle"
                elif msg_type == "XR":
                    if (
                        abs(vx) < 0.001
                        and abs(vy) < 0.001
                        and abs(vz) < 0.001
                    ):
                        ref_robot_pose = deepcopy(pose)
                        robot.send_cartesian_velocity(
                            0,
                            0,
                            0,
                            *orientation_twist,
                        )
                    else:
                        target_x, target_y = workspace.clamp_target(
                            ref_robot_pose.tool_pose_x + vz,
                            ref_robot_pose.tool_pose_y - vx,
                        )
                        target_z = ref_robot_pose.tool_pose_z + vy

                        x_speed = POSITION_GAIN * (target_x - pose.tool_pose_x)
                        y_speed = POSITION_GAIN * (target_y - pose.tool_pose_y)
                        z_speed = POSITION_GAIN * (target_z - pose.tool_pose_z)

                        limited_twist = workspace.constrain_motion(
                            pose,
                            x_speed,
                            y_speed,
                            z_speed,
                        )
                        robot.send_cartesian_velocity(
                            *limited_twist,
                            *orientation_twist,
                        )
                        auto_home_state = "idle"
                else:
                    robot.send_cartesian_velocity(
                        0,
                        0,
                        0,
                        *orientation_twist,
                    )

                feedback = workspace.feedback(pose)
                (
                    _controller_home_state,
                    auto_home_error,
                    auto_home_planned_duration_s,
                ) = robot.get_home_action_status()
                finite_joint_velocities = tuple(
                    abs(value)
                    for value in current_joint_velocities
                    if isfinite(value)
                )
                feedback["auto_home_state"] = auto_home_state
                feedback["auto_home_joint_error_deg"] = home_joint_error_deg
                feedback["auto_home_error"] = (
                    auto_home_error if auto_home_state != "idle" else ""
                )
                feedback["auto_home_planned_duration_s"] = (
                    auto_home_planned_duration_s
                )
                feedback["auto_home_max_joint_velocity_deg_s"] = max(
                    finite_joint_velocities,
                    default=0.0,
                )
                feedback["orientation_error_deg"] = orientation_error_deg
                feedback["orientation_target_yaw_deg"] = target_orientation_deg[2]
                feedback["tool_yaw_deg"] = float(pose.tool_pose_theta_z)
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
