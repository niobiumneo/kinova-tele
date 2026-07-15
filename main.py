import cv2
import json
import threading
from math import isfinite, sqrt
from time import time
import uvicorn
import numpy as np
import asyncio
from copy import deepcopy
import pyrealsense2 as rs
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# Pure Python Kinova Kortex API Imports
from kortex_api.TCPTransport import TCPTransport
from kortex_api.RouterClient import RouterClient
from kortex_api.SessionManager import SessionManager
from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
from kortex_api.autogen.client_stubs.BaseCyclicClientRpc import BaseCyclicClient
from kortex_api.autogen.messages import Session_pb2, Base_pb2

# --- CONFIGURATION ---
ROBOT_IP = "192.168.1.10"
ROBOT_PORT = 10000
USERNAME = "admin"
PASSWORD = "admin"

# --- SAFETY SETTINGS ---
VELOCITY_SCALE = 1.0
MAX_VELOCITY = 0.15  # m/s, per axis. Keep this low until you've verified directions.
# Change this line to set the origin to the robot's gripper
REFERENCE_FRAME = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE

# The robot base X/Y plane is assumed to be parallel to the work surface, with
# +X pointing forward and Y running across the table. These fixed base-frame
# coordinates put a 4 ft deep by 4 ft wide rectangle directly in front of the
# base. Adjust X_MIN or Y_CENTER to match the real table before operating.
# This application-level limiter is not a safety-rated substitute for matching
# Kortex protection zones or an external safety system.
FEET_TO_METERS = 0.3048
WORKSPACE_SIZE_X_M = 4.0 * FEET_TO_METERS
WORKSPACE_SIZE_Y_M = 4.0 * FEET_TO_METERS
WORKSPACE_X_MIN_M = 0.0
WORKSPACE_Y_CENTER_M = 0.0
WORKSPACE_SLOWDOWN_DISTANCE_M = 0.10
WORKSPACE_HAPTIC_THRESHOLD_M = 0.10

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
AUTO_HOME_POSITION_GAIN = 1.5
AUTO_HOME_MAX_VELOCITY_M_S = 0.08
AUTO_HOME_POSITION_TOLERANCE_M = 0.01
AUTO_HOME_TIMEOUT_S = 20.0


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


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


def auto_home_velocity(pose, target_pose):
    """Return a speed-limited straight-line velocity toward the startup pose."""
    error_x = target_pose.tool_pose_x - pose.tool_pose_x
    error_y = target_pose.tool_pose_y - pose.tool_pose_y
    error_z = target_pose.tool_pose_z - pose.tool_pose_z
    distance = sqrt(error_x**2 + error_y**2 + error_z**2)

    if distance <= AUTO_HOME_POSITION_TOLERANCE_M:
        return 0.0, 0.0, 0.0, distance

    speed_scale = min(
        AUTO_HOME_POSITION_GAIN,
        AUTO_HOME_MAX_VELOCITY_M_S / distance,
    )
    return (
        error_x * speed_scale,
        error_y * speed_scale,
        error_z * speed_scale,
        distance,
    )


class PlanarWorkspace:
    """Server-side rectangular X/Y workspace constraint."""

    def __init__(self):
        half_y = WORKSPACE_SIZE_Y_M / 2.0

        self.x_min = WORKSPACE_X_MIN_M
        self.x_max = WORKSPACE_X_MIN_M + WORKSPACE_SIZE_X_M
        self.y_min = WORKSPACE_Y_CENTER_M - half_y
        self.y_max = WORKSPACE_Y_CENTER_M + half_y

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
        }

    def describe(self):
        return (
            f"X=[{self.x_min:.3f}, {self.x_max:.3f}] m, "
            f"Y=[{self.y_min:.3f}, {self.y_max:.3f}] m"
        )


class KinovaController:
    def __init__(self):
        self.transport = None
        self.router = None
        self.base = None
        self.session_manager = None

    def connect(self):
        try:
            print(f"Connecting directly to Kinova Gen3 at {ROBOT_IP}...")
            self.transport = TCPTransport()
            error_callback = lambda kException: print(f"API Error: {kException}")
            self.router = RouterClient(self.transport, error_callback)
            self.transport.connect(ROBOT_IP, ROBOT_PORT)

            session_info = Session_pb2.CreateSessionInfo()
            session_info.username = USERNAME
            session_info.password = PASSWORD
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

    def send_cartesian_velocity(self, vx, vy, vz):
        """Send a base-frame Cartesian twist after applying final speed caps."""
        if not self.base:
            return

        safe_vx = max(-MAX_VELOCITY, min(MAX_VELOCITY, vx * VELOCITY_SCALE))
        safe_vy = max(-MAX_VELOCITY, min(MAX_VELOCITY, vy * VELOCITY_SCALE))
        safe_vz = max(-MAX_VELOCITY, min(MAX_VELOCITY, vz * VELOCITY_SCALE))

        # print(
        #     f"Commanding -> X: {safe_vx:.4f} m/s | Y: {safe_vy:.4f} m/s | Z: {safe_vz:.4f} m/s",
        #     end="\r",
        # )

        command = Base_pb2.TwistCommand()
        command.reference_frame = REFERENCE_FRAME
        command.twist.linear_x = safe_vx
        command.twist.linear_y = safe_vy
        command.twist.linear_z = safe_vz
        command.twist.angular_x = 0.0
        command.twist.angular_y = 0.0
        command.twist.angular_z = 0.0

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

    def get_current_robot_pose(self):
        """
        Fetches and prints the current Cartesian pose of the Kinova robot.
        """
        # Get the cyclic feedback (this contains the real-time feedback)
        feedback = self.baseCyclic.RefreshFeedback()
        
        # Extract Cartesian pose (end-effector)
        base = feedback.base
        # x = cartesian_pose.x
        # y = cartesian_pose.y
        # z = cartesian_pose.z
        # theta_x = cartesian_pose.theta_x
        # theta_y = cartesian_pose.theta_y
        # theta_z = cartesian_pose.theta_z
        
        # print(f"Current Position (m): X = {x:.3f}, Y = {y:.3f}, Z = {z:.3f}")
        # print(f"Current Orientation (deg): ThetaX = {theta_x:.3f}, ThetaY = {theta_y:.3f}, ThetaZ = {theta_z:.3f}")
        
        return base

    def stop(self):
        if self.base:
            try:
                self.base.Stop()
            except Exception:
                pass


# Initialize Robot Controller
robot = KinovaController()
workspace = None
home_pose = None
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def on_startup():
    global home_pose, workspace
    robot.connect()
    try:
        initial_pose = deepcopy(robot.get_current_robot_pose())
        workspace = PlanarWorkspace()
        if not workspace.contains(initial_pose):
            raise RuntimeError(
                "tool is outside the configured X/Y workspace; "
                f"tool=({initial_pose.tool_pose_x:.3f}, "
                f"{initial_pose.tool_pose_y:.3f}) m, "
                f"workspace={workspace.describe()}"
            )
        home_pose = deepcopy(initial_pose)
        print(f"Planar workspace latched: {workspace.describe()}")
        print(
            "Auto-home startup position: "
            f"X={home_pose.tool_pose_x:.3f}, "
            f"Y={home_pose.tool_pose_y:.3f}, "
            f"Z={home_pose.tool_pose_z:.3f} m"
        )
    except Exception as e:
        workspace = None
        home_pose = None
        print(f"Failed to latch planar workspace: {e}")


@app.on_event("shutdown")
def on_shutdown():
    robot.stop()


@app.get("/")
async def get_index():
    return FileResponse("index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("\n[WS] Quest client connected successfully!")
    if workspace is None or home_pose is None:
        await websocket.close(code=1011, reason="Robot workspace is unavailable")
        return

    ref_robot_pose = deepcopy(robot.get_current_robot_pose())
    last_gripper_position = None
    last_gripper_command_time = 0.0
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
                    robot.send_cartesian_velocity(0, 0, 0)
                    auto_home_active = False
                    if auto_home_state == "moving":
                        auto_home_state = "cancelled"
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
                grip_pressed = payload.get("grip_pressed") is True
                home_requested = payload.get("home_request") is True
                xr_presenting = payload.get("xr_presenting") is True
                controller_present = payload.get("controller_present") is True
                pose = deepcopy(robot.get_current_robot_pose())
                home_distance = sqrt(
                    (home_pose.tool_pose_x - pose.tool_pose_x) ** 2
                    + (home_pose.tool_pose_y - pose.tool_pose_y) ** 2
                    + (home_pose.tool_pose_z - pose.tool_pose_z) ** 2
                )

                if home_requested:
                    auto_home_active = True
                    auto_home_state = "moving"
                    auto_home_started_at = time()
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
                auto_home_timed_out = (
                    auto_home_active
                    and time() - auto_home_started_at > AUTO_HOME_TIMEOUT_S
                )

                if cancel_auto_home or auto_home_timed_out:
                    auto_home_active = False
                    auto_home_state = (
                        "timeout" if auto_home_timed_out else "cancelled"
                    )
                    robot.send_cartesian_velocity(0, 0, 0)
                elif auto_home_active:
                    home_vx, home_vy, home_vz, home_distance = auto_home_velocity(
                        pose,
                        home_pose,
                    )
                    if home_distance <= AUTO_HOME_POSITION_TOLERANCE_M:
                        robot.send_cartesian_velocity(0, 0, 0)
                        auto_home_active = False
                        auto_home_state = "complete"
                    else:
                        limited_twist = workspace.constrain_motion(
                            pose,
                            home_vx,
                            home_vy,
                            home_vz,
                        )
                        robot.send_cartesian_velocity(*limited_twist)
                        auto_home_state = "moving"

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
                    robot.send_cartesian_velocity(*limited_twist)
                    if keyboard_motion_requested:
                        auto_home_state = "idle"
                elif msg_type == "XR":
                    if (
                        abs(vx) < 0.001
                        and abs(vy) < 0.001
                        and abs(vz) < 0.001
                    ):
                        ref_robot_pose = deepcopy(pose)
                        robot.send_cartesian_velocity(0, 0, 0)
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
                        robot.send_cartesian_velocity(*limited_twist)
                        auto_home_state = "idle"
                else:
                    robot.send_cartesian_velocity(0, 0, 0)

                feedback = workspace.feedback(pose)
                feedback["auto_home_state"] = auto_home_state
                feedback["auto_home_distance_m"] = home_distance
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
    """Runs the FastAPI server in a background thread."""
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="error",
        ssl_keyfile="key.pem",
        ssl_certfile="cert.pem",
    )


if __name__ == "__main__":
    local_ip = "192.168.1.70"
    print("\n" + "=" * 50)
    print(" SERVER RUNNING (HTTPS/WSS REQUIRED FOR WEBXR) ")
    print(f" Quest Browser URL: https://{local_ip}:8000")
    print(" On first visit, accept the self-signed cert warning")
    print(" (Advanced -> Proceed) BEFORE trying to enter AR.")
    print("=" * 50 + "\n")

    # 1. Start FastAPI server in a background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()

    # 2. Native RealSense Pipeline (Bypasses /dev/media0)
    try:
        pipeline = rs.pipeline()
        config = rs.config()

        # Enable the standard color stream
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        print("RealSense camera active! Press 'Q' in the window to exit.")

        while True:
            # Wait for a coherent frame from the camera
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            # Convert the RealSense frame to an array OpenCV can display
            color_image = np.asanyarray(color_frame.get_data())

            cv2.imshow("Host PC - RealSense View", color_image)

            # Press 'q' to close the window
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"\nRealSense Pipeline failed: {e}")
        print("Check your USB cable! RealSense requires a fast USB 3.0 port.")
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
        cv2.destroyAllWindows()





# keyboard in VR




# import cv2
# import json
# import threading
# import uvicorn
# import numpy as np
# import pyrealsense2 as rs
# from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# from fastapi.responses import FileResponse

# # Pure Python Kinova Kortex API Imports
# from kortex_api.TCPTransport import TCPTransport
# from kortex_api.RouterClient import RouterClient
# from kortex_api.SessionManager import SessionManager
# from kortex_api.autogen.client_stubs.BaseClientRpc import BaseClient
# from kortex_api.autogen.messages import Session_pb2, Base_pb2

# # --- CONFIGURATION ---
# ROBOT_IP = "192.168.1.10"
# ROBOT_PORT = 10000
# USERNAME = "admin"
# PASSWORD = "admin"

# # --- SAFETY SETTINGS ---
# VELOCITY_SCALE = 1.0
# MAX_VELOCITY = 0.15
# REFERENCE_FRAME = Base_pb2.CARTESIAN_REFERENCE_FRAME_BASE

# class KinovaController:
#     def __init__(self):
#         self.transport = None
#         self.router = None
#         self.base = None
#         self.session_manager = None

#     def connect(self):
#         try:
#             print(f"Connecting directly to Kinova Gen3 at {ROBOT_IP}...")
#             self.transport = TCPTransport()
#             error_callback = lambda kException: print(f"API Error: {kException}")
#             self.router = RouterClient(self.transport, error_callback)
#             self.transport.connect(ROBOT_IP, ROBOT_PORT)

#             session_info = Session_pb2.CreateSessionInfo()
#             session_info.username = USERNAME
#             session_info.password = PASSWORD
#             session_info.session_inactivity_timeout = 60000
#             session_info.connection_inactivity_timeout = 2000

#             self.session_manager = SessionManager(self.router)
#             self.session_manager.CreateSession(session_info)
#             self.base = BaseClient(self.router)

#             try:
#                 self.base.ClearFaults()
#             except Exception:
#                 pass

#             try:
#                 servo_mode = Base_pb2.ServoingModeInformation()
#                 servo_mode.servoing_mode = Base_pb2.SINGLE_LEVEL_SERVOING
#                 self.base.SetServoingMode(servo_mode)
#                 print("Set to SINGLE_LEVEL_SERVOING.")
#             except Exception as e:
#                 print(f"SetServoingMode failed: {e}")

#             print("Successfully linked to Kortex API. Robot Ready.")
#         except Exception as e:
#             print(f"Hardware connection failed: {e}")

#     def send_planar_velocity(self, vx, vy):
#         if not self.base:
#             return

#         safe_vx = max(-MAX_VELOCITY, min(MAX_VELOCITY, vx * VELOCITY_SCALE))
#         safe_vy = max(-MAX_VELOCITY, min(MAX_VELOCITY, vy * VELOCITY_SCALE))
#         print(f"Commanding -> X: {safe_vx:.4f} m/s | Y: {safe_vy:.4f} m/s", end="\r")

#         command = Base_pb2.TwistCommand()
#         command.reference_frame = REFERENCE_FRAME
#         command.twist.linear_x = safe_vx
#         command.twist.linear_y = safe_vy
#         command.twist.linear_z = 0.0
#         command.twist.angular_x = 0.0
#         command.twist.angular_y = 0.0
#         command.twist.angular_z = 0.0

#         try:
#             self.base.SendTwistCommand(command)
#         except Exception as e:
#             print(f"\nSendTwistCommand failed: {e}")

#     def stop(self):
#         if self.base:
#             try:
#                 self.base.Stop()
#             except Exception:
#                 pass

# # Initialize Robot Controller
# robot = KinovaController()
# app = FastAPI()

# @app.on_event("startup")
# def on_startup():
#     robot.connect()

# @app.on_event("shutdown")
# def on_shutdown():
#     robot.stop()

# @app.get("/")
# async def get_index():
#     return FileResponse("index.html")

# @app.websocket("/ws")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     print("\n[WS] Quest 2 Client connected successfully!")
#     try:
#         while True:
#             data = await websocket.receive_text()
#             payload = json.loads(data)
#             vx = payload.get("vx", 0.0)
#             vy = payload.get("vy", 0.0)
#             robot.send_planar_velocity(vx, vy)
#     except WebSocketDisconnect:
#         print("\n[WS] Client disconnected. Stopping robot.")
#         robot.stop()
#     except Exception as e:
#         print(f"\n[WS] Handler error: {e}")
#         robot.stop()

# def start_server():
#     """Runs the FastAPI server in a background thread."""
#     uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")


# if __name__ == "__main__":
#     local_ip = "192.168.1.70" 
#     print("\n" + "=" * 50)
#     print(f" SERVER RUNNING OFFLINE (PURE HTTP) ")
#     print(f" Quest 2 Browser URL: http://{local_ip}:8000")
#     print("=" * 50 + "\n")
    
#     # 1. Start FastAPI server in a background thread
#     server_thread = threading.Thread(target=start_server, daemon=True)
#     server_thread.start()

#     # 2. Native RealSense Pipeline (Bypasses /dev/media0)
#     try:
#         pipeline = rs.pipeline()
#         config = rs.config()
        
#         # Enable the standard color stream
#         config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
#         pipeline.start(config)
        
#         print("RealSense camera active! Press 'Q' in the window to exit.")
        
#         while True:
#             # Wait for a coherent frame from the camera
#             frames = pipeline.wait_for_frames()
#             color_frame = frames.get_color_frame()
            
#             if not color_frame:
#                 continue
                
#             # Convert the RealSense frame to an array OpenCV can display
#             color_image = np.asanyarray(color_frame.get_data())
            
#             cv2.imshow("Host PC - RealSense View", color_image)
            
#             # Press 'q' to close the window
#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break
                
#     except Exception as e:
#         print(f"\nRealSense Pipeline failed: {e}")
#         print("Check your USB cable! RealSense requires a fast USB 3.0 port.")
#     finally:
#         try:
#             pipeline.stop()
#         except:
#             pass
#         cv2.destroyAllWindows()
