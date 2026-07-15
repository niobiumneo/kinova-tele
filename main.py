import cv2
import json
import threading
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

gain = 2

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
        """vx, vy, vz are normalized -1..1 commands (e.g. from a joystick or the
        hand-clutch offset in index.html), scaled here into real m/s."""
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
# Initialize Robot Controller
robot = KinovaController()
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def on_startup():
    robot.connect()


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
    ref_robot_pose = deepcopy(robot.get_current_robot_pose())
    last_gripper_position = None
    last_gripper_command_time = 0.0
    try:
        last_motion_message_time = time()
        watchdog_stopped_motion = False
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
                    watchdog_stopped_motion = True
                continue
            
            payload = json.loads(data)
            msg_type = payload.get("msg", None)

            gripper_position = payload.get("gripper_position")
            if isinstance(gripper_position, (int, float)) and not isinstance(
                gripper_position,
                bool,
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

            if msg_type != None:
                vx = payload.get("vx", 0.0)
                vy = payload.get("vy", 0.0)
                vz = payload.get("vz", 0.0)
                if  msg_type == "keyboard":
                    robot.send_cartesian_velocity(vx, vy, vz)
                else:
                    pose = deepcopy(robot.get_current_robot_pose())
                    if ((abs(vx) < 0.001) and (abs(vy) < 0.001) and (abs(vz) < 0.001)):
                        # print("/n/n/n/n/n/nHI!")
                        ref_robot_pose = deepcopy(pose)
                    elif msg_type == "XR":

                        x_speed = gain*((ref_robot_pose.tool_pose_x + vz) - pose.tool_pose_x)
                        y_speed = gain*((ref_robot_pose.tool_pose_y - vx) - pose.tool_pose_y)
                        z_speed = gain*((ref_robot_pose.tool_pose_z + vy) - pose.tool_pose_z)

                        # print(("X:", ref_robot_pose.tool_pose_x + vz, pose.tool_pose_x, vz, x_speed))
                        # print(("Y:", ref_robot_pose.tool_pose_y - vx, pose.tool_pose_y, -vx, y_speed))
                        # print(("Z:", ref_robot_pose.tool_pose_z + vy, pose.tool_pose_z, vy, z_speed))

                        robot.send_cartesian_velocity(x_speed, y_speed, z_speed)
                    else:
                        robot.send_cartesian_velocity(0,0,0)
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
