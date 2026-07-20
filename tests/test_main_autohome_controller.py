import importlib
import sys
import types
import unittest


def _install_module(name, *, package=False):
    module = types.ModuleType(name)
    if package:
        module.__path__ = []
    sys.modules[name] = module
    return module


class _Repeated(list):
    def __init__(self, factory):
        super().__init__()
        self._factory = factory

    def add(self):
        value = self._factory()
        self.append(value)
        return value


class _AngularWaypoint:
    def __init__(self):
        self.angles = []
        self.duration = 0.0


class _Waypoint:
    def __init__(self):
        self.name = ""
        self.angular_waypoint = _AngularWaypoint()


class _WaypointList:
    def __init__(self):
        self.duration = 0.0
        self.use_optimal_blending = False
        self.waypoints = _Repeated(_Waypoint)


class _EnumNames:
    names = {
        3: "TRAJECTORY_ERROR_TYPE_INVALID_DURATION",
        58: "CONTROL_INVALID_DURATION",
    }

    @classmethod
    def Name(cls, value):
        return cls.names[int(value)]


def _install_import_stubs():
    uvicorn = _install_module("uvicorn")
    uvicorn.run = lambda *args, **kwargs: None

    fastapi = _install_module("fastapi", package=True)

    class FastAPI:
        def __init__(self, *args, **kwargs):
            pass

        @staticmethod
        def _decorator(*args, **kwargs):
            return lambda function: function

        get = _decorator
        websocket = _decorator

    fastapi.FastAPI = FastAPI
    fastapi.WebSocket = type("WebSocket", (), {})
    fastapi.WebSocketDisconnect = type("WebSocketDisconnect", (Exception,), {})
    responses = _install_module("fastapi.responses")
    responses.FileResponse = type("FileResponse", (), {})

    _install_module("kortex_api", package=True)
    tcp = _install_module("kortex_api.TCPTransport")
    tcp.TCPTransport = type("TCPTransport", (), {})
    router = _install_module("kortex_api.RouterClient")
    router.RouterClient = type("RouterClient", (), {})
    session_manager = _install_module("kortex_api.SessionManager")
    session_manager.SessionManager = type("SessionManager", (), {})
    _install_module("kortex_api.autogen", package=True)
    _install_module("kortex_api.autogen.client_stubs", package=True)
    base_client = _install_module(
        "kortex_api.autogen.client_stubs.BaseClientRpc"
    )
    base_client.BaseClient = type("BaseClient", (), {})
    cyclic_client = _install_module(
        "kortex_api.autogen.client_stubs.BaseCyclicClientRpc"
    )
    cyclic_client.BaseCyclicClient = type("BaseCyclicClient", (), {})
    messages = _install_module("kortex_api.autogen.messages", package=True)
    session_pb2 = _install_module("kortex_api.autogen.messages.Session_pb2")
    base_pb2 = _install_module("kortex_api.autogen.messages.Base_pb2")
    messages.Session_pb2 = session_pb2
    messages.Base_pb2 = base_pb2

    base_pb2.CARTESIAN_REFERENCE_FRAME_BASE = 0
    base_pb2.EXECUTE_WAYPOINT_LIST = 41
    base_pb2.ACTION_END = 1
    base_pb2.ACTION_ABORT = 2
    base_pb2.ACTION_PREPROCESS_ABORT = 6
    base_pb2.WaypointList = _WaypointList
    base_pb2.NotificationOptions = type("NotificationOptions", (), {})
    base_pb2.TrajectoryErrorType = _EnumNames
    base_pb2.SubErrorCodes = _EnumNames


_install_import_stubs()
main = importlib.import_module("main")


class _Validation:
    def __init__(self, errors=()):
        self.trajectory_error_report = types.SimpleNamespace(
            trajectory_error_elements=list(errors)
        )


class _FakeBase:
    def __init__(self, errors=()):
        self.errors = tuple(errors)
        self.stop_calls = 0
        self.executed_waypoint_list = None
        self.unsubscribed = []

    def Stop(self):
        self.stop_calls += 1

    def ValidateWaypointList(self, waypoint_list):
        return _Validation(self.errors)

    def OnNotificationActionTopic(self, callback, options):
        self.callback = callback
        return "subscription"

    def ExecuteWaypointTrajectory(self, waypoint_list):
        self.executed_waypoint_list = waypoint_list

    def Unsubscribe(self, handle):
        self.unsubscribed.append(handle)


class KinovaAutoHomeControllerTests(unittest.TestCase):
    def make_controller(self, errors=()):
        controller = main.KinovaController()
        controller.base = _FakeBase(errors)
        return controller

    def test_prepare_stops_cartesian_control_before_settling(self):
        controller = self.make_controller()
        self.assertTrue(controller.prepare_joint_home())
        self.assertEqual(controller.base.stop_calls, 1)
        self.assertEqual(controller.get_home_action_state(), "settling")

    def test_start_builds_timed_waypoint_and_enters_moving(self):
        controller = self.make_controller()
        self.assertTrue(controller.prepare_joint_home())
        self.assertTrue(controller.start_joint_home((30.0, 20.0), (0.0, 20.0)))
        waypoint_list = controller.base.executed_waypoint_list
        waypoint = waypoint_list.waypoints[0]
        self.assertEqual(tuple(waypoint.angular_waypoint.angles), (30.0, 20.0))
        self.assertEqual(waypoint.angular_waypoint.duration, 6.0)
        state, error, duration = controller.get_home_action_status()
        self.assertEqual((state, error, duration), ("moving", "", 6.0))

    def test_already_home_completes_without_trajectory(self):
        controller = self.make_controller()
        self.assertFalse(controller.start_joint_home((0.1,), (0.0,)))
        self.assertEqual(controller.get_home_action_state(), "complete")
        self.assertIsNone(controller.base.executed_waypoint_list)

    def test_invalid_joint_feedback_is_reported_without_starting(self):
        controller = self.make_controller()
        self.assertFalse(controller.start_joint_home((0.0, 1.0), (0.0,)))
        state, detail, _duration = controller.get_home_action_status()
        self.assertEqual(state, "aborted")
        self.assertIn("different sizes", detail)
        self.assertIsNone(controller.base.executed_waypoint_list)

    def test_validation_reason_is_preserved(self):
        error = types.SimpleNamespace(
            error_type=3,
            message="duration is too short",
            waypoint_index=0,
            index=2,
            error_value=1.0,
            min_value=2.0,
            max_value=45.0,
        )
        controller = self.make_controller((error,))
        self.assertFalse(controller.start_joint_home((30.0,), (0.0,)))
        state, detail, _duration = controller.get_home_action_status()
        self.assertEqual(state, "aborted")
        self.assertIn("TRAJECTORY_ERROR_TYPE_INVALID_DURATION", detail)
        self.assertIn("duration is too short", detail)

    def test_notifications_are_filtered_and_abort_reason_is_reported(self):
        controller = self.make_controller()
        controller.start_joint_home((30.0,), (0.0,))
        unrelated = types.SimpleNamespace(
            action_event=main.Base_pb2.ACTION_END,
            handle=types.SimpleNamespace(action_type=999),
            abort_details=0,
        )
        controller._on_home_action_notification(unrelated)
        self.assertEqual(controller.get_home_action_state(), "moving")

        aborted = types.SimpleNamespace(
            action_event=main.Base_pb2.ACTION_ABORT,
            handle=types.SimpleNamespace(action_type=41),
            abort_details=58,
        )
        controller._on_home_action_notification(aborted)
        state, detail, _duration = controller.get_home_action_status()
        self.assertEqual(state, "aborted")
        self.assertIn("CONTROL_INVALID_DURATION (58)", detail)

    def test_late_cancel_notification_preserves_cancel_reason(self):
        controller = self.make_controller()
        controller.start_joint_home((30.0,), (0.0,))
        controller.cancel_joint_home("timeout", "planned timeout expired")

        late_abort = types.SimpleNamespace(
            action_event=main.Base_pb2.ACTION_ABORT,
            handle=types.SimpleNamespace(action_type=41),
            abort_details=58,
        )
        controller._on_home_action_notification(late_abort)
        state, detail, _duration = controller.get_home_action_status()
        self.assertEqual(state, "timeout")
        self.assertEqual(detail, "planned timeout expired")


class WorkspaceToleranceTests(unittest.TestCase):
    def test_auto_home_can_allow_small_boundary_feedback_noise(self):
        startup = types.SimpleNamespace(tool_pose_x=0.0, tool_pose_y=0.0)
        workspace = main.PlanarWorkspace(startup)
        pose = types.SimpleNamespace(
            tool_pose_x=workspace.x_max + 0.004,
            tool_pose_y=0.0,
        )
        self.assertFalse(workspace.contains(pose))
        self.assertTrue(workspace.contains(pose, tolerance_m=0.005))


if __name__ == "__main__":
    unittest.main()
