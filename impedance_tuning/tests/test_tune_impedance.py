import unittest

from impedance_tuning.impedance_controller import CartesianImpedanceController
from impedance_tuning.impedance_profile import conservative_tuning_profile
from impedance_tuning.tune_impedance import (
    RobotSnapshot,
    TuningRobot,
    ZERO_VECTOR,
    build_sample,
    cap_vector_per_axis,
    orientation_distance_degrees,
    smooth_sine_target,
    summarize_samples,
    wrapped_angle_difference_degrees,
)


class FakeTwistValues:
    pass


class FakeTwistCommand:
    def __init__(self):
        self.reference_frame = None
        self.twist = FakeTwistValues()
        self.duration = 0.0


class FakeAdmittance:
    def __init__(self):
        self.admittance_mode = None


class FakeBaseMessages:
    CARTESIAN_REFERENCE_FRAME_BASE = 1
    NULL_SPACE = 3
    DISABLED = 4
    TwistCommand = FakeTwistCommand
    Admittance = FakeAdmittance


class FakeBase:
    def __init__(self):
        self.twists = []
        self.admittance_modes = []
        self.stop_count = 0

    def SendTwistCommand(self, command):
        self.twists.append(command)

    def SetAdmittance(self, command):
        self.admittance_modes.append(command.admittance_mode)

    def Stop(self):
        self.stop_count += 1


def make_snapshot(position=ZERO_VECTOR):
    return RobotSnapshot(
        position_m=position,
        orientation_deg=ZERO_VECTOR,
        linear_velocity_m_s=ZERO_VECTOR,
        angular_velocity_deg_s=ZERO_VECTOR,
        raw_force_n=ZERO_VECTOR,
        raw_torque_nm=ZERO_VECTOR,
        wrench_available=True,
        joint_positions_deg=(),
        joint_velocities_deg_s=(),
        joint_torques_nm=(),
    )


class TuningHelpersTests(unittest.TestCase):
    def test_smooth_sine_starts_without_a_velocity_step(self):
        position, velocity = smooth_sine_target(0.0, 0.01, 0.2)

        self.assertEqual(position, 0.0)
        self.assertEqual(velocity, 0.0)

    def test_smooth_sine_continues_after_startup_ramp(self):
        position, velocity = smooth_sine_target(6.25, 0.01, 0.2)

        self.assertAlmostEqual(position, 0.01)
        self.assertAlmostEqual(velocity, 0.0)

    def test_per_axis_velocity_cap(self):
        self.assertEqual(
            cap_vector_per_axis((0.1, -0.2, 0.005), 0.01),
            (0.01, -0.01, 0.005),
        )

    def test_orientation_distance_handles_wrapped_yaw(self):
        distance = orientation_distance_degrees((0, 0, 179), (0, 0, -179))

        self.assertAlmostEqual(distance, 2.0, places=6)

    def test_joint_delta_wraps_at_360_degrees(self):
        self.assertAlmostEqual(wrapped_angle_difference_degrees(1.0, 359.0), 2.0)

    def test_summary_reports_return_error_and_no_fake_deadband_for_motion_test(self):
        profile = conservative_tuning_profile()
        controller = CartesianImpedanceController(
            profile.impedance,
            enabled=False,
        )
        output = controller.update(ZERO_VECTOR, 0.025)
        initial = make_snapshot()
        displaced = make_snapshot((0.01, 0.0, 0.0))
        sample = build_sample(
            elapsed_s=1.0,
            dt_s=0.025,
            mode="hold",
            output=output,
            snapshot=displaced,
            initial_snapshot=initial,
        )

        summary = summarize_samples(
            [sample],
            mode="hold",
            stop_reason="complete",
            completed=True,
        )

        self.assertAlmostEqual(summary["final_tcp_offset_m"], 0.01)
        self.assertEqual(summary["elapsed_s"], 1.0)
        self.assertIsNone(summary["recommended_deadband_from_this_run_n"])

    def test_robot_wrapper_sets_watchdog_and_disables_nullspace_on_stop(self):
        base = FakeBase()
        robot = TuningRobot("192.168.1.10", 10000, "user", "password")
        robot.base = base
        robot.Base_pb2 = FakeBaseMessages

        robot.send_twist((0.01, -0.02, 0.0), timeout_s=0.075)
        robot.set_native_nullspace_admittance(True)
        robot.stop()

        self.assertEqual(base.twists[0].twist.linear_x, 0.01)
        self.assertEqual(base.twists[0].twist.linear_y, -0.02)
        self.assertEqual(base.twists[0].duration, 0.075)
        self.assertEqual(base.admittance_modes, [3, 4])
        self.assertEqual(base.stop_count, 1)


if __name__ == "__main__":
    unittest.main()
