import unittest

import numpy as np

from impedance_tuning.simulate_impedance import (
    compute_joint_velocity,
    create_parser,
    damped_least_squares,
    rotation_error_vector,
    rotation_matrix_to_euler_xyz_degrees,
    wrapped_joint_error,
)


class SimulationMathTests(unittest.TestCase):
    def test_damped_inverse_and_null_projector_have_expected_shapes(self):
        jacobian = np.hstack((np.eye(6), np.ones((6, 1)) * 0.1))

        inverse, projector = damped_least_squares(jacobian, 0.01)

        self.assertEqual(inverse.shape, (7, 6))
        self.assertEqual(projector.shape, (7, 7))
        self.assertLess(np.linalg.norm(jacobian @ projector), 0.02)

    def test_joint_velocity_is_capped(self):
        jacobian = np.hstack((np.eye(6), np.zeros((6, 1))))
        velocity = compute_joint_velocity(
            jacobian=jacobian,
            task_twist=np.ones(6),
            posture_seed=np.ones(7),
            damping=0.01,
            max_joint_speed_rad_s=0.2,
        )

        self.assertTrue(np.all(np.abs(velocity) <= 0.2))

    def test_rotation_helpers_report_z_rotation(self):
        angle = np.deg2rad(15.0)
        current = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        euler = rotation_matrix_to_euler_xyz_degrees(current)
        error = rotation_error_vector(np.eye(3), current)

        self.assertAlmostEqual(euler[2], 15.0)
        self.assertAlmostEqual(np.rad2deg(error[2]), -15.0)

    def test_wrapped_joint_error_uses_short_path(self):
        target = np.deg2rad(np.array([1.0]))
        current = np.deg2rad(np.array([359.0]))

        error = wrapped_joint_error(target, current)

        self.assertAlmostEqual(np.rad2deg(error[0]), 2.0)

    def test_parser_has_physics_modes_and_no_session_timer(self):
        parser = create_parser()

        args = parser.parse_args(["--mode", "hold"])

        self.assertEqual(args.mode, "hold")
        self.assertFalse(hasattr(args, "duration"))


if __name__ == "__main__":
    unittest.main()
