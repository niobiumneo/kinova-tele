import unittest

from autohome import (
    joint_velocities_are_settled,
    max_joint_error_degrees,
    plan_home_duration_seconds,
    resolve_home_request,
    wrapped_joint_error_degrees,
)


class AutoHomeMathTests(unittest.TestCase):
    def test_wrapped_joint_error_uses_short_path(self):
        self.assertEqual(wrapped_joint_error_degrees(350.0, 10.0), 20.0)
        self.assertEqual(wrapped_joint_error_degrees(10.0, 350.0), -20.0)

    def test_max_joint_error_checks_configuration_size(self):
        with self.assertRaises(ValueError):
            max_joint_error_degrees((0.0,), (0.0, 1.0))

    def test_home_duration_has_acceleration_margin(self):
        duration = plan_home_duration_seconds(
            (0.0, 20.0),
            (30.0, 20.0),
            10.0,
        )
        self.assertEqual(duration, 6.0)

    def test_home_duration_uses_minimum_for_small_move(self):
        duration = plan_home_duration_seconds(
            (0.0,),
            (1.0,),
            10.0,
        )
        self.assertEqual(duration, 2.0)

    def test_settled_velocity_requires_complete_finite_sample(self):
        self.assertTrue(joint_velocities_are_settled((0.1, -0.2), 2, 0.25))
        self.assertFalse(joint_velocities_are_settled((0.1,), 2, 0.25))
        self.assertFalse(joint_velocities_are_settled((0.1, 0.3), 2, 0.25))
        self.assertFalse(
            joint_velocities_are_settled((0.1, float("nan")), 2, 0.25)
        )

    def test_home_request_sequence_is_consumed_once(self):
        last_id, requested = resolve_home_request(None, 0, False)
        self.assertEqual((last_id, requested), (0, False))

        last_id, requested = resolve_home_request(last_id, 1, True)
        self.assertEqual((last_id, requested), (1, True))

        last_id, requested = resolve_home_request(last_id, 1, True)
        self.assertEqual((last_id, requested), (1, False))

    def test_pending_first_request_survives_connection_start(self):
        self.assertEqual(
            resolve_home_request(None, 3, True),
            (3, True),
        )

    def test_legacy_home_request_without_id_still_works(self):
        self.assertEqual(
            resolve_home_request(4, None, True),
            (4, True),
        )


if __name__ == "__main__":
    unittest.main()
