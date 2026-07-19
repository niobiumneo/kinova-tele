import unittest

from impedance_tuning.impedance_controller import (
    CartesianImpedanceConfig,
    CartesianImpedanceController,
)


def make_config(**overrides):
    values = {
        "mass_kg": (1.0, 1.0, 1.0),
        "stiffness_n_m": (25.0, 25.0, 25.0),
        "damping_n_s_m": (10.0, 10.0, 10.0),
        "force_deadband_n": (0.0, 0.0, 0.0),
        "force_axis_sign": (1.0, 1.0, 1.0),
        "max_displacement_m": (0.2, 0.2, 0.2),
        "max_velocity_m_s": 0.5,
        "filter_cutoff_hz": 1000.0,
        "force_limit_n": 10.0,
        "force_release_n": 5.0,
        "contact_haptic_start_n": 1.0,
        "contact_haptic_full_scale_n": 5.0,
        "tare_samples": 0,
        "tare_max_force_n": 2.0,
        "min_dt_s": 0.001,
        "max_dt_s": 0.05,
    }
    values.update(overrides)
    return CartesianImpedanceConfig(**values)


class CartesianImpedanceControllerTests(unittest.TestCase):
    def test_tare_removes_stationary_force_bias(self):
        controller = CartesianImpedanceController(
            make_config(tare_samples=2, tare_max_force_n=4.0),
        )

        first = controller.update((2.0, -1.0, 0.5), 0.01)
        second = controller.update((2.0, -1.0, 0.5), 0.01)
        after_tare = controller.update((2.0, -1.0, 0.5), 0.01)

        self.assertEqual(first.state, "calibrating")
        self.assertEqual(second.state, "active")
        self.assertEqual(controller.force_bias_n, (2.0, -1.0, 0.5))
        self.assertAlmostEqual(after_tare.force_norm_n, 0.0)
        self.assertEqual(after_tare.compliance_velocity_m_s, (0.0, 0.0, 0.0))

    def test_tare_waits_until_allowed(self):
        controller = CartesianImpedanceController(
            make_config(tare_samples=1, tare_max_force_n=4.0),
        )

        waiting = controller.update(
            (1.0, 2.0, 3.0),
            0.01,
            allow_tare=False,
        )
        ready = controller.update((1.0, 2.0, 3.0), 0.01)

        self.assertEqual(waiting.state, "calibrating")
        self.assertEqual(waiting.tare_samples_collected, 0)
        self.assertEqual(ready.state, "active")
        self.assertEqual(controller.force_bias_n, (1.0, 2.0, 3.0))

    def test_tare_rejects_a_loaded_tool(self):
        controller = CartesianImpedanceController(
            make_config(tare_samples=1, tare_max_force_n=5.0),
        )

        loaded = controller.update((6.0, 0.0, 0.0), 0.01)
        ready = controller.update((1.0, 0.0, 0.0), 0.01)

        self.assertEqual(loaded.state, "tare_force_too_high")
        self.assertEqual(loaded.tare_samples_collected, 0)
        self.assertEqual(ready.state, "active")

    def test_force_limit_is_active_before_tare_finishes(self):
        controller = CartesianImpedanceController(
            make_config(
                tare_samples=2,
                tare_max_force_n=5.0,
                force_limit_n=8.0,
                force_release_n=4.0,
            ),
        )

        output = controller.update(
            (9.0, 0.0, 0.0),
            0.01,
            allow_tare=False,
            allow_force_limit_release=False,
        )

        self.assertEqual(output.state, "force_limited")
        self.assertTrue(output.force_limit_active)
        self.assertEqual(output.contact_haptic_intensity, 1.0)

    def test_positive_force_generates_positive_compliance_velocity(self):
        controller = CartesianImpedanceController(make_config())

        output = controller.update((3.0, 0.0, 0.0), 0.01)

        self.assertGreater(output.compliance_velocity_m_s[0], 0.0)
        self.assertGreater(output.compliance_displacement_m[0], 0.0)
        self.assertEqual(output.compliance_velocity_m_s[1:], (0.0, 0.0))

    def test_force_direction_can_be_inverted_per_axis(self):
        controller = CartesianImpedanceController(
            make_config(force_axis_sign=(-1.0, 1.0, 1.0)),
        )

        output = controller.update((3.0, 0.0, 0.0), 0.01)

        self.assertLess(output.compliance_velocity_m_s[0], 0.0)

    def test_native_force_bias_is_removed_before_frame_transform(self):
        controller = CartesianImpedanceController(
            make_config(tare_samples=1, tare_max_force_n=4.0),
        )
        rotate_z_90 = lambda force: (-force[1], force[0], force[2])
        controller.update(
            (1.0, 0.0, 0.0),
            0.01,
            force_transform=rotate_z_90,
        )

        output = controller.update(
            (3.0, 0.0, 0.0),
            0.01,
            force_transform=rotate_z_90,
        )

        self.assertAlmostEqual(output.effective_force_n[0], 0.0)
        self.assertGreater(output.effective_force_n[1], 0.0)
        self.assertGreater(output.compliance_velocity_m_s[1], 0.0)

    def test_deadband_rejects_small_force(self):
        controller = CartesianImpedanceController(
            make_config(force_deadband_n=(1.0, 1.0, 1.0)),
        )

        output = controller.update((0.5, -0.75, 0.25), 0.01)

        self.assertEqual(output.effective_force_n, (0.0, 0.0, 0.0))
        self.assertEqual(output.compliance_velocity_m_s, (0.0, 0.0, 0.0))

    def test_velocity_limiter_prevents_workspace_windup(self):
        controller = CartesianImpedanceController(make_config())

        output = controller.update(
            (3.0, 0.0, 0.0),
            0.01,
            velocity_limiter=lambda velocity: (0.0, velocity[1], velocity[2]),
        )

        self.assertEqual(output.compliance_velocity_m_s[0], 0.0)
        self.assertEqual(output.compliance_displacement_m[0], 0.0)

    def test_force_limit_stays_latched_until_input_is_released(self):
        controller = CartesianImpedanceController(
            make_config(force_limit_n=3.0, force_release_n=1.0),
        )

        limited = controller.update(
            (4.0, 0.0, 0.0),
            0.01,
            allow_force_limit_release=False,
        )
        still_limited = controller.update(
            (0.0, 0.0, 0.0),
            0.01,
            allow_force_limit_release=False,
        )
        released = controller.update(
            (0.0, 0.0, 0.0),
            0.01,
            allow_force_limit_release=True,
        )

        self.assertTrue(limited.force_limit_active)
        self.assertEqual(limited.state, "force_limited")
        self.assertTrue(still_limited.force_limit_active)
        self.assertFalse(released.force_limit_active)
        self.assertEqual(released.state, "active")

    def test_suspension_observes_force_but_clears_compliance_motion(self):
        controller = CartesianImpedanceController(make_config())
        controller.update((2.0, 0.0, 0.0), 0.01)

        suspended = controller.update(
            (2.0, 0.0, 0.0),
            0.01,
            allow_motion=False,
        )

        self.assertEqual(suspended.state, "suspended")
        self.assertGreater(suspended.force_norm_n, 0.0)
        self.assertEqual(suspended.compliance_velocity_m_s, (0.0, 0.0, 0.0))
        self.assertEqual(suspended.compliance_displacement_m, (0.0, 0.0, 0.0))

    def test_virtual_spring_returns_after_force_is_removed(self):
        controller = CartesianImpedanceController(make_config())
        output = None
        for _ in range(20):
            output = controller.update((2.0, 0.0, 0.0), 0.01)
        displacement_under_load = output.compliance_displacement_m[0]

        for _ in range(200):
            output = controller.update((0.0, 0.0, 0.0), 0.01)

        self.assertGreater(displacement_under_load, 0.0)
        self.assertLess(
            abs(output.compliance_displacement_m[0]),
            displacement_under_load,
        )

    def test_missing_wrench_disables_only_the_compliance_layer(self):
        controller = CartesianImpedanceController(make_config())

        output = controller.update(
            (1.0, 0.0, 0.0),
            0.01,
            wrench_available=False,
        )

        self.assertEqual(output.state, "wrench_unavailable")
        self.assertFalse(output.wrench_available)
        self.assertEqual(output.compliance_velocity_m_s, (0.0, 0.0, 0.0))

    def test_feedback_contains_pipeline_diagnostics(self):
        controller = CartesianImpedanceController(make_config())

        feedback = controller.update((2.0, 0.0, 0.0), 0.01).feedback()

        self.assertEqual(feedback["impedance_state"], "active")
        self.assertIn("impedance_force_norm_n", feedback)
        self.assertIn("impedance_velocity_x_m_s", feedback)
        self.assertIn("impedance_contact_haptic_intensity", feedback)

    def test_invalid_config_is_rejected(self):
        with self.assertRaises(ValueError):
            make_config(force_axis_sign=(1.0, 0.0, 1.0))
        with self.assertRaises(ValueError):
            make_config(force_limit_n=5.0, force_release_n=5.0)
        with self.assertRaises(ValueError):
            make_config(tare_samples=1.5)


if __name__ == "__main__":
    unittest.main()
