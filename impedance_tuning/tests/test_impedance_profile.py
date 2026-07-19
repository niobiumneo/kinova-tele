import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from impedance_tuning.impedance_profile import (
    SCHEMA_VERSION,
    conservative_tuning_profile,
    damping_ratios,
    load_impedance_profile,
    load_optional_impedance_profile,
    production_default_profile,
    profile_from_dict,
    profile_to_dict,
    save_impedance_profile,
)


class ImpedanceProfileTests(unittest.TestCase):
    def test_production_defaults_match_existing_controller_defaults(self):
        profile = production_default_profile()

        self.assertEqual(profile.schema_version, SCHEMA_VERSION)
        self.assertEqual(profile.impedance.mass_kg, (4.0, 4.0, 4.0))
        self.assertEqual(profile.impedance.stiffness_n_m, (180.0, 180.0, 220.0))
        self.assertEqual(profile.impedance.damping_n_s_m, (55.0, 55.0, 60.0))
        self.assertEqual(profile.runtime.wrench_frame, "base")

    def test_conservative_profile_is_motion_limited_and_near_critical(self):
        profile = conservative_tuning_profile()
        ratios = damping_ratios(profile.impedance)

        self.assertEqual(profile.impedance.max_displacement_m, (0.01, 0.01, 0.01))
        self.assertEqual(profile.impedance.max_velocity_m_s, 0.01)
        self.assertTrue(all(0.95 <= ratio <= 1.10 for ratio in ratios))

    def test_round_trip_preserves_a_complete_profile(self):
        expected = conservative_tuning_profile()

        encoded = profile_to_dict(expected)
        decoded = profile_from_dict(encoded)

        self.assertNotIn("duration_s", encoded["tuning"])
        self.assertEqual(decoded, expected)

    def test_partial_profile_inherits_production_defaults(self):
        profile = profile_from_dict(
            {
                "schema_version": 1,
                "name": "partial",
                "impedance": {"mass_kg": [5, 6, 7]},
            }
        )

        self.assertEqual(profile.impedance.mass_kg, (5.0, 6.0, 7.0))
        self.assertEqual(profile.impedance.stiffness_n_m, (180.0, 180.0, 220.0))

    def test_unknown_field_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown impedance"):
            profile_from_dict(
                {
                    "schema_version": 1,
                    "name": "typo",
                    "impedance": {"stifness_n_m": [1, 1, 1]},
                }
            )

    def test_boolean_numeric_value_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not a boolean"):
            profile_from_dict(
                {
                    "schema_version": 1,
                    "name": "boolean-mass",
                    "impedance": {"mass_kg": [True, 1, 1]},
                }
            )

    def test_invalid_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            profile_from_dict({"schema_version": 99, "name": "future"})

    def test_non_integer_schema_and_tare_count_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema_version must be an integer"):
            profile_from_dict({"schema_version": 1.0, "name": "float-schema"})
        with self.assertRaisesRegex(ValueError, "tare_samples must be an integer"):
            profile_from_dict(
                {
                    "schema_version": 1,
                    "name": "float-tare",
                    "impedance": {"tare_samples": 30.0},
                }
            )

    def test_invalid_test_envelope_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "peak speed"):
            profile_from_dict(
                {
                    "schema_version": 1,
                    "name": "unsafe-excitation",
                    "tuning": {
                        "excitation_amplitude_m": 0.03,
                        "max_tcp_deviation_m": 0.04,
                        "excitation_frequency_hz": 1.0,
                        "max_test_velocity_m_s": 0.02,
                    },
                }
            )

    def test_save_and_load_are_json_and_preserve_profile(self):
        profile = conservative_tuning_profile()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "profile.json"

            saved_path = save_impedance_profile(path, profile)
            with saved_path.open("r", encoding="utf-8") as handle:
                raw = json.load(handle)
            loaded = load_impedance_profile(path)

        self.assertEqual(raw["schema_version"], 1)
        self.assertEqual(loaded, profile)

    def test_optional_missing_profile_returns_production_defaults(self):
        with TemporaryDirectory() as directory:
            profile, loaded = load_optional_impedance_profile(
                Path(directory) / "missing.json"
            )

        self.assertFalse(loaded)
        self.assertEqual(profile, production_default_profile())


if __name__ == "__main__":
    unittest.main()
