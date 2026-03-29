import unittest

from validation.run_observational_validation import build_report


class OpenClusterValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.full_report = build_report()
        cls.report = cls.full_report["observational_validation"]

    def test_effective_migration_kernel_improves_fit(self):
        comp = self.report["comparison"]
        self.assertLess(comp["mean_abs_residual_dex"], comp["gas_only_mean_abs_residual_dex"])
        self.assertLess(comp["rmse_dex"], comp["gas_only_rmse_dex"])

    def test_report_exposes_raw_and_migration_adjusted_profiles(self):
        sim = self.report["simulation"]
        migration = self.report["migration_model"]
        bins = self.report["profile"]["binned_profile"]

        self.assertIn("sampled_gas_slope_dex_per_kpc", sim)
        self.assertIn("migration_adjusted_profile_feh", sim)
        self.assertIn("gas_profile_feh", sim)
        self.assertTrue(migration["enabled"])
        self.assertGreater(len(bins), 0)
        self.assertIn("simulated_feh_raw", bins[0])

    def test_report_exposes_stellar_kinematic_validation(self):
        kin = self.full_report["stellar_kinematic_validation"]
        self.assertIn("overall", kin)
        self.assertIn("age_binned_stats", kin)
        self.assertGreater(len(kin["age_binned_stats"]), 0)
        self.assertIn("avr", kin)
        self.assertIn("eccentricity", kin)
        self.assertIn("feh_guiding_radius", kin)

        self.assertGreater(kin["avr"]["sigma_R_age_correlation"], 0.0)
        self.assertGreater(kin["eccentricity"]["age_correlation"], 0.0)
        self.assertLess(kin["feh_guiding_radius"]["young"]["slope_dex_per_kpc"], 0.0)
        self.assertIn(
            kin["overall"]["status"],
            {"good", "mixed", "tension", "unknown"},
        )


if __name__ == "__main__":
    unittest.main()
