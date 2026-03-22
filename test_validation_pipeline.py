import unittest

from validation import run_validation_pipeline
from validation.report import format_markdown_report


class ValidationPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = run_validation_pipeline()

    def test_report_contains_expected_metrics(self):
        metric_ids = {item["id"] for item in self.report["results"]}
        self.assertIn("galaxy.solar_feh", metric_ids)
        self.assertIn("stellar.solar_teff", metric_ids)
        self.assertIn("planet.earth_surface_pressure_atm", metric_ids)
        self.assertEqual(self.report["summary"]["total"], len(metric_ids))

    def test_report_has_pass_and_fail_results(self):
        self.assertGreater(self.report["summary"]["passed"], 0)
        self.assertGreater(self.report["summary"]["failed"], 0)

    def test_markdown_renderer_mentions_key_sections(self):
        markdown = format_markdown_report(self.report)
        self.assertIn("GalaxyCSM Validation Report", markdown)
        self.assertIn("Largest Mismatches", markdown)
        self.assertIn("galaxy.solar_feh", markdown)
        self.assertIn("planet.earth_surface_pressure_atm", markdown)


if __name__ == "__main__":
    unittest.main()
