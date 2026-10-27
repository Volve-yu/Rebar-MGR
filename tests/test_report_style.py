import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src_report"))
from generate import _build_messages, _build_original_messages, _prompt_hash, _validate_concise_report


class ReportStyleTests(unittest.TestCase):
    def test_v2_keeps_original_protocol_and_adds_optional_summary(self):
        record = {"measurement": {"schema_version": "design_physical_v2",
                  "rebar_count": 3, "spacings_mm": [], "compliance": {}}}
        original = _build_messages(record, "measure_rule")
        self.assertEqual(original, _build_original_messages(record, "measure_rule"))
        concise = _build_messages(record, "measure_rule", "concise")
        self.assertIn("never more than six or 100 words", concise[0]["content"])
        self.assertEqual(original[1], concise[1])
        self.assertNotEqual(_prompt_hash(original), _prompt_hash(concise))

    def test_decimal_values_are_not_sentence_boundaries(self):
        _validate_concise_report(
            "Detected 3 rebars with mean spacing 185.65 mm. "
            "Maximum deviation is 8.32 mm. Failed interval: spacing[0]. "
            "Risk: WARNING; result: FAIL.")

    def test_rejects_truncation_and_excess_length(self):
        for report in ("Detected 3 rebars ...", "Detected 3 rebars\u2026",
                       "A complete sentence. " * 7, "Measured spacing is 185.65",
                       " ".join(["word"] * 101) + "."):
            with self.subTest(report=report), self.assertRaises(ValueError):
                _validate_concise_report(report)


if __name__ == "__main__":
    unittest.main()
