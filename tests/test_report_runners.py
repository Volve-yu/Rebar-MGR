import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


class ReportRunnerTests(unittest.TestCase):
    def test_full_input_merge_and_text_only_preflight(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            image = base / "image.png"
            Image.new("RGB", (8, 8), "white").save(image)
            measurement = {"schema_version": "design_physical_v2", "rebar_count": 2,
                           "spacings_mm": [100.0], "compliance": {"status": "N/A"}}
            row = {"subset": "site-1", "id": "1", "schema_version": "design_physical_v2",
                   "image_path": str(image), "seg_path": str(image),
                   "measurement": measurement, "reference_report": "Reference."}
            input_root = base / "inputs"
            qaep_root = base / "qaep"
            write_jsonl(input_root / "site-1_full_seed42/within/report_inputs.jsonl", [row])
            qaep_row = dict(row, setting="rgb_seg_measure_rule_qaep", prediction="Draft.",
                            qaep={"plan_text": "Describe the supplied evidence."})
            write_jsonl(qaep_root / "site-1/generations/rgb_seg_measure_rule_qaep.jsonl",
                        [qaep_row])

            merged = base / "merged"
            subprocess.run([sys.executable, str(ROOT / "scripts/prepare_full_report_inputs.py"),
                            "--inputs-root", str(input_root), "--qaep-root", str(qaep_root),
                            "--out-dir", str(merged), "--sites", "site-1"],
                           cwd=ROOT, check=True, capture_output=True, text=True)
            saved = json.loads((merged / "site-1_full_inputs_qaep.jsonl").read_text())
            self.assertEqual(saved["qaep"], qaep_row["qaep"])

            paired_root = base / "paired"
            for setting in ("rgb_measure", "rgb_measure_rule"):
                write_jsonl(paired_root / "site-1/generations" / (setting + ".jsonl"),
                            [dict(row, model="test-model", setting=setting,
                                  prediction="Paired report.")])
            result = subprocess.run(
                [sys.executable, str(ROOT / "scripts/run_measurement_only_reports.py"),
                 "--inputs-root", str(input_root), "--paired-root", str(paired_root),
                 "--out", str(base / "outputs"), "--sites", "site-1", "--model", "test-model"],
                cwd=ROOT, check=True, capture_output=True, text=True)
            self.assertIn("2 requests", result.stdout)
            self.assertFalse((base / "outputs").exists())


if __name__ == "__main__":
    unittest.main()
