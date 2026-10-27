"""Run text-only controls paired with the revised RGB report ablations."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import shutil
from datetime import datetime

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src_report"))
from generate import _build_messages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-api", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model", default="qwen3.8-omni-flash")
    parser.add_argument("--base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--inputs-root", type=Path, required=True,
                        help="Directory with <site>_full_seed42/within/report_inputs.jsonl")
    parser.add_argument("--paired-root", type=Path, required=True,
                        help="Directory with <site>/generations/rgb_measure*.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sites", nargs="+", default=["site-1", "site-2"])
    args = parser.parse_args()
    output = args.out
    if args.limit is not None:
        output = output / ("smoke_" + str(args.limit))
    jobs = []
    request_count = 0
    for site in args.sites:
        source = args.inputs_root / f"{site}_full_seed42/within/report_inputs.jsonl"
        rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not rows:
            raise ValueError(f"{site}: no input records")
        request_count += 2 * len(rows)
        for setting, paired in (("measure_only", "rgb_measure"), ("measure_rule", "rgb_measure_rule")):
            existing = args.paired_root / site / "generations" / f"{paired}.jsonl"
            paired_rows = [json.loads(line) for line in existing.read_text(encoding="utf-8").splitlines() if line.strip()]
            if {(r['subset'], r['id']) for r in rows} != {(r['subset'], r['id']) for r in paired_rows}:
                raise ValueError(f"Sample IDs differ: {existing}")
            if any(r.get("model") != args.model or r.get("report_style", "original") != "original" for r in paired_rows):
                raise ValueError(f"Model/style differs from paired RGB outputs: {existing}")
            paired_by_id = {(r['subset'], r['id']): r for r in paired_rows}
            for row in rows:
                previous = paired_by_id[(row['subset'], row['id'])]
                if any(row.get(key) != previous.get(key) for key in ("measurement", "reference_report")):
                    raise ValueError(f"Evidence/reference differs from RGB output: {site}/{row['id']}")
                messages = _build_messages(row, setting)
                paired_messages = _build_messages(row, paired)
                blocks = messages[1]["content"]
                if any(b.get("type") != "text" for b in blocks):
                    raise ValueError("Unexpected image in text-only control")
                paired_text = [b for b in paired_messages[1]["content"] if b.get("type") == "text"]
                if blocks != paired_text or messages[0] != paired_messages[0]:
                    raise ValueError("Prompt text differs between paired settings")
            target = output / site / "generations" / (setting + ".jsonl")
            jobs.append((source, target, setting))
    print(f"Preflight passed: same samples, model and prompt text; no images in text-only controls. {request_count} requests for a full run.")
    if not args.run_api:
        print("No API requests sent. Add --run-api to generate and evaluate.")
        return
    for source, target, setting in jobs:
        if target.exists():
            lines = target.read_text(encoding="utf-8").splitlines()
            successful = []
            for line in lines:
                if not line.strip():
                    continue
                record = json.loads(line)
                if not record.get("error") and str(record.get("prediction") or "").strip():
                    successful.append(line)
            failed = sum(bool(line.strip()) for line in lines) - len(successful)
            if failed:
                backup = target.parent.parent / "attempt_backups"
                backup.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, backup / (target.stem + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl"))
                target.write_text("\n".join(successful) + ("\n" if successful else ""), encoding="utf-8")
                print(f"Archived original output; retrying {failed} failed records in {target}")
        command = [sys.executable, "-u", str(CODE / "src_report/generate.py"),
                   "--input_jsonl", str(source), "--setting", setting,
                   "--out_jsonl", str(target), "--model", args.model,
                   "--base_url", args.base_url, "--resume"]
        if args.limit is not None:
            command += ["--limit", str(args.limit)]
        subprocess.run(command, cwd=CODE, check=True)
    for site in args.sites:
        files = [output / site / "generations" / (setting + ".jsonl") for setting in ("measure_only", "measure_rule")]
        for file in files:
            records = [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines() if line.strip()]
            if not records or any(r.get("error") or not r.get("prediction") for r in records):
                raise RuntimeError(f"Incomplete generation: {file}; rerun with --run-api")
        subprocess.run([sys.executable, str(CODE / "src_report/eval.py"), "--gen_files",
                        *map(str, files), "--out_dir", str(output / site / "metrics")], cwd=CODE, check=True)


if __name__ == "__main__":
    main()
