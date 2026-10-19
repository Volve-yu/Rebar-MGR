"""Print side-by-side reference and generated reports for local inspection.

Example:
  python src_report/inspect_reports.py \
      --gen_jsonl results_clean/report_generation/generations/full_seg_measure_rule_qaep.jsonl \
      --limit 8
"""

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def parse_args():
    p = argparse.ArgumentParser(description="Inspect generated reports against references.")
    p.add_argument("--gen_jsonl", required=True, type=Path)
    p.add_argument("--limit", type=int, default=10)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--show_measurement", action="store_true")
    p.add_argument("--show_qaep", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    shown = 0
    for idx, rec in enumerate(read_jsonl(args.gen_jsonl)):
        if idx < args.start:
            continue
        if shown >= args.limit:
            break
        shown += 1
        print("=" * 100)
        print(f"{idx + 1}. {rec.get('subset')}/{rec.get('id')}  setting={rec.get('setting')}")
        if rec.get("error"):
            print("[ERROR]")
            print(rec["error"])
        print("[REFERENCE]")
        print(rec.get("reference_report", ""))
        print("\n[PREDICTION]")
        print(rec.get("prediction", ""))
        if args.show_measurement:
            print("\n[MEASUREMENT]")
            print(json.dumps(rec.get("measurement"), ensure_ascii=False, indent=2))
        if args.show_qaep and rec.get("qaep") is not None:
            print("\n[QAEP]")
            qaep = rec.get("qaep") or {}
            slim = {
                "probabilities": qaep.get("probabilities"),
                "labels": qaep.get("labels"),
                "risk_tags": qaep.get("risk_tags"),
                "report_constraints": qaep.get("report_constraints"),
            }
            print(json.dumps(slim, ensure_ascii=False, indent=2))
    print("=" * 100)
    print(f"Shown {shown} record(s) from {args.gen_jsonl}")


if __name__ == "__main__":
    main()
