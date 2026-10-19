"""Build a weakly supervised QAEP dataset from report-generation inputs.

The script reads the existing report JSONL produced by build_report_inputs.py.
It extracts lightweight evidence-quality features from the predicted mask,
structured measurement record, optional RGB image, and optional GT mask.

Example:
  python src_report/build_qaep_dataset.py \
      --input_jsonl results_clean/report_generation/inputs/full.jsonl \
      --data_root data \
      --out_jsonl results_clean/report_generation/qaep/full_qaep_dataset.jsonl
"""

import argparse
import csv
from pathlib import Path

from qaep_utils import FEATURE_NAMES, TARGET_NAMES, extract_features, read_jsonl, weak_labels, write_jsonl


def parse_args():
    p = argparse.ArgumentParser(description="Build weakly supervised QAEP feature dataset.")
    p.add_argument("--input_jsonl", required=True, type=Path)
    p.add_argument("--data_root", default="data", type=Path)
    p.add_argument("--out_jsonl", required=True, type=Path)
    p.add_argument("--out_csv", default=None, type=Path)
    p.add_argument("--subsets", nargs="*", default=None,
                   help="Optional subset names to keep, e.g. site-1 site-2.")
    return p.parse_args()


def main():
    args = parse_args()
    records = read_jsonl(args.input_jsonl)
    if args.subsets:
        keep = set(args.subsets)
        records = [r for r in records if str(r.get("subset")) in keep]
    out_records = []

    for rec in records:
        features, quality = extract_features(rec, args.data_root)
        labels = weak_labels(rec, features, quality)
        out_records.append({
            "schema_version": rec.get("schema_version", "legacy"),
            "split": rec.get("split"),
            "subset": rec.get("subset"),
            "id": rec.get("id"),
            "variant": rec.get("variant"),
            "features": features,
            "quality": quality,
            "labels": labels,
            "measurement": rec.get("measurement"),
            "seg_path": rec.get("seg_path"),
            "image_path": rec.get("image_path"),
        })

    write_jsonl(args.out_jsonl, out_records)

    csv_path = args.out_csv
    if csv_path is None:
        csv_path = args.out_jsonl.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "subset",
        "id",
        "variant",
        "image_available",
    ] + FEATURE_NAMES + ["seg_iou", "seg_f1"] + TARGET_NAMES
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in out_records:
            row = {
                "subset": rec.get("subset"),
                "id": rec.get("id"),
                "variant": rec.get("variant"),
                "image_available": rec.get("quality", {}).get("image_available"),
            }
            row.update(rec["features"])
            row.update(rec["quality"])
            row.update(rec["labels"])
            writer.writerow(row)

    counts = {name: sum(int(r["labels"][name]) for r in out_records) for name in TARGET_NAMES}
    print(f"Wrote {args.out_jsonl}")
    print(f"Wrote {csv_path}")
    print(f"Records: {len(out_records)}")
    for name, pos in counts.items():
        print(f"{name}: positive={pos} negative={len(out_records) - pos}")


if __name__ == "__main__":
    main()
