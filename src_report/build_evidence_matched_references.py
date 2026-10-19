"""Build evidence-matched reference reports from report-input JSONL records.

These references are deterministic text renderings of the same structured
measurement record given to the report generator. They are useful for evaluating
language realization under controlled evidence. They should not be described as
manual ground-truth inspection reports.
"""

import argparse
import json
import math
from pathlib import Path


def read_jsonl(path: Path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def risk_level(m: dict):
    compliance = m.get("compliance", {}) or {}
    per = compliance.get("per_spacing", []) or []
    status = compliance.get("status", "N/A")
    if not per:
        if str(status).startswith("PASS"):
            return "ACCEPTABLE"
        if str(status).startswith("FAIL"):
            return "WARNING"
        return "UNKNOWN"
    max_dev = max(abs(float(p.get("deviation_from_mean_mm", 0.0))) for p in per)
    fail_ratio = sum(1 for p in per if not p.get("within_main_tol", True)) / len(per)
    if fail_ratio == 0 and max_dev <= 5:
        return "SAFE"
    if fail_ratio == 0 and max_dev <= 10:
        return "ACCEPTABLE"
    if fail_ratio <= 0.25 and max_dev <= 40:
        return "MARGINAL"
    if fail_ratio <= 0.5 or max_dev <= 60:
        return "WARNING"
    return "CRITICAL"


def trend_text(spacings):
    if len(spacings) < 2:
        return "insufficient", 0.0
    slope = (float(spacings[-1]) - float(spacings[0])) / max(len(spacings) - 1, 1)
    if slope > 2.0:
        trend = "increasing"
    elif slope < -2.0:
        trend = "decreasing"
    else:
        trend = "uniform"
    return trend, round(slope, 1)


def compliance_pass_fail(status):
    s = str(status).upper()
    if s.startswith("FAIL"):
        return "FAIL"
    if s.startswith("PASS"):
        return "PASS"
    return s


def build_reference(m: dict):
    compliance = m.get("compliance", {}) or {}
    status_raw = compliance.get("status", "N/A")
    status = compliance_pass_fail(status_raw)
    per = compliance.get("per_spacing", []) or []
    spacings = m.get("spacings_mm", []) or []
    risk = risk_level(m)
    trend, slope = trend_text(spacings)
    n = int(m.get("rebar_count", 0) or 0)
    mean_s = m.get("mean_spacing_mm")
    std_s = m.get("std_spacing_mm")

    devs = [abs(float(p.get("deviation_from_mean_mm", 0.0))) for p in per]
    max_dev = round(max(devs), 1) if devs else None
    fails = [p for p in per if not p.get("within_main_tol", True)]

    if mean_s is None or (isinstance(mean_s, float) and math.isnan(mean_s)):
        return (
            f"The inspection detected {n} vertical rebars, but the spacing "
            "measurement record was incomplete. The available evidence is "
            "insufficient for a reliable GB 50204-2015 spacing assessment. "
            "Additional measurement is required before acceptance. Result: N/A."
        )

    sentence1 = (
        f"The inspection detected {n} vertical rebars with {len(spacings)} "
        f"measured longitudinal spacings; the mean spacing was {float(mean_s):.1f} mm "
        f"and the standard deviation was {float(std_s):.1f} mm."
    )

    if fails:
        fail_desc = "; ".join(
            f"spacing[{int(p['index'])}]={float(p['spacing_mm']):.1f} mm "
            f"({float(p['deviation_from_mean_mm']):+.1f} mm)"
            for p in fails
        )
        sentence2 = (
            "According to the GB 50204-2015 +/-20 mm tolerance, "
            f"the non-compliant interval(s) were {fail_desc}, with a maximum "
            f"deviation of {max_dev:.1f} mm."
        )
    else:
        sentence2 = (
            "According to the GB 50204-2015 +/-20 mm tolerance, all measured "
            f"spacings were compliant, with a maximum deviation of {max_dev:.1f} mm."
        )

    fail_ratio = (len(fails) / len(per)) if per else 0.0
    sentence3 = (
        f"The risk level is {risk}, the non-compliant ratio is {fail_ratio:.3f}, "
        f"and the spacing trend is {trend} (slope={slope:+.1f} mm/gap)."
    )
    if status == "FAIL":
        sentence4 = (
            f"Correction and re-inspection are recommended before concrete placement. "
            f"Result: {status}."
        )
    else:
        sentence4 = (
            f"The installation can proceed to the next construction stage after "
            f"routine quality confirmation. Result: {status}."
        )
    return " ".join([sentence1, sentence2, sentence3, sentence4])


def parse_args():
    p = argparse.ArgumentParser(
        description="Build deterministic evidence-matched report references."
    )
    p.add_argument("--input_jsonl", required=True, type=Path)
    p.add_argument("--out_jsonl", required=True, type=Path)
    p.add_argument("--out_reports_json", default=None, type=Path)
    return p.parse_args()


def main():
    args = parse_args()
    records = read_jsonl(args.input_jsonl)
    out_records = []
    report_records = []
    for rec in records:
        ref = build_reference(rec.get("measurement") or {})
        new_rec = dict(rec)
        new_rec["reference_report"] = ref
        out_records.append(new_rec)
        report_records.append({
            "subset": rec.get("subset"),
            "id": rec.get("id"),
            "source": args.input_jsonl.as_posix(),
            "heading_type": "evidence_matched_reference",
            "reference_report": ref,
            "measurement": rec.get("measurement"),
        })

    write_jsonl(args.out_jsonl, out_records)
    if args.out_reports_json:
        args.out_reports_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_reports_json.write_text(
            json.dumps(report_records, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Wrote {args.out_jsonl}")
    if args.out_reports_json:
        print(f"Wrote {args.out_reports_json}")
    print(f"Records: {len(out_records)}")


if __name__ == "__main__":
    main()
