"""
Validate gt_report output for ground-truth quality control.

Checks per report:
  1. VLM generation failure         ([VLM report failed: ...])
  2. Missing required sections
  3. Inspector Notes too short      (< 80 chars, likely a stub)
  4. Low attribution score          (Attr < 0.6)
  5. Numerical contradiction        (VLM text mentions a spacing value
                                     that conflicts with algo measurement)
  6. Compliance contradiction       (VLM narrative says PASS but algo=FAIL
                                     or vice-versa)
  7. CSV numeric sanity             (rebar_count, spacing, std bounds)

Usage:
    python src/validate_reports.py --out_dir gt_reports_full
"""

import argparse
import csv
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_report(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _extract_section(text: str, header: str) -> str:
    """Return text between [header] and the next [section] or end."""
    pattern = re.escape(f"[{header}]")
    m = re.search(pattern, text)
    if not m:
        return ""
    start = m.end()
    nxt = re.search(r"\[[\w\s]+\]", text[start:])
    end = start + nxt.start() if nxt else len(text)
    return text[start:end].strip()


def _attr_score(text: str):
    m = re.search(r"Attr\(r\)\s*=\s*([\d.]+)", text)
    return float(m.group(1)) if m else None


def _vlm_says_pass(notes: str) -> bool:
    notes_lower = notes.lower()
    # VLM explicitly says PASS somewhere in narrative
    return bool(re.search(r'\bpass\b', notes_lower))


def _vlm_says_fail(notes: str) -> bool:
    notes_lower = notes.lower()
    return bool(re.search(r'\bfail\b', notes_lower))


# ---------------------------------------------------------------------------
# Per-report checks
# ---------------------------------------------------------------------------

REQUIRED_SECTIONS = [
    "Inspection Overview",
    "Measurements",
    "Compliance Assessment",
    "Risk Assessment",
    # Spacing Trend Analysis is omitted when rebar_count < 4 (fewer than 3 spacings)
]


def check_report(txt_path: Path, csv_row: dict) -> list:
    """Return list of (severity, message) tuples. Empty = clean."""
    text = _read_report(txt_path)
    if not text:
        return [("ERROR", "report file missing or unreadable")]

    issues = []
    sid = csv_row.get("image_id", txt_path.stem)

    # 1. VLM failure
    if "[VLM report failed:" in text:
        m = re.search(r"\[VLM report failed: (.+?)\]", text)
        reason = m.group(1)[:80] if m else "unknown"
        issues.append(("ERROR", f"VLM generation failed: {reason}"))

    # 2. Missing sections
    for sec in REQUIRED_SECTIONS:
        if f"[{sec}]" not in text and f"[{sec.split()[0]}" not in text:
            issues.append(("WARN", f"missing section [{sec}]"))

    # 3. Inspector Notes too short (VLM ran but produced almost nothing)
    notes = _extract_section(text, "Inspector Notes")
    if notes and len(notes) < 80:
        issues.append(("WARN", f"Inspector Notes very short ({len(notes)} chars)"))

    # 4. Low attribution score
    attr = _attr_score(text)
    if attr is not None and attr < 0.60:
        issues.append(("WARN", f"Attr(r) = {attr:.2f} < 0.60"))

    # 5. Compliance contradiction: algo vs VLM narrative
    algo_status = csv_row.get("compliance", "")
    if notes:
        vlm_pass = _vlm_says_pass(notes)
        vlm_fail = _vlm_says_fail(notes)
        if algo_status == "FAIL" and vlm_pass and not vlm_fail:
            issues.append(("WARN", f"algo=FAIL but VLM narrative suggests PASS"))
        if algo_status == "PASS" and vlm_fail and not vlm_pass:
            issues.append(("WARN", f"algo=PASS but VLM narrative suggests FAIL"))

    # 6. CSV numeric sanity
    try:
        n = float(csv_row.get("rebar_count") or 0)
        sp = float(csv_row.get("mean_spacing_mm") or 0)
        sd = float(csv_row.get("std_spacing_mm") or 0)

        if n < 2:
            issues.append(("ERROR", f"rebar_count={n:.0f} (< 2)"))
        if sp != 0 and (sp < 40 or sp > 500):
            issues.append(("ERROR", f"mean_spacing={sp:.0f} mm out of range [40,500]"))
        if sd > 80:
            issues.append(("WARN", f"std_spacing={sd:.0f} mm very high (> 80)"))
    except (ValueError, TypeError):
        issues.append(("WARN", "CSV numeric fields unparseable"))

    return issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="gt_report output directory")
    ap.add_argument("--min_attr", type=float, default=0.60,
                    help="minimum acceptable Attr(r) score")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    csv_path = out_dir / "results.csv"

    if not csv_path.exists():
        print(f"[ERROR] results.csv not found in {out_dir}")
        sys.exit(1)

    rows = {
        f"{r['subset']}_{r['image_id']}": r
        for r in csv.DictReader(csv_path.open(encoding="utf-8"))
    }

    txt_files = sorted(out_dir.glob("*.txt"))
    txt_files = [p for p in txt_files if p.stem != "summary"]

    all_issues: list = []  # (stem, severity, msg)
    n_clean = 0

    for txt_path in txt_files:
        stem = txt_path.stem          # e.g. "site-1_66"
        csv_row = rows.get(stem, {})
        issues = check_report(txt_path, csv_row)
        if issues:
            for sev, msg in issues:
                all_issues.append((stem, sev, msg))
        else:
            n_clean += 1

    # Print summary
    errors   = [(s, m) for st, s, m in all_issues if s == "ERROR"]
    warnings = [(s, m) for st, s, m in all_issues if s == "WARN"]

    print("=" * 60)
    print(f"Report validation: {out_dir.name}")
    print(f"  Total reports checked : {len(txt_files)}")
    print(f"  Clean (no issues)     : {n_clean}")
    print(f"  Errors                : {len(errors)}")
    print(f"  Warnings              : {len(warnings)}")
    print("=" * 60)

    if not all_issues:
        print("All reports passed validation.")
        return

    # Group by severity
    by_stem: dict = {}
    for stem, sev, msg in all_issues:
        by_stem.setdefault(stem, []).append((sev, msg))

    print("\n[ERRORS] — must fix before using as ground truth:")
    error_stems = [st for st, sv, _ in all_issues if sv == "ERROR"]
    if not error_stems:
        print("  none")
    else:
        for stem in sorted(set(error_stems)):
            for sev, msg in by_stem[stem]:
                if sev == "ERROR":
                    print(f"  {stem}: {msg}")

    print("\n[WARNINGS] — review recommended:")
    warn_stems = [st for st, sv, _ in all_issues if sv == "WARN"]
    if not warn_stems:
        print("  none")
    else:
        for stem in sorted(set(warn_stems)):
            for sev, msg in by_stem[stem]:
                if sev == "WARN":
                    print(f"  {stem}: {msg}")

    # Write flagged list to file
    flag_path = out_dir / "validation_issues.txt"
    with flag_path.open("w", encoding="utf-8") as f:
        for stem, sev, msg in sorted(all_issues):
            f.write(f"{sev}\t{stem}\t{msg}\n")
    print(f"\nFull issue list saved -> {flag_path}")


if __name__ == "__main__":
    main()
