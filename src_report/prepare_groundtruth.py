import argparse
import json
import re
from pathlib import Path


REPORT_NAME_RE = re.compile(r"^(site-[12])_(.+)\.txt$")
PROFESSIONAL_HEADING_RE = re.compile(
    r"^\s*#{3,6}\s*Professional\s+Narrative\s*:?\s*$", re.IGNORECASE
)

def is_stop_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith("["):
        return True
    if stripped.startswith("###"):
        return True
    if re.fullmatch(r"=+", stripped):
        return True
    return False


def clean_section(lines):
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    while lines and re.fullmatch(r"-{3,}", lines[0].strip()):
        lines.pop(0)
    while lines and re.fullmatch(r"-{3,}", lines[-1].strip()):
        lines.pop()
    return "\n".join(line.rstrip() for line in lines).strip()


def extract_after_heading(lines, heading_index: int) -> str:
    section = []
    for line in lines[heading_index + 1 :]:
        if is_stop_line(line):
            break
        section.append(line)
    return clean_section(section)


def find_narrative(text: str):
    lines = text.splitlines()

    for idx, line in enumerate(lines):
        if PROFESSIONAL_HEADING_RE.match(line):
            report = extract_after_heading(lines, idx)
            if report:
                return "professional_narrative", report
    return None, ""


def convert_groundtruth(input_dir: Path, output_dir: Path):
    records = []
    missing = []
    short = []

    for txt_path in sorted(input_dir.glob("*.txt")):
        match = REPORT_NAME_RE.match(txt_path.name)
        if not match:
            continue

        subset, sample_id = match.groups()
        text = txt_path.read_text(encoding="utf-8", errors="replace")
        heading_type, report = find_narrative(text)

        if not report:
            missing.append(str(txt_path))
            continue

        words = re.findall(r"\S+", report)
        if len(words) < 8:
            short.append(str(txt_path))

        records.append(
            {
                "subset": subset,
                "id": sample_id,
                "source_file": txt_path.as_posix(),
                "heading_type": heading_type,
                "reference_report": report,
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    reports_path = output_dir / "reports.json"
    reports_path.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    per_file_dir = output_dir / "by_sample"
    per_file_dir.mkdir(parents=True, exist_ok=True)
    for record in records:
        out_name = f"{record['subset']}_{record['id']}.json"
        (per_file_dir / out_name).write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    summary = {
        "input_dir": input_dir.as_posix(),
        "output_dir": output_dir.as_posix(),
        "num_records": len(records),
        "num_missing": len(missing),
        "num_short": len(short),
        "missing": missing,
        "short": short,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return reports_path, summary_path, summary


def main():
    parser = argparse.ArgumentParser(
        description="Extract concise report-generation references from groundtruth txt files."
    )
    parser.add_argument("--input_dir", default="groundtruth", type=Path)
    parser.add_argument("--output_dir", default="groundtruth_json", type=Path)
    args = parser.parse_args()

    reports_path, summary_path, summary = convert_groundtruth(args.input_dir, args.output_dir)
    print(f"Wrote reports: {reports_path}")
    print(f"Wrote summary: {summary_path}")
    print(
        "Converted {num_records} records; missing narrative: {num_missing}; short: {num_short}".format(
            **summary
        )
    )
    if summary["missing"]:
        print("Missing narrative files:")
        for path in summary["missing"][:20]:
            print(f"  {path}")
        if len(summary["missing"]) > 20:
            print(f"  ... {len(summary['missing']) - 20} more")


if __name__ == "__main__":
    main()
