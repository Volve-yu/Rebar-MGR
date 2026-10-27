"""Attach saved QAEP plans to Full report inputs without making API requests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows


def index_rows(rows: list[dict], path: Path) -> dict[tuple[str, str], dict]:
    indexed: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (str(row.get("subset", "")), str(row.get("id", "")))
        if not all(key) or key in indexed:
            raise ValueError(f"Invalid or duplicate sample key {key} in {path}")
        indexed[key] = row
    return indexed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs-root", type=Path, required=True,
                        help="Directory with <site>_full_seed42/within/report_inputs.jsonl")
    parser.add_argument("--qaep-root", type=Path, required=True,
                        help="Directory with <site>/generations/rgb_seg_measure_rule_qaep.jsonl")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sites", nargs="+", default=["site-1", "site-2"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    out_dir = args.out_dir

    prepared: dict[str, list[dict]] = {}
    for site in args.sites:
        input_path = (
            args.inputs_root
            / f"{site}_full_seed42"
            / "within"
            / "report_inputs.jsonl"
        )
        qaep_path = (
            args.qaep_root
            / site
            / "generations"
            / "rgb_seg_measure_rule_qaep.jsonl"
        )
        inputs = read_jsonl(input_path)
        prior = read_jsonl(qaep_path)
        input_by_key = index_rows(inputs, input_path)
        prior_by_key = index_rows(prior, qaep_path)

        if input_by_key.keys() != prior_by_key.keys():
            missing = sorted(input_by_key.keys() - prior_by_key.keys())[:5]
            extra = sorted(prior_by_key.keys() - input_by_key.keys())[:5]
            raise ValueError(
                f"{site}: sample IDs differ; missing={missing}, extra={extra}"
            )

        merged = []
        for row in inputs:
            key = (str(row["subset"]), str(row["id"]))
            qaep_row = prior_by_key[key]
            if row.get("schema_version") != "design_physical_v2":
                raise ValueError(f"{site}/{key[1]}: unexpected input schema")
            if row.get("measurement") != qaep_row.get("measurement"):
                raise ValueError(f"{site}/{key[1]}: measurement differs from QAEP run")
            for field in ("image_path", "seg_path"):
                if not Path(row.get(field, "")).is_file():
                    raise FileNotFoundError(
                        f"{site}/{key[1]}: missing {field}: {row.get(field)}"
                    )
            if qaep_row.get("setting") != "rgb_seg_measure_rule_qaep":
                raise ValueError(f"{site}/{key[1]}: unexpected QAEP source setting")
            if qaep_row.get("error") or not qaep_row.get("prediction"):
                raise ValueError(f"{site}/{key[1]}: QAEP source record is incomplete")
            plan = qaep_row.get("qaep")
            if not isinstance(plan, dict) or not plan.get("plan_text"):
                raise ValueError(f"{site}/{key[1]}: QAEP plan is missing")
            merged_row = dict(row)
            merged_row["qaep"] = plan
            merged.append(merged_row)
        prepared[site] = merged

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs = {site: out_dir / f"{site}_full_inputs_qaep.jsonl" for site in args.sites}
    manifest_path = out_dir / "manifest.json"
    existing = [path for path in [*outputs.values(), manifest_path] if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output already exists; choose another --out-dir or pass --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    for site, rows in prepared.items():
        with outputs[site].open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "purpose": "Full report generation inputs",
        "schema_version": "design_physical_v2",
        "qaep_source": str(args.qaep_root),
        "sites": {site: {"n": len(prepared[site]), "input": str(outputs[site])} for site in args.sites},
        "total": sum(len(rows) for rows in prepared.values()),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {manifest['total']} inputs with QAEP plans; no API requests made.")
    for site in args.sites:
        print(f"{site}: {len(prepared[site])} -> {outputs[site]}")


if __name__ == "__main__":
    main()
