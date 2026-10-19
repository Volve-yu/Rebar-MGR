"""Prepare segmentation sources for report-generation comparison.

The report input builder expects a uniform layout:

  {out_root}/{method}/{subset}/seg/{id}.png

This script copies prediction masks from our method and external baselines into
that layout, restricted to the requested split IDs.
"""

import argparse
import json
import shutil
from pathlib import Path


METHOD_SOURCES = {
    "ours_full": {
        "site-1": Path("predictions_measure_guided/full/site-1/seg"),
        "site-2": Path("predictions_measure_guided/full/site-2/seg"),
    },
    "deeplab": {
        "site-1": Path("predictions_clean/subset_site-1_baselines/deeplab/site-1/seg"),
        "site-2": Path("predictions_clean/subset_site-2_baselines/deeplab/site-2/seg"),
    },
    "gscnn": {
        "site-1": Path("predictions_clean/subset_site-1_baselines/gscnn/site-1/seg"),
        "site-2": Path("predictions_clean/subset_site-2_baselines/gscnn/site-2/seg"),
    },
    "mobile_seed": {
        "site-1": Path("predictions_clean/subset_site-1_baselines/mobile_seed/site-1/seg"),
        "site-2": Path("predictions_clean/subset_site-2_baselines/mobile_seed/site-2/seg"),
    },
    "sctransnet": {
        "site-1": Path("predictions_clean/subset_site-1_extra_baselines/sctransnet/site-1/seg"),
        "site-2": Path("predictions_clean/subset_site-2_extra_baselines/sctransnet/site-2/seg"),
    },
    "afenet": {
        "site-1": Path("predictions_clean/subset_site-1_extra_baselines/afenet/site-1/seg"),
        "site-2": Path("predictions_clean/subset_site-2_extra_baselines/afenet/site-2/seg"),
    },
}


def load_split_ids(data_root: Path, subset: str, split: str):
    path = data_root / subset / "gt_sets" / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="Copy segmentation masks into a uniform report-source layout."
    )
    p.add_argument("--data_root", default="data", type=Path)
    p.add_argument("--out_root", default="predictions_report_sources", type=Path)
    p.add_argument("--split", default="val")
    p.add_argument("--methods", nargs="+", default=list(METHOD_SOURCES.keys()))
    p.add_argument("--subsets", nargs="+", default=["site-1", "site-2"])
    return p.parse_args()


def main():
    args = parse_args()
    summaries = []

    for method in args.methods:
        if method not in METHOD_SOURCES:
            raise ValueError(f"Unknown method {method!r}. Valid: {sorted(METHOD_SOURCES)}")

        for subset in args.subsets:
            ids = load_split_ids(args.data_root, subset, args.split)
            src_dir = METHOD_SOURCES[method][subset]
            dst_dir = args.out_root / method / subset / "seg"
            dst_dir.mkdir(parents=True, exist_ok=True)

            copied = 0
            missing = []
            for sid in ids:
                src = src_dir / f"{sid}.png"
                dst = dst_dir / f"{sid}.png"
                if not src.exists():
                    missing.append(sid)
                    continue
                shutil.copy2(src, dst)
                copied += 1

            summaries.append({
                "method": method,
                "subset": subset,
                "source": src_dir.as_posix(),
                "target": dst_dir.as_posix(),
                "split": args.split,
                "expected": len(ids),
                "copied": copied,
                "missing": missing,
            })
            print(
                f"{method:12s} {subset:5s}: copied {copied}/{len(ids)} "
                f"from {src_dir}"
            )
            if missing:
                print(f"  missing: {', '.join(missing[:20])}")

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_root / "summary.json"
    summary_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
