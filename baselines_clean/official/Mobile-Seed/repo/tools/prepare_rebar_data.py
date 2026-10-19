from __future__ import annotations

"""
prepare_rebar_data.py -- prepare data_ms/ for Mobile-Seed official training.

Reads from:
  {project_root}/data/site-1/
  {project_root}/data/site-2/

Each subset layout:
  images/{id}.png
  segmentation/{id}.png
  edge/{id}.png
  gt_sets/train.txt
  gt_sets/val.txt

Writes to:
  {out_root}/images/{id}.png
  {out_root}/segmentation/{id}.png
  {out_root}/edge/{id}.png
  {out_root}/splits/train.txt
  {out_root}/splits/val.txt

IDs must be globally unique across subsets; script aborts on collision.
"""

import argparse
import shutil
import sys
from pathlib import Path


_DEFAULT_SUBSETS = ["site-1", "site-2"]
_FILE_DIRS = ["images", "segmentation", "edge"]


def _infer_project_root(script_path: Path) -> Path:
    # script lives at repo/tools/prepare_rebar_data.py
    # repo is inside baselines_clean/official/Mobile-Seed/
    # project root is 5 levels up
    return script_path.parents[4]


def _load_split(gt_sets_dir: Path, split: str) -> list[str]:
    txt = gt_sets_dir / f"{split}.txt"
    if not txt.exists():
        raise FileNotFoundError(f"Split file not found: {txt}")
    ids = [line.strip() for line in txt.read_text().splitlines() if line.strip()]
    return ids


def _check_files(subset_dir: Path, ids: list[str]) -> list[str]:
    missing = []
    for id_ in ids:
        for d in _FILE_DIRS:
            p = subset_dir / d / f"{id_}.png"
            if not p.exists():
                missing.append(str(p))
    return missing


def _copy_files(subset_dir: Path, ids: list[str], out_root: Path):
    for id_ in ids:
        for d in _FILE_DIRS:
            src = subset_dir / d / f"{id_}.png"
            dst = out_root / d / f"{id_}.png"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare data_ms/ for Mobile-Seed official training."
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root (gangjing/). Auto-inferred from script location if omitted.",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output directory (default: <script>/../../../data_ms relative to repo root).",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=_DEFAULT_SUBSETS,
        help="Subset directories to include, e.g. site-1 or site-2. Default: site-1 site-2.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip IDs with missing image/segmentation/edge files without prompting.",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else _infer_project_root(script_path)
    )
    if not (project_root / "data").exists():
        sys.exit(
            f"ERROR: data/ not found under project root {project_root}\n"
            "Pass --project-root explicitly."
        )

    out_root = (
        Path(args.out_root).resolve()
        if args.out_root
        else script_path.parents[1] / "data_ms"
    )

    print(f"Project root : {project_root}")
    print(f"Output root  : {out_root}")
    print()

    # --- collect IDs per split per subset ---
    all_ids: dict[str, dict[str, str]] = {"train": {}, "val": {}}

    for subset in args.subsets:
        subset_dir = project_root / "data" / subset
        if not subset_dir.exists():
            sys.exit(f"ERROR: subset directory not found: {subset_dir}")
        gt_sets_dir = subset_dir / "gt_sets"

        for split in ("train", "val"):
            ids = _load_split(gt_sets_dir, split)
            for id_ in ids:
                if id_ in all_ids[split]:
                    prev = all_ids[split][id_]
                    sys.exit(
                        f"ERROR: ID collision in split '{split}': '{id_}' "
                        f"appears in both '{prev}' and '{subset}'. "
                        "Rename IDs manually before running this script."
                    )
                all_ids[split][id_] = subset  # id -> source subset

    # --- check files exist ---
    total_missing = []
    invalid_ids: dict[str, set[str]] = {"train": set(), "val": set()}
    for split in ("train", "val"):
        for id_, subset in all_ids[split].items():
            subset_dir = project_root / "data" / subset
            for d in _FILE_DIRS:
                p = subset_dir / d / f"{id_}.png"
                if not p.exists():
                    total_missing.append(str(p))
                    invalid_ids[split].add(id_)

    if total_missing:
        print(f"WARNING: {len(total_missing)} missing file(s):")
        for p in total_missing[:20]:
            print(f"  {p}")
        if len(total_missing) > 20:
            print(f"  ... and {len(total_missing) - 20} more")
        print("Missing IDs will be skipped so split files only contain valid samples.")
        if not args.skip_missing:
            answer = input("Continue anyway? [y/N] ").strip().lower()
            if answer != "y":
                sys.exit("Aborted.")
        else:
            print("--skip-missing set: continuing without interactive prompt.")

    for split in ("train", "val"):
        for id_ in invalid_ids[split]:
            all_ids[split].pop(id_, None)

    # --- copy files ---
    for split in ("train", "val"):
        for id_, subset in all_ids[split].items():
            subset_dir = project_root / "data" / subset
            for d in _FILE_DIRS:
                src = subset_dir / d / f"{id_}.png"
                if not src.exists():
                    continue
                dst = out_root / d / f"{id_}.png"
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)

    # --- write split files ---
    splits_dir = out_root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "val"):
        ids_sorted = sorted(all_ids[split].keys())
        txt = splits_dir / f"{split}.txt"
        txt.write_text("\n".join(ids_sorted) + "\n")

    # --- report ---
    print(f"train : {len(all_ids['train'])} samples")
    print(f"val   : {len(all_ids['val'])} samples")
    print(f"missing files : {len(total_missing)}")
    skipped = sum(len(v) for v in invalid_ids.values())
    print(f"skipped IDs   : {skipped}")
    print()
    print("Output layout:")
    print(f"  {out_root}/images/")
    print(f"  {out_root}/segmentation/")
    print(f"  {out_root}/edge/")
    print(f"  {out_root}/splits/train.txt")
    print(f"  {out_root}/splits/val.txt")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
