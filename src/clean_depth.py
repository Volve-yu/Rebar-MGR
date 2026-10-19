"""
Clean up Depth folder: remove depth files that have no matching RGB image.
Run from project root: python src/clean_depth.py
"""

import os
import shutil
from pathlib import Path


def clean_depth(data_root="data", subsets=("site-1", "site-2"), dry_run=True):
    for subset in subsets:
        img_dir   = Path(data_root) / subset / "images"
        depth_dir = Path(data_root) / subset / "Depth"

        if not depth_dir.exists():
            print(f"{subset}/Depth: not found, skipping")
            continue

        img_ids   = {f.stem for f in img_dir.glob("*.png")}
        depth_ids = {f.stem for f in depth_dir.glob("*.png")}

        orphans = sorted(depth_ids - img_ids)
        print(f"\n{subset}:")
        print(f"  RGB: {len(img_ids)}  Depth: {len(depth_ids)}  Orphan depth: {len(orphans)}")

        if not orphans:
            print("  Nothing to delete.")
            continue

        print(f"  Orphan files: {orphans}")
        if dry_run:
            print("  [DRY RUN] Would delete the above files. Run with dry_run=False to actually delete.")
        else:
            for stem in orphans:
                f = depth_dir / f"{stem}.png"
                f.unlink()
                print(f"  Deleted: {f}")
            print(f"  Deleted {len(orphans)} files.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data")
    p.add_argument("--execute", action="store_true",
                   help="Actually delete files (default is dry run)")
    args = p.parse_args()

    print("=== Depth folder cleanup ===")
    clean_depth(args.data_root, dry_run=not args.execute)
    if not args.execute:
        print("\nRe-run with --execute to actually delete the orphan files.")
