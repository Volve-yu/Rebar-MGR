"""
Extract metric depth labels from 总数据 folder structure and write per-subset
depth_labels.txt files.

总数据 layout:
  总数据/
    上午数据/   -> corresponds to data/site-1
      0.4/Color/{id}.jpg   (shooting distance = 0.4 m)
      0.5/Color/{id}.jpg
      0.6/Color/{id}.jpg
    下午数据/   -> corresponds to data/site-2
      0.3/Color/{id}.jpg
      0.4/Color/{id}.jpg
      0.5/Color/{id}.jpg
      0.6/Color/{id}.jpg

Output: data/{subset}/depth_labels.txt
  One line per image id that has a known shooting distance:
    {id}\t{distance_metres}
  e.g.
    1\t0.4
    10\t0.5

Usage:
    python src/extract_depth_labels.py --raw_root "总数据" --data_root data
"""

import argparse
import os
from pathlib import Path


SESSION_TO_SUBSET = {
    "上午数据": "site-1",
    "下午数据": "site-2",
}


def extract(raw_root: str, data_root: str):
    raw_root = Path(raw_root)
    data_root = Path(data_root)

    for session, subset in SESSION_TO_SUBSET.items():
        session_dir = raw_root / session
        if not session_dir.exists():
            print(f"[SKIP] {session_dir} not found")
            continue

        # Build id -> distance mapping
        id_to_dist = {}
        for dist_folder in sorted(session_dir.iterdir()):
            if not dist_folder.is_dir():
                continue
            try:
                dist_m = float(dist_folder.name)
            except ValueError:
                continue
            color_dir = dist_folder / "Color"
            if not color_dir.exists():
                continue
            for f in color_dir.iterdir():
                if f.suffix.lower() == ".jpg":
                    img_id = f.stem
                    if img_id in id_to_dist:
                        print(f"  [WARN] duplicate id {img_id} in {subset}, "
                              f"keeping {id_to_dist[img_id]}m, ignoring {dist_m}m")
                    else:
                        id_to_dist[img_id] = dist_m

        # Filter: only keep IDs that exist in data/{subset}/images/
        img_dir = data_root / subset / "images"
        valid_ids = {f.stem for f in img_dir.glob("*.png")} if img_dir.exists() else set(id_to_dist)
        removed = [k for k in id_to_dist if k not in valid_ids]
        if removed:
            print(f"  [FILTER] {subset}: dropping {len(removed)} ids not in images/: {removed}")
        id_to_dist = {k: v for k, v in id_to_dist.items() if k in valid_ids}

        # Write output
        out_path = data_root / subset / "depth_labels.txt"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines = sorted(id_to_dist.items(), key=lambda x: int(x[0]))
        with open(out_path, "w", encoding="utf-8") as f:
            for img_id, dist in lines:
                f.write(f"{img_id}\t{dist}\n")

        print(f"{subset}: wrote {len(lines)} entries -> {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw_root", default="总数据",
                   help="Path to the 总数据 folder (relative to project root)")
    p.add_argument("--data_root", default="data")
    args = p.parse_args()
    extract(args.raw_root, args.data_root)
