"""Audit existing labels or generate REVIEW-REQUIRED detection-box drafts."""
import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
import cv2
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sites", nargs="+", default=["site-1", "site-2"])
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--restore-images", action="store_true",
                   help="Restore unchanged RGB files after moving labels to another machine")
    p.add_argument("--confirm-reviewed", action="store_true",
                   help="Only after visually checking/correcting ALL generated box labels")
    a = p.parse_args()
    if a.restore_images:
        with (a.out/"review.csv").open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            source = a.data_root/r["site"]/"images"/(r["id"]+".png")
            if sha(source) != r["image_sha256"]:
                raise ValueError("Source RGB differs from reviewed draft: "+str(source))
            dest = a.out/r["image"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(source, dest)
            elif sha(dest) != r["image_sha256"]:
                raise ValueError("Existing RGB differs: "+str(dest))
        print("Restored/checked", len(rows), "RGB images; labels unchanged")
        return
    if a.confirm_reviewed:
        manifest = a.out/"review.csv"
        with manifest.open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError("Empty review table")
        if any(r["reviewed"].strip().lower() != "yes" for r in rows):
            raise ValueError("Review each image, correct its label, then set reviewed=yes in review.csv")
        hashes = {}
        for r in rows:
            path = a.out/r["label"]
            values = [list(map(float, s.split())) for s in path.read_text().splitlines() if s.strip()]
            if not values or any(len(v) != 5 or v[0] != 0 or not all(0 <= x <= 1 for x in v[1:])
                                 or min(v[3:]) <= 0 for v in values):
                raise ValueError("Invalid YOLO label: "+str(path))
            if any(v[1]-v[3]/2 < -1e-6 or v[1]+v[3]/2 > 1+1e-6 or
                   v[2]-v[4]/2 < -1e-6 or v[2]+v[4]/2 > 1+1e-6 for v in values):
                raise ValueError("Box exceeds image: "+str(path))
            if sha(a.out/r["image"]) != r["image_sha256"]:
                raise ValueError("Image changed after draft creation")
            hashes[r["label"]] = sha(path)
            hashes[r["image"]] = r["image_sha256"]
        (a.out/"reviewed.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        print("Review confirmed for", len(rows), "images")
        return
    if a.out.exists() and not a.audit_only:
        raise FileExistsError(a.out)
    records, counts = [], {}
    for site in a.sites:
        splits = {s: (a.data_root/site/"gt_sets"/(s+".txt")).read_text().split() for s in ("train", "val")}
        if set(splits["train"]) & set(splits["val"]):
            raise ValueError("Train/val ID overlap: "+site)
        for split, ids in splits.items():
            if not ids or len(ids) != len(set(ids)):
                raise ValueError("Empty or duplicate split")
            counts[site+"/"+split] = len(ids)
            for sid in ids:
                source = a.data_root/site/"images"/(sid+".png")
                image = cv2.imread(str(source))
                mask = cv2.imread(str(a.data_root/site/"segmentation"/(sid+".png")), 0)
                xs = sorted(float(x) for x in (a.data_root/site/"rebar_labels"/(sid+".txt")).read_text().split())
                if image is None or mask is None or not xs:
                    raise ValueError("Missing inputs: "+site+"/"+sid)
                h, w = image.shape[:2]
                if min(xs) < 0 or max(xs) >= w or len(set(xs)) != len(xs):
                    raise ValueError("Invalid center references")
                if a.audit_only:
                    continue
                mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) > 127
                bounds = [0]+[int((x+y)/2) for x, y in zip(xs, xs[1:])]+[w]
                boxes, warnings = [], []
                for i, x in enumerate(xs):
                    lo, hi = bounds[i:i+2]
                    projection = mask[:, lo:hi].sum(0)
                    active = projection >= max(3, .15*h)
                    start = int(np.clip(round(x)-lo, 0, hi-lo-1))
                    possible = np.flatnonzero(active)
                    if not len(possible):
                        # A placeholder for manual correction, never a verified box.
                        left = max(lo, int(x)-3)
                        right = min(hi, int(x)+4)
                        boxes.append(f"0 {(left+right)/2/w:.8f} {0.5:.8f} {(right-left)/w:.8f} {1.:.8f}")
                        warnings.append("bar "+str(i+1)+": no foreground band; replace placeholder")
                        cv2.rectangle(image, (left, 0), (right-1, h-1), (0, 165, 255), 2)
                        continue
                    start = int(possible[np.argmin(abs(possible-start))])
                    left = right = start
                    while left > 0 and active[left-1]:
                        left -= 1
                    while right+1 < len(active) and active[right+1]:
                        right += 1
                    x1, x2 = lo+left, lo+right+1
                    ys = np.nonzero(mask[:, x1:x2])[0]
                    y1, y2 = int(ys.min()), int(ys.max())+1
                    boxes.append(f"0 {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}")
                    cv2.rectangle(image, (x1, y1), (x2-1, y2-1), (0, 0, 255), 2)
                    cv2.putText(image, str(i+1), (x1, max(15, y1+15)), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
                dest = a.out/site/"images"/split/source.name
                label = a.out/site/"labels"/split/(sid+".txt")
                preview = a.out/site/"preview"/split/source.name
                for path in (dest, label, preview):
                    path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
                label.write_text("\n".join(boxes)+"\n", encoding="utf-8")
                cv2.imwrite(str(preview), image)
                records.append(dict(site=site, split=split, id=sid, reviewed="no", warnings="; ".join(warnings),
                                    image=dest.relative_to(a.out).as_posix(), image_sha256=sha(dest),
                                    label=label.relative_to(a.out).as_posix()))
        if not a.audit_only:
            (a.out/site/"dataset.yaml").write_text(
                "path: "+json.dumps(str((a.out/site).resolve()))+"\ntrain: images/train\nval: images/val\nnames:\n  0: vertical_rebar\n", encoding="utf-8")
    print(json.dumps(counts, indent=2))
    if not a.audit_only:
        with (a.out/"review.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)


if __name__ == "__main__":
    main()
