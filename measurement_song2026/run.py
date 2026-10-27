"""Train independent site detectors; evaluate shared-scale 2D measurements."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
sys.path[:0] = [str(HERE), str(CODE/"src"), str(CODE/"scripts")]
from geometry import fit_axis, unique_centers
from inspection_data import AnnotationStore, matched_intervals
from audit_measurement_wholeset import load_scaled_gt_xs
from measure import RulerConfig, load_depth_labels
from experiment_snapshot import snapshot


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding="utf-8")


def reviewed(root):
    manifest = root/"reviewed.json"
    hashes = json.loads(manifest.read_text(encoding="utf-8"))
    if not hashes:
        raise ValueError("Empty reviewed dataset")
    for name, digest in hashes.items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
            raise ValueError("Reviewed data changed: "+name)
    return hashes


def train(a):
    import torch
    from ultralytics import YOLO
    import ultralytics
    from models import RebarLightTrainer
    import yaml
    hashes = reviewed(a.dataset)
    site_root = a.dataset/a.site
    source_before = snapshot(a.data_root, [a.site], ["train", "val"])
    for split in ("train", "val"):
        ids = source_before["members"][a.site+"/"+split]
        actual = {p.stem for p in (site_root/"images"/split).glob("*.png")}
        if actual != set(ids):
            raise ValueError("Detection dataset split differs from source: "+split)
        for sid in ids:
            image_key = f"{a.site}/images/{split}/{sid}.png"
            label_key = f"{a.site}/labels/{split}/{sid}.txt"
            expected = source_before["files"][f"{a.site}/images/{sid}.png"]
            if hashes.get(image_key) != expected or label_key not in hashes:
                raise ValueError("Missing or mismatched reviewed pair: "+image_key)
    cfg = yaml.safe_load((site_root/"dataset.yaml").read_text(encoding="utf-8"))
    cfg["path"] = str(site_root.resolve())
    if cfg["train"] != "images/train" or cfg["val"] != "images/val":
        raise ValueError("Unexpected detection dataset split paths")
    (site_root/"dataset.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    if a.out.exists():
        raise FileExistsError(a.out)
    # Hold initialization policy constant across all three architectures.
    model = YOLO("yolov8n.yaml" if a.model == "yolov8n" else "yolo11n.yaml")
    options = dict(data=str(site_root/"dataset.yaml"), epochs=a.epochs, imgsz=320,
                   batch=a.batch, device=a.device, seed=a.seed, workers=4,
                   lr0=.01, lrf=.01, pretrained=False, amp=False,
                   optimizer="SGD", patience=0,
                   project=str(a.out.parent.resolve()), name=a.out.name,
                   exist_ok=False, deterministic=True)
    if a.model == "rebar_light":
        options["trainer"] = RebarLightTrainer
    model.train(**options)
    if snapshot(a.data_root, [a.site], ["train", "val"]) != source_before:
        raise ValueError("Source dataset changed during training")
    dump(a.out/"experiment.json", dict(
        method=a.model, site=a.site, seed=a.seed, torch=torch.__version__,
        ultralytics=ultralytics.__version__, options={k:v for k,v in options.items() if k != "trainer"},
        reviewed_hashes=hashes, source_snapshot=source_before,
        protocol="within_site_2d_shared_laser_scale",
        reconstruction=("Paper-guided reconstruction; hidden expansion .5; GC ratio .25; "
                        "Figure 2 layer 8 retained; not author-verified architecture") if a.model == "rebar_light" else None,
        parameters=sum(p.numel() for p in model.model.parameters())))


def summarize(rows):
    n = sum(r["matched_intervals"] for r in rows)
    ref = sum(r["reference_intervals"] for r in rows)
    return dict(n_images=len(rows), matched_intervals=n, reference_intervals=ref,
                interval_coverage=n/ref if ref else None,
                spacing_mae_mm=sum(r["abs_error_sum"] for r in rows)/n if n else None,
                spacing_rmse_mm=(sum(r["squared_error_sum"] for r in rows)/n)**.5 if n else None,
                spacing_bias_mm=sum(r["signed_error_sum"] for r in rows)/n if n else None,
                count_mae=float(np.mean([r["count_abs_error"] for r in rows])),
                failure_rate=float(np.mean([r["predicted_count"] < 2 for r in rows])),
                fit_failure_rate=sum(r["failed_rois"] for r in rows)/max(1, sum(r["detected_rois"] for r in rows)),
                mean_detector_seconds=float(np.mean([r["detector_seconds"] for r in rows])),
                mean_detector_geometry_seconds=float(np.mean([r["detector_seconds"]+r["geometry_seconds"] for r in rows])),
                mean_geometry_seconds=float(np.mean([r["geometry_seconds"] for r in rows])))


def evaluate(a):
    import torch
    from ultralytics import YOLO
    import models  # Local trusted checkpoints deserialize reconstruction classes here.
    if a.out.exists():
        raise FileExistsError(a.out)
    if not 0 < a.hfov < 180 or a.center_tol_mm <= 0:
        raise ValueError("Invalid camera/matching settings")
    meta = json.loads((a.run/"experiment.json").read_text(encoding="utf-8"))
    if meta["site"] != a.site:
        raise ValueError("This entry is within-site only; do not mix sites")
    current = snapshot(a.data_root, [a.site], ["train", "val"])
    if current != meta["source_snapshot"]:
        raise ValueError("Source dataset changed since training")
    # The physical references are used only below for scoring, never for ROI fitting.
    store = AnnotationStore(a.annotations)
    distances = load_depth_labels(str(a.data_root/a.site/"depth_labels.txt"))
    model = YOLO(str(a.run/"weights"/"best.pt"))
    ids = current["members"][a.site+"/val"]
    first = str(a.data_root/a.site/"images"/(ids[0]+".png"))
    options = dict(imgsz=320, conf=a.conf, iou=.7, device=a.device, verbose=False)
    model.predict(first, **options)  # Exclude initialization/warm-up from timings.
    rows, detections = {m: [] for m in ("ht", "sht")}, []
    for sid in ids:
        path = a.data_root/a.site/"images"/(sid+".png")
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(path)
        h, w = image.shape[:2]
        distance = distances[sid]
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError("Invalid laser distance")
        mpp = RulerConfig(hfov_deg=a.hfov, image_width_px=w, image_height_px=h).mm_per_pixel(distance)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = model.predict(str(path), **options)[0]
        boxes = result.boxes.xyxy.cpu().numpy().tolist()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        detector_seconds = time.perf_counter()-start
        detections.append(dict(site=a.site, id=sid, boxes=boxes))
        xs = load_scaled_gt_xs(a.data_root, a.site, sid, w)
        truth = store.spacings(a.site, sid)
        if not xs or len(truth) != len(xs)-1:
            raise ValueError("Ground truth alignment mismatch")
        for method in rows:
            start = time.perf_counter()
            fitted = [fit_axis(image, b, method, threshold=a.hough_threshold) for b in boxes]
            pred = unique_centers([x for x in fitted if x is not None])
            geometry_seconds = time.perf_counter()-start
            _, gaps = matched_intervals(xs, pred, mpp, a.center_tol_mm)
            spacings = np.diff(pred)*mpp
            errors = [float(spacings[j]-truth[i]) for i, j in gaps]
            rows[method].append(dict(site=a.site, id=sid, predicted_centers=pred,
                predicted_spacings_mm=spacings.tolist(), predicted_count=len(pred),
                detected_rois=len(boxes), failed_rois=sum(x is None for x in fitted),
                reference_intervals=len(truth), matched_intervals=len(errors),
                count_abs_error=abs(len(pred)-len(xs)), abs_error_sum=sum(abs(x) for x in errors),
                squared_error_sum=sum(x*x for x in errors), signed_error_sum=sum(errors),
                detector_seconds=detector_seconds, geometry_seconds=geometry_seconds))
        print(a.site, sid, "detected", len(boxes), flush=True)
    a.out.mkdir(parents=True)
    (a.out/"detections.jsonl").write_text("".join(json.dumps(r)+"\n" for r in detections), encoding="utf-8")
    for method, records in rows.items():
        report = summarize(records)
        rng = np.random.default_rng(42)
        keys = ("spacing_mae_mm", "spacing_rmse_mm", "interval_coverage", "failure_rate")
        boot = [summarize([records[i] for i in rng.integers(0, len(records), len(records))]) for _ in range(a.bootstrap)]
        report["image_bootstrap_95ci"] = {
            k: np.percentile([v[k] for v in boot if v[k] is not None], [2.5, 97.5]).tolist()
            if any(v[k] is not None for v in boot) else None for k in keys}
        report["undefined_bootstrap_replicates"] = {k: sum(v[k] is None for v in boot) for k in keys}
        dump(a.out/(method+"_metrics.json"), report)
        (a.out/(method+"_per_image.jsonl")).write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")
    dump(a.out/"protocol.json", dict(site=a.site, hfov=a.hfov, center_tol_mm=a.center_tol_mm,
        conf=a.conf, hough_threshold=a.hough_threshold, annotation_sha256=store.sha256,
        checkpoint_sha256=hashlib.sha256((a.run/"weights"/"best.pt").read_bytes()).hexdigest(),
        scale="shared laser range, common planar pinhole approximation; no dense depth",
        scope="vertical family, center at image mid-row; horizontal x-spacing matching",
        warning="Matched-error metrics must be read together with coverage and failures"))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["train", "evaluate"])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--site", choices=["site-1", "site-2"], required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", default="0")
    p.add_argument("--dataset", type=Path)
    p.add_argument("--model", choices=["yolov8n", "yolo11n", "rebar_light"], default="rebar_light")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run", type=Path)
    p.add_argument("--annotations", type=Path)
    p.add_argument("--hfov", type=float, help="Verified camera horizontal field of view in degrees")
    p.add_argument("--center-tol-mm", type=float, default=20)
    p.add_argument("--conf", type=float, default=.25)
    p.add_argument("--hough-threshold", type=int, default=100)
    p.add_argument("--bootstrap", type=int, default=1000)
    a = p.parse_args()
    if a.mode == "train" and not a.dataset:
        p.error("train requires --dataset")
    if a.mode == "evaluate" and (not a.run or not a.annotations or a.hfov is None):
        p.error("evaluate requires --run, --annotations, and --hfov")
    if a.bootstrap < 0:
        p.error("bootstrap must be nonnegative")
    (train if a.mode == "train" else evaluate)(a)


if __name__ == "__main__":
    main()
