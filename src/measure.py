"""
Measurement module: pixel-to-metric conversion and GB 50204 compliance check.

Pipeline:
  1. Load shooting distance from depth_labels.txt
  2. Compute mm_per_pixel using pinhole camera model
  3. Run connected-component analysis on segmentation prediction
  4. Compute rebar spacing, diameter, count
  5. Check against GB 50204-2015 tolerances

Usage (standalone):
    python src/measure.py \
        --seg_path path/to/seg_pred.png \
        --image_id 1 \
        --subset site-1 \
        --data_root data

Usage (from Python):
    from measure import RulerConfig, load_depth_labels, measure_image
    cfg = RulerConfig()
    labels = load_depth_labels("data/site-1/depth_labels.txt")
    result = measure_image(seg_pred_binary, image_id="1", depth_labels=labels, cfg=cfg)
"""

import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    from scipy import ndimage
except ImportError:
    raise ImportError("scipy is required: pip install scipy")


# ---------------------------------------------------------------------------
# Camera / ruler configuration
# ---------------------------------------------------------------------------

@dataclass
class RulerConfig:
    """
    Camera parameters for RealSense D435 (default).
    Override for other cameras.
    """
    hfov_deg: float = 87.0          # horizontal field of view (degrees)
    image_width_px: int = 640       # width after resize (must match model input)
    image_height_px: int = 480      # height after resize

    # GB 50204-2015 tolerances (mm)
    # Spacing tolerances
    main_bar_spacing_tol_mm: float = 20.0   # main reinforcement
    stirrup_spacing_tol_mm: float = 10.0    # stirrups / ties

    # Minimum detectable rebar count to attempt measurement
    min_rebar_count: int = 2

    @property
    def hfov_rad(self) -> float:
        return math.radians(self.hfov_deg)

    def mm_per_pixel(self, distance_m: float) -> float:
        """
        Physical size of one pixel at the given shooting distance.
        Derived from pinhole camera model:
            mm_per_pixel = 2 * Z * tan(FOV/2) / image_width * 1000
        """
        return (2.0 * distance_m * math.tan(self.hfov_rad / 2.0)
                / self.image_width_px * 1000.0)


# ---------------------------------------------------------------------------
# Depth label loading
# ---------------------------------------------------------------------------

def load_depth_labels(txt_path: str) -> Dict[str, float]:
    """
    Load depth_labels.txt -> {image_id: distance_metres}.
    File format: one line per image, tab-separated: {id}\t{distance}
    """
    labels: Dict[str, float] = {}
    with open(txt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            img_id, dist = parts[0].strip(), parts[1].strip()
            try:
                labels[img_id] = float(dist)
            except ValueError:
                continue
    return labels


# ---------------------------------------------------------------------------
# Connected-component analysis
# ---------------------------------------------------------------------------

@dataclass
class RebarInstance:
    label: int
    centroid_px: Tuple[float, float]   # (x, y) in pixels
    area_px: int
    bbox_px: Tuple[int, int, int, int]  # (x_min, y_min, x_max, y_max)
    diameter_px: float                  # equivalent circular diameter


def extract_instances(seg_binary: np.ndarray,
                      min_area_px: int = 50) -> List[RebarInstance]:
    """
    Extract vertical rebar instances from a binary segmentation mask using
    column-projection peak detection.

    Connected-component analysis breaks when horizontal stirrups connect all
    vertical rebars into one large component (aspect ratio < 2.5, so the old
    filter misses it, centroid lands at the image center, bbox spans the full
    width). Column projection is immune: horizontal bars only raise the flat
    baseline; each vertical rebar produces a distinct tall local maximum.

    Steps:
      1. Sum foreground pixels per column  ->  1-D projection onto x-axis.
      2. Smooth lightly (uniform filter, 5 px) to suppress single-pixel noise.
      3. Find peaks: height >= 5 % of image height, prominence >= 1.5 %,
         minimum separation 8 px.
      4. Compute weighted centroid x and FWHM-based diameter per peak using
         scipy.signal.peak_widths (rel_height=0.5 = half-prominence level).
      5. Return one RebarInstance per peak, sorted left to right.

    seg_binary : (H, W) uint8 or bool, foreground = 1 / True
    min_area_px: discard peaks whose foreground pixel count is below this
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks, peak_widths as sp_peak_widths

    H, W = seg_binary.shape
    col_proj = seg_binary.sum(axis=0).astype(float)
    col_smooth = uniform_filter1d(col_proj, size=5)

    if col_smooth.max() == 0:
        return []

    min_height = H * 0.05       # rebar must cover >= 5 % of image height
    min_prom   = H * 0.015      # peak must stand out >= 1.5 % above neighbours
    min_distance = 30
    min_width_px  = 15.0

    peaks, _ = find_peaks(
        col_smooth,
        height=min_height,
        distance=min_distance,
        prominence=min_prom,
    )
    if len(peaks) == 0:
        return []

    # Width at half-prominence level  ->  robust rebar diameter estimate
    widths, _, left_ips, right_ips = sp_peak_widths(col_smooth, peaks, rel_height=0.5)

    # Compute valley-based hard boundaries between adjacent peaks.
    # This prevents FWHM bands from overlapping when two rebars are close together,
    # which would bias the centroid toward the midpoint of the pair.
    boundaries = [0] + [
        int(peaks[i] + np.argmin(col_smooth[peaks[i]:peaks[i+1]]))
        for i in range(len(peaks) - 1)
    ] + [W]

    instances: List[RebarInstance] = []
    for idx, (peak_x, w_px, l_ip, r_ip) in enumerate(
            zip(peaks, widths, left_ips, right_ips)):
        if w_px < min_width_px:
            continue
        # Use valley boundaries instead of FWHM bounds for centroid calculation
        l = boundaries[idx]
        r = boundaries[idx + 1] - 1

        # Weighted centroid x within the valley-bounded band
        band = col_smooth[l:r + 1]
        cx = float(l) + float(np.average(np.arange(len(band)),
                                          weights=band + 1e-9))

        # Centroid y, area, and tight bbox from actual mask pixels in band
        col_slice = seg_binary[:, l:r + 1]
        area = int(col_slice.sum())
        if area < min_area_px:
            continue

        ys, xs_local = np.where(col_slice)
        cy    = float(ys.mean())
        y_min = int(ys.min());  y_max = int(ys.max())
        x_min = l + int(xs_local.min())
        x_max = l + int(xs_local.max())

        instances.append(RebarInstance(
            label=idx + 1,
            centroid_px=(cx, cy),
            area_px=area,
            bbox_px=(x_min, y_min, x_max, y_max),
            diameter_px=float(w_px),
        ))

    return instances


def extract_instances_from_heatmap(heatmap: np.ndarray,
                                   seg_binary: np.ndarray,
                                   min_distance: int = 30,
                                   peak_threshold: float = 0.15,
                                   min_area_px: int = 50) -> List[RebarInstance]:
    """
    Detect rebar instances from the center heatmap (sigmoid output).

    heatmap    : (H, W) float32 in [0, 1], output of sigmoid(center_logit)
    seg_binary : (H, W) uint8 binary segmentation (used for area/diameter)
    min_distance: minimum pixel separation between adjacent peaks
    peak_threshold: minimum heatmap value to count as a peak

    Falls back to extract_instances(seg_binary) when no peaks are found.
    """
    from scipy.ndimage import uniform_filter1d
    from scipy.signal import find_peaks, peak_widths as sp_peak_widths

    H, W = heatmap.shape

    # Project heatmap onto x-axis using max pooling along y
    col_proj   = heatmap.max(axis=0).astype(float)
    col_smooth = uniform_filter1d(col_proj, size=5)

    if col_smooth.max() < peak_threshold:
        return extract_instances(seg_binary, min_area_px=min_area_px)

    peaks, props = find_peaks(
        col_smooth,
        height=peak_threshold,
        distance=min_distance,
        prominence=peak_threshold * 0.3,
    )
    if len(peaks) == 0:
        return extract_instances(seg_binary, min_area_px=min_area_px)

    widths, _, left_ips, right_ips = sp_peak_widths(col_smooth, peaks, rel_height=0.5)

    # Valley-based hard boundaries between adjacent peaks
    boundaries = [0] + [
        int(peaks[i] + np.argmin(col_smooth[peaks[i]:peaks[i + 1]]))
        for i in range(len(peaks) - 1)
    ] + [W]

    instances: List[RebarInstance] = []
    for idx, (peak_x, w_px) in enumerate(zip(peaks, widths)):
        l = boundaries[idx]
        r = boundaries[idx + 1] - 1

        band = col_smooth[l:r + 1]
        cx   = float(l) + float(np.average(np.arange(len(band)),
                                            weights=band + 1e-9))

        col_slice = seg_binary[:, l:r + 1]
        area = int(col_slice.sum())
        if area < min_area_px:
            continue

        ys, xs_local = np.where(col_slice) if area > 0 else (np.array([H // 2]), np.array([0]))
        cy    = float(ys.mean())
        y_min = int(ys.min());  y_max = int(ys.max())
        x_min = l + int(xs_local.min())
        x_max = l + int(xs_local.max())

        instances.append(RebarInstance(
            label=idx + 1,
            centroid_px=(cx, cy),
            area_px=area,
            bbox_px=(x_min, y_min, x_max, y_max),
            diameter_px=max(float(w_px), float(x_max - x_min)),
        ))

    return instances if instances else extract_instances(seg_binary, min_area_px=min_area_px)




@dataclass
class MeasurementResult:
    image_id: str
    distance_m: float
    mpp: float                          # mm per pixel
    rebar_count: int
    spacings_mm: List[float]            # adjacent spacings, left to right
    diameters_mm: List[float]           # per-instance equivalent diameter
    mean_spacing_mm: Optional[float]
    std_spacing_mm: Optional[float]
    mean_diameter_mm: Optional[float]
    compliance: Dict[str, object]       # GB 50204 compliance results
    warning: Optional[str] = None


def check_compliance(spacings_mm: List[float],
                     cfg: RulerConfig, design_rules=None) -> Dict[str, object]:
    """
    Check spacing against GB 50204-2015 tolerances.
    Returns a dict with pass/fail status and per-spacing deviations.
    """
    if design_rules is not None:
        from inspection_data import design_compliance
        return design_compliance(spacings_mm, design_rules)
    if not spacings_mm:
        return {"status": "N/A", "reason": "no spacings computed"}

    results = []
    for i, s in enumerate(spacings_mm):
        # Use stirrup tolerance as the stricter default; caller can override
        tol = cfg.stirrup_spacing_tol_mm
        deviation = s - (sum(spacings_mm) / len(spacings_mm))  # deviation from mean
        results.append({
            "index": i,
            "spacing_mm": round(s, 1),
            "deviation_from_mean_mm": round(deviation, 1),
            "within_stirrup_tol": abs(deviation) <= tol,
            "within_main_tol": abs(deviation) <= cfg.main_bar_spacing_tol_mm,
        })

    all_pass_stirrup = all(r["within_stirrup_tol"] for r in results)
    all_pass_main    = all(r["within_main_tol"]    for r in results)

    return {
        "status": "PASS" if all_pass_stirrup else ("PASS(main)" if all_pass_main else "FAIL"),
        "stirrup_tolerance_mm": cfg.stirrup_spacing_tol_mm,
        "main_bar_tolerance_mm": cfg.main_bar_spacing_tol_mm,
        "per_spacing": results,
    }


def measure_image(seg_pred: np.ndarray,
                  image_id: str,
                  depth_labels: Dict[str, float],
                  cfg: Optional[RulerConfig] = None,
                  min_area_px: int = 50,
                  instances_override: Optional[list] = None,
                  design_rules=None) -> MeasurementResult:
    """
    Full measurement pipeline for one image.

    seg_pred           : (H, W) binary array (0/1 or bool)
    instances_override : if provided, skip extract_instances() and use this list
    """
    if cfg is None:
        cfg = RulerConfig()

    if image_id not in depth_labels:
        return MeasurementResult(
            image_id=image_id, distance_m=float("nan"), mpp=float("nan"),
            rebar_count=0, spacings_mm=[], diameters_mm=[],
            mean_spacing_mm=None, std_spacing_mm=None, mean_diameter_mm=None,
            compliance={"status": "N/A", "reason": "no distance label"},
            warning=f"image_id '{image_id}' not found in depth_labels",
        )

    distance_m = depth_labels[image_id]
    mpp = cfg.mm_per_pixel(distance_m)

    # --- Instance extraction ---
    seg_binary = (seg_pred > 0).astype(np.uint8)
    instances = (instances_override if instances_override is not None
                 else extract_instances(seg_binary, min_area_px=min_area_px))
    n = len(instances)

    if n < cfg.min_rebar_count:
        return MeasurementResult(
            image_id=image_id, distance_m=distance_m, mpp=mpp,
            rebar_count=n, spacings_mm=[], diameters_mm=[],
            mean_spacing_mm=None, std_spacing_mm=None, mean_diameter_mm=None,
            compliance={"status": "N/A", "reason": f"only {n} rebar detected"},
            warning=f"fewer than {cfg.min_rebar_count} rebar detected",
        )

    # --- Spacings (x-direction only: horizontal distance between vertical rebars) ---
    spacings_mm: List[float] = []
    for i in range(n - 1):
        cx1 = instances[i].centroid_px[0]
        cx2 = instances[i + 1].centroid_px[0]
        spacings_mm.append((cx2 - cx1) * mpp)

    # --- Diameters ---
    diameters_mm = [inst.diameter_px * mpp for inst in instances]

    mean_spacing = float(np.mean(spacings_mm)) if spacings_mm else None
    std_spacing  = float(np.std(spacings_mm))  if spacings_mm else None
    mean_diam    = float(np.mean(diameters_mm)) if diameters_mm else None

    compliance = check_compliance(spacings_mm, cfg, design_rules)

    return MeasurementResult(
        image_id=image_id,
        distance_m=distance_m,
        mpp=round(mpp, 4),
        rebar_count=n,
        spacings_mm=[round(s, 1) for s in spacings_mm],
        diameters_mm=[round(d, 1) for d in diameters_mm],
        mean_spacing_mm=round(mean_spacing, 1) if mean_spacing is not None else None,
        std_spacing_mm=round(std_spacing, 1)   if std_spacing  is not None else None,
        mean_diameter_mm=round(mean_diam, 1)   if mean_diam    is not None else None,
        compliance=compliance,
    )


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------

def evaluate_dataset(seg_dir: str,
                     depth_labels: Dict[str, float],
                     cfg: Optional[RulerConfig] = None,
                     min_area_px: int = 50) -> List[MeasurementResult]:
    """
    Run measure_image on all .png files in seg_dir.
    seg_dir: folder of binary segmentation prediction images (0/255 PNG).
    """
    results = []
    for p in sorted(Path(seg_dir).glob("*.png")):
        img = np.array(Image.open(p).convert("L"))
        seg_binary = (img > 127).astype(np.uint8)
        result = measure_image(seg_binary, p.stem, depth_labels, cfg, min_area_px)
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(r: MeasurementResult):
    print(f"\n=== Image {r.image_id} ===")
    if r.warning:
        print(f"  WARNING: {r.warning}")
        return
    print(f"  Distance: {r.distance_m} m  |  mm/px: {r.mpp}")
    print(f"  Rebar count: {r.rebar_count}")
    print(f"  Spacings (mm): {r.spacings_mm}")
    print(f"  Mean spacing: {r.mean_spacing_mm} mm  |  Std: {r.std_spacing_mm} mm")
    print(f"  Diameters (mm): {r.diameters_mm}")
    print(f"  Mean diameter: {r.mean_diameter_mm} mm")
    print(f"  GB 50204 compliance: {r.compliance['status']}")
    if "per_spacing" in r.compliance:
        for s in r.compliance["per_spacing"]:
            flag = "OK" if s["within_stirrup_tol"] else "FAIL"
            print(f"    spacing[{s['index']}]: {s['spacing_mm']} mm  "
                  f"dev={s['deviation_from_mean_mm']:+.1f} mm  [{flag}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seg_path", required=True,
                   help="Path to binary segmentation prediction PNG (0/255)")
    p.add_argument("--image_id", required=True,
                   help="Image ID (must match depth_labels.txt)")
    p.add_argument("--subset", default="site-1", choices=["site-1", "site-2"])
    p.add_argument("--data_root", default="data")
    p.add_argument("--hfov", type=float, default=87.0,
                   help="Camera horizontal FOV in degrees (default: RealSense D435)")
    p.add_argument("--img_w", type=int, default=640)
    p.add_argument("--img_h", type=int, default=480)
    p.add_argument("--min_area_px", type=int, default=50)
    args = p.parse_args()

    cfg = RulerConfig(
        hfov_deg=args.hfov,
        image_width_px=args.img_w,
        image_height_px=args.img_h,
    )

    labels_path = Path(args.data_root) / args.subset / "depth_labels.txt"
    depth_labels = load_depth_labels(str(labels_path))

    seg_img = np.array(Image.open(args.seg_path).convert("L"))
    seg_binary = (seg_img > 127).astype(np.uint8)

    result = measure_image(seg_binary, args.image_id, depth_labels, cfg,
                           args.min_area_px)
    _print_result(result)
