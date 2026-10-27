"""Independent 2D adaptation of Song et al. (2026), not their RGB-D system."""
import math
import cv2
import numpy as np


def fit_axis(image, box, method="sht", angle_range=5, threshold=100):
    """Fit two boundaries and return their midpoint at the image mid-row.

    Only the predominantly vertical bar family is evaluated, matching the
    existing x-coordinate reference annotations. Failed fits return None.
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = box
    x1, y1 = max(0, int(math.floor(x1))), max(0, int(math.floor(y1)))
    x2, y2 = min(w, int(math.ceil(x2))), min(h, int(math.ceil(y2)))
    if x2-x1 < 5 or y2-y1 < 5:
        return None
    roi = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    edges = cv2.Canny(gray, 100, 200, apertureSize=3)
    ys, xs = np.nonzero(edges)
    if len(xs) < 4:
        return None
    points = np.column_stack((xs, ys)).astype(float)
    _, vectors = np.linalg.eigh(np.cov(points, rowvar=False))
    direction = vectors[:, -1]
    normal = np.array([direction[1], -direction[0]])
    if normal[0] < 0:
        normal = -normal
    theta = math.atan2(normal[1], normal[0])
    # Restrict to the vertical family; horizontal references are not available.
    if abs(theta) > math.pi/4:
        return None
    if method == "sht":
        angles = theta + np.deg2rad(np.arange(-angle_range, angle_range+1))
        radius = int(np.ceil(np.hypot(*edges.shape)))
        candidates = []
        for t in angles:
            rho = np.rint(xs*np.cos(t)+ys*np.sin(t)).astype(int)
            votes = np.bincount(rho+radius, minlength=2*radius+1)
            for k in np.flatnonzero(votes >= threshold):
                candidates.append((int(votes[k]), float(k-radius), float(t)))
        candidates.sort(reverse=True)
    elif method == "ht":
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
        candidates = []
        for rank, (r, t) in enumerate([] if lines is None else lines[:, 0]):
            if t > np.pi/2:
                r, t = -r, t-np.pi
            if abs(t) <= np.pi/4:
                candidates.append((-rank, float(r), float(t)))
    else:
        raise ValueError(method)
    if not candidates:
        return None
    # Separate the boundary peaks at the ROI mid-row. The paper does not
    # specify a complete peak-suppression rule; this is an explicit adaptation.
    mid = (y2-y1-1)/2
    first = candidates[0]
    first_x = (first[1]-mid*np.sin(first[2]))/np.cos(first[2])
    second = None
    for item in candidates[1:]:
        local_x = (item[1]-mid*np.sin(item[2]))/np.cos(item[2])
        if abs(local_x-first_x) >= 3 and abs(item[2]-first[2]) <= np.deg2rad(10):
            second = item
            break
    if second is None:
        return None
    centers = []
    for _, rho, t in (first, second):
        centers.append(x1+(rho-((h-1)/2-y1)*np.sin(t))/np.cos(t))
    x = float(np.mean(centers))
    return x if 0 <= x < w else None


def unique_centers(values, distance=3):
    result = []
    for x in sorted(values):
        if not result or x-result[-1] >= distance:
            result.append(float(x))
    return result
