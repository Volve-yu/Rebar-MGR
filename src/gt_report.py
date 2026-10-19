"""
Ground-truth report generator using Gemini Vision.

For each image Gemini is asked to locate the horizontal centroid (x-pixel)
of every distinct vertical rebar.  Spacings are then computed with the same
pinhole-camera scale conversion used by measure.py, and GB 50204-2015
compliance is checked with the identical function.

The output format is identical to report.py so the two can be compared
directly for the T-IM measurement-accuracy evaluation.

Usage:
    python src/gt_report.py \
        --data_root data \
        --subsets site-1 site-2 \
        --out_dir gt_reports \
        --api_key YOUR_GEMINI_API_KEY

    # Use val split only (default); process all images in val split:
    python src/gt_report.py --data_root data --out_dir gt_reports

Requirements:
    pip install google-genai

Output:
    gt_reports/
        results.csv              -- one row per image (same columns as run_measure.py)
        summary.txt              -- batch statistics
        {subset}_{id}.txt        -- per-image report
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).parent))
from measure import (RulerConfig, load_depth_labels, check_compliance,
                     MeasurementResult, measure_image, extract_instances)


# ---------------------------------------------------------------------------
# Knowledge Graph: GB 50204-2015 rebar-related clauses
# Each node: nominal_mm is used only for signed-deviation display;
# compliance is checked against tolerance bounds relative to the mean spacing.
# ---------------------------------------------------------------------------

GB50204_KG: Dict[str, dict] = {
    "longitudinal_spacing": {
        "tol_lower_mm": -20.0,
        "tol_upper_mm":  20.0,
        "clause": (
            "GB 50204-2015 Section 8.3.2: Allowable deviation for longitudinal "
            "main reinforcing bar spacing: ±20 mm."
        ),
    },
    "stirrup_spacing": {
        "tol_lower_mm": -10.0,
        "tol_upper_mm":  10.0,
        "clause": (
            "GB 50204-2015 Section 8.3.2: Allowable deviation for stirrup "
            "spacing: ±10 mm."
        ),
    },
    "rebar_diameter": {
        "tol_lower_mm": -3.0,
        "tol_upper_mm":  3.0,
        "clause": (
            "GB 50204-2015 Section 5.2.2: Diameter deviation shall conform to "
            "the product standard; typical tolerance ±3 mm for hot-rolled bars."
        ),
    },
}

# Few-shot CoT demonstrations (expert reasoning chains).
# Each entry: measurements summary + four-step reasoning chain + narrative.
FEW_SHOT_CHAINS = [
    {
        "summary": (
            "rebar_count=5, spacings_mm=[195, 202, 198, 205, 200], "
            "mean=200.0 mm, std=3.7 mm, diameter≈20 mm"
        ),
        "chain": (
            "Step 1 (Attribute identification): 5 rebars → 4 longitudinal "
            "spacings measured.\n"
            "Step 2 (Clause grounding): GB 50204-2015 §8.3.2 applies — "
            "tolerance ±20 mm for main longitudinal bars.\n"
            "Step 3 (Margin computation): deviations from mean = "
            "[-5, +2, -2, +5, 0] mm; all |deviation| ≤ 20 mm → I_a = 1 for all.\n"
            "Step 4 (Conclusion): overall status = PASS.\n"
            "### Professional Narrative:\n"
            "Five rebar instances were detected with spacings of 195, 202, "
            "198, 205, and 200 mm (mean 200.0 mm, std 3.7 mm). All spacings fall within "
            "the GB 50204-2015 longitudinal bar tolerance of ±20 mm. The installation "
            "is compliant. Result: PASS."
        ),
    },
    {
        "summary": (
            "rebar_count=4, spacings_mm=[185, 230, 195], "
            "mean=203.3 mm, std=19.4 mm, diameter≈16 mm"
        ),
        "chain": (
            "Step 1 (Attribute identification): 4 rebars → 3 longitudinal "
            "spacings measured.\n"
            "Step 2 (Clause grounding): GB 50204-2015 §8.3.2 — tolerance "
            "±20 mm for main longitudinal bars.\n"
            "Step 3 (Margin computation): deviations from mean = "
            "[-18.3, +26.7, -8.3] mm; spacing[1]=230 mm exceeds +20 mm → I_a = 0.\n"
            "Step 4 (Conclusion): overall status = FAIL.\n"
            "### Professional Narrative:\n"
            "Four rebar instances were detected with spacings of 185, 230, "
            "and 195 mm (mean 203.3 mm, std 19.4 mm). Spacing at position 1 (230 mm) "
            "deviates +26.7 mm from the mean, exceeding the GB 50204-2015 ±20 mm "
            "tolerance. The installation is non-compliant and requires correction "
            "before concrete placement. Result: FAIL."
        ),
    },
]


# ---------------------------------------------------------------------------
# Stage 1: Knowledge-graph-guided structured assessment
# ---------------------------------------------------------------------------

def _compute_risk_level(spacings_mm: List[float]) -> dict:
    """
    5-level risk classification from deviation magnitude and non-compliance rate.

    Let d_i = |s_i - mu| (absolute deviation from mean).
    f_nc = fraction of spacings with d_i > main_bar_tol (20 mm).
    d_max = max(d_i).

    Levels: SAFE -> ACCEPTABLE -> MARGINAL -> WARNING -> CRITICAL
    """
    if not spacings_mm:
        return {"level": "UNKNOWN", "max_dev_mm": None, "nc_ratio": None}

    mu = sum(spacings_mm) / len(spacings_mm)
    devs = [abs(s - mu) for s in spacings_mm]
    d_max = max(devs)
    nc_ratio = sum(1 for d in devs if d > 20) / len(spacings_mm)

    if nc_ratio == 0 and d_max <= 5:
        level = "SAFE"
    elif nc_ratio == 0 and d_max <= 10:
        level = "ACCEPTABLE"
    elif nc_ratio <= 0.25 and d_max <= 40:
        level = "MARGINAL"
    elif nc_ratio <= 0.5 or d_max <= 60:
        level = "WARNING"
    else:
        level = "CRITICAL"

    return {
        "level": level,
        "max_dev_mm": round(d_max, 1),
        "nc_ratio": round(nc_ratio, 3),
        "nc_count": sum(1 for d in devs if d > 20),
        "n_spacings": len(spacings_mm),
    }


def _trend_analysis(spacings_mm: List[float]) -> dict:
    """
    Detect monotonic trend in the left-to-right spacing sequence using
    linear regression slope and Kendall's tau.

    A positive slope means spacings widen toward the right (possible
    formwork drift or uneven rebar placement). A negative slope means
    spacings narrow. Both indicate systematic installation error beyond
    random construction tolerance.

    Returns dict with trend label, slope (mm per gap), and tau.
    """
    n = len(spacings_mm)
    if n < 3:
        return {"trend": "INSUFFICIENT_DATA", "slope_mm_per_gap": None, "tau": None}

    try:
        from scipy.stats import kendalltau as _kendalltau
        tau, p_val = _kendalltau(range(n), spacings_mm)
    except ImportError:
        tau, p_val = 0.0, 1.0

    # OLS slope (mm per gap index)
    mean_i = (n - 1) / 2.0
    mean_s = sum(spacings_mm) / n
    num = sum((i - mean_i) * (s - mean_s) for i, s in enumerate(spacings_mm))
    den = sum((i - mean_i) ** 2 for i in range(n))
    slope = num / den if den > 0 else 0.0

    # Classify: threshold 3 mm/gap to ignore noise
    if abs(slope) < 3.0:
        trend = "UNIFORM"
    elif slope > 0:
        trend = "INCREASING"
    else:
        trend = "DECREASING"

    return {
        "trend": trend,
        "slope_mm_per_gap": round(slope, 2),
        "tau": round(tau, 3) if tau is not None else None,
        "first_mm": round(spacings_mm[0], 1),
        "last_mm": round(spacings_mm[-1], 1),
    }


def graph_traverse(result: MeasurementResult) -> dict:
    """
    Map the measurement record to GB50204_KG nodes and compute per-attribute
    signed deviations and compliance indicators.  Also computes risk level
    and spacing trend analysis.

    Returns a structured assessment dict S used by Stage 2.
    """
    S: dict = {
        "rebar_count": result.rebar_count,
        "mean_spacing_mm": result.mean_spacing_mm,
        "std_spacing_mm": result.std_spacing_mm,
        "spacings_mm": result.spacings_mm,
        "mean_diameter_mm": result.mean_diameter_mm,
        "overall_status": result.compliance.get("status", "N/A"),
        "attributes": [],
    }

    mean = result.mean_spacing_mm
    node = GB50204_KG["longitudinal_spacing"]
    if mean is not None and result.spacings_mm:
        spacing_attrs = []
        for i, sp in enumerate(result.spacings_mm):
            margin = sp - mean
            compliant = node["tol_lower_mm"] <= margin <= node["tol_upper_mm"]
            spacing_attrs.append({
                "index": i,
                "measured_mm": round(sp, 1),
                "margin_mm": round(margin, 1),
                "tolerance": [node["tol_lower_mm"], node["tol_upper_mm"]],
                "clause": node["clause"],
                "compliant": compliant,
            })
        S["attributes"] = spacing_attrs

    if result.mean_diameter_mm is not None:
        d_node = GB50204_KG["rebar_diameter"]
        S["diameter_assessment"] = {
            "measured_mm": result.mean_diameter_mm,
            "clause": d_node["clause"],
            "note": "Diameter tolerance verified against product standard.",
        }

    # Risk level classification and trend analysis
    S["risk"] = _compute_risk_level(result.spacings_mm or [])
    S["trend"] = _trend_analysis(result.spacings_mm or [])

    return S


# ---------------------------------------------------------------------------
# Stage 2: Few-shot CoT prompt construction
# ---------------------------------------------------------------------------

def _build_cot_prompt(S: dict) -> str:
    """
    Build a few-shot chain-of-thought prompt for the VLM.
    S is the structured assessment from graph_traverse().
    """
    # Format the current assessment
    attr_lines = []
    for a in S.get("attributes", []):
        flag = "PASS" if a["compliant"] else "FAIL"
        attr_lines.append(
            f"  spacing[{a['index']}]: {a['measured_mm']} mm, "
            f"deviation {a['margin_mm']:+.1f} mm, "
            f"tol [{a['tolerance'][0]:+.0f}, {a['tolerance'][1]:+.0f}] mm, "
            f"clause: \"{a['clause']}\", status: {flag}"
        )
    attrs_str = "\n".join(attr_lines) if attr_lines else "  (no spacing data)"

    current = (
        f"rebar_count={S['rebar_count']}, "
        f"spacings_mm={S['spacings_mm']}, "
        f"mean={S['mean_spacing_mm']} mm, "
        f"std={S['std_spacing_mm']} mm"
        + (f", diameter≈{S['mean_diameter_mm']} mm" if S.get('mean_diameter_mm') else "")
    )

    # Risk level and trend summary for the prompt
    risk  = S.get("risk",  {})
    trend = S.get("trend", {})
    risk_str = (
        f"risk_level={risk.get('level','N/A')}, "
        f"max_deviation={risk.get('max_dev_mm','?')} mm, "
        f"non_compliant_ratio={risk.get('nc_ratio','?')}"
    ) if risk else ""
    trend_str = (
        f"trend={trend.get('trend','N/A')}, "
        f"slope={trend.get('slope_mm_per_gap','?')} mm/gap, "
        f"tau={trend.get('tau','?')}"
    ) if trend else ""

    # Build few-shot demonstrations
    demo_blocks = []
    for k, demo in enumerate(FEW_SHOT_CHAINS):
        demo_blocks.append(
            f"--- Example {k+1} ---\n"
            f"Measurements: {demo['summary']}\n"
            f"Reasoning chain:\n{demo['chain']}"
        )
    demos = "\n\n".join(demo_blocks)

    return (
        "You are a construction quality inspector. "
        "Use the four-step reasoning protocol shown in the examples below to "
        "assess the rebar installation and write an inspection report.\n\n"
        "RULES:\n"
        "1. Follow the four-step chain exactly: Attribute identification → "
        "Clause grounding → Margin computation → Compliance conclusion.\n"
        "2. After the chain, write a 2-4 sentence professional narrative under the exact heading "
        "'### Professional Narrative:'.\n"
        "3. Do NOT change any numerical values. "
        "Do NOT reference standards other than GB 50204-2015.\n\n"
        f"{demos}\n\n"
        "--- Current inspection ---\n"
        f"Measurements: {current}\n"
        f"Structured assessment (from knowledge graph traversal):\n{attrs_str}\n"
        f"Overall compliance: {S['overall_status']}\n"
        + (f"Risk level: {risk_str}\n" if risk_str else "")
        + (f"Spacing trend: {trend_str}\n" if trend_str else "")
        + "\nRules for narrative:\n"
        "- State the risk level and explain what it means.\n"
        "- If trend is INCREASING or DECREASING, flag it as a potential systematic error.\n"
        "- Do NOT change any numerical values.\n\n"
        "Reasoning chain and ### Professional Narrative:"
    )


# ---------------------------------------------------------------------------
# Stage 3: Numerical consistency verification
# ---------------------------------------------------------------------------

def _verify_consistency(report_text: str, result: MeasurementResult,
                        eps_mm: float = 1.0) -> tuple:
    """
    Extract numeric mm-values from the generated report text and compare
    against the algorithmic measurement record.

    Returns (is_consistent: bool, correction_instructions: str).
    Uses the algorithmic measurements as an external oracle (not VLM self-check).
    """
    nums = re.findall(r'(\d+\.?\d*)\s*mm', report_text)
    floats = [float(v) for v in nums]

    issues = []
    if result.mean_spacing_mm is not None:
        target = result.mean_spacing_mm
        # Flag any single number that is plausibly a mean spacing but wrong
        candidates = [v for v in floats if 30 < v < 600]
        for v in candidates:
            if abs(v - target) > eps_mm and abs(v - target) < 100:
                # Likely a misquoted mean spacing
                issues.append(
                    f"mean spacing should be {target} mm (found {v:.1f} mm in text)"
                )
                break  # report first mismatch only

    if result.std_spacing_mm is not None:
        target_std = result.std_spacing_mm
        small_vals = [v for v in floats if 0 < v < 30]
        for v in small_vals:
            if abs(v - target_std) > eps_mm and target_std > 0:
                issues.append(
                    f"spacing std should be {target_std} mm (found {v:.1f} mm in text)"
                )
                break

    if not issues:
        return True, ""

    correction = (
        "The report contains a numerical inconsistency with the algorithmic measurements.\n"
        "Please correct only the following values and rewrite the report:\n"
        + "\n".join(f"  - {issue}" for issue in issues)
        + "\nDo not change any other values. Reply with the corrected report only."
    )
    return False, correction


def _normalize_professional_narrative_heading(report_text: str) -> str:
    """Make the downstream reference extractor see one canonical narrative heading."""
    if re.search(r"^\s*#{3,6}\s*Professional\s+Narrative\s*:?\s*$",
                 report_text, flags=re.IGNORECASE | re.MULTILINE):
        return report_text

    normalized = re.sub(
        r"^\s*#{3,6}\s*Narrative\s*:?\s*$",
        "### Professional Narrative:",
        report_text,
        count=1,
        flags=re.IGNORECASE | re.MULTILINE,
    )
    if normalized != report_text:
        return normalized

    normalized = re.sub(
        r"(?im)^(\s*)Narrative\s*:\s*",
        r"\1### Professional Narrative:\n",
        report_text,
        count=1,
    )
    if normalized != report_text:
        return normalized

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", report_text) if p.strip()]
    if paragraphs:
        return "### Professional Narrative:\n" + paragraphs[-1]
    return report_text


def _generate_report_with_verification(
    image_pil: Image.Image,
    S: dict,
    result: MeasurementResult,
    model_name: str,
    api_key: str,
    base_url: Optional[str],
    max_retries: int = 2,
) -> str:
    """
    Stage 2 + Stage 3: generate report with few-shot CoT prompt, then verify
    numerical consistency against algorithmic measurements. Re-prompt once if
    inconsistency is detected.
    """
    prompt = _build_cot_prompt(S)
    try:
        report_text = _call_vlm(image_pil, prompt, model_name, api_key, base_url)
    except Exception as e:
        return f"[VLM generation failed: {e}]"

    for _ in range(max_retries):
        ok, correction = _verify_consistency(report_text, result)
        if ok:
            break
        # Re-prompt with correction instructions (Stage 3)
        correction_prompt = (
            f"Here is a draft inspection report:\n\n{report_text}\n\n{correction}"
        )
        try:
            report_text = _call_vlm(image_pil, correction_prompt,
                                     model_name, api_key, base_url)
        except Exception as e:
            report_text += f"\n[Correction attempt failed: {e}]"
            break

    return _normalize_professional_narrative_heading(report_text)


# ---------------------------------------------------------------------------
# Stage 4: Evidence attribution  Attr(r̂)  (eq:attr_set, eq:attr_acc)
# ---------------------------------------------------------------------------

def _attribute_claims(report_text: str, S: dict,
                      eps_mm: float = 1.0) -> tuple:
    """
    For each factual claim c_i extracted from report_text, check whether its
    attribution set E(c_i) is non-empty (traceable to S or GB50204_KG).

    Numerical claims: attributed if the value matches any entry in S within eps_mm.
    Regulatory claims: attributed if a recognised clause ID from GB50204_KG appears.

    Returns:
        attr_score   : float in [0, 1], Attr(r̂) = |attributed| / |claims|
        unattributed : list of claim strings where E(c_i) = empty set
    """
    # Build the set of attributable numerical values from S
    attr_values: set = set()
    for key in ("mean_spacing_mm", "std_spacing_mm", "mean_diameter_mm"):
        v = S.get(key)
        if v is not None:
            attr_values.add(float(v))
    if S.get("rebar_count") is not None:
        attr_values.add(float(S["rebar_count"]))
    for a in S.get("attributes", []):
        attr_values.add(float(a["measured_mm"]))
        attr_values.add(abs(float(a["tolerance"][0])))
        attr_values.add(abs(float(a["tolerance"][1])))
        attr_values.add(abs(float(a["margin_mm"])))   # signed deviation → attributed
    for sp in S.get("spacings_mm") or []:
        attr_values.add(float(sp))
    # Risk and trend derived values are also attributable
    if S.get("risk", {}).get("max_dev_mm") is not None:
        attr_values.add(float(S["risk"]["max_dev_mm"]))
    if S.get("trend", {}).get("slope_mm_per_gap") is not None:
        attr_values.add(abs(float(S["trend"]["slope_mm_per_gap"])))

    # Build recognisable clause identifiers from GB50204_KG
    clause_ids: list = ["GB 50204", "GB50204", "50204-2015"]
    for node in GB50204_KG.values():
        for m in re.finditer(r'Section\s+[\d.]+', node["clause"]):
            clause_ids.append(m.group(0))

    claims: list = []  # (claim_text: str, attributed: bool)

    # Numerical claims — all "X mm" or "Xmm" patterns
    for m in re.finditer(r'(\d+\.?\d*)\s*mm', report_text):
        val = float(m.group(1))
        attributed = any(abs(val - av) <= eps_mm for av in attr_values)
        claims.append((f"{val:.1f} mm", attributed))

    # Regulatory claims — standard / section references
    for m in re.finditer(
        r'GB\s*50204(?:-\d{4})?(?:\s+(?:Section|Clause)\s+[\d.]+)?'
        r'|(?:Section|Clause)\s+[\d.]+',
        report_text,
    ):
        ref = m.group(0).strip()
        attributed = any(cid in ref or ref in cid for cid in clause_ids)
        claims.append((ref, attributed))

    if not claims:
        return 1.0, []

    unattributed = [text for text, att in claims if not att]
    attr_score = round(sum(att for _, att in claims) / len(claims), 4)
    return attr_score, unattributed

def _call_vlm(image_pil: Image.Image, prompt: str,
              model_name: str, api_key: str,
              base_url: str = None,
              extra_images: list = None) -> str:
    """Call a vision-language model. Gemini models use google-genai; everything else
    uses the OpenAI-compatible chat/completions endpoint (works with any relay/proxy).
    extra_images: optional list of (Image, caption_str) tuples appended after the main image."""
    if model_name.startswith("gemini") and base_url is None:
        return _call_gemini(image_pil, prompt, model_name, api_key)
    return _call_openai(image_pil, prompt, model_name, api_key, base_url,
                        extra_images=extra_images)


def _call_gemini(image_pil: Image.Image, prompt: str,
                 model_name: str, api_key: str) -> str:
    try:
        from google import genai
    except ImportError:
        raise ImportError("google-genai is required: pip install google-genai")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model_name,
        contents=[prompt, image_pil],
    )
    return response.text


def _call_openai(image_pil: Image.Image, prompt: str,
                 model_name: str, api_key: str,
                 base_url: str = None,
                 extra_images: list = None) -> str:
    """OpenAI-compatible vision call via chat/completions.
    Works with any OpenAI-compatible provider (Dashscope, etc.).
    extra_images: list of (Image, caption_str) appended after the main image."""
    import base64, io
    try:
        import requests as _rq
    except ImportError:
        raise ImportError("requests is required: pip install requests")

    def _encode(img: Image.Image) -> str:
        img = img.copy()
        if max(img.size) > 1024:
            ratio = 1024 / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)),
                             Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    content = [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _encode(image_pil)}},
    ]
    if extra_images:
        for extra_img, caption in extra_images:
            content.append({"type": "text", "text": caption})
            content.append({"type": "image_url", "image_url": {"url": _encode(extra_img)}})

    endpoint = (base_url or "https://api.openai.com/v1").rstrip("/")
    hdrs = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = _rq.post(f"{endpoint}/chat/completions", headers=hdrs, json={
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 512,
    }, timeout=120)

    if not resp.ok:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")
    return resp.json()["choices"][0]["message"]["content"]


def _parse_json(text: str) -> dict:
    """Strip markdown fences then parse JSON."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip().rstrip("`").strip()
    return json.loads(text)




def _build_prompt(mpp: float, distance_m: float, img_w: int, img_h: int,
                  use_seg: bool = False,
                  algo_result: Optional[MeasurementResult] = None) -> str:
    """
    If algo_result is provided, the measurements are already computed.
    Ask VLM to write a professional inspection report based on those numbers.
    Otherwise, ask VLM to estimate measurements from the image.
    """
    if algo_result is not None:
        sp = algo_result.spacings_mm
        return (
            "You are a construction quality inspector writing an inspection report.\n\n"
            "The following measurements were computed algorithmically from the segmentation mask:\n"
            f"  - Rebar count: {algo_result.rebar_count}\n"
            f"  - Individual spacings (mm): {sp}\n"
            f"  - Mean spacing: {algo_result.mean_spacing_mm} mm, "
            f"Std: {algo_result.std_spacing_mm} mm\n"
            f"  - Mean diameter: {algo_result.mean_diameter_mm} mm\n"
            f"  - GB 50204-2015 compliance: {algo_result.compliance.get('status', 'N/A')}\n\n"
            "Write a concise professional inspection report (2-4 sentences) describing "
            "the rebar installation quality based on these numbers. "
            "Do NOT change or re-estimate any values. "
            "Reply in English only, plain text, no JSON."
        )

    if use_seg:
        seg_note = (
            "You are given TWO images:\n"
            "  Image 1: a binary segmentation mask — white regions are rebars, black is background.\n"
            "  Image 2: the original photo for visual reference.\n"
            "Use Image 1 as your primary measurement source. "
            "Use Image 2 only to confirm rebar orientation and count.\n\n"
        )
    else:
        seg_note = ""

    return (
        "You are a construction quality inspector measuring steel reinforcing bars (rebars).\n\n"
        + seg_note +
        f"Camera scale: {mpp:.4f} mm/pixel. Shooting distance: {distance_m} m.\n"
        f"Image size: {img_w}×{img_h} px (width×height).\n\n"
        "Task:\n"
        "1. Count the number of distinct VERTICAL rebar columns "
        "(ignore horizontal stirrups and crossing bars).\n"
        "2. For each adjacent pair of vertical rebars (left to right), compute the "
        "CENTER-TO-CENTER spacing in millimetres:\n"
        "   - Estimate the pixel distance between the centres of the two white bands in Image 1.\n"
        "   - Multiply by the scale above to get mm.\n"
        "   - Do NOT assume uniform spacing — measure each gap independently.\n"
        "   - If one gap is visibly wider than its neighbours, report a larger number.\n"
        "3. Estimate the typical rebar diameter in mm from the width of one white band in Image 1.\n\n"
        "Return ONLY a JSON object — no markdown, no explanation:\n"
        "{\n"
        '  "rebar_count": <integer>,\n'
        '  "spacings_mm": [<s1>, <s2>, ...],  // N-1 values, left-to-right, floats\n'
        '  "diameter_mm": <float>,             // typical bar diameter in mm\n'
        '  "confidence": "low" | "medium" | "high",\n'
        '  "notes": "<one sentence describing what you see>"\n'
        "}"
    )


# ---------------------------------------------------------------------------
# Per-image measurement via Gemini
# ---------------------------------------------------------------------------

def algo_measure(image_id: str,
                 seg_pil: Image.Image,
                 depth_labels: Dict[str, float],
                 cfg: RulerConfig) -> tuple:
    """
    Compute rebar measurements from the segmentation mask using the
    column-projection algorithm in measure.py.  Returns
    (MeasurementResult, centroids: List[float]) where centroids are the
    true x-pixel positions of each detected rebar.
    """
    seg_binary = (np.array(seg_pil.convert("L")) > 127).astype(np.uint8)
    result = measure_image(seg_binary, image_id, depth_labels, cfg)
    instances = extract_instances(seg_binary)
    centroids = [inst.centroid_px[0] for inst in instances]
    return result, centroids


def gt_label_measure(image_id: str,
                     seg_pil: Image.Image,
                     rebar_labels_path: Path,
                     depth_labels: Dict[str, float],
                     cfg: RulerConfig) -> tuple:
    """
    Compute rebar measurements using hand-labeled x-coordinates in
    rebar_labels/{id}.txt (one integer pixel per line, left to right).

    Spacings are derived directly from the GT labels, so they are not
    subject to column-projection peak detection errors.  Diameter is
    estimated from the segmentation mask FWHM at each GT x position.

    Returns (MeasurementResult, centroids: List[float]).
    """
    with open(rebar_labels_path) as f:
        gt_xs = sorted(int(l.strip()) for l in f if l.strip())

    _nan = float("nan")
    if image_id not in depth_labels:
        return MeasurementResult(
            image_id=image_id, distance_m=_nan, mpp=_nan,
            rebar_count=len(gt_xs), spacings_mm=[], diameters_mm=[],
            mean_spacing_mm=None, std_spacing_mm=None, mean_diameter_mm=None,
            compliance={"status": "N/A", "reason": "no distance label"},
        ), list(map(float, gt_xs))

    distance_m = depth_labels[image_id]
    mpp = cfg.mm_per_pixel(distance_m)

    # Spacings from GT labels (pixel distance * mm/px)
    spacings_px = [gt_xs[i + 1] - gt_xs[i] for i in range(len(gt_xs) - 1)]
    spacings_mm = [round(s * mpp, 1) for s in spacings_px]

    # Diameter cannot be reliably estimated from these segmentation masks:
    # annotators label broad margins around rebars, and horizontal stirrups
    # connect adjacent vertical rebars, inflating any width measurement.
    # Accurate diameter requires caliper measurement; set to None here.
    diameters_mm: List[float] = []

    mean_spacing_mm  = round(float(np.mean(spacings_mm)),  1) if spacings_mm  else None
    std_spacing_mm   = round(float(np.std(spacings_mm)),   1) if spacings_mm  else None
    mean_diameter_mm = None

    result = MeasurementResult(
        image_id=image_id,
        distance_m=distance_m,
        mpp=mpp,
        rebar_count=len(gt_xs),
        spacings_mm=spacings_mm,
        diameters_mm=diameters_mm,
        mean_spacing_mm=mean_spacing_mm,
        std_spacing_mm=std_spacing_mm,
        mean_diameter_mm=mean_diameter_mm,
        compliance=check_compliance(spacings_mm, cfg),
    )
    return result, list(map(float, gt_xs))


def gemini_measure(
    image_pil: Image.Image,
    image_id: str,
    depth_labels: Dict[str, float],
    cfg: RulerConfig,
    model_name: str,
    api_key: str,
    request_delay: float = 1.5,
    base_url: str = None,
    seg_pil: Optional[Image.Image] = None,
) -> tuple:
    """
    Run the full measurement pipeline for one image using a Vision LLM.

    If seg_pil is provided, the segmentation mask is overlaid on the image
    (rebar regions tinted green) before sending to the model, which gives
    cleaner boundary cues than the raw photo.

    Returns (MeasurementResult, centroids: List[float]).
    """
    if image_id not in depth_labels:
        return MeasurementResult(
            image_id=image_id, distance_m=float("nan"), mpp=float("nan"),
            rebar_count=0, spacings_mm=[], diameters_mm=[],
            mean_spacing_mm=None, std_spacing_mm=None, mean_diameter_mm=None,
            compliance={"status": "N/A", "reason": "no distance label"},
            warning=f"image_id '{image_id}' not found in depth_labels",
        ), []

    distance_m = depth_labels[image_id]
    mpp = cfg.mm_per_pixel(distance_m)

    # Build the image to send: seg mask first, original as reference
    use_seg = seg_pil is not None
    if use_seg:
        # Convert seg to RGB so JPEG encoding works
        seg_rgb = seg_pil.convert("RGB")
        send_img = seg_rgb
        extra_images = [(image_pil, "This is the original photo for visual reference.")]
    else:
        send_img = image_pil
        extra_images = None
    img_w, img_h = seg_pil.size if use_seg else image_pil.size
    prompt = _build_prompt(mpp, distance_m, img_w, img_h, use_seg=use_seg)

    # --- API call with retry ---
    data: Optional[dict] = None
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            raw = _call_vlm(send_img, prompt, model_name, api_key, base_url,
                            extra_images=extra_images)
            if not raw or not raw.strip():
                raise ValueError("empty response from model")
            print(f"    [debug] raw response: {raw[:300]}")
            data = _parse_json(raw)
            break
        except Exception as exc:
            last_error = exc
            print(f"    [debug] attempt {attempt+1} failed: {exc}")
            if attempt < 2:
                time.sleep(request_delay * (2 ** attempt))

    if data is None:
        return MeasurementResult(
            image_id=image_id, distance_m=distance_m, mpp=round(mpp, 4),
            rebar_count=0, spacings_mm=[], diameters_mm=[],
            mean_spacing_mm=None, std_spacing_mm=None, mean_diameter_mm=None,
            compliance={"status": "N/A", "reason": "gemini error"},
            warning=f"VLM API/parse error after 3 attempts: {last_error}",
        ), []

    n = int(data.get("rebar_count", 0))
    raw_spacings = data.get("spacings_mm", [])
    spacings_mm = [float(s) for s in raw_spacings if s is not None and float(s) > 0]
    diameter_mm = data.get("diameter_mm", None)
    try:
        diameter_mm = float(diameter_mm) if diameter_mm is not None else None
    except (TypeError, ValueError):
        diameter_mm = None

    # Reconstruct synthetic centroids for visualization only (not used for measurement)
    centroids: List[float] = []
    if spacings_mm:
        cx = img_w * 0.05  # start near left edge
        centroids.append(cx)
        for sp in spacings_mm:
            cx += sp / mpp
            centroids.append(cx)

    if n < cfg.min_rebar_count or len(spacings_mm) == 0:
        return MeasurementResult(
            image_id=image_id, distance_m=distance_m, mpp=round(mpp, 4),
            rebar_count=n, spacings_mm=[], diameters_mm=[],
            mean_spacing_mm=None, std_spacing_mm=None, mean_diameter_mm=None,
            compliance={"status": "N/A", "reason": f"only {n} rebar detected"},
            warning=f"fewer than {cfg.min_rebar_count} rebar detected",
        ), centroids

    mean_sp = float(np.mean(spacings_mm))
    std_sp = float(np.std(spacings_mm))
    compliance = check_compliance(spacings_mm, cfg)
    diameters_mm = [round(diameter_mm, 1)] if diameter_mm is not None else []

    return MeasurementResult(
        image_id=image_id,
        distance_m=distance_m,
        mpp=round(mpp, 4),
        rebar_count=n,
        spacings_mm=[round(s, 1) for s in spacings_mm],
        diameters_mm=diameters_mm,
        mean_spacing_mm=round(mean_sp, 1),
        std_spacing_mm=round(std_sp, 1),
        mean_diameter_mm=round(diameter_mm, 1) if diameter_mm is not None else None,
        compliance=compliance,
    ), centroids


# ---------------------------------------------------------------------------
# Report text (same structure as report.py _template_report_en)
# ---------------------------------------------------------------------------

def _render_report(r: MeasurementResult, model_name: str,
                   vlm_notes: str = "",
                   attr_score: Optional[float] = None,
                   unattributed_claims: Optional[list] = None) -> str:
    today = date.today().strftime("%Y-%m-%d")
    status = r.compliance.get("status", "N/A")

    lines = [
        "=" * 60,
        "Rebar Installation Quality Inspection Report (GT)",
        f"Date: {today}",
        f"Image ID: {r.image_id}",
        f"Generated by: {model_name} (algorithm + VLM)",
        "=" * 60,
        "",
        "[Inspection Overview]",
    ]
    if r.warning:
        lines += [f"WARNING: {r.warning}", ""]
        lines.append("=" * 60)
        return "\n".join(lines)

    lines.append(
        f"Shooting distance {r.distance_m} m, scale {r.mpp} mm/px, "
        f"{r.rebar_count} vertical rebar instances detected."
    )
    lines += ["", "[Measurements]"]

    if r.mean_spacing_mm is not None:
        lines += [
            f"Mean spacing: {r.mean_spacing_mm} mm (std: {r.std_spacing_mm} mm)",
            f"Individual spacings (mm): {r.spacings_mm}",
        ]
    if r.mean_diameter_mm is not None:
        lines.append(f"Estimated diameter: {r.mean_diameter_mm} mm")
    else:
        lines.append("Note: diameter not estimated from visual inspection.")

    lines += ["", "[Compliance Assessment — GB 50204-2015]"]
    if status == "PASS":
        lines.append(
            "All spacings within stirrup tolerance ±10 mm. Result: PASS."
        )
    elif status == "PASS(main)":
        lines.append(
            "Some spacings exceed stirrup tolerance (±10 mm) but are within "
            "main bar tolerance (±20 mm). Result: PASS(main)."
        )
    elif status == "FAIL":
        fail_items = [
            s for s in r.compliance.get("per_spacing", [])
            if not s["within_main_tol"]
        ]
        lines.append(
            "The following spacings exceed main bar tolerance ±20 mm. Result: FAIL."
        )
        for s in fail_items:
            lines.append(
                f"  spacing[{s['index']}]: {s['spacing_mm']} mm, "
                f"deviation {s['deviation_from_mean_mm']:+.1f} mm"
            )
    else:
        lines.append(
            f"Cannot assess: {r.compliance.get('reason', 'insufficient data')}."
        )

    lines += ["", "[Per-Spacing Detail]"]
    for s in r.compliance.get("per_spacing", []):
        flag = "OK" if s["within_stirrup_tol"] else (
            "OK(main)" if s["within_main_tol"] else "FAIL")
        lines.append(
            f"  spacing[{s['index']}]: {s['spacing_mm']} mm  "
            f"dev={s['deviation_from_mean_mm']:+.1f} mm  [{flag}]"
        )

    lines += ["", "[Recommendations]"]
    if status == "PASS":
        lines.append(
            "Rebar installation meets GB 50204-2015 requirements. "
            "Proceed to next construction stage."
        )
    elif status == "PASS(main)":
        lines.append(
            "Verify rebar type for spacings exceeding stirrup tolerance "
            "before final acceptance."
        )
    elif status == "FAIL":
        lines.append(
            "Non-compliant spacings must be corrected and re-inspected "
            "before concrete placement."
        )
    else:
        lines.append("Insufficient data. Retake image.")

    # Risk level and trend (computed without VLM)
    risk  = _compute_risk_level(r.spacings_mm or [])
    trend = _trend_analysis(r.spacings_mm or [])
    if risk.get("level") not in (None, "UNKNOWN"):
        lines += ["", "[Risk Assessment]",
                  f"Risk level: {risk['level']}  "
                  f"(max deviation: {risk['max_dev_mm']} mm, "
                  f"non-compliant: {risk['nc_count']}/{risk['n_spacings']} spacings, "
                  f"ratio: {risk['nc_ratio']:.1%})"]
    if trend.get("trend") not in (None, "INSUFFICIENT_DATA"):
        trend_desc = {
            "UNIFORM":    "No systematic spacing trend detected.",
            "INCREASING": "WARNING: spacings widen left-to-right "
                          "(possible formwork drift or uneven placement).",
            "DECREASING": "WARNING: spacings narrow left-to-right "
                          "(possible formwork drift or uneven placement).",
        }.get(trend["trend"], trend["trend"])
        lines += ["", "[Spacing Trend Analysis]",
                  f"Trend: {trend['trend']}  "
                  f"slope={trend['slope_mm_per_gap']:+.1f} mm/gap  "
                  f"tau={trend['tau']}",
                  trend_desc]

    if vlm_notes:
        lines += ["", "[Inspector Notes]", vlm_notes]

    if attr_score is not None:
        lines += ["", "[Evidence Attribution]",
                  f"Attr(r) = {attr_score:.4f}  "
                  f"({int(round(attr_score * 100))}% of claims traceable to S ∪ G)"]
        if unattributed_claims:
            lines.append("Unattributed claims (flagged for human review):")
            for c in unattributed_claims:
                lines.append(f"  ! {c}")
        else:
            lines.append("All claims attributed to structured assessment or KB.")

    lines += ["", "=" * 60]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def _save_viz_with_centroids(image_pil: Image.Image, centroids: list,
                              result: MeasurementResult, out_path: Path,
                              seg_pil: Optional[Image.Image] = None) -> None:
    """Draw algorithm measurements on the original image.
    If seg_pil is provided, the segmentation mask edges are overlaid in green."""
    img = image_pil.copy().convert("RGB")
    w, h = img.size

    # Overlay segmentation mask edges in green
    if seg_pil is not None:
        seg_arr = np.array(seg_pil.convert("L").resize((w, h), Image.NEAREST)) > 127
        img_arr = np.array(img)
        # Tint rebar foreground pixels with a subtle green overlay
        img_arr[seg_arr, 0] = (img_arr[seg_arr, 0] * 0.5).astype(np.uint8)
        img_arr[seg_arr, 1] = np.clip(img_arr[seg_arr, 1].astype(int) * 0.5 + 80, 0, 255).astype(np.uint8)
        img_arr[seg_arr, 2] = (img_arr[seg_arr, 2] * 0.5).astype(np.uint8)
        img = Image.fromarray(img_arr)

    draw = ImageDraw.Draw(img)

    # Black header bar
    status = result.compliance.get("status", "N/A")
    header = (f"ID:{result.image_id}  rebars:{result.rebar_count}  "
              f"mean:{result.mean_spacing_mm}mm  std:{result.std_spacing_mm}mm  {status}")
    draw.rectangle([0, 0, w, 22], fill=(0, 0, 0))
    draw.text((4, 4), header, fill=(255, 255, 255))

    if result.warning:
        draw.text((4, 26), f"WARN: {result.warning}", fill=(255, 80, 80))
        img.save(out_path)
        return

    # Draw vertical red lines at each detected centroid
    for cx in centroids:
        cx = int(cx)
        draw.line([(cx, 24), (cx, h - 1)], fill=(255, 50, 50), width=2)

    # Label spacings between adjacent centroids
    for i in range(len(centroids) - 1):
        cx1, cx2 = int(centroids[i]), int(centroids[i + 1])
        mid_x = (cx1 + cx2) // 2
        sp = result.spacings_mm[i] if i < len(result.spacings_mm) else 0
        draw.text((mid_x - 16, h - 22), f"{sp:.0f}", fill=(80, 255, 80))

    img.save(out_path)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------

def _is_anomaly(row: dict) -> tuple:
    """
    Return (is_anomaly: bool, reason: str).
    Anomaly criteria:
      1. Has a warning (detection failed)
      2. std_spacing_mm == 0.0  (perfectly uniform -> VLM likely guessed)
      3. std_spacing_mm < 0.5 and rebar_count >= 4  (suspiciously uniform)
      4. mean_spacing_mm < 10 or > 500  (physically implausible)
    """
    if row.get("warning"):
        return True, f"detection warning: {row['warning']}"

    try:
        std = float(row["std_spacing_mm"])
        mean = float(row["mean_spacing_mm"])
        count = int(row["rebar_count"])
    except (TypeError, ValueError):
        return True, "missing measurement data"

    if std == 0.0:
        return True, "std=0.0 (perfectly uniform, VLM likely assumed equal spacing)"
    if std < 0.5 and count >= 4:
        return True, f"std={std:.2f}mm suspiciously uniform for {count} rebars"
    if mean < 10:
        return True, f"mean_spacing={mean:.1f}mm too small (< 10mm)"
    if mean > 500:
        return True, f"mean_spacing={mean:.1f}mm too large (> 500mm)"

    return False, ""


def _img_to_data_url(img_path: Path) -> str:
    import base64
    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    suffix = img_path.suffix.lower().lstrip(".")
    mime = "jpeg" if suffix in ("jpg", "jpeg") else "png"
    return f"data:image/{mime};base64,{b64}"


def _build_anomaly_html(anomalies: list, out_dir: Path, img_dir_map: dict) -> Path:
    """
    Generate anomaly_review.html with side-by-side:
      left: original image
      right: viz image (if exists)
      below: spacing data + anomaly reason
    anomalies: list of (row_dict, reason_str, viz_path_or_None)
    img_dir_map: {subset: Path(images dir)}
    """
    cards = []
    for row, reason, viz_path in anomalies:
        sub = row["subset"]
        sid = row["image_id"]

        orig_path = img_dir_map.get(sub, Path(".")) / f"{sid}.png"
        orig_url = _img_to_data_url(orig_path) if orig_path.exists() else ""
        viz_url  = _img_to_data_url(viz_path)  if viz_path and viz_path.exists() else ""

        spacings_file = out_dir / f"{sub}_{sid}.txt"
        report_snippet = spacings_file.read_text(encoding="utf-8") if spacings_file.exists() else ""
        report_html = report_snippet.replace("&", "&amp;").replace("<", "&lt;").replace("\n", "<br>")

        orig_tag = f'<img src="{orig_url}" style="max-width:320px;border:2px solid #888">' if orig_url else "<em>image not found</em>"
        viz_tag  = f'<img src="{viz_url}"  style="max-width:320px;border:2px solid #f80">' if viz_url  else "<em>no viz (run with --visualize)</em>"

        cards.append(f"""
<div class="card">
  <h3>{sub} / {sid}</h3>
  <p class="reason">&#9888; {reason}</p>
  <div class="imgs">{orig_tag}{viz_tag}</div>
  <pre class="report">{report_html}</pre>
</div>""")

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>GT Anomaly Review</title>
<style>
  body {{ font-family: monospace; background:#1a1a1a; color:#ddd; padding:16px; }}
  h1   {{ color:#f80; }}
  .card {{ border:1px solid #444; border-radius:6px; padding:12px; margin:16px 0; background:#222; }}
  .card h3 {{ margin:0 0 6px; color:#6cf; }}
  .reason  {{ color:#f66; font-weight:bold; margin:4px 0 10px; }}
  .imgs    {{ display:flex; gap:12px; flex-wrap:wrap; margin-bottom:10px; }}
  .report  {{ background:#111; padding:8px; border-radius:4px; font-size:12px;
              white-space:pre-wrap; max-height:260px; overflow-y:auto; }}
</style>
</head>
<body>
<h1>GT Anomaly Review &mdash; {len(anomalies)} flagged image(s)</h1>
{"".join(cards) if cards else "<p>No anomalies detected.</p>"}
</body>
</html>"""

    out_path = out_dir / "anomaly_review.html"
    out_path.write_text(html, encoding="utf-8")
    return out_path


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def _flush_csv(rows: list, csv_path: Path) -> None:
    if not rows:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _load_requested_ids(ids_file: Optional[str]) -> Dict[str, set]:
    if not ids_file:
        return {}

    requested: Dict[str, set] = {}
    with open(ids_file, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = line.replace("\\", "/")
            name = Path(line).name
            m = re.match(r"^(site-[12])_(.+?)(?:\.txt)?$", name)
            if not m:
                raise ValueError(f"Invalid ids_file entry: {raw.rstrip()}")
            sub, sid = m.groups()
            requested.setdefault(sub, set()).add(sid)
    return requested


def run(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    viz_dir = out_dir / "viz"
    if args.visualize:
        viz_dir.mkdir(exist_ok=True)

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY", "") \
              or os.environ.get("GEMINI_API_KEY", "") \
              or os.environ.get("GOOGLE_API_KEY", "") \
              or os.environ.get("DASHSCOPE_API_KEY", "")
    if not api_key and not getattr(args, "no_vlm", False):
        print("[ERROR] API key required. Set --api_key or OPENAI_API_KEY / GEMINI_API_KEY env var.")
        sys.exit(1)

    cfg = RulerConfig(
        hfov_deg=args.hfov,
        image_width_px=args.img_w,
        image_height_px=args.img_h,
    )

    csv_rows: List[dict] = []
    viz_path_map: dict = {}   # (subset, image_id) -> Path or None
    img_dir_map:  dict = {}   # subset -> Path(images dir)

    # Resume: load rows already written in a previous partial run
    already_done: set = set()
    csv_path_existing = Path(args.out_dir) / "results.csv"
    if getattr(args, "resume", False) and csv_path_existing.exists():
        with open(csv_path_existing, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                csv_rows.append(row)
                already_done.add((row["subset"], row["image_id"]))
        print(f"[resume] loaded {len(csv_rows)} existing rows from {csv_path_existing}")

    requested_ids = _load_requested_ids(getattr(args, "ids_file", None))

    for sub in args.subsets:
        if requested_ids and sub not in requested_ids:
            continue
        sub_path = Path(args.data_root) / sub
        if not sub_path.exists():
            print(f"[SKIP] {sub_path} not found")
            continue

        labels_path = sub_path / "depth_labels.txt"
        if not labels_path.exists():
            print(f"[SKIP] {labels_path} not found")
            continue
        depth_labels = load_depth_labels(str(labels_path))

        split_file = sub_path / "gt_sets" / f"{args.split}.txt"
        if not split_file.exists():
            print(f"[SKIP] {split_file} not found")
            continue
        with open(split_file) as f:
            ids = [line.strip() for line in f if line.strip()]
        img_dir = sub_path / "images"
        img_dir_map[sub] = img_dir
        ids = [sid for sid in ids if (img_dir / f"{sid}.png").exists()]
        if requested_ids:
            ids = [sid for sid in ids if sid in requested_ids[sub]]
        if args.n > 0:
            ids = ids[: args.n]

        print(f"{sub}: {len(ids)} {args.split} images -> {args.model}")

        seg_dir        = sub_path / "segmentation"
        rebar_label_dir = sub_path / "rebar_labels"

        for i, sid in enumerate(ids):
            if (sub, sid) in already_done:
                print(f"  [{sub}/{sid}] skipping (resume)")
                vp = viz_dir / f"{sub}_{sid}.png" if args.visualize else None
                viz_path_map[(sub, sid)] = vp if (vp and vp.exists()) else None
                continue

            img_path = img_dir / f"{sid}.png"
            image_pil = Image.open(img_path).convert("RGB")

            # Load segmentation mask
            seg_pil = None
            seg_path = seg_dir / f"{sid}.png"
            if seg_path.exists():
                seg_pil = Image.open(seg_path).convert("L")

            # --- Step 1: Measurement (GT labels > seg algo > VLM fallback) ---
            rebar_label_path = rebar_label_dir / f"{sid}.txt"
            if rebar_label_path.exists() and seg_pil is not None:
                # Use hand-labeled rebar x-coordinates for precise spacing;
                # segmentation mask used only for diameter estimation.
                result, centroids = gt_label_measure(
                    sid, seg_pil, rebar_label_path, depth_labels, cfg)
            elif seg_pil is not None:
                result, centroids = algo_measure(sid, seg_pil, depth_labels, cfg)
            else:
                # Fallback: VLM measurement if no seg mask
                result, centroids = gemini_measure(
                    image_pil, sid, depth_labels, cfg,
                    model_name=args.model, api_key=api_key,
                    request_delay=args.delay, base_url=args.base_url or None,
                )

            # --- Step 2+3: Knowledge-grounded VLM report with CoT + verification ---
            vlm_notes = ""
            attr_score: Optional[float] = None
            unattributed_claims: list = []
            S: Optional[dict] = None
            if not args.no_vlm and not result.warning:
                try:
                    S = graph_traverse(result)
                    vlm_notes = _generate_report_with_verification(
                        image_pil, S, result,
                        args.model, api_key, args.base_url or None,
                    )
                    # --- Step 4: Evidence attribution  Attr(r̂) ---
                    if vlm_notes and not vlm_notes.startswith("[VLM"):
                        attr_score, unattributed_claims = _attribute_claims(
                            vlm_notes, S)
                    if i < len(ids) - 1:
                        time.sleep(args.delay)
                except Exception as e:
                    vlm_notes = f"[VLM report failed: {e}]"

            report_text = _render_report(result, args.model, vlm_notes=vlm_notes,
                                         attr_score=attr_score,
                                         unattributed_claims=unattributed_claims)
            (out_dir / f"{sub}_{sid}.txt").write_text(report_text, encoding="utf-8")

            vp = None
            if args.visualize:
                vp = viz_dir / f"{sub}_{sid}.png"
                _save_viz_with_centroids(image_pil, centroids, result, vp,
                                         seg_pil=seg_pil)
            viz_path_map[(sub, sid)] = vp

            status_str = result.compliance.get("status", "N/A")
            risk  = _compute_risk_level(result.spacings_mm or [])
            trend = _trend_analysis(result.spacings_mm or [])
            attr_str = f"{attr_score:.4f}" if attr_score is not None else ""
            print(f"  [{sub}/{sid}] rebar={result.rebar_count}  "
                  f"mean_spacing={result.mean_spacing_mm}mm  "
                  f"GB50204={status_str}  risk={risk.get('level','?')}  "
                  f"trend={trend.get('trend','?')}"
                  + (f"  Attr={attr_str}" if attr_str else "")
                  + (f"  WARN: {result.warning}" if result.warning else ""))

            csv_rows.append({
                "subset":             sub,
                "image_id":           sid,
                "distance_m":         result.distance_m,
                "mpp":                result.mpp,
                "rebar_count":        result.rebar_count,
                "mean_spacing_mm":    result.mean_spacing_mm,
                "std_spacing_mm":     result.std_spacing_mm,
                "mean_diameter_mm":   result.mean_diameter_mm,
                "compliance":         status_str,
                "risk_level":         risk.get("level", ""),
                "max_dev_mm":         risk.get("max_dev_mm", ""),
                "nc_ratio":           risk.get("nc_ratio", ""),
                "trend":              trend.get("trend", ""),
                "slope_mm_per_gap":   trend.get("slope_mm_per_gap", ""),
                "tau":                trend.get("tau", ""),
                "attr_score":         attr_score if attr_score is not None else "",
                "unattributed_count": len(unattributed_claims),
                "warning":            result.warning or "",
            })
            # Flush CSV after each image so an interrupted run can be --resume'd
            _flush_csv(csv_rows, out_dir / "results.csv")

    # --- CSV ---
    csv_path = out_dir / "results.csv"
    _flush_csv(csv_rows, csv_path)
    if csv_rows:
        print(f"\nCSV saved -> {csv_path}  ({len(csv_rows)} rows)")

    # --- Summary ---
    valid = [r for r in csv_rows if not r["warning"]]
    pass_n = sum(1 for r in valid if r["compliance"] == "PASS")
    pass_main_n = sum(1 for r in valid if r["compliance"] == "PASS(main)")
    fail_n = sum(1 for r in valid if r["compliance"] == "FAIL")
    na_n = len(csv_rows) - len(valid)

    summary_lines = [
        "=" * 60,
        "Rebar Inspection Ground-Truth Batch Summary",
        f"Model: {args.model}",
        f"Total images: {len(csv_rows)}",
        f"Valid measurements: {len(valid)}/{len(csv_rows)}",
        f"PASS: {pass_n}  |  PASS(main): {pass_main_n}  |  FAIL: {fail_n}  |  N/A: {na_n}",
    ]
    if valid:
        sp = [float(r["mean_spacing_mm"]) for r in valid if r.get("mean_spacing_mm") not in ("", "None", None)]
        if sp:
            summary_lines.append(
                f"Batch mean spacing: {np.mean(sp):.1f} mm  (std {np.std(sp):.1f} mm)"
            )
        if pass_n + pass_main_n + fail_n > 0:
            total_judged = pass_n + pass_main_n + fail_n
            summary_lines.append(
                f"GB 50204 PASS rate (of judged): "
                f"{pass_n}/{total_judged} ({100*pass_n/total_judged:.1f}%)"
            )
    summary_lines += ["", "--- Per-image reports below ---", ""]

    for row in csv_rows:
        report_file = out_dir / f"{row['subset']}_{row['image_id']}.txt"
        if report_file.exists():
            summary_lines.append(report_file.read_text(encoding="utf-8"))
            summary_lines.append("")

    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Summary -> {summary_path}")

    # --- Anomaly review HTML ---
    anomalies = []
    for row in csv_rows:
        flag, reason = _is_anomaly(row)
        if flag:
            vp = viz_path_map.get((row["subset"], row["image_id"]))
            anomalies.append((row, reason, vp))

    html_path = _build_anomaly_html(anomalies, out_dir, img_dir_map)
    print(f"Anomaly review ({len(anomalies)} flagged) -> {html_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Ground-truth report generation via Gemini Vision"
    )
    p.add_argument("--data_root", default="data")
    p.add_argument("--subsets",   nargs="+", default=["site-1", "site-2"])
    p.add_argument("--split",     default="val", choices=["train", "val", "trainval"])
    p.add_argument("--out_dir",   default="gt_reports")
    p.add_argument("--model",     default="gpt-4o",
                   help="Model name: gpt-4o, gpt-4o-mini, gemini-2.0-flash, etc.")
    p.add_argument("--api_key",   default=None,
                   help="API key (or set OPENAI_API_KEY / GEMINI_API_KEY env var)")
    p.add_argument("--base_url",  default=None,
                   help="Custom API base URL for OpenAI-compatible proxies (e.g. https://api.xxx.com/v1)")
    p.add_argument("--n",         type=int, default=0,
                   help="Images per subset to process (0 = all)")
    p.add_argument("--ids_file",  default=None,
                   help="Optional file listing subset/id entries to process, e.g. site-1_9.txt or site-2_164")
    p.add_argument("--delay",     type=float, default=1.5,
                   help="Seconds between API requests (default 1.5)")
    p.add_argument("--hfov",      type=float, default=87.0)
    p.add_argument("--img_w",     type=int, default=1280)
    p.add_argument("--img_h",     type=int, default=720)
    p.add_argument("--visualize", action="store_true",
                   help="Save annotated images to {out_dir}/viz/ for quality check")
    p.add_argument("--no_vlm", action="store_true",
                   help="Skip VLM report text; use algorithm measurements only (faster, no API cost)")
    p.add_argument("--resume", action="store_true",
                   help="Skip images already in results.csv; append new rows to existing CSV")
    run(p.parse_args())
