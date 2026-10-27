"""
generate.py -- generate inspection reports via Qwen-VL (OpenAI-compatible API).

Settings:
  measure_only      legacy diagnostic: measurement values only (no RGB)
  measure_rule      legacy diagnostic: measurement + GB 50204-2015 rule (no RGB)
  seg_only          legacy diagnostic: seg image only (no RGB)
  seg_measure       legacy diagnostic: seg image + measurement values (no RGB)
  seg_measure_rule  legacy diagnostic: seg image + measurement + rule (no RGB)
  seg_measure_rule_qaep  legacy diagnostic: seg image + measurement + rule + QAEP (no RGB)
  rgb_only          RGB image only
  rgb_rule          RGB image + GB 50204-2015 rule, no structured measurement
  rgb_measure       RGB + measurement values
  rgb_measure_rule  RGB + measurement values + GB 50204-2015 rule
  rgb_seg           RGB + seg image, no measurement values
  rgb_seg_rule      RGB + seg image + GB 50204-2015 rule, no measurement values
  rgb_seg_measure   RGB + seg image + measurement values
  rgb_seg_measure_rule       RGB + seg image + measurement + GB 50204-2015 rule
  rgb_seg_measure_rule_qaep  RGB + seg image + measurement + rule + QAEP evidence plan

Usage:
  python src_report/generate.py \\
      --input_jsonl results_clean/report_generation/qaep/cleanref_full_qaep_applied_by_subset.jsonl \\
      --setting rgb_seg_measure_rule_qaep \\
      --out_jsonl results_clean/report_generation/generations/downstream_rgb_seg_measure_rule_qaep.jsonl \\
      --model qwen-vl-plus \\
      --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 \\
      --limit 5
"""

import argparse
import base64
import hashlib
import json
import math
import os
import time
from pathlib import Path

import requests
from design_evidence import is_v2, risk_level, format_measurement, rule_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_SETTINGS = (
    "measure_only",
    "measure_rule",
    "seg_only",
    "seg_measure",
    "seg_measure_rule",
    "seg_measure_rule_qaep",
    "rgb_only",
    "rgb_rule",
    "rgb_measure",
    "rgb_measure_rule",
    "rgb_seg",
    "rgb_seg_rule",
    "rgb_seg_measure",
    "rgb_seg_measure_rule",
    "rgb_seg_measure_rule_qaep",
)

SYSTEM_PROMPT = (
    "You are a structural inspection report writer specialising in rebar "
    "(reinforcing bar) installation quality control. "
    "When given inspection data or a segmentation image, write a single "
    "Professional Narrative paragraph in English. "
    "Use a style close to a human inspection reference report: start from the "
    "inspection finding, state the detected rebar count and spacing statistics "
    "when provided, cite the GB 50204-2015 tolerance, identify abnormal "
    "spacing positions when present, state the risk level, mention spacing "
    "trend when provided, and close with a concise corrective or readiness "
    "statement. Requirements: one paragraph, exactly 4 sentences when "
    "measurement evidence is available; use only the numerical values provided "
    "and do not fabricate measurements; do not discuss diameter unless the "
    "user explicitly asks for a diameter assessment; do not include reasoning "
    "steps, bullet points, headers, or any preamble text; output only the "
    "paragraph text."
)


# ---------------------------------------------------------------------------
# Risk level
# ---------------------------------------------------------------------------

def compute_risk_level(m: dict) -> str:
    """
    Derive a graded risk level from measurement compliance data.

    This mirrors the risk rule used by src/gt_report.py:
      nc_ratio == 0 and max_dev <= 5   -> SAFE
      nc_ratio == 0 and max_dev <= 10  -> ACCEPTABLE
      nc_ratio <= 0.25 and max_dev <= 40 -> MARGINAL
      nc_ratio <= 0.5 or max_dev <= 60 -> WARNING
      otherwise -> CRITICAL
    """
    if is_v2(m):
        return risk_level(m)
    compliance = m.get("compliance", {})
    status = compliance.get("status", "N/A")

    per = compliance.get("per_spacing", [])
    if not per:
        if status == "PASS":
            return "ACCEPTABLE"
        if status == "PASS(main)":
            return "MARGINAL"
        if status == "FAIL":
            return "WARNING"
        return "UNKNOWN"

    max_dev = max(abs(p["deviation_from_mean_mm"]) for p in per)
    fails = [p for p in per if abs(p["deviation_from_mean_mm"]) > 20.0]
    fail_ratio = len(fails) / len(per)

    if fail_ratio == 0 and max_dev <= 5:
        return "SAFE"
    if fail_ratio == 0 and max_dev <= 10:
        return "ACCEPTABLE"
    if fail_ratio <= 0.25 and max_dev <= 40:
        return "MARGINAL"
    if fail_ratio <= 0.5 or max_dev <= 60:
        return "WARNING"
    return "CRITICAL"


# ---------------------------------------------------------------------------
# Measurement text formatter
# ---------------------------------------------------------------------------

def _format_measurement(m: dict, include_rule: bool = False) -> str:
    if is_v2(m):
        return format_measurement(m, include_rule)
    lines = []

    if m.get("warning"):
        lines.append(f"WARNING: {m['warning']}")
        lines.append(
            "Measurement could not be completed reliably; "
            "insufficient rebar instances were detected."
        )
        if include_rule:
            lines.append(
                "Standard: GB 50204-2015 requires spacing deviation "
                "within +/-20 mm (main bars) or +/-10 mm (stirrups)."
            )
        return "\n".join(lines)

    n         = m.get("rebar_count", 0)
    mean_s    = m.get("mean_spacing_mm")
    std_s     = m.get("std_spacing_mm")
    mean_d    = m.get("mean_diameter_mm")
    spacings  = m.get("spacings_mm", [])
    mpp       = m.get("mpp")
    dist      = m.get("distance_m")
    compliance = m.get("compliance", {})
    status    = compliance.get("status", "N/A")

    if dist is not None and not math.isnan(dist):
        lines.append(f"Shooting distance: {dist} m  |  mm/pixel: {mpp}")
    lines.append(f"Detected rebar count: {n}")
    if mean_s is not None:
        lines.append(f"Mean spacing: {mean_s} mm  (std: {std_s} mm)")
    if mean_d is not None:
        lines.append(f"Mean diameter: {mean_d} mm")

    if include_rule:
        lines.append(
            "Standard: GB 50204-2015 requires spacing deviation "
            "within +/-20 mm (main bars) or +/-10 mm (stirrups)."
        )

    risk = compute_risk_level(m)
    lines.append(f"Compliance result: {status}  (risk level: {risk})")

    per = compliance.get("per_spacing", [])

    if status == "FAIL" and per:
        fails = [p for p in per if not p.get("within_main_tol", True)]
        if fails:
            max_dev = max(abs(p["deviation_from_mean_mm"]) for p in fails)
            fail_str = "; ".join(
                f"spacing[{p['index']}]={p['spacing_mm']} mm "
                f"(dev={p['deviation_from_mean_mm']:+.1f} mm)"
                for p in fails
            )
            lines.append(
                f"Non-compliant spacings (exceeding +/-20 mm): {fail_str}"
            )
            lines.append(f"Maximum deviation: {max_dev:.1f} mm")

        if len(spacings) >= 2:
            slope = (spacings[-1] - spacings[0]) / max(len(spacings) - 1, 1)
            if slope > 2.0:
                trend = f"increasing ({slope:+.1f} mm/gap)"
            elif slope < -2.0:
                trend = f"decreasing ({slope:+.1f} mm/gap)"
            else:
                trend = "uniform"
            lines.append(f"Spacing trend: {trend}")

    elif status == "PASS(main)" and per:
        max_dev = max(abs(p["deviation_from_mean_mm"]) for p in per)
        lines.append(
            f"All spacings within +/-20 mm main-bar tolerance; "
            f"maximum deviation from mean: {max_dev:.1f} mm."
        )

    elif status == "PASS" and per:
        max_dev = max(abs(p["deviation_from_mean_mm"]) for p in per)
        lines.append(f"Maximum deviation from mean: {max_dev:.1f} mm.")

    if status != "FAIL" and len(spacings) >= 2:
        slope = (spacings[-1] - spacings[0]) / max(len(spacings) - 1, 1)
        if slope > 2.0:
            trend = f"increasing ({slope:+.1f} mm/gap)"
        elif slope < -2.0:
            trend = f"decreasing ({slope:+.1f} mm/gap)"
        else:
            trend = "uniform"
        lines.append(f"Spacing trend: {trend}")

    return "\n".join(lines)


def _format_qaep(qaep: dict) -> str:
    if not qaep:
        return ""

    tags = qaep.get("risk_tags", []) or []
    constraints = qaep.get("report_constraints", []) or []
    lines = ["Evidence-planning constraints for the final report:"]
    if tags:
        public_tags = [
            t for t in tags
            if t not in {"UNCERTAIN_SEGMENTATION", "LOW_IMAGE_QUALITY"}
        ]
        if public_tags:
            lines.append("Inspection tags: " + ", ".join(public_tags))
    if constraints:
        lines.append("Report constraints:")
        for c in constraints:
            if "probability" in c.lower():
                continue
            lines.append(f"- {c}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Image helper
# ---------------------------------------------------------------------------

def _img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _image_content_block(path: str) -> dict:
    b64 = _img_b64(path)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{b64}"},
    }


# ---------------------------------------------------------------------------
# Message builders
# ---------------------------------------------------------------------------

def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _build_original_messages(rec: dict, setting: str) -> list:
    m      = rec["measurement"]
    if is_v2(m):
        if setting not in VALID_SETTINGS:
            raise ValueError('Unknown setting: '+setting)
        content = []
        if 'rgb' in setting:
            content.append(_image_content_block(rec['image_path']))
        if 'seg' in setting:
            content.append(_image_content_block(rec['seg_path']))
        text = 'Write one professional inspection paragraph of six complete sentences using only the supplied evidence. '
        text += 'Retain the detailed inspection narrative, numerical findings, abnormal interval positions, risk level, and conclusion when supplied. '
        text += 'Do not claim statutory acceptance or infer precise spacings from an image alone. '
        if 'measure' in setting:
            text += '\n'+format_measurement(m, include_rule='rule' in setting)
        elif 'rule' in setting:
            text += '\n'+rule_text(m)
        if 'qaep' in setting:
            plan = rec.get('qaep', {})
            if not plan:
                raise ValueError('QAEP setting requires a predicted QAEP plan')
            text += '\n'+str(plan.get('plan_text', ''))
        content.append(_text_block(text))
        return [{'role':'system', 'content':'You write factual engineering inspection reports. Never invent missing measurements.'},
                {'role':'user', 'content':content}]

    if setting == "measure_only":
        mtext = _format_measurement(m, include_rule=False)
        user_text = (
            "Rebar installation inspection data:\n"
            f"{mtext}\n\n"
            "Write a Professional Narrative paragraph (4-6 sentences) "
            "summarising this rebar inspection."
        )
        user_content = user_text

    elif setting == "measure_rule":
        mtext = _format_measurement(m, include_rule=True)
        user_text = (
            "Rebar installation inspection data:\n"
            f"{mtext}\n\n"
            "Write a Professional Narrative paragraph using exactly four "
            "sentences. Incorporate the measurements, cite the GB 50204-2015 "
            "spacing tolerance, identify non-compliant spacing positions if "
            "present, include the maximum deviation and spacing trend when "
            "provided, and end with 'Result: PASS.' or 'Result: FAIL.'."
        )
        user_content = user_text

    elif setting == "seg_only":
        user_text = (
            "This is a binary segmentation mask of vertical rebar "
            "(white regions) in a concrete formwork photograph. "
            "Write a Professional Narrative paragraph (3-5 sentences) "
            "assessing the rebar installation quality based on what you "
            "observe in this segmentation mask."
        )
        user_content = [
            _image_content_block(rec["seg_path"]),
            _text_block(user_text),
        ]

    elif setting == "seg_measure":
        mtext = _format_measurement(m, include_rule=False)
        user_text = (
            "This is a binary segmentation mask of vertical rebar "
            "(white regions) in a concrete formwork photograph.\n\n"
            "Rebar installation inspection data:\n"
            f"{mtext}\n\n"
            "Write a Professional Narrative paragraph (4-6 sentences) "
            "summarising this inspection, incorporating the measurements above."
        )
        user_content = [
            _image_content_block(rec["seg_path"]),
            _text_block(user_text),
        ]

    elif setting == "seg_measure_rule":
        mtext = _format_measurement(m, include_rule=True)
        user_text = (
            "This is a binary segmentation mask of vertical rebar "
            "(white regions) in a concrete formwork photograph.\n\n"
            "Rebar installation inspection data:\n"
            f"{mtext}\n\n"
            "Write a Professional Narrative paragraph (4-6 sentences) "
            "summarising this inspection. Incorporate the measurements, "
            "reference the GB 50204-2015 standard, and state the compliance "
            "result using exactly the risk level provided above (do not "
            "change it). If FAIL, identify the non-compliant spacings and "
            "maximum deviation."
        )
        user_content = [
            _image_content_block(rec["seg_path"]),
            _text_block(user_text),
        ]

    elif setting == "seg_measure_rule_qaep":
        mtext = _format_measurement(m, include_rule=True)
        qtext = _format_qaep(rec.get("qaep") or {})
        risk = compute_risk_level(m)
        user_text = (
            "This is a binary segmentation mask of vertical rebar "
            "(white regions) in a concrete formwork photograph.\n\n"
            "Rebar installation inspection data:\n"
            f"{mtext}\n\n"
        )
        if qtext:
            user_text += f"{qtext}\n\n"
        user_text += (
            "Write a Professional Narrative paragraph using exactly four "
            "sentences. Use the evidence-planning constraints "
            "only to decide what facts or qualifications must be included; do "
            "not mention QAEP, model confidence, probabilities, labels, or "
            "internal planning terms in the final report. Incorporate the "
            "measurements, reference the GB 50204-2015 standard, and state the "
            "compliance result using exactly the risk level provided above (do "
            f"not change it). The only risk-level word you may use is {risk}; "
            "do not use SAFE, ACCEPTABLE, MARGINAL, WARNING, or CRITICAL as a "
            "generic adjective unless it is the provided risk level. If FAIL, "
            "identify every non-compliant spacing using the exact notation "
            "spacing[i], and include the maximum deviation. If a spacing trend "
            "is provided, include the trend using the words increasing, "
            "decreasing, or uniform. Follow this sentence plan: sentence 1 "
            "states detected rebar count, measured spacing count, mean spacing, "
            "and standard deviation; sentence 2 states the GB 50204-2015 "
            "tolerance, compliance result, non-compliant spacing positions if "
            "any, and maximum deviation; sentence 3 states the risk level and "
            "spacing trend; sentence 4 gives the corrective action or readiness "
            "statement and ends with 'Result: PASS.' or 'Result: FAIL.'"
        )
        user_content = [
            _image_content_block(rec["seg_path"]),
            _text_block(user_text),
        ]

    elif setting == "rgb_only":
        user_text = (
            "This is the original RGB image of a rebar installation inspection "
            "scene. Write a Professional Narrative paragraph (3-5 sentences) "
            "assessing the rebar installation quality based only on the visual "
            "image. Do not invent precise numerical measurements unless they are "
            "visually evident. Output only the paragraph text."
        )
        user_content = [
            _image_content_block(rec["image_path"]),
            _text_block(user_text),
        ]

    elif setting == "rgb_rule":
        user_text = (
            "This is the original RGB image of a rebar installation inspection "
            "scene. The relevant standard is GB 50204-2015, where longitudinal "
            "main-bar spacing deviation is assessed using a +/-20 mm tolerance. "
            "Write a Professional Narrative paragraph (3-5 sentences) for an "
            "inspection report based only on the image and this standard. Do "
            "not invent precise numerical measurements unless they are directly "
            "provided or visually evident. State uncertainty when the image "
            "alone is insufficient for millimeter-level spacing assessment. "
            "Output only the paragraph text."
        )
        user_content = [
            _image_content_block(rec["image_path"]),
            _text_block(user_text),
        ]

    elif setting == "rgb_measure":
        mtext = _format_measurement(m, include_rule=False)
        user_text = (
            "This is the original RGB image of a rebar installation inspection "
            "scene.\n\n"
            "Rebar installation inspection data:\n"
            f"{mtext}\n\n"
            "Write a Professional Narrative paragraph using exactly four "
            "sentences. Use the RGB image to verify the scene context and "
            "visible installation condition, and use the structured "
            "measurements as the only source of numerical spacing values. Do "
            "not invent measurements that are not provided. Output only the "
            "paragraph text."
        )
        user_content = [
            _image_content_block(rec["image_path"]),
            _text_block(user_text),
        ]

    elif setting == "rgb_measure_rule":
        mtext = _format_measurement(m, include_rule=True)
        user_text = (
            "This is the original RGB image of a rebar installation inspection "
            "scene.\n\n"
            "Rebar installation inspection data:\n"
            f"{mtext}\n\n"
            "Write a Professional Narrative paragraph using exactly four "
            "sentences. Use the RGB image to verify the scene context and "
            "visible installation condition, and use the structured "
            "measurements as the only source of numerical spacing values. "
            "Incorporate the measurements, reference the GB 50204-2015 "
            "standard, identify non-compliant spacing positions if present, "
            "and end with 'Result: PASS.' or 'Result: FAIL.'. Do not invent "
            "measurements that are not provided."
        )
        user_content = [
            _image_content_block(rec["image_path"]),
            _text_block(user_text),
        ]

    elif setting == "rgb_seg":
        user_text = (
            "You are given two visual inputs for the same rebar inspection "
            "sample. The first image is the original RGB construction-site "
            "photograph. The second image is the predicted binary segmentation "
            "mask of vertical rebars, where white regions indicate detected "
            "rebar areas. Write a Professional Narrative paragraph using "
            "3-5 sentences based only on these visual inputs. Do not invent "
            "precise numerical measurements unless they are visually evident. "
            "Output only the paragraph text."
        )
        user_content = [
            _image_content_block(rec["image_path"]),
            _image_content_block(rec["seg_path"]),
            _text_block(user_text),
        ]

    elif setting == "rgb_seg_rule":
        user_text = (
            "You are given two visual inputs for the same rebar inspection "
            "sample. The first image is the original RGB construction-site "
            "photograph. The second image is the predicted binary segmentation "
            "mask of vertical rebars, where white regions indicate detected "
            "rebar areas. The relevant standard is GB 50204-2015, where "
            "longitudinal main-bar spacing deviation is assessed using a "
            "+/-20 mm tolerance. Write a Professional Narrative paragraph "
            "using 3-5 sentences based only on the RGB image, segmentation "
            "mask, and this standard. Do not invent precise numerical "
            "measurements, spacing indices, compliance status, or risk level "
            "unless they are directly supported by the visual evidence. State "
            "uncertainty when the visual evidence is insufficient for "
            "millimetre-level spacing assessment. Output only the paragraph "
            "text."
        )
        user_content = [
            _image_content_block(rec["image_path"]),
            _image_content_block(rec["seg_path"]),
            _text_block(user_text),
        ]

    elif setting == "rgb_seg_measure":
        mtext = _format_measurement(m, include_rule=False)
        user_text = (
            "You are given two visual inputs for the same rebar inspection "
            "sample. The first image is the original RGB construction-site "
            "photograph. The second image is the predicted binary segmentation "
            "mask of vertical rebars, where white regions indicate detected "
            "rebar areas.\n\n"
            "Rebar installation inspection data:\n"
            f"{mtext}\n\n"
            "Write a Professional Narrative paragraph using exactly four "
            "sentences. Use the RGB image to verify the scene context and "
            "visible installation condition, use the segmentation mask as "
            "geometric evidence, and use the structured measurements as the "
            "only source of numerical spacing values. Do not invent "
            "measurements that are not provided."
        )
        user_content = [
            _image_content_block(rec["image_path"]),
            _image_content_block(rec["seg_path"]),
            _text_block(user_text),
        ]

    elif setting == "rgb_seg_measure_rule":
        mtext = _format_measurement(m, include_rule=True)
        user_text = (
            "You are given two visual inputs for the same rebar inspection "
            "sample. The first image is the original RGB construction-site "
            "photograph. The second image is the predicted binary segmentation "
            "mask of vertical rebars, where white regions indicate detected "
            "rebar areas.\n\n"
            "Rebar installation inspection data:\n"
            f"{mtext}\n\n"
            "Write a Professional Narrative paragraph using exactly four "
            "sentences. Use the RGB image to verify the scene context and "
            "visible installation condition, use the segmentation mask as "
            "geometric evidence, and use the structured measurements as the "
            "only source of numerical spacing values. Incorporate the "
            "measurements, reference the GB 50204-2015 standard, and state "
            "the compliance result using exactly the risk level provided "
            "above. If FAIL, identify the non-compliant spacings and maximum "
            "deviation. Do not invent measurements that are not provided."
        )
        user_content = [
            _image_content_block(rec["image_path"]),
            _image_content_block(rec["seg_path"]),
            _text_block(user_text),
        ]

    elif setting == "rgb_seg_measure_rule_qaep":
        mtext = _format_measurement(m, include_rule=True)
        qtext = _format_qaep(rec.get("qaep") or {})
        risk = compute_risk_level(m)
        user_text = (
            "You are given two visual inputs for the same rebar inspection "
            "sample. The first image is the original RGB construction-site "
            "photograph. The second image is the predicted binary segmentation "
            "mask of vertical rebars, where white regions indicate detected "
            "rebar areas.\n\n"
            "Rebar installation inspection data:\n"
            f"{mtext}\n\n"
        )
        if qtext:
            user_text += f"{qtext}\n\n"
        user_text += (
            "Write a Professional Narrative paragraph using exactly four "
            "sentences. Use the RGB image to verify the scene context and "
            "visible installation condition, use the segmentation mask as "
            "geometric evidence, and use the structured measurements as the "
            "only source of numerical spacing values. Use the evidence-planning "
            "constraints only to decide what facts or qualifications must be "
            "included; do not mention QAEP, model confidence, probabilities, "
            "labels, or internal planning terms in the final report. Incorporate "
            "the measurements, reference the GB 50204-2015 standard, and state "
            "the compliance result using exactly the risk level provided above "
            f"(do not change it). The only risk-level word you may use is "
            f"{risk}; do not use SAFE, ACCEPTABLE, MARGINAL, WARNING, or "
            "CRITICAL as a generic adjective unless it is the provided risk "
            "level. If FAIL, identify every non-compliant spacing using the "
            "exact notation spacing[i], and include the maximum deviation. If "
            "a spacing trend is provided, include the trend using the words "
            "increasing, decreasing, or uniform. Follow this sentence plan: "
            "sentence 1 states the visual inspection context, detected rebar "
            "count, measured spacing count, mean spacing, and standard "
            "deviation; sentence 2 states the GB 50204-2015 tolerance, "
            "compliance result, non-compliant spacing positions if any, and "
            "maximum deviation; sentence 3 states the risk level and spacing "
            "trend; sentence 4 gives the corrective action or readiness "
            "statement and ends with 'Result: PASS.' or 'Result: FAIL.'"
        )
        user_content = [
            _image_content_block(rec["image_path"]),
            _image_content_block(rec["seg_path"]),
            _text_block(user_text),
        ]

    else:
        raise ValueError(f"Unknown setting: {setting!r}")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return messages


# ---------------------------------------------------------------------------
# Prompt hash (sha256 over JSON-serialised messages, skipping base64 payload)
# ---------------------------------------------------------------------------

def _build_messages(rec: dict, setting: str, report_style: str = "original") -> list:
    messages = _build_original_messages(rec, setting)
    if report_style == "concise":
        messages[0]["content"] += (
            " Output only the final inspection summary as one plain paragraph. "
            "Aim for four short sentences, never more than six or 100 words. "
            "When supplied, summarize count and mean spacing, maximum design deviation, "
            "all failed zero-based interval indices, risk and final decision. "
            "List failed indices compactly; do not repeat every spacing measurement. "
            "Without measurements or design rules, state the missing evidence briefly "
            "and do not invent numbers or a compliance decision. "
            "Omit scene narration, reasoning, headings, Markdown and repeated disclaimers. "
            "Retain any necessary uncertainty or screening-only qualification briefly. "
            "Use complete sentences without ellipses."
        )
    elif report_style != "original":
        raise ValueError("Unknown report style: " + report_style)
    return messages


def _validate_concise_report(text: str) -> None:
    import re
    sentences = re.findall(r"[.!?](?:\s+|$)", text.strip())
    if "..." in text or "\u2026" in text:
        raise ValueError("Concise report contains ellipses; generate complete sentences.")
    if not 1 <= len(sentences) <= 6 or len(text.split()) > 100:
        raise ValueError("Concise report must contain 1-6 complete sentences and at most 100 words.")
    if not text.strip().endswith((".", "!", "?")):
        raise ValueError("Concise report ends with an incomplete sentence.")


def _prompt_hash(messages: list) -> str:
    def _strip_b64(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "image_url":
                return {"type": "image_url", "image_url": {"url": "<image>"}}
            return {k: _strip_b64(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_strip_b64(x) for x in obj]
        return obj

    stripped = _strip_b64(messages)
    canonical = json.dumps(stripped, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_api(
    messages: list,
    model: str,
    base_url: str,
    api_key: str,
    timeout: int = 90,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if not resp.ok:
        detail = resp.text.replace(api_key, "<redacted>")[:2000]
        raise requests.HTTPError(
            f"HTTP {resp.status_code}: {detail}", response=resp
        )
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_done(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                done.add((rec["subset"], rec["id"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate inspection reports via Qwen-VL."
    )
    p.add_argument("--input_jsonl", required=True,
                   help="Input jsonl from build_report_inputs.py")
    p.add_argument("--setting", required=True, choices=VALID_SETTINGS,
                   help="Prompt setting")
    p.add_argument("--report-style", choices=("original", "concise"), default="original",
                   help="Use concise for a final summary of at most six sentences; original preserves experiment prompts")
    p.add_argument("--out_jsonl", required=True,
                   help="Output jsonl path")
    p.add_argument("--model", default="qwen-vl-plus",
                   help="Model name (default: qwen-vl-plus)")
    p.add_argument("--base_url",
                   default="https://dashscope.aliyuncs.com/compatible-mode/v1",
                   help="API base URL")
    p.add_argument("--api_key", default=None,
                   help="API key (default: DASHSCOPE_API_KEY env var)")
    p.add_argument("--limit", type=int, default=None,
                   help="Stop after N generations (for smoke testing)")
    p.add_argument("--delay", type=float, default=1.5,
                   help="Seconds to sleep between requests (default: 1.5)")
    p.add_argument("--resume", action="store_true",
                   help="Skip (subset, id) pairs already present in out_jsonl")
    p.add_argument("--timeout", type=int, default=90,
                   help="Per-request timeout in seconds (default: 90)")
    return p.parse_args()


def main():
    args = parse_args()

    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit(
            "API key required: set DASHSCOPE_API_KEY or pass --api_key"
        )

    input_path = Path(args.input_jsonl)
    out_path   = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if line.strip() and json.loads(line).get("report_style", "original") != args.report_style:
                raise SystemExit("Report style differs from existing output; use a new --out_jsonl path.")

    # Load all input records
    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(records)} records from {input_path}")

    # Resume: find already-processed pairs
    done: set = set()
    if args.resume:
        done = _load_done(out_path)
        if done:
            print(f"Resume: skipping {len(done)} already-generated records.")

    n_generated = 0
    n_errors    = 0

    with open(out_path, "a", encoding="utf-8") as fout:
        for rec in records:
            subset = rec["subset"]
            sid    = rec["id"]

            if (subset, sid) in done:
                continue

            if args.limit is not None and n_generated >= args.limit:
                break

            messages = _build_messages(rec, args.setting, args.report_style)
            phash    = _prompt_hash(messages)

            out_rec = {
                "subset":           subset,
                "id":               sid,
                "variant":          rec.get("variant"),
                "setting":          args.setting,
                "model":            args.model,
                "prompt_hash":      phash,
                "report_style":     args.report_style,
                "reference_report": rec.get("reference_report", ""),
                "measurement":      rec.get("measurement"),
                "qaep":             rec.get("qaep"),
                "prediction":       None,
                "error":            None,
            }

            try:
                prediction = _call_api(
                    messages, args.model, args.base_url, api_key,
                    timeout=args.timeout,
                )
                if args.report_style == "concise":
                    out_rec["raw_prediction"] = prediction
                    _validate_concise_report(prediction)
                out_rec["prediction"] = prediction
                n_generated += 1
                print(f"  [{n_generated}] {subset}/{sid}  OK")
            except Exception as exc:
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    if exc.response.status_code in (401, 403, 404):
                        raise SystemExit(f"API configuration error; stopped: {exc}") from exc
                out_rec["error"] = str(exc)
                n_errors += 1
                print(f"  [ERR] {subset}/{sid}: {exc}")

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

            if args.delay > 0:
                time.sleep(args.delay)

    print(
        f"\nDone. generated={n_generated}  errors={n_errors}  "
        f"output={out_path}"
    )


if __name__ == "__main__":
    main()
