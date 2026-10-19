"""
Rebar-inspection adaptation of the ConstructionSite 10k VLM inference style.

The original repository uses VLM prompting for construction-site image
description and safety-rule VQA. This adapter keeps that downstream interface
idea, but reads our rebar evidence JSONL and writes the same JSONL schema used
by src_report/eval.py.

Example:
  python external_report_baselines/ConstructionSite-10k-Implementation/Inferences/rebar_query_openai_compatible.py ^
    --input_jsonl results_clean/report_generation/inputs_segsource_clean_ref/ours_full.jsonl ^
    --out_jsonl results_clean/report_generation/generations/external_constructionsite_rebar.jsonl ^
    --model qwen-vl-plus ^
    --base_url https://dashscope.aliyuncs.com/compatible-mode/v1 ^
    --limit 10
"""

import argparse
import base64
import json
import math
import os
import time
from pathlib import Path

import requests


SYSTEM_PROMPT = (
    "You are a construction site safety inspector. You view the provided "
    "visual evidence and answer your supervisor in a concise inspection-report "
    "style. Focus on observable construction quality evidence and the supplied "
    "rule facts."
)


def _fmt(v, ndigits=1):
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def _measurement_text(m: dict) -> str:
    if m.get("warning"):
        return f"Measurement warning: {m['warning']}"

    compliance = m.get("compliance") or {}
    per = compliance.get("per_spacing") or []
    fails = [p for p in per if not p.get("within_main_tol", True)]
    fail_text = "none"
    max_dev = None
    if per:
        max_dev = max(abs(p.get("deviation_from_mean_mm", 0.0)) for p in per)
    if fails:
        fail_text = "; ".join(
            f"spacing[{p.get('index')}]={_fmt(p.get('spacing_mm'))} mm "
            f"(dev={_fmt(p.get('deviation_from_mean_mm'))} mm)"
            for p in fails
        )

    spacings = m.get("spacings_mm") or []
    trend = "N/A"
    if len(spacings) >= 2:
        slope = (spacings[-1] - spacings[0]) / max(len(spacings) - 1, 1)
        if slope > 2.0:
            trend = f"increasing ({slope:+.1f} mm/gap)"
        elif slope < -2.0:
            trend = f"decreasing ({slope:+.1f} mm/gap)"
        else:
            trend = "uniform"

    return "\n".join(
        [
            f"Detected rebar count: {m.get('rebar_count')}",
            (
                f"Mean spacing: {_fmt(m.get('mean_spacing_mm'))} mm; "
                f"spacing std: {_fmt(m.get('std_spacing_mm'))} mm"
            ),
            f"Compliance status: {compliance.get('status', 'N/A')}",
            "Rule: GB 50204-2015 main-bar spacing deviation tolerance is +/-20 mm.",
            f"Non-compliant spacings: {fail_text}",
            f"Maximum deviation from mean: {_fmt(max_dev)} mm",
            f"Spacing trend: {trend}",
        ]
    )


def _img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _image_block(path: str) -> dict:
    suffix = Path(path).suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{_img_b64(path)}"},
    }


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


def _build_messages(rec: dict, image_mode: str, include_measurement: bool) -> list:
    content = []
    if image_mode in {"rgb", "rgb_seg"}:
        content.append(_image_block(rec["image_path"]))
    if image_mode in {"seg", "rgb_seg"}:
        content.append(_image_block(rec["seg_path"]))

    image_note = {
        "rgb": "The provided image is the RGB construction-site image.",
        "seg": "The provided image is a predicted rebar segmentation mask.",
        "rgb_seg": (
            "The first image is the RGB construction-site image; the second "
            "image is a predicted rebar segmentation mask."
        ),
    }[image_mode]

    if include_measurement:
        prompt = (
            "Please inspect the rebar installation evidence and generate one "
            "professional paragraph. "
            f"{image_note} Use the supplied measurement facts as authoritative; "
            "do not infer millimeter values from pixels. Mention the detected "
            "count, mean spacing, GB 50204-2015 +/-20 mm rule, compliance result, "
            "abnormal spacing positions if any, trend, and final corrective or "
            "readiness statement. Output only the paragraph.\n\n"
            "Measurement facts:\n"
            f"{_measurement_text(rec.get('measurement') or {})}"
        )
    else:
        prompt = (
            "Please inspect the rebar installation evidence and generate one "
            "professional paragraph in the style of a construction-site visual "
            "inspection report. "
            f"{image_note} The inspection target is rebar spacing quality under "
            "GB 50204-2015, whose main-bar spacing deviation tolerance is +/-20 mm. "
            "Use only the visual evidence provided. Do not invent exact millimeter "
            "spacings, measurement values, or deterministic compliance results when "
            "they cannot be derived from the image. Output only the paragraph."
        )
    content.append(_text_block(prompt))
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _call_api(messages, model, base_url, api_key, timeout):
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 512,
    }
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def _done_pairs(path: Path) -> set:
    done = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
            done.add((rec["subset"], str(rec["id"])))
        except Exception:
            pass
    return done


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--model", default="qwen-vl-plus")
    p.add_argument("--base_url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    p.add_argument("--api_key", default=None)
    p.add_argument("--image_mode", choices=["rgb", "seg", "rgb_seg"], default="rgb")
    p.add_argument("--include_measurement", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--delay", type=float, default=1.5)
    p.add_argument("--timeout", type=int, default=90)
    return p.parse_args()


def main():
    args = parse_args()
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY")
    if not api_key:
        raise SystemExit("API key required: set DASHSCOPE_API_KEY or pass --api_key")

    input_path = Path(args.input_jsonl)
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    done = _done_pairs(out_path) if args.resume else set()

    generated = 0
    with out_path.open("a", encoding="utf-8") as fout:
        for rec in records:
            key = (rec["subset"], str(rec["id"]))
            if key in done:
                continue
            if args.limit is not None and generated >= args.limit:
                break

            out_rec = {
                "subset": rec["subset"],
                "id": str(rec["id"]),
                "variant": rec.get("variant"),
                "setting": "external_constructionsite_rebar",
                "model": args.model,
                "reference_report": rec.get("reference_report", ""),
                "measurement": rec.get("measurement"),
                "prediction": None,
                "error": None,
            }
            try:
                messages = _build_messages(rec, args.image_mode, args.include_measurement)
                out_rec["prediction"] = _call_api(
                    messages, args.model, args.base_url, api_key, args.timeout
                )
                print(f"[OK] {rec['subset']}/{rec['id']}")
            except Exception as exc:
                out_rec["error"] = str(exc)
                print(f"[ERR] {rec['subset']}/{rec['id']}: {exc}")

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            generated += 1
            if args.delay > 0:
                time.sleep(args.delay)

    print(f"Wrote {out_path}")
    print(f"Processed: {generated}")


if __name__ == "__main__":
    main()
