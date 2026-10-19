"""
Rebar-report adaptation of the LLaVA-1.6 benchmark path in Clip2Safety.

The Clip2Safety repository evaluates several VLMs for safety-compliance
reasoning, including LLaVA-1.6. This script turns that notebook-style
inference path into a reproducible JSONL runner for our rebar report task.

Example on a GPU machine:
  python external_report_baselines/Clip2Safety/rebar_adapt/llava16_rebar_report.py \
    --input_jsonl results_clean/report_generation/inputs_segsource_clean_ref/ours_full.jsonl \
    --out_jsonl results_clean/report_generation/generations/external_clip2safety_llava16_rebar.jsonl \
    --limit 10 \
    --load_in_4bit
"""

import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageOps


def _fmt(v, ndigits=1):
    if v is None:
        return "N/A"
    if isinstance(v, float) and math.isnan(v):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.{ndigits}f}"
    return str(v)


def _measurement_prompt(m: dict) -> str:
    if m.get("warning"):
        return f"Measurement warning: {m['warning']}"

    compliance = m.get("compliance") or {}
    per = compliance.get("per_spacing") or []
    fails = [p for p in per if not p.get("within_main_tol", True)]
    fail_text = "none"
    if fails:
        fail_text = "; ".join(
            f"spacing[{p.get('index')}]={_fmt(p.get('spacing_mm'))} mm "
            f"dev={_fmt(p.get('deviation_from_mean_mm'))} mm"
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

    return (
        f"Detected rebar count: {m.get('rebar_count')}. "
        f"Mean spacing: {_fmt(m.get('mean_spacing_mm'))} mm; "
        f"spacing std: {_fmt(m.get('std_spacing_mm'))} mm. "
        f"GB 50204-2015 main-bar tolerance: +/-20 mm. "
        f"Compliance result: {compliance.get('status', 'N/A')}. "
        f"Non-compliant spacings: {fail_text}. "
        f"Spacing trend: {trend}."
    )


def _compose_rgb_seg(rgb_path: str, seg_path: str) -> Image.Image:
    rgb = Image.open(rgb_path).convert("RGB")
    seg = Image.open(seg_path).convert("L")
    seg = ImageOps.autocontrast(seg)
    seg = seg.resize(rgb.size, Image.Resampling.NEAREST).convert("RGB")
    canvas = Image.new("RGB", (rgb.width * 2, rgb.height), "white")
    canvas.paste(rgb, (0, 0))
    canvas.paste(seg, (rgb.width, 0))
    return canvas


def _load_image(rec: dict, image_mode: str) -> Image.Image:
    if image_mode == "rgb":
        return Image.open(rec["image_path"]).convert("RGB")
    if image_mode == "seg":
        return Image.open(rec["seg_path"]).convert("RGB")
    return _compose_rgb_seg(rec["image_path"], rec["seg_path"])


def _build_prompt(rec: dict) -> str:
    return _build_prompt_with_options(rec, include_measurement=True)


def _build_prompt_with_options(rec: dict, include_measurement: bool) -> str:
    if not include_measurement:
        return (
            "USER: <image>\n"
            "You are checking rebar installation quality. The image may "
            "contain the RGB scene and/or the predicted segmentation mask. "
            "Use only the visual evidence and the GB 50204-2015 rule that "
            "main-bar spacing deviation is assessed with a +/-20 mm tolerance. "
            "Write one professional inspection-report paragraph. Do not invent "
            "precise millimetre measurements, spacing indices, compliance "
            "status, or risk level when they cannot be determined from the "
            "visual evidence. State uncertainty when image evidence alone is "
            "insufficient for millimetre-level spacing assessment.\n"
            "ASSISTANT:"
        )

    return (
        "USER: <image>\n"
        "You are checking rebar installation quality. The image may contain "
        "the RGB scene and/or the predicted segmentation mask. Use the "
        "structured measurement facts below as authoritative. Write one "
        "professional inspection-report paragraph. Mention count, mean "
        "spacing, GB 50204-2015 +/-20 mm tolerance, compliance result, "
        "abnormal spacing positions if any, trend, and final corrective or "
        "readiness statement. Do not invent extra measurements.\n\n"
        f"Measurement facts: {_measurement_prompt(rec.get('measurement') or {})}\n"
        "ASSISTANT:"
    )


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
    p.add_argument("--model_id", default="llava-hf/llava-v1.6-mistral-7b-hf")
    p.add_argument("--image_mode", choices=["rgb", "seg", "rgb_seg"], default="rgb_seg")
    p.add_argument("--include_measurement", action="store_true")
    p.add_argument("--max_new_tokens", type=int, default=160)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--load_in_4bit", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    import torch
    from transformers import (
        BitsAndBytesConfig,
        LlavaNextForConditionalGeneration,
        LlavaNextProcessor,
    )

    quant = None
    if args.load_in_4bit:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

    processor = LlavaNextProcessor.from_pretrained(args.model_id)
    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
        quantization_config=quant,
    )
    model.eval()

    records = [
        json.loads(line)
        for line in Path(args.input_jsonl).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
                "setting": "external_clip2safety_llava16_rebar",
                "model": args.model_id,
                "reference_report": rec.get("reference_report", ""),
                "measurement": rec.get("measurement"),
                "prediction": None,
                "error": None,
            }
            try:
                image = _load_image(rec, args.image_mode)
                prompt = _build_prompt_with_options(
                    rec, include_measurement=args.include_measurement
                )
                inputs = processor(text=prompt, images=image, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                text = processor.batch_decode(
                    output,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
                if "ASSISTANT:" in text:
                    text = text.split("ASSISTANT:", 1)[1]
                out_rec["prediction"] = text.strip()
                print(f"[OK] {rec['subset']}/{rec['id']}")
            except Exception as exc:
                out_rec["error"] = str(exc)
                print(f"[ERR] {rec['subset']}/{rec['id']}: {exc}")

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()
            generated += 1

    print(f"Wrote {out_path}")
    print(f"Processed: {generated}")


if __name__ == "__main__":
    main()
