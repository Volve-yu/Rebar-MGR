"""
generate_hf.py -- generate inspection reports using a local HuggingFace VLM.

Supports:
  Qwen/Qwen2.5-VL-7B-Instruct      (via Qwen2_5VLForConditionalGeneration)
  llava-hf/llava-1.5-7b-hf         (via LlavaForConditionalGeneration)

Auto-detects model family from the model name/path; the caller does not need
to specify a --model_family flag.

Output format matches generate.py exactly:
  subset, id, variant, setting, model, prompt_hash,
  prediction, reference_report, measurement, error

Usage (Qwen2.5-VL):
  python src_report/generate_hf.py \\
      --input_jsonl results_clean/report_generation/inputs/full.jsonl \\
      --setting rgb_only \\
      --hf_model_path hf_models/Qwen2.5-VL-7B-Instruct \\
      --out_jsonl results_clean/report_generation/generations/baseline_qwen25vl_rgb_only.jsonl \\
      --device cuda --limit 5

Usage (LLaVA-1.5):
  python src_report/generate_hf.py \\
      --input_jsonl results_clean/report_generation/inputs/full.jsonl \\
      --setting rgb_only \\
      --hf_model_path hf_models/llava-1.5-7b-hf \\
      --out_jsonl results_clean/report_generation/generations/baseline_llava_rgb_only.jsonl \\
      --device cuda --limit 5
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Shared prompt text (must stay in sync with generate.py SYSTEM_PROMPT)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a structural inspection report writer specialising in rebar "
    "(reinforcing bar) installation quality control. "
    "When given inspection data or a segmentation image, write a single "
    "Professional Narrative paragraph in English. "
    "Requirements: exactly 2-4 sentences; use only the numerical values "
    "provided and do not fabricate measurements; do not include reasoning "
    "steps, bullet points, headers, or any preamble text; output only the "
    "paragraph text."
)

RGB_ONLY_USER = (
    "This is the original RGB image of a rebar installation inspection "
    "scene. Write a Professional Narrative paragraph (2-4 sentences) "
    "assessing the rebar installation quality based only on the visual "
    "image. Do not invent precise numerical measurements unless they are "
    "visually evident. Output only the paragraph text."
)

VALID_SETTINGS = ("rgb_only",)


# ---------------------------------------------------------------------------
# Model-family detection
# ---------------------------------------------------------------------------

def _detect_family(model_name_or_path: str) -> str:
    """Return 'qwen25vl' or 'llava15' based on the model name."""
    name = Path(model_name_or_path).name.lower()
    if "qwen2.5-vl" in name or "qwen2_5" in name or "qwen2-5" in name:
        return "qwen25vl"
    if "llava" in name:
        return "llava15"
    raise ValueError(
        f"Cannot detect model family from '{model_name_or_path}'. "
        "Expected a name containing 'qwen2.5-vl' or 'llava'."
    )


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class _Qwen25VLBackend:
    """Wrapper for Qwen2.5-VL-7B-Instruct."""

    def __init__(self, model_path: str, device: str, dtype: torch.dtype):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        device_map = "auto" if device.startswith("cuda") else None
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        self.device = device

    def generate(self, image: Image.Image, setting: str) -> str:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text",  "text": RGB_ONLY_USER},
                ],
            },
        ]
        # apply_chat_template needs the qwen-vl vision processor
        from qwen_vl_utils import process_vision_info  # installed with Qwen2.5-VL

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        # Trim prompt tokens from output
        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()


class _LLaVA15Backend:
    """Wrapper for llava-hf/llava-1.5-7b-hf."""

    def __init__(self, model_path: str, device: str, dtype: torch.dtype):
        from transformers import LlavaForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.model.eval()
        self.device = device

    def generate(self, image: Image.Image, setting: str) -> str:
        # LLaVA-1.5 expects a single <image> token in the prompt.
        # System + user joined because LLaVA-1.5 is single-turn.
        prompt = (
            f"USER: <image>\n{SYSTEM_PROMPT}\n\n{RGB_ONLY_USER}\nASSISTANT:"
        )
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        generated = output_ids[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(
            generated, skip_special_tokens=True
        )[0].strip()


def _load_backend(model_path: str, device: str, dtype: torch.dtype):
    family = _detect_family(model_path)
    if family == "qwen25vl":
        return _Qwen25VLBackend(model_path, device, dtype)
    if family == "llava15":
        return _LLaVA15Backend(model_path, device, dtype)
    raise ValueError(f"Unsupported family: {family}")


# ---------------------------------------------------------------------------
# Prompt hash (text-only; images are excluded from hash)
# ---------------------------------------------------------------------------

def _prompt_hash(setting: str, model_path: str) -> str:
    canonical = json.dumps(
        {"system": SYSTEM_PROMPT, "user": RGB_ONLY_USER,
         "setting": setting, "model": Path(model_path).name},
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


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
        description="Generate inspection reports with a local HuggingFace VLM."
    )
    p.add_argument("--input_jsonl",    required=True,
                   help="Input jsonl from build_report_inputs.py")
    p.add_argument("--setting",        default="rgb_only", choices=VALID_SETTINGS,
                   help="Prompt setting (default: rgb_only)")
    p.add_argument("--hf_model_path",  required=True,
                   help="Path to the HuggingFace model directory "
                        "(relative to CWD or absolute)")
    p.add_argument("--out_jsonl",      required=True,
                   help="Output jsonl path")
    p.add_argument("--device",         default="cuda",
                   help="Inference device: cuda or cpu (default: cuda)")
    p.add_argument("--dtype",          default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Model dtype (default: bfloat16)")
    p.add_argument("--limit",          type=int, default=None,
                   help="Stop after N generations")
    p.add_argument("--resume",         action="store_true",
                   help="Skip (subset, id) pairs already in out_jsonl")
    return p.parse_args()


def main():
    args = parse_args()

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    dtype = dtype_map[args.dtype]

    model_path = args.hf_model_path
    model_name = Path(model_path).name

    print(f"Loading model from: {model_path}  (dtype={args.dtype}, device={args.device})")
    backend = _load_backend(model_path, args.device, dtype)
    print("Model loaded.")

    input_path = Path(args.input_jsonl)
    out_path   = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(records)} records from {input_path}")

    done: set = set()
    if args.resume:
        done = _load_done(out_path)
        if done:
            print(f"Resume: skipping {len(done)} already-generated records.")

    phash = _prompt_hash(args.setting, model_path)

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

            out_rec = {
                "subset":           subset,
                "id":               sid,
                "variant":          rec.get("variant"),
                "setting":          args.setting,
                "model":            model_name,
                "prompt_hash":      phash,
                "reference_report": rec.get("reference_report", ""),
                "measurement":      rec.get("measurement"),
                "prediction":       None,
                "error":            None,
            }

            try:
                img_path = rec["image_path"]
                image = Image.open(img_path).convert("RGB")
                prediction = backend.generate(image, args.setting)
                out_rec["prediction"] = prediction
                n_generated += 1
                print(f"  [{n_generated}] {subset}/{sid}  OK")
            except Exception as exc:
                out_rec["error"] = str(exc)
                n_errors += 1
                print(f"  [ERR] {subset}/{sid}: {exc}", file=sys.stderr)

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            fout.flush()

    print(
        f"\nDone. generated={n_generated}  errors={n_errors}  "
        f"output={out_path}"
    )


if __name__ == "__main__":
    main()
