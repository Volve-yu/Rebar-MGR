"""Apply a trained QAEP model to report-generation inputs.

This script augments each input record with a `qaep` field that contains
predicted reliability labels, probabilities, risk tags, and a textual report
plan. The output can be consumed by src_report/generate.py as an additional
prompt conditioning signal.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from qaep_utils import (
    FEATURE_NAMES,
    extract_features,
    feature_vector,
    qaep_plan,
    read_jsonl,
    write_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser(description="Apply QAEP to report-generation inputs.")
    p.add_argument("--input_jsonl", required=True, type=Path)
    p.add_argument("--data_root", default="data", type=Path)
    p.add_argument("--model_path", required=True, type=Path)
    p.add_argument("--out_jsonl", required=True, type=Path)
    p.add_argument("--subsets", nargs="*", default=None,
                   help="Optional subset names to keep, e.g. site-1 site-2.")
    return p.parse_args()


def load_model(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def predict_prob(features, model):
    x = np.asarray(feature_vector(features), dtype=np.float32)
    mean = np.asarray(model["mean"], dtype=np.float32)
    std = np.asarray(model["std"], dtype=np.float32)
    x = (x - mean) / std

    probabilities = {}
    labels = {}
    for name, clf in model["classifiers"].items():
        if clf.get("constant"):
            prob = float(clf["probability"])
        else:
            w = np.asarray(clf["weights"], dtype=np.float32)
            b = float(clf["bias"])
            prob = float(sigmoid(x @ w + b))
        probabilities[name] = prob
        labels[name] = int(prob >= 0.5)
    return probabilities, labels


def main():
    args = parse_args()
    model = load_model(args.model_path)
    records = read_jsonl(args.input_jsonl)
    if args.subsets:
        keep = set(args.subsets)
        records = [r for r in records if str(r.get("subset")) in keep]
    out_records = []

    for rec in records:
        if rec.get('schema_version', 'legacy') != model.get('schema_version', 'legacy'):
            raise ValueError('QAEP model schema differs; retrain QAEP on the new evidence.')
        if rec.get('protocol') == 'cross_site' and rec.get('subset') in {s[0] for s in model.get('training_samples', [])}:
            raise ValueError('Cross-site QAEP model was trained on the target site')
        features, quality = extract_features(rec, args.data_root, include_gt=False)
        probabilities, labels = predict_prob(features, model)
        qaep = qaep_plan(rec, probabilities, labels)
        qaep["features"] = {k: float(v) for k, v in features.items()}
        qaep["quality"] = quality

        new_rec = dict(rec)
        new_rec["qaep"] = qaep
        out_records.append(new_rec)

    write_jsonl(args.out_jsonl, out_records)
    print(f"Wrote {args.out_jsonl}")
    print(f"Records: {len(out_records)}")


if __name__ == "__main__":
    main()
