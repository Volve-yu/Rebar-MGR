"""Train a lightweight QAEP classifier.

This is intentionally small and dependency-light: one binary logistic
classifier is trained for each QAEP target. The model is saved as JSON so it
can be applied on AutoDL without extra ML packages.

Example:
  python src_report/train_qaep.py \
      --train_jsonl results_clean/report_generation/qaep/full_qaep_dataset.jsonl \
      --model_out results_clean/report_generation/qaep/qaep_model.json \
      --metrics_out results_clean/report_generation/qaep/qaep_metrics.json
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np

from qaep_utils import FEATURE_NAMES, TARGET_NAMES, feature_vector, read_jsonl


def sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))


def binary_metrics(y_true, prob):
    pred = (prob >= 0.5).astype(np.int32)
    y = y_true.astype(np.int32)
    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(len(y), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def make_matrix(records):
    x = np.asarray([feature_vector(r["features"]) for r in records], dtype=np.float32)
    y = {
        name: np.asarray([int(r["labels"][name]) for r in records], dtype=np.float32)
        for name in TARGET_NAMES
    }
    return x, y


def train_logistic(x, y, epochs, lr, l2):
    if float(y.min()) == float(y.max()):
        return {
            "constant": True,
            "probability": float(y[0]),
            "weights": [],
            "bias": 0.0,
        }

    n, d = x.shape
    w = np.zeros(d, dtype=np.float32)
    b = 0.0
    for _ in range(epochs):
        prob = sigmoid(x @ w + b)
        err = prob - y
        grad_w = (x.T @ err) / n + l2 * w
        grad_b = float(err.mean())
        w -= lr * grad_w
        b -= lr * grad_b

    return {
        "constant": False,
        "weights": [float(v) for v in w],
        "bias": float(b),
    }


def predict_model(x, clf):
    if clf.get("constant"):
        return np.full((x.shape[0],), float(clf["probability"]), dtype=np.float32)
    w = np.asarray(clf["weights"], dtype=np.float32)
    b = float(clf["bias"])
    return sigmoid(x @ w + b)


def parse_args():
    p = argparse.ArgumentParser(description="Train QAEP lightweight classifiers.")
    p.add_argument("--train_jsonl", required=True, type=Path)
    p.add_argument("--val_jsonl", default=None, type=Path)
    p.add_argument("--model_out", required=True, type=Path)
    p.add_argument("--metrics_out", default=None, type=Path)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-4)
    return p.parse_args()


def main():
    args = parse_args()
    train_records = read_jsonl(args.train_jsonl)
    if not train_records:
        raise ValueError(f"No records in {args.train_jsonl}")
    schemas = {r.get('schema_version', 'legacy') for r in train_records}
    if len(schemas)!=1:
        raise ValueError('Do not mix evidence schemas in QAEP training')
    if 'design_physical_v2' in schemas and any(r.get('split')!='train' for r in train_records):
        raise ValueError('V2 QAEP training requires training-split inputs')

    train_x_raw, train_y = make_matrix(train_records)
    mean = train_x_raw.mean(axis=0)
    std = train_x_raw.std(axis=0)
    std[std < 1e-6] = 1.0
    train_x = (train_x_raw - mean) / std

    classifiers = {}
    train_metrics = {}
    for name in TARGET_NAMES:
        clf = train_logistic(train_x, train_y[name], args.epochs, args.lr, args.l2)
        classifiers[name] = clf
        train_prob = predict_model(train_x, clf)
        train_metrics[name] = binary_metrics(train_y[name], train_prob)

    model = {
        "schema_version": next(iter(schemas)),
        "training_samples": [[r['subset'], str(r['id'])] for r in train_records],
        "feature_names": FEATURE_NAMES,
        "target_names": TARGET_NAMES,
        "mean": [float(v) if math.isfinite(float(v)) else 0.0 for v in mean],
        "std": [float(v) if math.isfinite(float(v)) else 1.0 for v in std],
        "classifiers": classifiers,
    }

    metrics = {"train": train_metrics}
    if args.val_jsonl:
        val_records = read_jsonl(args.val_jsonl)
        val_x_raw, val_y = make_matrix(val_records)
        val_x = (val_x_raw - mean) / std
        val_metrics = {}
        for name in TARGET_NAMES:
            val_prob = predict_model(val_x, classifiers[name])
            val_metrics[name] = binary_metrics(val_y[name], val_prob)
        metrics["val"] = val_metrics

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.write_text(json.dumps(model, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    metrics_path = args.metrics_out or args.model_out.with_name(args.model_out.stem + "_metrics.json")
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Wrote {args.model_out}")
    print(f"Wrote {metrics_path}")
    for split, split_metrics in metrics.items():
        print(f"[{split}]")
        for name, row in split_metrics.items():
            print(
                f"  {name}: acc={row['accuracy']:.3f} "
                f"f1={row['f1']:.3f} p={row['precision']:.3f} r={row['recall']:.3f}"
            )


if __name__ == "__main__":
    main()
