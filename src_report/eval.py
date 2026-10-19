import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from design_evidence import is_v2, risk_level, deviation, is_failed


WORD_RE = re.compile(r"[A-Za-z0-9.+-]+")
NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?")
MAX_DEV_MISSING_ERROR_MM = 100.0
MAX_DEV_SCORE_SCALE_MM = 20.0
NUMERIC_FACT_TOL_MM = 1.0
RISK_ORDER = {
    "SAFE": 0,
    "ACCEPTABLE": 1,
    "MARGINAL": 2,
    "WARNING": 3,
    "HIGH_RISK": 3,
    "CRITICAL": 4,
}


def tokenize(text):
    return WORD_RE.findall((text or "").lower())


def ngrams(tokens, n):
    return [tuple(tokens[i:i + n]) for i in range(max(0, len(tokens) - n + 1))]


def bleu_score(pred, ref, max_n=4):
    pred_t = tokenize(pred)
    ref_t = tokenize(ref)
    if not pred_t or not ref_t:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        p_ngrams = Counter(ngrams(pred_t, n))
        r_ngrams = Counter(ngrams(ref_t, n))
        if not p_ngrams:
            precisions.append(0.0)
            continue
        overlap = sum(min(count, r_ngrams[gram]) for gram, count in p_ngrams.items())
        precisions.append((overlap + 1.0) / (sum(p_ngrams.values()) + 1.0))

    geo = math.exp(sum(math.log(p) for p in precisions) / max_n)
    bp = 1.0 if len(pred_t) > len(ref_t) else math.exp(1.0 - len(ref_t) / max(len(pred_t), 1))
    return bp * geo


def lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, 1):
            if x == y:
                cur.append(prev[j - 1] + 1)
            else:
                cur.append(max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(pred, ref):
    pred_t = tokenize(pred)
    ref_t = tokenize(ref)
    if not pred_t or not ref_t:
        return 0.0
    lcs = lcs_len(pred_t, ref_t)
    prec = lcs / len(pred_t)
    rec = lcs / len(ref_t)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def extract_status(text):
    t = (text or "").lower()
    result_match = re.search(r"\bresult\s*:\s*(pass|fail)\b", t)
    if result_match:
        return result_match.group(1).upper()
    if re.search(r"\bfail(?:ed|s)?\b|non-compliant|not compliant", t):
        return "FAIL"
    if re.search(r"\bpass(?:ed|es)?\b|pass\s*\(\s*main\s*\)|compliant|within acceptable", t):
        return "PASS"
    return None


def normalize_status(status):
    if status is None:
        return None
    s = str(status).upper()
    if s.startswith("PASS"):
        return "PASS"
    if s.startswith("FAIL"):
        return "FAIL"
    return s


def extract_risk(text):
    t = (text or "").lower()
    for key in ["critical", "high risk", "warning", "marginal", "acceptable", "safe"]:
        if key in t:
            return key.upper().replace(" ", "_")
    return None


def risk_level_error(pred_risk, target_risk):
    if pred_risk is None or target_risk is None:
        return None
    if pred_risk not in RISK_ORDER or target_risk not in RISK_ORDER:
        return None
    return abs(RISK_ORDER[pred_risk] - RISK_ORDER[target_risk])


def target_risk_from_measurement(m):
    if is_v2(m):
        return risk_level(m)
    compliance = m.get("compliance", {})
    per = compliance.get("per_spacing", []) or []
    status = compliance.get("status")
    if not per:
        return None

    devs = [abs(float(p.get("deviation_from_mean_mm", 0.0))) for p in per]
    max_dev = max(devs) if devs else 0.0
    fail_ratio = sum(1 for d in devs if d > 20.0) / len(per)

    if fail_ratio == 0 and max_dev <= 5:
        return "SAFE"
    if fail_ratio == 0 and max_dev <= 10:
        return "ACCEPTABLE"
    if fail_ratio <= 0.25 and max_dev <= 40:
        return "MARGINAL"
    if fail_ratio <= 0.5 or max_dev <= 60:
        return "WARNING"
    if status == "PASS":
        return "ACCEPTABLE"
    return "CRITICAL"


def measurement_facts(m):
    compliance = m.get("compliance", {}) or {}
    per = compliance.get("per_spacing", []) or []
    devs = [abs(float(deviation(p))) for p in per]
    fails = [p for p in per if is_failed(p)]
    max_dev = max(devs) if devs else None
    trend = None
    spacings = m.get("spacings_mm") or []
    if len(spacings) >= 2:
        slope = (float(spacings[-1]) - float(spacings[0])) / max(len(spacings) - 1, 1)
        if slope > 2.0:
            trend = "INCREASING"
        elif slope < -2.0:
            trend = "DECREASING"
        else:
            trend = "UNIFORM"

    return {
        "status": compliance.get("status"),
        "risk": target_risk_from_measurement(m),
        "rebar_count": m.get("rebar_count"),
        "num_spacings": len(spacings) if spacings else None,
        "mean_spacing": m.get("mean_spacing_mm"),
        "std_spacing": m.get("std_spacing_mm"),
        "spacings": spacings,
        "max_dev": max_dev,
        "fail_indices": {int(p.get("index")) for p in fails if p.get("index") is not None},
        "trend": trend,
    }


def contains_number_near(text, target, tol=1.0):
    if target is None:
        return None
    nums = [float(x) for x in NUM_RE.findall(text or "")]
    return any(abs(x - target) <= tol for x in nums)


def extract_max_dev_value(text):
    """
    Extract the reported maximum spacing deviation in millimetres.

    The matcher intentionally looks for a number near a maximum-deviation phrase
    rather than choosing the closest number in the whole report. This avoids
    giving credit for unrelated values such as the GB tolerance or individual
    spacing measurements.
    """
    t = text or ""
    patterns = [
        r"(?:maximum|max|largest|greatest)\s+(?:absolute\s+)?(?:spacing\s+)?(?:deviation|dev)"
        r"[^0-9+\-]{0,50}([-+]?\d+(?:\.\d+)?)\s*(?:mm)?",
        r"(?:deviation|dev)\s*(?:maximum|max)\s*[^0-9+\-]{0,50}([-+]?\d+(?:\.\d+)?)\s*(?:mm)?",
        r"(?:maxdev|max_dev)\s*[:=]?\s*([-+]?\d+(?:\.\d+)?)\s*(?:mm)?",
    ]
    for pattern in patterns:
        m = re.search(pattern, t, flags=re.IGNORECASE)
        if m:
            return abs(float(m.group(1)))

    # Some VLMs list abnormal intervals without explicitly writing "maximum",
    # e.g. "spacing[5]=83.7 mm, which deviates ... by -21.0 mm". In that case,
    # the largest signed deviation in a deviation-related sentence is used.
    signed_candidates = []
    for sent in re.split(r"[.;\n]", t):
        if "deviat" not in sent.lower():
            continue
        for m in re.finditer(r"(?<![/\w])[-+]\d+(?:\.\d+)?", sent):
            signed_candidates.append(abs(float(m.group(0))))
    if signed_candidates:
        return max(signed_candidates)

    return None


def extract_first_number(patterns, text):
    for pattern in patterns:
        m = re.search(pattern, text or "", flags=re.IGNORECASE)
        if m:
            return float(m.group(1))
    return None


def extract_rebar_count(text):
    value = extract_first_number([
        r"(?:detected\s+)?(?:rebar\s+)?count(?:\s+of|\s+is|:)?\s*(\d+)",
        r"detected\s+(\d+)\s+(?:vertical\s+)?rebars?",
        r"(\d+)\s+(?:vertical\s+)?rebars?",
    ], text)
    return int(value) if value is not None else None


def extract_mean_spacing_value(text):
    return extract_first_number([
        r"mean\s+spacing(?:\s+was|\s+is|:|\s+of)?\s*([-+]?\d+(?:\.\d+)?)\s*(?:mm)?",
        r"average\s+spacing(?:\s+was|\s+is|:|\s+of)?\s*([-+]?\d+(?:\.\d+)?)\s*(?:mm)?",
    ], text)


def extract_std_spacing_value(text):
    return extract_first_number([
        r"standard\s+deviation(?:\s+was|\s+is|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(?:mm)?",
        r"\bstd(?:\.|ev|andard\s+deviation)?(?:\s+was|\s+is|:)?\s*([-+]?\d+(?:\.\d+)?)\s*(?:mm)?",
    ], text)


def extract_spacing_value_claims(text):
    claims = []
    patterns = [
        r"spacing\s*\[\s*(\d+)\s*\]\s*(?:=|:|measures|is|was)?\s*([-+]?\d+(?:\.\d+)?)\s*(?:mm)?",
        r"spacing\s+(\d+)\s*(?:=|:|measures|is|was)\s*([-+]?\d+(?:\.\d+)?)\s*(?:mm)?",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, text or "", flags=re.IGNORECASE):
            claims.append((int(m.group(1)), float(m.group(2))))
    return claims


def has_no_failure_statement(text):
    t = (text or "").lower()
    return bool(re.search(
        r"no\s+(?:non[-\s]?compliant|abnormal|failed)\s+(?:spacing|interval|spacings|intervals)"
        r"|all\s+(?:spacing|spacings|intervals).{0,40}(?:within|compliant)",
        t,
    ))


def max_dev_error_and_score(pred, target):
    if target is None:
        return None, None, None
    pred_value = extract_max_dev_value(pred)
    if pred_value is None:
        return None, MAX_DEV_MISSING_ERROR_MM, 0.0
    err = abs(pred_value - float(target))
    score = max(0.0, 1.0 - err / MAX_DEV_SCORE_SCALE_MM)
    return pred_value, err, score


def extract_spacing_indices(text):
    found = set()
    # Require an explicit index marker: "mean spacing 103.7 mm" is not interval 103.
    for m in re.finditer(r"\bspacing\s*\[\s*(\d+)\s*\]", text or "", flags=re.IGNORECASE):
        found.add(int(m.group(1)))
    for m in re.finditer(r"\bspacing\s+index\s+(\d+)\b", text or "", flags=re.IGNORECASE):
        found.add(int(m.group(1)))
    return found


def fail_index_f1(pred_indices, target_indices):
    pred_indices = set(pred_indices or set())
    target_indices = set(target_indices or set())
    if not target_indices and not pred_indices:
        return 1.0
    if not target_indices or not pred_indices:
        return 0.0
    tp = len(pred_indices & target_indices)
    precision = tp / len(pred_indices) if pred_indices else 0.0
    recall = tp / len(target_indices) if target_indices else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def near_number(pred, target, tol=NUMERIC_FACT_TOL_MM):
    if pred is None or target is None:
        return False
    return abs(float(pred) - float(target)) <= tol


def key_fact_metrics(pred, facts):
    pred_status = extract_status(pred)
    pred_risk = extract_risk(pred)
    pred_count = extract_rebar_count(pred)
    pred_mean = extract_mean_spacing_value(pred)
    pred_std = extract_std_spacing_value(pred)
    pred_max_dev = extract_max_dev_value(pred)
    pred_fail_indices = extract_spacing_indices(pred)
    no_failure_statement = has_no_failure_statement(pred)

    target_status = normalize_status(facts["status"])
    target_risk = facts["risk"]
    target_fail_indices = facts["fail_indices"]

    slots = [
        ("status", target_status, pred_status, pred_status == target_status),
        ("risk", target_risk, pred_risk, pred_risk == target_risk),
        ("rebar_count", facts["rebar_count"], pred_count, pred_count == facts["rebar_count"]),
        ("mean_spacing", facts["mean_spacing"], pred_mean, near_number(pred_mean, facts["mean_spacing"])),
        ("std_spacing", facts["std_spacing"], pred_std, near_number(pred_std, facts["std_spacing"])),
        ("max_dev", facts["max_dev"], pred_max_dev, near_number(pred_max_dev, facts["max_dev"])),
    ]

    target_units = 0
    key_predicted_units = 0
    covered_units = 0
    key_correct_units = 0.0

    for _, target, pred_value, is_correct in slots:
        if target is None:
            continue
        target_units += 1
        if pred_value is not None:
            key_predicted_units += 1
            covered_units += 1
            if is_correct:
                key_correct_units += 1

    target_units += 1
    if target_fail_indices:
        if pred_fail_indices:
            key_predicted_units += 1
            covered_units += 1
            key_correct_units += fail_index_f1(pred_fail_indices, target_fail_indices)
    else:
        if no_failure_statement or pred_status == "PASS":
            key_predicted_units += 1
            covered_units += 1
            key_correct_units += 1.0 if not pred_fail_indices else 0.0
        elif pred_fail_indices:
            key_predicted_units += 1

    spacing_claims = extract_spacing_value_claims(pred)
    claim_predicted_units = key_predicted_units
    claim_correct_units = key_correct_units
    supported_spacing_claims = 0
    spacings = facts.get("spacings") or []
    for idx, value in spacing_claims:
        claim_predicted_units += 1
        if 0 <= idx < len(spacings) and near_number(value, spacings[idx]):
            supported_spacing_claims += 1
            claim_correct_units += 1

    precision = key_correct_units / key_predicted_units if key_predicted_units else 0.0
    recall = key_correct_units / target_units if target_units else 0.0
    key_fact_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )
    coverage = covered_units / target_units if target_units else None
    unsupported_rate = (
        max(0.0, claim_predicted_units - claim_correct_units) / claim_predicted_units
        if claim_predicted_units else 0.0
    )
    supported_claim_rate = 1.0 - unsupported_rate
    coverage_weighted_support = (
        coverage * supported_claim_rate if coverage is not None else None
    )
    fact_score = mean([key_fact_f1, coverage, coverage_weighted_support])

    return {
        "key_fact_precision": precision,
        "key_fact_recall": recall,
        "key_fact_f1": key_fact_f1,
        "evidence_coverage": coverage,
        "unsupported_claim_rate": unsupported_rate,
        "supported_claim_rate": supported_claim_rate,
        "coverage_weighted_support": coverage_weighted_support,
        "fact_score": fact_score,
        "pred_rebar_count": pred_count,
        "pred_mean_spacing": pred_mean,
        "pred_std_spacing": pred_std,
        "num_spacing_claims": len(spacing_claims),
        "num_supported_spacing_claims": supported_spacing_claims,
    }


def trend_match(pred, target):
    if target is None:
        return None
    t = (pred or "").lower()
    if target == "INCREASING":
        return int("increasing" in t or "widen" in t)
    if target == "DECREASING":
        return int("decreasing" in t or "narrow" in t)
    if target == "UNIFORM":
        return int("uniform" in t or "no clear systematic" in t or "consistent" in t)
    return None


def evaluate_record(rec):
    pred = rec.get("prediction") or ""
    ref = rec.get("reference_report") or ""
    facts = measurement_facts(rec.get("measurement") or {})

    pred_status = extract_status(pred)
    target_status = normalize_status(facts["status"])
    pred_risk = extract_risk(pred)
    target_risk = facts["risk"]
    pred_indices = extract_spacing_indices(pred)
    pred_max_dev, max_dev_abs_error, max_dev_score = max_dev_error_and_score(
        pred, facts["max_dev"]
    )
    risk_abs_error = risk_level_error(pred_risk, target_risk)

    fail_indices = facts["fail_indices"]
    if fail_indices:
        fail_idx_match = int(fail_indices.issubset(pred_indices))
    else:
        fail_idx_match = None
    fail_idx_f1 = fail_index_f1(pred_indices, fail_indices)
    trend = trend_match(pred, facts["trend"])
    risk_score = None
    if risk_abs_error is not None:
        risk_score = max(0.0, 1.0 - risk_abs_error / max(RISK_ORDER.values()))
    key_metrics = key_fact_metrics(pred, facts)

    return {
        "bleu1": bleu_score(pred, ref, max_n=1),
        "bleu2": bleu_score(pred, ref, max_n=2),
        "bleu3": bleu_score(pred, ref, max_n=3),
        "bleu4": bleu_score(pred, ref, max_n=4),
        "bleu": bleu_score(pred, ref, max_n=4),
        "rouge_l": rouge_l(pred, ref),
        "status_match": int(pred_status == target_status) if target_status else None,
        "risk_match": int(pred_risk == target_risk) if target_risk else None,
        "max_dev_match": int(contains_number_near(pred, facts["max_dev"], 1.0)),
        "fail_index_match": fail_idx_match,
        "risk_abs_error": risk_abs_error,
        "risk_score": risk_score,
        "pred_max_dev": pred_max_dev,
        "max_dev_abs_error": max_dev_abs_error,
        "max_dev_score": max_dev_score,
        "fail_index_f1": fail_idx_f1,
        "trend_match": trend,
        **key_metrics,
        "pred_len_words": len(tokenize(pred)),
    }


def mean(values):
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def evidence_information_accuracy(row):
    keys = [
        "status_acc",
        "risk_acc",
        "max_dev_acc",
        "fail_index_acc",
        "trend_acc",
    ]
    return mean([row.get(k) for k in keys])


def summarize_scores(path, records, scores, n_errors, subset="ALL"):
    row = {
        "file": path.name,
        "subset": subset,
        "variant": records[0].get("variant") if records else "",
        "setting": records[0].get("setting") if records else "",
        "n": len(records),
        "errors": n_errors,
        "bleu1": mean([s["bleu1"] for s in scores]),
        "bleu2": mean([s["bleu2"] for s in scores]),
        "bleu3": mean([s["bleu3"] for s in scores]),
        "bleu4": mean([s["bleu4"] for s in scores]),
        "bleu": mean([s["bleu"] for s in scores]),
        "rouge_l": mean([s["rouge_l"] for s in scores]),
        "status_acc": mean([s["status_match"] for s in scores]),
        "risk_acc": mean([s["risk_match"] for s in scores]),
        "risk_mae": mean([s["risk_abs_error"] for s in scores]),
        "risk_score": mean([s["risk_score"] for s in scores]),
        "max_dev_acc": mean([s["max_dev_match"] for s in scores]),
        "max_dev_mae": mean([s["max_dev_abs_error"] for s in scores]),
        "max_dev_score": mean([s["max_dev_score"] for s in scores]),
        "fail_index_acc": mean([s["fail_index_match"] for s in scores]),
        "fail_index_f1": mean([s["fail_index_f1"] for s in scores]),
        "trend_acc": mean([s["trend_match"] for s in scores]),
        "key_fact_precision": mean([s["key_fact_precision"] for s in scores]),
        "key_fact_recall": mean([s["key_fact_recall"] for s in scores]),
        "key_fact_f1": mean([s["key_fact_f1"] for s in scores]),
        "evidence_coverage": mean([s["evidence_coverage"] for s in scores]),
        "unsupported_claim_rate": mean([s["unsupported_claim_rate"] for s in scores]),
        "supported_claim_rate": mean([s["supported_claim_rate"] for s in scores]),
        "coverage_weighted_support": mean([s["coverage_weighted_support"] for s in scores]),
        "fact_score": mean([s["fact_score"] for s in scores]),
        "avg_pred_len": mean([s["pred_len_words"] for s in scores]),
    }
    row["eia"] = evidence_information_accuracy(row)
    return row


def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", default="results_clean/report_generation/generations", type=Path)
    ap.add_argument("--gen_files", nargs="*", default=None, type=Path,
                    help="Optional explicit jsonl files to evaluate instead of all files in gen_dir.")
    ap.add_argument("--out_dir", default="results_clean/report_generation/metrics", type=Path)
    ap.add_argument("--include_tests", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    detail_rows = []
    paths = args.gen_files if args.gen_files else sorted(args.gen_dir.glob("*.jsonl"))
    for path in paths:
        if not args.include_tests and path.name.startswith("test"):
            continue

        records = read_jsonl(path)
        scores = []
        scored_records = []
        n_errors = 0
        for rec in records:
            if rec.get("error"):
                n_errors += 1
                if not is_v2(rec.get('measurement') or {}):
                    continue
                rec = {**rec, 'prediction': ''}
            s = evaluate_record(rec)
            scores.append(s)
            scored_records.append((rec, s))
            detail_rows.append({
                "file": path.name,
                "subset": rec.get("subset"),
                "id": rec.get("id"),
                "variant": rec.get("variant"),
                "setting": rec.get("setting"),
                **s,
            })

        summary_rows.append(
            summarize_scores(path, records, scores, n_errors, subset="ALL")
        )

        subsets = sorted({str(rec.get("subset")) for rec, _ in scored_records})
        for subset in subsets:
            subset_pairs = [(rec, s) for rec, s in scored_records if str(rec.get("subset")) == subset]
            subset_records = [rec for rec, _ in subset_pairs]
            subset_scores = [s for _, s in subset_pairs]
            subset_errors = sum(
                1 for rec in records
                if str(rec.get("subset")) == subset and rec.get("error")
            )
            summary_rows.append(
                summarize_scores(
                    path, subset_records, subset_scores, subset_errors, subset=subset
                )
            )

    summary_csv = args.out_dir / "summary.csv"
    detail_csv = args.out_dir / "details.csv"

    fieldnames = [
        "file", "subset", "variant", "setting", "n", "errors",
        "bleu1", "bleu2", "bleu3", "bleu4", "bleu", "rouge_l",
        "status_acc", "risk_acc", "risk_mae", "risk_score",
        "max_dev_acc", "max_dev_mae", "max_dev_score",
        "fail_index_acc", "fail_index_f1",
        "trend_acc",
        "key_fact_precision", "key_fact_recall", "key_fact_f1",
        "evidence_coverage", "unsupported_claim_rate", "supported_claim_rate",
        "coverage_weighted_support",
        "fact_score", "eia", "avg_pred_len",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    if detail_rows:
        with open(detail_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
            w.writeheader()
            w.writerows(detail_rows)

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote {summary_csv}")
    print(f"Wrote {detail_csv}")
    for row in summary_rows:
        print(
            f"{row['file']} [{row['subset']}]: n={row['n']} "
            f"ROUGE-L={fmt(row['rouge_l'])} "
            f"BLEU-1={fmt(row['bleu1'])} BLEU-4={fmt(row['bleu4'])} "
            f"maxdev_mae={fmt(row['max_dev_mae'])} "
            f"failidx_f1={fmt(row['fail_index_f1'])} "
            f"keyfact_f1={fmt(row['key_fact_f1'])} "
            f"coverage={fmt(row['evidence_coverage'])} "
            f"unsupported={fmt(row['unsupported_claim_rate'])} "
            f"fact={fmt(row['fact_score'])}"
        )


if __name__ == "__main__":
    main()
