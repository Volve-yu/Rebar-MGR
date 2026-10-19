"""Local, frozen-prediction agreement against user-verified spacing annotations."""
import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = ROOT.parent
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from inspection_data import AnnotationStore, matched_intervals
from audit_measurement_wholeset import PeakConfig, extract_centers
from run_physical_inspection import load_samples

STANDARD_URL = 'https://gf.cabr-fire.com/article-12547.htm'
SCENARIOS = {'region_rules': None, 'exploratory_pm10': 10.0, 'exploratory_pm20': 20.0}


def read_jsonl(path):
    records = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    result = {(r['subset'], str(r['id'])): r for r in records}
    if len(result) != len(records):
        raise ValueError('Duplicate record: ' + str(path))
    return result


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision(value, rule):
    if value is None or not np.isfinite(value) or value <= 0:
        return 'N/A'
    delta = value - rule['design_spacing_mm']
    return 'PASS' if rule['lower_tolerance_mm'] - 1e-9 <= delta <= rule['upper_tolerance_mm'] + 1e-9 else 'FAIL'


def image_decision(values, rules):
    if not values or len(rules) != 1:
        return 'N/A'
    states = [decision(v, rules[0]) for v in values]
    if 'N/A' in states:
        return 'N/A'
    return 'FAIL' if 'FAIL' in states else 'PASS'


def counts(rows):
    # FAIL is the positive class. The final two bins preserve abstentions.
    keys = [('FAIL', 'FAIL'), ('PASS', 'FAIL'), ('PASS', 'PASS'),
            ('FAIL', 'PASS'), ('FAIL', 'N/A'), ('PASS', 'N/A')]
    return np.array([sum((r['reference_status'], r['prediction_status']) == key for r in rows)
                     for key in keys], dtype=float)


def divide(a, b):
    return float(a / b) if b else None


def classification(c):
    c = np.asarray(c, dtype=float)
    tp, fp, tn, fn, uf, up = c
    evaluated = tp + fp + tn + fn
    sensitivity, specificity = divide(tp, tp + fn), divide(tn, tn + fp)
    accuracy = divide(tp + tn, evaluated)
    chance = divide((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn), evaluated ** 2)
    return dict(n=int(c.sum()), evaluated_n=int(evaluated), unavailable_n=int(uf + up),
                reference_fail_n=int(tp + fn + uf), reference_pass_n=int(tn + fp + up),
                tp_fail=int(tp), fp_false_alarm=int(fp), tn_pass=int(tn), fn_false_pass=int(fn),
                reference_fail_unavailable_n=int(uf), reference_pass_unavailable_n=int(up),
                agreement=accuracy, agreement_over_all=divide(tp + tn, c.sum()),
                fail_recall=sensitivity, pass_specificity=specificity,
                false_pass_rate=divide(fn, tp + fn), false_alarm_rate=divide(fp, tn + fp),
                fail_detection_over_all_reference_fail=divide(tp, tp + fn + uf),
                balanced_accuracy=(sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
                fail_f1=divide(2 * tp, 2 * tp + fp + fn),
                kappa=divide(accuracy - chance, 1 - chance) if chance is not None else None)


def grouped_metrics(rows, image_keys, bootstrap, seed):
    grouped = {key: [] for key in image_keys}
    for row in rows:
        grouped[row['sample_key']].append(row)
    matrix = np.array([counts(grouped[k]) for k in image_keys])
    result = classification(matrix.sum(axis=0))
    groups = [[i for i, k in enumerate(image_keys) if k.startswith(site + '/')]
              for site in sorted({k.split('/')[0] for k in image_keys})]
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(bootstrap):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in groups])
        draws.append(classification(matrix[idx].sum(axis=0)))
    result['image_cluster_bootstrap_95ci'] = {}
    for name in ('agreement', 'agreement_over_all', 'false_pass_rate', 'false_alarm_rate', 'balanced_accuracy', 'kappa'):
        values = [d[name] for d in draws if d[name] is not None]
        result['image_cluster_bootstrap_95ci'][name] = {
            'bounds': np.percentile(values, [2.5, 97.5]).tolist() if values else None,
            'valid_resamples': len(values)}
    return result


def measurement_metrics(rows):
    errors = np.array([r['prediction_mm'] - r['reference_mm'] for r in rows
                       if r['prediction_mm'] is not None], dtype=float)
    return dict(matched_n=len(errors), reference_n=len(rows), coverage=divide(len(errors), len(rows)),
                mae_mm=float(np.abs(errors).mean()) if len(errors) else None,
                rmse_mm=float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
                bias_mm=float(errors.mean()) if len(errors) else None,
                p95_absolute_error_mm=float(np.percentile(np.abs(errors), 95)) if len(errors) else None,
                descriptive_error_thresholds={str(t): float(np.mean(np.abs(errors) <= t)) if len(errors) else None
                                              for t in (2, 5, 10)})


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs-root', type=Path, default=ROOT / 'runs/revision_full_seed42')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--annotations', type=Path, default=ROOT / 'annotation_all_images/all_images_annotations.csv')
    p.add_argument('--out-dir', type=Path)
    p.add_argument('--bootstrap', type=int, default=1000)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    if args.bootstrap < 0:
        p.error('--bootstrap must be nonnegative')
    out = args.out_dir or ROOT / 'results' / ('screening_agreement_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    if out.exists():
        raise FileExistsError('Use a new output directory: ' + str(out))
    store = AnnotationStore(args.annotations)
    intervals, images, sources = [], [], {}
    for site in ('site-1', 'site-2'):
        run = args.runs_root / (site + '_full_seed42')
        within = run / 'within'
        manifest = json.loads((within / 'manifest.json').read_text(encoding='utf-8'))
        if manifest['annotation_sha256'] != store.sha256 or manifest['split'] != 'val' or manifest['protocol'] != 'within_site':
            raise ValueError('Annotation/protocol mismatch: ' + site)
        split_path = args.data_root / site / 'gt_sets/val.txt'
        if sha(split_path) != manifest['split_sha256'][site]:
            raise ValueError('Validation split changed: ' + site)
        saved = read_jsonl(within / 'report_inputs.jsonl')
        refs = read_jsonl(within / 'physical_references.jsonl')
        config = SimpleNamespace(subsets=[site], split='val', data_root=args.data_root,
                                 pred_root=run / 'predictions/within', model='model', hfov=manifest['hfov'])
        samples = load_samples(config, store)
        expected = {(s['subset'], s['sid']) for s in samples}
        if set(saved) != expected or set(refs) != expected:
            raise ValueError('Missing or extra validation records: ' + site)
        sources[site] = dict(manifest=manifest, files={str(path): sha(path) for path in
                            (within / 'manifest.json', within / 'report_inputs.jsonl', within / 'physical_references.jsonl', split_path)},
                            mask_sha256={s['sid']: sha(Path(s['seg_path'])) for s in samples})
        for sample in samples:
            sid = sample['sid']
            key = site + '/' + sid
            pred_xs = extract_centers(sample['mask'], PeakConfig(**manifest['peak_config']))
            pred = [(b - a) * sample['mpp'] for a, b in zip(pred_xs, pred_xs[1:])]
            old = saved[(site, sid)]['measurement']
            if len(pred) != len(old['spacings_mm']) or not np.allclose(pred, old['spacings_mm'], rtol=0, atol=1e-6):
                raise ValueError('Frozen predictions not reproduced: ' + key)
            truth = sample['truth']
            old_truth = refs[(site, sid)]['measurement']['spacings_mm']
            if len(truth) != len(old_truth) or not np.allclose(truth, old_truth, rtol=0, atol=1e-9):
                raise ValueError('Physical reference mismatch: ' + key)
            if len(sample['rules']) != 1 or sample['rules'][0] != old['compliance'].get('rule'):
                raise ValueError('Mixed or changed region rule: ' + key)
            _, gaps = matched_intervals(sample['xs'], pred_xs, sample['mpp'], manifest['center_tol_mm'])
            mapping = dict(gaps)
            for scenario, tolerance in SCENARIOS.items():
                rule = dict(sample['rules'][0])
                if tolerance is not None:
                    rule.update(lower_tolerance_mm=-tolerance, upper_tolerance_mm=tolerance)
                common = dict(scenario=scenario, site=site, sample_key=key, **rule)
                images.append(dict(common, reference_status=image_decision(truth, [rule]),
                                   prediction_status=image_decision(pred, [rule]), reference_intervals=len(truth),
                                   predicted_intervals=len(pred), matched_intervals=len(gaps),
                                   unmatched_reference_intervals=len(truth) - len(gaps),
                                   unmatched_prediction_intervals=len(pred) - len(gaps)))
                for i, value in enumerate(truth):
                    j = mapping.get(i)
                    estimate = pred[j] if j is not None else None
                    boundary = min(abs(value - rule['design_spacing_mm'] - rule['lower_tolerance_mm']),
                                   abs(value - rule['design_spacing_mm'] - rule['upper_tolerance_mm']))
                    intervals.append(dict(common, reference_interval='d' + str(i + 1),
                                          predicted_interval='d' + str(j + 1) if j is not None else '',
                                          reference_mm=value, prediction_mm=estimate,
                                          boundary_distance_mm=boundary,
                                          reference_status=decision(value, rule), prediction_status=decision(estimate, rule)))
        print(site + ': checked ' + str(len(samples)) + ' validation images', flush=True)
    result = {}
    for scenario in SCENARIOS:
        result[scenario] = {}
        for site in ('site-1', 'site-2', 'pooled'):
            selected = [r for r in intervals if r['scenario'] == scenario and (site == 'pooled' or r['site'] == site)]
            selected_images = [r for r in images if r['scenario'] == scenario and (site == 'pooled' or r['site'] == site)]
            keys = [r['sample_key'] for r in selected_images]
            result[scenario][site] = dict(
                interval=grouped_metrics(selected, keys, args.bootstrap, args.seed),
                image=grouped_metrics(selected_images, keys, args.bootstrap, args.seed),
                measurement=measurement_metrics(selected),
                unmatched_prediction_intervals=sum(r['unmatched_prediction_intervals'] for r in selected_images),
                near_boundary_le5mm=classification(counts([r for r in selected if r['boundary_distance_mm'] <= 5])),
                away_from_boundary_gt5mm=classification(counts([r for r in selected if r['boundary_distance_mm'] > 5])))
    out.mkdir(parents=True)
    write_csv(out / 'interval_details.csv', intervals)
    write_csv(out / 'image_details.csv', images)
    metadata = dict(standard='GB 50204-2015, Table 5.5.3', standard_url=STANDARD_URL,
                    primary='region_rules', exploratory='pm10/pm20 do not establish the applicable rebar category',
                    reference_provenance='user-reported measurements, user-verified; no independent provenance audit',
                    image_rule='all predicted/annotated intervals within bounds; research screening, not code batch acceptance',
                    matching='original frozen monotone center matching; adjacent pairs only; no index-only pairing',
                    unavailable='unmatched reference gaps remain N/A; conditional metrics and coverage both reported',
                    bootstrap=dict(unit='image', stratified_by='site', repeats=args.bootstrap, seed=args.seed),
                    annotations_sha256=store.sha256, sources=sources, script_sha256=sha(Path(__file__)))
    for name, data in (('summary.json', result), ('manifest.json', metadata)):
        (out / name).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    lines = ['# Screening agreement', '', 'Primary: existing region rules. Other tolerances are exploratory.',
             'FAIL is positive; interval agreement is conditional on matched gaps. Always report coverage.',
             'Image screening is not a statutory batch acceptance procedure.', '',
             '| Scenario | Site | Images | Interval coverage | Interval agreement | False-pass rate | Image agreement | GT PASS images |',
             '|---|---|---:|---:|---:|---:|---:|---:|']
    def pct(value):
        return 'N/A' if value is None else f'{100 * value:.2f}%'
    for scenario, sites in result.items():
        for site, m in sites.items():
            lines.append(f"| {scenario} | {site} | {m['image']['n']} | {pct(m['measurement']['coverage'])} | "
                         f"{pct(m['interval']['agreement'])} | {pct(m['interval']['false_pass_rate'])} | "
                         f"{pct(m['image']['agreement'])} | {m['image']['reference_pass_n']} |")
    lines += ['', 'If reference PASS images are absent, image agreement cannot establish PASS/FAIL discrimination.',
              'No tolerance, matching parameter or model was selected using these agreement results.',
              'See summary.json for class counts, missing outputs, confidence intervals and error distributions.']
    (out / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))
    print('\nOutput: ' + str(out))


if __name__ == '__main__':
    main()
