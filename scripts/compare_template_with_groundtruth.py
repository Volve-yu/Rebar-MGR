"""Compare numerical template and archived Full reports against explicit groundtruth."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / 'src_report'))
from design_evidence import deviation, is_failed, rule_text, risk_level


def load(path):
    records = {}
    text = path.read_text(encoding='utf-8-sig')
    rows = json.loads(text) if text.lstrip().startswith('[') else [json.loads(line) for line in text.splitlines() if line.strip()]
    for row in rows:
        key = (str(row['subset']), str(row['id']))
        if key in records:
            raise ValueError(f'Duplicate sample {key} in {path}')
        records[key] = row
    return records


def numeric_template(m):
    """Render available numerical evidence; no image analysis or QAEP narrative."""
    parts = []
    if m.get('rebar_count') is not None:
        parts.append(f"Detected {m['rebar_count']} rebars.")
    values = m.get('spacings_mm') or []
    if values:
        parts.append('Measured spacings, left to right: ' + ', '.join(f'{v:.2f} mm' for v in values) + '.')
    if m.get('mean_spacing_mm') is not None:
        parts.append(f"Mean spacing: {m['mean_spacing_mm']:.2f} mm.")
    if m.get('std_spacing_mm') is not None:
        parts.append(f"Standard deviation: {m['std_spacing_mm']:.2f} mm.")
    compliance = m.get('compliance') or {}
    if compliance.get('rule'):
        parts.append(rule_text(m))
    per = compliance.get('per_spacing') or []
    if per:
        parts.append(f"Maximum deviation from design: {max(abs(deviation(p)) for p in per):.2f} mm.")
        failed = [p for p in per if is_failed(p)]
        parts.append('Failed intervals: ' + ('; '.join(
            f"spacing[{p['index']}]={p['spacing_mm']:.2f} mm" for p in failed
        ) or 'none') + '.')
    if compliance.get('rule') and per:
        parts.append(f"Risk level: {risk_level(m)}.")
    parts.append(f"Result: {compliance.get('status', 'N/A')}.")
    return ' '.join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--full-jsonl', nargs='+', type=Path, required=True)
    parser.add_argument('--groundtruth-jsonl', nargs='+', type=Path, required=True,
                        help='Reviewed records with subset, id, reference_report and measurement (groundtruth facts).')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--groundtruth-facts', nargs='+', type=Path,
                        help='Optional separate groundtruth measurement records, matched by subset and id.')
    parser.add_argument('--shared-only', action='store_true',
                        help='Evaluate only samples present in both Full and complete groundtruth.')
    args = parser.parse_args()
    truth = {}
    for path in args.groundtruth_jsonl:
        batch = load(path)
        if truth.keys() & batch.keys():
            raise ValueError('Duplicate groundtruth sample across files')
        truth.update(batch)
    if args.groundtruth_facts:
        facts = {}
        for path in args.groundtruth_facts:
            batch = load(path)
            if facts.keys() & batch.keys():
                raise ValueError('Duplicate groundtruth facts across files')
            facts.update(batch)
        truth = {key: dict(row, measurement=facts[key]['measurement']) for key,row in truth.items()
                 if key in facts and facts[key].get('measurement')}
    outputs = {'numeric_template': [], 'full_report': []}
    seen = set()
    for path in args.full_jsonl:
        for key, full in load(path).items():
            if key in seen:
                raise ValueError(f'Duplicate Full sample across files: {key}')
            seen.add(key)
            if full.get('error') or not str(full.get('prediction') or '').strip():
                raise ValueError(f'Full generation incomplete: {key}')
            gt = truth.get(key)
            if not gt or not str(gt.get('reference_report') or '').strip() or not gt.get('measurement'):
                if args.shared_only:
                    continue
                raise ValueError(f'Missing groundtruth report or structured groundtruth facts: {key}')
            predicted = full['measurement']
            for method in outputs:
                row = copy.deepcopy(full)
                row['setting'] = method
                row['reference_report'] = gt['reference_report']
                row['input_measurement'] = predicted
                # The evaluator derives factual targets from measurement, not report text.
                row['measurement'] = gt['measurement']
                row['prediction'] = numeric_template(predicted) if method == 'numeric_template' else full['prediction']
                row['evaluation_protocol'] = 'shared_reviewed_groundtruth'
                outputs[method].append(row)
    if not outputs['full_report']:
        raise ValueError('No samples with complete Full reports and groundtruth')
    if all(r['prediction'].strip() == r['reference_report'].strip() for r in outputs['numeric_template']):
        raise ValueError('All template outputs equal groundtruth: check for same-renderer self-comparison')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for method, rows in outputs.items():
        path = args.out_dir / f'{method}.jsonl'
        if path.exists():
            raise FileExistsError(f'Use a new output folder; refusing to overwrite {path}')
        files.append(path)
    for path, rows in zip(files, outputs.values()):
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    print(f"Matched {len(outputs['full_report'])} samples; excluded {len(seen)-len(outputs['full_report'])} without complete groundtruth.", flush=True)
    subprocess.run([sys.executable, str(CODE / 'src_report/eval.py'), '--gen_files',
                    *map(str, files), '--out_dir', str(args.out_dir / 'metrics')], check=True)
    print('Numerical fact metrics do not score visual-distribution descriptions; assess those separately.')


if __name__ == '__main__':
    main()
