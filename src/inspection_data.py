"""Validated physical references and design-based inspection rules (schema v2)."""
import csv
import hashlib
import math
from collections import defaultdict
from decimal import Decimal
from pathlib import Path

SCHEMA = 'design_physical_v2'


class AnnotationStore:
    def __init__(self, path):
        self.path = Path(path)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.samples = defaultdict(list)
        with self.path.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        seen = set()
        for row in rows:
            key = (row['sample_key'], row['interval_id'])
            if key in seen:
                raise ValueError('Duplicate interval: '+str(key))
            seen.add(key)
            if row.get('value_source') != 'user_reported_measurement' or row.get('review_status') != 'user_verified':
                raise ValueError('Unverified physical reference: '+str(key))
            if not row['region_id'].strip():
                raise ValueError('Missing region: '+str(key))
            for field in ('submitted_spacing_mm', 'design_spacing_mm', 'lower_tolerance_mm', 'upper_tolerance_mm'):
                value = float(row[field])
                if not math.isfinite(value):
                    raise ValueError('Nonfinite '+field+': '+str(key))
                row[field] = value
            if Decimal(str(row['submitted_spacing_mm'])) != Decimal(row['working_spacing_cm'])*10:
                raise ValueError('Unit mismatch: '+str(key))
            if row['submitted_spacing_mm'] <= 0 or row['design_spacing_mm'] <= 0:
                raise ValueError('Nonpositive spacing: '+str(key))
            if not row['lower_tolerance_mm'] <= 0 <= row['upper_tolerance_mm']:
                raise ValueError('Invalid signed tolerances: '+str(key))
            self.samples[row['sample_key']].append(row)
        if not rows:
            raise ValueError('Empty annotation table')
        for key, intervals in self.samples.items():
            intervals.sort(key=lambda row: int(row['interval_id'][1:]))
            for i, row in enumerate(intervals, 1):
                if row['interval_id'] != 'd'+str(i) or int(row['left_bar']) != i or int(row['right_bar']) != i+1:
                    raise ValueError('Nonconsecutive interval identifiers: '+key)

    def rows(self, subset, sid):
        key = subset+'/'+str(sid)
        if key not in self.samples:
            raise ValueError('Missing annotation: '+key)
        return self.samples[key]

    def spacings(self, subset, sid):
        return [r['submitted_spacing_mm'] for r in self.rows(subset, sid)]

    def rules(self, subset, sid):
        # Deliberately omit readings, reference coordinates and reference counts.
        fields = ('region_id', 'design_spacing_mm', 'lower_tolerance_mm', 'upper_tolerance_mm')
        return [dict(zip(fields, values)) for values in sorted({tuple(r[f] for f in fields) for r in self.rows(subset, sid)})]


def design_compliance(spacings, rules):
    base = dict(schema_version=SCHEMA, assessment_basis='user_confirmed_design_rules',
                index_convention='zero_based_prediction_order', per_spacing=[])
    if not spacings or not rules or len(rules) != 1:
        return dict(base, status='N/A', reason='missing spacing/rule or mixed regions require independent region localization')
    rule = rules[0]
    design = float(rule['design_spacing_mm'])
    lo, hi = float(rule['lower_tolerance_mm']), float(rule['upper_tolerance_mm'])
    if not all(math.isfinite(v) for v in (design, lo, hi)) or design <= 0 or not lo <= 0 <= hi:
        raise ValueError('Invalid design rule')
    per = []
    for i, value in enumerate(spacings):
        if not math.isfinite(value) or value <= 0:
            return dict(base, status='N/A', reason='invalid predicted spacing')
        delta = value-design
        passed = lo-1e-9 <= delta <= hi+1e-9
        per.append(dict(index=i, interval_id='d'+str(i+1), spacing_mm=value,
                        deviation_from_design_mm=delta, within_design_tol=passed, **rule))
    return dict(base, status='PASS' if all(p['within_design_tol'] for p in per) else 'FAIL',
                per_spacing=per, rule=rule)


def risk_level(measurement):
    """Experimental severity grading; no claim of a code-prescribed risk scale."""
    comp = measurement.get('compliance', {})
    if comp.get('schema_version') != SCHEMA or comp.get('status') == 'N/A':
        return 'UNKNOWN'
    per = comp.get('per_spacing', [])
    if not per:
        return 'UNKNOWN'
    maximum = max(abs(p['deviation_from_design_mm']) for p in per)
    ratio = sum(not p['within_design_tol'] for p in per)/len(per)
    if ratio == 0:
        return 'SAFE' if maximum <= 5 else 'ACCEPTABLE'
    if ratio <= .25 and maximum <= 40:
        return 'MARGINAL'
    if ratio <= .5 or maximum <= 60:
        return 'WARNING'
    return 'CRITICAL'


def measurement_record(spacings, count, rules, **metadata):
    mean = sum(spacings)/len(spacings) if spacings else None
    std = (sum((x-mean)**2 for x in spacings)/len(spacings))**.5 if spacings else None
    return dict(schema_version=SCHEMA, spacings_mm=spacings, rebar_count=count,
                mean_spacing_mm=mean, std_spacing_mm=std, mean_diameter_mm=None,
                compliance=design_compliance(spacings, rules), **metadata)


def matched_intervals(gt_xs, pred_xs, mpp, tolerance_mm):
    """Monotone one-to-one center matches; only consecutive matched pairs yield gaps."""
    i = j = 0
    pairs = []
    while i < len(gt_xs) and j < len(pred_xs):
        delta = (pred_xs[j]-gt_xs[i])*mpp
        if abs(delta) <= tolerance_mm:
            pairs.append((i, j)); i += 1; j += 1
        elif delta < 0:
            j += 1
        else:
            i += 1
    gaps = [(a, b) for (a, b), (c, d) in zip(pairs, pairs[1:]) if c == a+1 and d == b+1]
    return pairs, gaps
