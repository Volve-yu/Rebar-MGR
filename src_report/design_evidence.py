"""Shared v2 report facts, prompts and deterministic reporting baseline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from inspection_data import SCHEMA, risk_level


def is_v2(m):
    return m.get('schema_version') == SCHEMA or m.get('compliance', {}).get('schema_version') == SCHEMA


def deviation(p):
    return p.get('deviation_from_design_mm', p.get('deviation_from_mean_mm', 0.0))


def is_failed(p):
    if 'within_design_tol' in p:
        return not p['within_design_tol']
    return abs(p.get('deviation_from_mean_mm', 0.0)) > 20.0


def rule_text(m):
    rule = m.get('compliance', {}).get('rule')
    if not rule:
        return 'No unambiguous design rule is available; do not claim compliance.'
    return (f"Region {rule['region_id']}: design spacing {rule['design_spacing_mm']:g} mm; "
            f"allowed signed deviation [{rule['lower_tolerance_mm']:g}, {rule['upper_tolerance_mm']:g}] mm. "
            'These are supplied inspection rules, not a verified statutory acceptance certificate.')


def format_measurement(m, include_rule=False):
    per = m.get('compliance', {}).get('per_spacing', [])
    lines = [f"Detected rebar count: {m.get('rebar_count')}",
             f"Mean spacing: {m.get('mean_spacing_mm')} mm; standard deviation: {m.get('std_spacing_mm')} mm.",
             'Predicted spacings (mm), left to right: '+', '.join(f'{v:.2f}' for v in m.get('spacings_mm', [])),
             f"Compliance result: {m.get('compliance', {}).get('status', 'N/A')}; risk level: {risk_level(m)}."]
    if per:
        lines.append(f"Maximum deviation from design: {max(abs(deviation(p)) for p in per):.2f} mm.")
        lines.append('Failed intervals (zero-based): '+('; '.join(f"spacing[{p['index']}]={p['spacing_mm']:.2f} mm (deviation={deviation(p):+.2f} mm)" for p in per if is_failed(p)) or 'none'))
    if include_rule:
        lines.append(rule_text(m))
    if m.get('warning'):
        lines.append(str(m['warning']))
    return '\n'.join(lines)


def template_report(m):
    per = m.get('compliance', {}).get('per_spacing', [])
    mean, std = m.get('mean_spacing_mm'), m.get('std_spacing_mm')
    if mean is None or not per:
        return f"Detected {m.get('rebar_count', 0)} rebars. Spacing assessment is incomplete. Result: N/A. Further measurement is required."
    failed = [p for p in per if is_failed(p)]
    detail = '; '.join(f"spacing[{p['index']}]={p['spacing_mm']:.2f} mm (deviation={deviation(p):+.2f} mm)" for p in failed) or 'none'
    return (f"Detected {m['rebar_count']} rebars with mean spacing {mean:.2f} mm and standard deviation {std:.2f} mm. "
            f"The maximum deviation from design is {max(abs(deviation(p)) for p in per):.2f} mm; failed intervals: {detail}. "
            f"Risk level: {risk_level(m)}; result: {m['compliance']['status']}. "
            'This assessment is relative to the supplied design and tolerance settings.')
