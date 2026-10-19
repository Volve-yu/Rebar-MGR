"""
Report generation module: LLM-powered GB 50204 compliance report.

Takes MeasurementResult from measure.py and generates a structured
natural language compliance report via the Anthropic API.

Falls back to a template-based report if the API key is unavailable.

Usage (standalone):
    python src/report.py \
        --csv measure_output/results.csv \
        --out_dir reports

    python src/report.py \
        --csv measure_output/results.csv \
        --out_dir reports \
        --lang zh \
        --model claude-haiku-4-5

Usage (from Python):
    from report import generate_report, generate_batch_report
    from measure import MeasurementResult
    text = generate_report(result, lang="zh")
"""

import argparse
import csv
import os
import sys
import textwrap
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))
from measure import MeasurementResult, RulerConfig


# ---------------------------------------------------------------------------
# Prompt / template helpers
# ---------------------------------------------------------------------------

def _format_context_zh(r: MeasurementResult) -> str:
    """Build Chinese-language measurement context block."""
    lines = [
        f"图像编号: {r.image_id}",
        f"拍摄距离: {r.distance_m} m",
        f"像素换算比: {r.mpp} mm/px",
        f"检测钢筋根数: {r.rebar_count}",
    ]
    if r.mean_spacing_mm is not None:
        lines.append(f"平均间距: {r.mean_spacing_mm} mm")
        lines.append(f"间距标准差: {r.std_spacing_mm} mm")
        lines.append(f"各间距 (mm): {r.spacings_mm}")
    if r.mean_diameter_mm is not None:
        lines.append(f"平均直径: {r.mean_diameter_mm} mm")
        lines.append(f"各直径 (mm): {r.diameters_mm}")

    comp = r.compliance
    lines.append(f"GB 50204 合规状态: {comp.get('status', 'N/A')}")
    if "per_spacing" in comp:
        lines.append("逐间距合规详情:")
        for s in comp["per_spacing"]:
            flag = "合格" if s["within_stirrup_tol"] else (
                "主筋合格" if s["within_main_tol"] else "超差")
            lines.append(
                f"  间距[{s['index']}]: {s['spacing_mm']} mm  "
                f"偏差={s['deviation_from_mean_mm']:+.1f} mm  [{flag}]"
            )
    if r.warning:
        lines.append(f"警告: {r.warning}")
    return "\n".join(lines)


def _format_context_en(r: MeasurementResult) -> str:
    """Build English-language measurement context block."""
    lines = [
        f"Image ID: {r.image_id}",
        f"Shooting distance: {r.distance_m} m",
        f"Scale factor: {r.mpp} mm/px",
        f"Rebar count: {r.rebar_count}",
    ]
    if r.mean_spacing_mm is not None:
        lines.append(f"Mean spacing: {r.mean_spacing_mm} mm")
        lines.append(f"Spacing std dev: {r.std_spacing_mm} mm")
        lines.append(f"Individual spacings (mm): {r.spacings_mm}")
    if r.mean_diameter_mm is not None:
        lines.append(f"Mean diameter: {r.mean_diameter_mm} mm")
        lines.append(f"Individual diameters (mm): {r.diameters_mm}")

    comp = r.compliance
    lines.append(f"GB 50204 compliance: {comp.get('status', 'N/A')}")
    if "per_spacing" in comp:
        lines.append("Per-spacing compliance:")
        for s in comp["per_spacing"]:
            flag = "OK" if s["within_stirrup_tol"] else (
                "OK(main)" if s["within_main_tol"] else "FAIL")
            lines.append(
                f"  spacing[{s['index']}]: {s['spacing_mm']} mm  "
                f"dev={s['deviation_from_mean_mm']:+.1f} mm  [{flag}]"
            )
    if r.warning:
        lines.append(f"Warning: {r.warning}")
    return "\n".join(lines)


_SYSTEM_ZH = textwrap.dedent("""\
    你是一名专业的建筑工程质量检测员，熟悉 GB 50204-2015《混凝土结构工程施工质量验收规范》。
    你的任务是根据提供的钢筋间距和直径自动检测数据，撰写一份简洁专业的单图检测报告。

    报告要求：
    1. 语言简洁专业，使用工程术语
    2. 必须包含以下章节：检测概述、测量结果、合规性判定、处置建议
    3. 合规性判定需明确引用 GB 50204-2015 条款（箍筋间距允许偏差 ±10mm，主筋 ±20mm）
    4. 如存在超差，需指出具体哪个间距超差及偏差量
    5. 处置建议针对检测结论给出，合格时写"本区域钢筋安装质量符合规范要求，可进行下道工序"
    6. 不要重复原始数据表格，用文字描述关键指标即可
    7. 报告总长度控制在 300 字以内
""")

_SYSTEM_EN = textwrap.dedent("""\
    You are a professional construction quality inspector familiar with GB 50204-2015
    (Code for Acceptance of Constructional Quality of Concrete Structures).
    Write a concise, professional single-image inspection report based on the
    rebar spacing and diameter measurements provided.

    Requirements:
    1. Use professional engineering terminology
    2. Include exactly these sections: Inspection Overview, Measurements, Compliance Assessment, Recommendations
    3. Reference GB 50204-2015 explicitly (stirrup spacing tolerance ±10 mm, main bar ±20 mm)
    4. For any out-of-tolerance spacings, identify the specific index and deviation
    5. If compliant, state: "Rebar installation in this area meets GB 50204-2015 requirements"
    6. Do not repeat raw data tables; describe key metrics in prose
    7. Keep the report under 250 words
""")


def _build_prompt_zh(r: MeasurementResult) -> str:
    return f"请根据以下自动检测数据，撰写钢筋安装质量检测报告：\n\n{_format_context_zh(r)}"


def _build_prompt_en(r: MeasurementResult) -> str:
    return f"Write a rebar installation quality report based on the following measurement data:\n\n{_format_context_en(r)}"


# ---------------------------------------------------------------------------
# Template fallback (no API key required)
# ---------------------------------------------------------------------------

def _template_report_zh(r: MeasurementResult) -> str:
    today = date.today().strftime("%Y-%m-%d")
    status = r.compliance.get("status", "N/A")

    lines = [
        "=" * 60,
        "钢筋安装质量检测报告",
        f"检测日期：{today}",
        f"图像编号：{r.image_id}",
        "=" * 60,
        "",
        "【检测概述】",
        f"拍摄距离 {r.distance_m} m，像素换算比 {r.mpp} mm/px，",
        f"本图共检测到钢筋 {r.rebar_count} 根。",
        "",
        "【测量结果】",
    ]
    if r.mean_spacing_mm is not None:
        lines += [
            f"平均间距：{r.mean_spacing_mm} mm，标准差：{r.std_spacing_mm} mm",
            f"各间距（mm）：{r.spacings_mm}",
        ]
    if r.mean_diameter_mm is not None:
        lines.append(f"平均直径：{r.mean_diameter_mm} mm")

    lines += ["", "【合规性判定（GB 50204-2015）】"]
    if status == "PASS":
        lines.append("所有间距偏差均在箍筋允许偏差 ±10 mm 以内，判定：合格（PASS）。")
    elif status == "PASS(main)":
        lines.append("部分间距超过箍筋偏差 ±10 mm，但均在主筋偏差 ±20 mm 以内，判定：主筋合格（PASS(main)）。")
    elif status == "FAIL":
        fail_items = [
            s for s in r.compliance.get("per_spacing", [])
            if not s["within_main_tol"]
        ]
        lines.append(f"以下间距超出主筋允许偏差 ±20 mm，判定：不合格（FAIL）。")
        for s in fail_items:
            lines.append(
                f"  间距[{s['index']}]：{s['spacing_mm']} mm，"
                f"偏差 {s['deviation_from_mean_mm']:+.1f} mm"
            )
    else:
        lines.append(f"无法判定：{r.compliance.get('reason', '数据不足')}。")

    lines += ["", "【处置建议】"]
    if status == "PASS":
        lines.append("本区域钢筋安装质量符合规范要求，可进行下道工序。")
    elif status == "PASS(main)":
        lines.append("建议对超出箍筋偏差的间距进行复核，确认钢筋类型后再行判定。")
    elif status == "FAIL":
        lines.append("不合格间距需整改后重新检测，整改前不得进行混凝土浇筑。")
    else:
        lines.append(f"检测数据不足，建议重新拍摄。{r.warning or ''}")

    lines += ["", "=" * 60]
    return "\n".join(lines)


def _template_report_en(r: MeasurementResult) -> str:
    today = date.today().strftime("%Y-%m-%d")
    status = r.compliance.get("status", "N/A")

    lines = [
        "=" * 60,
        "Rebar Installation Quality Inspection Report",
        f"Date: {today}",
        f"Image ID: {r.image_id}",
        "=" * 60,
        "",
        "[Inspection Overview]",
        f"Shooting distance {r.distance_m} m, scale {r.mpp} mm/px, "
        f"{r.rebar_count} rebar instances detected.",
        "",
        "[Measurements]",
    ]
    if r.mean_spacing_mm is not None:
        lines += [
            f"Mean spacing: {r.mean_spacing_mm} mm (std: {r.std_spacing_mm} mm)",
            f"Individual spacings (mm): {r.spacings_mm}",
        ]
    if r.mean_diameter_mm is not None:
        lines.append(f"Mean diameter: {r.mean_diameter_mm} mm")

    lines += ["", "[Compliance Assessment — GB 50204-2015]"]
    if status == "PASS":
        lines.append("All spacings within stirrup tolerance ±10 mm. Result: PASS.")
    elif status == "PASS(main)":
        lines.append("Some spacings exceed stirrup tolerance (±10 mm) but are within main bar tolerance (±20 mm). Result: PASS(main).")
    elif status == "FAIL":
        fail_items = [
            s for s in r.compliance.get("per_spacing", [])
            if not s["within_main_tol"]
        ]
        lines.append("The following spacings exceed main bar tolerance ±20 mm. Result: FAIL.")
        for s in fail_items:
            lines.append(
                f"  spacing[{s['index']}]: {s['spacing_mm']} mm, "
                f"deviation {s['deviation_from_mean_mm']:+.1f} mm"
            )
    else:
        lines.append(f"Cannot assess: {r.compliance.get('reason', 'insufficient data')}.")

    lines += ["", "[Recommendations]"]
    if status == "PASS":
        lines.append("Rebar installation in this area meets GB 50204-2015 requirements. Proceed to next construction stage.")
    elif status == "PASS(main)":
        lines.append("Verify rebar type for spacings exceeding stirrup tolerance before final acceptance.")
    elif status == "FAIL":
        lines.append("Non-compliant spacings must be corrected and re-inspected before concrete placement.")
    else:
        lines.append(f"Insufficient data. Retake image. {r.warning or ''}")

    lines += ["", "=" * 60]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM-based report generation
# ---------------------------------------------------------------------------

def generate_report(
    result: MeasurementResult,
    lang: str = "en",
    model: str = "claude-haiku-4-5",
    api_key: Optional[str] = None,
    use_llm: bool = True,
) -> str:
    """
    Generate a compliance report for one MeasurementResult.

    lang    : "zh" (Chinese) or "en" (English)
    model   : Anthropic model ID
    api_key : Anthropic API key (falls back to ANTHROPIC_API_KEY env var)
    use_llm : set False to skip LLM and use template directly
    """
    if not use_llm:
        return (_template_report_zh if lang == "zh" else _template_report_en)(result)

    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        print("[report] ANTHROPIC_API_KEY not set, using template fallback")
        return (_template_report_zh if lang == "zh" else _template_report_en)(result)

    try:
        import anthropic
    except ImportError:
        print("[report] anthropic package not installed (pip install anthropic), using template fallback")
        return (_template_report_zh if lang == "zh" else _template_report_en)(result)

    system = _SYSTEM_ZH if lang == "zh" else _SYSTEM_EN
    prompt = _build_prompt_zh(result) if lang == "zh" else _build_prompt_en(result)

    client = anthropic.Anthropic(api_key=key)
    message = client.messages.create(
        model=model,
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


# ---------------------------------------------------------------------------
# Batch report from CSV
# ---------------------------------------------------------------------------

@dataclass
class _CsvRow:
    """Minimal reconstruction of MeasurementResult from results.csv columns."""
    image_id: str
    distance_m: float
    mpp: float
    rebar_count: int
    mean_spacing_mm: Optional[float]
    std_spacing_mm: Optional[float]
    mean_diameter_mm: Optional[float]
    compliance_status: str
    warning: str


def _csv_to_result(row: Dict[str, str]) -> MeasurementResult:
    """Reconstruct a MeasurementResult from a results.csv row (no per-spacing detail)."""
    def _f(v): return float(v) if v not in ("", "None", "nan") else None
    def _i(v): return int(v) if v not in ("", "None") else 0

    comp_status = row.get("compliance", "N/A")
    compliance: Dict = {"status": comp_status}

    return MeasurementResult(
        image_id=row["image_id"],
        distance_m=_f(row["distance_m"]) or float("nan"),
        mpp=_f(row["mpp"]) or float("nan"),
        rebar_count=_i(row["rebar_count"]),
        spacings_mm=[],
        diameters_mm=[],
        mean_spacing_mm=_f(row.get("mean_spacing_mm", "")),
        std_spacing_mm=_f(row.get("std_spacing_mm", "")),
        mean_diameter_mm=_f(row.get("mean_diameter_mm", "")),
        compliance=compliance,
        warning=row.get("warning") or None,
    )


def generate_batch_report(
    csv_path: str,
    out_dir: str,
    lang: str = "en",
    model: str = "claude-haiku-4-5",
    api_key: Optional[str] = None,
    use_llm: bool = True,
) -> str:
    """
    Generate one report file per image row in results.csv.
    Also writes a combined summary report (summary.txt).

    Returns the path to summary.txt.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    summary_lines_zh = [
        "=" * 60,
        "钢筋检测批次汇总报告" if lang == "zh" else "Rebar Inspection Batch Summary",
        f"来源：{csv_path}",
        f"共 {len(rows)} 张图像",
        "=" * 60,
        "",
    ]

    valid = [r for r in rows if not r.get("warning")]
    pass_n = sum(1 for r in valid if r.get("compliance") == "PASS")
    pass_main_n = sum(1 for r in valid if r.get("compliance") == "PASS(main)")
    fail_n = sum(1 for r in valid if r.get("compliance") == "FAIL")
    na_n = len(rows) - len(valid)

    if lang == "zh":
        summary_lines_zh += [
            f"有效测量：{len(valid)}/{len(rows)} 张",
            f"合格（PASS）：{pass_n} 张",
            f"主筋合格（PASS(main)）：{pass_main_n} 张",
            f"不合格（FAIL）：{fail_n} 张",
            f"无法测量（N/A）：{na_n} 张",
            "",
        ]
    else:
        summary_lines_zh += [
            f"Valid measurements: {len(valid)}/{len(rows)}",
            f"PASS: {pass_n}",
            f"PASS(main): {pass_main_n}",
            f"FAIL: {fail_n}",
            f"N/A: {na_n}",
            "",
        ]

    if valid:
        import statistics
        sp = [float(r["mean_spacing_mm"]) for r in valid if r.get("mean_spacing_mm") not in ("", "None")]
        dm = [float(r["mean_diameter_mm"]) for r in valid if r.get("mean_diameter_mm") not in ("", "None")]
        if sp and lang == "zh":
            summary_lines_zh.append(
                f"批次平均间距：{statistics.mean(sp):.1f} mm  "
                f"（标准差 {statistics.stdev(sp):.1f} mm）" if len(sp) > 1
                else f"批次平均间距：{sp[0]:.1f} mm"
            )
        elif sp:
            summary_lines_zh.append(
                f"Batch mean spacing: {statistics.mean(sp):.1f} mm  "
                f"(std {statistics.stdev(sp):.1f} mm)" if len(sp) > 1
                else f"Batch mean spacing: {sp[0]:.1f} mm"
            )
        if dm and lang == "zh":
            summary_lines_zh.append(
                f"批次平均直径：{statistics.mean(dm):.1f} mm  "
                f"（标准差 {statistics.stdev(dm):.1f} mm）" if len(dm) > 1
                else f"批次平均直径：{dm[0]:.1f} mm"
            )
        elif dm:
            summary_lines_zh.append(
                f"Batch mean diameter: {statistics.mean(dm):.1f} mm  "
                f"(std {statistics.stdev(dm):.1f} mm)" if len(dm) > 1
                else f"Batch mean diameter: {dm[0]:.1f} mm"
            )
        pass_rate = 100 * pass_n / len(valid) if valid else 0
        if lang == "zh":
            summary_lines_zh.append(f"GB 50204 合格率：{pass_rate:.1f}%")
        else:
            summary_lines_zh.append(f"GB 50204 pass rate: {pass_rate:.1f}%")

    summary_lines_zh.append("")

    # Per-image reports
    for row in rows:
        result = _csv_to_result(row)
        report_text = generate_report(result, lang=lang, model=model,
                                      api_key=api_key, use_llm=use_llm)
        img_path = out / f"{result.image_id}.txt"
        img_path.write_text(report_text, encoding="utf-8")
        print(f"  [{result.image_id}] -> {img_path.name}")

        summary_lines_zh.append(f"--- {result.image_id} ---")
        summary_lines_zh.append(report_text)
        summary_lines_zh.append("")

    summary_path = out / "summary.txt"
    summary_path.write_text("\n".join(summary_lines_zh), encoding="utf-8")
    return str(summary_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv",      required=True,
                   help="Path to results.csv from run_measure.py")
    p.add_argument("--out_dir",  default="reports",
                   help="Output directory for report files")
    p.add_argument("--lang",     default="en", choices=["zh", "en"])
    p.add_argument("--model",    default="claude-haiku-4-5",
                   help="Anthropic model ID")
    p.add_argument("--no_llm",   action="store_true",
                   help="Skip LLM call and use template-based report")
    p.add_argument("--api_key",  default=None,
                   help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    args = p.parse_args()

    summary = generate_batch_report(
        csv_path=args.csv,
        out_dir=args.out_dir,
        lang=args.lang,
        model=args.model,
        api_key=args.api_key,
        use_llm=not args.no_llm,
    )
    print(f"\nSummary report -> {summary}")
