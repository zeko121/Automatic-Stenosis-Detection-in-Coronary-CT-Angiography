"""Markdown report generator for evaluation results."""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from evaluation.composite_score import ModelScore


def generate_markdown_report(
    ranked_models: List[ModelScore],
    output_path: Optional[Path] = None,
    title: str = "Model Evaluation Report",
) -> str:
    lines = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append(f"# {title}")
    lines.append(f"Generated: {timestamp}")
    lines.append("")

    if not ranked_models:
        lines.append("No models evaluated.")
        report = "\n".join(lines)
        if output_path:
            output_path.write_text(report, encoding="utf-8")
        return report

    model_names = [m.model_name for m in ranked_models]
    n_models = len(model_names)

    lines.append("## Final Ranking")
    lines.append("")
    lines.append(_md_table(
        headers=["Rank", "Model", "Composite", "T1 Seg", "T2 Struct", "T3 Downstream", "T4 Robust"],
        rows=[
            [
                str(m.rank),
                m.model_name,
                f"{m.composite_score:.3f}",
                f"{m.tier1_score:.3f}",
                f"{m.tier2_score:.3f}",
                f"{m.tier3_score:.3f}",
                f"{m.tier4_score:.3f}",
            ]
            for m in ranked_models
        ],
    ))
    lines.append("")

    lines.append("## Tier 1 — Segmentation Quality")
    lines.append("")

    t1_keys = [
        ("dice_mean", "Dice", ".1%"),
        ("precision_mean", "Precision", ".1%"),
        ("recall_mean", "Recall", ".1%"),
        ("surface_dice_0.5mm_mean", "Surf Dice @0.5mm", ".1%"),
        ("surface_dice_1.0mm_mean", "Surf Dice @1.0mm", ".1%"),
        ("hd95_mean", "HD95 (mm)", ".2f"),
        ("asd_mean", "ASD (mm)", ".2f"),
        ("mare_mean", "MARE (mm)", ".2f"),
        ("msre_mean", "MSRE (mm)", "+.2f"),
    ]

    lines.append(_tier_table(ranked_models, "tier1_raw", t1_keys))
    lines.append("")

    lines.append("## Tier 2 — Structural Quality")
    lines.append("")

    t2_keys = [
        ("n_components_mean", "Components", ".1f"),
        ("largest_component_fraction_mean", "LCF", ".1%"),
        ("n_bifurcations_mean", "Bifurcations", ".1f"),
        ("n_endpoints_mean", "Endpoints", ".1f"),
        ("bifurcation_to_endpoint_ratio_mean", "Bif/EP Ratio", ".2f"),
        ("total_centerline_length_mm_mean", "CL Length (mm)", ".0f"),
        ("radius_mean_mm_mean", "Radius Mean (mm)", ".2f"),
        ("radius_plausibility_mean", "Radius Plausible", ".1%"),
        ("continuity_score_mean", "Continuity Score", ".3f"),
    ]

    lines.append(_tier_table(ranked_models, "tier2_raw", t2_keys))
    lines.append("")

    lines.append("## Tier 3 — Downstream Task Performance")
    lines.append("")

    lines.append("### 3a — Oracle Comparison (ImageCAS)")
    lines.append("")

    t3a_keys = [
        ("oracle_severity_agreement", "Severity Agreement", ".0%"),
        ("oracle_within_one_grade", "Within One Grade", ".0%"),
        ("oracle_mean_ref_radius_ratio", "Ref Radius Ratio", ".2f"),
        ("oracle_miss_rate", "Miss Rate", ".0%"),
        ("oracle_hallucination_rate", "Hallucination Rate", ".0%"),
        ("oracle_n_cases", "N Cases", "d"),
    ]

    lines.append(_tier_table(ranked_models, "tier3_raw", t3a_keys))
    lines.append("")

    lines.append("### 3b-i — Per-Side Evaluation (Primary)")
    lines.append("")
    lines.append("Per-side (Left/Right coronary) metrics are the primary real-world")
    lines.append("evaluation. Side assignment is reliable; sub-artery labeling (LAD/LCx) is not.")
    lines.append("")

    t3b_side_keys = [
        ("ziv_side_sensitivity", "Side Sensitivity", ".0%"),
        ("ziv_side_specificity", "Side Specificity", ".0%"),
        ("ziv_side_max_severity_agreement", "Max Severity Agreement", ".0%"),
        ("ziv_side_cohens_kappa", "Side Cohen's Kappa", ".2f"),
        ("ziv_n_cases", "N Cases", "d"),
    ]

    lines.append(_tier_table(ranked_models, "tier3_raw", t3b_side_keys))
    lines.append("")

    # side-level CIs
    side_ci_rows = []
    for m in ranked_models:
        t3 = m.tier3_raw
        sens_ci = t3.get("ziv_side_sensitivity_ci", [0, 0])
        spec_ci = t3.get("ziv_side_specificity_ci", [0, 0])
        if isinstance(sens_ci, (list, tuple)) and len(sens_ci) == 2:
            side_ci_rows.append([
                m.model_name,
                f"{sens_ci[0]:.0%} - {sens_ci[1]:.0%}",
                f"{spec_ci[0]:.0%} - {spec_ci[1]:.0%}",
            ])
    if side_ci_rows:
        lines.append("**95% Bootstrap CIs (Per-Side):**")
        lines.append("")
        lines.append(_md_table(
            headers=["Model", "Side Sensitivity CI", "Side Specificity CI"],
            rows=side_ci_rows,
        ))
        lines.append("")

    lines.append("### 3b-ii — Per-Artery Evaluation (Supplementary)")
    lines.append("")
    lines.append("Sub-artery labeling (LAD/LCx) is unreliable; per-side metrics above are primary.")
    lines.append("")

    t3b_keys = [
        ("ziv_sensitivity", "Sensitivity", ".0%"),
        ("ziv_specificity", "Specificity", ".0%"),
        ("ziv_agreement_rate", "Agreement Rate", ".0%"),
        ("ziv_within_one_grade", "Within One Grade", ".0%"),
        ("ziv_cohens_kappa", "Cohen's Kappa", ".2f"),
        ("ziv_n_cases", "N Cases", "d"),
    ]

    lines.append(_tier_table(ranked_models, "tier3_raw", t3b_keys))
    lines.append("")

    # per-artery CIs
    ci_rows = []
    for m in ranked_models:
        t3 = m.tier3_raw
        sens_ci = t3.get("ziv_sensitivity_ci", [0, 0])
        spec_ci = t3.get("ziv_specificity_ci", [0, 0])
        if isinstance(sens_ci, (list, tuple)) and len(sens_ci) == 2:
            ci_rows.append([
                m.model_name,
                f"{sens_ci[0]:.0%} - {sens_ci[1]:.0%}",
                f"{spec_ci[0]:.0%} - {spec_ci[1]:.0%}",
            ])
    if ci_rows:
        lines.append("**95% Bootstrap CIs (Per-Artery):**")
        lines.append("")
        lines.append(_md_table(
            headers=["Model", "Sensitivity CI", "Specificity CI"],
            rows=ci_rows,
        ))
        lines.append("")

    lines.append("## Tier 4 — Domain Transfer Robustness")
    lines.append("")

    t4_keys = [
        ("domain_gap", "Domain Gap", ".3f"),
        ("bifurcation_ratio", "Bif Ratio (Z/I)", ".2f"),
        ("length_ratio", "Length Ratio (Z/I)", ".2f"),
        ("radius_mean_ratio", "Radius Ratio (Z/I)", ".2f"),
        ("lcf_diff", "LCF Diff", ".2f"),
    ]

    lines.append(_tier_table(ranked_models, "tier4_raw", t4_keys))
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("**Tier weights:** T1=0.25, T2=0.15, T3=0.45, T4=0.15")
    lines.append("")
    lines.append("**Tiebreaker priority:** Ziv sensitivity > Surface Dice @0.5mm > Bifurcation count")
    lines.append("")
    lines.append("**Positive threshold:** >=25% stenosis is clinically significant")
    lines.append("")

    report = "\n".join(lines)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report, encoding="utf-8")

    return report


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_widths):
                col_widths[i] = max(col_widths[i], len(cell))

    def _pad(cells: List[str]) -> str:
        padded = [c.ljust(col_widths[i]) if i < len(col_widths) else c
                  for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    lines = [
        _pad(headers),
        "| " + " | ".join("-" * w for w in col_widths) + " |",
    ]
    for row in rows:
        lines.append(_pad(row))

    return "\n".join(lines)


def _tier_table(
    ranked_models: List[ModelScore],
    raw_attr: str,
    keys: List[tuple],
) -> str:
    headers = ["Metric"] + [m.model_name for m in ranked_models]
    rows = []

    for key, label, fmt in keys:
        row = [label]
        for m in ranked_models:
            raw = getattr(m, raw_attr, {})
            val = raw.get(key)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                row.append("N/A")
            elif fmt == "d":
                row.append(str(int(val)))
            elif "%" in fmt:
                row.append(f"{val:{fmt}}")
            else:
                row.append(f"{val:{fmt}}")
        rows.append(row)

    return _md_table(headers, rows)


def save_results_json(
    ranked_models: List[ModelScore],
    output_path: Path,
):
    data = {
        "generated": datetime.now().isoformat(),
        "models": [m.to_dict() for m in ranked_models],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
