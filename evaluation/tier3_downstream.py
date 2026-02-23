"""Tier 3 -- Downstream task performance.

3a: Oracle comparison (model vs GT-derived findings on ImageCAS).
3b: Ziv real-world comparison vs radiologist reports + bootstrap CIs.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class OracleComparison:
    """Model vs oracle (GT-derived) findings for one case."""
    case_id: str = ""
    n_arteries_compared: int = 0
    severity_matches: int = 0
    severity_agreement_rate: float = 0.0
    within_one_grade: int = 0
    within_one_grade_rate: float = 0.0

    ref_radius_ratios: List[float] = field(default_factory=list)
    mean_ref_radius_ratio: float = float("nan")  # model/oracle, >1 = inflation

    oracle_findings_count: int = 0
    model_findings_count: int = 0
    misses: int = 0
    hallucinations: int = 0
    miss_rate: float = 0.0
    hallucination_rate: float = 0.0

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "n_arteries_compared": self.n_arteries_compared,
            "severity_agreement_rate": self.severity_agreement_rate,
            "within_one_grade_rate": self.within_one_grade_rate,
            "mean_ref_radius_ratio": self.mean_ref_radius_ratio,
            "oracle_findings_count": self.oracle_findings_count,
            "model_findings_count": self.model_findings_count,
            "misses": self.misses,
            "hallucinations": self.hallucinations,
            "miss_rate": self.miss_rate,
            "hallucination_rate": self.hallucination_rate,
        }


@dataclass
class Tier3Metrics:
    model_name: str = ""

    # 3a: Oracle (ImageCAS)
    oracle_severity_agreement: float = 0.0
    oracle_within_one_grade: float = 0.0
    oracle_mean_ref_radius_ratio: float = float("nan")
    oracle_miss_rate: float = 0.0
    oracle_hallucination_rate: float = 0.0
    oracle_n_cases: int = 0

    # 3b: Ziv
    ziv_sensitivity: float = 0.0
    ziv_specificity: float = 0.0
    ziv_agreement_rate: float = 0.0
    ziv_within_one_grade: float = 0.0
    ziv_cohens_kappa: float = 0.0
    ziv_n_cases: int = 0

    ziv_sensitivity_ci: Tuple[float, float] = (0.0, 0.0)
    ziv_specificity_ci: Tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "oracle_severity_agreement": self.oracle_severity_agreement,
            "oracle_within_one_grade": self.oracle_within_one_grade,
            "oracle_mean_ref_radius_ratio": self.oracle_mean_ref_radius_ratio,
            "oracle_miss_rate": self.oracle_miss_rate,
            "oracle_hallucination_rate": self.oracle_hallucination_rate,
            "oracle_n_cases": self.oracle_n_cases,
            "ziv_sensitivity": self.ziv_sensitivity,
            "ziv_specificity": self.ziv_specificity,
            "ziv_agreement_rate": self.ziv_agreement_rate,
            "ziv_within_one_grade": self.ziv_within_one_grade,
            "ziv_cohens_kappa": self.ziv_cohens_kappa,
            "ziv_n_cases": self.ziv_n_cases,
            "ziv_sensitivity_ci": list(self.ziv_sensitivity_ci),
            "ziv_specificity_ci": list(self.ziv_specificity_ci),
        }


SEVERITY_ORDER = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
POSITIVE_THRESHOLD = 25.0


def run_oracle_pipeline(
    gt_mask: np.ndarray,
    temp_dir: Path,
    case_id: str = "",
    spacing_mm: float = 0.5,
) -> Optional[dict]:
    """Run postprocess -> centerline -> label -> stenosis on a mask. Returns findings or None."""
    import json
    import tempfile

    from pipeline.postprocess import postprocess_mask
    from pipeline.centerline import extract_vessel_tree
    from pipeline.label_arteries import label_arteries
    from pipeline.stenosis import StenosisDetector, StenosisConfig

    try:
        pp_mask, _metrics = postprocess_mask(gt_mask, verbose=False)

        if pp_mask.sum() == 0:
            logger.warning(f"Oracle {case_id}: empty mask after PP")
            return None

        tree, _skel, _dt = extract_vessel_tree(pp_mask)
        centerline_data = _vessel_tree_to_dict(tree, spacing_mm)

        labeling_result = label_arteries(centerline_data, vessel_mask=pp_mask,
                                         spacing_mm=spacing_mm, verbose=False)

        detector = StenosisDetector(StenosisConfig())
        findings = detector.detect(centerline_data, artery_labels=labeling_result.labels)

        severity_counts = {"Normal": 0, "Mild": 0, "Moderate": 0, "Severe": 0}
        for f in findings:
            severity_counts[f.severity] += 1

        max_severity = "Normal"
        for sev in ["Severe", "Moderate", "Mild"]:
            if severity_counts[sev] > 0:
                max_severity = sev
                break

        findings_dicts = []
        for f in findings:
            fd = {
                "segment_id": f.segment_id,
                "stenosis_percent": f.stenosis_percent,
                "severity": f.severity,
                "min_radius_mm": f.min_radius_mm,
                "reference_radius_mm": f.reference_radius_mm,
                "artery_name": f.artery_name,
                "artery_region": f.artery_region,
                "confidence": f.confidence,
                "segment_length_mm": f.segment_length_mm,
            }
            findings_dicts.append(fd)

        return {
            "summary": {
                "total_findings": len(findings),
                "max_severity": max_severity,
                "by_severity": severity_counts,
            },
            "findings": findings_dicts,
        }

    except Exception as e:
        logger.error(f"Oracle pipeline failed for {case_id}: {e}")
        return None


def _vessel_tree_to_dict(tree, spacing_mm: float = 0.5) -> dict:
    """Convert VesselTree to dict matching centerline.process() output format."""
    nodes_dict = {}
    for nid, node in tree.nodes.items():
        nodes_dict[str(nid)] = {
            "id": node.id,
            "position": node.position,
            "position_mm": node.position_mm,
            "radius_mm": node.radius_mm,
            "is_bifurcation": node.is_bifurcation,
            "is_endpoint": node.is_endpoint,
            "neighbors": node.neighbors,
        }

    segments_dict = {}
    for sid, seg in tree.segments.items():
        segments_dict[str(sid)] = {
            "id": seg.id,
            "node_ids": seg.node_ids,
            "centerline_points": (
                [list(p) for p in seg.centerline_points]
                if seg.centerline_points is not None else []
            ),
            "arc_length_mm": (
                list(seg.arc_length_mm) if seg.arc_length_mm is not None else []
            ),
            "radii_mm": list(seg.radii_mm) if seg.radii_mm is not None else [],
            "radii_smooth_mm": (
                list(seg.radii_smooth_mm) if seg.radii_smooth_mm is not None else []
            ),
            "length_mm": seg.length_mm,
        }

    return {
        "vessel_tree": {
            "nodes": nodes_dict,
            "segments": segments_dict,
            "edges": tree.edges,
            "total_length_mm": tree.total_length_mm,
            "num_bifurcations": tree.num_bifurcations,
            "num_endpoints": tree.num_endpoints,
        },
        "config": {"voxel_spacing_mm": spacing_mm},
    }


def compare_model_vs_oracle(
    model_findings: dict,
    oracle_findings: dict,
    case_id: str = "",
) -> OracleComparison:
    """Per-artery comparison: severity agreement, miss/hallucination rates."""
    from pipeline.compare_gt import normalize_artery_name

    comp = OracleComparison(case_id=case_id)

    def _aggregate_per_artery(findings_dict: dict) -> Dict[str, dict]:
        per_artery: Dict[str, dict] = {}
        for f in findings_dict.get("findings", []):
            name = normalize_artery_name(
                f.get("artery_region") or f.get("artery_name", "")
            )
            if name is None:
                continue
            if name not in per_artery or f.get("stenosis_percent", 0) > per_artery[name].get("stenosis_percent", 0):
                per_artery[name] = f
        return per_artery

    model_arteries = _aggregate_per_artery(model_findings)
    oracle_arteries = _aggregate_per_artery(oracle_findings)

    canonical = ["LM", "LAD", "LCx", "RCA"]
    matches = 0
    within_one = 0
    compared = 0

    for artery in canonical:
        m = model_arteries.get(artery)
        o = oracle_arteries.get(artery)

        m_sev = m.get("severity", "Normal") if m else "Normal"
        o_sev = o.get("severity", "Normal") if o else "Normal"

        compared += 1
        if m_sev == o_sev:
            matches += 1
        if abs(SEVERITY_ORDER.get(m_sev, 0) - SEVERITY_ORDER.get(o_sev, 0)) <= 1:
            within_one += 1

        if m and o:
            m_ref = m.get("reference_radius_mm", 0)
            o_ref = o.get("reference_radius_mm", 0)
            if o_ref > 0 and m_ref > 0:
                comp.ref_radius_ratios.append(m_ref / o_ref)

    comp.n_arteries_compared = compared
    comp.severity_matches = matches
    comp.severity_agreement_rate = matches / compared if compared > 0 else 0.0
    comp.within_one_grade = within_one
    comp.within_one_grade_rate = within_one / compared if compared > 0 else 0.0

    if comp.ref_radius_ratios:
        comp.mean_ref_radius_ratio = float(np.mean(comp.ref_radius_ratios))

    oracle_positive = {
        a for a, f in oracle_arteries.items()
        if f.get("stenosis_percent", 0) >= POSITIVE_THRESHOLD
    }
    model_positive = {
        a for a, f in model_arteries.items()
        if f.get("stenosis_percent", 0) >= POSITIVE_THRESHOLD
    }

    comp.oracle_findings_count = len(oracle_positive)
    comp.model_findings_count = len(model_positive)
    comp.misses = len(oracle_positive - model_positive)
    comp.hallucinations = len(model_positive - oracle_positive)
    comp.miss_rate = comp.misses / len(oracle_positive) if oracle_positive else 0.0
    comp.hallucination_rate = (
        comp.hallucinations / len(model_positive) if model_positive else 0.0
    )

    return comp


def compute_cohens_kappa(comparisons: list) -> float:
    """Cohen's kappa for severity agreement (Normal/Mild/Moderate/Severe)."""
    categories = list(SEVERITY_ORDER.keys())
    n_cat = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    matrix = np.zeros((n_cat, n_cat), dtype=int)
    for comp in comparisons:
        gt_i = cat_idx.get(comp.gt_severity, 0)
        pipe_i = cat_idx.get(comp.pipeline_severity, 0)
        matrix[gt_i, pipe_i] += 1

    n = matrix.sum()
    if n == 0:
        return 0.0

    po = np.trace(matrix) / n
    pe = np.sum(matrix.sum(axis=0) * matrix.sum(axis=1)) / (n * n)

    if pe >= 1.0:
        return 1.0 if po >= 1.0 else 0.0
    return float((po - pe) / (1 - pe))


def bootstrap_metric(
    case_results: List[dict],
    metric_fn,
    n_iterations: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, Tuple[float, float]]:
    """Case-level bootstrap. Returns (point_estimate, (lower_ci, upper_ci))."""
    rng = np.random.RandomState(seed)
    n = len(case_results)
    if n == 0:
        return 0.0, (0.0, 0.0)

    point = metric_fn(case_results)
    boot_vals = []
    for _ in range(n_iterations):
        sample = [case_results[i] for i in rng.randint(0, n, size=n)]
        boot_vals.append(metric_fn(sample))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_vals, 100 * alpha))
    upper = float(np.percentile(boot_vals, 100 * (1 - alpha)))

    return point, (lower, upper)


def _sensitivity_from_cases(case_results: List[dict]) -> float:
    tp = sum(c.get("true_positives", 0) for c in case_results)
    fn = sum(c.get("false_negatives", 0) for c in case_results)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def _specificity_from_cases(case_results: List[dict]) -> float:
    tn = sum(c.get("true_negatives", 0) for c in case_results)
    fp = sum(c.get("false_positives", 0) for c in case_results)
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def evaluate_ziv_cases(
    case_comparisons: List["ComparisonResult"],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict:
    """Aggregate Ziv results with bootstrap CIs and Cohen's kappa."""
    if not case_comparisons:
        return {}

    all_artery_comps = []
    case_dicts = []

    for cr in case_comparisons:
        all_artery_comps.extend(cr.artery_comparisons)
        case_dicts.append({
            "true_positives": cr.true_positives,
            "true_negatives": cr.true_negatives,
            "false_positives": cr.false_positives,
            "false_negatives": cr.false_negatives,
            "agreement_rate": cr.agreement_rate,
            "within_one_grade_rate": cr.within_one_grade_rate,
        })

    tp = sum(d["true_positives"] for d in case_dicts)
    tn = sum(d["true_negatives"] for d in case_dicts)
    fp = sum(d["false_positives"] for d in case_dicts)
    fn = sum(d["false_negatives"] for d in case_dicts)

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    agreement = float(np.mean([d["agreement_rate"] for d in case_dicts]))
    within_one = float(np.mean([d["within_one_grade_rate"] for d in case_dicts]))

    _, sens_ci = bootstrap_metric(case_dicts, _sensitivity_from_cases,
                                   n_bootstrap, ci, seed)
    _, spec_ci = bootstrap_metric(case_dicts, _specificity_from_cases,
                                   n_bootstrap, ci, seed)

    kappa = compute_cohens_kappa(all_artery_comps)

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "sensitivity_ci": sens_ci,
        "specificity_ci": spec_ci,
        "agreement_rate": agreement,
        "within_one_grade_rate": within_one,
        "cohens_kappa": kappa,
        "n_cases": len(case_comparisons),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def aggregate_oracle(results: List[OracleComparison]) -> Dict[str, float]:
    if not results:
        return {}

    agg: Dict[str, float] = {}
    agg["severity_agreement_mean"] = float(
        np.mean([r.severity_agreement_rate for r in results])
    )
    agg["within_one_grade_mean"] = float(
        np.mean([r.within_one_grade_rate for r in results])
    )

    all_ratios = []
    for r in results:
        all_ratios.extend(r.ref_radius_ratios)
    if all_ratios:
        agg["ref_radius_ratio_mean"] = float(np.mean(all_ratios))
        agg["ref_radius_ratio_std"] = float(np.std(all_ratios))

    total_oracle = sum(r.oracle_findings_count for r in results)
    total_misses = sum(r.misses for r in results)
    total_model = sum(r.model_findings_count for r in results)
    total_halluc = sum(r.hallucinations for r in results)

    agg["miss_rate"] = total_misses / total_oracle if total_oracle > 0 else 0.0
    agg["hallucination_rate"] = total_halluc / total_model if total_model > 0 else 0.0
    agg["n_cases"] = len(results)

    return agg


def build_tier3_metrics(
    model_name: str,
    oracle_results: Optional[List[OracleComparison]] = None,
    ziv_results: Optional[dict] = None,
) -> Tier3Metrics:
    m = Tier3Metrics(model_name=model_name)

    if oracle_results:
        oagg = aggregate_oracle(oracle_results)
        m.oracle_severity_agreement = oagg.get("severity_agreement_mean", 0.0)
        m.oracle_within_one_grade = oagg.get("within_one_grade_mean", 0.0)
        m.oracle_mean_ref_radius_ratio = oagg.get("ref_radius_ratio_mean", float("nan"))
        m.oracle_miss_rate = oagg.get("miss_rate", 0.0)
        m.oracle_hallucination_rate = oagg.get("hallucination_rate", 0.0)
        m.oracle_n_cases = int(oagg.get("n_cases", 0))

    if ziv_results:
        m.ziv_sensitivity = ziv_results.get("sensitivity", 0.0)
        m.ziv_specificity = ziv_results.get("specificity", 0.0)
        m.ziv_agreement_rate = ziv_results.get("agreement_rate", 0.0)
        m.ziv_within_one_grade = ziv_results.get("within_one_grade_rate", 0.0)
        m.ziv_cohens_kappa = ziv_results.get("cohens_kappa", 0.0)
        m.ziv_n_cases = ziv_results.get("n_cases", 0)
        m.ziv_sensitivity_ci = tuple(ziv_results.get("sensitivity_ci", (0.0, 0.0)))
        m.ziv_specificity_ci = tuple(ziv_results.get("specificity_ci", (0.0, 0.0)))

    return m
