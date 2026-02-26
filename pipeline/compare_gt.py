"""Compare pipeline findings against ground truth radiologist reports."""

import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

CANONICAL_ARTERIES = ["LM", "LAD", "LCx", "RCA"]
CANONICAL_SIDES = ["Left", "Right"]
POSITIVE_THRESHOLD = 25.0  # >=25% is clinically significant
SEVERITY_ORDER = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
_SEVERITY_FROM_ORDER = {v: k for k, v in SEVERITY_ORDER.items()}

# map artery names to coronary side
_SIDE_MAP = {
    "LM": "Left", "LAD": "Left", "LCx": "Left", "LCX": "Left",
    "D1": "Left", "D2": "Left", "OM": "Left", "OM1": "Left", "OM2": "Left",
    "Left Coronary": "Left",
    "RCA": "Right", "PDA": "Right", "PLB": "Right",
    "Right Coronary": "Right",
}

_NAME_MAP = {
    "LCX": "LCx", "lcx": "LCx", "Lcx": "LCx", "LCx": "LCx",
    "LM": "LM", "lm": "LM",
    "LAD": "LAD", "lad": "LAD",
    "RCA": "RCA", "rca": "RCA",
    "Left Coronary": "LAD",
    "Right Coronary": "RCA",
}

_REGION_PREFIXES = ["proximal", "mid", "distal"]


def normalize_artery_name(name):
    """Normalize to one of LM, LAD, LCx, RCA. Returns None if unmappable."""
    if not name:
        return None

    name = name.strip()

    if name in _NAME_MAP:
        return _NAME_MAP[name]

    # strip regional prefixes like "proximal LAD"
    lower = name.lower()
    for prefix in _REGION_PREFIXES:
        if lower.startswith(prefix + " "):
            remainder = name[len(prefix) + 1:].strip()
            return normalize_artery_name(remainder)

    # handle underscore-separated GT keys like "LAD_prox"
    if "_" in name:
        base = name.split("_")[0]
        return normalize_artery_name(base)

    return None


def artery_to_side(name, dominance=None):
    """map artery name to 'Left' or 'Right'. returns None if unmappable.

    PDA/PLB go to Right by default, but override to Left if dominance=='Left'.
    """
    if not name:
        return None

    name = name.strip()

    # direct lookup
    if name in _SIDE_MAP:
        side = _SIDE_MAP[name]
        # PDA/PLB override for left dominance
        base = name.split("_")[0] if "_" in name else name
        if base in ("PDA", "PLB") and dominance == "Left":
            return "Left"
        return side

    # strip regional prefixes
    lower = name.lower()
    for prefix in _REGION_PREFIXES:
        if lower.startswith(prefix + " "):
            remainder = name[len(prefix) + 1:].strip()
            return artery_to_side(remainder, dominance)

    # underscore-separated GT keys like "LAD_prox"
    if "_" in name:
        base = name.split("_")[0]
        return artery_to_side(base, dominance)

    # try via normalize_artery_name -> _SIDE_MAP
    canonical = normalize_artery_name(name)
    if canonical and canonical in _SIDE_MAP:
        return _SIDE_MAP[canonical]

    return None


def _severity_from_percent(pct):
    """classify stenosis percent into severity tier"""
    if pct < 25:
        return "Normal"
    elif pct < 50:
        return "Mild"
    elif pct < 70:
        return "Moderate"
    else:
        return "Severe"


@dataclass
class ArteryComparison:
    """Per-artery comparison result."""
    artery: str
    gt_severity: str
    gt_stenosis_percent: float
    pipeline_severity: str
    pipeline_stenosis_percent: float
    severity_match: bool
    classification: str  # TP, TN, FP, FN
    stenosis_diff: float

    def to_dict(self):
        return asdict(self)


@dataclass
class SideComparison:
    """comparison for one coronary side (Left or Right)"""
    side: str                                    # "Left" or "Right"

    # GT severity profile (only findings >= 25%)
    gt_severity_counts: dict = field(default_factory=dict)
    gt_positive_count: int = 0
    gt_max_severity: str = "Normal"
    gt_max_stenosis_percent: float = 0.0
    gt_segments_included: list = field(default_factory=list)

    # Pipeline severity profile (only findings >= 25%)
    pipeline_severity_counts: dict = field(default_factory=dict)
    pipeline_positive_count: int = 0
    pipeline_max_severity: str = "Normal"
    pipeline_max_stenosis_percent: float = 0.0

    any_positive_classification: str = "TN"   # TP/TN/FP/FN
    max_severity_match: bool = True
    max_stenosis_diff: float = 0.0
    detection_count_diff: int = 0

    def to_dict(self):
        return asdict(self)


@dataclass
class ComparisonResult:
    """Full pipeline vs GT comparison."""
    artery_comparisons: list = field(default_factory=list)
    agreement_rate: float = 0.0
    within_one_grade_rate: float = 0.0
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    sensitivity: float = 0.0
    specificity: float = 0.0
    unmapped_findings: list = field(default_factory=list)
    gt_metadata: dict = field(default_factory=dict)

    # per-side evaluation (primary)
    side_comparisons: list = field(default_factory=list)
    side_sensitivity: float = 0.0
    side_specificity: float = 0.0
    side_max_severity_agreement: float = 0.0

    def to_dict(self):
        return {
            "artery_comparisons": [c.to_dict() for c in self.artery_comparisons],
            "agreement_rate": self.agreement_rate,
            "within_one_grade_rate": self.within_one_grade_rate,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "unmapped_findings": self.unmapped_findings,
            "gt_metadata": self.gt_metadata,
            "side_comparisons": [c.to_dict() for c in self.side_comparisons],
            "side_sensitivity": self.side_sensitivity,
            "side_specificity": self.side_specificity,
            "side_max_severity_agreement": self.side_max_severity_agreement,
        }


def _normalize_severity_name(severity):
    """GT severity can be int or string -- normalize to string."""
    if isinstance(severity, int):
        return {0: "Normal", 1: "Mild", 2: "Moderate", 3: "Severe"}.get(severity, "Normal")
    if isinstance(severity, str):
        s = severity.strip().capitalize()
        if s in SEVERITY_ORDER:
            return s
    return "Normal"


def _build_side_profiles(gt_report, findings_list, dominance):
    """build GT and pipeline severity profiles per side.

    Returns (gt_side_data, pipe_side_data) where each is:
      { "Left": { "counts": {"Mild": 2, ...}, "max_pct": 75.0, "segments": [...] },
        "Right": ... }
    """
    training_labels = gt_report.get("training_labels", {})

    # --- GT side aggregation ---
    # prefer segments (per-finding granularity), fall back to arteries
    gt_segments = training_labels.get("segments", {})
    if not gt_segments:
        gt_segments = training_labels.get("arteries", {})

    gt_side = {}
    for side in CANONICAL_SIDES:
        gt_side[side] = {"counts": {}, "max_pct": 0.0, "segments": [], "positive_count": 0}

    for seg_key, seg_data in gt_segments.items():
        # determine side from the segment key itself, or from its artery sub-field
        side = artery_to_side(seg_key, dominance)
        if side is None:
            artery_field = seg_data.get("artery", "")
            side = artery_to_side(artery_field, dominance)
        if side is None:
            continue

        pct = float(seg_data.get("stenosis_percent", 0.0))
        if pct >= POSITIVE_THRESHOLD:
            sev = _severity_from_percent(pct)
            gt_side[side]["counts"][sev] = gt_side[side]["counts"].get(sev, 0) + 1
            gt_side[side]["positive_count"] += 1
            gt_side[side]["segments"].append(seg_key)
        if pct > gt_side[side]["max_pct"]:
            gt_side[side]["max_pct"] = pct

    # --- Pipeline side aggregation ---
    pipe_side = {}
    for side in CANONICAL_SIDES:
        pipe_side[side] = {"counts": {}, "max_pct": 0.0, "positive_count": 0}

    for finding in findings_list:
        artery_region = finding.get("artery_region", "")
        artery_name = finding.get("artery_name", "")
        side = artery_to_side(artery_region, dominance) or artery_to_side(artery_name, dominance)
        if side is None:
            continue

        pct = float(finding.get("stenosis_percent", 0.0))
        if pct >= POSITIVE_THRESHOLD:
            sev = _severity_from_percent(pct)
            pipe_side[side]["counts"][sev] = pipe_side[side]["counts"].get(sev, 0) + 1
            pipe_side[side]["positive_count"] += 1
        if pct > pipe_side[side]["max_pct"]:
            pipe_side[side]["max_pct"] = pct

    return gt_side, pipe_side


def compare_findings(pipeline_findings, gt_report):
    """Compare pipeline findings against GT report."""
    result = ComparisonResult()

    training_labels = gt_report.get("training_labels", {})
    gt_arteries = training_labels.get("arteries", {})
    dominance = training_labels.get("dominance")

    result.gt_metadata = {
        "serial_number": training_labels.get("serial_number"),
        "scan_date": gt_report.get("report_data", {}).get("scan_date"),
        "gender": training_labels.get("gender"),
        "dominance": dominance,
        "study_quality": gt_report.get("report_data", {}).get("study_quality"),
    }

    # aggregate per artery, keep max stenosis %
    findings_list = pipeline_findings.get("findings", [])
    pipeline_per_artery = {}
    unmapped = []

    for finding in findings_list:
        artery_name = finding.get("artery_name", "")
        artery_region = finding.get("artery_region", "")

        canonical = normalize_artery_name(artery_region) or normalize_artery_name(artery_name)

        if canonical is None:
            unmapped.append(finding)
            continue

        if canonical not in pipeline_per_artery:
            pipeline_per_artery[canonical] = finding
        else:
            if finding.get("stenosis_percent", 0) > pipeline_per_artery[canonical].get("stenosis_percent", 0):
                pipeline_per_artery[canonical] = finding

    result.unmapped_findings = unmapped

    # per-artery comparison (supplementary)
    matches = 0
    within_one = 0

    for artery in CANONICAL_ARTERIES:
        gt_key = artery.upper() if artery == "LCx" else artery  # GT uses "LCX"
        gt_data = gt_arteries.get(gt_key, {})
        gt_severity = _normalize_severity_name(gt_data.get("severity_name", gt_data.get("severity", "Normal")))
        gt_pct = float(gt_data.get("stenosis_percent", 0.0))

        pipe_finding = pipeline_per_artery.get(artery)
        if pipe_finding:
            pipe_severity = pipe_finding.get("severity", "Normal")
            pipe_pct = float(pipe_finding.get("stenosis_percent", 0.0))
        else:
            pipe_severity = "Normal"
            pipe_pct = 0.0

        severity_match = (gt_severity == pipe_severity)
        if severity_match:
            matches += 1

        gt_order = SEVERITY_ORDER.get(gt_severity, 0)
        pipe_order = SEVERITY_ORDER.get(pipe_severity, 0)
        if abs(gt_order - pipe_order) <= 1:
            within_one += 1

        gt_positive = gt_pct >= POSITIVE_THRESHOLD
        pipe_positive = pipe_pct >= POSITIVE_THRESHOLD

        if gt_positive and pipe_positive:
            classification = "TP"
            result.true_positives += 1
        elif not gt_positive and not pipe_positive:
            classification = "TN"
            result.true_negatives += 1
        elif not gt_positive and pipe_positive:
            classification = "FP"
            result.false_positives += 1
        else:
            classification = "FN"
            result.false_negatives += 1

        comp = ArteryComparison(
            artery=artery,
            gt_severity=gt_severity,
            gt_stenosis_percent=gt_pct,
            pipeline_severity=pipe_severity,
            pipeline_stenosis_percent=pipe_pct,
            severity_match=severity_match,
            classification=classification,
            stenosis_diff=pipe_pct - gt_pct,
        )
        result.artery_comparisons.append(comp)

    n_arteries = len(CANONICAL_ARTERIES)
    result.agreement_rate = matches / n_arteries if n_arteries > 0 else 0.0
    result.within_one_grade_rate = within_one / n_arteries if n_arteries > 0 else 0.0

    tp_fn = result.true_positives + result.false_negatives
    tn_fp = result.true_negatives + result.false_positives
    result.sensitivity = result.true_positives / tp_fn if tp_fn > 0 else 0.0
    result.specificity = result.true_negatives / tn_fp if tn_fp > 0 else 0.0

    # per-side comparison (primary)
    gt_side, pipe_side = _build_side_profiles(gt_report, findings_list, dominance)

    side_tp = 0
    side_tn = 0
    side_fp = 0
    side_fn = 0
    side_max_matches = 0

    for side in CANONICAL_SIDES:
        gs = gt_side[side]
        ps = pipe_side[side]

        gt_max_pct = gs["max_pct"]
        pipe_max_pct = ps["max_pct"]
        gt_max_sev = _severity_from_percent(gt_max_pct) if gt_max_pct >= POSITIVE_THRESHOLD else "Normal"
        pipe_max_sev = _severity_from_percent(pipe_max_pct) if pipe_max_pct >= POSITIVE_THRESHOLD else "Normal"

        gt_diseased = gs["positive_count"] > 0
        pipe_diseased = ps["positive_count"] > 0

        if gt_diseased and pipe_diseased:
            classification = "TP"
            side_tp += 1
        elif not gt_diseased and not pipe_diseased:
            classification = "TN"
            side_tn += 1
        elif not gt_diseased and pipe_diseased:
            classification = "FP"
            side_fp += 1
        else:
            classification = "FN"
            side_fn += 1

        max_sev_match = (gt_max_sev == pipe_max_sev)
        if max_sev_match:
            side_max_matches += 1

        sc = SideComparison(
            side=side,
            gt_severity_counts=dict(gs["counts"]),
            gt_positive_count=gs["positive_count"],
            gt_max_severity=gt_max_sev,
            gt_max_stenosis_percent=gt_max_pct,
            gt_segments_included=list(gs["segments"]),
            pipeline_severity_counts=dict(ps["counts"]),
            pipeline_positive_count=ps["positive_count"],
            pipeline_max_severity=pipe_max_sev,
            pipeline_max_stenosis_percent=pipe_max_pct,
            any_positive_classification=classification,
            max_severity_match=max_sev_match,
            max_stenosis_diff=pipe_max_pct - gt_max_pct,
            detection_count_diff=ps["positive_count"] - gs["positive_count"],
        )
        result.side_comparisons.append(sc)

    # side-level summary
    side_tp_fn = side_tp + side_fn
    side_tn_fp = side_tn + side_fp
    result.side_sensitivity = side_tp / side_tp_fn if side_tp_fn > 0 else 0.0
    result.side_specificity = side_tn / side_tn_fp if side_tn_fp > 0 else 0.0
    n_sides = len(CANONICAL_SIDES)
    result.side_max_severity_agreement = side_max_matches / n_sides if n_sides > 0 else 0.0

    return result


def compute_side_cohens_kappa(side_comparisons_list):
    """compute binary Cohen's kappa from a list of SideComparison objects.

    Designed for aggregation across multiple cases (not meaningful for a single case
    with only 2 sides). Each SideComparison contributes one observation.

    Returns kappa (float), or 0.0 if inputs are empty or degenerate.
    """
    if not side_comparisons_list:
        return 0.0

    n = len(side_comparisons_list)
    tp = sum(1 for sc in side_comparisons_list if sc.any_positive_classification == "TP")
    tn = sum(1 for sc in side_comparisons_list if sc.any_positive_classification == "TN")
    fp = sum(1 for sc in side_comparisons_list if sc.any_positive_classification == "FP")
    fn = sum(1 for sc in side_comparisons_list if sc.any_positive_classification == "FN")

    # observed agreement
    p_o = (tp + tn) / n

    # expected agreement by chance
    gt_pos = tp + fn
    gt_neg = tn + fp
    pipe_pos = tp + fp
    pipe_neg = tn + fn
    p_e = (gt_pos * pipe_pos + gt_neg * pipe_neg) / (n * n)

    if p_e == 1.0:
        return 1.0 if p_o == 1.0 else 0.0

    kappa = (p_o - p_e) / (1 - p_e)
    return kappa


def format_comparison_text(result):
    """Render comparison as formatted text for display."""
    lines = []

    lines.append("=" * 70)
    lines.append("GROUND TRUTH COMPARISON REPORT")
    lines.append("=" * 70)
    lines.append("")

    meta = result.gt_metadata
    if any(meta.values()):
        lines.append("-" * 70)
        lines.append("REPORT METADATA")
        lines.append("-" * 70)
        if meta.get("serial_number"):
            lines.append(f"  Serial Number:   {meta['serial_number']}")
        if meta.get("scan_date"):
            lines.append(f"  Scan Date:       {meta['scan_date']}")
        if meta.get("gender"):
            lines.append(f"  Gender:          {meta['gender']}")
        if meta.get("dominance"):
            lines.append(f"  Dominance:       {meta['dominance']}")
        if meta.get("study_quality"):
            lines.append(f"  Study Quality:   {meta['study_quality']}")
        lines.append("")


    if result.side_comparisons:
        lines.append("-" * 70)
        lines.append("PER-SIDE COMPARISON (PRIMARY)")
        lines.append("-" * 70)
        lines.append("")

        # summary table
        hdr = f"{'Side':<8} {'GT Max':<10} {'GT%':>5} {'Pipe Max':<10} {'Pipe%':>6} {'GT#':>4} {'Pipe#':>6} {'Binary':>7}"
        lines.append(hdr)
        lines.append("-" * len(hdr))

        for sc in result.side_comparisons:
            gt_pct_str = f"{sc.gt_max_stenosis_percent:.0f}%"
            pipe_pct_str = f"{sc.pipeline_max_stenosis_percent:.0f}%"
            line = (
                f"{sc.side:<8} "
                f"{sc.gt_max_severity:<10} "
                f"{gt_pct_str:>5} "
                f"{sc.pipeline_max_severity:<10} "
                f"{pipe_pct_str:>6} "
                f"{sc.gt_positive_count:>4} "
                f"{sc.pipeline_positive_count:>6} "
                f"{sc.any_positive_classification:>7}"
            )
            lines.append(line)

        lines.append("")

        # severity profile breakdown per side
        for sc in result.side_comparisons:
            lines.append(f"  {sc.side} side severity profile:")
            for tier in ["Severe", "Moderate", "Mild"]:
                gt_n = sc.gt_severity_counts.get(tier, 0)
                pipe_n = sc.pipeline_severity_counts.get(tier, 0)
                if gt_n > 0 or pipe_n > 0:
                    lines.append(f"    {tier:<10}  GT: {gt_n}  Pipeline: {pipe_n}")
            if sc.gt_segments_included:
                lines.append(f"    GT segments: {', '.join(sc.gt_segments_included)}")
            lines.append("")

        # side detection stats
        lines.append(f"  Side Detection (binary: any stenosis >=25%):")
        lines.append(f"    Sensitivity:              {result.side_sensitivity:.0%}")
        lines.append(f"    Specificity:              {result.side_specificity:.0%}")
        lines.append(f"    Max Severity Agreement:   {result.side_max_severity_agreement:.0%}")
        lines.append("")


    lines.append("-" * 70)
    lines.append("PER-ARTERY COMPARISON (SUPPLEMENTARY)")
    lines.append("-" * 70)
    lines.append("  Note: Sub-artery labeling (LAD vs LCx) is approximate.")
    lines.append("")

    header = f"{'Artery':<8} {'GT Severity':<14} {'GT %':>6} {'Pipe Severity':<14} {'Pipe %':>7} {'Match':>6} {'Diff':>7}"
    lines.append(header)
    lines.append("-" * len(header))

    for comp in result.artery_comparisons:
        match_str = "YES" if comp.severity_match else "NO"
        diff_str = f"{comp.stenosis_diff:+.0f}%" if comp.stenosis_diff != 0 else "0%"
        line = (
            f"{comp.artery:<8} "
            f"{comp.gt_severity:<14} "
            f"{comp.gt_stenosis_percent:5.0f}% "
            f"{comp.pipeline_severity:<14} "
            f"{comp.pipeline_stenosis_percent:6.1f}% "
            f"{match_str:>6} "
            f"{diff_str:>7}"
        )
        lines.append(line)

    lines.append("")

    lines.append("-" * 70)
    lines.append("PER-ARTERY SUMMARY METRICS")
    lines.append("-" * 70)
    lines.append(f"  Severity Agreement:      {result.agreement_rate:.0%} ({sum(1 for c in result.artery_comparisons if c.severity_match)}/4 arteries)")
    lines.append(f"  Within One Grade:        {result.within_one_grade_rate:.0%}")
    lines.append("")
    lines.append("  Detection Statistics (positive = >=25% stenosis):")
    lines.append(f"    True Positives:   {result.true_positives}")
    lines.append(f"    True Negatives:   {result.true_negatives}")
    lines.append(f"    False Positives:  {result.false_positives}")
    lines.append(f"    False Negatives:  {result.false_negatives}")
    lines.append(f"    Sensitivity:      {result.sensitivity:.0%}")
    lines.append(f"    Specificity:      {result.specificity:.0%}")

    if result.unmapped_findings:
        lines.append("")
        lines.append("-" * 70)
        lines.append(f"UNMAPPED PIPELINE FINDINGS ({len(result.unmapped_findings)})")
        lines.append("-" * 70)
        lines.append("These findings could not be mapped to a canonical artery:")
        for f in result.unmapped_findings:
            name = f.get("artery_region") or f.get("artery_name") or f"Segment {f.get('segment_id', '?')}"
            lines.append(f"  - {name}: {f.get('stenosis_percent', 0):.1f}% ({f.get('severity', '?')})")

    lines.append("")
    lines.append("=" * 70)

    return "\n".join(lines)
