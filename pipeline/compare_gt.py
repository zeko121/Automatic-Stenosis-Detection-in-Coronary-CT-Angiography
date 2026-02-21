"""Compare pipeline findings against ground truth radiologist reports."""

import logging
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)

CANONICAL_ARTERIES = ["LM", "LAD", "LCx", "RCA"]
POSITIVE_THRESHOLD = 25.0  # >=25% is clinically significant
SEVERITY_ORDER = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}

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


def compare_findings(pipeline_findings, gt_report):
    """Compare pipeline findings against GT report."""
    result = ComparisonResult()

    training_labels = gt_report.get("training_labels", {})
    gt_arteries = training_labels.get("arteries", {})

    result.gt_metadata = {
        "serial_number": training_labels.get("serial_number"),
        "scan_date": gt_report.get("report_data", {}).get("scan_date"),
        "gender": training_labels.get("gender"),
        "dominance": training_labels.get("dominance"),
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

    return result


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

    lines.append("-" * 70)
    lines.append("PER-ARTERY COMPARISON")
    lines.append("-" * 70)
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
    lines.append("SUMMARY METRICS")
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
