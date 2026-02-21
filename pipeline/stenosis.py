"""
Stenosis detection from centerline data. Classification into
Normal/Mild/Moderate/Severe based on clinical thresholds.
"""

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d, label as scipy_label

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class StenosisFinding:
    segment_id: int
    location_idx: int
    location_mm: float
    location_voxel: tuple
    stenosis_percent: float
    severity: str
    min_radius_mm: float
    reference_radius_mm: float
    segment_length_mm: float
    confidence: float
    artery_name: str = ""
    artery_region: str = ""
    start_arc_mm: float = 0.0
    end_arc_mm: float = 0.0


@dataclass
class StenosisConfig:
    normal_threshold: float = 0.25       # <25% Normal, 25-50% Mild, 50-70% Moderate, >=70% Severe
    mild_threshold: float = 0.50
    moderate_threshold: float = 0.70

    reference_percentile: float = 75.0
    smoothing_sigma: float = 1.0
    min_segment_points: int = 5
    min_stenosis_length_mm: float = 1.0

    bifurcation_suppression_mm: float = 2.0
    bifurcation_radius_factor: float = 1.5

    reference_exclude_ends_mm: float = 2.0

    endpoint_taper_distance_mm: float = 4.0
    endpoint_taper_fraction: float = 0.05

    stenosis_merge_distance_mm: float = 2.0


class StenosisDetector:

    def __init__(self, config=None):
        self.config = config or StenosisConfig()

    def classify_severity(self, stenosis_pct):
        if stenosis_pct < self.config.normal_threshold * 100:
            return "Normal"
        elif stenosis_pct < self.config.mild_threshold * 100:
            return "Mild"
        elif stenosis_pct < self.config.moderate_threshold * 100:
            return "Moderate"
        else:
            return "Severe"

    def compute_confidence(self, stenosis_pct, segment_length,
                           distance_from_end_mm=None, n_stenosis_points=None):
        base = min(1.0, stenosis_pct / 100.0 + 0.3)
        length_factor = min(1.0, segment_length / 20.0)

        if distance_from_end_mm is not None and n_stenosis_points is not None:
            position_factor = min(1.0, distance_from_end_mm / 5.0)
            points_factor = min(1.0, n_stenosis_points / 10.0)
            return min(1.0, base * (0.5 + 0.2 * length_factor
                                    + 0.15 * position_factor
                                    + 0.15 * points_factor))
        else:
            return min(1.0, base * (0.7 + 0.3 * length_factor))

    def _filter_endpoint_tapering(self, findings, segment_length_mm):
        threshold = max(
            self.config.endpoint_taper_distance_mm,
            segment_length_mm * self.config.endpoint_taper_fraction
        )
        return [f for f in findings
                if f.location_mm > threshold
                and f.location_mm < (segment_length_mm - threshold)]

    def _merge_nearby_stenoses(self, findings):
        if len(findings) <= 1:
            return findings

        findings.sort(key=lambda f: f.start_arc_mm)
        merged = [findings[0]]

        for f in findings[1:]:
            prev = merged[-1]
            gap = f.start_arc_mm - prev.end_arc_mm

            if gap < self.config.stenosis_merge_distance_mm:
                if f.stenosis_percent > prev.stenosis_percent:
                    prev.location_idx = f.location_idx
                    prev.location_mm = f.location_mm
                    prev.location_voxel = f.location_voxel
                prev.end_arc_mm = max(prev.end_arc_mm, f.end_arc_mm)
                prev.min_radius_mm = min(prev.min_radius_mm, f.min_radius_mm)
                prev.stenosis_percent = max(prev.stenosis_percent, f.stenosis_percent)
                prev.severity = self.classify_severity(prev.stenosis_percent)
            else:
                merged.append(f)

        return merged

    def analyze_segment(
        self,
        segment_id,
        radii_mm,
        arc_lengths_mm,
        centerline_points,
        segment_length_mm,
        start_is_bifurcation=False,
        end_is_bifurcation=False,
    ):
        n_points = len(radii_mm)
        if n_points < self.config.min_segment_points:
            logger.debug(f"Segment {segment_id}: skipped (only {n_points} points)")
            return []

        if n_points >= 3 and self.config.smoothing_sigma > 0:
            radii_smooth = gaussian_filter1d(radii_mm, sigma=self.config.smoothing_sigma)
        else:
            radii_smooth = radii_mm.copy()

        arc = np.array(arc_lengths_mm)

        suppress_mask = np.zeros(n_points, dtype=bool)
        if start_is_bifurcation:
            suppress_mask |= (arc < self.config.bifurcation_suppression_mm)
        if end_is_bifurcation:
            suppress_mask |= (arc > segment_length_mm - self.config.bifurcation_suppression_mm)

        local_median = np.median(radii_smooth)
        if local_median > 0:
            suppress_mask |= (radii_smooth > local_median * self.config.bifurcation_radius_factor)

        ref_exclude = self.config.reference_exclude_ends_mm
        valid_ref_mask = (arc >= ref_exclude) & (arc <= segment_length_mm - ref_exclude)
        valid_radii = radii_smooth[valid_ref_mask & ~suppress_mask]

        if len(valid_radii) >= 3:
            reference_radius = np.percentile(valid_radii, self.config.reference_percentile)
        else:
            unsuppressed = radii_smooth[~suppress_mask]
            if len(unsuppressed) >= 1:
                reference_radius = np.percentile(unsuppressed, self.config.reference_percentile)
            else:
                reference_radius = np.percentile(radii_smooth, self.config.reference_percentile)

        if reference_radius < 0.1:
            logger.debug(f"Segment {segment_id}: skipped (reference radius too small: {reference_radius:.3f}mm)")
            return []

        stenosis_pct = np.clip((1.0 - radii_smooth / reference_radius) * 100.0, 0, 100)
        stenosis_pct[suppress_mask] = 0

        stenosis_binary = (stenosis_pct >= self.config.normal_threshold * 100).astype(int)
        labeled, n_regions = scipy_label(stenosis_binary)

        findings = []
        for region_id in range(1, n_regions + 1):
            indices = np.where(labeled == region_id)[0]
            if len(indices) == 0:
                continue

            region_start_mm = arc[indices[0]]
            region_end_mm = arc[indices[-1]]
            length_mm = region_end_mm - region_start_mm
            if length_mm < self.config.min_stenosis_length_mm:
                continue

            peak_idx = indices[np.argmax(stenosis_pct[indices])]
            peak_stenosis = stenosis_pct[peak_idx]
            min_radius = radii_smooth[peak_idx]

            location_mm = arc[peak_idx] if peak_idx < len(arc) else 0.0
            if peak_idx < len(centerline_points):
                voxel = centerline_points[peak_idx]
                location_voxel = (int(voxel[0]), int(voxel[1]), int(voxel[2]))
            else:
                location_voxel = (0, 0, 0)

            dist_from_start = location_mm
            dist_from_end = segment_length_mm - location_mm
            distance_from_end_mm = min(dist_from_start, dist_from_end)

            severity = self.classify_severity(peak_stenosis)
            confidence = self.compute_confidence(
                peak_stenosis, segment_length_mm,
                distance_from_end_mm=distance_from_end_mm,
                n_stenosis_points=len(indices)
            )

            finding = StenosisFinding(
                segment_id=segment_id,
                location_idx=int(peak_idx),
                location_mm=float(location_mm),
                location_voxel=location_voxel,
                stenosis_percent=float(peak_stenosis),
                severity=severity,
                min_radius_mm=float(min_radius),
                reference_radius_mm=float(reference_radius),
                segment_length_mm=float(segment_length_mm),
                confidence=float(confidence),
                start_arc_mm=float(region_start_mm),
                end_arc_mm=float(region_end_mm),
            )
            findings.append(finding)

        findings = self._filter_endpoint_tapering(findings, segment_length_mm)
        findings = self._merge_nearby_stenoses(findings)

        return findings

    def detect(self, centerline_data, artery_labels=None):
        vessel_tree = centerline_data.get("vessel_tree", {})
        segments = vessel_tree.get("segments", {})
        nodes_dict = vessel_tree.get("nodes", {})

        if not segments:
            logger.warning("No segments found in centerline data")
            return []

        all_findings = []

        for seg_id_str, segment in segments.items():
            seg_id = int(seg_id_str)

            radii_mm = np.array(segment.get("radii_mm", []))
            arc_lengths = np.array(segment.get("arc_length_mm", []))
            centerline_pts = np.array(segment.get("centerline_points", []))
            segment_length = segment.get("length_mm", 0.0)

            if len(radii_mm) == 0:
                continue

            node_ids = segment.get("node_ids", [])
            start_node = nodes_dict.get(str(node_ids[0]), {}) if node_ids else {}
            end_node = nodes_dict.get(str(node_ids[-1]), {}) if node_ids else {}
            start_is_bif = start_node.get("is_bifurcation", False)
            end_is_bif = end_node.get("is_bifurcation", False)

            findings = self.analyze_segment(
                segment_id=seg_id,
                radii_mm=radii_mm,
                arc_lengths_mm=arc_lengths,
                centerline_points=centerline_pts,
                segment_length_mm=segment_length,
                start_is_bifurcation=start_is_bif,
                end_is_bifurcation=end_is_bif,
            )

            if artery_labels and seg_id in artery_labels:
                label = artery_labels[seg_id]
                for f in findings:
                    f.artery_name = label.artery_name if hasattr(label, 'artery_name') else str(label.get("artery_name", ""))
                    f.artery_region = label.full_name if hasattr(label, 'full_name') else str(label.get("full_name", ""))

            all_findings.extend(findings)

        all_findings.sort(key=lambda f: f.stenosis_percent, reverse=True)

        return all_findings



def process(
    input_path,
    output_path,
    config=None,
    artery_labels=None,
):
    start_time = time.time()
    logger.info(f"Processing stenosis detection: {input_path}")

    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Centerline file not found: {input_path}")

    with open(input_path, 'r') as f:
        centerline_data = json.load(f)

    config = config or StenosisConfig()
    detector = StenosisDetector(config)

    findings = detector.detect(centerline_data, artery_labels=artery_labels)

    severity_counts = {"Normal": 0, "Mild": 0, "Moderate": 0, "Severe": 0}
    for f in findings:
        severity_counts[f.severity] += 1

    max_severity = "Normal"
    for sev in ["Severe", "Moderate", "Mild", "Normal"]:
        if severity_counts[sev] > 0:
            max_severity = sev
            break

    vessel_tree = centerline_data.get("vessel_tree", {})
    n_segments = len(vessel_tree.get("segments", {}))
    n_nodes = len(vessel_tree.get("nodes", {}))

    output = {
        "summary": {
            "total_findings": len(findings),
            "max_severity": max_severity,
            "by_severity": severity_counts,
            "segments_analyzed": n_segments,
            "nodes_in_tree": n_nodes
        },
        "config": {
            "normal_threshold": config.normal_threshold,
            "mild_threshold": config.mild_threshold,
            "moderate_threshold": config.moderate_threshold,
            "reference_percentile": config.reference_percentile,
            "smoothing_sigma": config.smoothing_sigma,
            "min_segment_points": config.min_segment_points,
            "min_stenosis_length_mm": config.min_stenosis_length_mm,
            "bifurcation_suppression_mm": config.bifurcation_suppression_mm,
            "reference_exclude_ends_mm": config.reference_exclude_ends_mm,
            "endpoint_taper_distance_mm": config.endpoint_taper_distance_mm,
            "stenosis_merge_distance_mm": config.stenosis_merge_distance_mm,
        },
        "findings": [asdict(f) for f in findings]
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    elapsed = time.time() - start_time

    logger.info(f"Done: {len(findings)} findings ({max_severity}) from {n_segments} segments, {elapsed:.2f}s")

    return {
        "findings_count": len(findings),
        "max_severity": max_severity,
        "severity_counts": severity_counts,
        "segments_analyzed": n_segments,
        "output_path": str(output_path),
        "elapsed_seconds": elapsed
    }


if __name__ == "__main__":
    import sys

    test_input = "temp/case-30_centerline.json"
    test_output = "temp/case-30_stenosis.json"

    if len(sys.argv) >= 3:
        test_input = sys.argv[1]
        test_output = sys.argv[2]

    print(f"input:  {test_input}")
    print(f"output: {test_output}")

    result = process(test_input, test_output)
    print(f"\nfindings: {result['findings_count']}, max severity: {result['max_severity']}")
    print(f"by severity: {result['severity_counts']}")
    print(f"time: {result['elapsed_seconds']:.2f}s")
