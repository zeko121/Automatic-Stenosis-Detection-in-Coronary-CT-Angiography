"""Tier 2 -- Structural/topological quality (no GT needed).

Components, bifurcations, centerline length, radius plausibility, continuity score.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Tier2Metrics:
    case_id: str = ""
    n_components: int = 0
    largest_component_voxels: int = 0
    total_voxels: int = 0
    largest_component_fraction: float = 0.0

    n_bifurcations: int = 0
    n_endpoints: int = 0
    bifurcation_to_endpoint_ratio: float = 0.0
    total_centerline_length_mm: float = 0.0
    n_segments: int = 0

    radius_min_mm: float = 0.0
    radius_max_mm: float = 0.0
    radius_mean_mm: float = 0.0
    radius_plausibility: float = 0.0  # fraction in [0.2, 5.0] mm

    continuity_score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "n_components": self.n_components,
            "largest_component_voxels": self.largest_component_voxels,
            "total_voxels": self.total_voxels,
            "largest_component_fraction": self.largest_component_fraction,
            "n_bifurcations": self.n_bifurcations,
            "n_endpoints": self.n_endpoints,
            "bifurcation_to_endpoint_ratio": self.bifurcation_to_endpoint_ratio,
            "total_centerline_length_mm": self.total_centerline_length_mm,
            "n_segments": self.n_segments,
            "radius_min_mm": self.radius_min_mm,
            "radius_max_mm": self.radius_max_mm,
            "radius_mean_mm": self.radius_mean_mm,
            "radius_plausibility": self.radius_plausibility,
            "continuity_score": self.continuity_score,
        }


def compute_component_metrics(mask: np.ndarray) -> Dict[str, int]:
    from scipy.ndimage import label as scipy_label

    binary = mask.astype(bool)
    total = int(binary.sum())
    if total == 0:
        return {
            "n_components": 0,
            "largest_component_voxels": 0,
            "total_voxels": 0,
        }

    labeled, n_components = scipy_label(binary)
    if n_components == 0:
        return {
            "n_components": 0,
            "largest_component_voxels": 0,
            "total_voxels": total,
        }

    comp_sizes = np.bincount(labeled.ravel())[1:]
    largest = int(comp_sizes.max())

    return {
        "n_components": int(n_components),
        "largest_component_voxels": largest,
        "total_voxels": total,
    }


def compute_skeleton_metrics(
    mask: np.ndarray, centerline_config=None
) -> Dict[str, float]:
    from pipeline.centerline import extract_vessel_tree

    tree, _skeleton, _dt = extract_vessel_tree(mask, config=centerline_config)

    all_radii = []
    for seg in tree.segments.values():
        if seg.radii_smooth_mm:
            all_radii.extend(seg.radii_smooth_mm)
        elif seg.radii_mm:
            all_radii.extend(seg.radii_mm)

    radii_arr = np.array(all_radii) if all_radii else np.array([0.0])

    if len(all_radii) > 0:
        in_range = ((radii_arr >= 0.2) & (radii_arr <= 5.0)).sum()
        radius_plausibility = float(in_range / len(radii_arr))
    else:
        radius_plausibility = 0.0

    bif_to_ep = (
        tree.num_bifurcations / tree.num_endpoints
        if tree.num_endpoints > 0
        else 0.0
    )

    return {
        "n_bifurcations": tree.num_bifurcations,
        "n_endpoints": tree.num_endpoints,
        "bifurcation_to_endpoint_ratio": bif_to_ep,
        "total_centerline_length_mm": tree.total_length_mm,
        "n_segments": len(tree.segments),
        "radius_min_mm": float(radii_arr.min()) if len(radii_arr) > 0 else 0.0,
        "radius_max_mm": float(radii_arr.max()) if len(radii_arr) > 0 else 0.0,
        "radius_mean_mm": float(radii_arr.mean()) if len(radii_arr) > 0 else 0.0,
        "radius_plausibility": radius_plausibility,
    }


def compute_continuity_score(
    largest_component_fraction: float,
    n_bifurcations: int,
    total_length_mm: float,
    radius_plausibility: float,
    expected_bifurcations: int = 15,
    expected_length_mm: float = 200.0,
) -> float:
    """Weighted composite of LCF, bifurcations, length, radius plausibility."""
    return (
        0.4 * largest_component_fraction
        + 0.3 * min(1.0, n_bifurcations / expected_bifurcations)
        + 0.2 * min(1.0, total_length_mm / expected_length_mm)
        + 0.1 * radius_plausibility
    )


def evaluate_case(
    mask: np.ndarray,
    case_id: str = "",
    centerline_config=None,
    run_postprocess: bool = False,
    postprocess_config=None,
) -> Tier2Metrics:
    if run_postprocess:
        from pipeline.postprocess import postprocess_mask
        mask, _pp_metrics = postprocess_mask(mask, config=postprocess_config, verbose=False)

    metrics = Tier2Metrics(case_id=case_id)

    comp = compute_component_metrics(mask)
    metrics.n_components = comp["n_components"]
    metrics.largest_component_voxels = comp["largest_component_voxels"]
    metrics.total_voxels = comp["total_voxels"]
    metrics.largest_component_fraction = (
        comp["largest_component_voxels"] / comp["total_voxels"]
        if comp["total_voxels"] > 0
        else 0.0
    )

    if comp["total_voxels"] > 0:
        skel = compute_skeleton_metrics(mask, centerline_config)
        metrics.n_bifurcations = skel["n_bifurcations"]
        metrics.n_endpoints = skel["n_endpoints"]
        metrics.bifurcation_to_endpoint_ratio = skel["bifurcation_to_endpoint_ratio"]
        metrics.total_centerline_length_mm = skel["total_centerline_length_mm"]
        metrics.n_segments = skel["n_segments"]
        metrics.radius_min_mm = skel["radius_min_mm"]
        metrics.radius_max_mm = skel["radius_max_mm"]
        metrics.radius_mean_mm = skel["radius_mean_mm"]
        metrics.radius_plausibility = skel["radius_plausibility"]

    metrics.continuity_score = compute_continuity_score(
        largest_component_fraction=metrics.largest_component_fraction,
        n_bifurcations=metrics.n_bifurcations,
        total_length_mm=metrics.total_centerline_length_mm,
        radius_plausibility=metrics.radius_plausibility,
    )

    return metrics


def aggregate_tier2(results: List[Tier2Metrics]) -> Dict[str, float]:
    if not results:
        return {}

    keys = [
        "n_components", "largest_component_fraction",
        "n_bifurcations", "n_endpoints", "bifurcation_to_endpoint_ratio",
        "total_centerline_length_mm", "n_segments",
        "radius_mean_mm", "radius_plausibility", "continuity_score",
    ]
    agg: Dict[str, float] = {}

    for k in keys:
        vals = [float(getattr(r, k)) for r in results]
        agg[f"{k}_mean"] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))

    agg["n_cases"] = len(results)
    return agg
