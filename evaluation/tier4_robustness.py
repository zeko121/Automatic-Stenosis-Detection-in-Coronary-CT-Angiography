"""Tier 4 -- Domain transfer robustness (ImageCAS vs Ziv structural gap)."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class Tier4Metrics:
    model_name: str = ""
    domain_gap: float = float("nan")

    bifurcation_ratio: float = float("nan")
    length_ratio: float = float("nan")
    voxel_ratio: float = float("nan")
    lcf_diff: float = float("nan")
    radius_mean_ratio: float = float("nan")

    imagecas_bifurcations_mean: float = 0.0
    imagecas_length_mean: float = 0.0
    imagecas_voxels_mean: float = 0.0
    imagecas_lcf_mean: float = 0.0
    imagecas_radius_mean: float = 0.0

    ziv_bifurcations_mean: float = 0.0
    ziv_length_mean: float = 0.0
    ziv_voxels_mean: float = 0.0
    ziv_lcf_mean: float = 0.0
    ziv_radius_mean: float = 0.0

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "domain_gap": self.domain_gap,
            "bifurcation_ratio": self.bifurcation_ratio,
            "length_ratio": self.length_ratio,
            "voxel_ratio": self.voxel_ratio,
            "lcf_diff": self.lcf_diff,
            "radius_mean_ratio": self.radius_mean_ratio,
            "imagecas_bifurcations_mean": self.imagecas_bifurcations_mean,
            "imagecas_length_mean": self.imagecas_length_mean,
            "ziv_bifurcations_mean": self.ziv_bifurcations_mean,
            "ziv_length_mean": self.ziv_length_mean,
        }


def compute_domain_gap(
    imagecas_tier2: Dict[str, float],
    ziv_tier2: Dict[str, float],
    model_name: str = "",
) -> Tier4Metrics:
    """Domain gap from aggregated Tier 2 on ImageCAS vs Ziv."""
    m = Tier4Metrics(model_name=model_name)

    def _safe_ratio(a: float, b: float) -> float:
        if b > 0 and a > 0:
            return a / b
        return 1.0

    def _safe_get(d: Dict[str, float], key: str) -> float:
        return d.get(key, 0.0)

    m.imagecas_bifurcations_mean = _safe_get(imagecas_tier2, "n_bifurcations_mean")
    m.imagecas_length_mean = _safe_get(imagecas_tier2, "total_centerline_length_mm_mean")
    m.imagecas_voxels_mean = _safe_get(imagecas_tier2, "n_components_mean")
    m.imagecas_lcf_mean = _safe_get(imagecas_tier2, "largest_component_fraction_mean")
    m.imagecas_radius_mean = _safe_get(imagecas_tier2, "radius_mean_mm_mean")

    m.ziv_bifurcations_mean = _safe_get(ziv_tier2, "n_bifurcations_mean")
    m.ziv_length_mean = _safe_get(ziv_tier2, "total_centerline_length_mm_mean")
    m.ziv_voxels_mean = _safe_get(ziv_tier2, "n_components_mean")
    m.ziv_lcf_mean = _safe_get(ziv_tier2, "largest_component_fraction_mean")
    m.ziv_radius_mean = _safe_get(ziv_tier2, "radius_mean_mm_mean")

    m.bifurcation_ratio = _safe_ratio(m.ziv_bifurcations_mean, m.imagecas_bifurcations_mean)
    m.length_ratio = _safe_ratio(m.ziv_length_mean, m.imagecas_length_mean)
    m.voxel_ratio = _safe_ratio(m.ziv_voxels_mean, m.imagecas_voxels_mean)
    m.lcf_diff = abs(m.ziv_lcf_mean - m.imagecas_lcf_mean)
    m.radius_mean_ratio = _safe_ratio(m.ziv_radius_mean, m.imagecas_radius_mean)

    m.domain_gap = (
        0.3 * abs(1.0 - m.bifurcation_ratio)
        + 0.3 * abs(1.0 - m.length_ratio)
        + 0.2 * abs(1.0 - m.voxel_ratio)
        + 0.1 * m.lcf_diff
        + 0.1 * abs(1.0 - m.radius_mean_ratio)
    )

    return m
