"""Composite scoring and model ranking across all 4 tiers."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ModelScore:
    model_name: str = ""
    tier1_score: float = 0.0
    tier2_score: float = 0.0
    tier3_score: float = 0.0
    tier4_score: float = 0.0

    composite_score: float = 0.0
    rank: int = 0

    ziv_sensitivity: float = 0.0
    surface_dice_05mm: float = 0.0
    n_bifurcations: float = 0.0

    tier1_raw: Dict[str, float] = field(default_factory=dict)
    tier2_raw: Dict[str, float] = field(default_factory=dict)
    tier3_raw: Dict[str, float] = field(default_factory=dict)
    tier4_raw: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "tier1_score": self.tier1_score,
            "tier2_score": self.tier2_score,
            "tier3_score": self.tier3_score,
            "tier4_score": self.tier4_score,
            "composite_score": self.composite_score,
            "rank": self.rank,
            "ziv_sensitivity": self.ziv_sensitivity,
            "surface_dice_05mm": self.surface_dice_05mm,
            "n_bifurcations": self.n_bifurcations,
        }


def normalize_tier1(tier1_agg: Dict[str, float]) -> float:
    """Weighted combo of Dice, SurfDice, ASD, MARE, HD95 -> [0,1]."""
    dice = tier1_agg.get("dice_mean", 0.0)
    sd05 = tier1_agg.get("surface_dice_0.5mm_mean", 0.0)
    asd = tier1_agg.get("asd_mean", 0.0)
    mare = tier1_agg.get("mare_mean", 0.0)
    hd95 = tier1_agg.get("hd95_mean", 0.0)

    if np.isnan(mare):
        mare = 2.0  # worst case
    if np.isnan(asd):
        asd = 3.0
    if np.isnan(hd95):
        hd95 = 10.0

    return (
        0.30 * dice
        + 0.25 * sd05
        + 0.15 * max(0.0, 1.0 - asd / 3.0)
        + 0.15 * max(0.0, 1.0 - mare / 2.0)
        + 0.15 * max(0.0, 1.0 - hd95 / 10.0)
    )


def normalize_tier2(tier2_agg: Dict[str, float]) -> float:
    """Normalize Tier 2 to [0, 1]. Uses the continuity score directly."""
    return tier2_agg.get("continuity_score_mean", 0.0)


def normalize_tier3(tier3_dict: Dict[str, float]) -> float:
    # side-level is primary; fall back to per-artery if not computed (None)
    ziv_sens = tier3_dict.get("ziv_side_sensitivity")
    if ziv_sens is None:
        ziv_sens = tier3_dict.get("ziv_sensitivity", 0.0)
    ziv_spec = tier3_dict.get("ziv_side_specificity")
    if ziv_spec is None:
        ziv_spec = tier3_dict.get("ziv_specificity", 0.0)
    oracle_agree = tier3_dict.get("oracle_severity_agreement", 0.0)
    miss_rate = tier3_dict.get("oracle_miss_rate", 0.0)
    halluc_rate = tier3_dict.get("oracle_hallucination_rate", 0.0)

    return (
        0.30 * ziv_sens
        + 0.20 * ziv_spec
        + 0.20 * oracle_agree
        + 0.15 * (1.0 - miss_rate)
        + 0.15 * (1.0 - halluc_rate)
    )


def normalize_tier4(tier4_dict: Dict[str, float]) -> float:
    gap = tier4_dict.get("domain_gap", 0.5)
    if np.isnan(gap):
        gap = 0.5
    return max(0.0, 1.0 - min(1.0, gap))


def compute_composite(
    tier1_score: float,
    tier2_score: float,
    tier3_score: float,
    tier4_score: float,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """Weighted composite score."""
    if weights is None:
        weights = {"tier1": 0.25, "tier2": 0.15, "tier3": 0.45, "tier4": 0.15}

    return (
        weights["tier1"] * tier1_score
        + weights["tier2"] * tier2_score
        + weights["tier3"] * tier3_score
        + weights["tier4"] * tier4_score
    )


def rank_models(
    models: Dict[str, Dict[str, Dict[str, float]]],
    weights: Optional[Dict[str, float]] = None,
) -> List[ModelScore]:
    """Score and rank models, sorted by composite score (descending)."""
    scores: List[ModelScore] = []

    for name, tiers in models.items():
        t1 = tiers.get("tier1", {})
        t2 = tiers.get("tier2", {})
        t3 = tiers.get("tier3", {})
        t4 = tiers.get("tier4", {})

        ms = ModelScore(model_name=name)
        ms.tier1_score = normalize_tier1(t1)
        ms.tier2_score = normalize_tier2(t2)
        ms.tier3_score = normalize_tier3(t3)
        ms.tier4_score = normalize_tier4(t4)
        ms.composite_score = compute_composite(
            ms.tier1_score, ms.tier2_score, ms.tier3_score, ms.tier4_score, weights
        )

        # prefer side-level sensitivity when available
        ziv_side = t3.get("ziv_side_sensitivity")
        ms.ziv_sensitivity = ziv_side if ziv_side is not None else t3.get("ziv_sensitivity", 0.0)
        ms.surface_dice_05mm = t1.get("surface_dice_0.5mm_mean", 0.0)
        ms.n_bifurcations = t2.get("n_bifurcations_mean", 0.0)

        ms.tier1_raw = t1
        ms.tier2_raw = t2
        ms.tier3_raw = t3
        ms.tier4_raw = t4

        scores.append(ms)

    scores.sort(
        key=lambda s: (
            s.composite_score,
            s.ziv_sensitivity,
            s.surface_dice_05mm,
            s.n_bifurcations,
        ),
        reverse=True,
    )

    for i, s in enumerate(scores, 1):
        s.rank = i

    return scores
