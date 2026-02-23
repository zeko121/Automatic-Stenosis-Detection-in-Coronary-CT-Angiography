"""Tier 1 -- Segmentation quality metrics (requires GT masks).

Dice, Surface Dice, HD95, ASD, and radius accuracy (MARE/MSRE).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import distance_transform_edt


@dataclass
class Tier1Metrics:
    case_id: str = ""
    dice: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    iou: float = 0.0

    hd95: float = 0.0
    asd: float = 0.0
    surface_dice: Dict[float, float] = field(default_factory=dict)  # tol -> score

    mare: float = float("nan")
    msre: float = float("nan")
    n_radius_samples: int = 0

    def to_dict(self) -> dict:
        d = {
            "case_id": self.case_id,
            "dice": self.dice,
            "precision": self.precision,
            "recall": self.recall,
            "iou": self.iou,
            "hd95": self.hd95,
            "asd": self.asd,
            "mare": self.mare,
            "msre": self.msre,
            "n_radius_samples": self.n_radius_samples,
        }
        for tol, sd in self.surface_dice.items():
            d[f"surface_dice_{tol}mm"] = sd
        return d


def compute_volumetric_metrics(
    model_mask: np.ndarray, gt_mask: np.ndarray
) -> Dict[str, float]:
    """Dice, IoU, Precision, Recall."""
    m = model_mask.astype(bool)
    g = gt_mask.astype(bool)
    intersection = int(np.logical_and(m, g).sum())
    union = int(np.logical_or(m, g).sum())
    m_sum = int(m.sum())
    g_sum = int(g.sum())

    dice = 2 * intersection / (m_sum + g_sum) if (m_sum + g_sum) > 0 else 0.0
    iou = intersection / union if union > 0 else 0.0
    precision = intersection / m_sum if m_sum > 0 else 0.0
    recall = intersection / g_sum if g_sum > 0 else 0.0

    return {"dice": dice, "iou": iou, "precision": precision, "recall": recall}


def _extract_surface(mask: np.ndarray) -> np.ndarray:
    """Extract boundary voxels (foreground with at least one bg neighbor)."""
    from scipy.ndimage import binary_erosion
    eroded = binary_erosion(mask, iterations=1)
    surface = mask & ~eroded
    return surface


def compute_surface_distances(
    model_mask: np.ndarray,
    gt_mask: np.ndarray,
    spacing_mm: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (model_to_gt, gt_to_model) surface distances in mm."""
    m = model_mask.astype(bool)
    g = gt_mask.astype(bool)

    surf_m = _extract_surface(m)
    surf_g = _extract_surface(g)

    surf_m_coords = np.argwhere(surf_m)
    surf_g_coords = np.argwhere(surf_g)

    if len(surf_m_coords) == 0 or len(surf_g_coords) == 0:
        empty = np.array([float("inf")])
        return empty, empty

    dt_g = distance_transform_edt(~surf_g, sampling=spacing_mm)
    dt_m = distance_transform_edt(~surf_m, sampling=spacing_mm)

    m_to_g = dt_g[surf_m]
    g_to_m = dt_m[surf_g]

    return m_to_g, g_to_m


def compute_hd95(m_to_g: np.ndarray, g_to_m: np.ndarray) -> float:
    """95th percentile Hausdorff distance (mm)."""
    all_dists = np.concatenate([m_to_g, g_to_m])
    if len(all_dists) == 0:
        return float("inf")
    return float(np.percentile(all_dists, 95))


def compute_asd(m_to_g: np.ndarray, g_to_m: np.ndarray) -> float:
    """Average symmetric surface distance (mm)."""
    if len(m_to_g) == 0 or len(g_to_m) == 0:
        return float("inf")
    return float((m_to_g.mean() + g_to_m.mean()) / 2.0)


def compute_surface_dice(
    m_to_g: np.ndarray,
    g_to_m: np.ndarray,
    tolerance_mm: float,
) -> float:
    """Fraction of surface voxels within tolerance of the other surface."""
    if len(m_to_g) == 0 or len(g_to_m) == 0:
        return 0.0
    m_within = (m_to_g <= tolerance_mm).sum()
    g_within = (g_to_m <= tolerance_mm).sum()
    total = len(m_to_g) + len(g_to_m)
    return float((m_within + g_within) / total) if total > 0 else 0.0


def compute_radius_accuracy(
    model_mask: np.ndarray,
    gt_mask: np.ndarray,
    gt_centerline_points: np.ndarray,
    spacing_mm: float = 0.5,
) -> Dict[str, float]:
    """MARE and MSRE along GT centerline (positive MSRE = over-segmentation)."""
    if gt_centerline_points is None or len(gt_centerline_points) == 0:
        return {"mare": float("nan"), "msre": float("nan"), "n_samples": 0}

    dt_model = distance_transform_edt(model_mask.astype(bool), sampling=spacing_mm)
    dt_gt = distance_transform_edt(gt_mask.astype(bool), sampling=spacing_mm)

    pts = np.round(gt_centerline_points).astype(int)
    shape = np.array(gt_mask.shape)
    pts = np.clip(pts, 0, shape - 1)

    r_model = dt_model[pts[:, 0], pts[:, 1], pts[:, 2]]
    r_gt = dt_gt[pts[:, 0], pts[:, 1], pts[:, 2]]

    valid = r_gt > 0  # only inside GT vessel
    if valid.sum() == 0:
        return {"mare": float("nan"), "msre": float("nan"), "n_samples": 0}

    r_model_v = r_model[valid]
    r_gt_v = r_gt[valid]
    diff = r_model_v - r_gt_v

    return {
        "mare": float(np.mean(np.abs(diff))),
        "msre": float(np.mean(diff)),
        "n_samples": int(valid.sum()),
    }


def extract_gt_centerline_points(
    gt_mask: np.ndarray, min_vessel_size: int = 100
) -> Optional[np.ndarray]:
    """Skeletonize GT mask, return (N,3) centerline coords or None."""
    from skimage.morphology import skeletonize_3d

    binary = gt_mask.astype(bool)
    if binary.sum() < min_vessel_size:
        return None

    skeleton = skeletonize_3d(binary.astype(np.uint8))
    coords = np.argwhere(skeleton > 0)

    if len(coords) < 10:
        return None
    return coords


def evaluate_case(
    model_mask: np.ndarray,
    gt_mask: np.ndarray,
    case_id: str = "",
    spacing_mm: float = 0.5,
    surface_dice_tolerances: Optional[List[float]] = None,
    compute_radius: bool = True,
) -> Tier1Metrics:
    if surface_dice_tolerances is None:
        surface_dice_tolerances = [0.5, 1.0]

    metrics = Tier1Metrics(case_id=case_id)

    vol = compute_volumetric_metrics(model_mask, gt_mask)
    metrics.dice = vol["dice"]
    metrics.precision = vol["precision"]
    metrics.recall = vol["recall"]
    metrics.iou = vol["iou"]

    m_to_g, g_to_m = compute_surface_distances(model_mask, gt_mask, spacing_mm)
    metrics.hd95 = compute_hd95(m_to_g, g_to_m)
    metrics.asd = compute_asd(m_to_g, g_to_m)

    for tol in surface_dice_tolerances:
        metrics.surface_dice[tol] = compute_surface_dice(m_to_g, g_to_m, tol)

    if compute_radius:
        gt_pts = extract_gt_centerline_points(gt_mask)
        if gt_pts is not None:
            rad = compute_radius_accuracy(model_mask, gt_mask, gt_pts, spacing_mm)
            metrics.mare = rad["mare"]
            metrics.msre = rad["msre"]
            metrics.n_radius_samples = rad["n_samples"]

    return metrics


def aggregate_tier1(results: List[Tier1Metrics]) -> Dict[str, float]:
    """Mean and std of each metric across cases."""
    if not results:
        return {}

    keys = ["dice", "precision", "recall", "iou", "hd95", "asd", "mare", "msre"]
    agg: Dict[str, float] = {}

    for k in keys:
        vals = [getattr(r, k) for r in results if not np.isnan(getattr(r, k))]
        if vals:
            agg[f"{k}_mean"] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))
        else:
            agg[f"{k}_mean"] = float("nan")
            agg[f"{k}_std"] = float("nan")

    all_tols = set()
    for r in results:
        all_tols.update(r.surface_dice.keys())
    for tol in sorted(all_tols):
        vals = [r.surface_dice[tol] for r in results if tol in r.surface_dice]
        if vals:
            agg[f"surface_dice_{tol}mm_mean"] = float(np.mean(vals))
            agg[f"surface_dice_{tol}mm_std"] = float(np.std(vals))

    agg["n_cases"] = len(results)
    return agg
