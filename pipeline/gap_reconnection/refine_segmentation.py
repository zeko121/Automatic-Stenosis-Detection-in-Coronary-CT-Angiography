"""Refinement pipeline: gap-finding, ROI extraction, re-segmentation, and merge."""

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from pipeline.gap_reconnection.endpoint_classifier import EndpointInfo, classify_endpoints
from pipeline.gap_reconnection.gap_connector import (
    GapPair, find_gap_pairs, find_midpoint_gap_pairs, prepare_gap_rois,
)
from pipeline.gap_reconnection.resegment import resegment_gaps, merge_predictions
from pipeline.segment import load_model

logger = logging.getLogger(__name__)

_cached_gap_model = None
_cached_gap_config = None
_cached_gap_dir = None


def _get_or_load_gap_model(checkpoint_path, config_path, device):
    """Load gap model with caching to avoid repeated disk loads."""
    global _cached_gap_model, _cached_gap_config, _cached_gap_dir
    key = str(checkpoint_path)
    if _cached_gap_model is not None and _cached_gap_dir == key:
        logger.info("Using cached gap model")
        _cached_gap_model.to(device)
        _cached_gap_model.train(False)
        return _cached_gap_model, _cached_gap_config
    model, cfg = load_model(str(checkpoint_path), str(config_path), device)
    _cached_gap_model = model
    _cached_gap_config = cfg
    _cached_gap_dir = key
    return model, cfg


@dataclass
class RefinementConfig:
    """Tunable parameters for the refinement pipeline."""
    boundary_margin: int = 5
    large_radius_threshold_mm: float = 1.5
    thin_radius_threshold_mm: float = 0.8
    intensity_threshold: float = 0.5
    max_gap_distance_mm: float = 10.0
    min_alignment_score: float = 0.15
    min_radius_consistency: float = 0.35
    extension_mm: float = 3.0
    margin_mm: float = 5.0
    resegment_threshold: float = 0.3
    min_new_voxels: int = 10
    min_component_size: int = 50
    bce_threshold: float = 0.3
    main_threshold: float = 0.1
    gap_model_dir: str = r"..\..\models\2026-02-12_13-14-10"  # BCE model for gap filling
    voxel_spacing_mm: float = 0.5


def refine_mask(
    image_volume,
    prediction_mask,
    model_dir,
    gap_model_dir=None,
    device="cuda",
    config=None,
    enable=True,
    verbose=True,
    main_model_probs=None,
):
    """Refine segmentation by re-segmenting gap regions with the BCE gap model."""
    t0 = time.time()

    result = {
        "refined_mask": prediction_mask,
        "n_endpoints_found": 0,
        "n_invalid_endpoints": 0,
        "n_gaps_found": 0,
        "n_gaps_connected": 0,
        "n_voxels_added": 0,
        "gap_details": [],
        "endpoints": [],
        "runtime_sec": 0.0,
        "enabled": enable,
    }

    if not enable:
        if verbose:
            logger.info("Refinement disabled, returning original mask")
        return result

    if config is None:
        config = RefinementConfig()

    gap_dir = gap_model_dir or config.gap_model_dir
    gap_dir_path = Path(gap_dir)

    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")

    checkpoint_path = gap_dir_path / "checkpoints" / "best_model.pth"
    config_path = gap_dir_path / "run_config.json"

    if not checkpoint_path.exists():
        logger.error(f"Gap model checkpoint not found: {checkpoint_path}")
        result["runtime_sec"] = round(time.time() - t0, 2)
        return result

    if not config_path.exists():
        logger.error(f"Gap model config not found: {config_path}")
        result["runtime_sec"] = round(time.time() - t0, 2)
        return result

    if verbose:
        logger.info(f"Loading gap model from {gap_dir_path.name}...")

    gap_model, gap_model_config = _get_or_load_gap_model(
        checkpoint_path, config_path, torch_device
    )

    if verbose:
        logger.info(f"Gap model: {gap_model_config.model_type}, device={torch_device}")

    original_voxels = int(prediction_mask.sum())

    if verbose:
        logger.info("Classifying skeleton endpoints...")

    endpoints = classify_endpoints(
        prediction_mask=prediction_mask,
        image_volume=image_volume,
        model_probs=main_model_probs,
        voxel_spacing_mm=config.voxel_spacing_mm,
        boundary_margin=config.boundary_margin,
        large_radius_threshold_mm=config.large_radius_threshold_mm,
        thin_radius_threshold_mm=config.thin_radius_threshold_mm,
        intensity_threshold=config.intensity_threshold,
    )

    result["n_endpoints_found"] = len(endpoints)
    result["n_invalid_endpoints"] = sum(1 for ep in endpoints if not ep.is_valid)
    result["endpoints"] = [
        {
            "position": list(ep.position),
            "is_valid": ep.is_valid,
            "direction": list(ep.direction) if ep.direction else None,
            "radius_mm": ep.radius_mm,
            "reason": ep.reason,
            "confidence": ep.confidence,
            "component_label": ep.component_label,
        }
        for ep in endpoints
    ]

    if verbose:
        logger.info(f"{len(endpoints)} endpoints, {result['n_invalid_endpoints']} invalid")

    if result["n_invalid_endpoints"] < 1:
        if verbose:
            logger.info("No invalid endpoints, skipping refinement")
        result["runtime_sec"] = round(time.time() - t0, 2)
        return result

    if verbose:
        logger.info("Finding gap pairs...")

    gap_pairs = find_gap_pairs(
        endpoints=endpoints,
        prediction_mask=prediction_mask,
        max_gap_distance_mm=config.max_gap_distance_mm,
        min_alignment_score=config.min_alignment_score,
        min_radius_consistency=config.min_radius_consistency,
        voxel_spacing_mm=config.voxel_spacing_mm,
    )

    midpoint_pairs = find_midpoint_gap_pairs(
        endpoints=endpoints,
        prediction_mask=prediction_mask,
        max_gap_distance_mm=config.max_gap_distance_mm,
        min_radius_consistency=config.min_radius_consistency,
        voxel_spacing_mm=config.voxel_spacing_mm,
    )

    already_connected = set()
    for pair in gap_pairs:
        already_connected.add(pair.endpoint_a.component_label)
        already_connected.add(pair.endpoint_b.component_label)

    for pair in midpoint_pairs:
        if pair.endpoint_a.component_label not in already_connected:
            gap_pairs.append(pair)
            already_connected.add(pair.endpoint_a.component_label)

    result["n_gaps_found"] = len(gap_pairs)

    if not gap_pairs:
        if verbose:
            logger.info("No gap pairs found")
        result["runtime_sec"] = round(time.time() - t0, 2)
        return result

    gap_pairs = prepare_gap_rois(
        gap_pairs=gap_pairs,
        volume_shape=prediction_mask.shape,
        voxel_spacing_mm=config.voxel_spacing_mm,
        extension_mm=config.extension_mm,
        margin_mm=config.margin_mm,
        prediction_mask=prediction_mask,
        max_gap_distance_mm=config.max_gap_distance_mm,
    )

    if verbose:
        logger.info(f"{len(gap_pairs)} gap pairs with ROI tubes")
        logger.info("Re-segmenting gap regions...")

    roi_predictions, gap_details = resegment_gaps(
        image_volume=image_volume,
        prediction_mask=prediction_mask,
        gap_pairs=gap_pairs,
        model=gap_model,
        device=torch_device,
        threshold=config.resegment_threshold,
        main_model_probs=main_model_probs,
        bce_threshold=config.bce_threshold,
        main_threshold=config.main_threshold,
        verbose=verbose,
        voxel_spacing_mm=config.voxel_spacing_mm,
    )

    result["gap_details"] = gap_details

    if verbose:
        logger.info("Merging predictions...")

    refined_mask, n_truly_connected = merge_predictions(
        main_mask=prediction_mask,
        roi_predictions=roi_predictions,
        min_new_voxels=config.min_new_voxels,
        min_component_size=config.min_component_size,
        gap_pairs=gap_pairs,
    )

    refined_voxels = int(refined_mask.sum())
    n_voxels_added = refined_voxels - original_voxels
    if n_truly_connected is not None:
        n_gaps_connected = n_truly_connected
    else:
        n_gaps_connected = sum(
            1 for d in gap_details
            if d.get("status") == "success" and d.get("n_new_voxels", 0) >= config.min_new_voxels
        )

    result["refined_mask"] = refined_mask
    result["n_gaps_connected"] = n_gaps_connected
    result["n_voxels_added"] = n_voxels_added
    result["runtime_sec"] = round(time.time() - t0, 2)

    if verbose:
        logger.info(
            f"Refinement: {original_voxels:,} -> {refined_voxels:,} voxels "
            f"(+{n_voxels_added:,}), "
            f"{n_gaps_connected}/{len(gap_pairs)} gaps connected, "
            f"{result['runtime_sec']:.1f}s"
        )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result
