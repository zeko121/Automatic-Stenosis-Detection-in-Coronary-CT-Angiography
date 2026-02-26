"""Re-run inference on gap ROIs at lower threshold to reconnect fragmented vessels."""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import label as scipy_label
from skimage.graph import route_through_array
from skimage.morphology import remove_small_objects

from pipeline.gap_reconnection.gap_connector import GapPair

logger = logging.getLogger(__name__)


def _extract_patch(image_volume, bbox):
    """Extract sub-volume at the given bbox."""
    (z0, y0, x0), (z1, y1, x1) = bbox
    return image_volume[z0:z1, y0:y1, x0:x1].copy()


def _run_roi_inference(patch, model, device, threshold=0.3):
    """Run inference on a single ROI patch. Falls back to CPU on OOM."""
    import torch
    from monai.inferers import sliding_window_inference

    def _pad_to_divisible(arr, divisor=16):
        pad_widths = []
        for s in arr.shape:
            remainder = s % divisor
            pad_needed = (divisor - remainder) if remainder != 0 else 0
            pad_widths.append((0, pad_needed))
        if all(p == (0, 0) for p in pad_widths):
            return arr, arr.shape
        return np.pad(arr, pad_widths, mode='constant', constant_values=0), arr.shape

    def _infer(dev):
        padded_patch, orig_shape = _pad_to_divisible(patch, divisor=16)
        patch_tensor = torch.from_numpy(padded_patch).float().unsqueeze(0).unsqueeze(0).to(dev)

        with torch.no_grad():
            max_dim = max(padded_patch.shape)
            if max_dim <= 96:
                if dev.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        output = model(patch_tensor)
                else:
                    output = model(patch_tensor)
            else:
                roi_size = tuple(min(96, s) for s in padded_patch.shape)
                roi_size = tuple(max(16, (r // 16) * 16) for r in roi_size)
                if dev.type == "cuda":
                    with torch.amp.autocast("cuda"):
                        output = sliding_window_inference(
                            patch_tensor,
                            roi_size=roi_size,
                            sw_batch_size=1,
                            predictor=model,
                            overlap=0.5,
                            mode="gaussian",
                        )
                else:
                    output = sliding_window_inference(
                        patch_tensor,
                        roi_size=roi_size,
                        sw_batch_size=1,
                        predictor=model,
                        overlap=0.5,
                        mode="gaussian",
                    )
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
        return probs[:orig_shape[0], :orig_shape[1], :orig_shape[2]]

    try:
        probs = _infer(device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("CUDA OOM during ROI inference, falling back to CPU")
        torch.cuda.empty_cache()
        model.cpu()
        cpu_device = torch.device("cpu")
        probs = _infer(cpu_device)
        model.to(device)  # move back after

    mask = (probs > threshold).astype(np.uint8)
    return probs, mask


def _find_bridge_path(
    probs: np.ndarray,
    roi_mask: np.ndarray,
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    directional_weight: float = 0.0,
) -> Tuple[np.ndarray, float]:
    """Dijkstra shortest path through prob map (cost = 1 - prob)."""
    cost = 1.0 - np.clip(probs, 0.0, 0.99)

    # penalize voxels far from the start->end axis
    if directional_weight > 0:
        axis = np.array(end, dtype=float) - np.array(start, dtype=float)
        axis_length = np.linalg.norm(axis)
        normalizer = max(axis_length / 2.0, 1.0)

        if axis_length > 0:
            axis_unit = axis / axis_length
            zz, yy, xx = np.mgrid[
                0:probs.shape[0], 0:probs.shape[1], 0:probs.shape[2]
            ]
            coords_minus_start = np.stack([
                zz - start[0], yy - start[1], xx - start[2],
            ], axis=-1).astype(np.float32)
            proj = np.dot(coords_minus_start, axis_unit.astype(np.float32))
            proj_vec = proj[..., np.newaxis] * axis_unit.astype(np.float32)
            perp_vec = coords_minus_start - proj_vec
            perp_dist = np.linalg.norm(perp_vec, axis=-1)
            cost += directional_weight * (perp_dist / normalizer)

    cost[~roi_mask.astype(bool)] = 10.0  # high cost outside tube but finite

    shape = cost.shape
    s = tuple(np.clip(start, 0, np.array(shape) - 1))
    e = tuple(np.clip(end, 0, np.array(shape) - 1))
    cost[s] = 0.01
    cost[e] = 0.01

    try:
        path_indices, _ = route_through_array(
            cost, s, e, fully_connected=True, geometric=True,
        )
        path_coords = np.array(path_indices, dtype=int)
    except Exception:
        logger.warning("route_through_array failed, straight line fallback")
        n_pts = max(int(np.linalg.norm(np.array(end) - np.array(start))), 2)
        path_coords = np.round(
            np.linspace(start, end, n_pts)
        ).astype(int)

    for dim in range(3):
        path_coords[:, dim] = np.clip(path_coords[:, dim], 0, shape[dim] - 1)

    prob_vals = probs[path_coords[:, 0], path_coords[:, 1], path_coords[:, 2]]
    avg_prob = float(np.mean(prob_vals)) if len(prob_vals) > 0 else 0.0

    return path_coords, avg_prob


def _validate_bridge_path(
    path_coords: np.ndarray,
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    max_sinuosity: float = 1.8,
    max_deviation_voxels: float = 10.0,
) -> Tuple[bool, str]:
    """Check if bridge path is geometrically reasonable."""
    if len(path_coords) < 2:
        return True, ""

    steps = np.diff(path_coords.astype(float), axis=0)
    arc_length = float(np.sum(np.linalg.norm(steps, axis=1)))
    euclidean = float(np.linalg.norm(
        np.array(end, dtype=float) - np.array(start, dtype=float)
    ))
    sinuosity = arc_length / max(euclidean, 1e-6)

    if sinuosity > max_sinuosity:
        return False, (
            f"sinuosity {sinuosity:.2f} > {max_sinuosity:.1f} "
            f"(arc={arc_length:.1f}, euclid={euclidean:.1f})"
        )

    axis = np.array(end, dtype=float) - np.array(start, dtype=float)
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-6:
        return True, ""

    axis_unit = axis / axis_length
    pts = path_coords.astype(float) - np.array(start, dtype=float)
    proj = np.dot(pts, axis_unit)
    proj_vecs = proj[:, np.newaxis] * axis_unit
    perp_vecs = pts - proj_vecs
    perp_dists = np.linalg.norm(perp_vecs, axis=1)
    max_dev = float(np.max(perp_dists))

    if max_dev > max_deviation_voxels:
        return False, (
            f"max_deviation {max_dev:.1f} voxels > {max_deviation_voxels:.1f}"
        )

    return True, ""


def _dilate_path_to_bridge(
    path_coords: np.ndarray,
    volume_shape: Tuple[int, int, int],
    radius_a: float,
    radius_b: float,
) -> np.ndarray:
    """Dilate path into a tube with linearly interpolated radius."""
    bridge = np.zeros(volume_shape, dtype=np.uint8)
    n_pts = len(path_coords)

    if n_pts == 0:
        return bridge

    radius_a = max(radius_a, 1.5)  # min 0.75mm for connectivity
    radius_b = max(radius_b, 1.5)

    for i, (z, y, x) in enumerate(path_coords):
        t = i / max(n_pts - 1, 1)
        r = radius_a * (1 - t) + radius_b * t
        r_int = max(int(round(r)), 1)

        z_lo = max(0, z - r_int)
        z_hi = min(volume_shape[0], z + r_int + 1)
        y_lo = max(0, y - r_int)
        y_hi = min(volume_shape[1], y + r_int + 1)
        x_lo = max(0, x - r_int)
        x_hi = min(volume_shape[2], x + r_int + 1)

        for dz in range(z_lo, z_hi):
            for dy in range(y_lo, y_hi):
                for dx in range(x_lo, x_hi):
                    if (dz - z)**2 + (dy - y)**2 + (dx - x)**2 <= r_int**2:
                        bridge[dz, dy, dx] = 1

    return bridge


def _build_probability_bridge(
    combined_probs: np.ndarray,
    roi_mask: np.ndarray,
    existing_mask: np.ndarray,
    start: Tuple[int, int, int],
    end: Tuple[int, int, int],
    base_threshold: float = 0.3,
    min_threshold: float = 0.15,
) -> Tuple[Optional[np.ndarray], Dict]:
    """Build bridge by thresholding the prob map directly.

    Adapts threshold downward until endpoints are connected.
    """
    from scipy.ndimage import label as ndlabel

    roi_bool = roi_mask.astype(bool)
    existing_bool = existing_mask.astype(bool)
    threshold = base_threshold
    metadata = {"strategy": "probability_threshold", "threshold_used": threshold}

    while threshold >= min_threshold - 1e-6:
        candidate = (combined_probs > threshold) & roi_bool & ~existing_bool
        connected_region = candidate | existing_bool

        labeled, _ = ndlabel(connected_region)
        s = tuple(int(np.clip(c, 0, combined_probs.shape[i] - 1)) for i, c in enumerate(start))
        e = tuple(int(np.clip(c, 0, combined_probs.shape[i] - 1)) for i, c in enumerate(end))

        label_s = labeled[s]
        label_e = labeled[e]

        if label_s > 0 and label_e > 0 and label_s == label_e:
            metadata["threshold_used"] = round(threshold, 3)
            metadata["connected"] = True
            metadata["n_candidate_voxels"] = int(candidate.sum())
            return candidate.astype(np.uint8), metadata

        threshold -= 0.05

    metadata["connected"] = False
    metadata["threshold_used"] = round(base_threshold, 3)
    return None, metadata


def resegment_gaps(
    image_volume: np.ndarray,
    prediction_mask: np.ndarray,
    gap_pairs: List[GapPair],
    model,
    device,
    threshold: float = 0.3,
    main_model_probs: Optional[np.ndarray] = None,
    bce_threshold: float = 0.3,
    main_threshold: float = 0.1,
    verbose: bool = True,
    voxel_spacing_mm: float = 0.5,
    min_bridge_prob: float = 0.6,
    return_probs: bool = False,
) -> Tuple[np.ndarray, List[Dict]]:
    """Re-segment gap regions with model-guided bridge paths."""
    gap_details = []
    roi_predictions = []

    for i, pair in enumerate(gap_pairs):
        if pair.roi_bbox is None or pair.roi_mask is None:
            logger.warning(f"Gap {i}: missing ROI, skipping")
            gap_details.append({"gap_index": i, "status": "skipped", "reason": "no_roi"})
            roi_predictions.append(None)
            continue

        bbox = pair.roi_bbox
        (z0, y0, x0), (z1, y1, x1) = bbox

        detail = {
            "gap_index": i,
            "distance_mm": pair.distance_mm,
            "alignment_score": pair.alignment_score,
            "is_midpoint_pair": pair.is_midpoint_pair,
            "bbox": [list(bbox[0]), list(bbox[1])],
        }

        tube_mean_prob = None
        if main_model_probs is not None and pair.roi_mask is not None:
            main_probs_in_tube = main_model_probs[z0:z1, y0:y1, x0:x1]
            ms = tuple(min(a, b) for a, b in zip(main_probs_in_tube.shape, pair.roi_mask.shape))
            tube_probs = main_probs_in_tube[:ms[0], :ms[1], :ms[2]][
                pair.roi_mask[:ms[0], :ms[1], :ms[2]]
            ]
            if len(tube_probs) > 0:
                tube_mean_prob = float(np.mean(tube_probs))
                detail["tube_mean_prob"] = round(tube_mean_prob, 4)

                if tube_mean_prob < 0.03:
                    detail["status"] = "skipped"
                    detail["skip_reason"] = f"tube_occupancy {tube_mean_prob:.3f} < 0.03"
                    roi_predictions.append(None)
                    gap_details.append(detail)
                    continue

        try:
            patch = _extract_patch(image_volume, bbox)

            bce_probs, new_mask = _run_roi_inference(patch, model, device, threshold=threshold)

            existing_region = prediction_mask[z0:z1, y0:y1, x0:x1]
            roi_mask = pair.roi_mask

            if new_mask.shape != roi_mask.shape or new_mask.shape != existing_region.shape:
                logger.warning(
                    f"Gap {i}: shape mismatch, cropping to common shape"
                )
            min_shape = tuple(
                min(a, b, c)
                for a, b, c in zip(new_mask.shape, roi_mask.shape, existing_region.shape)
            )
            new_mask = new_mask[:min_shape[0], :min_shape[1], :min_shape[2]]
            bce_probs_cropped = bce_probs[:min_shape[0], :min_shape[1], :min_shape[2]]
            roi_cropped = roi_mask[:min_shape[0], :min_shape[1], :min_shape[2]]
            existing_cropped = existing_region[:min_shape[0], :min_shape[1], :min_shape[2]]

            # max of BCE and main model probs
            combined_probs = bce_probs_cropped.copy()
            if main_model_probs is not None:
                main_probs_region = main_model_probs[z0:z1, y0:y1, x0:x1]
                main_probs_region = main_probs_region[:min_shape[0], :min_shape[1], :min_shape[2]]
                combined_probs = np.maximum(combined_probs, main_probs_region)

            pos_a = np.array(pair.endpoint_a.position, dtype=int) - np.array([z0, y0, x0])
            pos_b = np.array(pair.endpoint_b.position, dtype=int) - np.array([z0, y0, x0])

            for dim in range(3):
                pos_a[dim] = np.clip(pos_a[dim], 0, min_shape[dim] - 1)
                pos_b[dim] = np.clip(pos_b[dim], 0, min_shape[dim] - 1)

            radius_a_vox = max(pair.endpoint_a.radius_mm / voxel_spacing_mm, 1.0)
            radius_b_vox = max(pair.endpoint_b.radius_mm / voxel_spacing_mm, 1.0)

            effective_min_prob = min_bridge_prob
            if tube_mean_prob is not None and 0.03 <= tube_mean_prob < 0.08:
                effective_min_prob = max(min_bridge_prob, 0.8)
                detail["tube_tier"] = "low_signal"
            elif tube_mean_prob is not None and tube_mean_prob >= 0.08:
                detail["tube_tier"] = "normal"
            else:
                detail["tube_tier"] = "unknown"

            tube_combined = combined_probs[roi_cropped.astype(bool)]
            mean_tube_combined = float(np.mean(tube_combined)) if len(tube_combined) > 0 else 0.0
            detail["mean_tube_combined_prob"] = round(mean_tube_combined, 4)

            prob_bridge_used = False
            if mean_tube_combined >= 0.15:
                # try probability thresholding first (preserves vessel shape)
                prob_bridge, prob_meta = _build_probability_bridge(
                    combined_probs, roi_cropped, existing_cropped,
                    tuple(pos_a), tuple(pos_b),
                    base_threshold=0.2, min_threshold=0.10,
                )
                detail.update({k: v for k, v in prob_meta.items() if k != "strategy"})

                if prob_bridge is not None:
                    masked_new = prob_bridge & roi_cropped & (existing_cropped == 0)
                    n_new_voxels = int(masked_new.sum())

                    euclid_dist = float(np.linalg.norm(
                        np.array(pos_b, dtype=float) - np.array(pos_a, dtype=float)
                    ))
                    avg_radius = (radius_a_vox + radius_b_vox) / 2.0
                    expected_cylinder = np.pi * avg_radius**2 * max(euclid_dist, 1.0)
                    if n_new_voxels > 3 * expected_cylinder and n_new_voxels > 100:  # too large
                        detail["prob_bridge_too_large"] = True
                        detail["prob_bridge_voxels"] = n_new_voxels
                        detail["expected_cylinder"] = int(expected_cylinder)
                    elif n_new_voxels >= 10:
                        detail["merge_strategy"] = "probability_threshold"
                        prob_bridge_used = True
                        if n_new_voxels > 0:
                            bridge_probs = combined_probs[masked_new.astype(bool)]
                            avg_prob = float(np.mean(bridge_probs))
                        else:
                            avg_prob = mean_tube_combined
                        detail["avg_probability"] = round(avg_prob, 4)

                if not prob_bridge_used:
                    detail["prob_threshold_failed"] = True

            if not prob_bridge_used:
                if mean_tube_combined >= 0.1:
                    euclid = float(np.linalg.norm(
                        np.array(pos_b, dtype=float) - np.array(pos_a, dtype=float)
                    ))

                    path_coords, avg_prob = _find_bridge_path(
                        combined_probs, roi_cropped, tuple(pos_a), tuple(pos_b),
                        directional_weight=0.3,
                    )

                    max_dev = max(2.0 * max(radius_a_vox, radius_b_vox), 5.0)
                    is_valid, reason = _validate_bridge_path(
                        path_coords, tuple(pos_a), tuple(pos_b),
                        max_sinuosity=1.8, max_deviation_voxels=max_dev,
                    )

                    steps = np.diff(path_coords.astype(float), axis=0)
                    arc_len = float(np.sum(np.linalg.norm(steps, axis=1))) if len(path_coords) > 1 else 0.0
                    sinuosity = arc_len / max(euclid, 1e-6)
                    detail["sinuosity"] = round(sinuosity, 3)
                    detail["path_length"] = len(path_coords)
                    detail["directional_weight"] = 0.3

                    if not is_valid:
                        n_pts = max(int(euclid), 2)
                        path_coords = np.round(np.linspace(pos_a, pos_b, n_pts)).astype(int)
                        for dim in range(3):
                            path_coords[:, dim] = np.clip(path_coords[:, dim], 0, min_shape[dim] - 1)
                        prob_vals = combined_probs[path_coords[:, 0], path_coords[:, 1], path_coords[:, 2]]
                        avg_prob = float(np.mean(prob_vals)) if len(prob_vals) > 0 else 0.0

                    detail["avg_probability"] = round(avg_prob, 4)

                    if avg_prob < effective_min_prob:
                        masked_new = np.zeros(min_shape, dtype=np.uint8)
                        detail["merge_strategy"] = "skipped_low_probability"
                        detail["skip_reason"] = f"avg_prob {avg_prob:.3f} < {effective_min_prob}"
                        n_new_voxels = 0
                    else:
                        thin_r_a = radius_a_vox * 0.7
                        thin_r_b = radius_b_vox * 0.7
                        bridge = _dilate_path_to_bridge(
                            path_coords, min_shape, thin_r_a, thin_r_b,
                        )
                        masked_new = bridge & roi_cropped & (existing_cropped == 0)
                        detail["merge_strategy"] = (
                            "dijkstra_thin" if is_valid else "dijkstra_thin_straight_fallback"
                        )
                        if not is_valid:
                            detail["fallback_reason"] = reason
                        n_new_voxels = int(masked_new.sum())
                else:
                    masked_new = np.zeros(min_shape, dtype=np.uint8)
                    detail["merge_strategy"] = "skipped_low_signal"
                    detail["skip_reason"] = f"mean_tube_combined {mean_tube_combined:.3f} < 0.1"
                    avg_prob = mean_tube_combined
                    detail["avg_probability"] = round(avg_prob, 4)
                    n_new_voxels = 0

            detail["n_new_voxels"] = n_new_voxels
            detail["status"] = "success"

            if return_probs:
                detail["bce_probs_roi"] = bce_probs_cropped
                detail["combined_probs_roi"] = combined_probs
                detail["roi_mask_for_probs"] = roi_cropped

            if verbose:
                logger.info(
                    f"  Gap {i}: dist={pair.distance_mm:.1f}mm, "
                    f"midpoint_pair={pair.is_midpoint_pair}, "
                    f"pos_a={pair.endpoint_a.position}, "
                    f"pos_b={pair.endpoint_b.position}, "
                    f"comp_a={pair.endpoint_a.component_label}, "
                    f"comp_b={pair.endpoint_b.component_label}, "
                    f"new_voxels={n_new_voxels}, "
                    f"merge={detail['merge_strategy']}, avg_prob={avg_prob:.3f}"
                )

            roi_predictions.append((bbox, masked_new.astype(np.uint8)))

        except Exception as e:
            detail["status"] = "error"
            detail["error"] = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Gap {i} failed: {detail['error']}")
            roi_predictions.append(None)

        gap_details.append(detail)

    return roi_predictions, gap_details


def _find_component_label_near(labeled, position, search_radius=3):
    """Find a non-zero component label at or near position."""
    z, y, x = int(position[0]), int(position[1]), int(position[2])
    if 0 <= z < labeled.shape[0] and 0 <= y < labeled.shape[1] and 0 <= x < labeled.shape[2]:
        lbl = labeled[z, y, x]
        if lbl > 0:
            return int(lbl)

    for r in range(1, search_radius + 1):
        z_lo = max(0, z - r)
        z_hi = min(labeled.shape[0], z + r + 1)
        y_lo = max(0, y - r)
        y_hi = min(labeled.shape[1], y + r + 1)
        x_lo = max(0, x - r)
        x_hi = min(labeled.shape[2], x + r + 1)
        region = labeled[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
        labels_in_region = region[region > 0]
        if len(labels_in_region) > 0:
            return int(np.bincount(labels_in_region).argmax())
    return 0


def merge_predictions(main_mask, roi_predictions, min_new_voxels=10,
                      min_component_size=50, gap_pairs=None):
    """Merge ROI predictions into main mask with connectivity verification."""
    merged = main_mask.copy().astype(np.uint8)

    n_added_total = 0
    n_rois_applied = 0
    applied_rois = []

    for idx, entry in enumerate(roi_predictions):
        if entry is None:
            continue
        bbox, new_voxels = entry
        n_new = int(new_voxels.sum())
        if n_new < min_new_voxels:
            logger.debug(f"Skipping ROI with only {n_new} new voxels (min={min_new_voxels})")
            continue

        (z0, y0, x0), (z1, y1, x1) = bbox
        region = merged[z0:z1, y0:y1, x0:x1]
        min_shape = tuple(min(a, b) for a, b in zip(region.shape, new_voxels.shape))
        region[:min_shape[0], :min_shape[1], :min_shape[2]] |= new_voxels[
            :min_shape[0], :min_shape[1], :min_shape[2]
        ]
        merged[z0:z1, y0:y1, x0:x1] = region

        n_added_total += n_new
        n_rois_applied += 1
        applied_rois.append((idx, bbox, new_voxels))

    n_truly_connected = None
    if gap_pairs is not None and applied_rois:
        labeled, _ = scipy_label(merged > 0)
        n_truly_connected = 0
        n_partial = 0

        for roi_idx, bbox, new_voxels in applied_rois:
            if roi_idx >= len(gap_pairs):
                continue
            pair = gap_pairs[roi_idx]
            if pair is None:
                continue

            pos_a = pair.endpoint_a.position
            pos_b = pair.endpoint_b.position

            label_a = _find_component_label_near(labeled, pos_a)
            label_b = _find_component_label_near(labeled, pos_b)

            if label_a > 0 and label_b > 0 and label_a == label_b:
                n_truly_connected += 1
                logger.info(f"ROI {roi_idx}: connected (label {label_a})")
            else:
                n_partial += 1
                logger.info(f"ROI {roi_idx}: partial bridge ({label_a} vs {label_b})")

        logger.info(f"Connectivity: {n_truly_connected} connected, {n_partial} partial")

    if min_component_size > 0 and n_added_total > 0:
        merged_bool = merged.astype(bool)
        merged_clean = remove_small_objects(merged_bool, min_size=min_component_size)
        merged = merged_clean.astype(np.uint8)

    logger.info(f"Merged {n_rois_applied} ROIs, +{n_added_total} voxels, final: {merged.sum():,}")
    return merged, n_truly_connected
