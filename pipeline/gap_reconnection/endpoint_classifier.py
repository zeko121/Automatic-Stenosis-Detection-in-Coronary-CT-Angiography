"""Classify skeleton endpoints as real vessel tips vs fragmentation gaps."""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, label as scipy_label
from skimage.morphology import skeletonize

logger = logging.getLogger(__name__)


@dataclass
class EndpointInfo:
    """Single endpoint with classification info."""
    position: tuple  # (z, y, x) voxel coords
    is_valid: bool = True  # True = real tip, False = gap
    direction: tuple = None         # unit vec pointing outward
    radius_mm: float = 0.0
    reason: str = ""
    confidence: float = 0.5
    component_label: int = 0


def _find_skeleton_endpoints(skeleton):
    """Find voxels with exactly 1 neighbor (26-connectivity)."""
    skel_points = np.argwhere(skeleton > 0)
    if len(skel_points) < 2:
        return [tuple(pt) for pt in skel_points]

    endpoints = []
    for pt in skel_points:
        z, y, x = pt
        z_lo, z_hi = max(0, z - 1), min(skeleton.shape[0], z + 2)
        y_lo, y_hi = max(0, y - 1), min(skeleton.shape[1], y + 2)
        x_lo, x_hi = max(0, x - 1), min(skeleton.shape[2], x + 2)
        neighborhood = skeleton[z_lo:z_hi, y_lo:y_hi, x_lo:x_hi]
        n_neighbors = int(neighborhood.sum()) - 1
        if n_neighbors == 1:
            endpoints.append((int(z), int(y), int(x)))

    return endpoints


def _estimate_direction(endpoint, skeleton, n_traceback=5):
    """Trace back along skeleton to estimate vessel direction at endpoint."""
    z, y, x = endpoint
    visited = {(z, y, x)}
    path = [(z, y, x)]

    current = (z, y, x)
    for _ in range(n_traceback):
        cz, cy, cx = current
        found_next = False
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = cz + dz, cy + dy, cx + dx
                    if (nz, ny, nx) in visited:
                        continue
                    if (0 <= nz < skeleton.shape[0] and
                            0 <= ny < skeleton.shape[1] and
                            0 <= nx < skeleton.shape[2] and
                            skeleton[nz, ny, nx] > 0):
                        visited.add((nz, ny, nx))
                        path.append((nz, ny, nx))
                        current = (nz, ny, nx)
                        found_next = True
                        break
                if found_next:
                    break
            if found_next:
                break
        if not found_next:
            break

    if len(path) < 2:
        return None

    start = np.array(path[-1], dtype=float)
    end = np.array(path[0], dtype=float)
    direction = end - start
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        return None

    direction = direction / norm
    return (float(direction[0]), float(direction[1]), float(direction[2]))


def _check_boundary_proximity(endpoint, volume_shape, margin_voxels=5):
    for coord, dim_size in zip(endpoint, volume_shape):
        if coord < margin_voxels or coord >= dim_size - margin_voxels:
            return True
    return False

def _check_intensity_beyond(
    endpoint,
    direction,
    image_volume,
    n_samples=5,
    sample_spacing_voxels=2.0,
    intensity_threshold=0.5,
):
    """Sample z-score values beyond endpoint to check if vessel continues."""
    direction = np.array(direction)
    endpoint = np.array(endpoint, dtype=float)

    intensities = []
    for i in range(1, n_samples + 1):
        sample_pos = endpoint + direction * sample_spacing_voxels * i
        sz, sy, sx = int(round(sample_pos[0])), int(round(sample_pos[1])), int(round(sample_pos[2]))
        if (0 <= sz < image_volume.shape[0] and
                0 <= sy < image_volume.shape[1] and
                0 <= sx < image_volume.shape[2]):
            intensities.append(float(image_volume[sz, sy, sx]))

    if not intensities:
        return False, 0.0

    mean_intensity = float(np.mean(intensities))
    continues = mean_intensity > intensity_threshold
    return continues, mean_intensity


def _check_model_probs_beyond(endpoint, direction, model_probs, prob_threshold=0.15,
                               n_samples=5, sample_spacing_voxels=2.0):
    """Check if model probability is elevated beyond the endpoint."""
    direction = np.array(direction, dtype=float)
    endpoint = np.array(endpoint, dtype=float)
    probs = []
    for i in range(1, n_samples + 1):
        pos = endpoint + direction * sample_spacing_voxels * i
        pos = np.round(pos).astype(int)
        if all(0 <= pos[d] < model_probs.shape[d] for d in range(3)):
            probs.append(float(model_probs[pos[0], pos[1], pos[2]]))
    if not probs:
        return False
    return np.mean(probs) > prob_threshold


def classify_endpoints(
    prediction_mask,
    image_volume=None,
    model_probs=None,
    voxel_spacing_mm=0.5,
    boundary_margin=5,
    large_radius_threshold_mm=1.5,
    thin_radius_threshold_mm=0.8,
    intensity_threshold=0.5,
    small_component_fraction=0.1,
):
    """Classify endpoints as valid tips or fragmentation gaps.

    Priority: boundary > radius > intensity > model probs > component size.
    """
    binary_mask = (prediction_mask > 0.5).astype(np.uint8)
    skeleton = skeletonize(binary_mask).astype(np.uint8)
    distance_map = distance_transform_edt(binary_mask) * voxel_spacing_mm

    labeled_mask, n_components = scipy_label(binary_mask)

    endpoint_positions = _find_skeleton_endpoints(skeleton)
    if not endpoint_positions:
        return []

    logger.info(f"Found {len(endpoint_positions)} endpoints in {n_components} components")

    component_sizes = np.bincount(labeled_mask.ravel())
    component_sizes[0] = 0
    largest_component = int(np.max(component_sizes)) if len(component_sizes) > 1 else 0
    small_component_threshold = max(500, int(largest_component * small_component_fraction))

    ep_count_by_comp = {}
    for ep in endpoint_positions:
        z, y, x = ep
        comp = int(labeled_mask[z, y, x])
        ep_count_by_comp[comp] = ep_count_by_comp.get(comp, 0) + 1

    results = []
    for ep in endpoint_positions:
        z, y, x = ep
        info = EndpointInfo(
            position=ep,
            radius_mm=float(distance_map[z, y, x]),
            component_label=int(labeled_mask[z, y, x]),
        )

        info.direction = _estimate_direction(ep, skeleton, n_traceback=5)

        component_size = int(component_sizes[info.component_label]) if info.component_label < len(component_sizes) else 0

        if _check_boundary_proximity(ep, prediction_mask.shape, boundary_margin):
            info.is_valid = True
            info.reason = "boundary"
            info.confidence = 0.9
            results.append(info)
            continue

        # large vessel abruptly stopping = likely a gap
        if info.radius_mm > large_radius_threshold_mm:
            info.is_valid = False
            info.reason = "large_radius_abrupt_stop"
            info.confidence = 0.8
            results.append(info)
            continue

        if image_volume is not None and info.direction is not None:
            continues, mean_intensity = _check_intensity_beyond(
                ep, info.direction, image_volume,
                intensity_threshold=intensity_threshold,
            )
            if continues:
                info.is_valid = False
                info.reason = f"contrast_continues (mean_zscore={mean_intensity:.2f})"
                info.confidence = 0.7
                results.append(info)
                continue

        model_prob_continues = False
        if model_probs is not None and info.direction is not None:
            model_prob_continues = _check_model_probs_beyond(
                ep, info.direction, model_probs, prob_threshold=0.15,
            )
            if model_prob_continues:
                info.is_valid = False
                info.reason = "model_prob_continues"
                info.confidence = 0.65
                results.append(info)
                continue

        # small component with <= 2 endpoints = simple fragment, likely a gap
        if component_size < small_component_threshold:
            n_eps_in_comp = ep_count_by_comp.get(info.component_label, 0)
            if n_eps_in_comp <= 2:
                info.is_valid = False
                info.reason = "small_component_fragment"
                info.confidence = 0.65
                results.append(info)
                continue
            # 3+ endpoints = branching sub-tree, fall through to other checks

        if info.radius_mm < thin_radius_threshold_mm:
            if component_size < small_component_threshold:
                info.is_valid = False
                info.reason = "thin_fragment_tip"
                info.confidence = 0.6
            else:
                info.is_valid = True
                info.reason = "thin_vessel_tip"
                info.confidence = 0.75
            results.append(info)
            continue

        info.is_valid = True
        info.reason = "default"
        info.confidence = 0.5
        results.append(info)

    n_valid = sum(1 for e in results if e.is_valid)
    n_invalid = sum(1 for e in results if not e.is_valid)
    logger.info(f"Classified: {n_valid} valid, {n_invalid} invalid endpoints")

    return results
