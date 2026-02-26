"""Find broken endpoint pairs and build cylindrical ROI tubes for re-segmentation."""

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_edt, label as scipy_label
from scipy.spatial import KDTree
from skimage.morphology import skeletonize

from pipeline.gap_reconnection.endpoint_classifier import EndpointInfo

logger = logging.getLogger(__name__)


@dataclass
class GapPair:
    """Two endpoints that should be reconnected."""
    endpoint_a: EndpointInfo
    endpoint_b: EndpointInfo
    distance_mm: float
    alignment_score: float = 0.0
    is_midpoint_pair: bool = False
    roi_bbox: tuple = None  # (min_corner, max_corner)
    roi_mask: np.ndarray = None


def _direction_alignment(dir_a, dir_b, pos_a, pos_b):
    """Score how well two endpoints face each other (0=bad, 1=perfect)."""
    if dir_a is None or dir_b is None:
        return 0.5

    dir_a = np.array(dir_a, dtype=float)
    dir_b = np.array(dir_b, dtype=float)
    pos_a = np.array(pos_a, dtype=float)
    pos_b = np.array(pos_b, dtype=float)

    ab = pos_b - pos_a
    ab_norm = np.linalg.norm(ab)
    if ab_norm < 1e-6:
        return 0.0
    ab = ab / ab_norm

    dot_a = float(np.dot(dir_a, ab))   # A should point toward B
    dot_b = float(np.dot(dir_b, -ab))  # B should point toward A

    score = max(0.0, (dot_a + dot_b) / 2.0)
    return score


def _radius_consistency(radius_a, radius_b):
    if radius_a <= 0 or radius_b <= 0:
        return 0.5
    ratio = min(radius_a, radius_b) / max(radius_a, radius_b)
    return ratio

def find_gap_pairs(
    endpoints,
    prediction_mask,
    max_gap_distance_mm=15.0,
    min_alignment_score=0.04,
    min_radius_consistency=0.3,
    voxel_spacing_mm=0.5,
):
    """Find invalid endpoint pairs that could be reconnected."""
    invalid_eps = [ep for ep in endpoints if not ep.is_valid]
    if len(invalid_eps) < 2:
        logger.info(f"Only {len(invalid_eps)} invalid endpoints, need at least 2")
        return []

    logger.info(f"Finding gap pairs among {len(invalid_eps)} invalid endpoints")

    positions = np.array([ep.position for ep in invalid_eps], dtype=float)
    max_gap_voxels = max_gap_distance_mm / voxel_spacing_mm
    tree = KDTree(positions)

    candidate_pairs = []
    pairs_seen = set()

    n_too_far = 0
    n_alignment_fail = 0
    n_radius_fail = 0
    n_same_component = 0

    for i in range(len(invalid_eps)):
        neighbors = tree.query_ball_point(positions[i], r=max_gap_voxels)
        for j in neighbors:
            if i >= j:
                continue
            pair_key = (i, j)
            if pair_key in pairs_seen:
                continue
            pairs_seen.add(pair_key)

            ep_a = invalid_eps[i]
            ep_b = invalid_eps[j]

            alignment = _direction_alignment(ep_a.direction, ep_b.direction, ep_a.position, ep_b.position)
            if alignment < min_alignment_score:
                n_alignment_fail += 1
                continue

            consistency = _radius_consistency(ep_a.radius_mm, ep_b.radius_mm)
            if consistency < min_radius_consistency:
                n_radius_fail += 1
                continue

            if ep_a.component_label == ep_b.component_label and ep_a.component_label > 0:
                n_same_component += 1
                continue

            distance_mm = float(np.linalg.norm(positions[i] - positions[j])) * voxel_spacing_mm
            candidate_pairs.append(GapPair(
                endpoint_a=ep_a,
                endpoint_b=ep_b,
                distance_mm=distance_mm,
                alignment_score=alignment,
            ))

    n_total_possible = len(invalid_eps) * (len(invalid_eps) - 1) // 2
    n_in_range = len(pairs_seen)
    n_too_far = n_total_possible - n_in_range

    if not candidate_pairs:
        logger.info(
            f"No candidate gap pairs after filtering. "
            f"{n_total_possible} possible, {n_too_far} too far (>{max_gap_distance_mm}mm), "
            f"{n_in_range} in range, {n_alignment_fail} bad alignment, "
            f"{n_radius_fail} bad radius, {n_same_component} same component"
        )
        return []

    # greedy MST-like selection to avoid redundant connections
    candidate_pairs.sort(key=lambda p: p.distance_mm)
    parent = {}

    def find(x):
        if x not in parent:
            parent[x] = x
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        parent[ra] = rb
        return True

    selected = []
    for pair in candidate_pairs:
        label_a = pair.endpoint_a.component_label
        label_b = pair.endpoint_b.component_label
        if union(label_a, label_b):
            selected.append(pair)

    logger.info(f"Selected {len(selected)} gap pairs from {len(candidate_pairs)} candidates")
    return selected



def find_midpoint_gap_pairs(
    endpoints,
    prediction_mask,
    max_gap_distance_mm=15.0,
    min_radius_consistency=0.3,
    voxel_spacing_mm=0.5,
    connected_component_labels=None,
):
    """Connect small-component endpoints to skeleton midpoints on larger components."""
    skeleton = skeletonize(prediction_mask > 0)
    dist_map = distance_transform_edt(prediction_mask > 0)

    if connected_component_labels is None:
        labeled, n_components = scipy_label(prediction_mask > 0)
    else:
        labeled = connected_component_labels
        n_components = int(labeled.max())

    if n_components < 2:
        logger.info("Only 1 component, no midpoint gap pairs needed")
        return []

    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0

    comp_skel = {}  # label -> (skeleton_pts, KDTree)
    for comp_label in range(1, n_components + 1):
        skel_mask = skeleton & (labeled == comp_label)
        pts = np.argwhere(skel_mask)
        if len(pts) > 0:
            comp_skel[comp_label] = (pts, KDTree(pts.astype(float)))

    max_gap_voxels = max_gap_distance_mm / voxel_spacing_mm

    eps_by_comp = {}
    for ep in endpoints:
        eps_by_comp.setdefault(ep.component_label, []).append(ep)

    candidate_pairs = []

    for src_label in range(1, n_components + 1):
        src_size = component_sizes[src_label]
        src_eps = eps_by_comp.get(src_label, [])
        if not src_eps:
            continue

        target_labels = [
            lbl for lbl in comp_skel
            if lbl != src_label and component_sizes[lbl] > src_size
        ]
        if not target_labels:
            continue

        for ep in src_eps:
            main_label = int(np.argmax(component_sizes))
            if src_label == main_label:
                continue  # never reconnect the largest component outward
            is_small = src_size < component_sizes[main_label] * 0.5
            if not is_small and ep.is_valid:
                continue

            pos = np.array(ep.position, dtype=float)
            best_dist = float('inf')
            best_target_label = None
            best_target_idx = None
            best_target_pts = None

            for tgt_label in target_labels:
                pts, tree = comp_skel[tgt_label]
                dist, idx = tree.query(pos)
                if dist < best_dist:
                    best_dist = dist
                    best_target_label = tgt_label
                    best_target_idx = idx
                    best_target_pts = pts

            if best_target_pts is None or best_dist > max_gap_voxels:
                continue

            target_pos = tuple(int(c) for c in best_target_pts[best_target_idx])
            target_radius_mm = float(dist_map[target_pos]) * voxel_spacing_mm

            consistency = _radius_consistency(ep.radius_mm, target_radius_mm)
            if consistency < min_radius_consistency:
                continue

            target_ep = EndpointInfo(
                position=target_pos,
                is_valid=False,
                direction=None,
                radius_mm=target_radius_mm,
                reason="midpoint_target",
                confidence=0.5,
                component_label=best_target_label,
            )

            distance_mm = float(best_dist) * voxel_spacing_mm

            candidate_pairs.append(GapPair(
                endpoint_a=ep,
                endpoint_b=target_ep,
                distance_mm=distance_mm,
                alignment_score=0.0,
                is_midpoint_pair=True,
            ))

    if not candidate_pairs:
        logger.info("No midpoint gap pairs found")
        return []

    # each source component connects at most once (shortest wins)
    candidate_pairs.sort(key=lambda p: p.distance_mm)
    connected_components = set()
    selected = []
    for pair in candidate_pairs:
        comp = pair.endpoint_a.component_label
        if comp in connected_components:
            continue
        connected_components.add(comp)
        selected.append(pair)

    logger.info(f"Selected {len(selected)} midpoint pairs from {len(candidate_pairs)} candidates")
    return selected


def extract_roi_tube(
    pos_a,
    pos_b,
    dir_a,
    dir_b,
    volume_shape,
    radius_a_mm=1.0,
    radius_b_mm=1.0,
    voxel_spacing_mm=0.5,
    extension_mm=3.0,
    margin_mm=5.0,
):
    """Build a cylindrical ROI mask between two endpoints."""
    pos_a = np.array(pos_a, dtype=float)
    pos_b = np.array(pos_b, dtype=float)

    axis = pos_b - pos_a
    axis_length = np.linalg.norm(axis)
    if axis_length < 1e-6:
        center = pos_a.astype(int)
        r = int(np.ceil((max(radius_a_mm, radius_b_mm) + margin_mm) / voxel_spacing_mm))
        bbox_min = tuple(max(0, int(c) - r) for c in center)
        bbox_max = tuple(min(s, int(c) + r + 1) for c, s in zip(center, volume_shape))
        shape = tuple(b - a for a, b in zip(bbox_min, bbox_max))
        mask = np.ones(shape, dtype=bool)
        return mask, (bbox_min, bbox_max)

    axis_unit = axis / axis_length

    extension_voxels = extension_mm / voxel_spacing_mm
    extended_a = pos_a - axis_unit * extension_voxels
    extended_b = pos_b + axis_unit * extension_voxels

    tube_radius_voxels = (max(radius_a_mm, radius_b_mm) + margin_mm) / voxel_spacing_mm

    pad = int(np.ceil(tube_radius_voxels)) + 1
    all_points = np.stack([extended_a, extended_b])
    bbox_min = tuple(max(0, int(np.floor(all_points[:, i].min())) - pad) for i in range(3))
    bbox_max = tuple(min(volume_shape[i], int(np.ceil(all_points[:, i].max())) + pad + 1) for i in range(3))

    shape = tuple(b - a for a, b in zip(bbox_min, bbox_max))
    if any(s <= 0 for s in shape):
        return np.zeros((1, 1, 1), dtype=bool), (bbox_min, bbox_max)

    zz, yy, xx = np.mgrid[
        bbox_min[0]:bbox_max[0],
        bbox_min[1]:bbox_max[1],
        bbox_min[2]:bbox_max[2],
    ]
    coords = np.stack([zz, yy, xx], axis=-1).astype(float)

    vec_to_a = coords - extended_a
    t = np.sum(vec_to_a * axis_unit, axis=-1)
    total_length = np.linalg.norm(extended_b - extended_a)

    projection = extended_a + np.outer(t.ravel(), axis_unit).reshape(coords.shape)
    perp_dist = np.linalg.norm(coords - projection, axis=-1)

    roi_mask = (perp_dist <= tube_radius_voxels) & (t >= 0) & (t <= total_length)

    return roi_mask, (bbox_min, bbox_max)


def build_component_roi(
    prediction_mask,
    source_component_label,
    volume_shape,
    max_gap_distance_mm=15.0,
    margin_mm=5.0,
    voxel_spacing_mm=0.5,
):
    """Build an expanded bbox ROI around a connected component."""
    labeled, _ = scipy_label(prediction_mask > 0)
    component_voxels = np.argwhere(labeled == source_component_label)

    if len(component_voxels) == 0:
        logger.warning(f"Component {source_component_label} has no voxels, fallback ROI")
        center = np.array(volume_shape) // 2
        r = 5
        bbox_min = tuple(max(0, int(c) - r) for c in center)
        bbox_max = tuple(min(s, int(c) + r + 1) for c, s in zip(center, volume_shape))
        shape = tuple(b - a for a, b in zip(bbox_min, bbox_max))
        return np.ones(shape, dtype=bool), (bbox_min, bbox_max)

    expand_voxels = int(np.ceil((max_gap_distance_mm + margin_mm) / voxel_spacing_mm))

    comp_min = component_voxels.min(axis=0)
    comp_max = component_voxels.max(axis=0)

    bbox_min = tuple(max(0, int(comp_min[i]) - expand_voxels) for i in range(3))
    bbox_max = tuple(min(volume_shape[i], int(comp_max[i]) + expand_voxels + 1) for i in range(3))

    shape = tuple(b - a for a, b in zip(bbox_min, bbox_max))
    roi_mask = np.ones(shape, dtype=bool)

    logger.debug(
        f"Component ROI label={source_component_label}: "
        f"bbox={bbox_min}->{bbox_max}, expand={expand_voxels}vox"
    )
    return roi_mask, (bbox_min, bbox_max)


def prepare_gap_rois(gap_pairs, volume_shape, voxel_spacing_mm=0.5,
                     extension_mm=3.0, margin_mm=5.0, prediction_mask=None,
                     max_gap_distance_mm=15.0):
    """Compute ROI tube for each gap pair (modifies pairs in place)."""
    for pair in gap_pairs:
        roi_mask, bbox = extract_roi_tube(
            pos_a=pair.endpoint_a.position,
            pos_b=pair.endpoint_b.position,
            dir_a=pair.endpoint_a.direction,
            dir_b=pair.endpoint_b.direction,
            volume_shape=volume_shape,
            radius_a_mm=pair.endpoint_a.radius_mm,
            radius_b_mm=pair.endpoint_b.radius_mm,
            voxel_spacing_mm=voxel_spacing_mm,
            extension_mm=extension_mm,
            margin_mm=margin_mm,
        )
        pair.roi_bbox = bbox
        pair.roi_mask = roi_mask

    logger.info(f"Prepared ROIs for {len(gap_pairs)} gap pairs")
    return gap_pairs
