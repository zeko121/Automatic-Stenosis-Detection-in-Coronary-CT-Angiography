"""
Label coronary arteries via connected-component analysis of the vessel mask.

Classifies left vs right coronary by x-position in RAS space, then
attempts LAD/LCx split at the first major bifurcation.
"""

import logging
import json
import time
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.ndimage import generate_binary_structure, label as scipy_label
from scipy.spatial import KDTree

logger = logging.getLogger(__name__)


@dataclass
class ArteryLabel:
    segment_id: int
    artery_name: str          # e.g. "Left Coronary", "Right Coronary", "LAD"
    region: str = ""          # "proximal", "mid", "distal", or ""
    full_name: str = ""
    tree_side: str = ""       # "left" or "right"
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class LabelingResult:
    labels: dict = field(default_factory=dict)
    left_ostium: list = None
    right_ostium: list = None
    aorta_center: list = None
    aorta_detected: bool = False
    ostia_detected: bool = False
    tree_sides: dict = field(default_factory=dict)
    elapsed_seconds: float = 0.0

    def to_dict(self):
        return {
            "labels": {str(k): v.to_dict() for k, v in self.labels.items()},
            "left_ostium": self.left_ostium,
            "right_ostium": self.right_ostium,
            "aorta_center": self.aorta_center,
            "aorta_detected": self.aorta_detected,
            "ostia_detected": self.ostia_detected,
            "tree_sides": {str(k): v for k, v in self.tree_sides.items()},
            "elapsed_seconds": self.elapsed_seconds,
        }


def map_segments_to_components(centerline_data, vessel_mask):
    """Map each centerline segment to a connected component by midpoint lookup."""
    struct = generate_binary_structure(3, 3)  # 26-connectivity
    labeled_mask, n_components = scipy_label(
        (vessel_mask > 0).astype(np.uint8), structure=struct
    )

    vessel_tree = centerline_data.get("vessel_tree", {})
    segments = vessel_tree.get("segments", {})

    seg_to_comp = {}

    for seg_id_str, seg in segments.items():
        sid = int(seg_id_str)
        pts = seg.get("centerline_points", [])
        if not pts:
            seg_to_comp[sid] = 0
            continue

        pts_arr = np.array(pts, dtype=float)
        n_pts = len(pts_arr)

        mid_idx = n_pts // 2
        mid_vox = np.round(pts_arr[mid_idx]).astype(int)
        comp_label = _safe_lookup(labeled_mask, mid_vox)

        if comp_label > 0:
            seg_to_comp[sid] = int(comp_label)
            continue

        sample_indices = [n_pts // 4, n_pts // 2, 3 * n_pts // 4]
        votes = []
        for idx in sample_indices:
            if idx < n_pts:
                vox = np.round(pts_arr[idx]).astype(int)
                cl = _safe_lookup(labeled_mask, vox)
                if cl > 0:
                    votes.append(cl)

        if votes:
            seg_to_comp[sid] = int(max(set(votes), key=votes.count))
        else:
            seg_to_comp[sid] = 0

    return seg_to_comp, labeled_mask, n_components


def _safe_lookup(labeled_mask, voxel):
    """Look up voxel in labeled mask; returns 0 if out of bounds."""
    shape = labeled_mask.shape
    if (voxel >= 0).all() and voxel[0] < shape[0] and voxel[1] < shape[1] and voxel[2] < shape[2]:
        return int(labeled_mask[voxel[0], voxel[1], voxel[2]])
    return 0


def classify_components(labeled_mask, n_components):
    """Classify components as left/right coronary by mean x-position in RAS."""
    if n_components == 0:
        return {}

    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0  # ignore background
    total_voxels = sizes.sum()

    if total_voxels == 0:
        return {}

    min_significant = max(1, int(total_voxels * 0.05))

    significant = []
    minor = []
    for comp_label in range(1, n_components + 1):
        if sizes[comp_label] >= min_significant:
            significant.append(comp_label)
        elif sizes[comp_label] > 0:
            minor.append(comp_label)

    result = {}

    for comp_label in minor:
        result[comp_label] = "Minor Vessel"

    if len(significant) == 0:
        for comp_label in minor:
            result[comp_label] = "Coronary"
        return result

    if len(significant) == 1:
        result[significant[0]] = "Coronary"
        return result

    # lower x in RAS = patient-left = left coronary
    comp_mean_x = {}
    for comp_label in significant:
        coords = np.argwhere(labeled_mask == comp_label)
        comp_mean_x[comp_label] = coords[:, 2].mean()

    sorted_comps = sorted(significant, key=lambda c: comp_mean_x[c])

    result[sorted_comps[0]] = "Left Coronary"
    result[sorted_comps[-1]] = "Right Coronary"

    if len(sorted_comps) > 2:
        left_x = comp_mean_x[sorted_comps[0]]
        right_x = comp_mean_x[sorted_comps[-1]]
        for comp_label in sorted_comps[1:-1]:
            cx = comp_mean_x[comp_label]
            if abs(cx - left_x) <= abs(cx - right_x):
                result[comp_label] = "Left Coronary"
            else:
                result[comp_label] = "Right Coronary"

    return result


def merge_nearby_components(
    labeled_mask,
    n_components,
    distance_threshold_mm=8.0,
    spacing_mm=0.5,
):
    """Merge nearby components (same-system fragments are typically 3-15mm apart)."""
    if n_components <= 1:
        return labeled_mask, n_components

    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0
    total_voxels = sizes.sum()

    if total_voxels == 0:
        return labeled_mask, n_components

    min_significant = max(1, int(total_voxels * 0.05))
    significant = [c for c in range(1, n_components + 1) if sizes[c] >= min_significant]

    if len(significant) <= 1:
        return labeled_mask, n_components

    distance_threshold_vox = distance_threshold_mm / spacing_mm

    max_sample = 2000
    comp_coords = {}
    for comp_label in significant:
        coords = np.argwhere(labeled_mask == comp_label)
        if len(coords) > max_sample:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(coords), max_sample, replace=False)
            coords = coords[idx]
        comp_coords[comp_label] = coords

    parent = {c: c for c in significant}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if sizes[ra] >= sizes[rb]:
            parent[rb] = ra
        else:
            parent[ra] = rb

    trees = {}
    for comp_label in significant:
        trees[comp_label] = KDTree(comp_coords[comp_label])

    for i, ca in enumerate(significant):
        for cb in significant[i + 1:]:
            if find(ca) == find(cb):
                continue  # already merged
            dists, _ = trees[ca].query(comp_coords[cb], k=1)
            min_dist = dists.min()
            if min_dist < distance_threshold_vox:
                union(ca, cb)
                logger.debug("merging components %d and %d (min dist: %.1f vox = %.1f mm)",
                             ca, cb, min_dist, min_dist * spacing_mm)

    roots = {find(c) for c in significant}
    if len(roots) == len(significant):
        return labeled_mask, n_components

    new_mask = labeled_mask.copy()
    label_remap = {}
    for comp_label in significant:
        root = find(comp_label)
        if root != comp_label:
            new_mask[labeled_mask == comp_label] = root
            label_remap[comp_label] = root

    unique_labels = sorted(set(new_mask[new_mask > 0]))
    compact_map = {old: new for new, old in enumerate(unique_labels, 1)}
    compact_mask = np.zeros_like(new_mask)
    for old, new in compact_map.items():
        compact_mask[new_mask == old] = new

    new_n = len(unique_labels)
    logger.info("component merge: %d -> %d components (threshold: %.1f mm)",
                n_components, new_n, distance_threshold_mm)
    return compact_mask, new_n


def _remap_segments_after_merge(seg_to_comp, labeled_mask, centerline_data):
    """Re-lookup segment->component after relabeling."""
    vessel_tree = centerline_data.get("vessel_tree", {})
    segments = vessel_tree.get("segments", {})
    new_seg_to_comp = {}

    for seg_id_str, seg in segments.items():
        sid = int(seg_id_str)
        pts = seg.get("centerline_points", [])
        if not pts:
            new_seg_to_comp[sid] = 0
            continue

        pts_arr = np.array(pts, dtype=float)
        n_pts = len(pts_arr)

        mid_vox = np.round(pts_arr[n_pts // 2]).astype(int)
        comp_label = _safe_lookup(labeled_mask, mid_vox)

        if comp_label > 0:
            new_seg_to_comp[sid] = int(comp_label)
            continue

        sample_indices = [n_pts // 4, n_pts // 2, 3 * n_pts // 4]
        votes = []
        for idx in sample_indices:
            if idx < n_pts:
                vox = np.round(pts_arr[idx]).astype(int)
                cl = _safe_lookup(labeled_mask, vox)
                if cl > 0:
                    votes.append(cl)

        if votes:
            new_seg_to_comp[sid] = int(max(set(votes), key=votes.count))
        else:
            new_seg_to_comp[sid] = 0

    return new_seg_to_comp


@dataclass
class RootedNode:
    node_id: int
    position: np.ndarray
    radius_mm: float
    parent_id: int = None
    children: list = field(default_factory=list)
    depth: int = 0
    distance_from_root_mm: float = 0.0


def build_rooted_tree(centerline_data, root_position, spacing_mm=0.5):
    """BFS from nearest node to root_position, building parent-child tree."""
    vessel_tree = centerline_data.get("vessel_tree", {})
    nodes = vessel_tree.get("nodes", {})
    segments = vessel_tree.get("segments", {})

    if not nodes:
        return {}

    node_positions = {}
    node_radii = {}
    node_adj = {}

    for nid_str, node in nodes.items():
        nid = int(nid_str)
        pos = np.array(node.get("position", [0, 0, 0]), dtype=float)
        node_positions[nid] = pos
        node_radii[nid] = node.get("radius_mm", 0)
        node_adj.setdefault(nid, set())

    for seg in segments.values():
        nids = seg.get("node_ids", [])
        for i in range(len(nids) - 1):
            node_adj.setdefault(nids[i], set()).add(nids[i + 1])
            node_adj.setdefault(nids[i + 1], set()).add(nids[i])

    all_ids = list(node_positions.keys())
    all_pos = np.array([node_positions[nid] for nid in all_ids])
    dists = np.linalg.norm(all_pos - root_position, axis=1)
    root_id = all_ids[np.argmin(dists)]

    rooted = {}
    queue = deque([(root_id, None, 0, 0.0)])
    visited = set()

    while queue:
        nid, parent_id, depth, dist_mm = queue.popleft()
        if nid in visited:
            continue
        visited.add(nid)

        rn = RootedNode(
            node_id=nid,
            position=node_positions[nid],
            radius_mm=node_radii[nid],
            parent_id=parent_id,
            depth=depth,
            distance_from_root_mm=dist_mm,
        )
        rooted[nid] = rn

        if parent_id is not None and parent_id in rooted:
            rooted[parent_id].children.append(nid)

        for nb in node_adj.get(nid, []):
            if nb not in visited:
                edge_len = np.linalg.norm(
                    node_positions[nb] - node_positions[nid]
                ) * spacing_mm
                queue.append((nb, nid, depth + 1, dist_mm + edge_len))

    return rooted


def find_segment_for_node(node_id, segments):
    for seg_id_str, seg in segments.items():
        if node_id in seg.get("node_ids", []):
            return int(seg_id_str)
    return None


def find_major_bifurcation(rooted_tree, root_id, min_child_ratio=0.35):
    """Walk from root to first node where 2+ children have substantial radius."""
    current = root_id
    visited = set()

    while current is not None:
        if current in visited:
            break
        visited.add(current)

        node = rooted_tree.get(current)
        if node is None:
            break

        children = node.children
        if len(children) >= 2:
            parent_r = node.radius_mm
            large_children = [
                c for c in children
                if rooted_tree[c].radius_mm >= min_child_ratio * parent_r
            ]
            if len(large_children) >= 2:
                return current

        if children:
            best_child = max(children, key=lambda c: rooted_tree[c].radius_mm)
            current = best_child
        else:
            break

    return None


def get_direction_vector(rooted_tree, start_id, max_steps=20):
    """Direction from start_id going distally (follows largest-radius child)."""
    positions = [rooted_tree[start_id].position.copy()]
    current = start_id
    for _ in range(max_steps):
        children = rooted_tree[current].children
        if not children:
            break
        best = max(children, key=lambda c: rooted_tree[c].radius_mm)
        positions.append(rooted_tree[best].position.copy())
        current = best

    if len(positions) < 2:
        return np.array([0, 0, 0], dtype=float)

    positions = np.array(positions)
    direction = positions[-1] - positions[0]
    norm = np.linalg.norm(direction)
    if norm > 0:
        direction /= norm
    return direction


def _collect_subtree_nodes(rooted_tree, start_id):
    nodes = set()
    queue = deque([start_id])
    while queue:
        nid = queue.popleft()
        if nid in nodes:
            continue
        nodes.add(nid)
        node = rooted_tree.get(nid)
        if node:
            for child in node.children:
                queue.append(child)
    return nodes


def assign_regions(labels, segments, rooted_tree, spacing_mm=0.5):
    """Assign proximal/mid/distal based on arc-length fraction."""
    artery_segments = {}
    for sid, label in labels.items():
        artery_segments.setdefault(label.artery_name, []).append(sid)

    for artery_name, seg_ids in artery_segments.items():
        if artery_name in ("Left Coronary", "Right Coronary", "Minor Vessel", "Coronary"):
            for sid in seg_ids:
                labels[sid].full_name = labels[sid].artery_name
            continue

        if any(c.isdigit() for c in artery_name) and artery_name not in ("LM",):
            for sid in seg_ids:
                labels[sid].full_name = labels[sid].artery_name
            continue

        total_length = 0.0
        seg_lengths = {}
        seg_cumulative_start = {}

        def _seg_root_dist(sid):
            seg = segments.get(str(sid), {})
            nids = seg.get("node_ids", [])
            if nids and nids[0] in rooted_tree:
                return rooted_tree[nids[0]].distance_from_root_mm
            return float('inf')

        sorted_sids = sorted(seg_ids, key=_seg_root_dist)

        cumulative = 0.0
        for sid in sorted_sids:
            seg = segments.get(str(sid), {})
            length = seg.get("length_mm", 0.0)
            seg_lengths[sid] = length
            seg_cumulative_start[sid] = cumulative
            cumulative += length
        total_length = cumulative

        if total_length <= 0:
            for sid in seg_ids:
                labels[sid].full_name = labels[sid].artery_name
            continue

        for sid in sorted_sids:
            mid_pos = seg_cumulative_start[sid] + seg_lengths[sid] / 2.0
            fraction = mid_pos / total_length

            if fraction < 0.33:
                region = "proximal"
            elif fraction < 0.67:
                region = "mid"
            else:
                region = "distal"

            labels[sid].region = region
            labels[sid].full_name = f"{region} {labels[sid].artery_name}"


def _try_lad_lcx_split(centerline_data, left_segment_ids, labels, spacing_mm=0.5):
    """Best-effort LAD/LCx split using direction vectors at the first bifurcation."""
    vessel_tree = centerline_data.get("vessel_tree", {})
    nodes = vessel_tree.get("nodes", {})
    segments = vessel_tree.get("segments", {})

    if not left_segment_ids or not nodes:
        return

    left_node_ids = set()
    for sid in left_segment_ids:
        seg = segments.get(str(sid), {})
        left_node_ids.update(seg.get("node_ids", []))

    if not left_node_ids:
        return

    best_radius = -1.0
    best_pos = None
    for nid in left_node_ids:
        node = nodes.get(str(nid), {})
        is_ep = node.get("is_endpoint", False)
        radius = node.get("radius_mm", 0)
        if is_ep and radius > best_radius:
            best_radius = radius
            best_pos = node.get("position", None)

    if best_pos is None:
        return

    root_pos = np.array(best_pos, dtype=float)

    tree = build_rooted_tree(centerline_data, root_pos, spacing_mm)
    if not tree:
        return

    root_id = min(tree.keys(), key=lambda k: tree[k].depth)

    bif_id = find_major_bifurcation(tree, root_id)
    if bif_id is None:
        return

    bif_node = tree[bif_id]
    children = bif_node.children
    if len(children) < 2:
        return

    child_info = []
    for child_id in children:
        direction = get_direction_vector(tree, child_id)
        radius = tree[child_id].radius_mm
        child_info.append((child_id, direction, radius))

    child_info.sort(key=lambda x: x[2], reverse=True)

    c0_id, c0_dir, _ = child_info[0]
    c1_id, c1_dir, _ = child_info[1]

    # LAD goes more anterior (higher y in RAS)
    score_0 = c0_dir[1] - c0_dir[2] * 0.5
    score_1 = c1_dir[1] - c1_dir[2] * 0.5

    if score_0 >= score_1:
        lad_root, lcx_root = c0_id, c1_id
    else:
        lad_root, lcx_root = c1_id, c0_id

    lad_nodes = _collect_subtree_nodes(tree, lad_root)
    lcx_nodes = _collect_subtree_nodes(tree, lcx_root)

    lm_nodes = set()
    current = root_id
    visited_path = set()
    while current is not None:
        if current in visited_path:
            break
        visited_path.add(current)
        lm_nodes.add(current)
        if current == bif_id:
            break
        ch = tree[current].children
        if not ch:
            break
        current = max(ch, key=lambda c: tree[c].radius_mm)

    for sid in left_segment_ids:
        seg = segments.get(str(sid), {})
        seg_nids = set(seg.get("node_ids", []))

        lm_overlap = len(seg_nids & lm_nodes)
        lad_overlap = len(seg_nids & lad_nodes)
        lcx_overlap = len(seg_nids & lcx_nodes)

        best_overlap = max(lm_overlap, lad_overlap, lcx_overlap)
        if best_overlap == 0:
            continue

        if lm_overlap == best_overlap and lm_overlap > len(seg_nids) * 0.3:
            labels[sid].artery_name = "LM"
            labels[sid].tree_side = "left"
            labels[sid].confidence = 0.7
            labels[sid].reason = "Left main (root to first bifurcation)"
        elif lad_overlap == best_overlap and lad_overlap > len(seg_nids) * 0.3:
            labels[sid].artery_name = "LAD"
            labels[sid].tree_side = "left"
            labels[sid].confidence = 0.7
            labels[sid].reason = "LAD (anterior branch at LM bifurcation)"
        elif lcx_overlap == best_overlap and lcx_overlap > len(seg_nids) * 0.3:
            labels[sid].artery_name = "LCx"
            labels[sid].tree_side = "left"
            labels[sid].confidence = 0.7
            labels[sid].reason = "LCx (lateral branch at LM bifurcation)"

    combined_labels = {sid: labels[sid] for sid in left_segment_ids if labels[sid].artery_name in ("LM", "LAD", "LCx")}
    if combined_labels:
        assign_regions(combined_labels, segments, tree, spacing_mm)
        for sid, lbl in combined_labels.items():
            labels[sid] = lbl

    logger.info("LAD/LCx split: %d LM, %d LAD, %d LCx segments",
                sum(1 for s in left_segment_ids if labels[s].artery_name == "LM"),
                sum(1 for s in left_segment_ids if labels[s].artery_name == "LAD"),
                sum(1 for s in left_segment_ids if labels[s].artery_name == "LCx"))


def label_arteries(
    centerline_data,
    ct_volume=None,
    vessel_mask=None,
    spacing_mm=0.5,
    verbose=True,
    distance_threshold_mm=8.0,
):
    """Classify segments into coronary arteries via connected components."""
    t0 = time.time()
    result = LabelingResult()
    vessel_tree = centerline_data.get("vessel_tree", {})
    segments = vessel_tree.get("segments", {})

    if not segments:
        logger.warning("no segments in centerline data, nothing to label")
        result.elapsed_seconds = time.time() - t0
        return result

    if vessel_mask is None:
        logger.warning("no vessel mask provided, labeling everything as generic Coronary")
        for seg_id_str in segments:
            sid = int(seg_id_str)
            result.labels[sid] = ArteryLabel(
                segment_id=sid,
                artery_name="Coronary",
                full_name="Coronary",
                tree_side="",
                confidence=0.5,
                reason="No vessel mask available for component classification",
            )
        result.elapsed_seconds = round(time.time() - t0, 2)
        return result

    if verbose:
        logger.info("mapping segments to components...")

    seg_to_comp, labeled_mask, n_components = map_segments_to_components(
        centerline_data, vessel_mask
    )

    if verbose:
        logger.info("  found %d components, %d segments mapped",
                     n_components, len(seg_to_comp))

    labeled_mask, n_components = merge_nearby_components(
        labeled_mask, n_components,
        distance_threshold_mm=distance_threshold_mm,
        spacing_mm=spacing_mm,
    )
    seg_to_comp = _remap_segments_after_merge(seg_to_comp, labeled_mask, centerline_data)

    if verbose:
        logger.info("  after merge: %d components, %d segments mapped",
                     n_components, len(seg_to_comp))

    if verbose:
        logger.info("classifying left vs right...")

    comp_names = classify_components(labeled_mask, n_components)

    if verbose:
        for cl, name in sorted(comp_names.items()):
            logger.info("  component %d -> %s", cl, name)

    if verbose:
        logger.info("assigning labels...")

    left_segment_ids = []
    right_segment_ids = []

    for seg_id_str in segments:
        sid = int(seg_id_str)
        comp_label = seg_to_comp.get(sid, 0)
        comp_name = comp_names.get(comp_label, "Coronary")

        if comp_label == 0:
            comp_name = "Coronary"
            confidence = 0.4
            reason = "Segment not mapped to any vessel component"
        else:
            sizes = np.bincount(labeled_mask.ravel())
            total_voxels = sizes[1:].sum() if len(sizes) > 1 else 1
            comp_size = sizes[comp_label] if comp_label < len(sizes) else 0
            comp_fraction = comp_size / max(1, total_voxels)

            if comp_fraction >= 0.15:
                confidence = 0.9
            elif comp_fraction >= 0.05:
                confidence = 0.7
            else:
                confidence = 0.6
            reason = f"Component {comp_label} ({comp_name})"

        tree_side = ""
        if "Left" in comp_name:
            tree_side = "left"
            left_segment_ids.append(sid)
        elif "Right" in comp_name:
            tree_side = "right"
            right_segment_ids.append(sid)

        result.labels[sid] = ArteryLabel(
            segment_id=sid,
            artery_name=comp_name,
            full_name=comp_name,
            tree_side=tree_side,
            confidence=confidence,
            reason=reason,
        )
        result.tree_sides[sid] = tree_side or "unknown"

    if left_segment_ids:
        if verbose:
            logger.info("attempting LAD/LCx split on %d left-system segments...",
                         len(left_segment_ids))
        try:
            _try_lad_lcx_split(centerline_data, left_segment_ids, result.labels, spacing_mm)
        except Exception as e:
            logger.debug("LAD/LCx split failed (non-fatal): %s", e)

    result.elapsed_seconds = round(time.time() - t0, 2)

    if verbose:
        labeled_count = sum(1 for l in result.labels.values()
                           if l.artery_name not in ("Unknown", ""))
        total = len(result.labels)
        logger.info("labeling done: %d/%d segments labeled (%.0f%%), took %.2fs",
                     labeled_count, total,
                     100 * labeled_count / total if total > 0 else 0,
                     result.elapsed_seconds)

    return result


def process(
    centerline_path,
    output_path,
    ct_volume_path=None,
    vessel_mask_path=None,
    spacing_mm=0.5,
    verbose=True,
):
    """Load data, run labeling, save results."""
    t0 = time.time()
    centerline_path = Path(centerline_path)
    output_path = Path(output_path)

    if not centerline_path.exists():
        return {"status": "error", "error": f"Centerline file not found: {centerline_path}"}

    with open(centerline_path, 'r') as f:
        centerline_data = json.load(f)

    vessel_mask = None
    if vessel_mask_path is not None:
        import zarr
        mask_path = Path(vessel_mask_path)
        if mask_path.exists():
            try:
                store = zarr.open_group(str(mask_path), mode='r')
                if 'mask' in store:
                    vessel_mask = store['mask'][:]
                    if verbose:
                        logger.info("loaded vessel mask: %s", vessel_mask.shape)
            except Exception as e:
                logger.warning("couldn't load vessel mask: %s", e)

    labeling_result = label_arteries(
        centerline_data=centerline_data,
        ct_volume=None,
        vessel_mask=vessel_mask,
        spacing_mm=spacing_mm,
        verbose=verbose,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "labeling": labeling_result.to_dict(),
        "input_path": str(centerline_path),
    }
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    elapsed = time.time() - t0

    labeled_count = sum(1 for l in labeling_result.labels.values()
                        if l.artery_name not in ("Unknown", ""))

    return {
        "status": "success",
        "total_segments": len(labeling_result.labels),
        "labeled_segments": labeled_count,
        "aorta_detected": labeling_result.aorta_detected,
        "ostia_detected": labeling_result.ostia_detected,
        "output_path": str(output_path),
        "elapsed_seconds": round(elapsed, 2),
    }
