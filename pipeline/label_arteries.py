"""
label coronary arteries using connected components from the vessel mask.

finds connected blobs, figures out which centerline segments live in which blob,
then classifies biggest blobs as left vs right by x-position. also tries to
split the left side into LAD/LCx at the first big bifurcaiton.
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
    """label for a single vessel segment"""
    segment_id: int
    artery_name: str
    region: str = ""
    full_name: str = ""
    tree_side: str = ""
    confidence: float = 0.0
    reason: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class LabelingResult:
    """everything that comes out of the labeling pipeline"""
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
    """map each centerline segment to a connected component via midpoint lookup"""
    struct = generate_binary_structure(3, 3)
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

        # try midpoint
        mid_idx = n_pts // 2
        mid_vox = np.round(pts_arr[mid_idx]).astype(int)
        comp_label = _safe_lookup(labeled_mask, mid_vox)

        if comp_label > 0:
            seg_to_comp[sid] = int(comp_label)
            continue

        # fallback: 3 point majority
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
    """lookup voxel in mask, 0 if oob"""
    shape = labeled_mask.shape
    if (voxel >= 0).all() and voxel[0] < shape[0] and voxel[1] < shape[1] and voxel[2] < shape[2]:
        return int(labeled_mask[voxel[0], voxel[1], voxel[2]])
    return 0


def classify_components(labeled_mask, n_components):
    """classify components as left/right coronary by x-position in RAS

    in RAS orientation, the x-axis points to patient's right side.
    so lower x-index in the array = patient right = right coronary,
    higher x-index = patient left = left coronary.
    """
    if n_components == 0:
        return {}

    sizes = np.bincount(labeled_mask.ravel())
    sizes[0] = 0
    total_voxels = sizes.sum()

    if total_voxels == 0:
        return {}

    # 5% threshold for significant
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

    # in RAS array: lower x-index = patient-right = right coronary
    comp_mean_x = {}
    for comp_label in significant:
        coords = np.argwhere(labeled_mask == comp_label)
        comp_mean_x[comp_label] = coords[:, 2].mean()

    sorted_comps = sorted(significant, key=lambda c: comp_mean_x[c])

    result[sorted_comps[0]] = "Right Coronary"
    result[sorted_comps[-1]] = "Left Coronary"

    # middle components go to whichever is closer
    if len(sorted_comps) > 2:
        right_x = comp_mean_x[sorted_comps[0]]
        left_x = comp_mean_x[sorted_comps[-1]]
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
    """merge components whose surfaces are close together

    fragmented vessels from the same system are usually 3-15mm apart,
    while left-to-right distance is 30-50mm. so 8mm threshold works well
    to reconnect fragments without merging left and right systems.
    uses union-find for efficient merging.
    """
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

    # subsample for speed
    max_sample = 2000
    comp_coords = {}
    for comp_label in significant:
        coords = np.argwhere(labeled_mask == comp_label)
        if len(coords) > max_sample:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(coords), max_sample, replace=False)
            coords = coords[idx]
        comp_coords[comp_label] = coords

    # union find
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
                continue
            dists, _ = trees[ca].query(comp_coords[cb], k=1)
            min_dist = dists.min()
            if min_dist < distance_threshold_vox:
                union(ca, cb)
                logger.debug("merging components %d and %d (min dist: %.1f vox = %.1f mm)",
                             ca, cb, min_dist, min_dist * spacing_mm)

    roots = {find(c) for c in significant}
    if len(roots) == len(significant):
        return labeled_mask, n_components

    # relabel
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
    """re-lookup segment->component after relabeling"""
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
    """node in the rooted vessel tree"""
    node_id: int
    position: np.ndarray
    radius_mm: float
    parent_id: int = None
    children: list = field(default_factory=list)
    depth: int = 0
    distance_from_root_mm: float = 0.0


def build_rooted_tree(centerline_data, root_position, spacing_mm=0.5):
    """bfs from nearest node to root_position"""
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
    """find which segment contains a node"""
    for seg_id_str, seg in segments.items():
        if node_id in seg.get("node_ids", []):
            return int(seg_id_str)
    return None


def find_major_bifurcation(rooted_tree, root_id, min_child_ratio=0.35):
    """walk from root until 2+ children have substantial radius"""
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
    """direction from start going distally"""
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
    """get all node ids in subtree rooted at start_id"""
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
    """assign proximal/mid/distal based on arc position"""
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


def _try_lad_lcx_split(centerline_data, left_segment_ids, labels, spacing_mm=0.5, vessel_mask=None):
    """lad/lcx split via junction graph instead of full centerline

    algorithm overview:
    1. build a graph of segment junctions (endpoints only, not all centerline pts)
    2. find LM root: the endpoint with highest z (most superior point)
    3. walk from root, at each step try to reconnect nearby junctions if
       theres a path through the vessel mask
    4. at bifurcations, check if 2+ branches are "major" (>= 10mm subtree)
       - if yes: this is the LAD/LCx split point
       - if no: absorb minor branches as LM, keep walking
    5. classify LAD vs LCx by y-direction (LAD goes anterior = lower y)
    6. label remaining segments by y-position relative to bifurcation
    7. fix borderline segments using neighbor connectivity
    """
    vessel_tree = centerline_data.get("vessel_tree", {})
    nodes = vessel_tree.get("nodes", {})
    segments = vessel_tree.get("segments", {})

    if not left_segment_ids or not nodes:
        return

    # build segment-level adjacency
    left_seg_set = set(left_segment_ids)
    junction_adj = {}
    seg_junctions = {}

    for sid in left_segment_ids:
        seg = segments.get(str(sid), {})
        nids = seg.get("node_ids", [])
        if len(nids) < 2:
            continue
        j_start, j_end = nids[0], nids[-1]
        seg_junctions[sid] = (j_start, j_end)
        junction_adj.setdefault(j_start, []).append((sid, j_end))
        junction_adj.setdefault(j_end, []).append((sid, j_start))

    if not junction_adj:
        return

    # reconnect on demand
    merge_dist_mm = 8.0

    all_junc_pos = {jid: np.array(nodes[str(jid)]["position"], dtype=float)
                    for jid in junction_adj}

    def _line_inside_mask(pos_a, pos_b, mask_vol):
        """check if line stays inside vessel mask

        samples points along the line and checks each is in mask.
        prevents reconnecting through empty space.
        """
        n_samples = max(3, int(np.linalg.norm(pos_b - pos_a)))
        for t in np.linspace(0, 1, n_samples):
            pt = pos_a + t * (pos_b - pos_a)
            idx = tuple(np.clip(np.round(pt).astype(int), 0,
                               np.array(mask_vol.shape) - 1))
            if mask_vol[idx] == 0:
                return False
        return True

    _vessel_mask = vessel_mask

    def _try_reconnect(from_jid, walked_set):
        """connect to nearby unvisited junctions through vessel mask

        called at each step of the walk to bridge gaps in the centerline.
        only connects if theres a valid path through vessel mask.
        """
        from_pos = all_junc_pos[from_jid]
        existing_neighbors = {other for _, other in junction_adj.get(from_jid, [])}
        connected = []
        for jid, pos in all_junc_pos.items():
            if jid in walked_set or jid == from_jid or jid in existing_neighbors:
                continue
            dist = np.linalg.norm(from_pos - pos) * spacing_mm
            if dist >= merge_dist_mm:
                continue
            if _vessel_mask is not None and _line_inside_mask(from_pos, pos, _vessel_mask):
                junction_adj[from_jid].append((-1, jid))
                junction_adj[jid].append((-1, from_jid))
                existing_neighbors.add(jid)
                connected.append(jid)
                logger.debug("LAD/LCx split: on-demand reconnect %d <-> %d (%.1fmm, inside vessel)",
                             from_jid, jid, dist)
        return connected[0] if connected else None

    # find LM root - highest z endpoint
    endpoints = []
    for jid in junction_adj:
        if len(junction_adj[jid]) == 1:
            pos = np.array(nodes.get(str(jid), {}).get("position", [0, 0, 0]), dtype=float)
            endpoints.append((jid, pos))

    if not endpoints:
        return

    root_jid = max(endpoints, key=lambda x: x[1][0])[0]

    # walk from root toward the LAD/LCx bifurcation
    # a branch is "major" if its subtree is at least 10mm total length
    min_major_length_mm = 10.0

    def _subtree_stats(start_jid, from_jid):
        """dfs to measure subtree length and check for downstream bifurcations"""
        total_length = 0.0
        n_junctions = 0
        vis = {from_jid}
        stack = [start_jid]
        while stack:
            jid = stack.pop()
            if jid in vis:
                continue
            vis.add(jid)
            n_junctions += 1
            for sid, other in junction_adj.get(jid, []):
                if other in vis:
                    continue
                if sid >= 0:
                    total_length += segments.get(str(sid), {}).get("length_mm", 0)
                stack.append(other)
        return total_length, n_junctions >= 3

    lm_path_segs = set()
    lm_path_nodes = {root_jid}
    current_jid = root_jid
    bif_jid = None
    bif_branches = []
    walked = {root_jid}

    for _ in range(50):
        _try_reconnect(current_jid, walked)

        neighbors = [(sid, other) for sid, other in junction_adj.get(current_jid, [])
                     if other not in walked]
        if not neighbors:
            break

        if len(neighbors) == 1:
            sid, next_jid = neighbors[0]
            if sid >= 0:
                lm_path_segs.add(sid)
            lm_path_nodes.add(next_jid)
            walked.add(next_jid)
            current_jid = next_jid
            continue

        # multiple branches
        branch_stats = []
        for sid, other_jid in neighbors:
            if sid >= 0:
                first_seg_len = segments.get(str(sid), {}).get("length_mm", 0)
                real_degree = len([s for s, _ in junction_adj.get(other_jid, []) if s >= 0])
                if first_seg_len < min_major_length_mm and real_degree <= 1:
                    branch_stats.append((sid, other_jid, first_seg_len, False, False))
                    continue
            length, has_bif = _subtree_stats(other_jid, current_jid)
            is_major = length >= min_major_length_mm
            branch_stats.append((sid, other_jid, length, has_bif, is_major))

        major_branches = [(sid, other, length) for sid, other, length, has_bif, is_major
                          in branch_stats if is_major]

        if len(major_branches) >= 2:
            bif_jid = current_jid
            for sid, other_jid, length in major_branches:
                if sid >= 0:
                    avg_r = np.mean(segments.get(str(sid), {}).get("radii_mm", [0]))
                else:
                    avg_r = nodes.get(str(other_jid), {}).get("radius_mm", 0)
                bif_branches.append((sid, other_jid, avg_r))
            break
        else:
            for sid, other, length, has_bif, is_major in branch_stats:
                if not is_major:
                    if sid >= 0:
                        lm_path_segs.add(sid)
                    lm_path_nodes.add(other)
                    walked.add(other)

            if len(major_branches) == 1:
                sid, next_jid, _ = major_branches[0]
                if sid >= 0:
                    lm_path_segs.add(sid)
                lm_path_nodes.add(next_jid)
                walked.add(next_jid)
                current_jid = next_jid
            else:
                if branch_stats:
                    best = max(branch_stats, key=lambda x: x[2])
                    sid, next_jid = best[0], best[1]
                    if sid >= 0:
                        lm_path_segs.add(sid)
                    lm_path_nodes.add(next_jid)
                    walked.add(next_jid)
                else:
                    break
                current_jid = next_jid

    if bif_jid is None:
        logger.debug("LAD/LCx split: no bifurcation with 2+ major branches found")
        return

    bif_pos = np.array(nodes[str(bif_jid)]["position"], dtype=float)
    logger.debug("LAD/LCx split: bifurcation at node %d, %d branches", bif_jid, len(bif_branches))

    # classify by y direction
    bif_branches.sort(key=lambda x: x[2], reverse=True)
    top_two = bif_branches[:2]

    branch_dirs = []
    for sid, child_jid, avg_r in top_two:
        if sid >= 0:
            seg = segments.get(str(sid), {})
            pts = np.array(seg.get("centerline_points", []))
            if len(pts) < 5:
                continue
            nids = seg.get("node_ids", [])
            if nids[0] == bif_jid:
                direction = pts[min(20, len(pts) - 1)] - pts[0]
            else:
                direction = pts[max(0, len(pts) - 21)] - pts[-1]
        else:
            target_pos = np.array(nodes.get(str(child_jid), {}).get("position", [0, 0, 0]), dtype=float)
            direction = target_pos - bif_pos
        norm = np.linalg.norm(direction)
        if norm > 0:
            direction = direction / norm
        branch_dirs.append((sid, child_jid, direction, avg_r))

    if len(branch_dirs) < 2:
        return

    # LAD runs anteriorly (lower y in RAS), LCx wraps posteriorly (higher y)
    # compare the y-component of each branch's direction vector
    d0_y = branch_dirs[0][2][1]
    d1_y = branch_dirs[1][2][1]

    if d0_y <= d1_y:
        lad_branch_sid = branch_dirs[0][0]
        lcx_branch_sid = branch_dirs[1][0]
    else:
        lad_branch_sid = branch_dirs[1][0]
        lcx_branch_sid = branch_dirs[0][0]

    logger.debug("LAD/LCx split: LAD=seg %d (y=%.2f), LCx=seg %d (y=%.2f)",
                 lad_branch_sid, min(d0_y, d1_y), lcx_branch_sid, max(d0_y, d1_y))

    # label LM
    lm_segs = lm_path_segs

    for sid in left_segment_ids:
        if sid in lm_segs:
            labels[sid].artery_name = "LM"
            labels[sid].full_name = "LM"
            labels[sid].confidence = 0.7
            labels[sid].reason = "Left main (root to first bifurcation)"

    # assign LAD/LCx by y position
    split_y = bif_pos[1]

    for sid in left_segment_ids:
        if sid in lm_segs:
            continue
        seg = segments.get(str(sid), {})
        pts = seg.get("centerline_points", [])
        if not pts:
            continue
        mid_y = pts[len(pts) // 2][1] if len(pts[0]) > 1 else 0
        if mid_y <= split_y:
            labels[sid].artery_name = "LAD"
            labels[sid].full_name = "LAD"
            labels[sid].confidence = 0.7
            labels[sid].reason = "LAD (anterior of bifurcation)"
        else:
            labels[sid].artery_name = "LCx"
            labels[sid].full_name = "LCx"
            labels[sid].confidence = 0.7
            labels[sid].reason = "LCx (posterior of bifurcation)"

    # fix borderline segments by neigbor connectivity
    # if ALL neighbors have the opposite label, flip this segment
    # catches edge cases where y-position alone got it wrong
    for sid in left_segment_ids:
        if sid in lm_segs:
            continue
        current_name = labels[sid].artery_name
        if current_name not in ("LAD", "LCx"):
            continue
        nids = segments.get(str(sid), {}).get("node_ids", [])
        if len(nids) < 2:
            continue
        neighbor_labels = set()
        for jid in [nids[0], nids[-1]]:
            for nsid, _ in junction_adj.get(jid, []):
                if nsid >= 0 and nsid != sid and nsid in left_seg_set and nsid not in lm_segs:
                    neighbor_labels.add(labels[nsid].artery_name)
        other = "LCx" if current_name == "LAD" else "LAD"
        if neighbor_labels == {other}:
            labels[sid].artery_name = other
            labels[sid].full_name = other
            labels[sid].reason = f"{other} (corrected by neighbor connectivity)"

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
    """main labeling function - classifies segments via connected componets

    pipeline:
    1. map each centerline segment to a connected component in vessel mask
    2. merge nearby components (fixes fragmented vessels)
    3. classify components as left/right by x-position
    4. try LAD/LCx split on left side
    """
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
        logger.info("mapping segments to vessel mask components...")

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
        logger.info("classifying components (left vs right)...")

    comp_names = classify_components(labeled_mask, n_components)

    if verbose:
        for cl, name in sorted(comp_names.items()):
            logger.info("  component %d -> %s", cl, name)

    if verbose:
        logger.info("assigning labels to segments...")

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

    # try lad/lcx split
    if left_segment_ids:
        if verbose:
            logger.info("attempting LAD/LCx split on %d left-system segments...",
                         len(left_segment_ids))
        try:
            _try_lad_lcx_split(centerline_data, left_segment_ids, result.labels, spacing_mm,
                               vessel_mask=vessel_mask)
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
    """pipeline entry point - loads data, runs labeling, saves results"""
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
