"""Centerline graph traversal and region masking helpers.

No Qt dependencies -- safe to import anywhere.
"""

from collections import deque

import numpy as np


def build_centerline_graph(centerline_data):
    """Build adjacency graph from centerline data. Returns (all_points, adjacency)."""
    vessel_tree = centerline_data.get("vessel_tree", {})
    segments = vessel_tree.get("segments", {})

    all_points = []    # (d, h, w, radius)
    node_indices = {}  # node_id -> index
    adjacency = {}     # index -> [neighbor indices]

    for seg_id, seg in segments.items():
        pts = seg.get("centerline_points", [])
        radii = seg.get("radii", [])
        if not pts:
            continue
        seg_indices = []
        for i, pt in enumerate(pts):
            r = radii[i] if i < len(radii) else 1.0
            idx = len(all_points)
            all_points.append((pt[0], pt[1], pt[2], r))
            seg_indices.append(idx)
            if idx not in adjacency:
                adjacency[idx] = []
        for i in range(len(seg_indices) - 1):
            adjacency[seg_indices[i]].append(seg_indices[i + 1])
            adjacency[seg_indices[i + 1]].append(seg_indices[i])

        node_ids = seg.get("node_ids", [])
        if node_ids and len(seg_indices) > 0:
            for ni, node_id in enumerate(node_ids):
                nid_str = str(node_id)
                ep_idx = seg_indices[0] if ni == 0 else seg_indices[-1]
                if nid_str not in node_indices:
                    node_indices[nid_str] = ep_idx
                else:
                    existing_idx = node_indices[nid_str]
                    if existing_idx != ep_idx:
                        adjacency.setdefault(existing_idx, []).append(ep_idx)
                        adjacency.setdefault(ep_idx, []).append(existing_idx)

    return all_points, adjacency


def bfs_path(start_idx, end_idx, adjacency):
    """BFS shortest path between two nodes. Falls back to [start, end] if disconnected."""
    if start_idx == end_idx:
        return [start_idx]

    visited = {start_idx}
    parent = {start_idx: None}
    queue = deque([start_idx])
    found = False

    while queue:
        current = queue.popleft()
        if current == end_idx:
            found = True
            break
        for neighbor in adjacency.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = current
                queue.append(neighbor)

    if not found:
        return [start_idx, end_idx]

    path = []
    node = end_idx
    while node is not None:
        path.append(node)
        node = parent[node]
    path.reverse()
    return path


def compute_region_mask(centerline_data, seg_mask, start_voxel, end_voxel):
    """Compute vessel region mask between two points via centerline BFS.

    Returns (region_mask, path_indices, all_points, status_msg).
    """
    from scipy.spatial import cKDTree

    all_points, adjacency = build_centerline_graph(centerline_data)

    if not all_points:
        return None, [], [], "No centerline points found"

    coords = np.array([(p[0], p[1], p[2]) for p in all_points])
    tree = cKDTree(coords)

    _, start_idx = tree.query(start_voxel)
    _, end_idx = tree.query(end_voxel)

    path_indices = bfs_path(start_idx, end_idx, adjacency)

    status_msg = ""
    if (len(path_indices) == 2
            and start_idx != end_idx
            and end_idx not in adjacency.get(start_idx, [])):
        status_msg = "Points on disconnected components - highlighting each point"

    # fill sphere at each path point, 1.5x radius (min 4 vox) for visibility
    mask_shape = seg_mask.shape
    region_mask = np.zeros(mask_shape, dtype=np.uint8)

    for idx in path_indices:
        d, h, w, radius = all_points[idx]
        d, h, w = int(round(d)), int(round(h)), int(round(w))
        r_vox = max(4, int(round(radius * 1.5)))

        d_lo = max(0, d - r_vox)
        d_hi = min(mask_shape[0], d + r_vox + 1)
        h_lo = max(0, h - r_vox)
        h_hi = min(mask_shape[1], h + r_vox + 1)
        w_lo = max(0, w - r_vox)
        w_hi = min(mask_shape[2], w + r_vox + 1)

        dd, hh, ww = np.ogrid[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi]
        dist_sq = (dd - d)**2 + (hh - h)**2 + (ww - w)**2
        region_mask[d_lo:d_hi, h_lo:h_hi, w_lo:w_hi][dist_sq <= r_vox**2] = 1

    region_mask = region_mask & (seg_mask > 0).astype(np.uint8)

    n_voxels = int(region_mask.sum())
    if not status_msg:
        status_msg = (
            f"Region: {len(path_indices)} centerline points, "
            f"{n_voxels:,} voxels highlighted"
        )

    return region_mask, path_indices, all_points, status_msg
