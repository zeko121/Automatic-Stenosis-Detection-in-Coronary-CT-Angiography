"""
Centerline extraction from segmentation masks via skeletonization.
"""

import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    label as scipy_label,
    gaussian_filter1d,
    binary_erosion,
    binary_closing,
    generate_binary_structure,
    map_coordinates,
)
from scipy.interpolate import splprep, splev
from scipy.spatial import KDTree
from skimage.morphology import skeletonize, remove_small_objects
import zarr


@dataclass
class CenterlineConfig:
    voxel_spacing_mm: float = 0.5

    closing_iterations: int = 2
    min_vessel_size_voxels: int = 100
    erosion_before_skeleton: bool = False

    neighbor_radius_voxels: float = 1.8  # ~sqrt(3) for 26-connectivity
    min_segment_points: int = 10
    radius_smoothing_sigma_mm: float = 1.0

    resample_spacing_mm: float = 0.3
    spline_smoothing_factor: float = 0.5

    bifurcation_merge_distance_voxels: float = 3.0


@dataclass
class VesselNode:
    id: int
    position: list       # [z, y, x] voxels
    position_mm: list    # [z, y, x] mm
    radius_mm: float
    is_bifurcation: bool = False
    is_endpoint: bool = False
    neighbors: list = field(default_factory=list)

    def to_dict(self):
        return {
            'id': self.id,
            'position': self.position,
            'position_mm': self.position_mm,
            'radius_mm': self.radius_mm,
            'is_bifurcation': self.is_bifurcation,
            'is_endpoint': self.is_endpoint,
            'neighbors': self.neighbors
        }


@dataclass
class VesselSegment:
    id: int
    node_ids: list
    centerline_points: list = None
    arc_length_mm: list = None
    radii_mm: list = None
    radii_smooth_mm: list = None
    length_mm: float = 0.0

    def to_dict(self):
        return {
            'id': self.id,
            'node_ids': self.node_ids,
            'centerline_points': self.centerline_points,
            'arc_length_mm': self.arc_length_mm,
            'radii_mm': self.radii_mm,
            'radii_smooth_mm': self.radii_smooth_mm,
            'length_mm': self.length_mm
        }



@dataclass
class VesselTree:
    nodes: dict = field(default_factory=dict)
    segments: dict = field(default_factory=dict)
    edges: list = field(default_factory=list)
    total_length_mm: float = 0.0
    num_bifurcations: int = 0
    num_endpoints: int = 0
    num_centerline_points: int = 0
    radius_min_mm: float = 0.0
    radius_max_mm: float = 0.0
    radius_mean_mm: float = 0.0

    def to_dict(self):
        return {
            'nodes': {str(k): v.to_dict() for k, v in self.nodes.items()},
            'segments': {str(k): v.to_dict() for k, v in self.segments.items()},
            'edges': self.edges,
            'total_length_mm': self.total_length_mm,
            'num_bifurcations': self.num_bifurcations,
            'num_endpoints': self.num_endpoints,
            'num_centerline_points': self.num_centerline_points,
            'radius_min_mm': self.radius_min_mm,
            'radius_max_mm': self.radius_max_mm,
            'radius_mean_mm': self.radius_mean_mm
        }


def preprocess_mask(mask, config):
    mask_clean = (mask > 0.5).astype(np.uint8)

    if config.erosion_before_skeleton:
        struct = generate_binary_structure(3, 1)
        mask_eroded = binary_erosion(mask_clean, structure=struct).astype(np.uint8)
        if mask_eroded.sum() > config.min_vessel_size_voxels:
            mask_clean = mask_eroded

    return mask_clean


def extract_skeleton(mask):
    distance_map = distance_transform_edt(mask)
    skeleton = skeletonize(mask).astype(np.uint8)
    return skeleton, distance_map


def extract_nodes(skeleton, distance_map, config):
    """Build nodes from skeleton voxels. Neighbors restricted to same component."""
    points = np.array(np.where(skeleton > 0)).T  # (N, 3) - [z, y, x]
    radii = distance_map[skeleton > 0] * config.voxel_spacing_mm

    component_labels, _ = scipy_label(skeleton, structure=generate_binary_structure(3, 3))
    node_component = component_labels[skeleton > 0]

    nodes = {}
    for i, (pt, r) in enumerate(zip(points, radii)):
        nodes[i] = VesselNode(
            id=i,
            position=pt.tolist(),
            position_mm=(pt * config.voxel_spacing_mm).tolist(),
            radius_mm=float(r)
        )

    if len(points) > 1:
        tree = KDTree(points)
        for i, pt in enumerate(points):
            neighbor_indices = tree.query_ball_point(pt, r=config.neighbor_radius_voxels)
            nodes[i].neighbors = [
                j for j in neighbor_indices
                if j != i and node_component[j] == node_component[i]
            ]

    return nodes


def classify_nodes(nodes):
    for node in nodes.values():
        n_neighbors = len(node.neighbors)
        node.is_endpoint = (n_neighbors == 1)
        node.is_bifurcation = (n_neighbors >= 3)


def merge_nearby_bifurcations(nodes, merge_distance_voxels=3.0):
    """Merge nearby bifurcation clusters into a single representative node."""
    bif_ids = [n.id for n in nodes.values() if n.is_bifurcation]
    if len(bif_ids) < 2:
        return nodes

    bif_points = np.array([nodes[i].position for i in bif_ids])
    tree = KDTree(bif_points)

    parent = {bid: bid for bid in bif_ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, bid in enumerate(bif_ids):
        nearby = tree.query_ball_point(bif_points[i], r=merge_distance_voxels)
        for j in nearby:
            if bif_ids[j] != bid:
                union(bid, bif_ids[j])

    groups = {}
    for bid in bif_ids:
        root = find(bid)
        groups.setdefault(root, []).append(bid)

    remap = {}
    for root, members in groups.items():
        if len(members) <= 1:
            continue
        rep = max(members, key=lambda m: nodes[m].radius_mm)
        for m in members:
            if m != rep:
                remap[m] = rep

    if not remap:
        return nodes

    for absorbed_id, rep_id in remap.items():
        absorbed = nodes[absorbed_id]
        rep = nodes[rep_id]
        for nbr in absorbed.neighbors:
            if nbr != rep_id and nbr not in remap:
                if nbr not in rep.neighbors:
                    rep.neighbors.append(nbr)

    for node in nodes.values():
        node.neighbors = [
            remap.get(nbr, nbr) for nbr in node.neighbors
            if nbr not in remap or remap[nbr] != node.id
        ]
        node.neighbors = list(set(node.neighbors))
        if node.id in node.neighbors:
            node.neighbors.remove(node.id)

    for absorbed_id in remap:
        del nodes[absorbed_id]

    classify_nodes(nodes)

    return nodes


def extract_edges(nodes):
    edges = set()
    for node in nodes.values():
        for neighbor_id in node.neighbors:
            edge = (min(node.id, neighbor_id), max(node.id, neighbor_id))
            edges.add(edge)
    return list(edges)

# TODO: might want to handle loops/cycles in the graph at some point

def extract_segments(nodes, config):
    segments = {}
    segment_id = 0
    visited_edges = set()

    start_nodes = [n for n in nodes.values() if n.is_endpoint or n.is_bifurcation]

    for start_node in start_nodes:
        for neighbor_id in start_node.neighbors:
            edge = (min(start_node.id, neighbor_id), max(start_node.id, neighbor_id))
            if edge in visited_edges:
                continue

            path = _trace_segment(start_node.id, neighbor_id, nodes, visited_edges)

            if len(path) >= config.min_segment_points:
                segments[segment_id] = VesselSegment(id=segment_id, node_ids=path)
                segment_id += 1

    return segments



def _trace_segment(start_id, next_id, nodes, visited_edges):
    path = [start_id, next_id]
    visited_edges.add((min(start_id, next_id), max(start_id, next_id)))

    current_id, prev_id = next_id, start_id

    while True:
        current_node = nodes[current_id]

        if current_node.is_bifurcation or current_node.is_endpoint:
            break

        # find next node (not where we came from)
        next_candidates = [n for n in current_node.neighbors if n != prev_id]
        if not next_candidates:
            break

        next_id = next_candidates[0]
        edge = (min(current_id, next_id), max(current_id, next_id))

        if edge in visited_edges:
            break

        visited_edges.add(edge)
        path.append(next_id)
        prev_id, current_id = current_id, next_id

    return path


def resample_centerline(points, spacing_mm=0.3, smoothing=0.5, voxel_spacing=0.5):
    """Fit spline to skeleton points and resample at uniform spacing."""
    if len(points) < 4:
        return points

    k = min(3, len(points) - 1)
    try:
        tck, u = splprep(
            [points[:, 0], points[:, 1], points[:, 2]],
            s=smoothing, k=k
        )
    except (ValueError, TypeError):
        return points

    diffs = np.diff(points, axis=0) * voxel_spacing
    raw_arc = np.sum(np.linalg.norm(diffs, axis=1))
    n_samples = max(4, int(raw_arc / spacing_mm))

    u_new = np.linspace(0, 1, n_samples)
    resampled = np.array(splev(u_new, tck)).T

    return resampled


def _compute_geometry_for_points(points, segment, distance_map, spacing, config):
    segment.centerline_points = points.tolist()

    arc_length = np.zeros(len(points))
    for i in range(1, len(points)):
        arc_length[i] = arc_length[i-1] + np.linalg.norm(points[i] - points[i-1]) * spacing
    segment.arc_length_mm = arc_length.tolist()
    segment.length_mm = float(arc_length[-1])

    radii = map_coordinates(distance_map, points.T, order=1, mode='nearest') * spacing
    segment.radii_mm = radii.tolist()

    sigma_voxels = config.radius_smoothing_sigma_mm / spacing
    if len(radii) >= 3:
        radii_smooth = gaussian_filter1d(radii, sigma=max(1, sigma_voxels))
    else:
        radii_smooth = radii.copy()
    segment.radii_smooth_mm = radii_smooth.tolist()

    return radii


def compute_segment_geometry(segment, nodes, distance_map, config):
    if len(segment.node_ids) < 2:
        return

    spacing = config.voxel_spacing_mm

    raw_points = np.array([nodes[nid].position for nid in segment.node_ids])

    points = resample_centerline(
        raw_points,
        spacing_mm=config.resample_spacing_mm,
        smoothing=config.spline_smoothing_factor,
        voxel_spacing=spacing
    )

    radii = _compute_geometry_for_points(points, segment, distance_map, spacing, config)

    # fall back to raw skeleton if spline wandered outside the mask
    if np.any(radii <= 0) and len(points) != len(raw_points):
        _compute_geometry_for_points(raw_points, segment, distance_map, spacing, config)


def extract_vessel_tree(mask, config=None):
    if config is None:
        config = CenterlineConfig()

    mask_clean = preprocess_mask(mask, config)

    if mask_clean.sum() < config.min_vessel_size_voxels:
        return VesselTree(), np.zeros_like(mask, dtype=np.uint8), np.zeros_like(mask, dtype=np.float32)

    skeleton, distance_map = extract_skeleton(mask_clean)

    nodes = extract_nodes(skeleton, distance_map, config)

    if len(nodes) < 2:
        tree = VesselTree(nodes=nodes)
        tree.num_centerline_points = len(nodes)
        return tree, skeleton, distance_map

    classify_nodes(nodes)
    nodes = merge_nearby_bifurcations(nodes, config.bifurcation_merge_distance_voxels)
    edges = extract_edges(nodes)
    segments = extract_segments(nodes, config)

    for seg in segments.values():
        compute_segment_geometry(seg, nodes, distance_map, config)

    tree = VesselTree(
        nodes=nodes,
        segments=segments,
        edges=edges,
        num_bifurcations=sum(1 for n in nodes.values() if n.is_bifurcation),
        num_endpoints=sum(1 for n in nodes.values() if n.is_endpoint),
        num_centerline_points=len(nodes)
    )

    tree.total_length_mm = sum(
        seg.length_mm for seg in segments.values()
    )

    all_radii = [n.radius_mm for n in nodes.values()]
    if all_radii:
        tree.radius_min_mm = float(np.min(all_radii))
        tree.radius_max_mm = float(np.max(all_radii))
        tree.radius_mean_mm = float(np.mean(all_radii))

    return tree, skeleton, distance_map


def process(input_path, output_path, voxel_spacing_mm=0.5, verbose=True, config=None):
    t0 = time.time()
    input_path = Path(input_path)
    output_path = Path(output_path)

    result = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "pending",
    }

    if not input_path.exists():
        result["status"] = "error"
        result["error"] = f"Input path does not exist: {input_path}"
        return result

    try:
        if verbose:
            print(f"Loading mask from {input_path.name}")

        store = zarr.open_group(str(input_path), mode='r')
        if 'mask' not in store:
            result["status"] = "error"
            result["error"] = "No 'mask' array found in input zarr"
            return result

        mask = store['mask'][:]

        if verbose:
            print(f"  Shape: {mask.shape}")
            print(f"  Vessel voxels: {mask.sum():,}")

        result["input_shape"] = list(mask.shape)
        result["input_vessel_voxels"] = int(mask.sum())

        config = config or CenterlineConfig(voxel_spacing_mm=voxel_spacing_mm)

        if verbose:
            print(f"Extracting centerlines (spacing={voxel_spacing_mm}mm)")

        tree, skeleton, distance_map = extract_vessel_tree(mask, config)

        if verbose:
            print(f"  Centerline points: {tree.num_centerline_points:,}")
            print(f"  Bifurcations: {tree.num_bifurcations}")
            print(f"  Endpoints: {tree.num_endpoints}")
            print(f"  Segments: {len(tree.segments)}")
            print(f"  Total length: {tree.total_length_mm:.1f} mm")
            if tree.radius_min_mm > 0:
                print(f"  Radius range: [{tree.radius_min_mm:.2f}, {tree.radius_max_mm:.2f}] mm")
                print(f"  Mean radius: {tree.radius_mean_mm:.2f} mm")

        if verbose:
            print(f"Saving to {output_path.name}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = {
            "vessel_tree": tree.to_dict(),
            "config": {
                "voxel_spacing_mm": config.voxel_spacing_mm,
                "closing_iterations": config.closing_iterations,
                "min_vessel_size_voxels": config.min_vessel_size_voxels,
                "min_segment_points": config.min_segment_points,
                "radius_smoothing_sigma_mm": config.radius_smoothing_sigma_mm,
                "resample_spacing_mm": config.resample_spacing_mm,
                "spline_smoothing_factor": config.spline_smoothing_factor,
                "bifurcation_merge_distance_voxels": config.bifurcation_merge_distance_voxels,
            },
            "input_path": str(input_path),
            "input_shape": list(mask.shape),
        }

        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        file_size_kb = output_path.stat().st_size / 1024

        if verbose:
            print(f"  Saved: {output_path.name} ({file_size_kb:.1f} KB)")

        elapsed = time.time() - t0

        result.update({
            "status": "success",
            "num_centerline_points": tree.num_centerline_points,
            "num_bifurcations": tree.num_bifurcations,
            "num_endpoints": tree.num_endpoints,
            "num_segments": len(tree.segments),
            "total_length_mm": round(tree.total_length_mm, 2),
            "radius_min_mm": round(tree.radius_min_mm, 3),
            "radius_max_mm": round(tree.radius_max_mm, 3),
            "radius_mean_mm": round(tree.radius_mean_mm, 3),
            "num_edges": len(tree.edges),
            "file_size_kb": round(file_size_kb, 2),
            "runtime_sec": round(elapsed, 2),
        })

        if verbose:
            print(f"Done in {elapsed:.2f}s")

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        if verbose:
            print(f"Error: {result['error']}")
            import traceback
            traceback.print_exc()

    return result


if __name__ == "__main__":
    import sys

    test_input = "temp/case-30_segmented.zarr"
    test_output = "temp/case-30_centerline.json"

    if len(sys.argv) >= 3:
        test_input = sys.argv[1]
        test_output = sys.argv[2]

    print("Centerline Extraction Test")
    print("-" * 40)

    # result = process(test_input, test_output, voxel_spacing_mm=0.3)
    result = process(
        test_input,
        test_output,
        voxel_spacing_mm=0.5,
        verbose=True
    )

    print()
    for key, value in result.items():
        print(f"  {key}: {value}")
