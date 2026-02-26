"""
Post-processing cleanup for segmentation masks.
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import (
    binary_closing,
    distance_transform_edt,
    generate_binary_structure,
    gaussian_filter,
    label as scipy_label,
)
from scipy.spatial import KDTree
import zarr

logger = logging.getLogger(__name__)


@dataclass
class PostprocessConfig:
    """Tuned on 100 ImageCAS cases -- most fancy steps ended up hurting Dice."""
    voxel_spacing_mm: float = 0.5

    # hole filling -- 4 iters fills gaps up to ~2mm
    enable_hole_filling: bool = False
    closing_iterations: int = 4

    enable_smoothing: bool = False
    smoothing_sigma: float = 0.5

    min_component_voxels: int = 200

    enable_bridging: bool = False
    max_bridge_distance_mm: float = 5.0
    bridge_radius_mm: float = 1.0

    enable_shape_filtering: bool = False
    min_elongation_ratio: float = 2.0

    enable_distance_filtering: bool = True
    max_distance_from_main_mm: float = 5.0

    enable_cycle_breaking: bool = False
    min_cycle_break_radius_mm: float = 0.5

    # protect top-K largest so we don't nuke the second coronary system
    num_protected_trees: int = 3
    protected_tree_min_fraction: float = 0.05

    min_total_voxels: int = 3000


@dataclass
class PostprocessMetrics:
    input_voxels: int = 0
    input_components: int = 0
    voxels_after_closing: int = 0
    voxels_after_smoothing: int = 0
    components_after_initial_removal: int = 0
    voxels_after_initial_removal: int = 0
    bridges_added: int = 0
    voxels_after_bridging: int = 0
    components_removed_shape: int = 0
    components_removed_distance: int = 0
    cycles_broken: int = 0
    output_voxels: int = 0
    output_components: int = 0

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}



def _count_components(mask):
    struct = generate_binary_structure(3, 3)
    _, n = scipy_label(mask, structure=struct)
    return n


def _label_components(mask):
    struct = generate_binary_structure(3, 3)
    return scipy_label(mask, structure=struct)

def _largest_component_mask(mask):
    labeled, n = _label_components(mask)
    if n == 0:
        return mask.copy()
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    largest_label = sizes.argmax()
    return (labeled == largest_label).astype(np.uint8)


def _identify_protected_trees(mask, config):
    labeled, n = _label_components(mask)
    if n == 0:
        return set(), labeled, n

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    total_fg = int(mask.sum())
    min_size = max(1, int(total_fg * config.protected_tree_min_fraction))

    sorted_labels = np.argsort(sizes)[::-1]
    protected = set()
    for lbl in sorted_labels:
        if sizes[lbl] == 0:
            break
        if len(protected) >= config.num_protected_trees:
            break
        if sizes[lbl] >= min_size:
            protected.add(int(lbl))

    return protected, labeled, n


def _check_safety_floor(mask, min_voxels, step_name):
    count = int(mask.sum())
    if count < min_voxels:
        logger.warning(
            "Safety floor hit after %s: %d voxels < %d minimum. "
            "Stopping further removal.",
            step_name, count, min_voxels
        )
        return True
    return False


def fill_holes(mask, config):
    struct = generate_binary_structure(3, 1)
    closed = binary_closing(
        mask, structure=struct, iterations=config.closing_iterations
    ).astype(np.uint8)

    before = int(mask.sum())
    after = int(closed.sum())
    if before > 0 and after > 2 * before:
        logger.warning(
            "Closing doubled voxel count (%d -> %d), reverting.", before, after
        )
        return mask.copy()

    return closed


def smooth_surface(mask, config):
    before = int(mask.sum())
    sigma = config.smoothing_sigma

    smoothed_float = gaussian_filter(mask.astype(np.float32), sigma=sigma)
    result = (smoothed_float >= 0.5).astype(np.uint8)
    after = int(result.sum())

    if before > 0 and after < 0.8 * before:
        sigma /= 2.0
        logger.info(
            "Smoothing removed >20%% voxels (%d->%d), retrying with sigma=%.2f",
            before, after, sigma
        )
        smoothed_float = gaussian_filter(mask.astype(np.float32), sigma=sigma)
        result = (smoothed_float >= 0.5).astype(np.uint8)

    return result



def remove_small_components(mask, min_size):
    labeled, n = _label_components(mask)
    if n == 0:
        return mask.copy(), 0, 0

    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0

    keep_labels = np.where(sizes >= min_size)[0]
    filtered = np.isin(labeled, keep_labels).astype(np.uint8)

    return filtered, n, n - len(keep_labels)


def _create_cylinder_bridge(start, end, radius_voxels, shape):
    bridge = np.zeros(shape, dtype=np.uint8)
    direction = end.astype(float) - start.astype(float)
    length = np.linalg.norm(direction)

    if length == 0:
        return bridge

    n_steps = int(np.ceil(length)) + 1
    for t in np.linspace(0, 1, n_steps):
        point = start + t * direction
        z, y, x = int(round(point[0])), int(round(point[1])), int(round(point[2]))

        for dz in range(-radius_voxels, radius_voxels + 1):
            for dy in range(-radius_voxels, radius_voxels + 1):
                for dx in range(-radius_voxels, radius_voxels + 1):
                    if dz * dz + dy * dy + dx * dx <= radius_voxels * radius_voxels:
                        nz, ny, nx = z + dz, y + dy, x + dx
                        if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                            bridge[nz, ny, nx] = 1
    return bridge


def bridge_to_main_tree(mask, config):
    """Bridge small components to nearest protected tree."""
    protected_labels, labeled, n = _identify_protected_trees(mask, config)
    if n <= 1:
        return mask.copy(), 0
    if not protected_labels:
        return mask.copy(), 0

    protected_mask = np.isin(labeled, list(protected_labels))
    protected_points = np.argwhere(protected_mask)
    if len(protected_points) == 0:
        return mask.copy(), 0
    protected_kd = KDTree(protected_points)

    max_dist_voxels = config.max_bridge_distance_mm / config.voxel_spacing_mm
    bridge_radius_voxels = max(1, int(round(config.bridge_radius_mm / config.voxel_spacing_mm)))

    bridged = mask.copy()
    num_bridges = 0

    for comp_label in range(1, n + 1):
        if comp_label in protected_labels:
            continue
        comp_points = np.argwhere(labeled == comp_label)
        if len(comp_points) == 0:
            continue

        sample_size = min(len(comp_points), 200)
        sample_idx = np.linspace(0, len(comp_points) - 1, sample_size, dtype=int)
        sample_pts = comp_points[sample_idx]

        dists, prot_indices = protected_kd.query(sample_pts)
        best_idx = np.argmin(dists)
        best_dist = dists[best_idx]

        if best_dist <= max_dist_voxels:
            comp_pt = sample_pts[best_idx]
            prot_pt = protected_points[prot_indices[best_idx]]
            bridge = _create_cylinder_bridge(
                comp_pt, prot_pt, bridge_radius_voxels, mask.shape
            )
            bridged = np.maximum(bridged, bridge)
            num_bridges += 1

    return bridged, num_bridges


def filter_by_shape(mask, config):
    """Remove blobby (non-elongated) components via PCA."""
    protected_labels, labeled, n = _identify_protected_trees(mask, config)
    if n <= 1:
        return mask.copy(), 0

    removed = 0
    result = mask.copy()

    for comp_label in range(1, n + 1):
        if comp_label in protected_labels:
            continue
        comp_points = np.argwhere(labeled == comp_label)
        if len(comp_points) < 3:
            continue

        centered = comp_points - comp_points.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = np.sort(eigenvalues)[::-1]

        if eigenvalues[1] > 0:
            elongation = eigenvalues[0] / eigenvalues[1]
        else:
            elongation = float('inf')  # perfectly 1D, keep it

        if elongation < config.min_elongation_ratio:
            result[labeled == comp_label] = 0
            removed += 1

    return result, removed



def filter_by_distance(mask, config):
    """Remove components far from all protected trees."""
    protected_labels, labeled, n = _identify_protected_trees(mask, config)
    if n <= 1:
        return mask.copy(), 0
    if not protected_labels:
        return mask.copy(), 0

    protected_mask = np.isin(labeled, list(protected_labels))
    protected_points = np.argwhere(protected_mask)
    if len(protected_points) == 0:
        return mask.copy(), 0
    protected_kd = KDTree(protected_points)

    max_dist_voxels = config.max_distance_from_main_mm / config.voxel_spacing_mm
    removed = 0
    result = mask.copy()

    for comp_label in range(1, n + 1):
        if comp_label in protected_labels:
            continue
        comp_points = np.argwhere(labeled == comp_label)
        if len(comp_points) == 0:
            continue

        sample_size = min(len(comp_points), 200)
        sample_idx = np.linspace(0, len(comp_points) - 1, sample_size, dtype=int)
        dists, _ = protected_kd.query(comp_points[sample_idx])
        min_dist = dists.min()

        if min_dist > max_dist_voxels:
            result[labeled == comp_label] = 0
            removed += 1

    return result, removed


def break_cycles(mask, config):
    """Break loops at thinnest points (coronaries are trees, cycles are artifacts)."""
    from skimage.morphology import skeletonize

    skeleton = skeletonize(mask > 0).astype(np.uint8)
    skel_points = np.argwhere(skeleton)

    if len(skel_points) < 10:
        return mask.copy(), 0

    kd = KDTree(skel_points)
    pairs = kd.query_pairs(r=1.8)  # 26-connectivity ~ sqrt(3)

    adj = {i: [] for i in range(len(skel_points))}
    for i, j in pairs:
        adj[i].append(j)
        adj[j].append(i)

    visited = set()
    parent = {}
    back_edges = []

    def dfs(node, par):
        visited.add(node)
        parent[node] = par
        for nb in adj[node]:
            if nb == par:
                continue
            if nb in visited:
                back_edges.append((node, nb))
            else:
                dfs(node=nb, par=node)

    import sys
    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(max(old_limit, len(skel_points) + 100))
    try:
        for i in range(len(skel_points)):
            if i not in visited:
                dfs(i, -1)
    except RecursionError:
        logger.warning("Recursion limit hit during cycle detection, skipping.")
        return mask.copy(), 0
    finally:
        sys.setrecursionlimit(old_limit)

    if not back_edges:
        return mask.copy(), 0

    dt = distance_transform_edt(mask > 0) * config.voxel_spacing_mm
    min_break_radius = config.min_cycle_break_radius_mm

    result = mask.copy()
    cycles_broken = 0
    # min_break_radius = np.median(dt[skeleton > 0]) * 0.5  # tried median, didn't help

    for edge_a, edge_b in back_edges:
        pt_a = skel_points[edge_a]
        pt_b = skel_points[edge_b]

        r_a = dt[tuple(pt_a)]
        r_b = dt[tuple(pt_b)]

        thin_pt = pt_a if r_a <= r_b else pt_b
        thin_r = min(r_a, r_b)

        if thin_r < min_break_radius:
            remove_r = max(1, int(round(1.0 / config.voxel_spacing_mm)))
            z, y, x = thin_pt
            for dz in range(-remove_r, remove_r + 1):
                for dy in range(-remove_r, remove_r + 1):
                    for dx in range(-remove_r, remove_r + 1):
                        if dz * dz + dy * dy + dx * dx <= remove_r * remove_r:
                            nz, ny, nx = z + dz, y + dy, x + dx
                            if 0 <= nz < mask.shape[0] and 0 <= ny < mask.shape[1] and 0 <= nx < mask.shape[2]:
                                result[nz, ny, nx] = 0
            cycles_broken += 1

    return result, cycles_broken


def postprocess_mask(mask, config=None, verbose=True):
    if config is None:
        config = PostprocessConfig()

    metrics = PostprocessMetrics()
    current = (mask > 0.5).astype(np.uint8)

    metrics.input_voxels = int(current.sum())
    metrics.input_components = _count_components(current)

    if metrics.input_voxels == 0:
        if verbose:
            logger.info("Empty mask, nothing to do.")
        return current, metrics

    if config.enable_hole_filling:
        if verbose:
            logger.info("Hole filling (closing, %d iters)", config.closing_iterations)
        current = fill_holes(current, config)
    metrics.voxels_after_closing = int(current.sum())

    if config.enable_smoothing:
        if verbose:
            logger.info("Surface smoothing (sigma=%.2f)", config.smoothing_sigma)
        current = smooth_surface(current, config)
    metrics.voxels_after_smoothing = int(current.sum())

    if verbose:
        logger.info("Removing small components (min=%d voxels)", config.min_component_voxels)
    current, n_before, n_removed = remove_small_components(current, config.min_component_voxels)
    metrics.components_after_initial_removal = n_before - n_removed
    metrics.voxels_after_initial_removal = int(current.sum())
    if verbose and n_removed > 0:
        logger.info("  dropped %d small components (%d -> %d)", n_removed, n_before, n_before - n_removed)

    if _check_safety_floor(current, config.min_total_voxels, "initial small removal"):
        metrics.output_voxels = int(current.sum())
        metrics.output_components = _count_components(current)
        return current, metrics

    if config.enable_bridging:
        if verbose:
            logger.info(
                "Gap bridging (max=%.1fmm, radius=%.1fmm)",
                config.max_bridge_distance_mm, config.bridge_radius_mm,
            )
        current, n_bridges = bridge_to_main_tree(current, config)
        metrics.bridges_added = n_bridges
        metrics.voxels_after_bridging = int(current.sum())
        if verbose and n_bridges > 0:
            logger.info("  added %d bridges", n_bridges)
    else:
        metrics.voxels_after_bridging = int(current.sum())

    if config.enable_shape_filtering:
        if verbose:
            logger.info("Shape filtering (min elongation=%.1f)", config.min_elongation_ratio)
        current, n_shape = filter_by_shape(current, config)
        metrics.components_removed_shape = n_shape
        if verbose and n_shape > 0:
            logger.info("  removed %d blobby components", n_shape)

        if _check_safety_floor(current, config.min_total_voxels, "shape filtering"):
            metrics.output_voxels = int(current.sum())
            metrics.output_components = _count_components(current)
            return current, metrics

    if config.enable_distance_filtering:
        if verbose:
            logger.info("Distance filtering (max=%.1fmm from main)", config.max_distance_from_main_mm)
        current, n_dist = filter_by_distance(current, config)
        metrics.components_removed_distance = n_dist
        if verbose and n_dist > 0:
            logger.info("  removed %d distant components", n_dist)
        # print(f"  DEBUG distance: kept {_count_components(current)} components")

        if _check_safety_floor(current, config.min_total_voxels, "distance filtering"):
            metrics.output_voxels = int(current.sum())
            metrics.output_components = _count_components(current)
            return current, metrics

    if config.enable_cycle_breaking:
        if verbose:
            logger.info("Cycle breaking (min radius=%.2fmm)", config.min_cycle_break_radius_mm)
        current, n_cycles = break_cycles(current, config)
        metrics.cycles_broken = n_cycles
        if verbose and n_cycles > 0:
            logger.info("  broken %d cycles", n_cycles)

        if _check_safety_floor(current, config.min_total_voxels, "cycle breaking"):
            metrics.output_voxels = int(current.sum())
            metrics.output_components = _count_components(current)
            return current, metrics

    if verbose:
        logger.info("Final small component removal")
    current, n_before, n_removed = remove_small_components(current, config.min_component_voxels)
    if verbose and n_removed > 0:
        logger.info("  removed %d small fragments", n_removed)

    metrics.output_voxels = int(current.sum())
    metrics.output_components = _count_components(current)

    return current, metrics


def process(input_path, output_path, config=None, verbose=True):
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

        store = zarr.open_group(str(input_path), mode="r")
        if "mask" not in store:
            result["status"] = "error"
            result["error"] = "No 'mask' array found in input zarr"
            return result

        mask = store["mask"][:]
        has_image = "image" in store
        image = store["image"][:] if has_image else None

        if verbose:
            print(f"  shape: {mask.shape}")
            print(f"  vessel voxels: {int(mask.sum()):,}")

        if verbose:
            print(f"Running post-processing pipeline")

        cleaned, metrics = postprocess_mask(mask, config=config, verbose=verbose)

        if verbose:
            print(f"Results:")
            print(f"  in:  {metrics.input_voxels:,} voxels, {metrics.input_components} components")
            print(f"  out: {metrics.output_voxels:,} voxels, {metrics.output_components} components")
            if metrics.bridges_added > 0:
                print(f"  bridges added: {metrics.bridges_added}")

        if verbose:
            print(f"Writing to {output_path.name}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        ZARR_V3 = zarr.__version__.startswith("3")
        if ZARR_V3:
            out_store = zarr.open_group(str(output_path), mode="w", zarr_version=3)
            chunks = tuple(min(64, s) for s in cleaned.shape)

            out_store.create_array(
                "mask", shape=cleaned.shape, dtype="uint8", chunks=chunks
            )[:] = cleaned

            if has_image:
                out_store.create_array(
                    "image", shape=image.shape, dtype="float32", chunks=chunks
                )[:] = image
        else:
            out_store = zarr.open_group(str(output_path), mode="w")
            out_store.create_dataset("mask", data=cleaned)
            if has_image:
                out_store.create_dataset("image", data=image)

        elapsed = time.time() - t0

        result.update({
            "status": "success",
            "input_shape": list(mask.shape),
            "input_voxels": metrics.input_voxels,
            "input_components": metrics.input_components,
            "output_voxels": metrics.output_voxels,
            "output_components": metrics.output_components,
            "bridges_added": metrics.bridges_added,
            "components_removed_shape": metrics.components_removed_shape,
            "components_removed_distance": metrics.components_removed_distance,
            "cycles_broken": metrics.cycles_broken,
            "runtime_sec": round(elapsed, 2),
        })

        if verbose:
            print(f"Done in {elapsed:.2f}s")  # TODO: add per-step timing breakdown

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        if verbose:
            print(f"\nError: {result['error']}")
            import traceback
            traceback.print_exc()

    return result
