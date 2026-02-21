"""3D vessel visualization using Plotly."""

import json
import logging
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, generate_binary_structure
from scipy.ndimage import label as scipy_label
from skimage import measure
import plotly.graph_objects as go
import zarr

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_mesh(binary_mask, spacing=(0.5, 0.5, 0.5), smooth=True, smooth_sigma=0.5):
    """Extract 3D mesh from binary mask via marching cubes."""
    if binary_mask.sum() == 0:
        logger.warning("empty mask, skipping mesh extraction")
        return None, None

    if smooth:
        mask_for_mesh = gaussian_filter(binary_mask.astype(float), sigma=smooth_sigma)
        level = 0.5
    else:
        mask_for_mesh = binary_mask.astype(float)
        level = 0.5

    try:
        verts, faces, _, _ = measure.marching_cubes(
            mask_for_mesh,
            level=level,
            spacing=spacing
        )
        return verts, faces
    except Exception as e:
        logger.error(f"Marching cubes failed: {e}")
        return None, None


def decimate_mesh(verts, faces, target_faces=50000):
    if len(faces) <= target_faces:
        return verts, faces

    indices = np.random.choice(len(faces), size=target_faces, replace=False)
    faces_decimated = faces[indices]

    unique_verts, inverse = np.unique(faces_decimated.ravel(), return_inverse=True)
    verts_decimated = verts[unique_verts]
    faces_remapped = inverse.reshape(-1, 3)

    return verts_decimated, faces_remapped


COMPONENT_COLORS = [
    "#E53E3E",  # red
    "#3182CE",  # blue
    "#38A169",  # green
    "#D69E2E",  # gold
    "#805AD5",  # purple
    "#DD6B20",  # orange
    "#E53E8C",  # pink
    "#2B6CB0",  # dark blue
    "#276749",  # dark green
    "#975A16",  # brown
]

SEVERITY_COLORS = {
    "Normal": "green",
    "Mild": "yellow",
    "Moderate": "orange",
    "Severe": "red"
}

SEVERITY_SIZES = {
    "Normal": 8,
    "Mild": 10,
    "Moderate": 14,
    "Severe": 18
}

ARTERY_COLORS = {
    "Left Coronary": "#FF4444",
    "Right Coronary": "#44BB44",
    "Minor Vessel": "#888888",
    "Coronary": "#CC44CC",   # single-component fallback
    "LM": "#FFD700",
    "LAD": "#FF4444",
    "LCx": "#4488FF",
    "RCA": "#44BB44",
    "D1": "#FF8844",
    "D2": "#FFAA66",
    "OM1": "#6688FF",
    "OM2": "#88AAFF",
    "PDA": "#66CC66",
    "Ramus": "#CC44CC",
    "AM1": "#88DD88",
    "AM2": "#AAEEBB",
    "Unknown": "#AAAAAA",
}


def create_3d_figure(
    segmentation_mask,
    stenosis_findings,
    spacing=(0.5, 0.5, 0.5),
    max_faces=80000,
    mesh_color="red",
    mesh_opacity=0.4,
    title="Vessel Segmentation with Stenosis Markers",
    highlight_mask=None,
    highlight_color="cyan",
    highlight_opacity=0.8,
    highlight_label="Gap-filled",
    highlight_mask_bad=None,
    highlight_color_bad="yellow",
    highlight_label_bad="Gap-filled (wrong)",
    baseline_mask=None,
    min_component_voxels=100,
    centerline_data=None,
    gt_mask=None,
    endpoint_data=None,
    artery_labels=None,
):
    """Build the 3D plotly figure with vessel mesh + stenosis markers."""
    traces = []

    if baseline_mask is not None:
        logger.info("rendering per-component meshes...")
        render_binary = (segmentation_mask > 0.5).astype(np.uint8)
        struct = generate_binary_structure(3, 3)  # 26-connectivity
        labeled, n_comp = scipy_label(render_binary, structure=struct)

        if n_comp == 0:
            logger.warning("baseline mask is empty")
        else:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0
            sorted_labels = np.argsort(sizes)[::-1]
            sorted_labels = [l for l in sorted_labels if sizes[l] >= min_component_voxels]

            logger.info(f"  {n_comp} components, {len(sorted_labels)} above {min_component_voxels} vox")

            faces_per_comp = max(5000, max_faces // max(1, len(sorted_labels)))

            for rank, comp_label in enumerate(sorted_labels):
                comp_mask = (labeled == comp_label).astype(np.uint8)
                comp_size = int(sizes[comp_label])
                color = COMPONENT_COLORS[rank % len(COMPONENT_COLORS)]

                c_verts, c_faces = extract_mesh(comp_mask, spacing=spacing)
                if c_verts is None or c_faces is None:
                    continue
                if len(c_faces) > faces_per_comp:
                    c_verts, c_faces = decimate_mesh(c_verts, c_faces, faces_per_comp)

                traces.append(go.Mesh3d(
                    x=c_verts[:, 2],
                    y=c_verts[:, 1],
                    z=c_verts[:, 0],
                    i=c_faces[:, 0],
                    j=c_faces[:, 1],
                    k=c_faces[:, 2],
                    color=color,
                    opacity=mesh_opacity,
                    name=f"Component {rank+1} ({comp_size:,} vox)",
                    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
                    hoverinfo="name",
                ))

            small_labels = [l for l in range(1, n_comp + 1) if sizes[l] > 0 and sizes[l] < min_component_voxels]
            if small_labels:
                small_mask = np.isin(labeled, small_labels).astype(np.uint8)
                small_total = int(small_mask.sum())
                if small_total > 0:
                    s_verts, s_faces = extract_mesh(small_mask, spacing=spacing)
                    if s_verts is not None and s_faces is not None:
                        if len(s_faces) > faces_per_comp:
                            s_verts, s_faces = decimate_mesh(s_verts, s_faces, faces_per_comp)
                        traces.append(go.Mesh3d(
                            x=s_verts[:, 2],
                            y=s_verts[:, 1],
                            z=s_verts[:, 0],
                            i=s_faces[:, 0],
                            j=s_faces[:, 1],
                            k=s_faces[:, 2],
                            color="#A0AEC0",
                            opacity=mesh_opacity * 0.6,
                            name=f"Small fragments ({len(small_labels)} comp, {small_total:,} vox)",
                            lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
                            hoverinfo="name",
                        ))

            logger.info(f"  {len(traces)} component meshes")
    else:
        binary_mask = (segmentation_mask > 0.5).astype(np.uint8)
        voxel_count = binary_mask.sum()

        logger.info(f"extracting mesh from {voxel_count:,} voxels")
        verts, faces = extract_mesh(binary_mask, spacing=spacing)

        if verts is not None and faces is not None:
            if len(faces) > max_faces:
                logger.info(f"decimating {len(faces):,} -> {max_faces:,} faces")
                verts, faces = decimate_mesh(verts, faces, max_faces)

            traces.append(go.Mesh3d(
                x=verts[:, 2],
                y=verts[:, 1],
                z=verts[:, 0],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color=mesh_color,
                opacity=mesh_opacity,
                name=f"Vessel ({voxel_count:,} voxels)",
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
                hoverinfo="name"
            ))
            logger.info(f"mesh: {len(verts):,} verts, {len(faces):,} faces")
        else:
            logger.warning("mesh extraction failed, showing markers only")

    if highlight_mask is not None:
        highlight_binary = (highlight_mask > 0.5).astype(np.uint8)
        highlight_count = int(highlight_binary.sum())
        if highlight_count > 0:
            label = (f"{highlight_label} (correct)" if highlight_mask_bad is not None
                     else highlight_label)
            logger.info(f"highlight mesh: {highlight_count:,} voxels")
            h_verts, h_faces = extract_mesh(highlight_binary, spacing=spacing)
            if h_verts is not None and h_faces is not None:
                if len(h_faces) > max_faces:
                    h_verts, h_faces = decimate_mesh(h_verts, h_faces, max_faces)
                traces.append(go.Mesh3d(
                    x=h_verts[:, 2],
                    y=h_verts[:, 1],
                    z=h_verts[:, 0],
                    i=h_faces[:, 0],
                    j=h_faces[:, 1],
                    k=h_faces[:, 2],
                    color=highlight_color,
                    opacity=highlight_opacity,
                    name=f"{label} ({highlight_count:,} voxels)",
                    lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5),
                    hoverinfo="name",
                ))
                logger.info(f"  {len(h_verts):,} verts, {len(h_faces):,} faces")
        else:
            logger.info("highlight mask empty, skipping")

    if highlight_mask_bad is not None:
        bad_binary = (highlight_mask_bad > 0.5).astype(np.uint8)
        bad_count = int(bad_binary.sum())
        if bad_count > 0:
            logger.info(f"bad-highlight mesh: {bad_count:,} voxels")
            b_verts, b_faces = extract_mesh(bad_binary, spacing=spacing)
            if b_verts is not None and b_faces is not None:
                if len(b_faces) > max_faces:
                    b_verts, b_faces = decimate_mesh(b_verts, b_faces, max_faces)
                traces.append(go.Mesh3d(
                    x=b_verts[:, 2],
                    y=b_verts[:, 1],
                    z=b_verts[:, 0],
                    i=b_faces[:, 0],
                    j=b_faces[:, 1],
                    k=b_faces[:, 2],
                    color=highlight_color_bad,
                    opacity=highlight_opacity,
                    name=f"{highlight_label_bad} ({bad_count:,} voxels)",
                    lighting=dict(ambient=0.6, diffuse=0.8, specular=0.5),
                    hoverinfo="name",
                ))
                logger.info(f"  {len(b_verts):,} verts, {len(b_faces):,} faces")
        else:
            logger.info("bad-highlight mask empty, skipping")

    if centerline_data is not None:
        vessel_tree = centerline_data.get("vessel_tree", {})
        segments = vessel_tree.get("segments", {})
        nodes = vessel_tree.get("nodes", {})

        n_segments = 0
        total_points = 0
        for seg_id, seg in segments.items():
            pts = seg.get("centerline_points")
            if not pts or len(pts) < 2:
                continue
            pts = np.array(pts, dtype=float)
            pts_mm = pts * np.array(spacing)

            seg_id_int = int(seg_id)
            seg_color = "white"
            seg_name = f"Seg {seg_id}"
            if artery_labels and seg_id_int in artery_labels:
                label = artery_labels[seg_id_int]
                a_name = label.artery_name if hasattr(label, 'artery_name') else label.get("artery_name", "")
                full_name = label.full_name if hasattr(label, 'full_name') else label.get("full_name", "")
                if a_name:
                    seg_color = ARTERY_COLORS.get(a_name, ARTERY_COLORS.get("Unknown", "white"))
                    seg_name = full_name or a_name

            traces.append(go.Scatter3d(
                x=pts_mm[:, 2],  # W -> X
                y=pts_mm[:, 1],  # H -> Y
                z=pts_mm[:, 0],  # D -> Z
                mode="lines",
                line=dict(color=seg_color, width=3),
                name=seg_name,
                showlegend=bool(artery_labels),
                hoverinfo="name",
            ))
            n_segments += 1
            total_points += len(pts)

        bif_x, bif_y, bif_z = [], [], []
        ep_x, ep_y, ep_z = [], [], []
        for node_id, node in nodes.items():
            pos = node.get("position", [0, 0, 0])
            pos_mm = [p * s for p, s in zip(pos, spacing)]
            if node.get("is_bifurcation"):
                bif_x.append(pos_mm[2])
                bif_y.append(pos_mm[1])
                bif_z.append(pos_mm[0])
            elif node.get("is_endpoint"):
                ep_x.append(pos_mm[2])
                ep_y.append(pos_mm[1])
                ep_z.append(pos_mm[0])

        if bif_x:
            traces.append(go.Scatter3d(
                x=bif_x, y=bif_y, z=bif_z,
                mode="markers",
                marker=dict(size=4, color="white", symbol="circle",
                            line=dict(color="black", width=1)),
                name=f"Bifurcations ({len(bif_x)})",
                hoverinfo="name",
            ))

        if ep_x:
            traces.append(go.Scatter3d(
                x=ep_x, y=ep_y, z=ep_z,
                mode="markers",
                marker=dict(size=3, color="#A0AEC0", symbol="circle"),
                name=f"Endpoints ({len(ep_x)})",
                hoverinfo="name",
            ))

        logger.info(f"centerline: {n_segments} segs, {total_points} pts, "
                     f"{len(bif_x)} bifs, {len(ep_x)} endpoints")

    if endpoint_data is not None and len(endpoint_data) > 0:
        valid_x, valid_y, valid_z, valid_hover = [], [], [], []
        invalid_x, invalid_y, invalid_z, invalid_hover = [], [], [], []

        for ep in endpoint_data:
            pos = ep.get("position", [0, 0, 0])
            pos_mm = [p * s for p, s in zip(pos, spacing)]
            is_valid = ep.get("is_valid", True)
            reason = ep.get("reason", "")
            radius = ep.get("radius_mm", 0)
            confidence = ep.get("confidence", 0)

            hover = (
                f"<b>{'Valid' if is_valid else 'Invalid'} Endpoint</b><br>"
                f"Reason: {reason}<br>"
                f"Radius: {radius:.2f}mm<br>"
                f"Confidence: {confidence:.0%}<br>"
                f"Pos: ({pos[0]}, {pos[1]}, {pos[2]})"
            )

            if is_valid:
                valid_x.append(pos_mm[2])
                valid_y.append(pos_mm[1])
                valid_z.append(pos_mm[0])
                valid_hover.append(hover)
            else:
                invalid_x.append(pos_mm[2])
                invalid_y.append(pos_mm[1])
                invalid_z.append(pos_mm[0])
                invalid_hover.append(hover)

        if valid_x:
            traces.append(go.Scatter3d(
                x=valid_x, y=valid_y, z=valid_z,
                mode="markers",
                marker=dict(size=6, color="#38A169", symbol="circle",
                            line=dict(color="white", width=1)),
                name=f"Valid Endpoints ({len(valid_x)})",
                hovertext=valid_hover,
                hoverinfo="text",
            ))

        if invalid_x:
            traces.append(go.Scatter3d(
                x=invalid_x, y=invalid_y, z=invalid_z,
                mode="markers",
                marker=dict(size=8, color="#E53E3E", symbol="diamond",
                            line=dict(color="white", width=1)),
                name=f"Invalid Endpoints ({len(invalid_x)})",
                hovertext=invalid_hover,
                hoverinfo="text",
            ))

        logger.info(f"endpoints: {len(valid_x)} valid, {len(invalid_x)} invalid")

    if stenosis_findings:
        logger.info(f"{len(stenosis_findings)} stenosis markers")

        for severity in ["Severe", "Moderate", "Mild", "Normal"]:
            findings_of_severity = [f for f in stenosis_findings if f.get("severity") == severity]

            if not findings_of_severity:
                continue

            x_coords = []
            y_coords = []
            z_coords = []
            hover_texts = []

            for finding in findings_of_severity:
                voxel = finding.get("location_voxel", [0, 0, 0])
                pos_mm = [v * s for v, s in zip(voxel, spacing)]

                x_coords.append(pos_mm[2])
                y_coords.append(pos_mm[1])
                z_coords.append(pos_mm[0])

                pct = finding.get("stenosis_percent", 0)
                seg_id = finding.get("segment_id", "?")
                min_r = finding.get("min_radius_mm", 0)
                ref_r = finding.get("reference_radius_mm", 0)
                conf = finding.get("confidence", 0)
                artery_name = finding.get("artery_name", "")
                artery_region = finding.get("artery_region", "")

                location_str = artery_region or artery_name or f"Segment {seg_id}"
                hover_text = (
                    f"<b>{location_str}</b><br>"
                    f"Stenosis: {pct:.1f}%<br>"
                    f"Severity: {severity}<br>"
                    f"Min radius: {min_r:.2f}mm<br>"
                    f"Ref radius: {ref_r:.2f}mm<br>"
                    f"Confidence: {conf:.0%}"
                )
                hover_texts.append(hover_text)

            traces.append(go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode="markers",
                marker=dict(
                    size=SEVERITY_SIZES.get(severity, 10),
                    color=SEVERITY_COLORS.get(severity, "white"),
                    symbol="diamond",
                    line=dict(color="black", width=1)
                ),
                name=f"{severity} ({len(findings_of_severity)})",
                hovertext=hover_texts,
                hoverinfo="text"
            ))

    n_analysis_traces = len(traces)

    gt_traces = []
    if gt_mask is not None:
        pred_binary = (segmentation_mask > 0.5).astype(np.uint8)
        gt_binary = (gt_mask > 0.5).astype(np.uint8)

        pred_count = int(pred_binary.sum())
        gt_count = int(gt_binary.sum())
        overlap_count = int((pred_binary & gt_binary).sum())

        logger.info(f"GT: pred={pred_count:,}, gt={gt_count:,}, overlap={overlap_count:,}")

        p_verts, p_faces = extract_mesh(pred_binary, spacing=spacing)
        if p_verts is not None and p_faces is not None:
            if len(p_faces) > max_faces:
                p_verts, p_faces = decimate_mesh(p_verts, p_faces, max_faces)
            gt_traces.append(go.Mesh3d(
                x=p_verts[:, 2], y=p_verts[:, 1], z=p_verts[:, 0],
                i=p_faces[:, 0], j=p_faces[:, 1], k=p_faces[:, 2],
                color="#E53E3E",
                opacity=0.35,
                name=f"Prediction ({pred_count:,} vox)",
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
                hoverinfo="name",
                visible=False,
            ))

        g_verts, g_faces = extract_mesh(gt_binary, spacing=spacing)
        if g_verts is not None and g_faces is not None:
            if len(g_faces) > max_faces:
                g_verts, g_faces = decimate_mesh(g_verts, g_faces, max_faces)
            gt_traces.append(go.Mesh3d(
                x=g_verts[:, 2], y=g_verts[:, 1], z=g_verts[:, 0],
                i=g_faces[:, 0], j=g_faces[:, 1], k=g_faces[:, 2],
                color="#38A169",
                opacity=0.35,
                name=f"Ground Truth ({gt_count:,} vox)",
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3),
                hoverinfo="name",
                visible=False,
            ))

        logger.info(f"GT traces: {len(gt_traces)}")

    all_traces = traces + gt_traces
    n_gt_traces = len(gt_traces)

    fig = go.Figure(data=all_traces)

    severity_counts = {}
    for f in stenosis_findings:
        sev = f.get("severity", "Unknown")
        severity_counts[sev] = severity_counts.get(sev, 0) + 1

    severity_summary = ", ".join([f"{k}: {v}" for k, v in severity_counts.items() if v > 0])
    if not severity_summary:
        severity_summary = "No stenoses found"

    updatemenus = []
    if n_gt_traces > 0:
        analysis_vis = [True] * n_analysis_traces + [False] * n_gt_traces
        gt_vis = [False] * n_analysis_traces + [True] * n_gt_traces

        updatemenus.append(dict(
            type="buttons",
            direction="left",
            x=0.5,
            xanchor="center",
            y=1.08,
            yanchor="top",
            buttons=[
                dict(
                    label="Analysis View",
                    method="update",
                    args=[{"visible": analysis_vis}],
                ),
                dict(
                    label="GT Comparison",
                    method="update",
                    args=[{"visible": gt_vis}],
                ),
            ],
            bgcolor="#F3F4F6",
            bordercolor="#D1D5DB",
            font=dict(size=12),
        ))

    fig.update_layout(
        title=dict(
            text=f"{title}<br><sub>{severity_summary}</sub>",
            x=0.5,
            xanchor="center"
        ),
        scene=dict(
            xaxis_title="X (mm)",
            yaxis_title="Y (mm)",
            zaxis_title="Z (mm)",
            aspectmode="data",
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        updatemenus=updatemenus if updatemenus else None,
    )

    return fig


def process(
    segmentation_path,
    stenosis_path,
    output_path,
    spacing=(0.5, 0.5, 0.5),
    max_faces=80000,
    title=None,
    highlight_mask=None,
    highlight_color="cyan",
    highlight_label="Gap-filled",
    highlight_mask_bad=None,
    highlight_color_bad="yellow",
    highlight_label_bad="Gap-filled (wrong)",
    baseline_mask=None,
    centerline_path=None,
    gt_mask=None,
    endpoint_data=None,
    mask_override=None,
    artery_labels=None,
):
    """Load seg + stenosis, build 3D figure, save as HTML."""
    start_time = time.time()
    logger.info(f"creating 3D viz: seg={segmentation_path}, stenosis={stenosis_path}")

    if mask_override is not None:
        segmentation_mask = mask_override
        logger.info(f"  mask override: {segmentation_mask.shape}, {int(segmentation_mask.sum()):,} vox")
    else:
        seg_path = Path(segmentation_path)
        if not seg_path.exists():
            raise FileNotFoundError(f"Segmentation file not found: {seg_path}")

        if seg_path.suffix == ".npy":
            segmentation_mask = np.load(seg_path)
        elif seg_path.suffix == ".zarr" or seg_path.is_dir():
            store = zarr.open_group(str(seg_path), mode='r')
            if 'mask' in store:
                segmentation_mask = store['mask'][:]
            else:
                raise ValueError(f"No 'mask' array found in zarr store: {seg_path}")
        else:
            raise ValueError(f"Unsupported segmentation format: {seg_path.suffix}")

    logger.info(f"  mask: {segmentation_mask.shape}, {segmentation_mask.sum():,} vox")

    stenosis_path = Path(stenosis_path)
    if not stenosis_path.exists():
        raise FileNotFoundError(f"Stenosis file not found: {stenosis_path}")

    with open(stenosis_path, 'r') as f:
        stenosis_data = json.load(f)

    findings = stenosis_data.get("findings", [])
    summary = stenosis_data.get("summary", {})
    logger.info(f"  {len(findings)} findings")

    if title is None:
        case_name = seg_path.stem.replace("_segmentation", "")
        max_severity = summary.get("max_severity", "Unknown")
        title = f"{case_name} | Max Severity: {max_severity}"

    cl_data = None
    if centerline_path is not None:
        cl_path = Path(centerline_path)
        if cl_path.exists():
            with open(cl_path, 'r') as f:
                cl_data = json.load(f)
            logger.info(f"  loaded centerline: {cl_path.name}")

    fig = create_3d_figure(
        segmentation_mask=segmentation_mask,
        stenosis_findings=findings,
        spacing=spacing,
        max_faces=max_faces,
        title=title,
        highlight_mask=highlight_mask,
        highlight_color=highlight_color,
        highlight_label=highlight_label,
        highlight_mask_bad=highlight_mask_bad,
        highlight_color_bad=highlight_color_bad,
        highlight_label_bad=highlight_label_bad,
        baseline_mask=baseline_mask,
        centerline_data=cl_data,
        gt_mask=gt_mask,
        endpoint_data=endpoint_data,
        artery_labels=artery_labels,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(output_path), include_plotlyjs="cdn")
    elapsed = time.time() - start_time
    file_size_kb = output_path.stat().st_size / 1024

    logger.info(f"  saved {output_path.name} ({file_size_kb:.1f} KB, {elapsed:.2f}s)")

    return {
        "output_path": str(output_path),
        "voxel_count": int(segmentation_mask.sum()),
        "findings_count": len(findings),
        "max_severity": summary.get("max_severity", "Unknown"),
        "file_size_kb": file_size_kb,
        "elapsed_seconds": elapsed
    }


if __name__ == "__main__":
    import sys

    test_seg = "temp/case-30_segmented.zarr"
    test_stenosis = "temp/case-30_stenosis.json"
    test_output = "temp/case-30_visualization.html"

    if len(sys.argv) >= 4:
        test_seg = sys.argv[1]
        test_stenosis = sys.argv[2]
        test_output = sys.argv[3]

    print(f"seg:      {test_seg}")
    print(f"stenosis: {test_stenosis}")
    print(f"output:   {test_output}")

    try:
        result = process(
            segmentation_path=test_seg,
            stenosis_path=test_stenosis,
            output_path=test_output,
            spacing=(0.5, 0.5, 0.5),
            max_faces=80000
        )

        print(f"\nvoxels={result['voxel_count']:,}, findings={result['findings_count']}, "
              f"severity={result['max_severity']}, {result['file_size_kb']:.1f} KB, "
              f"{result['elapsed_seconds']:.2f}s")
        print(f"output: {result['output_path']}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
