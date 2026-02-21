"""Slice viewer for CT volumes with segmentation overlay and stenosis markers."""

import numpy as np
import plotly.graph_objects as go

from pipeline.visualize import SEVERITY_COLORS, SEVERITY_SIZES

AXIS_MAP = {  # axis -> (dim_index, dim_label, x_label, y_label)
    "Axial (Z)":    (0, "Z", "X (Width)", "Y (Height)"),
    "Coronal (Y)":  (1, "Y", "X (Width)", "Z (Depth)"),
    "Sagittal (X)": (2, "X", "Y (Height)", "Z (Depth)"),
}

WINDOW_PRESETS = {
    "Default": (-3.0, 3.0),
    "Vessel Enhancement": (-1.0, 2.0),
    "Soft Tissue": (-2.0, 1.0),
}

MASK_OVERLAY_COLOR = "rgba(255, 80, 80, 0.35)"
FINDING_PROXIMITY = 2  # max slices away a finding still shows


def get_axis_dim(axis):
    return AXIS_MAP[axis][0]


def get_slice(volume, axis, slice_idx):
    """Extract a 2D slice from a 3D volume."""
    dim = get_axis_dim(axis)
    idx = max(0, min(slice_idx, volume.shape[dim] - 1))
    slicing = [slice(None)] * 3
    slicing[dim] = idx
    return volume[tuple(slicing)]

def get_nearby_findings(findings, axis, slice_idx, tolerance=FINDING_PROXIMITY):
    """Filter findings near the current slice."""
    if not findings:
        return []

    dim = get_axis_dim(axis)
    nearby = []
    for f in findings:
        loc = f.get("location_voxel")
        if loc and abs(loc[dim] - slice_idx) <= tolerance:
            nearby.append(f)
    return nearby


def _finding_2d_coords(finding, axis):
    """Project finding location onto the 2D slice plane."""
    loc = finding["location_voxel"]
    dim = get_axis_dim(axis)
    if dim == 0:
        return loc[2], loc[1]
    elif dim == 1:
        return loc[2], loc[0]
    else:
        return loc[1], loc[0]


def render_slice(
    image,
    mask,
    axis,
    slice_idx,
    findings=None,
    show_overlay=True,
    window_min=-3.0,
    window_max=3.0,
    spacing=(0.5, 0.5, 0.5),
):
    """Render a 2D slice as a Plotly figure with mask overlay and stenosis markers."""
    if axis not in AXIS_MAP:
        raise ValueError(f"Unknown axis: {axis}. Must be one of {list(AXIS_MAP.keys())}")

    dim, dim_label, x_label, y_label = AXIS_MAP[axis]
    slice_idx = max(0, min(slice_idx, image.shape[dim] - 1))
    mm_position = slice_idx * spacing[dim]

    image_slice = get_slice(image, axis, slice_idx)
    mask_slice = get_slice(mask, axis, slice_idx) if mask is not None else None

    traces = []

    traces.append(go.Heatmap(
        z=image_slice,
        colorscale="gray",
        zmin=window_min,
        zmax=window_max,
        showscale=True,
        colorbar=dict(title="Intensity", len=0.5, y=0.75),
        hovertemplate="x: %{x}<br>y: %{y}<br>value: %{z:.2f}<extra></extra>",
    ))

    if show_overlay and mask_slice is not None and mask_slice.any():
        overlay = np.where(mask_slice > 0, 1.0, np.nan)
        traces.append(go.Heatmap(
            z=overlay,
            colorscale=[[0, "rgba(0,0,0,0)"], [1, MASK_OVERLAY_COLOR]],
            zmin=0,
            zmax=1,
            showscale=False,
            hoverinfo="skip",
        ))

    if findings:
        nearby = get_nearby_findings(findings, axis, slice_idx)
        if nearby:
            for severity in ["Severe", "Moderate", "Mild", "Normal"]:
                sev_findings = [f for f in nearby if f.get("severity") == severity]
                if not sev_findings:
                    continue

                xs, ys, texts = [], [], []
                for f in sev_findings:
                    x, y = _finding_2d_coords(f, axis)
                    xs.append(x)
                    ys.append(y)
                    pct = f.get("stenosis_percent", 0)
                    seg = f.get("segment_id", "?")
                    texts.append(
                        f"{severity}: {pct:.1f}%<br>"
                        f"Segment {seg}<br>"
                        f"Radius: {f.get('min_radius_mm', 0):.2f}mm"
                    )

                traces.append(go.Scatter(
                    x=xs,
                    y=ys,
                    mode="markers",
                    marker=dict(
                        size=SEVERITY_SIZES.get(severity, 10),
                        color=SEVERITY_COLORS.get(severity, "white"),
                        symbol="cross",
                        line=dict(color="black", width=1.5),
                    ),
                    name=severity,
                    hovertext=texts,
                    hoverinfo="text",
                ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        uirevision=axis,
        title=dict(
            text=f"{axis} | Slice {slice_idx} | {mm_position:.1f} mm",
            font=dict(size=14),
        ),
        xaxis=dict(
            title=x_label,
            scaleanchor="y",
            constrain="domain",
            uirevision=axis,
        ),
        yaxis=dict(
            title=y_label,
            autorange="reversed",
            uirevision=axis,
        ),
        margin=dict(l=60, r=20, t=40, b=60),
        height=550,
        showlegend=bool(findings and get_nearby_findings(findings, axis, slice_idx)),
        legend=dict(
            x=1.02, y=1,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="gray",
            borderwidth=1,
        ),
    )

    return fig
