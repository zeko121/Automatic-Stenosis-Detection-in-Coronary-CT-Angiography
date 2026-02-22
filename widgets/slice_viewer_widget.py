"""2D slice viewer widget with MPR support, using pyqtgraph."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QComboBox, QSlider, QSpinBox, QCheckBox, QPushButton,
    QStackedWidget,
)
from PySide6.QtCore import Qt, Signal, QEvent, QTimer

import pyqtgraph as pg

from pipeline.slice_viewer import (
    AXIS_MAP, WINDOW_PRESETS, FINDING_PROXIMITY,
    get_axis_dim, get_slice, get_nearby_findings, _finding_2d_coords,
)
from pipeline.visualize import SEVERITY_COLORS


_SEVERITY_RGBA = {
    "Normal": (0, 200, 0, 220),
    "Mild": (255, 255, 0, 220),
    "Moderate": (255, 165, 0, 220),
    "Severe": (255, 0, 0, 220),
}

_SEVERITY_SIZES = {
    "Normal": 10,
    "Mild": 12,
    "Moderate": 16,
    "Severe": 20,
}

_MPR_LABEL = "All Views (MPR)"
_AXIS_NAMES = list(AXIS_MAP.keys())  # ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"]


class _SlicePanel:
    """One PlotWidget with image/mask/marker layers + crosshairs."""

    def __init__(self, axis_name: str):
        self.axis_name = axis_name

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(True)
        self.plot_widget.invertY(True)
        self.plot_widget.setLabel("bottom", "X")
        self.plot_widget.setLabel("left", "Y")

        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)

        self.mask_item = pg.ImageItem()
        self.mask_item.setZValue(1)
        self.plot_widget.addItem(self.mask_item)

        self.region_item = pg.ImageItem()
        self.region_item.setZValue(2.5)
        self.region_item.setVisible(False)
        self.plot_widget.addItem(self.region_item)

        self.marker_item = pg.ScatterPlotItem(
            pen=pg.mkPen('k', width=1.5), symbol='x', size=12,
        )
        self.marker_item.setZValue(3)
        self.plot_widget.addItem(self.marker_item)

        self.h_line = pg.InfiniteLine(
            angle=0, movable=True,
            pen=pg.mkPen('y', width=1.5, style=Qt.DashLine),
            hoverPen=pg.mkPen('y', width=2.5),
        )
        self.v_line = pg.InfiniteLine(
            angle=90, movable=True,
            pen=pg.mkPen('y', width=1.5, style=Qt.DashLine),
            hoverPen=pg.mkPen('y', width=2.5),
        )
        self.h_line.setVisible(False)
        self.v_line.setVisible(False)
        self.h_line.setZValue(4)
        self.v_line.setZValue(4)
        self.plot_widget.addItem(self.h_line)
        self.plot_widget.addItem(self.v_line)

    def clear(self):
        self.image_item.clear()
        self.mask_item.clear()
        self.mask_item.setVisible(False)
        self.region_item.clear()
        self.region_item.setVisible(False)
        self.marker_item.clear()
        self.marker_item.setVisible(False)
        self.h_line.setVisible(False)
        self.v_line.setVisible(False)


class SliceViewerWidget(QWidget):
    """2D slice viewer with zoom/pan, mask overlay, stenosis markers, and MPR."""

    finding_selected = Signal(int)  # Emitted when user selects a finding

    def __init__(self, parent=None):
        super().__init__(parent)

        self._image: Optional[np.ndarray] = None   # (D, H, W) float32
        self._mask: Optional[np.ndarray] = None     # (D, H, W) uint8
        self._findings: List[Dict] = []
        self._region_mask: Optional[np.ndarray] = None  # (D, H, W) uint8

        self._mpr_mode = False
        self._mpr_slices = [0, 0, 0]    # [z, y, x] — which slice each panel shows
        self._crosshair_pos = [0, 0, 0]  # [z, y, x] — where crosshairs point (click/drag)
        self._syncing_range = False       # guard against recursive range sync
        self._active_mpr_dim = 0         # which axis the slider controls (0=Z, 1=Y, 2=X)

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        layout = QHBoxLayout(self)

        ctrl_layout = QVBoxLayout()
        ctrl_layout.setSpacing(8)

        ctrl_layout.addWidget(QLabel("Jump to Finding"))
        self.finding_combo = QComboBox()
        self.finding_combo.setPlaceholderText("Select finding...")
        ctrl_layout.addWidget(self.finding_combo)

        ctrl_layout.addWidget(QLabel("View Axis"))
        self.axis_combo = QComboBox()
        self.axis_combo.addItems(_AXIS_NAMES + [_MPR_LABEL])
        ctrl_layout.addWidget(self.axis_combo)

        self._slice_label = QLabel("Slice")
        ctrl_layout.addWidget(self._slice_label)
        slice_row = QHBoxLayout()
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_spinbox = QSpinBox()
        self.slice_spinbox.setMinimum(0)
        self.slice_spinbox.setMaximum(100)
        slice_row.addWidget(self.slice_slider, stretch=3)
        slice_row.addWidget(self.slice_spinbox, stretch=1)
        ctrl_layout.addLayout(slice_row)

        self.overlay_checkbox = QCheckBox("Show vessel overlay")
        self.overlay_checkbox.setChecked(True)
        ctrl_layout.addWidget(self.overlay_checkbox)

        self.region_checkbox = QCheckBox("Region highlight (Cyan)")
        self.region_checkbox.setChecked(True)
        self.region_checkbox.setStyleSheet("color: #06B6D4;")
        self.region_checkbox.setVisible(False)
        ctrl_layout.addWidget(self.region_checkbox)

        ctrl_layout.addWidget(QLabel("Window Preset"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(list(WINDOW_PRESETS.keys()))
        ctrl_layout.addWidget(self.window_combo)

        self.reset_view_btn = QPushButton("Reset View")
        self.reset_view_btn.setToolTip("Reset zoom and pan to fit the image")
        ctrl_layout.addWidget(self.reset_view_btn)

        ctrl_layout.addStretch()

        ctrl_widget = QWidget()
        ctrl_widget.setLayout(ctrl_layout)
        ctrl_widget.setFixedWidth(200)
        layout.addWidget(ctrl_widget)

        self._view_stack = QStackedWidget()

        self._single_panel = _SlicePanel("Axial (Z)")
        self._view_stack.addWidget(self._single_panel.plot_widget)

        mpr_container = QWidget()
        mpr_layout = QHBoxLayout(mpr_container)
        mpr_layout.setContentsMargins(0, 0, 0, 0)
        mpr_layout.setSpacing(2)

        self._mpr_panels: List[_SlicePanel] = []
        for axis_name in _AXIS_NAMES:
            panel = _SlicePanel(axis_name)
            self._mpr_panels.append(panel)
            mpr_layout.addWidget(panel.plot_widget, stretch=1)
            panel.plot_widget.installEventFilter(self)

        self._view_stack.addWidget(mpr_container)
        self._view_stack.setCurrentIndex(0)

        for panel in self._mpr_panels:
            panel.plot_widget.getViewBox().disableAutoRange()

        layout.addWidget(self._view_stack, stretch=1)

        for panel in self._mpr_panels:
            panel.plot_widget.scene().sigMouseClicked.connect(
                lambda evt, p=panel: self._on_mpr_click(evt, p)
            )

        for panel in self._mpr_panels:
            panel.h_line.sigDragged.connect(
                lambda line, p=panel: self._on_crosshair_dragged(p, 'h')
            )
            panel.v_line.sigDragged.connect(
                lambda line, p=panel: self._on_crosshair_dragged(p, 'v')
            )

    def _connect_signals(self):
        self.slice_slider.valueChanged.connect(self._on_slider_changed)
        self.slice_spinbox.valueChanged.connect(self._on_spinbox_changed)
        self.axis_combo.currentTextChanged.connect(self._on_axis_changed)
        self.overlay_checkbox.stateChanged.connect(self._render)
        self.region_checkbox.stateChanged.connect(self._render)
        self.window_combo.currentTextChanged.connect(self._render)
        self.finding_combo.currentIndexChanged.connect(self._on_finding_selected)
        self.reset_view_btn.clicked.connect(self._on_reset_view)

    def set_data(self, image: np.ndarray, mask: Optional[np.ndarray],
                 findings: List[Dict]):
        self._image = image
        self._mask = mask
        self._findings = findings or []

        self.finding_combo.blockSignals(True)
        self.finding_combo.clear()
        self.finding_combo.addItem("")  # Empty placeholder
        for i, f in enumerate(self._findings):
            sev = f.get("severity", "?")
            pct = f.get("stenosis_percent", 0)
            seg = f.get("segment_id", "?")
            self.finding_combo.addItem(f"#{i+1}: {sev} ({pct:.1f}%) - Seg {seg}")
        self.finding_combo.blockSignals(False)

        center = [
            image.shape[0] // 2,
            image.shape[1] // 2,
            image.shape[2] // 2,
        ]
        self._mpr_slices = list(center)
        self._crosshair_pos = list(center)

        current = self.axis_combo.currentText()
        if current == _MPR_LABEL:
            self._mpr_mode = True
            self._active_mpr_dim = 0
            self._view_stack.setCurrentIndex(1)
            self._setup_mpr_slider()
            self._highlight_active_panel()
        else:
            self._mpr_mode = False
            self._view_stack.setCurrentIndex(0)
            self.slice_slider.setEnabled(True)
            self.slice_spinbox.setEnabled(True)
            self._update_slider_range()
            dim = get_axis_dim(current)
            mid = image.shape[dim] // 2
            self.slice_slider.blockSignals(True)
            self.slice_spinbox.blockSignals(True)
            self.slice_slider.setValue(mid)
            self.slice_spinbox.setValue(mid)
            self.slice_slider.blockSignals(False)
            self.slice_spinbox.blockSignals(False)

        self._render()

    def clear_data(self):
        self._image = None
        self._mask = None
        self._findings = []
        self._region_mask = None
        self._mpr_slices = [0, 0, 0]
        self._crosshair_pos = [0, 0, 0]
        self._active_mpr_dim = 0
        self._slice_label.setText("Slice")
        self.finding_combo.clear()
        self.region_checkbox.setVisible(False)
        self._single_panel.clear()
        for panel in self._mpr_panels:
            panel.clear()
            panel.plot_widget.setStyleSheet("")
        self.slice_slider.setValue(0)
        self.slice_spinbox.setValue(0)

    def _update_slider_range(self):
        if self._image is None or self._mpr_mode:
            return
        axis = self.axis_combo.currentText()
        if axis == _MPR_LABEL:
            return
        dim = get_axis_dim(axis)
        max_val = self._image.shape[dim] - 1
        self.slice_slider.blockSignals(True)
        self.slice_spinbox.blockSignals(True)
        self.slice_slider.setMaximum(max_val)
        self.slice_spinbox.setMaximum(max_val)
        if self.slice_slider.value() > max_val:
            self.slice_slider.setValue(max_val)
            self.slice_spinbox.setValue(max_val)
        self.slice_slider.blockSignals(False)
        self.slice_spinbox.blockSignals(False)

    def _setup_mpr_slider(self):
        self.slice_slider.setEnabled(True)
        self.slice_spinbox.setEnabled(True)
        if self._image is None:
            return
        dim = self._active_mpr_dim
        max_val = self._image.shape[dim] - 1
        current_val = min(self._mpr_slices[dim], max_val)
        self._slice_label.setText(f"Slice \u2014 {_AXIS_NAMES[dim]}")
        self.slice_slider.blockSignals(True)
        self.slice_spinbox.blockSignals(True)
        self.slice_slider.setMaximum(max_val)
        self.slice_spinbox.setMaximum(max_val)
        self.slice_slider.setValue(current_val)
        self.slice_spinbox.setValue(current_val)
        self.slice_slider.blockSignals(False)
        self.slice_spinbox.blockSignals(False)

    def _on_slider_changed(self, value):
        self.slice_spinbox.blockSignals(True)
        self.slice_spinbox.setValue(value)
        self.slice_spinbox.blockSignals(False)
        if self._mpr_mode and self._image is not None:
            delta = value - self._mpr_slices[self._active_mpr_dim]
            for i in range(3):
                self._mpr_slices[i] = max(0, min(
                    self._mpr_slices[i] + delta,
                    self._image.shape[i] - 1,
                ))
        self._render()

    def _on_spinbox_changed(self, value):
        self.slice_slider.blockSignals(True)
        self.slice_slider.setValue(value)
        self.slice_slider.blockSignals(False)
        if self._mpr_mode and self._image is not None:
            delta = value - self._mpr_slices[self._active_mpr_dim]
            for i in range(3):
                self._mpr_slices[i] = max(0, min(
                    self._mpr_slices[i] + delta,
                    self._image.shape[i] - 1,
                ))
        self._render()

    def _on_axis_changed(self, text):
        if text == _MPR_LABEL:
            self._mpr_mode = True
            self._active_mpr_dim = 0  # Default to Axial
            self._view_stack.setCurrentIndex(1)
            if self._image is not None:
                center = [
                    self._image.shape[0] // 2,
                    self._image.shape[1] // 2,
                    self._image.shape[2] // 2,
                ]
                self._mpr_slices = list(center)
                self._crosshair_pos = list(center)
            self._setup_mpr_slider()
            self._highlight_active_panel()
            self._render()
            self._schedule_auto_fit()
        else:
            self._mpr_mode = False
            self._slice_label.setText("Slice")  # Reset label
            self._view_stack.setCurrentIndex(0)
            self.slice_slider.setEnabled(True)
            self.slice_spinbox.setEnabled(True)
            self._update_slider_range()
            if self._image is not None:
                dim = get_axis_dim(text)
                mid = self._image.shape[dim] // 2
                self.slice_slider.blockSignals(True)
                self.slice_spinbox.blockSignals(True)
                self.slice_slider.setValue(mid)
                self.slice_spinbox.setValue(mid)
                self.slice_slider.blockSignals(False)
                self.slice_spinbox.blockSignals(False)
            self._render()

    def _on_finding_selected(self, index):
        if index <= 0 or self._image is None or not self._findings:
            return
        finding_idx = index - 1
        if finding_idx >= len(self._findings):
            return

        finding = self._findings[finding_idx]
        loc = finding.get("location_voxel", [0, 0, 0])

        if self._mpr_mode:
            target = [
                min(int(loc[0]), self._image.shape[0] - 1),
                min(int(loc[1]), self._image.shape[1] - 1),
                min(int(loc[2]), self._image.shape[2] - 1),
            ]
            self._mpr_slices = list(target)
            self._crosshair_pos = list(target)
            self._sync_slider_from_slices()
            self._render()
        else:
            self.axis_combo.blockSignals(True)
            self.axis_combo.setCurrentText("Axial (Z)")
            self.axis_combo.blockSignals(False)
            self._update_slider_range()
            slice_idx = min(int(loc[0]), self._image.shape[0] - 1)
            self.slice_slider.setValue(slice_idx)  # Triggers render

        self.finding_selected.emit(finding_idx)

    def navigate_to_finding(self, finding_index: int):
        """Navigate to a specific finding by index (e.g. from 3D click)."""
        if self._image is None or not self._findings:
            return
        if finding_index < 0 or finding_index >= len(self._findings):
            return
        self.finding_combo.setCurrentIndex(finding_index + 1)  # +1 for placeholder

    def navigate_to_voxel(self, d: int, h: int, w: int):
        """Navigate to voxel (d, h, w), switching to MPR if needed."""
        if self._image is None:
            return

        shape = self._image.shape
        d = max(0, min(d, shape[0] - 1))
        h = max(0, min(h, shape[1] - 1))
        w = max(0, min(w, shape[2] - 1))

        if not self._mpr_mode:
            self.axis_combo.blockSignals(True)
            self.axis_combo.setCurrentText(_MPR_LABEL)
            self.axis_combo.blockSignals(False)
            self._mpr_mode = True
            self._view_stack.setCurrentIndex(1)
            self._setup_mpr_slider()

        self._mpr_slices = [d, h, w]
        self._crosshair_pos = [d, h, w]
        self._sync_slider_from_slices()
        self._render()

    def set_region_mask(self, mask_3d: np.ndarray):
        self._region_mask = mask_3d.astype(np.uint8) if mask_3d is not None else None
        self.region_checkbox.setVisible(self._region_mask is not None)
        self.region_checkbox.setChecked(True)
        self._render()

    def clear_region_mask(self):
        self._region_mask = None
        self.region_checkbox.setVisible(False)
        for panel in self._mpr_panels:
            panel.region_item.clear()
            panel.region_item.setVisible(False)
        self._single_panel.region_item.clear()
        self._single_panel.region_item.setVisible(False)
        self._render()

    def _sync_slider_from_slices(self):
        if self._image is None:
            return
        dim = self._active_mpr_dim
        max_val = self._image.shape[dim] - 1
        val = max(0, min(self._mpr_slices[dim], max_val))
        self.slice_slider.blockSignals(True)
        self.slice_spinbox.blockSignals(True)
        self.slice_slider.setMaximum(max_val)
        self.slice_spinbox.setMaximum(max_val)
        self.slice_slider.setValue(val)
        self.slice_spinbox.setValue(val)
        self.slice_slider.blockSignals(False)
        self.slice_spinbox.blockSignals(False)

    def _on_reset_view(self):
        if self._mpr_mode:
            self._syncing_range = True
            for panel in self._mpr_panels:
                panel.plot_widget.autoRange()
            QTimer.singleShot(0, lambda: setattr(self, '_syncing_range', False))
        else:
            self._single_panel.plot_widget.autoRange()

    def _schedule_auto_fit(self, attempts=0):
        """Retry autoRange() until panels have valid geometry."""
        if not self._mpr_mode:
            return
        panel = self._mpr_panels[0]
        if panel.plot_widget.height() > 10 and panel.plot_widget.width() > 10:
            self._on_reset_view()
        elif attempts < 30:  # retry up to ~600ms
            QTimer.singleShot(20, lambda: self._schedule_auto_fit(attempts + 1))

    def _render(self, *_args):
        if self._image is None:
            return
        if self._mpr_mode:
            self._render_mpr()
        else:
            axis = self.axis_combo.currentText()
            if axis == _MPR_LABEL:
                return
            slice_idx = self.slice_slider.value()
            show_overlay = self.overlay_checkbox.isChecked()
            wmin, wmax = WINDOW_PRESETS.get(
                self.window_combo.currentText(), (-3.0, 3.0)
            )
            self._render_panel(
                self._single_panel, axis, slice_idx, show_overlay, wmin, wmax
            )

    def _render_mpr(self):
        show_overlay = self.overlay_checkbox.isChecked()
        wmin, wmax = WINDOW_PRESETS.get(
            self.window_combo.currentText(), (-3.0, 3.0)
        )
        for panel in self._mpr_panels:
            dim = get_axis_dim(panel.axis_name)
            slice_idx = self._mpr_slices[dim]
            self._render_panel(
                panel, panel.axis_name, slice_idx, show_overlay, wmin, wmax
            )
        self._update_crosshairs()

    def _render_panel(self, panel: _SlicePanel, axis: str, slice_idx: int,
                      show_overlay: bool, wmin: float, wmax: float):
        img_slice = get_slice(self._image, axis, slice_idx)

        clipped = np.clip(img_slice, wmin, wmax)
        if wmax > wmin:
            normalized = ((clipped - wmin) / (wmax - wmin) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(clipped, dtype=np.uint8)

        panel.image_item.setImage(
            normalized.T, autoLevels=False, levels=(0, 255)
        )

        if show_overlay and self._mask is not None:
            mask_slice = get_slice(self._mask, axis, slice_idx)
            if mask_slice.any():
                rgba = np.zeros((*mask_slice.shape, 4), dtype=np.uint8)
                rgba[mask_slice > 0] = [255, 80, 80, 90]
                panel.mask_item.setImage(rgba.transpose(1, 0, 2))
                panel.mask_item.setVisible(True)
            else:
                panel.mask_item.setVisible(False)
        else:
            panel.mask_item.setVisible(False)

        show_region = self.region_checkbox.isChecked() and self.region_checkbox.isVisible()
        if show_region and self._region_mask is not None:
            region_slice = get_slice(self._region_mask, axis, slice_idx)
            if region_slice.any():
                rgba = np.zeros((*region_slice.shape, 4), dtype=np.uint8)
                rgba[region_slice > 0] = [0, 210, 230, 100]
                panel.region_item.setImage(rgba.transpose(1, 0, 2))
                panel.region_item.setVisible(True)
            else:
                panel.region_item.setVisible(False)
        else:
            panel.region_item.setVisible(False)

        if self._findings:
            nearby = get_nearby_findings(self._findings, axis, slice_idx)
            if nearby:
                spots = []
                for f in nearby:
                    x, y = _finding_2d_coords(f, axis)
                    sev = f.get("severity", "Normal")
                    color = _SEVERITY_RGBA.get(sev, (255, 255, 255, 220))
                    size = _SEVERITY_SIZES.get(sev, 12)
                    pct = f.get("stenosis_percent", 0)
                    seg = f.get("segment_id", "?")
                    spots.append({
                        'pos': (x, y),
                        'size': size,
                        'brush': pg.mkBrush(*color),
                        'pen': pg.mkPen('k', width=1.5),
                        'symbol': 'x',
                        'data': f"{sev}: {pct:.1f}% (Seg {seg})",
                    })
                panel.marker_item.setData(spots)
                panel.marker_item.setVisible(True)
            else:
                panel.marker_item.clear()
                panel.marker_item.setVisible(False)
        else:
            panel.marker_item.clear()
            panel.marker_item.setVisible(False)

        if axis in AXIS_MAP:
            _, dim_label, x_label, y_label = AXIS_MAP[axis]
            panel.plot_widget.setLabel("bottom", x_label)
            panel.plot_widget.setLabel("left", y_label)
            panel.plot_widget.setTitle(
                f"{axis} | Slice {slice_idx} | {slice_idx * 0.5:.1f} mm"
            )

    def _update_crosshairs(self):
        cz, cy, cx = self._crosshair_pos  # D, H, W

        for panel in self._mpr_panels:
            dim = get_axis_dim(panel.axis_name)
            panel.h_line.blockSignals(True)
            panel.v_line.blockSignals(True)
            if dim == 0:    # Axial: H vs W plane
                panel.v_line.setValue(cx)
                panel.h_line.setValue(cy)
            elif dim == 1:  # Coronal: D vs W plane
                panel.v_line.setValue(cx)
                panel.h_line.setValue(cz)
            else:           # Sagittal: D vs H plane
                panel.v_line.setValue(cy)
                panel.h_line.setValue(cz)
            panel.h_line.setVisible(True)
            panel.v_line.setVisible(True)
            panel.h_line.blockSignals(False)
            panel.v_line.blockSignals(False)

    def _on_mpr_click(self, event, panel: _SlicePanel):
        if not self._mpr_mode or self._image is None:
            return

        pos = event.scenePos()
        mouse_point = panel.plot_widget.plotItem.vb.mapSceneToView(pos)
        click_x = int(round(mouse_point.x()))
        click_y = int(round(mouse_point.y()))

        dim = get_axis_dim(panel.axis_name)
        shape = self._image.shape

        if self._active_mpr_dim != dim:
            self._set_active_mpr_panel(dim)

        if dim == 0:    # Axial: click_x=W, click_y=H
            self._crosshair_pos[2] = max(0, min(click_x, shape[2] - 1))
            self._crosshair_pos[1] = max(0, min(click_y, shape[1] - 1))
        elif dim == 1:  # Coronal: click_x=W, click_y=D
            self._crosshair_pos[2] = max(0, min(click_x, shape[2] - 1))
            self._crosshair_pos[0] = max(0, min(click_y, shape[0] - 1))
        else:           # Sagittal: click_x=H, click_y=D
            self._crosshair_pos[1] = max(0, min(click_x, shape[1] - 1))
            self._crosshair_pos[0] = max(0, min(click_y, shape[0] - 1))

        self._update_crosshairs()

    def _on_crosshair_dragged(self, panel: _SlicePanel, direction: str):
        if not self._mpr_mode or self._image is None:
            return

        dim = get_axis_dim(panel.axis_name)
        shape = self._image.shape

        if direction == 'h':
            val = int(round(panel.h_line.value()))
        else:
            val = int(round(panel.v_line.value()))

        if dim == 0:    # Axial: h_line=H(cy), v_line=W(cx)
            if direction == 'h':
                self._crosshair_pos[1] = max(0, min(val, shape[1] - 1))
            else:
                self._crosshair_pos[2] = max(0, min(val, shape[2] - 1))
        elif dim == 1:  # Coronal: h_line=D(cz), v_line=W(cx)
            if direction == 'h':
                self._crosshair_pos[0] = max(0, min(val, shape[0] - 1))
            else:
                self._crosshair_pos[2] = max(0, min(val, shape[2] - 1))
        else:           # Sagittal: h_line=D(cz), v_line=H(cy)
            if direction == 'h':
                self._crosshair_pos[0] = max(0, min(val, shape[0] - 1))
            else:
                self._crosshair_pos[1] = max(0, min(val, shape[1] - 1))

        self._update_crosshairs()

    def _on_mpr_range_changed(self, source_panel: _SlicePanel, ranges):
        if self._syncing_range or not self._mpr_mode:
            return

        self._syncing_range = True
        try:
            x_range, y_range = ranges
            for panel in self._mpr_panels:
                if panel is not source_panel:
                    panel.plot_widget.blockSignals(True)
                    panel.plot_widget.setXRange(x_range[0], x_range[1], padding=0)
                    panel.plot_widget.setYRange(y_range[0], y_range[1], padding=0)
                    panel.plot_widget.blockSignals(False)
        finally:
            self._syncing_range = False

    def _set_active_mpr_panel(self, dim: int):
        self._active_mpr_dim = dim
        self._slice_label.setText(f"Slice \u2014 {_AXIS_NAMES[dim]}")
        self._sync_slider_from_slices()
        self._highlight_active_panel()

    def _highlight_active_panel(self):
        for i, panel in enumerate(self._mpr_panels):
            dim = get_axis_dim(panel.axis_name)
            if dim == self._active_mpr_dim:
                panel.plot_widget.setStyleSheet(
                    "border: 2px solid #00BCD4;"  # cyan highlight
                )
            else:
                panel.plot_widget.setStyleSheet("")

    def eventFilter(self, obj, event):
        if self._mpr_mode and self._image is not None:
            if event.type() == QEvent.Type.Wheel:
                for panel in self._mpr_panels:
                    if obj is panel.plot_widget:
                        dim = get_axis_dim(panel.axis_name)
                        if self._active_mpr_dim != dim:
                            self._set_active_mpr_panel(dim)
                        delta = 1 if event.angleDelta().y() > 0 else -1
                        for i in range(3):
                            self._mpr_slices[i] = max(0, min(
                                self._mpr_slices[i] + delta,
                                self._image.shape[i] - 1,
                            ))
                        self._sync_slider_from_slices()
                        self._render_mpr()
                        return True  # Consume event (no zoom)
        return super().eventFilter(obj, event)
