"""Stenosis Detection Desktop App — PySide6/PyQt."""

import json as _json
import re
import sys
import shutil
from pathlib import Path

_APP_DIR = Path(__file__).resolve().parent
_SOURCE_DIR = _APP_DIR.parent
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))

import numpy as np
import zarr

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit,
    QLabel, QTabWidget, QTextEdit, QGroupBox, QFrame,
    QFileDialog, QMessageBox, QCheckBox, QComboBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QUrl, QTimer
from PySide6.QtGui import QFont, QColor, QIcon
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineSettings
from PySide6.QtWebChannel import QWebChannel

from pipeline.runner import (
    PIPELINE_STAGES, TEMP_DIR,
    detect_input_type, generate_clinical_report,
    pipeline_stages,
)
from pipeline import slice_viewer
from pipeline.compare_gt import compare_findings, format_comparison_text
from pipeline.region_helpers import compute_region_mask
from widgets.slice_viewer_widget import SliceViewerWidget
from widgets.plotly_bridge import PlotlyBridge


class _ZarrLoaderWorker(QThread):
    """Background loader for zarr image+mask (keeps UI responsive)."""
    loaded = Signal(object, object, list)  # image, mask, findings

    def __init__(self, zarr_path: str, findings: list, parent=None):
        super().__init__(parent)
        self.zarr_path = zarr_path
        self.findings = findings

    def run(self):
        try:
            g = zarr.open_group(self.zarr_path, mode='r')
            image = g['image'][:]
            mask = g['mask'][:]
            self.loaded.emit(image, mask, self.findings)
        except Exception as e:
            print("Error loading slice viewer data:", e)


class RegionMaskWorker(QThread):
    """Background worker for region mask computation."""

    finished = Signal(object, object, object, str)  # region_mask, path_indices, all_points, status_msg
    error = Signal(str)

    def __init__(self, centerline_data, seg_mask, start_voxel, end_voxel, parent=None):
        super().__init__(parent)
        self._centerline_data = centerline_data
        self._seg_mask = seg_mask
        self._start_voxel = start_voxel
        self._end_voxel = end_voxel
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        if self._cancel_requested:
            return
        try:
            result = compute_region_mask(
                self._centerline_data, self._seg_mask,
                self._start_voxel, self._end_voxel
            )
            if self._cancel_requested:
                return
            region_mask, path_indices, all_points, status_msg = result
            self.finished.emit(region_mask, path_indices, all_points, status_msg)
        except Exception as exc:
            self.error.emit(str(exc))


class PipelineWorker(QThread):
    """Runs the stenosis detection pipeline in a background thread."""

    stage_update = Signal(int, str, list)  # stage_idx, state, status_lines
    finished = Signal(object)               # final result tuple (6 elements)
    error = Signal(str)                     # error message
    cancelled = Signal()                    # emitted when cancelled

    def __init__(self, input_path: str, enable_postprocess: bool = True,
                 enable_refinement: bool = False, model_dir=None, parent=None):
        super().__init__(parent)
        self.input_path = input_path
        self.enable_postprocess = enable_postprocess
        self.enable_refinement = enable_refinement
        self.model_dir = model_dir
        self._cancel_requested = False

    def cancel(self):
        self._cancel_requested = True

    def run(self):
        try:
            for stage_idx, state, status_lines, result in pipeline_stages(
                self.input_path,
                enable_postprocess=self.enable_postprocess,
                enable_refinement=self.enable_refinement,
                model_dir=self.model_dir,
            ):
                if self._cancel_requested:
                    self.cancelled.emit()
                    return
                if result is not None:
                    if state == "error":
                        self.error.emit(result[0])  # status_text contains error
                    else:
                        self.finished.emit(result)
                    return
                self.stage_update.emit(stage_idx, state, list(status_lines))
        except Exception as e:
            if not self._cancel_requested:
                self.error.emit(str(e))


class StepperWidget(QWidget):
    """Pipeline progress stepper (one indicator per stage)."""

    ICONS = {"waiting": "\u25CB", "running": "\u2699", "done": "\u2713", "failed": "\u2717"}
    COLORS = {
        "waiting": "#9CA3AF",
        "running": "#2563EB",
        "done": "#059669",
        "failed": "#DC2626",
    }
    BG_COLORS = {
        "waiting": "#F3F4F6",
        "running": "#DBEAFE",
        "done": "#D1FAE5",
        "failed": "#FEE2E2",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        self._icon_labels = []
        self._text_labels = []

        for name, desc in PIPELINE_STAGES:
            row = QHBoxLayout()
            row.setSpacing(8)

            icon_label = QLabel(self.ICONS["waiting"])
            icon_label.setFixedSize(28, 28)
            icon_label.setAlignment(Qt.AlignCenter)
            icon_label.setStyleSheet(
                f"background: {self.BG_COLORS['waiting']}; "
                f"color: {self.COLORS['waiting']}; "
                "border-radius: 14px; font-size: 13px; font-weight: bold;"
            )
            row.addWidget(icon_label)

            text_widget = QWidget()
            text_layout = QVBoxLayout(text_widget)
            text_layout.setContentsMargins(0, 0, 0, 0)
            text_layout.setSpacing(0)

            name_label = QLabel(name)
            name_label.setStyleSheet("font-size: 13px; font-weight: 500; color: #374151;")
            desc_label = QLabel(desc)
            desc_label.setStyleSheet("font-size: 11px; color: #9CA3AF;")
            text_layout.addWidget(name_label)
            text_layout.addWidget(desc_label)

            row.addWidget(text_widget, stretch=1)
            layout.addLayout(row)

            self._icon_labels.append(icon_label)
            self._text_labels.append((name_label, desc_label))

    def update_stage(self, stage_idx: int, state: str):
        if 0 <= stage_idx < len(self._icon_labels):
            icon = self.ICONS.get(state, self.ICONS["waiting"])
            color = self.COLORS.get(state, self.COLORS["waiting"])
            bg = self.BG_COLORS.get(state, self.BG_COLORS["waiting"])
            self._icon_labels[stage_idx].setText(icon)
            self._icon_labels[stage_idx].setStyleSheet(
                f"background: {bg}; color: {color}; "
                "border-radius: 14px; font-size: 13px; font-weight: bold;"
            )

    def set_all(self, state: str):
        for i in range(len(PIPELINE_STAGES)):
            self.update_stage(i, state)

    def reset(self):
        self.set_all("waiting")


def discover_gt_report(input_path: str):
    """Find and load the GT report JSON for a Ziv case, or return None."""
    path = Path(input_path).resolve()

    # zarr input: look for sibling <stem>Report.json
    if path.suffix.lower() == '.zarr':
        report_path = path.parent / f"{path.stem}Report.json"
        if report_path.exists():
            try:
                with open(report_path, "r") as f:
                    return _json.load(f)
            except Exception:
                pass

    case_id = None
    for part in path.parts:
        if re.match(r"^case-\d+$", part):
            case_id = part
            break

    if case_id is None:
        return None

    ziv_root = None
    for parent in path.parents:
        if (parent / case_id).is_dir():
            ziv_root = parent
            break

    if ziv_root is None:
        return None

    manifest_path = ziv_root / "ziv_manifest.json"
    if not manifest_path.exists():
        return None

    try:
        with open(manifest_path, "r") as f:
            manifest = _json.load(f)
    except Exception:
        return None

    case_entry = manifest.get("cases", {}).get(case_id)
    if not case_entry or not case_entry.get("gt_report"):
        return None

    gt_path = ziv_root / case_entry["gt_report"]
    if not gt_path.exists():
        return None

    try:
        with open(gt_path, "r") as f:
            return _json.load(f)
    except Exception:
        return None


def discover_models():
    """Scan models/ for valid model directories and return (label, Path) list."""
    models_root = Path(__file__).resolve().parent / "models"
    results = []
    if not models_root.exists():
        return results
    for d in sorted(models_root.iterdir()):
        if not d.is_dir():
            continue
        cfg_path = d / "run_config.json"
        ckpt_path = d / "checkpoints" / "best_model.pth"
        if not cfg_path.exists() or not ckpt_path.exists():
            continue
        try:
            with open(cfg_path, "r") as f:
                cfg = _json.load(f)
        except Exception:
            continue
        tc = cfg.get("training_config", cfg)
        model_type = tc.get("model_type", "?")
        loss_type = tc.get("loss_type", "")
        fm = cfg.get("final_metrics") or {}
        dice = fm.get("best_val_dice")
        parts = [d.name, model_type]
        if loss_type:
            parts.append(loss_type)
        if dice is not None:
            parts.append(f"Dice={dice:.4f}")
        else:
            desc = cfg.get("description", "")
            dice_match = re.search(r"best_val_dice=([\d.]+)", desc)
            if dice_match:
                parts.append(f"Dice={dice_match.group(1)}")
        label = "  |  ".join(parts)
        results.append((label, d))
    return results


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Stenosis Detection \u2014 University of Haifa & Ziv Medical Center"
        )
        self.resize(1400, 900)

        self._worker = None
        self._zarr_loader = None
        self._region_worker = None
        self._last_log_line_count = 0

        self._current_findings = []
        self._centerline_data = None
        self._current_mask = None
        self._selection_mode = "navigate"

        self._setup_ui()
        self._setup_web_channel()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(12, 12, 12, 12)

        header = QLabel("Coronary Artery Stenosis Detection System")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #1F2937;")
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)

        subtitle = QLabel(
            "University of Haifa & Ziv Medical Center | Research Demo"
        )
        subtitle.setStyleSheet("font-size: 12px; color: #6B7280;")
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #E5E7EB;")
        main_layout.addWidget(sep)

        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter, stretch=1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setSpacing(10)

        input_group = QGroupBox("Input")
        input_layout = QVBoxLayout(input_group)

        btn_row = QHBoxLayout()
        self.browse_btn = QPushButton("Browse Folder")
        self.browse_btn.clicked.connect(self._on_browse)
        self.example_btn = QPushButton("Load Ziv Example")
        self.example_btn.clicked.connect(self._on_load_example)
        btn_row.addWidget(self.browse_btn)
        btn_row.addWidget(self.example_btn)
        input_layout.addLayout(btn_row)

        self.path_input = QLineEdit()
        self.path_input.setPlaceholderText("Click 'Browse Folder' or type path...")
        input_layout.addWidget(self.path_input)

        left_layout.addWidget(input_group)

        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        self.model_combo = QComboBox()
        for label, path in discover_models():
            self.model_combo.addItem(label, userData=path)
        self.model_combo.currentIndexChanged.connect(self._on_model_selected)
        model_layout.addWidget(self.model_combo)
        self.model_detail = QLabel("")
        self.model_detail.setStyleSheet("font-size: 11px; color: #6B7280; padding: 2px;")
        self.model_detail.setWordWrap(True)
        model_layout.addWidget(self.model_detail)
        left_layout.addWidget(model_group)
        self._update_model_detail()

        action_row = QHBoxLayout()

        self.run_btn = QPushButton("Run Analysis")
        self.run_btn.setStyleSheet(
            "QPushButton { background-color: #2563EB; color: white; "
            "font-size: 14px; font-weight: bold; padding: 10px; "
            "border-radius: 6px; }"
            "QPushButton:hover { background-color: #1D4ED8; }"
            "QPushButton:disabled { background-color: #93C5FD; }"
        )
        self.run_btn.clicked.connect(self._on_run)
        action_row.addWidget(self.run_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setStyleSheet(
            "QPushButton { background-color: #DC2626; color: white; "
            "font-size: 14px; font-weight: bold; padding: 10px; "
            "border-radius: 6px; }"
            "QPushButton:hover { background-color: #B91C1C; }"
            "QPushButton:disabled { background-color: #FCA5A5; color: #FEE2E2; }"
        )
        self.cancel_btn.clicked.connect(self._on_cancel)
        self.cancel_btn.setEnabled(False)
        action_row.addWidget(self.cancel_btn)

        left_layout.addLayout(action_row)

        self.clear_btn = QPushButton("Clear Results")
        self.clear_btn.clicked.connect(self._on_clear)
        left_layout.addWidget(self.clear_btn)

        self.pp_checkbox = QCheckBox("Enable Post-Processing")
        self.pp_checkbox.setChecked(True)
        self.pp_checkbox.setToolTip(
            "When enabled, applies 8-step mask cleanup after segmentation.\n"
            "Disable to test raw segmentation output."
        )
        self.pp_checkbox.setStyleSheet("font-size: 12px; padding: 4px;")
        left_layout.addWidget(self.pp_checkbox)

        self.refine_checkbox = QCheckBox("Enable Re-segmentation (experimental)")
        self.refine_checkbox.setChecked(False)
        self.refine_checkbox.setToolTip(
            "When enabled, uses a separate BCE model to re-segment gap regions\n"
            "between disconnected vessel fragments. May improve connectivity."
        )
        self.refine_checkbox.setStyleSheet("font-size: 12px; padding: 4px;")
        left_layout.addWidget(self.refine_checkbox)

        status_group = QGroupBox("Pipeline Status")
        status_layout = QVBoxLayout(status_group)
        self.stepper = StepperWidget()
        status_layout.addWidget(self.stepper)
        left_layout.addWidget(status_group)

        log_group = QGroupBox("Detailed Log")
        log_group.setCheckable(True)
        log_group.setChecked(False)
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Consolas", 9))
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        log_group.toggled.connect(
            lambda checked: self.log_text.setVisible(checked)
        )
        self.log_text.setVisible(False)
        left_layout.addWidget(log_group)

        legend = QLabel(
            "Severity: Normal: <25% | Mild: 25-50% | "
            "Moderate: 50-70% | Severe: \u226570%"
        )
        legend.setStyleSheet("font-size: 11px; color: #6B7280; padding: 4px;")
        legend.setWordWrap(True)
        left_layout.addWidget(legend)

        left_layout.addStretch()

        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)

        results_label = QLabel("Results")
        results_label.setStyleSheet(
            "font-size: 16px; font-weight: bold; color: #374151;"
        )
        right_layout.addWidget(results_label)

        self.tabs = QTabWidget()

        viz_container = QWidget()
        viz_layout = QVBoxLayout(viz_container)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(0)

        viz_toolbar = QHBoxLayout()
        viz_toolbar.setContentsMargins(6, 4, 6, 4)

        self.viz_status_label = QLabel(
            "Click stenosis marker to jump | Ctrl+Click vessel to navigate"
        )
        self.viz_status_label.setStyleSheet(
            "font-size: 10px; color: #9CA3AF; padding: 0 8px;"
        )
        viz_toolbar.addWidget(self.viz_status_label)

        viz_toolbar.addStretch()

        _toolbar_btn_style = (
            "QPushButton { background-color: #374151; color: #D1D5DB; "
            "font-size: 11px; padding: 4px 12px; border-radius: 4px; "
            "border: 1px solid #4B5563; }"
            "QPushButton:hover { background-color: #4B5563; color: white; }"
            "QPushButton:disabled { color: #6B7280; border-color: #374151; }"
        )

        self.select_region_btn = QPushButton("Select Region")
        self.select_region_btn.setToolTip(
            "Click two points on the vessel to highlight a region in 2D"
        )
        self.select_region_btn.setStyleSheet(_toolbar_btn_style)
        self.select_region_btn.setEnabled(False)
        self.select_region_btn.clicked.connect(self._on_select_region_toggle)
        viz_toolbar.addWidget(self.select_region_btn)

        self.clear_region_btn = QPushButton("Clear Region")
        self.clear_region_btn.setToolTip("Remove the highlighted region from 2D view")
        self.clear_region_btn.setStyleSheet(_toolbar_btn_style)
        self.clear_region_btn.setEnabled(False)
        self.clear_region_btn.clicked.connect(self._on_clear_region)
        viz_toolbar.addWidget(self.clear_region_btn)

        self.reset_cam_btn = QPushButton("\u2302  Reset Camera")
        self.reset_cam_btn.setToolTip(
            "Reset 3D view to the default camera angle"
        )
        self.reset_cam_btn.setStyleSheet(_toolbar_btn_style)
        self.reset_cam_btn.setEnabled(False)
        self.reset_cam_btn.clicked.connect(self._on_reset_camera)
        viz_toolbar.addWidget(self.reset_cam_btn)
        viz_layout.addLayout(viz_toolbar)

        self.web_view = QWebEngineView()
        self.web_view.settings().setAttribute(
            QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True
        )
        self.web_view.loadFinished.connect(self._on_viz_loaded)
        self.web_view.setHtml(
            self._placeholder_html(
                "Run analysis to see interactive 3D visualization"
            )
        )
        viz_layout.addWidget(self.web_view, stretch=1)
        self.tabs.addTab(viz_container, "3D Visualization")

        self.slice_viewer = SliceViewerWidget()
        self.tabs.addTab(self.slice_viewer, "2D Slice Viewer")

        self.report_text = QTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setFont(QFont("Consolas", 10))
        self.report_text.setPlainText(
            "Run analysis to generate clinical report..."
        )
        self.tabs.addTab(self.report_text, "Clinical Report")

        self.json_text = QTextEdit()
        self.json_text.setReadOnly(True)
        self.json_text.setFont(QFont("Consolas", 10))
        self.tabs.addTab(self.json_text, "Raw Data (JSON)")

        self.gt_text = QTextEdit()
        self.gt_text.setReadOnly(True)
        self.gt_text.setFont(QFont("Consolas", 10))
        self.gt_text.setPlainText(
            "No ground truth report found for this case.\n\n"
            "GT reports are matched via data/ZIV/ziv_manifest.json."
        )
        self.tabs.addTab(self.gt_text, "GT Comparison")

        right_layout.addWidget(self.tabs, stretch=1)

        splitter.addWidget(right)

        splitter.setSizes([350, 1050])

        footer = QLabel(
            "This is an automated AI analysis for research purposes. "
            "Results should be validated by qualified medical professionals."
        )
        footer.setStyleSheet(
            "font-size: 11px; color: #9CA3AF; padding-top: 4px;"
        )
        footer.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(footer)

    @staticmethod
    def _placeholder_html(message: str) -> str:
        return (
            "<html><body style='display:flex; align-items:center; "
            "justify-content:center; height:100%; background:linear-gradient("
            "135deg,#f5f7fa,#e4e8eb); font-family:sans-serif;'>"
            "<p style='color:#888; font-size:16px;'>"
            + message
            + "</p></body></html>"
        )

    def _setup_web_channel(self):
        self._bridge = PlotlyBridge(self)
        self._bridge.set_spacing((0.5, 0.5, 0.5))

        self._web_channel = QWebChannel(self.web_view.page())
        self._web_channel.registerObject("bridge", self._bridge)
        self.web_view.page().setWebChannel(self._web_channel)

        self._bridge.stenosis_clicked.connect(self._on_3d_stenosis_clicked)
        self._bridge.vessel_clicked.connect(self._on_3d_vessel_clicked)
        self._bridge.region_selected.connect(self._on_3d_region_selected)
        self._bridge.status_message.connect(self._on_bridge_status)

    def _on_viz_loaded(self, ok: bool):
        """Nudge QWebEngineView to repaint (works around blank-until-hover Qt bug)."""
        if ok:
            sz = self.web_view.size()
            self.web_view.resize(sz.width() - 1, sz.height())
            self.web_view.resize(sz)
            self._inject_bridge_js()

    _RESET_CAMERA_JS = """
    (function() {
        var gd = document.querySelector('.plotly-graph-div');
        if (gd) {
            Plotly.relayout(gd, {
                'scene.camera': {
                    eye:    {x: 1.5, y: 1.5, z: 1.5},
                    center: {x: 0,   y: 0,   z: 0},
                    up:     {x: 0,   y: 0,   z: 1}
                }
            });
        }
    })();
    """

    def _on_reset_camera(self):
        self.web_view.page().runJavaScript(self._RESET_CAMERA_JS)

    _BRIDGE_JS = """
    (function() {
        if (window.__bridgeInitialized) return;
        window.__bridgeInitialized = true;

        window.__ctrlPressed = false;
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Control') window.__ctrlPressed = true;
        });
        document.addEventListener('keyup', function(e) {
            if (e.key === 'Control') window.__ctrlPressed = false;
        });
        window.addEventListener('blur', function() {
            window.__ctrlPressed = false;
        });

        window.__selectionMode = 'navigate';
        window.__selectionStart = null;

        function initBridge() {
            new QWebChannel(qt.webChannelTransport, function(channel) {
                window.__bridge = channel.objects.bridge;
                var checkPlotly = setInterval(function() {
                    var gd = document.querySelector('.plotly-graph-div');
                    if (gd && gd.data) {
                        clearInterval(checkPlotly);
                        setupClickHandler(gd);
                    }
                }, 200);
            });
        }

        function findStenosisIndex(x, y, z) {
            var findings = window.__stenosisFindings || [];
            var bestIdx = -1;
            var bestDist = Infinity;
            for (var i = 0; i < findings.length; i++) {
                var f = findings[i];
                var dx = f.x - x, dy = f.y - y, dz = f.z - z;
                var dist = dx*dx + dy*dy + dz*dz;
                if (dist < bestDist) { bestDist = dist; bestIdx = i; }
            }
            if (bestDist < 4.0) return bestIdx;
            return -1;
        }

        function isStenosisTrace(traceName) {
            if (!traceName) return false;
            var n = traceName.toLowerCase();
            return n.indexOf('severe') >= 0 || n.indexOf('moderate') >= 0 ||
                   n.indexOf('mild') >= 0 || n.indexOf('normal') >= 0;
        }

        function setupClickHandler(gd) {
            gd.on('plotly_click', function(eventData) {
                if (!eventData || !eventData.points || eventData.points.length === 0) return;
                var pt = eventData.points[0];
                var mm = [pt.x, pt.y, pt.z];
                var traceName = pt.data ? pt.data.name : '';

                if (traceName === '__selection__') return;

                if (isStenosisTrace(traceName)) {
                    var idx = findStenosisIndex(pt.x, pt.y, pt.z);
                    if (idx >= 0 && window.__bridge) {
                        window.__bridge.onStenosisClicked(JSON.stringify({
                            mm: mm, finding_index: idx
                        }));
                        return;
                    }
                }

                if (window.__selectionMode === 'select_start') {
                    window.__selectionStart = mm;
                    window.__selectionMode = 'select_end';
                    if (window.__bridge)
                        window.__bridge.onStatusUpdate('Click second point to complete region selection');
                    var _gd1 = gd, _mm1 = mm;
                    setTimeout(function() { addSelectionMarker(_gd1, _mm1, 'Start'); }, 0);
                    return;
                }

                if (window.__selectionMode === 'select_end') {
                    var _startMm = window.__selectionStart;
                    if (_startMm && mm[0] === _startMm[0] && mm[1] === _startMm[1] && mm[2] === _startMm[2]) {
                        return;
                    }
                    window.__selectionMode = 'navigate';
                    window.__selectionStart = null;
                    if (_startMm && window.__bridge) {
                        window.__bridge.onRegionSelected(JSON.stringify({
                            start_mm: _startMm, end_mm: mm
                        }));
                    }
                    var _gd2 = gd;
                    setTimeout(function() { clearSelectionMarkers(_gd2); }, 0);
                    return;
                }

                if (window.__ctrlPressed && window.__bridge) {
                    window.__bridge.onVesselClicked(JSON.stringify({mm: mm}));
                }
            });
        }

        function addSelectionMarker(gd, mm, label) {
            Plotly.addTraces(gd, {
                x: [mm[0]], y: [mm[1]], z: [mm[2]],
                mode: 'markers+text',
                marker: {size: 10, color: 'cyan', symbol: 'circle',
                         line: {color: 'white', width: 2}},
                text: [label], textposition: 'top center',
                textfont: {color: 'cyan', size: 12},
                name: '__selection__', showlegend: false,
                hoverinfo: 'skip', type: 'scatter3d'
            });
        }

        function clearSelectionMarkers(gd) {
            var toRemove = [];
            for (var i = 0; i < gd.data.length; i++) {
                if (gd.data[i].name === '__selection__') toRemove.push(i);
            }
            if (toRemove.length > 0) Plotly.deleteTraces(gd, toRemove);
        }

        var script = document.createElement('script');
        script.src = 'qrc:///qtwebchannel/qwebchannel.js';
        script.onload = initBridge;
        script.onerror = function() {
            if (typeof QWebChannel !== 'undefined') {
                initBridge();
            } else {
                console.error('[StenosisBridge] qrc:///qtwebchannel/qwebchannel.js failed to load.');
            }
        };
        document.head.appendChild(script);
    })();
    """

    def _inject_bridge_js(self):
        findings_js_items = []
        spacing = (0.5, 0.5, 0.5)
        for f in self._current_findings:
            voxel = f.get("location_voxel", [0, 0, 0])
            mm = [v * s for v, s in zip(voxel, spacing)]
            findings_js_items.append(
                f"{{x:{mm[2]:.4f},y:{mm[1]:.4f},z:{mm[0]:.4f}}}"
            )
        findings_js = "[" + ",".join(findings_js_items) + "]"

        inject = f"window.__stenosisFindings = {findings_js};\n" + self._BRIDGE_JS
        self.web_view.page().runJavaScript(inject)

    def _on_3d_stenosis_clicked(self, finding_index, voxel):
        self.tabs.setCurrentIndex(1)
        self.slice_viewer.navigate_to_finding(finding_index)

    def _on_3d_vessel_clicked(self, voxel):
        self.tabs.setCurrentIndex(1)
        self.slice_viewer.navigate_to_voxel(voxel[0], voxel[1], voxel[2])

    def _on_3d_region_selected(self, start_voxel, end_voxel):
        self.tabs.setCurrentIndex(1)
        self._compute_region_mask(start_voxel, end_voxel)

    def _on_bridge_status(self, message):
        self.viz_status_label.setText(message)

    def _set_selection_mode(self, mode):
        self._selection_mode = mode
        js = f"window.__selectionMode = '{mode}';"
        self.web_view.page().runJavaScript(js)
        if mode == "select_start":
            self.viz_status_label.setText("Click first point on vessel to start region selection")
            self.select_region_btn.setStyleSheet(
                "QPushButton { background-color: #06B6D4; color: white; "
                "font-size: 11px; padding: 4px 12px; border-radius: 4px; "
                "border: 1px solid #0891B2; }"
            )
        else:
            self.viz_status_label.setText(
                "Click stenosis marker to jump | Ctrl+Click vessel to navigate"
            )
            self.select_region_btn.setStyleSheet(
                "QPushButton { background-color: #374151; color: #D1D5DB; "
                "font-size: 11px; padding: 4px 12px; border-radius: 4px; "
                "border: 1px solid #4B5563; }"
                "QPushButton:hover { background-color: #4B5563; color: white; }"
                "QPushButton:disabled { color: #6B7280; border-color: #374151; }"
            )

    def _on_select_region_toggle(self):
        if self._selection_mode == "navigate":
            self._set_selection_mode("select_start")
        else:
            self._set_selection_mode("navigate")
            self.web_view.page().runJavaScript("""
                (function() {
                    var gd = document.querySelector('.plotly-graph-div');
                    if (gd) {
                        var toRemove = [];
                        for (var i = 0; i < gd.data.length; i++) {
                            if (gd.data[i].name === '__selection__') toRemove.push(i);
                        }
                        if (toRemove.length > 0) Plotly.deleteTraces(gd, toRemove);
                    }
                })();
            """)

    def _on_clear_region(self):
        self.slice_viewer.clear_region_mask()
        self.viz_status_label.setText(
            "Click stenosis marker to jump | Ctrl+Click vessel to navigate"
        )

    def _compute_region_mask(self, start_voxel, end_voxel):
        """Compute region mask between two centerline points (async)."""
        if self._centerline_data is None or self._current_mask is None:
            self.viz_status_label.setText("No centerline data available for region selection")
            return

        if self._region_worker is not None and self._region_worker.isRunning():
            self._region_worker.cancel()
            self._region_worker.wait(200)

        self.viz_status_label.setText("Computing region...")
        self.select_region_btn.setEnabled(False)

        self._region_worker = RegionMaskWorker(
            self._centerline_data, self._current_mask,
            start_voxel, end_voxel, parent=self
        )
        self._region_worker.finished.connect(self._on_region_mask_ready)
        self._region_worker.error.connect(self._on_region_mask_error)
        self._region_worker.start()

    def _on_region_mask_ready(self, region_mask, path_indices, all_points, status_msg):
        if region_mask is None:
            self.viz_status_label.setText(status_msg)
            self.select_region_btn.setEnabled(self._centerline_data is not None)
            return

        self.slice_viewer.set_region_mask(region_mask)
        self.viz_status_label.setText(status_msg)

        if path_indices:
            mid_idx = path_indices[len(path_indices) // 2]
            d, h, w, _ = all_points[mid_idx]
            self.slice_viewer.navigate_to_voxel(int(round(d)), int(round(h)), int(round(w)))
            self.slice_viewer._schedule_auto_fit()

        self.select_region_btn.setEnabled(self._centerline_data is not None)
        self.clear_region_btn.setEnabled(True)

    def _on_region_mask_error(self, msg):
        self.viz_status_label.setText(f"Region error: {msg}")
        self.select_region_btn.setEnabled(self._centerline_data is not None)

    def _on_browse(self):
        default_dir = str(Path(__file__).resolve().parent / "data")
        folder = QFileDialog.getExistingDirectory(
            self, "Select Patient DICOM Folder", default_dir
        )
        if folder:
            self.path_input.setText(folder)

    def _on_load_example(self):
        self.path_input.setText(
            str(Path(__file__).resolve().parent / "data" / "ziv" / "exampleCase.zarr")
        )

    def _on_model_selected(self, index):
        self._update_model_detail()

    def _update_model_detail(self):
        model_path = self.model_combo.currentData()
        if model_path is None:
            self.model_detail.setText("No models found")
            return
        cfg_path = model_path / "run_config.json"
        try:
            with open(cfg_path, "r") as f:
                cfg = _json.load(f)
            tc = cfg.get("training_config", cfg)
            parts = [f"Type: {tc.get('model_type', '?')}"]
            if tc.get("loss_type"):
                parts.append(f"Loss: {tc['loss_type']}")
            fm = cfg.get("final_metrics") or {}
            dice = fm.get("best_val_dice")
            if dice is not None:
                parts.append(f"Best Dice: {dice:.4f}")
            else:
                desc = cfg.get("description", "")
                dice_match = re.search(r"best_val_dice=([\d.]+)", desc)
                if dice_match:
                    parts.append(f"Best Dice: {dice_match.group(1)}")
            self.model_detail.setText("  |  ".join(parts))
        except Exception:
            self.model_detail.setText("Could not read model config")

    def _on_run(self):
        input_path = self.path_input.text().strip()
        if not input_path:
            QMessageBox.warning(
                self, "No Input",
                "Please enter a path to a DICOM folder, NIfTI file, or preprocessed .zarr.",
            )
            return

        if not Path(input_path).exists():
            QMessageBox.warning(
                self, "Path Not Found",
                "The path does not exist:\n" + input_path,
            )
            return

        if self._region_worker is not None and self._region_worker.isRunning():
            self._region_worker.cancel()
            self._region_worker.wait(200)
            self._region_worker = None

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Running...")
        self.cancel_btn.setEnabled(True)
        self.stepper.reset()
        self.log_text.clear()
        self._last_log_line_count = 0
        self.web_view.setHtml(
            self._placeholder_html("Processing...")
        )
        self.report_text.setPlainText("Processing...")
        self.json_text.clear()
        self.slice_viewer.clear_data()

        selected_model = self.model_combo.currentData()
        self._worker = PipelineWorker(
            input_path,
            enable_postprocess=self.pp_checkbox.isChecked(),
            enable_refinement=self.refine_checkbox.isChecked(),
            model_dir=str(selected_model) if selected_model else None,
        )
        self._worker.stage_update.connect(self._on_stage_update)
        self._worker.finished.connect(self._on_pipeline_finished)
        self._worker.error.connect(self._on_pipeline_error)
        self._worker.cancelled.connect(self._on_pipeline_cancelled)
        self._worker.start()

    def _on_cancel(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.cancel_btn.setText("Cancelling...")
            self.log_text.append("\nCancellation requested... stopping after current stage.")

    def _on_pipeline_cancelled(self):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Analysis")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel")
        self.stepper.set_all("failed")
        self.log_text.append("\nPipeline cancelled by user.")
        self.web_view.setHtml(
            self._placeholder_html("Analysis cancelled")
        )
        self.report_text.setPlainText("Analysis cancelled by user.")

    def _on_clear(self):
        if self._region_worker is not None and self._region_worker.isRunning():
            self._region_worker.cancel()
            self._region_worker.wait(200)
            self._region_worker = None

        self.stepper.reset()
        self.log_text.clear()
        self.web_view.setHtml(
            self._placeholder_html(
                "Run analysis to see interactive 3D visualization"
            )
        )
        self.reset_cam_btn.setEnabled(False)
        self.select_region_btn.setEnabled(False)
        self.clear_region_btn.setEnabled(False)
        self._current_findings = []
        self._centerline_data = None
        self._current_mask = None
        self._selection_mode = "navigate"
        self.report_text.setPlainText(
            "Run analysis to generate clinical report..."
        )
        self.json_text.clear()
        self.gt_text.setPlainText(
            "No ground truth report found for this case.\n\n"
            "GT reports are matched via data/ZIV/ziv_manifest.json."
        )
        self.slice_viewer.clear_data()
        self.viz_status_label.setText(
            "Click stenosis marker to jump | Ctrl+Click vessel to navigate"
        )

    def _on_stage_update(self, stage_idx: int, state: str, status_lines: list):
        self.stepper.update_stage(stage_idx, state)
        new_lines = status_lines[self._last_log_line_count:]
        if new_lines:
            for line in new_lines:
                self.log_text.append(line)
            self._last_log_line_count = len(status_lines)

    def _on_pipeline_finished(self, result):
        (status_text, findings_json, clinical_report,
         viz_path, segmented_zarr_path, findings_data) = result

        self.stepper.set_all("done")
        self.log_text.setPlainText(status_text)

        # Must be set before web_view.load() -- loadFinished reads _current_findings
        self._current_findings = (
            findings_data.get("findings", []) if findings_data else []
        )

        self._centerline_data = None
        self._current_mask = None
        if viz_path:
            viz_dir = Path(viz_path).parent
            cl_candidates = list(viz_dir.glob("*_centerline.json"))
            if cl_candidates:
                try:
                    with open(cl_candidates[0], 'r') as f:
                        self._centerline_data = _json.load(f)
                except Exception:
                    pass

        if viz_path and Path(viz_path).exists():
            self.web_view.load(
                QUrl.fromLocalFile(str(Path(viz_path).resolve()))
            )
            self.reset_cam_btn.setEnabled(True)
            self.select_region_btn.setEnabled(self._centerline_data is not None)
            self.clear_region_btn.setEnabled(True)
        else:
            self.web_view.setHtml(
                self._placeholder_html("Visualization not available")
            )
            self.reset_cam_btn.setEnabled(False)
            self.select_region_btn.setEnabled(False)
            self.clear_region_btn.setEnabled(False)

        if segmented_zarr_path and Path(segmented_zarr_path).exists():
            loaded_findings = (
                findings_data.get("findings", [])
                if findings_data else []
            )
            self._zarr_loader = _ZarrLoaderWorker(
                segmented_zarr_path, loaded_findings
            )
            self._zarr_loader.loaded.connect(self._on_zarr_loaded)
            self._zarr_loader.start()

        self.report_text.setPlainText(
            clinical_report or "Analysis did not produce a report."
        )

        self.json_text.setPlainText(findings_json or "{}")

        input_path = self.path_input.text().strip()
        gt_report = discover_gt_report(input_path)
        if gt_report and findings_data:
            try:
                comparison = compare_findings(findings_data, gt_report)
                self.gt_text.setPlainText(format_comparison_text(comparison))
            except Exception as e:
                self.gt_text.setPlainText(f"Error running GT comparison: {e}")
        else:
            self.gt_text.setPlainText(
                "No ground truth report found for this case.\n\n"
                "GT reports are matched via data/ZIV/ziv_manifest.json."
            )

        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Analysis")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel")

        self.tabs.setCurrentIndex(0)

    def _on_zarr_loaded(self, image, mask, findings):
        self.slice_viewer.set_data(image, mask, findings)
        self._current_mask = mask

    def _on_pipeline_error(self, error_msg: str):
        self.run_btn.setEnabled(True)
        self.run_btn.setText("Run Analysis")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Cancel")

        self.log_text.append("\nERROR: " + error_msg)
        self.web_view.setHtml(
            self._placeholder_html("Error occurred during analysis")
        )
        self.reset_cam_btn.setEnabled(False)
        self.report_text.setPlainText("Error: " + error_msg)

        QMessageBox.critical(self, "Pipeline Error", error_msg)

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(5000)
        if self._zarr_loader and self._zarr_loader.isRunning():
            self._zarr_loader.quit()
            self._zarr_loader.wait(3000)
        if self._region_worker and self._region_worker.isRunning():
            self._region_worker.cancel()
            self._region_worker.wait(1000)
        event.accept()


def main():
    print("Stenosis Detection Desktop App (PyQt)")
    print("Temp directory:", TEMP_DIR)

    available_models = discover_models()
    print(f"Found {len(available_models)} model(s)")
    if not available_models:
        print("WARNING: no valid models found in models/")

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    from widgets.splash_screen import SplashScreen
    splash = SplashScreen()
    splash.start()
    app.processEvents()

    window = MainWindow()
    app._main_window = window
    splash.finish(window)

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
