"""QWebChannel bridge: Plotly 3D JS click events -> Python Qt signals."""

import json
from typing import List, Optional, Tuple

from PySide6.QtCore import QObject, Signal, Slot


class PlotlyBridge(QObject):
    """Bridge between Plotly JS click events and Python Qt signals."""

    stenosis_clicked = Signal(int, list)    # (finding_index, voxel)
    vessel_clicked = Signal(list)            # (voxel,)
    region_selected = Signal(list, list)     # (start_voxel, end_voxel)
    status_message = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5)
        self._findings_mm: List[dict] = []

    def set_spacing(self, spacing: Tuple[float, float, float]):
        self._spacing = spacing

    def set_findings_mm(self, findings_mm: List[dict]):
        self._findings_mm = findings_mm

    def _mm_to_voxel(self, plotly_x: float, plotly_y: float, plotly_z: float) -> List[int]:
        """Reverse the mm mapping from visualize.py: x->W, y->H, z->D."""
        d = int(round(plotly_z / self._spacing[0]))
        h = int(round(plotly_y / self._spacing[1]))
        w = int(round(plotly_x / self._spacing[2]))
        return [d, h, w]

    @Slot(str)
    def onStenosisClicked(self, json_str: str):
        try:
            data = json.loads(json_str)
            mm = data.get("mm", [0, 0, 0])
            finding_index = data.get("finding_index", -1)
            voxel = self._mm_to_voxel(mm[0], mm[1], mm[2])
            self.stenosis_clicked.emit(finding_index, voxel)
        except Exception as e:
            self.status_message.emit(f"Stenosis click error: {e}")

    @Slot(str)
    def onVesselClicked(self, json_str: str):
        try:
            data = json.loads(json_str)
            mm = data.get("mm", [0, 0, 0])
            voxel = self._mm_to_voxel(mm[0], mm[1], mm[2])
            self.vessel_clicked.emit(voxel)
        except Exception as e:
            self.status_message.emit(f"Vessel click error: {e}")

    @Slot(str)
    def onRegionSelected(self, json_str: str):
        try:
            data = json.loads(json_str)
            start_mm = data.get("start_mm", [0, 0, 0])
            end_mm = data.get("end_mm", [0, 0, 0])
            start_voxel = self._mm_to_voxel(start_mm[0], start_mm[1], start_mm[2])
            end_voxel = self._mm_to_voxel(end_mm[0], end_mm[1], end_mm[2])
            self.region_selected.emit(start_voxel, end_voxel)
        except Exception as e:
            self.status_message.emit(f"Region select error: {e}")

    @Slot(str)
    def onStatusUpdate(self, message: str):
        self.status_message.emit(message)
