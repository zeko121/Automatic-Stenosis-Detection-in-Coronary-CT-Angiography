"""Environment configuration for evaluation (auto-detects Colab vs local)."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


def _is_colab() -> bool:
    try:
        import google.colab  # noqa: F401
        return True
    except ImportError:
        return False


def _detect_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


@dataclass
class EvalConfig:
    device: str = ""
    is_colab: bool = False

    project_root: Path = Path("")
    models_root: Path = Path("")
    imagecas_root: Path = Path("")
    ziv_root: Path = Path("")
    output_dir: Path = Path("")
    temp_dir: Path = Path("")

    imagecas_subfolders: List[str] = field(
        default_factory=lambda: ["1-200", "201-400", "401-600", "601-800", "801-1000"]
    )

    surface_dice_tolerances_mm: List[float] = field(default_factory=lambda: [0.5, 1.0])
    voxel_spacing_mm: float = 0.5
    bootstrap_n_iterations: int = 1000
    bootstrap_ci: float = 0.95

    tier_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "tier1": 0.25,
            "tier2": 0.15,
            "tier3": 0.45,
            "tier4": 0.15,
        }
    )

    max_cases: Optional[int] = None
    seed: int = 42

    def __post_init__(self):
        self.is_colab = _is_colab()
        if not self.device:
            self.device = _detect_device()

        if self.is_colab:
            self.project_root = Path("/content/drive/MyDrive")
            self.models_root = self.project_root / "models"
            self.imagecas_root = self.project_root / "imageCAS"
            self.ziv_root = self.project_root / "ZIVDICOM_NIFTI"
        else:
            self.project_root = Path(__file__).resolve().parent.parent  # repo root
            self.models_root = self.project_root / "models"
            self.imagecas_root = self.project_root / "data" / "imageCAS"
            self.ziv_root = self.project_root / "data" / "ziv"

        self.output_dir = self.project_root / "evaluation" / "results"
        self.temp_dir = self.project_root / "temp" / "eval"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


def discover_models(models_root: Path) -> Dict[str, Path]:
    """Return {model_name: model_dir} for dirs with checkpoint + config."""
    models: Dict[str, Path] = {}
    if not models_root.exists():
        return models
    for d in sorted(models_root.iterdir()):
        if not d.is_dir():
            continue
        ckpt = d / "checkpoints" / "best_model.pth"
        cfg = d / "run_config.json"
        if ckpt.exists() and cfg.exists():
            models[d.name] = d
    return models


def discover_imagecas_cases(
    root: Path, subfolders: Optional[List[str]] = None
) -> Dict[int, Path]:
    """Return {case_id: subfolder_path} for ImageCAS."""
    if subfolders is None:
        subfolders = ["1-200", "201-400", "401-600", "601-800", "801-1000"]
    cases: Dict[int, Path] = {}
    if not root.exists():
        return cases
    for sf_name in subfolders:
        sf = root / sf_name
        if not sf.is_dir():
            continue
        for f in sf.iterdir():
            if f.name.endswith(".img.nii.gz"):
                match = re.match(r"^(\d+)\.img\.nii\.gz$", f.name)
                if match:
                    cases[int(match.group(1))] = sf
    return cases


def imagecas_image_path(cases: Dict[int, Path], case_id: int) -> Path:
    return cases[case_id] / f"{case_id}.img.nii.gz"


def imagecas_label_path(cases: Dict[int, Path], case_id: int) -> Path:
    return cases[case_id] / f"{case_id}.label.nii.gz"


def discover_ziv_cases(root: Path) -> Dict[str, Path]:
    """Find Ziv case-XX directories."""
    cases: Dict[str, Path] = {}
    if not root.exists():
        return cases
    for d in sorted(root.iterdir()):
        if d.is_dir() and d.name.startswith("case-"):
            cases[d.name] = d
    return cases
