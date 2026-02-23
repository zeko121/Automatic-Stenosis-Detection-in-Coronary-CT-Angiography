# Stenosis Detection

Automated coronary artery stenosis detection from Cardiac CT Angiography (CCTA) scans. This system combines supervised deep learning for vessel segmentation with unsupervised computational geometry for stenosis measurement, delivered as a desktop application.

Developed at the University of Haifa, Department of Information Systems, in collaboration with Ziv Medical Center.

---

## Pipeline Overview

The system processes a CCTA scan through eight sequential stages:

```
DICOM/NIfTI  →  Preprocess  →  Segment  →  Post-process  →  Centerline  →  Label  →  Stenosis  →  Visualize
```

| Stage | Module | Description |
|---|---|---|
| DICOM to NIfTI | `dicom_to_nifti.py` | Converts a folder of DICOM files into a single NIfTI volume |
| Preprocessing | `preprocess.py` | HU windowing [-50, 700], RAS reorientation, 0.5 mm isotropic resampling, z-score normalization |
| Segmentation | `segment.py` | 3D U-Net inference with sliding window; outputs a binary vessel mask |
| Post-processing | `postprocess.py` | Distance-based fragment removal to clean spurious mask components |
| Centerline extraction | `centerline.py` | Topological skeletonization of the vessel mask; builds a graph with per-node radius estimates |
| Artery labeling | `label_arteries.py` | Component-based identification of left and right coronary artery territories |
| Stenosis detection | `stenosis.py` | Computes local narrowing as a fraction of the proximal reference diameter; assigns severity grade |
| Visualization | `visualize.py` | Interactive 3D Plotly HTML view with stenosis markers; JSON report |

---

## Project Structure

```
stenosis-detection/
├── app_qt.py              # PySide6 desktop application — main entry point
├── pipeline/              # Eight-stage processing pipeline
│   ├── __init__.py
│   ├── dicom_to_nifti.py  # DICOM folder → NIfTI volume
│   ├── preprocess.py      # Normalization, resampling, orientation alignment
│   ├── segment.py         # 3D U-Net inference (sliding window)
│   ├── postprocess.py     # Binary mask cleanup
│   ├── centerline.py      # Vessel skeleton extraction and radius measurement
│   ├── label_arteries.py  # Left/right coronary artery identification
│   ├── stenosis.py        # Stenosis detection and severity grading
│   ├── visualize.py       # 3D Plotly visualization and JSON report
│   ├── runner.py          # Pipeline orchestrator
│   ├── compare_gt.py      # Ground truth comparison utilities
│   ├── endpoint_classifier.py
│   ├── gap_connector.py
│   ├── region_helpers.py
│   ├── refine_segmentation.py
│   └── resegment.py
├── widgets/               # Qt UI components
│   ├── plotly_bridge.py   # Qt–Plotly HTML bridge
│   ├── slice_viewer_widget.py
│   └── splash_screen.py
├── evaluation/            # Four-tier evaluation framework
│   ├── config.py          # Environment detection and path configuration
│   ├── run_evaluation.py  # CLI orchestrator
│   ├── tier1_segmentation.py   # Dice, surface Dice, HD95
│   ├── tier2_structural.py     # Centerline and topology metrics
│   ├── tier3_downstream.py     # Stenosis detection performance
│   ├── tier4_robustness.py     # Cross-scanner and edge-case robustness
│   ├── composite_score.py      # Weighted composite across tiers
│   └── report_generator.py
├── models/                # Trained model checkpoints (Git LFS)
│   └── {run_id}/
│       ├── checkpoints/
│       │   └── best_model.pth
│       └── run_config.json
├── data/                  # Input data directory
│   ├── imageCAS/          # ImageCAS dataset (training/evaluation)
│   └── ZIV/               # Ziv Medical Center CCTA cases
└── tests/                 # Test suite
    ├── test_pipeline/     # Per-stage unit tests
    └── humanization/      # Snapshot regression tests
```

---

## Setup and Installation

**Requirements**: Python 3.10 or later.

```bash
pip install -r requirements.txt
```

### Model Placement

Place model checkpoint folders under `models/`:

```
models/
└── {run_id}/
    ├── checkpoints/
    │   └── best_model.pth
    └── run_config.json
```

Two trained models are included in this submission:

| Folder | Description |
|---|---|
| `2025-12-31_02-53-53` | MONAI 3D U-Net, DiceBCE loss — primary model |
| `2026-02-12_13-14-10` | MONAI 3D U-Net, BCE-only loss, high-recall variant for gap filling |

The `run_config.json` in each folder records the exact preprocessing parameters used during training. The pipeline reads this file at runtime to ensure inference preprocessing matches training.

---

## Usage

Launch the desktop application:

```bash
python app_qt.py
```

The application accepts two input formats:

- **DICOM folder**: a directory containing `.dcm` files (or extensionless DICOM files)
- **NIfTI file**: a `.nii` or `.nii.gz` file

Select the input path, choose a model from the dropdown, and click **Run**. The pipeline runs all eight stages sequentially, with progress reported per stage. On completion, an interactive 3D visualization opens in the built-in viewer and a JSON report is saved to the output directory.

### Evaluation Framework

To run quantitative evaluation against ground truth data:

```bash
python -m evaluation.run_evaluation --tiers 1,2 --max-cases 5
python -m evaluation.run_evaluation --tiers 1,2,3,4
python -m evaluation.run_evaluation --tiers 2 --dataset ziv
```

---

## Models

Both models share the same architecture: MONAI 3D U-Net with residual units, channels [32, 64, 128, 256, 512], batch normalization, 10% dropout, trained on 750 ImageCAS cases with 160x160x160 patches at 0.5 mm isotropic spacing.

**`2025-12-31_02-53-53` — DiceBCE model**
- Loss: combined Dice + BCE
- Inference threshold: 0.50
- Best Dice (with post-processing): 82.40% on 500-case ImageCAS evaluation

**`2026-02-12_13-14-10` — BCE-only model**
- Loss: BCE with positive class weight 5.0 (high-recall bias)
- Inference threshold: 0.80
- Best Dice (with post-processing): 82.95% on 500-case ImageCAS evaluation
- Trained with: AdamW, lr=1e-4, CosineAnnealing scheduler, 150 epochs, mixed precision

---

## Tech Stack

| Library | Version | Role |
|---|---|---|
| PyTorch | 2.4.1 | Model inference, CUDA acceleration |
| MONAI | 1.3.2 | U-Net architecture, sliding window inference |
| Zarr | 3.1.3 | Compressed intermediate volume storage |
| nibabel | 5.x | NIfTI I/O |
| SimpleITK | — | DICOM loading and orientation transforms |
| scikit-image | — | Skeletonization, morphological operations |
| scipy | 1.14.x | Graph analysis, distance transforms |
| PySide6 | — | Desktop application framework |
| Plotly | — | Interactive 3D visualization |
| pydicom | — | DICOM file parsing |

---

## Results

### Segmentation (ImageCAS, 500-case evaluation)

| Model | Loss | Threshold | Post-processing | Dice |
|---|---|---|---|---|
| DiceBCE | Dice + BCE | 0.50 | Yes | **82.40%** |
| BCE-only | BCE (w=5.0) | 0.80 | Yes | **82.95%** |

### Inference Performance

| Metric | Value |
|---|---|
| End-to-end pipeline time | ~84.5 s |
| Hardware | NVIDIA GTX 1650 (4 GB VRAM) |
| Sliding window patch size | 160 x 160 x 160 voxels |

### Domain Transfer

Zero-shot transfer from ImageCAS (Siemens scanner) to Ziv Medical Center (Philips scanner) is achieved by aligning preprocessing parameters — RAS orientation, HU window [-50, 700], 0.5 mm isotropic spacing — between training and inference. No Ziv data was used during training.

---

## Stenosis Classification

Stenosis severity is computed as:

```
stenosis_percent = 1 - (r_min / r_reference)
```

where `r_min` is the minimum radius within a candidate segment and `r_reference` is the proximal reference radius. Severity is classified as:

| Grade | Criterion |
|---|---|
| Normal | stenosis_percent < 25% |
| Mild | 25% <= stenosis_percent < 50% |
| Moderate | 50% <= stenosis_percent < 70% |
| Severe | stenosis_percent >= 70% |

---

## Preprocessing Parameters

All preprocessing is configured from the model's `run_config.json`. Default training parameters:

| Parameter | Value |
|---|---|
| HU window | [-50, 700] |
| Target spacing | 0.5 mm isotropic |
| Orientation | RAS |
| Z-score clip | [-3.0, 3.0] |
| Mask threshold | 0.3 (linear interpolation) |

---

## Authors and Acknowledgments

**University of Haifa** — Department of Information Systems

**Supervisor**: Prof. Mario Boley

**Clinical collaboration**: Ziv Medical Center, Safed — radiologists who provided CCTA data and ground truth annotations

**Training dataset**: ImageCAS (1,000 annotated CCTA volumes) — used under academic license for training and evaluation
