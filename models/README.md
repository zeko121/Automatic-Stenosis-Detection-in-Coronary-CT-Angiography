# Models

This directory contains trained segmentation model checkpoints for the coronary artery stenosis detection pipeline.

## Directory Structure

```
models/
├── 2025-12-31_02-53-53/
│   ├── run_config.json
│   └── checkpoints/best_model.pth
└── 2026-02-12_13-14-10/
    ├── run_config.json
    └── checkpoints/best_model.pth
```

---

## Model 1: 2025-12-31_02-53-53 — Primary DiceBCE Model

**Run name**: `unet_model`
**Description**: MONAI 3D U-Net trained with a combined Dice + BCE loss function.

### Architecture

| Parameter      | Value                        |
|----------------|------------------------------|
| Model type     | MONAI 3D U-Net               |
| In channels    | 1 (grayscale CT)             |
| Out channels   | 1 (binary vessel mask)       |
| Feature channels | [32, 64, 128, 256, 512]    |
| Strides        | [2, 2, 2, 2]                 |
| Residual units | 2 per block                  |
| Normalization  | Batch normalization           |
| Dropout        | 0.1                          |
| Patch size     | 160 x 160 x 160              |

### Loss Function

Combined Dice + Binary Cross-Entropy (DiceBCE).

### Performance

- **82.40% Dice** on 500-case ImageCAS held-out evaluation
- Inference threshold: 0.50
- Post-processing: distance-based fragment removal

### Preprocessing Parameters

Preprocessing parameters for this model are not fully enumerated in `run_config.json`. Use the defaults documented in `Preprocessing_v2.ipynb`:

| Parameter         | Value              |
|-------------------|--------------------|
| HU window         | [-50, 700]         |
| Target spacing    | 0.5 mm isotropic   |
| Orientation       | RAS                |
| Z-score clip      | [-3.0, 3.0]        |
| Mask threshold    | 0.3                |
| Resampling        | Linear interpolation |

---

## Model 2: 2026-02-12_13-14-10 — BCE-Only High-Recall Model

**Run name**: `BCE_ONLY`
**Description**: MONAI 3D U-Net trained with BCE-only loss, designed to maximize recall at the cost of some precision. Intended to be paired with post-processing to remove false positives.

### Architecture

| Parameter      | Value                        |
|----------------|------------------------------|
| Model type     | MONAI 3D U-Net               |
| In channels    | 1 (grayscale CT)             |
| Out channels   | 1 (binary vessel mask)       |
| Feature channels | [32, 64, 128, 256, 512]    |
| Strides        | [2, 2, 2, 2]                 |
| Residual units | 2 per block                  |
| Normalization  | Batch normalization           |
| Dropout        | 0.1                          |
| Patch size     | 160 x 160 x 160              |

### Loss Function

Binary Cross-Entropy only (`loss_type: BCE`).

| Parameter        | Value  |
|------------------|--------|
| BCE weight       | 1.0    |
| Dice weight      | 0.0    |
| BCE pos weight   | 5.0 (upweights vessel foreground) |

### Training Details

| Parameter              | Value                              |
|------------------------|------------------------------------|
| Optimizer              | AdamW                              |
| Learning rate          | 1e-4                               |
| Weight decay           | 1e-5                               |
| LR scheduler           | Cosine annealing                   |
| Max epochs             | 150                                |
| Batch size             | 4                                  |
| Samples per volume     | 4                                  |
| Foreground oversample  | 0.70                               |
| Augmentation           | Disabled                           |
| Mixed precision (AMP)  | Enabled                            |
| Early stopping patience| 30 epochs                          |
| Data split (train/val/test) | 75% / 15% / 10% (seed=42)    |
| Total training cases   | 750 of 1,000 ImageCAS cases        |
| Training hardware      | NVIDIA A100-SXM4-40GB (40 GB VRAM) |

### Preprocessing Parameters

Taken from the linked preprocessing run `run_2026-01-29_19-50_HU-50-700_0.5mm_RAS`:

| Parameter         | Value              |
|-------------------|--------------------|
| HU window         | [-50, 700]         |
| Target spacing    | 0.5 mm isotropic   |
| Orientation       | RAS                |
| Z-score clip      | [-3.0, 3.0]        |
| Mask threshold    | 0.3                |
| Resampling        | Linear interpolation |

### Performance

- **82.95% Dice** on 500-case ImageCAS held-out evaluation
- Inference threshold: 0.80 (raised to compensate for high recall bias)
- Post-processing: distance-based fragment removal

---

## Usage

The pipeline (`pipeline/runner.py`) reads `run_config.json` automatically to determine the correct preprocessing parameters for whichever model is selected. No manual parameter configuration is required.

To use a model, place its checkpoint folder under this `models/` directory and select it from the app's model picker. The expected structure per model run is:

```
models/{run_id}/
├── run_config.json       # Preprocessing + architecture parameters
└── checkpoints/
    └── best_model.pth    # Best validation checkpoint
```

### Inference Notes

- Inference uses sliding window with 160x160x160 patches.
- On a GTX 1650 (4 GB VRAM), the pipeline falls back to CPU if GPU memory is insufficient.
- On a GTX 1650, end-to-end pipeline runtime is approximately 84.5 seconds.
