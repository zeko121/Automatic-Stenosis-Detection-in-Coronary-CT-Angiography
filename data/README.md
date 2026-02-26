# Data Directory

This directory holds input data for the stenosis detection pipeline.

## Structure

```
data/
├── imageCAS/     # ImageCAS dataset (1,000 CCTA volumes with vessel annotations)
└── ziv/          # Ziv Medical Center dataset (25 patients, Philips scanner)
```

## Supported Input Formats

- **DICOM**: Place a folder of `.dcm` files. The pipeline will convert to NIfTI automatically.
- **NIfTI**: Place a single `.nii.gz` file.

## Included Examples

### ImageCAS (case 1)

Located at `imageCAS/1-200/`. Contains one sample scan tracked via Git LFS:

| File | Format | Size |
|------|--------|------|
| `1.img.nii.gz` | NIfTI (compressed) | ~92 MB |
| `1.label.nii.gz` | NIfTI (compressed) | ~2.5 MB |

### Ziv (preprocessed)

Located at `ziv/exampleCase.zarr`. A single preprocessed CCTA volume ready for pipeline inference:

| Property | Value |
|----------|-------|
| Shape | 336 x 336 x 296 |
| Dtype | float32 |
| Spacing | 0.5 mm isotropic |
| Normalization | Z-score clipped to [-3, 3] |
| Size on disk | ~64 MB |

## Notes

- Full datasets are not included in this repository due to size and privacy constraints.
- ImageCAS is publicly available at [ImageCAS dataset](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation).
- Ziv Medical Center data is private clinical data and cannot be redistributed.
