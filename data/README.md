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

## Notes

- Data files are not included in this repository due to size and privacy constraints.
- ImageCAS is publicly available at [ImageCAS dataset](https://github.com/XiaoweiXu/ImageCAS-A-Large-Scale-Dataset-and-Benchmark-for-Coronary-Artery-Segmentation).
- Ziv Medical Center data is private clinical data and cannot be redistributed.
