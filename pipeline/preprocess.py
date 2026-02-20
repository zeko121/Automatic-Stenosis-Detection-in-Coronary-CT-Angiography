"""
NIfTI to Zarr preprocessing. Params read from model's run_config.json.
"""

import json
import time
import shutil
import gc
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import nibabel as nib
from nibabel.orientations import io_orientation, axcodes2ornt, ornt_transform, aff2axcodes
from scipy.ndimage import gaussian_filter
from skimage.transform import resize
import zarr

# zarr v2 vs v3 compat
ZARR_V3 = zarr.__version__.startswith("3")
if ZARR_V3:
    from zarr.codecs import Blosc
else:
    from numcodecs import Blosc


@dataclass
class PreprocessConfig:
    hu_min: float = -50.0
    hu_max: float = 700.0
    target_spacing: tuple = (0.5, 0.5, 0.5)
    target_orientation: str = "RAS"
    zscore_clip: tuple = (-3.0, 3.0)
    zarr_chunk: tuple = (160, 160, 160)
    compressor: str = "zstd"
    compression_level: int = 3
    auto_crop_wide_fov: bool = True
    crop_target_fov_mm: float = 220.0
    crop_fov_threshold_mm: float = 280.0

    @classmethod
    def from_model_config(cls, config_path):
        with open(config_path, 'r') as f:
            config_data = json.load(f)

        preproc = config_data.get("preprocessing", {}).get("config", {}).get("parameters", {})

        # fallback to training_config
        if not preproc:
            preproc = config_data.get("training_config", {})

        return cls(
            hu_min=float(preproc.get("hu_min", -50.0)),
            hu_max=float(preproc.get("hu_max", 700.0)),
            target_spacing=tuple(preproc.get("target_spacing", [0.5, 0.5, 0.5])),
            target_orientation=preproc.get("target_orientation", "RAS"),
            zscore_clip=tuple(preproc.get("zscore_clip", [-3.0, 3.0])),
            zarr_chunk=tuple(preproc.get("zarr_chunk", [160, 160, 160])),
            compressor=preproc.get("compressor", "zstd"),
            compression_level=int(preproc.get("compression_level", 3)),
        )


def load_nifti(path):
    nii = nib.load(path)
    data = np.asarray(nii.dataobj, dtype=np.float32)
    spacing = tuple(float(s) for s in nii.header.get_zooms()[:3])
    return data, nii.affine, spacing


def get_orientation(affine):
    return ''.join(aff2axcodes(affine))

def reorient_to_target(data, affine, target="RAS"):
    current_orient = io_orientation(affine)
    current_code = get_orientation(affine)
    target_orient = axcodes2ornt(tuple(target))

    if np.array_equal(current_orient, target_orient):
        return data, current_code

    transform = ornt_transform(current_orient, target_orient)
    reoriented = nib.orientations.apply_orientation(data, transform)
    return reoriented, current_code


def resample_volume(data, orig_spacing, target_spacing, order=1):
    scale_factors = np.array(orig_spacing) / np.array(target_spacing)
    new_shape = tuple(np.round(np.array(data.shape) * scale_factors).astype(int))
    resampled = resize(
        data.astype(np.float32),
        new_shape,
        order=order,
        mode='edge',
        anti_aliasing=False,
        preserve_range=True
    )
    return resampled.astype(np.float32)


def apply_hu_window(data, hu_min, hu_max):
    np.clip(data, hu_min, hu_max, out=data)
    return data

def zscore_normalize(data, clip_range):
    mean = float(np.mean(data))
    std = max(float(np.std(data)), 1e-6)  # prevent div by zero

    normalized = (data - mean) / std
    np.clip(normalized, clip_range[0], clip_range[1], out=normalized)

    return normalized.astype(np.float32), mean, std


def write_zarr(output_path, image, attrs, config, temp_dir=None):
    output_path = Path(output_path)

    if temp_dir:
        write_path = Path(temp_dir) / output_path.name
        write_path.parent.mkdir(parents=True, exist_ok=True)
        if write_path.exists():
            shutil.rmtree(write_path)
    else:
        write_path = output_path
        if write_path.exists():
            shutil.rmtree(write_path)

    shuffle = 2 if config.compressor in ["zstd", "lz4", "blosclz"] else 0

    D, H, W = image.shape
    chunks = (
        min(config.zarr_chunk[0], D),
        min(config.zarr_chunk[1], H),
        min(config.zarr_chunk[2], W)
    )

    if ZARR_V3:
        compressor = [Blosc(cname=config.compressor, clevel=config.compression_level, shuffle=shuffle)]
        store = zarr.open_group(str(write_path), mode="w", zarr_version=3)
        store.create_array(
            "image",
            shape=image.shape,
            dtype="float32",
            chunks=chunks,
            compressors=compressor
        )[:] = image
    else:
        compressor = Blosc(cname=config.compressor, clevel=config.compression_level, shuffle=shuffle)
        store = zarr.open_group(str(write_path), mode="w")
        store.create_dataset(
            "image",
            data=image,
            dtype="float32",
            chunks=chunks,
            compressor=compressor
        )

    store.attrs.update(attrs)

    if temp_dir and write_path != output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(write_path, output_path)
        shutil.rmtree(write_path)

    size_mb = sum(f.stat().st_size for f in output_path.rglob('*') if f.is_file()) / (1024 * 1024)
    return size_mb


def crop_cardiac_roi(image, spacing, target_fov_mm=220.0, fov_threshold_mm=280.0, verbose=True):
    """Crop wide FOV to cardiac region using intensity peak detection."""
    fov_r = image.shape[0] * spacing[0]
    fov_a = image.shape[1] * spacing[1]

    if fov_r <= fov_threshold_mm and fov_a <= fov_threshold_mm:
        if verbose:
            print(f"   FOV ({fov_r:.0f} x {fov_a:.0f} mm) is within threshold, no crop needed")
        return image, False, None

    if verbose:
        print(f"   wide FOV detected: {fov_r:.0f} x {fov_a:.0f} mm (threshold={fov_threshold_mm}mm)")

    # middle 20% of z-axis where the heart usually sits
    z_size = image.shape[2]
    z_start = int(z_size * 0.4)
    z_end = int(z_size * 0.6)
    z_end = max(z_end, z_start + 1)

    slab = image[:, :, z_start:z_end].copy()
    np.clip(slab, 100, 700, out=slab)
    slab = slab - 100.0

    heat_map = slab.mean(axis=2)

    sigma_r = max(15.0 / spacing[0], 3.0)
    sigma_a = max(15.0 / spacing[1], 3.0)
    smoothed = gaussian_filter(heat_map, sigma=(sigma_r, sigma_a))

    peak_idx = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    heart_r, heart_a = int(peak_idx[0]), int(peak_idx[1])
    # print(f"   DEBUG smoothed max: {smoothed.max():.1f}")

    if verbose:
        print(f"   heart center at voxel ({heart_r}, {heart_a})")

    crop_r = int(round(target_fov_mm / spacing[0]))
    crop_a = int(round(target_fov_mm / spacing[1]))

    if fov_r > fov_threshold_mm:
        half_r = crop_r // 2
        r_start = heart_r - half_r
        r_end = r_start + crop_r
        if r_start < 0:
            r_start = 0
            r_end = min(crop_r, image.shape[0])
        if r_end > image.shape[0]:
            r_end = image.shape[0]
            r_start = max(0, r_end - crop_r)
    else:
        r_start, r_end = 0, image.shape[0]

    if fov_a > fov_threshold_mm:
        half_a = crop_a // 2
        a_start = heart_a - half_a
        a_end = a_start + crop_a
        if a_start < 0:
            a_start = 0
            a_end = min(crop_a, image.shape[1])
        if a_end > image.shape[1]:
            a_end = image.shape[1]
            a_start = max(0, a_end - crop_a)
    else:
        a_start, a_end = 0, image.shape[1]

    cropped = image[r_start:r_end, a_start:a_end, :]

    if verbose:
        new_fov_r = cropped.shape[0] * spacing[0]
        new_fov_a = cropped.shape[1] * spacing[1]
        print(f"   cropped: {image.shape} -> {cropped.shape}")
        print(f"   new FOV: {new_fov_r:.0f} x {new_fov_a:.0f} mm")

    return cropped, True, (heart_r, heart_a)


def process(input_path, output_path, model_config_path=None, config=None, verbose=True):
    t0 = time.time()
    input_path = Path(input_path)
    output_path = Path(output_path)

    result = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "pending",
    }

    if config is None:
        if model_config_path:
            config = PreprocessConfig.from_model_config(model_config_path)
            if verbose:
                print(f"loaded config from: {model_config_path}")
        else:
            config = PreprocessConfig()
            if verbose:
                print("using default config")

    result["config"] = {
        "hu_window": [config.hu_min, config.hu_max],
        "target_spacing": list(config.target_spacing),
        "target_orientation": config.target_orientation,
        "zscore_clip": list(config.zscore_clip),
    }

    if verbose:
        print(f"preprocessing config:")
        print(f"  HU window: [{config.hu_min}, {config.hu_max}]")
        print(f"  target spacing: {config.target_spacing} mm")
        print(f"  orientation: {config.target_orientation}")
        print(f"  z-score clip: {config.zscore_clip}")

    if not input_path.exists():
        result["status"] = "error"
        result["error"] = f"Input file does not exist: {input_path}"
        return result

    try:
        if verbose:
            print(f"\nloading NIfTI: {input_path.name}")

        image, affine, spacing = load_nifti(str(input_path))
        orig_shape = image.shape
        orig_orientation = get_orientation(affine)
        orig_hu_range = (float(image.min()), float(image.max()))

        if verbose:
            print(f"   shape: {orig_shape}")
            print(f"   spacing: {[f'{s:.3f}' for s in spacing]} mm")
            print(f"   orientation: {orig_orientation}")
            print(f"   HU range: [{orig_hu_range[0]:.0f}, {orig_hu_range[1]:.0f}]")

        if verbose:
            print(f"\nreorienting {orig_orientation} -> {config.target_orientation}")

        image, _ = reorient_to_target(image, affine, config.target_orientation)
        del affine
        gc.collect()

        if verbose:
            print(f"   shape after reorient: {image.shape}")

        was_cropped = False
        heart_center = None
        if config.auto_crop_wide_fov:
            if verbose:
                print(f"\nchecking FOV for cardiac crop (threshold={config.crop_fov_threshold_mm}mm)")

            image, was_cropped, heart_center = crop_cardiac_roi(
                image, spacing,
                target_fov_mm=config.crop_target_fov_mm,
                fov_threshold_mm=config.crop_fov_threshold_mm,
                verbose=verbose,
            )

            if was_cropped:
                gc.collect()

        if verbose:
            print(f"\napplying HU window [{config.hu_min}, {config.hu_max}]")

        image = apply_hu_window(image, config.hu_min, config.hu_max)

        if verbose:
            print(f"\nresampling to {config.target_spacing} mm")
            print(f"   original shape: {image.shape}")

        image = resample_volume(image, spacing, config.target_spacing, order=1)
        gc.collect()

        if verbose:
            print(f"   resampled shape: {image.shape}")

        if verbose:
            print(f"\nz-score normalizing (clip {config.zscore_clip})")

        image, zscore_mean, zscore_std = zscore_normalize(image, config.zscore_clip)
        # print(f"   DEBUG normalized dtype: {image.dtype}")

        if verbose:
            print(f"   pre-norm mean: {zscore_mean:.2f}, std: {zscore_std:.2f}")
            print(f"   post-norm mean: {image.mean():.4f}, std: {image.std():.4f}")
            print(f"   final range: [{image.min():.2f}, {image.max():.2f}]")

        if verbose:
            print(f"\nsaving to zarr")

        attrs = {
            "source_file": input_path.name,
            "original_shape": list(orig_shape),
            "original_spacing": list(spacing),
            "original_orientation": orig_orientation,
            "original_hu_range": list(orig_hu_range),
            "final_shape": list(image.shape),
            "final_spacing": list(config.target_spacing),
            "final_orientation": config.target_orientation,
            "hu_window": [config.hu_min, config.hu_max],
            "zscore_mean": round(zscore_mean, 4),
            "zscore_std": round(zscore_std, 4),
            "zscore_clip": list(config.zscore_clip),
            "pipeline_version": "1.0",
            "was_cropped": was_cropped,
        }
        if heart_center is not None:
            attrs["heart_center_voxel"] = list(heart_center)

        size_mb = write_zarr(output_path, image, attrs, config)

        if verbose:
            print(f"   saved: {output_path.name} ({size_mb:.1f} MB)")

        del image
        gc.collect()

        elapsed = time.time() - t0

        result.update({
            "status": "success",
            "original_shape": list(orig_shape),
            "original_spacing": list(spacing),
            "original_orientation": orig_orientation,
            "final_shape": attrs["final_shape"],
            "final_spacing": list(config.target_spacing),
            "final_orientation": config.target_orientation,
            "value_range": [float(attrs["zscore_clip"][0]), float(attrs["zscore_clip"][1])],
            "zscore_mean": round(zscore_mean, 4),
            "zscore_std": round(zscore_std, 4),
            "file_size_mb": round(size_mb, 2),
            "runtime_sec": round(elapsed, 2),
            "was_cropped": was_cropped,
        })
        if heart_center is not None:
            result["heart_center_voxel"] = list(heart_center)

        if verbose:
            print(f"\ndone! took {elapsed:.2f}s")

    except Exception as e:
        result["status"] = "error"
        result["error"] = f"{type(e).__name__}: {str(e)}"
        if verbose:
            print(f"\nerror: {result['error']}")
            import traceback
            traceback.print_exc()

    return result


def validate_zarr(zarr_path):
    zarr_path = Path(zarr_path)

    if not zarr_path.exists():
        return False, "Path does not exist", None

    try:
        store = zarr.open_group(str(zarr_path), mode="r")

        if "image" not in store:
            return False, "Missing 'image' array", None

        image = store["image"]
        attrs = dict(store.attrs)

        _ = image[0, 0, 0]

        metadata = {
            "shape": image.shape,
            "dtype": str(image.dtype),
            "attrs": attrs,
        }

        return True, f"Valid: shape={image.shape}", metadata

    except Exception as e:
        return False, f"Error: {e}", None


if __name__ == "__main__":
    import sys

    test_input = "temp/case-30.nii.gz"
    test_output = "temp/case-30.zarr"
    model_config = "models/2025-12-31_02-53-53/run_config.json"

    if len(sys.argv) >= 3:
        test_input = sys.argv[1]
        test_output = sys.argv[2]
    if len(sys.argv) >= 4:
        model_config = sys.argv[3]

    print("NIfTI to Zarr Preprocessing Test")

    result = process(test_input, test_output, model_config_path=model_config, verbose=True)

    print("\nResult:")
    for key, value in result.items():
        if key != "config":
            print(f"  {key}: {value}")

    if result["status"] == "success":
        print("\nValidation:")
        is_valid, msg, metadata = validate_zarr(test_output)
        print(f"  Valid: {is_valid}")
        print(f"  Message: {msg}")
        if metadata:
            print(f"  Shape: {metadata['shape']}")
            print(f"  Dtype: {metadata['dtype']}")
            attrs = metadata['attrs']
            print(f"  Original orientation: {attrs.get('original_orientation')}")
            print(f"  Final orientation: {attrs.get('final_orientation')}")
            print(f"  Final spacing: {attrs.get('final_spacing')}")
            print(f"  Value range (z-score clipped): {attrs.get('zscore_clip')}")
