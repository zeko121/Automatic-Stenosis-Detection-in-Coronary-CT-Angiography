"""
DICOM -> NIfTI conversion. Picks the best CCTA series from a folder
of mixed DICOM files and writes a single .nii.gz out.
"""

import time
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pydicom
import SimpleITK as sitk
import nibabel as nib
import numpy as np


@dataclass
class SeriesInfo:
    series_number: int
    description: str
    modality: str
    num_slices: int
    rows: int
    cols: int
    slice_thickness: float | None
    pixel_spacing: tuple[float, float] | None
    file_paths: list[Path] = field(default_factory=list)
    file_z_positions: list[float | None] = field(default_factory=list)  # parallel to file_paths
    phase: str | None = None
    is_ccta_candidate: bool = False
    selection_score: float = 0.0
    fov_mm: float | None = None
    reconstruction_diameter: float | None = None


@dataclass
class SeriesSelectionConfig:
    preferred_phases: list[str] = field(default_factory=lambda: ["75%", "70%", "80%"])
    exclude_keywords: list[str] = field(default_factory=lambda: [
        "LUNG", "Calcium Score", "locator", "tracker", "Tracker Graph",
        "Exam Summary", "SBI"
    ])
    min_slice_count: int = 200
    max_slice_thickness: float = 1.5
    optimal_thickness_range: tuple[float, float] = (0.5, 1.0)
    parallel_read: bool = True
    max_workers: int = 8
    cardiac_fov_max: float = 250.0
    borderline_fov_max: float = 300.0
    wide_fov_max: float = 350.0


def _read_dicom_metadata(filepath):
    try:
        ds = pydicom.dcmread(str(filepath), stop_before_pixels=True, force=True)
        z_position = None
        ipp = getattr(ds, 'ImagePositionPatient', None)
        if ipp and len(ipp) >= 3:
            try:
                z_position = float(ipp[2])
            except (ValueError, TypeError):
                pass
        # ReconstructionDiameter (0018,1100) tells us the FOV
        reconstruction_diameter = None
        rd = getattr(ds, 'ReconstructionDiameter', None)
        if rd is not None:
            try:
                reconstruction_diameter = float(rd)
            except (ValueError, TypeError):
                pass

        return {
            'series_number': getattr(ds, 'SeriesNumber', None),
            'series_description': str(getattr(ds, 'SeriesDescription', '')),
            'modality': getattr(ds, 'Modality', 'CT'),
            'rows': getattr(ds, 'Rows', None),
            'cols': getattr(ds, 'Columns', None),
            'slice_thickness': getattr(ds, 'SliceThickness', None),
            'pixel_spacing': getattr(ds, 'PixelSpacing', None),
            'instance_number': getattr(ds, 'InstanceNumber', 0),
            'filepath': filepath,
            'z_position': z_position,
            'reconstruction_diameter': reconstruction_diameter,
        }
    except Exception:
        return None


def _score_series(series_info, config):
    """Score a series for CCTA suitability (mutates in place)."""
    description = series_info.description.lower()

    for keyword in config.exclude_keywords:
        if keyword.lower() in description:
            series_info.selection_score = -1000
            series_info.is_ccta_candidate = False
            return

    if series_info.num_slices < config.min_slice_count:
        series_info.selection_score = -500
        series_info.is_ccta_candidate = False
        return

    score = 0
    series_info.is_ccta_candidate = True

    # more slices = better, cap at 400
    score += min(series_info.num_slices, 400)

    # cardiac phase bonus
    desc_upper = series_info.description.upper()
    for i, phase in enumerate(config.preferred_phases):
        if phase in desc_upper:
            series_info.phase = phase
            score += 100 - (i * 50)  # 75% gets +100, 70%/80% get +50
            break

    if series_info.slice_thickness:
        t = series_info.slice_thickness
        if config.optimal_thickness_range[0] <= t <= config.optimal_thickness_range[1]:
            score += 50
        elif t <= config.max_slice_thickness:
            score += 20
        else:
            score -= 100

    # prefer single-phase "75% Body" over multi-phase "Cardiac".
    # multi-phase cardiac series contain ALL phases interleaved and
    # produce corrupted volumes -- learned this the hard way
    if '75%' in desc_upper and 'body' in description:
        score += 200
    elif '75%' in desc_upper:
        pass  # already scored via phase scoring
    elif 'cardiac' in description and '75%' not in desc_upper:
        score -= 200  # penalize multi-phase without explicit phase tag
    if 'coronary' in description:
        score += 30

    # fov scoring -- penalize wide FOV (full-chest scans)
    if series_info.fov_mm is not None:
        fov = series_info.fov_mm
        if fov <= config.cardiac_fov_max:
            score += 80
        elif fov <= config.borderline_fov_max:
            pass
        elif fov <= config.wide_fov_max:
            score -= 150
        else:
            score -= 300

    # philips SEGMENT reconstructions tend to have wide FOV
    if 'segment' in description:
        score -= 30

    series_info.selection_score = score


def explore_dicom_folder(dicom_folder, config=None):
    """Walk a DICOM folder and return {series_number: SeriesInfo} with scores."""
    if config is None:
        config = SeriesSelectionConfig()

    dicom_folder = Path(dicom_folder)

    dicom_files = []
    for f in dicom_folder.rglob("*"):
        if f.is_file() and not f.name.startswith('.'):
            if f.suffix.lower() == '.dcm' or f.suffix == '':
                dicom_files.append(f)

    if not dicom_files:
        return {}

    series_files: dict[int, list[Path]] = defaultdict(list)
    series_z_positions: dict[int, list[float | None]] = defaultdict(list)
    series_metadata: dict[int, dict] = {}

    if config.parallel_read and len(dicom_files) > 50:
        with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
            futures = {executor.submit(_read_dicom_metadata, f): f for f in dicom_files}
            for future in as_completed(futures):
                result = future.result()
                if result and result['series_number'] is not None:
                    snum = result['series_number']
                    series_files[snum].append(result['filepath'])
                    series_z_positions[snum].append(result.get('z_position'))
                    if snum not in series_metadata:
                        series_metadata[snum] = result
    else:
        for fpath in dicom_files:
            result = _read_dicom_metadata(fpath)
            if result and result['series_number'] is not None:
                snum = result['series_number']
                series_files[snum].append(result['filepath'])
                series_z_positions[snum].append(result.get('z_position'))
                if snum not in series_metadata:
                    series_metadata[snum] = result

    series_dict: dict[int, SeriesInfo] = {}

    for series_num, files in series_files.items():
        meta = series_metadata.get(series_num, {})

        pixel_spacing = None
        if meta.get('pixel_spacing'):
            try:
                ps = meta['pixel_spacing']
                if hasattr(ps, '__iter__'):
                    pixel_spacing = (float(ps[0]), float(ps[1]))
            except (ValueError, TypeError):
                pass

        slice_thickness = None
        if meta.get('slice_thickness'):
            try:
                slice_thickness = float(meta['slice_thickness'])
            except (ValueError, TypeError):
                pass

        # fov from ReconstructionDiameter or pixel_spacing * cols
        recon_diam = meta.get('reconstruction_diameter')
        fov_mm = None
        if recon_diam is not None and recon_diam > 0:
            fov_mm = recon_diam
        elif pixel_spacing is not None and (meta.get('cols', 0) or 0) > 0:
            fov_mm = pixel_spacing[0] * (meta.get('cols', 0) or 0)

        series_info = SeriesInfo(
            series_number=series_num,
            description=meta.get('series_description', ''),
            modality=meta.get('modality', 'CT'),
            num_slices=len(files),
            rows=meta.get('rows', 0) or 0,
            cols=meta.get('cols', 0) or 0,
            slice_thickness=slice_thickness,
            pixel_spacing=pixel_spacing,
            file_paths=files,
            file_z_positions=series_z_positions.get(series_num, []),
            fov_mm=fov_mm,
            reconstruction_diameter=recon_diam,
        )

        _score_series(series_info, config)
        series_dict[series_num] = series_info

    return series_dict

def select_best_series(series_dict, manual_override=None):
    if manual_override is not None and manual_override in series_dict:
        return manual_override, f"Manual override: Series {manual_override}"

    candidates = [(snum, info) for snum, info in series_dict.items()
                  if info.is_ccta_candidate and info.selection_score > 0]

    if not candidates:
        return None, "No suitable CCTA series found"

    candidates.sort(key=lambda x: x[1].selection_score, reverse=True)
    best_num, best_series = candidates[0]

    thickness = f"{best_series.slice_thickness:.1f}" if best_series.slice_thickness else "?"
    reason = (
        f"Auto-selected: {best_series.description} "
        f"(score={best_series.selection_score:.0f}, "
        f"{best_series.num_slices} slices, {thickness}mm)"
    )

    return best_num, reason


def convert_series_to_nifti(series_info, output_path, verbose=True):
    try:
        if not series_info.file_paths:
            if verbose:
                print(f"  No files found for Series {series_info.series_number}")
            return False

        file_z_pairs = list(zip(series_info.file_paths, series_info.file_z_positions))

        missing_z = [i for i, (_, z) in enumerate(file_z_pairs) if z is None]
        if missing_z:
            if verbose:
                print(f"  Warning: {len(missing_z)} files missing z-position, reading from DICOM...")
            for idx in missing_z:
                fpath = file_z_pairs[idx][0]
                try:
                    ds = pydicom.dcmread(str(fpath), stop_before_pixels=True, force=True)
                    ipp = getattr(ds, 'ImagePositionPatient', None)
                    if ipp and len(ipp) >= 3:
                        file_z_pairs[idx] = (fpath, float(ipp[2]))
                    else:
                        file_z_pairs[idx] = (fpath, float(getattr(ds, 'InstanceNumber', 0) or 0))
                except Exception:
                    file_z_pairs[idx] = (fpath, 0.0)

        file_z_pairs.sort(key=lambda pair: pair[1])
        sorted_files = [pair[0] for pair in file_z_pairs]
        z_positions = [pair[1] for pair in file_z_pairs]

        # deduplicate slices at same z (multi-phase data that snuck thru)
        n_unique = len(set(round(z, 2) for z in z_positions))
        if n_unique < len(sorted_files) * 0.95:
            if verbose:
                print(f"  Warning: {len(sorted_files)} slices but only {n_unique} unique z-positions -- deduplicating")
            seen_z = set()
            deduped = []
            deduped_z = []
            for f, z in zip(sorted_files, z_positions):
                z_round = round(z, 2)
                if z_round not in seen_z:
                    seen_z.add(z_round)
                    deduped.append(f)
                    deduped_z.append(z)
            if verbose:
                print(f"  Kept {len(deduped)} unique slices (removed {len(sorted_files) - len(deduped)} duplicates)")
            sorted_files = deduped
            z_positions = deduped_z

        # some scanners merge 2+ table positions under one series number
        if len(z_positions) > 2:
            gaps = [z_positions[i+1] - z_positions[i] for i in range(len(z_positions) - 1)]
            median_gap = sorted(gaps)[len(gaps) // 2]
            gap_threshold = max(median_gap * 10, 5.0)  # at least 5mm
            large_gap_indices = [i for i, g in enumerate(gaps) if g > gap_threshold]

            if large_gap_indices:
                boundaries = [-1] + large_gap_indices + [len(sorted_files) - 1]
                clusters = []
                for i in range(len(boundaries) - 1):
                    start = boundaries[i] + 1
                    end = boundaries[i + 1] + 1
                    clusters.append((start, end))

                largest = max(clusters, key=lambda c: c[1] - c[0])
                if verbose:
                    print(f"  Warning: {len(large_gap_indices)} large z-gap(s) detected (>{gap_threshold:.1f}mm) -- multiple table positions")
                    for ci, (s, e) in enumerate(clusters):
                        marker = " <-- kept" if (s, e) == largest else ""
                        extent = z_positions[e-1] - z_positions[s]
                        print(f"    Cluster {ci}: {e-s} slices, z=[{z_positions[s]:.1f}, {z_positions[e-1]:.1f}], extent={extent:.1f}mm{marker}")
                sorted_files = sorted_files[largest[0]:largest[1]]
                z_positions = z_positions[largest[0]:largest[1]]

        file_paths = [str(f) for f in sorted_files]

        reader = sitk.ImageSeriesReader()
        reader.SetFileNames(file_paths)
        # print(f"  reading {len(file_paths)} files with SimpleITK...")
        image = reader.Execute()

        size = image.GetSize()
        spacing = image.GetSpacing()

        if size[2] < 50:
            if verbose:
                print(f"  Suspicious result: only {size[2]} slices")
            return False

        if spacing[2] > 50:
            if verbose:
                print(f"  Suspicious spacing: {spacing[2]:.1f}mm between slices")
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(image, str(output_path))

        if verbose:
            print(f"  Converted: {len(file_paths)} files -> {output_path.name}")
            print(f"  Size: {size}, Spacing: [{spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}]mm")

        return True

    except Exception as e:
        if verbose:
            print(f"  Conversion failed: {e}")
        return False


def validate_nifti(nifti_path, expected_slices=None):
    try:
        nii = nib.load(str(nifti_path))
        shape = nii.shape
        spacing = nib.affines.voxel_sizes(nii.affine)

        z_slices = shape[2] if len(shape) >= 3 else shape[0]
        z_spacing = spacing[2] if len(spacing) >= 3 else spacing[0]

        if z_slices < 50:
            return False, f"Only {z_slices} slices (likely scout images)"

        if z_spacing > 50:
            return False, f"{z_spacing:.1f}mm spacing (scout images)"

        if expected_slices and z_slices < expected_slices * 0.8:
            return False, f"Slice count mismatch: {z_slices} vs expected {expected_slices}"

        return True, f"Valid: shape={shape}, spacing=[{spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}]mm"

    except Exception as e:
        return False, f"Could not validate: {e}"

def process(input_path, output_path, series_number=None, verbose=True):
    """DICOM folder in, NIfTI file out."""
    t0 = time.time()
    input_path = Path(input_path)
    output_path = Path(output_path)

    result = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "status": "pending",
    }

    if not input_path.exists():
        result["status"] = "error"
        result["error"] = f"Input path does not exist: {input_path}"
        return result

    if verbose:
        print(f"Scanning DICOM folder: {input_path}")

    config = SeriesSelectionConfig()
    series_dict = explore_dicom_folder(input_path, config)

    if not series_dict:
        result["status"] = "error"
        result["error"] = "No DICOM series found in folder"
        return result

    result["series_found"] = len(series_dict)
    result["series_info"] = {
        snum: {
            "description": info.description,
            "num_slices": info.num_slices,
            "slice_thickness": info.slice_thickness,
            "score": info.selection_score,
            "is_candidate": info.is_ccta_candidate,
            "fov_mm": info.fov_mm,
        }
        for snum, info in series_dict.items()
    }

    if verbose:
        print(f"Found {len(series_dict)} series:")
        for snum, info in sorted(series_dict.items(), key=lambda x: x[1].selection_score, reverse=True):
            marker = "+" if info.is_ccta_candidate else "-"
            thickness = f"{info.slice_thickness:.2f}" if info.slice_thickness else "N/A"
            fov_str = f"{info.fov_mm:.0f}mm" if info.fov_mm else "N/A"
            print(f"  [{marker}] Series {snum}: {info.description} ({info.num_slices} slices, {thickness}mm, FOV={fov_str}, score={info.selection_score:.0f})")

    selected_series, selection_reason = select_best_series(series_dict, series_number)

    if selected_series is None:
        result["status"] = "error"
        result["error"] = selection_reason
        return result

    result["selected_series"] = selected_series
    result["selection_reason"] = selection_reason

    best_info = series_dict[selected_series]
    quality_warnings = []

    if best_info.fov_mm is not None and best_info.fov_mm > config.borderline_fov_max:
        quality_warnings.append(
            f"Selected series has wide FOV ({best_info.fov_mm:.0f}mm). "
            f"Model trained on cardiac close-up (150-250mm). Results may be unreliable."
        )

    if best_info.selection_score < 200:
        quality_warnings.append(
            f"Selected series has low quality score ({best_info.selection_score:.0f}). "
            f"No ideal cardiac series found in this scan."
        )

    if 'segment' in best_info.description.lower():
        quality_warnings.append(
            "Selected series uses SEGMENT reconstruction, which may have different FOV characteristics."
        )

    result["quality_warnings"] = quality_warnings

    if verbose:
        print(f"\nSelected: {selection_reason}")
        if quality_warnings:
            print(f"\n  WARNING(S):")
            for w in quality_warnings:
                print(f"    - {w}")

    series_info = series_dict[selected_series]

    if verbose:
        print(f"\nConverting Series {selected_series}...")

    success = convert_series_to_nifti(series_info, output_path, verbose)

    if not success:
        result["status"] = "error"
        result["error"] = "Conversion failed"
        return result

    is_valid, validation_msg = validate_nifti(output_path)

    if not is_valid:
        result["status"] = "error"
        result["error"] = f"Validation failed: {validation_msg}"
        return result

    nii = nib.load(str(output_path))
    shape = nii.shape
    spacing = nib.affines.voxel_sizes(nii.affine)
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    # TODO: maybe cache the nii object so we don't reload it in preprocess

    elapsed = time.time() - t0

    result.update({
        "status": "success",
        "output_shape": shape,
        "output_spacing": tuple(float(s) for s in spacing),
        "file_size_mb": round(file_size_mb, 2),
        "runtime_sec": round(elapsed, 2),
    })

    if verbose:
        print(f"\nSuccess!")
        print(f"  Output: {output_path}")
        print(f"  Shape: {shape}")
        print(f"  Spacing: [{spacing[0]:.2f}, {spacing[1]:.2f}, {spacing[2]:.2f}]mm")
        print(f"  File size: {file_size_mb:.2f} MB")
        print(f"  Runtime: {elapsed:.2f}s")

    return result


if __name__ == "__main__":
    # quick test with case-30
    import sys

    test_input = "data/ziv/case-30"
    test_output = "temp/case-30.nii.gz"

    if len(sys.argv) >= 3:
        test_input = sys.argv[1]
        test_output = sys.argv[2]

    print("DICOM to NIfTI Conversion Test")
    print("-" * 40)

    result = process(test_input, test_output, verbose=True)

    # print(f"raw result dict: {result}")
    print()
    print("Result:")
    for key, value in result.items():
        if key != "series_info":
            print(f"  {key}: {value}")
