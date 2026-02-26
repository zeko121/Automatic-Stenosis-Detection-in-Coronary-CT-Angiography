"""Pipeline runner -- orchestrates all 8 stages."""

import gc
import json
import shutil
import time
import traceback
from datetime import datetime
from pathlib import Path

from pipeline import dicom_to_nifti
from pipeline import preprocess
from pipeline import segment
from pipeline import postprocess
from pipeline import centerline
from pipeline import label_arteries
from pipeline import stenosis
from pipeline import visualize


PIPELINE_STAGES = [
    ("DICOM to NIfTI", "Converting DICOM files"),
    ("Preprocessing", "HU window, resample, normalize"),
    ("Segmentation", "AI vessel segmentation"),
    ("Post-processing", "Cleaning segmentation mask"),
    ("Centerlines", "Extracting vessel centerlines"),
    ("Artery Labeling", "Identifying coronary arteries"),
    ("Stenosis Detection", "Analyzing stenosis severity"),
    ("Visualization", "Generating 3D visualization"),
]

MODEL_DIR = Path(__file__).parent.parent / "models" / "2025-12-31_02-53-53"
GAP_MODEL_DIR = MODEL_DIR.parent / "2026-02-12_13-14-10"
TEMP_DIR = Path(__file__).parent.parent / "temp"
TEMP_DIR.mkdir(exist_ok=True)


def detect_input_type(input_path):
    """Return 'dicom', 'nifti', 'zarr', or 'unknown'."""
    path = Path(input_path)

    if path.is_file():
        if path.name.lower().endswith('.nii.gz') or path.suffix.lower() == '.nii':
            return 'nifti'
        if path.suffix.lower() == '.dcm':
            return 'dicom'

    if path.is_dir():
        # check for preprocessed zarr before DICOM (zarr dirs can contain misc files)
        if path.suffix.lower() == '.zarr' and (path / '.zgroup').is_file():
            try:
                import zarr as _zarr
                store = _zarr.open_group(str(path), mode='r')
                if 'image' in store:
                    return 'zarr'
            except Exception:
                pass

        dcm_files = list(path.glob('**/*.dcm')) + list(path.glob('**/*.DCM'))
        if dcm_files:
            return 'dicom'

        # extensionless files are sometimes DICOM
        for f in path.rglob('*'):
            if f.is_file() and not f.suffix:
                try:
                    import pydicom
                    pydicom.dcmread(str(f), stop_before_pixels=True)
                    return 'dicom'
                except Exception:
                    pass
                break

        nii_files = list(path.glob('**/*.nii*'))
        if nii_files:
            return 'nifti'

    return 'unknown'


def generate_clinical_report(case_name, findings_data, analysis_time):
    """Build plain-text clinical report."""
    summary = findings_data.get("summary", {})
    findings = findings_data.get("findings", [])

    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CORONARY ARTERY STENOSIS ANALYSIS REPORT")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"Patient/Case ID:    {case_name}")
    report_lines.append(f"Analysis Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Processing Time:    {analysis_time:.1f} seconds")
    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append("SUMMARY")
    report_lines.append("-" * 70)
    report_lines.append(f"Segments Analyzed:  {summary.get('segments_analyzed', 'N/A')}")
    report_lines.append(f"Total Findings:     {summary.get('total_findings', 0)}")
    report_lines.append(f"Maximum Severity:   {summary.get('max_severity', 'None')}")
    report_lines.append("")

    by_severity = summary.get("by_severity", {})
    report_lines.append("Severity Distribution:")
    report_lines.append(f"  - Severe (\u226570%):     {by_severity.get('Severe', 0)}")
    report_lines.append(f"  - Moderate (50-70%): {by_severity.get('Moderate', 0)}")
    report_lines.append(f"  - Mild (25-50%):     {by_severity.get('Mild', 0)}")
    report_lines.append(f"  - Normal (<25%):     {by_severity.get('Normal', 0)}")
    report_lines.append("")

    if findings:
        report_lines.append("-" * 70)
        report_lines.append("DETAILED FINDINGS")
        report_lines.append("-" * 70)

        severity_order = {"Severe": 0, "Moderate": 1, "Mild": 2, "Normal": 3}
        sorted_findings = sorted(findings, key=lambda x: severity_order.get(x.get("severity", "Normal"), 4))

        for i, finding in enumerate(sorted_findings, 1):
            severity = finding.get("severity", "Unknown")
            stenosis_pct = finding.get("stenosis_percent", 0)
            segment_id = finding.get("segment_id", "?")
            min_radius = finding.get("min_radius_mm", 0)
            ref_radius = finding.get("reference_radius_mm", 0)
            location = finding.get("location_voxel", [0, 0, 0])
            confidence = finding.get("confidence", 0)
            artery_name = finding.get("artery_name", "")
            artery_region = finding.get("artery_region", "")

            location_label = artery_region or artery_name or f"Segment {segment_id}"

            report_lines.append(f"\nFinding #{i}:")
            report_lines.append(f"  Location:          {location_label}")
            report_lines.append(f"  Segment ID:        {segment_id}")
            report_lines.append(f"  Severity:          {severity}")
            report_lines.append(f"  Stenosis:          {stenosis_pct:.1f}%")
            report_lines.append(f"  Minimum Radius:    {min_radius:.2f} mm")
            report_lines.append(f"  Reference Radius:  {ref_radius:.2f} mm")
            report_lines.append(f"  Location (voxel):  [{location[0]:.0f}, {location[1]:.0f}, {location[2]:.0f}]")
            report_lines.append(f"  Confidence:        {confidence:.0%}")
    else:
        report_lines.append("-" * 70)
        report_lines.append("No significant stenosis findings detected.")

    report_lines.append("")
    report_lines.append("-" * 70)
    report_lines.append("IMPRESSION")
    report_lines.append("-" * 70)

    if summary.get("max_severity") == "Severe":
        report_lines.append("SIGNIFICANT CORONARY ARTERY DISEASE DETECTED.")
        report_lines.append(f"Severe stenosis (\u226570%) identified in {by_severity.get('Severe', 0)} segment(s).")
        report_lines.append("Clinical correlation and further evaluation recommended.")
    elif summary.get("max_severity") == "Moderate":
        report_lines.append("MODERATE CORONARY ARTERY STENOSIS DETECTED.")
        report_lines.append(f"Moderate stenosis (50-70%) identified in {by_severity.get('Moderate', 0)} segment(s).")
        report_lines.append("Clinical follow-up recommended.")
    elif summary.get("max_severity") == "Mild":
        report_lines.append("MILD CORONARY ARTERY CHANGES DETECTED.")
        report_lines.append(f"Mild stenosis (25-50%) identified in {by_severity.get('Mild', 0)} segment(s).")
        report_lines.append("Routine follow-up as clinically indicated.")
    else:
        report_lines.append("NO SIGNIFICANT STENOSIS DETECTED.")
        report_lines.append("Coronary arteries appear within normal limits.")

    report_lines.append("")
    report_lines.append("=" * 70)
    report_lines.append("DISCLAIMER: This is an automated AI analysis for research purposes.")
    report_lines.append("Results should be validated by qualified medical professionals.")
    report_lines.append("University of Haifa & Ziv Medical Center")
    report_lines.append("=" * 70)

    return "\n".join(report_lines)


def pipeline_stages(input_path, enable_postprocess=True, enable_refinement=False,
                     model_dir=None):
    """Generator that yields (stage_idx, state, status_lines, result) per stage."""
    model_dir = Path(model_dir) if model_dir else MODEL_DIR
    start_time = time.time()
    case_name = Path(input_path).stem

    case_temp = TEMP_DIR / f"{case_name}_{int(time.time())}"
    case_temp.mkdir(exist_ok=True)

    status_lines = []

    def log(msg):
        status_lines.append(msg)
        try:
            print(msg)
        except UnicodeEncodeError:
            print(msg.encode('ascii', 'replace').decode('ascii'))

    try:
        input_type = detect_input_type(input_path)
        log(f"Case: {case_name}")
        log(f"Input: {input_path}")
        log(f"Detected type: {input_type}")

        if input_type == 'unknown':
            yield (-1, "error", status_lines,
                   ("Error: Could not detect input type. Please provide a DICOM folder, NIfTI file, or preprocessed .zarr.", "{}", "", None, None, None))
            return

        if input_type == 'zarr':
            log("\ninput is preprocessed .zarr — skipping conversion + preprocessing")
            zarr_path = Path(input_path)
            yield (0, "done", status_lines, None)
            yield (1, "done", status_lines, None)
        else:
            yield (0, "running", status_lines, None)
            if input_type == 'dicom':
                log("\nconverting DICOM to NIfTI...")

                nifti_path = case_temp / f"{case_name}.nii.gz"
                result = dicom_to_nifti.process(input_path, str(nifti_path))

                if result.get("status") != "success":
                    yield (-1, "error", status_lines,
                           (f"Error in DICOM conversion: {result.get('error', 'Unknown error')}", "{}", "", None))
                    return

                log(f"  Shape: {result['output_shape']}, Time: {result.get('runtime_sec', 0):.1f}s")
                if result.get("quality_warnings"):
                    for warning in result["quality_warnings"]:
                        log(f"  WARNING: {warning}")
                log("converting DICOM to NIfTI... \u2713")
                nifti_file = nifti_path
            else:
                log("\ninput is NIfTI, skipping conversion... \u2713")
                if Path(input_path).is_dir():
                    nii_files = list(Path(input_path).glob('*.nii*'))
                    if not nii_files:
                        yield (-1, "error", status_lines,
                               ("Error: No NIfTI files found in folder", "{}", "", None))
                        return
                    nifti_file = nii_files[0]
                else:
                    nifti_file = Path(input_path)
            yield (0, "done", status_lines, None)

            yield (1, "running", status_lines, None)
            log("\npreprocessing (HU window, resample, z-score)...")

            zarr_path = case_temp / f"{case_name}.zarr"
            model_config_path = model_dir / "run_config.json"
            result = preprocess.process(
                str(nifti_file), str(zarr_path), str(model_config_path), verbose=False
            )

            if result.get("status") != "success":
                yield (-1, "error", status_lines,
                       (f"Error in preprocessing: {result.get('error', 'Unknown error')}", "{}", "", None, None, None))
                return

            log(f"  Shape: {result.get('final_shape', result.get('output_shape', 'unknown'))}, Time: {result.get('runtime_sec', result.get('elapsed_seconds', 0)):.1f}s")
            log("preprocessing... \u2713")
            yield (1, "done", status_lines, None)

        yield (2, "running", status_lines, None)
        log("\nrunning vessel segmentation...")

        segmented_path = case_temp / f"{case_name}_segmented.zarr"
        result = segment.process(
            str(zarr_path), str(segmented_path), str(model_dir),
            patch_size=(160, 160, 160), overlap=0.5, verbose=False
        )

        if result.get("status") != "success":
            yield (-1, "error", status_lines,
                   (f"Error in segmentation: {result.get('error', 'Unknown error')}", "{}", "", None, None, None))
            return

        log(f"  Vessel voxels: {result['vessel_voxels']:,}, Time: {result.get('runtime_sec', result.get('elapsed_seconds', 0)):.1f}s")
        log("vessel segmentation... \u2713")

        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        yield (2, "done", status_lines, None)

        if enable_refinement:
            log("\n[Re-segmentation] Attempting to reconnect vessel fragments...")
            try:
                from pipeline.gap_reconnection.refine_segmentation import refine_mask
                import zarr as _zarr_ref
                _seg_group = _zarr_ref.open_group(str(segmented_path), mode='r')
                _seg_mask = _seg_group['mask'][:]
                _main_probs = _seg_group['probs'][:] if 'probs' in _seg_group else None
                _refine_image = _zarr_ref.open_group(str(zarr_path), mode='r')['image'][:]

                refine_result = refine_mask(
                    image_volume=_refine_image,
                    prediction_mask=_seg_mask,
                    model_dir=str(model_dir),
                    gap_model_dir=str(GAP_MODEL_DIR),
                    main_model_probs=_main_probs,
                )
                del _refine_image
                gc.collect()

                if refine_result['n_gaps_connected'] > 0:
                    refined_mask = refine_result['refined_mask']
                    _seg_out = _zarr_ref.open_group(str(segmented_path), mode='r+')
                    _seg_out['mask'][:] = refined_mask
                    log(f"  Re-segmentation: +{refine_result['n_voxels_added']:,} voxels, "
                        f"{refine_result['n_gaps_connected']} gaps connected "
                        f"({refine_result['runtime_sec']:.1f}s)")
                    del refined_mask
                else:
                    log(f"  Re-segmentation: no gaps connected "
                        f"(endpoints={refine_result['n_endpoints_found']}, "
                        f"gaps_found={refine_result['n_gaps_found']})")
                del _seg_mask, _main_probs
                gc.collect()
            except Exception as e:
                log(f"  Re-segmentation failed (non-fatal): {e}")

        yield (3, "running", status_lines, None)
        if enable_postprocess:
            log("\npost-processing segmentation mask...")

            postprocessed_path = case_temp / f"{case_name}_postprocessed.zarr"
            result = postprocess.process(
                str(segmented_path), str(postprocessed_path), verbose=False
            )

            if result.get("status") != "success":
                yield (-1, "error", status_lines,
                       (f"Error in post-processing: {result.get('error', 'Unknown error')}", "{}", "", None, None, None))
                return

            log(f"  {result['input_voxels']:,} -> {result['output_voxels']:,} voxels, "
                f"{result.get('bridges_added', 0)} bridges, "
                f"Time: {result.get('runtime_sec', 0):.1f}s")
            log("post-processing... \u2713")
            downstream_path = postprocessed_path
        else:
            log("\npost-processing... SKIPPED")
            downstream_path = segmented_path
        yield (3, "done", status_lines, None)

        yield (4, "running", status_lines, None)
        log("\nextracting vessel centerlines...")

        centerline_path = case_temp / f"{case_name}_centerline.json"
        result = centerline.process(
            str(downstream_path), str(centerline_path),
            voxel_spacing_mm=0.5, verbose=False
        )

        if result.get("status") != "success":
            yield (-1, "error", status_lines,
                   (f"Error in centerline extraction: {result.get('error', 'Unknown error')}", "{}", "", None, None, None))
            return

        log(f"  {result['num_centerline_points']} points, {result['num_bifurcations']} bifurcations, {result.get('runtime_sec', 0):.1f}s")
        log("extracting centerlines... \u2713")
        yield (4, "done", status_lines, None)

        yield (5, "running", status_lines, None)
        log("\nlabeling coronary arteries...")

        artery_labels_data = None
        try:
            import zarr as _zarr

            _mask_store = _zarr.open_group(str(downstream_path), mode='r')
            _vessel_mask = _mask_store['mask'][:] if 'mask' in _mask_store else None

            with open(centerline_path, 'r') as f:
                _cl_data = json.load(f)

            labeling_result = label_arteries.label_arteries(
                centerline_data=_cl_data,
                vessel_mask=_vessel_mask,
                spacing_mm=0.5,
                verbose=False,
            )

            artery_labels_data = labeling_result.labels
            labeled_count = sum(1 for l in artery_labels_data.values() if l.artery_name != "Unknown")
            log(f"  {labeled_count}/{len(artery_labels_data)} segments labeled, "
                f"{labeling_result.elapsed_seconds:.2f}s")

            label_path = case_temp / f"{case_name}_labels.json"
            with open(label_path, 'w') as f:
                json.dump(labeling_result.to_dict(), f, indent=2)

            del _vessel_mask
            gc.collect()

        except Exception as e:
            log(f"  Artery labeling failed (non-fatal): {e}")
            artery_labels_data = None

        log("artery labeling... \u2713")
        yield (5, "done", status_lines, None)
        yield (6, "running", status_lines, None)
        log("\ndetecting stenosis...")

        stenosis_path = case_temp / f"{case_name}_stenosis.json"
        result = stenosis.process(
            str(centerline_path), str(stenosis_path),
            artery_labels=artery_labels_data,
        )

        log(f"  {result['findings_count']} findings, max severity: {result['max_severity']}, {result.get('elapsed_seconds', result.get('runtime_sec', 0)):.2f}s")
        log("stenosis detection... \u2713")
        yield (6, "done", status_lines, None)

        yield (7, "running", status_lines, None)
        log("\ncreating 3D visualization...")

        viz_path = case_temp / f"{case_name}_visualization.html"
        result = visualize.process(
            str(downstream_path), str(stenosis_path), str(viz_path),
            spacing=(0.5, 0.5, 0.5), title=f"{case_name} | Stenosis Analysis",
            centerline_path=str(centerline_path),
            artery_labels=artery_labels_data,
        )

        log(f"  {result['file_size_kb']:.0f} KB, {result.get('elapsed_seconds', result.get('runtime_sec', 0)):.1f}s")
        log("3D visualization... \u2713")
        yield (7, "done", status_lines, None)

        with open(stenosis_path, 'r') as f:
            findings_data = json.load(f)

        elapsed_total = time.time() - start_time
        summary = findings_data.get("summary", {})

        log(f"\n{'='*50}")
        log("DONE")
        log(f"{'='*50}")
        log(f"total: {elapsed_total:.1f}s, findings: {summary.get('total_findings', 0)}, "
            f"max: {summary.get('max_severity', 'Unknown')}")

        findings_json = json.dumps(findings_data, indent=2)
        clinical_report = generate_clinical_report(case_name, findings_data, elapsed_total)

        yield (-1, "complete", status_lines,
               ("\n".join(status_lines), findings_json, clinical_report,
                str(viz_path), str(downstream_path), findings_data))

    except Exception as e:
        error_msg = f"Pipeline error: {str(e)}\n\n{traceback.format_exc()}"
        log(f"\nERROR: {error_msg}")
        try:
            if case_temp.exists():
                shutil.rmtree(case_temp)
        except Exception:
            pass
        yield (-1, "error", status_lines,
               ("\n".join(status_lines), "{}", "", None, None, None))

    finally:
        gc.collect()


def run_pipeline(input_path, progress=None, enable_postprocess=True,
                 enable_refinement=False, model_dir=None):
    """Run the full pipeline synchronously (non-generator wrapper)."""

    if not input_path or not input_path.strip():
        return "Error: Please enter a path to a DICOM folder, NIfTI file, or preprocessed .zarr.", "{}", "", None

    result = None
    for _stage_idx, _state, _lines, res in pipeline_stages(
        input_path, enable_postprocess=enable_postprocess,
        enable_refinement=enable_refinement, model_dir=model_dir,
    ):
        if res is not None:
            result = res

    if result is None:
        return "Error: pipeline did not complete.", "{}", "", None
    return result[0], result[1], result[2], result[3]
