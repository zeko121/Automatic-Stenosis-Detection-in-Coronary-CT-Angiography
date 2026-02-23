"""CLI orchestrator for model evaluation across all 4 tiers."""

import argparse
import gc
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from evaluation.config import (
    EvalConfig,
    discover_imagecas_cases,
    discover_models,
    discover_ziv_cases,
    imagecas_image_path,
    imagecas_label_path,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluation")


def _load_imagecas_masks(
    case_id: int,
    cases: Dict[int, Path],
    model_dir: Path,
    config: EvalConfig,
    temp_dir: Path,
) -> Optional[Dict[str, np.ndarray]]:
    """Preprocess + segment an ImageCAS case, return masks or None."""
    import nibabel as nib
    import zarr

    from pipeline.preprocess import process as preprocess_process
    from pipeline.segment import process as segment_process
    from pipeline.postprocess import postprocess_mask

    try:
        img_path = imagecas_image_path(cases, case_id)
        lbl_path = imagecas_label_path(cases, case_id)

        if not img_path.exists() or not lbl_path.exists():
            logger.warning(f"Case {case_id}: missing file")
            return None

        case_temp = temp_dir / f"case_{case_id:04d}"
        case_temp.mkdir(parents=True, exist_ok=True)

        zarr_path = case_temp / "preprocessed.zarr"
        seg_path = case_temp / "segmentation.zarr"

        preprocess_process(
            str(img_path), str(zarr_path),
            model_config_path=str(model_dir / "run_config.json"),
            verbose=False,
        )

        gt_nib = nib.load(str(lbl_path))
        gt_raw = gt_nib.get_fdata()

        g = zarr.open_group(str(zarr_path), mode="r")
        image = g["image"][:]

        from skimage.transform import resize
        gt_resampled = resize(
            gt_raw, image.shape, order=1, preserve_range=True, anti_aliasing=False
        )
        gt_mask = (gt_resampled > 0.3).astype(np.uint8)

        segment_process(
            str(zarr_path), str(seg_path), str(model_dir),
            verbose=False,
        )

        g_seg = zarr.open_group(str(seg_path), mode="r")
        model_mask_raw = g_seg["mask"][:]

        model_mask_pp, _ = postprocess_mask(model_mask_raw, verbose=False)

        return {
            "gt_mask": gt_mask,
            "model_mask": model_mask_pp,
            "model_mask_raw": model_mask_raw,
            "image": image,
        }

    except Exception as e:
        logger.error(f"Case {case_id} failed: {e}")
        traceback.print_exc()
        return None
    finally:
        gc.collect()


def _load_ziv_mask(
    case_name: str,
    case_path: Path,
    model_dir: Path,
    config: EvalConfig,
    temp_dir: Path,
) -> Optional[np.ndarray]:
    """Run full pipeline on a Ziv case, return post-processed mask."""
    from pipeline.preprocess import process as preprocess_process
    from pipeline.segment import process as segment_process
    from pipeline.postprocess import postprocess_mask
    from pipeline.runner import detect_input_type

    try:
        case_temp = temp_dir / case_name
        case_temp.mkdir(parents=True, exist_ok=True)

        input_type = detect_input_type(str(case_path))

        if input_type == "dicom":
            from pipeline.dicom_to_nifti import process as dicom_process
            nifti_path = case_temp / "converted.nii.gz"
            dicom_process(str(case_path), str(nifti_path), verbose=False)
            preprocess_input = str(nifti_path)
        else:
            preprocess_input = str(case_path)

        zarr_path = case_temp / "preprocessed.zarr"
        seg_path = case_temp / "segmentation.zarr"

        preprocess_process(
            preprocess_input, str(zarr_path),
            model_config_path=str(model_dir / "run_config.json"),
            verbose=False,
        )

        segment_process(
            str(zarr_path), str(seg_path), str(model_dir),
            verbose=False,
        )

        import zarr
        g_seg = zarr.open_group(str(seg_path), mode="r")
        model_mask_raw = g_seg["mask"][:]
        model_mask_pp, _ = postprocess_mask(model_mask_raw, verbose=False)

        return model_mask_pp

    except Exception as e:
        logger.error(f"Ziv case {case_name} failed: {e}")
        traceback.print_exc()
        return None
    finally:
        gc.collect()


def run_tier1(
    model_name: str,
    model_dir: Path,
    config: EvalConfig,
) -> Dict[str, float]:
    from evaluation.tier1_segmentation import (
        evaluate_case, aggregate_tier1,
    )

    cases = discover_imagecas_cases(config.imagecas_root, config.imagecas_subfolders)
    if not cases:
        logger.warning("No ImageCAS cases found — skipping Tier 1")
        return {}

    case_ids = sorted(cases.keys())
    if config.max_cases and config.max_cases < len(case_ids):
        rng = np.random.RandomState(config.seed)
        case_ids = sorted(rng.choice(case_ids, config.max_cases, replace=False))

    logger.info(f"Tier 1: evaluating {len(case_ids)} ImageCAS cases for {model_name}")

    results = []
    for i, cid in enumerate(case_ids, 1):
        logger.info(f"  [{i}/{len(case_ids)}] Case {cid}")
        data = _load_imagecas_masks(cid, cases, model_dir, config,
                                     config.temp_dir / model_name)
        if data is None:
            continue

        m = evaluate_case(
            data["model_mask"], data["gt_mask"],
            case_id=str(cid),
            spacing_mm=config.voxel_spacing_mm,
            surface_dice_tolerances=config.surface_dice_tolerances_mm,
        )
        results.append(m)
        logger.info(f"    Dice={m.dice:.3f} SD@0.5={m.surface_dice.get(0.5, 0):.3f} "
                     f"MARE={m.mare:.3f} MSRE={m.msre:+.3f}")

        del data
        gc.collect()

    agg = aggregate_tier1(results)
    logger.info(f"Tier 1 summary ({model_name}): Dice={agg.get('dice_mean', 0):.3f} "
                f"SD@0.5={agg.get('surface_dice_0.5mm_mean', 0):.3f}")
    return agg


def run_tier2(
    model_name: str,
    model_dir: Path,
    config: EvalConfig,
    dataset: str = "imagecas",
) -> Dict[str, float]:
    from evaluation.tier2_structural import (
        evaluate_case, aggregate_tier2,
    )

    results = []

    if dataset == "imagecas":
        cases = discover_imagecas_cases(config.imagecas_root, config.imagecas_subfolders)
        if not cases:
            logger.warning("No ImageCAS cases found — skipping Tier 2")
            return {}

        case_ids = sorted(cases.keys())
        if config.max_cases and config.max_cases < len(case_ids):
            rng = np.random.RandomState(config.seed)
            case_ids = sorted(rng.choice(case_ids, config.max_cases, replace=False))

        logger.info(f"Tier 2: evaluating {len(case_ids)} ImageCAS cases for {model_name}")

        for i, cid in enumerate(case_ids, 1):
            logger.info(f"  [{i}/{len(case_ids)}] Case {cid}")
            data = _load_imagecas_masks(cid, cases, model_dir, config,
                                         config.temp_dir / model_name)
            if data is None:
                continue

            m = evaluate_case(data["model_mask"], case_id=str(cid))
            results.append(m)
            logger.info(f"    Bif={m.n_bifurcations} LCF={m.largest_component_fraction:.2f} "
                         f"CL={m.total_centerline_length_mm:.0f}mm CS={m.continuity_score:.3f}")

            del data
            gc.collect()

    elif dataset == "ziv":
        ziv_cases = discover_ziv_cases(config.ziv_root)
        if not ziv_cases:
            logger.warning("No Ziv cases found — skipping Tier 2")
            return {}

        logger.info(f"Tier 2: evaluating {len(ziv_cases)} Ziv cases for {model_name}")

        for i, (name, path) in enumerate(sorted(ziv_cases.items()), 1):
            logger.info(f"  [{i}/{len(ziv_cases)}] {name}")
            mask = _load_ziv_mask(name, path, model_dir, config,
                                   config.temp_dir / model_name)
            if mask is None:
                continue

            m = evaluate_case(mask, case_id=name)
            results.append(m)
            logger.info(f"    Bif={m.n_bifurcations} LCF={m.largest_component_fraction:.2f} "
                         f"CS={m.continuity_score:.3f}")

            del mask
            gc.collect()

    agg = aggregate_tier2(results)
    logger.info(f"Tier 2 summary ({model_name}, {dataset}): "
                f"CS={agg.get('continuity_score_mean', 0):.3f}")
    return agg


def run_tier3_oracle(
    model_name: str,
    model_dir: Path,
    config: EvalConfig,
) -> List:
    from evaluation.tier3_downstream import (
        run_oracle_pipeline, compare_model_vs_oracle, OracleComparison,
    )
    from pipeline.postprocess import postprocess_mask
    from pipeline.centerline import extract_vessel_tree
    from pipeline.label_arteries import label_arteries
    from pipeline.stenosis import StenosisDetector, StenosisConfig

    cases = discover_imagecas_cases(config.imagecas_root, config.imagecas_subfolders)
    if not cases:
        logger.warning("No ImageCAS cases found — skipping Tier 3a")
        return []

    case_ids = sorted(cases.keys())
    if config.max_cases and config.max_cases < len(case_ids):
        rng = np.random.RandomState(config.seed)
        case_ids = sorted(rng.choice(case_ids, config.max_cases, replace=False))

    logger.info(f"Tier 3a: oracle comparison on {len(case_ids)} ImageCAS cases for {model_name}")

    results = []
    for i, cid in enumerate(case_ids, 1):
        logger.info(f"  [{i}/{len(case_ids)}] Case {cid}")
        data = _load_imagecas_masks(cid, cases, model_dir, config,
                                     config.temp_dir / model_name)
        if data is None:
            continue

        oracle_findings = run_oracle_pipeline(
            data["gt_mask"],
            config.temp_dir / model_name / f"oracle_{cid}",
            case_id=str(cid),
        )
        if oracle_findings is None:
            logger.warning(f"  Case {cid}: oracle pipeline failed")
            del data
            gc.collect()
            continue

        model_findings = run_oracle_pipeline(
            data["model_mask"],
            config.temp_dir / model_name / f"model_{cid}",
            case_id=str(cid),
        )
        if model_findings is None:
            logger.warning(f"  Case {cid}: model pipeline failed")
            del data
            gc.collect()
            continue

        comp = compare_model_vs_oracle(model_findings, oracle_findings, case_id=str(cid))
        results.append(comp)
        logger.info(f"    Agreement={comp.severity_agreement_rate:.0%} "
                     f"Misses={comp.misses} Halluc={comp.hallucinations}")

        del data
        gc.collect()

    return results


def run_tier3_ziv(
    model_name: str,
    model_dir: Path,
    config: EvalConfig,
) -> dict:
    from evaluation.tier3_downstream import evaluate_ziv_cases
    from pipeline.compare_gt import compare_findings

    ziv_cases = discover_ziv_cases(config.ziv_root)
    if not ziv_cases:
        logger.warning("No Ziv cases found — skipping Tier 3b")
        return {}

    comparisons = []
    for name, case_path in sorted(ziv_cases.items()):
        gt_path = case_path / "gt_report.json"
        if not gt_path.exists():
            continue

        # Run full pipeline on this case
        mask = _load_ziv_mask(name, case_path, model_dir, config,
                               config.temp_dir / model_name)
        if mask is None:
            continue

        try:
            from evaluation.tier3_downstream import run_oracle_pipeline
            findings = run_oracle_pipeline(
                mask, config.temp_dir / model_name / name, case_id=name,
            )
            if findings is None:
                continue

            gt_report = json.loads(gt_path.read_text(encoding="utf-8"))
            comp = compare_findings(findings, gt_report)
            comparisons.append(comp)
            logger.info(f"  {name}: Agreement={comp.agreement_rate:.0%} "
                         f"Sens={comp.sensitivity:.0%}")

        except Exception as e:
            logger.error(f"  {name} comparison failed: {e}")

        finally:
            del mask
            gc.collect()

    if not comparisons:
        logger.warning("No Ziv cases with GT reports — skipping Tier 3b")
        return {}

    return evaluate_ziv_cases(
        comparisons,
        n_bootstrap=config.bootstrap_n_iterations,
        ci=config.bootstrap_ci,
        seed=config.seed,
    )


def run_tier4(
    model_name: str,
    imagecas_tier2: Dict[str, float],
    ziv_tier2: Dict[str, float],
) -> dict:
    from evaluation.tier4_robustness import compute_domain_gap

    if not imagecas_tier2 or not ziv_tier2:
        logger.warning("Missing Tier 2 data for domain gap — skipping Tier 4")
        return {}

    result = compute_domain_gap(imagecas_tier2, ziv_tier2, model_name)
    logger.info(f"Tier 4 ({model_name}): domain_gap={result.domain_gap:.3f}")
    return result.to_dict()


def run_evaluation(args: argparse.Namespace):
    config = EvalConfig(
        max_cases=args.max_cases,
        seed=args.seed,
    )

    tiers = set(int(t) for t in args.tiers.split(","))
    logger.info(f"Running tiers: {sorted(tiers)}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Models root: {config.models_root}")

    if args.models:
        model_dirs = {}
        for m in args.models:
            p = Path(m)
            if not p.is_absolute():
                p = config.models_root / m
            if p.exists():
                model_dirs[p.name] = p
            else:
                logger.error(f"Model directory not found: {p}")
    else:
        model_dirs = discover_models(config.models_root)

    if not model_dirs:
        logger.error("No models found. Specify --models or check models directory.")
        return

    logger.info(f"Models to evaluate: {list(model_dirs.keys())}")

    all_results: Dict[str, Dict[str, Dict]] = {}

    for model_name, model_dir in model_dirs.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"{'='*60}")

        model_results: Dict[str, Dict] = {}
        t0 = time.time()

        if 1 in tiers:
            model_results["tier1"] = run_tier1(model_name, model_dir, config)

        imagecas_t2 = {}
        ziv_t2 = {}
        if 2 in tiers:
            dataset = args.dataset
            if dataset == "both" or dataset == "imagecas":
                imagecas_t2 = run_tier2(model_name, model_dir, config, "imagecas")
            if dataset == "both" or dataset == "ziv":
                ziv_t2 = run_tier2(model_name, model_dir, config, "ziv")
            model_results["tier2"] = imagecas_t2 or ziv_t2

        tier3_data = {}
        if 3 in tiers:
            oracle_results = run_tier3_oracle(model_name, model_dir, config)
            ziv_results = run_tier3_ziv(model_name, model_dir, config)

            from evaluation.tier3_downstream import (
                build_tier3_metrics, aggregate_oracle,
            )
            t3 = build_tier3_metrics(model_name, oracle_results,
                                      ziv_results if ziv_results else None)
            tier3_data = t3.to_dict()
            model_results["tier3"] = tier3_data

        if 4 in tiers:
            if not imagecas_t2 and 2 not in tiers:
                imagecas_t2 = run_tier2(model_name, model_dir, config, "imagecas")
            if not ziv_t2 and 2 not in tiers:
                ziv_t2 = run_tier2(model_name, model_dir, config, "ziv")
            model_results["tier4"] = run_tier4(model_name, imagecas_t2, ziv_t2)

        elapsed = time.time() - t0
        logger.info(f"Model {model_name} completed in {elapsed:.1f}s")

        all_results[model_name] = model_results

    from evaluation.composite_score import rank_models
    from evaluation.report_generator import (
        generate_markdown_report, save_results_json,
    )

    ranked = rank_models(all_results, config.tier_weights)

    report_path = config.output_dir / "evaluation_report.md"
    report = generate_markdown_report(ranked, report_path)

    json_path = config.output_dir / "evaluation_results.json"
    save_results_json(ranked, json_path)

    logger.info(f"\nReport saved to: {report_path}")
    logger.info(f"Results JSON saved to: {json_path}")

    print("\n" + "=" * 60)
    print("FINAL RANKING")
    print("=" * 60)
    for m in ranked:
        print(f"  #{m.rank}  {m.model_name:<30s}  Score: {m.composite_score:.3f}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Model Evaluation & Ranking Framework for Stenosis Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m evaluation.run_evaluation --tiers 2 --dataset ziv
  python -m evaluation.run_evaluation --tiers 1,2 --max-cases 5
  python -m evaluation.run_evaluation --tiers 1,2,3,4 --models model_a model_b
        """,
    )
    parser.add_argument(
        "--tiers", default="1,2,3,4",
        help="Comma-separated tier numbers to run (default: 1,2,3,4)",
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Model directory names (under models/) or absolute paths. "
             "If not specified, evaluates all models found.",
    )
    parser.add_argument(
        "--dataset", default="both", choices=["imagecas", "ziv", "both"],
        help="Dataset to evaluate on (default: both)",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Max cases per dataset (for quick testing)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for case sampling (default: 42)",
    )

    args = parser.parse_args()
    run_evaluation(args)


if __name__ == "__main__":
    main()
