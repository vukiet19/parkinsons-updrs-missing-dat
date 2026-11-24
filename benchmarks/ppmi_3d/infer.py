"""Predict original-unit UPDRS III from preprocessed MRI, without labels, folds or DaT.

python infer.py --bundle deployment/crad_f0_s0.pt --manifest mri_inputs.csv --output predictions.csv
python infer.py --run_dir runs_ppmi/off30/crad/fold0_seed0 --manifest mri_inputs.csv --output predictions.csv

CSV columns: subject_id,mri_path[,visit_id]. Relative image paths resolve against
the CSV's directory, or --image_root. Other columns are ignored. Multiple slices
of one subject/visit are averaged exactly as in the benchmark. A single model
is selected explicitly; no implicit fold/seed ensembling or score calculation.
Use evaluate.py for validation/test metrics of the original benchmark splits.
"""

import argparse
from pathlib import Path

import torch

from utils.io import write_csv, write_json
from utils.mri_inference import export_run, predict_mri, read_mri_manifest, sha256
from utils.repro import choose_device


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", type=Path)
    source.add_argument("--run_dir", type=Path, help="one completed fold*_seed* run, not a whole CV group")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--image_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", default="best", choices=("best", "last"))
    parser.add_argument("--reference_image", type=Path, help="run_dir mode only; embed/validate its affine")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.bundle and (args.reference_image is not None or args.checkpoint != "best"):
        parser.error("Bundle already fixes checkpoint/grid; --checkpoint last and --reference_image require --run_dir")
    if args.output.suffix.lower() != ".csv":
        parser.error("--output must end in .csv")
    if args.output.resolve() == args.manifest.resolve():
        parser.error("Output cannot overwrite the input manifest")
    report = args.output.with_suffix(".json")
    for path in (args.output, report):
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"{path} exists; select a new output or use --overwrite")
    rows = read_mri_manifest(args.manifest, args.image_root)
    if args.bundle:
        bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    else:
        bundle = export_run(args.run_dir, args.checkpoint, args.reference_image)
    device = choose_device(args.device)
    predictions = predict_mri(bundle, rows, device, args.batch_size)
    write_csv(args.output, predictions)
    write_json(report, dict(method=bundle["method"], role="student", provenance=bundle["provenance"],
                            bundle_sha256=sha256(args.bundle) if args.bundle else None,
                            input_manifest_sha256=sha256(args.manifest), n_rows=len(rows), n_units=len(predictions),
                            target_units="original UPDRS III units (training-fold scaler)",
                            grid_validation="shape_and_affine" if bundle["reference_grid"] else "shape_only",
                            benchmark_eligible=False, device=str(device), batch_size=args.batch_size,
                            note="Arbitrary MRI inference; use evaluate.py on stored splits for benchmark metrics."))
    print(f"{bundle['method']}: {len(rows)} MRI rows -> {len(predictions)} subject/visit predictions")
    print(f"Predictions: {args.output.resolve()}\nProvenance: {report.resolve()}")
    if bundle["reference_grid"] is None and bundle["protocol"]["data"]["spatial_dims"] == 3:
        print("Grid validation is shape-only; export with --reference_image to validate the affine too.")


if __name__ == "__main__":
    main()
