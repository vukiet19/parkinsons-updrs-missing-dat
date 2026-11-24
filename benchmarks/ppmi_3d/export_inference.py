"""Export one fold/seed's MRI-only student without training data or teacher dependencies.

python export_inference.py --run_dir runs_ppmi/off30/crad/fold0_seed0 --output deployment/crad_f0_s0.pt
Add --reference_image <preprocessed MRI.nii.gz> to embed the exact 3D affine.
AGMD embeds its graph geometry and uses its atlas grid automatically.
"""

import argparse
from pathlib import Path

import torch

from utils.mri_inference import export_run


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", type=Path, required=True, help="one completed fold*_seed* folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    parser.add_argument("--reference_image", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.output.suffix != ".pt":
        parser.error("--output must end in .pt")
    if args.output.resolve().is_relative_to(args.run_dir.resolve()):
        parser.error("Write deployment bundles outside the source run folder")
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; use --overwrite to replace a deployment bundle")
    bundle = export_run(args.run_dir, args.checkpoint, args.reference_image)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    grid = "shape + affine" if bundle["reference_grid"] is not None else "shape only (add --reference_image for affine validation)"
    print(f"Exported {bundle['method']} student ({bundle['provenance']}) -> {args.output.resolve()}")
    print(f"Input validation: {grid}. No DaT, teacher, training manifest or pretrained weights needed at inference.")


if __name__ == "__main__":
    main()
