"""Estimate an exploratory AAL3v1 shift from each approved OFF30 fold's training subjects.

    python scripts/agmd_atlas_shift.py --protocol configs/protocol_ppmi.yaml

For each outer fold, maximize the mean DaT in caudate and putamen using the training split.
The outputs go to the protocol's atlas_shift directory. A boundary optimum stops publication.
--mode shared also requires every training fold to agree.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.manifest import data_fingerprint, make_split, read_manifest, rows_of, verify_approved_data  # noqa: E402
from utils.config import config_hash, load_yaml, method_config_path, project_relative, resolve_path  # noqa: E402
from utils.io import write_json  # noqa: E402
from utils.repro import file_sha256  # noqa: E402

STRIATUM = (75, 76, 77, 78)                        # Caudate_L/R, Putamen_L/R (AAL3v1 label table, P1a)


def shift_labels(labels: np.ndarray, d) -> np.ndarray:
    """Translate a label volume by integer voxels d; vacated voxels become 0 (no wrap-around)."""
    out = np.zeros_like(labels)
    src = tuple(slice(max(0, -s), labels.shape[a] - max(0, s)) for a, s in enumerate(d))
    dst = tuple(slice(max(0, s), labels.shape[a] - max(0, -s)) for a, s in enumerate(d))
    out[dst] = labels[src]
    return out


def best_shift(mean_dat: np.ndarray, mask: np.ndarray, radius: int):
    vals = {d: float(mean_dat[shift_labels(mask, d)].mean())
            for d in itertools.product(range(-radius, radius + 1), repeat=3)}
    best = max(vals, key=vals.get)
    return best, vals[best], vals[(0, 0, 0)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", default="configs/protocol_ppmi.yaml")
    p.add_argument("--atlas", help="atlas override; requires --atlas_sha256")
    p.add_argument("--atlas_sha256")
    p.add_argument("--radius", type=int, default=4)
    p.add_argument("--out_csv", help="default: protocol output_root/atlas_shift/folds.csv")
    p.add_argument("--mode", choices=("per_fold", "shared"), default="per_fold")
    args = p.parse_args()
    if args.radius < 1 or bool(args.atlas) != bool(args.atlas_sha256):
        p.error("Provide a positive radius and pair --atlas with --atlas_sha256")
    protocol = load_yaml(resolve_path(args.protocol))
    config = method_config_path(protocol, "agmd")
    graph = load_yaml(config)["graph"]
    atlas_path = resolve_path(args.atlas or graph["atlas_path"])
    parent_hash = file_sha256(atlas_path)
    if parent_hash != (args.atlas_sha256 or graph["atlas_sha256"]):
        raise ValueError(f"Atlas SHA-256 mismatch: {atlas_path}")
    stats = {}
    rows = read_manifest(protocol, stats)
    fingerprint = data_fingerprint(protocol, rows)
    verify_approved_data(stats, fingerprint)
    out_csv = (resolve_path(args.out_csv) if args.out_csv else
               resolve_path(protocol["output_root"]) / "atlas_shift" / "folds.csv")
    if out_csv.resolve().is_relative_to(resolve_path(protocol["data"]["root"]).resolve()):
        p.error("Output must be outside the read-only source")
    atlas_img = nib.load(str(atlas_path))
    atlas = np.rint(np.asarray(atlas_img.dataobj)).astype(np.int16)
    stri = np.isin(atlas, STRIATUM)
    affine = atlas_img.affine
    cache = {}

    def dat(path):
        if path not in cache:
            img = nib.load(str(path))
            if img.shape != atlas.shape or not np.allclose(img.affine, atlas_img.affine, rtol=0, atol=1e-5):
                raise SystemExit(f"STOP: {path} is not on the atlas grid")
            voxels = img.get_fdata(dtype=np.float32)
            if not np.isfinite(voxels).all():
                raise ValueError(f"Non-finite DaT values: {path}")
            cache[path] = voxels
        return cache[path]

    table = []
    folds = {}
    for fold in protocol["data"]["split"]["folds"]:
        split = make_split(rows, fold, protocol)
        train = rows_of(rows, split.train)
        if not train or any(r.dat_path is None for r in train):
            raise ValueError(f"Fold {fold} needs paired training cases")
        mean = np.mean([dat(r.dat_path) for r in train], axis=0)
        d, v_best, v0 = best_shift(mean, stri, args.radius)
        folds[fold] = d
        table.append(dict(fold=fold, n_train_subjects=len(split.train), data_fingerprint=fingerprint,
                          shift_vox=" ".join(map(str, d)),
                            shift_mm=" ".join(f"{s:+.6g}" for s in affine[:3, :3] @ d),
                          on_boundary=max(abs(s) for s in d) == args.radius,
                          striatal_dat_at_0=round(v0, 4), striatal_dat_at_best=round(v_best, 4)))
        print(table[-1], flush=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    print(f"table -> {out_csv}")
    if any(row["on_boundary"] for row in table) or (args.mode == "shared" and len(set(folds.values())) != 1):
        print("STOP (D-034): a shift reached the search boundary or folds disagree in shared mode.")
        return 2

    files = {}
    for d in sorted(set(folds.values())):
        out = out_csv.parent / f"aal3v1_ppmi_shift_{d[0]}_{d[1]}_{d[2]}.nii.gz"
        nib.save(nib.Nifti1Image(shift_labels(atlas, d), atlas_img.affine), str(out))
        transform = np.eye(4)
        transform[:3, 3] = affine[:3, :3] @ d
        provenance = dict(method="integer voxel translation with zero fill", shift_vox=list(d),
                          source_atlas_sha256=parent_hash, world_transform=transform.tolist())
        files[d] = dict(atlas_path=project_relative(out), atlas_sha256=file_sha256(out),
                        transform_sha256=config_hash(provenance))
        write_json(out.with_name(out.name.removesuffix(".nii.gz") + ".json"),
                   dict(**provenance, **files[d], data_fingerprint=fingerprint,
                        source_folds=[fold for fold in folds if folds[fold] == d]))
    mapping = [dict(row, **files[folds[row["fold"]]]) for row in table]
    map_csv = out_csv.with_name(f"{out_csv.stem}_atlas_map.csv")
    with map_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(mapping[0]))
        writer.writeheader()
        writer.writerows(mapping)
    print(f"fold -> atlas map: {map_csv}")
    for row in mapping:
        print(f'python train.py --method agmd --config "{project_relative(config)}" --protocol "{args.protocol}" '
              f'--tag sens_atlas_shift --fold {row["fold"]} --seed all '
              f'--set "graph.atlas_path={row["atlas_path"]}" graph.atlas_sha256={row["atlas_sha256"]}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
