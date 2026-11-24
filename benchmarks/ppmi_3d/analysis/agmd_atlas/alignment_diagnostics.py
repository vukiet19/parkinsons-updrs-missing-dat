"""Quantify the fixed atlas-to-image overlap on a QC-approved cohort.

Run from benchmarks/ppmi_3d: python -I analysis/agmd_atlas/alignment_diagnostics.py --protocol configs/protocol_ppmi.yaml
Uses the specified atlas without fitting or publishing a transform. Use the separate
agmd_atlas_shift.py sensitivity tool for training-fold-only shift estimation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation

# Also support direct invocation with Python isolated mode (-I).
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agmd_atlas_analysis import (LABEL_TXT, checked_pair, qc_inputs, qc_parser,
                                 read_label_table, write_csv)
from utils.config import resolve_path
from utils.io import write_json


def summarize(values):
    return dict(min=min(values), median=float(np.median(values)), max=max(values))


def main():
    parser = qc_parser(__doc__)
    parser.add_argument("--labels", default=str(LABEL_TXT))
    args = parser.parse_args()
    out, cases, _, atlas, ref, evidence = qc_inputs(args)
    labels = read_label_table(resolve_path(args.labels))
    ids = {name: label for label, name in labels.items()}
    name_pairs = {"L": ("Caudate_L", "Putamen_L"), "R": ("Caudate_R", "Putamen_R")}
    striatum = {side: np.isin(atlas, [ids[name] for name in names]) for side, names in name_pairs.items()}
    neighbors = {side: binary_dilation(mask, iterations=3) for side, mask in striatum.items()}
    label_counts = np.bincount(atlas.ravel())
    out.mkdir(parents=True, exist_ok=True)

    rows, outside_rows = [], []
    for i, case in enumerate(cases, 1):
        mri, dat = checked_pair(case, ref)
        support = mri != 0
        outside = (atlas > 0) & ~support
        counts = np.bincount(atlas[outside], minlength=len(label_counts))
        for label in np.argsort(-counts)[:8]:
            if counts[label]:
                outside_rows.append(dict(case_index=i, label_id=int(label), roi=labels.get(label, str(label)),
                                         voxels_outside=int(counts[label]), voxels_label=int(label_counts[label])))
        row = dict(case_index=i, labelled_voxels_outside_mri=int(outside.sum()),
                   frac_atlas_outside_mri=float(outside.sum() / (atlas > 0).sum()))
        for side, roi in striatum.items():
            region = neighbors[side] & support
            if not region.any():
                raise ValueError(f"Striatal neighborhood is outside MRI support for review case {i}, side {side}")
            local_daT = dat[region]
            hot = region & (dat >= np.quantile(local_daT, 0.90))
            row[f"hot_in_atlas_{side}"] = float(roi[hot].mean())
            row[f"neighborhood_in_atlas_{side}"] = float(roi[region].mean())
            row[f"striatal_voxels_outside_mri_{side}"] = int((roi & ~support).sum())
        rows.append(row)

    write_csv(out, "alignment_diagnostics.csv", rows)
    if outside_rows:
        write_csv(out, "alignment_mask_outside_labels.csv", outside_rows)
    metrics = {key: summarize([r[key] for r in rows]) for key in rows[0] if key != "case_index"}
    evidence.update(metrics=metrics, hot_definition="upper decile of DaT within a 3-step face-dilated "
                    "ipsilateral striatal ROI intersected with the MRI nonzero mask; ties retained",
                    scope="anatomical review support only; fixed atlas, no transform fit")
    write_json(out / "alignment_diagnostics.json", evidence)
    print(f"Fixed-atlas diagnostics on {len(rows)} cases -> {out / 'alignment_diagnostics.json'}")


if __name__ == "__main__":
    main()
