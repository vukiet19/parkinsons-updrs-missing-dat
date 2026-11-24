"""Build data/manifests/ntua.csv from the existing NTUA subject table (one row per MRI slice).

Reads data/subjects.csv at the repository root (15 subjects in the original KD0/KD1 splits).
Output image paths are relative to the protocol's dataset_224_preprocessed directory.

    python scripts/make_ntua_manifest.py
"""

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parents[1] / "data" / "subjects.csv"
OUTPUT = ROOT / "data" / "manifests" / "ntua.csv"
PREFIX = "dataset_224_preprocessed/"


def main():
    with SOURCE.open(encoding="utf-8-sig", newline="") as handle:
        subjects = list(csv.DictReader(handle))
    rows = []
    for s in subjects:
        dat = s["dat_path"]
        if not dat.startswith(PREFIX):
            sys.exit(f"Unexpected DaT path: {dat}")
        for mri in s["mri_paths"].split(";"):
            if not mri.startswith(PREFIX):
                sys.exit(f"Unexpected MRI path: {mri}")
            rows.append(dict(subject_id=s["subject_id"], visit_id="BL", mri_path=mri[len(PREFIX):],
                             dat_path=dat[len(PREFIX):], updrs3=s["updrs"], aux_score=s["mmse"],
                             fold=s["outer_fold"]))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} rows, {len(subjects)} subjects -> {OUTPUT}")


if __name__ == "__main__":
    main()
