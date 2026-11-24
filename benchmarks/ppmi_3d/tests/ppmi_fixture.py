"""Synthetic paired OFF30 data for integration tests."""

from pathlib import Path

import nibabel as nib
import numpy as np
import yaml
from PIL import Image

from scripts import make_ppmi_manifest as mpm
from utils.config import load_yaml, resolve_path
from utils.io import write_csv, write_json


def write_fixture(folder: Path, protocol_path="configs/protocol_ppmi.yaml", *, size=None, n=30):
    protocol = load_yaml(resolve_path(protocol_path))
    protocol["data"].update(root=str(folder / "images"), manifest=str(folder / "manifest.csv"), cache_in_memory=False)
    protocol["data"]["qc"]["candidates"] = n
    pre = protocol["preprocessing"]
    if size is not None:
        pre["input_size"] = list(size)
    size = tuple(pre["input_size"])
    affine = np.asarray(pre["affine"], dtype=np.float64)
    root = Path(protocol["data"]["root"])
    root.mkdir(parents=True)
    x, y, z = np.ogrid[:size[0], :size[1], :size[2]]
    anatomy = ((2 * x + y + z) % 17).astype(np.float32)
    dat = (1 + ((x + y + z) % 8).astype(np.float32) / 8)
    ref = (dat == 1).astype(np.uint8)
    qc_rows = []
    for i in range(n):
        sid = f"FIX{i:02d}"
        case = root / sid
        (case / "paired_2mm").mkdir(parents=True)
        for rel, values in ((mpm.MRI_REL, (anatomy - 8 + i % 7) / 8), (mpm.DAT_REL, dat)):
            nib.save(nib.Nifti1Image(values, affine), str(case / rel))
        for rel in mpm.QC_IMAGES:
            path = case / rel
            path.parent.mkdir(exist_ok=True)
            Image.new("RGB", (8, 8), (i, 0, 0)).save(path)
        nib.save(nib.Nifti1Image(ref, affine), str(case / "DaT/04_occipital_pole_reference.nii.gz"))
        visit = protocol["data"]["qc"]["visits"][i % len(protocol["data"]["qc"]["visits"])]
        items = [dict(code=f"item{j:02d}", score=1 if j < i else 0) for j in range(33)]
        clinical = dict(visit=visit, PDSTATE="OFF", MoCA=dict(visit=visit, total=i % 31),
                        MDS_UPDRS_III=dict(visit=visit, total=sum(item["score"] for item in items), items=items))
        write_json(case / "case.json", dict(subject=sid, case_id=sid, clinical=clinical, visual_QC="pending"))
        qc_rows.append(dict(subject_id=sid, visual_qc="PASS", reviewer="synthetic fixture", date="2026-11-21", note=""))
    qc_csv = folder / "qc.csv"
    write_csv(qc_csv, qc_rows)
    cfg = folder / "protocol.yaml"
    cfg.write_text(yaml.safe_dump(protocol, sort_keys=False), encoding="utf-8")
    return cfg, protocol, qc_csv


def approved_fixture(folder: Path, protocol_path="configs/protocol_ppmi.yaml"):
    cfg, protocol, qc_csv = write_fixture(folder, protocol_path)
    rows, evidence = mpm.build_manifest(protocol, qc_csv, log=lambda *_: None)
    evidence.pop("exclusion_reasons")
    evidence.pop("excluded")
    record = mpm.publish_manifest(protocol, resolve_path(protocol["data"]["root"]),
                                  resolve_path(protocol["data"]["manifest"]), qc_csv, rows, evidence)
    return cfg, protocol, qc_csv, record
