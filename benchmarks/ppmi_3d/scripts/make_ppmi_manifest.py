"""Preflight PPMI images and clinical records, and publish after documented visual QC.

    python scripts/make_ppmi_manifest.py --qc_preview
    python scripts/make_ppmi_manifest.py --qc_csv data/manifests/ppmi_off30_qc.csv
    python scripts/make_ppmi_manifest.py --qc_csv data/manifests/ppmi_off30_qc.csv --publish

The preview is a reviewer worksheet with image paths and pending decisions.
Only the reviewed CSV can supply PASS/FAIL decisions. The source is read-only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.manifest import Row, approval_path, cohort_hash, data_fingerprint, make_split  # noqa: E402
from scripts.assign_folds import assign_folds  # noqa: E402
from utils.config import config_hash, load_yaml, method_config_path, project_relative, resolve_path  # noqa: E402
from utils.io import write_csv, write_json  # noqa: E402
from utils.repro import file_sha256, source_hash  # noqa: E402

MRI_REL, DAT_REL = "paired_2mm/MRI_zscore.nii.gz", "paired_2mm/DaT_REF_ratio.nii.gz"
QC_IMAGES = ("MRI/QC_affine.png", "DaT/QC_multislice.png")
QC_COLUMNS = ("subject_id", "visual_qc", "reviewer", "date", "note")
# Supported medication-state keys in the legacy flat schema.
MED_STATE_JSON_KEYS = ("med_state", "medication_state", "PDSTATE")
# Keys that match the "looks state-related" scan but are known to be unrelated.
KNOWN_UNRELATED_KEYS = ("MRI_execution", "label_date_precision")
SUSPECT_PATTERN = re.compile(r"med|state|on_off", re.IGNORECASE)
VALID_STATES = ("OFF", "ON")
QC_VALUES = ("PASS", "FAIL")
COLUMNS = ("subject_id", "visit_id", "mri_path", "dat_path", "updrs3", "aux_score", "fold", "med_state",
           "med_state_source", "status", "visual_qc", "case_sha256")


def _read_table(path: Path, required: tuple[str, ...]) -> list[dict]:
    return _parse_table(Path(path).read_bytes(), path, required)


def _parse_table(contents: bytes, path: Path, required: tuple[str, ...]) -> list[dict]:
    reader = csv.DictReader(io.StringIO(contents.decode("utf-8-sig"), newline=""))
    header = reader.fieldnames or []
    missing = [c for c in required if c not in header]
    if missing or len(header) != len(set(header)):
        raise ValueError(f"{path}: missing columns {missing} or duplicate headers")
    rows = []
    for row in reader:
        if None in row or None in row.values():
            raise ValueError(f"{path}: malformed row {reader.line_num}")
        rows.append({k: v.strip() for k, v in row.items()})
    return rows


def _normalize_state(value, where: str) -> str:
    state = str(value).strip().upper()
    if state not in VALID_STATES:
        raise ValueError(f"{where}: medication state {value!r} is not one of {VALID_STATES}")
    return state


def _one_row_per_subject(rows: list[dict], subjects: list[str], what: str) -> dict[str, dict]:
    counts = Counter(r["subject_id"] for r in rows)
    duplicated = sorted(s for s, n in counts.items() if n > 1)
    unknown = sorted(set(counts) - set(subjects))
    missing = sorted(set(subjects) - set(counts))
    if duplicated or unknown or missing:
        raise ValueError(f"{what}: duplicated {duplicated}, not in the data folder {unknown}, missing {missing}")
    return {r["subject_id"]: r for r in rows}


def scan_cases(data_root: Path) -> dict[str, dict]:
    """subject_id -> case.json content, for every sub-folder that has a case.json (loose files skipped)."""
    cases = {}
    for folder in sorted(p for p in data_root.iterdir() if p.is_dir()):
        case_file = folder / "case.json"
        if case_file.is_file():
            case = json.loads(case_file.read_text(encoding="utf-8-sig"))
            if not isinstance(case, dict):
                raise ValueError(f"{case_file}: expected an object")
            for field in ("subject", "case_id"):
                if field in case and str(case[field]).strip() != folder.name:
                    raise ValueError(f"{folder.name}: {field} differs from the directory name")
            cases[folder.name] = case
    if not cases:
        raise FileNotFoundError(f"No <subject>/case.json under {data_root}")
    return cases


def _clinical_section(case: dict, subject: str) -> dict | None:
    if "clinical" not in case:
        return None
    clinical = case["clinical"]
    if not isinstance(clinical, dict):
        raise ValueError(f"{subject}: case.json:clinical must be an object")
    return clinical


def _score(value, where: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{where}: expected a numeric score, got {value!r}") from error
    if isinstance(value, bool) or not math.isfinite(score):
        raise ValueError(f"{where}: expected a finite numeric score, got {value!r}")
    return score


def _visit(value, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where}: expected a non-empty visit")
    return value.strip()


def _check_items(record: dict, total: float, where: str) -> None:
    items = record.get("items")
    if not isinstance(items, list) or len(items) != 33:
        raise ValueError(f"{where}.items: expected 33 entries")
    codes = set()
    scores = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("code"), str) or not item["code"].strip():
            raise ValueError(f"{where}.items: expected a non-empty code")
        code, score = item["code"].strip(), item.get("score")
        if code in codes or type(score) is not int or not 0 <= score <= 4:
            raise ValueError(f"{where}.items: duplicate code or invalid integer score for {code}")
        codes.add(code)
        scores.append(score)
    if sum(scores) != total:
        raise ValueError(f"{where}.items: sum does not match total")


def clinical_fields(case: dict, sid: str, aux_max: float, *, off30=False) -> tuple[str, float, float | str]:
    """Read the visit, UPDRS III and optional auxiliary score for one case."""
    clinical = _clinical_section(case, sid)
    if clinical is None:
        if off30:
            raise ValueError(f"{sid}: OFF30 requires the nested clinical schema")
        if "MDS_UPDRS_III" not in case or "label_visit" not in case:
            raise ValueError(f"{sid}: case.json lacks MDS_UPDRS_III or label_visit")
        visit = _visit(case["label_visit"], f"{sid}: label_visit")
        updrs = _score(case["MDS_UPDRS_III"], f"{sid}: MDS_UPDRS_III")
        aux = _score(case["MoCA"], f"{sid}: MoCA") if "MoCA" in case else ""
        if aux != "" and not 0 <= aux <= aux_max:
            raise ValueError(f"{sid}: MoCA must be in [0, {aux_max}]")
        return visit, updrs, aux

    visit = _visit(clinical.get("visit"), f"{sid}: clinical.visit")
    scores = []
    for name, upper in (("MDS_UPDRS_III", 132), ("MoCA", aux_max)):
        record = clinical.get(name)
        where = f"{sid}: clinical.{name}"
        if not isinstance(record, dict) or _visit(record.get("visit"), f"{where}.visit") != visit:
            raise ValueError(f"{where}: expected a record for clinical.visit {visit!r}")
        score = _score(record.get("total"), f"{where}.total")
        if not 0 <= score <= upper:
            raise ValueError(f"{where}.total: expected a score in [0, {upper}], got {score}")
        if name == "MDS_UPDRS_III" and (off30 or "items" in record):
            _check_items(record, score, where)
        scores.append(score)
    updrs, moca = scores
    for key in ("label_visit", "visit"):
        if key in case and _visit(case[key], f"{sid}: {key}") != visit:
            raise ValueError(f"{sid}: {key} and clinical.visit disagree")
    for key, expected in (("MDS_UPDRS_III", updrs), ("MoCA", moca)):
        if key in case and _score(case[key], f"{sid}: {key}") != expected:
            raise ValueError(f"{sid}: flat and nested {key} scores disagree")
    return visit, updrs, moca


def check_images(data_root: Path, subjects: list[str], input_size, expected_affine=None) -> np.ndarray:
    """Validate the exact 3D grid and finite scaled voxels across both modalities."""
    import nibabel as nib
    reference = None
    for sid in subjects:
        affines = []
        for rel in (MRI_REL, DAT_REL):
            path = data_root / sid / rel
            if not path.is_file():
                raise FileNotFoundError(f"{sid}: missing {path}")
            if not path.resolve().is_relative_to(data_root.resolve()):
                raise ValueError(f"{sid}: {rel} points outside the data root")
            image = nib.load(str(path))
            if tuple(image.shape) != tuple(input_size):
                raise ValueError(f"{sid}: {rel} has shape {image.shape}, expected {tuple(input_size)}")
            affine = image.affine
            if not np.isfinite(affine).all() or (expected_affine is not None and (
                    not np.allclose(affine, expected_affine, rtol=0, atol=1e-5)
                    or not np.allclose(image.header.get_zooms(), nib.affines.voxel_sizes(expected_affine),
                                       rtol=0, atol=1e-5))):
                raise ValueError(f"{sid}: {rel} affine or spacing differs from the target grid")
            if not np.isfinite(image.get_fdata(dtype=np.float32)).all():
                raise ValueError(f"{sid}: {rel} contains non-finite voxels")
            affines.append(affine)
        if not np.allclose(affines[0], affines[1], rtol=0, atol=1e-5):
            raise ValueError(f"{sid}: MRI and DaT affines differ:\n{affines[0]}\n{affines[1]}")
        if reference is None:
            reference = affines[0]
        elif not np.allclose(reference, affines[0], rtol=0, atol=1e-5):
            raise ValueError(f"{sid}: affine differs from {subjects[0]}:\n{affines[0]}\n{reference}")
    return reference


def resolve_med_states(cases: dict[str, dict], subjects: list[str], med_state_csv=None, med_state_all=None,
                       med_state_source=None, log=print, *, require_clinical=False) -> dict[str, tuple[str, str]]:
    """subject -> (state, source), following the D-026 rules in the module docstring."""
    if med_state_csv and (med_state_all or med_state_source):
        raise ValueError("Declare the medication state with --med_state_csv OR --med_state_all, not both.")
    if bool(med_state_all) != bool(med_state_source and med_state_source.strip()):
        raise ValueError("--med_state_all needs --med_state_source (and vice versa).")
    declared: dict[str, tuple[str, str]] = {}
    if med_state_csv:
        table = _one_row_per_subject(_read_table(med_state_csv, ("subject_id", "med_state", "source")),
                                     subjects, f"med_state CSV {med_state_csv}")
        for sid, row in table.items():
            if not row["source"]:
                raise ValueError(f"med_state CSV {med_state_csv}: empty source for {sid}")
            declared[sid] = (_normalize_state(row["med_state"], f"{med_state_csv}:{sid}"), row["source"])
    elif med_state_all:
        state = _normalize_state(med_state_all, "--med_state_all")
        declared = {sid: (state, med_state_source) for sid in subjects}
    # Unknown key names that look state-related are only reported, never used.
    suspects = sorted({k for case in cases.values() for k in case
                       if SUSPECT_PATTERN.search(k) and k not in MED_STATE_JSON_KEYS and k not in KNOWN_UNRELATED_KEYS})
    log(f"case.json keys looking state-related but not in MED_STATE_JSON_KEYS: {suspects or 'none'}")
    result, undeclared = {}, []
    for sid in subjects:
        case = cases[sid]
        found = [(k, case[k]) for k in MED_STATE_JSON_KEYS if k in case]
        clinical = _clinical_section(case, sid)
        if require_clinical and (clinical is None or "PDSTATE" not in clinical):
            raise ValueError(f"{sid}: missing case.json:clinical.PDSTATE")
        if clinical is not None and "PDSTATE" in clinical:
            found.append(("clinical.PDSTATE", clinical["PDSTATE"]))
        json_state = None
        for key, value in found:
            state = _normalize_state(value, f"{sid}/case.json:{key}")
            if json_state is not None and state != json_state[1]:
                raise ValueError(f"{sid}: case.json keys disagree on the medication state: {found}")
            json_state = (key, state)
        external = declared.get(sid)
        if external and json_state and json_state[1] != external[0]:
            raise ValueError(f"{sid}: case.json key '{json_state[0]}' = {json_state[1]} but the declared "
                             f"medication state is {external[0]} ({external[1]})")
        if external and not require_clinical:
            result[sid] = external
        elif json_state is not None:
            result[sid] = (json_state[1], f"case.json:{json_state[0]}")
        else:
            undeclared.append(sid)
    if undeclared:
        raise ValueError("Medication state not declared (use --med_state_csv or --med_state_all + "
                         f"--med_state_source) and absent from case.json for: {undeclared}")
    return result


def read_qc(qc_csv: Path, subjects: list[str], contents: bytes | None = None) -> dict[str, dict]:
    contents = qc_csv.read_bytes() if contents is None else contents
    table = _one_row_per_subject(_parse_table(contents, qc_csv, QC_COLUMNS),
                                 subjects, f"QC file {qc_csv}")
    for sid, row in table.items():
        row["visual_qc"] = row["visual_qc"].upper()
        if row["visual_qc"] not in QC_VALUES:
            raise ValueError(f"QC file {qc_csv}: {sid} has visual_qc {row['visual_qc']!r}, expected {QC_VALUES}")
        if not row["reviewer"] or not row["date"] or (row["visual_qc"] == "FAIL" and not row["note"]):
            raise ValueError(f"QC file {qc_csv}: {sid} requires reviewer, date and a reason for FAIL")
        try:
            date.fromisoformat(row["date"])
        except ValueError as error:
            raise ValueError(f"QC file {qc_csv}: {sid} requires an ISO review date") from error
    return table


def inspect_candidates(protocol, data_root, med_state_csv=None, med_state_all=None, med_state_source=None, log=print):
    off30 = "qc" in protocol["data"]
    policy = protocol["data"].get("qc", {})
    cases = scan_cases(data_root)
    subjects = sorted(cases)
    if off30 and len(subjects) != policy["candidates"]:
        raise ValueError(f"Expected {policy['candidates']} candidates, found {len(subjects)}")
    aux_max = float(protocol["data"].get("aux", {}).get("max_value", 30.0))
    fields = {sid: clinical_fields(cases[sid], sid, aux_max, off30=off30) for sid in subjects}
    med = resolve_med_states(cases, subjects, med_state_csv, med_state_all, med_state_source, log,
                             require_clinical=off30)
    if off30:
        for sid in subjects:
            if fields[sid][0] not in policy["visits"]:
                raise ValueError(f"{sid}: clinical.visit is outside the configured visits")
            for rel in QC_IMAGES:
                if not (data_root / sid / rel).is_file():
                    raise FileNotFoundError(f"{sid}: missing visual QC image {rel}")
    pre = protocol["preprocessing"]
    affine = check_images(data_root, subjects, pre["input_size"], pre.get("affine"))
    return cases, fields, med, affine


def build_manifest(protocol: dict, qc_csv, med_state_csv=None, med_state_all=None, med_state_source=None,
                   fold_seed: int | None = None, data_root: Path | None = None, log=print) -> tuple[list[dict], dict]:
    """Validate the candidates and return manifest rows and QC evidence without writing files."""
    root = Path(data_root) if data_root is not None else resolve_path(protocol["data"]["root"])
    candidates = inspect_candidates(protocol, root, med_state_csv, med_state_all, med_state_source, log)
    return _build_rows(protocol, root, candidates, Path(qc_csv), fold_seed)


def _build_rows(protocol, root, candidates, qc_csv, fold_seed):
    cases, fields, med, _ = candidates
    subjects = sorted(cases)
    qc_bytes = qc_csv.read_bytes()
    qc = read_qc(qc_csv, subjects, qc_bytes)
    excluded = [sid for sid in subjects if qc[sid]["visual_qc"] == "FAIL"]
    kept = [sid for sid in subjects if qc[sid]["visual_qc"] == "PASS"]
    split_cfg = protocol["data"]["split"]
    folds = split_cfg["folds"]
    k = len(folds)
    fold_seed = split_cfg["split_seed"] if fold_seed is None else fold_seed
    if len(kept) < k:
        raise ValueError(f"Only {len(kept)} QC PASS cases for {k} folds")
    if "qc" in protocol["data"]:
        if fold_seed != split_cfg["split_seed"]:
            raise ValueError("OFF30 fold assignment must use the protocol split_seed")
        for sid in kept:
            if med[sid][0] != "OFF":
                raise ValueError(f"{sid}: retained case requires clinical.PDSTATE OFF")
    targets = {sid: fields[sid][1] for sid in kept}
    fold_of = {sid: folds[i] for sid, i in assign_folds(targets, k, fold_seed).items()}
    rows = [dict(subject_id=sid, visit_id=fields[sid][0], mri_path=f"{sid}/{MRI_REL}",
                 dat_path=f"{sid}/{DAT_REL}", updrs3=targets[sid], aux_score=fields[sid][2], fold=fold_of[sid],
                 med_state=med[sid][0], med_state_source=med[sid][1], status=str(cases[sid].get("status", "")),
                 visual_qc=qc[sid]["visual_qc"], case_sha256=file_sha256(root / sid / "case.json")) for sid in kept]
    cohort = _manifest_rows(root, rows)
    splits = [make_split(cohort, fold, protocol) for fold in folds]
    summary = dict(n_subjects=len(subjects), n_after_qc=len(kept), excluded=excluded,
                   exclusion_reasons={sid: qc[sid]["note"] for sid in excluded},
                   qc_sha256=hashlib.sha256(qc_bytes).hexdigest(),
                   qc_counts={state: sum(r["visual_qc"] == state for r in qc.values()) for state in QC_VALUES},
                   visits_after_qc=dict(Counter(r["visit_id"] for r in rows)),
                   splits={str(s.fold): {part: len(getattr(s, part)) for part in ("train", "val", "test")}
                           for s in splits}, fold_seed=fold_seed)
    return rows, summary


def _manifest_rows(root, rows):
    return [Row(r["subject_id"], r["visit_id"], root / r["mri_path"], root / r["dat_path"], r["updrs3"],
                float(r["aux_score"]) if r["aux_score"] != "" else math.nan, r["fold"], r["med_state"]) for r in rows]


def write_manifest(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def check_resources(protocol):
    import nibabel as nib

    resources = {}
    bb = protocol["backbone"]
    if bb.get("pretrained") == "medicalnet_r18_23dataset":
        resources["medicalnet"] = (bb["weights_path"], bb["weights_sha256"])
    if "qc" in protocol["data"]:
        graph = load_yaml(method_config_path(protocol, "agmd"))["graph"]
        if graph["nodes"] in ("atlas", "atlas_area"):
            resources["atlas"] = (graph["atlas_path"], graph["atlas_sha256"])
    result = {}
    for resource, (name, expected) in resources.items():
        path = resolve_path(name)
        actual = file_sha256(path)
        if actual != expected.lower():
            raise ValueError(f"{resource}: SHA-256 mismatch for {path}")
        if resource == "atlas":
            img = nib.load(str(path))
            pre = protocol["preprocessing"]
            if list(img.shape) != pre["input_size"] or not np.allclose(img.affine, pre["affine"], rtol=0, atol=1e-5):
                raise ValueError(f"Atlas grid differs from the protocol grid: {path}")
        result[resource] = dict(path=project_relative(path), sha256=actual)
    return result


def publish_manifest(protocol, root, out, qc_csv, rows, evidence):
    if file_sha256(qc_csv) != evidence["qc_sha256"]:
        raise ValueError("The reviewer CSV changed during validation")
    approval = approval_path(out)
    if any(path.resolve().is_relative_to(root.resolve()) for path in (out, approval)):
        raise ValueError("The manifest and approval must be outside the read-only source")
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_suffix(".publishing")
    with lock.open("x"):
        pass
    try:
        with TemporaryDirectory(dir=out.parent) as tmp:
            stage = Path(tmp) / out.name
            write_manifest(stage, rows)
            staged_protocol = {**protocol, "data": {**protocol["data"], "manifest": str(stage)}}
            cohort_protocol = {**protocol, "data": {**protocol["data"], "manifest": project_relative(out)}}
            fingerprint = data_fingerprint(staged_protocol, _manifest_rows(root, rows))
            record = dict(evidence, protocol=protocol["name"], cohort_hash=cohort_hash(cohort_protocol),
                          qc_csv=project_relative(qc_csv), manifest_sha256=file_sha256(stage),
                          data_fingerprint=fingerprint)
            if approval.exists():
                previous = json.loads(approval.read_text(encoding="utf-8"))
                identity = ("cohort_hash", "manifest_sha256", "qc_sha256", "data_fingerprint")
                if any(previous[key] != record[key] for key in identity):
                    raise ValueError(f"Published cohort differs: {approval}. Use a new protocol version.")
                if file_sha256(resolve_path(previous["qc_csv"])) != record["qc_sha256"]:
                    raise ValueError(f"Original reviewer evidence differs: {previous['qc_csv']}")
                record = previous
            if out.exists() and file_sha256(out) != record["manifest_sha256"]:
                raise ValueError(f"Published manifest differs: {out}. Use a new protocol version.")
            if not approval.exists():
                write_json(approval_path(stage), record)
                approval_path(stage).replace(approval)
            if not out.exists():
                stage.replace(out)
    finally:
        lock.unlink()
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--protocol", default="configs/protocol_ppmi.yaml")
    p.add_argument("--med_state_csv")
    p.add_argument("--med_state_all")
    p.add_argument("--med_state_source")
    p.add_argument("--qc_csv", help="reviewed PASS/FAIL CSV")
    p.add_argument("--qc_preview", nargs="?", const="", help="write a pending worksheet; optional destination")
    p.add_argument("--publish", action="store_true", help="publish the approved manifest")
    p.add_argument("--fold_seed", type=int)
    p.add_argument("--out", help="manifest destination for --publish (default: protocol data.manifest)")
    p.add_argument("--report", help="optional aggregate preflight JSON")
    args = p.parse_args()
    if (args.publish and not args.qc_csv) or (args.out and not args.publish):
        p.error("--publish requires --qc_csv and --out requires --publish")
    protocol = load_yaml(resolve_path(args.protocol))
    data = protocol["data"]
    root = resolve_path(data["root"]).resolve()
    canonical = resolve_path(data["manifest"])
    out = resolve_path(args.out) if args.out else canonical
    qc_csv = resolve_path(args.qc_csv) if args.qc_csv else None
    preview = (resolve_path(args.qc_preview or data["qc"]["template"]) if args.qc_preview is not None else None)
    report = resolve_path(args.report) if args.report else None
    protected = {canonical.resolve(), approval_path(canonical).resolve(), out.resolve(), approval_path(out).resolve()}
    if qc_csv:
        if qc_csv.resolve() in protected:
            p.error("Reviewer CSV must be separate from the manifest and its approval")
        protected.add(qc_csv.resolve())
    for path in (preview, report):
        if path:
            if path.resolve() in protected:
                p.error("Preview and report must have separate paths")
            protected.add(path.resolve())
    if any(path and path.resolve().is_relative_to(root) for path in (out, approval_path(out), preview, report)):
        p.error("Output paths must be outside the read-only source")
    candidates = inspect_candidates(protocol, root, resolve_path(args.med_state_csv) if args.med_state_csv else None,
                                    args.med_state_all, args.med_state_source)
    cases, fields, med, affine = candidates
    source_files = [root / sid / rel for sid in sorted(cases) for rel in ("case.json", MRI_REL, DAT_REL)]
    source_fingerprint = config_hash({path.relative_to(root).as_posix(): file_sha256(path) for path in source_files})
    evidence = dict(protocol=protocol["name"], config_hash=config_hash(protocol), source_hash=source_hash(),
                    source_fingerprint=source_fingerprint, n_subjects=len(cases), grid=affine.tolist(),
                    input_size=protocol["preprocessing"]["input_size"], finite_images=2 * len(cases),
                    visits=dict(Counter(values[0] for values in fields.values())),
                    med_state=dict(Counter(state for state, _ in med.values())),
                    source_visual_qc=dict(Counter(str(case.get("visual_QC", "unspecified")) for case in cases.values())),
                    resources=check_resources(protocol), qc_counts=dict(PASS=0, FAIL=0, pending=len(cases)),
                    status="awaiting_review")
    if preview:
        if preview.exists():
            raise FileExistsError(f"Review worksheet already exists: {preview}")
        write_csv(preview, [dict(subject_id=sid, visual_qc="pending", reviewer="", date="", note="",
                                 mri_qc=str(root / sid / QC_IMAGES[0]), dat_qc=str(root / sid / QC_IMAGES[1]))
                            for sid in sorted(cases)])
        print(f"Reviewer worksheet: {preview}")
    if qc_csv:
        rows, checked = _build_rows(protocol, root, candidates, qc_csv, args.fold_seed)
        evidence.update({k: v for k, v in checked.items() if k not in ("excluded", "exclusion_reasons")})
        evidence.update(status="review_validated")
        if args.publish:
            evidence = publish_manifest(protocol, root, out, qc_csv, rows, dict(evidence, status="published"))
            print(f"QC-approved manifest: {out}")
    if report:
        write_json(report, evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
