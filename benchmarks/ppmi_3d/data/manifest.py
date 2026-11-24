"""Read the image manifest and build subject-level train/val/test splits.

Manifest = CSV with one row per image (a 2D slice or a 3D volume):

    subject_id, visit_id, mri_path, dat_path, <target column>, <aux column>, fold
    [, med_state [, med_state_source]]

``dat_path`` and the aux column may be empty (missing modality / missing score). Paths are relative
to ``protocol.data.root``. ``fold`` is the outer fold of the subject.

Medication state (STEWARD D-026), with ``state = protocol.data.target.med_state`` and
``col = protocol.data.target.med_state_column``:
    no ``col`` column        -> every row kept, med_state "unknown", stats["column_present"] = False
    empty cell               -> row kept, med_state "unknown", counted in stats["unknown"]
    cell == state (any case) -> row kept, med_state = state, counted in stats["kept"][state]
    any other value          -> row dropped, counted in stats["dropped"][value] (never silently)
One summary line is printed per call; the same numbers are returned through ``stats``.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from utils.config import config_hash, project_relative, resolve_path
from utils.io import read_csv, read_json


@dataclass(frozen=True)
class Row:
    subject_id: str
    visit_id: str
    mri_path: Path
    dat_path: Path | None
    target: float
    aux: float          # NaN when missing; raw units (not normalised)
    fold: int
    med_state: str = "unknown"   # D-026; last field with a default so Row(**r.__dict__) keeps working

    @property
    def unit(self) -> str:
        return f"{self.subject_id}|{self.visit_id}"


def _float_or_nan(text: str | None) -> float:
    text = (text or "").strip()
    return float(text) if text else math.nan


def approval_path(manifest: Path) -> Path:
    return manifest.with_suffix(".approval.json")


def cohort_hash(protocol: dict) -> str:
    data = dict(protocol["data"])
    for key in ("root", "manifest"):
        data[key] = project_relative(resolve_path(data[key]))
    return config_hash(dict(name=protocol["name"], data=data, preprocessing=protocol["preprocessing"]))


def load_approval(protocol: dict) -> dict:
    manifest = resolve_path(protocol["data"]["manifest"])
    path = approval_path(manifest)
    if not manifest.is_file() or not path.is_file():
        raise FileNotFoundError(f"QC-approved manifest and approval required: {manifest}, {path}")
    approval = read_json(path)
    qc_csv = resolve_path(approval["qc_csv"])
    if (approval["cohort_hash"] != cohort_hash(protocol)
            or approval["manifest_sha256"] != hashlib.sha256(manifest.read_bytes()).hexdigest()
            or approval["qc_sha256"] != hashlib.sha256(qc_csv.read_bytes()).hexdigest()):
        raise ValueError(f"QC approval differs from the protocol, manifest or reviewer CSV: {path}")
    return approval


def verify_approved_data(stats: dict, fingerprint: str) -> None:
    approval = stats.get("qc_approval")
    if approval is not None and approval["data_fingerprint"] != fingerprint:
        raise ValueError("The data fingerprint differs from the QC publication.")


def read_manifest(protocol: dict, stats: dict | None = None) -> list[Row]:
    """Rows of the manifest after the medication-state rule; ``stats`` (if given) is filled in place."""
    data = protocol["data"]
    root = resolve_path(data["root"])
    manifest = resolve_path(data["manifest"])
    target_col, aux_col = data["target"]["column"], data["aux"]["column"]
    state, state_col = data["target"].get("med_state"), data["target"].get("med_state_column")
    approval = load_approval(protocol) if "qc" in data else None
    raws = read_csv(manifest)
    header = list(raws[0]) if raws else []
    present = bool(state and state_col and state_col in header)
    info = dict(declared=state, column=state_col, column_present=present, rows_in_manifest=len(raws),
                kept={}, unknown=0, dropped={}, sources={})
    if approval is not None:
        info["qc_approval"] = approval
    rows = []
    for raw in raws:
        med_state = "unknown"
        if present:
            # D-026: explicit per-row state; unknown is kept but counted, other states are dropped and counted.
            value = (raw.get(state_col) or "").strip()
            if "med_state_source" in raw:   # counted for every row read, dropped rows included
                source = (raw.get("med_state_source") or "").strip() or "(empty)"
                info["sources"][source] = info["sources"].get(source, 0) + 1
            if not value:
                info["unknown"] += 1
            elif value.upper() == str(state).upper():
                med_state = str(state)
                info["kept"][med_state] = info["kept"].get(med_state, 0) + 1
            else:
                info["dropped"][value] = info["dropped"].get(value, 0) + 1
                continue
        else:
            info["unknown"] += 1          # no column: the state of every row is unknown (logged below)
        dat = (raw.get("dat_path") or "").strip()
        row = Row(subject_id=raw["subject_id"].strip(), visit_id=(raw.get("visit_id") or "BL").strip(),
                  mri_path=(root / raw["mri_path"].strip()).resolve(),
                  dat_path=(root / dat).resolve() if dat else None,
                  target=float(raw[target_col]), aux=_float_or_nan(raw.get(aux_col)),
                  fold=int(raw["fold"]), med_state=med_state)
        if not math.isfinite(row.target):
            raise ValueError(f"Non-finite target for {row.unit}")
        for path in (row.mri_path, row.dat_path):
            if path is not None and not path.is_file():
                raise FileNotFoundError(path)
        rows.append(row)
    kept = ", ".join(f"{k}={v}" for k, v in info["kept"].items()) or "none"
    print(f"med_state: column '{state_col}' {'present' if present else 'ABSENT (all rows unknown)'}; "
          f"kept {kept}, unknown={info['unknown']}, dropped={info['dropped']} (sources: {info['sources']})",
          flush=True)
    if stats is not None:
        stats.clear()
        stats.update(info)
    if not rows:
        raise ValueError(f"Empty manifest (after the med_state filter): {manifest}")
    # A subject must live in exactly one fold, and one eval unit must have one target.
    folds, targets = {}, {}
    for row in rows:
        if folds.setdefault(row.subject_id, row.fold) != row.fold:
            raise ValueError(f"Subject {row.subject_id} appears in several folds.")
        if targets.setdefault(row.unit, row.target) != row.target:
            raise ValueError(f"Eval unit {row.unit} has conflicting targets.")
    if approval is not None:
        expected = data["split"]["folds"]
        if (len(folds) != len(rows) or len(rows) != approval["qc_counts"]["PASS"]
                or set(folds.values()) != set(expected) or info["kept"] != {state: len(raws)}
                or any(r.dat_path is None or not r.visit_id or not 0 <= r.aux <= data["aux"]["max_value"]
                       for r in rows)
                or any(r.get("visual_qc") != "PASS" or r.get("med_state_source") != "case.json:clinical.PDSTATE"
                       or not (r.get("visit_id") or "").strip() for r in raws)):
            raise ValueError(f"QC-approved cohort is inconsistent with the OFF30 protocol: {manifest}")
    return rows


def data_fingerprint(protocol: dict, rows: list[Row]) -> str:
    """SHA-256 over the manifest and every referenced image file."""
    digest = hashlib.sha256(resolve_path(protocol["data"]["manifest"]).read_bytes())
    for path in sorted({p for r in rows for p in (r.mri_path, r.dat_path) if p is not None}):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class Split:
    fold: int
    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]

    def as_dict(self) -> dict:
        return {k: list(v) if isinstance(v, tuple) else v for k, v in asdict(self).items()}


def make_split(rows: list[Row], fold: int, protocol: dict) -> Split:
    """Outer test = subjects of ``fold``; validation = target-stratified subset of the rest.

    The validation draw depends only on (split_seed, fold), never on the training seed or
    the method, so every method sees identical train/val/test subjects.
    """
    cfg = protocol["data"]["split"]
    subject_target: dict[str, list[float]] = {}
    subject_fold = {}
    for row in rows:
        subject_target.setdefault(row.subject_id, []).append(row.target)
        subject_fold[row.subject_id] = row.fold
    test = sorted(s for s, f in subject_fold.items() if f == fold)
    pool = sorted(s for s, f in subject_fold.items() if f != fold)
    if not test or len(pool) < 2:
        raise ValueError(f"Fold {fold} has no test subjects or too few training subjects.")
    n_val = max(1, int(round(cfg["val_fraction"] * len(pool))))
    rng = np.random.default_rng([int(cfg["split_seed"]), int(fold)])
    ordered = sorted(pool, key=lambda s: (float(np.mean(subject_target[s])), s))
    # One subject from each of n_val contiguous target bins (stratified by target).
    val = sorted(str(rng.choice(group)) for group in np.array_split(np.array(ordered), n_val))
    train = sorted(set(pool) - set(val))
    return Split(fold=fold, train=tuple(train), val=tuple(val), test=tuple(test))


def rows_of(rows: list[Row], subjects) -> list[Row]:
    wanted = set(subjects)
    return [row for row in rows if row.subject_id in wanted]


@dataclass(frozen=True)
class TargetScaler:
    """z-score of the target, fitted on the training eval units of one fold only."""

    mean: float
    std: float

    @classmethod
    def fit(cls, rows: list[Row]) -> "TargetScaler":
        values = np.array(list({row.unit: row.target for row in rows}.values()), dtype=np.float64)
        return cls(float(values.mean()), max(float(values.std()), 1e-6))

    def normalize(self, y):
        return (y - self.mean) / self.std

    def restore(self, z):
        return z * self.std + self.mean
