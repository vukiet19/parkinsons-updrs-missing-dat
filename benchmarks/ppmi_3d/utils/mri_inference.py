"""MRI-only deployment, separate from the fingerprint-checked benchmark evaluator.

Bundles contain plain metadata and tensors, loaded with weights_only=True. The
original architecture modules implement the forward pass; no teacher or loss is
needed. A bundle targets ONE fold/seed, with that training fold's target scaler.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from data.dataset import load_image
from utils.io import read_csv, read_json

FORMAT = "pd_bench_mri_only_v1"
METHODS = ("mri_only", "kd_basic", "crad", "cregkd", "dftd", "agmd", "shaspec")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def inference_protocol(protocol: dict) -> dict:
    """Keep architecture/preprocessing, never the training dataset or weight paths."""
    result = {key: copy.deepcopy(protocol[key]) for key in ("backbone", "head", "preprocessing")}
    result["data"] = {"spatial_dims": protocol["data"]["spatial_dims"]}
    result["backbone"]["pretrained"] = "none"
    for key in ("weights_path", "weights_sha256"):
        result["backbone"].pop(key, None)
    return result


def build_model(method: str, protocol: dict, cfg: dict, layout_payload=None):
    """Build only modules used by the MRI prediction, with no pretrained download."""
    if method not in METHODS:
        raise ValueError(f"No MRI-only deployment role for {method!r}; supported: {METHODS}")
    p = inference_protocol(protocol)
    if method in ("mri_only", "kd_basic"):
        from baselines.anchors.model import MRIOnlyNet
        return MRIOnlyNet(p)
    if method == "cregkd":
        from baselines.cregkd.model import CRegStudent
        c = copy.deepcopy(cfg)
        c["kd"]["enabled"] = False
        return CRegStudent(p, c)
    if method == "crad":
        from baselines.crad.model import CRADStudent
        return CRADStudent(p, cfg)
    if method == "dftd":
        from baselines.dftd.model import DFTDStudentOnly
        return DFTDStudentOnly(p, cfg)
    if method == "shaspec":
        from baselines.shaspec.model import ShaSpecModel
        return ShaSpecModel(p, cfg, specific=("mri",), dom_cls=False)
    from baselines.agmd.graph import GraphLayout
    from baselines.agmd.model import AGMDStudent
    layout = None if layout_payload is None else GraphLayout.from_deployment_payload(layout_payload)
    if layout is not None:
        from baselines.agmd.graph import resnet_stage_sizes
        sizes = resnet_stage_sizes(p["preprocessing"]["input_size"])
        if layout.layers != cfg["graph"]["layers"] or layout.stage_sizes != [sizes[l] for l in layout.layers]:
            raise ValueError("Embedded AGMD layout differs from the configured feature grid")
    return AGMDStudent(p, cfg, graph_layout=layout)


def student_state(method: str, cfg: dict, state: dict) -> dict:
    """Discard explicitly known training-only keys, then use strict load_state_dict."""
    ignored = {"kd_basic": ("regressors.",), "cregkd": ("afr.",),
               "shaspec": ("specific.dat.", "dom_cls.")}.get(method, ())
    full_dftd = method == "dftd" and cfg["mode"] == "full"
    dftd_ignored = ("enc_b.", "w_a.", "w_b.", "S_c_a.", "S_c_b.", "S_s_a.", "S_s_b.",
                    "D_s_a.", "D_s_b.", "iid.t_fuse0.", "iid.t_layers.", "iid.t_fuse.",
                    "iid.t_head.", "iid.lit.")
    out = {}
    for key, value in state.items():
        if key.startswith(ignored) or full_dftd and key.startswith(dftd_ignored):
            continue
        if full_dftd and key.startswith("iid.student."):
            key = "student." + key[len("iid.student."):]
        if not isinstance(value, torch.Tensor) or value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Invalid checkpoint tensor: {key}")
        out[key] = value
    return out


def reference_grid(path: Path, protocol: dict) -> dict:
    """Optional explicit reference grid; no reference patient is retained in a bundle."""
    import nibabel as nib
    image = nib.load(str(path))
    if len(image.shape) != 3 or list(image.shape) != protocol["preprocessing"]["input_size"]:
        raise ValueError("Reference image must have the trained 3D input shape")
    affine = protocol["preprocessing"].get("affine")
    if affine is not None and not np.allclose(image.affine, affine, rtol=0, atol=1e-5):
        raise ValueError("Reference image affine differs from the protocol grid")
    return dict(shape=list(image.shape), affine=image.affine.tolist())


def export_run(run_dir: Path, checkpoint="best", reference_image: Path | None = None) -> dict:
    """Read ONE completed run and its student checkpoint, never its data manifest."""
    if checkpoint not in ("best", "last"):
        raise ValueError("checkpoint must be best or last")
    run_dir = run_dir.resolve()
    meta = read_json(run_dir / "run.json")
    method, p, cfg = meta["method"], meta["protocol"], meta["config"]
    if meta["status"] != "complete" or method not in METHODS:
        raise ValueError("Expected a completed run with a supported MRI-only student")
    if method in ("dftd", "shaspec"):
        from baselines import get_method
        stages = get_method(method)(p, cfg).stages
        trained = [s for s in stages if s in meta["stages"] and not meta["stages"][s].get("skipped")]
        if not trained:
            raise ValueError("No trained stage in run metadata")
        stage = trained[-1]
    else:
        stage = "student"
    if stage not in meta["stages"] or meta["stages"][stage].get("skipped"):
        raise ValueError("Run has no completed student stage")
    path = run_dir / f"{stage}{'_last' if checkpoint == 'last' else ''}.pt"
    from utils.repro import load_run_checkpoint
    ckpt = load_run_checkpoint(path, meta, stage, checkpoint)
    with torch.random.fork_rng(devices=[]):
        model = build_model(method, p, cfg)
    model.load_state_dict(student_state(method, cfg, ckpt["state_dict"]), strict=True)
    layout = model.graph.layout.deployment_payload() if method == "agmd" else None
    pre = p["preprocessing"]
    grid = dict(shape=pre["input_size"], affine=pre["affine"]) if "affine" in pre else None
    if reference_image is not None:
        if p["data"]["spatial_dims"] != 3:
            raise ValueError("--reference_image only applies to 3D images")
        grid = reference_grid(reference_image, p)
    elif method == "agmd" and cfg["graph"]["nodes"] in ("atlas", "atlas_area"):
        from utils.config import resolve_path
        grid = reference_grid(resolve_path(cfg["graph"]["atlas_path"]), p)
    clean_cfg = copy.deepcopy(cfg)
    if method == "agmd":
        clean_cfg["graph"]["atlas_path"] = None  # geometry is embedded; no external file is required
    bundle = dict(format=FORMAT, method=method, role="student", protocol=inference_protocol(p), config=clean_cfg,
                  scaler=copy.deepcopy(meta["scaler"]), layout=layout, reference_grid=grid,
                  state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  provenance=dict(fold=meta["fold"], seed=meta["seed"], stage=stage, checkpoint=checkpoint,
                                  checkpoint_sha256=sha256(path), source_hash=meta.get("source_hash"),
                                  data_fingerprint=meta.get("data_fingerprint"), protocol_name=p.get("name")))
    validate_bundle(bundle)
    return bundle


def validate_bundle(bundle: dict) -> None:
    if bundle.get("format") != FORMAT or bundle.get("method") not in METHODS or bundle.get("role") != "student":
        raise ValueError("Unsupported MRI-only bundle format/method/role")
    scaler = bundle["scaler"]
    if not math.isfinite(float(scaler["mean"])) or not math.isfinite(float(scaler["std"])) or float(scaler["std"]) <= 0:
        raise ValueError("Bundle must contain a finite, positive training-fold target scaler")
    if bundle["method"] == "agmd" and bundle.get("layout") is None:
        raise ValueError("AGMD bundle must embed its trained graph layout")
    p = bundle["protocol"]
    dims = int(p["data"]["spatial_dims"])
    if dims not in (2, 3) or len(p["preprocessing"]["input_size"]) != dims:
        raise ValueError("Invalid input dimensions in bundle")
    for key, value in bundle["state_dict"].items():
        if not isinstance(value, torch.Tensor) or value.is_floating_point() and not torch.isfinite(value).all():
            raise ValueError(f"Invalid bundle tensor: {key}")


def load_bundle(bundle: dict, device):
    validate_bundle(bundle)
    with torch.random.fork_rng(devices=[]):
        model = build_model(bundle["method"], bundle["protocol"], bundle["config"], bundle.get("layout"))
    model.load_state_dict(bundle["state_dict"], strict=True)
    return model.to(device).eval().requires_grad_(False)


def read_mri_manifest(path: Path, image_root: Path | None = None) -> list[dict]:
    """Only three columns are used; labels, folds, aux scores and DaT columns are ignored."""
    root = image_root.resolve() if image_root else path.resolve().parent
    rows, seen = [], set()
    for raw in read_csv(path):
        sid, visit, image = (raw.get("subject_id") or "").strip(), (raw.get("visit_id") or "BL").strip(), (raw.get("mri_path") or "").strip()
        if not sid or not visit or not image or "|" in sid or "|" in visit:
            raise ValueError("Each row needs subject_id, mri_path and a valid optional visit_id (default BL)")
        file = (root / image).resolve()
        if not file.is_file():
            raise FileNotFoundError(file)
        key = (sid, visit, file)
        if key in seen:
            raise ValueError(f"Duplicate MRI row: {sid}|{visit}, {file}")
        seen.add(key)
        rows.append(dict(subject_id=sid, visit_id=visit, mri_path=file))
    if not rows:
        raise ValueError("Empty MRI manifest")
    return rows


class MRIRows(Dataset):
    def __init__(self, rows, bundle):
        self.rows, self.bundle = rows, bundle

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        path = self.rows[index]["mri_path"]
        p = self.bundle["protocol"]
        pre, dims = p["preprocessing"], p["data"]["spatial_dims"]
        if dims == 3:
            import nibabel as nib
            image = nib.load(str(path))
            if len(image.shape) != 3 or list(image.shape) != pre["input_size"]:
                raise ValueError(f"{path}: expected the preprocessed 3D grid {pre['input_size']}, got {image.shape}; no automatic resampling")
            grid = self.bundle.get("reference_grid")
            if grid is not None and not np.allclose(image.affine, np.asarray(grid["affine"]), atol=1e-4, rtol=0):
                raise ValueError(f"{path}: affine differs from the exported reference grid; register/preprocess upstream")
        return index, load_image(path, int(pre["mri_channels"]), pre["input_size"], pre["intensity"], dims)


@torch.inference_mode()
def predict_mri(bundle: dict, rows: list[dict], device, batch_size=1) -> list[dict]:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    model = load_bundle(bundle, device)
    loader = DataLoader(MRIRows(rows, bundle), batch_size=batch_size, shuffle=False, num_workers=0)
    preds = defaultdict(list)
    for indices, images in loader:
        images = images.to(device)
        if bundle["method"] == "shaspec":
            available = torch.ones(len(images), dtype=torch.bool, device=device)
            out = model({"mri": images}, dict(mri=available, dat=~available))
        else:
            out = model(images)
        z = out["pred"] if isinstance(out, dict) else out
        if tuple(z.shape) != (len(images),) or not torch.isfinite(z).all():
            raise FloatingPointError("Non-finite or wrongly shaped MRI prediction")
        for index, value in zip(indices.tolist(), z.float().cpu().tolist()):
            row = rows[index]
            pred = float(value) * float(bundle["scaler"]["std"]) + float(bundle["scaler"]["mean"])
            if not math.isfinite(pred):
                raise FloatingPointError("Non-finite prediction after restoring target units")
            preds[(row["subject_id"], row["visit_id"])].append(pred)
    return [dict(subject_id=sid, visit_id=visit, unit=f"{sid}|{visit}", y_pred=float(np.mean(values)), n_rows=len(values))
            for (sid, visit), values in sorted(preds.items())]
