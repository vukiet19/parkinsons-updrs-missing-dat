"""Review the configured AAL3v1 against an approved PPMI manifest, without fitting geometry.

Run from benchmarks/ppmi_3d: python -I analysis/agmd_atlas/agmd_atlas_analysis.py --protocol configs/protocol_ppmi.yaml
Outputs default to the protocol's agmd/atlas_qc directory. P1a stage analyses are optional.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import nibabel as nib
import nibabel.processing
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from baselines.agmd.graph import (GraphLayout, _knn_k, atlas_adjacency, private_label_sets, resample_labels,  # noqa: E402
                                  resnet_stage_sizes)
from models.backbones.resnet import ResNetEncoder  # noqa: E402
from data.manifest import data_fingerprint, read_manifest, verify_approved_data  # noqa: E402
from utils.config import config_hash, load_yaml, resolve_path  # noqa: E402
from utils.io import read_json, write_json  # noqa: E402
from utils.repro import file_sha256  # noqa: E402

HERE = Path(__file__).resolve().parent
ATLAS_1MM = HERE / "download" / "AAL3" / "AAL3v1_1mm.nii.gz"
LABEL_TXT = HERE / "download" / "AAL3" / "AAL3v1_1mm.nii.txt"
KEY_NAMES = ["Caudate_L", "Caudate_R", "Putamen_L", "Putamen_R", "Pallidum_L", "Pallidum_R",
             "SN_pc_L", "SN_pc_R", "SN_pr_L", "SN_pr_R"]
SHORT = {"Caudate_L": "CauL", "Caudate_R": "CauR", "Putamen_L": "PutL", "Putamen_R": "PutR", "Pallidum_L": "PalL",
         "Pallidum_R": "PalR", "SN_pc_L": "SNpcL", "SN_pc_R": "SNpcR", "SN_pr_L": "SNprL", "SN_pr_R": "SNprR"}
ORIG_MM = (12.0, 24.0)                  # nominal resolution of the official graph layers (14x17x14, 7x8x7)


def write_csv(out: Path, name: str, rows: list[dict]) -> None:
    with (out / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out / name} ({len(rows)} rows)")


def qc_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--protocol", default="configs/protocol_ppmi.yaml")
    parser.add_argument("--config", default="configs/baselines/agmd_ppmi.yaml")
    parser.add_argument("--atlas", help="path to the configured atlas (SHA-256 must match the config)")
    parser.add_argument("--out", help="output directory; default: protocol output_root/agmd/atlas_qc")
    return parser


def qc_inputs(args):
    protocol = load_yaml(resolve_path(args.protocol))
    config = load_yaml(resolve_path(args.config))
    stats = {}
    rows = read_manifest(protocol, stats)
    fingerprint = data_fingerprint(protocol, rows)
    verify_approved_data(stats, fingerprint)
    if "qc_approval" not in stats or not rows or any(row.dat_path is None for row in rows):
        raise ValueError("Atlas QC requires a published QC-approved paired manifest")
    graph = config["graph"]
    atlas_path = resolve_path(args.atlas or graph["atlas_path"])
    atlas_hash = file_sha256(atlas_path)
    if atlas_hash != graph["atlas_sha256"]:
        raise ValueError("Atlas SHA-256 differs from the AGMD config")
    ref = nib.load(str(rows[0].mri_path))
    image = nib.load(str(atlas_path))
    expected_shape = tuple(protocol["preprocessing"]["input_size"])
    expected_affine = np.asarray(protocol["preprocessing"]["affine"])
    for candidate in (ref, image):
        if candidate.shape != expected_shape or not np.allclose(candidate.affine, expected_affine, rtol=0, atol=1e-5):
            raise ValueError("Reference MRI or atlas grid differs from the protocol")
    atlas = np.asarray(image.dataobj)
    if not np.all(np.isfinite(atlas) & (atlas >= 0) & (atlas == np.rint(atlas))):
        raise ValueError("Atlas must contain finite, nonnegative integer labels")
    labels = set(np.unique(atlas))
    if not set(graph["atlas_key_rois"]) <= labels or 0 not in labels:
        raise ValueError("Atlas is missing configured key ROIs or background")
    out = resolve_path(args.out or Path(protocol["output_root"]) / "agmd" / "atlas_qc").resolve()
    if out.is_relative_to(resolve_path(protocol["data"]["root"]).resolve()):
        raise ValueError("QC outputs must be outside the read-only image source")
    approval = stats["qc_approval"]
    evidence = dict(protocol=protocol["name"], protocol_hash=config_hash(protocol), config_hash=config_hash(config),
                    protocol_sha256=file_sha256(resolve_path(args.protocol)), config_sha256=file_sha256(resolve_path(args.config)),
                    manifest_sha256=approval["manifest_sha256"], qc_sha256=approval["qc_sha256"],
                    data_fingerprint=fingerprint, atlas_path=str(atlas_path), atlas_sha256=atlas_hash,
                    shape=list(ref.shape), affine=ref.affine.tolist(), axis_codes=list(nib.aff2axcodes(ref.affine)),
                    n_cases=len(rows), n_labels_without_background=len(labels) - 1)
    return out, rows, config, atlas.astype(np.int16), ref, evidence


def checked_pair(row, ref):
    volumes = []
    for path in (row.mri_path, row.dat_path):
        image = nib.load(str(path))
        if image.shape != ref.shape or not np.allclose(image.affine, ref.affine, rtol=0, atol=1e-5):
            raise ValueError("MRI or DaT grid differs from the atlas reference")
        volume = image.get_fdata(dtype=np.float32)
        if not np.isfinite(volume).all():
            raise ValueError("MRI or DaT contains non-finite values")
        volumes.append(volume)
    if not np.any(volumes[0]):
        raise ValueError("MRI nonzero mask is empty")
    return volumes


def read_label_table(path: Path) -> dict[int, str]:
    table = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if parts:
            table[int(parts[0])] = parts[1]
    return table


def atlas_provenance(out, atlas, ref, config, source_path, label_path):
    table = read_label_table(label_path)
    keys = {name: label for label, name in table.items() if name in KEY_NAMES}
    if [keys[name] for name in KEY_NAMES] != config["graph"]["atlas_key_rois"]:
        raise ValueError("The ten configured ROI IDs disagree with the AAL3v1 label names")
    source = nib.load(str(source_path))
    source_labels = np.asarray(source.dataobj)
    rebuilt = nibabel.processing.resample_from_to(source, (ref.shape, ref.affine), order=0,
                                                 mode="constant", cval=0)
    if not np.array_equal(np.asarray(rebuilt.dataobj), atlas):
        raise ValueError("Configured atlas differs from header-based nearest resampling of AAL3v1")
    transform = dict(method="header-based nearest neighbor, constant zero fill", order=0,
                     world_transform=np.eye(4).tolist(),
                     target_voxel_to_source_voxel=np.linalg.solve(source.affine, ref.affine).tolist(),
                     source_affine=source.affine.tolist(), target_affine=ref.affine.tolist())
    voxel_volume = abs(np.linalg.det(ref.affine[:3, :3]))
    volumes = [dict(name=name, id=keys[name], source_voxels=int((source_labels == keys[name]).sum()),
                    target_voxels=int((atlas == keys[name]).sum()),
                    target_mm3=int((atlas == keys[name]).sum()) * voxel_volume) for name in KEY_NAMES]
    write_csv(out, "key_roi_volumes.csv", volumes)
    record = dict(source=str(source_path), source_sha256=file_sha256(source_path), labels=str(label_path),
                  labels_sha256=file_sha256(label_path), exact_voxel_reproduction=True, transform=transform,
                  transform_sha256=config_hash(transform))
    return keys, source_labels, record


def preprocessing_provenance(rows):
    evidence = []
    for i, row in enumerate(rows, 1):
        case = row.mri_path.parent.parent
        mri_run = case / "MRI" / "run.json"
        dat_run = case / "DaT" / "run.json"
        mr, dr = read_json(mri_run), read_json(dat_run)
        case_record = read_json(case / "case.json")
        template = Path(mr["template"]).name
        mri_pipeline = case_record["pipelines"]["MRI"]
        dat_registration = dr["registration_config"]
        if "Camacho" not in mri_pipeline or "PD25" not in template or dr["registration_mode"] != "identity":
            raise ValueError("Preprocessing records differ from the expected Camacho PD25 / DaT identity branches")
        evidence.append(dict(case_index=i, mri_template=template, mri_run_sha256=file_sha256(mri_run),
                             dat_run_sha256=file_sha256(dat_run),
                             mri_affine_sha256=file_sha256(case / "MRI" / "affine_0GenericAffine.mat"),
                             pd25_crop_sha256=file_sha256(case / "MRI" / "PD25_brain_crop.nii.gz"),
                             dat_registration=dat_registration,
                             dat_transform_sha256=config_hash(dict(world_transform=np.eye(4).tolist(),
                                                                    registration_config=dat_registration))))
    return evidence


def step_alignment(out, atlas, ref, keys, cases):
    labelled = atlas > 0
    roi_masks = {name: atlas == keys[name] for name in KEY_NAMES}
    rows = []
    summed = np.zeros((2, *ref.shape), dtype=np.float64)
    overlays = out / "overlays"
    overlays.mkdir(exist_ok=True)
    for i, case in enumerate(cases, 1):
        mri, dat = checked_pair(case, ref)
        mask = mri != 0
        brain = float(dat[mask].mean())
        if brain <= 0:
            raise ValueError("Mean DaT inside the MRI nonzero mask must be positive")
        row = dict(case_index=i, atlas_in_mri_mask=float(mask[labelled].mean()), dat_in_mri_mask=brain)
        for name, roi in roi_masks.items():
            row[f"mri_mask_{SHORT[name]}"] = float(mask[roi].mean())
            row[f"dat_{SHORT[name]}"] = float(dat[roi].mean())
        for roi, a, b in (("caudate", "Caudate_L", "Caudate_R"), ("putamen", "Putamen_L", "Putamen_R")):
            row[f"{roi}_over_brain"] = float(dat[roi_masks[a] | roi_masks[b]].mean()) / brain
        rows.append(row)
        summed[0] += mri
        summed[1] += dat
        qc_figure(mri, dat, atlas, keys, f"Case {i:02d}", overlays / f"case_{i:02d}.png")
    write_csv(out, "alignment_per_case.csv", rows)
    qc_figure(*(summed / len(rows)), atlas, keys, f"Cohort mean (n={len(rows)})", out / "qc_alignment.png")
    summary = {}
    for field in rows[0]:
        if field == "case_index":
            continue
        values = [r[field] for r in rows]
        summary[field] = dict(min=min(values), median=float(np.median(values)), max=max(values),
                              mean=float(np.mean(values)))
    summary["n_both_ratios_at_least_1_5"] = sum(min(r["caudate_over_brain"], r["putamen_over_brain"]) >= 1.5 for r in rows)
    summary["n_either_ratio_at_least_1_5"] = sum(max(r["caudate_over_brain"], r["putamen_over_brain"]) >= 1.5 for r in rows)
    index = ["# OFF30 atlas review", "", "Case numbers are one-based positions in the approved manifest.",
             "Match anatomical boundaries on MRI and use DaT contrast as supporting evidence.", "",
             "[Cohort mean overlay](qc_alignment.png) · [Per-case coverage and contrast](alignment_per_case.csv)", ""]
    index.extend(f"- [Case {i:02d}](overlays/case_{i:02d}.png)" for i in range(1, len(rows) + 1))
    (out / "review_index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    return summary


COLORS = [(255, 60, 60), (255, 150, 150), (60, 200, 60), (160, 255, 160), (70, 110, 255), (150, 180, 255),
          (255, 220, 0), (255, 240, 140), (230, 0, 230), (255, 150, 255)]


def _slice_rgb(vol, labels, axis, idx, keys, scale=4):
    if axis == 2:
        img, lab = vol[:, :, idx].T[::-1], labels[:, :, idx].T[::-1]          # axial: x right, y up
    elif axis == 1:
        img, lab = vol[:, idx, :].T[::-1], labels[:, idx, :].T[::-1]          # coronal: x right, z up
    else:
        img, lab = vol[idx, :, :].T[::-1], labels[idx, :, :].T[::-1]          # sagittal: y right, z up
    lo, hi = np.percentile(vol[vol != 0], [1, 99]) if np.any(vol != 0) else (0, 1)
    g = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.repeat((g * 255).astype(np.uint8)[..., None], 3, -1)
    rgb = rgb.repeat(scale, 0).repeat(scale, 1)
    lab = lab.repeat(scale, 0).repeat(scale, 1)
    for color, n in zip(COLORS, KEY_NAMES):
        m = lab == keys[n]
        edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
        rgb[edge] = color
    return Image.fromarray(rgb)


def qc_figure(mri, dat, atlas, keys, title, path):
    sections = []
    for section, names, lateral in (
        ("Striatum", KEY_NAMES[:6], "Putamen_L"),
        ("Substantia nigra", KEY_NAMES[6:], "SN_pc_L"),
    ):
        center = np.rint(np.argwhere(np.isin(atlas, [keys[n] for n in names])).mean(0)).astype(int)
        center[0] = int(round(np.argwhere(atlas == keys[lateral])[:, 0].mean()))
        planes = ((2, center[2]), (1, center[1]), (0, center[0]))
        volumes = [("MRI", mri)]
        if dat is not None:
            volumes.append(("DaT", dat))
        for modality, vol in volumes:
            tiles = [_slice_rgb(vol, atlas, axis, index, keys, scale=3) for axis, index in planes]
            caption = f"{title} | {section} | {modality} | axial z={center[2]}, coronal y={center[1]}, sagittal x={center[0]}"
            sections.append((caption, tiles))
    w = sum(tile.width for tile in sections[0][1]) + 40
    h = sum(tiles[0].height + 22 for _, tiles in sections) + 36
    canvas = Image.new("RGB", (w, h), "black")
    draw = ImageDraw.Draw(canvas)
    y = 0
    for caption, tiles in sections:
        draw.text((10, y + 5), caption, fill="white")
        x = 10
        for tile in tiles:
            canvas.paste(tile, (x, y + 22))
            x += tile.width + 10
        y += tiles[0].height + 22
    for i, (color, name) in enumerate(zip(COLORS, KEY_NAMES)):
        draw.text((10 + 65 * i, y + 8), SHORT[name], fill=color)
    canvas.save(path)


# ---------------------------------------------------------------------------------------------------------
# step 5: per-stage measurements
# ---------------------------------------------------------------------------------------------------------
def cell_index(in_shape, out_shape):
    """Flat cell id of every input voxel: cell = floor(i * out / in) per axis."""
    axes = [(np.arange(i) * o) // i for i, o in zip(in_shape, out_shape)]
    cx, cy, cz = np.meshgrid(*axes, indexing="ij")
    return (cx * out_shape[1] + cy) * out_shape[2] + cz


def coverage(atlas: np.ndarray, out_shape) -> np.ndarray:
    """(n_labels, n_cells) fraction of each cell covered by each label (background = label 0)."""
    n_lab = int(atlas.max()) + 1
    n_cells = int(np.prod(out_shape))
    cells = cell_index(atlas.shape, out_shape).ravel()
    counts = np.bincount(cells * n_lab + atlas.ravel().astype(np.int64), minlength=n_cells * n_lab)
    counts = counts.reshape(n_cells, n_lab).T.astype(np.float64)
    size = counts.sum(0)
    return counts / size, counts, size


def nearest_reference(atlas: np.ndarray, out_shape) -> np.ndarray:
    """Independent check of resample_labels: PyTorch 'nearest' samples src = floor(dst * in / out)."""
    idx = [np.floor(np.arange(o, dtype=np.float32) * np.float32(i / o)).astype(int) for i, o in zip(atlas.shape, out_shape)]
    return atlas[np.ix_(*idx)]


def step_stages(out_dir, atlas: np.ndarray, keys: dict[str, int]):
    print("\n[5] Stages")
    in_shape = list(atlas.shape)
    sizes = resnet_stage_sizes(in_shape)
    enc = ResNetEncoder(3, 1, "none").eval()
    with torch.no_grad():
        feats = enc(torch.zeros(1, 1, *in_shape))
    real = [list(f.shape[2:]) for f in feats]
    expected = [[25, 29, 25], [13, 15, 13], [7, 8, 7], [4, 4, 4]]
    print(f"    resnet_stage_sizes {sizes}; real forward {real}; as expected: {sizes == real == expected}")
    if not sizes == real == expected:
        raise SystemExit("STOP: stage sizes differ from the prompt (section 1)")
    A = torch.from_numpy(atlas.astype(np.int64))
    key_ids = [keys[n] for n in KEY_NAMES]
    stage_rows, nearest_maps, cov_all = [], {}, {}
    for s, out in enumerate(sizes):
        near = resample_labels(A, out).numpy()
        nearest_maps[s] = near
        ref_near = nearest_reference(atlas, out)
        cov, counts, size = coverage(atlas, out)
        cov_all[s] = cov
        sum_err = float(np.abs(cov.sum(0) - 1).max())
        vol_err = float(np.abs((cov * size).sum(1) - np.bincount(atlas.ravel(), minlength=cov.shape[0])).max())
        n_near = len(np.unique(near))
        n_area = int((cov.sum(1) > 0).sum())
        nom = 2 * [4, 8, 16, 32][s]
        fov = [round(i * 2 / o, 2) for i, o in zip(in_shape, out)]
        stage_rows.append(dict(stage=s, size="x".join(map(str, out)), cells=int(np.prod(out)), mm_nominal=nom,
                               mm_fov_xyz=str(fov), voxels_per_cell_min=int(size.min()), voxels_per_cell_max=int(size.max()),
                               nearest_labels_incl_bg=n_near, nearest_labels_excl_bg=n_near - int(0 in near),
                               nearest_key_rois_present=sum(int((near == k).any()) for k in key_ids),
                               area_labels_incl_bg=n_area, area_labels_excl_bg=n_area - int(cov[0].sum() > 0),
                               area_key_rois_present=sum(int((cov[k] > 0).any()) for k in key_ids),
                               nearest_equals_reference=bool(np.array_equal(near, ref_near)),
                               coverage_sum_max_err=sum_err, coverage_volume_max_err=vol_err))
        rows = []
        for n in KEY_NAMES:
            c = cov[keys[n]]
            rows.append(dict(stage=s, name=n, id=keys[n], nearest_cells=int((near == keys[n]).sum()),
                             area_cells_gt0=int((c > 0).sum()), area_cells_ge05=int((c >= 0.5).sum()),
                             area_sum_coverage=round(float(c.sum()), 3), area_max_coverage=round(float(c.max()), 3),
                             roi_voxels=int((atlas == keys[n]).sum())))
        write_csv(out_dir, f"key_roi_coverage_stage{s}.csv", rows)
        ov = []
        for i, a in enumerate(KEY_NAMES):
            for j, b in enumerate(KEY_NAMES):
                ca, cb = cov[keys[a]], cov[keys[b]]
                den = float(np.linalg.norm(ca) * np.linalg.norm(cb))
                ov.append(dict(stage=s, roi_a=a, roi_b=b, shared_cells=int(((ca > 0) & (cb > 0)).sum()),
                               cosine=round(float(ca @ cb) / den, 4) if den > 0 else ""))
        write_csv(out_dir, f"key_roi_overlap_stage{s}.csv", ov)
    write_csv(out_dir, "stage_geometry_and_labels.csv", stage_rows)
    for r in stage_rows:
        print(f"    {r}")
    return sizes, nearest_maps, cov_all


# ---------------------------------------------------------------------------------------------------------
# step 6: the four options
# ---------------------------------------------------------------------------------------------------------
def edges_like_graphlayout(labels: list[int], adjacent: set, keys: set) -> int:
    """Undirected anatomical edges kept by GraphLayout: (key ROI, adjacent ROI), both nodes present."""
    pos = {v: i for i, v in enumerate(labels)}
    directed = {(pos[u], pos[v]) for (u, v) in adjacent if u in pos and v in pos and (u in keys or v in keys)}
    return len({tuple(sorted(e)) for e in directed})


def step_options(out_dir, atlas: np.ndarray, keys: dict[str, int], sizes, cov_all, atlas_1mm: np.ndarray):
    print("\n[6] Options")
    A = torch.from_numpy(atlas.astype(np.int64))
    adj_grid = atlas_adjacency(A)                                   # what GraphLayout sees (atlas_path = PPMI grid)
    adj_1mm = atlas_adjacency(torch.from_numpy(atlas_1mm.astype(np.int64)))
    key_ids = {keys[n] for n in KEY_NAMES}
    name_of = {keys[n]: SHORT[n] for n in KEY_NAMES}
    rows, node_rows = [], []
    options = [("b", "nearest+private", [2, 3]), ("c", "nearest+private", [1, 2]),
               ("a", "area", [2, 3]), ("a+c", "area", [1, 2])]
    for code, kind, pair in options:
        for bg in (True, False):
            if kind == "nearest+private":
                maps = [resample_labels(A, sizes[l]) for l in pair]
                layer_labels = private_label_sets(maps, bg)
            else:
                layer_labels = []
                for l in pair:
                    present = [int(v) for v in np.nonzero(cov_all[l].sum(1) > 0)[0]]
                    layer_labels.append([v for v in present if bg or v != 0])
            for li, (l, labs) in enumerate(zip(pair, layer_labels)):
                kpres = [name_of[v] for v in sorted(key_ids) if v in labs]
                nom = 2 * [4, 8, 16, 32][l]
                row = dict(option=code, nodes=kind, stage=l, background=bg, n_nodes=len(labs),
                           k=_knn_k("sqrt_n", len(labs)) if len(labs) > 1 else 0,
                           key_rois_as_nodes=len(kpres), key_rois=" ".join(kpres),
                           anat_edges_grid_adjacency=edges_like_graphlayout(labs, adj_grid, key_ids),
                           anat_edges_1mm_adjacency=edges_like_graphlayout(labs, adj_1mm, key_ids),
                           mm_nominal=nom, ratio_to_orig_mm=round(nom / ORIG_MM[li], 2))
                rows.append(row)
                node_rows.append(dict(option=code, stage=l, background=bg, labels=" ".join(map(str, labs))))
    write_csv(out_dir, "options_summary.csv", rows)
    write_csv(out_dir, "nodes_per_stage.csv", node_rows)
    for row in rows:
        if row["background"]:
            print(f"    {row['option']:9s} stage {row['stage']}: nodes {row['n_nodes']:3d} k {row['k']:2d} key "
                  f"{row['key_rois_as_nodes']:2d} [{row['key_rois']}] edges {row['anat_edges_grid_adjacency']} "
                  f"(1mm adj {row['anat_edges_1mm_adjacency']}) {row['mm_nominal']} mm ({row['ratio_to_orig_mm']}x)")


def step_redundancy(out_dir, cov_all, keys: dict[str, int]):
    """Area nodes: how distinct are the node features? A node feature is the coverage-weighted mean over cells, so two
    nodes with cosine(coverage) ~ 1 get (nearly) the same feature, and at most rank(coverage) nodes are independent."""
    print("\n[6b] Area-node redundancy")
    rows, key_rows = [], []
    key_ids = [keys[n] for n in KEY_NAMES]
    pairs = {"L/R": [(a, a[:-1] + "R") for a in KEY_NAMES if a.endswith("_L")],
             "Cau/Put": [("Caudate_L", "Putamen_L"), ("Caudate_R", "Putamen_R")],
             "Put/Pal": [("Putamen_L", "Pallidum_L"), ("Putamen_R", "Pallidum_R")],
             "SNpc/SNpr": [("SN_pc_L", "SN_pr_L"), ("SN_pc_R", "SN_pr_R")]}
    for s, cov in cov_all.items():
        for bg in (True, False):
            ids = [i for i in range(cov.shape[0]) if cov[i].sum() > 0 and (bg or i != 0)]
            m = cov[ids]
            unit = m / np.linalg.norm(m, axis=1, keepdims=True)
            c = unit @ unit.T
            iu = np.triu_indices(len(ids), 1)
            vals = c[iu]
            rank = int(np.linalg.matrix_rank(m))
            rows.append(dict(stage=s, background=bg, n_nodes=len(ids), n_cells=cov.shape[1], rank_coverage=rank,
                             pairs=len(vals), pairs_cos_ge_0p9=int((vals >= 0.9).sum()),
                             pairs_cos_ge_0p99=int((vals >= 0.99).sum()),
                             nodes_with_a_twin_ge_0p9=int(((c >= 0.9).sum(1) > 1).sum()),
                             median_max_cos_to_other=round(float(np.median(np.max(c - 2 * np.eye(len(ids)), 1))), 4)))
        unit = cov[key_ids] / np.linalg.norm(cov[key_ids], axis=1, keepdims=True)
        kc = unit @ unit.T
        row = dict(stage=s, key_pairs_cos_ge_0p9=int((kc[np.triu_indices(10, 1)] >= 0.9).sum()),
                   key_max_cos=round(float(kc[np.triu_indices(10, 1)].max()), 4))
        for name, plist in pairs.items():
            row[f"cos_{name}"] = " ".join(f"{float(kc[KEY_NAMES.index(a), KEY_NAMES.index(b)]):.3f}" for a, b in plist)
        key_rows.append(row)
    write_csv(out_dir, "area_node_redundancy.csv", rows)
    write_csv(out_dir, "key_roi_overlap_summary.csv", key_rows)
    for r in rows + key_rows:
        print(f"    {r}")


def main():
    parser = qc_parser(__doc__)
    parser.add_argument("--atlas-source", default=str(ATLAS_1MM))
    parser.add_argument("--labels", default=str(LABEL_TXT))
    parser.add_argument("--stage-analysis", action="store_true", help="include stage geometry and node redundancy tables")
    args = parser.parse_args()
    out, rows, config, atlas, ref, evidence = qc_inputs(args)
    out.mkdir(parents=True, exist_ok=True)
    keys, source, provenance = atlas_provenance(out, atlas, ref, config, resolve_path(args.atlas_source),
                                                resolve_path(args.labels))
    evidence["atlas_provenance"] = provenance
    preprocessing = preprocessing_provenance(rows)
    evidence["preprocessing_provenance"] = preprocessing
    protocol = load_yaml(resolve_path(args.protocol))
    evidence["graph"] = GraphLayout(protocol, config).node_info()["layers"]
    evidence["alignment"] = step_alignment(out, atlas, ref, keys, rows)
    reference_hashes = {case["pd25_crop_sha256"] for case in preprocessing}
    if len(reference_hashes) == 1:
        template = nib.load(str(rows[0].mri_path.parent.parent / "MRI" / "PD25_brain_crop.nii.gz"))
        displayed_template = nibabel.processing.resample_from_to(template, (ref.shape, ref.affine), order=1)
        preview = out / "qc_pd25_template.png"
        qc_figure(displayed_template.get_fdata(dtype=np.float32), None, atlas, keys, "PD25 template crop", preview)
        evidence["pd25_template_overlay"] = dict(file=preview.name, crop_sha256=next(iter(reference_hashes)),
                                                  interpolation_order=1, atlas_transform_applied=False)
        with (out / "review_index.md").open("a", encoding="utf-8") as handle:
            handle.write("\n[PD25 template crop overlay](qc_pd25_template.png)\n")
    evidence["lattice_check"] = "PASS"
    evidence["anatomical_review"] = "REVIEW_REQUIRED"
    if args.stage_analysis:
        sizes, _, cov_all = step_stages(out, atlas, keys)
        step_options(out, atlas, keys, sizes, cov_all, source)
        step_redundancy(out, cov_all, keys)
    write_json(out / "atlas_qc.json", evidence)
    print(f"Grid and label provenance checked on {len(rows)} cases. Anatomy review packet: {out / 'review_index.md'}")


if __name__ == "__main__":
    main()
