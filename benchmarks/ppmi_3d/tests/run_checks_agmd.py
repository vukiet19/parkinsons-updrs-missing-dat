"""Section-9 checks for AGMD-dagger (no full training). CRAD/DFTD keep their own check scripts.

    python tests/run_checks_agmd.py      # writes notes/checks_report_agmd.md and notes/checks_report_agmd.json

Exit code 1 if any check fails. Results are measured, never estimated.
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import math
import shutil
import sys
import tempfile
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate as shared_evaluate  # noqa: E402
import train as shared_train  # noqa: E402
from baselines.agmd.graph import (gcn_norm, grid_masks, private_label_sets, resample_labels,  # noqa: E402
                                  semantic_adjacency)
from baselines.agmd.method import AGMDMethod  # noqa: E402
from baselines.agmd.uag import uag_weights  # noqa: E402
from data.dataset import ImageRows, collate, make_loader, to_device  # noqa: E402
from data.manifest import TargetScaler, data_fingerprint, make_split, read_manifest, rows_of  # noqa: E402
from utils.config import apply_overrides, load_yaml  # noqa: E402
from utils.inference import freeze  # noqa: E402
from utils.repro import choose_device, seed_everything  # noqa: E402

# P0a: --protocol / --report_suffix (defaults = previous behaviour). Check contents are unchanged.
_CLI = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_CLI.add_argument("--protocol", default="configs/protocol.yaml")
_CLI.add_argument("--report_suffix", default="", help="report -> notes/checks_report_agmd<suffix>.md/.json")
_CLI.add_argument("--report_dir", type=Path, default=ROOT / "notes", help="directory for the Markdown and JSON reports")
ARGS = _CLI.parse_args() if __name__ == "__main__" else _CLI.parse_args([])
PROTOCOL = load_yaml(Path(ARGS.protocol) if Path(ARGS.protocol).is_absolute() else ROOT / ARGS.protocol)
CONFIG = load_yaml(ROOT / "configs" / "baselines" / "agmd.yaml")
DEVICE = choose_device("auto")
ORIG = ROOT / "assets" / "source" / "AGMD"                       # official repo clone (read as text only, never executed)
RESULTS: list[dict] = []
LINES: list[str] = []
ROWS = read_manifest(PROTOCOL)
SPLIT0 = make_split(ROWS, 0, PROTOCOL)
TRAIN0 = rows_of(ROWS, SPLIT0.train)


def log(text=""):
    print(text, flush=True)
    LINES.append(text)


def check(name):
    def wrap(fn):
        def run():
            start = time.time()
            try:
                detail = fn() or ""
                RESULTS.append(dict(name=name, passed=True, detail=detail, seconds=round(time.time() - start, 1)))
                log(f"[PASS] {name} ({time.time() - start:.1f}s) {detail}")
            except Exception as error:
                RESULTS.append(dict(name=name, passed=False, detail=repr(error), seconds=round(time.time() - start, 1)))
                log(f"[FAIL] {name}: {error!r}")
                traceback.print_exc()
        return run
    return wrap


def cfg_with(*overrides):
    return apply_overrides(copy.deepcopy(CONFIG), list(overrides))


def real_batch(rows, n, load_dat=True):
    data = ImageRows(rows[:n], PROTOCOL, load_dat)
    batch = to_device(collate([data[i] for i in range(len(data))]), DEVICE)
    batch["y_z"] = TargetScaler.fit(rows).normalize(batch["y"])
    return batch


def nparams(m):
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------------------------------------
@check("1. Shape test (2D real NTUA batch; 3D synthetic grid; 3D atlas nodes) and parameter counts")
def shape_test():
    seed_everything(0)
    m = AGMDMethod(PROTOCOL, CONFIG)
    t, s = m.build("teacher").to(DEVICE), m.build("student").to(DEVICE)
    b = real_batch(TRAIN0, 4)
    ot, os_ = t(b["mri"], b["dat"]), s(b["mri"])
    log(f"    2D teacher: mri {tuple(b['mri'].shape)} dat {tuple(b['dat'].shape)} -> fused "
        f"{ {l: tuple(v.shape) for l, v in ot['feats'].items()} }, graph {tuple(ot['graph'].shape)}, "
        f"global {tuple(ot['global_'].shape)}, pred {tuple(ot['pred'].shape)}")
    log(f"    2D student: feats {[tuple(v.shape) for v in os_['feats'].values()]}, graph {tuple(os_['graph'].shape)}, "
        f"global {tuple(os_['global_'].shape)}")
    assert ot["graph"].shape == os_["graph"].shape == (4, 32, 64) and ot["pred"].shape == (4,)
    for l in (2, 3):
        assert ot["feats"][l].shape == os_["feats"][l].shape
    log(f"    params: teacher {nparams(t) / 1e6:.2f} M (CMT {nparams(t.cmt) / 1e6:.2f} M, GCN {nparams(t.graph) / 1e6:.3f} M,"
        f" readout {nparams(t.readout) / 1e6:.2f} M); student {nparams(s) / 1e6:.2f} M")
    # 3D, grid layout (64^3 -> stage 3/4 = 4^3 / 2^3, grid 2 per axis)
    p3 = apply_overrides(PROTOCOL, ["data.spatial_dims=3", "preprocessing.input_size=[64,64,64]",
                                    "backbone.pretrained=none", "preprocessing.mri_channels=1",
                                    "preprocessing.dat_channels=1"])
    c3 = cfg_with("graph.grid=[2,2]", "graph.grid_key_size=1")
    m3 = AGMDMethod(p3, c3)
    t3, s3 = m3.build("teacher").to(DEVICE), m3.build("student").to(DEVICE)
    x = torch.randn(2, 1, 64, 64, 64, device=DEVICE)
    o3 = t3(x, x.clone())
    assert o3["graph"].shape == (2, 16, 64) and s3(x)["pred"].shape == (2,)
    log(f"    3D grid: input 64^3 -> fused {[tuple(v.shape) for v in o3['feats'].values()]}, graph {tuple(o3['graph'].shape)}")
    # 3D, atlas layout: official raal3_1.pt if present (else synthetic labels), private-node scheme
    atlas_file = ORIG / "data" / "atlas" / "raal3_1.pt"
    tmp = Path(tempfile.mkdtemp(prefix="agmd_atlas_"))
    try:
        if atlas_file.exists():
            atlas_path, src = atlas_file, "official raal3_1.pt (56x68x56)"
        else:
            g = torch.arange(4 * 5 * 4).view(4, 5, 4) % 23
            atlas = g.repeat_interleave(8, 0).repeat_interleave(8, 1).repeat_interleave(8, 2)
            atlas_path, src = tmp / "atlas.pt", "synthetic 32x40x32"
            torch.save(atlas.float(), atlas_path)
        pa = apply_overrides(p3, ["preprocessing.input_size=[128,160,128]"])
        ca = cfg_with("graph.nodes=atlas", f"graph.atlas_path={atlas_path.as_posix()}")
        ma = AGMDMethod(pa, ca)
        sa = ma.build("student")
        lay = sa.graph.layout
        # reference: transcription of orig_repo/utils.py:109-118 on the same resampled maps
        atlas = torch.load(atlas_path, map_location="cpu", weights_only=True).round().long()
        maps = [resample_labels(atlas, sz) for sz in ([8, 10, 8], [4, 5, 4])]
        sets = [set(torch.unique(mm.flatten()).numpy().tolist()) for mm in maps]
        ref = [sorted(sets[0].difference(sets[1])), sorted(sets[1])]
        assert [sorted(n) for n in lay.node_ids] == ref, (lay.node_ids, ref)
        assert private_label_sets(maps, True) == ref
        n_anat = [int(lay.anatomical(i).sum()) // 2 for i in range(2)]
        xa = torch.randn(1, 1, 128, 160, 128)
        oa = sa(xa)
        log(f"    3D atlas ({src}): input 128x160x128 -> nodes per layer {lay.n_nodes} (private-node scheme = code), "
            f"k {lay.k}, anatomical edges {n_anat}, graph {tuple(oa['graph'].shape)}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return "2D/3D shapes OK"


@check("2. Graph: node = block mean, kNN graph (k = ceil(sqrt N), symmetric, no self loop), anatomical edges "
       "in teacher AND student, dense GCN == PyG GCNConv formula with duplicate edges")
def graph_test():
    seed_everything(0)
    m = AGMDMethod(PROTOCOL, CONFIG)
    t, s = m.build("teacher"), m.build("student")
    lay = s.graph.layout
    fmap = torch.randn(3, 8, 14, 14)
    h = lay.node_features(0, fmap)
    ref = torch.stack([blk.mean((2, 3)) for row in torch.tensor_split(fmap, 4, 2)
                       for blk in torch.tensor_split(row, 4, 3)], 1)
    assert torch.allclose(h, ref, atol=1e-6)
    masks = grid_masks([7, 7], 4)
    assert masks.sum(0).eq(1).all()                   # non-overlapping, full cover
    a = semantic_adjacency(h, lay.k[0])
    assert not a.requires_grad and torch.equal(a, a.transpose(1, 2)) and a.diagonal(dim1=1, dim2=2).eq(0).all()
    assert a.sum(-1).ge(lay.k[0]).all() and lay.k == [4, 4]
    # anatomical edges: central 2x2 blocks of a 4x4 grid and their 4-neighbours -> 12 undirected edges
    for i in range(2):
        aa = lay.anatomical(i)
        assert torch.equal(aa, aa.T) and int(aa.sum()) == 24
        assert torch.equal(aa, t.graph.layout.anatomical(i))
    keys = {5, 6, 9, 10}
    nz = {tuple(e) for e in lay.anatomical(0).nonzero().tolist()}
    assert all(u in keys or v in keys for u, v in nz)
    # dense GCN vs a sparse transcription of PyG gcn_norm + propagate (edge list = sem + anat, duplicates kept)
    conv = s.graph.gcn[0].conv1
    x = torch.randn(16, 256)
    a_sem = semantic_adjacency(torch.randn(1, 16, 32), 4)[0]
    a_f = a_sem + lay.anatomical(0)
    dense = conv(x[None], gcn_norm(a_f[None]))[0]
    edges = torch.cat([a_sem.nonzero(), lay.anatomical(0).nonzero(), torch.arange(16).repeat(2, 1).T])  # (src,dst)
    src, dst = edges[:, 0], edges[:, 1]
    deg = torch.zeros(16).index_add_(0, dst, torch.ones(len(dst)))
    norm = deg[src].pow(-0.5) * deg[dst].pow(-0.5)
    xw = conv.lin(x)
    sparse = torch.zeros_like(xw).index_add_(0, dst, xw[src] * norm[:, None]) + conv.bias
    err = float((dense - sparse).abs().max())
    assert err < 1e-5, err
    log(f"    kNN k={lay.k}; anatomical edges/layer = 12 (undirected, key blocks {sorted(keys)}); "
        f"teacher/student anatomical graphs identical; dense vs sparse GCN max |diff| = {err:.1e}")
    off = AGMDMethod(PROTOCOL, cfg_with("graph.anatomical=false")).build("student").graph.layout
    assert int(off.anatomical(0).sum()) == 0
    return "OK"


@check("3. CMT: attention over DaT positions sums to 1; 1/sqrt(C) scaling as Eq. 1 (code: none)")
def cmt_test():
    seed_everything(0)
    t = AGMDMethod(PROTOCOL, CONFIG).build("teacher").to(DEVICE).eval()
    tc = AGMDMethod(PROTOCOL, cfg_with("cmt.scale=none")).build("teacher").to(DEVICE).eval()
    tc.load_state_dict(t.state_dict())
    b = real_batch(TRAIN0, 2)
    out = []
    for model in (t, tc):
        mod = model.cmt["3"]
        mod.keep_attention = True
        with torch.no_grad():
            model(b["mri"], b["dat"])
        out.append(mod.last_attention)
        mod.keep_attention = False
    a_paper, a_code = out
    assert torch.allclose(a_paper.sum(-1), torch.ones_like(a_paper.sum(-1)), atol=1e-5)
    # same logits, different temperature: log-ratio of the two attentions differs by a factor sqrt(512)
    ent = lambda a: float(-(a * torch.log(a + 1e-12)).sum(-1).mean())
    log(f"    stage-4 attention {tuple(a_paper.shape)}; mean row entropy: scaled {ent(a_paper):.3f} vs "
        f"unscaled (code) {ent(a_code):.3f} (uniform = {math.log(a_paper.shape[-1]):.3f})")
    assert ent(a_paper) > ent(a_code)
    return "OK"


@check("4. Graph transformer: a subject's prediction does not depend on its batch (rule D fix); the code's "
       "TransformerFusion (batch as sequence) does")
def batch_independence_test():
    seed_everything(0)
    m = AGMDMethod(PROTOCOL, CONFIG)
    t, s = m.build("teacher").to(DEVICE).eval(), m.build("student").to(DEVICE).eval()
    b = real_batch(TRAIN0, 6)
    with torch.no_grad():
        full_s, one_s = s(b["mri"])["pred"], s(b["mri"][:1])["pred"]
        full_t, one_t = t(b["mri"], b["dat"])["pred"], t(b["mri"][:1], b["dat"][:1])["pred"]
        # readout alone, float64 on CPU: exact independence (GPU convs differ by ~1e-4 across batch sizes, TF32)
        ro = copy.deepcopy(s.readout).double().cpu().eval()
        tok = torch.randn(6, s.graph.n_nodes, s.graph.hidden, dtype=torch.float64)
        d_readout = float((ro(tok)[0] - ro(tok[:1])[0]).abs().max())
    d_ours = max(float((full_s[0] - one_s[0]).abs()), float((full_t[0] - one_t[0]).abs()))
    assert d_readout < 1e-10 and d_ours < 1e-3, (d_readout, d_ours)
    # transcription of orig_repo/model.py:245-253 at reduced width (d_model = N*64 = 2048)
    torch.manual_seed(0)
    layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8)            # batch_first=False as in the code
    code_tf = nn.TransformerEncoder(layer, num_layers=1).eval()
    x = torch.randn(6, 2048)
    with torch.no_grad():
        d_code = float((code_tf(x)[0] - code_tf(x[:1])[0]).abs().max())
    assert d_code > 1e-3, d_code
    log(f"    ours: readout (float64, CPU) |out(batch of 6) - out(alone)| = {d_readout:.1e}; full model on GPU "
        f"{d_ours:.1e} (cuDNN/TF32 numerical noise, z-units); code-style transformer output change "
        f"for the same subject = {d_code:.3f}")
    return "OK"


GROUPS = ("mri_encoder.stem", "mri_encoder.stages.0", "mri_encoder.stages.1", "mri_encoder.stages.2",
          "mri_encoder.stages.3", "graph.gcn", "readout.encoder", "readout.down", "head")


def reached(model, loss):
    params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, [p for _, p in params], retain_graph=True, allow_unused=True)
    hit = set()
    for (n, _), g in zip(params, grads):
        if g is not None and g.abs().sum() > 0:
            hit.update(gr for gr in GROUPS if n.startswith(gr))
    return hit


@check("5. Gradient test: teacher stage trains every teacher parameter; student stage: teacher frozen "
       "(grad None), each distillation term reaches exactly the expected student modules")
def gradient_test():
    seed_everything(0)
    m = AGMDMethod(PROTOCOL, CONFIG)
    t, s = m.build("teacher").to(DEVICE), m.build("student").to(DEVICE)
    b = real_batch(TRAIN0, 6)
    loss, _ = m.training_step("teacher", t, {}, b)
    loss.backward()
    no_grad = [n for n, p in t.named_parameters() if p.grad is None]
    assert not no_grad, no_grad
    t.zero_grad(set_to_none=True)
    freeze(t)
    out = s(b["mri"])
    with torch.no_grad():
        ot = t(b["mri"], b["dat"])
    terms = m.distill_terms(out, ot)
    enc = {"mri_encoder.stem", "mri_encoder.stages.0", "mri_encoder.stages.1", "mri_encoder.stages.2"}
    expect = {"shallow_2": enc,
              "shallow_3": enc | {"mri_encoder.stages.3"},
              "graph": enc | {"mri_encoder.stages.3", "graph.gcn"},
              "global": enc | {"mri_encoder.stages.3", "graph.gcn", "readout.encoder", "readout.down"},
              "logit": set(GROUPS)}
    for name, groups in expect.items():
        got = reached(s, terms[name].mean())
        assert got == groups, (name, sorted(got ^ groups))
        log(f"    {name:9s} -> {sorted(got)}")
    loss, _ = m.training_step("student", s, {"teacher": t}, b)
    loss.backward()
    assert all(p.grad is None for p in t.parameters())
    assert all(p.grad is not None for p in s.parameters())
    return "OK"


@check("6. UAG weights: error mode formula, code feature-entropy mode, batch reduction, none; detached, in [0, 1]; "
       "code cal_weight bug (rule D)")
def uag_test():
    seed_everything(0)
    m = AGMDMethod(PROTOCOL, CONFIG)
    t = freeze(m.build("teacher").to(DEVICE))
    b = real_batch(TRAIN0, 6)
    with torch.no_grad():
        ot = t(b["mri"], b["dat"])
    y = b["y_z"]
    w = uag_weights(CONFIG, ot, ot["pred"], y, [2, 3])
    ref = 1 - ((ot["pred"] - y).abs() / 1.0).clamp(0, 1)
    assert all(torch.allclose(v, ref) for v in w.values())
    cfe = cfg_with("uag.mode=feature_entropy", "uag.reduce=batch")
    wf = uag_weights(cfe, ot, ot["pred"], y, [2, 3])
    # transcription of orig_repo/model.py:356-366 (mean over the 2D spatial axes here)
    def code_w(feat):
        prob = F.softmax(feat, dim=-1)
        return 1 - (-(prob * torch.log(prob + 1e-10)).sum(1)) / math.log(feat.size(1))
    assert torch.allclose(wf["shallow_2"], code_w(ot["feats"][2].mean(dim=[2, 3])).mean().expand(6), atol=1e-6)
    assert torch.allclose(wf["graph"], code_w(ot["global_"]).mean().expand(6), atol=1e-6)
    wn = uag_weights(cfg_with("uag.mode=none"), ot, ot["pred"], y, [2, 3])
    assert all(torch.equal(v, torch.ones_like(v)) for v in wn.values())
    for d in (w, wf):
        assert all((not v.requires_grad) and v.min() >= 0 and v.max() <= 1 for v in d.values())
    # orig_repo/train_test.py:382-386 on 4-class probabilities: classes 2, 3 always get weight 0
    p = torch.softmax(torch.randn(1000, 4), -1)
    for label in range(4):
        cw = (p.max(1)[0] - label).abs()
        cw[cw > 1] = 1
        cw = 1 - cw
        if label >= 2:
            assert cw.max() == 0
    log(f"    error-mode w (fold-0 batch): {[round(v, 3) for v in ref.tolist()]}; code feature-entropy w: "
        f"shallow_2 {float(wf['shallow_2'][0]):.4f}, shallow_3 {float(wf['shallow_3'][0]):.4f}, "
        f"graph/global {float(wf['graph'][0]):.4f}; code cal_weight = 0 for every label 2/3 sample")
    return "OK"


@check("7. Overfit test (12 rows of 6 subjects, 100 epochs): teacher, AGMD student, student-only")
def overfit_test():
    rows = [r for r in TRAIN0 if r.dat_path is not None][:12]
    proto = apply_overrides(PROTOCOL, ["training.batch_size=12"])
    out = []
    for stage, cfg in (("teacher", CONFIG), ("student", CONFIG), ("student", cfg_with("kd.enabled=false"))):
        seed_everything(0)
        m = AGMDMethod(proto, cfg)
        frozen = {"teacher": freeze(teacher)} if stage == "student" and m.kd_enabled else {}
        model = m.build(stage).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        loader = make_loader(rows, proto, m.needs_dat(stage, "train"), shuffle=True, seed=0)
        hist = []
        for _ in range(100):
            model.train()
            for batch in loader:
                batch = to_device(batch, DEVICE)
                batch["y_z"] = TargetScaler.fit(rows).normalize(batch["y"])
                loss, terms = m.training_step(stage, model, frozen, batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
            hist.append((float(loss), float(terms["task"])))
        if stage == "teacher":
            teacher = model
        name = stage if m.kd_enabled else "student_only"
        out.append(f"{name}: loss {hist[0][0]:.3f} -> {hist[-1][0]:.4f} (task {hist[0][1]:.3f} -> {hist[-1][1]:.4f})")
        assert hist[-1][1] < 0.1 * hist[0][1], (name, hist[0], hist[-1])
    for line in out:
        log("    " + line)
    return "task loss < 10% of its start for all three"


@check("8. Loss test: every term finite; each option/term switchable; rows without DaT get the task loss only")
def loss_test():
    b = real_batch(TRAIN0, 6)
    b["has_dat"] = torch.tensor([True, True, False, True, False, True], device=DEVICE)
    variants = [[], ["uag.mode=feature_entropy", "uag.reduce=batch"], ["uag.mode=none"], ["uag.reduce=batch"],
                ["kd.logit=false"], ["graph.anatomical=false"], ["cmt.scale=none"], ["cmt.layers=[0,1,2,3]"],
                ["kd.w_shallow=0"], ["kd.w_graph=0"], ["kd.w_global=0"], ["kd.enabled=false"], ["graph.knn=8"],
                ["graph.grid=[7,4]"], ["uag.error_scale=0.5"]]
    lines = []
    for ov in variants:
        seed_everything(0)
        cfg = cfg_with(*ov)
        m = AGMDMethod(PROTOCOL, cfg)
        frozen = {}
        if m.kd_enabled:
            frozen["teacher"] = freeze(m.build("teacher").to(DEVICE))
        s = m.build("student").to(DEVICE)
        loss, terms = m.training_step("student", s, frozen, b)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(v) for v in terms.values()), (ov, terms)
        lines.append(f"{' '.join(ov) or 'default':45s} loss {float(loss):.4f} " +
                     " ".join(f"{k}={float(v):.4f}" for k, v in terms.items() if k != "task"))
    seed_everything(0)
    m = AGMDMethod(PROTOCOL, CONFIG)
    tch, s = freeze(m.build("teacher").to(DEVICE)), m.build("student").to(DEVICE)
    b["has_dat"][:] = False
    loss, terms = m.training_step("student", s, {"teacher": tch}, b)
    assert torch.equal(loss, terms["task"])
    for line in lines:
        log("    " + line)
    return f"{len(variants)} configurations finite; no-DaT batch = task loss only"


@check("9. MRI-only student: evaluation loader never opens DaT; prediction needs no 'dat'")
def mri_only_test():
    m = AGMDMethod(PROTOCOL, CONFIG)
    assert m.needs_dat("student", "eval") is False and m.needs_dat("teacher", "eval") is True
    s = m.build("student").to(DEVICE).eval()
    loader = make_loader(TRAIN0[:4], PROTOCOL, m.needs_dat("student", "eval"), shuffle=False, seed=0)
    batch = to_device(next(iter(loader)), DEVICE)
    assert "dat" not in batch and not batch["has_dat"].any()
    with torch.no_grad():
        pred = m.predict("student", s, batch)
    assert pred.shape == (4,) and torch.isfinite(pred).all()
    return "OK"


@check("10. train.py/evaluate.py integration: teacher+student, determinism, --teacher_from reuse and guard")
def integration_test():
    proto = apply_overrides(PROTOCOL, ["training.max_epochs=2"])
    tmp = Path(tempfile.mkdtemp(prefix="agmd_checks_"))
    try:
        fp = data_fingerprint(proto, ROWS)
        shared_train.run_one("agmd", AGMDMethod, proto, CONFIG, ROWS, fp, 0, 0, tmp / "a" / "fold0_seed0", DEVICE)
        shared_train.run_one("agmd", AGMDMethod, proto, CONFIG, ROWS, fp, 0, 0, tmp / "b" / "fold0_seed0", DEVICE)
        same = all(all(torch.equal(x, y) for x, y in zip(
            torch.load(tmp / d1 / "fold0_seed0" / f"{st}.pt", weights_only=False)["state_dict"].values(),
            torch.load(tmp / d2 / "fold0_seed0" / f"{st}.pt", weights_only=False)["state_dict"].values()))
            for st in ("teacher", "student") for d1, d2 in [("a", "b")])
        assert same
        cfg_s = cfg_with("uag.mode=none")
        shared_train.run_one("agmd", AGMDMethod, proto, cfg_s, ROWS, fp, 0, 0, tmp / "c", DEVICE, teacher_from=tmp / "a")
        meta = json.loads((tmp / "c" / "run.json").read_text(encoding="utf-8"))
        assert "reused_from" in meta["stages"]["teacher"] and not (tmp / "c" / "teacher.pt").exists()
        cache = {}
        preds, scores = shared_evaluate.evaluate_run("agmd", tmp / "c", "test", ["all"], DEVICE, False, cache)
        roles = sorted({s["role"] for s in scores})
        assert roles == ["student", "teacher", "train_mean"], roles
        try:
            shared_train.run_one("agmd", AGMDMethod, proto, cfg_with("graph.anatomical=false"), ROWS, fp, 0, 0,
                                 tmp / "d", DEVICE, teacher_from=tmp / "a")
        except ValueError as error:
            assert "teacher_config" in str(error)
        else:
            raise AssertionError("--teacher_from accepted a teacher with a different graph config")
        size = sum(p.stat().st_size for p in (tmp / "a" / "fold0_seed0").glob("*.pt"))
        log(f"    two identical runs -> identical teacher/student weights; student-side variant reuses the teacher; "
            f"evaluate roles {roles}; graph change refused; checkpoints per run = {size / 2**20:.1f} MiB")
        return "OK"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@check("11. Official-code defects confirmed statically (AST of LiuFei-AHU/AGMD@dc676d9, never executed)")
def static_code_test():
    if not (ORIG / "utils.py").exists():
        return f"SKIPPED: official repo not found at {ORIG}"
    utils_tree = ast.parse((ORIG / "utils.py").read_text(encoding="utf-8"))
    model_src = (ORIG / "model.py").read_text(encoding="utf-8")
    model_tree = ast.parse(model_src)
    fn = next(n for n in ast.walk(utils_tree) if isinstance(n, ast.FunctionDef) and n.name == "build_graph_structure_batch")
    params = [a.arg for a in fn.args.args]
    calls = [n for n in ast.walk(model_tree) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") ==
             "build_graph_structure_batch"]
    anat_call = next(c for c in calls if len(c.args) == 4)
    bound = params[3]
    fourth = ast.unparse(anat_call.args[3])
    k_expr = ast.unparse(anat_call.args[2])
    assert bound == "no_edge" and fourth == "anatomical_edges" and not anat_call.keywords      # G5
    assert "len(node_feature)" in k_expr                                                       # G6
    student_call = next(n for n in ast.walk(model_tree) if isinstance(n, ast.Call)
                        and getattr(n.func, "attr", "") == "build_graph" and n.lineno > 318)
    assert not any(k.arg == "cfg" for k in student_call.keywords) and len(student_call.args) < 4   # G9
    tel = next(n for n in ast.walk(model_tree) if isinstance(n, ast.Call)
               and getattr(n.func, "attr", "") == "TransformerEncoderLayer")
    assert not any(k.arg == "batch_first" for k in tel.keywords)                                # G7
    attn = [ln for ln in model_src.splitlines() if "einsum('bcdhw,bcDHW->bdhwDHW'" in ln][0]
    assert "sqrt" not in attn                                                                    # G4
    log(f"    G5: model.py:{anat_call.lineno} passes `{fourth}` as 4th positional argument -> parameter `{bound}` "
        f"of utils.build_graph_structure_batch{tuple(params)}; G6: k = `{k_expr}` (batch size); "
        f"G9: student build_graph without cfg (model.py:{student_call.lineno}); G7: TransformerEncoderLayer without "
        f"batch_first (model.py:{tel.lineno}); G4: no 1/sqrt(C) in `{attn.strip()}`")
    return "G4, G5, G6, G7, G9 confirmed"


def main():
    log(f"# AGMD-dagger checks ({time.strftime('%Y-%m-%d %H:%M')}, device={DEVICE}, torch {torch.__version__})")
    log()
    for fn in (shape_test, graph_test, cmt_test, batch_independence_test, gradient_test, uag_test, overfit_test,
               loss_test, mri_only_test, integration_test, static_code_test):
        fn()
    passed = sum(r["passed"] for r in RESULTS)
    log()
    log(f"{passed}/{len(RESULTS)} checks passed.")
    ARGS.report_dir.mkdir(parents=True, exist_ok=True)
    (ARGS.report_dir / f"checks_report_agmd{ARGS.report_suffix}.md").write_text("```text\n" + "\n".join(LINES) + "\n```\n", encoding="utf-8")
    (ARGS.report_dir / f"checks_report_agmd{ARGS.report_suffix}.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
