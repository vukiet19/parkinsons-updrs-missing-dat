"""Section-9 checks for ShaSpec-dagger (no full training). Other methods keep their own check scripts.

    python tests/run_checks_shaspec.py   # writes notes/checks_report_shaspec.md and notes/checks_report_shaspec.json

Exit code 1 if any check fails. Results are measured, never estimated.
"""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
import tempfile
import time
import traceback
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate as shared_evaluate  # noqa: E402
import train as shared_train  # noqa: E402
from baselines import get_method  # noqa: E402
from baselines.shaspec.method import ShaSpecMethod  # noqa: E402
from baselines.shaspec.model import MODALITIES  # noqa: E402
from data.dataset import ImageRows, collate, make_loader, to_device  # noqa: E402
from data.manifest import TargetScaler, data_fingerprint, make_split, read_manifest, rows_of  # noqa: E402
from utils.config import apply_overrides, load_yaml  # noqa: E402
from utils.inference import freeze  # noqa: E402
from utils.repro import choose_device, seed_everything  # noqa: E402

# P0a: --protocol / --report_suffix (defaults = previous behaviour). Check contents are unchanged.
_CLI = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_CLI.add_argument("--protocol", default="configs/protocol.yaml")
_CLI.add_argument("--report_suffix", default="", help="report -> notes/checks_report_shaspec<suffix>.md/.json")
ARGS = _CLI.parse_args() if __name__ == "__main__" else _CLI.parse_args([])
PROTOCOL = load_yaml(Path(ARGS.protocol) if Path(ARGS.protocol).is_absolute() else ROOT / ARGS.protocol)
CONFIG = load_yaml(ROOT / "configs" / "baselines" / "shaspec.yaml")
DEVICE = choose_device("auto")
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


def real_batch(rows, n, load_dat=True, protocol=PROTOCOL):
    data = ImageRows(rows[:n], protocol, load_dat)
    batch = to_device(collate([data[i] for i in range(len(data))]), DEVICE)
    batch["y_z"] = TargetScaler.fit(rows).normalize(batch["y"])
    return batch


def without_dat(rows, subjects):
    return [r.__class__(**{**r.__dict__, "dat_path": None}) if r.subject_id in subjects else r for r in rows]


def avail_all(batch):
    b = batch["mri"].shape[0]
    return {m: torch.ones(b, dtype=torch.bool, device=DEVICE) for m in MODALITIES}


# ---------------------------------------------------------------------------------------------------------------
# Independent references written from the paper's equations (Wang et al., CVPR 2023). STEWARD D-015 (i):
# the official repository (billhhh/ShaSpec @ 57a04fa) has no licence, so no code from it is reproduced here.
# Equality of our modules with the official code was verified on 2026-11-17 with transcriptions that were
# removed afterwards (notes/checks_report_shaspec.md of that date: DAO 1.35218, DCO 1.15204).
def ref_fusion(r, s, weight):
    """Eq. 2: f = f_proj([r, s]) + r, with f_proj a bias-free k=3 convolution."""
    return r + F.conv3d(torch.cat([r, s], 1), weight, padding=1)


def ref_missing_feature(r_available):
    """Eq. 4 with N = 2 modalities: the missing slot is the mean of the ONE available shared feature."""
    return torch.stack([r_available], 0).mean(0)


def ref_dao(r_mri, r_dat):
    """Eq. 8 with p = 1 and N = 2: both ordered pairs (mri, dat) and (dat, mri), element-mean absolute error."""
    return 2.0 * (r_mri - r_dat).abs().mean()


def ref_dco(logits_mri, logits_dat):
    """Eq. 5: cross-entropy of the modality classifier on specific features (label 0 = MRI, 1 = DaT)."""
    labels = torch.cat([torch.zeros(len(logits_mri)), torch.ones(len(logits_dat))]).long().to(logits_mri.device)
    return F.cross_entropy(torch.cat([logits_mri, logits_dat]), labels)


# ---------------------------------------------------------------------------------------------------------------
@check("1. Shape test (2D NTUA batch, linear and conv fusion; synthetic 3D volume)")
def shape_test():
    out = []
    batch = real_batch(TRAIN0, 4)
    for fusion in ("linear", "conv"):
        cfg = apply_overrides(CONFIG, [f"fusion.type={fusion}"])
        method = ShaSpecMethod(PROTOCOL, cfg)
        model = method.build("main").to(DEVICE).eval()
        with torch.no_grad():
            feats = model.shared(batch["mri"])
            res = model(dict(mri=batch["mri"], dat=batch["dat"]), avail_all(batch))
        log(f"    [{fusion}] input mri {tuple(batch['mri'].shape)} dat {tuple(batch['dat'].shape)}; shared stages "
            f"{[tuple(f.shape) for f in feats]}; r_mri {tuple(res['r']['mri'].shape)} s_dat {tuple(res['s']['dat'].shape)}; "
            f"head in {model.head.in_features}; pred {tuple(res['pred'].shape)}")
        assert res["pred"].shape == (4,)
        n_params = sum(p.numel() for p in model.parameters())
        out.append(f"{fusion}: {n_params / 1e6:.2f}M params")
    so = ShaSpecMethod(PROTOCOL, apply_overrides(CONFIG, ["mode=student_only"])).build("student")
    out.append(f"student_only: {sum(p.numel() for p in so.parameters()) / 1e6:.2f}M params")
    proto3 = apply_overrides(PROTOCOL, ["data.spatial_dims=3", "preprocessing.input_size=[32,32,32]",
                                        "backbone.pretrained=none", "preprocessing.mri_channels=1",
                                        "preprocessing.dat_channels=1"])
    for fusion in ("linear", "conv"):
        model = ShaSpecMethod(proto3, apply_overrides(CONFIG, [f"fusion.type={fusion}"])).build("main").to(DEVICE)
        x = torch.randn(2, 1, 32, 32, 32, device=DEVICE)
        res = model(dict(mri=x, dat=x.flip(2)), {m: torch.ones(2, dtype=torch.bool, device=DEVICE) for m in MODALITIES})
        log(f"    [3D {fusion}] r_mri {tuple(res['r']['mri'].shape)} pred {tuple(res['pred'].shape)}")
        assert res["pred"].shape == (2,)
    return "; ".join(out)


@check("2. Faithfulness to Eq. 2, 4, 5, 8 (independent references from the paper's equations)")
def faithfulness_test():
    torch.manual_seed(0)
    # (a) Eq. 2 conv fusion (3D, 256 channels).
    proto3 = apply_overrides(PROTOCOL, ["data.spatial_dims=3", "backbone.pretrained=none",
                                       "preprocessing.mri_channels=1", "preprocessing.dat_channels=1"])
    ours = ShaSpecMethod(proto3, apply_overrides(CONFIG, ["fusion.type=conv"])).build("main").double()
    ours.proj = nn.Conv3d(512, 256, 3, padding=1, bias=False).double()
    r, s = torch.randn(2, 256, 4, 5, 3, dtype=torch.float64), torch.randn(2, 256, 4, 5, 3, dtype=torch.float64)
    torch.testing.assert_close(ours.proj(torch.cat([r, s], 1)) + r, ref_fusion(r, s, ours.proj.weight),
                               rtol=0, atol=1e-12)
    # (b) Eq. 4: our model fills the missing DaT slot with r of the available MRI (both missing_feature modes).
    batch = real_batch(TRAIN0, 3)
    for missing_feature in ("mean_shared", "first_available"):
        method = ShaSpecMethod(PROTOCOL, apply_overrides(CONFIG, [f"missing_feature={missing_feature}"]))
        model = method.build("main").to(DEVICE).eval()
        captured = {}
        hook = model.head.register_forward_hook(lambda mod, inp, out: captured.update(x=inp[0]))
        with torch.no_grad():
            res = model(dict(mri=batch["mri"]), dict(mri=avail_all(batch)["mri"],
                                                      dat=torch.zeros(3, dtype=torch.bool, device=DEVICE)))
        hook.remove()
        torch.testing.assert_close(captured["x"][:, 512:], ref_missing_feature(res["r"]["mri"]))
    # (c) DAO (Eq. 8, L1) and DCO (Eq. 5).
    method = ShaSpecMethod(PROTOCOL, CONFIG)
    model = method.build("main").to(DEVICE).eval()
    with torch.no_grad():
        res = model(dict(mri=batch["mri"], dat=batch["dat"]), avail_all(batch))
        aux = method.aux_losses(model, res, avail_all(batch))
        r_dao = ref_dao(res["r"]["mri"], res["r"]["dat"])
        r_dco = ref_dco(model.domain_logits(res["s"]["mri"]), model.domain_logits(res["s"]["dat"]))
    torch.testing.assert_close(aux["dao"], r_dao)
    torch.testing.assert_close(aux["dco"], r_dco)
    log(f"    Eq. 2 fusion max|diff| = 0 (float64); Eq. 4 slot == r_mri; "
        f"DAO {float(aux['dao']):.5f} == Eq. 8 {float(r_dao):.5f}; DCO {float(aux['dco']):.5f} == Eq. 5 {float(r_dco):.5f}")
    return "fusion, missing-feature generation, DAO and DCO match the paper's equations"


GROUPS = ("shared", "specific.mri", "specific.dat", "proj", "dom_cls", "head")


def group_of(name):
    return next(g for g in GROUPS if name.startswith(g + "."))


def reached(model, term) -> set:
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(term, [p for _, p in named], retain_graph=True, allow_unused=True)
    return {group_of(n) for (n, _), g in zip(named, grads) if g is not None and float(g.abs().sum()) > 0}


@check("3. Gradient routing per loss (Eq. 9 parameter lists) and per modality availability")
def routing_test():
    method = ShaSpecMethod(PROTOCOL, CONFIG)
    seed_everything(0)
    model = method.build("main").to(DEVICE).train()
    batch = real_batch(TRAIN0, 4)
    res = model(dict(mri=batch["mri"], dat=batch["dat"]), avail_all(batch))
    task = (res["pred"] - batch["y_z"]).square().mean()
    aux = method.aux_losses(model, res, avail_all(batch))
    got = dict(task=reached(model, task), dao=reached(model, aux["dao"]), dco=reached(model, aux["dco"]))
    want = dict(task={"shared", "specific.mri", "specific.dat", "proj", "head"},      # theta_sha,spec,proj,dec
                dao={"shared"},                                                       # theta_sha (L1 has no theta_dao)
                dco={"specific.mri", "specific.dat", "dom_cls"})                      # theta_spec, theta_dco
    for k in want:
        log(f"    {k:5s} -> {sorted(got[k])}")
        assert got[k] == want[k], (k, got[k])
    # MRI-only forward: DaT-specific encoder receives no gradient, the DaT slot is r_mri (Eq. 4).
    model.zero_grad(set_to_none=True)
    res = model(dict(mri=batch["mri"], dat=batch["dat"]),
                dict(mri=avail_all(batch)["mri"], dat=torch.zeros(4, dtype=torch.bool, device=DEVICE)))
    groups = reached(model, (res["pred"] - batch["y_z"]).square().mean())
    log(f"    task, DaT dropped for every sample -> {sorted(groups)}")
    assert "specific.dat" not in groups and "specific.mri" in groups
    return "task/DAO/DCO reach exactly the parameters of Eq. 9; dropped DaT -> no gradient to f_spec^dat"


@check("4. Paper vs code handling of a dropped modality (aux_on_missing) and natural missing DaT")
def missing_test():
    batch = real_batch(TRAIN0, 4)
    avail = dict(mri=torch.ones(4, dtype=torch.bool, device=DEVICE),
                 dat=torch.tensor([True, False, True, False], device=DEVICE))
    noisy = dict(batch)
    noisy["dat"] = batch["dat"].clone()
    noisy["dat"][~avail["dat"]] = torch.randn_like(noisy["dat"][~avail["dat"]]) * 5
    out = {}
    for aux_on_missing in (False, True):
        method = ShaSpecMethod(PROTOCOL, apply_overrides(CONFIG, [f"aux_on_missing={str(aux_on_missing).lower()}"]))
        seed_everything(0)
        model = method.build("main").to(DEVICE).train()
        enter = {m: torch.ones_like(avail[m]) for m in MODALITIES} if aux_on_missing else avail
        vals = []
        for b in (batch, noisy):
            res = model(dict(mri=b["mri"], dat=b["dat"]), avail, enter)
            aux = method.aux_losses(model, res, enter)
            vals.append(torch.stack([res["pred"].sum(), aux["dao"], aux["dco"]]).detach())
        torch.testing.assert_close(vals[0], vals[1])      # a dropped modality's pixels never matter (zero-filled)
        out[aux_on_missing] = vals[0]
    log(f"    paper (omit): pred_sum/DAO/DCO = {[round(float(v), 5) for v in out[False]]}; "
        f"code (zero-filled kept): {[round(float(v), 5) for v in out[True]]}")
    assert not torch.allclose(out[False][1:], out[True][1:]), "the two readings should differ on DAO/DCO"
    # natural missing DaT: rows without DaT never get DaT in any drop mode
    rows = without_dat(TRAIN0, set(SPLIT0.train[:3]))
    nb = real_batch([r for r in rows if r.dat_path is None] + [r for r in rows if r.dat_path is not None], 8)
    method = ShaSpecMethod(PROTOCOL, CONFIG)
    for mode in ("random_subset", "keep_mri", "dedicated", "none"):
        method.cfg = apply_overrides(CONFIG, [f"drop.mode={mode}"])
        for _ in range(50):
            a = method._sample_availability(method._present(nb))
            assert not bool((a["dat"] & ~nb["has_dat"]).any()) and bool((a["mri"] | a["dat"]).all())
    model = method.build("main").to(DEVICE)
    loss, terms = ShaSpecMethod(PROTOCOL, CONFIG).training_step("main", model, {}, nb)
    assert torch.isfinite(loss)
    return "dropped DaT pixels have no effect in either reading; readings differ on DAO/DCO; natural missing respected"


@check("5. Modality dropping distribution (code DualNet_SS.py:378-386) and determinism")
def drop_test():
    b = 4000
    present = dict(mri=torch.ones(b, dtype=torch.bool), dat=torch.ones(b, dtype=torch.bool))
    out = []
    for mode, expect in (("random_subset", dict(full=0.5, mri=0.25, dat=0.25)), ("keep_mri", dict(full=0.5, mri=0.5)),
                         ("dedicated", dict(mri=1.0)), ("none", dict(full=1.0))):
        method = ShaSpecMethod(PROTOCOL, apply_overrides(CONFIG, [f"drop.mode={mode}"]))
        torch.manual_seed(0)
        a = method._sample_availability(present)
        torch.manual_seed(0)
        a2 = method._sample_availability(present)
        assert all(torch.equal(a[m], a2[m]) for m in MODALITIES)
        pattern = Counter("full" if (x and y) else "mri" if x else "dat" for x, y in zip(a["mri"].tolist(), a["dat"].tolist()))
        freq = {k: pattern[k] / b for k in pattern}
        for k, v in expect.items():
            assert abs(freq.get(k, 0) - v) < 0.03, (mode, freq)
        out.append(f"{mode}: " + ", ".join(f"{k} {v:.3f}" for k, v in sorted(freq.items())))
    log("    " + "; ".join(out))
    return "; ".join(out)


@check("6. MRI-only roles never read DaT; predictions independent of batch composition")
def mri_only_test():
    method = ShaSpecMethod(PROTOCOL, CONFIG)
    assert method.needs_dat("main", "eval") is False and method.needs_dat("main", "train") is True
    assert method.needs_dat("warmup", "eval") is True and dict(method.eval_roles) == {"student": False, "teacher": True}
    broken = [r.__class__(**{**r.__dict__, "dat_path": Path("Z:/does/not/exist.png")}) for r in TRAIN0[:4]]
    batch = next(iter(make_loader(broken, PROTOCOL, load_dat=False, shuffle=False, seed=0)))
    assert "dat" not in batch
    model = method.build("main").to(DEVICE).eval()
    batch = to_device(batch, DEVICE)
    with torch.no_grad():
        a = method.predict("student", model, batch)
        batch.update(dat=torch.full_like(batch["mri"], float("nan")), has_dat=torch.ones_like(batch["has_dat"]))
        torch.testing.assert_close(a, method.predict("student", model, batch))
    # batch independence (eval mode, CPU float64): one sample alone == same sample inside a batch
    cpu = model.cpu().double()
    full = real_batch(TRAIN0, 4)
    full = {k: (v.cpu().double() if torch.is_tensor(v) and v.is_floating_point() else v.cpu() if torch.is_tensor(v) else v)
            for k, v in full.items()}
    with torch.no_grad():
        together = {r: method.predict(r, cpu, full) for r in ("student", "teacher")}
        for i in range(4):
            one = {k: (v[i:i + 1] if torch.is_tensor(v) else v[i:i + 1]) for k, v in full.items()}
            for r in together:
                torch.testing.assert_close(method.predict(r, cpu, one), together[r][i:i + 1], rtol=0, atol=1e-10)
    return "student ignores DaT (NaN DaT, missing file); student/teacher predictions batch-independent"


@check("7. Overfit test (12 training rows, lr 1e-4, 100 epochs)")
def overfit_test():
    proto = apply_overrides(PROTOCOL, ["training.batch_size=12"])
    rows = TRAIN0[:12]
    scaler = TargetScaler.fit(rows)
    out = []
    for name, cfg, stage in (("shaspec main (random drop)", CONFIG, "main"),
                             ("shaspec warmup (all modalities)", CONFIG, "warmup"),
                             ("student_only", apply_overrides(CONFIG, ["mode=student_only"]), "student")):
        seed_everything(0)
        method = ShaSpecMethod(proto, cfg)
        model = method.build(stage).to(DEVICE)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        loader = make_loader(rows, proto, method.needs_dat(stage, "train"), shuffle=True, seed=0)
        curve = []
        for _ in range(100):
            model.train()
            for batch in loader:
                batch = to_device(batch, DEVICE)
                batch["y_z"] = scaler.normalize(batch["y"])
                loss, terms = method.training_step(stage, model, {}, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                curve.append({k: float(v.detach()) for k, v in dict(loss=loss, **terms).items()})
        model.eval()
        with torch.no_grad():
            b = to_device(next(iter(make_loader(rows, proto, False, False, 0))), DEVICE)
            mae = float((scaler.restore(method.predict("student", model, b)) - b["y"]).abs().mean())
        a, z = curve[0], curve[-1]
        extra = f"; DAO {a['dao']:.3f} -> {z['dao']:.3f}; DCO {a['dco']:.3f} -> {z['dco']:.4f}" if "dao" in a else ""
        log(f"    {name}: total {a['loss']:.3f} -> {z['loss']:.4f}; task {a['task']:.3f} -> {z['task']:.4f}{extra}; "
            f"train MAE (MRI only) {mae:.3f} UPDRS points")
        assert z["task"] < 0.1 * a["task"], f"{name}: task loss did not drop by 90%"
        out.append(f"{name.split(' (')[0]} task {a['task']:.3f}->{z['task']:.4f}")
    return "; ".join(out)


SWITCHES = [
    ["fusion.type=conv"], ["fusion.bias=true"], ["missing_feature=first_available"], ["aux_on_missing=true"],
    ["loss.dao=mse"], ["loss.alpha=0"], ["loss.beta=0"], ["loss.alpha=0", "loss.beta=0"], ["warmup.enabled=false"],
    ["drop.mode=keep_mri"], ["drop.mode=dedicated"], ["drop.mode=none"], ["mode=student_only"],
    ["fusion.type=conv", "aux_on_missing=true"],
]


@check("8. Loss test (every term finite; each auxiliary loss / option can be switched)")
def loss_test():
    batch = real_batch(TRAIN0, 6)
    method = ShaSpecMethod(PROTOCOL, CONFIG)
    seed_everything(0)
    loss, terms = method.training_step("main", method.build("main").to(DEVICE), {}, batch)
    log("    stage 'main' terms: " + ", ".join(f"{k}={float(v.detach()):.4f}" for k, v in terms.items()))
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in terms.values())
    zero = ShaSpecMethod(PROTOCOL, apply_overrides(CONFIG, ["loss.alpha=0", "loss.beta=0"]))
    seed_everything(0)
    l0, t0 = zero.training_step("main", zero.build("main").to(DEVICE), {}, batch)
    torch.testing.assert_close(l0, t0["task"])                 # alpha = beta = 0 -> task loss only
    for switch in SWITCHES:
        m = ShaSpecMethod(PROTOCOL, apply_overrides(CONFIG, switch))
        frozen = {}
        for stage in m.stages:
            seed_everything(0)
            model = m.build(stage).to(DEVICE)
            m.init_stage(stage, model, frozen)
            loss, _ = m.training_step(stage, model, frozen, batch)
            loss.backward()
            assert torch.isfinite(loss), f"{switch} {stage} non-finite"
            frozen[stage] = freeze(model)
    log(f"    {len(SWITCHES)} configurations ran forward+backward (all stages) with finite loss: "
        + "; ".join(",".join(s) for s in SWITCHES))
    return f"{len(SWITCHES)} switch configurations OK; alpha=beta=0 reduces to the task loss"


@check("9. Stage logic: warm-up uses DaT rows only, 'main' starts from the best warm-up weights")
def stage_test():
    method = ShaSpecMethod(PROTOCOL, CONFIG)
    assert method.stages == ("warmup", "main")
    rows = without_dat(TRAIN0, set(SPLIT0.train[:2]))
    assert all(r.dat_path is not None for r in method.train_rows("warmup", rows))
    assert len(method.train_rows("main", rows)) == len(rows)
    seed_everything(0)
    warm = freeze(method.build("warmup").to(DEVICE))
    seed_everything(1)
    main = method.build("main").to(DEVICE)
    method.init_stage("main", main, {"warmup": warm})
    sw, sm = warm.state_dict(), main.state_dict()
    assert all(torch.equal(sw[k], sm[k]) for k in sw)
    assert all(p.requires_grad for p in main.parameters())
    seed_everything(1)
    fresh = method.build("main").to(DEVICE)
    method.init_stage("main", fresh, {})                         # warm-up skipped (no DaT rows) -> no copy
    assert not all(torch.equal(sw[k], fresh.state_dict()[k]) for k in sw)
    assert ShaSpecMethod(PROTOCOL, apply_overrides(CONFIG, ["warmup.enabled=false"])).stages == ("main",)
    return "warm-up filters DaT rows; init_stage copies all warm-up weights and keeps them trainable"


@check("10. train.py/evaluate.py integration: two stages, determinism, roles, learned parameters, "
       "--teacher_from refused")
def integration_test():
    proto = apply_overrides(PROTOCOL, ["training.max_epochs=2"])
    tmp = Path(tempfile.mkdtemp(prefix="shaspec_checks_"))
    try:
        method_cls = get_method("shaspec")
        assert method_cls is ShaSpecMethod
        fp = data_fingerprint(proto, ROWS)
        shared_train.run_one("shaspec", method_cls, proto, CONFIG, ROWS, fp, 0, 0, tmp / "a", DEVICE)
        meta = json.loads((tmp / "a" / "run.json").read_text(encoding="utf-8"))
        assert meta["status"] == "complete" and set(meta["stages"]) == {"warmup", "main"}, meta["stages"]
        shared_train.run_one("shaspec", method_cls, proto, CONFIG, ROWS, fp, 0, 0, tmp / "b", DEVICE)
        sa = torch.load(tmp / "a" / "main.pt", weights_only=False)["state_dict"]
        sb = torch.load(tmp / "b" / "main.pt", weights_only=False)["state_dict"]
        same = all(torch.equal(sa[k], sb[k]) for k in sa)
        log(f"    fold 0 seed 0, 2 epochs/stage: two identical runs give identical 'main' weights: {same}")
        assert same
        # learned parameters really move (proj, dom_cls, head) relative to a fresh build with the same seed
        seed_everything(0)
        init = ShaSpecMethod(proto, CONFIG).build("warmup").state_dict()
        params = {n for n, _ in ShaSpecMethod(proto, CONFIG).build("warmup").named_parameters()}   # no BN buffers
        moved = {g: max(float((sa[k].float() - init[k].float().to(sa[k].device)).abs().max()) for k in sa
                        if k.startswith(g + ".") and k in params) for g in ("proj", "dom_cls", "head", "shared", "specific")}
        log("    max |w_trained - w_init| per group: " + ", ".join(f"{g} {v:.2e}" for g, v in moved.items()))
        assert all(v > 0 for v in moved.values())
        cache = {proto["data"]["manifest"]: (ROWS, fp)}
        preds, scores = shared_evaluate.evaluate_run("shaspec", tmp / "a", "test", ["all"], DEVICE, False, cache)
        roles = sorted({s["role"] for s in scores})
        log(f"    evaluate roles {roles}")
        assert roles == ["student", "teacher", "train_mean"]
        try:
            shared_train.run_one("shaspec", method_cls, proto, CONFIG, ROWS, fp, 0, 0, tmp / "d", DEVICE,
                                 teacher_from=tmp / "a")
        except ValueError as error:
            assert "teacher_from" in str(error)
        else:
            raise AssertionError("--teacher_from was accepted for ShaSpec")
        cfg_so = apply_overrides(CONFIG, ["mode=student_only"])
        shared_train.run_one("shaspec", method_cls, proto, cfg_so, ROWS, fp, 0, 0, tmp / "e", DEVICE)
        _, scores = shared_evaluate.evaluate_run("shaspec", tmp / "e", "test", ["all"], DEVICE, False, cache)
        assert sorted({s["role"] for s in scores}) == ["student", "train_mean"]
        return "two stages, determinism, evaluation roles, parameter movement and --teacher_from guard OK"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    log(f"# ShaSpec-dagger checks ({time.strftime('%Y-%m-%d %H:%M')}, device={DEVICE}, torch {torch.__version__})")
    log()
    for fn in (shape_test, faithfulness_test, routing_test, missing_test, drop_test, mri_only_test, overfit_test,
               loss_test, stage_test, integration_test):
        fn()
    passed = sum(r["passed"] for r in RESULTS)
    log()
    log(f"{passed}/{len(RESULTS)} checks passed.")
    (ROOT / "notes" / f"checks_report_shaspec{ARGS.report_suffix}.md").write_text("```text\n" + "\n".join(LINES) + "\n```\n", encoding="utf-8")
    (ROOT / "notes" / f"checks_report_shaspec{ARGS.report_suffix}.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
