"""Section-9 checks for DFTD-dagger (no full training). CRAD keeps its own tests/run_checks.py.

    python tests/run_checks_dftd.py      # writes notes/checks_report_dftd.md and notes/checks_report_dftd.json

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
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate as shared_evaluate  # noqa: E402
import train as shared_train  # noqa: E402
from baselines.dftd.encoders import region_vectors  # noqa: E402
from baselines.dftd.method import DFTDMethod  # noqa: E402
from baselines.dftd.mi import PairDiscriminator, RegionCritic, jsd_terms, negative_index  # noqa: E402
from data.dataset import ImageRows, collate, make_loader, to_device  # noqa: E402
from data.manifest import TargetScaler, data_fingerprint, make_split, read_manifest, rows_of  # noqa: E402
from utils.config import apply_overrides, load_yaml, resolve_path  # noqa: E402
from utils.inference import freeze  # noqa: E402
from utils.io import read_csv, write_csv  # noqa: E402
from utils.repro import choose_device, seed_everything  # noqa: E402

# P0a: --protocol / --report_suffix (defaults = previous behaviour). Check contents are unchanged.
_CLI = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_CLI.add_argument("--protocol", default="configs/protocol.yaml")
_CLI.add_argument("--report_suffix", default="", help="report -> notes/checks_report_dftd<suffix>.md/.json")
_CLI.add_argument("--report_dir", type=Path, default=ROOT / "notes", help="directory for the Markdown and JSON reports")
ARGS = _CLI.parse_args() if __name__ == "__main__" else _CLI.parse_args([])
PROTOCOL = load_yaml(Path(ARGS.protocol) if Path(ARGS.protocol).is_absolute() else ROOT / ARGS.protocol)
CONFIG = load_yaml(ROOT / "configs" / "baselines" / "dftd.yaml")
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


def real_batch(rows, n, load_dat=True):
    data = ImageRows(rows[:n], PROTOCOL, load_dat)
    batch = to_device(collate([data[i] for i in range(len(data))]), DEVICE)
    batch["y_z"] = TargetScaler.fit(rows).normalize(batch["y"])
    return batch


def without_dat(rows, subjects):
    return [r.__class__(**{**r.__dict__, "dat_path": None}) if r.subject_id in subjects else r for r in rows]


GROUPS = ("enc_a.stem", "enc_a.trunk", "enc_a.head_c", "enc_a.head_s", "enc_b.stem", "enc_b.trunk", "enc_b.head_c",
          "enc_b.head_s", "w_a", "w_b", "S_c_a", "S_c_b", "S_s_a", "S_s_b", "D_s_a", "D_s_b", "iid.student",
          "iid.t_fuse0", "iid.t_layers", "iid.t_fuse", "iid.t_head", "iid.lit")


def group_of(name):
    return next((g for g in GROUPS if name.startswith(g + ".") or name == g), name.split(".")[0])


def reached(model, term) -> set:
    """Parameter groups that receive a non-zero gradient from one loss term."""
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(term, [p for _, p in named], retain_graph=True, allow_unused=True)
    return {group_of(n) for (n, _), g in zip(named, grads) if g is not None and float(g.abs().sum()) > 0}


@check("1. Shape test (2D real NTUA batch, 3D synthetic batch) and parameter counts")
def shape_test():
    method = DFTDMethod(PROTOCOL, CONFIG)
    batch = real_batch(TRAIN0, 4)
    model = method.build("complete").to(DEVICE).eval()
    with torch.no_grad():
        ea, eb = model.enc_a(batch["mri"]), model.enc_b(batch["dat"])
        f_a, f_b = torch.cat([ea["c"], ea["s"]], 1), torch.cat([eb["c"], eb["s"]], 1)
        out = model.iid(f_a, f_b, ea["c"])
    log(f"    2D input mri {tuple(batch['mri'].shape)}, dat {tuple(batch['dat'].shape)}")
    for name, e in (("MRI", ea), ("DaT", eb)):
        log(f"    {name}: m {tuple(e['m'].shape)}, c {tuple(e['c'].shape)}, s {tuple(e['s'].shape)}, "
            f"regions r {tuple(e['r'].shape)}")
    log(f"    f_a {tuple(f_a.shape)}; c'_b {tuple(out['cb_hat'].shape)}; student o_i "
        + ", ".join(str(tuple(o.shape)) for o in out["s_feats"]) + "; teacher o_i "
        + ", ".join(str(tuple(o.shape)) for o in out["t_feats"]) + f"; y_s {tuple(out['pred_s'].shape)}, "
        f"y_t {tuple(out['pred_t'].shape)}")
    assert ea["r"].shape == (4, 16, 256) and ea["c"].shape == (4, 512) and out["pred_t"].shape == (4,)
    for role in ("student", "teacher", "teacher_imputed"):
        assert method.predict(role, model, batch).shape == (4,)
    total = sum(p.numel() for p in model.parameters())
    infer = sum(p.numel() for p in model.enc_a.parameters()) + sum(p.numel() for p in model.iid.student.parameters())
    log(f"    parameters: total {total / 1e6:.2f} M (checkpoint ~{total * 4 / 2**20:.0f} MiB); "
        f"MRI-only inference path {infer / 1e6:.2f} M")
    proto3 = apply_overrides(PROTOCOL, ["data.spatial_dims=3", "backbone.pretrained=none",
                                        "preprocessing.mri_channels=1", "preprocessing.dat_channels=1",
                                        "preprocessing.input_size=[64,64,64]"])
    m3 = DFTDMethod(proto3, CONFIG)
    model3 = m3.build("complete").to(DEVICE).eval()
    x = torch.randn(2, 1, 64, 64, 64, device=DEVICE)
    b3 = dict(mri=x, dat=torch.randn_like(x), has_dat=torch.ones(2, dtype=torch.bool, device=DEVICE),
              subject_id=["a", "b"], y_z=torch.zeros(2, device=DEVICE))
    with torch.no_grad():
        e3 = model3.enc_a(x)
        for role in ("student", "teacher", "teacher_imputed"):
            assert m3.predict(role, model3, b3).shape == (2,)
    log(f"    3D (64^3): m {tuple(e3['m'].shape)}, regions r {tuple(e3['r'].shape)} (4x4x4 = 64 as in the paper)")
    assert e3["r"].shape == (2, 64, 256)
    loss, _ = m3.training_step("complete", model3.train(), {}, b3)
    assert torch.isfinite(loss)
    return f"2D 16 regions, 3D 64 regions; {total / 1e6:.1f} M parameters"


@check("2. Region split = non-overlapping block means")
def region_test():
    m = torch.rand(2, 3, 14, 14)
    r = region_vectors(m, 4)
    edges = [0, 4, 8, 11, 14]               # torch.tensor_split(14, 4) -> 4, 4, 3, 3
    ref = torch.stack([m[:, :, edges[i]:edges[i + 1], edges[j]:edges[j + 1]].mean((2, 3))
                       for i in range(4) for j in range(4)], 1)
    torch.testing.assert_close(r, ref)
    m3 = torch.rand(1, 2, 8, 8, 8)
    assert region_vectors(m3, 4).shape == (1, 64, 2)
    torch.testing.assert_close(region_vectors(m3, 4)[0, 0], m3[0, :, :2, :2, :2].mean((1, 2, 3)))
    return "14x14 -> blocks 4,4,3,3 per axis, no overlap"


EXPECT_ALG1 = {   # Alg. 1: which formula updates which parameters
    "task_s": {"iid.student"},
    "kd_output": {"iid.student"},
    "kd_feature": {"iid.student"},
    "lit": {"iid.lit"},
    "rel": {"enc_a.stem", "enc_a.trunk", "enc_a.head_c", "enc_b.stem", "enc_b.trunk", "enc_b.head_c",
            "w_a", "w_b", "S_c_a", "S_c_b"},
    "spe_mi": {"enc_a.stem", "enc_a.trunk", "enc_a.head_s", "enc_b.stem", "enc_b.trunk", "enc_b.head_s",
               "w_a", "w_b", "S_s_a", "S_s_b"},
    "adv_bce": {"enc_a.stem", "enc_a.trunk", "enc_a.head_s", "enc_b.stem", "enc_b.trunk", "enc_b.head_s",
                "D_s_a", "D_s_b"},
}


@check("3. Gradient routing per loss term (alg1 = Alg. 1 lines 5/7/9/11; joint = Eq. 23 end-to-end)")
def routing_test():
    batch = real_batch(TRAIN0, 8)
    out = []
    for routing in ("alg1", "joint"):
        seed_everything(0)
        method = DFTDMethod(PROTOCOL, apply_overrides(CONFIG, [f"gradient_routing={routing}"]))
        model = method.build("complete").to(DEVICE).train()
        loss, terms = method.training_step("complete", model, {}, batch)
        table = {k: reached(model, terms[k]) for k in
                 ("task_s", "task_t", "kd_output", "kd_feature", "lit", "rel", "spe_mi", "adv_bce")}
        for k, v in table.items():
            log(f"    {routing:5s} {k:10s} -> {sorted(v)}")
        everything = reached(model, loss)
        missing = [g for g in GROUPS if g not in everything]
        assert not missing, f"{routing}: groups without gradient from the total loss: {missing}"
        if routing == "alg1":
            for k, groups in EXPECT_ALG1.items():
                assert table[k] == groups, f"alg1 {k}: {sorted(table[k])} != {sorted(groups)}"
            assert not any(g.startswith(("enc_", "w_", "S_", "D_")) for g in table["task_t"]), "task reached M_RAD"
        else:
            assert {"enc_a.head_c", "enc_a.head_s"} <= table["task_s"], "joint: task loss must reach the MRI encoder"
            assert "enc_a.head_c" in table["lit"] and "enc_b.head_c" in table["lit"]
        out.append(f"{routing}: every group reached by the total loss")
    return "; ".join(out) + "; alg1 matrix matches Alg. 1 exactly"


@check("4. Gradient reversal: D_s minimises its BCE, the encoder side receives the opposite gradient")
def grl_test():
    torch.manual_seed(0)
    disc = PairDiscriminator(4, 4, [8, 4])
    c, s = torch.randn(6, 4), torch.randn(6, 4, requires_grad=True)
    k, valid = negative_index([str(i) for i in range(6)], "other_subject", "cpu")
    loss = disc(c, s, k, valid, reverse_c=False)
    g_rev = torch.autograd.grad(loss, s)[0]
    s2 = s.detach().clone().requires_grad_(True)
    joint = disc.net(torch.cat([c, s2], 1)).squeeze(1)
    prod = disc.net(torch.cat([c[valid], s2[k][valid]], 1)).squeeze(1)
    plain = F.binary_cross_entropy_with_logits(joint, torch.zeros_like(joint)) + \
        F.binary_cross_entropy_with_logits(prod, torch.ones_like(prod))
    torch.testing.assert_close(g_rev, -torch.autograd.grad(plain, s2)[0])
    torch.testing.assert_close(loss, plain)


@check("5. MI estimator: Deep InfoMax JSD loss is bounded (>= 0); Eq. 4 as printed is unbounded (rule D)")
def jsd_test():
    torch.manual_seed(0)
    res = {}
    for est in ("jsd_dim", "paper_literal"):
        critic = RegionCritic(8, 4, [16, 8])
        g, r = torch.randn(16, 8), torch.randn(16, 3, 4)
        k, valid = negative_index([str(i) for i in range(16)], "any", "cpu")
        opt = torch.optim.Adam(critic.parameters(), lr=0.05)
        curve = []
        for _ in range(300):
            loss = critic(g, r, k, valid, est).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            curve.append(float(loss))
        res[est] = (curve[0], curve[-1], min(curve))
    log(f"    jsd_dim: {res['jsd_dim'][0]:.3f} -> {res['jsd_dim'][1]:.4f} (min {res['jsd_dim'][2]:.4f}); "
        f"paper_literal: {res['paper_literal'][0]:.3f} -> {res['paper_literal'][1]:.1f}")
    assert res["jsd_dim"][2] >= 0
    assert res["paper_literal"][1] < -10 and res["paper_literal"][1] < res["paper_literal"][0] - 10
    return f"jsd_dim stays >= 0; literal Eq. 4 falls to {res['paper_literal'][1]:.0f} after 300 steps"


@check("6. Negative pairs never mix rows of the same subject; no partner -> no MI term, finite loss")
def negative_test():
    ids = ["A", "A", "B", "B", "C", "A", "C", "B"]
    for _ in range(50):
        k, valid = negative_index(ids, "other_subject", "cpu")
        assert valid.all() and all(ids[j] != ids[int(k[j])] for j in range(len(ids)))
    k, valid = negative_index(["A", "A", "A"], "other_subject", "cpu")
    assert not valid.any()
    z = jsd_terms(torch.randn(3, 4, requires_grad=True), torch.randn(3, 4), valid, "jsd_dim")
    assert float(z.abs().sum()) == 0
    method = DFTDMethod(PROTOCOL, CONFIG)
    model = method.build("complete").to(DEVICE).train()
    one = [r for r in TRAIN0 if r.subject_id == TRAIN0[0].subject_id]
    batch = real_batch(one, len(one))
    loss, terms = method.training_step("complete", model, {}, batch)
    assert torch.isfinite(loss)
    return f"single-subject batch ({len(one)} rows): rel={float(terms['rel']):.1f}, loss finite"


@check("7. Stage 2: starts from stage-1 weights; M_RAD frozen (grad None, weights and BN buffers unchanged)")
def stage2_test():
    seed_everything(0)
    method = DFTDMethod(PROTOCOL, CONFIG)
    stage1 = method.build("complete").to(DEVICE).train()
    opt = torch.optim.Adam(stage1.parameters(), lr=1e-4)
    loss, _ = method.training_step("complete", stage1, {}, real_batch(TRAIN0, 8))
    loss.backward()
    opt.step()
    frozen = {"complete": freeze(stage1)}
    model = method.build("incomplete").to(DEVICE)
    method.init_stage("incomplete", model, frozen)
    for k, v in stage1.state_dict().items():
        assert torch.equal(v, model.state_dict()[k]), f"{k} not copied from stage 1"
    params = [p for p in model.parameters() if p.requires_grad]
    trainable = {group_of(n) for n, p in model.named_parameters() if p.requires_grad}
    assert all(g.startswith("iid.") for g in trainable), trainable
    opt2 = torch.optim.Adam(params, lr=1e-3)
    before = copy.deepcopy(model.state_dict())
    model.train()
    assert all(not m.training for m in model.rad_modules()), "M_RAD left eval mode"
    batch = real_batch(without_dat(TRAIN0, set(SPLIT0.train)), 8, load_dat=False)
    assert "dat" not in batch
    loss, terms = method.training_step("incomplete", model, frozen, batch)
    opt2.zero_grad()
    loss.backward()
    opt2.step()
    assert all(p.grad is None for m in model.rad_modules() for p in m.parameters())
    changed = {group_of(k) for k, v in model.state_dict().items() if not torch.equal(v, before[k])}
    assert changed and all(g.startswith("iid.") for g in changed), changed
    log("    stage-2 terms: " + ", ".join(f"{k}={float(v):.4f}" for k, v in terms.items()))
    return f"trainable groups {sorted(trainable)}; changed after one step: {sorted(changed)}"


@check("8. Overfit test (12 rows of 6 subjects, 100 epochs): full DFTD stage 1 and student-only")
def overfit_test():
    proto = apply_overrides(PROTOCOL, ["training.batch_size=12"])
    rows = TRAIN0[:12]
    scaler = TargetScaler.fit(rows)
    out = []
    for name, cfg, stage in (("dftd", CONFIG, "complete"),
                             ("student_only", apply_overrides(CONFIG, ["mode=student_only"]), "student")):
        seed_everything(0)
        method = DFTDMethod(proto, cfg)
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
        log(f"    {name}: total {a['loss']:.3f} -> {z['loss']:.3f}; student task {a['task_s']:.4f} -> "
            f"{z['task_s']:.4f}" + (f"; teacher task {a['task_t']:.4f} -> {z['task_t']:.4f}; L_LIT "
                                    f"{a['lit']:.2f} -> {z['lit']:.2f}; rel {a['rel']:.3f} -> {z['rel']:.3f}; "
                                    f"spe_mi {a['spe_mi']:.3f} -> {z['spe_mi']:.3f}; adv BCE {a['adv_bce']:.3f} "
                                    f"-> {z['adv_bce']:.3f}" if "lit" in a else "")
            + f"; train MAE {mae:.3f} UPDRS points")
        assert z["task_s"] < 0.1 * a["task_s"], f"{name}: student task loss did not drop by 90%"
        out.append(f"{name} task_s {a['task_s']:.3f}->{z['task_s']:.4f}")
    return "; ".join(out)


SWITCHES = [
    ["gradient_routing=joint"], ["regions.weights=uniform"], ["critic.negatives=any"], ["iid.fusion=linear"],
    ["kd.detach_teacher=false"], ["kd.detach_student_into_teacher=true"], ["kd.output_kd=l1"],
    ["kd.fd_layers=[3]"], ["regions.grid_dat=2"], ["regions.grid_mri=2", "regions.grid_dat=1"],
    ["weights.rel=0"], ["weights.spe=0"], ["weights.adv=0"], ["weights.lit=0"], ["kd.alpha=0"], ["kd.beta=0"],
    ["stage2.enabled=false"], ["mode=student_only"], ["critic.estimator=paper_literal"],
]


@check("9. Loss test (every term finite; each auxiliary loss / option can be switched)")
def loss_test():
    batch = real_batch(TRAIN0, 6)
    nodat = real_batch(without_dat(TRAIN0, set(SPLIT0.train)), 6, load_dat=False)
    method = DFTDMethod(PROTOCOL, CONFIG)
    seed_everything(0)
    model = method.build("complete").to(DEVICE)
    loss, terms = method.training_step("complete", model, {}, batch)
    log("    stage 'complete' terms: " + ", ".join(f"{k}={float(v):.4f}" for k, v in terms.items()))
    assert torch.isfinite(loss) and all(torch.isfinite(v) for v in terms.values())
    for switch in SWITCHES:
        m = DFTDMethod(PROTOCOL, apply_overrides(CONFIG, switch))
        frozen = {}
        for stage in m.stages:
            model = m.build(stage).to(DEVICE)
            if hasattr(m, "init_stage"):
                m.init_stage(stage, model, frozen)
            b = nodat if stage == "incomplete" else batch
            loss, _ = m.training_step(stage, model, frozen, b)
            loss.backward()
            assert torch.isfinite(loss), f"{switch} {stage} non-finite"
            frozen[stage] = freeze(model)
    log(f"    {len(SWITCHES)} configurations ran forward+backward (both stages) with finite loss: "
        + "; ".join(",".join(s) for s in SWITCHES))
    return f"{len(SWITCHES)} switch configurations OK"


@check("10. MRI-only roles (student, teacher_imputed) never read DaT")
def mri_only_test():
    method = DFTDMethod(PROTOCOL, CONFIG)
    assert method.needs_dat("complete", "eval") is False and method.needs_dat("incomplete", "eval") is False
    assert dict(method.eval_roles) == {"student": False, "teacher": True, "teacher_imputed": False}
    broken = [r.__class__(**{**r.__dict__, "dat_path": Path("Z:/does/not/exist.png")}) for r in TRAIN0[:3]]
    batch = next(iter(make_loader(broken, PROTOCOL, load_dat=False, shuffle=False, seed=0)))
    assert "dat" not in batch
    model = method.build("complete").to(DEVICE).eval()
    batch = to_device(batch, DEVICE)
    with torch.no_grad():
        a = {r: method.predict(r, model, batch) for r in ("student", "teacher_imputed")}
        batch.update(aux=batch["aux"] * 0 + 99, y=batch["y"] * 0 - 99,
                     dat=torch.full_like(batch["mri"], float("nan")))
        for r in a:
            torch.testing.assert_close(a[r], method.predict(r, model, batch))


@check("11. train.py/evaluate.py integration: stage skip on NTUA, stage 2 with simulated missing DaT, "
       "determinism, --teacher_from refused")
def integration_test():
    proto = apply_overrides(PROTOCOL, ["training.max_epochs=2"])
    tmp = Path(tempfile.mkdtemp(prefix="dftd_checks_"))
    try:
        method_cls = DFTDMethod
        fp = data_fingerprint(proto, ROWS)
        shared_train.run_one("dftd", method_cls, proto, CONFIG, ROWS, fp, 0, 0, tmp / "a", DEVICE)
        meta = json.loads((tmp / "a" / "run.json").read_text(encoding="utf-8"))
        assert meta["status"] == "complete" and meta["stages"]["incomplete"].get("skipped"), meta["stages"]
        shared_train.run_one("dftd", method_cls, proto, CONFIG, ROWS, fp, 0, 0, tmp / "b", DEVICE)
        sa = torch.load(tmp / "a" / "complete.pt", weights_only=False)["state_dict"]
        sb = torch.load(tmp / "b" / "complete.pt", weights_only=False)["state_dict"]
        same = all(torch.equal(sa[k], sb[k]) for k in sa)
        log(f"    NTUA fold 0: stage 'incomplete' skipped; two identical runs give identical weights: {same}")
        assert same
        del sa, sb
        missing = set(SPLIT0.train[:3])                   # simulate 3 training subjects without DaT
        mixed = copy.deepcopy(proto)
        manifest = tmp / "missing_dat.csv"
        manifest_rows = read_csv(resolve_path(proto["data"]["manifest"]))
        for row in manifest_rows:
            if row["subject_id"].strip() in missing:
                row["dat_path"] = ""
        write_csv(manifest, manifest_rows)
        mixed["data"]["manifest"] = str(manifest)
        rows = read_manifest(mixed)
        shared_train.run_one("dftd", method_cls, mixed, CONFIG, rows, data_fingerprint(mixed, rows),
                             0, 0, tmp / "c", DEVICE)
        meta = json.loads((tmp / "c" / "run.json").read_text(encoding="utf-8"))
        assert not meta["stages"]["incomplete"].get("skipped"), meta["stages"]
        s1 = torch.load(tmp / "c" / "complete.pt", weights_only=False)["state_dict"]
        s2 = torch.load(tmp / "c" / "incomplete.pt", weights_only=False)["state_dict"]
        rad_same = all(torch.equal(s1[k], s2[k]) for k in s1 if not k.startswith("iid."))
        iid_diff = any(not torch.equal(s1[k], s2[k]) for k in s1 if k.startswith("iid."))
        assert rad_same and iid_diff
        del s1, s2
        _, scores = shared_evaluate.evaluate_run("dftd", tmp / "c", "test", ["all"], DEVICE, False, {})
        roles = sorted({s["role"] for s in scores})
        log(f"    simulated missing DaT for {sorted(missing)}: stage 2 trained; M_RAD identical to stage 1, "
            f"M_IID changed; evaluate roles {roles}")
        assert roles == ["student", "teacher", "teacher_imputed", "train_mean"]
        try:
            shared_train.run_one("dftd", method_cls, proto, CONFIG, ROWS, fp, 0, 0, tmp / "d", DEVICE,
                                 teacher_from=tmp / "a")
        except ValueError as error:
            assert "teacher_from" in str(error)
        else:
            raise AssertionError("--teacher_from was accepted for DFTD")
        return "skip, stage 2, evaluation roles, determinism and --teacher_from guard OK"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    log(f"# DFTD-dagger checks ({time.strftime('%Y-%m-%d %H:%M')}, device={DEVICE}, torch {torch.__version__})")
    log()
    for fn in (shape_test, region_test, routing_test, grl_test, jsd_test, negative_test, stage2_test,
               overfit_test, loss_test, mri_only_test, integration_test):
        fn()
    passed = sum(r["passed"] for r in RESULTS)
    log()
    log(f"{passed}/{len(RESULTS)} checks passed.")
    ARGS.report_dir.mkdir(parents=True, exist_ok=True)
    report_name = f"checks_report_dftd{ARGS.report_suffix}"
    (ARGS.report_dir / f"{report_name}.md").write_text("```text\n" + "\n".join(LINES) + "\n```\n", encoding="utf-8")
    (ARGS.report_dir / f"{report_name}.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
