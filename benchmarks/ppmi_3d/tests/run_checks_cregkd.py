"""Section-9 checks for CReg-KD-dagger (no full training). CRAD and DFTD keep their own scripts.

    python tests/run_checks_cregkd.py    # writes notes/checks_report_cregkd.md and notes/checks_report_cregkd.json

The legacy checks use the NTUA protocol by default. Check 10 uses synthetic volumes on the
OFF30 grid and the shared MedicalNet weights without opening the OFF30 manifest or images.

Exit code 1 if any check fails.
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
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import evaluate as shared_evaluate  # noqa: E402
import train as shared_train  # noqa: E402
from baselines.cregkd.afr import AFR, phi  # noqa: E402
from baselines.cregkd.losses import gate, hint_loss, output_kd, teacher_hints  # noqa: E402
from baselines.cregkd.method import CRegKDMethod  # noqa: E402
from data.dataset import ImageRows, collate, make_loader, to_device  # noqa: E402
from data.manifest import TargetScaler, data_fingerprint, make_split, read_manifest, rows_of  # noqa: E402
from utils.config import apply_overrides, load_yaml  # noqa: E402
from utils.inference import freeze  # noqa: E402
from utils.repro import choose_device, seed_everything  # noqa: E402

# P0a: --protocol / --report_suffix (defaults = previous behaviour). Check contents are unchanged.
_CLI = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_CLI.add_argument("--protocol", default="configs/protocol.yaml")
_CLI.add_argument("--report_suffix", default="", help="report -> notes/checks_report_cregkd<suffix>.md/.json")
ARGS = _CLI.parse_args() if __name__ == "__main__" else _CLI.parse_args([])
PROTOCOL = load_yaml(Path(ARGS.protocol) if Path(ARGS.protocol).is_absolute() else ROOT / ARGS.protocol)
CONFIG = load_yaml(ROOT / "configs" / "baselines" / "cregkd.yaml")
DEVICE = choose_device("auto")
RESULTS: list[dict] = []
LINES: list[str] = []


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
            except Exception as error:  # report and continue with the other checks
                RESULTS.append(dict(name=name, passed=False, detail=repr(error), seconds=round(time.time() - start, 1)))
                log(f"[FAIL] {name}: {error!r}")
                traceback.print_exc()
        return run
    return wrap


ROWS = read_manifest(PROTOCOL)
SPLIT0 = make_split(ROWS, 0, PROTOCOL)
TRAIN0 = rows_of(ROWS, SPLIT0.train)


def real_batch(rows, load_dat=True, protocol=PROTOCOL):
    data = ImageRows(rows, protocol, load_dat)
    batch = to_device(collate([data[i] for i in range(len(data))]), DEVICE)
    batch["y_z"] = TargetScaler.fit(rows).normalize(batch["y"])
    return batch


def without_dat(rows, subjects):
    return [r.__class__(**{**r.__dict__, "dat_path": None}) if r.subject_id in subjects else r for r in rows]


def n_params(module):
    return sum(p.numel() for p in module.parameters()) / 1e6


def built(cfg=CONFIG, protocol=PROTOCOL, seed=0):
    seed_everything(seed)
    method = CRegKDMethod(protocol, cfg)
    teacher = freeze(method.build("teacher").to(DEVICE)) if method.kd_enabled else None
    student = method.build("student").to(DEVICE)
    return method, teacher, student


# ---------------------------------------------------------------------------------------------
# Literal ports of the official code, used as references (Apache-2.0, github.com/podismine/CReg-KD @ 9d9d301
# and github.com/podismine/BrainAgeReg @ 134ea1e). Kept verbatim except for indentation / argument passing.
class _OfficialRefine(nn.Module):
    """orig_repo/models/context.py:5-60 (3D, config.ab passed directly)."""

    def __init__(self, channels, ab):
        super().__init__()
        self.ab = ab
        for i, c in enumerate(channels, start=1):
            setattr(self, f"conv{i}", nn.Sequential(
                nn.Conv3d(c, c, kernel_size=3, stride=1, padding=1, bias=False, groups=c),
                nn.Conv3d(c, c, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(c)))

    def forward(self, x_t, x_s, weight):
        fea_t = x_t['fea']
        loss = 0.
        fea_s = x_s['fea']
        t1 = self.conv1(fea_s[0])
        t2 = self.conv2(fea_s[1])
        t3 = self.conv3(fea_s[2])
        t4 = self.conv4(fea_s[3])
        loss += 1 * sum([self.at_loss(f_s, f_t.detach(), weight) for f_s, f_t in zip([t1, t2, t3, t4], fea_t)])
        return loss

    def at_loss(self, f_s, f_t, weight):
        out = ((self.at(f_s) - self.at(f_t)).pow(2) * weight[:, None]).mean()
        return out

    def at(self, f):
        if self.ab == 0:
            xx = f.pow(2).mean(1).view(f.size(0), -1)
        elif self.ab == 1:
            xx = f.mean(1).pow(2).view(f.size(0), -1)
        elif self.ab == 2:
            xx = f.mean(1).view(f.size(0), -1)
        elif self.ab == 3:
            xx = f.view(f.size(0), -1)
        return xx


def _official_gate(T_pred, target, scale):
    """BrainAgeReg run.py:161-162 (regression form of Eq. 8)."""
    weight = abs(T_pred - target) / scale; weight[weight > 1] = 1
    weight = 1 - weight
    return weight


def _official_weight_func(func, x, y, weight):
    """orig_repo/run_student_gkd.py:70-75."""
    n = x.shape[0]
    ret = 0.
    for i in range(n):
        ret += func(x[i], y[i]) * weight[i]
    return ret / n


# ---------------------------------------------------------------------------------------------
@check("1. Shape test (2D real NTUA batch; 3D synthetic batch) and parameter counts")
def shape_test():
    batch = real_batch(TRAIN0[:4])
    method, teacher, student = built()
    with torch.no_grad():
        ot, os_ = teacher(batch["mri"], batch["dat"]), student(batch["mri"])
        refined = student.afr(os_["feats"])
    log(f"    2D input mri {tuple(batch['mri'].shape)}, dat {tuple(batch['dat'].shape)}")
    log(f"    teacher (late) MRI feats {[tuple(f.shape) for f in ot['feats']]}; DaT feats "
        f"{[tuple(f.shape) for f in ot['feats_dat']]}; pred {tuple(ot['pred'].shape)}")
    log(f"    student feats {[tuple(f.shape) for f in os_['feats']]}; AFR out {[tuple(f.shape) for f in refined]}; "
        f"phi {[tuple(phi(f, 'pow_avg').shape) for f in refined]}")
    for a, b in zip(os_["feats"], refined):
        assert a.shape == b.shape
    for a, b in zip(os_["feats"], ot["feats"]):
        assert a.shape == b.shape
    afr_p = n_params(student.afr)
    log(f"    parameters: teacher {n_params(teacher):.2f} M; student {n_params(student):.2f} M of which AFR "
        f"{afr_p:.3f} M (training only); MRI-only inference path {n_params(student) - afr_p:.2f} M")
    early = apply_overrides(CONFIG, ["teacher.fusion=early"])
    _, te, _ = built(early)
    with torch.no_grad():
        oe = te(batch["mri"], batch["dat"])
    log(f"    teacher (early, 6-channel input) feats {[tuple(f.shape) for f in oe['feats']]}; {n_params(te):.2f} M")
    p3 = apply_overrides(PROTOCOL, ["data.spatial_dims=3", "backbone.pretrained=none", "preprocessing.mri_channels=1",
                                    "preprocessing.dat_channels=1"])
    m3, t3, s3 = built(protocol=p3)
    mri, dat = torch.randn(2, 1, 64, 64, 64, device=DEVICE), torch.randn(2, 1, 48, 48, 48, device=DEVICE)
    with torch.no_grad():
        o3t, o3s = t3(mri, dat), s3(mri)
        r3 = s3.afr(o3s["feats"])
        h3 = teacher_hints(o3t["feats"], o3t["feats_dat"], "concat")
    log(f"    3D: mri (2,1,64^3) dat (2,1,48^3): student feats {[tuple(f.shape) for f in o3s['feats']]}; "
        f"AFR {[tuple(f.shape) for f in r3]}; concat hints {[tuple(f.shape) for f in h3]}")
    assert all(a.shape[2:] == b.shape[2:] for a, b in zip(r3, h3))
    return f"student {n_params(student) - afr_p:.2f} M at inference + AFR {afr_p:.3f} M"


@check("2. Module correctness vs the official code: AFR + phi + L_HT (context.py), gate (BrainAgeReg run.py), "
       "weight_func")
def official_test():
    seed_everything(0)
    chans = (8, 16, 32, 64)
    ours = AFR(3, chans, "code").to(DEVICE)
    out = []
    for ab, mode in enumerate(("pow_avg", "avg_pow", "mean", "raw")):
        ref = _OfficialRefine(chans, ab).to(DEVICE)
        src = list(ours.state_dict().values())
        ref.load_state_dict({k: v.clone() for k, v in zip(ref.state_dict().keys(), src)})
        fs = [torch.randn(3, c, 6, 5, 4, device=DEVICE) for c in chans]
        ft = [torch.randn(3, c, 6, 5, 4, device=DEVICE) for c in chans]
        w = torch.rand(3, device=DEVICE)
        a = ref({"fea": ft}, {"fea": fs}, w)
        b, _ = hint_loss(ours(fs), ft, w, mode, "sum")
        torch.testing.assert_close(a, b)
        out.append(f"{mode} {float(b.detach()):.4f}")
    log(f"    L_HT (ours == official Refine, same weights): {', '.join(out)}")
    # phi on [MRI || DaT] = mean of the two pow-avg maps (G2 concat)
    fm, fd = torch.randn(2, 8, 5, 5, device=DEVICE), torch.randn(2, 8, 5, 5, device=DEVICE)
    torch.testing.assert_close(phi(torch.cat([fm, fd], 1), "pow_avg"), 0.5 * (phi(fm, "pow_avg") + phi(fd, "pow_avg")))
    # gate == BrainAgeReg run.py:161-162, values in [0, 1], 0 beyond eta, no gradient path
    pt = torch.randn(64, device=DEVICE, requires_grad=True)
    y = torch.randn(64, device=DEVICE)
    for eta in (0.25, 1.0, 5.0):
        g = gate(pt, y, eta)
        torch.testing.assert_close(g, _official_gate(pt.detach().clone(), y, eta))
        assert g.min() >= 0 and g.max() <= 1 and not g.requires_grad
        assert torch.all(g[(pt - y).abs() >= eta] == 0)
    # gated output KD == weight_func(MSE)
    ps, pt2, w = torch.randn(7, device=DEVICE), torch.randn(7, device=DEVICE), torch.rand(7, device=DEVICE)
    torch.testing.assert_close((w * output_kd(ps, pt2, "mse")).mean(),
                               _official_weight_func(lambda a, b: (a - b) ** 2, ps, pt2, w))
    return "L_HT for all four phi, concat-phi identity, gate (eta 0.25/1/5) and weight_func match"


@check("3. Gradient flow per loss term (torch.autograd.grad): task, output KD, hint KD; teacher frozen")
def gradient_test():
    method, teacher, student = built()
    batch = real_batch(TRAIN0[:4])
    _, terms = method.training_step("student", student, {"teacher": teacher}, batch)
    groups = dict(encoder=list(student.encoder.parameters()), head=list(student.head.parameters()),
                  afr=list(student.afr.parameters()))
    report = []
    for term in ("task", "kd_output", "kd_hint"):
        flat = [p for g in groups.values() for p in g]
        grads = torch.autograd.grad(terms[term], flat, retain_graph=True, allow_unused=True)
        it = iter(grads)
        touched = {}
        for name, params in groups.items():
            gs = [next(it) for _ in params]
            touched[name] = any(g is not None and float(g.abs().sum()) > 0 for g in gs)
        report.append(f"{term}: " + ", ".join(f"{k}={'Y' if v else '-'}" for k, v in touched.items()))
        expected = dict(task=dict(encoder=True, head=True, afr=False),
                        kd_output=dict(encoder=True, head=True, afr=False),
                        kd_hint=dict(encoder=True, head=False, afr=True))[term]
        assert touched == expected, (term, touched)
    assert all(not p.requires_grad for p in teacher.parameters()) and not teacher.training
    assert not terms["gate_mean"].requires_grad
    for line in report:
        log(f"    {line}")
    log(f"    teacher: requires_grad False on all {sum(1 for _ in teacher.parameters())} tensors, eval mode; "
        f"gate detached")
    return "; ".join(report)


@check("4. Teacher fixed during the student stage (weights and BN buffers unchanged after 5 steps)")
def frozen_test():
    method, teacher, student = built()
    before = copy.deepcopy(teacher.state_dict())
    opt = torch.optim.Adam([p for p in student.parameters() if p.requires_grad], lr=1e-3)
    batch = real_batch(TRAIN0[:4])
    student.train()
    for _ in range(5):
        loss, _ = method.training_step("student", student, {"teacher": teacher}, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    after = teacher.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert all(p.grad is None for p in teacher.parameters())
    return f"{len(before)} teacher tensors identical; teacher .grad all None"


@check("5. Overfit test (12 rows, 100 epochs): teacher, CReg-KD student, student-only")
def overfit_test():
    proto = apply_overrides(PROTOCOL, ["training.batch_size=12"])
    rows = TRAIN0[:12]
    scaler = TargetScaler.fit(rows)
    out = []
    frozen = {}
    for name, cfg, stage in (("teacher", CONFIG, "teacher"), ("cregkd student", CONFIG, "student"),
                             ("student_only", apply_overrides(CONFIG, ["kd.enabled=false"]), "student")):
        seed_everything(0)
        method = CRegKDMethod(proto, cfg)
        model = method.build(stage).to(DEVICE)
        opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-4)
        loader = make_loader(rows, proto, method.needs_dat(stage, "train"), shuffle=True, seed=0)
        curve = []
        for _ in range(100):
            model.train()
            for batch in loader:
                batch = to_device(batch, DEVICE)
                batch["y_z"] = scaler.normalize(batch["y"])
                loss, terms = method.training_step(stage, model, frozen, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                curve.append({k: float(v.detach()) for k, v in dict(loss=loss, **terms).items()})
        if stage == "teacher":
            frozen["teacher"] = freeze(model)
        a, z = curve[0], curve[-1]
        extra = (f"; kd_output {a['kd_output']:.4f} -> {z['kd_output']:.4f}; kd_hint {a['kd_hint']:.4f} -> "
                 f"{z['kd_hint']:.4f}; gate mean {a['gate_mean']:.3f} -> {z['gate_mean']:.3f}"
                 if "kd_hint" in a else "")
        log(f"    {name}: total {a['loss']:.4f} -> {z['loss']:.4f}; task {a['task']:.4f} -> {z['task']:.4f}{extra}")
        assert z["task"] < 0.1 * a["task"], f"{name}: task loss did not drop by 90%"
        if "kd_hint" in a:
            assert z["kd_hint"] < a["kd_hint"], "hint loss did not decrease"
        out.append(f"{name} task {a['task']:.3f}->{z['task']:.4f}")
    return "; ".join(out)


SWITCHES = [
    ["gate.enabled=false"], ["gate.eta=0.25"], ["afr.enabled=false"], ["afr.form=paper"], ["afr.phi=avg_pow"],
    ["afr.phi=mean"], ["afr.phi=raw"], ["afr.teacher_hint=concat"], ["teacher.fusion=early"], ["kd.lambda_1=0"],
    ["kd.lambda_2=0"], ["kd.output_form=l1"], ["kd.reduce_layers=mean"], ["kd.layers=[3]"], ["kd.enabled=false"],
]


@check("6. Loss test (every term finite and >= 0; each option / auxiliary loss can be switched)")
def loss_test():
    batch = real_batch(TRAIN0[:4])
    method, teacher, student = built()
    loss, terms = method.training_step("student", student, {"teacher": teacher}, batch)
    assert all(torch.isfinite(v) and float(v) >= 0 for v in terms.values())
    log("    default: " + ", ".join(f"{k} {float(v):.4f}" for k, v in terms.items()))
    ok = []
    for sw in SWITCHES:
        cfg = apply_overrides(CONFIG, sw)
        m, t, s = built(cfg)
        loss, terms = m.training_step("student", s, {"teacher": t} if t is not None else {}, batch)
        loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(v) for v in terms.values()), sw
        if sw == ["kd.lambda_1=0"] or sw == ["kd.lambda_2=0"]:
            key = "kd_hint" if sw == ["kd.lambda_1=0"] else "kd_output"
            torch.testing.assert_close(loss, terms["task"] + 0.8 * terms[key])
        if sw == ["gate.enabled=false"]:
            assert float(terms["gate_mean"]) == 1.0
        ok.append(sw[0])
    tl, tt = method.training_step("teacher", built()[1].train().requires_grad_(True), {}, batch)
    assert torch.isfinite(tl)
    return f"{len(ok)} switches OK; teacher task {float(tl):.4f}"


@check("7. KD masking: rows without DaT get the task loss only; a batch without DaT gives loss == task")
def masking_test():
    method, teacher, student = built()
    rows = without_dat(TRAIN0[:4], {TRAIN0[0].subject_id})
    batch = real_batch(rows)
    n_dat = int(batch["has_dat"].sum())
    loss, terms = method.training_step("student", student, {"teacher": teacher}, batch)
    sub = {k: (v[batch["has_dat"]] if torch.is_tensor(v) and v.ndim and v.shape[0] == len(rows) else v)
           for k, v in batch.items()}
    sub["unit"] = [u for u, h in zip(batch["unit"], batch["has_dat"].tolist()) if h]
    student.eval()   # BN in eval so that the sub-batch forward is comparable row by row
    loss_a, terms_a = method.training_step("student", student, {"teacher": teacher}, batch)
    loss_b, terms_b = method.training_step("student", student, {"teacher": teacher}, sub)
    torch.testing.assert_close(terms_a["kd_output"], terms_b["kd_output"])
    torch.testing.assert_close(terms_a["kd_hint"], terms_b["kd_hint"])
    none = real_batch(without_dat(TRAIN0[:4], {r.subject_id for r in TRAIN0[:4]}))
    loss_n, terms_n = method.training_step("student", student, {"teacher": teacher}, none)
    torch.testing.assert_close(loss_n, terms_n["task"])
    return f"{n_dat}/4 rows with DaT: KD terms equal those of the DaT-only sub-batch; no-DaT batch -> task only"


@check("8. MRI-only student never reads DaT and AFR is not on the inference path")
def mri_only_test():
    method, _, student = built()
    assert method.needs_dat("student", "eval") is False and method.needs_dat("teacher", "eval") is True
    broken = [r.__class__(**{**r.__dict__, "dat_path": Path("Z:/does/not/exist.png")}) for r in TRAIN0[:3]]
    batch = next(iter(make_loader(broken, PROTOCOL, load_dat=False, shuffle=False, seed=0)))
    assert "dat" not in batch
    batch = to_device(batch, DEVICE)
    student.eval()
    with torch.no_grad():
        a = method.predict("student", student, batch)
        for p in student.afr.parameters():
            p.add_(torch.randn_like(p))
        batch.update(dat=torch.full_like(batch["mri"], float("nan")), y=batch["y"] * 0 - 99)
        b = method.predict("student", student, batch)
    torch.testing.assert_close(a, b)
    return "predict(student) unchanged with DaT = NaN, y = -99 and perturbed AFR weights"


@check("9. train.py / evaluate.py integration: determinism, roles, AFR learns, --teacher_from reuse and guard")
def integration_test():
    proto = apply_overrides(PROTOCOL, ["training.max_epochs=2"])
    tmp = Path(tempfile.mkdtemp(prefix="cregkd_checks_"))
    try:
        fp = data_fingerprint(proto, ROWS)
        shared_train.run_one("cregkd", CRegKDMethod, proto, CONFIG, ROWS, fp, 0, 0, tmp / "a" / "fold0_seed0", DEVICE)
        shared_train.run_one("cregkd", CRegKDMethod, proto, CONFIG, ROWS, fp, 0, 0, tmp / "b" / "fold0_seed0", DEVICE)
        same = True
        for stage in ("teacher", "student"):
            sa = torch.load(tmp / "a" / "fold0_seed0" / f"{stage}.pt", weights_only=False)["state_dict"]
            sb = torch.load(tmp / "b" / "fold0_seed0" / f"{stage}.pt", weights_only=False)["state_dict"]
            same &= all(torch.equal(sa[k], sb[k]) for k in sa)
        assert same, "two identical runs gave different weights"
        # every learnable AFR tensor moved away from its initialisation
        seed_everything(0)
        init = CRegKDMethod(proto, CONFIG).build("student").afr.state_dict()
        trained = {k[len("afr."):]: v for k, v in sa.items() if k.startswith("afr.")}
        moved = [k for k in init if init[k].is_floating_point() and not k.endswith("running_var")
                 and not k.endswith("running_mean") and not torch.equal(init[k], trained[k])]
        n_learn = sum(1 for k in init if init[k].is_floating_point() and "running" not in k)
        log(f"    two identical runs -> identical teacher and student weights; AFR tensors updated: "
            f"{len(moved)}/{n_learn}")
        assert len(moved) == n_learn
        cache = {}
        _, scores = shared_evaluate.evaluate_run("cregkd", tmp / "a" / "fold0_seed0", "test", ["all"], DEVICE, False, cache)
        roles = sorted({s["role"] for s in scores})
        assert roles == ["student", "teacher", "train_mean"], roles
        # --teacher_from: reuse with a student-side change, refuse with a teacher-side change
        cfg_s = apply_overrides(CONFIG, ["afr.form=paper"])
        shared_train.run_one("cregkd", CRegKDMethod, proto, cfg_s, ROWS, fp, 0, 0, tmp / "c" / "fold0_seed0", DEVICE,
                             teacher_from=tmp / "a")
        meta = json.loads((tmp / "c" / "fold0_seed0" / "run.json").read_text(encoding="utf-8"))
        assert "reused_from" in meta["stages"]["teacher"] and not (tmp / "c" / "fold0_seed0" / "teacher.pt").exists()
        _, scores_c = shared_evaluate.evaluate_run("cregkd", tmp / "c" / "fold0_seed0", "test", ["all"], DEVICE, False, cache)
        ta = [s for s in scores if s["role"] == "teacher"]
        tc = [s for s in scores_c if s["role"] == "teacher"]
        assert ta[0]["mae"] == tc[0]["mae"], "reused teacher scores differ"
        try:
            shared_train.run_one("cregkd", CRegKDMethod, proto, apply_overrides(CONFIG, ["teacher.fusion=early"]),
                                 ROWS, fp, 0, 0, tmp / "d" / "fold0_seed0", DEVICE, teacher_from=tmp / "a")
        except ValueError as error:
            assert "teacher_config" in str(error)
        else:
            raise AssertionError("--teacher_from accepted a teacher with a different fusion")
        log(f"    evaluate roles {roles}; --teacher_from reused the teacher for afr.form=paper (same teacher MAE "
            f"{ta[0]['mae']:.3f}) and refused teacher.fusion=early")
        return "determinism, roles, AFR updated, teacher reuse and guard OK"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)



@check("10. OFF30 3D: MedicalNet features, AFR, gated KD, student gradients and frozen teacher")
def off30_3d_test():
    protocol = load_yaml(ROOT / "configs" / "protocol_ppmi.yaml")
    method, teacher, student = built(protocol=protocol)
    size = tuple(protocol["preprocessing"]["input_size"])
    expected = [(64, 25, 29, 25), (128, 13, 15, 13), (256, 7, 8, 7), (512, 4, 4, 4)]
    mri = torch.randn(2, 1, *size, device=DEVICE)
    dat = torch.randn_like(mri)
    dat[1].fill_(float("nan"))
    has_dat = torch.tensor([True, False], device=DEVICE)
    eta = float(CONFIG["gate"]["eta"])
    errors = eta * torch.tensor([0.0, 0.25, 0.5, 1.0, 2.0], device=DEVICE)
    weights = gate(errors, torch.zeros_like(errors), eta)
    assert torch.isfinite(weights).all() and ((0 <= weights) & (weights <= 1)).all()
    assert (weights.diff() <= 0).all()
    torch.testing.assert_close(weights, errors.new_tensor([1, 0.75, 0.5, 0, 0]))

    student.eval()
    with torch.no_grad():
        out_t = teacher(mri[:1], dat[:1])
        out_s = student(mri)
        refined = student.afr(out_s["feats"])
        hints = teacher_hints(out_t["feats"], out_t["feats_dat"], CONFIG["afr"]["teacher_hint"])
        assert [tuple(f.shape) for f in refined] == [(2, *shape) for shape in expected]
        assert [tuple(f.shape) for f in out_s["feats"]] == [(2, *shape) for shape in expected]
        assert [tuple(f.shape) for f in hints] == [(1, *shape) for shape in expected]
        assert [tuple(phi(f, "pow_avg").shape) for f in hints] == [(1, n) for n in (18125, 2535, 392, 64)]
        assert out_t["pred"].shape == (1,) and out_s["pred"].shape == (2,)
        y_z = torch.cat((out_t["pred"] + 0.25 * eta, mri.new_tensor([-1.0])))
        batch = dict(mri=mri, dat=dat, has_dat=has_dat, y_z=y_z)
        paired = {key: value[:1] for key, value in batch.items()}
        loss, mixed = method.training_step("student", student, {"teacher": teacher}, batch)
        _, paired_terms = method.training_step("student", student, {"teacher": teacher}, paired)
        for name in ("kd_output", "kd_hint", "gate_mean", "gate_zero_frac"):
            torch.testing.assert_close(mixed[name], paired_terms[name])
        torch.testing.assert_close(mixed["gate_mean"], mri.new_tensor(0.75))
        torch.testing.assert_close(mixed["task"], (out_s["pred"] - y_z).square().mean())
        torch.testing.assert_close(loss, mixed["task"] + CONFIG["kd"]["lambda_1"] * mixed["kd_output"]
                                   + CONFIG["kd"]["lambda_2"] * mixed["kd_hint"])
        no_dat = dict(batch, has_dat=torch.zeros_like(has_dat))
        task_only, masked = method.training_step("student", student, {}, no_dat)
        torch.testing.assert_close(task_only, mixed["task"])
        assert masked["kd_output"].item() == masked["kd_hint"].item() == 0

    teacher_before = {key: value.clone() for key, value in teacher.state_dict().items()}
    student.train()
    loss, terms = method.training_step("student", student, {"teacher": teacher}, batch)
    assert torch.isfinite(loss) and all(torch.isfinite(v) and v >= 0 for v in terms.values())
    assert all(terms[name] > 0 for name in ("task", "kd_output", "kd_hint"))
    loss.backward()
    for module in (student.encoder, student.head, student.afr):
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(g.count_nonzero() > 0 for g in grads)
    torch.optim.Adam(student.parameters(), lr=protocol["training"]["lr"]).step()
    assert not teacher.training
    assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    assert any(name.endswith("running_mean") for name, _ in teacher.named_buffers())
    assert all(torch.equal(before, teacher.state_dict()[name]) for name, before in teacher_before.items())
    return (f"synthetic {size}, four AFR/hint shapes, 1/2 paired, gate monotone, Adam and frozen BN OK; "
            f"params: teacher {n_params(teacher):.3f} M, student {n_params(student):.3f} M, "
            f"AFR {n_params(student.afr):.3f} M")


def main():
    log(f"# CReg-KD-dagger checks ({time.strftime('%Y-%m-%d %H:%M')}, device={DEVICE}, torch {torch.__version__})")
    log()
    for fn in (shape_test, official_test, gradient_test, frozen_test, overfit_test, loss_test, masking_test,
               mri_only_test, integration_test, off30_3d_test):
        fn()
    passed = sum(r["passed"] for r in RESULTS)
    log()
    log(f"{passed}/{len(RESULTS)} checks passed.")
    (ROOT / "notes" / f"checks_report_cregkd{ARGS.report_suffix}.md").write_text("```text\n" + "\n".join(LINES) + "\n```\n",
                                                            encoding="utf-8")
    (ROOT / "notes" / f"checks_report_cregkd{ARGS.report_suffix}.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
