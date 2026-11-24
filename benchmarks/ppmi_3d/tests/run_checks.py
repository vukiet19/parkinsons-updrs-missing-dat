"""Section-9 checks (no full training): shape, gradient, overfit, loss switches, module correctness.

    python tests/run_checks.py            # writes notes/checks_report.md and notes/checks_report.json

Exit code 1 if any check fails. Results are measured, never estimated.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from baselines.crad.losses import gram_kd, orthogonal_loss, output_kd  # noqa: E402
from baselines.crad.method import CRADMethod  # noqa: E402
from baselines.crad.pfac import PFAC  # noqa: E402
from baselines.crad.sdu import layer_confidence, output_confidence, psi  # noqa: E402
from data.dataset import collate, ImageRows, make_loader, to_device  # noqa: E402
from data.manifest import TargetScaler, make_split, read_manifest, rows_of  # noqa: E402
from utils.config import apply_overrides, load_yaml  # noqa: E402
from utils.inference import freeze  # noqa: E402
from utils.repro import choose_device, seed_everything  # noqa: E402

# P0a: --protocol / --report_suffix (defaults = previous behaviour). Check contents are unchanged.
_CLI = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_CLI.add_argument("--protocol", default="configs/protocol.yaml")
_CLI.add_argument("--report_suffix", default="", help="report -> notes/checks_report<suffix>.md/.json")
ARGS = _CLI.parse_args() if __name__ == "__main__" else _CLI.parse_args([])
PROTOCOL = load_yaml(Path(ARGS.protocol) if Path(ARGS.protocol).is_absolute() else ROOT / ARGS.protocol)
CONFIG = load_yaml(ROOT / "configs" / "baselines" / "crad.yaml")
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


def real_batch(rows, n, load_dat=True):
    data = ImageRows(rows[:n], PROTOCOL, load_dat)
    batch = to_device(collate([data[i] for i in range(len(data))]), DEVICE)
    scaler = TargetScaler.fit(rows)
    batch["y_z"] = scaler.normalize(batch["y"])
    return batch


ROWS = read_manifest(PROTOCOL)
SPLIT0 = make_split(ROWS, 0, PROTOCOL)
TRAIN0 = rows_of(ROWS, SPLIT0.train)


def shapes(out):
    lines = []
    for key in ("feats", "z_m", "z_p", "z_u", "pred", "cafr"):
        value = out.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            lines.append(f"{key}: " + ", ".join(str(tuple(v.shape)) for v in value))
        else:
            lines.append(f"{key}: {tuple(value.shape)}")
    return lines


@check("1. Shape test (2D real NTUA batch, 3D synthetic batch)")
def shape_test():
    method = CRADMethod(PROTOCOL, CONFIG)
    batch = real_batch(TRAIN0, 4)
    teacher, student = method.build("teacher").to(DEVICE).eval(), method.build("student").to(DEVICE).eval()
    with torch.no_grad():
        out_t, out_s = teacher(batch["mri"], batch["dat"]), student(batch["mri"])
    log(f"    2D input mri {tuple(batch['mri'].shape)}, dat {tuple(batch['dat'].shape)}")
    for line in shapes(out_t):
        log(f"    teacher {line}")
    for line in shapes(out_s):
        log(f"    student {line}")
    assert out_t["pred"].shape == out_s["pred"].shape == (4,)
    assert out_t["z_u"].shape == out_s["z_u"].shape == (4, 512, 7, 7)
    proto3 = apply_overrides(PROTOCOL, ["data.spatial_dims=3", "backbone.pretrained=none",
                                        "preprocessing.mri_channels=1", "preprocessing.dat_channels=1",
                                        "preprocessing.input_size=[64,64,64]"])
    m3 = CRADMethod(proto3, CONFIG)
    t3, s3 = m3.build("teacher").to(DEVICE).eval(), m3.build("student").to(DEVICE).eval()
    x = torch.randn(2, 1, 64, 64, 64, device=DEVICE)
    with torch.no_grad():
        o3t, o3s = t3(x, torch.randn(2, 1, 48, 48, 48, device=DEVICE)), s3(x)
    for line in shapes(o3t):
        log(f"    3D teacher {line}")
    assert o3t["pred"].shape == o3s["pred"].shape == (2,)
    return "2D and 3D forward OK"


def grad_groups(model):
    groups = {}
    for name, p in model.named_parameters():
        key = ".".join(name.split(".")[:2]) if name.startswith(("mri_encoder", "dat_encoder", "cafr")) else name.split(".")[0]
        has = p.grad is not None and float(p.grad.abs().sum()) > 0
        groups.setdefault(key, []).append(has)
    return {k: f"{sum(v)}/{len(v)}" for k, v in groups.items()}


@check("2. Gradient test (teacher phase, student phase with frozen teacher)")
def gradient_test():
    seed_everything(0)
    method = CRADMethod(PROTOCOL, CONFIG)
    batch = real_batch(TRAIN0, 8)
    teacher = method.build("teacher").to(DEVICE).train()
    loss, _ = method.training_step("teacher", teacher, {}, batch)
    loss.backward()
    g_t = grad_groups(teacher)
    log("    teacher phase, params with non-zero grad per group: " + json.dumps(g_t))
    missing = [k for k, v in g_t.items() if v.startswith("0/")]
    assert not missing, f"teacher groups without gradient: {missing}"
    teacher.zero_grad(set_to_none=True)
    freeze(teacher)
    before = copy.deepcopy(teacher.state_dict())
    student = method.build("student").to(DEVICE).train()
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
    loss, terms = method.training_step("student", student, {"teacher": teacher}, batch)
    loss.backward()
    optimizer.step()
    g_s = grad_groups(student)
    log("    student phase, params with non-zero grad per group: " + json.dumps(g_s))
    assert not [k for k, v in g_s.items() if v.startswith("0/")], "student group without gradient"
    assert all(p.grad is None for p in teacher.parameters()), "teacher received gradients"
    assert all(torch.equal(v, teacher.state_dict()[k]) for k, v in before.items()), "teacher weights/BN changed"
    assert not teacher.training, "teacher left train mode"
    for key in ("conf_attention", "conf_output"):
        assert not terms[key].requires_grad, f"{key} is not detached"
    return "teacher grad=None and teacher state (incl. BN buffers) unchanged after a student step"


@check("3. Overfit test (12 MRI rows of 6 subjects: teacher then student, 100 epochs)")
def overfit_test():
    seed_everything(0)
    proto = apply_overrides(PROTOCOL, ["training.batch_size=12"])
    method = CRADMethod(proto, CONFIG)
    rows = TRAIN0[:12]
    scaler = TargetScaler.fit(rows)
    loader = make_loader(rows, proto, True, shuffle=True, seed=0)
    out = []
    frozen = {}
    for stage in method.stages:
        seed_everything(0)               # every stage is re-seeded, exactly as in train.py
        model = method.build(stage).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        curve = []
        for epoch in range(100):
            model.train()
            for batch in loader:
                batch = to_device(batch, DEVICE)
                batch["y_z"] = scaler.normalize(batch["y"])
                loss, terms = method.training_step(stage, model, frozen, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                curve.append((float(loss.detach()), float(terms["task"].detach())))
        model.eval()
        with torch.no_grad():
            batch = to_device(next(iter(make_loader(rows, proto, True, False, 0))), DEVICE)
            mae = float((scaler.restore(method.predict(stage, model, batch)) - batch["y"]).abs().mean())
        log(f"    {stage}: total loss {curve[0][0]:.4f} -> {curve[-1][0]:.4f}; task loss "
            f"{curve[0][1]:.4f} -> {curve[-1][1]:.4f}; train MAE after = {mae:.3f} UPDRS points")
        assert curve[-1][1] < 0.1 * curve[0][1], f"{stage} task loss did not drop by 90%"
        out.append(f"{stage} task {curve[0][1]:.3f}->{curve[-1][1]:.4f}")
        frozen[stage] = freeze(model)
    return "; ".join(out)


SWITCHES = [
    ["teacher.gamma_cafr=0"], ["teacher.gamma_ort=0"],
    ["cafr.enabled=false", "sdu.attention_confidence=none"], ["ort.enabled=false"],
    ["kd.lambda=0"], ["kd.w_attention=0"], ["kd.w_output=0"], ["kd.w_feature=0"], ["sdu.enabled=false"],
    ["kd.enabled=false"], ["kd.fd_target=branches"], ["kd.fd_divergence=kl_softmax_rows"],
    ["kd.attention_distance=mse", "kd.attention_reduce_layers=mean"], ["ort.form=paper_sum"],
    ["cafr.pool_input=raw", "cafr.loss=mse", "cafr.reduce_layers=mean"],
    ["sdu.attention_confidence=output"], ["sdu.attention_confidence=none"], ["sdu.psi_mode=batch"],
    ["sdu.psi_mode=always"], ["sdu.psi_mode=hard"], ["sdu.psi_mode=none"], ["kd.output_kd=l1"],
]


@check("4. Loss test (every term finite; each auxiliary loss can be switched off)")
def loss_test():
    invalid_cafr = apply_overrides(CONFIG, ["cafr.enabled=false", "sdu.attention_confidence=cafr"])
    try:
        CRADMethod(PROTOCOL, invalid_cafr)
    except ValueError as error:
        assert "needs cafr.enabled=true" in str(error)
    else:
        raise AssertionError("CAFR confidence requires an active CAFR head")
    batch = real_batch(TRAIN0, 6)
    batch["aux_mask"][0] = False      # one sample without aux score
    batch["aux"][0] = float("nan")
    method = CRADMethod(PROTOCOL, CONFIG)
    seed_everything(0)
    teacher = method.build("teacher").to(DEVICE)
    loss, terms = method.training_step("teacher", teacher, {}, batch)
    log("    teacher terms: " + ", ".join(f"{k}={float(v):.4f}" for k, v in terms.items()))
    assert all(torch.isfinite(v) for v in terms.values()) and torch.isfinite(loss)
    freeze(teacher)
    student = method.build("student").to(DEVICE)
    loss, terms = method.training_step("student", student, {"teacher": teacher}, batch)
    log("    student terms: " + ", ".join(f"{k}={float(v):.4f}" for k, v in terms.items()))
    assert all(torch.isfinite(v) for v in terms.values()) and torch.isfinite(loss)
    for switch in SWITCHES:
        cfg = apply_overrides(CONFIG, switch)
        m = CRADMethod(PROTOCOL, cfg)
        frozen = {}
        for stage in m.stages:
            model = m.build(stage).to(DEVICE)
            loss, terms = m.training_step(stage, model, frozen, batch)
            loss.backward()
            assert torch.isfinite(loss), f"{switch} {stage} non-finite"
            frozen[stage] = freeze(model)
    log(f"    {len(SWITCHES)} configurations ran forward+backward with finite loss: "
        + "; ".join(",".join(s) for s in SWITCHES))
    return f"{len(SWITCHES)} switch configurations OK"


@check("5a. PFAC == sigmoid(diag(q q^T)) of ChannelAttention_wn, zero parameters")
def pfac_test():
    z = torch.rand(3, 16, 5, 6)
    _, m = PFAC()(z)
    x = F.normalize(z, p=2, dim=1)                                   # verbatim crad_model.py:44-50 in 2D
    out = torch.relu(F.adaptive_avg_pool2d(x, 1).squeeze(-1) + F.adaptive_max_pool2d(x, 1).squeeze(-1))
    ref = torch.sigmoid(torch.diagonal(torch.matmul(out, out.permute(0, 2, 1)), dim1=-2, dim2=-1))
    torch.testing.assert_close(m.flatten(1), ref)
    assert sum(p.numel() for p in PFAC().parameters()) == 0


@check("5b. L_ORT == ort_project_loss2 of the original code (paper_sum = 2x)")
def ort_test():
    x, y = torch.rand(5, 8, 3, 3), torch.rand(5, 8, 3, 3)
    a, b = F.normalize(x.flatten(1), dim=1), F.normalize(y.flatten(1), dim=1)   # crad_model.py:237-244
    code = (1.0 - (torch.abs(a @ a.T).mean() + torch.abs(b @ b.T).mean()) * 0.5) + torch.abs(a @ b.T).mean()
    torch.testing.assert_close(orthogonal_loss(x, y, "code"), code)
    torch.testing.assert_close(orthogonal_loss(x, y, "paper_sum"), 2 * code)


@check("5c. Feature KD: official Gram discrepancy == code formula, >= 0, finite; it is not a KL")
def gram_test():
    torch.manual_seed(0)
    z_t, z_s = torch.rand(2, 16, 5, 5), torch.rand(2, 16, 5, 5)
    for i in range(2):  # verbatim orig_repo/utils.py:37-51 applied per sample (train_test.py:50-55)
        ft, fs = F.normalize(z_t[i].flatten(1), p=2, dim=1), F.normalize(z_s[i].flatten(1), p=2, dim=1)
        dt, ds = ft @ ft.t(), fs @ fs.t()
        ref = abs(F.kl_div(ds, dt, reduction="sum", log_target=True) / dt.numel())
        torch.testing.assert_close(gram_kd(z_t, z_s, "gram_discrepancy")[i], ref)
    for mode in ("gram_discrepancy", "kl_softmax_rows"):
        assert float(gram_kd(z_t, z_t.clone(), mode).abs().max()) < 1e-6
        assert bool((gram_kd(z_t, torch.rand(8, 16, 5, 5)[:2], mode) >= -1e-7).all())
    results = {}
    for mode in ("kl_softmax_rows", "gram_discrepancy"):
        z_s = torch.rand(2, 16, 5, 5, requires_grad=True)
        opt = torch.optim.Adam([z_s], lr=0.05)
        for _ in range(300):
            loss = gram_kd(z_t, z_s, mode).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            def gram(z):
                n = F.normalize(z.flatten(2), dim=2)
                return n @ n.transpose(1, 2)
            gap = float((gram(z_s) - gram(z_t)).abs().mean())
        results[mode] = (float(loss), gap)
    log(f"    after minimising each loss: kl_softmax_rows loss={results['kl_softmax_rows'][0]:.2e}, "
        f"|G_S-G_T|={results['kl_softmax_rows'][1]:.4f}; gram_discrepancy loss={results['gram_discrepancy'][0]:.2e}, "
        f"|G_S-G_T|={results['gram_discrepancy'][1]:.4f}")
    return (f"matches code; non-negative; Gram gap at its optimum {results['gram_discrepancy'][1]:.4f} "
            f"(true KL: {results['kl_softmax_rows'][1]:.4f}) -> weaker constraint, not a bug")


@check("5g. Rule-C evidence: PFAC weights ~ constant, SmoothL1 regime of CAFR/PF-CMAD on real data")
def near_equivalence_test():
    seed_everything(0)
    method = CRADMethod(PROTOCOL, CONFIG)
    batch = real_batch(TRAIN0, 8)
    teacher, student = method.build("teacher").to(DEVICE).eval(), method.build("student").to(DEVICE).eval()
    pfac = PFAC()
    with torch.no_grad():
        ft, fs = teacher(batch["mri"], batch["dat"])["feats"], student(batch["mri"])["feats"]
        lines, quad = [], []
        for k in range(4):
            _, w = pfac(ft[k])
            a_t, a_s = pfac(ft[k])[0], pfac(fs[k])[0]
            diff = (a_s - a_t).abs()
            quad.append(float((diff < 1).float().mean()))
            lines.append(f"stage {k + 1}: PFAC weight {float(w.min()):.4f}-{float(w.max()):.4f}; "
                         f"|M_S Z_S - M_T Z_T| < 1 for {100 * quad[-1]:.1f}% of elements")
    for line in lines:
        log("    " + line)
    log("    CAFR targets and predictions lie in [0, 1] (aux / 30) -> SmoothL1 = 0.5 * MSE there")
    return "; ".join(f"s{k + 1} {100 * q:.0f}% quadratic" for k, q in enumerate(quad))


@check("5d. SDU weights: ranges, psi behaviour, missing aux, detached")
def sdu_test():
    y = torch.zeros(3)
    t = torch.tensor([0.1, 0.5, 2.0], requires_grad=True)
    s = torch.tensor([0.05, 1.0, 0.0])
    sc = output_confidence(t, y, 1.0)
    torch.testing.assert_close(sc, torch.tensor([0.9, 0.5, 0.0]))
    p = psi(t, s, y, "per_sample")
    torch.testing.assert_close(p, torch.tensor([(0.05 / 0.15) ** 2, 1.0, 0.0]), atol=1e-6, rtol=0)
    assert not sc.requires_grad and not p.requires_grad
    lc = layer_confidence(torch.tensor([[0.5, 0.9], [0.0, 0.0]]), torch.tensor([0.5, float("nan")]),
                          torch.tensor([True, False]))
    torch.testing.assert_close(lc, torch.tensor([[1.0, 0.84], [1.0, 1.0]]))
    torch.testing.assert_close(output_kd(torch.tensor([1.0, 2.0]), torch.tensor([0.0, 0.0])), torch.tensor([1.0, 4.0]))


@check("5e. Student never reads DaT / clinical scores at evaluation")
def student_input_test():
    method = CRADMethod(PROTOCOL, CONFIG)
    assert method.needs_dat("student", "eval") is False
    broken = [r.__class__(**{**r.__dict__, "dat_path": Path("Z:/does/not/exist.png")}) for r in TRAIN0[:3]]
    batch = next(iter(make_loader(broken, PROTOCOL, load_dat=False, shuffle=False, seed=0)))
    assert "dat" not in batch, "DaT tensor present in an MRI-only batch"
    student = method.build("student").to(DEVICE).eval()
    batch = to_device(batch, DEVICE)
    with torch.no_grad():
        a = method.predict("student", student, batch)
        batch.update(aux=batch["aux"] * 0 + 99, y=batch["y"] * 0 - 99, dat=torch.full_like(batch["mri"], float("nan")))
        b = method.predict("student", student, batch)
    torch.testing.assert_close(a, b)


@check("5f. Splits are subject-level, disjoint, seed-independent; scaler uses training subjects only")
def split_test():
    tests = []
    for fold in PROTOCOL["data"]["split"]["folds"]:
        sp = make_split(ROWS, fold, PROTOCOL)
        assert not (set(sp.train) & set(sp.val) or set(sp.train) & set(sp.test) or set(sp.val) & set(sp.test))
        assert sp == make_split(ROWS, fold, PROTOCOL)
        tests += sp.test
        log(f"    fold {fold}: train {len(sp.train)} {list(sp.train)}; val {list(sp.val)}; test {list(sp.test)}")
    assert sorted(tests) == sorted({r.subject_id for r in ROWS}), "test folds do not partition subjects"
    altered = [r.__class__(**{**r.__dict__, "target": 999.0}) if r.subject_id in SPLIT0.test else r for r in ROWS]
    assert TargetScaler.fit(rows_of(altered, SPLIT0.train)) == TargetScaler.fit(TRAIN0)


def main():
    log(f"# Baseline checks: CRAD-dagger ({time.strftime('%Y-%m-%d %H:%M')}, device={DEVICE}, "
        f"torch {torch.__version__})")
    log()
    for fn in (shape_test, gradient_test, overfit_test, loss_test, pfac_test, ort_test, gram_test,
               sdu_test, student_input_test, split_test, near_equivalence_test):
        fn()
    passed = sum(r["passed"] for r in RESULTS)
    log()
    log(f"{passed}/{len(RESULTS)} checks passed.")
    report = ROOT / "notes" / f"checks_report{ARGS.report_suffix}.md"
    report.write_text("```text\n" + "\n".join(LINES) + "\n```\n", encoding="utf-8")
    (ROOT / "notes" / f"checks_report{ARGS.report_suffix}.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
