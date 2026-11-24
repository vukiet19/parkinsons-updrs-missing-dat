"""Paired comparison of two evaluated methods on the same eval units (run evaluate.py first).

    python compare.py --a runs\\crad --b runs\\crad_student_only
    python compare.py --a runs\\crad --b runs\\crad --role_b train_mean
    python compare.py --a runs_ppmi/off30/mm_teacher --role_a teacher --b runs_ppmi/off30/mri_only
    python compare.py --a runs_ppmi/off30/crad --b runs_ppmi/off30/mri_only --stat hier_bootstrap
    python compare.py --group configs\\comparisons_primary_ppmi.yaml --stat hier_bootstrap --holm

--stat (STEWARD D-011, D-038):
  wilcoxon (default, unchanged numbers): per unit, absolute errors averaged over seeds; delta = MAE_A - MAE_B
      (negative favours A), two-sided Wilcoxon signed-rank test and a unit-level bootstrap 95% CI of delta.
  hier_bootstrap: |y_hat - y| per (unit, seed) for A and B (same units, same seeds); each of n_boot draws
      (RNG seed 0) resamples units with replacement, then seeds with replacement within each drawn unit;
      delta* = mean_A - mean_B on the draw. Reports observed delta, percentile 95% CI and the two-sided
      p = 2 min(P(delta* <= 0), P(delta* >= 0)), floored at 1/n_boot and capped at 1.
  nb_ttest: corrected resampled t-test (Nadeau & Bengio 2003; repeated k-fold form, Bouckaert & Frank 2004):
      d_j = MAE_A - MAE_B per (fold, seed), J = k r, t = mean(d) / sqrt((1/J + n_test/n_train) var(d, ddof=1)),
      df = J - 1; n_test / n_train (train + val subjects) are read from each run.json (test split only).
--group <yaml> [--holm]: every pair of the group with --stat; --holm adds Holm-Bonferroni adjusted p-values.

Output: <--out_dir, default parent folder of --a>/comparisons/<a>-<role>_vs_<b>-<role>_<split>_<stat>.json
(group: group_<name>_<split>_<stat>[_holm].json). The statistic is recorded in the file name and the JSON.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

from data.manifest import cohort_hash
from utils.config import load_yaml, resolve_path
from utils.io import read_csv, read_json, write_json

STATS = ("wilcoxon", "hier_bootstrap", "nb_ttest")


def load(run_root, role, split):
    root = resolve_path(run_root)
    rows = [r for r in read_csv(root / f"predictions_{split}.csv") if r["role"] == role]
    if not rows:
        raise ValueError(f"No '{role}' predictions in {root}")
    errors, truth = defaultdict(list), {}
    for r in rows:
        errors[r["unit"]].append(abs(float(r["y_pred"]) - float(r["y_true"])))
        truth[r["unit"]] = float(r["y_true"])
    runs = read_json(root / f"summary_{split}.json")["runs"]
    return root, {u: float(np.mean(v)) for u, v in errors.items()}, truth, runs


def wilcoxon_compare(a_dir, role_a, b_dir, role_b, split, n_boot=10000) -> dict:
    """The original (pre-P0c) comparison; numbers unchanged."""
    root_a, err_a, y_a, runs_a = load(a_dir, role_a, split)
    root_b, err_b, y_b, runs_b = load(b_dir, role_b, split)
    if set(err_a) != set(err_b) or any(y_a[u] != y_b[u] for u in y_a):
        raise ValueError("The two result sets do not cover the same units with the same targets.")
    if sorted(runs_a) != sorted(runs_b):
        raise ValueError(f"Different fold/seed runs: {runs_a} vs {runs_b}")
    units = sorted(err_a)
    a, b = np.array([err_a[u] for u in units]), np.array([err_b[u] for u in units])
    diff = a - b
    rng = np.random.default_rng(0)
    boot = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
    wilcoxon_p = float(stats.wilcoxon(a, b).pvalue) if np.any(diff != 0) else 1.0
    return dict(a=f"{root_a.name}:{role_a}", b=f"{root_b.name}:{role_b}", split=split,
                n_units=len(units), mae_a=float(a.mean()), mae_b=float(b.mean()), delta_mae=float(diff.mean()),
                delta_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                wilcoxon_p=wilcoxon_p, units_a_better=int((diff < 0).sum()), units_b_better=int((diff > 0).sum()),
                stat="wilcoxon", p=wilcoxon_p)


# ---- hierarchical bootstrap ------------------------------------------------------------------------------
def error_matrix(run_root, role, split) -> tuple[Path, list[str], list[int], np.ndarray]:
    """|y_hat - y| as a (unit, seed) matrix; every unit must have exactly one prediction per seed."""
    root = resolve_path(run_root)
    rows = [r for r in read_csv(root / f"predictions_{split}.csv") if r["role"] == role]
    if not rows:
        raise ValueError(f"No '{role}' predictions in {root}")
    cells = {}
    for r in rows:
        key = (r["unit"], int(float(r["seed"])))
        if key in cells:
            raise ValueError(f"{root}: several predictions for unit {key[0]} seed {key[1]}")
        cells[key] = abs(float(r["y_pred"]) - float(r["y_true"]))
    units, seeds = sorted({u for u, _ in cells}), sorted({s for _, s in cells})
    missing = [(u, s) for u in units for s in seeds if (u, s) not in cells]
    if missing:
        raise ValueError(f"{root}: incomplete unit x seed matrix, e.g. {missing[:3]}")
    return root, units, seeds, np.array([[cells[(u, s)] for s in seeds] for u in units])


def hier_bootstrap_delta(err_a: np.ndarray, err_b: np.ndarray, n_boot: int = 10000, rng_seed: int = 0) -> dict:
    """Paired two-level bootstrap of delta = MAE_A - MAE_B over (unit, seed) error matrices of equal shape."""
    if err_a.shape != err_b.shape:
        raise ValueError(f"A and B differ in units x seeds: {err_a.shape} vs {err_b.shape}")
    d = err_a - err_b
    n_units, n_seeds = d.shape
    rng = np.random.default_rng(rng_seed)
    units = rng.integers(0, n_units, size=(n_boot, n_units))
    seeds = rng.integers(0, n_seeds, size=(n_boot, n_units, n_seeds))
    boot = d[units[..., None], seeds].mean(axis=(1, 2))
    p = 2.0 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
    p = min(1.0, max(p, 1.0 / n_boot))
    return dict(delta_mae=float(d.mean()), mae_a=float(err_a.mean()), mae_b=float(err_b.mean()),
                delta_ci95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], p=p,
                n_units=n_units, n_seeds=n_seeds, n_boot=n_boot, rng_seed=rng_seed)


def hier_bootstrap_compare(a_dir, role_a, b_dir, role_b, split, n_boot=10000) -> dict:
    root_a, units_a, seeds_a, ea = error_matrix(a_dir, role_a, split)
    root_b, units_b, seeds_b, eb = error_matrix(b_dir, role_b, split)
    if units_a != units_b or seeds_a != seeds_b:
        raise ValueError(f"A and B must cover the same units and seeds: {len(units_a)}x{seeds_a} vs {len(units_b)}x{seeds_b}")
    res = hier_bootstrap_delta(ea, eb, n_boot)
    return dict(a=f"{root_a.name}:{role_a}", b=f"{root_b.name}:{role_b}", split=split, stat="hier_bootstrap",
                seeds=seeds_a, **res)


# ---- corrected resampled t-test ----------------------------------------------------------------------------
def nb_ttest_from_d(d, n_test: float, n_train: float) -> dict:
    d = np.asarray(d, dtype=np.float64)
    J = len(d)
    if J < 2:
        raise ValueError("nb_ttest needs at least two (fold, seed) differences")
    mean, var = float(d.mean()), float(d.var(ddof=1))
    denom = np.sqrt((1.0 / J + n_test / n_train) * var)
    if denom == 0:
        t, p = (0.0, 1.0) if mean == 0 else (float(np.sign(mean) * np.inf), 0.0)
    else:
        t = mean / denom
        p = float(2 * stats.t.sf(abs(t), df=J - 1))
    return dict(t=float(t), df=J - 1, p=p, J=J, mean_d=mean, var_d=var, n_test=n_test, n_train=n_train)


def _fold_seed_mae(root: Path, role: str, split: str) -> dict[tuple[int, int], float]:
    out = defaultdict(list)
    for r in read_csv(root / f"predictions_{split}.csv"):
        if r["role"] == role:
            out[(int(float(r["fold"])), int(float(r["seed"])))].append(abs(float(r["y_pred"]) - float(r["y_true"])))
    if not out:
        raise ValueError(f"No '{role}' predictions in {root}")
    return {k: float(np.mean(v)) for k, v in out.items()}


def split_sizes(root: Path) -> tuple[float, float, dict]:
    """Mean n_test and n_train (= train + val subjects) over the runs of ``root``, read from run.json."""
    sizes = {}
    for meta_path in sorted(root.glob("fold*_seed*/run.json")):
        split = read_json(meta_path)["split"]
        sizes[meta_path.parent.name] = (len(split["test"]), len(split["train"]) + len(split["val"]))
    if not sizes:
        raise ValueError(f"{root}: no run.json to read the split sizes from")
    n_test = float(np.mean([v[0] for v in sizes.values()]))
    n_train = float(np.mean([v[1] for v in sizes.values()]))
    return n_test, n_train, sizes


def nb_ttest_compare(a_dir, role_a, b_dir, role_b, split) -> dict:
    if split != "test":
        raise ValueError("nb_ttest is defined on the outer test folds (--split test)")
    root_a, root_b = resolve_path(a_dir), resolve_path(b_dir)
    ma, mb = _fold_seed_mae(root_a, role_a, split), _fold_seed_mae(root_b, role_b, split)
    if set(ma) != set(mb):
        raise ValueError(f"A and B cover different (fold, seed) runs: {sorted(ma)} vs {sorted(mb)}")
    keys = sorted(ma)
    n_test, n_train, sizes = split_sizes(root_a)
    _, _, sizes_b = split_sizes(root_b)
    if {k: v for k, v in sizes.items()} != {k: v for k, v in sizes_b.items() if k in sizes}:
        raise ValueError("A and B runs have different split sizes")
    res = nb_ttest_from_d([ma[k] - mb[k] for k in keys], n_test, n_train)
    return dict(a=f"{root_a.name}:{role_a}", b=f"{root_b.name}:{role_b}", split=split, stat="nb_ttest",
                runs=[f"fold{f}_seed{s}" for f, s in keys], d=[ma[k] - mb[k] for k in keys],
                mae_a=float(np.mean([ma[k] for k in keys])), mae_b=float(np.mean([mb[k] for k in keys])),
                delta_mae=res["mean_d"], **res)


def _off30_pair_key(root, role, split, runs):
    summary = read_json(root / f"summary_{split}.json")
    names = summary["runs"]
    if (not names or len(names) != len(set(names)) or not set(names) <= runs.keys()
            or summary.get("checkpoint", "best") != ("last" if split.endswith("_last") else "best")):
        raise ValueError(f"{root}: invalid OFF30 summary metadata.")
    part = split.removesuffix("_last")
    provenance = {}
    cohort = None
    for name in names:
        meta = runs[name]
        if "qc" not in meta.get("protocol", {}).get("data", {}) or meta["status"] != "complete":
            raise ValueError(f"{root}: comparison requires completed OFF30 runs.")
        identity = (meta["fold"], meta["seed"])
        if identity in provenance:
            raise ValueError(f"{root}: duplicate OFF30 fold/seed identity.")
        approval = meta.get("med_state", {}).get("qc_approval")
        if approval is None or approval["data_fingerprint"] != meta["data_fingerprint"]:
            raise ValueError(f"{root}: OFF30 comparison requires matching QC approval and fingerprint.")
        current = (cohort_hash(meta["protocol"]), meta["data_fingerprint"], approval)
        if cohort is None:
            cohort = current
        if current != cohort:
            raise ValueError(f"{root}: comparison group mixes OFF30 cohort identities.")
        provenance[identity] = (*current, meta["split"])

    expected = {(fold, seed, sid) for (fold, seed), (_, _, _, members) in provenance.items() for sid in members[part]}
    cells = {}
    for row in read_csv(root / f"predictions_{split}.csv"):
        if row["role"] != role:
            continue
        key = (int(row["fold"]), int(row["seed"]), row["subject_id"])
        if key in cells:
            raise ValueError(f"{root}: duplicate predictions for a subject, fold and seed.")
        cells[key] = (row["unit"], float(row["y_true"]))
    if expected != cells.keys():
        raise ValueError(f"{root}: predictions differ from the subjects in the OFF30 split.")
    return provenance, cells


def compare(a_dir, role_a, b_dir, role_b, split, stat, n_boot=10000) -> dict:
    root_a, root_b = resolve_path(a_dir), resolve_path(b_dir)
    run_maps = [{path.parent.name: read_json(path) for path in root.glob("fold*_seed*/run.json")}
                for root in (root_a, root_b)]
    if any("qc" in meta.get("protocol", {}).get("data", {}) for runs in run_maps for meta in runs.values()):
        left = _off30_pair_key(root_a, role_a, split, run_maps[0])
        right = _off30_pair_key(root_b, role_b, split, run_maps[1])
        if left != right:
            raise ValueError("Paired OFF30 results require the same approval, data, subjects, folds, seeds and targets.")
    if stat == "wilcoxon":
        return wilcoxon_compare(a_dir, role_a, b_dir, role_b, split, n_boot)
    if stat == "hier_bootstrap":
        return hier_bootstrap_compare(a_dir, role_a, b_dir, role_b, split, n_boot)
    if stat == "nb_ttest":
        return nb_ttest_compare(a_dir, role_a, b_dir, role_b, split)
    raise ValueError(f"Unknown --stat {stat}; choose from {STATS}")


def holm(pvalues) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values, in the input order."""
    p = np.asarray(pvalues, dtype=np.float64)
    m = len(p)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted.tolist()


def _summary_line(r: dict) -> str:
    extra = f"95% CI {r['delta_ci95'][0]:+.3f}, {r['delta_ci95'][1]:+.3f}; " if "delta_ci95" in r else ""
    extra += f"t = {r['t']:.3f}, J = {r['J']}, n_test/n_train = {r['n_test']:g}/{r['n_train']:g}; " if r["stat"] == "nb_ttest" else ""
    return (f"{r['a']} MAE {r['mae_a']:.3f} | {r['b']} MAE {r['mae_b']:.3f} | delta {r['delta_mae']:+.3f} "
            f"({extra}{r['stat']} p = {r['p']:.4g})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a")
    p.add_argument("--b")
    p.add_argument("--role_a", default="student")
    p.add_argument("--role_b", default="student")
    p.add_argument("--split", default="test")
    p.add_argument("--n_boot", type=int, default=10000)
    p.add_argument("--stat", default="wilcoxon", choices=STATS)
    p.add_argument("--out_dir", help="default: <parent folder of --a>/comparisons")
    p.add_argument("--group", help="YAML with name, split and pairs [{a, role_a, b, role_b}]")
    p.add_argument("--holm", action="store_true", help="Holm-Bonferroni over the --group pairs")
    args = p.parse_args()
    if args.holm and not args.group:
        p.error("--holm needs --group")
    if args.group:
        group = load_yaml(resolve_path(args.group))
        split = group.get("split", args.split)
        results = [compare(pair["a"], pair.get("role_a", "student"), pair["b"], pair.get("role_b", "student"),
                           split, args.stat, args.n_boot) for pair in group["pairs"]]
        if args.holm:
            for r, adj in zip(results, holm([r["p"] for r in results])):
                r["p_holm"] = adj
        out_dir = resolve_path(args.out_dir) if args.out_dir else resolve_path(group["pairs"][0]["a"]).parent / "comparisons"
        name = f"group_{group.get('name', Path(args.group).stem)}_{split}_{args.stat}{'_holm' if args.holm else ''}.json"
        write_json(out_dir / name, dict(group=args.group, stat=args.stat, split=split, holm=args.holm,
                                        n_comparisons=len(results), comparisons=results))
        for r in results:
            print(_summary_line(r) + (f" | Holm p = {r['p_holm']:.4g}" if args.holm else ""))
        print(f"Saved: {out_dir / name}")
        return
    if not (args.a and args.b):
        p.error("--a and --b are required without --group")
    result = compare(args.a, args.role_a, args.b, args.role_b, args.split, args.stat, args.n_boot)
    root_a, root_b = resolve_path(args.a), resolve_path(args.b)
    out_dir = resolve_path(args.out_dir) if args.out_dir else root_a.parent / "comparisons"
    out = out_dir / f"{root_a.name}-{args.role_a}_vs_{root_b.name}-{args.role_b}_{args.split}_{args.stat}.json"
    write_json(out, result)
    if args.stat == "wilcoxon":   # unchanged console line
        print(f"{result['a']} MAE {result['mae_a']:.3f} | {result['b']} MAE {result['mae_b']:.3f} | "
              f"delta {result['delta_mae']:+.3f} (95% CI {result['delta_ci95'][0]:+.3f}, {result['delta_ci95'][1]:+.3f}) | "
              f"Wilcoxon p = {result['wilcoxon_p']:.3f} | n = {result['n_units']}")
    else:
        print(_summary_line(result))
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
