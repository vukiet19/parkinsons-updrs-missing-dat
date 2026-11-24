"""Summarize pooled out-of-fold test results on the approved shared OFF30 cohort.

python scripts/summarize_pilot.py [--checkpoint best|last] [--config <yaml>]
Run evaluate.py for all methods first.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.manifest import make_split, read_manifest
from utils.config import load_yaml, resolve_path
from utils.io import read_json, write_csv, write_json


def collect(entries, checkpoint="best", *, protocol):
    suffix = "_last" if checkpoint == "last" else ""
    cohort_stats = {}
    cohort = read_manifest(protocol, cohort_stats)
    expected_units = len({r.unit for r in cohort})
    approval = cohort_stats["qc_approval"]
    folds, seeds = protocol["data"]["split"]["folds"], protocol["training"]["seeds"]
    expected = {(f, s) for f in folds for s in seeds}
    splits = {fold: make_split(cohort, fold, protocol).as_dict() for fold in folds}
    rows = []
    for entry in entries:
        root = resolve_path(entry["run_dir"])
        summary = read_json(root / f"summary_test{suffix}.json")
        if summary["method"] != entry["method"] or summary["split"] != "test":
            raise ValueError(f"{root}: wrong method/split in summary")
        selected = [r for r in summary["summary"] if r["role"] == entry["role"]]
        if len(selected) != 1:
            raise ValueError(f"{root}: expected exactly one summary for role {entry['role']}")
        r = selected[0]
        runs = summary["runs"]
        if len(runs) != len(set(runs)):
            raise ValueError(f"{root}: duplicate runs in summary")
        metas = [read_json(root / run / "run.json") for run in runs]
        if not metas:
            raise ValueError(f"{root}: no source runs")
        observed = {(m["fold"], m["seed"]) for m in metas}
        if observed != expected or len(metas) != len(expected) or r["scope"] != "pooled_oof":
            raise ValueError(f"{root}: incomplete final fold x seed grid")
        if any(m["status"] != "complete" or m["method"] != entry["method"]
               or any(m["protocol"][key] != protocol[key] for key in ("name", "data", "preprocessing"))
               or m["split"] != splits[m["fold"]] or m["data_fingerprint"] != approval["data_fingerprint"]
               for m in metas):
            raise ValueError(f"{root}: inconsistent run status/method/protocol")
        if r["n_units"] != expected_units or r["n_seeds"] != len(seeds):
            raise ValueError(f"{root}: unexpected cohort size/seed count")
        rows.append(dict(name=entry["name"], method=entry["method"], role=entry["role"],
                         protocol=protocol["name"], n_units=r["n_units"], n_seeds=r["n_seeds"],
                         checkpoint=checkpoint, **{k: r[k] for k in ("mae_mean", "mae_sd", "rmse_mean", "rmse_sd")},
                         pearson_r=r["pearson_r_mean"], r2=r["r2_mean"], run_dir=str(root)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot_summary_ppmi.yaml")
    parser.add_argument("--checkpoint", choices=("best", "last"), default="best")
    args = parser.parse_args()
    config = load_yaml(resolve_path(args.config))
    rows = collect(config["entries"], args.checkpoint, protocol=load_yaml(resolve_path(config["protocol"])))
    stem = str(resolve_path(config["output"])) + ("_last" if args.checkpoint == "last" else "")
    write_csv(Path(stem + ".csv"), rows)
    note = "Descriptive metrics on the approved shared cohort. The MRI+DaT reference uses both modalities at test."
    write_json(Path(stem + ".json"), dict(note=note, rows=rows))
    lines = ["# PPMI-OFF30 pooled test results", "", note, "",
             "| Method | N units | Seeds | MAE | RMSE | R² |", "|---|---:|---:|---|---|---|"]
    lines += [f"| {r['name']} | {r['n_units']} | {r['n_seeds']} | {r['mae_mean']:.3f} ± {r['mae_sd']:.3f} | "
              f"{r['rmse_mean']:.3f} ± {r['rmse_sd']:.3f} | {r['r2']:.3f} |" for r in rows]
    Path(stem + ".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Combined descriptive table: {stem}.md")


if __name__ == "__main__":
    main()
