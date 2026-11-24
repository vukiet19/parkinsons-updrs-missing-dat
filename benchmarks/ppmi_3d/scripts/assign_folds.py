"""Add/overwrite the ``fold`` column of a manifest with subject-level, target-stratified folds.

For a new dataset (e.g. PPMI). Subjects are sorted by mean target and dealt into k folds in
shuffled blocks of k, so every fold covers the whole severity range. Deterministic given --seed.

    python scripts/assign_folds.py --manifest data/manifests/ppmi.csv --k 5 --target updrs3 --seed 0
"""

import argparse
import csv
from collections import defaultdict

import numpy as np


def assign_folds(targets: dict[str, float], k: int, seed: int) -> dict[str, int]:
    """Subject -> fold. Sort by (target, id), deal blocks of k subjects, permute folds per block.

    Shared with scripts/make_ppmi_manifest.py so every manifest uses the same fold algorithm."""
    order = sorted(targets, key=lambda s: (targets[s], s))
    rng = np.random.default_rng(seed)
    fold_of = {}
    for start in range(0, len(order), k):
        block = order[start:start + k]
        for subject, fold in zip(block, rng.permutation(k)[:len(block)]):
            fold_of[subject] = int(fold)
    return fold_of


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--target", default="updrs3")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    with open(args.manifest, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    targets = defaultdict(list)
    for row in rows:
        targets[row["subject_id"]].append(float(row[args.target]))
    fold_of = assign_folds({s: np.mean(v) for s, v in targets.items()}, args.k, args.seed)
    for row in rows:
        row["fold"] = fold_of[row["subject_id"]]
    with open(args.manifest, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)
    counts = np.bincount(list(fold_of.values()), minlength=args.k)
    print(f"{len(fold_of)} subjects -> folds {counts.tolist()}")


if __name__ == "__main__":
    main()
