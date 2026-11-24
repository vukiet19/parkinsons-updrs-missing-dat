"""Delete the *.pt checkpoints of an evaluated run group to save disk space (predictions are kept).

    python scripts/prune_checkpoints.py --run_dir runs\\cregkd_sens_eta025            # dry run
    python scripts/prune_checkpoints.py --run_dir runs\\cregkd_sens_eta025 --delete   # delete
    python scripts/prune_checkpoints.py --runs_root runs_ppmi/off30 --run_dir runs_ppmi/off30/cregkd_sens_eta025 --delete

Refuses unless the group is evaluated on val and test (predictions_{val,test}.csv exist) and every run is
complete; refuses to delete a teacher.pt -- or its teacher_last.pt sibling (D-013) -- that another run
group under --runs_root reuses (run.json "reused_from"). 2D (NTUA) training is bit-exact, so a deleted
checkpoint can be regenerated with the same train.py command; 3D (PPMI) runs are close but not bit-exact.
Appends what was deleted to <run_dir>/pruned.txt.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.config import resolve_path  # noqa: E402
from utils.io import read_json  # noqa: E402


def reused_paths(runs_root: Path) -> set[Path]:
    out = set()
    for meta in runs_root.glob("*/fold*_seed*/run.json"):
        for info in read_json(meta).get("stages", {}).values():
            if isinstance(info, dict) and "reused_from" in info:
                path = resolve_path(info["reused_from"]).resolve()
                out.add(path)
                out.add(path.with_name(path.stem + "_last" + path.suffix))   # teacher_last.pt counts as teacher.pt
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--delete", action="store_true")
    p.add_argument("--runs_root", default="runs", help="folder whose run groups may reuse a teacher")
    args = p.parse_args()
    root = resolve_path(args.run_dir)
    for split in ("val", "test"):
        if not (root / f"predictions_{split}.csv").exists():
            sys.exit(f"{root}: predictions_{split}.csv missing -- run evaluate.py first.")
    runs = sorted(root.glob("fold*_seed*/run.json"))
    if not runs or any(read_json(r)["status"] != "complete" for r in runs):
        sys.exit(f"{root}: missing or incomplete runs.")
    protected = reused_paths(resolve_path(args.runs_root))
    targets = [pt for pt in sorted(root.glob("fold*_seed*/*.pt")) if pt.resolve() not in protected]
    kept = [pt for pt in sorted(root.glob("fold*_seed*/*.pt")) if pt.resolve() in protected]
    size = sum(pt.stat().st_size for pt in targets)
    print(f"{len(targets)} checkpoints, {size / 2**20:.1f} MiB" + (f"; kept (reused elsewhere): {len(kept)}" if kept else ""))
    if not args.delete:
        print("dry run; pass --delete to remove them")
        return
    for pt in targets:
        pt.unlink()
    with (root / "pruned.txt").open("a", encoding="utf-8") as fh:
        fh.write(f"{time.strftime('%Y-%m-%d %H:%M')} deleted {len(targets)} checkpoints ({size} bytes): "
                 + ", ".join(str(pt.relative_to(root)) for pt in targets) + "\n")
    print("deleted")


if __name__ == "__main__":
    main()
