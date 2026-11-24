#!/usr/bin/env bash
# CReg-KD-dagger: evaluate the main runs, then train/evaluate every sensitivity variant and prune its
# checkpoints (predictions are kept; training is deterministic). Run from benchmarks/ppmi_3d with Git Bash:
#     bash scripts/run_cregkd_variants.sh
# PowerShell equivalents of each command are in README.md (section CReg-KD-dagger).
set -euo pipefail
PY=.venv/Scripts/python.exe
LOG=runs/logs
mkdir -p "$LOG"

evaluate() {   # $1 = run group
  $PY evaluate.py --method cregkd --run_dir "$1" --split val
  $PY evaluate.py --method cregkd --run_dir "$1" --split test
}

evaluate runs/cregkd > "$LOG/cregkd_main_eval.log" 2>&1

# Student-side variants reuse the frozen teacher of runs/cregkd (identical protocol/split/teacher config).
for spec in "sens_afr_paper afr.form=paper" "sens_eta025 gate.eta=0.25" \
            "sens_hint_concat afr.teacher_hint=concat" "abl_no_gate gate.enabled=false"; do
  set -- $spec
  tag=$1; shift
  df -h /d | tail -1 >> "$LOG/cregkd_$tag.log"
  $PY train.py --method cregkd --tag "$tag" --teacher_from runs/cregkd --set "$@" >> "$LOG/cregkd_$tag.log" 2>&1
  evaluate "runs/cregkd_$tag" >> "$LOG/cregkd_$tag.log" 2>&1
  $PY scripts/prune_checkpoints.py --run_dir "runs/cregkd_$tag" --delete >> "$LOG/cregkd_$tag.log" 2>&1
done

# Teacher-side variant (G1 b): early fusion retrains the teacher.
tag=sens_teacher_early
df -h /d | tail -1 >> "$LOG/cregkd_$tag.log"
$PY train.py --method cregkd --tag "$tag" --set teacher.fusion=early >> "$LOG/cregkd_$tag.log" 2>&1
evaluate "runs/cregkd_$tag" >> "$LOG/cregkd_$tag.log" 2>&1
$PY scripts/prune_checkpoints.py --run_dir "runs/cregkd_$tag" --delete >> "$LOG/cregkd_$tag.log" 2>&1
echo "variants done"
