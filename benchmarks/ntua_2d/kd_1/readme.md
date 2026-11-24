# kd_1 — CRAD-Inspired Baseline

This baseline keeps the kd_0 teacher (MRI + DaT) and MRI-only student, then adds three CRAD-inspired components:

- **CAFR:** predicts MMSE from teacher MRI features at four encoder stages. MMSE is an auxiliary training target, not a student input at test time.
- **PFAC / PF-CMAD:** computes parameter-free channel attention and aligns attended teacher/student MRI features at four stages.
- **SDU:** confidence-weights the attention and UPDRS output distillation losses.

This is an adaptation of CRAD for 2D images and UPDRS III regression, not a reproduction of the paper's original benchmark.

## Run

Run these commands from the legacy directory after preparing the data and installing dependencies (see [README.md](../README.md)):

```powershell
python .\kd_1\train.py --check
python .\kd_1\train.py --device cuda
python .\kd_1\test.py --device cuda --role student
```

Use `--device cpu` if CUDA is unavailable. Training uses the same 3-fold (5/5/5) protocol as kd_0. Testing reports MAE and RMSE in original UPDRS III points; the student uses MRI only.

- Settings: `config.json`; model: `model.py`; total loss: `loss.py`.
- Components: `components/cafr.py`, `components/pfac.py`, `components/attention.py`, `components/sdu.py`.
- Runs and checkpoints: `kd_1/runs/<run_name>/`.
- Test a specific run: `python .\kd_1\test.py --run .\kd_1\runs\<run_name> --role student`.
- Quick pipeline check (one epoch, random weights; not a research result): `python .\kd_1\train.py --smoke`.

Architecture: [architecture.png](architecture.png). Method formulas and source attribution: [method.md](../method.md).
