# kd_0 — Basic Teacher–Student Baseline

The teacher receives MRI + DaT. The student receives MRI only. Both use a fine-tuned 2D ResNet-18. The student is trained with true UPDRS III labels and an MSE distillation loss from the teacher.

## Run

Run these commands from the legacy directory after preparing the data and installing dependencies (see [README.md](../README.md)):

```powershell
python .\kd_0\train.py --check
python .\kd_0\train.py --device cuda
python .\kd_0\test.py --device cuda --role student
```

Use `--device cpu` if CUDA is unavailable. Training uses 3 outer folds (5/5/5), selects epochs with inner validation, and refits on each fold's 10 training subjects. Testing reports MAE and RMSE in original UPDRS III points. Student-only evaluation is the main result.

- Settings: `config.json`; model: `model.py`; losses: `loss.py`.
- Runs and checkpoints: `kd_0/runs/<run_name>/`.
- Test a specific run: `python .\kd_0\test.py --run .\kd_0\runs\<run_name> --role student`.
- Quick pipeline check (one epoch, random weights; not a research result): `python .\kd_0\train.py --smoke`.

Architecture: [architecture.png](architecture.png).
