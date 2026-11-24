# NTUA · 2D benchmark

KD0 and KD1 predict UPDRS III with a 2D ResNet-18. Both use an MRI+DaT teacher and an MRI-only student. The original protocol uses three fixed folds with five patients per test fold.

- **kd_0:** supervised student regression + basic teacher–student output distillation.
- **kd_1:** kd_0 plus CAFR, PFAC/PF-CMAD attention distillation, and SDU confidence weighting.

## Data

Place the local 2D images in `dataset_224_preprocessed/` and the manifest in `data/subjects.csv` at the top level of the repository. Run commands from `benchmarks/ntua_2d/`.

`subjects.csv` must contain these columns: `subject_id`, `mri_paths`, `dat_path`, `updrs`, `mmse`, `outer_fold`, `inner_0`, `inner_1`, `inner_2`. Image paths in the CSV are relative to the repository root. Keep medical images and clinical labels private; do not commit them.

## Install dependencies

From this benchmark directory, create and activate a Python 3.12 environment. Install a PyTorch/torchvision build suitable for your CPU or CUDA setup, then install the remaining requirements.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If CUDA is required, verify that your installed PyTorch build supports it with `python -c "import torch; print(torch.cuda.is_available())"`.

## Train and evaluate

First check that the data and folds are readable.

```powershell
python .\kd_0\train.py --check
python .\kd_1\train.py --check
```

Train and test each baseline.

```powershell
python .\kd_0\train.py --device cuda
python .\kd_0\test.py --device cuda --role student

python .\kd_1\train.py --device cuda
python .\kd_1\test.py --device cuda --role student
```

Use `--device cpu` on a CPU-only machine. Training selects epochs using inner validation, then refits each outer fold on its 10 training subjects. Testing reports MAE and RMSE in the original UPDRS III score units. The student is the primary result; it uses MRI only.

Each run is saved under `kd_0/runs/` or `kd_1/runs/`. Test defaults to the latest completed non-smoke run. To evaluate a specific run, pass it explicitly, for example:

```powershell
python .\kd_1\test.py --run .\kd_1\runs\<run_name> --role student --device cuda
```

Optional pipeline check (one epoch, random initialization; **not** a research result):

```powershell
python .\kd_1\train.py --smoke
python .\kd_1\test.py --run .\kd_1\runs\<smoke_run> --role student
```

## Compare KD0 and KD1

After testing both runs, compare their held-out predictions:

```powershell
python .\compare.py --kd0 .\kd_0\runs\<kd0_run> --kd1 .\kd_1\runs\<kd1_run>
```

The comparison checks that the runs use matching data, folds and shared settings. [KD0](kd_0/readme.md) · [KD1](kd_1/readme.md) · [Method notes](method.md) · [KD1 editable diagram](docs/figures/kd1-crad-adaptation.drawio.png)
