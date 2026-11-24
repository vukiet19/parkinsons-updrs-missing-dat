# PPMI OFF30 · 3D benchmark

Predict MDS-UPDRS Part III scores measured in the OFF medication state. The protocol holds out patients across five folds. Methods use the same subject splits, with 17 train, 7 validation and 6 test subjects per fold for OFF30.

## Setup

Run commands from this directory using Python 3.12. Create and activate a virtual environment. For GPU training, install the matching PyTorch and torchvision builds first. [PyTorch installation](https://pytorch.org/get-started/locally/)

```bash
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# Bash: source .venv/bin/activate
python -m pip install -r requirements.txt
python tests/run_checks_off30.py
python tests/run_checks_mri_inference.py
```

The checks use synthetic data.

## Prepare the data

1. Set `data.root` in the protocol to your local PPMI-OFF30 directory. Each subject needs `case.json`, `paired_2mm/MRI_zscore.nii.gz` and `paired_2mm/DaT_REF_ratio.nii.gz`. Images share the protocol’s 2 mm grid. `case.json` supplies same-visit OFF MDS-UPDRS III and MoCA. [Protocol](configs/protocol_ppmi.yaml)
2. Place `resnet_18_23dataset.pth` at `backbone.weights_path` and prepare the local AAL3 atlas for AGMD. The manifest release checks the required assets. [MedicalNet weights](https://github.com/Tencent/MedicalNet) · [AAL3 setup](data/atlas/README.md)
3. Use the QC preview to prepare the manifest, or reuse an existing reviewed QC CSV with `subject_id,visual_qc,reviewer,date,note`. Publish the manifest locally after review.

```bash
python scripts/make_ppmi_manifest.py --qc_preview
python scripts/make_ppmi_manifest.py --qc_csv data/manifests/ppmi_off30_qc.csv --publish
```

## Train and evaluate

```bash
python train.py --protocol configs/protocol_ppmi.yaml --method mri_only --fold 0 --seed 0 --pset training.max_epochs=2 --tag smoke
python evaluate.py --method mri_only --run_dir runs_ppmi/off30/mri_only_smoke/fold0_seed0 --split test --roles student
```

Use `mri_only`, `kd_basic`, `crad`, `cregkd`, `dftd`, `agmd` or `shaspec` for MRI-only evaluation. Use `mm_teacher` and `--roles teacher` for the MRI+DaT reference. For full runs, pass `--fold all --seed all`, use a fresh run directory and use the configured epoch budget.

The primary checkpoint is selected by validation MAE. Evaluation saves held-out predictions and metrics including MAE and RMSE in UPDRS points and R². Run `python compare.py --help` for matched comparisons. For MRI-only inference outside the evaluator, run `python export_inference.py --help` and `python infer.py --help`.

[Method implementations](baselines/) · [Configurations and tuning grids](configs/) · [Third-party notices](licenses/)
