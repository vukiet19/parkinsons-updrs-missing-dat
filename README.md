# Parkinson’s motor severity prediction with missing DaT imaging

Can structural MRI predict MDS-UPDRS Part III motor scores when DaT imaging is unavailable at test time? This repository contains the baselines used to study that question. Paired MRI and DaT are available during training for methods that use both modalities.

| Benchmark | What it contains |
| --- | --- |
| [PPMI OFF30 · 3D](benchmarks/ppmi_3d/README.md) | Current volumetric benchmark with seven MRI-only baselines and a paired MRI+DaT reference. Reports MAE, RMSE and R² on held-out subjects. |
| [NTUA · 2D](benchmarks/ntua_2d/README.md) | Original KD0 and KD1 slice-based baselines with three subject-level folds. |
| [Research timeline](docs/TIMELINE.md) | Reading, protocol decisions and pilot results since August 2026, in Vietnamese. |

```text
benchmarks/
├── ppmi_3d/     # Volumetric models, protocol and evaluation tools
└── ntua_2d/     # Original KD0 and KD1 experiments
docs/
└── TIMELINE.md
```

## Quick check

Use Python 3.12. Create and activate a virtual environment, then run the PPMI checks with synthetic data.

```bash
cd benchmarks/ppmi_3d
python -m pip install -r requirements.txt
python tests/run_checks_off30.py
```

Each benchmark guide covers input files, training and evaluation. Patient images, clinical tables and checkpoints are stored locally. Third-party notices are kept with each benchmark.
