# AAL3v1 atlas for AGMD

Obtain `AAL3v1_1mm.nii.gz` from the official AAL3 distribution. See the [atlas provider](https://www.gin.cnrs.fr/en/tools/aal/) and the [AAL3 notice](../../licenses/AAL3_NOTICE.txt).

Resample the label volume with nearest-neighbor interpolation to the `preprocessing.input_size` and `preprocessing.affine` in `configs/protocol_ppmi.yaml`. Save the integer label map as `data/atlas/aal3v1_ppmi_97x115x97.nii.gz`. Run the example from `benchmarks/ppmi_3d/` with the source atlas in the working directory.

```bash
python -c "import nibabel as n, numpy as np, yaml; from nibabel.processing import resample_from_to; p=yaml.safe_load(open('configs/protocol_ppmi.yaml'))['preprocessing']; a=resample_from_to(n.load('AAL3v1_1mm.nii.gz'), (tuple(p['input_size']), p['affine']), order=0); n.save(n.Nifti1Image(np.asarray(a.dataobj, dtype=np.int16), a.affine), 'data/atlas/aal3v1_ppmi_97x115x97.nii.gz')"
```

After verifying voxel labels and alignment, use the SHA-256 of your local file as `graph.atlas_sha256` in `configs/baselines/agmd_ppmi.yaml`. The hash checks the exact file, including its header and compression. Audit anatomical alignment with the local MRI template before training AGMD. Keep the atlas image local.
