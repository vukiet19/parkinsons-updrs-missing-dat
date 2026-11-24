# Implementation notes

The two architectures follow the approved `kd_0_vs_kd_1.drawio`. MRI and DaT images are handled by separate trainable ImageNet-initialized ResNet18 encoders. The student uses the same MRI backbone family. Residual stages 1-4 return tensors with 64/128/256/512 channels; the stem is not a fifth distillation stage.

MRI vectors are globally pooled per image and averaged within a subject. The teacher concatenates the MRI subject vector and DaT vector, then applies a 1024-to-256 fusion layer. The student applies a 512-to-256 projection. Both regressors are 256-to-64-to-1 with ReLU and dropout 0.1. No adapters, orthogonal projection, class labels, probability logits, or clinical input branches are added.

## kd_0

Let y be train-standardized UPDRS III, t the teacher prediction, and s the student prediction:

```text
Teacher: mean_subjects (t - y)^2
Student: mean_subjects (s - y)^2 + output_weight * mean_subjects (s - stop_gradient(t))^2
```

## CAFR

Each MRI stage has an image-only GAP -> Linear(C,32) -> ReLU -> Linear(32,1) head. Average image predictions within the subject. The auxiliary target is MMSE/30. The auxiliary MSE averages equally over subjects and stages. Its gradient updates the MRI encoder as well as the auxiliary heads. True MMSE is never supplied to model.forward.

This follows the paper's GAP-based intermediate supervision and the approved diagram. The repository's default `LayerScore_PFCAE_wn` instead pools the PFAC-weighted map for its auxiliary head; that code detail is not silently treated as identical to the paper diagram.

## PFAC / PF-CMAD

For feature map Z with shape [images,C,H,W]:

```text
N = L2-normalize(Z, channel dimension)
q = ReLU(spatial_mean(N) + spatial_max(N))
M = sigmoid(q^2)
A = M * Z
```

`q^2` is exactly `diag(q @ q.T)`, the expression used by CRAD's default `ChannelAttention_wn`. This avoids allocating an unnecessary C-by-C matrix. PFAC introduces zero trainable parameters; gradients still flow through it to the student encoder. Other repository variants (`wnma`, `wnma_paca`) are not mixed into this implementation.

At each of four MRI stages, compare student and detached teacher attended maps with mean squared error over channels and pixels. Average errors over images within a subject, multiply by that subject's stage confidence, then average over subjects and stages. MRI is compared only against the same MRI images and stages, never pixelwise against DaT. This uses squared feature discrepancy as in the paper; the published training script uses SmoothL1Loss instead.

## SDU for regression

Let m_k be a teacher CAFR prediction and m the true MMSE/30:

```text
layer_confidence_k = 1 - clamp((m_k - m)^2, 0, 1)
```

This is the intermediate-confidence rule of paper Eq. 14, with explicit normalized MMSE. For scalar UPDRS output, adapt confidence to regression:

```text
e_t = abs(t - y)       # train-standardized UPDRS units
e_s = abs(s - y)
confidence = 1 - clamp(e_t, 0, 1)
penalty = (e_s / (e_t + e_s + 1e-8))^2 if e_s < e_t else 1
output_confidence = confidence * penalty
```

The penalty is computed per subject rather than using the original training code's batch-mean branch. This is an explicit regression adaptation, not a claim of identical classification calibration. All confidence tensors are computed under no_grad. They are training weights, not calibrated uncertainty estimates. The teacher is fixed and used in eval mode when supervising the student.

```text
kd_1 teacher = UPDRS_MSE + cafr_weight * MMSE_MSE
kd_1 student = UPDRS_MSE
             + output_weight * mean(output_confidence * output_KD)
             + attention_weight * mean(layer_confidence * attention_KD)
```

Weights in config are fixed starting choices, not validated optimal values. Teacher predictions on its own training subjects follow ordinary teacher-student CRAD training, rather than the cross-fit targets used by the separate Ridge project. Training confidence can therefore be optimistic. Outer-test labels never set these weights, epochs, or scalers.

## Attribution

Adapted from Liu et al., **CRAD: Cognitive Aware Feature Refinement with Missing Modalities for Early Alzheimer's Progression Prediction**, Computerized Medical Imaging and Graphics 126 (2025), 102664.

- Paper DOI: https://doi.org/10.1016/j.compmedimag.2025.102664
- Official repository: https://github.com/LiuFei-AHU/CRAD
- Reference implementation: `models/crad_model.py` (`ChannelAttention_wn`, `LayerScore_PFCAE_wn`) and `train_test.py`.

The CRAD reference repository uses Apache-2.0. A copy is included at `licenses/CRAD.txt`. This project changes the image dimensionality, backbone, target, output KD, and SDU's output error calculation, and implements the approved subset CAFR/PF-CMAD/SDU. It does not assert reproduction of the full CRAD architecture or original reported performance.
