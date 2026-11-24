"""Common reference models shared by every baseline (STEWARD D-005; review Q13). Not baselines themselves.

mri_only   : shared ResNet-18 + linear head on MRI alone (the MRI-only floor every method must beat).
mm_teacher : two ResNet-18 encoders (MRI, DaT), GAP each, concatenate, shared head; uses DaT at test time,
             so it is an upper reference, not a missing-modality method.
"""
