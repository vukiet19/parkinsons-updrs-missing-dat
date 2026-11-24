"""Method registry used by the shared train.py / evaluate.py.

Every method implements the same small interface (see ``baselines/crad/method.py``):

    stages: tuple[str, ...]            # trained in order; earlier stages are frozen and handed on
    primary_role: str                  # the MRI-only model that is reported (e.g. "student")
    build(stage) -> nn.Module
    train_rows(stage, rows) -> rows    # rows usable for training this stage
    needs_dat(stage, phase) -> bool    # phase in {"train", "eval"}; eval of an MRI-only role is False
    training_step(stage, model, frozen, batch) -> (loss, {name: tensor})
    predict(stage, model, batch) -> tensor (B,)   # z-scored target units

Optional (added for methods whose teacher and student live in ONE model, e.g. DFTD, also used by ShaSpec;
CRAD uses none):

    init_stage(stage, model, frozen) -> None      # called after build(stage), before the optimiser is
                                                  # created, e.g. to warm-start from the previous stage
    eval_roles: tuple[(role, needs_dat), ...]     # evaluate.py scores these roles from the checkpoint of
                                                  # the LAST trained stage via predict(role, ...);
                                                  # without it every stage is one role (CRAD behaviour)

Optional (added for CReg-KD, also used by AGMD; CRAD falls back to the default ("teacher", "cafr", "ort")):

    teacher_sections: tuple[str, ...]             # method-config keys compared by --teacher_from
    describe(stage, model) -> list[str]           # lines that train.py writes to train.log after building a
                                                  # stage (added for AGMD, P1b)

train.py skips a stage whose ``train_rows`` is empty (logged as skipped in run.json), and refuses
``--teacher_from`` for methods without a separately trained "teacher" stage.
"""

from importlib import import_module

_REGISTRY = {
    "crad": "baselines.crad.method:CRADMethod",
    "dftd": "baselines.dftd.method:DFTDMethod",
    "cregkd": "baselines.cregkd.method:CRegKDMethod",
    "agmd": "baselines.agmd.method:AGMDMethod",
    "shaspec": "baselines.shaspec.method:ShaSpecMethod",
    "kd_basic": "baselines.kd_basic.method:KDBasicMethod",   # basic-KD anchor (STEWARD D-010, D-041)
    # Common anchors (STEWARD D-005), shared by every baseline; not baselines themselves.
    "mri_only": "baselines.anchors.method:MRIOnlyMethod",
    "mm_teacher": "baselines.anchors.method:MMTeacherMethod",
}


def get_method(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"Unknown method '{name}'. Available: {sorted(_REGISTRY)}")
    module, cls = _REGISTRY[name].split(":")
    return getattr(import_module(module), cls)
