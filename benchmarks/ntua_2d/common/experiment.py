"""Fixed 3-fold evaluation with epoch selection inside each outer training set."""

import importlib
import platform
import statistics
from dataclasses import asdict
from pathlib import Path

import torch
import torchvision

from .data import ROOT, TargetScale, data_fingerprint, read_subjects
from .fit import fit_model, freeze_teacher
from .utils import new_directory, source_hash, write_csv, write_json


def baseline_modules(baseline):
    models = importlib.import_module(f'{baseline}.model')
    objective = importlib.import_module(f'{baseline}.loss').Loss
    return models, objective


def save_checkpoint(path, model, role, baseline, config, fold, train, test, scale, fingerprint, epochs):
    payload = dict(format_version=1, baseline=baseline, role=role, outer_fold=fold,
                   config=config, scale=asdict(scale), data_fingerprint=fingerprint,
                   train_subjects=[s.subject_id for s in train],
                   test_subjects=[s.subject_id for s in test], epochs=epochs,
                   state_dict={key: value.detach().cpu() for key, value in model.state_dict().items()})
    torch.save(payload, path)


def run_training(baseline, config, device, output=None, folds=(0, 1, 2), smoke=False):
    subjects = read_subjects()
    fingerprint = data_fingerprint(subjects)
    models, objective = baseline_modules(baseline)
    run = new_directory(ROOT / baseline / 'runs', output, 'smoke' if smoke else 'train')
    metadata = dict(format_version=1, baseline=baseline, status='running',
                    purpose='smoke' if smoke else 'experiment', config=config, folds=list(folds),
                    data_fingerprint=fingerprint, source_hash=source_hash(ROOT),
                    python=platform.python_version(), torch=str(torch.__version__),
                    torchvision=str(torchvision.__version__), device=str(device))
    write_json(run / 'run.json', metadata)
    history, selections = [], []

    # Progress is mirrored to a small log so long runs remain easy to inspect.
    with (run / 'train.log').open('w', encoding='utf-8', buffering=1) as file:
        def log(message):
            print(message, flush=True)
            file.write(message + '\n')

        log(f'{baseline}: {len(subjects)} subjects; device={device}; output={run}')
        try:
            for outer in folds:
                train = [s for s in subjects if s.outer_fold != outer]
                test = [s for s in subjects if s.outer_fold == outer]
                folder = run / f'fold_{outer}'
                folder.mkdir()
                # Outer-test scores never participate in training or epoch selection.
                write_json(folder / 'split.json', dict(
                    train=[s.subject_id for s in train], test=[s.subject_id for s in test]))
                teacher_epochs, student_epochs = [], []
                for inner in range(config['inner_folds']):
                    inner_train = [s for s in train if s.inner_folds[outer] != inner]
                    validation = [s for s in train if s.inner_folds[outer] == inner]
                    assert set(s.subject_id for s in inner_train).isdisjoint(s.subject_id for s in validation)
                    scale = TargetScale.fit(inner_train)
                    seed = config['seed'] + outer * 100 + inner * 10
                    context = dict(stage='selection', outer_fold=outer, inner_fold=inner)
                    log(f'Outer {outer}, inner {inner}: {len(inner_train)} train / {len(validation)} validation')
                    teacher, selected_t, rows = fit_model(
                        models.Teacher, objective, 'teacher', inner_train, validation, scale,
                        config, device, seed, config['epochs'], context=context, log=log)
                    history.extend(rows)
                    student, selected_s, rows = fit_model(
                        models.Student, objective, 'student', inner_train, validation, scale,
                        config, device, seed + 1, config['epochs'], teacher=teacher, context=context, log=log)
                    history.extend(rows)
                    teacher_epochs.append(selected_t['epochs'])
                    student_epochs.append(selected_s['epochs'])
                    selections.append(dict(outer_fold=outer, inner_fold=inner,
                        train_subjects=';'.join(s.subject_id for s in inner_train),
                        validation_subjects=';'.join(s.subject_id for s in validation),
                        teacher_epoch=selected_t['epochs'], teacher_mae=selected_t['validation_mae'],
                        student_epoch=selected_s['epochs'], student_mae=selected_s['validation_mae']))
                    del teacher, student
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    write_csv(run / 'history.csv', history)
                    write_csv(run / 'selection.csv', selections)

                # Refit on all ten outer-train subjects for the selected epoch counts.
                epochs_t = max(1, round(statistics.median(teacher_epochs)))
                epochs_s = max(1, round(statistics.median(student_epochs)))
                scale = TargetScale.fit(train)
                context = dict(stage='refit', outer_fold=outer, inner_fold='')
                seed = config['seed'] + outer * 100 + 90
                log(f'Outer {outer}: refit on 10 subjects; teacher={epochs_t}, student={epochs_s} epochs')
                teacher, _, rows = fit_model(models.Teacher, objective, 'teacher', train, None, scale,
                    config, device, seed, epochs_t, context=context, log=log)
                history.extend(rows)
                save_checkpoint(folder / 'teacher.pt', teacher, 'teacher', baseline, config,
                                outer, train, test, scale, fingerprint, epochs_t)
                student, _, rows = fit_model(models.Student, objective, 'student', train, None, scale,
                    config, device, seed + 1, epochs_s, teacher=teacher, context=context, log=log)
                history.extend(rows)
                save_checkpoint(folder / 'student.pt', student, 'student', baseline, config,
                                outer, train, test, scale, fingerprint, epochs_s)
                write_json(folder / 'scale.json', asdict(scale))
                write_csv(run / 'history.csv', history)
                write_csv(run / 'selection.csv', selections)
                del teacher, student
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            metadata['status'] = 'complete'
            write_json(run / 'run.json', metadata)
            write_json(ROOT / baseline / ('latest_smoke.json' if smoke else 'latest_run.json'),
                       dict(run=str(run)))
            log('Training complete. Run test.py to evaluate held-out subjects.')
        except Exception as error:
            metadata.update(status='failed', error=str(error))
            write_json(run / 'run.json', metadata)
            raise
    return run
