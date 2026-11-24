"""Train one teacher or student on a permitted subject partition."""

import copy
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from .data import Images, collate_subjects
from .utils import seed_everything


def move_batch(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def make_loader(subjects, include_dat, config, seed, shuffle):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Images(subjects, include_dat), batch_size=config['batch_size'],
                      shuffle=shuffle, generator=generator, num_workers=0,
                      collate_fn=collate_subjects)


@torch.inference_mode()
def validation_mae(model, loader, scale, device):
    model.eval()
    errors = []
    for batch in loader:
        batch = move_batch(batch, device)
        prediction = scale.restore(model(batch)['prediction'])
        errors.extend((prediction - batch['target']).abs().cpu().tolist())
    return sum(errors) / len(errors)


# A fixed teacher has neither parameter updates nor BatchNorm-statistic updates.
def freeze_teacher(teacher):
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher


def fit_model(model_class, loss_class, role, train_subjects, validation_subjects,
              scale, config, device, seed, epochs, teacher=None, context=None, log=None):
    seed_everything(seed)
    model = model_class(weights=config['weights']).to(device)
    objective = loss_class(config).to(device)
    train_loader = make_loader(train_subjects, True, config, seed, True)
    validation_loader = (make_loader(validation_subjects, role == 'teacher', config, seed, False)
                         if validation_subjects else None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    if teacher is not None:
        freeze_teacher(teacher)
    amp = config['amp'] and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    best_mae, best_epoch, best_state, stale = float('inf'), 0, None, 0
    history = []
    context = context or {}

    for epoch in range(1, epochs + 1):
        model.train()
        totals, seen = {}, 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            target = scale.normalize(batch['target'])
            optimizer.zero_grad(set_to_none=True)
            precision = torch.autocast('cuda', dtype=torch.float16) if amp else nullcontext()
            with precision:
                output = model(batch)
                if role == 'teacher':
                    loss, terms = objective.teacher(output, batch, target)
                else:
                    with torch.no_grad():
                        teacher_output = teacher(batch)
                    loss, terms = objective.student(output, teacher_output, batch, target)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite {role} loss at epoch {epoch}.')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch['subject_id'])
            values = dict(total=loss, **terms)
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value.detach()) * count
            seen += count
        row = dict(**context, role=role, epoch=epoch, n_train=len(train_subjects),
                   **{key: value / seen for key, value in totals.items()})
        if validation_loader is not None:
            mae = validation_mae(model, validation_loader, scale, device)
            row['validation_mae'] = mae
            if mae < best_mae:
                best_mae, best_epoch, stale = mae, epoch, 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale += 1
        else:
            best_epoch = epoch
        history.append(row)
        if log:
            score = f" val_MAE={row['validation_mae']:.3f}" if 'validation_mae' in row else ''
            log(f"{role} epoch {epoch}/{epochs}: loss={row['total']:.5f}{score}")
        if validation_loader is not None and stale >= config['patience']:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, dict(epochs=best_epoch, validation_mae=best_mae if validation_loader else None), history
