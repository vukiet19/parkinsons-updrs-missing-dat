"""Short, identical train/test entry points for both experiments."""

import argparse
from pathlib import Path

import torch

from .data import ROOT, data_fingerprint, read_subjects
from .evaluate import run_test
from .experiment import run_training
from .utils import choose_device, read_json


def train_main(baseline):
    parser = argparse.ArgumentParser(description=f'Train {baseline}: 3 outer subject folds (5/5/5).')
    parser.add_argument('--config', type=Path, default=ROOT / baseline / 'config.json')
    parser.add_argument('--output', type=Path, help='New run directory; existing folders are refused.')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--epochs', type=int, help='Maximum epochs per inner fit; selection uses validation only.')
    parser.add_argument('--batch-size', type=int, help='Number of patients per batch, not MRI slices.')
    parser.add_argument('--weights', choices=['imagenet', 'none'])
    parser.add_argument('--inner-folds', type=int, choices=[1, 3])
    parser.add_argument('--folds', type=int, nargs='+', choices=[0, 1, 2], default=[0, 1, 2])
    parser.add_argument('--smoke', action='store_true', help='All requested folds, 1 epoch, 1 inner fold, random initialization.')
    parser.add_argument('--check', action='store_true', help='Inspect data/folds without creating or training models.')
    args = parser.parse_args()
    try:
        subjects = read_subjects()
        if args.check:
            print('15 subjects / 28 MRI / 15 DaT; outer folds = 5/5/5')
            print('Data fingerprint:', data_fingerprint(subjects))
            return
        config = read_json(args.config)
        for option, value in [('epochs', args.epochs), ('batch_size', args.batch_size),
                              ('weights', args.weights), ('inner_folds', args.inner_folds)]:
            if value is not None:
                config[option] = value
        if args.smoke:
            config.update(epochs=1, inner_folds=1, weights='none', amp=False)
        if len(set(args.folds)) != len(args.folds):
            raise ValueError('A fold must not be listed twice.')
        if (config['epochs'] < 1 or config['batch_size'] < 1 or config['patience'] < 1
                or config['learning_rate'] <= 0 or config['weight_decay'] < 0
                or config['gradient_clip'] <= 0 or config['cpu_threads'] < 1
                or config['inner_folds'] not in (1, 3) or config['weights'] not in ('imagenet', 'none')):
            raise ValueError('Invalid training configuration.')
        for key in ('output_weight', 'cafr_weight', 'attention_weight'):
            if key in config and config[key] < 0:
                raise ValueError(f'{key} must be nonnegative.')
        torch.set_num_threads(config['cpu_threads'])
        run_training(baseline, config, choose_device(args.device), args.output,
                     tuple(args.folds), args.smoke)
    except (ValueError, OSError, FloatingPointError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')


def test_main(baseline):
    parser = argparse.ArgumentParser(description=f'Evaluate saved {baseline} checkpoints on outer-test subjects.')
    parser.add_argument('--run', type=Path, help='Training run; defaults to the latest completed non-smoke run.')
    parser.add_argument('--output', type=Path, help='New results directory; existing folders are refused.')
    parser.add_argument('--role', choices=['student', 'teacher', 'both'], default='student')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args()
    try:
        if args.run:
            run = args.run
        else:
            pointer = ROOT / baseline / 'latest_run.json'
            if not pointer.exists():
                raise ValueError('No completed experiment. Train first or pass --run explicitly for a smoke run.')
            run = Path(read_json(pointer)['run'])
        config = read_json(run / 'run.json')['config']
        torch.set_num_threads(config['cpu_threads'])
        run_test(baseline, run, choose_device(args.device), args.role, args.output)
    except (ValueError, OSError, FloatingPointError, RuntimeError) as error:
        parser.exit(1, f'Error: {error}\n')
