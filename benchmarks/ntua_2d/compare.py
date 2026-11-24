"""Compare paired student OOF predictions from completed kd_0 and kd_1 tests."""

import argparse
import csv
from pathlib import Path

from common.evaluate import metrics
from common.utils import new_directory, read_json, write_csv, write_json


def load(run):
    meta = read_json(run / 'run.json')
    test = Path(read_json(run / 'latest_test.json')['test'])
    test_meta = read_json(test / 'test.json')
    if not test_meta['full_cv']:
        raise ValueError('Comparison requires all three outer folds.')
    with (test / 'predictions.csv').open(encoding='utf-8', newline='') as file:
        rows = [r for r in csv.DictReader(file) if r['role'] == 'student']
    if len(rows) != 15 or len({r['subject_id'] for r in rows}) != 15:
        raise ValueError('Each test must contain exactly 15 unique student OOF predictions.')
    return meta, {r['subject_id']: dict(fold=int(r['fold']), y_true=float(r['y_true']),
                                     y_pred=float(r['y_pred'])) for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kd0', type=Path, required=True, help='Completed kd_0 run with test results.')
    parser.add_argument('--kd1', type=Path, required=True, help='Completed kd_1 run with test results.')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--allow-smoke', action='store_true', help='Only validate the comparison workflow, not model quality.')
    args = parser.parse_args()
    try:
        a, rows_a = load(args.kd0)
        b, rows_b = load(args.kd1)
        if a['baseline'] != 'kd_0' or b['baseline'] != 'kd_1':
            raise ValueError('Pass kd_0 to --kd0 and kd_1 to --kd1.')
        if a['data_fingerprint'] != b['data_fingerprint']:
            raise ValueError('The experiments used different datasets or folds.')
        if a['source_hash'] != b['source_hash']:
            raise ValueError('Shared implementation changed between runs; repeat the paired experiment.')
        for key, value in a['config'].items():
            if key not in ('cafr_weight', 'attention_weight') and b['config'].get(key) != value:
                raise ValueError(f'Common settings differ: {key}')
        if (a['purpose'] != 'experiment' or b['purpose'] != 'experiment') and not args.allow_smoke:
            raise ValueError('Smoke runs do not measure improvement. Pass --allow-smoke only to check the workflow.')
        if set(rows_a) != set(rows_b):
            raise ValueError('Student test subjects do not match.')
        for sid in rows_a:
            if any(rows_a[sid][key] != rows_b[sid][key] for key in ('fold', 'y_true')):
                raise ValueError(f'Fold or label mismatch: {sid}')
        results = []
        for fold in [0, 1, 2, 'all']:
            ids = [sid for sid in rows_a if fold == 'all' or rows_a[sid]['fold'] == fold]
            m0 = metrics([rows_a[sid] for sid in ids])
            m1 = metrics([rows_b[sid] for sid in ids])
            result = dict(fold=fold, n_subjects=len(ids), kd0_mae=m0['mae'], kd1_mae=m1['mae'],
                delta_mae=m1['mae'] - m0['mae'], kd0_rmse=m0['rmse'], kd1_rmse=m1['rmse'],
                delta_rmse=m1['rmse'] - m0['rmse'])
            results.append(result)
            print(f"fold={fold}: MAE {m0['mae']:.3f} -> {m1['mae']:.3f}; "
                  f"RMSE {m0['rmse']:.3f} -> {m1['rmse']:.3f}")
        out = new_directory(Path(__file__).parent / 'comparisons', args.output, 'compare')
        write_csv(out / 'metrics.csv', results)
        write_json(out / 'comparison.json', dict(kd0=str(args.kd0.resolve()), kd1=str(args.kd1.resolve()),
            purpose='smoke' if args.allow_smoke else 'experiment', delta='kd_1 minus kd_0; negative favors kd_1'))
        print(f'Comparison: {out}')
    except (ValueError, OSError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    main()
