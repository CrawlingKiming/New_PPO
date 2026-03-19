import argparse
import glob
import json
import os
from collections import defaultdict

from tensorboard.backend.event_processing import event_accumulator


def load_scalars(event_path, tag):
    ea = event_accumulator.EventAccumulator(event_path, size_guidance={'scalars': 0})
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def parse_algo(run_name):
    return run_name.split('_', 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, help='Root directory to search for event files')
    parser.add_argument('--tag', default='charts/episodic_return', help='TensorBoard scalar tag')
    parser.add_argument('--glob', default='**/events.out.tfevents.*', help='Glob to find event files')
    parser.add_argument('--out', required=True, help='Output JSON path')
    args = parser.parse_args()

    event_paths = glob.glob(os.path.join(args.root, args.glob), recursive=True)
    if not event_paths:
        raise SystemExit(f'No event files found under {args.root}')

    runs = []
    for path in event_paths:
        scalars = load_scalars(path, args.tag)
        if not scalars:
            continue
        last_step, last_value = scalars[-1]
        run_dir = os.path.dirname(path)
        run_name = os.path.basename(run_dir)
        env_id = os.path.basename(os.path.dirname(run_dir))
        algo = parse_algo(run_name)
        runs.append(
            {
                'env_id': env_id,
                'algo': algo,
                'run_dir': run_dir,
                'last_step': int(last_step),
                'last_value': float(last_value),
            }
        )

    if not runs:
        raise SystemExit(f'No scalars found for tag: {args.tag}')

    env_algo_to_vals = defaultdict(list)
    for run in runs:
        env_algo_to_vals[(run['env_id'], run['algo'])].append(run['last_value'])

    summary = []
    for (env_id, algo), vals in sorted(env_algo_to_vals.items()):
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        summary.append(
            {
                'env_id': env_id,
                'algo': algo,
                'num_runs': len(vals),
                'mean_last': mean,
                'std_last': var ** 0.5,
            }
        )

    output = {
        'root': args.root,
        'tag': args.tag,
        'runs': runs,
        'summary': summary,
    }

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
