import argparse
import glob
import os
from collections import defaultdict

from tensorboard.backend.event_processing import event_accumulator


def load_scalars(event_path, tag):
    ea = event_accumulator.EventAccumulator(event_path, size_guidance={'scalars': 0})
    ea.Reload()
    if tag not in ea.Tags().get('scalars', []):
        return []
    return [(e.step, e.value) for e in ea.Scalars(tag)]


def aggregate(events, tag):
    step_to_vals = defaultdict(list)
    for path in events:
        scalars = load_scalars(path, tag)
        for step, value in scalars:
            step_to_vals[step].append(value)
    steps = sorted(step_to_vals.keys())
    means = []
    stds = []
    for step in steps:
        vals = step_to_vals[step]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        means.append(mean)
        stds.append(var ** 0.5)
    return steps, means, stds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='.', help='Root directory to search for event files')
    parser.add_argument('--tag', default='charts/episodic_return', help='TensorBoard scalar tag')
    parser.add_argument('--out', default='aggregate.png', help='Output plot path')
    parser.add_argument('--glob', default='**/events.out.tfevents.*', help='Glob to find event files')
    args = parser.parse_args()

    event_paths = glob.glob(os.path.join(args.root, args.glob), recursive=True)
    if not event_paths:
        raise SystemExit(f'No event files found under {args.root}')

    steps, means, stds = aggregate(event_paths, args.tag)
    if not steps:
        raise SystemExit(f'No scalars found for tag: {args.tag}')

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise SystemExit(f'matplotlib is required to plot: {exc}')

    plt.figure(figsize=(8, 4))
    plt.plot(steps, means, label='mean')
    plt.fill_between(steps, [m - s for m, s in zip(means, stds)], [m + s for m, s in zip(means, stds)], alpha=0.2)
    plt.xlabel('step')
    plt.ylabel(args.tag)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
