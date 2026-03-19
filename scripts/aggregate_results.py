import argparse
import json
import os


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)


def build_index(summary):
    index = {}
    for item in summary:
        key = (item['env_id'], item['algo'])
        index[key] = item
    return index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Results JSON from collect_results.py')
    parser.add_argument('--algo-a', required=True, help='First algorithm name')
    parser.add_argument('--algo-b', required=True, help='Second algorithm name')
    parser.add_argument('--out', default=None, help='Optional CSV output path')
    args = parser.parse_args()

    data = load_json(args.input)
    summary = data.get('summary', [])
    index = build_index(summary)

    envs = sorted({item['env_id'] for item in summary})
    rows = []
    for env_id in envs:
        a = index.get((env_id, args.algo_a))
        b = index.get((env_id, args.algo_b))
        if not a or not b:
            continue
        rows.append(
            (
                env_id,
                a['mean_last'],
                a['std_last'],
                b['mean_last'],
                b['std_last'],
                a['mean_last'] - b['mean_last'],
            )
        )

    if not rows:
        raise SystemExit('No overlapping envs found for the selected algorithms.')

    header = ['env_id', f'{args.algo_a}_mean', f'{args.algo_a}_std', f'{args.algo_b}_mean', f'{args.algo_b}_std', 'diff_mean']
    lines = [','.join(header)]
    for row in rows:
        lines.append(','.join([row[0]] + [f'{v:.6f}' if isinstance(v, float) else str(v) for v in row[1:]]))

    table = '\n'.join(lines)
    print(table)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(table + '\n')
        print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
