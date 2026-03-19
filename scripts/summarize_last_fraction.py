import argparse
import csv
import glob
import json
import math
import os
import re
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

from tensorboard.backend.event_processing import event_accumulator


EVENT_GLOB = "**/events.out.tfevents.*"
SEED_RE = re.compile(r"_seed_(\d+)")


def load_scalars(event_path, tag):
    ea = event_accumulator.EventAccumulator(event_path, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return []
    return [float(event.value) for event in ea.Scalars(tag)]


def extract_seed(run_dir):
    match = SEED_RE.search(os.path.basename(run_dir))
    if not match:
        raise ValueError(f"Could not parse seed from run directory: {run_dir}")
    return int(match.group(1))


def population_std(values):
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(var)


def discover_methods(root, requested_methods):
    if requested_methods:
        return requested_methods
    methods = []
    for entry in sorted(os.listdir(root)):
        path = os.path.join(root, entry)
        if os.path.isdir(path):
            methods.append(entry)
    return methods


def build_per_run(root, methods, tag, tail_fraction):
    tasks = []
    for method in methods:
        method_root = os.path.join(root, method)
        event_paths = glob.glob(os.path.join(method_root, EVENT_GLOB), recursive=True)
        for event_path in sorted(event_paths):
            tasks.append((method, event_path, tag, tail_fraction))
    return tasks


def process_event_file(task):
    method, event_path, tag, tail_fraction = task
    run_dir = os.path.dirname(event_path)
    env = os.path.basename(os.path.dirname(run_dir))
    if env == method:
        env = os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
    seed = extract_seed(run_dir)
    values = load_scalars(event_path, tag)
    if not values:
        return None
    tail_points = max(1, math.ceil(len(values) * tail_fraction))
    tail_values = values[-tail_points:]
    return {
        "method": method,
        "env": env,
        "seed": seed,
        "num_points": len(values),
        "tail_points": tail_points,
        "tail_mean": sum(tail_values) / len(tail_values),
        "run_dir": run_dir,
    }


def collect_per_run(root, methods, tag, tail_fraction, workers):
    tasks = build_per_run(root, methods, tag, tail_fraction)
    per_run = []
    if workers <= 1:
        for task in tasks:
            item = process_event_file(task)
            if item is not None:
                per_run.append(item)
        return per_run

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for item in executor.map(process_event_file, tasks, chunksize=1):
            if item is not None:
                per_run.append(item)
    return per_run


def build_summary(per_run):
    grouped = defaultdict(list)
    for item in per_run:
        grouped[(item["method"], item["env"])].append(item)

    summary = []
    for (method, env), runs in sorted(grouped.items()):
        runs = sorted(runs, key=lambda item: item["seed"])
        tail_means = [item["tail_mean"] for item in runs]
        seeds = [item["seed"] for item in runs]
        summary.append(
            {
                "method": method,
                "env": env,
                "num_seeds": len(runs),
                "seeds": seeds,
                "mean_of_seed_means": sum(tail_means) / len(tail_means),
                "std_of_seed_means": population_std(tail_means),
            }
        )
    return summary


def write_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Root directory containing method subdirectories.")
    parser.add_argument("--tag", default="charts/episodic_return", help="TensorBoard scalar tag.")
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.1,
        help="Fraction of the scalar history to average from the end of each run.",
    )
    parser.add_argument(
        "--methods",
        nargs="*",
        default=None,
        help="Optional method directory names to include. Defaults to every direct child under --root.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of worker processes for parsing event files.",
    )
    parser.add_argument("--json-out", required=True, help="Output JSON path.")
    parser.add_argument("--summary-csv-out", required=True, help="Output summary CSV path.")
    parser.add_argument("--per-run-csv-out", required=True, help="Output per-run CSV path.")
    args = parser.parse_args()

    methods = discover_methods(args.root, args.methods)
    per_run = collect_per_run(args.root, methods, args.tag, args.tail_fraction, args.workers)
    if not per_run:
        raise SystemExit(f"No matching scalar data found under {args.root}")

    summary = build_summary(per_run)
    payload = {
        "root": args.root,
        "tag": args.tag,
        "tail_fraction": args.tail_fraction,
        "per_run": per_run,
        "summary": summary,
    }

    os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_csv_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.per_run_csv_out) or ".", exist_ok=True)

    with open(args.json_out, "w") as f:
        json.dump(payload, f, indent=2)

    summary_rows = []
    for item in summary:
        summary_rows.append(
            {
                "method": item["method"],
                "env": item["env"],
                "num_seeds": item["num_seeds"],
                "seeds": "|".join(str(seed) for seed in item["seeds"]),
                "mean_of_seed_means": f'{item["mean_of_seed_means"]:.6f}',
                "std_of_seed_means": f'{item["std_of_seed_means"]:.6f}',
            }
        )

    per_run_rows = []
    for item in per_run:
        per_run_rows.append(
            {
                "method": item["method"],
                "env": item["env"],
                "seed": item["seed"],
                "num_points": item["num_points"],
                "tail_points": item["tail_points"],
                "tail_mean": f'{item["tail_mean"]:.6f}',
                "run_dir": item["run_dir"],
            }
        )

    write_csv(
        args.summary_csv_out,
        ["method", "env", "num_seeds", "seeds", "mean_of_seed_means", "std_of_seed_means"],
        summary_rows,
    )
    write_csv(
        args.per_run_csv_out,
        ["method", "env", "seed", "num_points", "tail_points", "tail_mean", "run_dir"],
        per_run_rows,
    )

    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.summary_csv_out}")
    print(f"Wrote {args.per_run_csv_out}")


if __name__ == "__main__":
    main()
