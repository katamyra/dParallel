#!/usr/bin/env python3
"""
Summarize evaluation results for multiple experiment types under an output dir.

Usage:
    python summarize_experiments.py [output_dir]

By default it looks at:
    /home/hice1/rbansal66/scratch/dParallel/output

and expects subdirectories like:
    - dparallel/
    - dparallel_prophet/
    - dparallel_prophet_dynamic/
    - dparallel_prophet_ema/

Each experiment directory is expected to contain task subdirectories
such as `gsm8k/`, each with:
    - speed_metrics.json
    - a model subdirectory containing `results_*.json` from lm-eval
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple


def load_speed_metrics(metrics_path: str) -> Dict[str, Any]:
    """Load speed metrics from JSON file."""
    try:
        with open(metrics_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {metrics_path}")
        return {}


def find_results_file(task_dir: Path) -> Optional[Path]:
    """Find the results JSON file in model subdirectories."""
    if not task_dir.exists():
        return None

    # Look for results files in model subdirectories
    for model_dir in task_dir.iterdir():
        if model_dir.is_dir():
            for results_file in model_dir.glob("results_*.json"):
                return results_file

    # Fallback: check for results.json directly in task_dir
    results_path = task_dir / "results.json"
    if results_path.exists():
        return results_path

    return None


def load_accuracy_results(results_path: Optional[Path]) -> Dict[str, Any]:
    """Load accuracy results from lm-eval JSON file."""
    if results_path is None or not results_path.exists():
        return {}

    try:
        with open(results_path, "r") as f:
            data = json.load(f)
            return data.get("results", {})
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {results_path}")
        return {}


def extract_primary_accuracy(
    accuracy_results: Dict[str, Any]
) -> Optional[Tuple[str, str, float]]:
    """
    Heuristically extract a primary accuracy metric.

    Returns:
        (task_name, metric_name, value) or None if not found.
    """
    for task_name, metrics in accuracy_results.items():
        if not isinstance(metrics, dict):
            continue

        # Prefer a metric with 'flexible-extract' in the name if present
        preferred: List[Tuple[str, float]] = []
        fallback: List[Tuple[str, float]] = []
        for metric_name, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            if "flexible-extract" in metric_name:
                preferred.append((metric_name, float(value)))
            else:
                fallback.append((metric_name, float(value)))

        if preferred:
            metric_name, value = preferred[0]
            return task_name, metric_name, value
        if fallback:
            metric_name, value = fallback[0]
            return task_name, metric_name, value

    return None


def format_metrics(
    speed_metrics: Dict[str, Any],
    accuracy_results: Dict[str, Any],
    exp_name: str,
    task_name: str,
) -> str:
    """Format metrics into a readable string."""
    output: List[str] = []
    output.append(f"\n{'='*80}")
    output.append(f"Results for: {exp_name}  |  Task: {task_name}")
    output.append(f"{'='*80}")

    # Speed metrics
    if speed_metrics:
        output.append("\nSpeed Metrics:")
        output.append(
            f"  Total Time:           {speed_metrics.get('total_time_seconds', 0):.2f} seconds"
        )
        output.append(
            f"  Total Tokens:         {speed_metrics.get('total_tokens', 0):,}"
        )
        output.append(
            f"  Total NFE:            {speed_metrics.get('total_nfe', 0):,}"
        )
        output.append(
            f"  Throughput:           {speed_metrics.get('tokens_per_second', 0):.2f} tokens/sec"
        )
        output.append(
            f"  Avg NFE per token:    {speed_metrics.get('avg_nfe_per_token', 0):.4f}"
        )
        if "num_ranks" in speed_metrics:
            output.append(
                f"  Number of GPUs:       {speed_metrics.get('num_ranks', 1)}"
            )
    else:
        output.append("\nSpeed Metrics: Not available")

    # Accuracy metrics
    if accuracy_results:
        output.append("\nAccuracy Metrics:")
        for t_name, metrics in accuracy_results.items():
            output.append(f"\n  Task: {t_name}")
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        output.append(f"    {metric_name}: {value:.4f}")
    else:
        output.append("\nAccuracy Metrics: Not available")

    output.append(f"{'='*80}\n")
    return "\n".join(output)


def main() -> None:
    # Get output directory from command line or use default
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("/home/hice1/rbansal66/scratch/dParallel/output")

    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        sys.exit(1)

    print(f"Scanning directory: {output_dir}")

    # Experiment types we care about under the output directory
    experiment_types = [
        "dparallel",
        "dparallel_prophet",
        "dparallel_prophet_dynamic",
        "dparallel_prophet_em_aggressive",
        "dparallel_prophet_ema_relaxed",
    ]

    detailed_blocks: List[str] = []
    summary_rows: List[str] = []

    # Header for compact summary table
    summary_rows.append(
        "Experiment,Task,PrimaryMetricName,PrimaryAccuracy,AvgNFEperToken"
    )

    for exp in experiment_types:
        exp_dir = output_dir / exp
        if not exp_dir.exists():
            continue

        # Discover task subdirectories dynamically
        task_dirs = [d for d in exp_dir.iterdir() if d.is_dir()]
        if not task_dirs:
            continue

        for task_dir in task_dirs:
            task_name = task_dir.name
            speed_path = task_dir / "speed_metrics.json"
            results_path = find_results_file(task_dir)

            if not speed_path.exists() and (results_path is None or not results_path.exists()):
                # Nothing to summarize for this (exp, task) pair
                continue

            speed_metrics = load_speed_metrics(str(speed_path))
            accuracy_results = load_accuracy_results(results_path)

            # Detailed block
            block = format_metrics(speed_metrics, accuracy_results, exp, task_name)
            print(block)
            detailed_blocks.append(block)

            # Compact summary row
            avg_nfe = speed_metrics.get("avg_nfe_per_token")
            primary = extract_primary_accuracy(accuracy_results)

            if primary is not None:
                p_task, metric_name, acc_val = primary
                summary_rows.append(
                    f"{exp},{task_name},{metric_name},{acc_val:.6f},{avg_nfe if isinstance(avg_nfe, (int, float)) else ''}"
                )
            else:
                summary_rows.append(
                    f"{exp},{task_name},,,{avg_nfe if isinstance(avg_nfe, (int, float)) else ''}"
                )

    # Save detailed summary to file
    detailed_path = output_dir / "experiments_detailed_summary.txt"
    with open(detailed_path, "w") as f:
        f.write("\n".join(detailed_blocks))

    # Save compact CSV-like summary (accuracy + NFE) to file
    compact_path = output_dir / "experiments_accuracy_nfe_summary.csv"
    with open(compact_path, "w") as f:
        f.write("\n".join(summary_rows))

    print(f"\n✓ Detailed summary saved to: {detailed_path}")
    print(f"✓ Accuracy + NFE summary saved to: {compact_path}")


if __name__ == "__main__":
    main()


