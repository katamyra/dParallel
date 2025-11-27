#!/usr/bin/env python3
"""
Script to summarize evaluation results including accuracy and speed metrics.
Usage: python summarize_results.py [output_dir]
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, Any

def load_speed_metrics(metrics_path: str) -> Dict[str, Any]:
    """Load speed metrics from JSON file."""
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {metrics_path}")
        return {}

def find_results_file(task_dir: Path) -> Path:
    """Find the results JSON file in model subdirectories."""
    if not task_dir.exists():
        return None
    
    # Look for results files in model subdirectories
    for model_dir in task_dir.iterdir():
        if model_dir.is_dir():
            # Look for any results_*.json file
            for results_file in model_dir.glob("results_*.json"):
                return results_file
    
    # Fallback: check for results.json directly in task_dir
    results_path = task_dir / 'results.json'
    if results_path.exists():
        return results_path
    
    return None

def load_accuracy_results(results_path: Path) -> Dict[str, Any]:
    """Load accuracy results from lm-eval JSON file."""
    if results_path is None or not results_path.exists():
        return {}
    
    try:
        with open(results_path, 'r') as f:
            data = json.load(f)
            return data.get('results', {})
    except json.JSONDecodeError:
        print(f"Warning: Could not parse {results_path}")
        return {}

def format_metrics(speed_metrics: Dict, accuracy_results: Dict, model_name: str) -> str:
    """Format metrics into a readable string."""
    output = []
    output.append(f"\n{'='*80}")
    output.append(f"Results for: {model_name}")
    output.append(f"{'='*80}")
    
    # Speed metrics
    if speed_metrics:
        output.append("\nSpeed Metrics:")
        output.append(f"  Total Time:           {speed_metrics.get('total_time_seconds', 0):.2f} seconds")
        output.append(f"  Total Tokens:         {speed_metrics.get('total_tokens', 0):,}")
        output.append(f"  Total NFE:            {speed_metrics.get('total_nfe', 0):,}")
        output.append(f"  Throughput:           {speed_metrics.get('tokens_per_second', 0):.2f} tokens/sec")
        output.append(f"  Avg NFE per token:    {speed_metrics.get('avg_nfe_per_token', 0):.2f}")
        if 'num_ranks' in speed_metrics:
            output.append(f"  Number of GPUs:       {speed_metrics.get('num_ranks', 1)}")
    else:
        output.append("\nSpeed Metrics: Not available")
    
    # Accuracy metrics
    if accuracy_results:
        output.append("\nAccuracy Metrics:")
        for task_name, metrics in accuracy_results.items():
            output.append(f"\n  Task: {task_name}")
            # Extract main accuracy metric
            if isinstance(metrics, dict):
                for metric_name, value in metrics.items():
                    if isinstance(value, (int, float)):
                        output.append(f"    {metric_name}: {value:.4f}")
    else:
        output.append("\nAccuracy Metrics: Not available")
    
    output.append(f"{'='*80}\n")
    return "\n".join(output)

def compare_results(baseline_speed: Dict, baseline_acc: Dict, 
                   dparallel_speed: Dict, dparallel_acc: Dict) -> str:
    """Generate comparison between baseline and dParallel."""
    output = []
    output.append(f"\n{'='*80}")
    output.append("Comparison: dParallel vs Baseline")
    output.append(f"{'='*80}")
    
    # Speed comparison
    if baseline_speed and dparallel_speed:
        baseline_throughput = baseline_speed.get('tokens_per_second', 0)
        dparallel_throughput = dparallel_speed.get('tokens_per_second', 0)
        
        if baseline_throughput > 0:
            speedup = dparallel_throughput / baseline_throughput
            output.append(f"\nSpeed Comparison:")
            output.append(f"  Baseline throughput:   {baseline_throughput:.2f} tokens/sec")
            output.append(f"  dParallel throughput:  {dparallel_throughput:.2f} tokens/sec")
            output.append(f"  Speedup:               {speedup:.2f}x")
            
        baseline_nfe = baseline_speed.get('avg_nfe_per_token', 0)
        dparallel_nfe = dparallel_speed.get('avg_nfe_per_token', 0)
        if baseline_nfe > 0:
            nfe_reduction = (1 - dparallel_nfe / baseline_nfe) * 100
            output.append(f"\n  Baseline NFE/token:    {baseline_nfe:.2f}")
            output.append(f"  dParallel NFE/token:   {dparallel_nfe:.2f}")
            output.append(f"  NFE reduction:         {nfe_reduction:.1f}%")
    
    # Accuracy comparison
    if baseline_acc and dparallel_acc:
        output.append(f"\nAccuracy Comparison:")
        for task_name in baseline_acc:
            if task_name in dparallel_acc:
                output.append(f"\n  Task: {task_name}")
                baseline_metrics = baseline_acc[task_name]
                dparallel_metrics = dparallel_acc[task_name]
                
                if isinstance(baseline_metrics, dict) and isinstance(dparallel_metrics, dict):
                    for metric_name in baseline_metrics:
                        if metric_name in dparallel_metrics:
                            baseline_val = baseline_metrics[metric_name]
                            dparallel_val = dparallel_metrics[metric_name]
                            if isinstance(baseline_val, (int, float)) and isinstance(dparallel_val, (int, float)):
                                diff = dparallel_val - baseline_val
                                output.append(f"    {metric_name}:")
                                output.append(f"      Baseline:  {baseline_val:.4f}")
                                output.append(f"      dParallel: {dparallel_val:.4f}")
                                output.append(f"      Difference: {diff:+.4f}")
    
    output.append(f"{'='*80}\n")
    return "\n".join(output)

def main():
    # Get output directory from command line or use default
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path("/home/hice1/jzhang3463/scratch/CS4644-DeepLearning/dParallel/output")
    
    if not output_dir.exists():
        print(f"Error: Output directory {output_dir} does not exist")
        sys.exit(1)
    
    print(f"Scanning directory: {output_dir}")
    
    # Look for all task subdirectories
    tasks = ['gsm8k', 'minerva_math', 'humaneval', 'mbpp']
    
    all_results = []
    
    for task in tasks:
        # Load baseline results
        baseline_dir = output_dir / 'baseline' / task
        baseline_speed_path = baseline_dir / 'speed_metrics.json'
        baseline_results_path = find_results_file(baseline_dir)
        
        # Load dParallel results
        dparallel_dir = output_dir / 'dparallel' / task
        dparallel_speed_path = dparallel_dir / 'speed_metrics.json'
        dparallel_results_path = find_results_file(dparallel_dir)
        
        # Check if any results exist for this task
        paths_to_check = [baseline_speed_path, dparallel_speed_path]
        if baseline_results_path:
            paths_to_check.append(baseline_results_path)
        if dparallel_results_path:
            paths_to_check.append(dparallel_results_path)
        
        if not any([p and p.exists() for p in paths_to_check if p is not None]):
            continue
        
        print(f"\n{'#'*80}")
        print(f"# Task: {task.upper()}")
        print(f"{'#'*80}")
        
        # Load all metrics
        baseline_speed = load_speed_metrics(str(baseline_speed_path))
        baseline_acc = load_accuracy_results(baseline_results_path)
        dparallel_speed = load_speed_metrics(str(dparallel_speed_path))
        dparallel_acc = load_accuracy_results(dparallel_results_path)
        
        # Print individual results
        if baseline_speed or baseline_acc:
            result = format_metrics(baseline_speed, baseline_acc, f"Baseline ({task})")
            print(result)
            all_results.append(result)
        
        if dparallel_speed or dparallel_acc:
            result = format_metrics(dparallel_speed, dparallel_acc, f"dParallel ({task})")
            print(result)
            all_results.append(result)
        
        # Print comparison
        if (baseline_speed or baseline_acc) and (dparallel_speed or dparallel_acc):
            comparison = compare_results(baseline_speed, baseline_acc, 
                                       dparallel_speed, dparallel_acc)
            print(comparison)
            all_results.append(comparison)
    
    # Save summary to file
    summary_path = output_dir / 'evaluation_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("\n".join(all_results))
    
    print(f"\n✓ Summary saved to: {summary_path}")

if __name__ == "__main__":
    main()