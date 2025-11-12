# Evaluation Logging Guide

This guide explains how to log and analyze accuracy and speed metrics for dParallel and baseline LLaDA evaluations.

## What's New

### 1. Speed Metrics Logging
The evaluation script now automatically logs speed metrics including:
- **Total time**: Total wall-clock time for generation
- **Total tokens**: Number of tokens generated
- **Total NFE**: Number of function evaluations (model forward passes)
- **Throughput**: Tokens per second
- **Average NFE per token**: Efficiency metric

Speed metrics are:
- Aggregated across all GPU ranks automatically
- Saved to `speed_metrics.json` in the output directory
- Printed to stdout with clear formatting

### 2. Accuracy Results Logging
The lm-evaluation-harness automatically saves accuracy results to:
- `results.json`: Overall accuracy metrics for all tasks
- Individual sample files when `--log_samples` is used

## Output Directory Structure

After running evaluations, your output directory will be organized as:

```
/home/hice1/rbansal66/scratch/dParallel/output/
├── baseline/
│   ├── gsm8k/
│   │   ├── speed_metrics.json     # Speed metrics for baseline
│   │   ├── results.json            # Accuracy results from lm-eval
│   │   └── rank_*.jsonl            # Per-rank generation outputs
│   ├── minerva_math/
│   ├── humaneval/
│   └── mbpp/
└── dparallel/
    ├── gsm8k/
    │   ├── speed_metrics.json     # Speed metrics for dParallel
    │   ├── results.json            # Accuracy results from lm-eval
    │   └── rank_*.jsonl            # Per-rank generation outputs
    ├── minerva_math/
    ├── humaneval/
    └── mbpp/
```

## Running Evaluations

### Run a specific task evaluation:
```bash
cd /home/hice1/rbansal66/scratch/dParallel/LLaDA

# For GSM8K
bash eval.sh   # Then comment out other tasks if needed

# Or run specific commands from eval.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py \
  --tasks gsm8k --num_fewshot 0 --confirm_run_unsafe_code --model llada_dist \
  --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,show_speed=True,task="gsm8k",save_dir=/home/hice1/rbansal66/scratch/dParallel/output/baseline/gsm8k \
  --output_path /home/hice1/rbansal66/scratch/dParallel/output/baseline/gsm8k
```

### What to expect during evaluation:
1. Progress bars for dataset loading and context building
2. Generation progress with individual Q&A printed
3. **Speed metrics summary** printed at the end with clear separators:
   ```
   ================================================================================
   SPEED METRICS (Aggregated across 4 ranks):
     Total time taken: 1234.56 seconds
     Total tokens generated: 50,000
     Total NFE: 12,800,000
     Throughput: 40.50 tokens/sec
     Average NFE per token: 256.00
   ================================================================================
   ```
4. **Accuracy results table** printed by lm-eval harness
5. Results automatically saved to JSON files

## Viewing Results

### Option 1: Use the Summary Script (Recommended)

```bash
cd /home/hice1/rbansal66/scratch/dParallel/LLaDA
python summarize_results.py [output_dir]

# Default output directory
python summarize_results.py

# Or specify custom directory
python summarize_results.py /path/to/output
```

This script will:
- Scan for all completed evaluations
- Print formatted results for each task
- Show side-by-side comparisons of baseline vs dParallel
- Calculate speedup and accuracy differences
- Save a summary to `evaluation_summary.txt`

### Option 2: Manual Inspection

```bash
# View speed metrics
cat /home/hice1/rbansal66/scratch/dParallel/output/baseline/gsm8k/speed_metrics.json
cat /home/hice1/rbansal66/scratch/dParallel/output/dparallel/gsm8k/speed_metrics.json

# View accuracy results
cat /home/hice1/rbansal66/scratch/dParallel/output/baseline/gsm8k/results.json
cat /home/hice1/rbansal66/scratch/dParallel/output/dparallel/gsm8k/results.json

# Use jq for pretty printing (if available)
jq . /home/hice1/rbansal66/scratch/dParallel/output/baseline/gsm8k/speed_metrics.json
```

### Option 3: Check Log Files

If you're running via SLURM or redirecting output:
```bash
# Check stdout for accuracy tables
tail -100 /path/to/your/logfile.out

# Check stderr for progress and speed metrics
tail -100 /path/to/your/logfile.err
```

## Key Metrics to Compare

### Speed Metrics
- **Throughput (tokens/sec)**: Higher is better
- **Average NFE per token**: Lower means more efficient (fewer model calls per token)
- **Total time**: Total wall-clock time for the evaluation

### Accuracy Metrics
- **GSM8K**: `exact_match` or `acc` (accuracy)
- **Minerva Math**: `exact_match` per sub-task (algebra, geometry, etc.)
- **HumanEval**: `pass@1` (requires postprocessing)
- **MBPP**: `pass@1` (requires postprocessing)

## Troubleshooting

### No speed metrics saved
- Check that `show_speed=True` is set in `--model_args`
- Verify the `save_dir` path is writable
- Check for errors in stderr log

### No accuracy results saved
- Ensure `--output_path` is specified in the command
- Check that the evaluation completed successfully
- Look for `results.json` in the output directory

### Results look wrong
- Verify you're comparing the same tasks with same parameters
- Check that both baseline and dParallel evaluations completed
- Ensure you're using the correct threshold values for dParallel

## Example: Complete Workflow

```bash
# 1. Run evaluations
cd /home/hice1/rbansal66/scratch/dParallel/LLaDA
bash eval.sh  # Or run specific tasks

# 2. Wait for completion, then check results
python summarize_results.py

# 3. View detailed results
less /home/hice1/rbansal66/scratch/dParallel/output/evaluation_summary.txt

# 4. Or manually inspect specific metrics
cat /home/hice1/rbansal66/scratch/dParallel/output/baseline/gsm8k/speed_metrics.json
cat /home/hice1/rbansal66/scratch/dParallel/output/dparallel/gsm8k/speed_metrics.json
```

## Notes

- Speed metrics are only computed when `show_speed=True` is in model_args
- Metrics are automatically aggregated across all GPU ranks
- The evaluation framework saves checkpoints, so you can resume interrupted evaluations
- For HumanEval and MBPP, remember to run the postprocessing scripts as noted in eval.sh

