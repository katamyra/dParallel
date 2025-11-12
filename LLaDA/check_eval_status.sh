#!/bin/bash
# Quick script to check evaluation status

OUTPUT_DIR="${1:-/home/hice1/rbansal66/scratch/dParallel/output}"

echo "============================================"
echo "Evaluation Status Check"
echo "============================================"
echo "Output directory: $OUTPUT_DIR"
echo ""

TASKS=("gsm8k" "minerva_math" "humaneval" "mbpp")
MODELS=("baseline" "dparallel")

for task in "${TASKS[@]}"; do
    echo "----------------------------------------"
    echo "Task: $task"
    echo "----------------------------------------"
    
    for model in "${MODELS[@]}"; do
        DIR="$OUTPUT_DIR/$model/$task"
        
        if [ -d "$DIR" ]; then
            echo "  $model:"
            
            # Check for speed metrics
            if [ -f "$DIR/speed_metrics.json" ]; then
                echo "    ✓ Speed metrics available"
                # Extract key metrics if jq is available
                if command -v jq &> /dev/null; then
                    THROUGHPUT=$(jq -r '.tokens_per_second' "$DIR/speed_metrics.json" 2>/dev/null)
                    if [ "$THROUGHPUT" != "null" ] && [ -n "$THROUGHPUT" ]; then
                        echo "      Throughput: $THROUGHPUT tokens/sec"
                    fi
                fi
            else
                echo "    ✗ Speed metrics missing"
            fi
            
            # Check for accuracy results
            if [ -f "$DIR/results.json" ]; then
                echo "    ✓ Accuracy results available"
                # Extract accuracy if jq is available
                if command -v jq &> /dev/null; then
                    # Try to extract common accuracy metrics
                    ACC=$(jq -r ".results.${task}.acc // .results.${task}.exact_match // .results.${task}.pass_at_1 // \"N/A\"" "$DIR/results.json" 2>/dev/null)
                    if [ "$ACC" != "null" ] && [ -n "$ACC" ] && [ "$ACC" != "N/A" ]; then
                        echo "      Accuracy: $ACC"
                    fi
                fi
            else
                echo "    ✗ Accuracy results missing"
            fi
            
            # Count rank files
            RANK_FILES=$(find "$DIR" -name "rank_*.jsonl" 2>/dev/null | wc -l)
            if [ "$RANK_FILES" -gt 0 ]; then
                echo "    ✓ Generation outputs: $RANK_FILES rank files"
            fi
            
        else
            echo "  $model: Not started"
        fi
        echo ""
    done
done

echo "============================================"
echo "Summary"
echo "============================================"

# Count completed evaluations
TOTAL_SPEED=0
TOTAL_ACC=0
TOTAL_POSSIBLE=$((${#TASKS[@]} * ${#MODELS[@]}))

for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        DIR="$OUTPUT_DIR/$model/$task"
        [ -f "$DIR/speed_metrics.json" ] && ((TOTAL_SPEED++))
        [ -f "$DIR/results.json" ] && ((TOTAL_ACC++))
    done
done

echo "Speed metrics completed: $TOTAL_SPEED / $TOTAL_POSSIBLE"
echo "Accuracy results completed: $TOTAL_ACC / $TOTAL_POSSIBLE"
echo ""

if [ "$TOTAL_SPEED" -eq "$TOTAL_POSSIBLE" ] && [ "$TOTAL_ACC" -eq "$TOTAL_POSSIBLE" ]; then
    echo "✓ All evaluations complete! Run 'python summarize_results.py' to generate report."
elif [ "$TOTAL_SPEED" -gt 0 ] || [ "$TOTAL_ACC" -gt 0 ]; then
    echo "⚠ Evaluations in progress or partially complete."
else
    echo "⚠ No evaluations found. Have you run eval.sh yet?"
fi
echo "============================================"

