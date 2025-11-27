# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true

dataset="$1"
mode="$2"  # optional: for minerva_math, can be "baseline" or "dparallel"


############################################### gsm8k evaluations ###############################################
if [ -z "$dataset" ] || [ "$dataset" = "gsm8k" ]; then
    task=gsm8k
    length=256
    block_length=32
    num_fewshot=0
    steps=256
    save_dir=/home/hice1/jzhang3463/scratch/CS4644-DeepLearning/dParallel/output

    # baseline
    CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="gsm8k",save_dir=${save_dir}/baseline/${task} \
    --output_path ${save_dir}/baseline/${task}

    # dParallel
    CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k",save_dir=${save_dir}/dparallel/${task} \
    --output_path ${save_dir}/dparallel/${task}
fi




############################################### minerva_math evaluations ###############################################
if [ -z "$dataset" ] || [ "$dataset" = "minerva_math" ]; then
    task=minerva_math
    length=256
    block_length=32
    num_fewshot=4
    steps=256
    save_dir=/home/hice1/jzhang3463/scratch/CS4644-DeepLearning/dParallel/output

    # baseline (runs if mode is empty or "baseline")
    if [ -z "$mode" ] || [ "$mode" = "baseline" ]; then
        CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="minerva_math",save_dir=${save_dir}/baseline/${task} \
        --output_path ${save_dir}/baseline/${task}
    fi

    # dParallel (runs if mode is empty or "dparallel")
    if [ -z "$mode" ] || [ "$mode" = "dparallel" ]; then
        CUDA_VISIBLE_DEVICES=0,1 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
        --confirm_run_unsafe_code --model llada_dist \
        --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="minerva_math",save_dir=${save_dir}/dparallel/${task} \
        --output_path ${save_dir}/dparallel/${task}
    fi
fi



############################################### humaneval evaluations ###############################################
if [ -z "$dataset" ] || [ "$dataset" = "humaneval" ]; then
    task=humaneval
    length=256
    block_length=32
    num_fewshot=0
    steps=256
    save_dir=/home/hice1/jzhang3463/scratch/CS4644-DeepLearning/dParallel/output

    # baseline
    CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="humaneval",save_dir=${save_dir}/baseline/${task} \
    --output_path ${save_dir}/baseline/${task} --log_samples

    # dparallel
    CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.5,show_speed=True,task="humaneval",save_dir=${save_dir}/dparallel/${task} \
    --output_path ${save_dir}/dparallel/${task} --log_samples

    ## NOTICE: use postprocess for humaneval
    model_dir=${save_dir}/dparallel/humaneval/Zigeng__dParallel-LLaDA-8B-instruct
    samples_file=$(ls -t "${model_dir}"/samples_humaneval_*.jsonl | head -n 1)
    python postprocess_code_humaneval.py "${samples_file}"
fi





############################################### mbpp evaluations ###############################################
if [ -z "$dataset" ] || [ "$dataset" = "mbpp" ]; then
    task=mbpp
    length=256
    block_length=32
    num_fewshot=3
    steps=256
    save_dir=/home/hice1/jzhang3463/scratch/CS4644-DeepLearning/dParallel/output

    # baseline
    CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="mbpp",save_dir=${save_dir}/baseline/${task} \
    --output_path ${save_dir}/baseline/${task} --log_samples

    # parallel
    CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.45,show_speed=True,task="mbpp",save_dir=${save_dir}/dparallel/${task} \
    --output_path ${save_dir}/dparallel/${task} --log_samples

    ## NOTICE: use postprocess for mbpp
    model_dir=${save_dir}/dparallel/mbpp/Zigeng__dParallel-LLaDA-8B-instruct
    samples_file=$(ls -t "${model_dir}"/samples_mbpp_*.jsonl | head -n 1)
    python postprocess_code_mbpp.py "${samples_file}"
fi