# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true


############################################### gsm8k evaluations ###############################################
task=gsm8k
length=256
block_length=32
num_fewshot=0
steps=256
save_dir=/home/hice1/rbansal66/scratch/dParallel/output

# # baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="gsm8k",save_dir=${save_dir}/baseline/${task} \
# --output_path ${save_dir}/baseline/${task} --log_samples

# dParallel
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k",constraints_text="200:The|201:answer|202:is",save_dir=${save_dir}/dparallel/${task} \
# --output_path ${save_dir}/dparallel/${task} --log_samples

# # dParallel+prophet (piecewise thresholds)
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k",use_prophet=True,analyze_gap=True,log_gap_trace=True,constraints_text="200:The|201:answer|202:is",save_dir=${save_dir}/dparallel_prophet/${task} \
# --output_path ${save_dir}/dparallel_prophet/${task} --log_samples

# dParallel+prophet with dynamic progress-based thresholds (sweep over dyn_alpha)
dyn_alphas=(0.5 1.0 2.0 4.0)
for alpha in "${dyn_alphas[@]}"; do
  echo "Running dParallel+Prophet dynamic thresholding with dyn_alpha=${alpha}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29602 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
  --confirm_run_unsafe_code --model llada_dist \
  --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k",use_prophet=True,dynamic_threshold=True,dyn_alpha=${alpha},analyze_gap=True,log_gap_trace=True,constraints_text="200:The|201:answer|202:is",save_dir=${save_dir}/dparallel_prophet_dynamic_alpha${alpha}/${task} \
  --output_path ${save_dir}/dparallel_prophet_dynamic_alpha${alpha}/${task} --log_samples
done

# dParallel+prophet with EMA / z-score-based thresholds (sweep over ema_k and ema_min_progress)
ema_ks=(0.5 1.0 2.0)
ema_min_progresses=(0.1 0.2 0.3)
for k in "${ema_ks[@]}"; do
  for p in "${ema_min_progresses[@]}"; do
    echo "Running dParallel+Prophet EMA thresholding with ema_k=${k}, ema_min_progress=${p}"
    CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29603 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
    --confirm_run_unsafe_code --model llada_dist \
    --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k",use_prophet=True,ema_threshold=True,ema_k=${k},ema_min_progress=${p},analyze_gap=True,log_gap_trace=True,constraints_text="200:The|201:answer|202:is",save_dir=${save_dir}/dparallel_prophet_ema_k${k}_p${p}/${task} \
    --output_path ${save_dir}/dparallel_prophet_ema_k${k}_p${p}/${task} --log_samples
  done
done


############################################### minerva_math evaluations ###############################################
# task=minerva_math
# length=256
# block_length=32
# num_fewshot=4
# steps=256
# save_dir=/home/hice1/rbansal66/scratch/dParallel/output

# # baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="minerva_math",save_dir=${save_dir}/baseline/${task} \
# --output_path ${save_dir}/baseline/${task} --log_samples

# # dParallel
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="minerva_math",save_dir=${save_dir}/dparallel/${task} \
# --output_path ${save_dir}/dparallel/${task} --log_samples



############################################### humaneval evaluations ###############################################
# task=humaneval
# length=256
# block_length=32
# num_fewshot=0
# steps=256
# save_dir=/home/hice1/rbansal66/scratch/dParallel/output

# # baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="humaneval",save_dir=${save_dir}/baseline/${task} \
# --output_path ${save_dir}/baseline/${task} --log_samples

# # dparallel
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.5,show_speed=True,task="humaneval",save_dir=${save_dir}/dparallel/${task} \
# --output_path ${save_dir}/dparallel/${task} --log_samples

# ## NOTICE: use postprocess for humaneval
# python postprocess_code_humaneval.py {the samples_xxx.jsonl file under output_path}





############################################### mbpp evaluations ###############################################
# task=mbpp
# length=256
# block_length=32
# num_fewshot=3
# steps=256
# save_dir=/home/hice1/rbansal66/scratch/dParallel/output

# # baseline
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="mbpp",save_dir=${save_dir}/baseline/${task} \
# --output_path ${save_dir}/baseline/${task} --log_samples

# # parallel
# CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
# --confirm_run_unsafe_code --model llada_dist \
# --model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=${length},steps=${steps},block_length=${block_length},threshold=0.45,show_speed=True,task="mbpp",save_dir=${save_dir}/dparallel/${task} \
# --output_path ${save_dir}/dparallel/${task} --log_samples

## NOTICE: use postprocess for mbpp
# python postprocess_code_mbpp.py {the samples_xxx.jsonl file under output_path}

## NOTICE: for gsm8k with Prophet+dParallel, use grep-based postprocess:
# python postprocess_gsm8k_grep.py {the samples_gsm8k_*.jsonl file under output_path}

