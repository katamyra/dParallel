# cd LLaDA
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from generate_earlyexit import generate as generate_early
from generate import generate as generate_normal
import torch

device = 'cuda'
model = LLaDAModelLM.from_pretrained('Zigeng/dParallel-LLaDA-8B-instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('Zigeng/dParallel-LLaDA-8B-instruct', trust_remote_code=True)

prompt = "Write a function to find the similar elements from the given two tuple lists."

m = [{"role": "user", "content": prompt}, ]
prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

input_ids = tokenizer(prompt)['input_ids']
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

# -- prophet
out, nfe_early, gap_data = generate_early(model, input_ids, steps=256, gen_length=256, block_length=32, analyze_gap=True, answer_start_pos=input_ids.shape[1] + 200, early_exit_thresholds={'early': 7.5, 'mid': 5.0, 'late': 2.5})


generated_text = tokenizer.decode(out[0, input_ids.shape[1]:], skip_special_tokens=True)
print(f"Generated: {generated_text}")
print(f"NFE: {nfe_early}")
print(f"Early exit: {gap_data['exit_info']['early_exit_triggered']} at step {gap_data['exit_info']['exit_decision_step']}")

# -- normal
out = generate_normal(model, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., threshold=0.5,remasking='low_confidence')

print("Response:",tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])
print("NFE:",out[1])