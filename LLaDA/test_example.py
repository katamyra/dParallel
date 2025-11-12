print("starting")
from transformers import AutoTokenizer
print("importing model")
from model.modeling_llada import LLaDAModelLM
print("importing generate")
from generate import generate
print("importing torch")
import torch
print("done importing")

device = 'cuda'
model = LLaDAModelLM.from_pretrained('Zigeng/dParallel-LLaDA-8B-instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('Zigeng/dParallel-LLaDA-8B-instruct', trust_remote_code=True)

prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Please reason step by step, and put your final answer within \\boxed{}."

m = [{"role": "user", "content": prompt}, ]
prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

input_ids = tokenizer(prompt)['input_ids']
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

out = generate(model, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., threshold=0.5,remasking='low_confidence')
print("Response:",tokenizer.batch_decode(out[0][:, input_ids.shape[1]:], skip_special_tokens=True)[0])
print("NFE:",out[1])