# cd LLaDA
from transformers import AutoTokenizer
from model.modeling_llada import LLaDAModelLM
from generate import generate, generate_prophet
import torch

def _parse_constraints(text: str, tokenizer) -> dict[int, int]:
    """Parse constraint string like "120:THE|121:ANSWER" into position->token_id dict."""
    constraints: dict[int, int] = {}
    if text is None or text.strip() == "":
        return constraints
    for part in text.split('|'):
        if ':' not in part:
            continue
        pos_str, word = part.split(':', 1)
        try:
            pos = int(pos_str.strip())
        except ValueError:
            continue
        word = word.strip()
        # Prepend space for tokenization consistency
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        for i, tid in enumerate(ids):
            constraints[pos + i] = tid
    return constraints

device = 'cuda'
# dParallel_Dream_7B_Instruct
# model = LLaDAModelLM.from_pretrained('Zigeng/dParallel-LLaDA-8B-instruct', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device).eval()
model = LLaDAModelLM.from_pretrained('Zigeng/dParallel-LLaDA-8B-instruct', torch_dtype=torch.bfloat16).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained('Zigeng/dParallel-LLaDA-8B-instruct', trust_remote_code=True)

# prompt = "Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?"
# prompt = "Julie is reading a 120-page book. Yesterday, she was able to read 12 pages and today, she read twice as many pages as yesterday. If she wants to read half of the remaining pages tomorrow, how many pages should she read?"
# prompt = "Tobias is buying a new pair of shoes that costs $95. He has been saving up his money each month for the past three months. He gets a $5 allowance a month. He also mows lawns and shovels driveways. He charges $15 to mow a lawn and $7 to shovel. After buying the shoes, he has $15 in change. If he mows 4 lawns, how many driveways did he shovel?"
# prompt = "Rachel and Sara want to attend a beauty and modeling contest. They both want to buy new pairs of shoes and dresses. Sara buys a pair of shoes which costs $50 and a dress which costs $200. How much should Rachel budget if she wants to spend twice as much as what Sara spent on the pair of shoes and dress?"
prompt = "It's Ava's birthday party. Her parents bought a unicorn piñata for $13 and filled it with all of her favorite treats. They bought 4 bags of Reese's for $9 per bag, 3 bags of Snickers for $5 per bag, and 5 bags of Skittles for $7 per bag. How much did the unicorn piñata and the treats cost altogether?"

m = [{"role": "user", "content": prompt}, ]
prompt = tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)

input_ids = tokenizer(prompt)['input_ids']
input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

# -- prophet
constraints_text="200:The|201:answer|202:is"
constraints = _parse_constraints(constraints_text, tokenizer)
answer_start = max(constraints.keys()) + 2 if constraints else 0
answer_start_pos = input_ids.shape[1] + answer_start

out, nfe_early, gap_data = generate_prophet(model, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., threshold=0.5,remasking='low_confidence', analyze_gap=True, answer_start_pos=answer_start_pos, early_exit_thresholds={'early': 9.0, 'mid': 7.0, 'late': 5.0})

generated_text = tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
print(f"Generated: {generated_text}")
print(f"NFE: {nfe_early}")
print(f"Early exit: {gap_data['exit_info']['early_exit_triggered']} at step {gap_data['exit_info']['exit_decision_step']}")

print("\n--------------------------------\n")

# -- normal
out, nfe = generate(model, input_ids, steps=256, gen_length=256, block_length=32, temperature=0., threshold=0.5,remasking='low_confidence')

print("Response:",tokenizer.batch_decode(out[:, input_ids.shape[1]:], skip_special_tokens=True)[0])
print("NFE:",nfe)
# print(f"Early exit: {gap_data['exit_info']['early_exit_triggered']} at step {gap_data['exit_info']['exit_decision_step']}")