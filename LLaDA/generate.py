# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
import os
from transformers import AutoTokenizer, AutoModel
from model.modeling_llada import LLaDAModelLM

def add_gumbel_noise(logits, temperature):
    '''
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    '''
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1) # b, l

    if remasking == 'low_confidence':
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1) # b, l
    elif remasking == 'random':
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


def get_transfer_index_entropy(logits, temperature, remasking, mask_index, x, num_transfer_tokens, entropy_threshold=None):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
    
    # Calculate entropy instead of confidence
    p = F.softmax(logits.to(torch.float64), dim=-1)
    
    if remasking == 'low_confidence':
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)  # b, l
    elif remasking == 'random':
        # For random remasking, use random entropy values
        entropy = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)
    
    x0 = torch.where(mask_index, x0, x)
    
    # Use entropy instead of confidence (note: higher entropy means lower confidence)
    # So we set entropy to +inf for non-masked positions (to exclude them from selection)
    entropy_for_selection = torch.where(mask_index, entropy, torch.inf)
    
    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    
    if entropy_threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    
    for j in range(entropy_for_selection.shape[0]):
        # Select tokens with lowest entropy (highest confidence)
        # topk with largest=False gives us the smallest values (lowest entropy)
        _, select_index = torch.topk(entropy_for_selection[j], k=num_transfer_tokens[j], largest=False)
        transfer_index[j, select_index] = True
        
        if entropy_threshold is not None:
            # Only keep tokens with entropy below threshold (high confidence)
            # If entropy > threshold, set transfer_index to False
            for k in range(1, num_transfer_tokens[j]):  # Skip the first one (lowest entropy)
                if entropy[j, select_index[k]] > entropy_threshold:
                    transfer_index[j, select_index[k]] = False
    
    return x0, transfer_index


def should_early_exit(current_step, max_steps, answer_gap, thresholds=None):
    """Phase-aware early exit strategy."""
    if answer_gap is None:
        return False
    
    # Use default or provided thresholds
    if thresholds is None:
        thresholds = {'early': 7.5, 'mid': 5.0, 'late': 2.5}
    
    progress = current_step / max_steps
    
    # Phase-based thresholds
    if progress < 0.33:  # Early phase
        return answer_gap >= thresholds.get('early', 7.5)
    elif progress < 0.67:  # Mid phase  
        return answer_gap >= thresholds.get('mid', 5.0)
    else:  # Late phase
        return answer_gap >= thresholds.get('late', 2.5)


@ torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    threshold=None
):
    '''
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    '''
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)

        i = 0

        while True:
            nfe += 1
            mask_index = (x == mask_id)

            logits = model(x).logits
            mask_index[:, block_end:] = 0

            if threshold is not None:
                x0, transfer_index = get_transfer_index_entropy(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)
            else:
                x0, transfer_index = get_transfer_index(logits, temperature, remasking, mask_index, x, num_transfer_tokens[:, i] if threshold is None else None, threshold)

            x[transfer_index] = x0[transfer_index]
            i += 1
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
        # if x[:, prompt.shape[1] + (num_block + 1) * block_length-1] == 126081:
        #     x[:, prompt.shape[1] + (num_block + 1) * block_length:] = 126081
        #     break
    return x, nfe


@torch.no_grad()
def generate_prophet(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.,
    remasking='low_confidence',
    mask_id=126336,
    threshold=None,
    analyze_gap=False,
    answer_start_pos=None,
    early_exit_thresholds=None
):
    '''LLaDA generation with Prophet early exit mechanism based on logits gap.'''
    
    # Initialize
    early_exit_triggered = False
    exit_decision_step = None
    
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(model.device)
    x[:, :prompt.shape[1]] = prompt.clone()
        
    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks
    
    global_step = 0
    max_steps = steps
    
    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = (x[:, block_start:block_end] == mask_id)
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
        
        for i in range(steps_per_block):
            global_step += 1
            mask_index = (x == mask_id)
            
            logits = model(x).logits
            
            # Early exit check based on logits gap
            if analyze_gap and answer_start_pos is not None:
                # Calculate answer region
                answer_length = 5
                answer_positions = list(range(answer_start_pos, min(prompt.shape[1] + gen_length, answer_start_pos + answer_length)))
                
                # Analyze gap in answer region
                gen_start = prompt.shape[1]
                gen_logits = logits[:, gen_start:, :]
                
                answer_gaps = []
                for pos in answer_positions:
                    if pos >= gen_start and pos < logits.shape[1]:
                        rel_pos = pos - gen_start
                        if rel_pos < gen_logits.shape[1]:
                            # Get top-2 logits
                            top2_vals, _ = torch.topk(gen_logits[:, rel_pos, :], k=2, dim=-1)
                            gap = (top2_vals[0, 0] - top2_vals[0, 1]).item()
                            answer_gaps.append(gap)
                
                # Check early exit condition
                if answer_gaps and not early_exit_triggered:
                    avg_answer_gap = sum(answer_gaps) / len(answer_gaps)
                    
                    if should_early_exit(global_step, max_steps, avg_answer_gap, early_exit_thresholds):
                        print(f"Early exit at step {global_step}/{max_steps} with gap={avg_answer_gap:.3f}")
                        exit_decision_step = global_step
                        early_exit_triggered = True
                        
                        # Fill remaining masks
                        logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                        x0 = torch.argmax(logits_with_noise, dim=-1)
                        remaining_mask = (x == mask_id)
                        x[remaining_mask] = x0[remaining_mask]
                        break
            
            # Mask out tokens beyond current block
            mask_index[:, block_end:] = False
            
            if threshold is not None:
                x0, transfer_index = get_transfer_index_entropy(
                    logits, temperature, remasking, mask_index, x, 
                    num_transfer_tokens[:, i], threshold
                )
            else:
                x0, transfer_index = get_transfer_index(
                    logits, temperature, remasking, mask_index, x, 
                    num_transfer_tokens[:, i], threshold
                )
                        
            x[transfer_index] = x0[transfer_index]
            if (x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length] == mask_id).sum() == 0:
                break
        
        # Break outer loop if early exit triggered
        if early_exit_triggered:
            break
    
    # Return results
    if analyze_gap:
        gap_data = {
            'exit_info': {
                'early_exit_triggered': early_exit_triggered,
                'exit_decision_step': exit_decision_step,
                'total_steps': max_steps,
                'actual_steps': exit_decision_step if early_exit_triggered else max_steps
            }
        }
        return x, global_step, gap_data
    
    return x, global_step