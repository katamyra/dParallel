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
    """Original phase-aware early exit strategy using piecewise thresholds."""
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
    early_exit_thresholds=None,
    # Dynamic thresholding options (progress-based)
    dynamic_threshold=False,
    dyn_tau_min=None,
    dyn_tau_max=None,
    dyn_alpha=1.0,
    # EMA / z-score based early-exit
    ema_threshold=False,
    ema_k=2.0,
    ema_min_progress=0.3,
    # Optional logging of gap schedule for offline calibration
    log_gap_trace=False,
):
    '''LLaDA generation with Prophet early exit mechanism based on logits gap.
    
    Early-exit modes:
      - Default (no dynamic_threshold / ema_threshold): piecewise thresholds
        via should_early_exit(..., early_exit_thresholds).
      - dynamic_threshold=True: progress-based schedule
            tau(progress) = tau_min + (tau_max - tau_min) * (1 - progress) ** alpha
        where progress = current_step / max_steps.
      - ema_threshold=True: per-instance adaptive threshold using a z-score
        criterion:
            exit if progress >= ema_min_progress and
                    gap_t >= mean_gap_t + ema_k * std_gap_t
    '''
    
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

    # State for EMA / z-score thresholding
    gap_ema = None
    sq_ema = None
    ema_beta = 0.9
    ema_count = 0
    # Optional trace of (step, progress, avg_gap) for calibration
    gap_trace = [] if (analyze_gap and log_gap_trace) else None
    
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
                    progress = global_step / max_steps

                    # Optionally record trace for offline calibration
                    if gap_trace is not None:
                        gap_trace.append(
                            {
                                "step": int(global_step),
                                "progress": float(progress),
                                "avg_gap": float(avg_answer_gap),
                            }
                        )

                    # Decide whether to early exit
                    exit_now = False

                    if ema_threshold:
                        # Update running mean / variance of answer gap
                        if gap_ema is None:
                            gap_ema = avg_answer_gap
                            sq_ema = avg_answer_gap * avg_answer_gap
                            ema_count = 1
                        else:
                            gap_ema = ema_beta * gap_ema + (1.0 - ema_beta) * avg_answer_gap
                            sq_ema = ema_beta * sq_ema + (1.0 - ema_beta) * (avg_answer_gap * avg_answer_gap)
                            ema_count += 1

                        if ema_count >= 2 and progress >= ema_min_progress:
                            var = max(sq_ema - gap_ema * gap_ema, 1e-6)
                            std = var ** 0.5
                            z = (avg_answer_gap - gap_ema) / (std + 1e-6)
                            if z >= ema_k:
                                exit_now = True
                    elif dynamic_threshold:
                        # Derive tau_max / tau_min if not provided explicitly
                        tau_max = dyn_tau_max if dyn_tau_max is not None else (early_exit_thresholds or {}).get('early', 7.5)
                        tau_min = dyn_tau_min if dyn_tau_min is not None else (early_exit_thresholds or {}).get('late', 2.5)
                        # Progress-based threshold schedule
                        tau = tau_min + (tau_max - tau_min) * (1.0 - progress) ** dyn_alpha
                        exit_now = avg_answer_gap >= tau
                    else:
                        exit_now = should_early_exit(global_step, max_steps, avg_answer_gap, early_exit_thresholds)

                    if exit_now:
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
        # Optionally include full gap trace for offline calibration
        if gap_trace is not None:
            gap_data['gap_trace'] = gap_trace
        return x, global_step, gap_data
    
    return x, global_step