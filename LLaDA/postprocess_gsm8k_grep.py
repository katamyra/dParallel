#!/usr/bin/env python

# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Grep-style GSM8K postprocessing:
#   If the gold numeric answer appears anywhere in the generated text,
#   we count the sample as correct. This is robust to Prophet+dParallel
#   placing the final boxed answer in an unexpected location.

import json
import re
import sys
from typing import Optional


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract canonical numeric answer from GSM8K-style answer string.

    Prefer the number after a '####' marker, otherwise fall back to the
    last number in the string. Returns a normalized string with commas
    removed and trailing periods stripped, e.g. '1,234.' -> '1234'.
    """
    if text is None:
        return None

    # Prefer explicit '#### answer' pattern
    m = re.search(r"####\s*([-+]?[0-9][0-9,\.]*)", text)
    if not m:
        # Fall back: any number, use the last one
        nums = re.findall(r"[-+]?[0-9][0-9,\.]*", text)
        if not nums:
            return None
        raw = nums[-1]
    else:
        raw = m.group(1)

    # Normalize: strip spaces, commas, trailing dots
    norm = raw.strip().replace(",", "")
    norm = norm.rstrip(".")
    return norm


def extract_prediction_text(sample: dict) -> str:
    """Get the raw generated text from a lm-eval-harness samples jsonl record.

    For GSM8K with multiple filters (strict-match / flexible-extract),
    the JSONL contains duplicate entries per doc. We always want the
    *full generation text* here, not the filtered numeric string
    (and certainly not '[invalid]' from strict-match), so we:
      1. Prefer `resps` (the full decoded output).
      2. Only fall back to `filtered_resps` if `resps` is missing.
    """
    # Prefer raw responses
    rs = sample.get("resps")
    if isinstance(rs, list) and rs:
        first = rs[0]
        if isinstance(first, list) and first:
            return str(first[0])
        return str(first)

    # Fallback: filtered responses (e.g., numeric extraction)
    fr = sample.get("filtered_resps")
    if isinstance(fr, list) and fr:
        if isinstance(fr[0], list):
            return str(fr[0][0])
        return str(fr[0])

    # Older lm-eval versions sometimes use 'decoded'
    return str(sample.get("decoded", ""))


def extract_gold_answer(sample: dict) -> Optional[str]:
    """Get normalized gold numeric answer from a samples jsonl record."""
    target = sample.get("target")
    if isinstance(target, str):
        ans = extract_numeric_answer(target)
        if ans is not None:
            return ans

    doc = sample.get("doc", {})
    if isinstance(doc, dict):
        # GSM8K docs typically store the full solution in 'answer'
        maybe = doc.get("answer")
        if isinstance(maybe, str):
            ans = extract_numeric_answer(maybe)
            if ans is not None:
                return ans

    return None


def contains_gold_number(prediction: str, gold_num: str) -> bool:
    """Return True if any numeric token in prediction equals gold_num."""
    if gold_num is None:
        return False

    nums = re.findall(r"[-+]?[0-9][0-9,\.]*", prediction)
    for n in nums:
        norm = n.strip().replace(",", "").rstrip(".")
        if norm == gold_num:
            return True
    return False


def main(path: str):
    total = 0
    correct = 0

    for sample in read_jsonl(path):
        # GSM8K samples JSONL contains one entry per (doc, filter_name).
        # We only want to count each problem once, and to align with the
        # "flexible-extract" metric in results_*.json we *skip* strict-match
        # entries here.
        if sample.get("filter") == "strict-match":
            continue

        total += 1

        gold = extract_gold_answer(sample)
        pred_text = extract_prediction_text(sample)

        if gold is None:
            # If we cannot parse a numeric gold answer, skip this item
            continue

        if contains_gold_number(pred_text, gold):
            correct += 1

    if total == 0:
        print("No samples found.")
        return

    acc = correct / total
    print(f"GSM8K grep-style accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/samples_gsm8k_*.jsonl", file=sys.stderr)
        sys.exit(1)
    main(sys.argv[1])


