import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from tqdm import tqdm
import math

def evaluate_autoregressive(model, dataset, tokenizer, device, seqlen):
    logging.info("Starting autoregressive evaluation...")
    loss = nn.CrossEntropyLoss()
    tot = 0.0
    correct_predictions = 0
    total_predictions = 0
    max_memory = 0
    memory_usage = []
    times = []
    # 拼接所有样本文本
    texts = [ex["text"] for ex in dataset if ex["text"].strip()]
    full_text = "\n\n".join(texts)
    input_ids = tokenizer(full_text, return_tensors="pt").input_ids[:, :seqlen].to(device)
    cache = {"past": None}
    with torch.no_grad():
        attention_mask = torch.ones((1, input_ids.numel()), device=device)
        for i in tqdm(range(input_ids.numel()), desc="Autoregressive evaluation"):
            tick = time.time()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids[:, i:i+1],
                    past_key_values=cache["past"],
                    attention_mask=attention_mask[:, :(i+1)].reshape((1, -1)),
                )
            times.append(time.time() - tick)
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024
            max_memory = max(max_memory, current_memory)
            memory_usage.append(current_memory)
            if i != input_ids.numel() - 1:
                tot += loss(out.logits[0].to(device), input_ids[:, (i+1)].to(device)).float()
                predictions = torch.argmax(out.logits[0], dim=-1)
                actual_next_token = input_ids[:, (i+1)]
                correct_predictions += (predictions == actual_next_token).sum().item()
                total_predictions += 1
            cache["past"] = out.past_key_values
            del out
    median_time = torch.tensor(times).median().item()
    ppl = torch.exp(tot / (input_ids.numel() - 1)).item()
    accuracy = 100 * correct_predictions / total_predictions
    logging.info("Autoregressive evaluation completed")
    return {
        'PPL': ppl,
        'Accuracy': accuracy,
        'Max Memory (MB)': max_memory,
        'Median Time (s)': median_time,
        'Memory Usage': memory_usage,
        'Times': times
    }

def evaluate_strict(model, dataset, tokenizer, device, max_samples=100, seqlen=2048):
    logging.info("Starting strict evaluation...")
    logging.info(f"seqlen: {seqlen}")
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    sample_count = 0
    if device == "cuda":
        torch.cuda.synchronize()
    pbar = tqdm(total=max_samples, desc="Strict evaluation")
    for example in dataset:
        if sample_count >= max_samples:
            break
        text = example["text"]
        if not text.strip():
            continue
        try:
            input_ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
            if len(input_ids) < 2:
                continue
            input_ids = input_ids[:seqlen]
            if len(input_ids) < 2:
                continue
            for i in range(1, len(input_ids)):
                context = input_ids[:i].unsqueeze(0)
                target = input_ids[i].unsqueeze(0)
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = model(context)
                    logits = outputs.logits[:, -1, :]
                logits = logits.to(device)
                target = target.to(device)
                log_prob = F.log_softmax(logits, dim=-1)
                token_log_prob = log_prob[0, target.item()]
                total_loss -= token_log_prob.item()
                pred_token = torch.argmax(logits, dim=-1)[0]
                if pred_token.item() == target.item():
                    correct_tokens += 1
                total_tokens += 1
                if device == "cuda":
                    torch.cuda.synchronize()
            sample_count += 1
            pbar.update(1)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logging.error(f"CUDA error occurred at sample {sample_count}, token {i}: {str(e)}")
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    pbar.close()
    ppl = math.exp(total_loss / total_tokens)
    acc = correct_tokens / total_tokens
    logging.info("Strict evaluation completed")
    return ppl, acc 

def evaluate_sliding_window(model, dataset, tokenizer, device, window_size=2048, stride=1, max_samples=100):
    """
    使用滑动窗口对模型进行评测。
    先用长度1到window_size的窗口分别预测下一个token，
    然后用固定长度window_size的窗口滑动预测，直到样本结束。
    统计整体的 loss、ppl、accuracy。
    """
    logging.info("Starting sliding window evaluation...")
    total_loss = 0.0
    total_tokens = 0
    correct_tokens = 0
    sample_count = 0
    if device == "cuda":
        torch.cuda.synchronize()
    pbar = tqdm(total=max_samples, desc="Sliding window evaluation")
    for example in dataset:
        if sample_count >= max_samples:
            break
        text = example["text"]
        if not text.strip():
            continue
        try:
            input_ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)
            L = len(input_ids)
            if L < 2:
                continue
            # 1. 前期递增窗口
            for win_len in range(1, min(window_size, L)):
                context = input_ids[:win_len].unsqueeze(0)
                target = input_ids[win_len].unsqueeze(0)
                with torch.no_grad():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        outputs = model(context)
                    logits = outputs.logits[:, -1, :]
                logits = logits.to(device)
                target = target.to(device)
                log_prob = F.log_softmax(logits, dim=-1)
                token_log_prob = log_prob[0, target.item()]
                total_loss -= token_log_prob.item()
                pred_token = torch.argmax(logits, dim=-1)[0]
                if pred_token.item() == target.item():
                    correct_tokens += 1
                total_tokens += 1
                if device == "cuda":
                    torch.cuda.synchronize()
            # 2. 固定窗口滑动
            if L > window_size:
                for start in range(0, L - window_size, stride):
                    context = input_ids[start:start+window_size].unsqueeze(0)
                    target = input_ids[start+window_size].unsqueeze(0)
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            outputs = model(context)
                        logits = outputs.logits[:, -1, :]
                    logits = logits.to(device)
                    target = target.to(device)
                    log_prob = F.log_softmax(logits, dim=-1)
                    token_log_prob = log_prob[0, target.item()]
                    total_loss -= token_log_prob.item()
                    pred_token = torch.argmax(logits, dim=-1)[0]
                    if pred_token.item() == target.item():
                        correct_tokens += 1
                    total_tokens += 1
                    if device == "cuda":
                        torch.cuda.synchronize()
            sample_count += 1
            pbar.update(1)
        except RuntimeError as e:
            if "CUDA" in str(e):
                logging.error(f"CUDA error occurred at sample {sample_count}: {str(e)}")
                if device == "cuda":
                    torch.cuda.empty_cache()
                continue
            else:
                raise e
    pbar.close()
    if total_tokens == 0:
        ppl = float('inf')
        acc = 0.0
    else:
        ppl = math.exp(total_loss / total_tokens)
        acc = correct_tokens / total_tokens
    logging.info("Sliding window evaluation completed")
    return ppl, acc

# a method of comparing the changes between base model and quantized model
# def compare_models(model1, model2, dataset, tokenizer, device, max_samples=100, seqlen=2048):