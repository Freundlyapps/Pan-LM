#!/usr/bin/env python3
"""QLoRA training — the config measured in preflight.py, nothing invented.

Run standalone or from the Train tab:
    ~/.venvs/punjabi-lm/bin/python train_job.py --dataset v1 --epochs 3

Everything here that looks unusual was forced by measurement on an RTX 2070 (see
STATUS.md): no expandable_segments (breaks CUDA under WSL2), no
prepare_model_for_kbit_training (fp32 upcast wants ~11 GB), regex-scoped LoRA targets
(a plain suffix list hits the vision/audio towers), and chunked cross-entropy (the
[seq x 262144] logits tensor is the real bottleneck; liger fails on sm_75).
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import argparse
import json
import math
import time
from pathlib import Path

LORA_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"


def load_jsonl(p):
    rows = []
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def encode(rows, tok, seq_len):
    """Chat-template the messages and mask the prompt so loss is on the answer only."""
    import torch
    out = []
    for r in rows:
        msgs = r["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        prompt = tok.apply_chat_template(msgs[:1], tokenize=False, add_generation_prompt=True)
        ids = tok(full, truncation=True, max_length=seq_len).input_ids
        n_prompt = len(tok(prompt, truncation=True, max_length=seq_len).input_ids)
        labels = list(ids)
        for i in range(min(n_prompt, len(labels))):
            labels[i] = -100                      # train on the completion only
        if sum(1 for l in labels if l != -100) < 8:
            continue                              # nothing to learn from
        out.append((torch.tensor(ids), torch.tensor(labels)))
    return out


def chunked_ce(model, ids, labels, chunk):
    """Loss without materializing [B, T, 262144]. See STATUS.md for why this exists."""
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    lang, head = base.model.language_model, base.lm_head
    cap = getattr(getattr(base.config, "text_config", base.config),
                  "final_logit_softcapping", None)

    h = lang(input_ids=ids).last_hidden_state[:, :-1, :]
    tgt = labels[:, 1:]

    def piece(hc, tc):
        lg = head(hc).float()
        if cap:
            lg = cap * torch.tanh(lg / cap)
        return F.cross_entropy(lg.reshape(-1, lg.size(-1)), tc.reshape(-1),
                               ignore_index=-100, reduction="sum")

    total, n = None, 0
    for i in range(0, h.size(1), chunk):
        hc, tc = h[:, i:i + chunk], tgt[:, i:i + chunk]
        valid = int((tc != -100).sum())
        if valid == 0:
            continue
        part = checkpoint(piece, hc, tc, use_reentrant=False)
        total = part if total is None else total + part
        n += valid
    if total is None:
        return None
    return total / max(n, 1)


def train(cfg, ds_version, epochs, lr, out_name, job=None, max_steps=0, save_every=100):
    log = (job.log if job else print)
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    import bitsandbytes as bnb

    data_dir = Path(cfg["data_dir"]) / "datasets" / ds_version
    train_rows = load_jsonl(data_dir / "train.jsonl")
    eval_path = data_dir / "eval.jsonl"
    eval_rows = load_jsonl(eval_path) if eval_path.exists() else []
    log(f"dataset {ds_version}: {len(train_rows)} train / {len(eval_rows)} eval")

    model_id = cfg["base_model"]
    seq_len, r, ce_chunk = cfg["seq_len"], cfg["lora_r"], cfg["ce_chunk"]
    tok = AutoTokenizer.from_pretrained(model_id)

    train_data = encode(train_rows, tok, seq_len)
    eval_data = encode(eval_rows, tok, seq_len)
    # shortest-first keeps early steps cheap and surfaces an OOM on the longest sample
    # at a predictable point rather than randomly hours in
    train_data.sort(key=lambda x: len(x[0]))
    log(f"encoded {len(train_data)} train / {len(eval_data)} eval sequences "
        f"(max {max((len(x[0]) for x in train_data), default=0)} tokens)")

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,     # Turing has no bf16 silicon
        llm_int8_enable_fp32_cpu_offload=True)

    log("loading base model in 4-bit ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_cfg, dtype="auto",
        device_map={"model.vision_tower": "cpu", "model.audio_tower": "cpu", "": 0},
        attn_implementation="eager")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=r, lora_alpha=r * 2, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=LORA_TARGETS))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"LoRA r={r}: {n_train:,} trainable")

    opt = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    accum = 16
    steps_total = max_steps or math.ceil(len(train_data) * epochs / accum)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps_total, 1))

    out_dir = Path(cfg["data_dir"]) / "adapters" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    hist = []
    if job:
        job.total = steps_total

    log(f"training: {epochs} epoch(s), ~{steps_total} optimizer steps, accum={accum}")
    model.train()
    step = micro = 0
    t0 = time.time()
    stop = False
    for ep in range(epochs):
        if stop:
            break
        for ids, labels in train_data:
            if job and job.stopped:
                log("stop requested")
                stop = True
                break
            ids = ids.unsqueeze(0).to("cuda")
            labels = labels.unsqueeze(0).to("cuda")
            try:
                loss = chunked_ce(model, ids, labels, ce_chunk)
                if loss is None:
                    continue
                (loss / accum).backward()
            except torch.cuda.OutOfMemoryError:
                log(f"OOM on a {ids.shape[1]}-token sample — skipped, cache cleared")
                opt.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            micro += 1
            if micro % accum == 0:
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if job:
                    job.done = step
                if step % 5 == 0 or step == 1:
                    vram = torch.cuda.max_memory_reserved() / 1024**3
                    el = time.time() - t0
                    hist.append({"step": step, "loss": float(loss), "epoch": ep + 1})
                    log(f"step {step}/{steps_total} ep{ep+1} loss {float(loss):.4f} "
                        f"lr {sched.get_last_lr()[0]:.2e} {vram:.2f}GB "
                        f"{el/max(step,1):.1f}s/step")
                if step % save_every == 0:
                    model.save_pretrained(out_dir)
                    log(f"checkpoint saved at step {step}")
                if max_steps and step >= max_steps:
                    stop = True
                    break

    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    log(f"saved adapter -> {out_dir}")

    if eval_data:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad():
            for ids, labels in eval_data[:40]:
                l = chunked_ce(model, ids.unsqueeze(0).to("cuda"),
                               labels.unsqueeze(0).to("cuda"), ce_chunk)
                if l is not None:
                    tot += float(l)
                    n += 1
        if n:
            log(f"eval loss {tot/n:.4f} over {n} example(s)  (ppl {math.exp(tot/n):.1f})")
    log(f"FINISHED in {(time.time()-t0)/60:.0f} min")
    return str(out_dir)


def main():
    import config
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-steps", type=int, default=0)
    args = ap.parse_args()
    cfg = config.load()
    train(cfg, args.dataset, args.epochs, args.lr,
          args.out or f"{args.dataset}-r{cfg['lora_r']}", max_steps=args.max_steps)


if __name__ == "__main__":
    main()
