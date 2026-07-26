#!/usr/bin/env python3
"""Generation — base vs. tuned, the only honest way to see if training helped.

Loads the 4-bit base once and hot-swaps LoRA adapters onto it, so comparing base against
a tuned adapter (or two adapters) does not reload 9.6 GB each time.
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

from pathlib import Path

_STATE = {"model": None, "tok": None, "base_id": None, "adapters": set()}


def _ensure_base(cfg, log=print):
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
    if _STATE["model"] is not None and _STATE["base_id"] == cfg["base_model"]:
        return
    log("loading base model in 4-bit ...")
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16, llm_int8_enable_fp32_cpu_offload=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"], quantization_config=bnb, dtype="auto",
        device_map={"model.vision_tower": "cpu", "model.audio_tower": "cpu", "": 0},
        attn_implementation="eager")
    model.config.use_cache = True                     # generation wants the KV cache
    _STATE.update(model=model, base_id=cfg["base_model"],
                  tok=AutoTokenizer.from_pretrained(cfg["base_model"]), adapters=set())


def _load_adapter(cfg, adapter_dir, log=print):
    """Attach an adapter under its own name; reuse if already attached."""
    from peft import PeftModel
    name = Path(adapter_dir).name
    m = _STATE["model"]
    if name in _STATE["adapters"]:
        return name
    if not isinstance(m, object) or not hasattr(m, "load_adapter"):
        _STATE["model"] = PeftModel.from_pretrained(m, adapter_dir, adapter_name=name)
    else:
        _STATE["model"].load_adapter(adapter_dir, adapter_name=name)
    _STATE["adapters"].add(name)
    log(f"adapter '{name}' ready")
    return name


def generate(cfg, prompt, adapter=None, temp=0.7, top_p=0.9, max_new=400,
             repetition_penalty=1.15, log=print):
    """One generation. adapter=None or 'base' uses the untuned model.

    'base' is served from the SAME loaded model with adapters disabled, whether or not
    any adapter is attached yet — so a base/tuned comparison never reloads the 9.6 GB base.

    Decoding note: a high repetition_penalty (was 1.3) starves the common vocabulary on long
    generations and pushes the model into rare/garbage tokens (other scripts, brackets) — the
    'salad' failure. 1.15 + no_repeat_ngram_size stops loops without that collapse.
    """
    import contextlib
    import torch
    _ensure_base(cfg, log)
    tok = _STATE["tok"]
    want_base = (not adapter) or adapter == "base"

    if not want_base:
        name = _load_adapter(cfg, Path(cfg["data_dir"]) / "adapters" / adapter, log)
        _STATE["model"].set_adapter(name)
    model = _STATE["model"]

    msgs = [{"role": "user", "content": prompt}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    if not isinstance(enc, torch.Tensor):        # transformers 5.x returns a BatchEncoding
        enc = enc["input_ids"]
    inputs = enc.to("cuda")
    # For base output, disable any attached adapter for the duration of the call.
    disable = (want_base and hasattr(model, "disable_adapter"))
    with torch.no_grad(), (model.disable_adapter() if disable
                           else contextlib.nullcontext()):
        out = model.generate(inputs, max_new_tokens=max_new, do_sample=temp > 0,
                             temperature=max(temp, 1e-4), top_p=top_p,
                             repetition_penalty=repetition_penalty, no_repeat_ngram_size=4,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()


def compare(cfg, prompt, adapter, **kw):
    """Base vs. tuned on the same prompt."""
    base = generate(cfg, prompt, adapter="base", **kw)
    tuned = generate(cfg, prompt, adapter=adapter, **kw) if adapter else "(no adapter chosen)"
    return base, tuned


def list_adapters(cfg):
    d = Path(cfg["data_dir"]) / "adapters"
    if not d.exists():
        return []
    return sorted([p.name for p in d.iterdir() if (p / "adapter_config.json").exists()],
                  reverse=True)
