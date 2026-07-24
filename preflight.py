#!/usr/bin/env python3
"""Preflight: can this GPU actually QLoRA-train Gemma 4 E2B?

Loads the model in 4-bit NF4, attaches LoRA, and runs real training steps on Gurmukhi
text while measuring peak VRAM, peak system RAM, and loss stability. Prints a GO/NO-GO
verdict and the settings that fit.

This is the go/no-go gate — nothing else in the project is worth building until it passes.
Later wired to the Train tab's "Preflight" button.

Usage:
  ~/.venvs/punjabi-lm/bin/python preflight.py
  ~/.venvs/punjabi-lm/bin/python preflight.py --seq-len 512 --lora-r 8
"""
import os

# DO NOT enable expandable_segments here. It is standard anti-fragmentation advice on
# native Linux, but under WSL2 it relies on CUDA virtual-memory APIs (cuMemCreate/cuMemMap)
# that the GPU-paravirtualization layer supports poorly: every allocation failed with
# "CUDA driver error: out of memory" while nvidia-smi still reported 1.3 GB free.
# Measured 2026-07-24 — removing it was the difference between no training and training.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "garbage_collection_threshold:0.8")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import argparse
import resource
import subprocess
import sys
import time
from pathlib import Path

MODEL_DEFAULT = "google/gemma-4-E2B-it"

# Real Gurmukhi, not lorem ipsum — tokenizer behaviour on this script is part of what
# we are measuring. Falls back to this if the Transcribe corpus is unavailable.
FALLBACK_TEXT = """ਚੰਨ ਚਾਨਣੀ ਰਾਤ ਤਾਰਾ ਕੋਈ ਕੋਈ ਆ
ਵਿਆਹ ਮੁਕਲਾਵਾ ਕੱਠਾ ਜੋੜੀ ਨਵੀਂ ਨਰੋਈ ਆ
ਮੇਰੇ ਹੋਠ ਪਿਆਸੇ ਸੀ, ਓਦੇ ਮੂੰਹ ਦੇ ਹਾਸੇ ਸੀ
ਜਦ ਮੈਂ ਸ਼ਰਬਤ ਦੀ ਬੋਤਲ ਵਿੱਚ ਜੀਭ ਡਬੋਈ ਆ
ਉਹ ਫੁੱਲ ਖਿੜਿਆ ਤੋਰੀ ਦਾ, ਰੰਗ ਪਤਲੀ ਗੋਰੀ ਦਾ
ਬੁੱਕਲ ਦੇ ਵਿੱਚ ਇੰਝ ਲੱਗਦੀ ਜਿਵੇਂ ਕੂੰਜ ਲੁਕੋਈ ਆ"""

CORPUS_SAMPLE = Path("/mnt/e/Transcribe/corpus/03. CHAN CHANNI RAAT.txt")


def chunked_ce_loss(model, input_ids, chunk):
    """Cross-entropy without ever materializing the full [B, T, 262144] logits tensor.

    That tensor is the real VRAM bottleneck: ~0.65 GB per 128 tokens, doubled by the
    softcapping temp. liger-kernel solves this with fused triton kernels, but every liger
    kernel fails to compile on Turing (ptxas exit 255 for sm_75), so this does it in plain
    PyTorch instead.

    The trick is torch.utils.checkpoint: logits for a chunk are freed after the forward
    pass and recomputed during backward, so only one chunk's logits are ever resident.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    lang, lm_head = base.model.language_model, base.lm_head
    softcap = getattr(getattr(base.config, "text_config", base.config),
                      "final_logit_softcapping", None)

    h = lang(input_ids=input_ids).last_hidden_state[:, :-1, :]   # drop last position
    tgt = input_ids[:, 1:]                                        # next-token targets

    def piece(hc, tc):
        logits = lm_head(hc).float()
        if softcap:
            logits = softcap * torch.tanh(logits / softcap)
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               tc.reshape(-1), reduction="sum")

    total, n = None, 0
    for i in range(0, h.size(1), chunk):
        hc, tc = h[:, i:i + chunk], tgt[:, i:i + chunk]
        part = checkpoint(piece, hc, tc, use_reentrant=False)
        total = part if total is None else total + part
        n += tc.numel()
    return total / max(n, 1)


def gpu_free_mib():
    """Free VRAM as the driver sees it — includes Windows-side usage under WSL2."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        total, used, free = (int(x) for x in out.stdout.strip().split(","))
        return total, used, free
    except Exception:
        return None, None, None


def peak_rss_gb():
    """Peak resident set size of this process. ru_maxrss is KiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


def load_model(model_id, bnb_config, device_map, max_memory, log):
    """Gemma 4 is multimodal; the text-only path may live under a different auto-class.

    Try the causal-LM class first (what we want for text training), then fall back
    through the multimodal classes the model card mentions.
    """
    import transformers
    candidates = [
        "AutoModelForCausalLM",
        "AutoModelForMultimodalLM",
        "AutoModelForImageTextToText",
        "AutoModelForPreTraining",
    ]
    errors = []
    for name in candidates:
        cls = getattr(transformers, name, None)
        if cls is None:
            errors.append(f"{name}: not present in transformers {transformers.__version__}")
            continue
        try:
            log(f"   trying {name} ...")
            kw = dict(
                quantization_config=bnb_config,
                dtype="auto",
                device_map=device_map,
                attn_implementation="eager",  # Turing: no flash-attn-2
            )
            if max_memory:
                kw["max_memory"] = max_memory
            model = cls.from_pretrained(model_id, **kw)
            log(f"   loaded via {name}")
            return model, name
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:200]}")
    raise RuntimeError("could not load model.\n  " + "\n  ".join(errors))


def main():
    ap = argparse.ArgumentParser(description="Gemma 4 E2B QLoRA preflight on 8GB VRAM")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--attn-only", action="store_true",
                    help="LoRA on attention projections only (lower memory)")
    ap.add_argument("--cpu-offload", action="store_true",
                    help="Spill layers that don't fit onto CPU RAM (device_map=auto). "
                         "Slower per step, but survives a tight GPU.")
    ap.add_argument("--gpu-mem", default="5GiB",
                    help="VRAM cap when --cpu-offload is set (leave headroom for activations)")
    ap.add_argument("--cpu-mem", default="5GiB",
                    help="CPU RAM cap when --cpu-offload is set")
    ap.add_argument("--peft-prepare", action="store_true",
                    help="Use peft prepare_model_for_kbit_training (upcasts to fp32; OOMs here)")
    ap.add_argument("--keep-towers", dest="text_only", action="store_false",
                    help="Keep vision/audio towers on GPU (default: park them on CPU)")
    ap.add_argument("--tagged", action="store_true",
                    help="Wrap the sample in structural tags (tagfmt.build_compact)")
    ap.add_argument("--chunked-ce", action="store_true",
                    help="Chunked cross-entropy in pure PyTorch. Avoids the full logits "
                         "tensor without triton, so it works on Turing (liger does not).")
    ap.add_argument("--ce-chunk", type=int, default=128,
                    help="Tokens per loss chunk when --chunked-ce is set")
    ap.add_argument("--liger", action="store_true",
                    help="Fused linear+cross-entropy via liger-kernel. Never materializes the "
                         "[seq x 262144] logits tensor -- the real VRAM bottleneck.")
    args = ap.parse_args()

    log = lambda m: print(m, flush=True)
    t0 = time.time()

    log("=" * 64)
    log("PREFLIGHT — Gemma 4 E2B QLoRA on RTX 2070 (8 GB, Turing)")
    log("=" * 64)

    total, used, free = gpu_free_mib()
    if total:
        log(f"GPU before load : {used} MiB used / {total} MiB total  ->  {free} MiB free")

    import torch
    log(f"torch           : {torch.__version__}  CUDA {torch.version.cuda}")
    if not torch.cuda.is_available():
        log("\nFAIL: CUDA not available.")
        return 2
    cap = torch.cuda.get_device_capability(0)
    log(f"device          : {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}")
    log(f"bf16 supported  : {torch.cuda.is_bf16_supported()}  (Turing = False, so fp16)")

    import transformers
    from transformers import AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    log(f"transformers    : {transformers.__version__}")

    # ---- tokenizer + Gurmukhi fertility -------------------------------------------
    log("\n-- tokenizer --")
    tok = AutoTokenizer.from_pretrained(args.model)
    text = CORPUS_SAMPLE.read_text(encoding="utf-8") if CORPUS_SAMPLE.exists() else FALLBACK_TEXT
    src = "Transcribe corpus" if CORPUS_SAMPLE.exists() else "built-in sample"
    if args.tagged:
        import tagfmt
        text = tagfmt.build_compact(text, title="ਚੰਨ ਚਾਨਣੀ ਰਾਤ", form="kali",
                                    theme="ਵਿਆਹ ਮੁਕਲਾਵਾ")
        src += " + structural tags"
    n_tok = len(tok(text).input_ids)
    n_word = len(text.split())
    log(f"   vocab {len(tok):,} | sample: {src}")
    log(f"   {n_word} words -> {n_tok} tokens  =  {n_tok / max(n_word,1):.2f} tokens/word")
    log(f"   (>4 means Gurmukhi is expensive: shorter usable context, slower generation)")

    # ---- load in 4-bit ------------------------------------------------------------
    if args.liger:
        # Must patch BEFORE the model is constructed -- it swaps the loss path in the class.
        from liger_kernel.transformers import apply_liger_kernel_to_gemma4
        # rms_norm and geglu default to True but their triton kernels fail to compile on
        # Turing (ptxas exit 255 for sm_75). Only the fused linear+CE is needed here --
        # that is the one that avoids materializing [seq x 262144] logits.
        apply_liger_kernel_to_gemma4(
            fused_linear_cross_entropy=True, cross_entropy=False,
            rms_norm=False, geglu=False, rope=False, layer_norm=False)
        log("\n-- liger fused linear+CE patched (rms_norm/geglu off: no sm_75 support) --")

    log("\n-- loading model in 4-bit NF4 --")
    # Any CPU dispatch of a quantized model needs this flag -- including the cheap
    # text-only trick of parking the vision/audio towers off-GPU.
    any_cpu = args.cpu_offload or args.text_only
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,  # Turing has no bf16 silicon
        llm_int8_enable_fp32_cpu_offload=any_cpu,
    )
    if args.cpu_offload:
        device_map, max_memory = "auto", {0: args.gpu_mem, "cpu": args.cpu_mem}
        log(f"   GPU+CPU: cap {args.gpu_mem} VRAM / {args.cpu_mem} CPU RAM")
    else:
        device_map, max_memory = {"": 0}, None
        log("   GPU-only (add --cpu-offload to spill onto CPU RAM)")

    if args.text_only:
        # E2B is multimodal. For a text-only songwriter the vision and audio towers are
        # never invoked, so parking them in CPU RAM frees VRAM at zero cost -- no PCIe
        # traffic, because nothing ever reads them.
        if isinstance(device_map, dict):
            device_map = {"model.vision_tower": "cpu", "model.audio_tower": "cpu",
                          **device_map}
        log("   text-only: vision + audio towers -> CPU")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model, cls_used = load_model(args.model, bnb_config, device_map, max_memory, log)

    dm = getattr(model, "hf_device_map", {}) or {}
    if dm:
        from collections import Counter
        log(f"   placement: {dict(Counter(str(d) for d in dm.values()))}")
    on_gpu = sum(p.numel() for p in model.parameters() if p.device.type == "cuda")
    on_cpu = sum(p.numel() for p in model.parameters() if p.device.type == "cpu")
    log(f"   params: {on_gpu/1e9:.2f}B on GPU, {on_cpu/1e9:.2f}B on CPU")

    after_load = torch.cuda.memory_allocated() / 1024**3
    log(f"   VRAM after load: {after_load:.2f} GB allocated")

    # ---- attach LoRA --------------------------------------------------------------
    log("\n-- attaching LoRA --")
    if args.peft_prepare:
        # peft's helper upcasts every non-quantized param to fp32. Gemma 4 keeps ~2.8B
        # params as 16-bit embedding tables (262K vocab + per-layer embeddings), so this
        # tries to allocate ~11 GB and OOMs on an 8 GB card. Off by default.
        log("   using peft prepare_model_for_kbit_training (fp32 upcast — memory heavy)")
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        log("   skipping peft fp32 upcast (keeps embeddings 16-bit)")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    # Target the LANGUAGE MODEL only. A bare ["q_proj", ...] list matches by suffix across
    # the whole model and also hits the vision/audio towers, whose projections are wrapped
    # in Gemma4ClippableLinear -- a type peft cannot adapt. A regex scopes it properly, and
    # we don't want to train the towers for a text-only songwriter anyway.
    inner = r"self_attn\.[qkvo]_proj" if args.attn_only else \
            r"(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
    targets = rf".*language_model\.layers\.\d+\.{inner}"
    model = get_peft_model(model, LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=targets,
    ))
    n_lora = len([n for n, _ in model.named_modules() if n.endswith("lora_A.default")])
    log(f"   adapters injected into {n_lora} modules")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    log(f"   r={args.lora_r} on {len(targets)} module types")
    log(f"   trainable {trainable:,} / {total_p:,}  ({100*trainable/total_p:.3f}%)")

    # ---- train steps --------------------------------------------------------------
    log(f"\n-- running {args.steps} steps @ seq_len={args.seq_len} batch={args.batch_size} --")
    import bitsandbytes as bnb
    opt = bnb.optim.PagedAdamW8bit(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4)

    ids = tok(text, return_tensors="pt", truncation=True, max_length=args.seq_len,
              padding="max_length").input_ids
    batch = ids.repeat(args.batch_size, 1).to("cuda")
    labels = batch.clone()

    model.train()
    losses, step_times, bad = [], [], []
    for step in range(1, args.steps + 1):
        s = time.time()
        try:
            if args.chunked_ce:
                loss = chunked_ce_loss(model, batch, args.ce_chunk)
            else:
                loss = model(input_ids=batch, labels=labels).loss
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
        except torch.cuda.OutOfMemoryError as e:
            log(f"\n   OOM at step {step}: {str(e)[:150]}")
            log(f"   peak VRAM before OOM: {torch.cuda.max_memory_reserved()/1024**3:.2f} GB")
            return 3
        dt = time.time() - s
        step_times.append(dt)
        v = loss.item()
        losses.append(v)
        if v != v or v in (float("inf"), float("-inf")):
            bad.append(step)
        if step == 1 or step % 5 == 0 or step == args.steps:
            log(f"   step {step:3d}  loss {v:8.4f}  {dt:5.2f}s  "
                f"VRAM {torch.cuda.max_memory_reserved()/1024**3:5.2f} GB peak")

    # ---- verdict ------------------------------------------------------------------
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_res = torch.cuda.max_memory_reserved() / 1024**3
    rss = peak_rss_gb()
    total, used, free = gpu_free_mib()
    budget = total / 1024 if total else 8.0

    log("\n" + "=" * 64)
    log("RESULT")
    log("=" * 64)
    log(f"model class      : {cls_used}")
    log(f"peak VRAM alloc  : {peak_alloc:.2f} GB")
    log(f"peak VRAM resvd  : {peak_res:.2f} GB   <- the number that must fit")
    log(f"GPU total        : {budget:.2f} GB")
    log(f"headroom         : {budget - peak_res:.2f} GB")
    log(f"peak system RAM  : {rss:.2f} GB (this process; PLE offload + paged optimizer live here)")
    log(f"loss             : {losses[0]:.4f} -> {losses[-1]:.4f}"
        f"   (min {min(losses):.4f}, max {max(losses):.4f})")
    log(f"non-finite steps : {bad if bad else 'none'}")
    med = sorted(step_times)[len(step_times)//2]
    log(f"median step      : {med:.2f}s")
    log(f"elapsed          : {time.time()-t0:.0f}s")

    ok = not bad and peak_res < budget * 0.95
    log("")
    if ok:
        log(f"VERDICT: GO — fits with {budget - peak_res:.2f} GB headroom, loss finite.")
        if med > 8:
            log("  NOTE: steps are slow. Under WSL2 the driver spills VRAM into system RAM")
            log("        instead of OOM-ing, which looks like this. Watch for it during training.")
        est = med * 3000 / 60  # ~3000 steps for ~2-3k pairs over 3 epochs
        log(f"  At this rate a ~3000-step run takes roughly {est:.0f} min.")
    elif bad:
        log("VERDICT: NO-GO — non-finite loss (fp16 overflow on Turing).")
        log("  Try: fp32 layer norms, or lower LR, or fall back to google/gemma-3-4b-it.")
    else:
        log(f"VERDICT: TIGHT — peak {peak_res:.2f} GB of {budget:.2f} GB.")
        log("  Ladder: --seq-len 512, then --lora-r 8, then --attn-only,")
        log("          then fall back to google/gemma-3-4b-it.")
    log("=" * 64)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
