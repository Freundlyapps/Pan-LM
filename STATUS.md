# Status — Punjabi Song & Story LM

Recovery notes, in case the session is lost. Resume the conversation with:

```bash
cd "/mnt/e/projects/Punjabi LM"
claude --continue
```

Full plan: `~/.claude/plans/tranquil-wishing-cerf.md`

## Goal

Fine-tune **Gemma 4 E2B** (QLoRA) to write Punjabi songs and stories in **Gurmukhi**, in the
register of classic Punjabi folk. Corpus comes from the user's own audio in `/mnt/e`, transcribed
by the existing `/mnt/e/Transcribe` pipeline, with a human edit gate before training.
Personal, non-commercial — weights are not distributed.

Deliverable is a **Gradio platform** (port 7861) where every stage is manually triggered:
Library → Transcribe (batches of 10) → **Editor (approve/reject gate)** → Text import →
Dataset → Train → Test → Settings.

## Environment (done)

- venv: `~/.venvs/punjabi-lm` — created with `virtualenv` (no `python3-venv`, no sudo needed;
  same pattern as `Transcribe/setup.sh`).
- torch 2.6.0+cu124 · transformers 5.14.1 · peft 0.19.1 · trl 1.9.0 · bitsandbytes 0.49.2 ·
  accelerate 1.14.0
- `build-essential` + `python3.10-dev` installed via apt. **Both were required**: bitsandbytes
  0.49 JIT-compiles 4-bit kernels through triton, which needs `gcc` *and* `Python.h`.
  Verified working: `Linear4bit` forward pass succeeds.
- HF token cached; access confirmed (HTTP 200) to `google/gemma-4-E2B-it`, `google/gemma-4-E2B`,
  and the fallback `google/gemma-3-4b-it`.
- Weights downloading to `~/.cache/huggingface/hub/models--google--gemma-4-E2B-it`
  (~10 GB total: E2B is 5.1B *raw* params in bf16; the 2.3B "effective" figure excludes
  per-layer embeddings).

## Hardware facts (measured, not assumed)

- RTX 2070, 8192 MiB VRAM, Turing **sm_75**. ~700 MiB is held by the Windows side under WSL2,
  leaving ~7.3 GiB free.
- `torch.cuda.is_bf16_supported()` returns **True but this is misleading** — torch 2.6 defaults
  to `including_emulation=True`. Turing has no bf16 silicon. **Use fp16.**
- No flash-attn-2 on Turing → `attn_implementation="eager"`.
- Host has 16 GB RAM; WSL2 default gives it 50% = 7.71 GB.

## Pending: apply `.wslconfig`

`/mnt/c/Users/sayad/.wslconfig` was created (10 GB RAM + 8 GB swap, WSLg off) but is
**not yet active**. Apply from Windows PowerShell:

```powershell
wsl --shutdown
```

Then reopen the terminal and `claude --continue`. Only do this when no download or training
job is running — background jobs do not survive.

Why it matters: Gemma 4's Per-Layer Embeddings offload to CPU RAM, and `paged_adamw_8bit` pages
optimizer state there too. Both hit the same pool. 10 GB (not 12) so Windows keeps ~6 GB —
starving the host makes it swap, which drags the GPU driver down with it.

## Content types — THREE, never mixed (`tagfmt.py`)

| Type | Form | Function | Tags |
|---|---|---|---|
| **song** — geet, kali, tappa | verse, sung | `build_song()` | Suno-standard |
| **qissa kav** — Heer, Sassi Punnu, Mirza Sahiban | narrative **verse**, metrical | `build_qissa()` | `[Bait n]` |
| **story** — short / long | **prose** | `build_story()` | `[Scene n]` |

**Qissa kav is NOT a story.** It is metrical, rhymed, sung/recited verse that carries a plot.
Waris Shah's Heer is in *baint* meter — that meter was his signature innovation. Tagging a qissa
as prose would destroy the meter and teach the model to write it as flat narration.

Punjabi folk song structure is **mukhda** (refrain, a.k.a. *sthayi*) + **antara** (verse), with
the mukhda repeating after every antara. That maps exactly onto Suno's `[Chorus]` / `[Verse N]`,
so we use the standard English tags: Gemma has seen them in pretraining, they cost 3-5 tokens,
and the output stays directly pasteable into Suno. *Kali* is a *chhand* (quatrain) form with
strict rules (types: Suchchi, Amba, Roopa), popularised by Kuldeep Manak.

Measured tag costs — all trivial against the ~660 spare tokens at seq 1024:
`[Chorus]` 3 · `[Verse 1]` 5 · `[Intro]` 3 · `[Bridge]` 3 · `[Outro]` 4 · `[Scene 1]` 5 ·
`[Bait]` 4 · `[Style: ...]` 7. Latin-script tags cost 3-5; Gurmukhi equivalents cost 7-9
(`ਅੰਤਰਾ` = 4 tokens alone), so English tags are both cheaper *and* standard.

Tagged song on the real transcript: 243 raw -> 366 tokens (+51%), 658 spare at seq 1024.
Most of that growth is the repeated chorus, which is the actual song form, not tag overhead.

## FINAL CONFIG (measured 2026-07-24)

```
seq 1024 · LoRA r16 · all-linear · chunked CE (chunk 128) · fp16 · NF4 4-bit · towers on CPU
→ 7.29 GB VRAM (+0.71 headroom), 2.99 s/step, ~375 Punjabi words per example
```

```bash
~/.venvs/punjabi-lm/bin/python preflight.py --seq-len 1024 --lora-r 16 --chunked-ce --ce-chunk 128
```

### Chunked cross-entropy is what made this possible

The `[seq x 262144]` logits tensor was the whole bottleneck: +0.65 GB per 128 tokens without it.
`chunked_ce_loss()` in `preflight.py` computes hidden states once, then applies `lm_head` +
cross-entropy in 128-token chunks under `torch.utils.checkpoint` — logits are freed after forward
and recomputed in backward, so only one chunk is ever resident.

| seq | plain CE | chunked CE | step (CCE) |
|---|---|---|---|
| 256 | 7.68 GB | - | - |
| 320 | 7.99 GB | - | - |
| 384 | 8.34 GB over | - | - |
| 512 | 8.98 GB over | **7.03 GB** | 2.12s |
| 1024 | ~11.6 GB | **7.29 GB** | 2.99s |
| 2048 | - | 7.59 GB | 11.69s (O(n^2) attention — too slow) |

Doubling 512 -> 1024 costs only +0.26 GB instead of +2.6 GB. seq 512 is even *faster* with CCE
(2.12s vs 2.42s) — less allocator pressure outweighs the recompute.

### Ruled out by measurement — do not retry

- **CPU offload (`--cpu-offload`)**: accelerate pushes so much to CPU the model executes there.
  GPU sat at 5% utilisation and never reached a training step. PLE is looked up per-layer
  per-token (35 round trips per forward), so it is the worst possible tensor to put over PCIe.
- **liger-kernel**: every kernel fails on Turing — `ptxas ... --gpu-name=sm_75` exits 255. This
  includes `rms_norm`, `geglu`, *and* `fused_linear_cross_entropy`. liger needs sm_80+.
  The pure-PyTorch chunked CE above replaces it with no triton dependency.

**New tight resource is system RAM, not VRAM**: peaks at ~8.7 GB of the 9.72 GB WSL now has.

## PREFLIGHT RESULT: GO (measured 2026-07-24)

Gemma 4 E2B QLoRA **trains on this card**. fp16 is stable on Turing — loss fell smoothly with
zero non-finite steps across every config tried.

| Config | Peak VRAM | Headroom | Step | Loss (8 steps) | |
|---|---|---|---|---|---|
| seq 512, r16, all-linear | 8.98 GB | -0.98 | 2.52s | 9.30 -> 6.29 | spills to RAM |
| seq 384, r8, attn-only | 8.14 GB | -0.14 | 2.16s | 9.20 -> 7.05 | over |
| seq 256, r16, all-linear | 7.68 GB | +0.32 | 2.29s | 8.44 -> **3.64** | tight, learns best |
| **seq 256, r8, attn-only** | **7.52 GB** | **+0.48** | **1.82s** | 8.44 -> 6.12 | **safe** |

Fixed cost is **6.03 GB of base weights** — bitsandbytes does not quantize embeddings, and
the 262K vocab plus per-layer embeddings dominate. That leaves only ~1.5 GB for everything else.
Tokenizer fertility on real Gurmukhi: **2.73 tokens/word** (good; a 256-token window ~= 94 words).

### Four traps found the hard way — do not re-introduce

1. **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` breaks training under WSL2.** Standard
   anti-fragmentation advice on native Linux, but it uses CUDA virtual-memory APIs the WSL
   paravirtualization layer supports poorly. Every allocation failed with `CUDA driver error:
   out of memory` while nvidia-smi still showed 1.3 GB free. Removing it was the difference
   between no training and training. Now set to `garbage_collection_threshold:0.8`.
2. **Do not call `peft.prepare_model_for_kbit_training`.** It upcasts non-quantized params to
   fp32; on Gemma 4's ~2.8B of 16-bit embeddings that is ~11 GB. Enable gradient checkpointing
   and `enable_input_require_grads()` directly instead.
3. **Scope LoRA targets with a regex, not a suffix list.** `["q_proj", ...]` matches by suffix
   across the whole model and hits the vision/audio towers, whose projections are
   `Gemma4ClippableLinear` — a type peft cannot adapt. Use
   `.*language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)`.
4. **Any CPU dispatch of a quantized model needs `llm_int8_enable_fp32_cpu_offload=True`** —
   including just parking the vision/audio towers off-GPU (which saves 0.25 GB and is free,
   since text-only inference never invokes them).

### CPU offload is blocked on system RAM

`--cpu-offload` currently fails: with only 7.7 GB of RAM (and the process already peaking at
6.4-6.6 GB), accelerate has nowhere to put the weights and spills them to **disk**
(`placement: {'disk': 5, 'cpu': 1}`, 0.00 GB on GPU). Raising WSL to 10 GB should unlock it,
which would in turn allow seq 512 + r16 all-linear — the configuration that learns best.

## Next step

Apply `.wslconfig` (see above), then re-test:

```bash
~/.venvs/punjabi-lm/bin/python preflight.py --steps 8 --seq-len 512 --lora-r 16 \
    --cpu-offload --gpu-mem 5.5GiB --cpu-mem 7GiB
```

Watch **step time**: if offload pushes it past ~10s, it is a false economy — take
`--seq-len 256 --lora-r 16` (7.68 GB) instead, which trains well and needs no offload.

Fallback that is known-good today, no restart required:
`--seq-len 256 --lora-r 8 --attn-only`

## Original preflight command reference

```bash
cd "/mnt/e/projects/Punjabi LM"
~/.venvs/punjabi-lm/bin/python preflight.py --steps 20 --seq-len 1024 --lora-r 16
# if it OOMs or reports TIGHT:
~/.venvs/punjabi-lm/bin/python preflight.py --steps 20 --seq-len 1024 --lora-r 16 --cpu-offload
```

Reports peak VRAM, peak system RAM, loss stability, Gurmukhi tokens/word, and median step time.

If it is tight, the ladder is: `--seq-len 512` → `--lora-r 8` → `--attn-only` → `--cpu-offload`
→ fall back to `google/gemma-3-4b-it`.

**Watch step time.** Under WSL2 the driver spills VRAM into system RAM instead of OOM-ing, so
training does not crash — it silently runs 10-50x slower. A sudden collapse in step rate means
spilling; abort and reduce rather than waiting it out.

## Corpus (not yet transcribed)

Core Punjabi folk, ~503 tracks: `Mohd Sadiq` (190), `Didar Sandhu` (150), `Manak Kuldeep` (137),
`Khiadan De Din Chaar` (26). Excluded as Hindi: `Kishore Da`, `rafi`. Undecided until
transcription quality is judged: `nusrat` (91, qawwali), `sukhwinder` (125, Bollywood-Punjabi),
`Music` (269, mostly Urdu ghazal + Osho Hindi — cherry-pick only).

~500 tracks is only ~300-500K tokens: enough to teach **style, form and register**, not enough
to teach Gemma more Punjabi. Stories come from text pasted into the UI.

## Known blocker downstream

**GGUF export is broken for E2B/E4B** — llama.cpp issue #22243. PLE weights load but the
per-layer signal never enters the forward graph, silently degrading quality. Serve the merged
model via transformers instead; treat any GGUF as a draft build and A/B it.
