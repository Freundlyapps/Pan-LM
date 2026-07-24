# CLAUDE.md

Guidance for Claude Code (and humans) working in this repo.

## What this is

**Pan-LM** — a local training platform that fine-tunes **Gemma 4 E2B** (QLoRA) to write
**Punjabi songs and stories in Gurmukhi**, in the register of classic Punjabi folk. It turns the
user's own audio into a reviewed text corpus, then trains and tests a personal, non-commercial
style model. Everything is driven from a Gradio UI where each stage is manually triggered and
**nothing reaches training without a human approve gate**.

Pipeline: **audio → (Demucs → IndicConformer dual-view → Claude reconstruction) → human edit/approve
→ tagged dataset → QLoRA → base-vs-tuned test.** The audio→text half reuses the companion
`/mnt/e/Transcribe` repo via subprocess.

## Copyright — read first

`data/` contains reconstructed lyrics of **copyrighted commercial recordings**. It is `.gitignore`d
and must **never** be committed or published — this repo is **code only**. The reconstruction step
frames the task honestly as cleaning up the user's own recording; **do not add jailbreaks** to defeat
the model's copyright refusals (the honest framing is what's appropriate). Refused tracks are flagged
`REFUSED-copyright` and left for the user to hand-edit or drop.

## Run it

```bash
./run.sh                                        # Gradio UI → http://localhost:7861
./overnight.sh --folder "NAME" --batch 10       # unattended batched transcription
tail -f data/overnight.log ; touch data/STOP    # watch / stop cleanly
~/.venvs/punjabi-lm/bin/python preflight.py --seq-len 1024 --lora-r 16 --chunked-ce
```

View the UI from the **Windows** browser (WSL forwards localhost). Read Gurmukhi there, **not in the
terminal** — it mangles conjuncts.

## Environment (outside the repo)

- venv `~/.venvs/punjabi-lm` — torch 2.6+cu124, transformers 5.14, peft, trl, bitsandbytes, gradio.
  Created with `virtualenv` (no `python3-venv`); needs system `build-essential` + `python3.10-dev`
  (bitsandbytes JIT-compiles 4-bit kernels through triton).
- Transcription uses the **Transcribe repo's** own `~/.venvs/demucs` venv via subprocess — its
  launchers set `LD_LIBRARY_PATH`, which we strip so torch uses its bundled CUDA libs.
- Gemma 4 weights cache in `~/.cache/huggingface` (~9.6 GB, gated — HF login required).

## Modules

- `app.py` — Gradio UI, 8 tabs (Library, Transcribe, Editor, Text import, Dataset, Train, Test,
  Settings). Tab wiring + thin handlers only.
- `config.py` / `config.yaml` — settings; `state.py` — SQLite (`data/state.db`), one row per item,
  the pipeline state machine (`pending→transcribed→edited→approved/rejected`).
- `jobs.py` — background job manager; GPU jobs (`transcribe`/`train`/`generate`) are mutually
  exclusive (the 8 GB card can't share).
- `library.py` — audio scan + per-track transcription; `fast_transcribe.py` + `conformer_batch.py` —
  batched path that loads Demucs/IndicConformer **once per batch** (~16% faster).
- `quality.py` — the corpus guard (script mix, ASR-loop vs refrain, length) → ok/warn/reject.
- `tagfmt.py` — Suno-standard structural tags for the 3 content types.
- `dataset.py` — reverse-instruction dataset builder; `train_job.py` — QLoRA; `infer.py` — base-vs-
  tuned generation; `translit.py` — roman→Gurmukhi typing helper (Google Input Tools).
- `overnight.py`/`preflight.py` — standalone runners.

## Hardware constraints (measured — see STATUS.md, do not re-derive)

RTX 2070 **8 GB, Turing sm_75**; 16 GB host, WSL2 given 10 GB via `.wslconfig`.

- **fp16 only** — no bf16 silicon (`is_bf16_supported()` lies; it counts emulation). `eager` attention
  (no flash-attn-2).
- **Final config: seq 1024 · LoRA r16 all-linear · chunked CE (128) · NF4 4-bit · towers on CPU →
  7.29 GB, ~2.9 s/step.**
- Four traps that WILL break training if reintroduced: `expandable_segments` (breaks CUDA on WSL2);
  `prepare_model_for_kbit_training` (fp32 upcast → ~11 GB); plain-suffix LoRA targets (hit the
  vision/audio towers' `Gemma4ClippableLinear`, unsupportable — scope with a `language_model` regex);
  liger-kernel (every kernel fails `ptxas` on sm_75 — hence the pure-PyTorch chunked CE).
- **CPU offload of transformer layers is dead** (runs on CPU, 5% GPU util). Only the unused
  vision/audio towers go to CPU.
- **GGUF export is broken** for E2B (llama.cpp #22243 — PLE skipped, silent quality loss). Serve the
  merged model via transformers.

## Content types — never mix (see tagfmt.py)

**song** (geet/kali/tappa, sung verse) · **qissa kav** (Heer, Sassi Punnu — narrative *verse*, metrical,
NOT prose) · **story** (prose). Tags are Suno-standard English (`[Verse]`, `[Chorus]`, `[Bridge]`,
`[Intro]`, `[Outro]`) — Gemma has seen them and lyrics stay Suno-pasteable. mukhda→`[Chorus]`,
antara→`[Verse N]`.

## Conventions

Match existing style: stdlib `argparse`/small focused functions, `log=print` callbacks so the UI can
capture output, venvs outside the repo, torch kept out of the whisper venv. The **human review gate is
the point** — never auto-approve, never widen the quality gate to pass more without measuring.
