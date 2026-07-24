#!/usr/bin/env python3
"""Batched IndicConformer — RUNS INSIDE THE DEMUCS VENV (torch present).

The single-track pipeline reloads IndicConformer for every file and every chunk view
(2 loads/track). This loads it ONCE and streams a whole batch through it, both views per
file. GPU util on the single-track path was 1-6% — almost all wall-clock was model
loading, not inference. This removes that.

Reuses the transcription helpers from Transcribe/conformer_infer.py so behaviour matches
the proven pipeline exactly; only the loop structure changes.

Input  (stdin JSON): {"files": ["a.wav", ...], "lang": "pa", "decoding": "rnnt",
                      "chunks": ["window", "vad"]}
Output (stdout JSON): {"a.wav": {"window": [segs], "vad": [segs]}, ...}
Logs to stderr so stdout stays clean JSON.
"""
import json
import sys

sys.path.insert(0, "/mnt/e/Transcribe")
import torch
import conformer_infer as ci   # load_audio_mono16k, vad_chunks, window_chunks, to_text


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def transcribe(model, wav, duration, lang, decoding, chunk, device):
    if chunk == "window":
        regions = ci.window_chunks(duration)
    else:
        regions = ci.vad_chunks(wav)
    segs = []
    with torch.no_grad():
        for a, b in regions:
            i0, i1 = int(a * ci.TARGET_SR), int(b * ci.TARGET_SR)
            piece = wav[i0:i1]
            if piece.numel() < ci.TARGET_SR * 0.1:
                continue
            try:
                out = model(piece.unsqueeze(0).to(device), lang, decoding)
            except Exception as e:
                log(f"  segment {a:.1f}-{b:.1f}s failed: {e}")
                continue
            text = ci.to_text(out).strip()
            if text:
                segs.append({"start": round(a, 3), "end": round(b, 3), "text": text})
    return segs


def main():
    req = json.load(sys.stdin)
    files = req["files"]
    lang = req.get("lang", "pa")
    decoding = req.get("decoding", "rnnt")
    chunks = req.get("chunks", ["window", "vad"])
    model_id = req.get("model", "ai4bharat/indic-conformer-600m-multilingual")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"[batch] loading {model_id} on {device} (ONCE for {len(files)} file(s)) ...")
    from transformers import AutoModel
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()
    log("[batch] model resident — streaming files")

    result = {}
    for i, f in enumerate(files, 1):
        try:
            wav, duration = ci.load_audio_mono16k(f)
        except Exception as e:
            log(f"[{i}/{len(files)}] {f}: load failed: {e}")
            result[f] = {c: [] for c in chunks}
            continue
        result[f] = {}
        for c in chunks:
            segs = transcribe(model, wav, duration, lang, decoding, c, device)
            result[f][c] = segs
            log(f"[{i}/{len(files)}] {f} [{c}] {len(segs)} segment(s)")
        del wav
        if device == "cuda":
            torch.cuda.empty_cache()

    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
