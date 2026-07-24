#!/usr/bin/env python3
"""Fast batched transcription — models load ONCE per batch, not per track.

Single-track pipeline.sh does, per track: 1 Demucs load + 2 IndicConformer loads. Over a
batch that is dozens of cold starts on a GPU that then sits at 1-6% util. This does:

    Demucs  (one subprocess, all files)         -> vocals wavs   [model loaded once]
    conformer_batch (one subprocess, all files) -> raw segments  [model loaded once]
    Opus reconstruct (per file, API, no GPU)     -> clean text

Same models, same dual-view (window+vad), same reconstruction prompt as the proven
pipeline — only the load structure changes. Drop-in alternative to library.transcribe_batch.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import library
import quality
import state


def _demucs_batch(demucs_py, files, model_name, device, outdir, job):
    """One Demucs invocation over every file — loads the model once."""
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)      # torch uses its own bundled CUDA libs
    cmd = [demucs_py, "-m", "demucs", "-n", model_name, "--two-stems", "vocals",
           "-d", device, "-o", str(outdir)] + [str(f) for f in files]
    job.log(f"[demucs] separating {len(files)} file(s) in one pass ...")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    job.proc = proc
    for line in proc.stdout:
        if job.stopped:
            proc.terminate()
            break
        if "%" in line or "error" in line.lower():
            job.log("  " + line.strip()[:100])
    proc.wait()
    # Demucs writes <outdir>/<model>/<stem>/vocals.wav
    out = {}
    for f in files:
        v = Path(outdir) / model_name / Path(f).stem / "vocals.wav"
        out[str(f)] = str(v) if v.exists() else None
    return out


def _conformer_batch(demucs_py, vocals, lang, job):
    """One conformer_batch subprocess over every vocals file — loads the model once."""
    files = [v for v in vocals.values() if v]
    if not files:
        return {}
    req = {"files": files, "lang": lang, "decoding": "rnnt",
           "chunks": ["window", "vad"]}
    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    script = str(Path(__file__).resolve().parent / "conformer_batch.py")
    job.log(f"[conformer] transcribing {len(files)} file(s), dual-view, model loaded once ...")
    proc = subprocess.Popen([demucs_py, script], env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    job.proc = proc
    out, err = proc.communicate(json.dumps(req))
    for line in err.splitlines():
        if "[batch]" in line or "segment(s)" in line:
            job.log("  " + line.strip())
    if proc.returncode != 0:
        job.log(f"[conformer] FAILED rc={proc.returncode}")
        return {}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except Exception as e:
        job.log(f"[conformer] bad output: {e}")
        return {}


def fast_batch(con, cfg, folder, limit, claude_model, job, item_ids=None):
    """Batched replacement for library.transcribe_batch. Same DB contract."""
    sys.path.insert(0, cfg["transcribe_repo"])
    import clean  # build_prompt, run_claude, make_ctx — the proven reconstruction

    demucs_py = str(Path(cfg["venv"].replace("punjabi-lm", "demucs")) / "bin" / "python")
    if not Path(demucs_py).exists():
        demucs_py = str(Path.home() / ".venvs" / "demucs" / "bin" / "python")
    if not Path(demucs_py).exists():
        job.log(f"ERROR: demucs venv python not found at {demucs_py}")
        return

    if item_ids:
        qs = ",".join("?" * len(item_ids))
        items = con.execute(f"SELECT * FROM items WHERE id IN ({qs}) ORDER BY folder,title",
                            list(item_ids)).fetchall()
    else:
        items = state.pending_for_transcribe(con, folder, limit)
    job.total = len(items)
    if not items:
        job.log("nothing pending")
        return

    files = [Path(it["path"]) for it in items if Path(it["path"]).exists()]
    job.log(f"fast batch: {len(files)} track(s) — models load ONCE for the whole batch")

    with tempfile.TemporaryDirectory(prefix="fastbatch_") as tmp:
        # 1) Demucs, all files, one load
        vocals = _demucs_batch(demucs_py, files, "htdemucs", "cuda", tmp, job)
        if job.stopped:
            job.log("stopped after demucs")
            return
        # 2) IndicConformer, all files, one load, dual-view
        segs_by_file = _conformer_batch(demucs_py, vocals, "pa", job)

        out_dir = Path(cfg["data_dir"]) / "corpus"
        (out_dir / "raw").mkdir(parents=True, exist_ok=True)

        # 3) reconstruct per file (API, no GPU)
        for it in items:
            if job.stopped:
                job.log("stopped before reconstruction of remaining tracks")
                break
            audio = Path(it["path"])
            vpath = vocals.get(str(audio))
            fseg = segs_by_file.get(vpath, {}) if vpath else {}
            blocks = []
            for c in ("window", "vad"):
                lines = "\n".join(s["text"].strip() for s in fseg.get(c, [])
                                  if s["text"].strip())
                blocks.append(f"--- Transcript from conformer-{c} ---\n{lines}")
            combined = "\n\n".join(blocks)
            (out_dir / "raw" / f"{audio.stem}.txt").write_text(combined + "\n",
                                                               encoding="utf-8")
            if not combined.replace("-", "").strip():
                state.set_fields(con, it["id"], state="transcribed", note="empty ASR")
                job.log(f"  {audio.name}: empty ASR")
                continue
            ctx = clean.make_ctx(audio)
            final_raw = clean.run_claude(clean.build_prompt("song", ctx, 2),
                                         combined, claude_model)
            body = library.strip_commentary(final_raw)
            m = quality.score(body)
            state.set_fields(con, it["id"], state="transcribed", clean_text=body,
                             final_text=body, raw_text=combined + "\n",
                             note=f"{m['verdict']}: {'; '.join(m['reasons'])}"
                                  if m["reasons"] else "ok")
            job.done += 1
            job.log(f"  {audio.name} -> [{m['verdict']}] {m['chars']}ch "
                    f"gurmukhi={m['script_gurmukhi']:.0%}")

    job.log(f"fast batch done: {job.done}/{job.total}")
