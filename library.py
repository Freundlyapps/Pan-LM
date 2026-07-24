#!/usr/bin/env python3
"""Library scan and the transcription job that feeds off it."""
import os
from pathlib import Path

import jobs
import state

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".wma", ".aac"}


def scan(con, cfg, log=print):
    """Walk the configured roots and register audio files. Never clobbers existing rows."""
    found = new = 0
    excl = set(cfg.get("exclude_dirs", []))
    for root in cfg["source_roots"]:
        root = Path(root)
        if not root.exists():
            log(f"missing root: {root}")
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in excl and not d.startswith(".")]
            rel = Path(dirpath).relative_to(root)
            folder = rel.parts[0] if rel.parts else root.name
            for fn in filenames:
                if Path(fn).suffix.lower() not in AUDIO_EXTS:
                    continue
                p = Path(dirpath) / fn
                found += 1
                before = con.total_changes
                state.upsert_item(con, p, Path(fn).stem, folder)
                if con.total_changes > before:
                    new += 1
    log(f"scan complete: {found} audio files, {new} new")
    return found, new


def strip_commentary(text):
    """Drop the model's own prose from a reconstruction.

    The Transcribe repo's clean.py asks Claude to reconstruct lyrics, and Claude sometimes
    prefixes an explanation ("Looking at both transcripts, I'll reconstruct...") before the
    Gurmukhi. Measured on the very first test track: 52% Latin, which the quality gate
    rejects outright.

    Lyrics are ~100% Gurmukhi, so keeping only predominantly-Gurmukhi lines removes the
    commentary without touching the content. Applied here rather than by editing the
    Transcribe repo, which is a working system we do not want to disturb.
    """
    import tagfmt
    keep, seen_content = [], False
    for line in text.splitlines():
        s = line.strip()
        if not s:
            if seen_content:
                keep.append("")
            continue
        if tagfmt.gurmukhi_ratio(s) >= 0.5:
            keep.append(s)
            seen_content = True
    return "\n".join(keep).strip()


def register_uploads(con, cfg, files, folder="(uploaded)"):
    """Copy uploaded audio into data/uploads and register it like any library track."""
    import shutil
    dest_dir = Path(cfg["data_dir"]) / "uploads"
    dest_dir.mkdir(parents=True, exist_ok=True)
    added = []
    for f in files or []:
        src = Path(getattr(f, "name", f))
        if src.suffix.lower() not in AUDIO_EXTS:
            continue
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
        state.upsert_item(con, dest, dest.stem, folder)
        row = con.execute("SELECT id FROM items WHERE path=?", (str(dest),)).fetchone()
        added.append((row["id"], dest.name))
    return added


def pending_choices(con, folder, limit=500):
    """Labels for the track picker: 'id — title' so the id is recoverable on submit."""
    rows = state.pending_for_transcribe(con, folder, limit)
    return [f"{r['id']} — {r['title']}" for r in rows]


def transcribe_batch(con, cfg, folder, limit, mode, claude_model, job, item_ids=None):
    """Run the existing Transcribe pipeline over the next N pending items.

    Deliberately reuses /mnt/e/Transcribe/pipeline.sh as a subprocess rather than importing
    it: that repo keeps torch in a separate venv and its launchers set LD_LIBRARY_PATH.
    Shelling out is the contract it already documents.
    """
    repo = Path(cfg["transcribe_repo"])
    pipeline = repo / "pipeline.sh"
    if not pipeline.exists():
        job.log(f"ERROR: {pipeline} not found — check Settings")
        return

    if item_ids:
        qs = ",".join("?" * len(item_ids))
        items = con.execute(
            f"SELECT * FROM items WHERE id IN ({qs}) ORDER BY folder, title",
            list(item_ids)).fetchall()
        job.log(f"explicit selection: {len(items)} track(s)")
    else:
        items = state.pending_for_transcribe(con, folder, limit)
    job.total = len(items)
    if not items:
        job.log("nothing pending in that selection")
        return

    out_dir = Path(cfg["data_dir"]) / "corpus"
    out_dir.mkdir(parents=True, exist_ok=True)
    job.log(f"batch: {len(items)} item(s) | mode={mode} | model={claude_model}")

    for i, item in enumerate(items, 1):
        if job.stopped:
            job.log("stopped — finished items are kept")
            break
        audio = Path(item["path"])
        job.log(f"\n[{i}/{len(items)}] {audio.name}")
        if not audio.exists():
            state.set_fields(con, item["id"], state="rejected", note="file missing")
            job.log("   file missing; marked rejected")
            continue

        cmd = [str(pipeline), str(audio), "--mode", mode, "--out", str(out_dir)]
        if claude_model:
            cmd += ["--claude-model", claude_model]
        rc = jobs.run_stream(job, cmd, cwd=str(repo))

        clean = out_dir / f"{audio.stem}.txt"
        raw = out_dir / "raw" / f"{audio.stem}.txt"
        if rc == 0 and clean.exists():
            import quality
            body = strip_commentary(clean.read_text(encoding="utf-8"))
            m = quality.score(body)
            state.set_fields(
                con, item["id"], state="transcribed",
                clean_text=body, final_text=body,
                raw_text=raw.read_text(encoding="utf-8") if raw.exists() else None,
                note=f"{m['verdict']}: {'; '.join(m['reasons'])}" if m["reasons"] else "ok")
            job.done += 1
            icon = {"ok": "OK", "warn": "WARN", "reject": "REJECT"}[m["verdict"]]
            job.log(f"   -> transcribed [{icon}] {m['chars']}ch "
                    f"gurmukhi={m['script_gurmukhi']:.0%}")
            if m["reasons"]:
                job.log(f"      {'; '.join(m['reasons'])}")
        else:
            state.set_fields(con, item["id"], note=f"pipeline exit {rc}")
            job.log(f"   FAILED (exit {rc})")

    job.log(f"\nbatch finished: {job.done}/{job.total} transcribed")
