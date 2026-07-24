#!/usr/bin/env python3
"""Unattended batch transcription — safe to leave running overnight.

Runs the library in small batches with a cooldown between each, so an 8-hour session
doesn't cook the GPU. Every batch commits to SQLite as it goes, so nothing is lost if
the machine is interrupted: rerunning simply picks up whatever is still `pending`.

    ./overnight.sh --folder "Manak Kuldeep" --batch 10 --cooldown 180

Watch it:  tail -f data/overnight.log
Stop it:   touch data/STOP        (finishes the current track, then exits cleanly)
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import config
import fast_transcribe
import jobs
import library
import state

HERE = Path(__file__).resolve().parent
STOP_FILE = None


def gpu_stats():
    """(temperature C, free MiB) or (None, None)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        t, f = out.stdout.strip().split("\n")[0].split(",")
        return int(t), int(f)
    except Exception:
        return None, None


def cool_down(seconds, temp_limit, log):
    """Sleep between batches, and keep sleeping while the GPU is above temp_limit."""
    log(f"cooldown {seconds}s")
    for _ in range(seconds):
        if STOP_FILE.exists():
            return
        time.sleep(1)
    waited = 0
    while waited < 600:                      # never wait more than 10 extra minutes
        t, _ = gpu_stats()
        if t is None or t <= temp_limit:
            if t is not None:
                log(f"GPU {t}C — resuming")
            return
        log(f"GPU {t}C > {temp_limit}C — waiting")
        time.sleep(30)
        waited += 30


def main():
    global STOP_FILE
    ap = argparse.ArgumentParser(description="Unattended batch transcription")
    ap.add_argument("--folder", default="all")
    ap.add_argument("--batch", type=int, default=10, help="tracks per batch")
    ap.add_argument("--cooldown", type=int, default=180, help="seconds between batches")
    ap.add_argument("--max", type=int, default=0, help="stop after N tracks (0 = all)")
    ap.add_argument("--temp-limit", type=int, default=78, help="pause above this GPU temp")
    ap.add_argument("--mode", default="song", choices=["song", "story"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--slow", action="store_true",
                    help="Use the per-track pipeline (reloads models each track). "
                         "Default is the fast batched path: models load once per batch.")
    args = ap.parse_args()

    cfg = config.load()
    con = state.connect(config.data_path(cfg, "state.db"))
    model = args.model or cfg["claude_model"]
    STOP_FILE = Path(cfg["data_dir"]) / "STOP"
    STOP_FILE.unlink(missing_ok=True)
    logf = Path(cfg["data_dir"]) / "overnight.log"
    logf.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
        print(line, flush=True)
        with logf.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    remaining = len(state.pending_for_transcribe(con, args.folder, 100000))
    target = min(remaining, args.max) if args.max else remaining
    log("=" * 64)
    log(f"START folder={args.folder!r} batch={args.batch} cooldown={args.cooldown}s "
        f"model={model}")
    log(f"{remaining} pending; will do {target}")
    t, f = gpu_stats()
    log(f"GPU {t}C, {f} MiB free")
    eta = timedelta(seconds=int(target * 169 + (target / args.batch) * args.cooldown))
    log(f"estimated {eta} (at ~2m49s/track + cooldowns)")

    done = errors = batch_no = 0
    t0 = time.time()
    while done < target:
        if STOP_FILE.exists():
            log("STOP file present — exiting cleanly")
            break
        batch_no += 1
        n = min(args.batch, target - done)
        log(f"--- batch {batch_no}: {n} track(s) | {done}/{target} done ---")

        job = jobs.Job(f"overnight-{batch_no}")
        try:
            if args.slow or args.mode == "story":
                library.transcribe_batch(con, cfg, args.folder, n, args.mode, model, job)
            else:
                fast_transcribe.fast_batch(con, cfg, args.folder, n, model, job)
        except Exception as e:
            log(f"BATCH ERROR {type(e).__name__}: {e}")
            errors += 1
            if errors >= 3:
                log("3 batch errors — stopping to avoid burning the night on a broken setup")
                break

        # surface the per-track verdicts the batch recorded
        for line in job.text(400).splitlines():
            if "->" in line or "FAILED" in line or "[" in line:
                log("  " + line.strip())

        if job.done == 0:
            log("batch produced nothing — stopping rather than looping uselessly")
            break
        done += job.done
        el = time.time() - t0
        rate = el / max(done, 1)
        log(f"progress {done}/{target} · {el/60:.0f}m elapsed · "
            f"~{(target-done)*rate/60:.0f}m left")

        if done < target:
            cool_down(args.cooldown, args.temp_limit, log)

    counts = state.counts(con)
    log(f"FINISHED {done} track(s) in {(time.time()-t0)/60:.0f}m · states={counts}")
    log("review them in the UI: http://localhost:%s" % cfg["port"])


if __name__ == "__main__":
    sys.exit(main())
