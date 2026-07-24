#!/usr/bin/env python3
"""Background job manager — start / stop / live log, off the Gradio thread.

Long work (transcription, training) must never block the UI, and must be stoppable
mid-batch without losing finished work. Jobs run as subprocesses so a crash or a kill
takes the job down, not the app.
"""
import subprocess
import threading
import time
from collections import deque


class Job:
    def __init__(self, name):
        self.name = name
        self.lines = deque(maxlen=2000)
        self.proc = None
        self.thread = None
        self.running = False
        self.stopped = False
        self.done = 0
        self.total = 0
        self.started = None

    def log(self, msg):
        self.lines.append(msg.rstrip())

    def text(self, n=60):
        return "\n".join(list(self.lines)[-n:])

    def status(self):
        if self.running:
            el = int(time.time() - (self.started or time.time()))
            prog = f"{self.done}/{self.total}" if self.total else str(self.done)
            return f"RUNNING · {prog} · {el//60}m{el%60:02d}s"
        if self.stopped:
            return f"STOPPED · {self.done}/{self.total} done"
        if self.started:
            return f"FINISHED · {self.done}/{self.total}"
        return "idle"


def gpu_free_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().split("\n")[0])
    except Exception:
        return None


class JobManager:
    """One named job of each kind at a time. Starting a second is refused, not queued.

    GPU jobs are additionally mutually exclusive across names: Demucs, IndicConformer and
    training all want the same 8 GB card, and training needs ~7.3 GB of it. Letting a
    transcription batch overlap a training run would OOM one or both.
    """

    GPU_JOBS = {"transcribe", "train", "generate"}

    def __init__(self):
        self.jobs = {}
        self._lock = threading.Lock()

    def get(self, name):
        with self._lock:
            return self.jobs.setdefault(name, Job(name))

    def is_running(self, name):
        return self.get(name).running

    def gpu_busy(self, exclude=None):
        """Name of the GPU job currently running, if any."""
        for n in self.GPU_JOBS:
            if n != exclude and self.jobs.get(n) and self.jobs[n].running:
                return n
        return None

    def start(self, name, target, needs_gpu=None):
        """target(job) runs in a worker thread and should poll job.stopped."""
        job = self.get(name)
        if job.running:
            return False, f"{name} is already running"
        if needs_gpu is None:
            needs_gpu = name in self.GPU_JOBS
        if needs_gpu:
            busy = self.gpu_busy(exclude=name)
            if busy:
                return False, (f"GPU is busy with '{busy}' — stop it first. "
                               f"Transcription and training cannot share this 8 GB card.")
        job.lines.clear()
        job.running, job.stopped = True, False
        job.done, job.total, job.started = 0, 0, time.time()

        def wrap():
            try:
                target(job)
            except Exception as e:
                job.log(f"ERROR: {type(e).__name__}: {e}")
            finally:
                job.running = False

        job.thread = threading.Thread(target=wrap, daemon=True)
        job.thread.start()
        return True, f"{name} started"

    def stop(self, name):
        job = self.get(name)
        if not job.running:
            return f"{name} is not running"
        job.stopped = True
        job.log("--- stop requested; finishing current item ---")
        if job.proc and job.proc.poll() is None:
            job.proc.terminate()
        return f"{name} stopping"


def run_stream(job, cmd, cwd=None, env=None):
    """Run a subprocess, streaming stdout into the job log. Returns exit code."""
    job.log(f"$ {' '.join(str(c) for c in cmd)}")
    job.proc = subprocess.Popen(
        cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, errors="replace")
    for line in job.proc.stdout:
        job.log(line)
        if job.stopped:
            job.proc.terminate()
            break
    job.proc.wait()
    return job.proc.returncode


MANAGER = JobManager()
