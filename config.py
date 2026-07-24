#!/usr/bin/env python3
"""Configuration — everything the UI needs, nothing hardcoded in the app."""
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
CONFIG_PATH = HERE / "config.yaml"

DEFAULTS = {
    "source_roots": ["/mnt/e"],
    "transcribe_repo": "/mnt/e/Transcribe",
    "venv": str(Path.home() / ".venvs" / "punjabi-lm"),
    "data_dir": str(HERE / "data"),
    "port": 7861,                      # 7860 belongs to the Transcribe UI
    "batch_size": 10,
    # Opus, not Haiku. A/B tested on Manak's "Chhetee Kar Sarvan Bacha" (2026-07-24):
    #   haiku  285s  8 lines  — dropped ~64% of the song, ignored the "no English" rule,
    #                           left ਦਸ ਰੱਥ ("ten chariot") for ਦਸ਼ਰਥ and ਦੰਤ ("tooth") for ਦੈਂਤ
    #   sonnet 140s 21 lines  — recovered the narrative and proper nouns
    #   opus    25s 22 lines  — same, plus better anti-drift (ਗੜਵਾ, closest to the raw sound)
    # Reconstruction quality is permanent: a bad transcript corrupts the corpus forever.
    # One call per track over ~500 tracks is a one-time cost worth paying.
    "claude_model": "claude-opus-4-8",
    "base_model": "google/gemma-4-E2B-it",
    # measured on an RTX 2070 8GB — see STATUS.md before changing these
    "seq_len": 1024,
    "lora_r": 16,
    "ce_chunk": 128,
    "exclude_dirs": ["$RECYCLE.BIN", "System Volume Information", "projects",
                     "Recovery", "$AV_ASW"],
}


def load():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        loaded = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        cfg.update(loaded)
    return cfg


def save(cfg):
    CONFIG_PATH.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True))
    return cfg


def data_path(cfg, *parts):
    p = Path(cfg["data_dir"]).joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
