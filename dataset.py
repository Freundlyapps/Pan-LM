#!/usr/bin/env python3
"""Dataset builder — approved items become training examples.

Raw lyrics alone only teach continuation. To make a model that takes an instruction, we
use REVERSE-INSTRUCTION: Claude reads a finished song and writes the prompt that would
have produced it. The outputs stay authentic Punjabi (they are the real transcripts);
only the prompts are synthetic. That is the whole point — a model trained on translated
or model-written Punjabi would sound like translated Punjabi.

Three example shapes per item:
    instruct     instruction  -> full tagged text
    theme        title/theme  -> full tagged text
    continue     opening      -> rest of the song
"""
import json
import random
import re
import subprocess
import time
from pathlib import Path

import quality
import state
import tagfmt

META_PROMPT = """You are given a Punjabi {kind} in Gurmukhi. Produce JSON only, no prose.

{{"theme": "<2-5 words, Gurmukhi>",
  "mood": "<one word, English>",
  "form": "<kali|geet|tappa|qissa|story>",
  "summary": "<one sentence, English>",
  "instructions": ["<3 different natural user requests, in English, that would produce
                    this exact piece. Vary specificity: one vague, one with theme, one
                    with form+style. Do not quote its lines.>"]}}"""


def claude_json(text, kind, model, timeout=180):
    """Ask Claude for metadata. Returns dict or None — callers must tolerate None."""
    try:
        p = subprocess.run(["claude", "-p", META_PROMPT.format(kind=kind), "--model", model],
                           input=text, capture_output=True, text=True, timeout=timeout)
        if p.returncode != 0:
            return None
        m = re.search(r"\{.*\}", p.stdout, re.S)
        return json.loads(m.group(0)) if m else None
    except Exception:
        return None


def tag(item, meta):
    """Apply the structural tags appropriate to the item's kind."""
    text = item["final_text"] or item["clean_text"] or ""
    kind = item["kind"] or "song"
    title = item["title"] or ""
    theme = (meta or {}).get("theme") or item["theme"] or ""
    form = (meta or {}).get("form") or item["form"] or ("kali" if kind == "song" else "")
    if kind == "story":
        return tagfmt.build_story(text, title=title, theme=theme)
    if kind == "qissa":
        return tagfmt.build_qissa([s.splitlines() for s in tagfmt.split_stanzas(text)],
                                  title=title, characters=item["artist"] or "")
    return tagfmt.build_song(text, title=title, form=form or "kali", theme=theme,
                             artist=item["artist"] or "")


def examples_for(item, meta, kinds_on):
    """Build the chat-format examples for one item."""
    tagged = tag(item, meta)
    kind = item["kind"] or "song"
    title = item["title"] or ""
    out = []

    if kinds_on.get("instruct", True) and meta and meta.get("instructions"):
        for ins in meta["instructions"][:3]:
            out.append({"type": "instruct",
                        "messages": [{"role": "user", "content": ins},
                                     {"role": "assistant", "content": tagged}]})

    if kinds_on.get("theme", True):
        theme = (meta or {}).get("theme") or item["theme"] or title
        if theme:
            artist = item["artist"] or ""
            ask = f"Write a Punjabi {kind} about {theme}"
            if artist:
                ask += f", in the style of {artist}"
            out.append({"type": "theme",
                        "messages": [{"role": "user", "content": ask},
                                     {"role": "assistant", "content": tagged}]})

    if kinds_on.get("continue", True):
        lines = tagged.splitlines()
        cut = next((i for i, l in enumerate(lines) if l.startswith("[Verse 2]")), 0)
        if cut > 2:
            out.append({"type": "continue",
                        "messages": [{"role": "user",
                                      "content": "Continue this Punjabi song:\n\n"
                                                 + "\n".join(lines[:cut])},
                                     {"role": "assistant",
                                      "content": "\n".join(lines[cut:])}]})
    return out


def build(con, cfg, version, model, kinds_on, eval_frac, job, seed=0):
    """Walk approved items, generate metadata, write train/eval JSONL."""
    items = [r for r in state.query(con, "approved")]
    job.total = len(items)
    if not items:
        job.log("no approved items — approve some in the Editor first")
        return None

    out_dir = Path(cfg["data_dir"]) / "datasets" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    job.log(f"building '{version}' from {len(items)} approved item(s), model={model}")

    rows, skipped = [], 0
    for i, item in enumerate(items, 1):
        if job.stopped:
            job.log("stopped")
            break
        text = item["final_text"] or item["clean_text"] or ""
        m = quality.score(text, item["kind"] or "song")
        if m["verdict"] == "reject":
            skipped += 1
            job.log(f"[{i}/{len(items)}] SKIP {item['title']}: {'; '.join(m['reasons'])}")
            continue
        meta = claude_json(text, item["kind"] or "song", model)
        ex = examples_for(item, meta, kinds_on)
        for e in ex:
            e["item_id"] = item["id"]
            e["title"] = item["title"]
        rows.extend(ex)
        job.done += 1
        job.log(f"[{i}/{len(items)}] {item['title']}: {len(ex)} example(s)"
                + ("" if meta else "  (metadata failed — theme/continue only)"))

    if not rows:
        job.log("no examples produced")
        return None

    # split by ITEM, never by example: three examples from the same song sharing a
    # train/eval boundary would leak the answer into evaluation
    ids = sorted({r["item_id"] for r in rows})
    random.Random(seed).shuffle(ids)
    n_eval = max(1, int(len(ids) * eval_frac)) if eval_frac > 0 else 0
    eval_ids = set(ids[:n_eval])

    train = [r for r in rows if r["item_id"] not in eval_ids]
    ev = [r for r in rows if r["item_id"] in eval_ids]
    for name, data in (("train", train), ("eval", ev)):
        p = out_dir / f"{name}.jsonl"
        with p.open("w", encoding="utf-8") as fh:
            for r in data:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        job.log(f"wrote {p.name}: {len(data)} example(s)")

    meta_out = {"version": version, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "items": job.done, "skipped": skipped, "examples": len(rows),
                "train": len(train), "eval": len(ev), "eval_items": len(eval_ids),
                "model": model, "types": kinds_on}
    (out_dir / "meta.json").write_text(json.dumps(meta_out, indent=2, ensure_ascii=False))
    job.log(f"done: {len(rows)} examples from {job.done} items ({skipped} skipped)")
    return meta_out


def list_versions(cfg):
    d = Path(cfg["data_dir"]) / "datasets"
    if not d.exists():
        return []
    return sorted([p.name for p in d.iterdir() if (p / "train.jsonl").exists()],
                  reverse=True)


def stats(cfg, version):
    d = Path(cfg["data_dir"]) / "datasets" / version
    mp = d / "meta.json"
    if not mp.exists():
        return "no such dataset"
    m = json.loads(mp.read_text())
    lines = [f"**{version}** — {m['created']}",
             f"- {m['items']} items → {m['examples']} examples "
             f"({m['train']} train / {m['eval']} eval)",
             f"- {m['skipped']} skipped by the quality gate",
             f"- metadata model: {m['model']}"]
    return "\n".join(lines)


def preview(cfg, version, n=3):
    p = Path(cfg["data_dir"]) / "datasets" / version / "train.jsonl"
    if not p.exists():
        return "(no dataset)"
    out = []
    with p.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= n:
                break
            r = json.loads(line)
            out.append(f"### [{r['type']}] {r['title']}\n"
                       f"**USER:** {r['messages'][0]['content'][:300]}\n\n"
                       f"**ASSISTANT:**\n{r['messages'][1]['content'][:700]}")
    return "\n\n---\n\n".join(out)
