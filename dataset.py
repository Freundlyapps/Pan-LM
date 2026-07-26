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


# --- Long-story handling ------------------------------------------------------
# A story longer than the training window (cfg["seq_len"]) would be SILENTLY
# truncated at train time (train_job.encode uses truncation=True) — the model
# would only ever see the opening. So the user pastes the whole story as ONE
# reviewed item and we split it HERE into a continuation chain: the first chunk is
# produced from the instruction/theme, each later chunk from a "continue this
# story: <tail>" prompt. Nothing truncates, and the model learns the real
# narrative flow instead of disconnected fragments. Split points are scene-aligned
# (build_story's [Scene N] markers), so chunks never cut mid-paragraph.
TAIL_TOK = 224                        # context a continuation prompt shows of the prior chunk
_SENT = re.compile(r"(?<=[।!?\.])\s+")
_SCENE = re.compile(r"(?m)(?=^\[Scene \d+\])")


def load_tokenizer(cfg, job=None):
    """The real tokenizer makes chunk sizing exact. If it can't load (offline / gated),
    fall back to a char estimate — callers pass the result straight to the helpers below."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(cfg["base_model"])
    except Exception as e:                # noqa: BLE001 — any load failure is non-fatal
        if job:
            job.log(f"tokenizer unavailable ({type(e).__name__}); using char estimate for chunking")
        return None


def tok_len(tok, text):
    """Token count via the real tokenizer, else a conservative char over-estimate —
    over-counting splits sooner, which is the safe direction (never truncate)."""
    if tok is None:
        return int(len(text) / 1.6)
    return len(tok(text, add_special_tokens=False).input_ids)


def _pack(units, tok, budget):
    """Greedily pack text units into chunks, each <= budget tokens (never merge across
    a unit boundary that would overflow)."""
    chunks, cur = [], ""
    for u in units:
        u = u.strip("\n")
        if not u:
            continue
        cand = f"{cur}\n{u}" if cur else u
        if cur and tok_len(tok, cand) > budget:
            chunks.append(cur)
            cur = u
        else:
            cur = cand
    if cur:
        chunks.append(cur)
    return chunks


def split_story(tagged, tok, budget):
    """Split a tagged story into scene-aligned chunks, each <= budget tokens.
    Returns [] when the whole story already fits — the caller then keeps the normal
    single-answer path."""
    if tok_len(tok, tagged) <= budget:
        return []
    segs = _SCENE.split(tagged)
    header = segs[0] if segs and not segs[0].lstrip().startswith("[Scene") else ""
    scenes = segs[1:] if header else segs
    units = []
    for sc in scenes:
        if tok_len(tok, sc) <= budget:
            units.append(sc)
        else:                             # one scene overflows the window — split by sentence
            units.extend(_pack(_SENT.split(sc.strip()), tok, budget))
    chunks = _pack(units, tok, budget)
    if header.strip() and chunks:         # the [Form/Title/Theme] header rides with chunk 0
        chunks[0] = f"{header.rstrip()}\n{chunks[0]}"
    return chunks


def _tail(text, tok, cap):
    """Trailing scene(s) of a chunk, <= cap tokens — the context a continuation prompt shows."""
    keep = []
    for line in reversed(text.strip().splitlines()):
        if keep and tok_len(tok, "\n".join([line, *keep])) > cap:
            break
        keep.insert(0, line)
    return "\n".join(keep)


def _story_chain(item, meta, kinds_on, chunks, tok):
    """Instruction/theme -> first chunk, then a 'continue' example for each later chunk."""
    title = item["title"] or ""
    head = chunks[0]
    out = []
    if kinds_on.get("instruct", True) and meta and meta.get("instructions"):
        for ins in meta["instructions"][:3]:
            out.append({"type": "instruct",
                        "messages": [{"role": "user", "content": ins},
                                     {"role": "assistant", "content": head}]})
    if kinds_on.get("theme", True):
        theme = (meta or {}).get("theme") or item["theme"] or title
        if theme:
            ask = f"Write a Punjabi story about {theme}"
            if item["artist"]:
                ask += f", in the style of {item['artist']}"
            out.append({"type": "theme",
                        "messages": [{"role": "user", "content": ask},
                                     {"role": "assistant", "content": head}]})
    if kinds_on.get("continue", True):
        for i in range(1, len(chunks)):
            ctx = _tail(chunks[i - 1], tok, TAIL_TOK)
            out.append({"type": "continue",
                        "messages": [{"role": "user",
                                      "content": "Continue this Punjabi story:\n\n" + ctx},
                                     {"role": "assistant", "content": chunks[i]}]})
    return out


def examples_for(item, meta, kinds_on, tok=None, seq_len=1024):
    """Build the chat-format examples for one item."""
    tagged = tag(item, meta)
    kind = item["kind"] or "song"
    title = item["title"] or ""
    out = []

    # A story too long for the window is turned into a continuation chain (above),
    # so it trains whole without truncation. Short stories fall through unchanged.
    if kind == "story":
        chunk_budget = max(256, seq_len - 320)      # leave room for template + tail context
        chunks = split_story(tagged, tok, chunk_budget)
        if chunks:
            return _story_chain(item, meta, kinds_on, chunks, tok)

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

    seq_len = cfg.get("seq_len", 1024)
    tok = load_tokenizer(cfg, job)        # sizes long-story chunks to the real window
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
        ex = examples_for(item, meta, kinds_on, tok=tok, seq_len=seq_len)
        for e in ex:
            e["item_id"] = item["id"]
            e["title"] = item["title"]
        rows.extend(ex)
        job.done += 1
        is_story = (item["kind"] or "song") == "story"
        chained = sum(1 for e in ex if e["type"] == "continue") if is_story else 0
        note = ""
        if not meta:
            note = "  (metadata failed — theme/continue only)"
        elif chained:
            note = f"  (long story → chained into {chained + 1} parts)"
        job.log(f"[{i}/{len(items)}] {item['title']}: {len(ex)} example(s)" + note)

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
