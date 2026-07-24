#!/usr/bin/env python3
"""Quality gate — keep garbage out of the training set.

A 500-song corpus is small enough that a handful of bad transcripts measurably degrades
the model: wrong Gurmukhi spellings get memorised, ASR loops teach repetition, and Hindi
or Urdu leakage teaches code-switching in a Gurmukhi-only model.

Every check here is derived from failure modes actually observed in the raw output of
/mnt/e/Transcribe — looped lines, misheard words, truncated captures.

Verdicts:
    ok      safe to train on
    warn    usable, but a human should look
    reject  do not train on this
"""
import re
import unicodedata

GURMUKHI = re.compile(r"[਀-੿]")
DEVANAGARI = re.compile(r"[ऀ-ॿ]")
ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")

MIN_CHARS = 120
MIN_LINES = 4


def _letters(text):
    return [c for c in text if c.isalpha()]


def script_mix(text):
    ls = _letters(text)
    if not ls:
        return {"gurmukhi": 0.0, "devanagari": 0.0, "arabic": 0.0, "latin": 0.0}
    n = len(ls)
    return {
        "gurmukhi": sum(bool(GURMUKHI.match(c)) for c in ls) / n,
        "devanagari": sum(bool(DEVANAGARI.match(c)) for c in ls) / n,
        "arabic": sum(bool(ARABIC.match(c)) for c in ls) / n,
        "latin": sum(bool(LATIN.match(c)) for c in ls) / n,
    }


def loop_score(lines):
    """Distinguish a refrain from an ASR loop.

    Total repeat count does NOT separate them. Measured on Manak's Puran Bhagat: the
    mukhda appears 8 times at lines 3,7,11,15,19,23,27,31 — perfectly regular, zero
    adjacent. That is the form, not a fault; an earlier "reject at 6 repeats" rule threw
    out 18% of the corpus, every one of them a well-formed kali.

    What actually marks a loop is ADJACENCY — the recognizer emitting the same line
    back-to-back. A refrain always has verse content between its occurrences.

    Returns (duplicate fraction, max total repeats, max CONSECUTIVE repeats).
    """
    if not lines:
        return 0.0, 0, 0
    counts = {}
    for l in lines:
        counts[l] = counts.get(l, 0) + 1
    worst = max(counts.values())
    dup = sum(c - 1 for c in counts.values() if c > 1)

    run = best_run = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        best_run = max(best_run, run)
    return dup / len(lines), worst, best_run


def intraline_repeat(lines):
    """Longest run of a phrase repeated inside a single line — a classic ASR artifact."""
    worst = 0
    for l in lines:
        w = l.split()
        # up to 10-word phrases: the conformer commonly re-emits a whole sung line,
        # which a 4-word window silently misses
        for size in range(2, 11):
            i = 0
            while i + 2 * size <= len(w):
                if w[i:i + size] == w[i + size:i + 2 * size]:
                    run = 2
                    j = i + 2 * size
                    while j + size <= len(w) and w[j:j + size] == w[i:i + size]:
                        run += 1
                        j += size
                    worst = max(worst, run)
                    i = j
                else:
                    i += 1
    return worst


def score(text, kind="song"):
    """Return metrics plus a verdict and human-readable reasons."""
    text = unicodedata.normalize("NFC", text or "")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    mix = script_mix(text)
    dup_ratio, worst_rep, consec = loop_score(lines)
    intra = intraline_repeat(lines)

    m = {
        "chars": len(text.strip()),
        "lines": len(lines),
        "unique_lines": len(set(lines)),
        **{f"script_{k}": round(v, 3) for k, v in mix.items()},
        "dup_ratio": round(dup_ratio, 3),
        "max_line_repeat": worst_rep,
        "max_consecutive": consec,
        "intraline_repeat": intra,
    }

    reasons, verdict = [], "ok"

    def bad(msg):
        reasons.append(msg)

    if m["chars"] < MIN_CHARS:
        bad(f"too short ({m['chars']} chars)")
        verdict = "reject"
    if m["lines"] < MIN_LINES:
        bad(f"only {m['lines']} lines")
        verdict = "reject"
    if mix["gurmukhi"] < 0.80:
        bad(f"Gurmukhi only {mix['gurmukhi']:.0%}")
        verdict = "reject"
    elif mix["gurmukhi"] < 0.95:
        bad(f"Gurmukhi {mix['gurmukhi']:.0%} — some foreign script")
        verdict = max(verdict, "warn", key=["ok", "warn", "reject"].index)
    if mix["devanagari"] > 0.02:
        bad(f"Devanagari {mix['devanagari']:.0%} (Hindi leakage)")
        verdict = "reject"
    if mix["arabic"] > 0.02:
        bad(f"Arabic script {mix['arabic']:.0%} (Urdu leakage)")
        verdict = "reject"
    if mix["latin"] > 0.10:
        bad(f"Latin {mix['latin']:.0%}")
        verdict = max(verdict, "warn", key=["ok", "warn", "reject"].index)
    # Adjacent repetition is the loop signal. Folk singing repeats half-lines constantly
    # ("ਮਾਵਾਂ ਹੁੰਦੀਆਂ ਮਤੇਈਆਂ ਬੁਰੀਆਂ ਵੇ ਮਾਵਾਂ ਹੁੰਦੀਆਂ"), so the intraline bar is high.
    if intra >= 4:
        bad(f"phrase repeats {intra}x inside a line (ASR loop)")
        verdict = "reject"
    elif intra == 3:
        bad(f"phrase repeats {intra}x inside a line")
        verdict = max(verdict, "warn", key=["ok", "warn", "reject"].index)
    if consec >= 3:
        bad(f"same line {consec}x back-to-back (ASR loop)")
        verdict = "reject"
    elif consec == 2:
        bad("a line repeats back-to-back")
        verdict = max(verdict, "warn", key=["ok", "warn", "reject"].index)
    # NOT a rule: high total repeat count. A mukhda returning after every antara is the
    # song form — see loop_score(). Only near-total duplication is suspicious.
    if dup_ratio > 0.70:
        bad(f"{dup_ratio:.0%} of lines are duplicates")
        verdict = max(verdict, "warn", key=["ok", "warn", "reject"].index)
    if m["unique_lines"] <= 3 and m["lines"] > 8:
        bad(f"only {m['unique_lines']} unique lines in {m['lines']}")
        verdict = "reject"

    m["verdict"] = verdict
    m["reasons"] = reasons
    return m


def summary(m):
    icon = {"ok": "✅", "warn": "⚠️", "reject": "❌"}[m["verdict"]]
    head = (f"{icon} **{m['verdict'].upper()}** · {m['chars']} chars · {m['lines']} lines "
            f"({m['unique_lines']} unique) · Gurmukhi {m['script_gurmukhi']:.0%}")
    return head + ("\n\n- " + "\n- ".join(m["reasons"]) if m["reasons"] else "")
