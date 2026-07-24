#!/usr/bin/env python3
"""Roman -> Gurmukhi phonetic transliteration, for editing without a Punjabi keyboard.

Uses Google Input Tools (the same engine as easypunjabityping.com) — type "muklawa" and
get ਮੁਕਲਾਵਾ, with alternatives. Falls back to a rough local map if offline, so the Editor
never hard-breaks without internet.
"""
import json
import urllib.parse
import urllib.request

_CACHE = {}

# Rough offline fallback only. The API is far better; this just keeps typing usable
# without internet. Order matters: multi-char keys first.
_FALLBACK = [
    ("aa", "ਾ"), ("ee", "ੀ"), ("oo", "ੂ"), ("ai", "ੈ"), ("au", "ੌ"),
    ("kh", "ਖ"), ("gh", "ਘ"), ("ch", "ਚ"), ("jh", "ਝ"), ("th", "ਥ"),
    ("dh", "ਧ"), ("ph", "ਫ"), ("bh", "ਭ"), ("sh", "ਸ਼"), ("nn", "ੰਨ"),
    ("a", "ਅ"), ("i", "ਇ"), ("u", "ਉ"), ("e", "ੇ"), ("o", "ੋ"),
    ("k", "ਕ"), ("g", "ਗ"), ("j", "ਜ"), ("t", "ਤ"), ("d", "ਦ"),
    ("n", "ਨ"), ("p", "ਪ"), ("b", "ਬ"), ("m", "ਮ"), ("y", "ਯ"),
    ("r", "ਰ"), ("l", "ਲ"), ("v", "ਵ"), ("w", "ਵ"), ("s", "ਸ"), ("h", "ਹ"),
]


def options(word, n=5):
    """Return up to n Gurmukhi candidates for one roman word."""
    word = word.strip()
    if not word:
        return []
    if word in _CACHE:
        return _CACHE[word]
    try:
        url = "https://inputtools.google.com/request?" + urllib.parse.urlencode(
            {"itc": "pa-t-i0-und", "num": n, "text": word})
        with urllib.request.urlopen(url, timeout=8) as r:
            d = json.load(r)
        cands = d[1][0][1] if d and d[0] == "SUCCESS" else []
    except Exception:
        cands = [_fallback_word(word)]
    _CACHE[word] = cands
    return cands


def _fallback_word(word):
    out, w = "", word.lower()
    while w:
        for k, v in _FALLBACK:
            if w.startswith(k):
                out += v
                w = w[len(k):]
                break
        else:
            out += w[0]
            w = w[1:]
    return out


def to_gurmukhi(text):
    """Transliterate a whole line/phrase, best candidate per word. Keeps Gurmukhi as-is."""
    out = []
    for tok in text.split(" "):
        if not tok or any("਀" <= c <= "੿" for c in tok):
            out.append(tok)                     # already Gurmukhi or empty
        else:
            cands = options(tok, 1)
            out.append(cands[0] if cands else tok)
    return " ".join(out)
