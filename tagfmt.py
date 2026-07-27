#!/usr/bin/env python3
"""Structural tag format for Punjabi songs and stories.

Why tags: training examples are windows, but a song is a whole. Tags carry song-level
context (refrain, rhyme, position) into every window, so the model learns architecture
-- mukhda/antara alternation, rhyme discipline, narrative progression -- rather than
just local phrasing.

Test:  ~/.venvs/punjabi-lm/bin/python tagfmt.py
"""
import re
import unicodedata
from pathlib import Path

GURMUKHI = re.compile(r"[਀-੿]")

# Section tags. Punjabi names, because the model already knows these words and they cost
# fewer tokens than inventing English labels for Punjabi forms.
MUKHDA = "ਮੁਖੜਾ"    # refrain / hook
ANTARA = "ਅੰਤਰਾ"    # verse
SATHAI = "ਸਥਾਈ"     # opening line of a kali

# Duet (dogana) voice markers. In a dogana two voices trade stanzas — the ਸਵਾਲ-ਜਵਾਬ
# (question-answer) that defines the form. ASR can't tell who sang, so the human marks a
# stanza's voice while editing by starting it with one of these; the tagger turns it into
# Suno's [Verse N: Female] / [Verse N: Male] speaker labels — the tags that render duets
# correctly in Suno. Both a Gurmukhi word and a short roman letter are accepted.
VOICE_TOKENS = {
    "ਕੁੜੀ": "Female", "ਕੁੜੀਆਂ": "Female", "ਕ": "Female", "f": "Female", "female": "Female",
    "ਮੁੰਡਾ": "Male", "ਮੁੰਡੇ": "Male", "ਮ": "Male", "m": "Male", "male": "Male",
    "ਦੋਵੇਂ": "Both", "ਦੋਨੋਂ": "Both", "b": "Both", "both": "Both",
}
_VOICE_RE = re.compile(r"^\s*([^\s:–\-]+)\s*[:\-–]\s*(.*)$")


def strip_voice(line):
    """If a line begins with a voice marker (ਕੁੜੀ:/ਮੁੰਡਾ:/ਦੋਵੇਂ: or F:/M:/B:), return
    (voice, rest-of-line). Otherwise (None, line). Roman markers are case-insensitive."""
    m = _VOICE_RE.match(line)
    if not m:
        return None, line
    tok, rest = m.group(1), m.group(2)
    voice = VOICE_TOKENS.get(tok) or VOICE_TOKENS.get(tok.lower())
    return (voice, rest.strip()) if voice else (None, line)


def has_voice_markers(text):
    return any(strip_voice(l)[0] for l in text.splitlines())


# Section overrides. The mukhda/antara heuristic in build_song labels every non-refrain
# stanza [Verse N] and always repeats the refrain as [Outro] — it has no notion of a
# distinct closing stanza, a bridge, or an intro. So, exactly like the duet voice markers,
# the editor can force a stanza's section by starting it with a marker. An explicit [Outro]
# also suppresses the automatic refrain-outro (the "[Chorus] and [Outro] are identical"
# case). Gurmukhi or short English, case-insensitive.
SECTION_TOKENS = {
    "ਇੰਟਰੋ": "Intro", "intro": "Intro",
    "ਮੁਖੜਾ": "Chorus", "ਸਥਾਈ": "Chorus", "mukhda": "Chorus", "chorus": "Chorus",
    "ਅੰਤਰਾ": "Verse", "antara": "Verse", "verse": "Verse",
    "ਪੁਲ": "Bridge", "bridge": "Bridge",
    "ਅੰਤ": "Outro", "outro": "Outro", "end": "Outro",
}
_SECTION_RE = re.compile(r"^\s*([A-Za-z਀-੿]+)\s*[:：]\s*(.*)$")


def strip_section(line):
    """If a line begins with a section marker (ਅੰਤ:/outro:, ਪੁਲ:/bridge:, ਮੁਖੜਾ:/chorus:,
    ਅੰਤਰਾ:/verse:, ਇੰਟਰੋ:/intro:), return (Section, rest-of-line). Otherwise (None, line)."""
    m = _SECTION_RE.match(line)
    if not m:
        return None, line
    tok, rest = m.group(1), m.group(2)
    sec = SECTION_TOKENS.get(tok) or SECTION_TOKENS.get(tok.lower())
    return (sec, rest.strip()) if sec else (None, line)


def _stanza_section(stanza):
    """(Section|None, stanza-without-its-marker-line). A leading marker forces the section."""
    lines = stanza.splitlines()
    if not lines:
        return None, stanza
    sec, first = strip_section(lines[0].strip())
    if not sec:
        return None, stanza
    body = ([first] if first.strip() else []) + lines[1:]
    return sec, "\n".join(body).strip()


def gurmukhi_ratio(text):
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(GURMUKHI.match(c)) for c in letters) / len(letters)


def rhyme_key(line, n=3):
    """Trailing characters of a line — a crude but effective rhyme signature.

    Punjabi folk rhyme is overwhelmingly end-rhyme on the final word, so the last few
    characters capture it without needing a phonetic dictionary.
    """
    w = [w for w in re.sub(r"[^\w\s਀-੿]", "", line).split() if w]
    return w[-1][-n:] if w else ""


def split_stanzas(text):
    return [s.strip() for s in re.split(r"\n\s*\n", text.strip()) if s.strip()]


# Story scene segmentation. A story's [Scene N] tags come from its paragraph breaks, but a
# pasted / OCR'd story is often one unbroken block (no blank lines) — which would collapse
# the whole thing into a single [Scene 1]. So: blank-line paragraphs win when present (the
# author's real breaks); otherwise we segment the block into evenly sized scenes at sentence
# boundaries so a flat wall of prose still carries structure. These fallback breaks are
# size-based, not semantic — inserting a blank line in the Editor always overrides them.
STORY_LONG_WORDS = 400          # a story past this many words is labelled "long", else "short"
SCENE_SENTENCES = 6             # sentences per scene when segmenting an unbroken block
_SENT_END = re.compile(r"[।!?.]\s+")


def word_count(text):
    return len(re.findall(r"\S+", text))


def story_scenes(text):
    """Scene bodies for a story, preserving the original text verbatim (only cutting it).
    Blank-line paragraphs are used as-is; an unbroken block is split every SCENE_SENTENCES
    sentences."""
    text = text.strip()
    paras = split_stanzas(text)
    if len(paras) > 1:
        return paras                        # author supplied real paragraph breaks
    ends = [m.end() for m in _SENT_END.finditer(text)]
    if len(ends) < SCENE_SENTENCES:         # too short to bother segmenting
        return [text]
    cuts = ends[SCENE_SENTENCES - 1::SCENE_SENTENCES]
    scenes, start = [], 0
    for c in cuts:
        chunk = text[start:c].strip()
        if chunk:
            scenes.append(chunk)
        start = c
    tail = text[start:].strip()
    if tail:
        scenes.append(tail)
    return scenes


def detect_mukhda(stanzas):
    """The most-repeated line across stanzas is the refrain."""
    counts = {}
    for st in stanzas:
        for line in st.splitlines():
            line = line.strip()
            if line:
                counts[line] = counts.get(line, 0) + 1
    if not counts:
        return None
    best, n = max(counts.items(), key=lambda kv: kv[1])
    return best if n >= 2 else None


def detect_bridge(bodies, main_rhyme):
    """Index of the bridge (ਪੁਲ) stanza, or None.

    A bridge departs from the song: different rhyme, and it appears once, late. So the
    signal is a stanza whose end-rhyme breaks the dominant scheme, sitting in the back
    half of the song. Heuristic by nature — the Editor lets you override it.
    """
    if len(bodies) < 3 or not main_rhyme:
        return None
    odd = [i for i, b in enumerate(bodies)
           if b and rhyme_key(b[-1]) != main_rhyme]
    # exactly one departure, and not the opening stanza
    if len(odd) == 1 and odd[0] >= len(bodies) // 2:
        return odd[0]
    return None


def build_song(text, title, form="kali", theme="", artist="", style_extra="",
               intro="", bridge_at=None):
    """SONG scheme — Suno-standard tags.

    Three content types exist and must not be mixed:
      song      geet / kali / tappa       verse, sung        -> this function
      qissa kav Heer, Sassi Punnu         narrative VERSE    -> build_qissa()
      story     short / long              prose              -> build_story()

    Punjabi folk structure is mukhda (refrain, a.k.a. sthayi) + antara (verse), with the
    mukhda repeating after each antara. That maps exactly onto Suno's [Chorus]/[Verse N],
    so we use the standard tags: Gemma has seen them in pretraining, they cost 3-5 tokens,
    and the output stays pasteable into Suno.
    """
    text = unicodedata.normalize("NFC", text)

    # A dogana (duet) is voice-attributed — route to the speaker-tagged builder when the
    # editor marked voices, or when the form itself says duet.
    if has_voice_markers(text) or form.lower() in ("dogana", "duet", "dohra"):
        return build_duet(text, title, form=form, theme=theme, artist=artist,
                          style_extra=style_extra, intro=intro)

    stanzas = split_stanzas(text)
    # A stanza may carry an editor section override (ਅੰਤ:/outro:, ਪੁਲ:/bridge:, …). The
    # refrain is detected from the marker-stripped text so a marked stanza still counts.
    parsed = [_stanza_section(st) for st in stanzas]
    mukhda = detect_mukhda([txt for _, txt in parsed])

    bodies = []            # (section|None, [verse lines])
    for sec, txt in parsed:
        verse = [l.strip() for l in txt.splitlines()
                 if l.strip() and l.strip() != mukhda]
        if verse or sec:
            bodies.append((sec, verse))

    rhymes = [rhyme_key(v[-1]) for _, v in bodies if v]
    rhyme = max(set(rhymes), key=rhymes.count) if rhymes else ""

    style = f"Punjabi folk {form}"
    if artist:
        style += f", {artist}"
    if style_extra:
        style += f", {style_extra}"

    out = [f"[Style: {style}]"]
    if title:
        out.append(f"[Title: {title}]")
    if theme:
        out.append(f"[Theme: {theme}]")
    if rhyme:
        out.append(f"[Rhyme: {rhyme}]")

    if intro:
        out += ["[Intro]", intro]

    if bridge_at is None:
        bridge_at = detect_bridge([v for _, v in bodies], rhyme)

    # An explicit [Outro] marker means the song ends on that stanza — don't also tack the
    # refrain on as an outro (that's the "[Chorus] and [Outro] are identical" complaint).
    explicit_outro = any(sec == "Outro" for sec, _ in bodies)

    # mukhda repeats after every antara — that alternation IS the form, so emit it. A marked
    # stanza jumps straight to its own section (no preceding [Chorus]).
    vnum = 0
    for i, (sec, verse) in enumerate(bodies):
        if sec in ("Intro", "Outro"):
            out.append(f"[{sec}]")
            out.extend(verse)
            continue
        if sec == "Chorus":
            out += ["[Chorus]", *(verse or ([mukhda] if mukhda else []))]
            continue
        if mukhda:
            out += ["[Chorus]", mukhda]
        if sec == "Bridge" or (sec is None and i == bridge_at):
            out.append("[Bridge]")          # ਪੁਲ — departs from the main rhyme
        else:
            vnum += 1
            out.append(f"[Verse {vnum}]")
        out.extend(verse)
    if mukhda and not explicit_outro:
        out += ["[Outro]", mukhda]
    return "\n".join(out)


def _voice_segments(text):
    """Split a duet into (voice, [lines]) segments, honouring the editor's markers.

    A marker sets the voice for what follows until the next marker or a blank line. A blank
    line ends a segment (stanza break); the voice persists only within a marked run. Lines
    before any marker come back as voice=None so they still render as plain verses.
    """
    segments, voice, lines = [], None, []
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            if lines:
                segments.append((voice, lines))
                lines = []
            continue
        v, rest = strip_voice(s)
        if v:
            if lines:
                segments.append((voice, lines))
                lines = []
            voice = v
            if rest:
                lines.append(rest)
        else:
            lines.append(s)
    if lines:
        segments.append((voice, lines))
    return segments


def build_duet(text, title, form="dogana", theme="", artist="", style_extra="", intro=""):
    """DOGANA (duet) scheme — two voices trade stanzas, tagged with Suno speaker labels.

    Emits [Verse N: Female] / [Verse N: Male] (and [Chorus: Both] for a shared refrain) —
    the exact tags that make Suno render a Punjabi duet with alternating singers. Voice
    attribution comes from the markers the editor typed (see strip_voice); it cannot be
    inferred from ASR, so an unmarked stanza is left as a plain [Verse N].
    """
    text = unicodedata.normalize("NFC", text)
    segments = _voice_segments(text)

    # Refrain: the single line repeated across segments — usually sung by both.
    counts = {}
    for _, lines in segments:
        for l in lines:
            counts[l] = counts.get(l, 0) + 1
    mukhda = None
    if counts:
        best, n = max(counts.items(), key=lambda kv: kv[1])
        mukhda = best if n >= 2 else None

    rhymes = [rhyme_key(lines[-1]) for _, lines in segments if lines]
    rhyme = max(set(rhymes), key=rhymes.count) if rhymes else ""

    style = f"Punjabi folk {form} (duet)"
    if artist:
        style += f", {artist}"
    if style_extra:
        style += f", {style_extra}"

    out = [f"[Style: {style}]"]
    if title:
        out.append(f"[Title: {title}]")
    if theme:
        out.append(f"[Theme: {theme}]")
    if rhyme:
        out.append(f"[Rhyme: {rhyme}]")
    if intro:
        out += ["[Intro]", intro]

    vnum = 0
    for voice, lines in segments:
        body = [l for l in lines if l != mukhda]
        is_refrain = mukhda and not body           # this segment is only the refrain
        if is_refrain:
            out += [f"[Chorus: {voice}]" if voice and voice != "Both" else "[Chorus: Both]",
                    mukhda]
            continue
        vnum += 1
        out.append(f"[Verse {vnum}: {voice}]" if voice else f"[Verse {vnum}]")
        if mukhda and mukhda in lines:             # refrain tucked into a voice's stanza
            out += body + ["[Chorus: Both]", mukhda]
        else:
            out.extend(body)
    return "\n".join(out)


def build_qissa(stanzas, title, position=None, total=None,
                characters="", meter="baint", episode=""):
    """QISSA KAV scheme — narrative verse (Heer Waris Shah, Sassi Punnu, Mirza Sahiban).

    Not a story: it is metrical, rhymed, sung/recited verse that happens to carry a plot.
    Tagging it as prose would destroy the meter. Waris Shah's Heer is in *baint* meter.

    Because a qissa runs to thousands of stanzas, a training window is a slice — so the
    header carries narrative position, letting the model learn arc as well as meter.
    """
    out = [f"[Form: qissa kav]", f"[Meter: {meter}]"]
    if title:
        out.append(f"[Title: {title}]")
    if characters:
        out.append(f"[Characters: {characters}]")
    if episode:
        out.append(f"[Episode: {episode}]")
    if position and total:
        out.append(f"[Position: {position}/{total}]")
    for i, st in enumerate(stanzas, 1):
        out.append(f"[Bait {i}]")
        out.extend(st if isinstance(st, list) else st.splitlines())
    return "\n".join(out)


def build_story(text, title, kind=None, theme="", characters=""):
    """STORY scheme — prose. Plain narrative, no meter, no rhyme.

    kind=None (the default) auto-labels the story "short" or "long" by length; pass "short"
    or "long" explicitly to override. Scenes come from paragraph breaks, or from sentence-
    grouping when the text is one unbroken block (see story_scenes)."""
    text = unicodedata.normalize("NFC", text)
    if kind is None:
        kind = "long" if word_count(text) > STORY_LONG_WORDS else "short"
    out = [f"[Form: {kind} story]"]
    if title:
        out.append(f"[Title: {title}]")
    if theme:
        out.append(f"[Theme: {theme}]")
    if characters:
        out.append(f"[Characters: {characters}]")
    for i, para in enumerate(story_scenes(text), 1):
        out.append(f"[Scene {i}]")
        out.append(para)
    return "\n".join(out)


def build_essay(text, title, form="essay", theme=""):
    """ESSAY scheme (ਲੇਖ) — expository prose: essay / article / blog. Not a story: no plot,
    characters or scenes; it argues or reflects on a topic. So it is tagged with [Section N]
    (discursive parts), never [Scene N]. `form` labels the sub-kind (essay/article/blog).
    Segmentation reuses story_scenes — paragraph breaks, or sentence-grouping for one block."""
    text = unicodedata.normalize("NFC", text)
    out = [f"[Form: {form}]"]
    if title:
        out.append(f"[Title: {title}]")
    if theme:
        out.append(f"[Theme: {theme}]")
    for i, para in enumerate(story_scenes(text), 1):
        out.append(f"[Section {i}]")
        out.append(para)
    return "\n".join(out)


def build_compact(text, title, form="kali", theme="", kind="song"):
    """Compact scheme — same structure, far fewer tokens.

    Measured tag costs in Gemma's tokenizer (the reason this scheme exists):
      antara / verse -> 1 token      ਅੰਤਰਾ -> 4      ਮੁਖੜਾ -> 5
      [V2/4]         -> 6 tokens     <ਅੰਤਰਾ n="2" of="4"> -> 13
    Closing tags are dropped entirely (0 tokens) — a blank line already separates
    stanzas, so </...> buys nothing but costs 3-6 tokens every section.
    """
    text = unicodedata.normalize("NFC", text)
    stanzas = split_stanzas(text)
    mukhda = detect_mukhda(stanzas)

    bodies = []
    for st in stanzas:
        verse = [l.strip() for l in st.splitlines()
                 if l.strip() and l.strip() != mukhda]
        if verse:
            bodies.append(verse)

    rhymes = [rhyme_key(b[-1]) for b in bodies if b]
    rhyme = max(set(rhymes), key=rhymes.count) if rhymes else ""

    head = f'<{kind} {form}'
    if title:
        head += f' "{title}"'
    if theme:
        head += f' theme="{theme}"'
    if rhyme:
        head += f' rhyme="{rhyme}"'
    head += f' n={len(bodies)}>'

    out = [head]
    if mukhda:
        out.append(f"[M] {mukhda}")
    for i, verse in enumerate(bodies, 1):
        out.append(f"[V{i}/{len(bodies)}]")
        out.extend(verse)
    return "\n".join(out)


def build(text, title, form="kali", theme="", kind="song"):
    """Verbose XML-style scheme. Kept for comparison — costs ~42% overhead."""
    text = unicodedata.normalize("NFC", text)
    stanzas = split_stanzas(text)
    mukhda = detect_mukhda(stanzas)

    body_stanzas = []
    for st in stanzas:
        lines = [l.strip() for l in st.splitlines() if l.strip()]
        # Drop the refrain from verse bodies; it lives in the header instead, so every
        # window sees it without paying for it repeatedly inside the text.
        verse = [l for l in lines if l != mukhda]
        if verse:
            body_stanzas.append(verse)

    rhymes = [rhyme_key(l[-1]) for l in body_stanzas if l]
    rhyme = max(set(rhymes), key=rhymes.count) if rhymes else ""

    head = [f'<{kind} title="{title}" form="{form}"']
    if theme:
        head.append(f'theme="{theme}"')
    if rhyme:
        head.append(f'rhyme="{rhyme}"')
    head.append(f'sections="{len(body_stanzas)}">')
    out = [" ".join(head)]
    if mukhda:
        out.append(f"<{MUKHDA}>{mukhda}</{MUKHDA}>")
    for i, verse in enumerate(body_stanzas, 1):
        out.append(f'<{ANTARA} n="{i}" of="{len(body_stanzas)}">')
        out.extend(verse)
        out.append(f"</{ANTARA}>")
    out.append(f"</{kind}>")
    return "\n".join(out)


def main():
    import os
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    from transformers import AutoTokenizer

    src = Path("/mnt/e/Transcribe/corpus/03. CHAN CHANNI RAAT.txt")
    raw = src.read_text(encoding="utf-8")
    kw = dict(title="ਚੰਨ ਚਾਨਣੀ ਰਾਤ", form="kali", theme="ਵਿਆਹ ਮੁਕਲਾਵਾ")
    compact, verbose = build_compact(raw, **kw), build(raw, **kw)

    print("=" * 60)
    print("COMPACT SCHEME (recommended)")
    print("=" * 60)
    print(compact)

    tok = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
    n = lambda s: len(tok(s).input_ids)
    n_raw, n_c, n_v = n(raw), n(compact), n(verbose)

    print("\n" + "=" * 60)
    print("TOKEN COST")
    print("=" * 60)
    print(f"{'scheme':10} {'tokens':>7} {'overhead':>10} {'vs raw':>8}")
    print(f"{'raw':10} {n_raw:7d} {'-':>10} {'-':>8}")
    print(f"{'compact':10} {n_c:7d} {n_c-n_raw:10d} {100*(n_c-n_raw)/n_raw:7.1f}%")
    print(f"{'verbose':10} {n_v:7d} {n_v-n_raw:10d} {100*(n_v-n_raw)/n_raw:7.1f}%")
    print(f"\ngurmukhi ratio : {gurmukhi_ratio(raw):.3f}")
    print(f"fits seq 1024  : {'YES' if n_c <= 1024 else 'NO'} "
          f"({1024 - n_c} spare with compact)")


if __name__ == "__main__":
    main()
