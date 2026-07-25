#!/usr/bin/env python3
"""Scanned-PDF -> Gurmukhi text, for importing a phone-scanned book.

A phone photo of a page has no text layer, so plain extraction gets nothing — the page
has to be *read*. We render each page to an image with PyMuPDF and hand it to the same
`claude` CLI the rest of the pipeline uses (no API key, uses your existing auth). Claude
vision reads printed Gurmukhi — including the conjuncts (ਸ਼, ੰ, ੍ਰ) that Tesseract mangles —
far better than an offline OCR engine on a phone scan.

Same honest framing as the audio reconstruction: this is transcribing *your own scan of
your own book*, nothing more. If Claude declines a page as copyrighted, the page is flagged
REFUSED-copyright and left blank for you to type by hand — we do not jailbreak around it.

The output is a rough draft on purpose. Every page still goes through the Editor gate before
it can become training data — OCR just saves you copy-typing a book.
"""
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF

import library  # strip_commentary()

# {name} is filled with the image basename. The explicit "use your Read tool" instruction
# is load-bearing: without it the CLI sometimes treats the filename as literal text and
# never opens the image (returns "I don't see an image").
OCR_PROMPT = (
    "Use your Read tool to open the image file {name} in the current directory. It is a "
    "scanned page from my own Punjabi book. Transcribe the Punjabi (Gurmukhi) text on the "
    "page exactly as printed, preserving line breaks and stanza breaks. Output ONLY the "
    "Gurmukhi text — no translation, no romanisation, no commentary, no headers. Keep the "
    "reading order top-to-bottom. If a word is genuinely unclear, give your best single "
    "reading rather than a guess-list. If the page is blank or has no Punjabi text, output "
    "exactly: (blank)"
)

# Response looks like the model never actually opened the image — retry once.
_MISS = ("i don't see an image", "no image", "couldn't open", "could not open",
         "needs permission", "wasn't able to open", "provide the image")


def page_count(pdf_path):
    with fitz.open(pdf_path) as doc:
        return doc.page_count


def render_pages(pdf_path, out_dir, dpi=220, first=1, last=None):
    """Render pages [first..last] (1-based, inclusive) to PNGs. Yields (page_no, png_path)."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    with fitz.open(pdf_path) as doc:
        last = last or doc.page_count
        for n in range(first, min(last, doc.page_count) + 1):
            pix = doc.load_page(n - 1).get_pixmap(matrix=mat)
            png = Path(out_dir) / f"page_{n:04d}.png"
            pix.save(png)
            yield n, str(png)


def ocr_page(png_path, model, timeout=180):
    """Read one page image with Claude vision via the CLI. Returns (text, refused).

    The CLI's file-read tool is permission-scoped to its working directory, so we run it
    with cwd = the image's folder and reference the image by basename — an absolute path in
    /tmp is silently refused (returns nothing), which would look like a blank page.
    """
    png = Path(png_path)
    prompt = OCR_PROMPT.format(name=png.name)
    raw = ""
    for attempt in range(2):
        try:
            p = subprocess.run(
                ["claude", "-p", prompt, "--model", model], cwd=str(png.parent),
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=timeout)
        except Exception as e:
            return f"[ERROR {type(e).__name__}: {e}]", False
        if p.returncode != 0:
            return f"[ERROR claude rc={p.returncode}: {p.stderr.strip()[:200]}]", False
        raw = (p.stdout or "").strip()
        low = raw.lower()
        if not any(m in low for m in _MISS):
            break            # image was read — accept; else loop once more
    low = raw.lower()
    # Honest refusal handling — flag, never jailbreak.
    if not raw or ("can't" in low and "copyright" in low) or "i cannot help" in low:
        return "", True
    if raw == "(blank)":
        return "", False
    return library.strip_commentary(raw), False


def pdf_to_text(pdf_paths, model, first=1, last=None, log=print):
    """Generator over one or more PDFs. Yields the accumulated text after each page so the
    UI can stream it into the editable box. Pages are separated by a marker the user can
    split on in the Editor."""
    out = []
    with tempfile.TemporaryDirectory(prefix="panlm-ocr-") as tmp:
        for pdf_path in pdf_paths:
            name = Path(pdf_path).name
            total = page_count(pdf_path)
            log(f"{name}: {total} page(s)")
            for n, png in render_pages(pdf_path, tmp, first=first, last=last):
                text, refused = ocr_page(png, model)
                if refused:
                    log(f"  p{n}: REFUSED-copyright — left blank")
                    out.append(f"----- {name} p{n}  [REFUSED-copyright: type by hand] -----")
                elif text:
                    ratio_ok = sum(c.isspace() for c in text) < len(text)
                    log(f"  p{n}: {len(text.split())} words" + ("" if ratio_ok else " (empty?)"))
                    out.append(f"----- {name} p{n} -----\n{text}")
                else:
                    log(f"  p{n}: blank")
                    out.append(f"----- {name} p{n}  [blank] -----")
                yield "\n\n".join(out)
    log("done")
