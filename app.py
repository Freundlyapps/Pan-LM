#!/usr/bin/env python3
"""Punjabi LM — training platform.

Every stage is manually triggered. Nothing runs on its own, and nothing reaches a dataset
without passing through the Editor's approve gate.

    ./run.sh      ->  http://localhost:7861
"""
import gradio as gr

import config
import dataset
import infer
import jobs
import library
import pdftext
import quality
import state
import tagfmt
import translit
import train_job

CFG = config.load()
CON = state.connect(config.data_path(CFG, "state.db"))
KINDS = ["song", "qissa", "story", "lekh"]


# ---------------------------------------------------------------- helpers
def stats_md():
    c = state.counts(CON)
    total = sum(c.values())
    if not total:
        return "**No items yet** — run a scan in the Library tab."
    order = ["pending", "transcribed", "edited", "approved", "rejected"]
    bits = " · ".join(f"**{c[s]}** {s}" for s in order if c.get(s))
    return f"{bits} · **{total}** total"


def folder_choices():
    return ["all"] + [f"{f}  ({n})" for f, n in state.folders(CON)]


def _folder(sel):
    return "all" if not sel or sel == "all" else sel.split("  (")[0]


def item_rows(fstate, folder, kind):
    rows = state.query(CON, fstate, _folder(folder), kind)
    return [[r["id"], r["title"], r["folder"], r["kind"], r["state"],
             (r["final_text"] or "")[:60].replace("\n", " ")] for r in rows]


def needs_review_ids():
    return [r["id"] for r in state.query(CON, "transcribed")] + \
           [r["id"] for r in state.query(CON, "edited")]


def review_queue(scope="needs review"):
    """Ids to walk through in the Editor, by queue scope. 'approved'/'rejected' let you page
    back into already-decided items to fix a mistake; 'all' walks everything."""
    if scope == "needs review":
        ids = needs_review_ids()
    elif scope in ("approved", "rejected"):
        ids = [r["id"] for r in state.query(CON, scope)]
    else:                                      # "all"
        ids = [r["id"] for r in state.query(CON)]
    return sorted(set(ids))


def neighbour_id(current, step, scope):
    """Next/prev id in the review queue relative to `current`."""
    q = review_queue(scope)
    if not q:
        return current
    if not current or int(current) not in q:
        return q[0]
    i = q.index(int(current))
    return q[(i + step) % len(q)]


# ---------------------------------------------------------------- library
def do_scan():
    lines = []
    library.scan(CON, CFG, log=lines.append)
    return "\n".join(lines), stats_md(), gr.update(choices=folder_choices())


def do_hint(folder, hint):
    f = _folder(folder)
    if f == "all":
        return "pick a specific folder"
    state.set_folder_hint(CON, f, lang_hint=hint)
    return f"{f} marked as {hint}"


# ---------------------------------------------------------------- transcribe
def list_pending(folder):
    """Populate the track picker for the chosen folder."""
    ch = library.pending_choices(CON, _folder(folder))
    return gr.update(choices=ch, value=[]), f"{len(ch)} pending in this folder"


def pick_first_n(folder, n):
    ch = library.pending_choices(CON, _folder(folder))[: int(n)]
    return gr.update(choices=library.pending_choices(CON, _folder(folder)), value=ch)


def _ids(picked):
    return [int(p.split(" — ")[0]) for p in (picked or [])]


def start_transcribe(folder, limit, mode, model, picked):
    if jobs.MANAGER.is_running("transcribe"):
        return "already running"
    ids = _ids(picked)
    ok, msg = jobs.MANAGER.start(
        "transcribe",
        lambda job: library.transcribe_batch(
            CON, CFG, _folder(folder), int(limit), mode, model or None, job,
            item_ids=ids or None))
    return f"{msg} ({len(ids)} picked)" if ids else f"{msg} (next {int(limit)})"


def do_upload(files, folder):
    added = library.register_uploads(CON, CFG, files, folder or "(uploaded)")
    if not added:
        return "no audio files recognised", stats_md(), gr.update()
    labels = [f"{i} — {n}" for i, n in added]
    return (f"added {len(added)}: {', '.join(n for _, n in added[:5])}"
            + ("…" if len(added) > 5 else ""),
            stats_md(), gr.update(choices=labels, value=labels))


def poll_transcribe():
    j = jobs.MANAGER.get("transcribe")
    return j.status(), j.text(), stats_md()


# ---------------------------------------------------------------- editor
def load_item(item_id):
    if not item_id:
        return ("", "", "", "song", "", "", "", "no item selected")
    r = state.get(CON, int(item_id))
    if not r:
        return ("", "", "", "song", "", "", "", f"item {item_id} not found")
    txt = r["final_text"] or r["clean_text"] or ""
    note = quality.summary(quality.score(txt, r["kind"] or "song")) if txt else "(empty)"
    return (r["raw_text"] or "(no raw transcript)", txt, r["title"], r["kind"] or "song",
            r["artist"] or "", r["form"] or "", r["theme"] or "",
            f"**state: {r['state']}** · {note}")


def load_with_pos(item_id, scope):
    """load_item plus a 'position in queue' line and the resolved id (for Prev/Next)."""
    out = load_item(item_id)
    q = review_queue(scope)
    pos = ""
    if item_id and int(item_id) in q:
        pos = f"  ·  {q.index(int(item_id)) + 1} / {len(q)} in '{scope}'"
    elif q:
        pos = f"  ·  {len(q)} in '{scope}'"
    return (int(item_id) if item_id else None, *out[:7], out[7] + pos)


def go(step, current, scope, text, title, kind, artist, form, theme):
    """Auto-save the current song, then load the neighbour — so edit → Next never loses work."""
    if current and (text or "").strip():
        state.save_edit(CON, int(current), text)
        state.set_fields(CON, int(current), title=title, kind=kind, artist=artist,
                         form=form, theme=theme)
    nid = neighbour_id(current, step, scope)
    return load_with_pos(nid, scope)


def open_from_table(fstate, folder, kind, evt: gr.SelectData):
    """Click a Library row -> its id, ready to open in the Editor."""
    rows = state.query(CON, fstate, _folder(folder), kind)
    if evt.index and evt.index[0] < len(rows):
        return rows[evt.index[0]]["id"]
    return None


def phon_convert(roman):
    """Roman -> Gurmukhi, showing the best guess plus alternatives to pick from."""
    if not roman.strip():
        return "", "type romanized Punjabi above, e.g. muklawa naroi"
    best = translit.to_gurmukhi(roman.strip())
    # per-word alternatives for the last word, the one you're most likely tuning
    last = roman.strip().split(" ")[-1]
    alts = translit.options(last, 5)
    alt_line = ("alternatives for '%s': %s" % (last, "  ".join(alts))) if alts else ""
    return best, alt_line


def append_to_text(current, addition):
    if not addition.strip():
        return current
    sep = "" if current.endswith("\n") or not current else "\n"
    return current + sep + addition.strip()


def find_replace(text, find, repl, all_occ):
    if not find:
        return text, "enter text to find"
    if find not in text:
        return text, f"'{find}' not found"
    n = text.count(find)
    new = text.replace(find, repl) if all_occ else text.replace(find, repl, 1)
    did = n if all_occ else 1
    return new, f"replaced {did} of {n} occurrence(s)"


def save_item(item_id, text, title, kind, artist, form, theme):
    if not item_id:
        return "no item"
    i = int(item_id)
    prev = state.get(CON, i)
    was_approved = bool(prev) and prev["state"] == "approved"
    state.save_edit(CON, i, text)
    state.set_fields(CON, i, title=title, kind=kind, artist=artist, form=form, theme=theme)
    if was_approved:
        return "saved — sent back to review (was approved); Approve again to re-add it"
    return "saved"


def delete_current(item_id, scope, confirm):
    """Remove the loaded item, then move to the next in the current queue. Guarded by a
    confirm checkbox so a misclick can't wipe good work. An audio track is *reset* to pending
    (kept for transcription, not lost); a pasted-text item is hard-deleted."""
    if not item_id:
        return (*load_with_pos(None, scope), stats_md())
    i = int(item_id)
    r = state.get(CON, i)
    if not r:
        return (*load_with_pos(None, scope), stats_md())
    if not confirm:
        cur = load_with_pos(i, scope)
        return (cur[0], *cur[1:8], "⚠ tick 'confirm delete' first — then click Delete. "
                + cur[8], stats_md())
    is_audio = bool(r["path"])                 # a real track: keep the file, just reset it
    nxt = neighbour_id(i, 1, scope)
    if is_audio:
        state.reset_item(CON, i)
        verb = f"reset audio item {i} ('{r['title']}') to pending — transcribe it later"
    else:
        state.delete_item(CON, i)
        verb = f"🗑 deleted item {i} ('{r['title']}')"
    nxt = None if nxt == i else nxt
    res = load_with_pos(nxt, scope)
    return (res[0], *res[1:8], f"{verb}. {res[8]}", stats_md())


def decide(item_id, approve, text="", kind="song", title=None, force=False):
    """Approving a `reject`-scored item requires the override checkbox.

    The corpus is only ~500 songs, so a few bad transcripts measurably damage the model.
    The gate is deliberately hard to click through by accident — but never impossible,
    because the detector is heuristic and you are the judge.

    Approve persists the two fields that actually change training — `type` (which builder
    runs) and `title` (the model learns to emit it) — so you can Approve without a prior
    Save. The optional fields (artist/form/theme) still need Save; they're overrides for
    metadata the Dataset step otherwise auto-generates.
    """
    if not item_id:
        return "no item", stats_md()
    # Persist the on-screen text BEFORE deciding, so an edit + Approve never loses the edit.
    if text:
        state.save_edit(CON, int(item_id), text)
    if approve:
        m = quality.score(text or "", kind)
        if m["verdict"] == "reject" and not force:
            return (f"❌ BLOCKED — {'; '.join(m['reasons'])}. "
                    "Fix the text, or tick 'override' if the checker is wrong."), stats_md()
        fields = {"state": "approved", "kind": kind,
                  "note": f"approved ({m['verdict']}{' FORCED' if force else ''})"}
        if title is not None and title.strip():
            fields["title"] = title.strip()
        state.set_fields(CON, int(item_id), **fields)
        return f"✅ saved + approved ({m['verdict']})", stats_md()
    state.set_fields(CON, int(item_id), state="rejected")
    return "saved + rejected", stats_md()


def preview_tags(item_id, text, title, kind, artist, form, theme):
    if not text.strip():
        return "(nothing to preview)"
    if kind == "song":
        return tagfmt.build_song(text, title=title, form=form or "kali",
                                 theme=theme, artist=artist)
    if kind == "story":
        return tagfmt.build_story(text, title=title, theme=theme)
    if kind == "lekh":
        return tagfmt.build_essay(text, title=title, theme=theme, form=form or "essay")
    return tagfmt.build_qissa([s.splitlines() for s in tagfmt.split_stanzas(text)],
                              title=title, characters=artist)


# ---------------------------------------------------------------- import
def do_import(title, text, kind, artist, form, theme):
    if not text.strip():
        return "paste some text first", stats_md(), None
    if not title.strip():
        return "give it a title", stats_md(), None
    i = state.add_text_item(CON, title.strip(), text.strip(), kind,
                            artist=artist, form=form, theme=theme)
    # Return the new id so the click chain can open it straight in the Editor — the queue
    # holds 100s of items, so "review it in the Editor" only helps if we jump you to it.
    return (f"imported as item {i} — opened in the Editor tab; review & Approve it there",
            stats_md(), i)


def do_pdf(files, model, first, last, existing):
    """Stream Claude-vision OCR of scanned PDF(s) into the import box. Generator: yields
    (status, text) as each page comes back so you watch the book fill in."""
    if not files:
        yield "drop one or more scanned PDFs first", existing or ""
        return
    paths = [f.name if hasattr(f, "name") else f for f in files]
    model = (model or CFG["claude_model"]).strip()
    first = max(1, int(first or 1))
    last = int(last) if last else None
    logs = []

    def log(m):
        logs.append(m)

    head = (existing.rstrip() + "\n\n") if existing and existing.strip() else ""
    last_text = existing or ""
    for acc in pdftext.pdf_to_text(paths, model, first=first, last=last, log=log):
        last_text = head + acc
        yield "OCR… " + (logs[-1] if logs else ""), last_text
    yield ("OCR done — review/split it, set title+type, then Import. "
           "Any [REFUSED-copyright] page you type by hand."), last_text


# ---------------------------------------------------------------- dataset
def start_dataset(version, model, t_ins, t_theme, t_cont, eval_frac, max_cont):
    if not version.strip():
        return "name the dataset version first"
    kinds_on = {"instruct": t_ins, "theme": t_theme, "continue": t_cont}
    ok, msg = jobs.MANAGER.start(
        "dataset",
        lambda job: dataset.build(CON, CFG, version.strip(), model, kinds_on,
                                  float(eval_frac), job, max_continue=int(max_cont)),
        needs_gpu=False)          # metadata comes from the claude CLI, not the GPU
    return msg


def poll_dataset():
    j = jobs.MANAGER.get("dataset")
    return j.status(), j.text(), gr.update(choices=dataset.list_versions(CFG))


# ---------------------------------------------------------------- train
def gpu_line():
    t, f = None, jobs.gpu_free_mib()
    busy = jobs.MANAGER.gpu_busy()
    s = f"GPU free: **{f} MiB**" if f else "GPU: unknown"
    return s + (f" · ⚠️ busy with **{busy}**" if busy else " · free")


def start_train(ds, epochs, lr, out, max_steps):
    if not ds:
        return "pick a dataset first", gpu_line()
    busy = jobs.MANAGER.gpu_busy()
    if busy:
        return (f"❌ GPU is busy with '{busy}'. Training needs 7.3 GB of your 8 GB — "
                "stop that job first."), gpu_line()
    name = out.strip() or f"{ds}-r{CFG['lora_r']}"
    ok, msg = jobs.MANAGER.start(
        "train",
        lambda job: train_job.train(CFG, ds, int(epochs), float(lr), name, job,
                                    max_steps=int(max_steps)))
    return msg, gpu_line()


def poll_train():
    j = jobs.MANAGER.get("train")
    return j.status(), j.text(), gpu_line()


# ---------------------------------------------------------------- settings
def save_settings(roots, repo, port, batch, cmodel, bmodel, seq, r, chunk):
    CFG.update({
        "source_roots": [x.strip() for x in roots.splitlines() if x.strip()],
        "transcribe_repo": repo, "port": int(port), "batch_size": int(batch),
        "claude_model": cmodel, "base_model": bmodel,
        "seq_len": int(seq), "lora_r": int(r), "ce_chunk": int(chunk),
    })
    config.save(CFG)
    return "saved to config.yaml (port change needs a restart)"


# ---------------------------------------------------------------- ui
with gr.Blocks(title="Punjabi LM") as demo:
    gr.Markdown("# ਪੰਜਾਬੀ LM — training platform")
    stats = gr.Markdown(stats_md())

    with gr.Tabs() as main_tabs:
        with gr.Tab("Library"):
            gr.Markdown("Scan your source roots, then tag folders by language so Hindi/Urdu "
                        "material stays out of a Gurmukhi-only model.")
            with gr.Row():
                scan_btn = gr.Button("Scan library", variant="primary")
                f_state = gr.Dropdown(["all"] + state.STATES, value="all", label="state")
                f_folder = gr.Dropdown(folder_choices(), value="all", label="folder")
                f_kind = gr.Dropdown(["all"] + KINDS, value="all", label="type")
                refresh = gr.Button("Refresh")
            with gr.Row():
                hint = gr.Dropdown(["punjabi", "hindi", "urdu", "mixed", "unknown"],
                                   value="punjabi", label="language of selected folder")
                hint_btn = gr.Button("Tag folder")
            scan_log = gr.Textbox(label="log", lines=3)
            table = gr.Dataframe(
                headers=["id", "title", "folder", "type", "state", "preview"],
                interactive=False, wrap=True, label="items")

            scan_btn.click(do_scan, outputs=[scan_log, stats, f_folder])
            refresh.click(item_rows, [f_state, f_folder, f_kind], table)
            hint_btn.click(do_hint, [f_folder, hint], scan_log)

        with gr.Tab("Transcribe"):
            gr.Markdown(f"Runs `{CFG['transcribe_repo']}/pipeline.sh` in batches. "
                        "Stop is safe — finished tracks stay finished, and the next run skips them.")
            with gr.Row():
                t_folder = gr.Dropdown(folder_choices(), value="all", label="folder")
                t_limit = gr.Number(CFG["batch_size"], label="batch size", precision=0)
                t_mode = gr.Radio(["song", "story"], value="song", label="mode")
                t_model = gr.Textbox(CFG["claude_model"], label="claude model")
            with gr.Row():
                t_list = gr.Button("List pending tracks")
                t_firstn = gr.Button("Tick first N")
                t_clear = gr.Button("Clear selection")
            t_pick = gr.Dropdown([], value=[], multiselect=True, label="tracks to transcribe "
                                 "(leave empty to just take the next N)", filterable=True)
            with gr.Accordion("Upload audio from anywhere", open=False):
                t_up = gr.File(file_count="multiple", label="drop mp3/wav/m4a/flac here")
                t_upfolder = gr.Textbox("(uploaded)", label="assign to folder")
                t_upbtn = gr.Button("Add to library")
            with gr.Row():
                t_start = gr.Button("Start batch", variant="primary")
                t_stop = gr.Button("Stop")
                t_poll = gr.Button("Refresh log")
            t_status = gr.Textbox(label="status", value="idle")
            t_log = gr.Textbox(label="live log", lines=18, max_lines=18)

            t_list.click(list_pending, t_folder, [t_pick, t_status])
            t_firstn.click(pick_first_n, [t_folder, t_limit], t_pick)
            t_clear.click(lambda: gr.update(value=[]), outputs=t_pick)
            t_upbtn.click(do_upload, [t_up, t_upfolder], [t_status, stats, t_pick])
            t_start.click(start_transcribe, [t_folder, t_limit, t_mode, t_model, t_pick],
                          t_status)
            t_stop.click(lambda: jobs.MANAGER.stop("transcribe"), outputs=t_status)
            t_poll.click(poll_transcribe, outputs=[t_status, t_log, stats])
            timer = gr.Timer(2.0)
            timer.tick(poll_transcribe, outputs=[t_status, t_log, stats])

        with gr.Tab("Editor", id="editor"):
            gr.Markdown("**The gate.** Nothing reaches a dataset until you approve it here. "
                        "Raw ASR on the left is your evidence for fixing misheard words.\n\n"
                        "**Fix a mistakenly-approved item:** set **queue → approved**, Prev/Next to "
                        "it, edit, then **Save** (it drops back to review) and **Approve** again — "
                        "or **Reject** it. Save always sends an approved item back to the gate, so a "
                        "correction can never sneak into a dataset unreviewed.")
            with gr.Row():
                e_id = gr.Number(label="item id", precision=0)
                e_load = gr.Button("Load", variant="primary")
                e_prevbtn = gr.Button("← Prev")
                e_nextbtn = gr.Button("Next →")
                e_scope = gr.Dropdown(["needs review", "approved", "rejected", "all"],
                                     value="needs review", label="queue")
            e_note = gr.Markdown()
            with gr.Row():
                e_raw = gr.Textbox(label="raw ASR (both views)", lines=20, interactive=False)
                e_text = gr.Textbox(label="final text — edit freely", lines=20)
            with gr.Row():
                e_title = gr.Textbox(label="title")
                e_kind = gr.Dropdown(KINDS, value="song", label="type")
                e_artist = gr.Textbox(label="artist")
                e_form = gr.Textbox(label="form (kali/geet/tappa/dogana)")
                e_theme = gr.Textbox(label="theme")
            gr.Markdown(
                "**Duet (dogana)?** Start a stanza with `ਕੁੜੀ:` / `ਮੁੰਡਾ:` / `ਦੋਵੇਂ:` (or "
                "`F:` / `M:` / `B:`) to mark who sings it — the tagged preview turns these into "
                "Suno's `[Verse 1: Female]` / `[Verse 2: Male]` / `[Chorus: Both]`. Solo songs "
                "need no marks.\n\n"
                "**Wrong section tag?** The auto-tagger labels every non-refrain stanza "
                "`[Verse N]` and repeats the refrain as `[Outro]`. To override, start a stanza "
                "with `ਅੰਤ:` (outro) · `ਪੁਲ:` (bridge) · `ਇੰਟਰੋ:` (intro) · `ਮੁਖੜਾ:` (chorus) — "
                "English `outro:` / `bridge:` / `intro:` / `chorus:` work too. A stanza marked "
                "`ਅੰਤ:` becomes the closing `[Outro]` with its own words, and the duplicate "
                "refrain-outro is dropped.")
            with gr.Accordion("⌨ Punjabi typing helper — type roman, get Gurmukhi", open=True):
                with gr.Row():
                    e_roman = gr.Textbox(label="romanized (e.g. muklawa naroi koonj)",
                                         scale=3)
                    e_conv = gr.Button("→ Gurmukhi", scale=1)
                e_guru = gr.Textbox(label="Gurmukhi result (edit, then append)", scale=3, buttons=["copy"])
                e_alts = gr.Markdown()
                e_append = gr.Button("↓ append to final text")
                gr.Markdown("**Fix a garbled word:** find it, type the replacement above, paste "
                            "into 'replace with'.")
                with gr.Row():
                    e_find = gr.Textbox(label="find")
                    e_repl = gr.Textbox(label="replace with")
                    e_all = gr.Checkbox(True, label="all")
                    e_dofr = gr.Button("Replace")
            with gr.Row():
                e_save = gr.Button("Save")
                e_ok = gr.Button("Approve", variant="primary")
                e_no = gr.Button("Reject")
                e_prev = gr.Button("Preview tags")
                e_force = gr.Checkbox(False, label="override quality block")
            with gr.Row():
                e_del = gr.Button("Delete item")
                e_delok = gr.Checkbox(False, label="confirm delete")
                gr.Markdown("Delete an empty/junk item. A pasted text item is removed; a real "
                            "audio track is reset to **pending** (kept for transcription).")
            e_msg = gr.Textbox(label="", lines=1)
            e_tagged = gr.Textbox(label="tagged preview (what training sees)", lines=14, buttons=["copy"])

            nav_out = [e_id, e_raw, e_text, e_title, e_kind, e_artist, e_form, e_theme, e_note]
            nav_in = [e_id, e_scope, e_text, e_title, e_kind, e_artist, e_form, e_theme]
            e_load.click(lambda i, o: load_with_pos(i, o), [e_id, e_scope], nav_out)
            e_prevbtn.click(lambda *a: go(-1, *a), nav_in, nav_out)
            e_nextbtn.click(lambda *a: go(1, *a), nav_in, nav_out)
            e_save.click(save_item, [e_id, e_text, e_title, e_kind, e_artist, e_form, e_theme],
                         e_msg)
            e_ok.click(lambda i, t, k, ti, f: decide(i, True, text=t, kind=k, title=ti, force=f),
                       [e_id, e_text, e_kind, e_title, e_force], [e_msg, stats])
            e_no.click(lambda i: decide(i, False), e_id, [e_msg, stats])
            e_del.click(delete_current, [e_id, e_scope, e_delok], nav_out + [stats])
            e_prev.click(preview_tags,
                         [e_id, e_text, e_title, e_kind, e_artist, e_form, e_theme], e_tagged)
            e_conv.click(phon_convert, e_roman, [e_guru, e_alts])
            e_roman.submit(phon_convert, e_roman, [e_guru, e_alts])
            e_append.click(append_to_text, [e_text, e_guru], e_text)
            e_dofr.click(find_replace, [e_text, e_find, e_repl, e_all], [e_text, e_msg])

        # Library row click -> load that item straight into the Editor above.
        table.select(open_from_table, [f_state, f_folder, f_kind], e_id).then(
            lambda i, o: load_with_pos(i, o), [e_id, e_scope], nav_out)

        with gr.Tab("Text import"):
            gr.Markdown("Paste songs, qissa or stories. They join the same Editor queue as "
                        "transcribed audio.\n\n"
                        "**Long stories: paste the whole thing as one item — no need to split.** "
                        "The Dataset builder automatically breaks a story longer than the training "
                        "window into a *continuation chain* (part 1 from the instruction, each later "
                        "part from a “continue this story…” prompt), so nothing is truncated and the "
                        "narrative flow is preserved.")
            with gr.Row():
                i_title = gr.Textbox(label="title")
                i_kind = gr.Dropdown(KINDS, value="story", label="type")
                i_artist = gr.Textbox(label="artist / poet")
                i_form = gr.Textbox(label="form")
                i_theme = gr.Textbox(label="theme")
            with gr.Accordion("Scanned PDF → text (phone-scanned book, Claude-vision OCR)",
                              open=False):
                gr.Markdown(
                    "Drop phone-scanned PDF(s). Each page is read by Claude vision (your own "
                    "scan, honest transcription — no jailbreak) and streamed into the box below, "
                    "page by page, separated by `----- file pN -----`. It's a **rough draft**: "
                    "fix it here or in the Editor before Import. A `[REFUSED-copyright]` page is "
                    "left blank for you to type by hand.")
                with gr.Row():
                    i_pdf = gr.File(file_count="multiple", file_types=[".pdf"],
                                    label="drop scanned PDF(s)")
                    i_pdf_model = gr.Textbox(CFG["claude_model"], label="vision model")
                    i_pdf_first = gr.Number(1, label="first page", precision=0)
                    i_pdf_last = gr.Number(0, label="last page (0 = end)", precision=0)
                i_pdf_btn = gr.Button("Convert PDF → text")
            i_text = gr.Textbox(label="text", lines=18)
            with gr.Row():
                i_btn = gr.Button("Import", variant="primary")
                i_open = gr.Button("↳ Open in Editor")
            i_msg = gr.Textbox(label="", lines=1)
            i_pdf_btn.click(do_pdf, [i_pdf, i_pdf_model, i_pdf_first, i_pdf_last, i_text],
                            [i_msg, i_text])
            # Import, then load the new item into the Editor above so it's ready to review.
            i_btn.click(do_import, [i_title, i_text, i_kind, i_artist, i_form, i_theme],
                        [i_msg, stats, e_id]).then(
                        lambda i, o: load_with_pos(i, o) if i else (gr.skip(),) * 9,
                        [e_id, e_scope], nav_out)
            # One-click jump: reload the just-imported item and switch to the Editor tab.
            i_open.click(lambda i, s: load_with_pos(i, s), [e_id, e_scope], nav_out).then(
                        lambda: gr.Tabs(selected="editor"), None, main_tabs)

        with gr.Tab("Dataset"):
            gr.Markdown(
                "Builds training examples from **approved items only**. Reverse-instruction: "
                "Claude reads each finished song and writes the prompt that would have "
                "produced it — real Punjabi output, synthetic prompt. Runs on CPU/API, so it "
                "is safe while transcription holds the GPU.")
            with gr.Row():
                d_ver = gr.Textbox("v1", label="version name")
                d_model = gr.Textbox(CFG["claude_model"], label="metadata model")
                d_eval = gr.Number(0.1, label="eval fraction", precision=2)
            with gr.Row():
                d_ins = gr.Checkbox(True, label="instruction → song")
                d_theme = gr.Checkbox(True, label="theme → song")
                d_cont = gr.Checkbox(True, label="opening → continuation")
                d_maxcont = gr.Number(6, label="max continue / story (0 = all)", precision=0)
            gr.Markdown("*A long story yields many “continue” fragments. Capping them per story "
                        "keeps whole-story examples from being drowned out — the fix for the model "
                        "learning to write fragments instead of full stories. 6 is a good default; "
                        "0 = uncapped.*")
            with gr.Row():
                d_build = gr.Button("Build dataset", variant="primary")
                d_stop = gr.Button("Stop")
                d_poll = gr.Button("Refresh")
            d_status = gr.Textbox(label="status", value="idle")
            d_log = gr.Textbox(label="log", lines=12, max_lines=12)
            with gr.Row():
                d_pick = gr.Dropdown(dataset.list_versions(CFG), label="existing datasets")
                d_show = gr.Button("Show stats + preview")
            d_stats = gr.Markdown()
            d_prev = gr.Markdown()

            d_build.click(start_dataset,
                          [d_ver, d_model, d_ins, d_theme, d_cont, d_eval, d_maxcont],
                          d_status)
            d_stop.click(lambda: jobs.MANAGER.stop("dataset"), outputs=d_status)
            d_poll.click(poll_dataset, outputs=[d_status, d_log, d_pick])
            gr.Timer(3.0).tick(poll_dataset, outputs=[d_status, d_log, d_pick])
            d_show.click(lambda v: (dataset.stats(CFG, v), dataset.preview(CFG, v)),
                         d_pick, [d_stats, d_prev])

        with gr.Tab("Train"):
            gr.Markdown(
                f"QLoRA on **{CFG['base_model']}** — seq {CFG['seq_len']}, r{CFG['lora_r']}, "
                f"chunked CE {CFG['ce_chunk']}. Measured at **7.29 GB / 2.93 s per step** on "
                "this card (see `STATUS.md`). Training and transcription cannot share the GPU.")
            tr_gpu = gr.Markdown(gpu_line())
            with gr.Row():
                tr_ds = gr.Dropdown(dataset.list_versions(CFG), label="dataset")
                tr_ep = gr.Number(3, label="epochs", precision=0)
                tr_lr = gr.Number(1e-4, label="learning rate")
                tr_out = gr.Textbox("", label="adapter name (blank = auto)")
                tr_max = gr.Number(0, label="max steps (0 = all)", precision=0)
            with gr.Row():
                tr_start = gr.Button("Start training", variant="primary")
                tr_stop = gr.Button("Stop")
                tr_poll = gr.Button("Refresh")
            tr_status = gr.Textbox(label="status", value="idle")
            tr_log = gr.Textbox(label="live log", lines=18, max_lines=18)

            tr_start.click(start_train, [tr_ds, tr_ep, tr_lr, tr_out, tr_max],
                           [tr_status, tr_gpu])
            tr_stop.click(lambda: jobs.MANAGER.stop("train"), outputs=tr_status)
            tr_poll.click(poll_train, outputs=[tr_status, tr_log, tr_gpu])
            gr.Timer(3.0).tick(poll_train, outputs=[tr_status, tr_log, tr_gpu])

        with gr.Tab("Test"):
            gr.Markdown(
                "Base vs. tuned on the same prompt — the only honest way to see if training "
                "helped. Needs the GPU, so stop transcription/training first.")
            tst_gpu = gr.Markdown(gpu_line())
            with gr.Row():
                tst_adapter = gr.Dropdown(infer.list_adapters(CFG), label="tuned adapter")
                tst_refresh = gr.Button("↻ adapters")
            tst_prompt = gr.Textbox(
                "ਮਾਨਕ ਦੇ ਅੰਦਾਜ਼ ਵਿੱਚ ਵਿਛੋੜੇ ਦੀ ਕਲੀ ਲਿਖੋ",
                label="prompt", lines=2)
            with gr.Row():
                tst_temp = gr.Slider(0.0, 1.5, 0.9, label="temperature")
                tst_topp = gr.Slider(0.1, 1.0, 0.95, label="top-p")
                tst_max = gr.Number(400, label="max new tokens", precision=0)
            tst_go = gr.Button("Generate base vs tuned", variant="primary")
            with gr.Row():
                tst_base = gr.Textbox(label="BASE (untuned)", lines=20)
                tst_tuned = gr.Textbox(label="TUNED", lines=20)

            def run_compare(adapter, prompt, temp, topp, mx):
                busy = jobs.MANAGER.gpu_busy()
                if busy:
                    return f"GPU busy with '{busy}'", "", gpu_line()
                try:
                    b, t = infer.compare(CFG, prompt, adapter or None, temp=temp,
                                         top_p=topp, max_new=int(mx))
                except Exception as e:
                    return f"ERROR: {type(e).__name__}: {e}", "", gpu_line()
                return b, t, gpu_line()

            tst_refresh.click(lambda: gr.update(choices=infer.list_adapters(CFG)),
                              outputs=tst_adapter)
            tst_go.click(run_compare, [tst_adapter, tst_prompt, tst_temp, tst_topp, tst_max],
                         [tst_base, tst_tuned, tst_gpu])

        with gr.Tab("Settings"):
            s_roots = gr.Textbox("\n".join(CFG["source_roots"]), label="source roots (one/line)",
                                 lines=3)
            s_repo = gr.Textbox(CFG["transcribe_repo"], label="Transcribe repo")
            with gr.Row():
                s_port = gr.Number(CFG["port"], label="port", precision=0)
                s_batch = gr.Number(CFG["batch_size"], label="default batch", precision=0)
                s_cmodel = gr.Textbox(CFG["claude_model"], label="claude model")
            gr.Markdown("**Training defaults** — measured on an RTX 2070 8GB. "
                        "See `STATUS.md` before changing.")
            with gr.Row():
                s_bmodel = gr.Textbox(CFG["base_model"], label="base model")
                s_seq = gr.Number(CFG["seq_len"], label="seq len", precision=0)
                s_r = gr.Number(CFG["lora_r"], label="lora r", precision=0)
                s_chunk = gr.Number(CFG["ce_chunk"], label="CE chunk", precision=0)
            s_btn = gr.Button("Save settings", variant="primary")
            s_msg = gr.Textbox(label="", lines=1)
            s_btn.click(save_settings,
                        [s_roots, s_repo, s_port, s_batch, s_cmodel, s_bmodel, s_seq, s_r,
                         s_chunk], s_msg)


if __name__ == "__main__":
    # Gradio 6 moved `theme` from the Blocks constructor to launch().
    demo.launch(server_name="0.0.0.0", server_port=CFG["port"], theme=gr.themes.Soft())
