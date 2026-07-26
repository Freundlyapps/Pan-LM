#!/usr/bin/env python3
"""SQLite state — one row per item, tracked through the pipeline.

Why a database rather than files: the UI has to filter and sort a thousand items, and a
batch that dies halfway must resume without redoing finished work. Both are painful with
scattered text files.

Pipeline states:
    pending -> transcribed -> edited -> approved
                                     \\-> rejected
Only `approved` items may enter a dataset.
"""
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id          INTEGER PRIMARY KEY,
    path        TEXT UNIQUE,          -- source audio; NULL for pasted text
    title       TEXT NOT NULL,
    folder      TEXT,
    kind        TEXT DEFAULT 'song',  -- song | qissa | story
    state       TEXT DEFAULT 'pending',
    raw_text    TEXT,                 -- both ASR views, for the editor
    clean_text  TEXT,                 -- Claude reconstruction
    final_text  TEXT,                 -- what the human approved
    artist      TEXT, form TEXT, theme TEXT,
    duration    REAL,
    note        TEXT,
    updated_at  REAL
);
CREATE TABLE IF NOT EXISTS folders (
    path        TEXT PRIMARY KEY,
    lang_hint   TEXT DEFAULT 'unknown',   -- punjabi | hindi | urdu | mixed
    include     INTEGER DEFAULT 1
);
CREATE TABLE IF NOT EXISTS edits (
    id INTEGER PRIMARY KEY, item_id INTEGER, text TEXT, ts REAL
);
CREATE INDEX IF NOT EXISTS idx_items_state  ON items(state);
CREATE INDEX IF NOT EXISTS idx_items_folder ON items(folder);
"""

STATES = ["pending", "transcribed", "edited", "approved", "rejected"]


def connect(db_path):
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def upsert_item(con, path, title, folder, duration=None, kind="song"):
    """Insert if new; never clobber existing work on rescan."""
    con.execute(
        "INSERT INTO items (path, title, folder, duration, kind, updated_at) "
        "VALUES (?,?,?,?,?,?) ON CONFLICT(path) DO NOTHING",
        (str(path), title, folder, duration, kind, time.time()))
    con.commit()


def add_text_item(con, title, text, kind="story", **meta):
    cur = con.execute(
        "INSERT INTO items (path, title, folder, kind, state, clean_text, final_text, "
        "artist, form, theme, updated_at) VALUES (NULL,?,?,?,'transcribed',?,?,?,?,?,?)",
        (title, "(pasted)", kind, text, text, meta.get("artist"), meta.get("form"),
         meta.get("theme"), time.time()))
    con.commit()
    return cur.lastrowid


def set_fields(con, item_id, **fields):
    if not fields:
        return
    fields["updated_at"] = time.time()
    cols = ", ".join(f"{k}=?" for k in fields)
    con.execute(f"UPDATE items SET {cols} WHERE id=?", (*fields.values(), item_id))
    con.commit()


def save_edit(con, item_id, text):
    """Persist an edit and keep history, so a bad edit is recoverable."""
    con.execute("INSERT INTO edits (item_id, text, ts) VALUES (?,?,?)",
                (item_id, text, time.time()))
    set_fields(con, item_id, final_text=text, state="edited")


def delete_item(con, item_id):
    """Hard-delete an item and its edit history — for junk (e.g. an empty item approved by
    accident). An *audio* track re-appears as 'pending' on the next Library scan (the file is
    untouched); a pasted-text item is gone for good, so it's the real delete for imports."""
    con.execute("DELETE FROM edits WHERE item_id=?", (item_id,))
    con.execute("DELETE FROM items WHERE id=?", (item_id,))
    con.commit()


def reset_item(con, item_id):
    """Send an item back to the start of the pipeline: clear its approval/edit and mark it
    'pending' again. Right for an audio track that was mis-approved empty — it keeps the track
    (and its id) and re-enters the transcribe queue instead of being deleted."""
    con.execute("DELETE FROM edits WHERE item_id=?", (item_id,))
    set_fields(con, item_id, state="pending", raw_text=None, clean_text=None,
               final_text=None, note=None)


def get(con, item_id):
    return con.execute("SELECT * FROM items WHERE id=?", (item_id,)).fetchone()


def query(con, state=None, folder=None, kind=None, limit=5000):
    sql, args = "SELECT * FROM items WHERE 1=1", []
    if state and state != "all":
        sql, args = sql + " AND state=?", args + [state]
    if folder and folder != "all":
        sql, args = sql + " AND folder=?", args + [folder]
    if kind and kind != "all":
        sql, args = sql + " AND kind=?", args + [kind]
    sql += " ORDER BY folder, title LIMIT ?"
    return con.execute(sql, args + [limit]).fetchall()


def counts(con):
    rows = con.execute("SELECT state, COUNT(*) n FROM items GROUP BY state").fetchall()
    return {r["state"]: r["n"] for r in rows}


def folders(con):
    rows = con.execute(
        "SELECT folder, COUNT(*) n FROM items GROUP BY folder ORDER BY folder").fetchall()
    return [(r["folder"], r["n"]) for r in rows]


def set_folder_hint(con, path, lang_hint=None, include=None):
    con.execute("INSERT INTO folders (path) VALUES (?) ON CONFLICT(path) DO NOTHING", (path,))
    if lang_hint is not None:
        con.execute("UPDATE folders SET lang_hint=? WHERE path=?", (lang_hint, path))
    if include is not None:
        con.execute("UPDATE folders SET include=? WHERE path=?", (int(include), path))
    con.commit()


def folder_hints(con):
    return {r["path"]: dict(r) for r in con.execute("SELECT * FROM folders").fetchall()}


def pending_for_transcribe(con, folder, limit):
    """Next N untranscribed audio items — this is what makes batches resumable."""
    return con.execute(
        "SELECT * FROM items WHERE state='pending' AND path IS NOT NULL "
        "AND (? = 'all' OR folder = ?) ORDER BY folder, title LIMIT ?",
        (folder, folder, limit)).fetchall()
