# Punjabi LM — your walkthrough

**Start the app:**
```bash
cd "/mnt/e/projects/Punjabi LM" && ./run.sh
```
Then open **http://localhost:7861** in your Windows browser.

Overnight transcription (survives closing everything):
```bash
./overnight.sh --folder "Mohd Sadiq" --batch 10 --cooldown 180
tail -f data/overnight.log     # watch
touch data/STOP                # stop cleanly
```

---

## The flow — 6 steps

**1. Library** — *Scan library* finds all audio. Tag each folder's language (keep Hindi/Urdu out).

**2. Transcribe** — pick folder → *List pending tracks* → *Tick first N* (or hand-pick) → *Start batch*.
Stop is safe; finished tracks stay done. ~3.5 min/track. Can't run while training.

**3. Editor** ← *the real work.* Load an item. Raw ASR (both views) on the left is your evidence;
fix the final text on the right. Set title/artist/form/theme. **Approve** or **Reject**.
- Only *approved* songs reach a dataset.
- Garbage is blocked at approval (tick *override* if the checker is wrong).
- `REFUSED-copyright` items have empty text — hand-edit from the raw ASR, or reject.

**4. Dataset** — name a version → *Build dataset*. Turns approved songs into training pairs
(instruction→song, theme→song, opening→continuation). Runs on API, safe while transcribing.

**5. Train** — pick dataset → *Start training*. QLoRA, ~2.5 h. Needs the GPU free.

**6. Test** — pick adapter → *Generate base vs tuned*. Side-by-side proof it learned.

---

## Current state (2026-07-24)

- Manak Kuldeep done: **98 usable songs**, 25 copyright-refused, 13 rejected.
- Nothing approved yet — **your review (step 3) is next.**
- Other folders not transcribed: Mohd Sadiq (190), Didar Sandhu (150).

---

## The faster pipeline idea (not built yet)

**The problem:** `pipeline.sh` is invoked **once per track**, so Demucs + IndicConformer
**reload from scratch every single track**. GPU sat at 1–6% util — the ~3.5 min/track is almost
all model *loading*, not inference. This is the cold-start you noticed.

**The fix:** load the models **once per batch** and stream all tracks through them.
Expected ~30–40% faster (≈2.2 min/track), turning a 20 h folder run into ~13 h.

**Why not done yet:** it means modifying the Transcribe pipeline to keep models resident, and
testing it needs the GPU — which was busy all night. Worth doing before the big folders
(Mohd Sadiq + Didar Sandhu = ~340 tracks). Ask me to build it when ready.
