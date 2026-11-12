"""Build model inputs (full / partial context) and the long-format clinician reference table.

Outputs in the data directory (see paths.py):
    inputs.jsonl      one row per (corpus, transcript_id, condition): text
    refs.csv          corpus, transcript_id, instrument, item, ref   (--refs-only rebuilds just this file)
    speakers.csv      primary-corpus speaker -> role assignment, for checking
    transcripts_txt/  <transcript_id>.txt full-context text, for the verbatim-quote check in analysis/evaluate.py

    python scoring/prepare.py
"""
import json
import os
import re

import pandas as pd

from paths import AIIMS as AI, DATA as OUT, EDAIC as ED

PARTIAL = 0.40
SHEETS = {"PHQ-9": ("PHQ9.csv", 9), "HAM-D": ("HAMD.csv", 17), "HAM-A": ("HAMA.csv", 14), "MADRS": ("MADRS.csv", 10)}
PHQ_TEXT = {"not at all": 0, "several days": 1, "more than half the days": 2, "nearly every day": 3}


def score(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in PHQ_TEXT:
        return PHQ_TEXT[s]
    m = re.match(r"\d+", s)
    return int(m.group()) if m else None


def refs_aiims():
    rows = []
    for instr, (f, k) in SHEETS.items():
        d = pd.read_csv(os.path.join(AI, f)).drop(columns=["Patient Name"], errors="ignore").dropna(subset=["Patient ID"])
        d["Patient ID"] = d["Patient ID"].str.strip()
        cols = d.columns[1:k + 1]
        d["n"] = d[cols].notna().sum(1)
        # duplicated IDs: keep the most complete row
        d = d.sort_values("n", ascending=False).drop_duplicates("Patient ID")
        for _, r in d.iterrows():
            vals = [score(r[c]) for c in cols]
            if all(v is None for v in vals):
                continue
            for i, v in enumerate(vals, 1):
                rows.append(dict(corpus="AIIMS", transcript_id=r["Patient ID"], instrument=instr, item=i, ref=v))
    return rows


def refs_edaic():
    # detailed_lables.csv carries item-level PHQ-8 for all 275 sessions, including the test partition
    d = pd.read_csv(os.path.join(ED, "labels", "detailed_lables.csv"))
    cols = [c for c in d.columns if c.startswith("PHQ8_")][:8]
    rows = []
    for _, r in d.iterrows():
        for i, c in enumerate(cols, 1):
            rows.append(dict(corpus="E-DAIC", transcript_id=str(r.Participant), instrument="PHQ-8", item=i, ref=int(r[c])))
    return rows


def clinician(t):
    """The interviewer is taken to be the speaker with the most question-mark utterances."""
    q = t[t.transcript.astype(str).str.contains(r"\?")].speaker.value_counts()
    return q.index[0] if len(q) else t.speaker.iloc[0]


def aiims_inputs():
    rows, spk = [], []
    for d in sorted(x for x in os.listdir(AI) if x.endswith("-P")):
        t = pd.read_csv(os.path.join(AI, d, f"{d}_transcript.csv")).dropna(subset=["transcript"])
        c = clinician(t)
        t["role"] = ["CLINICIAN" if s == c else "PATIENT" for s in t.speaker]
        spk.append(dict(transcript_id=d[:-2], clinician=c, n_speakers=t.speaker.nunique(),
                        clin_utts=(t.role == "CLINICIAN").sum(), pat_utts=(t.role == "PATIENT").sum()))
        lines = [f"{r}: {x.strip()}" for r, x in zip(t.role, t.transcript.astype(str))]
        rows.append(dict(corpus="AIIMS", transcript_id=d[:-2], condition="full", text="\n".join(lines)))
        # partial: keep the dialogue up to and including the first 40% of patient utterances
        pat = [i for i, r in enumerate(t.role) if r == "PATIENT"]
        cut = pat[max(int(round(PARTIAL * len(pat))) - 1, 0)] + 1 if pat else len(lines)
        rows.append(dict(corpus="AIIMS", transcript_id=d[:-2], condition="partial", text="\n".join(lines[:cut])))
    return rows, spk


def edaic_inputs():
    rows = []
    split = {}
    for s in ("train", "dev", "test"):
        for p in pd.read_csv(os.path.join(ED, "labels", f"{s}_split.csv")).Participant_ID:
            split[str(p)] = s
    tdir = os.path.join(ED, "transcripts")
    for f in sorted(os.listdir(tdir)):
        if not f.endswith("_Transcript.csv"):
            continue
        pid = f.split("_")[0]
        t = pd.read_csv(os.path.join(tdir, f)).dropna(subset=["Text"])
        lines = [f"PARTICIPANT: {x.strip()}" for x in t.Text.astype(str)]
        n = max(int(round(PARTIAL * len(lines))), 1)
        for cond, ls in (("full", lines), ("partial", lines[:n])):
            rows.append(dict(corpus="E-DAIC", transcript_id=pid, condition=cond, split=split.get(pid), text="\n".join(ls)))
    return rows


if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    if "--refs-only" in sys.argv:  # leave inputs.jsonl (and any masked rows) untouched
        pd.DataFrame(refs_aiims() + refs_edaic()).to_csv(os.path.join(OUT, "refs.csv"), index=False)
        sys.exit()
    ai, spk = aiims_inputs()
    rows = ai + edaic_inputs()
    with open(os.path.join(OUT, "inputs.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tdir = os.path.join(OUT, "transcripts_txt")
    os.makedirs(tdir, exist_ok=True)
    for r in rows:
        if r["condition"] == "full":
            with open(os.path.join(tdir, f"{r['transcript_id']}.txt"), "w", encoding="utf-8") as f:
                f.write(r["text"])
    pd.DataFrame(spk).to_csv(os.path.join(OUT, "speakers.csv"), index=False)
    refs = pd.DataFrame(refs_aiims() + refs_edaic())
    refs.to_csv(os.path.join(OUT, "refs.csv"), index=False)
    e = pd.DataFrame(rows)
    print(e.groupby(["corpus", "condition"]).size())
    print(refs.dropna().groupby(["corpus", "instrument"]).transcript_id.nunique())
