"""Filter a *_preds.csv to (transcript, instrument) pairs with complete clinician references.

    python scoring/to_analysis.py results/llama3b_AIIMS_preds.csv results/llama3b_AIIMS_eval.csv [--split test]
"""
import argparse

import pandas as pd

from instruments import items

ap = argparse.ArgumentParser()
ap.add_argument("src")
ap.add_argument("dst")
ap.add_argument("--split")
ap.add_argument("--min-n", type=int, default=5, help="drop instruments with fewer complete transcripts")
a = ap.parse_args()

p = pd.read_csv(a.src, dtype={"transcript_id": str})
if a.split:
    p = p[p.split.isin(a.split.split(","))]
n_items = {i: len(items(i)) for i in p.instrument.unique()}
# a pair is kept only if every item has a reference score and a prediction, in every condition
g = p.groupby(["transcript_id", "instrument", "condition"])
ok = g[["ref"]].apply(lambda x: x.ref.notna().sum() == n_items[x.name[1]] and len(x) == n_items[x.name[1]])
n_cond = p.condition.nunique()
ok = ok.groupby(level=[0, 1]).agg(lambda x: x.all() and len(x) == n_cond)
keep = ok[ok].index
out = p.set_index(["transcript_id", "instrument"]).loc[keep].reset_index()
n = out.groupby("instrument").transcript_id.nunique()
for i in n[n < a.min_n].index:
    print(f"dropping {i}: only {n[i]} transcript(s) with complete references")
out = out[out.instrument.isin(n[n >= a.min_n].index)]
out["ref"] = out["ref"].astype(int)
out.to_csv(a.dst, index=False)
print(out.groupby("instrument").transcript_id.nunique().to_string())
