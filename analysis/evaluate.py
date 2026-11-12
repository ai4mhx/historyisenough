"""Evaluation: agreement with CIs, ICC, Bland-Altman, severity bands, evidence tags, calibration, safety.

Input: one long-format CSV, one row per (transcript, instrument, item, condition):

    transcript_id, instrument, item, condition, pred, ref, tag, confidence
    optional: rater2      second clinician's item score (human-human agreement)
              masked_item 1 if this item's evidence was masked in that transcript
              rationale   model rationale text (quote-hallucination check)

instrument in {PHQ-9, HAM-D, HAM-A, MADRS, PHQ-8}; condition in {full, masked, partial};
tag in {E, I, A}. For E-DAIC use instrument PHQ-8 and condition full.

Optional: --covariates CSV with transcript_id plus subgroup columns (diagnosis, gender, ...),
          --transcripts DIR with <transcript_id>.txt for the quote check.

Usage:
    python evaluate.py predictions.csv --out results/ [--weights linear] [--bins 10]
    python evaluate.py --demo --out demo_results/      # synthetic data smoke test
"""
import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score, roc_auc_score

ITEM_MAX = {"PHQ-9": 3, "PHQ-8": 3, "HAM-A": 4, "MADRS": 6, "HAM-D": 4}
# HAM-D items 4-6, 12-14, 16-17 are scored 0-2; the rest 0-4.
HAMD_02 = {4, 5, 6, 12, 13, 14, 16, 17}
# Lower bounds of severity bands above the lowest band.
BANDS = {"PHQ-9": [5, 10, 15, 20], "PHQ-8": [5, 10, 15, 20], "HAM-D": [8, 17, 24],
         "MADRS": [7, 20, 35], "HAM-A": [18, 25, 31]}
SUICIDE_ITEMS = {("PHQ-9", 9), ("HAM-D", 3), ("MADRS", 10)}
B = 2000
RNG = np.random.default_rng(2026)


def item_max(instr, item):
    if instr == "HAM-D" and int(item) in HAMD_02:
        return 2
    return ITEM_MAX[instr]


def wkappa(a, b, labels, weights):
    if len(set(a)) < 2 and len(set(b)) < 2:
        return np.nan
    return cohen_kappa_score(a, b, labels=labels, weights=weights)


def icc21(x, y):
    """Two-way random effects, absolute agreement, single measure (Shrout & Fleiss ICC(2,1))."""
    m = np.column_stack([x, y]).astype(float)
    n, k = m.shape
    gm = m.mean()
    msr = k * ((m.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((m.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((m - m.mean(1, keepdims=True) - m.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def resample(df):
    """One cluster-bootstrap replicate: whole transcripts drawn with replacement, relabelled as distinct."""
    groups = df.attrs.get("_groups")
    if groups is None:
        groups = [np.asarray(ix) for ix in df.groupby("transcript_id", sort=False).indices.values()]
        df.attrs["_groups"] = groups
    pick = RNG.integers(0, len(groups), len(groups))
    idx = np.concatenate([groups[i] for i in pick])
    lab = np.concatenate([np.full(len(groups[i]), k) for k, i in enumerate(pick)])
    out = df.iloc[idx].copy()
    out.attrs = {}
    out["transcript_id"] = lab
    return out


def cluster_boot(df, fn, b=None):
    """Percentile CI from resampling whole transcripts. fn(df) -> float."""
    b = b or B
    df = df.copy()
    est = fn(df)
    reps = []
    for _ in range(b):
        try:
            reps.append(fn(resample(df)))
        except ValueError:
            continue
    reps = np.array([r for r in reps if np.isfinite(r)])
    lo, hi = np.percentile(reps, [2.5, 97.5]) if len(reps) else (np.nan, np.nan)
    return est, lo, hi, reps


def fmt(est, lo, hi, d=2):
    return f"{est:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


def totals(df):
    return df.groupby(["transcript_id", "instrument", "condition"], as_index=False)[["pred", "ref"]].sum()


# ------------------------------------------------------------------ item level (Fig. 2, Supp Table 1)
def item_table(df, weights):
    rows = []
    for (instr, item, cond), g in df.groupby(["instrument", "item", "condition"]):
        labels = list(range(item_max(instr, item) + 1))
        f = lambda s: wkappa(s.ref, s.pred, labels, weights)
        k, lo, hi, _ = cluster_boot(g, f)
        row = dict(instrument=instr, item=item, condition=cond, n=len(g), kappa=k, ci_lo=lo, ci_hi=hi,
                   exact_acc=(g.pred == g.ref).mean(), within1=((g.pred - g.ref).abs() <= 1).mean())
        if "rater2" in g and g.rater2.notna().sum() >= 5:
            h = g.dropna(subset=["rater2"])
            row["human_kappa"] = wkappa(h.ref.astype(int), h.rater2.astype(int), labels, weights)
            row["human_n"] = len(h)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["condition", "instrument", "item"])


# ------------------------------------------------------------------ scale level (Table 2)
def scale_table(df, weights):
    rows = []
    for (instr, cond), g in df.groupby(["instrument", "condition"]):
        t = totals(g)
        stats_fns = {
            "pearson_r": lambda s: stats.pearsonr(*_tot(s))[0],
            "spearman_rho": lambda s: stats.spearmanr(*_tot(s))[0],
            "icc21": lambda s: icc21(*_tot(s)),
            "mae": lambda s: np.mean(np.abs(np.subtract(*_tot(s)))),
            "rmse": lambda s: np.sqrt(np.mean(np.subtract(*_tot(s)) ** 2)),
            "mean_item_kappa": lambda s: np.nanmean([
                wkappa(h.ref, h.pred, list(range(item_max(instr, it) + 1)), weights)
                for it, h in s.groupby("item")]),
        }
        row = dict(instrument=instr, condition=cond, n=t.transcript_id.nunique())
        for name, fn in stats_fns.items():
            e, lo, hi, _ = cluster_boot(g, fn, b=None if name != "mean_item_kappa" else min(B, 500))
            row[name] = fmt(e, lo, hi)
        d = t.pred - t.ref
        row["ba_bias"] = round(d.mean(), 2)
        row["ba_loa"] = f"{d.mean() - 1.96 * d.std(ddof=1):.2f} to {d.mean() + 1.96 * d.std(ddof=1):.2f}"
        # Proportional bias: does under-scoring grow with severity?
        slope = stats.linregress((t.pred + t.ref) / 2, d)
        row["ba_trend_slope"] = round(slope.slope, 3)
        row["ba_trend_p"] = slope.pvalue
        row["baseline_mean_mae"] = round(_mean_baseline_mae(g), 2)
        if instr in BANDS:
            cut = BANDS[instr]
            bp, br = np.digitize(t.pred, cut), np.digitize(t.ref, cut)
            row["band_exact"] = round((bp == br).mean(), 3)
            row["band_wkappa"] = round(wkappa(br, bp, list(range(len(cut) + 1)), weights), 3)
        if instr == "PHQ-8":
            y = (t.ref >= 10).astype(int)
            if y.nunique() == 2:
                row["auc"] = fmt(*cluster_boot(g, lambda s: roc_auc_score(*_auc(s)))[:3])
                pos = t.pred >= 10
                row["sensitivity"] = round((pos & (y == 1)).sum() / (y == 1).sum(), 3)
                row["specificity"] = round((~pos & (y == 0)).sum() / (y == 0).sum(), 3)
        rows.append(row)
    return pd.DataFrame(rows)


def _tot(s):
    t = s.groupby("transcript_id")[["pred", "ref"]].sum()
    return t.pred.values, t.ref.values


def _auc(s):
    t = s.groupby("transcript_id")[["pred", "ref"]].sum()
    return (t.ref >= 10).astype(int), t.pred


def _mean_baseline_mae(g):
    """Trivial baseline: each item predicted by its leave-one-transcript-out mean."""
    errs = []
    for tid, h in g.groupby("transcript_id"):
        rest = g[g.transcript_id != tid].groupby("item").ref.mean()
        errs.append(abs(h.item.map(rest).sum() - h.ref.sum()))
    return float(np.mean(errs))


# ------------------------------------------------------------------ degradation test
def degradation_tests(df):
    rows = []
    instrs = sorted(df.instrument.unique())
    t = totals(df)

    def drop(tt, instr):
        f = tt[(tt.instrument == instr) & (tt.condition == "full")]
        p = tt[(tt.instrument == instr) & (tt.condition == "partial")]
        return stats.pearsonr(f.pred, f.ref)[0] - stats.pearsonr(p.pred, p.ref)[0]

    for i, a in enumerate(instrs):
        for b_ in instrs[i + 1:]:
            if not {"full", "partial"} <= set(t.condition):
                continue
            est = drop(t, a) - drop(t, b_)
            reps = []
            for _ in range(B):
                tt = resample(t)
                try:  # a resample can hold < 2 transcripts of an instrument (patients rated on different scales)
                    reps.append(drop(tt, a) - drop(tt, b_))
                except ValueError:
                    continue
            reps = np.array([r for r in reps if np.isfinite(r)])
            p = 2 * min((reps <= 0).mean(), (reps >= 0).mean())
            rows.append(dict(contrast=f"drop({a}) - drop({b_})", estimate=round(est, 3),
                             ci=f"{np.percentile(reps, 2.5):.3f} to {np.percentile(reps, 97.5):.3f}",
                             p_two_sided=round(max(p, 1 / B), 4)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ evidence tags
def tag_table(df):
    rows = []
    for (cond, tag), g in df.groupby(["condition", "tag"]):
        e, lo, hi, _ = cluster_boot(g, lambda s: (s.pred == s.ref).mean(), b=min(B, 1000))
        rows.append(dict(condition=cond, tag=tag, n=len(g), share=len(g) / (df.condition == cond).sum(),
                         exact_acc=fmt(e, lo, hi), mean_pred=g.pred.mean(), mean_ref=g.ref.mean()))
    out = pd.DataFrame(rows)
    if "masked_item" in df:
        m = df[df.masked_item == 1]
        full = df[df.condition == "full"].merge(
            m[["transcript_id", "instrument", "item"]].drop_duplicates(), on=["transcript_id", "instrument", "item"])
        mm = m[m.condition == "masked"]
        out = pd.concat([out, pd.DataFrame([
            dict(condition="masked items only: full", tag="all", n=len(full), exact_acc=(full.pred == full.ref).mean(),
                 share=np.nan),
            dict(condition="masked items only: masked", tag="all", n=len(mm), exact_acc=(mm.pred == mm.ref).mean(),
                 share=np.nan),
            *[dict(condition="masked items only: masked", tag=t, n=(mm.tag == t).sum(), share=(mm.tag == t).mean())
              for t in "EIA"]])])
    return out


def quote_check(df, tdir):
    """Share of E-tagged rationales whose quoted spans (single or double quotes) occur verbatim in the transcript."""
    norm = lambda s: re.sub(r"\W+", " ", str(s).lower()).strip()
    hits, total = 0, 0
    for _, r in df[(df.tag == "E") & df.rationale.notna()].iterrows():
        path = os.path.join(tdir, f"{r.transcript_id}.txt")
        if not os.path.exists(path):
            continue
        text = norm(open(path, encoding="utf-8").read())
        for qt in re.findall(r"(?:^|\s)[\"“‘']([^\"“”‘’]{3,}?)[\"”’'](?=[\s,.;:)]|$)", str(r.rationale)):
            total += 1
            hits += norm(qt) in text
    return pd.DataFrame([dict(quotes_checked=total, verbatim=hits, verbatim_rate=hits / total if total else np.nan)])


# ------------------------------------------------------------------ calibration
def ece(s, bins):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(s.confidence, edges[1:-1]), 0, bins - 1)
    correct = (s.pred == s.ref).values
    out = 0.0
    for b_ in range(bins):
        m = idx == b_
        if m.any():
            out += m.mean() * abs(correct[m].mean() - s.confidence.values[m].mean())
    return out


def calibration(df, bins):
    rows, rel = [], []
    for cond, g in df.groupby("condition"):
        e, lo, hi, _ = cluster_boot(g, lambda s: ece(s, bins), b=min(B, 1000))
        rows.append(dict(condition=cond, instrument="pooled", ece=fmt(e, lo, hi, 3), bins=bins))
        for instr, h in g.groupby("instrument"):
            rows.append(dict(condition=cond, instrument=instr, ece=round(ece(h, bins), 3), bins=bins))
        edges = np.linspace(0, 1, bins + 1)
        idx = np.clip(np.digitize(g.confidence, edges[1:-1]), 0, bins - 1)
        for b_ in range(bins):
            m = idx == b_
            if m.any():
                rel.append(dict(condition=cond, bin=f"{edges[b_]:.1f}-{edges[b_ + 1]:.1f}", n=int(m.sum()),
                                share=m.mean(), mean_conf=g.confidence.values[m].mean(),
                                accuracy=(g.pred.values[m] == g.ref.values[m]).mean()))
    return pd.DataFrame(rows), pd.DataFrame(rel)


def risk_coverage(df):
    rows = []
    for cond, g in df.groupby("condition"):
        g = g.sort_values("confidence", ascending=False)
        correct = (g.pred == g.ref).values
        n = len(g)
        for cov in np.arange(0.1, 1.01, 0.1):
            k = max(1, int(round(cov * n)))
            rows.append(dict(condition=cond, coverage=round(cov, 1), accuracy=correct[:k].mean(),
                             conf_threshold=g.confidence.values[k - 1]))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ suicidality
def suicidality(df):
    rows = []
    for (instr, item) in SUICIDE_ITEMS:
        for cond, g in df[(df.instrument == instr) & (df.item == item)].groupby("condition"):
            pos = g[g.ref > 0]
            fn = (pos.pred == 0).sum()
            rows.append(dict(instrument=instr, item=item, condition=cond, n=len(g), n_ref_positive=len(pos),
                             false_negatives=int(fn), fnr=fn / len(pos) if len(pos) else np.nan,
                             under_scored=(g.pred < g.ref).mean(), over_scored=(g.pred > g.ref).mean(),
                             share_tag_A_or_I=g.tag.isin(["A", "I"]).mean(), mean_conf=g.confidence.mean()))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ subgroups
def subgroups(df, cov, weights):
    d = df[df.condition == "full"].merge(cov, on="transcript_id")
    rows = []
    for col in [c for c in cov.columns if c != "transcript_id"]:
        for (lvl, instr), g in d.groupby([col, "instrument"]):
            t = totals(g)
            if len(t) < 5:
                continue
            rows.append(dict(variable=col, level=lvl, instrument=instr, n=len(t),
                             pearson_r=round(stats.pearsonr(t.pred, t.ref)[0], 3),
                             mae=round((t.pred - t.ref).abs().mean(), 2)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def figures(out, item_df, rel, rc, df):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 600})
    mm = 1 / 25.4
    cols = {"full": "#1f4e79", "masked": "#3b7dbf", "partial": "#e07b39"}
    # Fig. 4: reliability by condition, risk-coverage, bin histogram (full)
    fig, ax = plt.subplots(1, 3, figsize=(180 * mm, 58 * mm))
    ax[0].plot([0, 1], [0, 1], ls="--", c="grey", lw=0.6)
    for cond, g in rel.groupby("condition"):
        ax[0].plot(g.mean_conf, g.accuracy, marker="o", ms=3, lw=0.9, c=cols.get(cond), label=f"{cond} context")
        rcg = rc[rc.condition == cond]
        ax[1].plot(rcg.coverage * 100, rcg.accuracy, marker="o", ms=3, lw=0.9, c=cols.get(cond), label=f"{cond} context")
    ax[0].set(xlabel="Mean stated confidence", ylabel="Exact-match accuracy", xlim=(0, 1), ylim=(0, 1))
    ax[1].set(xlabel="Coverage (% of items accepted, most confident first)", ylabel="Accuracy of accepted items")
    full = rel[rel.condition == "full"]
    ax[2].bar(range(len(full)), full.share * 100, color="#3b7dbf")
    ax[2].set_xticks(range(len(full)))
    ax[2].set_xticklabels(full.bin, rotation=45, ha="right")
    ax[2].set(xlabel="Confidence bin", ylabel="Share of item predictions (%)")
    ax[0].legend(frameon=False)
    for a, l in zip(ax, "abc"):
        a.text(-0.2, 1.05, l, transform=a.transAxes, fontweight="bold", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "Fig4_calibration.png"), bbox_inches="tight")
    fig.savefig(os.path.join(out, "Fig4_calibration.pdf"), bbox_inches="tight")
    plt.close(fig)
    # Supplementary: Bland-Altman per instrument (full context)
    t = totals(df[df.condition == "full"])
    instrs = sorted(t.instrument.unique())
    fig, ax = plt.subplots(1, len(instrs), figsize=(180 * mm, 50 * mm), squeeze=False)
    for a, instr in zip(ax[0], instrs):
        g = t[t.instrument == instr]
        m, d = (g.pred + g.ref) / 2, g.pred - g.ref
        a.scatter(m, d, s=8, c="#1f4e79", edgecolor="white", lw=0.3)
        for y, ls in ((d.mean(), "-"), (d.mean() + 1.96 * d.std(), "--"), (d.mean() - 1.96 * d.std(), "--")):
            a.axhline(y, ls=ls, c="#e07b39", lw=0.7)
        a.set(title=instr, xlabel="Mean of model and clinician total", ylabel="Model − clinician")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "SuppFig_bland_altman.png"), bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ demo data
def demo():
    rows = []
    n_items = {"PHQ-9": 9, "HAM-D": 17, "HAM-A": 14, "MADRS": 10}
    for t in range(46):
        sev = RNG.normal(0, 1)
        for instr, n in n_items.items():
            for it in range(1, n + 1):
                mx = item_max(instr, it)
                ref = int(np.clip(np.round(mx / 2 + sev * mx / 3 + RNG.normal(0, 0.6)), 0, mx))
                r2 = int(np.clip(ref + RNG.integers(-1, 2) * (RNG.random() < 0.3), 0, mx))
                masked = int(RNG.random() < 0.1)
                for cond, noise in (("full", 0.6), ("masked", 0.9), ("partial", 1.2)):
                    nz = noise * (1.8 if masked and cond == "masked" else 1)
                    pred = int(np.clip(np.round(ref + RNG.normal(0, nz)), 0, mx))
                    conf = float(np.clip(0.9 - 0.25 * abs(pred - ref) + RNG.normal(0, 0.12), 0.01, 0.99))
                    tag = RNG.choice(list("EIA"), p=[0.6, 0.3, 0.1] if cond == "full" else [0.3, 0.5, 0.2])
                    rows.append(dict(transcript_id=f"T{t:02d}", instrument=instr, item=it, condition=cond,
                                     pred=pred, ref=ref, tag=tag, confidence=conf, rater2=r2, masked_item=masked))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", nargs="?")
    ap.add_argument("--out", default="results")
    ap.add_argument("--weights", default="quadratic", choices=["linear", "quadratic"])
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--covariates")
    ap.add_argument("--transcripts")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()
    global B
    B = a.boot
    df = demo() if a.demo else pd.read_csv(a.csv)
    df["item"] = df["item"].astype(int)
    os.makedirs(a.out, exist_ok=True)
    w = lambda name, x: x.to_csv(os.path.join(a.out, name), index=False)

    items = item_table(df, a.weights); w("supp_table1_item_kappa.csv", items)
    w("table2_scale_level.csv", scale_table(df, a.weights))
    w("degradation_tests.csv", degradation_tests(df))
    w("evidence_tags.csv", tag_table(df))
    cal, rel = calibration(df, a.bins); w("calibration_ece.csv", cal); w("calibration_bins.csv", rel)
    rc = risk_coverage(df); w("risk_coverage.csv", rc)
    w("suicidality.csv", suicidality(df))
    if a.covariates:
        w("subgroups.csv", subgroups(df, pd.read_csv(a.covariates), a.weights))
    if a.transcripts and "rationale" in df:
        w("quote_check.csv", quote_check(df, a.transcripts))
    figures(a.out, items, rel, rc, df)
    print(f"Wrote results to {a.out}/ (kappa weights: {a.weights}, ECE bins: {a.bins}, bootstrap: {B})")


if __name__ == "__main__":
    main()
