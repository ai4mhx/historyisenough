"""Table 3: comparison with general-purpose models, from the per-model outputs of evaluate.py.

    python analysis/make_table3.py llama3b llama8b llama70b [--results results/]

Reads <results>/<model>_{AIIMS,EDAICtest}/table2_scale_level.csv, calibration_ece.csv and quote_check.csv;
writes <results>/table3_comparison.csv and table3_comparison.md, and prints a per-model summary.
"""
import argparse
import os

import pandas as pd

NAME = {"llama3b": "LLaMA 3.2-3B", "llama8b": "LLaMA 3.1-8B", "llama70b": "LLaMA 3.1-70B", "study": "Study model"}
ORDER = ["HAM-A", "HAM-D", "MADRS", "PHQ-9", "PHQ-8"]


def fmt(s):
    """'0.09 (-0.50–0.46)' -> '0.09 (−0.50 to 0.46)'."""
    s = str(s).replace("–", " to ").replace("-", "−")
    return s.replace("−0.00", "0.00")


def point(s):
    return str(s).split(" ")[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("--results", default=os.path.join(os.path.dirname(__file__), "..", "results"))
    a = ap.parse_args()

    rows, summary = [], []
    for m in a.models:
        for corpus in ("AIIMS", "EDAICtest"):
            d = os.path.join(a.results, f"{m}_{corpus}")
            t = pd.read_csv(os.path.join(d, "table2_scale_level.csv"))
            t["k"] = t.instrument.map(ORDER.index)
            t["c"] = t.condition.map({"full": 0, "masked": 1, "partial": 2})
            for _, r in t.sort_values(["k", "c"]).iterrows():
                label = f"E-DAIC {r.instrument}" if corpus == "EDAICtest" else r.instrument
                rows.append({"Model": NAME.get(m, m), "Instrument (n)": f"{label} ({int(r.n)})",
                             "Context": r.condition.capitalize(),
                             "Pearson r (95% CI)": fmt(r.pearson_r),
                             "Mean item κ (95% CI)": fmt(r.mean_item_kappa),
                             "MAE (mean predictor)": f"{float(point(r.mae)):.2f} ({float(r.baseline_mean_mae):.2f})"})
            e = pd.read_csv(os.path.join(d, "calibration_ece.csv"))
            e = e[(e.instrument == "pooled") & (e.condition == "full")]
            qc = os.path.join(d, "quote_check.csv")
            q = pd.read_csv(qc).verbatim_rate.iloc[0] if os.path.exists(qc) else float("nan")
            summary.append(dict(model=m, corpus=corpus, ece_full=point(e.ece.iloc[0]) if len(e) else "",
                                verbatim_quotes=f"{q:.0%}" if q == q else ""))
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(a.results, "table3_comparison.csv"), index=False)
    with open(os.path.join(a.results, "table3_comparison.md"), "w") as f:
        f.write("| " + " | ".join(out.columns) + " |\n|" + "---|" * len(out.columns) + "\n")
        for r in out.itertuples(index=False):
            f.write("| " + " | ".join(map(str, r)) + " |\n")
    print(out.to_string(index=False))
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
