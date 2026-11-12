"""Figs 1-3 of the paper in npj house style (Arial, 7-8 pt at 180 mm width, no red/green).

The plotted values are those reported for the domain-adapted study model. To draw bootstrap
95% CIs, pass the CI columns produced by analysis/evaluate.py into fig2() / fig3().
Fig. 4 (calibration) and the Bland-Altman plot are written by analysis/evaluate.py.

    python figures/make_figures.py      # writes Fig1-3 as .png and .pdf in the current directory
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 600, "pdf.fonttype": 42, "svg.fonttype": "none",
})
MM = 1 / 25.4
NAVY, BLUE, LIGHT, ORANGE, GREY = "#1f4e79", "#3b7dbf", "#a9c8e6", "#e07b39", "#8c8c8c"


def panel_label(ax, s, x=-0.02, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom", ha="right")


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- Fig. 1
def fig1():
    fig = plt.figure(figsize=(180 * MM, 72 * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 180); ax.set_ylim(0, 72); ax.axis("off")
    cols = [
        ("a", "Clinical corpora", [
            ("Primary corpus", "46 psychiatric history-taking\ntranscripts (AIIMS Delhi)\nHAM-D, HAM-A, MADRS, PHQ-9"),
            ("External validation corpus", "E-DAIC virtual-agent interviews\nofficial test partition (n = 56)\nself-reported PHQ-8"),
        ]),
        ("b", "Context conditions", [
            ("Full context", "complete transcript"),
            ("Masked context", "3–5 item-relevant spans\nreplaced by [MASKED]"),
            ("Partial context", "first 40% of patient\nutterances retained"),
        ]),
        ("c", "Item-level inference", [
            ("Domain-adapted LLaMA 3.2-3B", "history-taking model; no\nscoring-specific fine-tuning"),
            ("Instrument-specific prompt", "role, transcript, item anchors,\nJSON output specification"),
            ("Output per item", "score, evidence tag (E/I/A),\nconfidence, rationale"),
        ]),
        ("d", "Evaluation", [
            ("Item level", "weighted Cohen’s κ vs\nclinician reference"),
            ("Scale level", "Pearson r, Spearman ρ,\nMAE of total scores"),
            ("Reliability", "expected calibration error,\nreliability diagram, AUC-ROC"),
        ]),
    ]
    w, gap, x0 = 40.5, 5.5, 1.5
    for k, (lab, title, boxes) in enumerate(cols):
        x = x0 + k * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 1.5), w, 66, boxstyle="round,pad=0,rounding_size=2",
                                    fc="#f4f7fb", ec=GREY, lw=0.6))
        ax.text(x + 1.5, 69.5, lab, fontsize=9, fontweight="bold", va="center")
        ax.text(x + w / 2, 64, title, fontsize=7.5, fontweight="bold", ha="center", va="center")
        n = len(boxes); top, bottom = 60, 4
        h = (top - bottom - (n - 1) * 3) / n
        for j, (head, body) in enumerate(boxes):
            y = top - (j + 1) * h - j * 3
            fc = [LIGHT, "#d7e6f4", "#fbe3d3"][j] if k == 1 else "white"
            ax.add_patch(FancyBboxPatch((x + 2, y), w - 4, h, boxstyle="round,pad=0,rounding_size=1.2",
                                        fc=fc, ec=NAVY, lw=0.6))
            ax.text(x + w / 2, y + h * 0.70, head, fontsize=7, fontweight="bold", ha="center", va="center")
            ax.text(x + w / 2, y + h * 0.33, body, fontsize=6.3, ha="center", va="center", linespacing=1.25)
            if k == 2 and j < n - 1:
                ax.annotate("", (x + w / 2, y - 2.8), (x + w / 2, y), arrowprops=dict(arrowstyle="-|>", lw=0.7, color=NAVY))
        if k < 3:
            ax.annotate("", (x + w + gap - 0.3, 34), (x + w + 0.3, 34),
                        arrowprops=dict(arrowstyle="-|>", lw=0.9, color="black"))
    save(fig, "Fig1_study_design")


# ---------------------------------------------------------------- Fig. 2
PHQ9 = [("1 Anhedonia", .69), ("2 Depressed mood", .72), ("3 Sleep", .65), ("4 Energy/fatigue", .63),
        ("5 Appetite", .61), ("6 Guilt/worthlessness", .66), ("7 Concentration", .67),
        ("8 Psychomotor", .52), ("9 Self-harm", .43)]
HAMD = [("1 Depressed mood", .68), ("2 Guilt", .65), ("3 Suicide", .44), ("4 Insomnia (early)", .66),
        ("5 Insomnia (middle)", .63), ("6 Insomnia (late)", .62), ("7 Work/activities", .60),
        ("8 Retardation", .38), ("9 Agitation", .41)]
MADRS = [("1 Apparent sadness", .71), ("2 Reported sadness", .73), ("3 Inner tension", .74),
         ("4 Reduced sleep", .72), ("5 Reduced appetite", .68), ("6 Concentration", .67),
         ("7 Lassitude", .65), ("8 Inability to feel", .63), ("9 Pessimistic thoughts", .68),
         ("10 Suicidal thoughts", .45)]
BANDS = [(0.21, 0.40, "Fair"), (0.41, 0.60, "Moderate"), (0.61, 0.80, "Substantial")]


def band_colour(k):
    return ORANGE if k <= 0.40 else (BLUE if k <= 0.60 else NAVY)


def fig2(panels=(("PHQ-9", PHQ9), ("HAM-D (items 1–9)", HAMD), ("MADRS", MADRS)), ci=None):
    fig, axes = plt.subplots(1, len(panels), figsize=(180 * MM, 62 * MM))
    for i, (ax, (name, items)) in enumerate(zip(axes, panels)):
        labels = [a for a, _ in items][::-1]; vals = [b for _, b in items][::-1]
        for lo, hi, txt in BANDS:
            ax.axvspan(lo, hi, color="#eef2f7" if txt == "Moderate" else "white", zorder=0)
        for b in (0.40, 0.60, 0.80):
            ax.axvline(b + 0.005, color=GREY, lw=0.4, ls=(0, (2, 2)), zorder=1)
        ys = range(len(vals))
        if ci is not None and name in ci:
            lo, hi = zip(*ci[name][::-1])
            ax.hlines(ys, lo, hi, color="black", lw=0.7, zorder=2)
        ax.scatter(vals, ys, s=16, c=[band_colour(v) for v in vals], edgecolor="black", lw=0.4, zorder=3)
        for y, v in zip(ys, vals):
            ax.text(v + 0.025, y, f"{v:.2f}", va="center", fontsize=6)
        ax.set_yticks(list(ys)); ax.set_yticklabels(labels)
        ax.set_xlim(0.2, 0.9); ax.set_xticks([0.2, 0.4, 0.6, 0.8]); ax.set_ylim(-0.6, len(vals) - 0.4)
        ax.set_xlabel("Weighted Cohen’s κ"); ax.set_title(name, fontsize=7.5, fontweight="bold")
        panel_label(ax, "abc"[i], x=-0.55 if i == 0 else -0.5, y=1.06)
    handles = [plt.Line2D([], [], marker="o", ls="", mfc=c, mec="black", mew=0.4, ms=4.5, label=l)
               for c, l in ((ORANGE, "Fair (0.21\u20130.40)"), (BLUE, "Moderate (0.41\u20130.60)"),
                            (NAVY, "Substantial (0.61\u20130.80)"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(w_pad=1.2, rect=(0, 0.06, 1, 1))
    save(fig, "Fig2_item_kappa")


# ---------------------------------------------------------------- Fig. 3
R = {"PHQ-9": (.81, .68, .57), "MADRS": (.83, .71, .60), "HAM-D": (.76, .59, .47), "HAM-A": (.74, .56, .44)}
TAGS = {"Full context": (62.3, 28.4, 9.3), "Masked context": (26.2, 52.1, 21.7)}


def fig3(ci=None):
    fig, (a, b) = plt.subplots(1, 2, figsize=(180 * MM, 68 * MM), gridspec_kw={"width_ratios": [1.45, 1]})
    conds = ["Full", "Masked", "Partial"]; mk = {"PHQ-9": "o", "MADRS": "s", "HAM-D": "^", "HAM-A": "D"}
    cols = {"PHQ-9": NAVY, "MADRS": BLUE, "HAM-D": ORANGE, "HAM-A": "#9a5b2e"}
    for k, (name, v) in enumerate(R.items()):
        xs = [c + (k - 1.5) * 0.06 for c in range(3)]
        if ci is not None and name in ci:
            for x, (lo, hi) in zip(xs, ci[name]):
                a.vlines(x, lo, hi, color=cols[name], lw=0.7)
        a.plot(xs, v, marker=mk[name], ms=4, lw=0.9, color=cols[name], mec="black", mew=0.3,
               label=f"{name} (Δ = −{v[0] - v[2]:.2f})")
    a.set_xticks(range(3)); a.set_xticklabels([f"{c} context" for c in conds]); a.set_xlim(-0.35, 2.35)
    a.set_ylim(0.3, 0.9); a.set_ylabel("Pearson r, model vs clinician total score")
    a.legend(frameon=False, fontsize=6.3, loc="lower left", title="Δ, full → partial", title_fontsize=6.3)
    panel_label(a, "a", x=-0.12)
    bottom = [0, 0]; names = list(TAGS); colours = [NAVY, LIGHT, ORANGE]
    for t, (lab, col) in enumerate(zip(["Explicit", "Inferred", "Absent"], colours)):
        vals = [TAGS[n][t] for n in names]
        b.bar(names, vals, bottom=bottom, width=0.55, color=col, edgecolor="white", lw=0.5, label=lab)
        for x, (v, bt) in enumerate(zip(vals, bottom)):
            b.text(x, bt + v / 2, f"{v:.1f}%", ha="center", va="center", fontsize=6.3,
                   color="white" if col == NAVY else "black")
        bottom = [bt + v for bt, v in zip(bottom, vals)]
    b.set_ylim(0, 100); b.set_ylabel("Share of item predictions (%)")
    b.legend(frameon=False, fontsize=6.3, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    panel_label(b, "b", x=-0.18)
    fig.tight_layout(w_pad=2)
    save(fig, "Fig3_degradation_tags")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
