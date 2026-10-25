"""Build report figures and LaTeX tables from results/. Run: python make_report.py"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG = HERE / "figures"
TAB = HERE / "tables"
FIG.mkdir(exist_ok=True)
TAB.mkdir(exist_ok=True)

BUDGETS = [6, 5, 4.5, 4, 3.5, 3]
BITS = [8, 6, 4, 3, 2]
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"  # dataviz slots 1-3
INK, INK2, AXIS, MUTED = "#0b0b0b", "#52514e", "#c3c2b7", "#898781"
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "pdf.fonttype": 42,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight",
})

# ---------------- load ----------------
sens = pd.read_csv(RES / "sens.csv")
res = pd.read_csv(RES / "results.csv")
base = res[res.method == "unquantized"].iloc[0]
BASE_SENS, BASE_TEST = float(base.ppl_sens), float(base.ppl_test)

rows = res[res.method != "unquantized"].copy()
rows["budget"] = rows.budget.astype(float)
below_bits = rows[rows.method == "uniform_below"].set_index("budget").avg_bits
# A uniform_above row that repeats uniform_below (same bit width) is a duplicate.
dup = rows.apply(
    lambda r: r.method == "uniform_above" and r.budget in below_bits.index
    and below_bits[r.budget] == r.avg_bits, axis=1)
clean = rows[~dup].copy()
assert len(clean) == 33, len(clean)

p8 = sens[sens.bits == 8].set_index("unit")
p8 = p8.assign(order=p8.layer * 2 + (p8.part == "mlp").astype(int)).sort_values("order")
ORDER = list(p8.index)
YLAB = [f"L{l} {p}" for l, p in zip(p8.layer, p8.part)]
PIV = sens.pivot_table(index="unit", columns="bits", values="delta_ppl")[BITS].loc[ORDER]
PART = p8.part.reindex(PIV.index)

greedy = clean[clean.method == "greedy"].sort_values("avg_bits")
uni = clean[clean.method.str.startswith("uniform")].drop_duplicates("avg_bits").sort_values("avg_bits")
rnd = clean[clean.method.str.startswith("random")]
rs = rnd.groupby("budget").agg(med=("ppl_test", "median"), lo=("ppl_test", "min"),
                               hi=("ppl_test", "max"), bits=("avg_bits", "mean"))


def load_alloc(b):
    return json.loads((RES / "allocation" / f"allocation_greedy_{b:g}.json").read_text())


# ---------------- figures ----------------
def fig_heatmap():
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, ax = plt.subplots(figsize=(3.4, 7.0))
    im = ax.imshow(np.clip(PIV.values, 0, 1.5), aspect="auto", cmap=cmap,
                   vmin=0, vmax=1.5, interpolation="nearest")
    ax.set_xticks(range(len(BITS)))
    ax.set_xticklabels(BITS)
    ax.set_yticks(range(len(YLAB)))
    ax.set_yticklabels(YLAB, fontsize=5)
    ax.set_xlabel("bits")
    cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cb.set_label("ΔPPL (clipped to [0, 1.5])")
    fig.savefig(FIG / "fig1_sensitivity_heatmap.pdf")
    plt.close(fig)


def fig_parts():
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = list(range(len(BITS)))
    for part, color, label in [("mixer", BLUE, "Mixer"), ("mlp", ORANGE, "MLP")]:
        sub = PIV[PART == part]
        for _, row in sub.iterrows():
            ax.plot(x, row.values, color=color, alpha=0.2, lw=0.7)
        ax.plot(x, sub.median().values, color=color, lw=2.2, marker="o", ms=3.5,
                label=f"{label}: median over units")
    ax.set_yscale("symlog", linthresh=0.01)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(BITS)
    ax.set_xlabel("bits")
    ax.set_ylabel("ΔPPL (symlog scale)")
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(FIG / "fig2_sensitivity_by_part.pdf")
    plt.close(fig)


def fig_ppl_bits():
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.plot(greedy.avg_bits, greedy.ppl_test, color=BLUE, marker="o", lw=1.6, label="Greedy")
    ax.plot(uni.avg_bits, uni.ppl_test, color=ORANGE, marker="s", lw=1.4, ls="--", label="Uniform")
    ax.errorbar(rs.bits, rs.med, yerr=[rs.med - rs.lo, rs.hi - rs.med], color=AQUA,
                marker="^", ls="none", capsize=3, label="Random: median, min to max (3 seeds)")
    ax.axhline(BASE_TEST, color=INK2, lw=0.9, ls=":")
    ax.text(3.05, BASE_TEST * 1.1, f"unquantized ({BASE_TEST:.2f})", ha="left", fontsize=7, color=INK2)
    ax.set_yscale("log")
    ax.set_xlabel("average bits per weight (decoder linears)")
    ax.set_ylabel("test perplexity (log scale)")
    ax.legend(frameon=False, fontsize=7, loc="upper center")
    fig.savefig(FIG / "fig3_ppl_vs_bits.pdf")
    plt.close(fig)


def fig_bits():
    levels = [2, 3, 4, 6, 8]
    cmap = ListedColormap(["#cde2fb", "#86b6ef", "#3987e5", "#184f95", "#0d366b"])
    mat = np.array([[load_alloc(b)[u] for u in ORDER] for b in BUDGETS], dtype=int)
    idx = np.vectorize(levels.index)(mat)
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.imshow(idx, aspect="auto", cmap=cmap, vmin=-0.5, vmax=4.5, interpolation="nearest")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=4.5,
                    color="white" if idx[i, j] >= 2 else INK)
    ax.set_yticks(range(len(BUDGETS)))
    ax.set_yticklabels([f"{b:g}" for b in BUDGETS])
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(YLAB, rotation=90, fontsize=4.5)
    ax.set_ylabel("budget (bits)")
    fig.savefig(FIG / "fig4_greedy_bits.pdf")
    plt.close(fig)


def fig_pred():
    pts = clean.assign(cat=clean.method.map(
        lambda m: "greedy" if m == "greedy" else ("uniform" if m.startswith("uniform") else "random")))
    pts = pts.drop_duplicates(subset=["cat", "avg_bits", "pred_ppl_sens", "ppl_sens"])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for cat, color, marker, label in [("greedy", BLUE, "o", "Greedy"), ("uniform", ORANGE, "s", "Uniform"),
                                      ("random", AQUA, "^", "Random (3 seeds)")]:
        s = pts[pts.cat == cat]
        ax.scatter(s.pred_ppl_sens, s.ppl_sens, color=color, marker=marker, s=22, label=label)
    lo, hi = 5, 1e7
    ax.plot([lo, hi], [lo, hi], ls="--", color=MUTED, lw=0.9, label="prediction = measurement")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("predicted sensitivity PPL (additive)")
    ax.set_ylabel("measured sensitivity PPL")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.savefig(FIG / "fig5_pred_vs_measured.pdf")
    plt.close(fig)


# ---------------- tables ----------------
def fmt(x):
    if abs(x) >= 1e6:
        return f"{x / 1e6:.2f} M"
    if abs(x) >= 1e3:
        return f"{x:,.0f}"
    return f"{x:.2f}"


def write_table(name, colspec, header, body):
    lines = [f"\\begin{{tabular}}{{{colspec}}}", r"\toprule"] + header + [r"\midrule"] + body + [r"\bottomrule", r"\end{tabular}"]
    (TAB / name).write_text("\n".join(lines) + "\n")


def tab_main():
    body = []
    for b in BUDGETS:
        g = clean.query("budget == @b and method == 'greedy'").iloc[0]
        ub = clean.query("budget == @b and method == 'uniform_below'").iloc[0]
        ua = clean.query("budget == @b and method == 'uniform_above'")
        ua_cells = (r"\multicolumn{2}{c}{omitted}" if ua.empty
                    else f"{ua.iloc[0].avg_bits:.0f} & {fmt(ua.iloc[0].ppl_test)}")
        body.append(f"{b:g} & {g.avg_bits:.2f} & {fmt(g.ppl_test)} & {ub.avg_bits:.0f} & "
                    f"{fmt(ub.ppl_test)} & {ua_cells} & {fmt(rs.loc[b, 'med'])} \\\\")
    header = [r"Budget $B$ & \multicolumn{2}{c}{Greedy} & \multicolumn{2}{c}{Uniform, $\lfloor B \rfloor$}"
              r" & \multicolumn{2}{c}{Uniform, $\lceil B \rceil$} & Random median \\",
              r" & bits & PPL & bits & PPL & bits & PPL & PPL \\"]
    write_table("tab_main.tex", "rrrrrrrr", header, body)


def tab_random():
    pr = rnd.pivot(index="budget", columns="method", values="ppl_test")
    body = []
    for b in BUDGETS:
        cells = " & ".join(fmt(pr.loc[b, f"random_seed_{s}"]) for s in (0, 1, 2))
        body.append(f"{b:g} & {rs.loc[b, 'bits']:.2f} & {cells} & {fmt(rs.loc[b, 'med'])} \\\\")
    header = [r"Budget $B$ & mean bits & seed 0 & seed 1 & seed 2 & median \\"]
    write_table("tab_random.tex", "rrrrrr", header, body)


def tab_shares():
    params = p8.params
    body = []
    for b in BUDGETS:
        a = pd.Series(load_alloc(b)).reindex(params.index)
        share = [params[a == k].sum() / params.sum() * 100 for k in (2, 3, 4, 6, 8)]
        avg = {}
        for part in ("mixer", "mlp"):
            m = PART.reindex(params.index) == part
            avg[part] = (a[m] * params[m]).sum() / params[m].sum()
        body.append(f"{b:g} & {avg['mixer']:.2f} & {avg['mlp']:.2f} & "
                    + " & ".join(f"{s:.1f}" for s in share) + r" \\")
    header = [r"Budget $B$ & mixer bits & MLP bits & 2-bit & 3-bit & 4-bit & 6-bit & 8-bit \\",
              r" & & & \multicolumn{5}{c}{share of parameters (\%)} \\"]
    write_table("tab_shares.tex", "rrrrrrrr", header, body)


def tab_outliers():
    body = []
    for b in (4, 3, 2):
        top = sens[sens.bits == b].nlargest(3, "delta_ppl")
        for _, r in top.iterrows():
            layer, part = r.unit.split("_")[1], r.unit.split("_")[2]
            body.append(f"{b} & L{layer} {part} & {r.params / 1e6:.1f} & {r.delta_ppl:.3f} \\\\")
        body.append(r"\midrule")
    body = body[:-1]
    header = [r"Bits & Unit & Parameters (M) & $\Delta$PPL \\"]
    write_table("tab_outliers.tex", "llrr", header, body)


# ---------------- numbers used in the text ----------------
def numbers():
    print(f"baseline sens {BASE_SENS:.6f}  test {BASE_TEST:.6f}  ratio {BASE_TEST / BASE_SENS:.3f}")
    print(f"clean method rows {len(clean)}; dropped duplicate uniform_above rows {int(dup.sum())}")

    total2 = sens[sens.bits == 2].delta_ppl.sum()
    top2 = sens[(sens.bits == 2)].nlargest(1, "delta_ppl").iloc[0]
    print(f"2-bit summed dPPL {total2:.1f}; largest unit {top2.unit} = {top2.delta_ppl / total2 * 100:.1f}%")

    drops = np.max(PIV.values[:, :-1] - PIV.values[:, 1:], axis=1)
    print("direction-correct: units where lowering bits lowers dPPL somewhere:", int((drops > 0).sum()),
          "of", len(drops))
    for t in (0.05, 0.1, 0.2):
        print(f"   units with a drop > {t}: {int((drops > t).sum())}")
    print("code-style non_monotone count (as logged):",
          sum(not (g.sort_values("bits")["delta_ppl"].to_numpy()[1:] >=
                   g.sort_values("bits")["delta_ppl"].to_numpy()[:-1]).all()
              for _, g in sens.groupby("unit")))

    print("\ngreedy vs uniform interpolation (log-linear in avg bits):")
    ub_bits, ub_log = uni.avg_bits.to_numpy(), np.log(uni.ppl_test.to_numpy())
    for _, g in greedy.iterrows():
        if ub_bits.min() <= g.avg_bits <= ub_bits.max():
            interp = float(np.exp(np.interp(g.avg_bits, ub_bits, ub_log)))
            print(f"  {g.avg_bits:.3f} bits: greedy {g.ppl_test:.3f}, interp {interp:.3f}, ratio {interp / g.ppl_test:.2f}")
    print("greedy test vs baseline (%):", {b: round((float(greedy[greedy.budget == b].ppl_test.iloc[0]) / BASE_TEST - 1) * 100, 1) for b in BUDGETS})

    print("\nprediction vs measurement (unique points):")
    pts = clean.drop_duplicates(subset=["method", "avg_bits", "pred_ppl_sens", "ppl_sens"])
    rho = pts[["pred_ppl_sens", "ppl_sens"]].rank().corr().iloc[0, 1]
    print(f"  Spearman over {len(pts)} unique points: {rho:.3f}")
    g_ratio = (greedy.ppl_sens / greedy.pred_ppl_sens).round(3).tolist()
    print("  greedy measured/predicted by budget", dict(zip(greedy.budget.tolist(), g_ratio)))
    r_ratio = (rnd.ppl_sens / rnd.pred_ppl_sens)
    print("  random measured/predicted range", round(r_ratio.min(), 2), round(r_ratio.max(), 2))
    print("  median |log10 ratio| avg_bits >= 3.9:",
          round(float(np.median(np.abs(np.log10(pts[pts.avg_bits >= 3.9].ppl_sens / pts[pts.avg_bits >= 3.9].pred_ppl_sens)))), 3))

    print("\nallocations with negative-cost downgrade steps on their path (correct path rule lo >= bits):")
    for b in BUDGETS:
        a = load_alloc(b)
        n_units, n_steps = 0, 0
        for u, bits in a.items():
            g_ = sens[sens.unit == u].set_index("bits").delta_ppl
            steps = [(hi, lo) for hi, lo in [(8, 6), (6, 4), (4, 3), (3, 2)] if lo >= bits]
            neg = [s for s in steps if g_[s[1]] - g_[s[0]] < 0]
            n_units += bool(neg)
            n_steps += len(neg)
        print(f"  budget {b:g}: units {n_units}, negative steps {n_steps}")


if __name__ == "__main__":
    fig_heatmap()
    fig_parts()
    fig_ppl_bits()
    fig_bits()
    fig_pred()
    tab_main()
    tab_random()
    tab_shares()
    tab_outliers()
    numbers()
