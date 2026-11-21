"""All analysis tables and figures, built only from reports/results/. Run after the experiments:

python scripts/make_tables.py && python scripts/analyze.py

Writes reports/results/{length_bias,stress_test,llm_subset_qwk,feasibility}.csv and reports/figures/*.png.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aes.data import ROOT, SCORE_RANGES, load_asap  # noqa: E402
from aes.metrics import per_prompt_qwk  # noqa: E402
from aes.tracking import RESULTS_DIR  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
# Display names and fixed colors: a model keeps its color in every figure (validated categorical palette).
MODELS = {
    "length_only_ridge": ("Length only (Ridge)", "#eda100"),
    "features_ridge": ("Features (Ridge)", "#e87ba4"),
    "features_lgbm": ("Features (LightGBM)", "#e87ba4"),
    "lstm": ("LSTM", "#1baf7a"),
    "cnn_lstm": ("CNN-LSTM", "#1baf7a"),
    "bert_base": ("BERT-base", "#008300"),
    "deberta_v3_base": ("DeBERTa-v3-base", "#2a78d6"),
    "deberta_v3_base_thr": ("DeBERTa-v3-base + thresholds", "#2a78d6"),
    "llm_gemini_zero": ("Gemini 3.5 Flash-Lite zero-shot", "#eb6834"),
    "llm_gemini_few": ("Gemini 3.5 Flash-Lite few-shot", "#eb6834"),
}
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


def load(name, prefix="preds"):
    files = sorted((RESULTS_DIR / name).glob(f"{prefix}_fold*.csv"))
    return pd.concat(map(pd.read_csv, files)) if files else None


def available():
    return [m for m in MODELS if load(m) is not None]


def norm_err(df):
    rng = df.prompt.map(lambda p: SCORE_RANGES[p][1] - SCORE_RANGES[p][0])
    return (df.y_pred - df.y_true) / rng


def length_bias(models, words):
    """Per model: mean over prompts of corr(error, length), and corr(prediction, length) vs corr(truth, length)."""
    rows = []
    for m in models:
        df = load(m).assign(words=lambda d: d.essay_id.map(words))
        df["err"] = norm_err(df)
        per = df.groupby("prompt").apply(lambda g: pd.Series({
            "corr_error_length": g.err.corr(g.words),
            "corr_pred_length": g.y_pred.corr(g.words),
            "corr_true_length": g.y_true.corr(g.words),
        }), include_groups=False)
        rows.append({"model": m, **per.mean().round(3).to_dict()})
    return pd.DataFrame(rows)


def stress_test(models):
    """Mean score change when essays get longer without getting better, as % of each prompt's score range."""
    rows = []
    for m in models:
        base = load(m)
        for variant in ("offtopic", "repeat"):
            s = load(m, f"stress_{variant}_preds")
            if s is None:
                continue
            j = base.merge(s, on=["essay_id", "fold"], suffixes=("", "_s"))
            rng = j.prompt.map(lambda p: SCORE_RANGES[p][1] - SCORE_RANGES[p][0])
            shift = (j.y_pred_s - j.y_pred) / rng
            rows.append({"model": m, "variant": variant, "mean_shift_pct_of_range": round(100 * shift.mean(), 2),
                         "pct_essays_up": round(100 * (shift > 0).mean(), 1),
                         "pct_essays_down": round(100 * (shift < 0).mean(), 1)})
    return pd.DataFrame(rows)


def llm_subset(models):
    """Every model's QWK on exactly the essays the LLM graded (fold-0 test subset), so the comparison is fair."""
    llm = [m for m in models if m.startswith("llm_")]
    if not llm:
        return None
    ids = set(load(llm[0]).essay_id)
    table = {}
    for m in models:
        df = load(m)
        df = df[(df.fold == 0) & df.essay_id.isin(ids)]
        if len(df) == len(ids):
            table[m] = per_prompt_qwk(df)
    t = pd.DataFrame(table).T
    t.columns = [f"P{c}" for c in t.columns]
    t["Avg"] = t.mean(axis=1)
    return t.round(3)


def feasibility(models):
    rows = []
    for m in models:
        metas = [json.loads(f.read_text()) for f in sorted((RESULTS_DIR / m).glob("meta_fold*.json"))]
        log = RESULTS_DIR / m / "llm_log.csv"
        row = {"model": m}
        if metas:
            row |= {"device": metas[0]["device"], "train_minutes_total": round(sum(x["train_minutes"] for x in metas), 1),
                    "sec_per_essay": round(np.mean([x["test_sec_per_essay"] for x in metas]), 4)}
        if log.exists():
            lg = pd.read_csv(log)
            row |= {"device": "Gemini API (free tier)", "train_minutes_total": 0,
                    "sec_per_essay": round(lg.latency_s.mean(), 2),
                    "input_tokens_per_essay": round(lg.input_tokens.mean()),
                    "output_tokens_per_essay": round(lg.output_tokens.mean()),
                    "invalid_answers": int(lg.invalid.sum()), "essays": len(lg)}
        rows.append(row)
    return pd.DataFrame(rows)


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.set_facecolor(SURFACE)


def fig_qwk(table: pd.DataFrame):
    t = table.copy()
    t.index = [MODELS.get(i.split(" (")[0], (i,))[0] if not i.startswith("human") else "Human vs human" for i in t.index]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(t) + 1.2), facecolor=SURFACE)
    im = ax.imshow(t.values, cmap="Blues", vmin=0.3, vmax=0.9, aspect="auto")
    ax.set_xticks(range(t.shape[1]), t.columns)
    ax.set_yticks(range(t.shape[0]), t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="white" if v > 0.7 else INK)
    ax.set_title("QWK per prompt (5-fold average)", loc="left", color=INK)
    ax.tick_params(length=0, colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.colorbar(im, ax=ax, fraction=0.025)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "qwk_per_prompt.png", dpi=200)
    plt.close(fig)


def fig_length(models, words):
    """Mean signed error by essay-length quintile (within prompt). Flat at 0 = no length bias."""
    pick = [m for m in ("length_only_ridge", "features_ridge", "lstm", "deberta_v3_base", "llm_gemini_zero") if m in models]
    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor=SURFACE)
    _style(ax)
    ax.axhline(0, color=MUTED, lw=1)
    ends = []
    for m in pick:
        df = load(m).assign(words=lambda d: d.essay_id.map(words))
        df["err"] = norm_err(df)
        df["q"] = df.groupby("prompt").words.transform(lambda w: pd.qcut(w.rank(method="first"), 5, labels=False))
        y = df.groupby("q").err.mean() * 100
        name, color = MODELS[m]
        ax.plot(y.index + 1, y.values, color=color, lw=2, marker="o", ms=8, mec=SURFACE, mew=2)
        ends.append([y.values[-1], name])
    lo, hi = ax.get_ylim()
    gap = 0.05 * (hi - lo)
    ends.sort()
    for i in range(1, len(ends)):  # nudge end labels apart so they never overlap
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y_end, name in ends:
        ax.annotate(name, (5, y_end), xytext=(10, 0), textcoords="offset points", va="center", fontsize=9, color=INK)
    ax.set_xticks(range(1, 6), ["shortest\n20%", "2", "3", "4", "longest\n20%"])
    ax.set_xlim(0.8, 6.8)
    ax.set_ylabel("mean error (% of score range)\n< 0 under-scores, > 0 over-scores", color=MUTED)
    ax.set_title("Prediction error by essay length", loc="left", color=INK)
    ax.grid(axis="y", color="#e4e3df", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "length_bias.png", dpi=200)
    plt.close(fig)


def fig_stress(stress: pd.DataFrame):
    if stress.empty:
        return
    p = stress.pivot(index="model", columns="variant", values="mean_shift_pct_of_range")
    p = p.loc[[m for m in MODELS if m in p.index]]
    labels = {"offtopic": ("+ off-topic paragraph", "#2a78d6"), "repeat": ("essay pasted twice", "#eb6834")}
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(p) + 1.5), facecolor=SURFACE)
    _style(ax)
    ax.axvline(0, color=MUTED, lw=1)
    h = 0.38
    for k, v in enumerate(c for c in ("offtopic", "repeat") if c in p.columns):
        y = np.arange(len(p)) + (k - 0.5) * (h + 0.02)
        ax.barh(y, p[v], height=h, color=labels[v][1], label=labels[v][0])
        for yi, val in zip(y, p[v]):
            ax.text(val, yi, f" {val:+.1f}" if val >= 0 else f"{val:+.1f} ", va="center",
                    ha="left" if val >= 0 else "right", fontsize=8, color=INK)
    ax.set_yticks(range(len(p)), [MODELS[m][0] for m in p.index])
    ax.invert_yaxis()
    ax.set_xlabel("mean score change (% of score range)", color=MUTED)
    ax.set_title("Stress test: does a longer (not better) essay score higher?", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "stress_test.png", dpi=200)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    models = available()
    asap = load_asap()
    words = asap.set_index("essay_id").essay.str.split().str.len()

    lb = length_bias(models, words)
    st = stress_test(models)
    fe = feasibility(models)
    sub = llm_subset(models)
    lb.to_csv(RESULTS_DIR / "length_bias.csv", index=False)
    st.to_csv(RESULTS_DIR / "stress_test.csv", index=False)
    fe.to_csv(RESULTS_DIR / "feasibility.csv", index=False)
    for t in (lb, st, fe):
        print(t.to_string(index=False), "\n")
    if sub is not None:
        sub.to_csv(RESULTS_DIR / "llm_subset_qwk.csv")
        print(sub.to_string(), "\n")

    table = RESULTS_DIR / "qwk_table.csv"
    if table.exists():
        fig_qwk(pd.read_csv(table, index_col=0))
    fig_length(models, words)
    fig_stress(st)
    print("figures in", FIG_DIR)


if __name__ == "__main__":
    main()
