"""Fill reports/REPORT.template.md with numbers and tables from reports/results/, write REPORT.md and REPORT.pdf.

python scripts/build_report.py            # needs make_tables.py and analyze.py to have run first
Placeholders: {{qwk_table}} etc. for tables; {{qwk.<model>}}, {{shift.<model>.<variant>}}, ... for single numbers.

Why a template with placeholders instead of writing numbers in the report? Every number is then read from the
result files, so a rerun updates the report automatically and no number can be mistyped or out of date.
The README's results table is regenerated the same way.
"""
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown
import pandas as pd

from analyze import MODELS
from aes.data import ROOT, load_asap
from aes.tracking import RESULTS_DIR

REPORTS = ROOT / "reports"


def nice(name: str) -> str:
    """Display name for an experiment folder or table row, e.g. 'deberta_v3_base (5 folds)' -> 'DeBERTa-v3-base'."""
    base = name.split(" (")[0]
    return "Human vs human" if base.startswith("human") else MODELS.get(base, (base,))[0]


def md_table(df: pd.DataFrame, index=True) -> str:
    """Markdown table with 3 decimals; pandas marks numeric columns right-aligned."""
    return df.to_markdown(index=index, floatfmt=".3f")


def values_and_tables():
    """Two lookups for the template: v = single numbers (as formatted text), t = whole tables with captions."""
    v, t = {"n_essays": f"{len(load_asap()):,}"}, {}
    q = pd.read_csv(RESULTS_DIR / "qwk_table.csv", index_col=0)
    # Every QWK value becomes addressable, e.g. {{qwk.bert_base}} (average) or {{qwk.bert_base.P8}} (one prompt).
    for name, row in q.iterrows():
        key = "human" if name.startswith("human") else name.split(" (")[0]
        v[f"qwk.{key}"] = f"{row.Avg:.3f}"
        for c in q.columns:
            v[f"qwk.{key}.{c}"] = f"{row[c]:.3f}"
    q = q[[i.startswith("human") or "(5 folds)" in i for i in q.index]].sort_values("Avg", ascending=False)
    q.index = [nice(i) for i in q.index]  # LLM (subset only) is in Table 2; partial runs are left out
    t["qwk_table"] = "<p class=\"caption\"><b>Table 1. </b>QWK per prompt, averaged over the 5 folds (higher is better).</p>\n\n" + md_table(q)

    f = RESULTS_DIR / "llm_subset_qwk.csv"
    if f.exists():
        s = pd.read_csv(f, index_col=0)
        for name, row in s.iterrows():
            v[f"sub.{name}"] = f"{row.Avg:.3f}"
        s.index = map(nice, s.index)
        t["llm_subset_table"] = ("<p class=\"caption\"><b>Table 2. </b>QWK on the 200-essay subset (fold-0 test, 25 per prompt) graded by every "
                                 "model, including the LLM. Per-prompt values rest on 25 essays each, so they are noisy.</p>\n\n"
                                 + md_table(s))

    lb = pd.read_csv(RESULTS_DIR / "length_bias.csv")
    for r in lb.itertuples():
        v[f"lcorr.{r.model}"], v[f"pcorr.{r.model}"] = f"{r.corr_error_length:+.2f}", f"{r.corr_pred_length:.2f}"
    v["tcorr"] = f"{lb.corr_true_length.iloc[0]:.2f}"  # same for all 5-fold models (same essays, same human scores)
    lb.model = lb.model.map(nice)
    lb.columns = ["Model", "corr(error, length)", "corr(prediction, length)", "corr(human score, length)"]
    t["length_table"] = ("<p class=\"caption\"><b>Table 3. </b>Correlation with essay length (in words), averaged over prompts. Errors that grow "
                         "with length show as a non-zero first column.</p>\n\n" + md_table(lb, index=False))

    st = pd.read_csv(RESULTS_DIR / "stress_test.csv")
    for r in st.itertuples():
        v[f"shift.{r.model}.{r.variant}"] = f"{r.mean_shift_pct_of_range:+.1f}%"
        v[f"up.{r.model}.{r.variant}"] = f"{r.pct_essays_up:.0f}%"
    st.model = st.model.map(nice)
    st.columns = ["Model", "Variant", "Mean score change (% of range)", "% essays scored higher", "% essays scored lower"]
    t["stress_table"] = ("<p class=\"caption\"><b>Table 4. </b>Stress test on all test essays: score change after making the essay longer "
                         "but not better.</p>\n\n" + md_table(st, index=False))

    fe = pd.read_csv(RESULTS_DIR / "feasibility.csv")
    for r in fe.to_dict("records"):
        v[f"sec.{r['model']}"] = f"{r['sec_per_essay']:.4g}"
        if pd.notna(r.get("train_minutes_total")):
            v[f"train.{r['model']}"] = f"{r['train_minutes_total']:.0f}"
        if "invalid_answers" in r and pd.notna(r.get("invalid_answers")):
            v[f"invalid.{r['model']}"] = f"{int(r['invalid_answers'])}/{int(r['essays'])}"
    fe.model = fe.model.map(nice)
    t["feasibility_table"] = ("<p class=\"caption\"><b>Table 5. </b>Cost and speed. Times are per essay at prediction time; training time is the "
                              "total over 5 folds. All runs used free resources.</p>\n\n" + md_table(fe.astype(object).fillna("–"), index=False))
    v.update(extra_values())
    return v, t


def extra_values() -> dict:
    """Derived numbers quoted in the discussion text."""
    from analyze import available, load, norm_err
    from aes.metrics import per_prompt_qwk

    v = {}
    for m in available():
        v[f"bias.{m}"] = f"{100 * norm_err(load(m)).mean():+.1f}%"  # mean signed error, % of score range
    q = pd.read_csv(RESULTS_DIR / "qwk_table.csv", index_col=0)
    human = q.loc[[i for i in q.index if i.startswith("human")][0]]
    for m in ("deberta_v3_base", "deberta_v3_base_thr"):
        row = q.loc[[i for i in q.index if i.split(" (")[0] == m][0]]
        v[f"beats_human.{m}"] = str(int((row.drop("Avg") > human.drop("Avg")).sum()))
    for short, long in (("lstm", "lstm_e50"), ("cnn_lstm", "cnn_lstm_e50")):
        a, b = load(short), load(long)
        if a is not None and b is not None:
            shared = sorted(set(a.fold) & set(b.fold))
            qa = per_prompt_qwk(a[a.fold.isin(shared)]).mean()
            qb = per_prompt_qwk(b[b.fold.isin(shared)]).mean()
            v[f"e30.{short}"], v[f"e50.{short}"], v[f"efolds.{short}"] = f"{qa:.3f}", f"{qb:.3f}", str(len(shared))
    return v


def fill(template: str, v: dict, t: dict) -> str:
    """Replace every {{key}} in the template; unknown keys are marked visibly in the output and listed as a warning."""
    missing = set()

    def sub(m):
        """Replacement for one {{key}} match: a table first, then a single value, else a visible marker."""
        key = m.group(1)
        if key in t:
            return t[key]
        if key in v:
            return v[key]
        missing.add(key)
        return f"**[missing: {key}]**"

    out = re.sub(r"\{\{([\w.\-]+)\}\}", sub, template)
    if missing:
        print("WARNING, missing values:", sorted(missing))
    return out


CSS = """
/* Print layout in the style of a conference paper: serif text, ruled tables, numbered pages. */
@page { size: A4; margin: 22mm 20mm 20mm; @bottom-center { content: counter(page); font: 9pt Cambria, Georgia, serif; color: #555; } }
body { font: 10.5pt/1.5 Cambria, Georgia, 'Times New Roman', serif; color: #1a1a1a; max-width: 44em; margin: auto;
       text-align: justify; hyphens: auto; font-variant-numeric: oldstyle-nums; }
h1 { font-size: 19pt; font-weight: normal; text-align: center; margin: 0 0 .3em; letter-spacing: .01em; }
.authors { text-align: center; font-variant: small-caps; letter-spacing: .06em; font-size: 11.5pt; margin: 0; }
.venue { text-align: center; font-size: 9pt; color: #555; margin: .2em 0 1.4em; }
.venue a { color: #555; text-decoration: none; }
.abstract { margin: 0 2.5em 1.2em; font-size: 9.5pt; }
.abstract-label { font-variant: small-caps; font-weight: bold; letter-spacing: .04em; }
.note { font-size: 8.5pt; color: #666; font-style: italic; text-align: center; }
h2 { font-size: 12pt; font-variant: small-caps; letter-spacing: .05em; font-weight: bold; margin: 1.6em 0 .5em;
     break-after: avoid; }
h2::after { content: ""; display: block; width: 2.2em; border-bottom: 1.5px solid #8b1e1e; margin-top: .25em; }
p, li { orphans: 3; widows: 3; }
ul, ol { padding-left: 1.4em; }
li { margin: .15em 0; }
strong { font-weight: bold; }
a { color: #1a1a1a; }
code { font: 8.5pt Consolas, 'Courier New', monospace; text-align: left; }
.caption { font-size: 9pt; font-variant: small-caps; letter-spacing: .03em; margin: 1.2em 0 .2em; text-align: left; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; margin: .3em 0 1em; font-variant-numeric: lining-nums tabular-nums;
        border-top: 1.4px solid #1a1a1a; border-bottom: 1.4px solid #1a1a1a; break-inside: avoid; }
thead th { border-bottom: .7px solid #1a1a1a; font-weight: bold; }
th, td { padding: 2.5px 6px; text-align: left; vertical-align: top; }
th:first-child, td:first-child { text-align: left; }
figure { margin: 1.2em 0; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; }
figcaption { font-size: 8.5pt; text-align: left; margin-top: .4em; color: #333; }
"""


def to_pdf(md_text: str):
    """Markdown -> HTML (with CSS above) -> PDF, printed by a headless Edge/Chrome.

    Why a browser? It is already installed on the machine and supports print CSS (page size, page numbers),
    so no LaTeX or extra PDF library is needed.
    """
    html = REPORTS / "REPORT.html"
    md_text = re.sub(r"(?m)^( {1,3})(?=\S)", lambda m: m.group(1) * 2, md_text)  # 2-space nesting -> 4 for python-markdown
    body = markdown.markdown(md_text, extensions=["tables"])
    html.write_text(f"<!doctype html><meta charset='utf-8'><style>{CSS}</style>{body}", encoding="utf-8")
    browser = next((p for p in (shutil.which("msedge"), shutil.which("chrome"),
                                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                                r"C:\Program Files\Google\Chrome\Application\chrome.exe") if p and Path(p).exists()), None)
    if browser is None:
        print("No Edge/Chrome found: open reports/REPORT.html and print to PDF manually.")
        return
    pdf = REPORTS / "REPORT.pdf"
    with tempfile.TemporaryDirectory() as profile:  # own profile, so an already-open browser can't swallow the job
        subprocess.run([browser, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
                        f"--print-to-pdf={pdf}", html.as_uri()], check=True, capture_output=True, timeout=120)
    print("wrote", pdf) if pdf.exists() else print("PDF not written: open reports/REPORT.html and print to PDF")


def main():
    v, t = values_and_tables()
    # 1) report as Markdown (readable on GitHub), 2) README results table, 3) PDF for submission.
    text = fill((REPORTS / "REPORT.template.md").read_text(encoding="utf-8"), v, t)
    (REPORTS / "REPORT.md").write_text(text, encoding="utf-8")
    print("wrote", REPORTS / "REPORT.md")
    readme = ROOT / "README.md"
    rd = readme.read_text(encoding="utf-8")
    start, end = "<!-- results:start -->", "<!-- results:end -->"  # only the text between these markers is replaced
    table = t["qwk_table"].split("\n\n", 1)[1]
    readme.write_text(rd[:rd.index(start) + len(start)] + "\n" + table + "\n" + rd[rd.index(end):], encoding="utf-8")
    to_pdf(text)


if __name__ == "__main__":
    main()
