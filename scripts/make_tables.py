"""Build the results table (QWK per model x prompt) from reports/results/*/preds_fold*.csv.

python scripts/make_tables.py   ->  reports/results/qwk_table.csv and qwk_table.md
"""
import pandas as pd

from aes.metrics import per_prompt_qwk
from aes.data import ROOT
from aes.tracking import RESULTS_DIR


def main():
    table = {}
    # One row per experiment folder; smoke-test folders (quick local checks) are skipped.
    for d in sorted(p for p in RESULTS_DIR.glob("*") if p.is_dir() and not p.name.startswith("smoke_")):
        files = sorted(d.glob("preds_fold*.csv"))
        if files:
            # The fold count is kept in the row name so partial runs are never mistaken for complete ones.
            table[f"{d.name} ({len(files)} folds)"] = per_prompt_qwk(pd.concat(map(pd.read_csv, files)))
    human = RESULTS_DIR / "human_agreement.csv"  # from scripts/human_agreement.py
    if human.exists():
        table["human (rater1 vs rater2)"] = pd.read_csv(human, index_col="prompt").qwk
    if not table:
        print("no predictions found")
        return
    t = pd.DataFrame(table).T
    t.columns = [f"P{c}" for c in t.columns]
    t["Avg"] = t.mean(axis=1)  # unweighted mean over the 8 prompts, as in the paper
    t = t.round(3).sort_values("Avg")
    t.to_csv(RESULTS_DIR / "qwk_table.csv")
    (RESULTS_DIR / "qwk_table.md").write_text(t.to_markdown())
    print(t.to_string())
    update_readme(t)


def update_readme(t: pd.DataFrame):
    """Replace the README's results table with the current numbers, so the README is never out of date.

    Only models evaluated on all 5 folds (plus the human row) are shown: partial runs and the LLM, which graded a
    200-essay subset, are not comparable with 5-fold averages.
    """
    readme = ROOT / "README.md"
    start, end = "<!-- results:start -->", "<!-- results:end -->"
    text = readme.read_text(encoding="utf-8")
    if start not in text:
        return
    full = t[[i.startswith("human") or "(5 folds)" in i for i in t.index]].sort_values("Avg", ascending=False)
    full.index = [nice_name(i) for i in full.index]
    table = full.to_markdown(floatfmt=".3f")
    readme.write_text(text[:text.index(start) + len(start)] + "\n" + table + "\n" + text[text.index(end):], encoding="utf-8")


def nice_name(row: str) -> str:
    """'deberta_v3_base (5 folds)' -> 'DeBERTa-v3-base', using the display names defined in analyze.py."""
    from analyze import MODELS  # imported here: analyze.py loads matplotlib, which make_tables otherwise does not need
    base = row.split(" (")[0]
    return "Human vs human" if base.startswith("human") else MODELS.get(base, (base,))[0]


if __name__ == "__main__":
    main()
