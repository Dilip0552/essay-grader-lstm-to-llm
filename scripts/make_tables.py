"""Build the results table (QWK per model x prompt) from reports/results/*/preds_fold*.csv."""
import pandas as pd

from aes.metrics import per_prompt_qwk
from aes.tracking import RESULTS_DIR


def main():
    table = {}
    for d in sorted(p for p in RESULTS_DIR.glob("*") if p.is_dir() and not p.name.startswith("smoke_")):
        files = sorted(d.glob("preds_fold*.csv"))
        if files:
            table[f"{d.name} ({len(files)} folds)"] = per_prompt_qwk(pd.concat(map(pd.read_csv, files)))
    human = RESULTS_DIR / "human_agreement.csv"  # from scripts/human_agreement.py
    if human.exists():
        table["human (rater1 vs rater2)"] = pd.read_csv(human, index_col="prompt").qwk
    if not table:
        print("no predictions found")
        return
    t = pd.DataFrame(table).T
    t.columns = [f"P{c}" for c in t.columns]
    t["Avg"] = t.mean(axis=1)
    t = t.round(3).sort_values("Avg")
    t.to_csv(RESULTS_DIR / "qwk_table.csv")
    (RESULTS_DIR / "qwk_table.md").write_text(t.to_markdown())
    print(t.to_string())


if __name__ == "__main__":
    main()
