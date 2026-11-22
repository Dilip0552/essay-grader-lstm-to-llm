"""Human-human agreement (rater1 vs rater2 QWK) per prompt: the ceiling models are compared against.

Prompts 1, 7 and 8 give the final score as roughly the sum of the two raters, so each rater uses a smaller
scale than the final score; agreement is measured on the raters' own scale.

python scripts/human_agreement.py   ->  reports/results/human_agreement.csv
"""
from aes.data import load_asap
from aes.metrics import qwk
from aes.tracking import RESULTS_DIR


def main():
    df = load_asap()
    # No fixed min/max is passed: each prompt's rater scale is taken from the scores that actually occur.
    t = df.groupby("prompt").apply(lambda g: qwk(g.rater1, g.rater2), include_groups=False).rename("qwk")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(RESULTS_DIR / "human_agreement.csv")
    print(t.round(3).to_string(), f"\navg {t.mean():.3f}")


if __name__ == "__main__":
    main()
