"""Rubric-prompted LLM grader on a fixed subset: fold-0 test, N essays per prompt.

Why a subset and not all 13k essays? The free Gemini tier allows only a limited number of requests per day, so we
grade 25 essays per prompt (200 total) and compare every other model on exactly the same 200 essays (Table 2 of the
report), which keeps the comparison fair.

python scripts/run_llm.py --config configs/llm_gemini_zero.yaml
python scripts/run_llm.py --config configs/llm_gemini_few.yaml
python scripts/run_llm.py --config configs/llm_gemini_zero.yaml --limit 2   # quick check: 2 essays per prompt
"""
import argparse

import numpy as np
import pandas as pd

from aes.data import PROMPTS, SCORE_RANGES, load_fold
from aes.models.llm import LLM_SCALE, Cache, GeminiBackend, build_prompt, load_rubrics, parse_score
from aes.tracking import RESULTS_DIR, save_predictions
from aes.utils import load_config

# Name in the config -> backend class. Adding another provider (or a local model) means adding one entry here;
# it only needs a generate(prompt) -> (text, info) method.
BACKENDS = {"gemini": GeminiBackend}


def subset(test: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """n random test essays per prompt. Fixed seed = the same essays every run, so zero- and few-shot are comparable."""
    return test.groupby("prompt").sample(n, random_state=seed).reset_index(drop=True)


def few_shot_examples(train: pd.DataFrame, prompt: int, k: int, seed: int):
    """k training essays spread from low to high score, with their score on the single-rater scale."""
    # Why spread across the range? Examples at the 10th, 50th and 90th percentile of scores show the model what a
    # weak, an average and a strong essay look like, which anchors its scale better than random examples.
    # Why from the training split? Using test essays as examples would leak the answers into the prompt.
    g = train[train.prompt == prompt].sample(frac=1, random_state=seed).sort_values("score", kind="stable")
    picks = g.iloc[(np.linspace(0.1, 0.9, k) * (len(g) - 1)).astype(int)].itertuples()
    # For prompts whose final score is the sum of two raters, show rater 1's score: that is the scale the LLM uses.
    return [(r.essay, r.rater1 if LLM_SCALE[prompt][2] == 2 else r.score) for r in picks]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int, help="essays per prompt (overrides config) for a quick check")
    args = ap.parse_args()
    cfg = load_config(args.config)
    name = cfg["name"] if not args.limit else f"smoke_{cfg['name']}"

    data = load_fold(0)
    essays = subset(data["test"], args.limit or cfg["per_prompt"], cfg["seed"])
    rubrics = load_rubrics()
    backend = BACKENDS[cfg["backend"]](cfg["model"], cfg["min_interval"], cfg.get("thinking_budget"))
    cache = Cache(cfg["model"])
    shots = {p: few_shot_examples(data["train"], p, cfg["shots"], cfg["seed"]) for p in PROMPTS} if cfg["shots"] else {}
    # If the LLM never gives a usable score, fall back to the prompt's median training score. This is the
    # least-harmful guess, and the essay is still counted as "invalid" in the log so the failure stays visible.
    fallback = data["train"].groupby("prompt").score.median().round().astype(int)

    rows = []
    for i, e in enumerate(essays.itertuples(), 1):
        lo, hi, mult = LLM_SCALE[e.prompt]
        prompt = build_prompt(e.prompt, rubrics[e.prompt], e.essay, shots.get(e.prompt, ()))
        rec, tries = None, 0
        for tries in (1, 2):  # one retry on an invalid answer, with the cache bypassed
            # Cache first: a rerun after a crash or quota stop costs no API calls for essays already graded.
            hit = cache.get(prompt) if tries == 1 else None
            if hit is None:
                # The retry adds a stricter instruction; it is not cached, so the original prompt keeps its answer.
                text, info = backend.generate(prompt if tries == 1 else prompt + "\n\nReply with the JSON object only.")
                hit = {"text": text, **info, "cached": False}
                if tries == 1:
                    cache.put(prompt, hit)
            else:
                hit = {**hit, "cached": True}
            rec = hit
            if parse_score(rec["text"], lo, hi) is not None:
                break
        raw = parse_score(rec["text"], lo, hi)
        lo_f, hi_f = SCORE_RANGES[e.prompt]
        # Convert one rater's score to the final scale (x2 for prompts 1, 7, 8) and clip to the official range.
        pred = fallback[e.prompt] if raw is None else int(np.clip(raw * mult, lo_f, hi_f))
        rows.append({"essay_id": e.essay_id, "prompt": e.prompt, "fold": 0, "y_true": e.score, "y_pred": pred,
                     "llm_raw_score": raw, "invalid": raw is None, "tries": tries, "cached": rec["cached"],
                     "latency_s": rec.get("latency_s"), "input_tokens": rec.get("input_tokens"),
                     "output_tokens": rec.get("output_tokens")})
        print(f"{i}/{len(essays)} prompt {e.prompt}: true {e.score} pred {pred}{' (invalid)' if raw is None else ''}")

    # The log is committed, so it holds no essay text and no model explanations: those may quote the essay,
    # and the ASAP rules forbid sharing essays. Explanations stay in the gitignored cache.
    log = pd.DataFrame(rows)
    save_predictions(log, name, 0)
    log.to_csv(RESULTS_DIR / name / "llm_log.csv", index=False)
    fresh = log[~log.cached]  # cached answers took ~0 s, so latency is measured on fresh calls only
    print(f"invalid: {log.invalid.sum()}/{len(log)} | mean latency (uncached) "
          f"{fresh.latency_s.mean() if len(fresh) else float('nan'):.2f}s | "
          f"tokens in/out per essay {log.input_tokens.mean():.0f}/{log.output_tokens.mean():.0f}")


if __name__ == "__main__":
    main()
