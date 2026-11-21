# Plan

Team project (4 members). Deliverables: this GitHub repo and a written report.

Data rule (from the Kaggle asap-aes rules): we may publish summary and analysis, but never the essays. Prediction CSVs (IDs, scores) are fine. Essay text must never appear in commits, notebook outputs, logs, or few-shot prompt files. Compute: Kaggle notebooks (free GPU) and a local RTX 3050 (4 GB).

How Kaggle is used: a notebook clones this repo, installs it, trains, and writes prediction CSVs to `/kaggle/working/`. We download those CSVs into `reports/results/` and commit them. The code and results live in GitHub; Kaggle is only the GPU.

## Steps (in order)

| # | Task | Runs on | Status |
|---|---|---|---|
| 1 | Skeleton, data loader with nea folds, QWK, tests | local | done |
| 2 | Download ASAP data; check the loader on the real file | local | |
| 3 | Human agreement (rater1 vs rater2 QWK) + EDA plots | local CPU | |
| 4 | Feature baseline: `length_only_ridge`, `features_ridge`, `features_lgbm` | local CPU | |
| 5 | DeBERTa-v3-base fine-tune, 5 folds (+ BERT-base), Kaggle notebook | Kaggle GPU, ~2–3 h | |
| 6 | LSTM / CNN-LSTM reproduction, 5 folds | local 3050 or Kaggle | |
| 7 | LLM grader, zero-shot vs few-shot, 200-essay subset (fold 0 test) | Gemini free API (backend swappable) | |
| 8 | Per-prompt rounding thresholds on DeBERTa | local | |
| 9 | Length bias: error vs length correlation + padding stress test | local / Kaggle | |
| 10 | Results table, plots, report, README results | local | |

Step 5 starts first because it is the longest GPU job; steps 3, 4, 6 and 7 run while it trains.

## Cut for time
DeBERTa-large; multiple seeds on every fold (we use 3 seeds on fold 0 only); the LLM on all essays (we use a fixed 200-essay subset).

## Design notes
- One standard output: `reports/results/<experiment>/preds_fold{k}.csv`, written by `aes.tracking.save_predictions`. The tables and plots read only these files.
- LSTM follows the paper: one model per prompt, 4000-word vocabulary, 300-unit LSTM, mean-over-time pooling, sigmoid output on [0, 1] scores, RMSprop, best epoch chosen on dev QWK.
- DeBERTa: one model for all prompts, [0, 1] target, MSE loss, head+tail truncation to 512 tokens.
- LLM: Gemini free tier. Its terms allow Google to use inputs to improve products, so the report states that the 200 subset essays were sent to it. Only the minimum is sent: one essay per call, no IDs. For prompts 1, 7 and 8 the final score is two raters added together, so we ask for one rater's score and double it. Responses are cached on disk. Invalid answers are retried once, then counted.
