# Contributing

## First time

```bash
git clone https://github.com/Dilip0552/essay-grader-lstm-to-llm.git
cd essay-grader-lstm-to-llm
pip install -r requirements.txt -e .
```
Then follow README "Data" to get the dataset. Never commit `data/`, model weights, or `.env`.

## Every task

1. Get the latest main: `git checkout main && git pull`
2. Make a branch named after your issue: `git checkout -b feat/12-lstm-model` (12 is the issue number)
3. Work, then commit small steps with clear messages:
   `git add <files>` then `git commit -m "Add LSTM model with mean-over-time pooling"`
4. Run `pytest` before pushing.
5. Push your branch: `git push -u origin feat/12-lstm-model`
6. Open a pull request on GitHub into `main`. Write `Closes #12` in the description and ask Dilip (@Dilip0552) to review.
7. If main moved on while you worked, run `git pull origin main` on your branch, fix any conflicts, and push again.

## Code rules

- Put code in `src/aes/`, and a runnable entry point in `scripts/` that reads a YAML from `configs/`.
- Use `aes.data.load_fold(k)` for data. Never make your own splits.
- Use `aes.utils.set_seed(cfg["seed"])` at the start of every script.
- Save test predictions with `aes.tracking.save_predictions(...)`. That one call puts your model in the results table.
- Don't paste numbers into the README or report by hand. They come from `scripts/make_tables.py`.
- Commit from your own GitHub account on your own machine.
