"""Fine-tune a pretrained encoder (DeBERTa-v3 / BERT) to regress per-prompt-normalized scores.

One model for all 8 prompts. The text is prefixed with its prompt number so the model knows which task it is
grading; the target is the score scaled to [0, 1] within that prompt's range.
"""
import json
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from aes.data import from_unit, load_fold, to_unit
from aes.metrics import per_prompt_qwk
from aes.stress import stress_variants
from aes.tracking import RESULTS_DIR, save_predictions


def encode(texts, prompts, tok, max_len, head):
    """Tokenize with head+tail truncation: keep the first `head` tokens and the last ones that still fit."""
    texts = [f"Prompt {p}: {t}" for t, p in zip(texts, prompts)]
    ids = tok(texts, add_special_tokens=False)["input_ids"]
    room = max_len - 2  # [CLS] ... [SEP], the single-sequence format of both BERT and DeBERTa
    out = []
    for x in ids:
        if len(x) > room:
            x = x[:head] + x[-(room - head):]
        out.append([tok.cls_token_id, *x, tok.sep_token_id])
    return out


def _batches(ids, targets, tok, batch_size, shuffle, seed=0):
    def collate(idx):
        seqs = [ids[i] for i in idx]
        L = max(map(len, seqs))
        input_ids = torch.full((len(seqs), L), tok.pad_token_id)
        mask = torch.zeros((len(seqs), L), dtype=torch.long)
        for r, s in enumerate(seqs):
            input_ids[r, :len(s)] = torch.tensor(s)
            mask[r, :len(s)] = 1
        return input_ids, mask, torch.tensor(targets[idx], dtype=torch.float)

    g = torch.Generator().manual_seed(seed)
    return DataLoader(np.arange(len(ids)), batch_size=batch_size, shuffle=shuffle, generator=g, collate_fn=collate)


@torch.no_grad()
def predict(model, ids, tok, batch_size, device):
    model.eval()
    order = np.argsort([len(x) for x in ids])  # length-sorted batches = less padding = faster
    sorted_ids = [ids[i] for i in order]
    out = []
    for input_ids, mask, _ in _batches(sorted_ids, np.zeros(len(ids)), tok, batch_size * 2, shuffle=False):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits.float().squeeze(-1).cpu())
    raw = np.empty(len(ids))
    raw[order] = torch.cat(out).numpy()
    return raw


def _frame(df, fold, raw):
    return pd.DataFrame({"essay_id": df.essay_id, "prompt": df.prompt, "fold": fold, "y_true": df.score,
                         "y_pred": from_unit(raw, df.prompt), "y_raw": raw})


def train_fold(cfg: dict, fold: int, limit: int | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_fold(fold)
    if limit:  # smoke tests only
        data = {k: v.sample(min(limit, len(v)), random_state=0).reset_index(drop=True) for k, v in data.items()}
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    enc = {k: encode(v.essay.tolist(), v.prompt.tolist(), tok, cfg["max_len"], cfg["head_tokens"]) for k, v in data.items()}
    y_train = to_unit(data["train"].score, data["train"].prompt)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model"], num_labels=1, hidden_dropout_prob=cfg["dropout"], attention_probs_dropout_prob=cfg["dropout"]
    ).float().to(device)  # fp32 master weights; transformers>=5 keeps the checkpoint's fp16 dtype otherwise
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loader = _batches(enc["train"], y_train, tok, cfg["batch_size"], shuffle=True, seed=cfg["seed"])
    steps = len(loader) * cfg["epochs"] // cfg["grad_accum"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup_ratio"] * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda" and cfg["fp16"])

    best, best_state, history, t0 = -1.0, None, [], time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        losses = []
        for step, (input_ids, mask, y) in enumerate(loader, 1):
            with torch.autocast(device.type, enabled=device.type == "cuda" and cfg["fp16"]):
                pred = model(input_ids=input_ids.to(device), attention_mask=mask.to(device)).logits.squeeze(-1)
            loss = torch.nn.functional.mse_loss(pred.float(), y.to(device)) / cfg["grad_accum"]
            scaler.scale(loss).backward()
            if step % cfg["grad_accum"] == 0 or step == len(loader):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                sched.step()
            losses.append(loss.item() * cfg["grad_accum"])
        dev_qwk = per_prompt_qwk(_frame(data["dev"], fold, predict(model, enc["dev"], tok, cfg["batch_size"], device))).mean()
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_qwk": dev_qwk,
                        "minutes": (time.time() - t0) / 60})
        print(history[-1])
        if dev_qwk > best:
            best, best_state = dev_qwk, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    name = cfg["name"]
    save_predictions(_frame(data["dev"], fold, predict(model, enc["dev"], tok, cfg["batch_size"], device)), name, fold, "dev")
    t1 = time.time()
    scores = save_predictions(_frame(data["test"], fold, predict(model, enc["test"], tok, cfg["batch_size"], device)),
                              name, fold)
    sec_per_essay = (time.time() - t1) / len(data["test"])

    test = data["test"]
    for variant, texts in stress_variants(test.essay.tolist()).items():
        ids = encode(texts, test.prompt.tolist(), tok, cfg["max_len"], cfg["head_tokens"])
        save_predictions(_frame(test, fold, predict(model, ids, tok, cfg["batch_size"], device)), name, fold, f"stress_{variant}")

    out = RESULTS_DIR / name
    pd.DataFrame(history).to_csv(out / f"history_fold{fold}.csv", index=False)
    meta = {"fold": fold, "best_dev_qwk": best, "test_qwk": scores.mean(), "train_minutes": history[-1]["minutes"],
            "test_sec_per_essay": sec_per_essay, "device": torch.cuda.get_device_name() if device.type == "cuda" else "cpu",
            "config": cfg}
    (out / f"meta_fold{fold}.json").write_text(json.dumps(meta, indent=2))
    return scores
