"""Taghipour & Ng (2016) neural AES: embeddings -> (optional CNN) -> LSTM -> mean-over-time -> sigmoid.

One model per prompt, trained on scores scaled to [0, 1]; the epoch with the best dev QWK is kept.
Deviation from the paper: embeddings are randomly initialized (the paper used pretrained 50-d embeddings).

Design choices (all from the paper unless noted):
- One model per prompt: each prompt has its own topic, genre and score scale, so the paper trains separately.
- Mean over time (average of the LSTM's output at every word) instead of only the last output: essays are long,
  and the last state mostly remembers the end of the essay; the paper found mean-over-time works best.
- Sigmoid output + scores scaled to [0, 1]: the output can never leave the valid range.
- MSE loss: a regression on the score, so bigger mistakes are penalised more, like in QWK.
"""
import re
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from aes.data import SCORE_RANGES, from_unit
from aes.metrics import qwk

# Reserved vocabulary ids: padding, unknown word, any number.
PAD, UNK, NUM = 0, 1, 2
# Tokens: anonymisation tags (@person1), words (with apostrophes), numbers, or single punctuation characters.
_TOKEN = re.compile(r"@[a-z]+\d*|[a-z]+(?:'[a-z]+)?|\d+(?:[.,]\d+)*|[^\sa-z\d]")


def tokenize(text: str) -> list:
    """Lowercase; '@PERSON1' -> '@person' (ASAP anonymization); numbers -> <num> (as in nea)."""
    return ["<num>" if t[0].isdigit() else t.rstrip("0123456789") for t in _TOKEN.findall(text.lower())]


def build_vocab(texts, size: int) -> dict:
    """Map the `size` most frequent training tokens (specials included) to ids; everything else becomes <unk>."""
    counts = Counter(t for x in texts for t in tokenize(x))
    counts.pop("<num>", None)  # already a reserved id
    vocab = {"<pad>": PAD, "<unk>": UNK, "<num>": NUM}
    for w, _ in counts.most_common(size - len(vocab)):
        vocab[w] = len(vocab)
    return vocab


def to_ids(texts, vocab) -> list:
    """Essays as lists of token ids; an empty essay becomes [<unk>] so it still has length 1."""
    return [[vocab.get(t, UNK) for t in tokenize(x)] or [UNK] for x in texts]


class NeaModel(nn.Module):
    """The paper's architecture: embedding -> optional convolution -> LSTM -> mean over time -> linear -> sigmoid."""

    def __init__(self, vocab_size, emb_dim, hidden, cnn_filters, cnn_window, dropout, init_bias):
        """cnn_filters=0 builds the plain LSTM; init_bias sets the starting output (see train_prompt)."""
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)
        # Optional convolution over neighbouring words (the CNN-LSTM variant); padding keeps the sequence length.
        self.cnn = nn.Conv1d(emb_dim, cnn_filters, cnn_window, padding=cnn_window // 2) if cnn_filters else None
        self.lstm = nn.LSTM(cnn_filters or emb_dim, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(hidden, 1)
        nn.init.constant_(self.out.bias, init_bias)  # start predicting the mean training score

    def forward(self, ids, lengths):
        """ids: (batch, time) padded token ids; lengths: true essay lengths. Returns one [0, 1] score per essay."""
        x = self.emb(ids)
        if self.cnn is not None:
            x = self.cnn(x.transpose(1, 2)).transpose(1, 2)  # Conv1d expects (batch, channels, time)
        # Packing makes the LSTM skip padding positions, so short essays are not diluted by zeros.
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True)
        mot = h.sum(1) / lengths.unsqueeze(1).to(h.dtype)  # mean over time, padding excluded (padded h is 0)
        return torch.sigmoid(self.out(self.drop(mot))).squeeze(-1)  # sigmoid keeps the output in [0, 1]


def _batch(seqs, device):
    """Pad a list of id sequences into one tensor, and return it with the true lengths."""
    lengths = torch.tensor([len(s) for s in seqs])
    ids = torch.zeros(len(seqs), int(lengths.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
    return ids.to(device), lengths.to(device)


@torch.no_grad()
def predict(model, seqs, device, batch_size=64):
    """Model outputs in [0, 1] for every essay, in the original order."""
    model.eval()
    # Batch essays of similar length together to minimise padding, then write results back to their positions.
    order = np.argsort([len(s) for s in seqs])
    out = np.empty(len(seqs))
    for i in range(0, len(seqs), batch_size):
        idx = order[i:i + batch_size]
        out[idx] = model(*_batch([seqs[j] for j in idx], device)).cpu().numpy()
    return out


def train_prompt(cfg, prompt, train, dev, y_train, device):
    """Train on one prompt; return (model, vocab, best dev QWK, best epoch)."""
    vocab = build_vocab(train.essay, cfg["vocab_size"])  # built from training essays only
    tr, dv = to_ids(train.essay, vocab), to_ids(dev.essay, vocab)
    # Output bias = logit of the mean target, so the untrained model already predicts the average score.
    mean = float(np.clip(y_train.mean(), 1e-3, 1 - 1e-3))
    model = NeaModel(len(vocab), cfg["emb_dim"], cfg["hidden"], cfg["cnn_filters"] if cfg["cnn"] else 0,
                     cfg["cnn_window"], cfg["dropout"], init_bias=float(np.log(mean / (1 - mean)))).to(device)
    opt = torch.optim.RMSprop(model.parameters(), lr=cfg["lr"], alpha=0.9, eps=1e-6)  # optimiser used in the paper
    y = torch.tensor(y_train, dtype=torch.float)
    rng = np.random.default_rng(cfg["seed"])
    lo, hi = SCORE_RANGES[prompt]
    best, best_epoch, best_state = -2.0, 0, None
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        # Shuffle each epoch and split into batches of about batch_size essays.
        for idx in np.array_split(rng.permutation(len(tr)), max(1, len(tr) // cfg["batch_size"])):
            loss = nn.functional.mse_loss(model(*_batch([tr[i] for i in idx], device)), y[idx].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["clip_norm"])  # guards against exploding LSTM gradients
            opt.step()
        # Model selection as in the paper: keep the weights from the epoch with the best dev QWK.
        dev_qwk = qwk(dev.score, from_unit(predict(model, dv, device), dev.prompt), lo, hi)
        if dev_qwk > best:
            best, best_epoch = dev_qwk, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, vocab, best, best_epoch
