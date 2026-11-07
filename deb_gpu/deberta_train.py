"""NLP Disaster Tweets - RoBERTa/DeBERTa 5-fold fine-tune for Kaggle GPU notebook.

Heuristics from top public solutions:
- Do NOT over-clean text (destroys transformers on this dataset). Use raw text.
- max_len 96 is enough for tweets; lr 2e-5; 2 epochs per fold.
- 5-fold stratified CV, probability average, apply known text-dup leak.
- Community references: RoBERTa-base 5-fold ~0.839, RoBERTa-large ~0.845.

Expects the competition data attached (competition_sources: nlp-getting-started).
Saves /kaggle/working/submission.csv for collection.
"""
import os
import re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "roberta-large"  # community-verified ~0.845; roberta-base got 0.8003 OOF
SEED = 42
NFOLD = 5
EPOCHS = 2
BATCH = 8
MAX_LEN = 128
LR = 2e-5

torch.manual_seed(SEED)
np.random.seed(SEED)

INPUT_DIR = "/kaggle/input/nlp-getting-started"
train = pd.read_csv(f"{INPUT_DIR}/train.csv")
test = pd.read_csv(f"{INPUT_DIR}/test.csv")
y = train["target"].values

# raw text, only fill NaN (no heavy cleaning - heuristics from top solutions)
train_text = train["text"].fillna("").tolist()
test_text = test["text"].fillna("").tolist()

# ---------- leak: duplicated / near-duplicated texts ----------
def normalize(s):
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\d+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

train_text_lbl = train.groupby("text")["target"].agg(["mean", "count"])
leak_map = {t: int(r["mean"]) for t, r in train_text_lbl.iterrows() if r["mean"] in (0.0, 1.0)}
exact_hit = pd.Series(test_text).map(leak_map)

norm_train = pd.Series(train_text).apply(normalize)
norm_test = pd.Series(test_text).apply(normalize)
ntl = pd.DataFrame({"n": norm_train, "t": y}).groupby("n")["t"].agg(["mean", "count"])
norm_map = {n_: int(r["mean"]) for n_, r in ntl.iterrows() if r["mean"] in (0.0, 1.0)}
norm_hit = norm_test.map(norm_map)
print(f"leak: exact={exact_hit.notna().sum()} normalized={norm_hit.notna().sum()}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TweetDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.enc = tokenizer(
            texts, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt"
        )
        self.labels = labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
oof = np.zeros(len(y))
oof_scores = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_text, y)):
    config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = int(np.ceil(len(tr_idx) / BATCH)) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
    ds_tr = TweetDataset([train_text[i] for i in tr_idx], y[tr_idx])
    ds_va = TweetDataset([train_text[i] for i in va_idx], y[va_idx])
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False)

    model.train()
    for epoch in range(EPOCHS):
        for step, batch in enumerate(dl_tr):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 100 == 0:
                print(f"  fold {fold} epoch {epoch} step {step}/{len(dl_tr)} loss {loss.item():.4f}", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dl_va:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch).logits
            preds.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
    oof[va_idx] = np.concatenate(preds)
    f1 = f1_score(y[va_idx], (oof[va_idx] > 0.5).astype(int))
    oof_scores.append(f1)
    print(f"fold {fold}: OOF F1 = {f1:.5f}", flush=True)
    del model, ds_tr, ds_va, dl_tr, dl_va
    torch.cuda.empty_cache()

print(f"OOF F1 mean = {np.mean(oof_scores):.5f} (+/- {np.std(oof_scores):.5f})")
print(f"OOF F1 overall = {f1_score(y, (oof > 0.5).astype(int)):.5f}")

# ---------- inference: average over folds (rerun each fold's model is costly,
# so train final model on full data; heuristics: full-data model ~= fold avg) ----------
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=2)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, config=config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = int(np.ceil(len(y) / BATCH)) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
ds_tr = TweetDataset(train_text, y)
dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)
model.train()
for epoch in range(EPOCHS):
    for step, batch in enumerate(dl_tr):
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f"  full epoch {epoch} step {step}/{len(dl_tr)} loss {loss.item():.4f}", flush=True)

ds_te = TweetDataset(test_text)
dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False)
model.eval()
probs = []
with torch.no_grad():
    for batch in dl_te:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch).logits
        probs.append(torch.softmax(out, dim=1)[:, 1].cpu().numpy())
test_proba = np.concatenate(probs)

# threshold from OOF
best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.40, 0.61, 0.01):
    f1 = f1_score(y, (oof > thr).astype(int))
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
print(f"best thr {best_thr:.2f} OOF F1 {best_f1:.5f}")

pred = (test_proba > best_thr).astype(int)
mask = exact_hit.notna()
pred[mask.values] = exact_hit[mask].values.astype(int)
mask2 = norm_hit.notna() & ~exact_hit.notna()
pred[mask2.values] = norm_hit[mask2].values.astype(int)

sub = pd.DataFrame({"id": test["id"], "target": pred})
sub.to_csv("/kaggle/working/submission.csv", index=False)
np.save("/kaggle/working/test_proba.npy", test_proba)
np.save("/kaggle/working/oof_proba.npy", oof)
print("submission rows:", len(sub), "pos rate:", sub["target"].mean().round(4))
print("done")
