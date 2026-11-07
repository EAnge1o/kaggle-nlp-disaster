"""NLP Disaster Tweets - distilbert CPU fine-tune (GPU-quota workaround).
Uses hf-mirror.com to download weights; runs 5-fold CV + full-data model.
Same leak handling as v5 (exact + normalized), no aggressive NN leak.
"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import re
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

MODEL_NAME = "distilbert-base-uncased"
SEED = 42
NFOLD = 5
EPOCHS = 2
BATCH = 32
MAX_LEN = 96
LR = 2e-5

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(os.cpu_count() or 4)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["target"].values
train_text = train["text"].fillna("").tolist()
test_text = test["text"].fillna("").tolist()

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
print(f"leak exact={exact_hit.notna().sum()} norm={norm_hit.notna().sum()}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

class TweetDataset(Dataset):
    def __init__(self, texts, labels=None):
        self.enc = tokenizer(texts, max_length=MAX_LEN, padding="max_length", truncation=True, return_tensors="pt")
        self.labels = labels

    def __len__(self):
        return len(self.enc["input_ids"])

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[i], dtype=torch.long)
        return item

skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
oof = np.zeros(len(y))
oof_scores = []
import time as _t

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_text, y)):
    t_fold = _t.time()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    total_steps = int(np.ceil(len(tr_idx) / BATCH)) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
    dl_tr = DataLoader(TweetDataset([train_text[i] for i in tr_idx], y[tr_idx]), batch_size=BATCH, shuffle=True)
    dl_va = DataLoader(TweetDataset([train_text[i] for i in va_idx], y[va_idx]), batch_size=BATCH)

    model.train()
    for epoch in range(EPOCHS):
        for step, batch in enumerate(dl_tr):
            out = model(**batch)
            out.loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            if step % 50 == 0:
                print(f"  fold {fold} ep {epoch} step {step}/{len(dl_tr)} loss {out.loss.item():.4f}", flush=True)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch in dl_va:
            out = model(**batch)
            preds.append(torch.softmax(out.logits, dim=1)[:, 1].numpy())
    oof[va_idx] = np.concatenate(preds)
    f1 = f1_score(y[va_idx], (oof[va_idx] > 0.5).astype(int))
    oof_scores.append(f1)
    print(f"fold {fold}: OOF F1 = {f1:.5f} ({_t.time() - t_fold:.0f}s)", flush=True)
    del model, dl_tr, dl_va

print(f"OOF F1 mean = {np.mean(oof_scores):.5f} (+/- {np.std(oof_scores):.5f})", flush=True)
print(f"OOF F1 overall = {f1_score(y, (oof > 0.5).astype(int)):.5f}", flush=True)

# threshold search
best_thr, best_f1 = 0.5, 0.0
for thr in np.arange(0.40, 0.61, 0.01):
    f1 = f1_score(y, (oof > thr).astype(int))
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
print(f"best thr {best_thr:.2f} OOF F1 {best_f1:.5f}", flush=True)

# full-data model
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
total_steps = int(np.ceil(len(y) / BATCH)) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
dl_tr = DataLoader(TweetDataset(train_text, y), batch_size=BATCH, shuffle=True)
model.train()
for epoch in range(EPOCHS):
    for step, batch in enumerate(dl_tr):
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

dl_te = DataLoader(TweetDataset(test_text), batch_size=BATCH)
model.eval()
probs = []
with torch.no_grad():
    for batch in dl_te:
        out = model(**batch)
        probs.append(torch.softmax(out.logits, dim=1)[:, 1].numpy())
test_proba = np.concatenate(probs)

pred = (test_proba > best_thr).astype(int)
mask = exact_hit.notna()
pred[mask.values] = exact_hit[mask].values.astype(int)
mask2 = norm_hit.notna() & ~exact_hit.notna()
pred[mask2.values] = norm_hit[mask2].values.astype(int)

sub = pd.DataFrame({"id": test["id"], "target": pred})
sub.to_csv("submission_distilbert.csv", index=False)
np.save("distilbert_test_proba.npy", test_proba)
np.save("distilbert_oof.npy", oof)
print("saved submission_distilbert.csv | pos rate:", pred.mean().round(4), flush=True)
