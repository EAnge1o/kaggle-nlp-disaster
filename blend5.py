"""Final 5-way blend: roberta-large + roberta-base + distilv1 + distilv2 + tfidf."""
import os
import re
from itertools import product
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["target"].values

probas = []
for name, oof_path, test_path in [
    ("roberta_l", "downloads_roberta_large/oof_proba.npy", "downloads_roberta_large/test_proba.npy"),
    ("roberta_b", "downloads_deberta/oof_proba.npy", "downloads_deberta/test_proba.npy"),
    ("distilv1", "downloads_cloud/oof_proba.npy", "downloads_cloud/test_proba.npy"),
    ("distilv2", "downloads_cloud_v2/oof_proba.npy", "downloads_cloud_v2/test_proba.npy"),
    ("tfidf", "tfidf_oof.npy", "tfidf_test.npy"),
]:
    if os.path.exists(oof_path) and os.path.exists(test_path):
        oof = np.load(oof_path)
        tst = np.load(test_path)
        if len(oof) == len(y) and len(tst) == len(test):
            probas.append((name, oof, tst))
            print(f"{name}: OOF F1 = {f1_score(y, (oof > 0.5).astype(int)):.5f}")
        else:
            print(f"{name}: length mismatch, skipped")
    else:
        print(f"{name}: missing, skipped")

n = len(probas)
print(f"blending {n} models")
best_f1, best_w = 0.0, None
grid = np.arange(0.0, 1.001, 0.05)
for ws in product(grid, repeat=n):
    if abs(sum(ws) - 1.0) > 1e-9:
        continue
    ens = sum(w * p[1] for w, p in zip(ws, probas))
    f1 = f1_score(y, (ens > 0.5).astype(int))
    if f1 > best_f1:
        best_f1, best_w = f1, ws
print(f"best weights: {dict(zip([p[0] for p in probas], [round(w,2) for w in best_w]))}")
print(f"blend OOF F1 @0.5 = {best_f1:.5f}")

ens = sum(w * p[1] for w, p in zip(best_w, probas))
best_thr, best_f1_t = 0.5, 0.0
for thr in np.arange(0.40, 0.601, 0.01):
    f1 = f1_score(y, (ens > thr).astype(int))
    if f1 > best_f1_t:
        best_f1_t, best_thr = f1, thr
print(f"best thr {best_thr:.2f} -> OOF F1 {best_f1_t:.5f}")

test_ens = sum(w * p[2] for w, p in zip(best_w, probas))
pred = (test_ens > best_thr).astype(int)

def normalize(s):
    s = s.lower()
    s = re.sub(r"https?://\S+|www\.\S+", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\d+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

train_text_lbl = train.groupby("text")["target"].agg(["mean", "count"])
leak_map = {t: int(r["mean"]) for t, r in train_text_lbl.iterrows() if r["mean"] in (0.0, 1.0)}
exact_hit = test["text"].fillna("").astype(str).map(leak_map)
norm_train = train["text"].fillna("").astype(str).apply(normalize)
norm_test = test["text"].fillna("").astype(str).apply(normalize)
ntl = pd.DataFrame({"n": norm_train, "t": y}).groupby("n")["t"].agg(["mean", "count"])
norm_map = {n_: int(r["mean"]) for n_, r in ntl.iterrows() if r["mean"] in (0.0, 1.0)}
norm_hit = norm_test.map(norm_map)

mask = exact_hit.notna()
pred[mask.values] = exact_hit[mask].values.astype(int)
mask2 = norm_hit.notna() & ~exact_hit.notna()
pred[mask2.values] = norm_hit[mask2].values.astype(int)

sub = pd.DataFrame({"id": test["id"], "target": pred})
sub.to_csv("submission_final_blend5.csv", index=False)
print("saved submission_final_blend5.csv | pos rate:", pred.mean().round(4))
