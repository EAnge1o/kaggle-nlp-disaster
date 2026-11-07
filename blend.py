"""Blend transformer (distilbert) + TF-IDF ensemble probabilities.
Weight search on OOF, apply exact/norm leak, save submission.
"""
import re
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["target"].values

oof_db = np.load("downloads_cloud/oof_proba.npy")
test_db = np.load("downloads_cloud/test_proba.npy")
oof_tf = np.load("tfidf_oof.npy")
test_tf = np.load("tfidf_test.npy")

print(f"distilbert OOF F1: {f1_score(y, (oof_db > 0.5).astype(int)):.5f}")
print(f"tfidf OOF F1:      {f1_score(y, (oof_tf > 0.5).astype(int)):.5f}")

best_f1, best_w = 0.0, None
for w in np.arange(0.0, 1.01, 0.05):
    ens = w * oof_db + (1 - w) * oof_tf
    f1 = f1_score(y, (ens > 0.5).astype(int))
    if f1 > best_f1:
        best_f1, best_w = f1, round(w, 2)
print(f"best blend: w_distilbert={best_w}, OOF F1 = {best_f1:.5f}")

test_ens = best_w * test_db + (1 - best_w) * test_tf

# threshold search on OOF blend
oof_ens = best_w * oof_db + (1 - best_w) * oof_tf
best_thr, best_f1_t = 0.5, 0.0
for thr in np.arange(0.40, 0.61, 0.01):
    f1 = f1_score(y, (oof_ens > thr).astype(int))
    if f1 > best_f1_t:
        best_f1_t, best_thr = f1, thr
print(f"best thr {best_thr:.2f} -> OOF F1 {best_f1_t:.5f}")

pred = (test_ens > best_thr).astype(int)

# leaks (same as v5)
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
sub.to_csv("submission_blend.csv", index=False)
print("saved submission_blend.csv | pos rate:", pred.mean().round(4))
