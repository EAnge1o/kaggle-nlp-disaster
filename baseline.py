"""NLP Disaster Tweets - heuristic baseline v1.
TF-IDF (word + char n-grams) with linear models, 5-fold stratified CV.
Fast feasible solution first, iterate later.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

SEED = 42
NFOLD = 5

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

X_tr = train["text"].fillna("")
X_te = test["text"].fillna("")
y = train["target"].values

# word 1-2 grams + char 2-5 grams, sublinear tf, common heuristic text config
vec = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
    strip_accents="unicode",
)
tfidf = vec.fit_transform(pd.concat([X_tr, X_te]))
Xtr = tfidf[: len(X_tr)]
Xte = tfidf[len(X_tr):]

skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
folds = list(skf.split(Xtr, y))

results = {}
oof_store = {}
for name, model in [
    ("lr", LogisticRegression(C=1.0, solver="liblinear", random_state=SEED, max_iter=1000)),
    ("svc", LinearSVC(C=1.0, random_state=SEED, max_iter=5000)),
    ("nb", MultinomialNB(alpha=0.5)),
]:
    oof = np.zeros(len(y))
    for tr_idx, va_idx in folds:
        model.fit(Xtr[tr_idx], y[tr_idx])
        p = model.predict(Xtr[va_idx])
        oof[va_idx] = p
    f1 = f1_score(y, oof)
    results[name] = f1
    oof_store[name] = oof
    print(f"{name}: OOF F1 = {f1:.5f}")

best = max(results, key=results.get)
print(f"\nbest = {best} ({results[best]:.5f})")

# refit best on full data, predict test
model = {
    "lr": LogisticRegression(C=1.0, solver="liblinear", random_state=SEED, max_iter=1000),
    "svc": LinearSVC(C=1.0, random_state=SEED, max_iter=5000),
    "nb": MultinomialNB(alpha=0.5),
}[best]
model.fit(Xtr, y)
pred = model.predict(Xte).astype(int)

sub = pd.DataFrame({"id": test["id"], "target": pred})
sub.to_csv(f"submission_{best}_v1.csv", index=False)
print("submission rows:", len(sub), "| pos rate:", sub["target"].mean().round(4))
print("saved submission_" + best + "_v1.csv")
