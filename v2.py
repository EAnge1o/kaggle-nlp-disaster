"""NLP Disaster Tweets - heuristic v2.
Adds: light cleaning, separate word+char tf-idf hstack, keyword/location/count
features, and the known train/test duplicated-text leak (checked for conflicts).
"""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
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
y = train["target"].values

URL_RE = re.compile(r"https?://\S+|www\.\S+")
USER_RE = re.compile(r"@\w+")
HTML_RE = re.compile(r"&amp;|&lt;|&gt;|&quot;|&#\d+;")

def clean(s):
    if not isinstance(s, str):
        return ""
    s = s.replace("&amp;", " and ").replace("&lt;", " < ").replace("&gt;", " > ")
    s = HTML_RE.sub(" ", s)
    s = URL_RE.sub(" url ", s)
    s = USER_RE.sub(" user ", s)
    return s

X_tr_raw = train["text"].fillna("").apply(clean)
X_te_raw = test["text"].fillna("").apply(clean)
all_raw = pd.concat([X_tr_raw, X_te_raw])

word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=40000, sublinear_tf=True)
W = word_vec.fit_transform(all_raw)
C = char_vec.fit_transform(all_raw)
tfidf = sp.hstack([W, C]).tocsr()
Xtr, Xte = tfidf[: len(X_tr_raw)], tfidf[len(X_tr_raw):]

# --- metadata features (heuristic: count/structure signals) ---
def meta(df):
    t = df["text"].fillna("").astype(str)
    out = pd.DataFrame(index=df.index)
    out["n_words"] = t.str.split().str.len()
    out["n_unique"] = t.str.split().apply(lambda x: len(set(x)))
    out["n_url"] = t.str.count(r"https?://|www\.")
    out["n_hashtag"] = t.str.count(r"#\w+")
    out["n_mention"] = t.str.count(r"@\w+")
    out["n_digits"] = t.str.count(r"\d")
    out["n_punct"] = t.str.count(r"[!?]")
    out["n_upper"] = t.apply(lambda s: sum(1 for ch in s if ch.isupper()))
    out["char_len"] = t.str.len()
    out["kw_missing"] = df["keyword"].isna().astype(int)
    out["loc_missing"] = df["location"].isna().astype(int)
    out["loc_has_num"] = df["location"].fillna("").str.contains(r"\d", regex=True).astype(int)
    return out

# keyword label encoding (221 categories) - sparse one-hot
kw_cats = pd.Categorical(pd.concat([train["keyword"], test["keyword"]]).fillna("__nan__"))
kw_onehot = pd.get_dummies(kw_cats, sparse=True)
kw_tr, kw_te = kw_onehot.iloc[: len(train)].values, kw_onehot.iloc[len(train):].values

m_tr = meta(train).values
m_te = meta(test).values

def build(X, m, kw):
    return sp.hstack([X, sp.csr_matrix(m.astype(float)), sp.csr_matrix(kw)]).tocsr()

Ftr = build(Xtr, m_tr, kw_tr)
Fte = build(Xte, m_te, kw_te)

skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
folds = list(skf.split(Ftr, y))

results, oof_store = {}, {}
for name, model in [
    ("svc", LinearSVC(C=1.0, random_state=SEED, max_iter=10000)),
    ("lr", LogisticRegression(C=1.0, solver="liblinear", random_state=SEED, max_iter=1000)),
    ("nb", MultinomialNB(alpha=0.5)),
]:
    oof = np.zeros(len(y))
    for tr_idx, va_idx in folds:
        model.fit(Ftr[tr_idx], y[tr_idx])
        oof[va_idx] = model.predict(Ftr[va_idx])
    f1 = f1_score(y, oof)
    results[name] = f1
    oof_store[name] = oof
    print(f"{name}: OOF F1 = {f1:.5f}")

best = max(results, key=results.get)
print(f"\nbest = {best} ({results[best]:.5f})")

# --- known leak: duplicated texts between train and test ---
# copy train label only when unambiguous
train_text_lbl = train.groupby("text")["target"].agg(["mean", "count"])
leak_map = {}
for txt, row in train_text_lbl.iterrows():
    if row["count"] > 0 and row["mean"] in (0.0, 1.0):
        leak_map[txt] = int(row["mean"])
te_texts = test["text"].fillna("").astype(str)
pred = np.array([])  # placeholder
leak_hit = te_texts.map(leak_map)
n_leak = leak_hit.notna().sum()
print(f"unambiguous leak matches: {n_leak} / {len(test)}")
print("leak label dist:", leak_hit.dropna().astype(int).value_counts().to_dict())

model = {
    "svc": LinearSVC(C=1.0, random_state=SEED, max_iter=10000),
    "lr": LogisticRegression(C=1.0, solver="liblinear", random_state=SEED, max_iter=1000),
    "nb": MultinomialNB(alpha=0.5),
}[best]
model.fit(Ftr, y)
pred = model.predict(Fte).astype(int)

use_leak = True
if use_leak:
    mask = leak_hit.notna()
    pred[mask.values] = leak_hit[mask].values.astype(int)
    print("leak applied, pos rate now:", pred.mean().round(4))

sub = pd.DataFrame({"id": test["id"], "target": pred})
sub.to_csv(f"submission_{best}_v2.csv", index=False)
print("saved submission_" + best + "_v2.csv")
