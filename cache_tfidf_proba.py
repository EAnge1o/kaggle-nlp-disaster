"""Cache v5-style TF-IDF ensemble probabilities (OOF + test) for blending
with the transformer model. Reuses v5 pipeline: weighted lr/svc/sgd with
pseudo-label augmentation.
"""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold

SEED = 42
NFOLD = 5

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = train["target"].values
n_tr = len(train)

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

def normalize(s):
    s = clean(s).lower()
    s = re.sub(r"https?://\S+|www\.\S+", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    s = re.sub(r"\d+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

X_tr_raw = train["text"].fillna("").apply(clean)
X_te_raw = test["text"].fillna("").apply(clean)
all_raw = pd.concat([X_tr_raw, X_te_raw])

word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True)
char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_features=40000, sublinear_tf=True)
tfidf = sp.hstack([word_vec.fit_transform(all_raw), char_vec.fit_transform(all_raw)]).tocsr()
Xtr, Xte = tfidf[:n_tr], tfidf[n_tr:]
svd = TruncatedSVD(n_components=300, random_state=SEED)
S = svd.fit_transform(tfidf)
Str, Ste = S[:n_tr], S[n_tr:]

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

kw_cats = pd.Categorical(pd.concat([train["keyword"], test["keyword"]]).fillna("__nan__"))
kw_onehot = pd.get_dummies(kw_cats, sparse=True)
kw_tr, kw_te = kw_onehot.iloc[:n_tr].values, kw_onehot.iloc[n_tr:].values
m_tr, m_te = meta(train).values, meta(test).values

def build(X, m, kw):
    return sp.hstack([X, sp.csr_matrix(m.astype(float)), sp.csr_matrix(kw)]).tocsr()

Ftr = build(Xtr, m_tr, kw_tr)
Fte = build(Xte, m_te, kw_te)
skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
folds = list(skf.split(Ftr, y))

# base OOF probas (no pseudo) for weights
def cv_proba(fn, X):
    p = np.zeros(len(y))
    for tr_idx, va_idx in folds:
        p[va_idx] = fn(X, tr_idx, va_idx)
    return p

def lr_fn(X, tr_idx, va_idx):
    m_ = LogisticRegression(C=1.0, solver="liblinear", random_state=SEED, max_iter=2000).fit(X[tr_idx], y[tr_idx])
    return m_.predict_proba(X[va_idx])[:, 1]

def svc_fn(X, tr_idx, va_idx):
    m_ = LinearSVC(C=0.1, dual=False, tol=1e-3, max_iter=20000, random_state=SEED).fit(X[tr_idx], y[tr_idx])
    d = m_.decision_function(X[va_idx])
    return 1 / (1 + np.exp(-d))

def sgd_fn(X, tr_idx, va_idx):
    m_ = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-4, random_state=SEED).fit(X[tr_idx], y[tr_idx])
    return m_.predict_proba(X[va_idx])[:, 1]

p_lr = cv_proba(lr_fn, Ftr)
p_svc = cv_proba(svc_fn, Ftr)
p_sgd = cv_proba(sgd_fn, Str)

# pseudo-label augmentation as v5: leak + high-conf
train_text_lbl = train.groupby("text")["target"].agg(["mean", "count"])
leak_map = {t: int(r["mean"]) for t, r in train_text_lbl.iterrows() if r["mean"] in (0.0, 1.0)}
exact_hit = test["text"].fillna("").astype(str).map(leak_map)
norm_train = train["text"].fillna("").astype(str).apply(normalize)
norm_test = test["text"].fillna("").astype(str).apply(normalize)
ntl = pd.DataFrame({"n": norm_train, "t": y}).groupby("n")["t"].agg(["mean", "count"])
norm_map = {n_: int(r["mean"]) for n_, r in ntl.iterrows() if r["mean"] in (0.0, 1.0)}
norm_hit = norm_test.map(norm_map)
leak_label = exact_hit.combine_first(norm_hit)

m_lr0 = LogisticRegression(C=1.0, solver="liblinear", random_state=SEED, max_iter=2000).fit(Ftr, y)
m_svc0 = LinearSVC(C=0.1, dual=False, tol=1e-3, max_iter=20000, random_state=SEED).fit(Ftr, y)
m_sgd0 = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-4, random_state=SEED).fit(Str, y)
q0 = (0.8 * m_lr0.predict_proba(Fte)[:, 1]
      + 0.1 / (1 + np.exp(-m_svc0.decision_function(Fte)))
      + 0.1 * m_sgd0.predict_proba(Ste)[:, 1])
pseudo = leak_label.copy()
conf = (q0 > 0.9) | (q0 < 0.1)
pseudo = pseudo.combine_first(pd.Series(np.where(conf, (q0 > 0.5).astype(int), np.nan), index=test.index))
add_mask = pseudo.notna()
y_plus = np.concatenate([y, pseudo[add_mask].values.astype(int)])
all_plus = pd.concat([X_tr_raw, X_te_raw[add_mask.values]])
Wp = word_vec.transform(all_plus)
Cp = char_vec.transform(all_plus)
tfidf_plus = sp.hstack([Wp, Cp]).tocsr()
kw_plus = sp.vstack([kw_tr, kw_te[add_mask.values]])
m_plus = np.vstack([m_tr, m_te[add_mask.values]])
Fplus = build(tfidf_plus, m_plus, kw_plus)

# OOF probas on original train with augmented model (v5 style)
oof_p_lr = np.zeros(len(y))
oof_p_svc = np.zeros(len(y))
skf2 = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
for tr_idx, va_idx in skf2.split(Fplus, y_plus):
    va_orig = va_idx[va_idx < len(y)]
    m_ = LogisticRegression(C=1.0, solver="liblinear", random_state=SEED, max_iter=2000).fit(Fplus[tr_idx], y_plus[tr_idx])
    oof_p_lr[va_orig] = m_.predict_proba(Fplus[va_orig])[:, 1]
    m_ = LinearSVC(C=0.1, dual=False, tol=1e-3, max_iter=20000, random_state=SEED).fit(Fplus[tr_idx], y_plus[tr_idx])
    d = m_.decision_function(Fplus[va_orig])
    oof_p_svc[va_orig] = 1 / (1 + np.exp(-d))
oof_tf = 0.8 * oof_p_lr + 0.1 * oof_p_svc + 0.1 * p_sgd

m_lr_f = LogisticRegression(C=1.0, solver="liblinear", random_state=SEED, max_iter=2000).fit(Fplus, y_plus)
m_svc_f = LinearSVC(C=0.1, dual=False, tol=1e-3, max_iter=20000, random_state=SEED).fit(Fplus, y_plus)
m_sgd_f = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-4, random_state=SEED).fit(Str, y)
test_tf = (0.8 * m_lr_f.predict_proba(Fte)[:, 1]
           + 0.1 / (1 + np.exp(-m_svc_f.decision_function(Fte)))
           + 0.1 * m_sgd_f.predict_proba(Ste)[:, 1])

np.save("tfidf_oof.npy", oof_tf)
np.save("tfidf_test.npy", test_tf)
from sklearn.metrics import f1_score
print("tfidf ensemble OOF F1:", round(f1_score(y, (oof_tf > 0.5).astype(int)), 5))
print("saved tfidf_oof.npy / tfidf_test.npy")
