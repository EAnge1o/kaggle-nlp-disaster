"""NLP Disaster Tweets - heuristic v3.
Fix SVM convergence (SGD), add OOF keyword target-encoding, ensemble
LR + SGD + NB via probability averaging, SVD variant, and near-duplicate
leak extension (normalized text matching).
"""
import re
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

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
W = word_vec.fit_transform(all_raw)
C = char_vec.fit_transform(all_raw)
tfidf = sp.hstack([W, C]).tocsr()
Xtr, Xte = tfidf[:n_tr], tfidf[n_tr:]

# SVD variant (denoise, speeds linear models)
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

m_tr = meta(train).values
m_te = meta(test).values

def build(X, m, kw):
    return sp.hstack([X, sp.csr_matrix(m.astype(float)), sp.csr_matrix(kw)]).tocsr()

Ftr = build(Xtr, m_tr, kw_tr)
Fte = build(Xte, m_te, kw_te)

skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=SEED)
folds = list(skf.split(Ftr, y))

def run_cv(name, Xa, Xb, models, use_meta=False):
    oof_proba = {k: np.zeros(len(y)) for k in models}
    for tr_idx, va_idx in folds:
        Xa_tr, Xa_va = Xa[tr_idx], Xa[va_idx]
        # OOF keyword target encoding (smoothed) inside fold
        kw_enc_tr = np.zeros(len(tr_idx))
        kw_enc_va = np.zeros(len(va_idx))
        if use_meta:
            kw_tr_col = pd.Series(train["keyword"].fillna("__nan__").values[tr_idx])
            kw_va_col = pd.Series(train["keyword"].fillna("__nan__").values[va_idx])
            global_mean = y[tr_idx].mean()
            agg = pd.DataFrame({"t": y[tr_idx]}).groupby(kw_tr_col)["t"].agg(["sum", "count"])
            enc_map = ((agg["sum"] + 10 * global_mean) / (agg["count"] + 10)).to_dict()
            kw_enc_tr = kw_tr_col.map(enc_map).fillna(global_mean).values
            kw_enc_va = kw_va_col.map(enc_map).fillna(global_mean).values
        for k, model in models.items():
            trX = sp.hstack([Xa_tr, sp.csr_matrix(kw_enc_tr.reshape(-1, 1))]).tocsr() if use_meta else Xa_tr
            vaX = sp.hstack([Xa_va, sp.csr_matrix(kw_enc_va.reshape(-1, 1))]).tocsr() if use_meta else Xa_va
            model.fit(trX, y[tr_idx])
            oof_proba[k][va_idx] = model.predict_proba(vaX)[:, 1]
    for k in models:
        f1 = f1_score(y, (oof_proba[k] > 0.5).astype(int))
        print(f"  {name} {k}: OOF F1 = {f1:.5f}")
    proba = np.mean([oof_proba[k] for k in models], axis=0)
    f1 = f1_score(y, (proba > 0.5).astype(int))
    print(f"  {name} ensemble: OOF F1 = {f1:.5f}")
    return proba

models_full = {
    "lr": LogisticRegression(C=1.0, solver="liblinear", random_state=SEED, max_iter=1000),
    "sgd": SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-4, random_state=SEED),
    "nb": MultinomialNB(alpha=0.5),
}
models_svd = {
    "lr": LogisticRegression(C=1.0, solver="liblinear", random_state=SEED, max_iter=1000),
    "sgd": SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-4, random_state=SEED),
    "nb": MultinomialNB(alpha=0.5),  # NB needs non-negative; SVD can be negative -> skip
}

print("== full tf-idf + meta + kw onehot ==")
oof_full = run_cv("full", Ftr, Fte, models_full, use_meta=True)

print("== tf-idf + SVD300 ==")
# NB on SVD: clip negatives to 0 for a simple variant
def run_cv_svd(Xa, Xb):
    oof = {k: np.zeros(len(y)) for k in ("lr", "sgd", "nb")}
    for tr_idx, va_idx in folds:
        for k, model in {
            "lr": LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
            "sgd": SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-4, random_state=SEED),
            "nb": MultinomialNB(alpha=0.5),
        }.items():
            A = Xa[tr_idx] if k != "nb" else np.clip(Xa[tr_idx], 0, None)
            B = Xa[va_idx] if k != "nb" else np.clip(Xa[va_idx], 0, None)
            model.fit(A, y[tr_idx])
            oof[k][va_idx] = model.predict_proba(B)[:, 1]
    for k in ("lr", "sgd", "nb"):
        print(f"  svd {k}: OOF F1 = {f1_score(y, (oof[k] > 0.5).astype(int)):.5f}")
    proba = np.mean([oof[k] for k in ("lr", "sgd", "nb")], axis=0)
    print(f"  svd ensemble: OOF F1 = {f1_score(y, (proba > 0.5).astype(int)):.5f}")
    return proba

print("== SVD300 models ==")
oof_svd = run_cv_svd(Str, Ste)

# pick better OOF ensemble
use_svd = f1_score(y, (oof_svd > 0.5).astype(int)) > f1_score(y, (oof_full > 0.5).astype(int))
print(f"\nchoosing {'SVD' if use_svd else 'full'} pipeline (ensemble OOF)")

# --- leaks: exact text dup + normalized near-dup ---
train_text_lbl = train.groupby("text")["target"].agg(["mean", "count"])
leak_map = {}
for txt, row in train_text_lbl.iterrows():
    if row["mean"] in (0.0, 1.0):
        leak_map[txt] = int(row["mean"])
te_texts = test["text"].fillna("").astype(str)
exact_hit = te_texts.map(leak_map)

# normalized matching: use train normalized text -> label where unambiguous
norm_train = train["text"].fillna("").astype(str).apply(normalize)
norm_test = test["text"].fillna("").astype(str).apply(normalize)
ntl = pd.DataFrame({"n": norm_train, "t": y}).groupby("n")["t"].agg(["mean", "count"])
norm_map = {}
for n_, row in ntl.iterrows():
    if row["count"] >= 1 and row["mean"] in (0.0, 1.0):
        norm_map[n_] = int(row["mean"])
norm_hit = norm_test.map(norm_map)

print(f"exact leak: {exact_hit.notna().sum()}, normalized leak: {norm_hit.notna().sum()}")

# --- final fit on full data with chosen pipeline, apply leaks, predict ---
def final_predict():
    if use_svd:
        kw_enc_col = pd.Series(train["keyword"].fillna("__nan__"))
        global_mean = y.mean()
        agg = pd.DataFrame({"t": y}).groupby(kw_enc_col)["t"].agg(["sum", "count"])
        enc_map = ((agg["sum"] + 10 * global_mean) / (agg["count"] + 10)).to_dict()
        kw_enc_tr = kw_enc_col.map(enc_map).fillna(global_mean).values
        kw_enc_te = pd.Series(test["keyword"].fillna("__nan__")).map(enc_map).fillna(global_mean).values
        A = sp.hstack([sp.csr_matrix(Str), sp.csr_matrix(kw_enc_tr.reshape(-1, 1))]).tocsr()
        B = sp.hstack([sp.csr_matrix(Ste), sp.csr_matrix(kw_enc_te.reshape(-1, 1))]).tocsr()
    else:
        kw_enc_col = pd.Series(train["keyword"].fillna("__nan__"))
        global_mean = y.mean()
        agg = pd.DataFrame({"t": y}).groupby(kw_enc_col)["t"].agg(["sum", "count"])
        enc_map = ((agg["sum"] + 10 * global_mean) / (agg["count"] + 10)).to_dict()
        kw_enc_tr = kw_enc_col.map(enc_map).fillna(global_mean).values
        kw_enc_te = pd.Series(test["keyword"].fillna("__nan__")).map(enc_map).fillna(global_mean).values
        A = sp.hstack([Ftr, sp.csr_matrix(kw_enc_tr.reshape(-1, 1))]).tocsr()
        B = sp.hstack([Fte, sp.csr_matrix(kw_enc_te.reshape(-1, 1))]).tocsr()
    proba = np.zeros(len(test))
    for k, model in {
        "lr": LogisticRegression(C=1.0, max_iter=1000, random_state=SEED),
        "sgd": SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=2000, tol=1e-4, random_state=SEED),
        "nb": MultinomialNB(alpha=0.5),
    }.items():
        AA, BB = A, B
        if k == "nb" and use_svd:
            AA, BB = np.clip(A, 0, None), np.clip(B, 0, None)
        model.fit(AA, y)
        proba += model.predict_proba(BB)[:, 1]
    proba /= 3.0
    pred = (proba > 0.5).astype(int)
    # apply leaks: exact first, then normalized where exact missing
    mask = exact_hit.notna()
    pred[mask.values] = exact_hit[mask].values.astype(int)
    mask2 = norm_hit.notna() & ~exact_hit.notna()
    pred[mask2.values] = norm_hit[mask2].values.astype(int)
    return pred

pred = final_predict()
sub = pd.DataFrame({"id": test["id"], "target": pred})
sub.to_csv("submission_ensemble_v3.csv", index=False)
print("pos rate:", pred.mean().round(4), "| saved submission_ensemble_v3.csv")
