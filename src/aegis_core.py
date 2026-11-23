# =====================================================================
# aegis_core.py  —  AEGIS-MD core library
# Adaptive Evidence-Gap Investigation of Safety signals for Medical Devices
#
# Shared by all notebooks (each notebook writes this file to disk first,
# so every notebook is self-contained on Kaggle).
# =====================================================================
import os, re, glob, json, math, time, warnings
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
from scipy import stats, sparse

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Configuration (change here, or override in a notebook before use)
# ---------------------------------------------------------------------
CFG = dict(
    DATA_START="2014-01",   # first month of MAUDE data pulled
    DATA_END="2025-12",     # last month of MAUDE data pulled
    PANEL_START="2016-01",  # first evaluation month t
    PANEL_END="2025-06",    # last evaluation month t (needs HORIZON months of labels after it)
    TRAIN_END="2020-12",    # t <= TRAIN_END  -> train
    VAL_END="2022-06",      # TRAIN_END < t <= VAL_END -> validation ; later -> test
    HORIZON=6,              # predict a new recall initiated in (t, t+HORIZON]
    WASHOUT=6,              # drop t if a recall of that code was initiated in (t-WASHOUT, t]
    RECENT=3,               # recent window = months t-2..t
    BASELINE=12,            # baseline window = 12 months before the recent window
    LABEL_CLASSES=["Class I"],   # ["Class I"] (primary) or ["Class I","Class II"]
    K_PER_GAP=10,           # documents read per evidence-gap retrieval
    SEED=42,
)

DEEP_SOURCES = ["mechanism", "investigation", "batch", "precedent"]
TIER0_SOURCES = ["statistical", "temporal", "ml_anomaly", "severity", "problem_shift"]

# Gap specifications: what each evidence gap needs, and how it is queried.
GAP_SPECS = {
    "mechanism": dict(
        question="Is there a consistent, specific failure mechanism behind the recent reports?",
        proto="The device failed or malfunctioned in a specific, repeatable way; the report describes the failure mechanism and the component involved.",
        terms=["failure", "failed", "malfunction", "broke", "crack", "leak", "error", "alarm", "component"],
        use_emerging=True),
    "investigation": dict(
        question="What did the manufacturer's own investigation of returned devices conclude?",
        proto="The manufacturer evaluated the returned device; the investigation confirmed the reported failure and identified the root cause.",
        terms=["evaluation", "evaluated", "investigation", "returned", "analysis", "confirmed", "root", "cause", "found", "reproduced"],
        use_emerging=False),
    "batch": dict(
        question="Are the affected units concentrated in specific lots, batches or models?",
        proto="Multiple units from the same lot or manufacturing batch were affected by the same manufacturing defect.",
        terms=["lot", "batch", "manufacturing", "units", "same", "defect", "production", "nonconformance", "supplier"],
        use_emerging=False),
    "precedent": dict(
        question="Does this failure resemble defects that previously led to recalls?",
        proto="The failure resembles a known product defect that previously led to a corrective action or product recall.",
        terms=["defect", "design", "manufacturing", "failure", "component"],
        use_emerging=True),
}
GENERIC_QUERY = dict(
    proto="A report of an adverse event or problem involving the medical device.",
    terms=["adverse", "event", "problem", "malfunction", "patient", "device"])

# ---------------------------------------------------------------------
# Month helpers  (month index mi = year*12 + month-1)
# ---------------------------------------------------------------------
def ym(s):
    y, m = map(int, str(s)[:7].split("-")); return y * 12 + m - 1

def ym_str(mi):
    mi = int(mi); return f"{mi // 12}-{mi % 12 + 1:02d}"

def to_mi(series):
    d = pd.to_datetime(series, errors="coerce", format="mixed") if hasattr(pd, "to_datetime") else series
    return (d.dt.year * 12 + d.dt.month - 1).astype("float")

def mi_from_yyyymmdd(s):
    s = str(s or "")
    s = s.replace("-", "")
    if len(s) < 6 or not s[:6].isdigit():
        return np.nan
    return int(s[:4]) * 12 + int(s[4:6]) - 1

def split_of(mi, cfg=CFG):
    if mi <= ym(cfg["TRAIN_END"]): return "train"
    if mi <= ym(cfg["VAL_END"]): return "val"
    return "test"

# ---------------------------------------------------------------------
# File discovery (Kaggle: /kaggle/working and attached inputs)
# ---------------------------------------------------------------------
def data_roots():
    roots = []
    if os.environ.get("AEGIS_DIR"): roots.append(os.environ["AEGIS_DIR"])
    roots += ["/kaggle/working", "/kaggle/input", "."]
    return [r for r in roots if os.path.isdir(r)]

def find_file(name, required=True):
    for r in data_roots():
        p = os.path.join(r, name)
        if os.path.exists(p): return p
        hits = glob.glob(os.path.join(r, "**", name), recursive=True)
        if hits: return sorted(hits, key=len)[0]
    if required:
        raise FileNotFoundError(f"{name} not found. Attach the previous notebook's output as an input (Add Input -> Notebook Output).")
    return None

def out_dir():
    d = os.environ.get("AEGIS_OUT") or ("/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
    os.makedirs(d, exist_ok=True); return d

# ---------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------
STOP = set("""a an the and or of to in on for with without by from at as is are was were be been being this that these those it its
into than then there their they them he she his her patient patients device devices reported report reports reportedly per also
not no yes any all some such which who whom what when where while during after before about over under between via
had has have having did does do done will would could should may might can cannot unable however therefore further
information additional event events date dates time times use used using user customer facility hospital
mfr manufacturer company firm product products model lot serial number numbers unknown na n/a""".split())
TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]{2,}")

def tokenize(s, keep_stop=False):
    toks = TOKEN_RE.findall(str(s or "").lower())
    return toks if keep_stop else [t for t in toks if t not in STOP]

# Leakage guard: MAUDE narratives can be updated by supplemental reports
# after a recall. Every sentence that mentions a recall / field action is removed.
RECALL_MENTION = re.compile(
    r"(recall|recalled|field safety|field action|field correction|correction and removal|"
    r"\bfsca\b|\bfsn\b|z-\d{3,4}-\d{4}|urgent medical device|advisory notice|removal notice)", re.I)
SENT_SPLIT = re.compile(r"(?<=[\.\!\?;])\s+")

def scrub_recall_mentions(text):
    if not text: return "", 0
    sents = SENT_SPLIT.split(str(text))
    keep = [s for s in sents if not RECALL_MENTION.search(s)]
    return " ".join(keep), len(sents) - len(keep)

# Regex stance (CPU fallback when no NLI model is available)
CONFIRM_RE = re.compile(
    r"(was|were|has been|have been|is) confirmed|confirmed the (reported|complaint|failure|issue|allegation)|"
    r"root cause (was|is|has been) (determined|identified|attributed|found)|"
    r"(failure|problem|issue|defect) (was|is|were) (found|observed|reproduced|duplicated|verified)|"
    r"(found|observed) to be (cracked|broken|defective|damaged|leaking|fractured|separated|occluded)|"
    r"(evaluation|analysis|investigation|inspection) (found|revealed|identified|determined|showed)", re.I)
REFUTE_RE = re.compile(
    r"could not be (confirmed|reproduced|duplicated|determined|verified)|"
    r"no (device )?(malfunction|anomal\w*|defect|problem|issue|fault|failure)s? (was|were )?(found|identified|observed|detected|noted)|"
    r"(was|were|has not been|have not been) not (been )?returned|not returned for (evaluation|analysis)|unable to (confirm|reproduce|duplicate|verify)|"
    r"functioned (as intended|properly|normally|as designed)|(met|meets|within) (all )?(specification|specifications)|"
    r"\b(user|use) error\b|not confirmed|unconfirmed|no evidence of (a )?(device )?(malfunction|defect)", re.I)

def regex_stance(text):
    t = str(text or "")
    if not t.strip(): return np.nan, np.nan
    c = 0.85 if CONFIRM_RE.search(t) else 0.10
    r = 0.85 if REFUTE_RE.search(t) else 0.10
    return c, r

# ---------------------------------------------------------------------
# Sparse BM25 index over all records (vectorised; fast per-case scoring)
# ---------------------------------------------------------------------
class BM25Matrix:
    def __init__(self, texts, k1=1.2, b=0.75, min_df=3, max_features=60000):
        from sklearn.feature_extraction.text import CountVectorizer
        self.cv = CountVectorizer(tokenizer=tokenize, lowercase=False, token_pattern=None,
                                  min_df=min_df, max_features=max_features)
        tf = self.cv.fit_transform(texts).tocsr().astype(np.float32)
        N = tf.shape[0]
        df = np.diff(tf.tocsc().indptr)
        idf = np.log(1 + (N - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        avgdl = max(dl.mean(), 1.0)
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)
        w = idf[tf.col] * tf.data * (k1 + 1) / denom
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
        self.vocab = self.cv.vocabulary_

    def qvec(self, tokens):
        idx = sorted({self.vocab[t] for t in tokens if t in self.vocab})
        v = np.zeros(self.W.shape[1], dtype=np.float32); v[idx] = 1.0
        return v

    def score(self, rows, qv):
        return np.asarray(self.W[rows] @ qv).ravel()

def _minmax(x):
    x = np.asarray(x, dtype=np.float32)
    lo, hi = (x.min(), x.max()) if len(x) else (0, 0)
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)

def hybrid_retrieve(rows, bm25, emb, q_tokens, proto_vec, k, alpha=0.5):
    """Hybrid sparse (BM25) + dense (embedding) retrieval restricted to candidate rows."""
    rows = np.asarray(rows)
    if len(rows) <= k: return rows
    s_bm = _minmax(bm25.score(rows, bm25.qvec(q_tokens)))
    s_dn = _minmax(emb[rows].astype(np.float32) @ proto_vec.astype(np.float32))
    s = alpha * s_bm + (1 - alpha) * s_dn
    return rows[np.argsort(-s, kind="stable")[:k]]

# ---------------------------------------------------------------------
# Tier-0 (routine surveillance) detectors
# ---------------------------------------------------------------------
def mann_kendall(x):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 4 or np.all(x == x[0]): return 0.0, 0.0, 1.0
    s = 0.0
    for i in range(n - 1): s += np.sign(x[i + 1:] - x[i]).sum()
    _, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - np.sum(counts * (counts - 1) * (2 * counts + 5))) / 18.0
    z = 0.0 if var <= 0 else ((s - 1) / math.sqrt(var) if s > 0 else ((s + 1) / math.sqrt(var) if s < 0 else 0.0))
    tau = s / (0.5 * n * (n - 1))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return tau, z, p

def sen_slope(x):
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 3: return 0.0
    sl = [(x[j] - x[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    return float(np.median(sl))

def jsd(p_counts, q_counts):
    keys = set(p_counts) | set(q_counts)
    if not keys: return 0.0
    p = np.array([p_counts.get(k, 0) for k in keys], float) + 0.5
    q = np.array([q_counts.get(k, 0) for k in keys], float) + 0.5
    p /= p.sum(); q /= q.sum(); m = 0.5 * (p + q)
    return float(0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m)))

def tier0_for_code(n_by_month, months, agg, cfg=CFG):
    """Tier-0 evidence for every month of one product code.
    n_by_month: dict mi->total reports (openFDA counts); agg: dict mi->dict of sample aggregates."""
    R, B = cfg["RECENT"], cfg["BASELINE"]
    rows, cusum = [], 0.0
    for t in months:
        rec = [n_by_month.get(m, 0) for m in range(t - R + 1, t + 1)]
        base = [n_by_month.get(m, 0) for m in range(t - R - B + 1, t - R + 1)]
        mu = max(np.mean(base), 0.25)
        n_rec = sum(rec)
        p_pois = stats.poisson.sf(n_rec - 1, mu * R)
        last12 = [n_by_month.get(m, 0) for m in range(t - 11, t + 1)]
        tau, z_mk, p_mk = mann_kendall(last12)
        slope = sen_slope(last12) / (np.mean(last12) + 1.0)
        persist = sum(stats.poisson.sf(n_by_month.get(m, 0) - 1, mu) < 0.05 for m in range(t - 5, t + 1))
        n_t = n_by_month.get(t, 0)
        cusum = max(0.0, cusum + (n_t - mu) / math.sqrt(mu + 1.0) - 0.5)
        # sample-based structured aggregates (severity, problem-code mix, reporting reliability)
        def pool(ms):
            out = dict(n=0, serious=0, mfr_src=0, text=0, prob=Counter())
            for m in ms:
                a = agg.get(m)
                if a is None: continue
                out["n"] += a["n"]; out["serious"] += a["serious"]; out["mfr_src"] += a["mfr_src"]
                out["text"] += a["text"]; out["prob"].update(a["prob"])
            return out
        w = pool(range(t - R + 1, t + 1)); b = pool(range(t - R - B + 1, t - R + 1))
        p1 = (w["serious"] + 0.5) / (w["n"] + 1); p0 = (b["serious"] + 0.5) / (b["n"] + 1)
        pp = (w["serious"] + b["serious"] + 1) / (w["n"] + b["n"] + 2)
        se = math.sqrt(pp * (1 - pp) * (1 / (w["n"] + 1) + 1 / (b["n"] + 1)))
        rows.append(dict(
            mi=t, n_recent=n_rec, base_mu=mu,
            pois_score=min(-math.log10(max(p_pois, 1e-12)), 12.0),
            log_ratio=math.log((n_rec + 0.5) / (mu * R + 0.5)),
            mk_z=z_mk, mk_tau=tau, sen_norm=slope, persist=persist, cusum=cusum,
            sev_z=(p1 - p0) / se if se > 0 else 0.0, sev_share=p1,
            prob_jsd=jsd(w["prob"], b["prob"]),
            reliab=(w["mfr_src"] + 1) / (w["n"] + 2), completeness=(w["text"] + 1) / (w["n"] + 2),
            n_sample_recent=w["n"]))
    return pd.DataFrame(rows)

TIER0_GROUPS = {
    "statistical": ["pois_score", "log_ratio", "cusum"],
    "temporal": ["mk_z", "sen_norm", "persist"],
    "ml_anomaly": ["if_score"],
    "severity": ["sev_z", "sev_share"],
    "problem_shift": ["prob_jsd"],
}

def emerging_terms(win_prob, base_prob, topn=3):
    tw = sum(win_prob.values()) + 1; tb = sum(base_prob.values()) + 1
    sc = {k: math.log((v + 1) / tw) - math.log((base_prob.get(k, 0) + 1) / tb) for k, v in win_prob.items()}
    top = [k for k, _ in sorted(sc.items(), key=lambda kv: -kv[1])[:topn]]
    toks = []
    for k in top: toks += tokenize(k)
    return top, toks

# ---------------------------------------------------------------------
# Deep evidence extracted from a set of retrieved documents S
# ---------------------------------------------------------------------
DEEP_FEATURES = {
    "mechanism": ["mech_coh", "mech_shift"],
    "investigation": ["inv_confirm", "inv_refute", "inv_cov"],
    "batch": ["lot_hhi", "model_hhi", "lot_cov"],
    "precedent": ["prec_top", "prec_max"],
}

def _hhi(values):
    v = [x for x in values if x]
    if len(v) < 2: return 0.0
    c = np.array(list(Counter(v).values()), float); s = c / c.sum()
    return float((s ** 2).sum())

def deep_evidence(S, R, ctx, sources=DEEP_SOURCES):
    """S: array of record row-ids; R: dict of record arrays; ctx: case context (baseline centroid etc.).
    Returns dict feature->value plus per-source reliability rho_<source>."""
    out = {}
    S = np.asarray(S, dtype=int)
    if len(S) == 0:
        for s in sources:
            for f in DEEP_FEATURES[s]: out[f] = np.nan
            out["rho_" + s] = 0.0
        return out
    rho = 0.5 + 0.5 * float(np.mean(R["has_text"][S] & R["mand_src"][S]))
    if "mechanism" in sources:
        E = R["emb"][S].astype(np.float32)
        c = E.mean(0); cn = c / (np.linalg.norm(c) + 1e-9)
        out["mech_coh"] = float((E @ cn).mean()) - ctx["base_coh"]
        out["mech_shift"] = float(1 - cn @ ctx["base_cen"]) if ctx["base_cen"] is not None else 0.0
        out["rho_mechanism"] = rho
    if "investigation" in sources:
        has = R["has_mfr"][S]
        if has.any():
            out["inv_confirm"] = float(np.nanmean(R["p_confirm"][S][has]))
            out["inv_refute"] = float(np.nanmean(R["p_refute"][S][has]))
        else:
            out["inv_confirm"] = out["inv_refute"] = np.nan
        out["inv_cov"] = float(has.mean())
        out["rho_investigation"] = rho
    if "batch" in sources:
        lots = R["lot"][S]; models = R["model"][S]
        out["lot_hhi"] = _hhi(lots); out["model_hhi"] = _hhi(models)
        out["lot_cov"] = float(np.mean([bool(x) for x in lots]))
        out["rho_batch"] = rho
    if "precedent" in sources:
        ps = np.sort(R["prec_sim"][S])[::-1]
        ps = ps[np.isfinite(ps)]
        out["prec_top"] = float(ps[:3].mean()) if len(ps) else np.nan
        out["prec_max"] = float(ps[0]) if len(ps) else np.nan
        out["rho_precedent"] = rho
    return out

LEVEL_FEATURES = {"mech_coh", "inv_confirm", "inv_refute", "lot_hhi", "model_hhi", "prec_top", "prec_max"}

def contrastive_gap_evidence(S_w, S_b, R, ctx, source, rate=None):
    """Query-conditioned contrastive evidence: the same gap query is run on the recent window
    (S_w) and on the code's own baseline window (S_b); level features are reported as
    window-minus-baseline differences so that query-induced selection bias cancels out."""
    fw = deep_evidence(S_w, R, ctx, sources=[source])
    fb = deep_evidence(S_b, R, ctx, sources=[source]) if len(S_b) else None
    out = {}
    for f in DEEP_FEATURES[source]:
        v = fw.get(f, np.nan)
        if f in LEVEL_FEATURES and fb is not None and np.isfinite(fb.get(f, np.nan)) and np.isfinite(v):
            v = v - fb[f]
        out[f] = v
    if source == "mechanism" and len(S_w) and len(S_b):
        cw = R["emb"][S_w].astype(np.float32).mean(0); cb = R["emb"][S_b].astype(np.float32).mean(0)
        out["mech_shift"] = float(1 - (cw @ cb) / (np.linalg.norm(cw) * np.linalg.norm(cb) + 1e-9))
    out["rate_" + source] = rate if rate is not None else np.nan
    out["rho_" + source] = fw.get("rho_" + source, 0.0)
    return out

def query_rate_shift(win, base, emb, proto, thr):
    """Change in the share of reports that match a gap query (dense similarity above a fixed
    corpus-level threshold) between the recent and the baseline window. No reading needed."""
    if len(win) == 0: return np.nan
    hw = float((emb[win].astype(np.float32) @ proto >= thr).sum())
    hb = float((emb[base].astype(np.float32) @ proto >= thr).sum()) if len(base) else 0.0
    return math.log((hw + 0.5) / (len(win) + 1)) - math.log((hb + 0.5) / (len(base) + 1))

CONTRAST_FEATURES = {s: DEEP_FEATURES[s] + ["rate_" + s] for s in DEEP_FEATURES}

# ---------------------------------------------------------------------
# Reliability-discounted Bayesian evidence fusion + Value-of-Information
# ---------------------------------------------------------------------
def _logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def _sigmoid(z): return 1 / (1 + np.exp(-z))

class SourceCalibrator:
    """Maps one evidence source's raw features to a log-likelihood ratio (LLR)
    relative to the base rate: LLR = logit(P(y|x)) - logit(a)."""
    def __init__(self, feats, C=0.5):
        self.feats, self.C = feats, C
    def _prep(self, X):
        X = X[self.feats].astype(float).copy()
        miss = X.isna().all(axis=1).values
        X = X.fillna(self.med)
        return X.values, miss
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        self.med = X[self.feats].median()
        self.med = self.med.fillna(0.0)
        Xv, miss = self._prep(X)
        self.sc = StandardScaler().fit(Xv[~miss])
        self.lr = LogisticRegression(C=self.C, max_iter=2000).fit(self.sc.transform(Xv[~miss]), y[~miss])
        self.la = _logit(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        return self
    def llr(self, X):
        Xv, miss = self._prep(X)
        p = self.lr.predict_proba(self.sc.transform(Xv))[:, 1]
        l = _logit(p) - self.la
        l[miss] = 0.0          # absent evidence carries no information
        return np.clip(l, -6.0, 6.0)   # bound any single source's influence

class EvidenceFusion:
    """z = b0 + sum_k w_k * rho_k * LLR_k   (reliability-discounted log-odds pooling).
    Calibrators are fit on TRAIN; pooling weights on VALIDATION; VoI uses out-of-fold
    class-conditional LLR distributions from TRAIN."""
    def __init__(self, groups, use_rho=True, C=0.5):
        self.groups, self.use_rho, self.C = groups, use_rho, C
    def _L(self, X, rho=None):
        L = np.column_stack([self.cal[s].llr(X) for s in self.groups])
        if self.use_rho and rho is not None:
            L = L * rho
        return L
    def _rho(self, X):
        return np.column_stack([X["rho_" + s].fillna(1.0).values if ("rho_" + s) in X else np.ones(len(X))
                                for s in self.groups])
    def fit(self, Xtr, ytr, Xva, yva, groups_tr=None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupKFold
        self.cal = {s: SourceCalibrator(f, self.C).fit(Xtr, ytr) for s, f in self.groups.items()}
        Lva = self._L(Xva, self._rho(Xva))
        self.stack = LogisticRegression(C=1.0, max_iter=2000).fit(Lva, yva)
        self.w = self.stack.coef_.ravel(); self.b0 = float(self.stack.intercept_[0])
        # out-of-fold LLR samples (train) for VoI predictive distributions
        oof = np.zeros((len(Xtr), len(self.groups)))
        g = groups_tr if groups_tr is not None else np.arange(len(Xtr)) % 5
        for tr, te in GroupKFold(n_splits=5).split(Xtr, ytr, g):
            for j, (s, f) in enumerate(self.groups.items()):
                c = SourceCalibrator(f, self.C).fit(Xtr.iloc[tr], ytr[tr])
                oof[te, j] = c.llr(Xtr.iloc[te])
        oof = oof * (self._rho(Xtr) if self.use_rho else 1.0)
        rng = np.random.default_rng(0)
        self.samples = {}
        for j, s in enumerate(self.groups):
            for yv in (0, 1):
                v = oof[ytr == yv, j]; v = v[v != 0] if (v != 0).any() else v
                self.samples[(s, yv)] = rng.choice(v, size=min(400, len(v)), replace=len(v) < 400) if len(v) else np.zeros(1)
        return self
    def contributions(self, X):
        return self._L(X, self._rho(X)) * self.w
    def predict_logit(self, X, mask=None):
        C = self.contributions(X)
        if mask is not None: C = C * mask
        return self.b0 + C.sum(1)
    def predict(self, X, mask=None):
        return _sigmoid(self.predict_logit(X, mask))

def bayes_risk(p, tau):
    return np.minimum(tau * (1 - p), (1 - tau) * p)

def voi(z, j_name, fusion, tau, j_idx):
    """Expected reduction in Bayes decision risk if evidence source j is acquired."""
    p = _sigmoid(z)
    l1 = fusion.samples[(j_name, 1)]; l0 = fusion.samples[(j_name, 0)]
    w = fusion.w[j_idx]
    e1 = bayes_risk(_sigmoid(z + w * l1), tau).mean()
    e0 = bayes_risk(_sigmoid(z + w * l0), tau).mean()
    return float(bayes_risk(p, tau) - (p * e1 + (1 - p) * e0))

def entropy_gain(z, j_name, fusion, j_idx):
    def H(p): p = np.clip(p, 1e-9, 1 - 1e-9); return -(p * np.log(p) + (1 - p) * np.log(1 - p))
    p = _sigmoid(z); w = fusion.w[j_idx]
    e1 = H(_sigmoid(z + w * fusion.samples[(j_name, 1)])).mean()
    e0 = H(_sigmoid(z + w * fusion.samples[(j_name, 0)])).mean()
    return float(H(p) - (p * e1 + (1 - p) * e0))

# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------
def metrics(y, p):
    from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
    y = np.asarray(y); p = np.asarray(p)
    if y.sum() == 0 or y.sum() == len(y): return dict(AUPRC=np.nan, AUROC=np.nan, Brier=np.nan)
    return dict(AUPRC=average_precision_score(y, p), AUROC=roc_auc_score(y, p), Brier=brier_score_loss(y, np.clip(p, 0, 1)))

def cluster_bootstrap_diff(y, p_a, p_b, clusters, n_boot=1000, seed=0, metric="AUPRC"):
    """Paired cluster (product-code) bootstrap of metric(p_a) - metric(p_b)."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    f = average_precision_score if metric == "AUPRC" else roc_auc_score
    rng = np.random.default_rng(seed)
    cl = np.asarray(clusters); u = np.unique(cl)
    idx = {c: np.where(cl == c)[0] for c in u}
    diffs = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[c] for c in rng.choice(u, len(u), replace=True)])
        yy = y[pick]
        if yy.sum() == 0 or yy.sum() == len(yy): continue
        diffs.append(f(yy, p_a[pick]) - f(yy, p_b[pick]))
    d = np.array(diffs)
    return dict(diff=float(f(y, p_a) - f(y, p_b)), lo=float(np.percentile(d, 2.5)),
                hi=float(np.percentile(d, 97.5)), p_le0=float((d <= 0).mean()))

def cluster_bootstrap_ci(y, p, clusters, n_boot=1000, seed=0, metric="AUPRC"):
    from sklearn.metrics import average_precision_score, roc_auc_score
    f = average_precision_score if metric == "AUPRC" else roc_auc_score
    rng = np.random.default_rng(seed); cl = np.asarray(clusters); u = np.unique(cl)
    idx = {c: np.where(cl == c)[0] for c in u}; vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx[c] for c in rng.choice(u, len(u), replace=True)])
        if 0 < y[pick].sum() < len(pick): vals.append(f(y[pick], p[pick]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

def lead_time_table(cases, p, tau, events, horizon):
    """For every recall event in the evaluated period, earliest alert month before it."""
    df = cases[["code", "mi"]].copy(); df["p"] = p
    rows = []
    for code, e in events:
        w = df[(df.code == code) & (df.mi >= e - horizon) & (df.mi < e)]
        if len(w) == 0: continue
        hit = w[w.p >= tau]
        rows.append(dict(code=code, event_mi=e, detected=len(hit) > 0,
                         lead=(e - hit.mi.min()) if len(hit) else 0))
    return pd.DataFrame(rows)
