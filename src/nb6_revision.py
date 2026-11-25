# %% [markdown]
# # AEGIS-MD · Notebook 6 — Revision analyses (CPU)
# Inputs: attach the output of the **full pipeline** notebook (records.pkl, record_ann.pkl, rec_emb.npy,
# proto_emb.pkl, counts.pkl, codes.pkl, recalls.pkl).
#
# Re-builds the benchmark from the enriched records (same code as Notebook 3) and adds every analysis requested
# in review:
# * paired product-code cluster-bootstrap differences (ΔAUPRC, ΔAUROC) for the pre-specified comparisons;
# * disproportionality baselines (PRR, ROR, EBGM / gamma-Poisson shrinker), recall-history baseline,
#   supervised text baseline (narrative embeddings + logistic regression);
# * calibration (intercept, slope, reliability curve), alert burden, precision@k for a monthly review queue;
# * repeated random reading (10 draws), 3- and 12-month horizons;
# * ablations on the validation period (inner temporal split of training), source-correlation check;
# * per-event and per-code results, leave-one-code-out influence;
# * initial-report-only sensitivity (MAUDE supplements removed), washout after any recall class;
# * pooled gap evidence (k = 30 per gap, and gap evidence averaged over 3 months);
# * benchmark datasheet (codes, events) and blind labelling sheets for stance and leakage checks.

# %%
# %%writefile-core

# %%
import os, sys, json, time, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats, optimize, special
sys.path.insert(0, os.getcwd())
from aegis_core import *
OUT = out_dir(); FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)
T_START = time.time()
def lap(msg): print(f"[{(time.time() - T_START) / 60:5.1f} min] {msg}", flush=True)
N_BOOT = int(os.environ.get("AEGIS_NBOOT", "2000"))
N_RAND = int(os.environ.get("AEGIS_NRAND", "10"))

R = pd.read_pickle(find_file("records.pkl"))
ANN = pd.read_pickle(find_file("record_ann.pkl"))
E = np.load(find_file("rec_emb.npy"))
PROTO = pd.read_pickle(find_file("proto_emb.pkl"))
COUNTS = pd.read_pickle(find_file("counts.pkl"))
CODES = pd.read_pickle(find_file("codes.pkl"))
REC = pd.read_pickle(find_file("recalls.pkl"))
assert (ANN.rid.values == R.rid.values).all() and len(E) == len(R)
lap(f"{len(R)} records | {len(CODES)} codes | {len(REC)} recall rows")

MAND = R.source.str.lower().str.contains("manufacturer|user facility|distributor|importer").values
RA = dict(emb=E, has_text=(R.text_event.fillna("").str.len() > 20).values, mand_src=MAND,
          has_mfr=ANN.has_mfr.values.astype(bool), p_confirm=ANN.p_confirm.values, p_refute=ANN.p_refute.values,
          lot=R.lot.fillna("").replace({"UNK": "", "UNKNOWN": "", "NA": "", "N/A": "", "NI": ""}).values,
          model=R.model.fillna("").replace({"UNK": "", "UNKNOWN": "", "NA": "", "N/A": "", "NI": ""}).values,
          prec_sim=ANN.prec_sim.values.astype(np.float32))
TOR = R.type_of_report.fillna("").str.lower() if "type_of_report" in R else pd.Series([""] * len(R))
INITIAL_ONLY = ~TOR.str.contains("follow|supplement|additional|correction").values
lap(f"initial-only records: {INITIAL_ONLY.mean():.3f}")

# %% [markdown]
# ## 1. Recall events, labels at 3 / 6 / 12 months, case panel

# %%
W = CFG["WASHOUT"]
lab = REC[REC.klass.isin(CFG["LABEL_CLASSES"])]
ev = lab.groupby(["code", "event_id"]).agg(init_mi=("init_mi", "min"), posted_mi=("posted_mi", "min"),
                                           n_rows=("rec_id", "size")).reset_index()
EVENTS = ev.groupby("code").init_mi.apply(lambda s: sorted(set(int(x) for x in s))).to_dict()
ANY_EV = REC[REC.klass.str.startswith("Class")].groupby("code").init_mi.apply(lambda s: sorted(set(int(x) for x in s))).to_dict()
panel = set(CODES.code)
ev_panel = ev[ev.code.isin(panel)].copy()
lap(f"Class I recall rows in panel codes: {len(lab[lab.code.isin(panel)])} -> events (res_event_number): {len(ev_panel)} "
    f"-> distinct (code, initiation month): {sum(len(EVENTS.get(c, [])) for c in panel)}")
unclassified = (~REC.klass.str.startswith("Class")).mean()

R_SER = R.event_type.str.lower().isin(["death", "injury"]).values
cnt = COUNTS.set_index(["code", "mi"]).n.to_dict()
agg = {}
for (c, m), g in R.groupby(["code", "mi"]):
    pc = Counter()
    for s in g.problems.fillna(""):
        for p in s.split(";"):
            p = p.strip()
            if p: pc[p] += 1
    ii = g.index.values
    agg[(c, m)] = dict(n=len(g), serious=int(R_SER[ii].sum()), mfr_src=int(MAND[ii].sum()),
                       text=int(RA["has_text"][ii].sum()), prob=pc)

p0, p1 = ym(CFG["PANEL_START"]), ym(CFG["PANEL_END"])
first_m = ym(CFG["DATA_START"]) + CFG["RECENT"] + CFG["BASELINE"]
rows = []
for c in CODES.code:
    nb = {m: v for (cc, m), v in cnt.items() if cc == c}
    ag = {m: v for (cc, m), v in agg.items() if cc == c}
    t0 = tier0_for_code(nb, range(first_m, p1 + 1), ag)
    t0 = t0[t0.mi >= p0].copy(); t0["code"] = c
    evs = EVENTS.get(c, []); anyev = ANY_EV.get(c, [])
    for h in (3, 6, 12):
        t0[f"y{h}"] = [int(any(t < e <= t + h for e in evs)) for t in t0.mi]
    t0["washout"] = [int(any(t - W < e <= t for e in evs)) for t in t0.mi]
    t0["washout_any"] = [int(any(t - W < e <= t for e in anyev)) for t in t0.mi]
    rows.append(t0)
CASES = pd.concat(rows, ignore_index=True)
CASES = CASES[CASES.washout == 0].reset_index(drop=True)
CASES["y"] = CASES.y6
CASES["split"] = [split_of(m) for m in CASES.mi]
CASES["case_id"] = np.arange(len(CASES))
from sklearn.ensemble import IsolationForest
IF_FEATS = ["pois_score", "log_ratio", "mk_z", "sen_norm", "persist", "cusum", "sev_z", "prob_jsd"]
iso = IsolationForest(n_estimators=300, contamination="auto", random_state=CFG["SEED"]).fit(
    CASES.loc[CASES.split == "train", IF_FEATS].fillna(0))
CASES["if_score"] = -iso.score_samples(CASES[IF_FEATS].fillna(0))

y = CASES.y.values; split = CASES.split.values; code = CASES.code.values; MI = CASES.mi.values
tr, va, te = split == "train", split == "val", split == "test"
tra, trb = tr & (MI <= ym("2019-06")), tr & (MI > ym("2019-06"))   # inner temporal split for validation ablations

# benchmark summary with events per split, and the code(s) absent from a split
def ev_in(lo, hi):
    return [(c, e) for c, evs in EVENTS.items() if c in panel for e in evs if lo < e <= hi]
SPL = {"train": (ym(CFG["PANEL_START"]) - 1, ym(CFG["TRAIN_END"])), "val": (ym(CFG["TRAIN_END"]), ym(CFG["VAL_END"])),
       "test": (ym(CFG["VAL_END"]), ym(CFG["PANEL_END"]))}
summ = []
for s, (lo, hi) in SPL.items():
    m = split == s
    summ.append(dict(split=s, months=f"{ym_str(lo + 1)}..{ym_str(hi)}", cases=int(m.sum()), positives=int(y[m].sum()),
                     prevalence=round(float(y[m].mean()), 4), codes=int(pd.Series(code[m]).nunique()),
                     events_with_label_in_split=len(ev_in(lo, hi + CFG["HORIZON"])),
                     events_initiated_in_split=len(ev_in(lo, hi)),
                     codes_missing=";".join(sorted(panel - set(code[m])))))
SUMM = pd.DataFrame(summ); SUMM.to_csv(f"{OUT}/benchmark_summary_v2.csv", index=False); print(SUMM.to_string())
EV_YEAR = pd.Series([e // 12 for c, evs in EVENTS.items() if c in panel for e in evs]).value_counts().sort_index()
EV_YEAR.rename("class_I_events").to_csv(f"{OUT}/events_per_year.csv")
print("Class I events per initiation year (panel):", EV_YEAR.to_dict())
miss = {c: dict(n_cases=int((code == c).sum()), n_test=int(((code == c) & te).sum()),
                months_with_reports_test=int(sum(1 for (cc, m), v in cnt.items() if cc == c and m > ym(CFG["VAL_END"]) and v > 0)))
        for c in panel - set(code[te])}
print("codes with no test cases:", miss)

# datasheet: product codes and recall events
names = CODES.set_index("code").device_name.to_dict()
DS = CODES[["code", "group", "device_name"]].copy() if "group" in CODES else CODES[["code", "device_name"]].copy()
DS["maude_reports_2014_2025"] = DS.code.map(COUNTS.groupby("code").n.sum()).fillna(0).astype(int)
DS["sampled_records"] = DS.code.map(R.code.value_counts()).fillna(0).astype(int)
for s in ["train", "val", "test"]:
    DS[f"cases_{s}"] = DS.code.map(pd.Series(code[split == s]).value_counts()).fillna(0).astype(int)
DS["class_I_events_2016_2025"] = DS.code.map(lambda c: sum(1 for e in EVENTS.get(c, []) if p0 <= e <= p1 + CFG["HORIZON"]))
DS.to_csv(f"{OUT}/datasheet_product_codes.csv", index=False)
ev_panel["init_month"] = ev_panel.init_mi.map(ym_str); ev_panel["posted_month"] = ev_panel.posted_mi.map(lambda v: ym_str(v) if v == v else "")
ev_panel["split_of_initiation"] = [("pre-panel" if e < p0 else split_of(e)) for e in ev_panel.init_mi]
ev_panel.drop(columns=["init_mi", "posted_mi"]).to_csv(f"{OUT}/datasheet_recall_events.csv", index=False)
lap("cases built")

# %% [markdown]
# ## 2. Evidence features under each reading strategy (same definitions as Notebook 3)

# %%
doc_text = (R.text_event.fillna("") + " " + R.text_mfr.fillna("") + " " + R.problems.fillna("").str.replace(";", " "))
BM = BM25Matrix(doc_text.values)
BY = R.groupby(["code", "mi"]).rid.apply(np.array).to_dict()
def rows_in(c, lo, hi):
    parts = [BY[(c, m)] for m in range(lo, hi + 1) if (c, m) in BY]
    return np.concatenate(parts) if parts else np.array([], dtype=int)

K = CFG["K_PER_GAP"]; KS = [10, 20, 40]
_samp = np.random.default_rng(0).choice(len(E), min(50000, len(E)), replace=False)
THR = {g: float(np.quantile(E[_samp].astype(np.float32) @ PROTO[g], 0.90)) for g in GAP_SPECS}
STRATS = ["full", "full_init", "gap", "gap_nc", "gap30"] + [f"generic{k}" for k in KS] + \
         [f"random{k}_s{s}" for k in KS for s in range(N_RAND)]
feat = {s: [] for s in STRATS}
S_GAP, S_GAPB, S_GAP30 = {}, {}, {}
DIM = E.shape[1]
TXT = np.zeros((len(CASES), 2 * DIM), np.float32)
rng_b = np.random.default_rng(CFG["SEED"])
rngs = {s: np.random.default_rng(1000 + s) for s in range(N_RAND)}
for i, r in CASES.iterrows():
    c, t = r.code, int(r.mi)
    win = rows_in(c, t - CFG["RECENT"] + 1, t)
    base = rows_in(c, t - CFG["RECENT"] - CFG["BASELINE"] + 1, t - CFG["RECENT"])
    base_all = base
    if len(base) > 400: base = rng_b.choice(base, 400, replace=False)
    if len(base):
        Eb = E[base].astype(np.float32); cb = Eb.mean(0); cb /= (np.linalg.norm(cb) + 1e-9)
        ctx = dict(base_cen=cb, base_coh=float((Eb @ cb).mean()))
    else:
        ctx = dict(base_cen=None, base_coh=0.0); cb = np.zeros(DIM, np.float32)
    if len(win):
        cw = E[win].astype(np.float32).mean(0); TXT[i, :DIM] = cw; TXT[i, DIM:] = cw - cb * np.linalg.norm(cw)
    wp, bp = Counter(), Counter()
    for m in range(t - CFG["RECENT"] + 1, t + 1): wp.update(agg.get((c, m), {}).get("prob", {}))
    for m in range(t - CFG["RECENT"] - CFG["BASELINE"] + 1, t - CFG["RECENT"] + 1): bp.update(agg.get((c, m), {}).get("prob", {}))
    em_phr, em_tok = emerging_terms(wp, bp)
    dn_tok = tokenize(names.get(c, ""))

    f = deep_evidence(win, RA, ctx); f["n_docs"] = len(win); feat["full"].append(f)
    wi = win[INITIAL_ONLY[win]] if len(win) else win
    f = deep_evidence(wi, RA, ctx); f["n_docs"] = len(wi); feat["full_init"].append(f)
    g_out, g_nc, g30 = {"n_window": len(win)}, {"n_window": len(win)}, {"n_window": len(win)}
    sl, bl, sl30 = {}, {}, {}
    for gname, spec in GAP_SPECS.items():
        q = spec["terms"] + (em_tok + dn_tok if spec["use_emerging"] else [])
        rate = query_rate_shift(win, base_all, E, PROTO[gname], THR[gname])
        S = hybrid_retrieve(win, BM, E, q, PROTO[gname], K)
        Sb = hybrid_retrieve(base, BM, E, q, PROTO[gname], K) if len(base) else np.array([], dtype=int)
        S3 = hybrid_retrieve(win, BM, E, q, PROTO[gname], 3 * K)
        Sb3 = hybrid_retrieve(base, BM, E, q, PROTO[gname], 3 * K) if len(base) else np.array([], dtype=int)
        sl[gname], bl[gname], sl30[gname] = S, Sb, (S3, Sb3)
        g_nc.update(deep_evidence(S, RA, ctx, sources=[gname]))
        g_out.update(contrastive_gap_evidence(S, Sb, RA, ctx, gname, rate=rate))
        g30.update(contrastive_gap_evidence(S3, Sb3, RA, ctx, gname, rate=rate))
    feat["gap"].append(g_out); feat["gap_nc"].append(g_nc); feat["gap30"].append(g30)
    S_GAP[r.case_id], S_GAPB[r.case_id], S_GAP30[r.case_id] = sl, bl, sl30
    gq = GENERIC_QUERY["terms"] + dn_tok
    for k in KS:
        S = hybrid_retrieve(win, BM, E, gq, PROTO["generic"], k)
        f = deep_evidence(S, RA, ctx); f["n_docs"] = len(S); feat[f"generic{k}"].append(f)
        for s in range(N_RAND):
            S = rngs[s].choice(win, min(k, len(win)), replace=False) if len(win) else win
            f = deep_evidence(S, RA, ctx); f["n_docs"] = len(S); feat[f"random{k}_s{s}"].append(f)
    if i % 1000 == 0: lap(f"features {i}/{len(CASES)}")
FEAT = {s: pd.DataFrame(v, index=CASES.case_id.values) for s, v in feat.items()}

# gap evidence pooled over the current and two previous case months of the same code (past only)
G = FEAT["gap"].copy(); G["code"] = code; G["mi"] = MI
key = {(c, m): i for i, (c, m) in enumerate(zip(code, MI))}
gcols = [c for c in FEAT["gap"].columns if c != "n_window"]
pooled = np.full((len(CASES), len(gcols)), np.nan)
GV = FEAT["gap"][gcols].values
for i, (c, m) in enumerate(zip(code, MI)):
    idx = [key[(c, mm)] for mm in (m - 2, m - 1, m) if (c, mm) in key]
    pooled[i] = np.nanmean(GV[idx], axis=0)
FEAT["gap_pool3"] = pd.DataFrame(pooled, columns=gcols, index=CASES.case_id.values)
lap("features done")

# %% [markdown]
# ## 3. Disproportionality (PRR, ROR, EBGM) and recall-history features

# %%
# report weights: full monthly count / sampled reports, so that sampled problem-code counts estimate full counts
nsamp = R.groupby(["code", "mi"]).size().to_dict()
wrec = np.array([cnt.get((c, m), n) / max(nsamp.get((c, m), 1), 1) for c, m, n in
                 zip(R.code.values, R.mi.values, [nsamp.get((c, m), 1) for c, m in zip(R.code.values, R.mi.values)])])
wrec = np.where(np.isfinite(wrec) & (wrec > 0), wrec, 1.0)
PROB_SETS = [tuple(sorted({p.strip() for p in s.split(";") if p.strip()})) for s in R.problems.fillna("")]
month_tot, month_cp, month_p = Counter(), Counter(), Counter()     # weighted reports per (code,m), (code,p,m), (p,m)
month_all = Counter()
for c, m, w_, ps in zip(R.code.values, R.mi.values, wrec, PROB_SETS):
    month_tot[(c, m)] += w_; month_all[m] += w_
    for p in ps:
        month_cp[(c, p, m)] += w_; month_p[(p, m)] += w_
cp_by_cm = {}
for (c, p, m), v in month_cp.items(): cp_by_cm.setdefault((c, m), {})[p] = v
p_by_m = {}
for (p, m), v in month_p.items(): p_by_m.setdefault(m, {})[p] = v

def window_cells(c, t):
    ms = range(t - CFG["RECENT"] + 1, t + 1)
    a = Counter(); Nc = 0.0; Np = Counter(); N = 0.0
    for m in ms:
        Nc += month_tot.get((c, m), 0.0); N += month_all.get(m, 0.0)
        a.update(cp_by_cm.get((c, m), {})); Np.update(p_by_m.get(m, {}))
    return a, Nc, Np, N

# DuMouchel gamma-Poisson shrinker: prior = mixture of two gammas, fit by marginal likelihood on training cells
def nb_logpmf(n, E_, a, b):
    return (special.gammaln(a + n) - special.gammaln(a) - special.gammaln(n + 1)
            + a * np.log(b / (b + E_)) + n * np.log(E_ / (b + E_)))
cells = []
for i in np.where(tr)[0][::3]:
    a, Nc, Np, N = window_cells(code[i], int(MI[i]))
    for p, n in a.items():
        if N > 0: cells.append((n, Nc * Np[p] / N))
cells = np.array(cells) if cells else np.zeros((0, 2))
def negll(th):
    a1, b1, a2, b2 = np.exp(th[:4]); w_ = special.expit(th[4])
    l1 = nb_logpmf(cells[:, 0], cells[:, 1], a1, b1); l2 = nb_logpmf(cells[:, 0], cells[:, 1], a2, b2)
    return -np.sum(np.logaddexp(np.log(w_) + l1, np.log(1 - w_) + l2))
if len(cells) > 50:
    opt = optimize.minimize(negll, x0=np.log([0.2, 0.1, 2.0, 4.0]).tolist() + [0.0], method="L-BFGS-B",
                            bounds=[(-7, 7)] * 4 + [(-6, 6)])
    A1, B1, A2, B2 = np.exp(opt.x[:4]); PW = special.expit(opt.x[4])
else:
    A1, B1, A2, B2, PW = 0.2, 0.1, 2.0, 4.0, 0.3
GPS_PRIOR = dict(alpha1=A1, beta1=B1, alpha2=A2, beta2=B2, w=PW, n_cells=len(cells))
print("GPS prior:", {k: round(float(v), 4) for k, v in GPS_PRIOR.items()})
def ebgm(n, E_):
    l1 = np.log(PW) + nb_logpmf(n, E_, A1, B1); l2 = np.log(1 - PW) + nb_logpmf(n, E_, A2, B2)
    q = np.exp(l1 - np.logaddexp(l1, l2))
    elog = q * (special.digamma(A1 + n) - np.log(B1 + E_)) + (1 - q) * (special.digamma(A2 + n) - np.log(B2 + E_))
    return np.exp(elog)

disp = np.zeros((len(CASES), 3))
for i in range(len(CASES)):
    a, Nc, Np, N = window_cells(code[i], int(MI[i]))
    best = [0.0, 0.0, 0.0]
    if a and N > Nc > 0:
        ps = list(a.keys()); av = np.array([a[p] for p in ps]); npv = np.array([Np[p] for p in ps])
        b_ = Nc - av; c_ = npv - av; d_ = (N - Nc) - c_
        av5, b5, c5, d5 = av + .5, np.clip(b_, 0, None) + .5, np.clip(c_, 0, None) + .5, np.clip(d_, 0, None) + .5
        lprr = np.log((av5 / (av5 + b5)) / (c5 / (c5 + d5)))
        se_prr = np.sqrt(1 / av5 - 1 / (av5 + b5) + 1 / c5 - 1 / (c5 + d5))
        lror = np.log(av5 * d5 / (b5 * c5)); se_ror = np.sqrt(1 / av5 + 1 / b5 + 1 / c5 + 1 / d5)
        ok = av >= 3
        if ok.any():
            best[0] = float(np.max((lprr - 1.96 * se_prr)[ok])); best[1] = float(np.max((lror - 1.96 * se_ror)[ok]))
        Ev = Nc * npv / N
        best[2] = float(np.log(np.max(ebgm(av, np.clip(Ev, 1e-6, None)))))
    disp[i] = best
CASES["prr_lb"], CASES["ror_lb"], CASES["log_ebgm"] = disp[:, 0], disp[:, 1], disp[:, 2]

# recall history (public at time t: posted before t)
RH = []
recs_by_code = {c: g for c, g in REC.groupby("code")}
for c, t in zip(code, MI):
    g = recs_by_code.get(c)
    if g is None: RH.append((0, 0, 120, 120)); continue
    pub = g[g.posted_mi < t]
    p1_ = pub[pub.klass.isin(CFG["LABEL_CLASSES"])]
    RH.append((pub.event_id.nunique(), p1_.event_id.nunique(),
               min(120, t - pub.posted_mi.max()) if len(pub) else 120, min(120, t - p1_.posted_mi.max()) if len(p1_) else 120))
RH = np.array(RH, float)
CASES["rh_n_any"], CASES["rh_n_c1"], CASES["rh_since_any"], CASES["rh_since_c1"] = RH.T
RH_F = ["rh_n_any", "rh_n_c1", "rh_since_any", "rh_since_c1"]
lap("disproportionality + recall history done")

# %% [markdown]
# ## 4. Models

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score
T0F = sum(TIER0_GROUPS.values(), []); DF = sum(DEEP_FEATURES.values(), [])
GBM_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                  early_stopping=True, validation_fraction=0.15, random_state=CFG["SEED"])
G0 = dict(TIER0_GROUPS); GA = {**TIER0_GROUPS, **DEEP_FEATURES}; GC = {**TIER0_GROUPS, **CONTRAST_FEATURES}
ALERT_RATE = 0.10
def thr(p_val): return float(np.quantile(p_val, 1 - ALERT_RATE))
def Xof(strategy, cases=CASES):
    X = cases.set_index("case_id")[T0F + RH_F].copy()
    if strategy: X = X.join(FEAT[strategy])
    return X.reset_index(drop=True)
def gbm(X, yy, fit_mask):
    m = HistGradientBoostingClassifier(**GBM_PARAMS).fit(X[fit_mask], yy[fit_mask]); return m.predict_proba(X)[:, 1]
def minmax_tr(s): s = np.nan_to_num(np.asarray(s, float)); return (s - s[tr].min()) / (s[tr].max() - s[tr].min() + 1e-9)
def fusion(groups, strategy, yy=None, fit=(None, None), use_rho=True):
    yy = y if yy is None else yy; a, b = fit if fit[0] is not None else (tr, va)
    X = Xof(strategy); fz = EvidenceFusion(groups, use_rho=use_rho).fit(X[a], yy[a], X[b], yy[b], groups_tr=code[a])
    return fz.predict(X), fz, X
def lr_select(X, yy, a=None, b=None):
    a = tr if a is None else a; b = va if b is None else b
    sc = StandardScaler().fit(X[a]); Xs = sc.transform(X)
    best = None
    for C in [0.001, 0.01, 0.1, 1.0]:
        m = LogisticRegression(C=C, max_iter=3000).fit(Xs[a], yy[a]); s = average_precision_score(yy[b], m.predict_proba(Xs[b])[:, 1])
        if best is None or s > best[0]: best = (s, C, m)
    return best[2].predict_proba(Xs)[:, 1], best[1]

PRED, DOCS, FAMILY = {}, {}, {}
def add(name, p, docs=0.0, fam=""):
    PRED[name] = np.asarray(p, float); DOCS[name] = float(docs); FAMILY[name] = fam

for f, nm in [("pois_score", "Poisson exceedance"), ("cusum", "Poisson CUSUM"), ("mk_z", "Mann-Kendall trend"),
              ("if_score", "Isolation Forest"), ("prr_lb", "PRR (max lower 95% bound)"),
              ("ror_lb", "ROR (max lower 95% bound)"), ("log_ebgm", "EBGM (max, gamma-Poisson shrinker)")]:
    add(nm, minmax_tr(CASES[f].values), fam="routine-unsupervised")
add("Recall history (logistic regression)", lr_select(CASES[RH_F].values, y)[0], fam="supervised")
add("GBM, count features", gbm(CASES[T0F].values, y, tr), fam="supervised")
add("GBM, count + recall history", gbm(CASES[T0F + RH_F].values, y, tr), fam="supervised")
pca = PCA(n_components=min(32, TXT.shape[1] - 1), random_state=0).fit(TXT[tr]); TXTP = pca.transform(TXT)
p_txt, C_txt = lr_select(TXTP, y); add("Text model (embeddings + LR)", p_txt, docs=FEAT["full"].n_docs[te].mean(), fam="supervised-text")
p_txt2, _ = lr_select(np.column_stack([TXTP, CASES[T0F].values]), y)
add("Text model + count features (LR)", p_txt2, docs=FEAT["full"].n_docs[te].mean(), fam="supervised-text")
Xfull = np.column_stack([CASES[T0F].values, FEAT["full"][DF].values])
add("GBM, count + full-reading narrative features", gbm(Xfull, y, tr), docs=FEAT["full"].n_docs[te].mean(), fam="supervised")
add("Fusion, count evidence only", fusion(G0, None)[0], fam="fusion")
GH = {**TIER0_GROUPS, "history": RH_F}; GAH = {**GA, "history": RH_F}
add("Fusion, count evidence + recall history", fusion(GH, None)[0], fam="fusion")
P_FULL, FZF, XF = fusion(GA, "full"); add("Fusion + full reading", P_FULL, docs=FEAT["full"].n_docs[te].mean(), fam="fusion+reading")
add("Fusion + full reading + recall history", fusion(GAH, "full")[0], docs=FEAT["full"].n_docs[te].mean(), fam="fusion+reading")
add("GBM, count + recall history + full-reading narrative features", gbm(np.column_stack([Xfull, CASES[RH_F].values]), y, tr), docs=FEAT["full"].n_docs[te].mean(), fam="supervised")
add("Fusion + full reading, initial reports only", fusion(GA, "full_init")[0], docs=FEAT["full_init"].n_docs[te].mean(), fam="sensitivity")
for k in KS:
    add(f"Fusion + generic retrieval (k={k})", fusion(GA, f"generic{k}")[0], docs=FEAT[f"generic{k}"].n_docs[te].mean(), fam="fusion+reading")
RAND = {}
for k in KS:
    for s in range(N_RAND):
        RAND[(k, s)] = fusion(GA, f"random{k}_s{s}")[0]
    add(f"Fusion + random reading (k={k})", RAND[(k, 0)], docs=FEAT[f"random{k}_s0"].n_docs[te].mean(), fam="fusion+reading")
lap("baselines + fusion done")

# AEGIS gap policies (same selection rule as Notebook 4: cheapest validation setting within 2% of best val AUPRC)
P_G4, FZ, XG = fusion(GC, "gap")
SRC = list(GC.keys()); n0 = len(TIER0_GROUPS); DEEP_IDX = {s: SRC.index(s) for s in DEEP_SOURCES}
CONTRIB = FZ.contributions(XG); Z0 = FZ.b0 + CONTRIB[:, :n0].sum(1); TAU = thr(FZ.predict(XG)[va])
CIDS = CASES.case_id.values
def docs_read(cid, acquired, S_=S_GAP, B_=S_GAPB):
    if not acquired: return 0
    w_ = set().union(*[set(np.asarray(S_[cid][g]).tolist()) for g in acquired])
    b_ = set().union(*[set(np.asarray(B_[cid].get(g, [])).tolist()) for g in acquired])
    return len(w_) + len(b_)
def run_policy(policy, eps=0.0, B=4):
    P = np.zeros(len(CASES)); D = np.zeros(len(CASES)); A = []
    for i in range(len(CASES)):
        z = Z0[i]; acq = []
        for _ in range(B):
            rem = [g for g in DEEP_SOURCES if g not in acq]
            if not rem: break
            if policy == "fixed": g = rem[0]
            else:
                v = {g: (voi(z, g, FZ, TAU, DEEP_IDX[g]) if policy == "voi" else entropy_gain(z, g, FZ, DEEP_IDX[g])) for g in rem}
                g = max(v, key=v.get)
                if v[g] <= eps: break
            acq.append(g); z += CONTRIB[i, DEEP_IDX[g]]
        P[i] = 1 / (1 + np.exp(-z)); D[i] = docs_read(CIDS[i], acq); A.append(acq)
    return P, D, A
EPS = [0.0, 1e-5, 5e-5, 1e-4, 3e-4, 1e-3, 3e-3]
runs = []
for eps in EPS:
    for pol in ["voi", "entropy"]:
        e = eps if pol == "voi" else eps * 20
        P, D, A = run_policy(pol, eps=e)
        runs.append(dict(pol=pol, eps=e, P=P, D=D, val=average_precision_score(y[va], P[va]), vdocs=D[va].mean()))
best_val = max(r_["val"] for r_ in runs)
pick = sorted([r_ for r_ in runs if r_["val"] >= 0.98 * best_val], key=lambda r_: r_["vdocs"])[0]
add(f"AEGIS gap retrieval (VoI, {'decision risk' if pick['pol'] == 'voi' else 'information'})", pick["P"], docs=pick["D"][te].mean(), fam="AEGIS")
for pol in ["voi", "entropy"]:
    if pol == pick["pol"]: continue
    sub = [r_ for r_ in runs if r_["pol"] == pol]; bv = max(r_["val"] for r_ in sub)
    alt = sorted([r_ for r_ in sub if r_["val"] >= 0.98 * bv], key=lambda r_: r_["vdocs"])[0]
    add(f"AEGIS gap retrieval (VoI, {'decision risk' if pol == 'voi' else 'information'})", alt["P"], docs=alt["D"][te].mean(), fam="AEGIS")
P_all, D_all, _ = run_policy("fixed", B=4)
add("AEGIS, all four gaps", P_all, docs=D_all[te].mean(), fam="AEGIS")
D30 = np.array([len(set().union(*[set(np.asarray(S_GAP30[c][g][0]).tolist()) | set(np.asarray(S_GAP30[c][g][1]).tolist()) for g in DEEP_SOURCES])) for c in CIDS])
add("Gap retrieval, all four gaps, k=30 per gap", fusion(GC, "gap30")[0], docs=D30[te].mean(), fam="AEGIS-variant")
D3m = np.array([D_all[[key[(c, mm)] for mm in (m - 2, m - 1, m) if (c, mm) in key]].sum() for c, m in zip(code, MI)])
add("Gap retrieval, all four gaps, pooled over 3 months", fusion(GC, "gap_pool3")[0], docs=D3m[te].mean(), fam="AEGIS-variant")

# VoI-gated full reading (Notebook 4, section 4b)
SRCF = list(GA.keys()); DIDX = {s: SRCF.index(s) for s in DEEP_SOURCES}
CF = FZF.contributions(XF); Z0F = FZF.b0 + CF[:, :n0].sum(1); TAUF = thr(P_FULL[va])
NW = FEAT["full"].n_docs.fillna(0).values.astype(float)
def run_gate(policy, eps):
    P = np.zeros(len(CASES)); D = np.zeros(len(CASES))
    for i in range(len(CASES)):
        z = Z0F[i]; acq = []
        for _ in range(4):
            rem = [g for g in DEEP_SOURCES if g not in acq]
            if not rem: break
            v = {g: (voi(z, g, FZF, TAUF, DIDX[g]) if policy == "voi" else entropy_gain(z, g, FZF, DIDX[g])) for g in rem}
            g = max(v, key=v.get)
            if v[g] <= eps: break
            acq.append(g); z += CF[i, DIDX[g]]
        P[i] = 1 / (1 + np.exp(-z)); D[i] = NW[i] if acq else 0.0
    return P, D
gr = []
for pol, grid in [("voi", [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]), ("entropy", [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2])]:
    for eps in grid:
        P, D = run_gate(pol, eps); gr.append(dict(P=P, D=D, val=average_precision_score(y[va], P[va]), vdocs=D[va].mean()))
bv = max(g_["val"] for g_ in gr); gpick = sorted([g_ for g_ in gr if g_["val"] >= 0.98 * bv], key=lambda g_: g_["vdocs"])[0]
add("AEGIS adaptive investigation (VoI-gated full reading)", gpick["P"], docs=gpick["D"][te].mean(), fam="AEGIS")
lap("AEGIS policies done")

# %% [markdown]
# ## 5. Main results table, paired differences, calibration, alert burden, precision@k

# %%
TEST_EVENTS = [(c, e) for c, evs in EVENTS.items() if c in set(code[te]) for e in evs
               if e > ym(CFG["VAL_END"]) + 1 and e <= ym(CFG["PANEL_END"]) + CFG["HORIZON"]]
def _lg(p): p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def calib(yy, p):
    """Calibration intercept (calibration-in-the-large, slope fixed at 1) and calibration slope."""
    lp = _lg(p); yy = np.asarray(yy, float)
    def nll(v, slope=None):
        a = v[0]; b = v[1] if slope is None else slope
        z = a + b * lp; return -np.sum(yy * z - np.logaddexp(0, z))
    cil = optimize.minimize(lambda v: nll([v[0]], slope=1.0), x0=[0.0], method="BFGS").x[0]
    ab_ = optimize.minimize(nll, x0=[0.0, 1.0], method="BFGS").x
    return float(cil), float(ab_[1])
def burden(p, tau, mask=te):
    df = pd.DataFrame(dict(code=code[mask], mi=MI[mask], y=y[mask], p=p[mask]))
    alerts = df.p >= tau; code_years = len(df) / 12.0
    prec_k = {}
    for k in (5, 10):
        prec_k[k] = df.groupby("mi").apply(lambda g: g.nlargest(k, "p").y.sum() / k).mean()
    return dict(alerts_per_code_year=alerts.sum() / code_years, false_alerts_per_code_year=(alerts & (df.y == 0)).sum() / code_years,
                PPV=float(df.y[alerts].mean()) if alerts.any() else np.nan, sensitivity_month=float(alerts[df.y == 1].mean()),
                prec_at_5=prec_k[5], prec_at_10=prec_k[10])
ROWS = []
def _ci(*a, **k):
    try: return cluster_bootstrap_ci(*a, **k)
    except Exception: return (np.nan, np.nan)
for name, p in PRED.items():
  try:
    m = metrics(y[te], p[te]); lo, hi = _ci(y[te], p[te], code[te], n_boot=N_BOOT // 2)
    lo2, hi2 = _ci(y[te], p[te], code[te], n_boot=N_BOOT // 2, metric="AUROC")
    tau = thr(p[va]); lt = lead_time_table(CASES[te].reset_index(drop=True), p[te], tau, TEST_EVENTS, CFG["HORIZON"])
    cil, slope = calib(y[te], p[te]) if FAMILY[name] != "routine-unsupervised" else (np.nan, np.nan)
    b = burden(p, tau)
    ROWS.append(dict(method=name, family=FAMILY[name], reports_read=round(DOCS[name], 1), AUPRC=m["AUPRC"], AUPRC_lo=lo, AUPRC_hi=hi,
                     AUROC=m["AUROC"], AUROC_lo=lo2, AUROC_hi=hi2,
                     Brier=m["Brier"] if FAMILY[name] != "routine-unsupervised" else np.nan,
                     calib_intercept=cil, calib_slope=slope, events_detected=lt.detected.mean() if len(lt) else np.nan,
                     median_lead=lt.loc[lt.detected, "lead"].median() if len(lt) and lt.detected.any() else np.nan, **b))
  except Exception as ex:
    print("metrics failed:", name, type(ex).__name__, str(ex)[:100])
RES = pd.DataFrame(ROWS); RES.to_csv(f"{OUT}/results_test_v2.csv", index=False)
print(RES[["method", "reports_read", "AUPRC", "AUPRC_lo", "AUPRC_hi", "AUROC", "calib_slope", "prec_at_10", "events_detected"]].round(3).to_string())

# repeated random reading
rr = []
for k in KS if 0 < y[te].sum() < te.sum() else []:
    a_ = [average_precision_score(y[te], RAND[(k, s)][te]) for s in range(N_RAND)]
    b_ = [roc_auc_score(y[te], RAND[(k, s)][te]) for s in range(N_RAND)]
    rr.append(dict(k=k, reports_read=FEAT[f"random{k}_s0"].n_docs[te].mean(), AUPRC_mean=np.mean(a_), AUPRC_sd=np.std(a_, ddof=1),
                   AUROC_mean=np.mean(b_), AUROC_sd=np.std(b_, ddof=1), n_draws=N_RAND))
RR = pd.DataFrame(rr); RR.to_csv(f"{OUT}/random_reading_repeats.csv", index=False); print(RR.round(4))

# paired differences (pre-specified)
gen20 = "Fusion + generic retrieval (k=20)"; aeg = [n for n in PRED if n.startswith("AEGIS gap retrieval")][0]
best_disp = max(["PRR (max lower 95% bound)", "ROR (max lower 95% bound)", "EBGM (max, gamma-Poisson shrinker)"],
                key=lambda n: average_precision_score(y[va], PRED[n][va]))
COMPS = [("H1", "Fusion + full reading", "Fusion, count evidence only"),
         ("H1", "GBM, count + full-reading narrative features", "GBM, count features"),
         ("H1", "Fusion + full reading", best_disp),
         ("H2", gen20, "Fusion + full reading"),
         ("H2", gen20, "Fusion, count evidence only"),
         ("H3", aeg, gen20),
         ("H3", "AEGIS, all four gaps", "Fusion, count evidence only"),
         ("H3", "Gap retrieval, all four gaps, pooled over 3 months", "AEGIS, all four gaps"),
         ("H3", "Gap retrieval, all four gaps, k=30 per gap", "AEGIS, all four gaps"),
         ("other", "Text model (embeddings + LR)", "Fusion + full reading"),
         ("other", "Fusion + full reading", "Recall history (logistic regression)"),
         ("history", "Fusion + full reading + recall history", "Recall history (logistic regression)"),
         ("history", "Fusion + full reading + recall history", "Fusion, count evidence + recall history"),
         ("other", "Fusion + full reading, initial reports only", "Fusion + full reading")]
sig = []
for h, a_, b_ in (COMPS if 0 < y[te].sum() < te.sum() else []):
    for met in ["AUPRC", "AUROC"]:
        d = cluster_bootstrap_diff(y[te], PRED[a_][te], PRED[b_][te], code[te], n_boot=N_BOOT, metric=met)
        sig.append(dict(hypothesis=h, method_a=a_, method_b=b_, metric=met, delta=d["diff"], lo=d["lo"], hi=d["hi"],
                        p_one_sided=d["p_le0"]))
SIG = pd.DataFrame(sig); SIG.to_csv(f"{OUT}/paired_differences.csv", index=False); print(SIG.round(4).to_string())
lap("main results + paired differences done")

# %% [markdown]
# ## 6. Horizons 3 and 12 months; washout after any recall class

# %%
hz = []
for h in (3, 6, 12):
  try:
    yy = CASES[f"y{h}"].values
    ph = {"Fusion, count evidence only": fusion(G0, None, yy)[0], "Fusion + full reading": fusion(GA, "full", yy)[0],
          gen20: fusion(GA, "generic20", yy)[0], "GBM, count features": gbm(CASES[T0F].values, yy, tr),
          "GBM, count + full-reading narrative features": gbm(Xfull, yy, tr), "AEGIS, all four gaps": fusion(GC, "gap", yy)[0],
          "EBGM (max, gamma-Poisson shrinker)": PRED["EBGM (max, gamma-Poisson shrinker)"]}
    for nm, p in ph.items():
        m = metrics(yy[te], p[te]); lo, hi = cluster_bootstrap_ci(yy[te], p[te], code[te], n_boot=N_BOOT // 4)
        hz.append(dict(horizon=h, method=nm, prevalence=yy[te].mean(), AUPRC=m["AUPRC"], AUPRC_lo=lo, AUPRC_hi=hi, AUROC=m["AUROC"]))
    d = cluster_bootstrap_diff(yy[te], ph["Fusion + full reading"][te], ph["Fusion, count evidence only"][te], code[te], n_boot=N_BOOT // 2)
    hz.append(dict(horizon=h, method="Δ full reading − count only", prevalence=yy[te].mean(), AUPRC=d["diff"], AUPRC_lo=d["lo"], AUPRC_hi=d["hi"]))
  except Exception as ex:
    print("horizon failed:", h, type(ex).__name__, str(ex)[:120])
HZ = pd.DataFrame(hz); HZ.to_csv(f"{OUT}/horizon_sensitivity.csv", index=False); print(HZ.round(4).to_string())
wa = []
mask = te & (CASES.washout_any.values == 0)
for nm in ["Fusion, count evidence only", "Fusion + full reading", gen20, "GBM, count + full-reading narrative features", "AEGIS, all four gaps"]:
    m = metrics(y[mask], PRED[nm][mask]); wa.append(dict(method=nm, cases=int(mask.sum()), positives=int(y[mask].sum()), AUPRC=m["AUPRC"], AUROC=m["AUROC"]))
pd.DataFrame(wa).to_csv(f"{OUT}/washout_any_class.csv", index=False); print(pd.DataFrame(wa).round(4))
lap("horizons done")

# %% [markdown]
# ## 7. Ablations on the validation period (calibrators fit on train ≤ 2019-06, weights on 2019-07..2020-12)

# %%
ab = []
def val_metrics(name, groups, strategy, use_rho=True, docs=np.nan):
  try:
    p = fusion(groups, strategy, fit=(tra, trb), use_rho=use_rho)[0]
    pt = fusion(groups, strategy, use_rho=use_rho)[0]
    ab.append(dict(variant=name, val_AUPRC=average_precision_score(y[va], p[va]), val_AUROC=roc_auc_score(y[va], p[va]),
                   test_AUPRC=average_precision_score(y[te], pt[te]), test_AUROC=roc_auc_score(y[te], pt[te])))
  except Exception as ex:
    print("ablation failed:", name, type(ex).__name__, str(ex)[:120])
val_metrics("Fusion, count evidence only", G0, None)
val_metrics("Fusion + full reading (reference)", GA, "full")
for drop in DEEP_SOURCES:
    val_metrics(f"Full reading without {drop} evidence", {k: v for k, v in GA.items() if k != drop}, "full")
val_metrics("Full reading without contradictory evidence", {k: [f for f in v if f != "inv_refute"] for k, v in GA.items()}, "full")
val_metrics("Full reading without reliability discounting", GA, "full", use_rho=False)
val_metrics("AEGIS, all four gaps (reference)", GC, "gap")
val_metrics("AEGIS without query-rate statistic", {k: [f for f in v if not f.startswith("rate_")] for k, v in GC.items()}, "gap")
val_metrics("AEGIS without contrast", GA, "gap_nc")
val_metrics("AEGIS without reliability discounting", GC, "gap", use_rho=False)
val_metrics("AEGIS without contradictory evidence", {k: [f for f in v if f != "inv_refute"] for k, v in GC.items()}, "gap")
for drop in DEEP_SOURCES:
    val_metrics(f"AEGIS without {drop} evidence", {k: v for k, v in GC.items() if k != drop}, "gap")
AB = pd.DataFrame(ab); AB.to_csv(f"{OUT}/ablations_validation.csv", index=False); print(AB.round(4).to_string())
# source correlation (fusion assumes conditional independence of sources)
Lsrc = pd.DataFrame(FZF.contributions(XF)[tr], columns=SRCF)
CORR = Lsrc.corr(method="spearman"); CORR.to_csv(f"{OUT}/source_llr_correlation.csv")
off = CORR.where(~np.eye(len(CORR), dtype=bool)).abs()
print("max |Spearman| between source log-odds contributions:", round(float(np.nanmax(off.values)), 3))
WEIGHTS = dict(zip(SRCF, FZF.w.round(4).tolist())); WEIGHTS["b0"] = round(FZF.b0, 4)
lap("ablations done")

# %% [markdown]
# ## 8. Per-event and per-code results

# %%
pe = []
pc_rank = {n: pd.Series(PRED[n][te]).rank(pct=True).values for n in ["Fusion, count evidence only", "Fusion + full reading", gen20]}
dte = CASES[te].reset_index(drop=True)
for c, e in TEST_EVENTS:
    w_ = ((dte.code == c) & (dte.mi >= e - CFG["HORIZON"]) & (dte.mi < e)).values
    if not w_.any(): continue
    row = dict(code=c, event_month=ym_str(e), positive_months=int(w_.sum()))
    for n, rk in pc_rank.items(): row[f"max_pct_rank::{n}"] = float(rk[w_].max())
    for n in ["Fusion, count evidence only", "Fusion + full reading", gen20, "GBM, count + full-reading narrative features"]:
        row[f"detected::{n}"] = bool((PRED[n][te][w_] >= thr(PRED[n][va])).any())
    pe.append(row)
PE = pd.DataFrame(pe); PE.to_csv(f"{OUT}/per_event_results.csv", index=False)
def boot_mean(x, n=2000, seed=0):
    r_ = np.random.default_rng(seed); x = np.asarray(x, float)
    b_ = [r_.choice(x, len(x)).mean() for _ in range(n)]; return float(x.mean()), float(np.percentile(b_, 2.5)), float(np.percentile(b_, 97.5))
EV_DET = {} if not len(PE) else {n: boot_mean(PE[f"detected::{n}"]) for n in ["Fusion, count evidence only", "Fusion + full reading", gen20, "GBM, count + full-reading narrative features"]}
print("event-level detection (mean, 95% CI):", {k: tuple(round(v, 3) for v in t) for k, t in EV_DET.items()})
# leave-one-code-out influence on ΔAUPRC (full reading − count only)
codes_te = sorted(set(code[te])); loo = []
pa, pb = PRED["Fusion + full reading"], PRED["Fusion, count evidence only"]
for c in (codes_te if 0 < y[te].sum() < te.sum() else []):
    m = te & (code != c)
    if y[m].sum() == 0: continue
    loo.append(dict(code=c, positives_in_code=int(y[te & (code == c)].sum()),
                    delta_AUPRC=average_precision_score(y[m], pa[m]) - average_precision_score(y[m], pb[m])))
LOO = pd.DataFrame(loo, columns=["code", "positives_in_code", "delta_AUPRC"]).sort_values("delta_AUPRC")
LOO.to_csv(f"{OUT}/leave_one_code_out.csv", index=False)
if len(LOO): print("LOO ΔAUPRC range:", round(float(LOO.delta_AUPRC.min()), 4), round(float(LOO.delta_AUPRC.max()), 4))
lap("per-event done")

# %% [markdown]
# ## 9. Figures

# %%
plt.rcParams.update({"font.size": 9})
def make_figures():
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    r0 = RES.set_index("method")
    ax.axhline(r0.loc["Fusion, count evidence only", "AUPRC"], color="#888", ls=":", lw=1)
    ax.text(DOCS["Fusion + full reading"] + 2, r0.loc["Fusion, count evidence only", "AUPRC"] + 0.002, "Count evidence only", ha="right", fontsize=8, color="#555")
    ax.axhline(y[te].mean(), color="#bbb", ls="--", lw=1)
    ax.text(DOCS["Fusion + full reading"] + 2, y[te].mean() + 0.002, "Prevalence", ha="right", fontsize=8, color="#777")
    ax.errorbar(RR.reports_read, RR.AUPRC_mean, yerr=RR.AUPRC_sd, fmt="o--", color="#9a9a9a", capsize=3, label=f"Random reading (mean ± SD, {N_RAND} draws)")
    gx = [DOCS[f"Fusion + generic retrieval (k={k})"] for k in KS]; gy = [r0.loc[f"Fusion + generic retrieval (k={k})", "AUPRC"] for k in KS]
    ax.plot(gx, gy, "s-", color="#5f5f5f", label="Generic retrieval")
    ax.plot([DOCS["Fusion + full reading"]], [r0.loc["Fusion + full reading", "AUPRC"]], "D", color="k", label="Full reading")
    an = [n for n in PRED if FAMILY[n] in ("AEGIS",)]
    an = sorted(an, key=lambda n: DOCS[n])
    ax.plot([DOCS[n] for n in an], [r0.loc[n, "AUPRC"] for n in an], "o-", color="#1f5fbf", label="AEGIS policies")
    for n in ["Gap retrieval, all four gaps, k=30 per gap", "Gap retrieval, all four gaps, pooled over 3 months"]:
        ax.plot([DOCS[n]], [r0.loc[n, "AUPRC"]], "^", color="#1f5fbf", mfc="white")
    ax.plot([], [], "^", color="#1f5fbf", mfc="white", label="Gap retrieval, larger/pooled evidence")
    ax.set_xlabel("Reports read per case (mean)"); ax.set_ylabel("Test AUPRC")
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.legend(frameon=False, fontsize=7.5, loc="lower right"); fig.tight_layout()
    fig.savefig(f"{FIG}/fig_auprc_vs_reports.png", dpi=300); fig.savefig(f"{FIG}/fig_auprc_vs_reports.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    for n, col in [("Fusion, count evidence only", "#9a9a9a"), ("Fusion + full reading", "k"), ("GBM, count + full-reading narrative features", "#1f5fbf")]:
        p = PRED[n][te]; q = pd.qcut(pd.Series(p).rank(method="first"), 10, labels=False)
        d = pd.DataFrame(dict(p=p, y=y[te], q=q)).groupby("q").mean()
        ax.plot(d.p, d.y, "o-", color=col, ms=4, label=n)
    mx = max(0.4, float(np.nanmax([PRED[n][te].max() for n in PRED if FAMILY[n] != "routine-unsupervised"])))
    ax.plot([0, mx], [0, mx], ":", color="#bbb"); ax.set_xlim(0, mx); ax.set_ylim(0, mx)
    ax.set_xlabel("Predicted probability (decile mean)"); ax.set_ylabel("Observed proportion"); ax.legend(frameon=False, fontsize=7)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    fig.tight_layout(); fig.savefig(f"{FIG}/fig_calibration.png", dpi=300); fig.savefig(f"{FIG}/fig_calibration.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})

    if len(PE):
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        a_ = PE["max_pct_rank::Fusion, count evidence only"]; b_ = PE["max_pct_rank::Fusion + full reading"]
        ax.scatter(a_, b_, s=18, color="#1f5fbf", alpha=.8)
        ax.plot([0, 1], [0, 1], ":", color="#bbb"); ax.axhline(0.9, color="#ddd", lw=.8); ax.axvline(0.9, color="#ddd", lw=.8)
        ax.set_xlabel("Count evidence only: best percentile rank before recall"); ax.set_ylabel("Full reading: best percentile rank before recall")
        ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.01)
        for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
        fig.tight_layout(); fig.savefig(f"{FIG}/fig_per_event.png", dpi=300); fig.savefig(f"{FIG}/fig_per_event.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
try:
    make_figures()
except Exception as ex:
    print("figures failed:", type(ex).__name__, str(ex)[:150])

# %% [markdown]
# ## 10. Blind labelling sheets (stance validation, leakage check)

# %%
rs = np.random.default_rng(2026)
cand = np.where(ANN.has_mfr.values & np.isfinite(ANN.p_confirm.values))[0]
if len(cand):
    sc = ANN.p_confirm.values[cand] - ANN.p_refute.values[cand]
    qs = pd.qcut(pd.Series(sc).rank(method="first"), 5, labels=False).values
    pick_ = np.concatenate([rs.choice(cand[qs == q], min(40, (qs == q).sum()), replace=False) for q in range(5)])
    rs.shuffle(pick_)
    sheet = pd.DataFrame(dict(item_id=[f"S{i:03d}" for i in range(len(pick_))], product_code=R.code.values[pick_],
                              manufacturer_narrative=R.text_mfr.values[pick_],
                              label="", notes=""))
    sheet.to_csv(f"{OUT}/labelling_stance_200.csv", index=False)
    pd.DataFrame(dict(item_id=sheet.item_id, rid=R.rid.values[pick_], p_confirm=ANN.p_confirm.values[pick_],
                      p_refute=ANN.p_refute.values[pick_])).to_csv(f"{OUT}/labelling_stance_200_KEY_do_not_open.csv", index=False)
pos_rows = np.concatenate([rows_in(c, int(t) - 2, int(t)) for c, t in zip(code[y == 1], MI[y == 1])]) if (y == 1).any() else np.array([], int)
pos_rows = np.unique(pos_rows)
lk = np.concatenate([rs.choice(pos_rows, min(200, len(pos_rows)), replace=False),
                     rs.choice(np.setdiff1d(np.arange(len(R)), pos_rows), 100, replace=False)])
rs.shuffle(lk)
SUSPECT = re.compile(r"(recall|field|correction|removal|advisory|notice|fsn|fsca|corrective action|withdraw|quarantin|retrofit|upgrade kit|customer letter|dear (doctor|customer|healthcare))", re.I)
leak = pd.DataFrame(dict(item_id=[f"L{i:03d}" for i in range(len(lk))], product_code=R.code.values[lk],
                         event_narrative=R.text_event.values[lk], manufacturer_narrative=R.text_mfr.values[lk],
                         mentions_recall_or_field_action="", notes=""))
leak.to_csv(f"{OUT}/labelling_leakage_300.csv", index=False)
auto_hits = (leak.event_narrative.fillna("") + " " + leak.manufacturer_narrative.fillna("")).str.contains(SUSPECT)
corpus_hits = (R.text_event.fillna("") + " " + R.text_mfr.fillna("")).str.contains(SUSPECT).mean()
lap(f"labelling sheets written; residual suspect-term rate: sample {auto_hits.mean():.3f}, corpus {corpus_hits:.3f}")

# %%
meta = dict(gps_prior={k: float(v) for k, v in GPS_PRIOR.items()}, fusion_weights_full=WEIGHTS, text_model_C=C_txt,
            text_pca_var=float(pca.explained_variance_ratio_.sum()), gbm_params=GBM_PARAMS, alert_rate=ALERT_RATE,
            n_boot=N_BOOT, n_random_draws=N_RAND, best_disproportionality=best_disp, aegis_pick=dict(policy=pick["pol"], eps=pick["eps"]),
            unclassified_recall_rows=float(unclassified), initial_only_share=float(INITIAL_ONLY.mean()),
            event_detection=EV_DET, positive_months_per_event=float(PE.positive_months.mean()) if len(PE) else None,
            max_abs_source_corr=float(np.nanmax(off.values)), codes_without_test_cases=miss,
            residual_suspect_term_rate=float(corpus_hits), test_events=len(TEST_EVENTS),
            thresholds=dict(THR), runtime_min=(time.time() - T_START) / 60)
json.dump(meta, open(f"{OUT}/revision_meta.json", "w"), indent=1, default=str)
PRED_DF = CASES[["case_id", "code", "mi", "split", "y", "y3", "y12", "washout_any"]].copy()
for k_, v_ in PRED.items(): PRED_DF[k_] = v_
PRED_DF.to_pickle(f"{OUT}/predictions_v2.pkl")
lap("DONE")
