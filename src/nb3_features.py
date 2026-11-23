# %% [markdown]
# # AEGIS-MD · Notebook 3 — Benchmark construction, routine detectors & gap-driven retrieval (CPU)
# Inputs: attach the output of **Notebook 2**.
#
# 1. Builds the **leakage-free recall benchmark**: one case per (product code, month *t*); label = a new recall of the target class is initiated in *(t, t+6]*; months right after a recall are washed out (stimulated reporting); strict temporal train / validation / test split.
# 2. **Tier-0 routine surveillance evidence** (what standard post-market surveillance already sees): Poisson exceedance, CUSUM, Mann–Kendall trend & Sen slope, persistence, Isolation Forest anomaly, severity shift, problem-code distribution shift (Jensen–Shannon).
# 3. **Deep evidence** that requires *reading* reports, obtained through different retrieval strategies over the recent reports of that code:
#    * **Gap-driven hybrid retrieval** (ours): one BM25 + dense query per evidence gap (mechanism, manufacturer investigation, batch/lot, recall precedent), conditioned on the signal's emerging problem terms;
#    * generic-query RAG, random reading, and full reading (upper bound).

# %%
# %%writefile-core

# %%
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from aegis_core import *
from collections import Counter
OUT = out_dir()

R = pd.read_pickle(find_file("records.pkl"))
ANN = pd.read_pickle(find_file("record_ann.pkl"))
E = np.load(find_file("rec_emb.npy"))
PROTO = pd.read_pickle(find_file("proto_emb.pkl"))
COUNTS = pd.read_pickle(find_file("counts.pkl"))
CODES = pd.read_pickle(find_file("codes.pkl"))
REC = pd.read_pickle(find_file("recalls.pkl"))
assert (ANN.rid.values == R.rid.values).all() and len(E) == len(R)
print(len(R), "records |", len(CODES), "codes |", len(REC), "recall rows")

MAND = R.source.str.lower().str.contains("manufacturer|user facility|distributor|importer").values
RA = dict(emb=E, has_text=(R.text_event.fillna("").str.len() > 20).values, mand_src=MAND,
          has_mfr=ANN.has_mfr.values.astype(bool), p_confirm=ANN.p_confirm.values, p_refute=ANN.p_refute.values,
          lot=R.lot.fillna("").replace({"UNK": "", "UNKNOWN": "", "NA": "", "N/A": "", "NI": ""}).values,
          model=R.model.fillna("").replace({"UNK": "", "UNKNOWN": "", "NA": "", "N/A": "", "NI": ""}).values,
          prec_sim=ANN.prec_sim.values.astype(np.float32))

# %% [markdown]
# ## 1. Labels (recall ground truth) and case panel

# %%
H, W = CFG["HORIZON"], CFG["WASHOUT"]
lab = REC[REC.klass.isin(CFG["LABEL_CLASSES"])]
ev = lab.groupby(["code", "event_id"]).init_mi.min().reset_index()
EVENTS = ev.groupby("code").init_mi.apply(lambda s: sorted(set(int(x) for x in s))).to_dict()
print("label classes:", CFG["LABEL_CLASSES"], "| recall events in panel codes:",
      sum(len(EVENTS.get(c, [])) for c in CODES.code))

R_SER = R.event_type.str.lower().isin(["death", "injury"]).values
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

cnt = COUNTS.set_index(["code", "mi"]).n.to_dict()
p0, p1 = ym(CFG["PANEL_START"]), ym(CFG["PANEL_END"])
first_m = ym(CFG["DATA_START"]) + CFG["RECENT"] + CFG["BASELINE"]
rows = []
for c in CODES.code:
    nb = {m: v for (cc, m), v in cnt.items() if cc == c}
    ag = {m: v for (cc, m), v in agg.items() if cc == c}
    t0 = tier0_for_code(nb, range(first_m, p1 + 1), ag)
    t0 = t0[t0.mi >= p0].copy(); t0["code"] = c
    evs = EVENTS.get(c, [])
    t0["y"] = [int(any(t < e <= t + H for e in evs)) for t in t0.mi]
    t0["washout"] = [int(any(t - W < e <= t for e in evs)) for t in t0.mi]
    rows.append(t0)
CASES = pd.concat(rows, ignore_index=True)
CASES = CASES[CASES.washout == 0].reset_index(drop=True)
CASES["split"] = [split_of(m) for m in CASES.mi]
CASES["case_id"] = np.arange(len(CASES))
print(CASES.groupby("split").y.agg(["count", "sum", "mean"]))

# Isolation Forest on routine multivariate surveillance features (fit on train only)
from sklearn.ensemble import IsolationForest
IF_FEATS = ["pois_score", "log_ratio", "mk_z", "sen_norm", "persist", "cusum", "sev_z", "prob_jsd"]
iso = IsolationForest(n_estimators=300, contamination="auto", random_state=CFG["SEED"]).fit(
    CASES.loc[CASES.split == "train", IF_FEATS].fillna(0))
CASES["if_score"] = -iso.score_samples(CASES[IF_FEATS].fillna(0))

# %% [markdown]
# ## 2. Retrieval index (BM25 sparse + dense embeddings)

# %%
t0 = time.time()
doc_text = (R.text_event.fillna("") + " " + R.text_mfr.fillna("") + " " + R.problems.fillna("").str.replace(";", " "))
BM = BM25Matrix(doc_text.values)
print("BM25 vocab", len(BM.vocab), f"{time.time() - t0:.0f}s")
BY = R.groupby(["code", "mi"]).rid.apply(np.array).to_dict()
names = CODES.set_index("code").device_name.to_dict()

def rows_in(c, lo, hi):
    parts = [BY[(c, m)] for m in range(lo, hi + 1) if (c, m) in BY]
    return np.concatenate(parts) if parts else np.array([], dtype=int)

# %% [markdown]
# ## 3. Deep evidence under each retrieval strategy

# %%
K = CFG["K_PER_GAP"]; KS = [10, 20, 40]
# corpus-level similarity thresholds (90th percentile) for the query-rate statistic
_samp = rng_thr = np.random.default_rng(0).choice(len(E), min(50000, len(E)), replace=False)
THR = {g: float(np.quantile(E[_samp].astype(np.float32) @ PROTO[g], 0.90)) for g in GAP_SPECS}
rng = np.random.default_rng(CFG["SEED"])
feat = {s: [] for s in ["full", "gap", "gap_nc"] + [f"generic{k}" for k in KS] + [f"random{k}" for k in KS]}
S_GAP, S_GAPB, S_GEN, CTX = {}, {}, {}, {}
t0 = time.time()
for i, r in CASES.iterrows():
    c, t = r.code, int(r.mi)
    win = rows_in(c, t - CFG["RECENT"] + 1, t)
    base = rows_in(c, t - CFG["RECENT"] - CFG["BASELINE"] + 1, t - CFG["RECENT"])
    base_all = base
    if len(base) > 400: base = rng.choice(base, 400, replace=False)
    if len(base):
        Eb = E[base].astype(np.float32); cb = Eb.mean(0); cb /= (np.linalg.norm(cb) + 1e-9)
        ctx = dict(base_cen=cb, base_coh=float((Eb @ cb).mean()))
    else:
        ctx = dict(base_cen=None, base_coh=0.0)
    wp, bp = Counter(), Counter()
    for m in range(t - CFG["RECENT"] + 1, t + 1): wp.update(agg.get((c, m), {}).get("prob", {}))
    for m in range(t - CFG["RECENT"] - CFG["BASELINE"] + 1, t - CFG["RECENT"] + 1): bp.update(agg.get((c, m), {}).get("prob", {}))
    em_phr, em_tok = emerging_terms(wp, bp)
    dn_tok = tokenize(names.get(c, ""))
    CTX[r.case_id] = dict(emerging=em_phr, n_window=len(win))

    f = deep_evidence(win, RA, ctx); f["n_docs"] = len(win); feat["full"].append(f)
    g_out, g_nc, s_lists, b_lists = {"n_window": len(win)}, {"n_window": len(win)}, {}, {}
    for gname, spec in GAP_SPECS.items():
        q = spec["terms"] + (em_tok + dn_tok if spec["use_emerging"] else [])
        S = hybrid_retrieve(win, BM, E, q, PROTO[gname], K)
        Sb = hybrid_retrieve(base, BM, E, q, PROTO[gname], K) if len(base) else np.array([], dtype=int)
        s_lists[gname] = S; b_lists[gname] = Sb
        g_nc.update(deep_evidence(S, RA, ctx, sources=[gname]))
        g_out.update(contrastive_gap_evidence(S, Sb, RA, ctx, gname,
                                              rate=query_rate_shift(win, base_all, E, PROTO[gname], THR[gname])))
    feat["gap"].append(g_out); feat["gap_nc"].append(g_nc); S_GAP[r.case_id] = s_lists; S_GAPB[r.case_id] = b_lists
    gq = GENERIC_QUERY["terms"] + dn_tok
    for k in KS:
        S = hybrid_retrieve(win, BM, E, gq, PROTO["generic"], k)
        if k == K: S_GEN[r.case_id] = S
        f = deep_evidence(S, RA, ctx); f["n_docs"] = len(S); feat[f"generic{k}"].append(f)
        S = rng.choice(win, min(k, len(win)), replace=False) if len(win) else win
        f = deep_evidence(S, RA, ctx); f["n_docs"] = len(S); feat[f"random{k}"].append(f)
    if i % 1000 == 0: print(i, "/", len(CASES), f"{time.time() - t0:.0f}s")

FEAT = {s: pd.DataFrame(v, index=CASES.case_id.values) for s, v in feat.items()}
print({s: v.shape for s, v in FEAT.items()})

# %%
CASES.to_pickle(f"{OUT}/cases.pkl")
pd.to_pickle(FEAT, f"{OUT}/deep_features.pkl")
pd.to_pickle(dict(S_GAP=S_GAP, S_GAPB=S_GAPB, S_GEN=S_GEN, CTX=CTX), f"{OUT}/retrieval_sets.pkl")
pd.to_pickle(EVENTS, f"{OUT}/events.pkl")
summ = CASES.groupby("split").agg(cases=("y", "size"), positives=("y", "sum"), codes=("code", "nunique"))
summ.to_csv(f"{OUT}/benchmark_summary.csv"); print(summ)
# carry forward files needed by Notebooks 4-5
import shutil
for f in ["records.pkl", "recalls.pkl", "codes.pkl", "record_ann.pkl", "recall_texts.pkl", "enrich_meta.json", "data_summary.json"]:
    src = find_file(f, required=False)
    if src and os.path.abspath(src) != os.path.abspath(f"{OUT}/{f}"): shutil.copy(src, f"{OUT}/{f}")
