# %% [markdown]
# # AEGIS-MD · Notebook 4 — Evidence fusion, Value-of-Information gap policy & full evaluation (CPU)
# Inputs: attach the outputs of **Notebook 2** and **Notebook 3**.
#
# * **Reliability-discounted Bayesian evidence fusion**: every evidence source is converted to a calibrated log-likelihood ratio (fit on train), discounted by the reliability of the documents it came from, and pooled with weights learned on validation. Contradictory evidence (e.g. a manufacturer investigation that could not confirm the failure) enters as negative log-odds.
# * **Evidence-gap analysis as Value of Information (VoI)**: for each missing evidence type, the expected reduction in Bayes decision risk if it were retrieved. The investigation loop retrieves the highest-VoI gap, re-evaluates, and stops when no gap is worth its reading cost.
# * Baselines: Poisson, CUSUM, Mann–Kendall, Isolation Forest, gradient boosting (routine and full features), fusion with generic-query RAG, random reading and full reading.
# * Metrics on the held-out **test period**: AUPRC, AUROC, Brier, documents read, recall events detected and **lead time before recall initiation**, product-code cluster bootstrap CIs.

# %%
# %%writefile-core

# %%
import os, sys, json, time
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.getcwd())
from aegis_core import *
OUT = out_dir(); FIG = os.path.join(OUT, "figures"); os.makedirs(FIG, exist_ok=True)

CASES = pd.read_pickle(find_file("cases.pkl"))
FEAT = pd.read_pickle(find_file("deep_features.pkl"))
RS = pd.read_pickle(find_file("retrieval_sets.pkl"))
EVENTS = pd.read_pickle(find_file("events.pkl"))
y = CASES.y.values; split = CASES.split.values; code = CASES.code.values
tr, va, te = split == "train", split == "val", split == "test"
print(CASES.groupby("split").y.agg(["size", "sum"]))
ALERT_RATE = 0.10       # alert threshold = top 10% of validation risk scores
RESULTS, PRED = [], {}

def thr(p_val): return float(np.quantile(p_val, 1 - ALERT_RATE))
test_events = [(c, e) for c, evs in EVENTS.items() if c in set(code[te])
               for e in evs if e > ym(CFG["VAL_END"]) + 1 and e <= ym(CFG["PANEL_END"]) + CFG["HORIZON"]]
print("test recall events:", len(test_events))

def record(name, p_all, docs=0.0, family="", tau=None):
    m = metrics(y[te], p_all[te]); lo, hi = cluster_bootstrap_ci(y[te], p_all[te], code[te], n_boot=500)
    tau = thr(p_all[va]) if tau is None else tau
    lt = lead_time_table(CASES[te].reset_index(drop=True), p_all[te], tau, test_events, CFG["HORIZON"])
    det = lt.detected.mean() if len(lt) else np.nan
    lead = lt.loc[lt.detected, "lead"].median() if len(lt) and lt.detected.any() else np.nan
    RESULTS.append(dict(method=name, family=family, docs_read=round(float(docs), 2), AUPRC=m["AUPRC"],
                        AUPRC_lo=lo, AUPRC_hi=hi, AUROC=m["AUROC"], Brier=m["Brier"],
                        events_detected=det, median_lead_months=lead))
    PRED[name] = p_all
    print(f"{name:38s} docs={docs:6.1f}  AUPRC={m['AUPRC']:.3f} [{lo:.3f},{hi:.3f}]  AUROC={m['AUROC']:.3f}  detected={det:.2f}  lead={lead}")

# %% [markdown]
# ## 1. Routine surveillance baselines (unsupervised scores)

# %%
for f, nm in [("pois_score", "Poisson exceedance"), ("cusum", "Poisson CUSUM"), ("mk_z", "Mann-Kendall trend"),
              ("if_score", "Isolation Forest")]:
    s = CASES[f].fillna(0).values
    s = (s - s[tr].min()) / (s[tr].max() - s[tr].min() + 1e-9)
    record(nm, s, family="routine-unsupervised")

# %% [markdown]
# ## 2. Supervised black-box baselines

# %%
from sklearn.ensemble import HistGradientBoostingClassifier
T0F = sum(TIER0_GROUPS.values(), [])
DF = sum(DEEP_FEATURES.values(), [])
def gbm(X):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                       early_stopping=True, validation_fraction=0.15, random_state=CFG["SEED"])
    m.fit(X[tr], y[tr]); return m.predict_proba(X)[:, 1]
X0 = CASES[T0F].values
Xfull = np.column_stack([X0, FEAT["full"][DF].values])
record("GBM (routine features)", gbm(X0), family="supervised")
record("GBM (routine + all deep, full reading)", gbm(Xfull), docs=FEAT["full"].n_docs[te].mean(), family="supervised")

# %% [markdown]
# ## 3. Evidence fusion with different reading strategies

# %%
G0 = dict(TIER0_GROUPS)
GA = {**TIER0_GROUPS, **DEEP_FEATURES}
def Xof(strategy):
    X = CASES.set_index("case_id")[T0F].copy()
    if strategy: X = X.join(FEAT[strategy])
    return X.reset_index(drop=True)

fz0 = EvidenceFusion(G0).fit(Xof(None)[tr], y[tr], Xof(None)[va], y[va], groups_tr=code[tr])
record("Fusion (routine evidence only)", fz0.predict(Xof(None)), family="fusion")

for s in ["random10", "random20", "random40", "generic10", "generic20", "generic40", "full"]:
    X = Xof(s); fz = EvidenceFusion(GA).fit(X[tr], y[tr], X[va], y[va], groups_tr=code[tr])
    lab = {"full": "Fusion + full reading (all recent reports)"}.get(s, f"Fusion + {'generic-query RAG' if s.startswith('generic') else 'random reading'} (k={s[-2:]})")
    record(lab, fz.predict(X), docs=FEAT[s].n_docs[te].mean(), family="fusion+reading")

# %% [markdown]
# ## 4. AEGIS: gap-driven retrieval + VoI investigation loop

# %%
XG = Xof("gap")
GC = {**TIER0_GROUPS, **CONTRAST_FEATURES}          # contrastive gap evidence + query-rate statistic
FZ = EvidenceFusion(GC).fit(XG[tr], y[tr], XG[va], y[va], groups_tr=code[tr])
SRC = list(GC.keys()); n0 = len(TIER0_GROUPS)
DEEP_IDX = {s: SRC.index(s) for s in DEEP_SOURCES}
CONTRIB = FZ.contributions(XG)                      # (n_cases, n_sources)  w * rho * LLR
Z0 = FZ.b0 + CONTRIB[:, :n0].sum(1)
TAU = thr(FZ.predict(XG)[va])
print("pooling weights:", dict(zip(SRC, np.round(FZ.w, 3))), "| tau =", round(TAU, 4))

def docs_read(cid, acquired):
    """Reports read = retrieved recent-window reports + retrieved baseline reports (both are read)."""
    if not acquired: return 0
    S, Sb = RS["S_GAP"][cid], RS.get("S_GAPB", {}).get(cid, {})
    w = set().union(*[set(S[g].tolist()) for g in acquired])
    b = set().union(*[set(np.asarray(Sb.get(g, [])).tolist()) for g in acquired]) if Sb else set()
    return len(w) + len(b)

CIDS = CASES.case_id.values
def run_policy(policy, eps=0.0, B=4, seed=0):
    rng = np.random.default_rng(seed)
    P = np.zeros(len(CASES)); D = np.zeros(len(CASES)); ACQ = []
    for i in range(len(CASES)):
        z = Z0[i]; acq = []
        for step in range(B):
            rem = [g for g in DEEP_SOURCES if g not in acq]
            if not rem: break
            if policy == "voi":
                v = {g: voi(z, g, FZ, TAU, DEEP_IDX[g]) for g in rem}
                g = max(v, key=v.get)
                if v[g] <= eps: break
            elif policy == "entropy":
                v = {g: entropy_gain(z, g, FZ, DEEP_IDX[g]) for g in rem}
                g = max(v, key=v.get)
                if v[g] <= eps: break
            elif policy == "fixed":
                g = rem[0]
            elif policy == "random":
                g = rem[rng.integers(len(rem))]
            acq.append(g); z += CONTRIB[i, DEEP_IDX[g]]
        P[i] = 1 / (1 + np.exp(-z)); D[i] = docs_read(CIDS[i], acq); ACQ.append(acq)
    return P, D, ACQ

curve = []
for B in [1, 2, 3, 4]:
    for pol in ["fixed", "random"]:
        P, D, _ = run_policy(pol, B=B)
        record(f"Gap retrieval, {pol} order (B={B})", P, docs=D[te].mean(), family="gap-ablation")
        curve.append(dict(policy=pol, param=B, docs=D[te].mean(), AUPRC=metrics(y[te], P[te])["AUPRC"]))
EPS = [0.0, 1e-5, 5e-5, 1e-4, 3e-4, 1e-3, 3e-3]
voi_runs = {}
for eps in EPS:
    for pol in ["voi", "entropy"]:
        e = eps if pol == "voi" else eps * 20
        P, D, A = run_policy(pol, eps=e)
        voi_runs[(pol, eps)] = (P, D, A)
        curve.append(dict(policy=pol, param=e, docs=D[te].mean(), AUPRC=metrics(y[te], P[te])["AUPRC"],
                          val_AUPRC=metrics(y[va], P[va])["AUPRC"], val_docs=D[va].mean()))
CURVE = pd.DataFrame(curve); CURVE.to_csv(f"{OUT}/efficiency_curve.csv", index=False)

# choose the operating point on VALIDATION only: among both VoI utilities (decision risk / information),
# the cheapest setting whose validation AUPRC is within 2% of the best validation AUPRC
cv = CURVE[CURVE.policy.isin(["voi", "entropy"])].copy()
ok = cv[cv.val_AUPRC >= 0.98 * cv.val_AUPRC.max()].sort_values("val_docs")
POL_STAR = ok.iloc[0].policy
EPS_STAR = [e for e in EPS if abs((e if POL_STAR == "voi" else e * 20) - ok.iloc[0].param) < 1e-12][0]
P_A, D_A, A_A = voi_runs[(POL_STAR, EPS_STAR)]
UTIL = {"voi": "decision-risk", "entropy": "information"}[POL_STAR]
record(f"AEGIS (VoI gap policy, {UTIL} utility)", P_A, docs=D_A[te].mean(), family="AEGIS")
for pol in ["voi", "entropy"]:
    if pol == POL_STAR: continue
    sub = CURVE[CURVE.policy == pol]; best_e = sub.sort_values("val_docs")[sub.val_AUPRC >= 0.98 * sub.val_AUPRC.max()].iloc[0].param
    e0 = [e for e in EPS if abs((e if pol == "voi" else e * 20) - best_e) < 1e-12][0]
    P_o, D_o, _ = voi_runs[(pol, e0)]
    record(f"Gap retrieval, VoI {'decision-risk' if pol == 'voi' else 'information'} utility (alt.)", P_o, docs=D_o[te].mean(), family="gap-ablation")
P_all, D_all, _ = run_policy("fixed", B=4)
record("AEGIS-all-gaps (every gap retrieved)", P_all, docs=D_all[te].mean(), family="AEGIS")

# %% [markdown]
# ## 4b. AEGIS adaptive investigation: VoI decides *which signals* to investigate and *which evidence* to read
# Evidence for an investigated case is extracted from the full set of recent reports of that product code;
# cost = number of reports read. Compared with reading everything, triage by routine risk, and random triage.

# %%
XF = Xof("full")
FZF = EvidenceFusion(GA).fit(XF[tr], y[tr], XF[va], y[va], groups_tr=code[tr])
SRCF = list(GA.keys()); DIDX = {s: SRCF.index(s) for s in DEEP_SOURCES}
CF = FZF.contributions(XF); Z0F = FZF.b0 + CF[:, :n0].sum(1)
NW = FEAT["full"].n_docs.fillna(0).values.astype(float)
TAUF = thr(FZF.predict(XF)[va])
P_FULL = FZF.predict(XF)

def run_gate(policy, eps=0.0, B=4):
    P = np.zeros(len(CASES)); D = np.zeros(len(CASES)); ACQ = []
    for i in range(len(CASES)):
        z = Z0F[i]; acq = []
        for _ in range(B):
            rem = [g for g in DEEP_SOURCES if g not in acq]
            if not rem: break
            if policy == "voi": v = {g: voi(z, g, FZF, TAUF, DIDX[g]) for g in rem}
            else: v = {g: entropy_gain(z, g, FZF, DIDX[g]) for g in rem}
            g = max(v, key=v.get)
            if v[g] <= eps: break
            acq.append(g); z += CF[i, DIDX[g]]
        P[i] = 1 / (1 + np.exp(-z)); D[i] = NW[i] if acq else 0.0; ACQ.append(acq)
    return P, D, ACQ

def triage(order_score, frac, seed=0):
    """Read (all deep evidence) for the top `frac` of cases ranked by order_score."""
    k = int(round(frac * len(CASES)))
    pick = np.zeros(len(CASES), bool)
    if k: pick[np.argsort(-order_score)[:k]] = True
    z = Z0F + np.where(pick, CF[:, n0:].sum(1), 0.0)
    return 1 / (1 + np.exp(-z)), np.where(pick, NW, 0.0)

gate_curve, gate_runs = [], {}
for pol, grid in [("voi", [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]), ("entropy", [0.0, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.2])]:
    for eps in grid:
        P, D, A = run_gate(pol, eps); gate_runs[(pol, eps)] = (P, D, A)
        gate_curve.append(dict(policy="AEGIS-" + pol, param=eps, docs=D[te].mean(), read_frac=(D[te] > 0).mean(),
                               AUPRC=metrics(y[te], P[te])["AUPRC"], AUROC=metrics(y[te], P[te])["AUROC"],
                               val_AUPRC=metrics(y[va], P[va])["AUPRC"], val_docs=D[va].mean()))
rng_t = np.random.default_rng(7)
for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
    for nm, sc in [("triage-by-routine-risk", Z0F), ("random-triage", rng_t.random(len(CASES)))]:
        P, D = triage(sc, frac)
        gate_curve.append(dict(policy=nm, param=frac, docs=D[te].mean(), read_frac=(D[te] > 0).mean(),
                               AUPRC=metrics(y[te], P[te])["AUPRC"], AUROC=metrics(y[te], P[te])["AUROC"],
                               val_AUPRC=metrics(y[va], P[va])["AUPRC"], val_docs=D[va].mean()))
GCURVE = pd.DataFrame(gate_curve); GCURVE.to_csv(f"{OUT}/adaptive_investigation_curve.csv", index=False)
print(GCURVE.round(4).to_string())

gv = GCURVE[GCURVE.policy.str.startswith("AEGIS")]
ok = gv[gv.val_AUPRC >= 0.98 * gv.val_AUPRC.max()].sort_values("val_docs").iloc[0]
pol_g = ok.policy.split("-")[1]; eps_g = ok.param
P_G, D_G, A_G = gate_runs[(pol_g, eps_g)]
record(f"AEGIS adaptive investigation (VoI-{pol_g})", P_G, docs=D_G[te].mean(), family="AEGIS-adaptive")
frac_match = float((D_G[te] > 0).mean())
for nm, sc in [("Triage by routine risk (matched reading)", Z0F), ("Random triage (matched reading)", np.random.default_rng(11).random(len(CASES)))]:
    P, D = triage(sc, frac_match)
    record(nm, P, docs=D[te].mean(), family="AEGIS-adaptive-baseline")
print(f"AEGIS adaptive: investigates {frac_match:.1%} of test product-code-months, reads {D_G[te].mean():.1f} reports/case vs {NW[te].mean():.1f} for full reading")

# %% [markdown]
# ## 5. Ablations (component contribution)

# %%
XNC = Xof("gap_nc")
fz_ncon = EvidenceFusion(GA).fit(XNC[tr], y[tr], XNC[va], y[va], groups_tr=code[tr])
record("Ablation: gap retrieval without contrast", fz_ncon.predict(XNC),
       docs=np.mean([len(set().union(*[set(RS["S_GAP"][c][g].tolist()) for g in DEEP_SOURCES])) for c in CIDS[te]]), family="ablation")
g_nrate = {k: [f for f in v if not f.startswith("rate_")] for k, v in GC.items()}
fz_nrate = EvidenceFusion(g_nrate).fit(XG[tr], y[tr], XG[va], y[va], groups_tr=code[tr])
record("Ablation: no query-rate statistic", fz_nrate.predict(XG), docs=D_all[te].mean(), family="ablation")
fz_nr = EvidenceFusion(GC, use_rho=False).fit(XG[tr], y[tr], XG[va], y[va], groups_tr=code[tr])
record("Ablation: no reliability discounting", fz_nr.predict(XG), docs=D_all[te].mean(), family="ablation")
g_nc = {k: [f for f in v if f != "inv_refute"] for k, v in GC.items()}
fz_nc = EvidenceFusion(g_nc).fit(XG[tr], y[tr], XG[va], y[va], groups_tr=code[tr])
record("Ablation: no contradictory evidence", fz_nc.predict(XG), docs=D_all[te].mean(), family="ablation")
for drop in DEEP_SOURCES:
    g = {k: v for k, v in GC.items() if k != drop}
    fz = EvidenceFusion(g).fit(XG[tr], y[tr], XG[va], y[va], groups_tr=code[tr])
    record(f"Ablation: without {drop} evidence", fz.predict(XG), docs=D_all[te].mean(), family="ablation")

# %% [markdown]
# ## 6. Significance (paired product-code cluster bootstrap on the test period)

# %%
RES = pd.DataFrame(RESULTS)
aegis = [r for r in RESULTS if r["family"] == "AEGIS-adaptive"][0]["method"]
aegis_gap = [r for r in RESULTS if r["family"] == "AEGIS"][0]["method"]
comps = [aegis_gap, "Triage by routine risk (matched reading)", "Random triage (matched reading)", "Fusion (routine evidence only)", "Fusion + generic-query RAG (k=20)", "Fusion + generic-query RAG (k=40)",
         "Fusion + random reading (k=20)", "GBM (routine features)", "GBM (routine + all deep, full reading)",
         "Fusion + full reading (all recent reports)", "Isolation Forest", "Poisson exceedance"]
sig = []
for cmp_ in comps:
    if cmp_ not in PRED: continue
    for met in ["AUPRC", "AUROC"]:
        d = cluster_bootstrap_diff(y[te], PRED[aegis][te], PRED[cmp_][te], code[te], n_boot=1000, metric=met)
        sig.append(dict(comparison=f"{aegis} vs {cmp_}", metric=met, **d))
SIG = pd.DataFrame(sig); SIG.to_csv(f"{OUT}/significance.csv", index=False)
print(SIG.round(4).to_string())

# %% [markdown]
# ## 7. Does gap-driven retrieval find the *actual* future recall's failure mode? (root-cause alignment)

# %%
REC = pd.read_pickle(find_file("recalls.pkl")); RT = pd.read_pickle(find_file("recall_texts.pkl"))
E = np.load(find_file("rec_emb.npy"), mmap_mode="r"); ER = np.load(find_file("recall_emb.npy"))
tidx = {t: i for i, t in enumerate(RT.text.values)}
REC["text"] = (REC.reason.fillna("") + " " + REC.root_cause.fillna("")).str.strip()
lab = REC[REC.klass.isin(CFG["LABEL_CLASSES"])]
rows = []
rng = np.random.default_rng(1)
for i in np.where(te & (y == 1))[0]:
    c, t, cid = code[i], int(CASES.mi.values[i]), CIDS[i]
    fut = lab[(lab.code == c) & (lab.init_mi > t) & (lab.init_mi <= t + CFG["HORIZON"])].text
    vid = [tidx[x] for x in fut if x in tidx]
    if not vid: continue
    Rv = ER[vid].astype(np.float32)
    def align(S):
        S = np.asarray(S, int)
        return float((E[S].astype(np.float32) @ Rv.T).max()) if len(S) else np.nan
    Sg, Sm = RS["S_GEN"].get(cid, []), RS["S_GAP"][cid]["mechanism"]
    win_n = RS["CTX"][cid]["n_window"]
    union = set()
    for g in DEEP_SOURCES: union |= set(RS["S_GAP"][cid][g].tolist())
    rows.append(dict(case_id=cid, gap_mechanism=align(Sm), generic=align(Sg),
                     gap_union=align(list(union)), n_window=win_n))
ALIGN = pd.DataFrame(rows)
if len(ALIGN):
    from scipy.stats import wilcoxon
    ok_ = ALIGN.dropna()
    w = wilcoxon(ok_.gap_mechanism, ok_.generic) if len(ok_) > 10 else None
    print(ALIGN[["gap_mechanism", "generic", "gap_union"]].describe().round(3))
    print("Wilcoxon gap(mechanism, k=10) vs generic (k=10):", w)
    ALIGN.to_csv(f"{OUT}/root_cause_alignment.csv", index=False)

# %% [markdown]
# ## 8. Figures and tables

# %%
RES = pd.DataFrame(RESULTS); RES.to_csv(f"{OUT}/results_test.csv", index=False)
print(RES[["method", "docs_read", "AUPRC", "AUPRC_lo", "AUPRC_hi", "AUROC", "Brier", "events_detected", "median_lead_months"]].round(3).to_string())

fig, ax = plt.subplots(figsize=(7.5, 5))
for pol, mk in [("voi", "o-"), ("entropy", "s--"), ("fixed", "^-"), ("random", "v:")]:
    d = CURVE[CURVE.policy == pol].sort_values("docs")
    ax.plot(d.docs, d.AUPRC, mk, label={"voi": "VoI, decision-risk utility", "entropy": "VoI, information utility",
                                         "fixed": "Gap retrieval, fixed order", "random": "Gap retrieval, random order"}[pol])
for s, mk in [("generic", "D"), ("random", "x")]:
    pts = RES[RES.method.str.contains("generic-query" if s == "generic" else "random reading")]
    ax.plot(pts.docs_read, pts.AUPRC, mk, ls="-", alpha=.7, label="Generic-query RAG" if s == "generic" else "Random reading")
for pol, mk in [("AEGIS-voi", "o-"), ("AEGIS-entropy", "s-"), ("triage-by-routine-risk", "^--"), ("random-triage", "x:")]:
    d = GCURVE[GCURVE.policy == pol].sort_values("docs")
    ax.plot(d.docs, d.AUPRC, mk, lw=2 if pol.startswith("AEGIS") else 1, label={"AEGIS-voi": "AEGIS adaptive (VoI decision-risk)", "AEGIS-entropy": "AEGIS adaptive (VoI information)", "triage-by-routine-risk": "Triage by routine risk", "random-triage": "Random triage"}[pol])
f0 = RES[RES.method == "Fusion (routine evidence only)"].AUPRC.values[0]
ff = RES[RES.method.str.startswith("Fusion + full")]
ax.axhline(f0, color="gray", ls=":", label="Routine evidence only")
ax.plot(ff.docs_read, ff.AUPRC, "*", ms=14, color="k", label="Full reading")
ax.set_xlabel("Mean documents read per product-code-month"); ax.set_ylabel("Test AUPRC")
ax.set_title("Evidence efficiency: detection quality vs reading cost"); ax.legend(fontsize=8); ax.grid(alpha=.3)
fig.tight_layout(); fig.savefig(f"{FIG}/fig_efficiency_curve.png", dpi=200)

fig, ax = plt.subplots(figsize=(7, 4))
main = RES[RES.family.isin(["routine-unsupervised", "supervised", "fusion", "AEGIS", "AEGIS-adaptive", "AEGIS-adaptive-baseline"]) | RES.method.str.contains("k=20|full reading")]
main = main.sort_values("AUPRC")
ax.barh(main.method, main.AUPRC, xerr=[main.AUPRC - main.AUPRC_lo, main.AUPRC_hi - main.AUPRC], color="#4C72B0")
ax.axvline(y[te].mean(), color="r", ls=":", label="prevalence"); ax.set_xlabel("Test AUPRC (95% cluster-bootstrap CI)")
ax.legend(); fig.tight_layout(); fig.savefig(f"{FIG}/fig_method_comparison.png", dpi=200)

acq = pd.Series([g for a, s in zip(A_G, te) if s for g in a]).value_counts()
n_gaps = pd.Series([len(a) for a, s in zip(A_G, te) if s]).value_counts().sort_index()
fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
acq.plot.bar(ax=axs[0], color="#55A868"); axs[0].set_title("Evidence gaps retrieved (test)")
n_gaps.plot.bar(ax=axs[1], color="#C44E52"); axs[1].set_title("Gaps retrieved per case"); axs[1].set_xlabel("# gaps")
fig.tight_layout(); fig.savefig(f"{FIG}/fig_gap_usage.png", dpi=200)

contrib = pd.DataFrame(np.abs(CF[te]), columns=SRCF).mean().sort_values()
fig, ax = plt.subplots(figsize=(6, 3.5)); contrib.plot.barh(ax=ax, color="#8172B2")
ax.set_xlabel("Mean |weighted log-odds contribution| (test)"); fig.tight_layout(); fig.savefig(f"{FIG}/fig_source_contributions.png", dpi=200)

# investigation trajectories (AEGIS adaptive) for NB5 explanations
traj = []
for i in np.where(te)[0]:
    z = Z0F[i]; steps = [dict(stage="routine", p=float(1 / (1 + np.exp(-z))))]
    for g in A_G[i]:
        z += CF[i, DIDX[g]]; steps.append(dict(stage=g, p=float(1 / (1 + np.exp(-z)))))
    traj.append(dict(case_id=int(CIDS[i]), code=code[i], mi=int(CASES.mi.values[i]), y=int(y[i]),
                     p_final=float(P_G[i]), acquired=A_G[i], steps=steps,
                     contrib={s: float(CF[i, j]) for j, s in enumerate(SRCF)}))
TAU = TAUF
pd.to_pickle(dict(traj=traj, tau=TAU, eps=eps_g, policy=pol_g, weights=dict(zip(SRCF, FZF.w.tolist())), b0=FZF.b0),
             f"{OUT}/aegis_test_trajectories.pkl")
PRED_DF = CASES[["case_id", "code", "mi", "split", "y"]].copy()
for k, v in PRED.items(): PRED_DF[k] = v
PRED_DF.to_pickle(f"{OUT}/predictions.pkl")
json.dump(dict(adaptive_policy=pol_g, adaptive_eps=eps_g, gap_policy=POL_STAR, eps_star=EPS_STAR, tau=TAU, alert_rate=ALERT_RATE, test_events=len(test_events),
               weights=dict(zip(SRCF, FZF.w.round(4).tolist()))), open(f"{OUT}/aegis_config.json", "w"), indent=1)
print("saved results, figures ->", OUT)
