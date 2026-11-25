# %% [markdown]
# # AEGIS-MD revision extras
# Reads the saved predictions of the main-panel revision run and the random-panel run, and prints
# calibration bins, paired comparisons against the best count detector, alert workload at realistic
# prevalence, validation-period checks of the initial-reports analysis, and per-event ranks.

# %%
import glob, os, json
import numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from scipy import optimize
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
P = {}
for f in glob.glob("/kaggle/input/**/predictions_v2.pkl", recursive=True):
    df = pd.read_pickle(f); ncode = df.code.nunique()
    P["random" if ncode > 90 else "main"] = (df, os.path.dirname(f))
    print(f, len(df), "rows", ncode, "codes")
OUT = "/kaggle/working"; os.makedirs(OUT, exist_ok=True)
FULL, CNT = "Fusion + full reading", "Fusion, count evidence only"
INIT, GEN = "Fusion + full reading, initial reports only", "Fusion + generic retrieval (k=20)"
RH, CUSUM = "Recall history (logistic regression)", "Poisson CUSUM"
GBMC, GBMN = "GBM, count features", "GBM, count + full-reading narrative features"
GBMH = "GBM, count + recall history + full-reading narrative features"
def lg(p): p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def calib(y, p):
    lp = lg(p); y = np.asarray(y, float)
    def nll(v, s=None):
        z = v[0] + (v[1] if s is None else s) * lp; return -np.sum(y * z - np.logaddexp(0, z))
    a = optimize.minimize(lambda v: nll([v[0]], 1.0), [0.0], method="BFGS").x[0]
    b = optimize.minimize(nll, [0.0, 1.0], method="BFGS").x[1]
    return a, b
def paired(df, a, b, n=2000, seed=1):
    codes = df.code.unique(); g = {c: np.where(df.code.values == c)[0] for c in codes}
    y = df.y.values; pa, pb = df[a].values, df[b].values
    base = (AP(y, pa) - AP(y, pb), AUC(y, pa) - AUC(y, pb)); r = np.random.default_rng(seed); out = []
    for _ in range(n):
        idx = np.concatenate([g[c] for c in r.choice(codes, len(codes))])
        if y[idx].sum() == 0 or y[idx].sum() == len(idx): continue
        out.append((AP(y[idx], pa[idx]) - AP(y[idx], pb[idx]), AUC(y[idx], pa[idx]) - AUC(y[idx], pb[idx])))
    o = np.array(out)
    return dict(a=a, b=b, dAUPRC=base[0], lo=np.percentile(o[:, 0], 2.5), hi=np.percentile(o[:, 0], 97.5), p1=float((o[:, 0] <= 0).mean()),
                dAUROC=base[1], lo2=np.percentile(o[:, 1], 2.5), hi2=np.percentile(o[:, 1], 97.5))
def burden(df_te, p_val, name):
    tau = np.quantile(p_val, 0.9); al = df_te[name].values >= tau; y = df_te.y.values; cy = len(df_te) / 12
    pk = df_te.groupby("mi").apply(lambda g: g.nlargest(10, name).y.sum() / 10).mean()
    return dict(method=name, alerts_per_code_year=al.sum() / cy, false_alerts_per_code_year=(al & (y == 0)).sum() / cy,
                PPV=y[al].mean() if al.any() else np.nan, prec_at_10=pk, codes_per_month=df_te.groupby("mi").size().mean())

# %%
for panel, (df, d) in P.items():
    print("\n==================", panel, "panel ==================")
    te, va = df[df.split == "test"].reset_index(drop=True), df[df.split == "val"].reset_index(drop=True)
    print("test cases", len(te), "positives", int(te.y.sum()), "prev", round(te.y.mean(), 4), "| val prev", round(va.y.mean(), 4),
          "| train prev", round(df[df.split == "train"].y.mean(), 4))
    meth = [m for m in [CNT, FULL, INIT, GEN, RH, CUSUM, GBMC, GBMN, GBMH, "PRR (max lower 95% bound)"] if m in df]
    # calibration
    rows = []
    for m in [CNT, FULL, INIT, GEN, RH, GBMN, GBMH]:
        if m not in df: continue
        a, b = calib(te.y, te[m]); av, bv = calib(va.y, va[m])
        rows.append(dict(method=m, mean_pred_test=te[m].mean(), obs_test=te.y.mean(), cal_int_test=a, cal_slope_test=b,
                         mean_pred_val=va[m].mean(), obs_val=va.y.mean(), cal_int_val=av, cal_slope_val=bv))
    C = pd.DataFrame(rows); print("CALIBRATION\n" + C.round(3).to_string()); C.to_csv(f"{OUT}/calib_{panel}.csv", index=False)
    bins = []
    for m in [FULL, CNT, RH, GBMN]:
        if m not in df: continue
        q = pd.qcut(te[m].rank(method="first"), 10, labels=False)
        for k, g in te.groupby(q): bins.append(dict(method=m, decile=int(k), n=len(g), mean_pred=g[m].mean(), obs=g.y.mean()))
    B = pd.DataFrame(bins); print("CALBINS\n" + B.round(4).to_string()); B.to_csv(f"{OUT}/calbins_{panel}.csv", index=False)
    # validation check of initial-only and other key models
    v = [dict(method=m, val_AUPRC=AP(va.y, va[m]), val_AUROC=AUC(va.y, va[m]), test_AUPRC=AP(te.y, te[m]), test_AUROC=AUC(te.y, te[m])) for m in meth]
    print("VALTEST\n" + pd.DataFrame(v).round(3).to_string())
    # paired comparisons
    comps = [(FULL, CUSUM), (FULL, CNT), (GBMN, GBMC), (GBMN, CUSUM), (FULL, "PRR (max lower 95% bound)"), (INIT, FULL), (GBMH, RH), (FULL, RH)]
    S = pd.DataFrame([paired(te, a, b) for a, b in comps if a in te and b in te])
    print("PAIRED\n" + S.round(4).to_string()); S.to_csv(f"{OUT}/paired_extra_{panel}.csv", index=False)
    vS = pd.DataFrame([paired(va, INIT, FULL)]); print("PAIRED_VAL initial-only vs full\n" + vS.round(4).to_string())
    # workload
    W = pd.DataFrame([burden(te, va[m].values, m) for m in [CNT, FULL, INIT, GEN, RH, CUSUM, GBMN, GBMH] if m in te])
    print("BURDEN\n" + W.round(3).to_string()); W.to_csv(f"{OUT}/burden_{panel}.csv", index=False)
    # per-event ranks
    pef = glob.glob(f"{d}/per_event_results.csv")
    if pef:
        pe = pd.read_csv(pef[0]); rk = {m: te[m].rank(pct=True).values for m in [CNT, FULL]}
        def mi_of(s): y_, m_ = map(int, s.split("-")); return y_ * 12 + m_ - 1
        mis = te.mi.values; base = int(np.median(mis)); rows = []; cover = np.zeros(len(te), bool); cnt = np.zeros(len(te), int)
        # detect month-index convention
        for _, r in pe.iterrows():
            e = mi_of(r.event_month)
            if abs(e - base) > 400: e = e - (2016 * 12)  # fall back if months are indexed from 2016-01
            w = ((te.code.values == r.code) & (mis >= e - 6) & (mis < e))
            if w.sum() == 0: continue
            cover |= w; cnt += w
            rows.append(dict(code=r.code, event_month=r.event_month, n=int(w.sum()), count_rank=rk[CNT][w].max(), full_rank=rk[FULL][w].max()))
        R = pd.DataFrame(rows); R.to_csv(f"{OUT}/per_event_ranks_{panel}.csv", index=False)
        print("PEREVENT n_events", len(R), "below_diag", int((R.full_rank < R.count_rank).sum()), "above", int((R.full_rank > R.count_rank).sum()),
              "| sum positive months", int(R.n.sum()), "distinct positive case-months", int((cover & (te.y.values == 1)).sum()),
              "| case-months in >1 window", int((cnt > 1).sum()), "| positives", int(te.y.sum()))
        print("PERROWS"); print(R.round(4).to_csv(index=False))
print("ALL DONE")
