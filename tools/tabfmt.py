"""Single source for every test-period number in Table 3 and Supplementary Table S3."""
import math, pandas as pd
R = pd.read_csv("results_test_v2.csv").set_index("method")
RR = pd.read_csv("random_reading_repeats.csv").set_index("k")
def n0(x): return str(int(math.floor(float(x) + 0.5)))
def f3(x): return "–" if pd.isna(x) else f"{x:.3f}"
def f2(x): return "–" if pd.isna(x) else f"{x:.2f}"
def ci(m, lo, hi): return f"{m:.3f} ({lo:.3f}–{hi:.3f})"
def reads(m):
    if m.startswith("Fusion + random reading"): return n0(RR.loc[int(m.split("k=")[1].rstrip(")")), "reports_read"])
    return n0(R.loc[m, "reports_read"])
def auprc(m):
    if m.startswith("Fusion + random reading"):
        k = int(m.split("k=")[1].rstrip(")")); return f"{RR.loc[k,'AUPRC_mean']:.3f} ± {RR.loc[k,'AUPRC_sd']:.3f} (SD)"
    r = R.loc[m]; return ci(r.AUPRC, r.AUPRC_lo, r.AUPRC_hi)
def auroc(m, with_ci=True):
    if m.startswith("Fusion + random reading"):
        k = int(m.split("k=")[1].rstrip(")")); return f"{RR.loc[k,'AUROC_mean']:.3f} ± {RR.loc[k,'AUROC_sd']:.3f} (SD)"
    r = R.loc[m]; return ci(r.AUROC, r.AUROC_lo, r.AUROC_hi) if with_ci else f"{r.AUROC:.3f}"
def rnd(m): return m.startswith("Fusion + random reading")
def table3_row(m, label):
    r = R.loc[m]
    if rnd(m): return [label, reads(m), auprc(m), auroc(m), "–", "–", "–", "–"]
    return [label, reads(m), auprc(m), auroc(m), f2(r.calib_intercept), f2(r.calib_slope), f2(r.prec_at_10), f"{r.events_detected*100:.0f}%"]
def s3_rows():
    out = []
    for m in R.index:
        r = R.loc[m]
        if rnd(m): out.append([m + ", mean of 10 draws", reads(m), auprc(m), f"{RR.loc[int(m.split('k=')[1].rstrip(')')),'AUROC_mean']:.3f}", "–", "–", "–", "–", "–"]); continue
        out.append([m, reads(m), auprc(m), f"{r.AUROC:.3f}", f3(r.Brier), f2(r.calib_intercept), f2(r.calib_slope), f2(r.PPV), f2(r.prec_at_10)])
    return out
def val(m, col): return R.loc[m, col]
