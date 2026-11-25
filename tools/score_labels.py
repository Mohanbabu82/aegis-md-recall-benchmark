"""Score the manual labelling sheets for the AEGIS-MD revision.

1. labelling_stance_200.csv  — fill column `label` with one of: confirmed / not_confirmed / unclear
   (leave the KEY file closed until you finish labelling).
2. labelling_leakage_300.csv — fill column `mentions_recall_or_field_action` with Y or N.

Usage:  python score_labels.py labelling_stance_200.csv labelling_stance_200_KEY_do_not_open.csv labelling_leakage_300.csv
Prints Cohen's kappa / F1 for the NLI stance score and the residual leakage rate with a 95% CI.
"""
import sys
import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score

def wilson(k, n, z=1.96):
    if n == 0: return (np.nan, np.nan)
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)

stance, key, leak = sys.argv[1:4]
S = pd.read_csv(stance).merge(pd.read_csv(key), on="item_id")
S["label"] = S.label.astype(str).str.strip().str.lower().str.replace(" ", "_")
S = S[S.label.isin(["confirmed", "not_confirmed", "unclear"])]
print(f"stance items labelled: {len(S)}  ({S.label.value_counts().to_dict()})")
clear = S[S.label != "unclear"].copy()
clear["pred"] = np.where(clear.p_confirm >= clear.p_refute, "confirmed", "not_confirmed")
print("Cohen's kappa (model vs human, clear items):", round(cohen_kappa_score(clear.label, clear.pred), 3))
print("macro F1:", round(f1_score(clear.label, clear.pred, average="macro"), 3))
print("AUROC of p_confirm - p_refute for 'confirmed':",
      round(roc_auc_score((clear.label == "confirmed").astype(int), clear.p_confirm - clear.p_refute), 3))

L = pd.read_csv(leak)
v = L.mentions_recall_or_field_action.astype(str).str.strip().str.upper().str[:1]
L = L[v.isin(["Y", "N"])]; k = int((v[v.isin(["Y", "N"])] == "Y").sum()); n = len(L)
lo, hi = wilson(k, n)
print(f"leakage check: {k}/{n} narratives still mention a recall or field action ({k / max(n, 1):.1%}; 95% CI {lo:.1%}-{hi:.1%})")
