# %% [markdown]
# # AEGIS-MD · Notebook 2 — Record enrichment (GPU T4, Internet ON)
# Inputs: attach the output of **Notebook 1** (`records.pkl`, `recalls.pkl`).
#
# For every MAUDE record this notebook computes, once:
# * a dense **embedding** of the narrative (`BAAI/bge-small-en-v1.5`), used for hybrid retrieval and failure-mechanism evidence;
# * **NLI stance** of the manufacturer's investigation narrative (`cross-encoder/nli-deberta-v3-base`): P(problem *confirmed*) and P(problem *refuted / not confirmed*) → supporting vs **contradictory** evidence;
# * **precedent similarity**: maximum similarity to any historical recall reason that was *publicly posted before the report was received* (time-aware, leakage-free).
#
# Accelerator: **GPU T4 ×1** (≈10–25 min). Without a GPU it falls back to TF-IDF/LSA embeddings and regex stance (much weaker; use only for debugging).

# %%
# %%writefile-core

# %%
import os, sys, json, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from aegis_core import *
OUT = out_dir()

R = pd.read_pickle(find_file("records.pkl"))
REC = pd.read_pickle(find_file("recalls.pkl"))
print("records", len(R), "recalls", len(REC))

FORCE_CPU = os.environ.get("AEGIS_CPU") == "1"
try:
    import torch
    DEVICE = "cuda" if (torch.cuda.is_available() and not FORCE_CPU) else "cpu"
except Exception:
    torch = None; DEVICE = "cpu"
print("device:", DEVICE)
EMB_MODEL = "BAAI/bge-small-en-v1.5"
NLI_MODEL = "cross-encoder/nli-deberta-v3-base"

doc_text = (R.text_event.fillna("") + " " + R.text_mfr.fillna("") + " Problems: " + R.problems.fillna("").str.replace(";", ", ")).str.slice(0, 2000)
recall_text_all = (REC.reason.fillna("") + " " + REC.root_cause.fillna("")).str.strip()

# unique recall texts with earliest public availability month
RT = (pd.DataFrame({"text": recall_text_all, "posted_mi": REC.posted_mi, "rec_id": REC.rec_id})
      .query("text.str.len() > 15", engine="python")
      .sort_values("posted_mi").drop_duplicates("text").reset_index(drop=True))
print("unique recall texts:", len(RT))

proto_texts = {g: GAP_SPECS[g]["proto"] for g in GAP_SPECS}
proto_texts["generic"] = GENERIC_QUERY["proto"]

# %% [markdown]
# ## 1. Dense embeddings

# %%
t0 = time.time()
USE_ST = False
if DEVICE == "cuda":
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(EMB_MODEL, device=DEVICE)
        st.half()
        enc = lambda texts, bs=256: st.encode(list(texts), batch_size=bs, normalize_embeddings=True,
                                               show_progress_bar=True, convert_to_numpy=True)
        E = enc(doc_text.values)
        E_rec = enc(RT.text.values)
        qpre = "Represent this sentence for searching relevant passages: "
        P = {k: enc([qpre + v])[0] for k, v in proto_texts.items()}
        USE_ST = True; ENC_NAME = EMB_MODEL
    except Exception as ex:
        print("SentenceTransformer failed -> LSA fallback:", ex)
if not USE_ST:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    tf = TfidfVectorizer(tokenizer=tokenize, token_pattern=None, lowercase=False, min_df=2, max_features=50000, sublinear_tf=True)
    X = tf.fit_transform(doc_text.values)
    svd = TruncatedSVD(n_components=min(128, X.shape[1] - 1), random_state=0).fit(X)
    def enc(texts):
        Z = svd.transform(tf.transform(list(texts))); return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    E = enc(doc_text.values); E_rec = enc(RT.text.values)
    P = {k: enc([v])[0] for k, v in proto_texts.items()}
    ENC_NAME = "tfidf-lsa (fallback)"
E = E.astype(np.float16); E_rec = E_rec.astype(np.float16)
print("embeddings", E.shape, f"{time.time() - t0:.0f}s", ENC_NAME)

# %% [markdown]
# ## 2. NLI stance of manufacturer investigation narratives (support vs contradiction)

# %%
H_CONFIRM = "The manufacturer confirmed the reported device problem or identified a device-related root cause."
H_REFUTE = "The manufacturer could not confirm the reported problem, found no device malfunction, or the device was not returned."
has_mfr = (R.text_mfr.fillna("").str.len() > 30).values
p_conf = np.full(len(R), np.nan); p_ref = np.full(len(R), np.nan)
idx = np.where(has_mfr)[0]
NLI_NAME = "regex (fallback)"
if DEVICE == "cuda" and len(idx):
    try:
        from sentence_transformers import CrossEncoder
        ce = CrossEncoder(NLI_MODEL, device=DEVICE, max_length=256)
        ce.model.half()
        lab = {v.lower(): int(k) for k, v in ce.model.config.id2label.items()}
        ent = lab.get("entailment")
        prem = R.text_mfr.values[idx]
        for H, dest in ((H_CONFIRM, p_conf), (H_REFUTE, p_ref)):
            logits = ce.predict([(p, H) for p in prem], batch_size=128, show_progress_bar=True, apply_softmax=False)
            pr = np.exp(logits - logits.max(1, keepdims=True)); pr /= pr.sum(1, keepdims=True)
            dest[idx] = pr[:, ent]
        NLI_NAME = NLI_MODEL
    except Exception as ex:
        print("NLI failed -> regex fallback:", ex)
if NLI_NAME.startswith("regex"):
    for i in idx:
        p_conf[i], p_ref[i] = regex_stance(R.text_mfr.values[i])
print("stance done:", NLI_NAME, "| mean confirm", np.nanmean(p_conf).round(3), "| mean refute", np.nanmean(p_ref).round(3))

# %% [markdown]
# ## 3. Time-aware precedent similarity (record vs recalls posted before the record month)

# %%
rec_av = RT.posted_mi.values.astype(np.int64)
order = np.argsort(R.mi.values, kind="stable")
prec_sim = np.full(len(R), np.nan, dtype=np.float32); prec_arg = np.full(len(R), -1, dtype=np.int64)
CH = 4096
if torch is not None and DEVICE == "cuda":
    Er = torch.tensor(E_rec, device="cuda", dtype=torch.float16); av = torch.tensor(rec_av, device="cuda")
    for s in range(0, len(order), CH):
        ii = order[s:s + CH]
        q = torch.tensor(E[ii], device="cuda", dtype=torch.float16)
        S = (q @ Er.T).float()
        m = torch.tensor(R.mi.values[ii].astype(np.int64), device="cuda")
        S[av[None, :] >= m[:, None]] = -2.0
        v, a = S.max(1)
        prec_sim[ii] = v.cpu().numpy(); prec_arg[ii] = a.cpu().numpy()
else:
    Erf = E_rec.astype(np.float32)
    for s in range(0, len(order), CH):
        ii = order[s:s + CH]
        S = E[ii].astype(np.float32) @ Erf.T
        S[rec_av[None, :] >= R.mi.values[ii][:, None]] = -2.0
        prec_sim[ii] = S.max(1); prec_arg[ii] = S.argmax(1)
prec_sim[prec_sim <= -1.5] = np.nan
prec_rec = np.where(prec_arg >= 0, RT.rec_id.values[np.clip(prec_arg, 0, None)], -1)
prec_rec[np.isnan(prec_sim)] = -1
print("precedent sim: mean", np.nanmean(prec_sim).round(3))

# %%
ann = pd.DataFrame(dict(rid=R.rid.values, has_mfr=has_mfr, p_confirm=p_conf, p_refute=p_ref,
                        prec_sim=prec_sim, prec_rec_id=prec_rec))
ann.to_pickle(f"{OUT}/record_ann.pkl")
np.save(f"{OUT}/rec_emb.npy", E)
np.save(f"{OUT}/recall_emb.npy", E_rec)
RT[["rec_id", "posted_mi", "text"]].to_pickle(f"{OUT}/recall_texts.pkl")
pd.to_pickle({k: np.asarray(v, np.float32) for k, v in P.items()}, f"{OUT}/proto_emb.pkl")
json.dump(dict(encoder=ENC_NAME, nli=NLI_NAME, n_records=len(R), n_recall_texts=len(RT),
               h_confirm=H_CONFIRM, h_refute=H_REFUTE), open(f"{OUT}/enrich_meta.json", "w"), indent=1)
# carry forward NB1 data so later notebooks only need this notebook as input
for f in ["records.pkl", "recalls.pkl", "counts.pkl", "codes.pkl", "data_summary.json"]:
    src = find_file(f, required=False)
    if src and os.path.abspath(src) != os.path.abspath(f"{OUT}/{f}"):
        import shutil; shutil.copy(src, f"{OUT}/{f}")
print("saved to", OUT)
