# %% [markdown]
# # AEGIS-MD · Notebook 5 — Evidence-grounded LLM explanations + faithfulness evaluation (GPU T4, Internet ON)
# Inputs: attach outputs of **Notebook 2**, **Notebook 3** and **Notebook 4**.
#
# For the top test-period alerts, an **evidence dossier** is assembled from the AEGIS investigation trajectory (routine signal → gaps retrieved → evidence for / against → posterior). An open LLM (`Qwen2.5-3B-Instruct`, fp16 on a T4) writes an investigation summary **that may only cite dossier items**. We then measure:
# * **citation validity** (every cited ID exists in the dossier),
# * **faithfulness** (NLI entailment of each cited sentence by its cited source),
# * **root-cause anticipation**: similarity between the explanation and the text of the recall that was actually initiated later (true alerts only), comparing AEGIS dossiers with generic-RAG dossiers.

# %%
# %%writefile-core

# %%
import os, sys, json, re, time
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from aegis_core import *
OUT = out_dir()
N_CASES = 40
LLM_NAME = "Qwen/Qwen2.5-3B-Instruct"      # or "Qwen/Qwen2.5-7B-Instruct" with LOAD_4BIT=True
LOAD_4BIT = False

T = pd.read_pickle(find_file("aegis_test_trajectories.pkl"))
CASES = pd.read_pickle(find_file("cases.pkl")).set_index("case_id")
R = pd.read_pickle(find_file("records.pkl")); ANN = pd.read_pickle(find_file("record_ann.pkl"))
REC = pd.read_pickle(find_file("recalls.pkl")); RS = pd.read_pickle(find_file("retrieval_sets.pkl"))
CODES = pd.read_pickle(find_file("codes.pkl")).set_index("code")
tau = T["tau"]
traj = sorted([t for t in T["traj"] if t["p_final"] >= tau], key=lambda d: -d["p_final"])
# keep at most 3 months per product code so the case set is diverse
seen, SEL = Counter(), []
for t in traj:
    if seen[t["code"]] < 3: SEL.append(t); seen[t["code"]] += 1
    if len(SEL) >= N_CASES: break
print("alerts at tau:", len(traj), "| selected:", len(SEL), "| true alerts among selected:", sum(t["y"] for t in SEL))

try:
    import torch; GPU = torch.cuda.is_available()
except Exception:
    torch = None; GPU = False
print("GPU:", GPU)

# %% [markdown]
# ## 1. Evidence dossiers

# %%
def snip(s, n=320):
    s = re.sub(r"\s+", " ", str(s or "")).strip(); return s[:n] + ("…" if len(s) > n else "")

def dossier(t, mode="aegis"):
    cid = t["case_id"]; c = CASES.loc[cid]; code = t["code"]
    items, lines = {}, []
    lines.append(f"Device product code {code} ({CODES.device_name.get(code, '')}); assessment month {ym_str(t['mi'])}.")
    lines.append(f"Routine surveillance: {int(c.n_recent)} reports in last 3 months vs baseline {c.base_mu * 3:.1f} expected "
                 f"(Poisson score {c.pois_score:.1f}); Mann-Kendall z={c.mk_z:.2f}; serious-event share {c.sev_share:.2f} "
                 f"(shift z={c.sev_z:.2f}); problem-code shift JSD={c.prob_jsd:.3f}.")
    if mode == "aegis":
        lines.append("Investigation trajectory: " + " -> ".join(f"{s['stage']} (risk {s['p']:.2f})" for s in t["steps"]))
        groups = {g: RS["S_GAP"][cid][g][:3] for g in t["acquired"]}
        if not groups: groups = {"mechanism": RS["S_GAP"][cid]["mechanism"][:3]}
    else:
        groups = {"retrieved": np.asarray(RS["S_GEN"].get(cid, []))[:6]}
    for g, S in groups.items():
        if g in GAP_SPECS: lines.append(f"Evidence gap '{g}': {GAP_SPECS[g]['question']}")
        for rid in S:
            r = R.iloc[int(rid)]; a = ANN.iloc[int(rid)]
            key = f"MDR:{r.mdr_key}"
            stance = ""
            if a.has_mfr and np.isfinite(a.p_confirm):
                stance = f" [manufacturer finding: confirmed={a.p_confirm:.2f}, not-confirmed={a.p_refute:.2f}]"
            txt = f"{snip(r.text_event, 260)} | Manufacturer: {snip(r.text_mfr, 220)} | lot {r.lot or 'n/a'}, model {r.model or 'n/a'}{stance}"
            items[key] = txt; lines.append(f"[{key}] {txt}")
            if g == "precedent" and a.prec_rec_id >= 0:
                pr = REC[REC.rec_id == a.prec_rec_id]
                if len(pr) and pr.posted_mi.iloc[0] < t["mi"]:
                    k2 = f"RECALL:{pr.recall_number.iloc[0] or pr.rec_id.iloc[0]}"
                    items[k2] = snip(pr.reason.iloc[0], 300)
                    lines.append(f"[{k2}] earlier recall (similarity {a.prec_sim:.2f}): {items[k2]}")
    if mode == "aegis":
        keep = set(TIER0_SOURCES) | set(t["acquired"])
        con = {k: v for k, v in t["contrib"].items() if abs(v) > 0.05 and k in keep}
        lines.append("Log-odds contributions (positive supports a safety signal, negative contradicts it): " +
                     ", ".join(f"{k} {v:+.2f}" for k, v in sorted(con.items(), key=lambda kv: -abs(kv[1]))))
        lines.append(f"Final AEGIS risk: {t['p_final']:.2f} (alert threshold {tau:.2f}).")
    return "\n".join(lines), items

PROMPT = """You are assisting a medical-device post-market safety reviewer. Using ONLY the evidence dossier below, write a concise investigation summary with these headings:
1. Signal  2. Supporting evidence  3. Contradicting or weakening evidence  4. Remaining evidence gaps  5. Suggested next step for the human reviewer.
Rules: cite every factual statement with the bracketed ID(s) from the dossier, e.g. [MDR:1234] or [RECALL:Z-0001-2020]. Do not invent facts, numbers or IDs. Do not claim a confirmed causal defect; this is a screening signal for human review. Max 220 words.

EVIDENCE DOSSIER:
{dossier}
"""

# %% [markdown]
# ## 2. Generate explanations

# %%
gen = None
if GPU:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(LLM_NAME)
        kw = dict(device_map="cuda", dtype=torch.float16)
        if LOAD_4BIT:
            from transformers import BitsAndBytesConfig
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        llm = AutoModelForCausalLM.from_pretrained(LLM_NAME, **kw).eval()
        def gen(text):
            msgs = [{"role": "user", "content": PROMPT.format(dossier=text)}]
            prompt = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            enc = tok(prompt, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = llm.generate(**enc, max_new_tokens=380, do_sample=False, pad_token_id=tok.eos_token_id)
            return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception as ex:
        print("LLM unavailable -> template explanations:", ex); gen = None

def template_expl(text, items):
    ids = list(items)[:4]
    return ("1. Signal: " + text.split("\n")[1] + "\n2. Supporting evidence: " + "; ".join(f"{items[i][:120]} [{i}]" for i in ids[:2]) +
            "\n3. Contradicting evidence: see manufacturer findings " + " ".join(f"[{i}]" for i in ids[2:4]) +
            "\n4. Remaining gaps: not all reports were read.\n5. Next step: human review of cited reports.")

EXPL = []
t0 = time.time()
for t in SEL:
    for mode in ["aegis", "generic"]:
        d, items = dossier(t, mode)
        e = gen(d) if gen else template_expl(d, items)
        EXPL.append(dict(case_id=t["case_id"], code=t["code"], mi=t["mi"], y=t["y"], p=t["p_final"], mode=mode,
                         dossier=d, items=items, explanation=e))
    print(len(EXPL), f"{time.time() - t0:.0f}s")

# %% [markdown]
# ## 3. Faithfulness & root-cause anticipation

# %%
CITE = re.compile(r"\[((?:MDR|RECALL):[^\]\s,;]+)\]")
nli = None
if GPU:
    try:
        from sentence_transformers import CrossEncoder
        nli = CrossEncoder("cross-encoder/nli-deberta-v3-base", device="cuda", max_length=384)
        ENT = {v.lower(): int(k) for k, v in nli.model.config.id2label.items()}["entailment"]
    except Exception as ex:
        print("NLI unavailable:", ex)
emb_model = None
if GPU:
    try:
        from sentence_transformers import SentenceTransformer
        emb_model = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    except Exception as ex:
        print("embedder unavailable:", ex)

lab = REC[REC.klass.isin(CFG["LABEL_CLASSES"])]
rows = []
for x in EXPL:
    sents = [s for s in re.split(r"(?<=[\.\!\?])\s+|\n", x["explanation"]) if len(s.strip()) > 15]
    cited = [(s, CITE.findall(s)) for s in sents]
    all_ids = [i for _, ids in cited for i in ids]
    valid = [i in x["items"] for i in all_ids]
    sup = []
    if nli is not None:
        pairs = [(x["items"][i], CITE.sub("", s)) for s, ids in cited for i in ids if i in x["items"]]
        if pairs:
            lg = nli.predict(pairs, apply_softmax=True); sup = list(lg[:, ENT] > 0.5)
    rc = np.nan
    if x["y"] == 1 and emb_model is not None:
        fut = lab[(lab.code == x["code"]) & (lab.init_mi > x["mi"]) & (lab.init_mi <= x["mi"] + CFG["HORIZON"])]
        if len(fut):
            v = emb_model.encode([x["explanation"]] + list((fut.reason + " " + fut.root_cause.fillna("")).values),
                                 normalize_embeddings=True)
            rc = float((v[1:] @ v[0]).max())
    rows.append(dict(case_id=x["case_id"], mode=x["mode"], y=x["y"], n_sent=len(sents),
                     cited_sent_frac=np.mean([bool(ids) for _, ids in cited]) if cited else 0,
                     n_citations=len(all_ids), citation_validity=np.mean(valid) if valid else np.nan,
                     nli_supported=np.mean(sup) if len(sup) else np.nan, rootcause_sim=rc))
FAITH = pd.DataFrame(rows)
print(FAITH.groupby("mode")[["cited_sent_frac", "citation_validity", "nli_supported", "rootcause_sim"]].mean().round(3))
if FAITH.rootcause_sim.notna().sum() > 10:
    from scipy.stats import wilcoxon
    pv = FAITH.pivot_table(index="case_id", columns="mode", values="rootcause_sim").dropna()
    if len(pv) > 5: print("root-cause anticipation, AEGIS vs generic dossier:", wilcoxon(pv.aegis, pv.generic))
FAITH.to_csv(f"{OUT}/explanation_faithfulness.csv", index=False)

with open(f"{OUT}/case_studies.md", "w") as f:
    for x in EXPL:
        if x["mode"] != "aegis": continue
        f.write(f"## {x['code']} · {ym_str(x['mi'])} · risk {x['p']:.2f} · {'TRUE recall followed' if x['y'] else 'no recall within 6 months'}\n\n")
        f.write("### Dossier\n```\n" + x["dossier"] + "\n```\n### Explanation\n" + x["explanation"] + "\n\n")
pd.to_pickle(EXPL, f"{OUT}/explanations.pkl")
json.dump(dict(llm=LLM_NAME if gen else "template", nli=bool(nli), n_cases=len(SEL)), open(f"{OUT}/explain_meta.json", "w"))
print("saved case_studies.md, explanation_faithfulness.csv")
