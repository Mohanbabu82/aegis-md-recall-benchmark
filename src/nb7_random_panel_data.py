# %% [markdown]
# # AEGIS-MD · Notebook 7 — Random product-code panel (CPU, Internet ON)
# Sensitivity analysis for the outcome-selected main panel. Product codes are sampled **at random from the FDA
# device classification database**, without looking at recalls; eligibility uses only report volume in
# 2014–2015, before the evaluation period (≥ 24 reports, i.e. about one a month).
#
# Input: attach the **full pipeline** output (for `recalls.pkl`). Output: `records.pkl`, `counts.pkl`
# (complete monthly counts from the bulk files), `codes.pkl`, `recalls.pkl`, `all_code_counts.pkl`.
# Then run Notebook 8 (enrichment + evaluation) with this notebook's output attached.

# %%
# %%writefile-core

# %%
import os, sys, json, time, io, zipfile, hashlib, heapq, re as _re
import numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.getcwd())
from aegis_core import *
OUT = out_dir()
N_PANEL = 150            # random product codes kept
N_CAND = 900             # random candidates tracked during the bulk pass
PER_MONTH = 30           # reports sampled per code-month (bottom-k hashing)
MIN_PRE = 24             # eligibility: >= 24 reports received 2014-01..2015-12 (before the panel period)
SEED = 2026
BASE = "https://api.fda.gov"

REC = pd.read_pickle(find_file("recalls.pkl"))
print("recall rows:", len(REC))

# %% [markdown]
# ## 1. All product codes in the device classification database → random candidates

# %%
def get_json(endpoint, params):
    for a in range(5):
        r = requests.get(BASE + endpoint, params=params, timeout=60); time.sleep(0.4)
        if r.status_code == 404: return {"results": []}
        if r.status_code == 429: time.sleep(20 * (a + 1)); continue
        r.raise_for_status(); return r.json()
    raise RuntimeError(endpoint)
allc = []
for dc in ["1", "2", "3", "U", "N", "f"]:
    skip = 0
    while True:
        js = get_json("/device/classification.json", {"search": f'device_class:"{dc}"', "limit": 1000, "skip": skip})
        res = js.get("results", [])
        allc += [(str(r.get("product_code", "")).upper(), r.get("device_name", ""), r.get("device_class", ""),
                  r.get("medical_specialty_description", "")) for r in res]
        tot = js.get("meta", {}).get("results", {}).get("total", 0); skip += 1000
        if not res or skip >= tot or skip > 25000: break
CLS = pd.DataFrame(allc, columns=["code", "device_name", "device_class", "specialty"]).drop_duplicates("code")
CLS = CLS[CLS.code.str.len() == 3].reset_index(drop=True)
print("product codes in classification database:", len(CLS), CLS.device_class.value_counts().to_dict())
cand = list(np.random.default_rng(SEED).permutation(CLS.code.values)[:N_CAND])
cand_set = set(cand)

# %% [markdown]
# ## 2. One pass over the bulk MAUDE files: complete counts for every code, bottom-k samples for candidates

# %%
meta = requests.get(BASE + "/download.json", timeout=120).json()
parts_all = meta["results"]["device"]["event"]["partitions"]
def part_q(p):
    m = _re.match(r"(\d{4}) Q(\d)", p.get("display_name", ""))
    return int(m.group(1)) * 12 + (int(m.group(2)) - 1) * 3 if m else None
lo_, hi_ = ym(CFG["DATA_START"]), ym(CFG["DATA_END"])
sel = [p for p in parts_all if part_q(p) is not None and lo_ - 2 <= part_q(p) <= hi_]
print("bulk partitions:", len(sel), "| MB:", round(sum(float(p["size_mb"]) for p in sel)))
def fetch(url):
    for a in range(6):
        try:
            r = requests.get(url, timeout=600); r.raise_for_status(); return r.content
        except Exception as ex_:
            print("  retry", a + 1, str(ex_)[:100]); time.sleep(15 * (a + 1))
    raise RuntimeError(url)
def hkey(k): return int(hashlib.md5(k.encode()).hexdigest()[:12], 16)

def _j(x):
    if isinstance(x, list): return ";".join(str(v) for v in x if v is not None)
    return "" if x is None else str(x)
def parse_event(e, code):
    dev = (e.get("device") or [{}])
    d = next((x for x in dev if str(x.get("device_report_product_code", "")).upper() == code), dev[0] if dev else {})
    txt_e, txt_m = [], []
    for t in e.get("mdr_text") or []:
        tt = str(t.get("text_type_code", "")).lower()
        (txt_m if "manufacturer" in tt else txt_e).append(str(t.get("text") or ""))
    te_, n1 = scrub_recall_mentions(" ".join(txt_e)[:3000]); tm_, n2 = scrub_recall_mentions(" ".join(txt_m)[:3000])
    return dict(mdr_key=str(e.get("mdr_report_key", "")), code=code, mi=mi_from_yyyymmdd(e.get("date_received")),
                event_type=str(e.get("event_type", "")), source=str(e.get("report_source_code", "")),
                type_of_report=_j(e.get("type_of_report") or []),
                mfr=str(d.get("manufacturer_d_name", "") or ""), brand=str(d.get("brand_name", "") or ""),
                model=str(d.get("model_number", "") or "").strip().upper(), lot=str(d.get("lot_number", "") or "").strip().upper(),
                evaluated=str(d.get("device_evaluated_by_manufacturer", "") or ""), problems=_j(e.get("product_problems") or []),
                text_event=te_[:1200], text_mfr=tm_[:1200], scrubbed=n1 + n2)

ALLCNT = Counter()
HEAP = {}                       # (code, mi) -> max-heap of (-h, mdr_key, record)
t_start = time.time(); n_bad = 0
with ThreadPoolExecutor(2) as pool:
    futs = {i: pool.submit(fetch, sel[i]["file"]) for i in range(min(2, len(sel)))}
    for i, p in enumerate(sel):
        blob = futs.pop(i).result()
        if i + 2 < len(sel): futs[i + 2] = pool.submit(fetch, sel[i + 2]["file"])
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            js = json.loads(z.read(z.namelist()[0]))
        del blob
        for e in js.get("results", []):
            mi = mi_from_yyyymmdd(e.get("date_received"))
            if mi != mi: continue
            mi = int(mi)
            pcs = {str(d.get("device_report_product_code", "")).upper() for d in (e.get("device") or [])}
            for c in pcs:
                ALLCNT[(c, mi)] += 1
                if c not in cand_set: continue
                h = hkey(str(e.get("mdr_report_key", "")) + c); k = (c, mi); hp = HEAP.setdefault(k, [])
                if len(hp) < PER_MONTH or -hp[0][0] > h:
                    try:
                        rr = parse_event(e, c)
                    except Exception:
                        n_bad += 1; continue
                    item = (-h, rr["mdr_key"], rr)
                    if len(hp) < PER_MONTH: heapq.heappush(hp, item)
                    else: heapq.heapreplace(hp, item)
        del js
        print(f"{p['display_name']}: {i + 1}/{len(sel)} files | {(time.time() - t_start) / 60:.1f} min | sampled cells {len(HEAP)}", flush=True)
print("malformed records skipped:", n_bad)

# %% [markdown]
# ## 3. Eligibility (pre-period volume only) and the random panel

# %%
ac = pd.DataFrame([(c, m, n) for (c, m), n in ALLCNT.items()], columns=["code", "mi", "n"])
ac.to_pickle(f"{OUT}/all_code_counts.pkl")
pre = ac[ac.mi.between(ym("2014-01"), ym("2015-12"))].groupby("code").n.sum()
eligible = [c for c in cand if pre.get(c, 0) >= MIN_PRE]
panel = eligible[:N_PANEL]
print(f"candidates {len(cand)} | eligible {len(eligible)} | panel {len(panel)}")
codes = CLS[CLS.code.isin(panel)].copy(); codes["group"] = "random"
lab = REC[REC.klass.isin(CFG["LABEL_CLASSES"])]
codes["has_class_I_2016_2025"] = codes.code.isin(set(lab[lab.init_mi.between(ym("2016-01"), ym("2025-12"))].code))
print("random-panel codes with a Class I recall initiated 2016-2025:", int(codes.has_class_I_2016_2025.sum()), "of", len(codes))
codes.to_pickle(f"{OUT}/codes.pkl")
counts = ac[ac.code.isin(panel)].reset_index(drop=True); counts.to_pickle(f"{OUT}/counts.pkl")
rows = [rr for (c, m), hp in HEAP.items() if c in set(panel) for _, _, rr in hp]
R = pd.DataFrame(rows)
R = R[R.mi.notna()].drop_duplicates(["mdr_key", "code"]).sort_values(["code", "mi"]).reset_index(drop=True)
R["mi"] = R.mi.astype(int); R["rid"] = np.arange(len(R))
R.to_pickle(f"{OUT}/records.pkl"); REC.to_pickle(f"{OUT}/recalls.pkl")
json.dump(dict(n_classification_codes=len(CLS), n_candidates=len(cand), n_eligible=len(eligible), n_panel=len(panel),
               n_records=len(R), per_month=PER_MONTH, min_pre_reports=MIN_PRE, seed=SEED,
               panel_codes_with_class_I=int(codes.has_class_I_2016_2025.sum()), malformed=n_bad,
               bulk_files=len(sel), snapshot=meta.get("meta", {}).get("last_updated", "")),
          open(f"{OUT}/data_summary.json", "w"), indent=1)
print("DONE: records", len(R))
