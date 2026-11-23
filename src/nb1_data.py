# %% [markdown]
# # AEGIS-MD · Notebook 1 — Real data acquisition from openFDA (CPU, Internet ON)
# **Adaptive Evidence-Gap Investigation of Safety signals for Medical Devices**
#
# Downloads only **real public FDA data** through the openFDA API:
# 1. Device recalls (`/device/recall.json`) and enforcement reports (`/device/enforcement.json`), joined to get the **recall class** (Class I/II/III). These are the **ground-truth labels** and the corpus of historical **precedents**.
# 2. Selects a panel of device **product codes** (codes with and without Class I recalls).
# 3. Monthly MAUDE report **counts** per product code (`count=date_received`).
# 4. A monthly sample of MAUDE **records** per product code (narratives, manufacturer investigation text, lot/model numbers, problem codes, event type, report source).
#
# **Before running:** Settings → Internet **ON**. Strongly recommended: get a free openFDA API key (https://open.fda.gov/apis/authentication/) and add it as a Kaggle secret named `OPENFDA_API_KEY` (Add-ons → Secrets). Without a key openFDA allows only ~1,000 requests/day.
#
# The record download is **resumable**: it stops after `TIME_BUDGET_MIN` minutes and saves progress. Just run the notebook again (or "Save & Run All" again) until it prints `ALL CODES DONE`.

# %%
# %%writefile-core

# %%
import os, sys, json, time, math, random
import numpy as np, pandas as pd
sys.path.insert(0, os.getcwd())
from aegis_core import *

OUT = out_dir()
TIME_BUDGET_MIN = 38          # stop record download after this many minutes (resume on next run)
N_POS_CODES = 50              # product codes with >=1 Class I recall in the label window
N_NEG_CODES = 30              # product codes with no Class I recall in the whole period
RECORDS_PER_MONTH = 60        # MAUDE records sampled per code-month
MIN_TOTAL_REPORTS = 600       # code must have >= this many MAUDE reports in DATA_START..DATA_END
MAX_TOTAL_REPORTS = 400000
MOCK = os.environ.get("AEGIS_MOCK") == "1"   # local testing only

API_KEY = os.environ.get("OPENFDA_API_KEY", "")
try:
    from kaggle_secrets import UserSecretsClient
    API_KEY = UserSecretsClient().get_secret("OPENFDA_API_KEY") or API_KEY
except Exception as _ex:
    print("secret not available:", type(_ex).__name__, str(_ex)[:150])
print("openFDA API key:", "found" if API_KEY else "NOT found (limited to ~1000 requests/day)")
print(json.dumps(CFG, indent=1))

# %%
import requests
BASE = "https://api.fda.gov"
SLEEP = 0.26 if API_KEY else 0.35
_N_CALLS = 0

def get_json(endpoint, params, retries=5):
    """GET with retries. openFDA returns HTTP 404 when a search has no matches -> empty result."""
    global _N_CALLS
    if MOCK:
        import mock_openfda; _N_CALLS += 1
        return mock_openfda.respond(endpoint, dict(params))
    p = dict(params)
    if API_KEY: p["api_key"] = API_KEY
    for a in range(retries):
        try:
            r = requests.get(BASE + endpoint, params=p, timeout=60)
            _N_CALLS += 1
            time.sleep(SLEEP)
            if r.status_code == 404: return {"results": [], "meta": {"results": {"total": 0}}}
            if r.status_code == 403:
                raise PermissionError(f"openFDA 403 for {endpoint}: {r.text[:150]}")
            if r.status_code == 429:
                time.sleep(20 * (a + 1)); continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print("  retry", a + 1, endpoint, str(e)[:120]); time.sleep(3 * (a + 1))
    raise RuntimeError(f"openFDA request failed: {endpoint} {params}")

def paged(endpoint, search, limit=1000, max_skip=25000):
    out, skip = [], 0
    while True:
        js = get_json(endpoint, {"search": search, "limit": limit, "skip": skip})
        res = js.get("results", [])
        out += res
        total = js.get("meta", {}).get("results", {}).get("total", 0)
        skip += limit
        if not res or skip >= total: break
        if skip + limit > max_skip + 1000:
            print(f"  WARNING: {search} has {total} hits; capped at {max_skip + 1000}. Narrow the date window."); break
    return out

# %% [markdown]
# ## 1. Recalls + enforcement reports → recall class (ground truth)

# %%
_cached = find_file("recalls.pkl", required=False)
y0, y1 = ym(CFG["DATA_START"]) // 12 - 2, ym(CFG["DATA_END"]) // 12 + 1   # recalls a bit wider than MAUDE window
recall_rows, enf_rows = [], []
for y in (range(y0, y1 + 1) if not _cached else []):
    got = []
    for fmt in (f"[{y}0101 TO {y}1231]", f"[{y}-01-01 TO {y}-12-31]"):
        got = paged("/device/recall.json", f"event_date_initiated:{fmt}")
        if got: break
    recall_rows += got
    enf = paged("/device/enforcement.json", f"recall_initiation_date:[{y}0101 TO {y}1231]")
    enf_rows += enf
    print(y, "recalls:", len(got), " enforcement:", len(enf))

def first(x):
    if isinstance(x, list): return x[0] if x else None
    return x

if _cached:
    rec = pd.read_pickle(_cached); print("loaded cached recalls:", _cached)
else:
  rec = pd.DataFrame([dict(
      event_id=str(r.get("res_event_number") or ""),
      recall_number=str(r.get("product_res_number") or "").strip(),
      code=str(first(r.get("product_code")) or "").strip().upper(),
      firm=r.get("recalling_firm"),
      init_mi=mi_from_yyyymmdd(r.get("event_date_initiated")),
      posted_mi=mi_from_yyyymmdd(r.get("event_date_posted")),
      reason=(r.get("reason_for_recall") or "")[:1500],
      root_cause=r.get("root_cause_description") or "",
      device_name=first((r.get("openfda") or {}).get("device_name")) or (r.get("product_description") or "")[:200],
      klass_rec=str(r.get("classification") or r.get("recall_class") or ""),
  ) for r in recall_rows])
  enf = pd.DataFrame([dict(recall_number=str(e.get("recall_number") or "").strip(),
                           klass_enf=str(e.get("classification") or "")) for e in enf_rows]).drop_duplicates("recall_number")
  rec = rec.merge(enf, on="recall_number", how="left")
  rec["klass"] = np.where(rec.klass_rec.str.startswith("Class"), rec.klass_rec, rec.klass_enf.fillna(""))
  rec = rec[(rec.code != "") & rec.init_mi.notna()].copy()
  rec["posted_mi"] = rec.posted_mi.fillna(rec.init_mi + 2)      # availability time of the recall text
  rec = rec.drop_duplicates(["event_id", "recall_number", "code"]).reset_index(drop=True)
  rec["rec_id"] = np.arange(len(rec))
  print("recall rows:", len(rec), "| class coverage:", (rec.klass.str.startswith("Class")).mean().round(3))
  print(rec.klass.value_counts().head())
rec.to_pickle(f"{OUT}/recalls.pkl")

# %% [markdown]
# ## 2. Select product-code panel

# %%
d0 = f"{CFG['DATA_START'].replace('-', '')}01"; d1 = f"{CFG['DATA_END'].replace('-', '')}31"
_cc = find_file("codes.pkl", required=False)
COUNT_CACHE = {}
def code_daily_counts(c):
    if c not in COUNT_CACHE:
        js = get_json("/device/event.json", {
            "search": f'device.device_report_product_code:"{c}" AND date_received:[{d0} TO {d1}]', "count": "date_received"})
        COUNT_CACHE[c] = js.get("results", [])
    return COUNT_CACHE[c]
vol = pd.DataFrame()
if not _cc:
    try:
        js = get_json("/device/event.json", {"search": f"date_received:[{d0} TO {d1}]",
                                             "count": "device.device_report_product_code.exact", "limit": 1000})
        vol = pd.DataFrame(js.get("results", [])).rename(columns={"term": "code", "count": "n_maude"})
        vol["code"] = vol.code.str.upper()
        print("codes with MAUDE volume:", len(vol))
    except PermissionError as ex:
        # openFDA requires an API key for cross-code aggregate counts. Fallback: candidate codes from the recall
        # database (positives: Class I recall in the label window; negatives: recalled devices that never had a
        # Class I recall), volume from per-code count queries (allowed without a key).
        print("aggregate count needs a key -> recall-database candidate pool.", str(ex)[:100])
        lab_lo0, lab_hi0 = ym(CFG["PANEL_START"]), ym(CFG["PANEL_END"]) + CFG["HORIZON"]
        isl = rec.klass.isin(CFG["LABEL_CLASSES"])
        pos_pool = sorted(set(rec[isl & rec.init_mi.between(lab_lo0, lab_hi0)].code))
        neg_pool = sorted(set(rec[rec.init_mi.between(ym(CFG["DATA_START"]), ym(CFG["DATA_END"]))].code) - set(rec[isl].code))
        r0 = np.random.default_rng(CFG["SEED"])
        pool = list(r0.permutation(pos_pool)[:N_POS_CODES * 3]) + list(r0.permutation(neg_pool)[:N_NEG_CODES * 4])
        print("candidate pool:", len(pos_pool), "pos /", len(neg_pool), "neg; querying", len(pool))
        vol = pd.DataFrame([(c, sum(r["count"] for r in code_daily_counts(c))) for c in pool], columns=["code", "n_maude"])
if _cc:
    codes = pd.read_pickle(_cc); print("loaded cached code panel:", _cc)
lab_lo, lab_hi = ym(CFG["PANEL_START"]), ym(CFG["PANEL_END"]) + CFG["HORIZON"]
is_lab = rec.klass.isin(CFG["LABEL_CLASSES"])
pos_codes = set(rec[is_lab & rec.init_mi.between(lab_lo, lab_hi)].code)
ever_lab = set(rec[is_lab].code)
if not _cc:
  cand = vol[(vol.n_maude >= MIN_TOTAL_REPORTS) & (vol.n_maude <= MAX_TOTAL_REPORTS)].copy()
  rng = np.random.default_rng(CFG["SEED"])
  pos = cand[cand.code.isin(pos_codes)]
  neg = cand[~cand.code.isin(ever_lab)]
  pos = pos.sample(min(N_POS_CODES, len(pos)), random_state=CFG["SEED"])
  neg = neg.sample(min(N_NEG_CODES, len(neg)), random_state=CFG["SEED"])
  codes = pd.concat([pos.assign(group="pos"), neg.assign(group="neg")]).reset_index(drop=True)
  names = rec.groupby("code").device_name.agg(lambda s: s.dropna().iloc[0] if s.notna().any() else "")
  codes["device_name"] = codes.code.map(names).fillna("")
  print(codes.group.value_counts())
codes.to_pickle(f"{OUT}/codes.pkl")

# %% [markdown]
# ## 3. Monthly MAUDE counts per product code (full counts, not a sample)

# %%
_ck = find_file("counts.pkl", required=False)
cnt_rows = []
for i, c in enumerate(codes.code if not _ck else []):
    # openFDA returns every daily bucket for date counts, so one call per code covers the whole period
    for r in code_daily_counts(c):
        cnt_rows.append((c, mi_from_yyyymmdd(r["time"]), r["count"]))
    if i % 10 == 0: print(i, c, "calls so far", _N_CALLS)
counts = pd.read_pickle(_ck) if _ck else pd.DataFrame(cnt_rows, columns=["code", "mi", "n"]).groupby(["code", "mi"], as_index=False).n.sum()
counts.to_pickle(f"{OUT}/counts.pkl")
# fill device names that recalls could not provide (negative codes) from the classification endpoint
for i in codes.index[codes.device_name == ""]:
    js = get_json("/device/classification.json", {"search": f'product_code:"{codes.at[i, "code"]}"', "limit": 1})
    if js.get("results"): codes.at[i, "device_name"] = js["results"][0].get("device_name", "")
codes.to_pickle(f"{OUT}/codes.pkl")
print("count rows:", len(counts))

# %% [markdown]
# ## 4. Monthly sample of MAUDE records (resumable)

# %%
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
    te, n1 = scrub_recall_mentions(" ".join(txt_e)[:3000])
    tm, n2 = scrub_recall_mentions(" ".join(txt_m)[:3000])
    probs = e.get("product_problems") or []
    tor = e.get("type_of_report") or []
    return dict(mdr_key=str(e.get("mdr_report_key", "")), code=code,
                mi=mi_from_yyyymmdd(e.get("date_received")),
                event_type=str(e.get("event_type", "")), source=str(e.get("report_source_code", "")),
                type_of_report=_j(tor),
                mfr=str(d.get("manufacturer_d_name", "") or ""), brand=str(d.get("brand_name", "") or ""),
                model=str(d.get("model_number", "") or "").strip().upper(),
                lot=str(d.get("lot_number", "") or "").strip().upper(),
                evaluated=str(d.get("device_evaluated_by_manufacturer", "") or ""),
                problems=_j(probs),
                text_event=te[:1200], text_mfr=tm[:1200], scrubbed=n1 + n2)

state_path = f"{OUT}/records_progress.json"
state = json.load(open(state_path)) if os.path.exists(state_path) else {"done": []}
# allow resuming from a previous version of this notebook attached as input
prev = find_file("records_progress.json", required=False)
if prev and prev != state_path and not state["done"]:
    state = json.load(open(prev))
    for f in glob.glob(os.path.join(os.path.dirname(prev), "records_*.pkl")):
        if not os.path.exists(os.path.join(OUT, os.path.basename(f))):
            pd.read_pickle(f).to_pickle(os.path.join(OUT, os.path.basename(f)))
    print("resumed from attached input:", len(state["done"]), "codes done")

# Without an API key the records come from openFDA's bulk quarterly files (no rate limit). Within each
# product-code-month a uniform random sample is kept by bottom-k hashing of the report key.
_have_records = find_file("records.pkl", required=False) is not None   # data notebook already attached
USE_BULK = (not API_KEY) and not MOCK and not _have_records
code_set = set(codes.code)
if USE_BULK:
    import io, zipfile, hashlib, re as _re
    from concurrent.futures import ThreadPoolExecutor
    meta = requests.get(BASE + "/download.json", timeout=120).json()
    parts_all = meta["results"]["device"]["event"]["partitions"]
    def part_q(p):
        m = _re.match(r"(\d{4}) Q(\d)", p.get("display_name", ""))
        return int(m.group(1)) * 12 + (int(m.group(2)) - 1) * 3 if m else None
    lo_, hi_ = ym(CFG["DATA_START"]), ym(CFG["DATA_END"])
    sel = [p for p in parts_all if part_q(p) is not None and lo_ - 2 <= part_q(p) <= hi_]
    print("bulk partitions:", len(sel), "| total MB:", round(sum(float(p["size_mb"]) for p in sel)))
    state.setdefault("bulk_done", [])
    def hkey(k): return int(hashlib.md5(k.encode()).hexdigest()[:12], 16)
    def fetch(url):
        for a in range(6):
            try:
                r = requests.get(url, timeout=600); r.raise_for_status(); return r.content
            except Exception as ex_:
                print("  download retry", a + 1, str(ex_)[:100]); time.sleep(15 * (a + 1))
        raise RuntimeError(url)
    todo = [p for p in sel if p["file"] not in state["bulk_done"]]
    t_start = time.time()
    with ThreadPoolExecutor(2) as pool:
        futs = {i: pool.submit(fetch, todo[i]["file"]) for i in range(min(2, len(todo)))}
        for i, p in enumerate(todo):
            if (time.time() - t_start) / 60 > TIME_BUDGET_MIN:
                print("Time budget reached - run this cell again to continue."); break
            blob = futs.pop(i).result()
            if i + 2 < len(todo): futs[i + 2] = pool.submit(fetch, todo[i + 2]["file"])
            with zipfile.ZipFile(io.BytesIO(blob)) as z:
                js = json.loads(z.read(z.namelist()[0]))
            del blob
            keep = {}; n_bad = 0
            for e in js.get("results", []):
                pcs = {str(d.get("device_report_product_code", "")).upper() for d in (e.get("device") or [])}
                for c in pcs & code_set:
                    try:
                        r = parse_event(e, c)
                    except Exception as ex_:
                        n_bad += 1; continue
                    if r["mi"] != r["mi"]: continue
                    keep.setdefault((c, int(r["mi"])), []).append((hkey(r["mdr_key"] + c), r))
            del js
            rows = []
            for g, lst in keep.items():
                lst.sort(key=lambda x: x[0]); rows += [dict(r, h=h) for h, r in lst[:RECORDS_PER_MONTH]]
            pd.DataFrame(rows).to_pickle(f"{OUT}/bulk_{len(state['bulk_done']):04d}.pkl")
            state["bulk_done"].append(p["file"]); json.dump(state, open(state_path, "w"))
            print(f"{p['display_name']}: kept {len(rows)} (skipped {n_bad} malformed) | {len(state['bulk_done'])}/{len(sel)} files | {(time.time() - t_start) / 60:.1f} min", flush=True)
        for f_ in futs.values(): f_.cancel()
    if len(state["bulk_done"]) >= len(sel):
        Bk = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(f"{OUT}/bulk_*.pkl"))], ignore_index=True)
        Bk = Bk.sort_values("h").groupby(["code", "mi"], group_keys=False).head(RECORDS_PER_MONTH)
        for c, g in Bk.groupby("code"): g.drop(columns="h").to_pickle(f"{OUT}/records_{c}.pkl")
        state["done"] = list(codes.code); json.dump(state, open(state_path, "w"))
        for f in glob.glob(f"{OUT}/bulk_*.pkl"): os.remove(f)
        del Bk

cmap = counts.set_index(["code", "mi"]).n.to_dict()
t_start = time.time()
months = range(ym(CFG["DATA_START"]), ym(CFG["DATA_END"]) + 1)
for c in (codes.code if not (USE_BULK or _have_records) else []):
    if c in state["done"]: continue
    if (time.time() - t_start) / 60 > TIME_BUDGET_MIN:
        print(f"Time budget reached. {len(state['done'])}/{len(codes)} codes done. RUN THE NOTEBOOK AGAIN to continue."); break
    rows = []
    for m in months:
        if cmap.get((c, m), 0) == 0: continue
        y, mo = m // 12, m % 12 + 1
        last = pd.Period(f"{y}-{mo:02d}").days_in_month
        js = get_json("/device/event.json", {
            "search": f'device.device_report_product_code:"{c}" AND date_received:[{y}{mo:02d}01 TO {y}{mo:02d}{last}]',
            "limit": RECORDS_PER_MONTH})
        rows += [parse_event(e, c) for e in js.get("results", [])]
    pd.DataFrame(rows).to_pickle(f"{OUT}/records_{c}.pkl")
    state["done"].append(c); json.dump(state, open(state_path, "w"))
    print(f"{c}: {len(rows)} records | {len(state['done'])}/{len(codes)} | {(time.time() - t_start) / 60:.1f} min | calls {_N_CALLS}")

# %%
if _have_records:
    print("records.pkl found in attached input - skipping download")
elif len(state["done"]) == len(codes):
    parts = [pd.read_pickle(f"{OUT}/records_{c}.pkl") for c in codes.code if os.path.exists(f"{OUT}/records_{c}.pkl")]
    R = pd.concat(parts, ignore_index=True)
    R = R[R.mi.notna()].drop_duplicates(["mdr_key", "code"]).reset_index(drop=True)
    R["mi"] = R.mi.astype(int); R["rid"] = np.arange(len(R))
    R.to_pickle(f"{OUT}/records.pkl")
    for c in codes.code:
        p = f"{OUT}/records_{c}.pkl"
        if os.path.exists(p): os.remove(p)
    print("ALL CODES DONE. records:", len(R), "| with manufacturer narrative:", (R.text_mfr.str.len() > 20).mean().round(3),
          "| sentences scrubbed for recall mentions:", int(R.scrubbed.sum()))
    summary = dict(n_codes=len(codes), n_records=len(R), n_recalls=len(rec), n_count_rows=len(counts),
                   label_classes=CFG["LABEL_CLASSES"], api_calls=_N_CALLS)
    json.dump(summary, open(f"{OUT}/data_summary.json", "w"), indent=1); print(summary)
