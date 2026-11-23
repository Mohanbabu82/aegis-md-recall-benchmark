"""Local stand-in for the openFDA API (testing only). Returns responses shaped like openFDA JSON."""
import re, numpy as np

FAILS = ["battery overheated and the housing melted", "catheter tip separated during insertion",
         "pump delivered incorrect dose due to software error", "seal leaked causing fluid ingress",
         "sensor gave falsely low readings", "connector cracked under normal use"]
GENERIC = ["patient reported discomfort after use", "device alarmed during therapy and was replaced",
           "user reported difficulty operating the device", "device did not function as expected per user",
           "minor skin irritation noted at the site", "device was replaced as a precaution"]
PROBS_G = ["Use of Device Problem", "Device Alarm System", "Adverse Event Without Identified Device or Use Problem",
           "Patient Device Interaction Problem"]
PROBS_F = ["Overheating of Device", "Separation Problem", "Incorrect Dose Delivered", "Fluid Leak",
           "Incorrect Measurement", "Crack"]

class World:
    def __init__(self, seed=7):
        rng = np.random.default_rng(seed)
        self.codes = [f"Q{chr(65 + i // 26)}{chr(65 + i % 26)}" for i in range(40)]
        self.base = {c: float(rng.uniform(4, 40)) for c in self.codes}
        self.fail = {c: int(rng.integers(len(FAILS))) for c in self.codes}
        self.recalls = []   # (code, init_mi, klass, event_id, recall_number)
        self.surge = {}     # (code, mi) -> intensity
        rid = 0
        for i, c in enumerate(self.codes):
            n_ev = int(rng.integers(1, 4)) if i < 25 else 0
            for _ in range(n_ev):
                e = int(rng.integers(2016 * 12 + 3, 2025 * 12 + 10))
                rid += 1
                self.recalls.append((c, e, "Class I", f"E{rid}", f"Z-{rid:04d}-{e // 12}"))
                for k in range(1, 6): self.surge[(c, e - k)] = 1.0 - 0.15 * (k - 1)
            # decoy surges (no recall) and class II recalls
            for _ in range(int(rng.integers(0, 3))):
                e = int(rng.integers(2015 * 12, 2025 * 12 + 10))
                for k in range(0, 3): self.surge[(c, e - k)] = -0.8
            if rng.random() < 0.5:
                e = int(rng.integers(2014 * 12, 2026 * 12)); rid += 1
                self.recalls.append((c, e, "Class II", f"E{rid}", f"Z-{rid:04d}-{e // 12}"))
        # other-device historical recalls (precedent corpus)
        for j in range(300):
            e = int(rng.integers(2010 * 12, 2026 * 12)); rid += 1
            self.recalls.append((f"X{j:03d}", e, ["Class I", "Class II", "Class III"][j % 3], f"E{rid}", f"Z-{rid:04d}-{e // 12}"))

    def n(self, c, m):
        rng = np.random.default_rng(hash((c, m)) % 2**32)
        s = self.surge.get((c, m), 0)
        if c not in self.base: return 0
        lam = self.base[c] * (1 + 0.003 * (m - 2014 * 12)) * (1.0 + (0.9 * s if s > 0 else (0.7 if s < 0 else 0)))
        return int(rng.poisson(lam))

    def records(self, c, m, limit):
        n = min(self.n(c, m), limit); rng = np.random.default_rng((hash((c, m, "r")) % 2**32))
        s = self.surge.get((c, m), 0); out = []
        for k in range(n):
            sig = s > 0 and rng.random() < 0.25 + 0.4 * s
            decoy = s < 0 and rng.random() < 0.4
            f = self.fail[c]
            if sig:
                ev = f"It was reported that the {FAILS[f]}. Patient required intervention."
                mf = rng.choice(["The device was returned and evaluated. The reported failure was confirmed; root cause was determined to be a component defect.",
                                 "Investigation found the failure was reproduced in lot testing. This device is subject to recall Z-1111-2020.",
                                 "Device not returned; investigation ongoing."], p=[.6, .2, .2])
                lot = f"L{f}{m % 3}"; prob = PROBS_F[f]; et = rng.choice(["Injury", "Malfunction"], p=[.5, .5])
            else:
                ev = str(rng.choice(GENERIC)) + "."
                if decoy: ev = f"It was reported that the {FAILS[(f + 1) % len(FAILS)]}."
                mf = rng.choice(["The device was evaluated and no malfunction was found. The device functioned as intended.",
                                 "The device was not returned for evaluation; the complaint could not be confirmed.",
                                 "Probable cause was user error.", ""], p=[.35, .3, .15, .2])
                lot = f"L{int(rng.integers(1000))}"; prob = str(rng.choice(PROBS_G)) if not decoy else PROBS_F[(f + 1) % 6]
                et = rng.choice(["Malfunction", "Injury", "Death", "Other"], p=[.6, .3, .02, .08])
            day = int(rng.integers(1, 28))
            out.append({"mdr_report_key": f"{c}{m}{k}", "date_received": f"{m // 12}{m % 12 + 1:02d}{day:02d}",
                        "event_type": str(et), "report_source_code": str(rng.choice(["Manufacturer report", "Voluntary report", "User facility report"], p=[.7, .15, .15])),
                        "type_of_report": ["Initial submission"], "product_problems": [prob],
                        "device": [{"device_report_product_code": c, "manufacturer_d_name": "ACME MED", "brand_name": "ACME " + c,
                                    "model_number": f"M{int(rng.integers(3))}", "lot_number": lot, "device_evaluated_by_manufacturer": "Y"}],
                        "mdr_text": [{"text_type_code": "Description of Event or Problem", "text": ev},
                                     {"text_type_code": "Additional Manufacturer Narrative", "text": str(mf)}]})
        return out

W = None
def world():
    global W
    if W is None: W = World()
    return W

def _rng_years(s):
    m = re.search(r"\[(\d{4})-?(\d{2})-?(\d{2}) TO (\d{4})-?(\d{2})-?(\d{2})\]", s)
    return (int(m.group(1)) * 12 + int(m.group(2)) - 1, int(m.group(4)) * 12 + int(m.group(5)) - 1) if m else None

def respond(endpoint, p):
    w = world(); s = p.get("search", ""); lim = int(p.get("limit", 1)); skip = int(p.get("skip", 0))
    def page(res):
        return {"meta": {"results": {"total": len(res), "skip": skip, "limit": lim}}, "results": res[skip:skip + lim]}
    if endpoint == "/device/recall.json":
        lo, hi = _rng_years(s)
        res = [{"res_event_number": e, "product_res_number": rn, "product_code": c, "recalling_firm": "ACME MED",
                "event_date_initiated": f"{m // 12}-{m % 12 + 1:02d}-10", "event_date_posted": f"{(m + 1) // 12}-{(m + 1) % 12 + 1:02d}-02",
                "reason_for_recall": (f"The {FAILS[w.fail[c]]} in affected lots." if c in w.fail else f"Units may exhibit {FAILS[int(e[1:]) % 6]}."),
                "root_cause_description": "Component design", "openfda": {"device_name": f"Device {c}"}}
               for c, m, k, e, rn in w.recalls if lo <= m <= hi]
        return page(res)
    if endpoint == "/device/enforcement.json":
        lo, hi = _rng_years(s)
        return page([{"recall_number": rn, "classification": k} for c, m, k, e, rn in w.recalls if lo <= m <= hi])
    if endpoint == "/device/classification.json":
        return {"results": [{"device_name": "Mock device"}]}
    if endpoint == "/device/event.json":
        if p.get("count") == "device.device_report_product_code.exact":
            import os
            if os.environ.get("AEGIS_MOCK_403"): raise PermissionError("openFDA 403 API_KEY_MISSING (mock)")
            return {"results": [{"term": c, "count": int(w.base[c] * 140)} for c in w.codes]}
        code = re.search(r'product_code:"(\w+)"', s).group(1)
        lo, hi = _rng_years(s)
        if p.get("count") == "date_received":
            return {"results": [{"time": f"{m // 12}{m % 12 + 1:02d}15", "count": w.n(code, m)} for m in range(lo, hi + 1) if w.n(code, m) > 0]}
        return {"meta": {"results": {"total": w.n(code, lo)}}, "results": w.records(code, lo, lim)}
    return {"results": []}
