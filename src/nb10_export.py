# %% [markdown]
# # AEGIS-MD export of datasheets and labelling sheets
# Copies the benchmark datasheets and the blind labelling sheets written by the revision runs into
# small JSON parts (each under 90 KB) so they can be added to the public repository.

# %%
import glob, os, json
OUT = "/kaggle/working/export"; os.makedirs(OUT, exist_ok=True)
WANT = ["labelling_stance_200.csv", "labelling_stance_200_KEY_do_not_open.csv", "labelling_leakage_300.csv",
        "datasheet_product_codes.csv", "datasheet_recall_events.csv", "benchmark_summary_v2.csv", "events_per_year.csv",
        "data_summary.json"]
index = []
for f in sorted(glob.glob("/kaggle/input/**/*", recursive=True)):
    b = os.path.basename(f)
    if b not in WANT: continue
    panel = "random" if "random-panel" in f else "main"
    txt = open(f, encoding="utf-8").read()
    lines = txt.splitlines(keepends=True)
    parts, cur = [], ""
    for ln in lines:
        if len(cur) + len(ln) > 85000 and cur: parts.append(cur); cur = ""
        cur += ln
    parts.append(cur)
    for i, p in enumerate(parts):
        name = f"{panel}__{b}__part{i:02d}.json"
        json.dump({"panel": panel, "file": b, "part": i, "n_parts": len(parts), "text": p}, open(f"{OUT}/{name}", "w"))
        index.append(name)
    print(panel, b, len(txt), "chars", len(parts), "parts", f)
json.dump(index, open(f"{OUT}/index.json", "w"))
print("ALL DONE", len(index))
