import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd
from PIL import Image
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5})
# Figure 3: per-event best rank
d = pd.read_csv("extras/perevent_main.csv")
fig, ax = plt.subplots(figsize=(5.2, 4.8))
below = d.full_rank < d.count_rank
ax.plot([0, 1], [0, 1], ls=":", color="#999", lw=1)
ax.axhline(0.9, color="#ccc", lw=0.8); ax.axvline(0.9, color="#ccc", lw=0.8)
ax.scatter(d.count_rank[~below], d.full_rank[~below], s=22, color="#1f5fbf", alpha=0.85, label=f"Ranked higher with narratives ({(~below).sum()})")
ax.scatter(d.count_rank[below], d.full_rank[below], s=24, facecolor="white", edgecolor="#b5651d", lw=1.2, label=f"Ranked lower with narratives ({below.sum()})")
ax.set_xlim(0, 1.01); ax.set_ylim(0, 1.01)
ax.set_xlabel("Count evidence only: best percentile rank before recall")
ax.set_ylabel("Full reading: best percentile rank before recall")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=7.5, loc="lower right")
fig.tight_layout(); fig.savefig("Figure3.png", dpi=300); fig.savefig("Figure3.tif", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)
# Supplementary Figure S1: calibration
sty = {"Fusion + full reading": ("o-", "#1f5fbf"), "Fusion, count evidence only": ("s--", "#888"),
       "Recall history (logistic regression)": ("^-.", "#d9822b"), "GBM, count + full-reading narrative features": ("D:", "#333")}
lab = {"Fusion + full reading": "Fusion + full reading", "Fusion, count evidence only": "Fusion, count evidence only",
       "Recall history (logistic regression)": "Recall history (LR)", "GBM, count + full-reading narrative features": "GBM, count + narrative features"}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.7))
for ax, f, lim, ttl in [(axs[0], "extras/calbins_main.csv", 0.35, "A. Main panel (test prevalence 0.099)"), (axs[1], "extras/calbins_random.csv", 0.12, "B. Random panel (test prevalence 0.027)")]:
    c = pd.read_csv(f)
    ax.plot([0, lim], [0, lim], color="#bbb", lw=1)
    for m, (ls, col) in sty.items():
        g = c[c.method == m]; ax.plot(g.mean_pred, g.obs, ls, color=col, ms=4, lw=1, label=lab[m])
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_title(ttl, fontsize=8.5, loc="left")
    ax.set_xlabel("Mean predicted risk (decile)"); ax.set_ylabel("Observed proportion positive")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
axs[0].legend(frameon=False, fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig("FigureS1_calibration.png", dpi=300); plt.close(fig)
for f in ["Figure1_design.tif", "Figure2.tif", "Figure3.tif"]:
    im = Image.open(f); print(f, im.mode, im.size, im.info.get("dpi"))
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[-1] if im.mode in ("RGBA", "LA") else None); im = bg
    im.save(f, compression="tiff_lzw", dpi=(600, 600))
    im2 = Image.open(f); print(" ->", im2.mode, im2.size, im2.info.get("dpi"))
