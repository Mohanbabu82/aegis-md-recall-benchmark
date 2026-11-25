# %% [markdown]
# # AEGIS-MD · Notebook 8 — Random-panel sensitivity: enrichment (GPU) + evaluation
# Input: attach the output of **Notebook 7** (random product-code panel). Accelerator: GPU T4.
# Runs the Notebook 2 enrichment (embeddings, NLI stance, precedent similarity) on the random-panel records,
# then the Notebook 6 analyses on that panel.

# %%
import os
os.environ["AEGIS_NRAND"] = "3"
os.environ["AEGIS_NBOOT"] = "1000"
