# AEGIS-MD: results on real openFDA data (Kaggle run, 23 Nov 2026)

## Data (all real; nothing synthetic)

- **Recalls:** 40,636 FDA recall records (2012–2026), joined to enforcement reports to get the recall class (98.4% have one): 3,669 Class I, 35,320 Class II, 999 Class III.
- **Device panel:** 76 product codes.
  - 50 have a Class I recall between 2016 and 2025.
  - 26 comparison codes were recalled only for less serious reasons (Class II or III), never Class I.
- **Adverse-event reports:** 334,438 MAUDE reports from openFDA's bulk files, a uniform random sample of up to 60 per code per month (2014–2025), plus the full monthly report counts for each code.
- **Benchmark:**
  - One case per product code per month. A case is positive if a new Class I recall starts in the next 6 months.
  - Leakage guards: the 6 months after each recall are dropped; reports count only from the date FDA received them; older recalls count only after FDA posted them; sentences mentioning a recall are removed from report text.

| Split | Period | Cases | Positive | Prevalence |
|---|---|---|---|---|
| Train | 2016-01 to 2020-12 | 4,329 | 197 | 4.6% |
| Validation | 2021-01 to 2022-06 | 1,244 | 79 | 6.4% |
| Test | 2022-07 to 2025-06 | 2,407 | 238 | 9.9% (67 recall events) |

- **Text models (GPU):** report text embedded with BAAI/bge-small-en-v1.5. Manufacturer investigation notes scored by cross-encoder/nli-deberta-v3-base for whether they confirm or rule out the reported problem.

## Results on the test period (95% CIs from a bootstrap over product codes)

| Method | Reports read per case | AUPRC [95% CI] | AUROC | Recall events flagged |
|---|---|---|---|---|
| Poisson exceedance | 0 | 0.137 [0.084, 0.217] | 0.574 | 38% |
| Poisson CUSUM | 0 | 0.136 [0.082, 0.217] | 0.545 | 27% |
| Mann–Kendall trend | 0 | 0.132 [0.080, 0.231] | 0.580 | 22% |
| Isolation Forest | 0 | 0.118 [0.076, 0.203] | 0.505 | 29% |
| GBM on routine count features | 0 | 0.146 [0.093, 0.217] | 0.602 | 33% |
| Fusion, routine evidence only | 0 | 0.133 [0.081, 0.228] | 0.538 | 36% |
| Fusion + generic-query RAG (k=20) | 18.5 | 0.170 [0.091, 0.278] | 0.635 | 35% |
| Fusion + random reading (k=40) | 34.2 | 0.176 [0.095, 0.287] | 0.636 | 31% |
| **Fusion + reading all recent reports** | 107.9 | **0.177 [0.103, 0.276]** | **0.660** | 33% |
| **GBM, routine + all narrative evidence** | 107.9 | **0.183 [0.109, 0.285]** | **0.663** | 42% |
| AEGIS gap retrieval (VoI policy) | 25.6 | 0.117 [0.075, 0.191] | 0.543 | 40% |
| AEGIS gap retrieval, all 4 gaps | 54.7 | 0.121 [0.073, 0.199] | 0.546 | 40% |
| AEGIS adaptive investigation (VoI-gated) | 107.8 | 0.140 [0.094, 0.216] | 0.608 | 26% |
| Ablation: gap retrieval without contrast | 24.3 | 0.121 | 0.534 | 36% |

Median lead time for flagged recall events: 5–6 months before the recall started.

## What the real data show

1. **Report counts alone barely predict Class I recalls.** Poisson, CUSUM, trend and Isolation Forest score AUROC 0.50–0.58, close to chance.
2. **Reading what the reports say helps.** Adding evidence from the report text (failure mechanism, the manufacturer's investigation findings, lot/model concentration, similarity to past recalls) raises AUROC from about 0.54–0.60 to 0.66. AUPRC rises from 0.13 to 0.18, against a test prevalence of 0.099. The confidence intervals overlap, so this is a trend, not a proven gain.
3. **The proposed gap-targeted retrieval and the VoI policy did not beat reading everything or simple generic retrieval** (AUROC about 0.54–0.61). Contrastive querying did not fix this.
   - Likely reason: each gap's evidence comes from only 10 reports chosen by a query. That sample is noisy and biased by the selection, while the whole-window statistics are stable.

## Implications for the paper

- As it stands, the evidence supports a **benchmark and empirical-study** framing:
  - the first leakage-controlled MAUDE-to-recall benchmark;
  - count-based surveillance is weak for predicting Class I recalls;
  - narrative evidence adds signal.
- It does **not** support claiming that AEGIS is a better detector.
- Options before targeting a Q1 journal:
  - predict recalls at the firm/manufacturer level instead of the product-code level;
  - use Class I + II recalls as labels;
  - use more reports per month;
  - retrieve evidence once per gap and pool it across months instead of using 10 reports per gap.
