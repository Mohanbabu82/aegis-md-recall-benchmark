# AEGIS-MD benchmark datasheet

All files were written by the Kaggle revision runs (`src/nb6_revision.py`) from the openFDA snapshot used in the paper (main panel 23 November 2026; random panel 20 November 2026). No patient-level data are included; MAUDE narratives are public and de-identified by FDA.

## `datasheet/` – panels, recall events and splits

| File | Panel | Contents |
| --- | --- | --- |
| `main_product_codes.csv` | Main (76 codes) | One row per FDA product code: `code`, `group` (`pos` = 50 codes with a Class I recall in 2016–2025; `neg` = 26 comparison codes recalled only in other classes), `device_name`, `maude_reports_2014_2025` (all reports received), `sampled_records` (reports in the study sample), `cases_train`, `cases_val`, `cases_test` (code-months per split after washout), `class_I_events_2016_2025`. |
| `main_recall_events.csv` | Main | One row per Class I recall event: `code`, `event_id` (openFDA `res_event_number`), `n_rows` (recall records in the event), `init_month` (recall initiation), `posted_month` (FDA posting; earlier recalls are used as predictors only from this month), `split_of_initiation` (`pre-panel`, `train`, `val`, `test`). |
| `main_benchmark_summary.csv` | Main | Cases, positives, prevalence, codes and events per split. |
| `main_events_per_year.csv` | Main | Class I events by initiation year. |
| `random_product_codes.csv` | Random (110 codes) | Same columns; `group` = `random`. |
| `random_recall_events.csv` | Random | Same columns. |
| `random_benchmark_summary.csv` | Random | Same columns as the main summary. |
| `random_events_per_year.csv` | Random | Class I events by initiation year. |
| `random_panel_selection.json` | Random | How the random panel was drawn: 900 of 7,103 classification codes (seed 2026); 110 had at least 24 reports received in 2014–2015; 20 had a Class I recall. |

Splits are by month of the case: train 2016-01 to 2020-12, validation 2021-01 to 2022-06, test 2022-07 to 2025-06. A case (product code, month t) is positive if a new Class I recall begins in months t+1 to t+6; months within 6 months after a Class I recall initiation are excluded (washout). Product code CBK has no test cases because all its test months fall in washout windows.

## `labelling/` – blind manual-validation sheets (main panel)

These sheets were prepared for human review but were **not labelled** for the study (see Supplementary Methods S1.6 of the paper). They are released so the stance signal and the leakage filter can be checked independently.

| File | Contents |
| --- | --- |
| `labelling_stance_200.csv` | 200 manufacturer narratives, 40 from each fifth of the stance score p(confirmed) − p(not confirmed), shuffled. Fill `label` with `confirmed`, `not_confirmed` or `unclear`. |
| `labelling_stance_200_KEY_do_not_open.csv` | Model scores for the stance items (`item_id`, `rid`, `p_confirm`, `p_refute`). **Do not open before labelling.** |
| `labelling_leakage_300.csv` | 300 narratives after recall-sentence filtering (200 from reports received in a positive case month or the 2 months before it, 100 at random). Fill `mentions_recall_or_field_action` with `Y` or `N`. |

Score completed sheets with `python tools/score_labels.py data/labelling/labelling_stance_200.csv data/labelling/labelling_stance_200_KEY_do_not_open.csv data/labelling/labelling_leakage_300.csv` (Cohen's kappa, macro F1 and AUROC for stance; residual leakage rate with a Wilson 95% interval).
