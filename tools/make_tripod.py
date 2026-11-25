import re
pages = open("ms.txt").read().split("\f")
def L(ph):
    ph = re.sub(r"\s+", " ", ph)
    for pi, pg in enumerate(pages, 1):
        for ln in pg.split("\n"):
            m = re.match(r"\s*(\d+)\s{2,}(.*)", ln)
            if m and ph in re.sub(r"\s+", " ", m.group(2)): return f"p. {pi}, line {m.group(1)}"
    raise SystemExit("not found: " + ph)
NA = "Not applicable"
rows = [
("Title", "1", "Identify the study as developing or evaluating a prediction model, the target population and the outcome", L("Do adverse event narratives anticipate")),
("Abstract", "2", "See the TRIPOD+AI for Abstracts checklist", L("Objective:") + " (structured abstract)"),
("Background", "3a", "Healthcare context and rationale, with reference to existing models", L("The Manufacturer and User Facility") + "; " + L("Table 1 summarizes recent studies")),
("Background", "3b", "Target population, intended purpose and intended users", L("For safety teams") + " (regulators, manufacturers, hospital safety teams; ranked review queue)"),
("Background", "3c", "Known health inequalities between sociodemographic groups", NA + ": the unit of analysis is a device product code; no patient-level data were used"),
("Objectives", "4", "Study objectives; development and/or evaluation", L("We built the benchmark and tested")),
("Data", "5a", "Data sources for development and evaluation; rationale; representativeness", L("All data came from openFDA") + "; " + L("The main panel comprised")),
("Data", "5b", "Dates of data, start and end of accrual and follow-up", L("All data came from openFDA") + "; " + L("Table 2. Benchmark splits")),
("Participants", "6a", "Study setting, number and location of centres", L("All data came from openFDA") + " (US national passive surveillance; single data source)"),
("Participants", "6b", "Eligibility criteria", L("The main panel comprised") + "; " + L("We repeated the main comparisons")),
("Participants", "6c", "Treatments received and how they were handled", NA),
("Data preparation", "7", "Data pre-processing and quality checking", L("Recall events") + "; " + L("The unit of analysis") + "; Supplementary Methods S1.1"),
("Outcome", "8a", "Outcome definition, time horizon, how and when assessed, rationale", L("The unit of analysis") + "; " + L("Recall events")),
("Outcome", "8b", "Qualifications of outcome assessors if subjective", NA + ": outcome taken from FDA recall classification"),
("Outcome", "8c", "Blinding of outcome assessment", NA + ": recall class assigned by FDA, independent of the predictors"),
("Predictors", "9a", "Choice of initial predictors and any pre-selection", L("For each case we computed") + "; " + L("Narratives were embedded")),
("Predictors", "9b", "Definition of all predictors, how and when measured", L("For each case we computed") + "–" + L("We varied which reports") + "; Supplementary Methods S1.1–S1.5"),
("Predictors", "9c", "Qualifications of predictor assessors if subjective", NA + ": predictors computed automatically"),
("Sample size", "10", "How the study size was arrived at and whether it was sufficient", L("The main panel comprised") + "; " + L("With 76 codes and 55") + " (no formal sample size calculation; limitation stated)"),
("Missing data", "11", "Handling of missing data", "Supplementary Methods S1.2 (median imputation in calibrators; absent sources contribute zero log-odds)"),
("Analytical methods", "12a", "How data were used and partitioned", L("Data were split by time") + "; " + L("Models were trained on the training")),
("Analytical methods", "12b", "Handling of predictors (functional form, rescaling, transformation)", L("Each source was converted") + "; Supplementary Methods S1.2, S1.4"),
("Analytical methods", "12c", "Model type, rationale, building steps, tuning, internal validation", L("Each source was converted") + "; " + L("Gradient-boosted trees") + "; Supplementary Table S2"),
("Analytical methods", "12d", "Heterogeneity across clusters", L("Confidence intervals came from") + " (bootstrap over product codes); leave-one-code-out analysis, " + L("At the same threshold")),
("Analytical methods", "12e", "Performance measures and plots, and model comparison", L("Models were trained on the training") + "; " + L("Hypotheses were tested")),
("Analytical methods", "12f", "Model updating (eg, recalibration)", "No updating was performed; need for recalibration discussed at " + L("All models underestimated risk on the test")),
("Analytical methods", "12g", "How predictions were calculated for evaluation", L("Each source was converted") + "; code in the repository (Data Availability)"),
("Class imbalance", "13", "Class imbalance methods", "None used; AUPRC chosen as primary metric, " + L("We built the benchmark and tested")),
("Fairness", "14", "Approaches to model fairness", NA + ": no individual-level or sociodemographic data"),
("Model output", "15", "Model output and any classification thresholds", L("Models were trained on the training") + " (risk probabilities; alert threshold = top 10% of validation scores)"),
("Training versus evaluation", "16", "Differences between development and evaluation data", L("Prevalence rose across the splits") + "; " + L("We repeated the main comparisons")),
("Ethical approval", "17", "Ethics committee approval or waiver", L("The study used only public")),
("Open science", "18a", "Funding", L("This research received no")),
("Open science", "18b", "Conflicts of interest", L("The authors have no competing")),
("Open science", "18c", "Protocol", L("All source data are public") + " (no protocol prepared)"),
("Open science", "18d", "Registration", L("All source data are public") + " (not registered)"),
("Open science", "18e", "Data sharing", L("All source data are public")),
("Open science", "18f", "Code sharing", L("All source data are public")),
("Patient and public involvement", "19", "Patient and public involvement", L("Patients and the public were not")),
("Participants (results)", "20a", "Flow of participants, numbers with and without the outcome", L("The benchmark held 7,980") + "; Table 2; Figure 1"),
("Participants (results)", "20b", "Characteristics, key dates, sample size, outcome events, missing data", L("The benchmark held 7,980") + "; Supplementary Table S1"),
("Participants (results)", "20c", "Comparison of development and evaluation data", L("Prevalence rose across the splits") + "; Table 2"),
("Model development", "21", "Number of participants and outcome events in each analysis", "Table 2; " + L("The benchmark held 7,980") + "; Supplementary Table S6 note"),
("Model specification", "22", "Details of the full model to allow predictions and third-party evaluation", "Supplementary Methods S1.2, Supplementary Table S2; code and fitted weights in the repository (Data Availability)"),
("Model performance", "23a", "Performance with confidence intervals; plots", "Table 3; " + L("Count-based detectors were close") + "; Figures 2–3; Supplementary Table S3, Figure S1"),
("Model performance", "23b", "Heterogeneity in performance across clusters", L("At the same threshold") + " (per event, leave-one-code-out); " + L("On the random panel (110")),
("Model updating", "24", "Results of any model updating", NA + ": no updating performed"),
("Interpretation", "25", "Overall interpretation, including fairness, with previous studies", L("In this leakage-controlled benchmark")),
("Limitations", "26", "Limitations and their effects on bias, uncertainty and generalisability", L("The main panel was selected on the outcome")),
("Usability", "27a", "Handling of poor-quality or unavailable input data in implementation", L("Report content was a sample") + "; Supplementary Methods S1.2"),
("Usability", "27b", "User interaction and expertise required", L("For safety teams") + " (human review queue)"),
("Usability", "27c", "Next steps for future research", L("Future work should")),
]
out = ["# TRIPOD+AI CHECKLIST", "",
"Do adverse event narratives anticipate Class I medical device recalls? A leakage-controlled MAUDE benchmark. Locations refer to page and line numbers of the submitted manuscript file (AEGIS-MD_JAMIA_Manuscript_v3.docx) unless a Supplementary item is named. Item descriptions are abbreviated from the TRIPOD+AI statement (Collins et al., BMJ 2024;385:e078378); the official form is available from the TRIPOD website.", "",
"| Section/topic | Item | Checklist item (abbreviated) | Where reported |", "| --- | --- | --- | --- |"]
out += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
open("tripod_checklist.md", "w").write("\n".join(out) + "\n")
print(len(rows))
