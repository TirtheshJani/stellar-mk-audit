# Project Plan: Physics-Informed LightGBM MK Classifier on Gaia-ESO UVES

## Paper in one paragraph

A solo-authored short paper for arXiv astro-ph.IM, then submission to Research in Astronomy and Astrophysics or Open Journal of Astrophysics. Claim: tree-based stellar classifiers can be audited for physics-awareness via causal masked-line ablation, a test that permutation importance (Li, Lin and Qiu 2019) and saliency maps (Candebat et al. 2024) cannot provide. Headline result: a LightGBM MK classifier on Gaia-ESO FLAMES-UVES spectra either does or does not show class-specific accuracy drops when canonical MK diagnostic regions (H Balmer, Mg b, Na D) are masked. Either outcome is publishable. Supported by permutation importance, TreeSHAP, sliding-window occlusion, and quantitative peak-matching against a canonical MK line list.

## Timeline

12 weeks at side-project cadence: roughly 8 to 10 hours per week, evenings and weekends. Not 4 to 6 weeks. Do not compress further.

| Phase | Weeks | Deliverable |
|---|---|---|
| 0. Pre-flight decisions | week 0 | Three decisions locked |
| 1. Data and labels | weeks 1 to 2 | features.npz, labels.parquet, per-class counts |
| 2. Baseline classifier | weeks 3 to 4 | trained model, metrics.json, confusion matrix |
| 3. Interpretability triangulation | weeks 5 to 6 | perm, SHAP, occlusion + agreement report |
| 4. Causal ablation + line matching | weeks 7 to 8 | ablation.csv with CIs, line_match.csv |
| 5. External benchmark + figures | week 9 | benchmark confusion matrix, two figures |
| 6. Drafting | weeks 10 to 11 | full manuscript |
| 7. Submission | week 12 | endorsement secured, arXiv posted |

## Non-negotiables

- Finish this paper before starting anything else.
- Honest in-progress assessments at each phase gate. If a gate fails, pivot or kill.
- Document hyperparameters and random seeds completely (Li Lin Qiu's failure mode).
- Publish code and data at submission time (Candebat's standard).
- No em dashes in the manuscript or the code comments.

---

## Phase 0: Pre-flight decisions (week 0)

Three decisions gate the project. Make them before touching any code.

### Decision 1: Wavelength window

**Option A (narrow, default).** Keep 4800 to 6800 Å. Drop `Ca_HK` and `TiO_bands` from `LINE_SETS`. Physics claim restricted to Hα, Hβ, Mg b, Na D. M-star audit becomes impossible because TiO bands are out of window; drop M from the label set accordingly.

**Option B (wide).** Extend to ~3800 to 7200 Å. Requires either combining UVES U520 (4150 to 6200 Å) and U580 (4800 to 6800 Å) setups, or restricting the sample to stars observed in U520. Gains Ca II H and K and (partially) TiO γ band at 7050-7100 Å.

**How to decide in an hour:** Open the HDF5, filter to survey='ges', and count per-spectrum coverage of (3800, 4800, 6800, 7200) Å. If Option B leaves fewer than 3000 spectra, default to Option A.

- [ ] Query HDF5 for UVES wavelength coverage per spectrum
- [ ] Count spectra surviving each option's window
- [ ] Lock window choice in a decision log file

### Decision 2: External benchmark

**Option A (cheapest).** GES DR5.1 recommended parameters table may contain a spectral-type column (MK from Gaia-ESO pipeline). Check Vizier J/A+A/692/A228 companion tables. If present, use it.

**Option B (most defensible).** Pickles (1998) template matching via chi-squared. Independent physics. Most work.

**Option C (middle ground).** SIMBAD cross-match by CNAME or coordinates, pull `sp_type`. Heterogeneous quality but fast.

- [ ] Check GES DR5.1 parameters table columns
- [ ] If GES sp_type present, lock Option A; otherwise choose B or C
- [ ] Lock benchmark choice in the decision log

### Decision 3: Surviving classes

GES UVES is FGK-heavy by design. Run per-class counts before writing code.

- [ ] Build a throwaway labels table from GES T_eff
- [ ] Report per-class counts
- [ ] Drop classes with fewer than 50 spectra
- [ ] Lock the surviving class set (likely F, G, K; possibly A and M)

### Phase 0 kill criteria

If the window decision leaves fewer than 2000 spectra, or if only two MK classes survive with ≥ 50 samples, the paper does not work as framed. Pivot to **preprocessing-methods paper** (candidate B from your memory) or **normalization ablation on LAMOST** where the sample sizes are not a constraint.

---

## Phase 1: Data and labels (weeks 1 to 2)

Work the code plan's `build_labels.py` and `build_features.py`. Changes from the original plan noted below.

### Tasks

- [ ] Implement `src/interpret/labels.py` with self-bootstrapping GES parameters fetch
- [ ] Add `boundary_distance_k` column (distance from T_eff to nearest class edge)
- [ ] Implement `build_labels.py` CLI
- [ ] Write `tests/test_labels.py`
- [ ] Implement `src/interpret/features.py` with the window chosen in Phase 0
- [ ] Implement `build_features.py` CLI
- [ ] Write `tests/test_features.py`
- [ ] Add mask-coverage test: every `LINE_SETS` entry must intersect the feature window
- [ ] Run end-to-end: produce `features.npz` and `ges_mk_labels.parquet`
- [ ] Write a one-page data description (sample sizes, SNR distribution, class balance)

### Watchouts

- Do not use full 17 000-pixel resolution. Rebin ~5 native pixels → ~1000 features (plan is correct; confirm after window lock).
- If Option A window is locked, update `LINE_SETS` in `src/interpret/lines.py` BEFORE running tests.
- Impute missing flux with class-conditional bin means computed on train only, never on test. Leakage is easy here.

### Gate to Phase 2

- [ ] features.npz loads, shape matches (N, n_bins)
- [ ] Per-class counts documented; surviving classes locked
- [ ] All Phase 1 tests pass
- [ ] Decision log updated with any deviations

---

## Phase 2: Baseline classifier (weeks 3 to 4)

### Tasks

- [ ] Implement `src/interpret/classifier.py` (LightGBM, `class_weight='balanced'`, documented hyperparameters)
- [ ] Implement `train_classifier.py` CLI
- [ ] 5-fold stratified CV for robust metric estimation
- [ ] Save `metrics.json` with overall accuracy, macro-F1, per-class precision/recall/F1, confusion matrix
- [ ] Produce confusion matrix figure for internal review
- [ ] Run on boundary-distance-filtered subset as sensitivity check

### Gate to Phase 3

- [ ] Per-class recall ≥ 0.6 on every surviving class with ≥ 100 training samples
- [ ] Macro-F1 ≥ 0.7
- [ ] Confusion matrix inspected; class confusions make physical sense (adjacent MK classes mix, non-adjacent should not)
- [ ] If a class has recall below 0.4, either drop it or add a note explaining why the ablation result for that class will be uninterpretable

### Kill criteria

- If baseline macro-F1 is below 0.5, the model is not good enough to audit. Either revisit feature engineering (unlikely to help at this late stage) or restrict the paper to two or three best-performing classes only. Worst case: pivot to the preprocessing-methods paper.

---

## Phase 3: Interpretability triangulation (weeks 5 to 6)

### Tasks

- [ ] Implement `src/interpret/importance.py` (sklearn `permutation_importance`, `n_repeats=10`, validation set)
- [ ] Implement `src/interpret/shap_explain.py` (TreeExplainer, stratified subsample cap 1000 rows)
- [ ] Add bootstrap CIs on SHAP rankings (Jaccard stability across 100 bootstraps)
- [ ] Implement `src/interpret/occlusion.py` sliding-window function (NOT masked-line here; that is Phase 4)
- [ ] Implement `run_interpret.py` CLI that produces three artifact files
- [ ] Add three-way agreement report: perm vs SHAP, perm vs occlusion, SHAP vs occlusion, and three-way intersection
- [ ] Flag any pairwise Jaccard below 0.5 in the log (do not hide the disagreement)
- [ ] Produce per-class importance figure (one panel per surviving class)

### Gate to Phase 4

- [ ] All three methods produce non-degenerate outputs
- [ ] Pairwise Jaccard between perm and SHAP on top-20 features ≥ 0.5 (relaxed from 0.6 in original plan because three-way is more informative than two-way stringency)
- [ ] Per-class importance figure shows class-specific wavelength preferences, not uniform importance

### Watchouts

- If SHAP takes longer than 2 hours per class to compute, reduce subsample further or down-sample the feature set.
- Do not report Gini/impurity importance as a primary result. It is biased toward high-cardinality features and Li, Lin and Qiu explicitly rejected it.

---

## Phase 4: Causal ablation + line matching (weeks 7 to 8)

This is the headline. Both pieces are new work beyond the original code plan.

### Tasks

- [ ] Extend `src/interpret/occlusion.py` with `masked_line_ablation`
- [ ] Implement bootstrap CIs (500 resamples of test set, 95% CI) on delta-accuracy per class per line set
- [ ] Implement random-window null distribution (100 random windows matched to line-set total width per line set)
- [ ] Report p-value of line-set delta-accuracy against random-window null
- [ ] Implement `src/interpret/line_match.py`:
  - [ ] `detect_peaks(importance, wave_centers, prominence)`
  - [ ] `match_peaks_to_lines(peaks, MK_LINES, tolerance_aa)`
  - [ ] `compute_match_metrics(matches, MK_LINES_in_window)` → precision, recall, Jaccard
- [ ] Sweep tolerance at 1, 2, 5, 10 Å
- [ ] Implement `ablation.py` CLI producing `masked_line_ablation.csv` with columns: `line_set, mk_class, n_test, baseline_acc, masked_acc, delta_acc_mean, delta_acc_ci_low, delta_acc_ci_high, p_value_vs_random`
- [ ] Produce `line_match.csv` and summary JSON
- [ ] Write `tests/test_ablation.py` including mask-coverage assertion

### Gate to Phase 5 (falsifiable physics claim)

- [ ] For at least two of (H_balmer on A, Mg_b on G/K, Na_D on K), p-value against random null ≤ 0.01 AND delta_acc_ci_low > 0
- [ ] Line-matching precision ≥ 0.5 at 5 Å tolerance against MK_LINES_in_window

### Kill criteria (and fallbacks)

- If NO line set shows p ≤ 0.05 against random null for any class, the model is not physics-aware in the way the paper assumes. This is actually still publishable, but the paper has to be rewritten from "here is how to audit" to "audit reveals shortcut learning in tree-based stellar classifiers." Budget an extra two weeks for that rewrite.
- If the peak-matching precision is below 0.3 at 10 Å tolerance, the line-matching contribution is weak; keep it but de-emphasize, lead with the ablation.

---

## Phase 5: External benchmark + figures (week 9)

### Tasks

- [ ] Implement `src/interpret/benchmark.py` per Phase 0 decision (GES sp_type / SIMBAD / Pickles)
- [ ] Produce benchmark confusion matrix: LightGBM vs external labels
- [ ] Document disagreement rate and whether disagreements concentrate on class boundaries
- [ ] Implement `make_figure.py`:
  - [ ] Figure 1: main result (importance + spectrum + line markers, single panel)
  - [ ] Figure 2: per-class importance panel (one subplot per class)
- [ ] Produce confusion matrix figure for the paper (different from Phase 2 internal version)
- [ ] Produce ablation summary figure (bar chart of delta-accuracy with CIs and random-null band)

### Gate to Phase 6

- [ ] Benchmark agreement with external labels ≥ 70% on classes where both agree at >100 samples (honest threshold; Li Lin Qiu got 93% but on different task setup)
- [ ] All figures render cleanly at print resolution
- [ ] Figure 1 tells the whole paper's story in one image

---

## Phase 6: Drafting (weeks 10 to 11)

### Writing order

Write in this sequence. Do not start with the introduction.

1. **Methods** (week 10, early). Write while you still remember every decision.
2. **Results** (week 10, mid). One subsection per main artifact: baseline performance, triangulation agreement, ablation, line matching, benchmark.
3. **Discussion** (week 10, late). What the ablation shows about physics-awareness. Honest limitations: scale (~5K vs 50K), no calibrated uncertainty, single survey, single model family.
4. **Introduction** (week 11, early). Two paragraphs differentiating from Li Lin Qiu 2019 and Candebat et al. 2024 (use the exact framing from the gap map).
5. **Abstract** (week 11, mid). Last.
6. **Conclusion** (week 11, mid). One paragraph. What you did, what the headline finding was, what comes next.

### Writing tasks

- [ ] Methods draft
- [ ] Results draft with all figures embedded
- [ ] Discussion draft including limitations honestly
- [ ] Introduction with two prior-work paragraphs (Li Lin Qiu, Candebat)
- [ ] Abstract (four to six sentences, claim-forward)
- [ ] Conclusion
- [ ] Full manuscript read-through for em dashes (you do not use them)
- [ ] Cross-check every numerical claim against the artifact CSVs and JSONs
- [ ] Compile reference list; verify every citation resolves to a real paper

### Honest acknowledgments to include

- No calibrated Bayesian uncertainty (Candebat D1)
- Small sample relative to Candebat's 50K catalogue (D3)
- Single survey (single instrument, single wavelength window)
- Benchmark is X (whatever Phase 0 locked)
- FGK-dominated training set means OB (and possibly M) claims are weak or absent

---

## Phase 7: Submission (week 12)

### Endorsement

- [ ] Identify three to five active astro-ph.IM authors in ML-on-stellar-spectra (Candebat, Sacco, Magrini, Belfiore team; Bailer-Jones; Ting Yuan-Sen; Leung and Bovy; Fabbro)
- [ ] Confirm each has an active arXiv endorsement record
- [ ] Draft a 4-sentence endorsement request: who you are, one-line paper summary, abstract attached, thank you
- [ ] Send to the top 1 choice; wait 5 business days before emailing number 2

### Submission tasks

- [ ] Compile manuscript in arXiv-ready LaTeX (A&A style class is fine for astro-ph.IM)
- [ ] Prepare GitHub repo: code, tests, README, license (MIT or BSD-3)
- [ ] Mint Zenodo DOI for features.npz and trained model
- [ ] Add DOI and GitHub URL to paper
- [ ] Post arXiv preprint
- [ ] Submit to RAA or Open Journal of Astrophysics within one week of arXiv

---

## Kill criteria summary

Any of these trips, stop and reassess with the feedback loop (writing group, senior reader, or me).

- Phase 0: fewer than 2000 spectra in window, or only 2 classes survive
- Phase 2: macro-F1 below 0.5
- Phase 4: no line set shows p ≤ 0.05 AND you are not prepared to rewrite as a shortcut-learning paper
- Any phase: 50% over the time budget without visible progress

Off-ramp: preprocessing-methods paper (candidate B from memory).

---

## Decision log (update as you go)

Keep a single file `decisions.md` in the repo root. Every time a choice is made, log date, decision, reasoning, what would reverse it.

- [ ] Date, window choice, reasoning
- [ ] Date, benchmark choice, reasoning
- [ ] Date, surviving classes, reasoning
- [ ] Date, any deviations from this plan

---

## What NOT to do

- Do not add a second classifier for comparison (scope creep; Candebat's territory)
- Do not attempt T_eff or log g regression (scope creep; Candebat's territory)
- Do not try to match Candebat's uncertainty rigor (you will lose the time budget)
- Do not skip the external benchmark (gap map D2; this is where Li Lin Qiu lost credibility)
- Do not publish without Zenodo DOI and GitHub archive (gap map D5)
- Do not cite the priors as "Liu et al. 2019" (correct authors: Li, Lin and Qiu)
- Do not describe Candebat as using UVES (they use GIRAFFE HR10 and HR21)
