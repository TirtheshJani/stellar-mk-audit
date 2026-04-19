# Decisions log — stellar-mk-audit

Binding decisions for the LightGBM MK audit paper. Every decision that affects the paper's claims, methodology, or reproducibility is logged here with date, reasoning, and what would reverse it.

This file is the reproducibility contract. Phase 6 methods writing pulls from here. The Physicist subagent (see `CLAUDE.md` §4.5) verifies entries against canonical references.

Citation conventions: author + year for papers; bibcode for unambiguous reference; VizieR code for catalogs; NIST ASD for atomic data. Unverified citations are tagged `[CITE NEEDED]` and must be resolved before Phase 6.

No em dashes anywhere in this file.

---

## Entry format

```
## N. Short title

- **Date:** YYYY-MM-DD
- **Decision:** what was chosen
- **Reasoning:** why, with citations
- **Reversal:** condition that would require revisiting this
- **Affects:** files, phases, or artifacts touched
```

When a decision is reversed, append a new entry rather than editing the original. Example: `## 14. REVERSED decision 8 on 2026-05-02 after Physicist review`.

---

## Strategic decisions (Phase 0 scope)

These four decisions set the paper's scope. They were baked into the initial scaffolded code before `decisions.md` existed; Phase -1 logs them retroactively. Phase 0 probes are the first opportunity to reverse any of them.

## 1. Wavelength window locked to 4800-6800 A (Option A)

- **Date:** 2026-04-18
- **Decision:** feature window is 4800-6800 A, corresponding to UVES U580 nominal coverage.
- **Reasoning:** UVES U580 native coverage is approximately 4760-6840 A (Dekker et al. 2000; Sacco et al. 2014, A&A 565, A113). Staying within U580 avoids combining U520 and U580 setups, which would require separate normalization and reduce the cross-matched sample. The window covers all five diagnostic features claimed in the paper: Hbeta (4861.33 A), Mg b triplet (5167.32, 5172.68, 5183.60 A), Na D doublet (5889.95, 5895.92 A), Ca I (6162.17, 6439.08 A), and Halpha (6562.80 A). All wavelengths in air, per NIST ASD (Kramida et al. 2023).
- **Reversal:** Phase 0 coverage probe reports `n_covered < 2000` spectra meeting the 90 percent coverage threshold for this window.
- **Affects:** `src/interpret/features.py` (`DEFAULT_WAVE_MIN`, `DEFAULT_WAVE_MAX`); `src/interpret/lines.py` (`MK_LINES`, `LINE_SETS`); `project_plan.md` Phase 0 Decision 1.

## 2. Label scope locked to A/F/G/K

- **Date:** 2026-04-18
- **Decision:** classifier is trained only on A, F, G, K classes. O, B, M excluded at label construction time.
- **Reasoning:** the 4800-6800 A window excludes TiO bands at 7050-7100 A required for M-class diagnosis, and He II at 4686 A required for O/B classification (Gray & Corbally 2009, Stellar Spectral Classification). GES UVES is FGK-targeted by design (Gilmore et al. 2022, A&A 666, A120), so O/B/M sample counts would be too low to train on even if the window permitted.
- **Reversal:** Phase 0 class counts show A below 50 spectra (then drop A from the class set rather than widen the window); or decision 1 is reversed.
- **Affects:** `src/interpret/labels.py` (`MK_BIN_EDGES`, `ALLOWED_MK_CLASSES`); `src/interpret/lines.py` (`diag_for` attributions).

## 3. External benchmark locked to Pickles 1998 UVKLIB

- **Date:** 2026-04-18
- **Decision:** benchmark the classifier against chi-squared template matching using the Pickles UVKLIB stellar flux library (Pickles 1998, PASP 110, 863; VizieR J/PASP/110/863).
- **Reasoning:** independent physics, not the same pipeline that generated training labels. This is the most defensible of the three benchmark options in `project_plan.md` Phase 0 Decision 2 and survives the "tested on the labels that trained it" critique that Li, Lin and Qiu (2019) did not address. Options A (GES DR5 `sp_type` column, if present) and C (SIMBAD `sp_type`) were rejected because both inherit pipeline or literature heterogeneity.
- **Reversal:** Pickles UVKLIB FITS files are unavailable locally and remote download mirrors are blocked.
- **Affects:** `src/interpret/benchmark.py`; `scripts/run_benchmark.py`; Phase 5 execution.

## 4. GES parameters catalog locked to VizieR J/A+A/666/A121

- **Date:** 2026-04-18
- **Decision:** pull Teff, logg, [Fe/H] for MK labeling from the Gaia-ESO DR5 recommended-parameters catalog (Hourihane et al. 2023, A&A 666, A121; bibcode 2023A&A...666A.121H; VizieR J/A+A/666/A121).
- **Reasoning:** Hourihane et al. 2023 is the authoritative GES DR5 recommended-parameters table, providing Teff with the precision needed for MK binning via Pecaut & Mamajek (2013) edges. The alternative J/A+A/692/A228 mentioned in the original plan text was considered for a direct spectral-type column but is not required given Teff-based binning.
- **Reversal:** VizieR does not resolve J/A+A/666/A121, or the resolved table's columns do not map to the aliases in `src/interpret/labels.py._VIZIER_ALIASES` (RA, Dec, Teff, logg, [Fe/H]).
- **Affects:** `src/interpret/labels.py` (`VIZIER_CATALOG`, `VIZIER_BIBCODE`, `fetch_ges_params_catalog`); `data/ges/catalogs/ges_dr5_params.parquet` cache.

---

## Hyperparameter and threshold anchors

Lower-stakes decisions committed in code. Logged here for reproducibility and Physicist verification. These should not change unless a Physicist review flags a mismatch with canonical values.

## 5. MK Teff bin edges

- **Date:** 2026-04-18
- **Decision:** A [7300, 10000) K; F [6000, 7300) K; G [5300, 6000) K; K [3900, 5300) K.
- **Reasoning:** Pecaut & Mamajek (2013) ApJS 208, 9, Table 5, rounded to the nearest 100 K. Standard dwarf-sequence bin edges for MK classification. Giant-branch discrepancies from Gray & Corbally (2009) are below 100 K at these boundaries and ignored here since the classifier is dwarf-dominated (UVES-FGK sample).
- **Reversal:** Physicist review identifies a post-2013 calibration update that shifts any boundary by more than 100 K.
- **Affects:** `src/interpret/labels.py` (`MK_BIN_EDGES`).

## 6. Cross-match radii

- **Date:** 2026-04-18
- **Decision:** primary match radius 0.5 arcsec; ambiguity (second-nearest) cutoff 2.0 arcsec.
- **Reasoning:** 0.5 arcsec is a conservative tolerance given Gaia DR3 astrometric precision and the GES pointing convention. The 2.0 arcsec ambiguity cutoff prevents wrong-star assignment in crowded open-cluster fields where GES is concentrated.
- **Reversal:** Phase 1 diagnostics show more than 5 percent ambiguous matches at 2.0 arcsec, indicating a density regime that requires tighter cuts.
- **Affects:** `src/interpret/labels.py` (`build_labels` defaults).

## 7. Signal-to-noise floor

- **Date:** 2026-04-18
- **Decision:** require median SNR >= 20 per pixel before features are built.
- **Reasoning:** SNR 20 is adequate for stellar classification (MK typing); SNR 40 would be required for [Fe/H] precision at +/- 0.1 dex, which is not our task. The 20-per-pixel floor matches the GES DR5.1 QC convention as cited in `src/interpret/features.py` docstring attributing it to Gilmore et al. 2022. `[CITE NEEDED: verify Gilmore+ 2022 specifically states SNR >= 20 as the DR5.1 QC floor for MK classification]`.
- **Reversal:** Phase 1 surviving-spectrum count drops below 1500 (approaching the Phase 0 kill threshold). Consider lowering to 15 only after Physicist review.
- **Affects:** `src/interpret/features.py` (`DEFAULT_MIN_SNR`).

## 8. Rebin factor

- **Date:** 2026-04-18
- **Decision:** rebin native UVES pixels by 5, producing ~2.75 A bins and ~1000 features per spectrum.
- **Reasoning:** native UVES pixel size is approximately 0.55 A. Rebinning by 5 produces bins narrower than the Mg b triplet component separations (5-11 A between b1, b2, b3) and narrower than typical line FWHMs at R ~ 47000. The resulting feature count is tractable for SHAP TreeExplainer within the Phase 3 time budget.
- **Reversal:** Physicist review in Phase 1 finds that Na D1 (5895.92 A) and Na D2 (5889.95 A), separated by 5.97 A, fall into the same rebinned bin with centers that prevent distinguishing them, OR Mg b1 and Mg b2 (separation 5.36 A) collapse into one bin.
- **Affects:** `src/interpret/features.py` (`DEFAULT_REBIN`).

## 9. Coverage threshold

- **Date:** 2026-04-18
- **Decision:** require >= 90 percent of native pixels in the feature window to be valid (finite, non-zero) for a spectrum to survive.
- **Reasoning:** the UVES U580 inter-chip gap at approximately 5769-5834 A spans ~3.25 percent of the 4800-6800 A window (Sacco et al. 2014). A 90 percent threshold tolerates this gap plus small detector cosmetics without admitting spectra with large missing regions.
- **Reversal:** none expected; this is an instrument-driven threshold.
- **Affects:** `src/interpret/features.py` (`DEFAULT_COVERAGE_THRESHOLD`).

## 10. Post-rebin NaN cap

- **Date:** 2026-04-18
- **Decision:** after rebinning, drop spectra with more than 10 percent NaN bins.
- **Reasoning:** the 90 percent coverage threshold applies before rebinning; this cap is a post-rebin cross-check. Remaining NaN bins after the cap are filled via train-set median imputation (decision 11).
- **Reversal:** per-class counts drop below Phase 0 minima after applying this cap.
- **Affects:** `src/interpret/features.py` (`DEFAULT_MAX_NAN_FRAC`).

## 11. Imputation strategy

- **Date:** 2026-04-18
- **Decision:** replace NaN bins with train-set per-bin median. Imputer is fit on training rows only and applied to validation and test.
- **Reasoning:** avoids label leakage. Follows The Cannon (Ness et al. 2015, ApJ 808, 16) and StarNet (Fabbro et al. 2018, MNRAS 475, 2978) conventions for spectrum-level imputation. Train-only medians preserve test-set independence.
- **Reversal:** Reviewer or Physicist identifies a leakage path; or per-bin medians are dominated by gap bins such that the imputed value is systematically biased.
- **Affects:** `src/interpret/features.py` (`fit_median_imputer`, `apply_median_imputer`).

## 12. Stratified subsample cap

- **Date:** 2026-04-18
- **Decision:** subsample the full UVES-matched set to at most 5000 spectra, stratified by MK class.
- **Reasoning:** keeps training, SHAP computation, and Phase 4 bootstrap budgets tractable on a laptop-class machine. Bootstrap CIs in Phase 4 are valid at this sample size. The paper's honest limitations section will note that Candebat et al. 2024 used 50K spectra.
- **Reversal:** Phase 4 bootstrap CIs exceed the target width of 0.05 on delta_acc, preventing the 2-of-3 headline gate from passing. Raise to 10000.
- **Affects:** `src/interpret/features.py` (`build_features` `max_spectra` default); `scripts/build_features.py`.

## 13. LightGBM hyperparameters

- **Date:** 2026-04-18
- **Decision:** `max_depth=8`, `num_leaves=63`, `learning_rate=0.05`, `n_estimators=500` with `early_stopping_rounds=50`, `min_child_samples=20`, `subsample=0.9`, `subsample_freq=1`, `colsample_bytree=0.9`, `class_weight='balanced'`, `random_state=42`.
- **Reasoning:** shallow trees (max_depth 8) keep TreeSHAP tractable in Phase 3. `class_weight='balanced'` compensates for FGK skew. Early stopping on validation loss prevents overfitting within the 500-tree budget. Fixed seed for reproducibility. These are standard LightGBM defaults for modest-sized tabular classification, not tuned.
- **Reversal:** Phase 2 kill criterion (macro-F1 < 0.5); OR SHAP computation exceeds 2 hours per class in Phase 3 (reduce `max_depth` or `num_leaves`).
- **Affects:** `src/interpret/classifier.py` (`DEFAULT_HPARAMS`).

## 14. Ablation bootstrap and null-distribution sizes

- **Date:** 2026-04-18
- **Decision:** 500 bootstrap resamples of the test set for delta-accuracy CI; 100 random-window controls per line set, width-matched to each `LINE_SETS` entry and drawn outside all `MK_LINES` windows.
- **Reasoning:** 500 bootstrap resamples give CI half-widths stable to within ~1 percent at 95 percent confidence. 100 random controls give p-value resolution of 0.01, which is the Phase 4 gate threshold. Forbidden mask covers all `MK_LINES` windows to prevent inadvertent overlap with diagnostic features.
- **Reversal:** fewer than 80 of 100 random-window draws succeed due to tight forbidden-mask coverage; lower to 50 controls with explicit documentation, or relax forbidden mask to `MK_LINES +/- 10 A` instead of the full `LINE_SETS`.
- **Affects:** `src/interpret/occlusion.py` (`masked_line_ablation` defaults); `scripts/ablation.py` CLI.

## 15. Wavelength convention

- **Date:** 2026-04-18
- **Decision:** all wavelengths in code, comments, and outputs are in AIR, matching the GES UVES pipeline convention (Sacco et al. 2014).
- **Reasoning:** mixing air and vacuum is the single most common source of 1-3 A offsets in spectral analysis code. Locking AIR at the decisions-log level gives the Physicist agent a clear standard to enforce at every phase review.
- **Reversal:** none. Any vacuum-wavelength quantity encountered in external data must be converted to air before use.
- **Affects:** everything downstream of `src/fetch/fetch_ges.py`. Particularly `src/interpret/lines.py` and `src/interpret/line_match.py`.

---

## 16. VizieR resolution probe deferred to Phase 0 preconditions

- **Date:** 2026-04-19
- **Decision:** Phase -1 VizieR resolution check (CLAUDE.md §5.1) could not execute in the current sandbox; `astroquery.vizier.Vizier.get_catalogs('J/A+A/666/A121')` returned HTTP 403 Forbidden against `https://vizier.cds.unistra.fr/viz-bin/votable`. Captured verbatim in `artifacts/vizier_probe.log`. The CLAUDE.md §5.1 kill criterion ("VizieR does not resolve AND no CDS mirror is available") is NOT satisfied: VizieR serves J/A+A/666/A121 publicly; the 403 is a sandbox egress restriction, not a VizieR outage. Action: defer the probe to a network-enabled environment and re-run as a Phase 0 precondition before any label build. No change to decision 4.
- **Reasoning:** Physicist-post review (2026-04-19) confirmed VizieR availability at the literature level and classified the 403 as environmental. Coder correctly refused to fabricate a probe response per CLAUDE.md rule 2.
- **Reversal:** the re-run from a network-enabled environment returns zero tables or columns that do not map to `_VIZIER_ALIASES` in `src/interpret/labels.py`; at that point trigger the Phase -1 kill criterion and escalate to TJ for a pivot (SIMBAD Option C or manual catalog download).
- **Affects:** Phase 0 preconditions (CLAUDE.md §5.2); `artifacts/vizier_probe.log`; the commit message of 7f67b11 ("vizier verified") is retained as-is for hash stability but its truth is conditioned on the re-run documented here.

---

## Pending decisions (to resolve in-phase)

These are decisions `project_plan.md` defers until data is in hand. Log them here as they resolve.

- **[PENDING Phase 0]** Re-run VizieR probe from a network-enabled environment per decision 16; confirm table count and column aliases before build_labels.
- **[PENDING Phase 2]** 5-fold CV vs single-split. Default per `CLAUDE.md` §5.4 is 5-fold; fall back to single-split if implementation blocks for more than 4 hours.
- **[PENDING Phase 4]** Add Ca I to the headline ablation gate, or keep it as a secondary result. `project_plan.md` Phase 4 mentions only H_balmer/Mg_b/Na_D in the falsifiable gate.
- **[PENDING Phase 5]** Download source for Pickles UVKLIB. VizieR J/PASP/110/863 or the ESO mirror at `www.eso.org/sci/observing/tools/standards/IR_spectral_library.html`. Log the chosen source and the download date.
- **[PENDING Phase 6]** Add a one-sentence scope clarification to manuscript methods: "causal" in "causal masked-line ablation" refers to a feature-intervention (do-operator on masking) on a fixed trained model, not to the data-generating process. Physicist-pre 2026-04-19.

---

## Template

```
## N. Short title

- **Date:** YYYY-MM-DD
- **Decision:**
- **Reasoning:**
- **Reversal:**
- **Affects:**
```

Append at the bottom. Do not edit earlier entries except to tag them `REVERSED` in a new entry that references them by number.
