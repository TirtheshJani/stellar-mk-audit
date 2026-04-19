# Claude Code execution plan: stellar-mk-audit

Drop this file at the repo root as `CLAUDE_CODE_EXECUTION.md` (or rename to `CLAUDE.md` to make Claude Code load it automatically on startup). Pair it with `project_plan.md` (the 12-week plan, source of truth for phase gates) and the forthcoming `decisions.md` (runtime decision log).

This document is written for Claude Code as the reader. Imperative voice. Do what the gates say. Do not invent scope.

---

## 1. Mission

Run a solo-authored short paper to arXiv astro-ph.IM, then submit to Research in Astronomy and Astrophysics or Open Journal of Astrophysics. Claim: tree-based stellar classifiers can be audited for physics-awareness via causal masked-line ablation.

You are the execution agent. You are not the author. You do not write the manuscript. You build artifacts, run tests, log decisions, and stop when a gate fails.

---

## 2. Hard rules (non-negotiable)

1. **No em dashes.** Not in code comments, commit messages, docstrings, markdown, or shell output you author. If you catch one in pre-existing text you are editing, remove it.
2. **Do not fabricate.** If a test fails, report it. If astroquery returns no rows, say so. Never synthesize a VizieR response, a metric value, or a file that "should" exist.
3. **Do not exceed scope.** If the plan asks for one classifier, do not add a second. If the plan asks for A/F/G/K, do not add M. Scope creep is the primary failure mode of this paper.
4. **Commit after each passing gate.** Commit message format: `phase{N}: <one-line summary>`. No pushes unless TJ asks.
5. **Log every binding decision to `decisions.md`** with date, decision, reasoning, and what would reverse it.
6. **Kill criteria are real.** If a kill criterion trips, stop. Do not "try one more thing." Report the failure and wait for TJ.
7. **Escalate after 2 failed fix attempts** on the same gate. Do not keep iterating silently.
8. **Use AIR wavelengths throughout.** GES UVES pipeline convention. If you see a VACUUM wavelength anywhere, flag it.

---

## 3. File scope

**You may create or modify:**
- `src/interpret/*`
- `scripts/*`
- `tests/*`
- `pyproject.toml`, `requirements.txt`, `README.md`, `LICENSE`
- `decisions.md`, `artifacts/` (new)
- `data/ges/catalogs/` (cache for VizieR downloads)

**Do not modify:**
- `src/preprocess/*`, `src/fetch/*`, `src/utils/*` (imported from the GONS pipeline, treated as read-only). If a bug is found there, open an issue note in `decisions.md` and add a thin wrapper in `src/interpret/` instead.
- `.git/*`

**Do not create:**
- Parallel classifiers (scope creep)
- T_eff or log g regression heads (Candebat's territory)
- A second paper's worth of infrastructure

---

## 4. Current repo state (audit baseline, 2026-04-18)

Assume this is the starting state. Verify before Phase -1.

- `pyproject.toml` is still the GONS project (wrong name, wrong description, torch as core dep, lightgbm hidden behind `[interpret]` extra).
- `README.md` is one line.
- `decisions.md` does not exist.
- `src/interpret/__init__.py` references `plan-stellar-wild-manatee.md` which is not in the repo.
- `src/interpret/features.py` hardcodes the 4800-6800 A window (Phase 0 Decision 1 silently committed).
- `src/interpret/labels.py` excludes O/B/M at construction (Phase 0 Decision 3 silently committed).
- `src/interpret/labels.py` targets VizieR `J/A+A/666/A121` (Hourihane+ 2023 GES DR5). TJ's plan text cited `J/A+A/692/A228`. The code's choice is defensible; the decision is not logged.
- `src/preprocess/` and `src/fetch/` are full GONS copies including APOGEE and GALAH paths not used here.
- `src/interpret/occlusion.py` uses `CONTINUUM_FILL = 1.0` without asserting the feature matrix is continuum-normalized to that level.
- `src/interpret/classifier.py` single-split only. Plan Phase 2 calls for 5-fold CV. Reconcile in Phase 2.
- Tests exist for labels, features, lines, line_coverage, benchmark, xmatch. Not verified that they pass.
- Scripts exist for build_labels, build_features, train_classifier, run_interpret, ablation, run_benchmark, make_figure. Not verified that they run.

---

## 5. Phase -1: repo health (prerequisites, 1 day)

Nothing else runs until this is green.

### 5.1 Fix `pyproject.toml`

Change:
- `name` → `"stellar-mk-audit"`
- `description` → `"Physics-informed audit of LightGBM stellar MK classifier on Gaia-ESO UVES spectra."`
- `keywords` → `["astronomy", "stellar-classification", "interpretability", "lightgbm", "gaia-eso"]`
- `[project.urls]` → both point to `https://github.com/TirtheshJani/stellar-mk-audit`
- Core `dependencies` → drop `torch`. Keep numpy, pandas, requests, astropy, pyarrow, h5py, pyyaml, matplotlib, tqdm, scipy.
- Promote `lightgbm>=4.0`, `shap>=0.42`, `scikit-learn>=1.3`, `astroquery>=0.4.7` from `[interpret]` extra into core `dependencies`.
- Keep `[dev]`, `[notebooks]`, `[dvc]` extras. Drop `[astronn]` and `[interpret]`.

Mirror the core deps into `requirements.txt`. Drop `torch`, `astroNN`, `dvc`.

### 5.2 Write a real `README.md`

Sections:
1. One-paragraph abstract (copy from `project_plan.md` top).
2. Install block: `pip install -e ".[dev]"`.
3. How to reproduce: `python scripts/build_labels.py ...` etc. through `make_figure.py`.
4. Data provenance: GES DR5 (Hourihane+ 2023), Pickles 1998.
5. License: MIT.

Keep under 200 lines. No badges unless CI is real.

### 5.3 Create `decisions.md`

Seed it with the four decisions the code has already baked in silently. Each gets a dated entry with the reversal condition:

1. **2026-04-18 Window locked to 4800-6800 A (Option A).** Reasoning: UVES U580 native coverage; avoids combining setups. Reversal: Phase 0 coverage probe shows fewer than 2000 spectra survive.
2. **2026-04-18 Label scope locked to A/F/G/K.** Reasoning: window excludes TiO bands needed for M-class audit; UVES sample is FGK-heavy. Reversal: class counts in Phase 0 show A below 50 spectra (then drop A), or window changes.
3. **2026-04-18 Benchmark locked to Pickles 1998 UVKLIB.** Reasoning: independent physics, survives the "tested on the labels that trained it" critique. Reversal: Pickles FITS files unavailable.
4. **2026-04-18 GES params catalog locked to VizieR `J/A+A/666/A121` (Hourihane+ 2023).** Reasoning: DR5 recommended parameters with Teff/logg/[Fe/H]. Reversal: VizieR does not resolve or columns do not match `_VIZIER_ALIASES` in `labels.py`.

### 5.4 Fix `src/interpret/__init__.py`

Remove the reference to `plan-stellar-wild-manatee.md`. Replace with a reference to `project_plan.md` and `decisions.md`.

### 5.5 Verify VizieR resolution (10 min)

```bash
python -c "
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = 5
t = Vizier.get_catalogs('J/A+A/666/A121')
print(f'n_tables={len(t)}')
for i, tbl in enumerate(t):
    print(f'table {i}: {len(tbl)} rows')
    print(f'  columns: {tbl.colnames[:12]}...')
"
```

Gate: catalog resolves, at least one table returned, column names intersect `_VIZIER_ALIASES` for RA, Dec, Teff. If not, stop and escalate.

### 5.6 Run existing tests

```bash
pip install -e ".[dev]"
pytest tests/ -x -v 2>&1 | tee artifacts/pytest_phase_minus_1.log
```

Expected outcome: some tests pass, some may fail because fixtures/data are not set up. Triage:
- If tests reference files under `data/` that do not exist, log which ones and whether they are smoke tests (need synthetic fixtures) or integration tests (need real HDF5).
- Do not invent passing output. Paste the tail of `pytest` verbatim into the phase -1 report.

### 5.7 Trim GONS leftovers (optional but recommended)

`src/preprocess/` and `src/fetch/` contain APOGEE and GALAH code. Either:
- (a) Delete APOGEE/GALAH-only files after confirming nothing in `src/interpret/` imports them, OR
- (b) Leave them and add a line to `README.md` saying "preprocess and fetch include APOGEE/GALAH paths inherited from the upstream GONS pipeline; only UVES paths are exercised here."

Ask TJ before doing (a). Default: do (b). 2 minutes.

### 5.8 Acceptance gate for Phase -1

- `pyproject.toml` name, description, URLs, deps correct
- `README.md` has abstract + install + reproduce + data provenance + license
- `decisions.md` has the four seed entries
- VizieR probe returns a non-empty catalog with expected columns
- `pytest tests/ -x` either passes or has a documented list of expected failures with reasons

Commit: `phase-1: repo health fixes (pyproject, readme, decisions log)`.

---

## 6. Phase 0: pre-flight probes (1 day)

### 6.1 Preconditions

- Phase -1 acceptance met
- Path to the regridded HDF5 is known. Ask TJ if not. Do not guess.

### 6.2 Coverage probe (Decision 1)

```bash
python -c "
from src.interpret.features import coverage_probe
r = coverage_probe(h5_path='<PATH>', wave_min=4800, wave_max=6800)
print(r)
"
```

Gate: `n_covered >= 2000`. If fewer, stop and escalate. Do not quietly lower the threshold.

### 6.3 Class count probe (Decision 3 verification)

```bash
python -c "
from pathlib import Path
from src.interpret.labels import build_labels
df, stats = build_labels(
    h5_path='<PATH>',
    cache_dir=Path('data/ges/catalogs'),
    allow_drop_underfilled=True,
)
print(stats)
print(df['mk_class'].value_counts())
"
```

Gate: at least 3 MK classes with n >= 50. If only 2 survive, stop and escalate.

Log: per-class counts to `decisions.md` under the 2026-04-18 label-scope entry.

### 6.4 Benchmark availability check (Decision 2 verification)

```bash
ls data/pickles/uk*.fits | head -5 || echo "NO PICKLES FITS FOUND"
```

If Pickles FITS not present, add a subtask: "download Pickles UVKLIB to `data/pickles/` before Phase 5." Do not block Phase 1-4 on this, but do log it.

### 6.5 Acceptance gate for Phase 0

- Coverage probe: `n_covered >= 2000` documented in `decisions.md`
- Class count probe: at least 3 surviving classes each with >= 50 spectra, documented in `decisions.md`
- Benchmark availability: known (present or explicitly deferred)

Commit: `phase0: coverage and class probes passed (n=X, classes=Y)`.

---

## 7. Phase 1: data and labels (weeks 1-2)

### 7.1 Preconditions

- Phase 0 acceptance met
- VizieR cache at `data/ges/catalogs/ges_dr5_params.parquet` exists (will be created on first `build_labels` call)

### 7.2 Run

```bash
mkdir -p artifacts
python scripts/build_labels.py \
    --h5-path <PATH> \
    --cache-dir data/ges/catalogs \
    --out artifacts/ges_mk_labels.parquet \
    2>&1 | tee artifacts/build_labels.log

python scripts/build_features.py \
    --h5-path <PATH> \
    --labels artifacts/ges_mk_labels.parquet \
    --out artifacts/features.npz \
    2>&1 | tee artifacts/build_features.log
```

If the scripts have arg names that differ from above, read the script first and adapt. Do not invent args.

### 7.3 Tests

```bash
pytest tests/test_labels.py tests/test_features.py tests/test_lines.py tests/test_line_coverage.py -v
```

Add a test if missing: `tests/test_line_coverage.py` must assert that every `LINE_SETS` entry intersects the feature window `[4800, 6800]` after Phase 0 window lock.

### 7.4 Acceptance gate

- `artifacts/features.npz` exists and loads with expected keys: `X`, `y`, `wave_centers`, `train_idx`, `val_idx`, `test_idx`, `boundary_distance_k`, `dwarf_flag`, `groups`, `median_imputer`
- Shape sanity: `X.shape[0] == len(y)`, `X.shape[1]` close to 1000 (400 native pixels per 2 A after rebin of 5)
- No NaN in `X` after imputation
- Per-class counts in `y` match `decisions.md` Phase 0 numbers within 5 percent (some drop from SNR cut is expected and fine)
- All four listed tests pass

### 7.5 Self-review pass (run these yourself before declaring green)

**Reviewer checks:**
- Determinism: every `random_state` and `seed` in Phase 1 code is set explicitly
- No duplication of utilities from `src/preprocess/`
- Type hints on every public function
- `grep -rn "Liu et al"` in changed files returns nothing (correct authors are Li, Lin and Qiu)

**Physicist checks:**
- `wave_centers` spans 4800-6800 A with gaps consistent with the UVES inter-chip gap at 5769-5834 A OR the imputer has filled them
- Per-class Teff distributions look right: A peaks near 7500-8000 K, F near 6500, G near 5700, K near 4500
- No spectrum with SNR < 20 survived

Commit: `phase1: features and labels built (N=<X>, n_bins=<Y>)`.

---

## 8. Phase 2: baseline classifier (weeks 3-4)

### 8.1 Preconditions

- `artifacts/features.npz` exists
- Labels cover at least 3 MK classes each with >= 100 training samples

### 8.2 Reconcile the CV question

`project_plan.md` Phase 2 calls for 5-fold stratified CV. `src/interpret/classifier.py` is single-split. Pick one and log it to `decisions.md`:

- (a) Add a `train_cv()` function to `classifier.py` that runs 5-fold stratified CV on `train + val` combined, reports mean ± std of macro-F1, then fits the final model on the full train+val for test evaluation. More defensible, 2 hours of work.
- (b) Stay single-split. Justify in `decisions.md` with "fixed split chosen for reproducibility and because bootstrap CIs in Phase 4 provide the uncertainty quantification." Less defensible, 5 minutes.

Default to (a). If TJ has not responded in the conversation about this, do (a) unless it blocks for more than 4 hours.

### 8.3 Run

```bash
python scripts/train_classifier.py \
    --features artifacts/features.npz \
    --model-out artifacts/lgbm_mk.pkl \
    --metrics-out artifacts/metrics.json \
    2>&1 | tee artifacts/train_classifier.log
```

### 8.4 Acceptance gate

- `metrics.json` exists
- `macro_f1 >= 0.7`
- Per-class `recall >= 0.6` for every class with >= 100 training samples
- Confusion matrix: off-diagonal mass concentrated on adjacent MK classes (F confuses with G, G with F or K; A does not confuse with K)

### 8.5 Kill criterion

If `macro_f1 < 0.5`: stop. Do not fine-tune hyperparameters beyond one round. Report to TJ. Pivot options are documented in `project_plan.md` Phase 2 kill criteria.

If a single class has `recall < 0.4`: flag it in `decisions.md`, note that Phase 4 ablation results for that class will be uninterpretable, continue.

### 8.6 Self-review pass

**Physicist checks:**
- Confusion matrix pattern: expect A ↔ F confusion, F ↔ G confusion, G ↔ K confusion, minimal A ↔ K confusion. If A ↔ K is significant, something is wrong with features or labels.
- On the boundary-filtered subset (|boundary_distance_k| > 200 K), accuracy should be meaningfully higher than on the full test set. If not, the classifier is memorizing something other than temperature.

Commit: `phase2: baseline classifier (macro-f1=<X>, per-class recall ok)`.

---

## 9. Phase 3: interpretability triangulation (weeks 5-6)

### 9.1 Preconditions

- `artifacts/lgbm_mk.pkl` exists
- Macro-F1 >= 0.7

### 9.2 Run

```bash
python scripts/run_interpret.py \
    --model artifacts/lgbm_mk.pkl \
    --features artifacts/features.npz \
    --out-dir artifacts/interpret \
    2>&1 | tee artifacts/run_interpret.log
```

Expected outputs:
- `artifacts/interpret/perm_importance.npz`
- `artifacts/interpret/shap_values.npz`
- `artifacts/interpret/occlusion_trace.npz`
- `artifacts/interpret/triangulation_report.json`

### 9.3 Acceptance gate

- All three methods produce non-degenerate outputs (non-constant importance traces, not all zeros)
- Pairwise Jaccard(top-20 perm, top-20 SHAP) >= 0.5
- Per-class importance figures show class-specific wavelength preferences (A peaks near H Balmer, G/K peaks near Mg b and Na D)

### 9.4 Watchouts

- If SHAP takes longer than 2 hours per class: reduce `shap_subsample` or skip classes with n < 200 for SHAP and document in `decisions.md`
- Do not report Gini/impurity importance as a primary result. Li, Lin and Qiu rejected it, we reject it.

### 9.5 Self-review pass

**Physicist checks:**
- Top-10 perm importance bins for class A include bins within 2 A of 4861 (Hβ) or 6563 (Hα)
- Top-10 perm importance bins for class G or K include bins within 2 A of 5169-5184 (Mg b) or 5890-5896 (Na D)
- If these physics predictions fail, do not "tune until they pass." Log the failure. It is an interesting negative result and changes the paper's framing (shortcut learning).

Commit: `phase3: triangulation (jaccard perm-shap=<X>)`.

---

## 10. Phase 4: causal ablation + line matching (weeks 7-8)

This is the headline. Extra care.

### 10.1 Preconditions

- Phase 3 outputs exist
- Feature matrix median is close to 1.0 (continuum-normalized). Verify before running ablation:

```bash
python -c "
import numpy as np
d = np.load('artifacts/features.npz')
print(f'X median: {np.nanmedian(d[\"X\"]):.3f}')
print(f'X 1st percentile: {np.nanpercentile(d[\"X\"], 1):.3f}')
print(f'X 99th percentile: {np.nanpercentile(d[\"X\"], 99):.3f}')
"
```

Gate: `0.5 < median < 1.5`. If not, `CONTINUUM_FILL = 1.0` in `occlusion.py` is wrong and ablation will produce garbage. Stop and escalate.

### 10.2 Run

```bash
python scripts/ablation.py \
    --model artifacts/lgbm_mk.pkl \
    --features artifacts/features.npz \
    --out-dir artifacts/ablation \
    --n-bootstrap 500 \
    --n-random-controls 100 \
    --seed 42 \
    2>&1 | tee artifacts/ablation.log

python -c "
from src.interpret.line_match import sweep_tolerances, save_sweep
import numpy as np
d = np.load('artifacts/interpret/perm_importance.npz')
results = sweep_tolerances(d['importance_mean'], d['wave_centers'])
save_sweep('artifacts/ablation/line_match.csv', 'artifacts/ablation/line_match.json', results)
"
```

### 10.3 Acceptance gate (falsifiable physics claim)

For at least **two** of these three pairings:
- `line_set=H_balmer`, `mk_class=A`
- `line_set=Mg_b`, `mk_class=G` or `mk_class=K`
- `line_set=Na_D`, `mk_class=K`

Both must hold:
- `p_value_vs_random <= 0.01`
- `delta_acc_ci_low > 0` (accuracy actually drops when masked, with CI excluding zero) — careful: `delta_acc` is negative when masking hurts, so the condition is really `delta_acc_ci_high < 0`. Confirm the sign convention in `masked_line_ablation` before asserting.

Line-matching precision at 5 A tolerance >= 0.5 against `MK_LINES_in_window`.

### 10.4 Watchouts

- `occlusion.py._sample_random_windows` can return `[]` if the forbidden mask (all line bins) is too restrictive. The resulting `null_deltas` array can be short or empty. If fewer than 50 of the 100 random controls succeed, log a warning and document in the results. Do not silently proceed with 10 controls.
- The sign convention: `delta_acc_mean` is `masked - baseline`. A good line set has `delta_acc < 0` (masking hurts). The p-value is `mean(null_deltas <= delta)`, so small p means the line set is MORE damaging than random. Confirm this matches the test assertion before running.

### 10.5 Kill criterion / pivot

If **no** line set shows p <= 0.05 against random null for any class:

The model is not physics-aware as assumed. **This is still publishable** but the paper rewrites from "here is how to audit" to "audit reveals shortcut learning in tree-based stellar classifiers." Budget an extra 2 weeks, flag to TJ, continue to Phase 5.

If line-matching precision at 10 A tolerance < 0.3: keep the line-matching contribution in the paper but de-emphasize. Lead with the ablation.

### 10.6 Self-review pass

**Physicist checks:**
- Masked-line ablation directionality: expected signs are `(H_balmer, A) negative`, `(Mg_b, G) negative`, `(Mg_b, K) negative`, `(Na_D, K) negative`. Any sign flip is a red flag worth reporting.
- Random-null