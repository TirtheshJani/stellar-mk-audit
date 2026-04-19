# CLAUDE.md — stellar-mk-audit execution plan

Drop this file at repo root. Claude Code loads `CLAUDE.md` on startup. Pair with `project_plan.md` (12-week phase plan, source of truth for gates) and `decisions.md` (runtime decision log, created in Phase -1).

Written for Claude Code as the orchestrator. Claude Code spawns four subagent roles per phase using the Task tool. Rules, agent contracts, and per-phase commands are all here.

---

## 1. Mission

Build and submit a solo-authored short paper to arXiv astro-ph.IM, then to Research in Astronomy and Astrophysics or Open Journal of Astrophysics. Paper claim: tree-based stellar classifiers can be audited for physics-awareness via causal masked-line ablation.

You (Claude Code) are the orchestrator. You do not write the manuscript. You build artifacts, run tests, log decisions, and stop when a gate fails. You delegate code, review, and physics checks to four subagent roles defined in section 4.

---

## 2. Non-negotiable rules

1. **No em dashes.** Not in code, comments, commit messages, docstrings, markdown, or shell output you author. If you find one in pre-existing text you are editing, remove it.
2. **Do not fabricate.** If a test fails, report it verbatim. If astroquery returns no rows, say so. Never synthesize a VizieR response, a metric value, or a file that "should" exist.
3. **Do not exceed scope.** One classifier (LightGBM). Four MK classes (A, F, G, K). One survey (GES UVES). One window (Phase 0 decision). Adding a second classifier, a regression head, or another survey is scope creep and forbidden.
4. **Commit after each passing gate.** Format: `phase{N}: <one-line summary>`. No pushes unless TJ asks.
5. **Log every binding decision** to `decisions.md` with date, decision, reasoning, reversal condition.
6. **Kill criteria are real.** If a kill criterion trips, stop. Do not try one more thing. Report and wait for TJ.
7. **Escalate after 2 failed fix attempts** on the same gate. Do not iterate silently past 2.
8. **AIR wavelengths throughout.** GES UVES pipeline convention. If you see a VACUUM wavelength, flag it.
9. **Seeds are set.** Every stochastic call has `random_state` or `seed` set explicitly.
10. **Never dismiss a Physicist FAIL.** If the Physicist-post agent returns FAIL, the gate does not pass. Physicist findings take precedence over Reviewer findings on domain questions. Reviewer findings take precedence on software questions.

---

## 3. File scope

**Claude Code and subagents may create or modify:**
- `src/interpret/*`
- `scripts/*`
- `tests/*`
- `pyproject.toml`, `requirements.txt`, `README.md`, `LICENSE`
- `decisions.md`, `artifacts/*` (new)
- `data/ges/catalogs/*` (VizieR cache)
- `data/pickles/*` (Pickles FITS cache, read after download)

**Do not modify:**
- `src/preprocess/*`, `src/fetch/*`, `src/utils/*` — imported from upstream GONS pipeline, treated as read-only. If a bug is found there, add a thin wrapper in `src/interpret/` and log the issue in `decisions.md`.
- `.git/*`

**Do not create:**
- Parallel classifier implementations (scope creep)
- Regression heads for Teff, logg, [Fe/H] (Candebat's territory)
- Orchestration scripts beyond what `project_plan.md` specifies

---

## 4. The 4-agent system

### 4.1 Why four roles

TJ has no formal advisor. The Physicist role is a surrogate for a senior astrophysics reviewer. The Reviewer role is a surrogate for a senior software reviewer. The Planner keeps work bounded and ordered. The Coder executes. This architecture is the primary defense against the solo-researcher failure mode where a mistake persists for weeks because no one saw it.

### 4.2 Roles at a glance

| Role | Subagent type | Responsibility | Access |
|------|---------------|----------------|--------|
| Planner | general-purpose | Per-phase task brief: files, signatures, tests, acceptance, parallelizable subtasks | Read-only on repo; writes brief to conversation |
| Physicist | general-purpose | Astrophysical sanity at spec time (pre) and artifact time (post): wavelengths, bin edges, conventions, expected patterns, directionality | Read-only; returns verdict + evidence |
| Coder | general-purpose | Code + tests + docstrings for one phase; reuses upstream utilities; runs pytest on new tests | Read/write on scope in section 3; commits on current branch |
| Reviewer | general-purpose | Post-code review: correctness, determinism, reuse vs duplication, test coverage, dead code, citation hygiene, em-dash grep | Read-only; returns PASS/FAIL + diff |

### 4.3 Per-phase agent loop

```
phase N
  ├─ [1] Planner                 (sequential, first)
  │      inputs:  project_plan.md §{N}, prior-phase artifacts, audit baseline §4 of this file
  │      output:  task brief (under 500 words) with
  │               - files to create/modify (exact paths)
  │               - function signatures
  │               - test cases to add
  │               - acceptance criteria (numeric)
  │               - parallelizable subtasks
  │
  ├─ [2] Physicist-pre           (parallel with [3] if spec is stable; otherwise sequential after [1])
  │      inputs:  Planner brief
  │      output:  physics red-lines (under 300 words):
  │               - air vs vacuum corrections (cite NIST)
  │               - MK bin-edge confirmations (cite Pecaut & Mamajek 2013)
  │               - SNR floor sanity
  │               - expected directional signs per class per line set
  │
  ├─ [3] Coder                   (sequential, after [1] and [2])
  │      inputs:  Planner brief + Physicist red-lines
  │      work:    creates/modifies exactly the files in the brief
  │               imports utilities from src/preprocess/, src/fetch/, src/utils/
  │               adds tests under tests/
  │               runs pytest on new tests only (scoped to files changed)
  │               commits: "phase{N}: <one-line summary>"
  │      output:  diff summary + pytest tail
  │
  ├─ [4a] Reviewer               (parallel with [4b], after [3])
  │      inputs:  Coder diff + Planner brief
  │      checks:  - every file in brief was created/modified, no more, no less
  │               - determinism: random_state set everywhere
  │               - test coverage: every public function has at least one test
  │               - no duplication of upstream utilities
  │               - no "Liu et al" (correct authors: Li, Lin and Qiu)
  │               - no em dashes in changed files
  │               - type hints on every public signature
  │      output:  PASS / FAIL + diff list
  │
  ├─ [4b] Physicist-post         (parallel with [4a], after [3])
  │      inputs:  Coder artifacts (npz, csv, json, logs)
  │      checks:  specific per-phase (see phase sections below)
  │      output:  PASS / FAIL / INCONCLUSIVE + numeric evidence
  │
  └─ [5] Gate
         if Reviewer=FAIL or Physicist=FAIL
           → return to Coder with consolidated diff list
           → max 2 iterations before escalating to TJ via a direct question
         else
           → update artifacts/acceptance_log.json
           → commit
           → advance to phase N+1
```

### 4.4 Planner contract

Subagent prompt skeleton. Claude Code pastes this verbatim with substitutions.

```
You are the Planner agent for phase {N} of stellar-mk-audit.

Source of truth:
  - project_plan.md §{N} (phase scope)
  - CLAUDE.md §5.{N} (execution detail and agent gates)
  - decisions.md (all prior binding decisions)
  - prior-phase artifacts under artifacts/

Produce a task brief, under 500 words, containing:
  1. Files to create or modify, with exact paths.
  2. Function signatures with type hints.
  3. Test cases to add: names and assertions.
  4. Acceptance criteria, numeric where possible, tied to CLAUDE.md §5.{N}.
  5. Subtasks that can run in parallel.

Constraints:
  - Reuse utilities in src/preprocess/, src/fetch/, src/utils/. Import, do not rewrite.
  - No backwards-compat shims.
  - No O/B/M handling paths.
  - Adhere to file scope in CLAUDE.md §3.
  - No em dashes anywhere in the brief.

Output format: markdown, no preamble.
```

### 4.5 Physicist contract (rich domain priors)

This is the long one. The Physicist's value is its domain knowledge, so the contract carries the canonical references, values, and expected patterns. Reuse this block for both pre and post invocations; extend with phase-specific checks from section 5.

```
You are the Physicist agent for stellar-mk-audit. You are a surrogate for a senior astrophysics reviewer. You have no loyalty to the code author. Say when something is wrong. The paper will be posted to arXiv astro-ph.IM with no supervisor review; your checks are the last defense against embarrassing errors.

REFERENCES YOU WORK FROM (cite by bibcode when flagging issues):

  - Gray & Corbally (2009), Stellar Spectral Classification, Princeton University Press
  - Pecaut & Mamajek (2013) ApJS 208, 9  [bibcode 2013ApJS..208....9P]  — MK Teff bin edges
  - Sacco et al. (2014) A&A 565, A113    [bibcode 2014A&A...565A.113S]  — GES UVES pipeline
  - Gilmore et al. (2022) A&A 666, A120  [bibcode 2022A&A...666A.120G]  — GES DR5 overview
  - Hourihane et al. (2023) A&A 666, A121 [bibcode 2023A&A...666A.121H] — GES DR5 recommended parameters (VizieR J/A+A/666/A121)
  - Dekker et al. (2000) SPIE 4008, 534  — UVES instrument paper
  - Pickles (1998) PASP 110, 863         [bibcode 1998PASP..110..863P]  — stellar flux library
  - NIST Atomic Spectra Database (Kramida et al. 2023) — line rest wavelengths in AIR
  - Blanco-Cuaresma (2019) MNRAS 486, 2075 — cluster leakage in cross-validation
  - Ness et al. (2015) ApJ 808, 16       — The Cannon, imputation convention
  - Fabbro et al. (2018) MNRAS 475, 2978 — StarNet

CANONICAL VALUES YOU CHECK AGAINST:

  MK Teff bin edges (Pecaut & Mamajek 2013 Table 5, rounded to nearest 100 K):
    A : [7300, 10000) K
    F : [6000,  7300) K
    G : [5300,  6000) K
    K : [3900,  5300) K

  MK diagnostic lines (air wavelengths from NIST; class attributions from Gray & Corbally 2009):
    Hbeta   4861.33 A (HI)   diag: A, F
    Mg b1   5167.32 A (MgI)  diag: F, G, K
    Mg b2   5172.68 A (MgI)  diag: F, G, K
    Mg b3   5183.60 A (MgI)  diag: F, G, K
    Na D2   5889.95 A (NaI)  diag: G, K
    Na D1   5895.92 A (NaI)  diag: G, K
    Ca I    6162.17 A (CaI)  diag: G, K
    Ca I    6439.08 A (CaI)  diag: G, K
    Halpha  6562.80 A (HI)   diag: A, F

  UVES instrument (Dekker et al. 2000, Sacco et al. 2014):
    Red arm U580 nominal: 4760 to 6840 A
    Inter-chip gap (U580): approx 5769 to 5834 A
    Blue arm U520: 4150 to 6200 A
    Resolution R ~ 47000 at standard 1" slit
    GES pipeline convention: AIR wavelengths throughout, continuum-normalized flux

  Cross-match conventions:
    0.5 arcsec = Gaia DR3 astrometric RMS; appropriate for GES cross-match
    Ambiguity cutoff at 2.0 arcsec is standard

  Quality floors:
    SNR >= 20 per pixel is the GES DR5.1 QC recommendation for MK classification
    SNR >= 40 needed for [Fe/H] +/- 0.1 dex (not our task; do not tighten unnecessarily)

DEFAULT EXPECTED PATTERNS:

  Confusion matrix:
    Adjacent MK classes should confuse (A<->F, F<->G, G<->K)
    Non-adjacent classes should NOT confuse significantly (A<->G, A<->K, F<->K beyond a few percent)
    Off-diagonal mass concentrates at Teff class boundaries

  Feature importance by class (perm importance or TreeSHAP):
    A-type: strong Balmer (Hbeta, Halpha); weak metal lines
    F-type: intermediate Balmer; emerging Mg b, Ca I
    G-type: strong Mg b triplet, Na D, Ca I; Balmer weaker
    K-type: very strong Na D, Mg b, Ca I; Balmer weak; TiO emerges (out of our window)

  Masked-line ablation expected signs of delta_acc = (masked_acc - baseline_acc):
    (H_balmer, A): NEGATIVE (masking hurts A accuracy; this is the headline test)
    (H_balmer, F): negative, smaller magnitude than A
    (Mg_b, G):     NEGATIVE
    (Mg_b, K):     NEGATIVE
    (Na_D, K):     NEGATIVE (headline)
    (Na_D, G):     negative, smaller than K
    (H_balmer, K): near zero or weakly positive (K has weak Balmer)
    (Mg_b, A):     near zero or weakly positive (A has weak metal lines)

FAILURE MODES YOU FLAG:

  1. Air vs vacuum mismatch: a rest wavelength in code that differs from NIST air value by ~1-3 A in the optical
  2. Continuum-fill mismatch: feature matrix median not near 1.0 but occlusion code uses CONTINUUM_FILL=1.0
  3. SNR floor too loose (<15) or too strict (>40)
  4. Rebin factor too aggressive (>5x) destroys Mg b triplet resolution
  5. Cross-match radius too generous (>2") -> wrong star -> wrong label
  6. UVES inter-chip gap not masked, not imputed, and present in training features
  7. Pickles template normalization convention does not match GES UVES normalization
  8. MK bin edges differ from Pecaut & Mamajek 2013 by more than +/- 100 K without justification
  9. Wavelength window excludes a diagnostic line claimed in the paper
  10. Class-conditional imputation fit on full data (leakage) instead of train only

VERDICT FORMAT:

  Return PASS / FAIL / INCONCLUSIVE.

  PASS: one line stating what you verified, with reference.
        example: "PASS: Mg b triplet rest wavelengths match NIST air values within 0.01 A (Kramida+ 2023)."

  FAIL: specific quantity, observed value, expected value, source, remediation.
        example: "FAIL: lines.py line 37 has Halpha=6562.80; NIST air is 6562.80 A so this is correct.
                 But lines.py line 33 has Hbeta=4861.33 while NIST air is 4861.33 A.
                 Corrections: none needed for these two lines. However Na_D1 at 5895.92 A should be cross-checked."
        (Invent a real FAIL only when evidence warrants; never manufacture findings.)

  INCONCLUSIVE: what is missing to decide.
        example: "INCONCLUSIVE: cannot verify continuum-fill convention without seeing the median of
                 features.npz; recommend adding a sanity-check print to build_features.py output."

Phase-specific additional checks will be appended to this prompt at invocation time.
Under 300 words per response (priors section excluded).
```

### 4.6 Coder contract

```
You are the Coder agent for phase {N} of stellar-mk-audit.

Inputs (paste verbatim):
  - Planner brief: {paste}
  - Physicist red-lines: {paste}

Repo branch: {branch}
Working directory: {path}

Rules:
  - Create or modify exactly the files listed in the Planner brief. No more, no less.
  - Reuse utilities from src/preprocess/, src/fetch/, src/utils/. Import, do not copy.
  - Add tests under tests/ matching the test cases in the brief.
  - Run pytest only on the new or changed test files, -x to fail fast.
  - Commit with message: "phase{N}: <one-line summary>".
  - Do NOT push.
  - Do NOT modify files outside the listed paths.
  - Do NOT add em dashes.
  - Do NOT add "Liu et al" citations (correct authors are Li, Lin and Qiu).

Return:
  - diff summary (file -> lines added/removed)
  - pytest tail (last 30 lines) verbatim
  - one-sentence self-assessment: "ready for review" or "blocked by X"
```

### 4.7 Reviewer contract

```
You are the Reviewer agent for phase {N} of stellar-mk-audit.

Inputs:
  - Coder diff: {paste or path}
  - Planner brief: {paste}

Check, in order, and stop reporting after the first FAIL of each category:

  1. File scope: every file in the Planner brief was created or modified. No files outside scope.
  2. No duplication: functions available in src/preprocess/, src/fetch/, src/utils/ are imported, not re-implemented.
  3. Determinism: every np.random, sklearn split, LightGBM call, SHAP call has random_state or seed set.
  4. Test coverage: every public function in changed files has at least one test.
  5. No shims, no dead code, no "// removed" comments, no unused _vars.
  6. Citation hygiene: grep -rn "Liu et al" in changed files returns nothing.
  7. No em dashes: grep for U+2014 and U+2013 in changed files returns nothing.
  8. Type hints on every public signature.
  9. Logging: public functions log inputs, outputs, and decisions at INFO level.

Return PASS or FAIL with a consolidated diff list of required changes.
If PASS, state one-line confidence summary.
If FAIL, list every finding with file:line and a suggested fix.
Under 300 words.
```

### 4.8 Invoking subagents in Claude Code

In Claude Code, subagents are spawned with the Task tool, `subagent_type="general-purpose"`. Multiple Task calls in the same message run in parallel. Use this pattern:

- **Planner**: one Task call, wait for result, paste into context.
- **Physicist-pre**: one Task call after Planner returns (or parallel if the spec is obvious).
- **Coder**: one Task call after Planner and Physicist-pre are both in context.
- **Reviewer + Physicist-post**: two Task calls in the SAME message (parallel).
- **Gate check**: do in the orchestrator (Claude Code itself), not a subagent.

Every subagent invocation includes the relevant contract prompt above plus phase-specific additions from section 5.

### 4.9 Escalation

Escalate to TJ via a direct question in the main conversation when:
- Reviewer or Physicist returns FAIL twice on the same gate with the same root cause
- A kill criterion trips
- External data is unavailable (VizieR, Pickles) and no fallback is documented
- A phase exceeds its time budget by 50 percent with no visible progress

Never silently proceed past an escalation trigger.

---

## 5. Per-phase execution

Each phase section below states: preconditions, Planner brief seed (what Claude Code passes to Planner), phase-specific Physicist checks, commands, acceptance gate, kill criteria, commit message.

### 5.1 Phase -1: repo health

**Preconditions:** none (this is the starting phase).

**Audit baseline (known issues, 2026-04-18):**
- `pyproject.toml` still reads as the upstream GONS project: wrong name, wrong description, `torch` as core dep, LightGBM/SHAP behind `[interpret]` extra, URLs pointing to `StellarSpectraWithGONS`.
- `README.md` is one line.
- `decisions.md` does not exist.
- `src/interpret/__init__.py` references `plan-stellar-wild-manatee.md` (not in repo).
- `src/interpret/features.py` hardcodes the 4800-6800 A window silently (Phase 0 Decision 1 already committed).
- `src/interpret/labels.py` excludes O/B/M at construction silently (Phase 0 Decision 3 already committed).
- `src/interpret/labels.py` targets VizieR `J/A+A/666/A121` (Hourihane+ 2023); TJ's plan text cited `J/A+A/692/A228`. Code is likely correct; decision is not logged.
- `src/preprocess/` and `src/fetch/` include APOGEE and GALAH code paths not used here.
- `src/interpret/occlusion.py` uses `CONTINUUM_FILL = 1.0` without asserting features are normalized to that level.
- `src/interpret/classifier.py` is single-split; `project_plan.md` Phase 2 calls for 5-fold CV. Reconcile in Phase 2.
- Tests exist but are not verified to pass.

**Planner brief seed:**
"Bring repo to green-test state. Rewrite `pyproject.toml` to reflect the LightGBM audit paper. Write a real `README.md`. Create `decisions.md` and log the four decisions the code has already committed silently. Verify VizieR resolution for `J/A+A/666/A121`. Run existing tests and report status. Trim or document the GONS leftovers per CLAUDE.md §3."

**Physicist-pre checks (specific to this phase):**
- Confirm bibcode `2023A&A...666A.121H` maps to Hourihane et al. 2023 and that table is GES DR5 recommended parameters.
- Confirm MK bin edges in `labels.py` match Pecaut & Mamajek 2013 within +/- 100 K.
- Confirm that excluding O/B/M at label time is scientifically justified given the UVES window 4800-6800 A lacks TiO and He II lines needed for O/B/M classification.

**Physicist-post checks:**
- `decisions.md` entries each cite a scientific reference where applicable.
- `README.md` correctly describes the pipeline as AIR-wavelength, continuum-normalized, GES UVES U580.

**Commands:**

```bash
# Edit pyproject.toml (Coder delegated)
# Edit README.md (Coder delegated)
# Create decisions.md (Coder delegated)
# Fix src/interpret/__init__.py reference

# Verify VizieR resolves
python -c "
from astroquery.vizier import Vizier
Vizier.ROW_LIMIT = 5
t = Vizier.get_catalogs('J/A+A/666/A121')
print(f'n_tables={len(t)}')
for i, tbl in enumerate(t):
    print(f'table {i}: {len(tbl)} rows, columns: {tbl.colnames[:12]}')
"

# Install and test
pip install -e '.[dev]'
mkdir -p artifacts
pytest tests/ -x -v 2>&1 | tee artifacts/pytest_phase_minus_1.log
```

**decisions.md seed entries** (Coder writes these; dates are today's date at runtime):

1. Window locked to 4800-6800 A (Option A). Reasoning: UVES U580 native coverage; avoids combining setups; covers Hbeta, Mg b, Na D, Ca I, Halpha. Reversal: Phase 0 coverage probe shows fewer than 2000 spectra survive.
2. Label scope locked to A/F/G/K. Reasoning: window excludes TiO bands needed for M; window excludes He II needed for O/B; UVES sample is FGK-heavy by design (Gilmore+ 2022). Reversal: Phase 0 class counts show A below 50 spectra.
3. Benchmark locked to Pickles 1998 UVKLIB (VizieR J/PASP/110/863). Reasoning: independent physics, survives "tested on labels that trained it" critique. Reversal: Pickles FITS unavailable locally and remote mirrors blocked.
4. GES params catalog locked to VizieR J/A+A/666/A121 (Hourihane et al. 2023). Reasoning: GES DR5 recommended parameters with Teff/logg/[Fe/H]. Reversal: VizieR does not resolve, or columns do not match `_VIZIER_ALIASES` in labels.py.

**Acceptance gate:**
- `pyproject.toml` renamed, core deps include lightgbm/shap/sklearn/astroquery, no torch.
- `README.md` contains: abstract, install block, reproduce block, data provenance, license.
- `decisions.md` has the four seed entries with real dates.
- VizieR probe returns at least one table with columns resolving to RA, Dec, Teff, logg, [Fe/H].
- `pytest tests/ -x` either passes or has a documented list of expected failures with reasons.

**Kill criterion:** VizieR does not resolve and no CDS mirror is available. Stop and ask TJ whether to pivot to SIMBAD (Option C) or download the catalog manually.

**Commit message:** `phase-1: repo health (pyproject, readme, decisions log, vizier verified)`

---

### 5.2 Phase 0: pre-flight probes

**Preconditions:** Phase -1 acceptance met. Regridded HDF5 path known.

**Planner brief seed:**
"Run the coverage probe and class-count probe against the real HDF5. Log counts to decisions.md. Verify Pickles FITS availability (or schedule a download task). Do not write or modify source; this phase is measurement only."

**Physicist-pre checks:**
- Window 4800-6800 A covers all nine MK_LINES entries. Confirm.
- Coverage threshold 0.90 is appropriate given the UVES inter-chip gap is about 3.25 percent of the window.
- Class-count floor of 50 per class is the minimum for meaningful per-class recall; 100+ preferred.

**Physicist-post checks:**
- Per-class Teff distribution peaks match expectation: A near 7500-8000 K, F near 6500, G near 5700, K near 4500.
- If any class peak is more than 300 K off expectation, something is wrong with bin edges or catalog columns.

**Commands:**

```bash
# Coverage probe (Decision 1 verification)
python -c "
from src.interpret.features import coverage_probe
r = coverage_probe(h5_path='<PATH>', wave_min=4800, wave_max=6800)
print(r)
"

# Class-count probe (Decision 3 verification)
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
print('Teff means by class:')
print(df.groupby('mk_class')['teff_k'].agg(['mean', 'std', 'count']))
"

# Benchmark availability
ls data/pickles/uk*.fits 2>/dev/null | head -5 || echo "NO PICKLES FITS FOUND"
```

**Acceptance gate:**
- Coverage probe: `n_covered >= 2000`, logged to decisions.md.
- Class-count probe: at least 3 surviving classes each with n >= 50, logged.
- Teff per-class means within +/- 300 K of expectation (Physicist-post).
- Benchmark: Pickles present, OR explicit download task added to Phase 5 preconditions.

**Kill criterion:** `n_covered < 2000` OR fewer than 3 classes with n >= 50. Stop and escalate.

**Commit message:** `phase0: probes passed (n_covered=<X>, classes=<Y>, pickles=<status>)`

---

### 5.3 Phase 1: data and labels

**Preconditions:** Phase 0 acceptance met.

**Planner brief seed:**
"Run build_labels and build_features scripts end-to-end. Produce artifacts/ges_mk_labels.parquet and artifacts/features.npz. Add tests/test_line_coverage.py if missing: assert every LINE_SETS entry intersects the feature window after Phase 0 window lock. Document the final per-class sample sizes."

**Physicist-pre checks:**
- `match_radius_arcsec=0.5` is the Gaia DR3 astrometric RMS; appropriate.
- `ambiguity_radius_arcsec=2.0` prevents wrong-star matching in crowded fields.
- `min_snr=20.0` per GES DR5.1 QC.
- `rebin_factor=5` produces ~2.75 A bins; Mg b triplet is resolved (separations 5-11 A); Na D doublet at 5.97 A separation may or may not be resolved depending on exact bin centers. Flag if Na D1 and Na D2 land in the same bin.

**Physicist-post checks:**
- `wave_centers` covers 4800-6800 A with at most one gap at the inter-chip region (5769-5834 A); if imputed, the imputer median for those bins should differ from the rest by less than 10 percent.
- `features.npz['X']` median should be close to 1.0 (continuum-normalized).
- Per-class Teff distributions in the final label set still peak at expected values.
- No class has n_test (test set) less than 20; report if any do.

**Commands:**

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

pytest tests/test_labels.py tests/test_features.py tests/test_lines.py tests/test_line_coverage.py -v
```

If script arg names differ from above, Coder reads the script and adapts. Do not invent arg names.

**Acceptance gate:**
- `artifacts/features.npz` loads with keys `X, y, wave_centers, train_idx, val_idx, test_idx, boundary_distance_k, dwarf_flag, groups, median_imputer`.
- `X.shape[0] == len(y)`.
- `X.shape[1]` close to 1000 within 20 percent.
- No NaN in X after imputation.
- Per-class counts in y match decisions.md Phase 0 numbers within 10 percent.
- All four tests pass.
- Physicist-post PASS.

**Kill criterion:** final n after SNR cut drops below 1500. Stop.

**Commit message:** `phase1: features and labels (n=<X>, n_bins=<Y>, classes=<Z>)`

---

### 5.4 Phase 2: baseline classifier

**Preconditions:** `artifacts/features.npz` exists; per-class n_train >= 100 for at least 3 classes.

**Reconcile CV decision (add to decisions.md before running):**

Pick one, justify, log:
- (a) Add `train_cv()` to classifier.py: 5-fold stratified CV on train+val combined, report mean/std of macro-F1, then fit final model on full train+val for test eval. Aligns with project_plan.md Phase 2.
- (b) Stay single-split. Justify with "bootstrap CIs in Phase 4 provide uncertainty; single split preserves reproducibility."

Default: (a). If blocked more than 4 hours, fall back to (b) with explicit reasoning logged.

**Planner brief seed:**
"Train the LightGBM classifier per the CV choice logged to decisions.md. Produce artifacts/lgbm_mk.pkl and artifacts/metrics.json. Inspect the confusion matrix. Run a sensitivity check on the boundary-distance-filtered subset (|boundary_distance_k| > 200)."

**Physicist-pre checks:**
- `class_weight='balanced'` is appropriate given FGK-heavy sample.
- `max_depth=8` and `num_leaves=63` keep trees shallow enough for SHAP; do not inflate.
- `n_estimators=500` with early stopping at 50 rounds is standard; do not change.

**Physicist-post checks:**
- Confusion pattern: adjacent-class off-diagonal dominates. A<->K confusion under 2 percent.
- Per-class recall highest on largest-sample class; lowest on A (expected given UVES under-sampling of A).
- Sensitivity check: accuracy on boundary-filtered subset is higher than full-test accuracy by at least 3 percentage points. If not, something is wrong (memorizing noise rather than temperature).

**Commands:**

```bash
python scripts/train_classifier.py \
    --features artifacts/features.npz \
    --model-out artifacts/lgbm_mk.pkl \
    --metrics-out artifacts/metrics.json \
    2>&1 | tee artifacts/train_classifier.log

# Boundary sensitivity (Coder may embed this in train_classifier or run separately)
python -c "
import numpy as np, json, pickle
from src.interpret.classifier import evaluate
d = np.load('artifacts/features.npz')
with open('artifacts/lgbm_mk.pkl', 'rb') as f: m = pickle.load(f)
test = d['test_idx']
bd = d['boundary_distance_k'][test]
mask = np.abs(bd) > 200
print(f'full test acc: {(m.predict(d[\"X\"][test]) == d[\"y\"][test]).mean():.4f}')
print(f'bd>200 test acc: {(m.predict(d[\"X\"][test][mask]) == d[\"y\"][test][mask]).mean():.4f}')
"
```

**Acceptance gate:**
- `metrics.json` exists.
- `macro_f1 >= 0.7`.
- Per-class recall >= 0.6 for every class with >= 100 training samples.
- Confusion: A<->K confusion under 2 percent.
- Boundary-filtered accuracy at least 3 percentage points above full-test accuracy.
- Physicist-post PASS.

**Kill criteria:**
- `macro_f1 < 0.5`: stop. Pivot options in project_plan.md.
- Any class recall < 0.4: do not stop, but log in decisions.md that Phase 4 ablation results for that class will be uninterpretable.

**Commit message:** `phase2: classifier (macro-f1=<X>, recall-ok)`

---

### 5.5 Phase 3: interpretability triangulation

**Preconditions:** Phase 2 acceptance met.

**Planner brief seed:**
"Produce permutation importance, TreeSHAP, and sliding-window occlusion on the validation set. Write a triangulation report: pairwise Jaccard on top-20 features for perm-vs-SHAP, perm-vs-occlusion, SHAP-vs-occlusion, plus three-way intersection. Produce a per-class importance figure showing class-specific wavelength preferences."

**Physicist-pre checks:**
- Permutation importance scoring should be 'accuracy' (not 'neg_log_loss') for a classification audit.
- `n_repeats=10` gives reasonable CI on permutation.
- SHAP TreeExplainer on a stratified subsample of 1000 rows is the budget ceiling; do not exceed.
- Sliding window: 50 A wide with 25 A stride resolves Mg b triplet (21 A wide) and Na D doublet (6 A wide) adequately.

**Physicist-post checks (CRITICAL):**
- Top-10 perm-importance bins for class A: at least one bin within 2 A of 4861.33 (Hbeta) OR 6562.80 (Halpha). If neither, flag.
- Top-10 perm-importance bins for class G or K: at least one bin within 2 A of 5167-5184 (Mg b) OR 5889-5896 (Na D). If neither, flag.
- If expected peaks are missing, do NOT tune until they appear. Log the negative result. The paper framing shifts to shortcut learning, which is also publishable (project_plan.md Phase 4 kill-criteria pivot).

**Commands:**

```bash
python scripts/run_interpret.py \
    --model artifacts/lgbm_mk.pkl \
    --features artifacts/features.npz \
    --out-dir artifacts/interpret \
    2>&1 | tee artifacts/run_interpret.log

# Expected outputs:
#   artifacts/interpret/perm_importance.npz
#   artifacts/interpret/shap_values.npz
#   artifacts/interpret/occlusion_trace.npz
#   artifacts/interpret/triangulation_report.json
```

**Acceptance gate:**
- All three methods produce non-constant outputs (std > 0 across features).
- Jaccard(top-20 perm, top-20 SHAP) >= 0.5.
- Per-class importance figure renders; class-specific peaks visible.
- Physicist-post PASS or an explicit "shortcut-learning pivot" flag in decisions.md.

**Watchouts:**
- If SHAP takes more than 2 hours per class, reduce `shap_subsample` or skip classes with n < 200 for SHAP; document in decisions.md.
- Do not report Gini/impurity importance as a primary result. It is biased toward high-cardinality features; Li, Lin and Qiu explicitly rejected it.

**Commit message:** `phase3: triangulation (jaccard perm-shap=<X>, physicist=PASS)`

---

### 5.6 Phase 4: causal ablation + line matching (HEADLINE)

**Preconditions:** Phase 3 acceptance met. Feature matrix median near 1.0 verified (sanity check below).

**Continuum-fill sanity check (run before ablation):**

```bash
python -c "
import numpy as np
d = np.load('artifacts/features.npz')
med = float(np.nanmedian(d['X']))
p1 = float(np.nanpercentile(d['X'], 1))
p99 = float(np.nanpercentile(d['X'], 99))
print(f'X median={med:.3f}, 1%={p1:.3f}, 99%={p99:.3f}')
assert 0.5 < med < 1.5, f'median {med} out of range; CONTINUUM_FILL=1.0 is wrong'
print('OK: CONTINUUM_FILL=1.0 is consistent with feature matrix.')
"
```

If this fails, stop. Fix `CONTINUUM_FILL` to `np.nanmedian(X)` or fix the upstream normalization. Do not run ablation with a mismatched fill value.

**Planner brief seed:**
"Run masked-line ablation with 500 bootstrap resamples and 100 random-window controls per line set. Compute per-class delta-accuracy with 95 percent bootstrap CIs. Run line-matching at tolerances 1, 2, 5, 10 A. Produce ablation.csv with columns line_set, mk_class, n_test, baseline_acc, masked_acc, delta_acc_mean, delta_acc_ci_low, delta_acc_ci_high, p_value_vs_random."

**Physicist-pre checks:**
- Random-window controls must have total width matched to line-set width AND must avoid all MK_LINES regions. Verify the forbidden-mask logic in occlusion.py.
- Bootstrap resamples test rows with replacement; CIs are on delta_acc, not on masked_acc alone.
- Expected signs (reminder):
  - (H_balmer, A): delta_acc_ci_high < 0 (headline test)
  - (Mg_b, G) or (Mg_b, K): delta_acc_ci_high < 0
  - (Na_D, K): delta_acc_ci_high < 0

**Physicist-post checks (CRITICAL):**
- For at least 2 of the 3 headline pairs, `delta_acc_ci_high < 0` AND `p_value_vs_random <= 0.01`.
- Directional sign of delta_acc matches expected pattern above for at least 5 of the 8 (line_set, class) combinations.
- Line-matching precision at 5 A tolerance >= 0.5 against `MK_LINES_in_window`.
- Random-window controls: at least 80 of 100 successfully drawn for each line set (log failures).

**Commands:**

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
save_sweep('artifacts/ablation/line_match.csv',
           'artifacts/ablation/line_match.json', results)
for r in results:
    print(r)
"
```

**Acceptance gate:**
- `artifacts/ablation/masked_line_ablation.csv` exists with required columns.
- At least 2 of 3 headline pairs pass (p <= 0.01, delta_acc_ci_high < 0).
- Line-matching precision at 5 A tolerance >= 0.5.
- Physicist-post PASS.

**Kill criterion / pivot:**

If NO line set shows p <= 0.05 against random null for any class:
- The model is not physics-aware as assumed.
- This is still publishable as "audit reveals shortcut learning in tree-based stellar classifiers."
- Add 2 weeks to Phase 6 budget. Escalate to TJ with the finding. Continue to Phase 5.

If line-matching precision at 10 A tolerance < 0.3: keep line-matching in the paper but de-emphasize; lead with ablation.

**Commit message:** `phase4: ablation (<N>/3 headline pairs pass, line-match-precision-5A=<X>)`

---

### 5.7 Phase 5: external benchmark + figures

**Preconditions:** Phase 4 acceptance met (or pivot to shortcut-learning framing logged). Pickles UVKLIB FITS present in `data/pickles/`.

**Pickles download (if not already present):**

```bash
# Pickles 1998 is at VizieR J/PASP/110/863. UVKLIB templates are also mirrored
# at ESO: https://www.eso.org/sci/observing/tools/standards/IR_spectral_library.html
# Download UVKLIB to data/pickles/. Ask TJ if network access is restricted.
```

**Planner brief seed:**
"Run Pickles benchmark on the test set. Produce benchmark confusion matrix vs classifier predictions. Produce two figures for the paper: (1) main result (importance trace + spectrum + line markers, single panel); (2) per-class importance (one subplot per class). Produce confusion matrix figure and ablation summary bar chart."

**Physicist-pre checks:**
- Pickles templates are on flux (not flambda) in absolute units; continuum-normalize before chi-squared matching.
- UVKLIB filename convention: `uk{N}.fits` with N in 1-131, per Pickles 1998 Table 3.
- Collapsing Pickles MK types to A/F/G/K/OTHER: 'A*V' -> 'A', 'F*V' -> 'F', etc. Luminosity class ignored.

**Physicist-post checks:**
- Benchmark confusion concentrated on diagonal for F/G/K.
- A row in benchmark confusion may have high OTHER count because Pickles has many A subtypes; expected.
- Figure 1 tells the paper's story in one image (importance aligns with MK_LINES for at least one class).

**Commands:**

```bash
python scripts/run_benchmark.py \
    --model artifacts/lgbm_mk.pkl \
    --features artifacts/features.npz \
    --pickles-dir data/pickles \
    --out-dir artifacts/benchmark \
    2>&1 | tee artifacts/run_benchmark.log

python scripts/make_figure.py \
    --importance artifacts/interpret/perm_importance.npz \
    --ablation artifacts/ablation/masked_line_ablation.csv \
    --metrics artifacts/metrics.json \
    --benchmark artifacts/benchmark/benchmark_report.json \
    --out-dir artifacts/figures \
    2>&1 | tee artifacts/make_figure.log
```

**Acceptance gate:**
- `artifacts/benchmark/benchmark_report.json` exists with `agreement_rate` field.
- `agreement_rate >= 0.70` on classes where both agree at n > 100 (honest threshold; Li, Lin and Qiu reported 0.93 on a different setup).
- All figures render at 300 DPI for print.
- Figure 1 is readable standalone.
- Physicist-post PASS.

**Commit message:** `phase5: benchmark (agreement=<X>) and figures`

---

### 5.8 Phase 6: drafting

**You (Claude Code) do not write the manuscript. TJ writes. You review.**

**Preconditions:** Phases 1-5 all green.

**Planner brief seed:**
"Produce a review checklist for TJ's draft. Do not write prose. Cross-check every numerical claim in the draft against the artifact CSVs and JSONs. Produce a list of discrepancies."

**Physicist-post role during drafting:**
When TJ shares a draft section, the Physicist agent reviews:
- Every claimed numerical value matches the artifact value within quoted precision.
- Every cited line wavelength matches NIST air.
- No em dashes.
- No "Liu et al" (correct: Li, Lin and Qiu).
- Candebat et al. 2024 cited as using GIRAFFE HR10 and HR21, NOT UVES.
- Honest limitations section includes: no calibrated Bayesian uncertainty, small sample vs Candebat 50K, single survey, FGK-only.

**Commit message:** `phase6: draft review pass <N>`

---

### 5.9 Phase 7: submission

**Preconditions:** Phase 6 final draft approved by TJ.

**Planner brief seed:**
"Compile manuscript LaTeX. Mint Zenodo DOI. Prepare arXiv submission. Do NOT submit; stop at 'ready to upload'. TJ performs the submission."

**Physicist-post role:**
- Confirm arXiv category astro-ph.IM is correct for an interpretability-focused paper (not astro-ph.SR which is for stellar astrophysics results).
- Confirm endorser candidates are currently active on arXiv (search recent submissions).

**Commands:**

```bash
# Coder prepares LaTeX + Zenodo metadata; does NOT push or submit.
# Final submission is TJ manual action.
```

**Commit message:** `phase7: submission-ready`

---

## 6. Artifacts register

Expected artifacts by phase. `artifacts/` is gitignored except for `artifacts/acceptance_log.json`.

| Phase | Artifacts |
|-------|-----------|
| -1    | `pytest_phase_minus_1.log` |
| 0     | entries in `decisions.md` |
| 1     | `ges_mk_labels.parquet`, `features.npz`, `build_labels.log`, `build_features.log` |
| 2     | `lgbm_mk.pkl`, `metrics.json`, `train_classifier.log` |
| 3     | `interpret/perm_importance.npz`, `interpret/shap_values.npz`, `interpret/occlusion_trace.npz`, `interpret/triangulation_report.json` |
| 4     | `ablation/masked_line_ablation.csv`, `ablation/line_match.csv`, `ablation/line_match.json`, `ablation.log` |
| 5     | `benchmark/benchmark_confusion.csv`, `benchmark/benchmark_report.json`, `figures/*.pdf`, `figures/*.png` |
| 6     | draft discrepancy report (conversation-only) |
| 7     | `submission/manuscript.tex`, `submission/zenodo_metadata.json` |

---

## 7. Acceptance log

After each passing gate, append to `artifacts/acceptance_log.json`:

```json
{
  "phase": "<N>",
  "date": "<ISO date>",
  "commit": "<git hash>",
  "reviewer": "PASS",
  "physicist": "PASS",
  "gates": {
    "gate_name_1": {"required": "<value>", "observed": "<value>", "status": "PASS"},
    "gate_name_2": {"required": "<value>", "observed": "<value>", "status": "PASS"}
  },
  "notes": "<one line>"
}
```

This file is the reproducibility contract. TJ reviews it before submission.

---

## 8. When in doubt

- Read `project_plan.md` for the phase scope and kill criteria.
- Read `decisions.md` for prior binding decisions.
- Invoke the Physicist before making any scientific choice, even if the Planner did not specify one.
- Do not invent scope. Do not fabricate. Do not iterate past 2 failures. Stop and ask TJ.
