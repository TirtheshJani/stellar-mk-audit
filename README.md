# stellar-mk-audit

## Abstract

This repository audits a LightGBM classifier that assigns Morgan-Keenan (MK)
spectral classes A, F, G, and K to Gaia-ESO Survey (GES) UVES U580 stellar
spectra. The audit triangulates three interpretability methods (permutation
importance, TreeSHAP, and sliding-window occlusion) and adds a causal
masked-line ablation that tests whether the classifier relies on physically
diagnostic absorption lines (Balmer series, Mg b triplet, Na D doublet, Ca I).
Results are benchmarked against the Pickles 1998 UVKLIB template library as an
independent-physics cross-check. All spectra are processed in AIR wavelengths
over 4800 to 6800 Angstrom and are continuum-normalized before feature
extraction.

## Install

```bash
pip install -e ".[dev]"
```

## Reproduce

The end-to-end pipeline is run as a sequence of scripts under `scripts/`:

1. `scripts/build_labels.py` resolves GES parameters via VizieR and constructs
   the A/F/G/K label set with boundary-distance diagnostics.
2. `scripts/build_features.py` extracts the rebinned, continuum-normalized
   feature matrix from the UVES HDF5 store.
3. `scripts/train_classifier.py` fits the LightGBM model and produces
   `metrics.json`.
4. `scripts/run_interpret.py` computes permutation importance, TreeSHAP, and
   sliding-window occlusion on the held-out validation set.
5. `scripts/ablation.py` executes the causal masked-line ablation with
   bootstrap confidence intervals and random-window controls.
6. `scripts/run_benchmark.py` performs the Pickles 1998 UVKLIB cross-check.
7. `scripts/make_figure.py` assembles publication figures.

See `CLAUDE.md` for the phased execution plan and `project_plan.md` for the
12-week scope. Binding decisions (wavelength window, class scope, catalog
choice, benchmark) are logged in `decisions.md`.

## Data provenance

- Gaia-ESO Survey DR5 recommended parameters: VizieR `J/A+A/666/A121`
  (Hourihane et al. 2023, A&A 666, A121).
- Spectra: UVES U580 setup, red arm, covering 4800 to 6800 Angstrom in AIR
  wavelengths, continuum-normalized per the GES pipeline convention
  (Sacco et al. 2014, A&A 565, A113).
- Benchmark templates: Pickles 1998 UVKLIB stellar flux library
  (PASP 110, 863; VizieR `J/PASP/110/863`).

## GONS leftovers

The directories `src/preprocess/` and `src/fetch/` contain APOGEE and GALAH
code paths inherited from the upstream GONS pipeline. Only the UVES paths are
exercised in this audit. These directories are treated as read-only per
`CLAUDE.md` section 3; any required adjustments are made in thin wrappers
under `src/interpret/`.

## License

MIT. See `LICENSE`.
