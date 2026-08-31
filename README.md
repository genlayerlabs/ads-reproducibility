# Aggregate Disambiguation Systems: reproducibility bundle

Code, tests, generated figures, and sanitized pilot data for Part 1,
*Aggregate Disambiguation Systems*. The repository contains both the
deterministic numerical studies used in the paper and the executable empirical
companion. Release `v1.0.0` is the archival bundle associated with the first
arXiv version.

The empirical study evaluates frozen equivalence-principle problems with a
declared population of LLM configurations, computes the finite-sample
operational certificate, and emits publication artifacts with complete
provenance. The theoretical and synthetic calculations are under
[`theory/part1/`](theory/part1/README.md).

The current dataset is a **50-item pilot seed set**, not a confirmatory sample.
No command interprets it as evidence about a universal LLM or workload
population. Each record is treated as a pre-frozen workload unit; this pilot
does not claim to sample a workload-generator law. The normative scientific
contract is [PROTOCOL.md](PROTOCOL.md).
The operational command sequence is [docs/RUNBOOK.md](docs/RUNBOOK.md).
The declared evaluator catalogue and unavailable requested aliases are recorded
in [docs/MODEL_POPULATION.md](docs/MODEL_POPULATION.md).

## Current state

- Seed dataset imported and hash-pinned under `configs/problems/`.
- Pilot manifest declares evaluator families, row sampling, prompts, failures,
  statistical budgets, and the implemented GenLayer panel grid.
- The router client targets `https://router.ygr.ai/v1` and records the complete
  sanitized `x_router` trace returned by Unhardcoded.
- Validation, resumable execution, analysis, figures, tables, and regression
  tests are available from the Makefile.
- The completed run pilot-21f-final-20260803-01 contains 2,000 unique cells
  and is released under results/paper/ as an explicitly diagnostic case study.
  The sanitized binary cell ledger and realized evaluator-row design are
  included so the reported aggregates can be recomputed without provider
  credentials.
- Confirmatory execution remains blocked until a larger independently labelled
  workload is frozen and preregistered.
- Part 1's deterministic figures, exact security curves, operational Monte
  Carlo study, published CSV outputs, and numerical regression tests are
  released under `theory/part1/`.

## Safety and credentials

Real calls are never made by `make validate`, `make test`, or `make analyze`.
Every live execution requires `--confirm-spend`; the full pilot additionally
requires `CONFIRM_LIVE_PILOT=YES`.

Credential lookup order is:

1. `LLM_API_KEY` or `UNHARDCODED_API_KEY` in the process environment;
2. `.env.local` and then `.env` in this repository (both gitignored);
3. `MY_DEV_KEY` in the adjacent local `../llm-policy-host/.env` checkout.

Only the key's presence is ever logged.  No key, authorization header, provider
credential, raw digest, or sibling secret file is copied into a run artifact.

## Setup

With Nix:

```bash
nix-shell --run 'uv sync --frozen'
nix-shell --run 'make validate test'
```

Or with an existing `uv` installation:

```bash
uv sync --frozen
make validate test
```

## Router checks and live execution

Catalog/credential probe without inference spend:

```bash
nix-shell --run 'make probe'
```

Live checks are deliberately staged. The frozen load candidate permits at most
three concurrent evaluator calls globally, serializes calls within each model
family, and has no artificial inter-call delay:

```bash
STAGE1_RUN_ID=stage1-YYYYMMDD-HHMM nix-shell --run 'make smoke-one'
STAGE2_RUN_ID=stage2-YYYYMMDD-HHMM nix-shell --run 'make smoke-two'
SMOKE_RUN_ID=smoke-YYYYMMDD-HHMM nix-shell --run 'make smoke'
FAMILY_SMOKE_RUN_ID=family-smoke-YYYYMMDD-HHMM nix-shell --run 'make smoke-families'
STRESS_RUN_ID=stress-YYYYMMDD-HHMM nix-shell --run 'make stress-families'
```

Full 50-by-40 pilot (2,000 evaluator calls):

```bash
CONFIRM_LIVE_PILOT=YES nix-shell --run 'make pilot'
```

Runs are resumable. Each cell is keyed by item and evaluator-row index; a
successful or conservatively mapped malformed cell is not called twice.

## Analysis

Recompute every released vote-based diagnostic from the sanitized ledgers,
without credentials or provider calls:

```bash
nix-shell --run 'make refresh-release'
```

The following command is for a local checkout that also contains the original
non-public run records:

```bash
nix-shell --run 'make analyze paper-artifacts'
```

The analysis reports:

- per-item acceptance rates and Clopper--Pearson intervals;
- simultaneous mass-controlled resolvability bounds over the declared panel
  grid, with exact exterior inversion and a Hoeffding benchmark;
- threshold and certification-boundary mass diagnostics for every panel size;
- agreement with pilot construction labels on resolvable items;
- disagreement/calibration on constructed unresolvable items;
- input-presentation-stress outcomes;
- exact fixed-catalogue and frozen-census coverage;
- evaluator-population and failure-policy sensitivity;
- catalogue plug-in security profiles for both fixed Byzantine share and targeted
  census contamination;
- provider/model, error, latency, token, and cost summaries.

Pilot outputs may be cited only as diagnostic case-study evidence. They do not
support a workload-population or external-truth claim. An immutable
confirmatory release must pass the acceptance gates in
[PROTOCOL.md](PROTOCOL.md) before either claim is made.

## Repository interface

```text
make validate         # schemas, hashes, manifest, and any existing run matrix
make refresh-release  # recompute public diagnostics from sanitized ledgers
make analyze          # analyze a local RUN_ID containing original run records
make paper-artifacts  # publication artifacts from original run records
make test             # empirical and Part 1 numerical regression tests
make theory-figures   # deterministic Part 1 figures and exact security CSV
make theory-monte-carlo # four predeclared 2,000-repetition scenarios
make probe            # authenticated catalog probe, no inference
make smoke            # guarded four-cell live call
make smoke-families   # guarded one-call coverage of every evaluator family
make stress-families  # guarded 3-item load gate over every evaluator family
make pilot            # guarded complete pilot
```

No live call is made by `make refresh-release`, `make test`, or any
`theory-*` target. See [`theory/part1/README.md`](theory/part1/README.md) for
the mapping from scripts to paper figures and tables.

## Citation and license

Use [`CITATION.cff`](CITATION.cff) or the citation information attached to the
`v1.0.0` release. Unless a file states otherwise, the repository is licensed
under the [MIT License](LICENSE).
