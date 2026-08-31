# Publication artifacts

This directory contains the deliberately released publication bundle for
pilot-21f-final-20260803-01. The bundle is **diagnostic only**: the 50 frozen
items are a stratified seed catalogue with construction labels, not an IID
sample from an identified workload-generator law and not independently
adjudicated external truth.

global_verdicts.csv is the sanitized 2,000-cell binary ledger and
evaluator_rows.csv records the realized 40-row evaluator design. The figures,
tables, summaries, and both ledgers are hash-bound by provenance.json. Raw
provider text and routing traces remain outside Git; no credential or
authorization material is included.

The refreshed binary-vote release separates three reference objects:

- `finite_catalogue_coverage.csv` gives evaluator-uncertainty certificates for
  the fully enumerated 50-unit catalogue;
- `finite_census_per_item.csv` and `finite_census_coverage.csv` give exact
  without-replacement results for the frozen 40-row census;
- `operational_coverage.csv` gives the exact and Hoeffding outer-sampling
  calculations, explicitly as non-population diagnostics for this stratified
  catalogue.

`population_sensitivity.csv`, `failure_policy_sensitivity.csv`, and
`finite_census_security.csv` contain descriptive sensitivity analyses and state
their claim scope in each row.

The manuscript may use this bundle as an implementation and case-study
diagnostic. A confirmatory population claim requires a new immutable run that
passes the acceptance gates in PROTOCOL.md.
