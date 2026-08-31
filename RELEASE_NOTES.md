# Release v1.0.0

This is the reproducibility bundle for the first arXiv version of *Aggregate
Disambiguation Systems*.

It contains:

- the sanitized 2,000-cell binary pilot ledger and realized 40-row evaluator
  design;
- exact fixed-catalogue, frozen-census, and outer-sampling diagnostics;
- the frozen pilot manifest, prompts, dataset, schemas, dependency lock, and
  analysis implementation;
- Part 1's deterministic figures, operational Monte Carlo study, exact
  security curves, published CSV outputs, and numerical regression tests; and
- provenance hashes identifying the original execution revision and the clean
  offline analysis revision.

The 50 pilot cases are a manually constructed diagnostic catalogue. They are
not an IID workload sample, do not support a workload-population claim, and
were not independently adjudicated as semantic truth. Raw provider text and
routing traces are not included; their immutable hashes remain in the
provenance record.
