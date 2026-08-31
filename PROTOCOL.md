# Part 1 real-LLM resolvability study

## Status

This is the execution specification for the standalone empirical companion to
Part 1, version 5.5. This repository is intended to be pinned by the paper as a
Git submodule after an immutable empirical release exists.

- Repository: **genlayerlabs/ads-reproducibility**
- Proposed submodule path: **external/ads-llm-resolvability-study**
- Current state: the completed and reviewed run
  pilot-21f-final-20260803-01 is released as a diagnostic case study; it is not
  a confirmatory workload-population result.
- Mathematical reference: Part 1, especially the familywise and
  mass-controlled operational certificates.

The paper repository retains the earlier discursive design notes. This document
is the normative execution contract for this study.

The bundled 50-item pilot treats every record as an already frozen workload
unit (U). It diagnoses the evaluator population and implementation; it does
not identify an i.i.d. workload-generator population. The confirmatory phase
must independently generate, adjudicate, and freeze the larger workload sample
before evaluator calls begin.

## 1. Scientific question

For a declared workload-generator population and a declared evaluator
population, determine whether a finite evaluator panel can reproduce the ideal
population decision for most generated resolutions at predeclared error and
confidence levels.

For a fixed problem \(X\), the primary estimand is

\[
R_{X;K,\delta}
=
\Pr_{A\sim\pi}
\left(
e_{X,K}(\operatorname{Solve}(A,X))\leq\delta
\right).
\]

For multiple problems, the population unit must include the problem, generator
configuration, and frozen resolution. The primary estimand is then

\[
R_{\Pi;K,\delta}
=
\Pr_{U\sim\Pi}\left(e_K(U)\leq\delta\right),
\]

where \(\Pi\) is a predeclared workload-generator law. The observed benchmark
list is not automatically a population.

The primary confirmatory claim has the form:

> With calibration confidence at least \(1-\eta_E-\eta_G\), the selected panel
> size \(\widehat K\) has ideal resolvability coverage at least \(1-\beta\),
> with pointwise fresh-panel error at most \(\delta\) on that covered mass.

On the same event, the fresh-resolution deployment disagreement is bounded by

\[
\delta+(1-\delta)\beta.
\]

This is a coordination claim relative to declared populations and rules. It is
not a semantic-truth claim. External correctness is a separate endpoint and
requires independent labels.

## 2. Pilot and confirmatory phases

The project must have two distinct phases.

### 2.1 Pilot

The pilot may be used to:

- test prompts, parsers, tool policies, and retry behavior;
- estimate cost, latency, malformed-output rates, and provider limits;
- obtain a rough distribution of acceptance means for power simulation;
- choose feasible values of \(A\), \(M\), and the candidate grid;
- detect implementation problems in the generation-evaluation pipeline.

Pilot resolutions and votes must not be reused in the confirmatory analysis if
they influenced any primary design choice.

### 2.2 Confirmatory run

Before the first confirmatory model call, freeze and hash a preregistration
manifest containing every item in Section 3. The confirmatory analysis must run
from that manifest without interactive parameter changes.

If the confirmatory certificate fails, the result is reported as
**inconclusive at the declared design**. Parameters may be changed only for a
new run with new generator draws and a new preregistration hash.

## 3. Required preregistration

The manifest must declare:

1. Problem or workload population and its weights.
2. Generator population \(\pi\), including model versions, prompts, decoding
   parameters, tools, seeds, time window, and weights.
3. Task-specific evaluator laws \(\nu_X\), with the same metadata.
4. Whether the target is a finite catalogue, an i.i.d. superpopulation, or a
   deterministic approximation to a limiting population.
5. Component schema, binary component-verdict parser, deterministic global
   rule \(G\), threshold \(\tau\), quota, and tie policy.
6. Candidate grid \(\mathcal J\) of \((K,\delta)\) pairs.
7. Required coverage \(1-\beta\).
8. Calibration budgets \(\eta_E,\eta_G\) and evaluator mass charge \(\xi_E\).
9. Generator count \(A\), evaluator-row count \(M\), and power calculation.
10. Smallest-passing-\(K\) or other predeclared selection rule.
11. Eligibility rules, self-evaluation policy, blinding policy, and duplicate
    configuration policy.
12. Treatment of timeouts, refusals, malformed output, tool errors, retries,
    provider errors, and missing cells.
13. Confirmatory start and end conditions, including provider-version drift.
14. Alternative population weightings used only for predeclared sensitivity
    analysis.
15. External truth source and metrics, if any.

Prompts and rules may be referenced by content hash, but the referenced files
must be stored in the experiment repository.

## 4. Sampling design

### 4.1 Generator columns

Draw \(A\) independent generator configurations from the declared
workload-generator law. Generate each complete resolution once, store it
verbatim, assign it a content hash, and freeze it before evaluation begins.

Generator identity and unnecessary metadata should be hidden from evaluators.
No resolution may be edited after any confirmatory vote is observed.

### 4.2 Evaluator rows

Draw \(M\) evaluator rows independently of the generator sample. One row
contains every configuration and private randomness needed to evaluate all
eligible frozen resolutions.

Using the same evaluator row across columns is allowed and expected. Outcomes
within one row may therefore be strongly dependent across resolutions. The
operational theorems require the rows to be i.i.d. under the declared design
and each fixed-column count to have the stated binomial marginal; they do not
require column independence.

If calls require per-cell randomness, all per-cell seeds for row \(b\) must be
derived from a preregistered row seed without using observed outputs.

### 4.3 Independence of the two samples

The generator sample and evaluator-row sample must be independent. Evaluator
selection, prompts, retries, or tool access may not be changed in response to
the generated resolutions, except through rules frozen in the preregistration.

### 4.4 Complete matrix requirement

The primary certificate assumes that every eligible generator column is
evaluated by the same \(M\) sampled rows. A sparse or adaptively expanded
matrix is a different sampling design and must not be analyzed with the primary
theorem without a new justification.

If self-evaluation is excluded, either:

- use disjoint generator and evaluator populations; or
- predeclare the column-specific eligible population and use an analysis that
  accounts for the resulting unequal row sets.

The confirmatory default should be disjoint roles because it preserves the
clean rectangular design.

## 5. Execute-then-vote pipeline

For every problem \(X\), generator draw \(a\), and evaluator row \(b\):

1. Generate and freeze \(T_a=\operatorname{Solve}(a,X)\).
2. Give evaluator \(b\) the canonical \(X\), the identical frozen \(T_a\), and
   the preregistered evaluation prompt and tools.
3. Record the component vector
   \(\mathbf V_{b,a}\in\{0,1\}^{N_X}\).
4. Compute the global vote
   \(Y_{b,a}=G_X(\mathbf V_{b,a})\) in common deterministic code.
5. Store raw output, parsed output, status, timestamps, model identity, tool
   trace references, and hashes.

The model must not compute the final population decision. It only produces the
component verdicts. The experiment code applies \(G_X\), the panel quota, and
all certificates.

Malformed or missing component outputs follow the preregistered rule. The
recommended conservative default is to map them to rejection while also
reporting their rate separately.

## 6. Choosing \(A\) and \(M\)

For the mass-controlled certificate, a necessary design check for a
non-vacuous target is

\[
A
\geq
\frac{\log(|\mathcal J|/\eta_G)}
     {2(\beta-\xi_E)^2},
\qquad
0<\xi_E<\beta.
\]

This condition is not a power guarantee. Power also depends on the mass of
generator columns whose acceptance means lie near the pointwise certification
boundary.

There is no distribution-free choice of \(M\) that guarantees useful power.
Choose \(M\) through simulation using:

- the preregistered \(\mathcal J,\delta,\beta,\eta_E,\eta_G,\xi_E\);
- conservative pilot estimates or a grid of plausible acceptance-mean laws;
- both independent-column and strong shared-row dependence scenarios;
- the exact interval and certification implementation that will analyze the
  confirmatory data.

The simulation must report the probability of certifying the target, not only
the average lower bound. The confirmatory \(A\) and \(M\) are frozen before new
resolutions are generated.

The full cost is approximately \(A\times M\) evaluator calls for the declared
workload sample. If this is unaffordable, reduce the scientific claim or
develop and prove a different sampling design; do not silently reinterpret a
sparse matrix as the complete design.

## 7. Primary analysis

For each generator column \(a\), compute

\[
C_a=\sum_{b=1}^M Y_{b,a}.
\]

The default primary analysis is the mass-controlled certificate:

1. Construct a two-sided Clopper--Pearson interval
   \(I_a=I_{M,\eta_E\xi_E}(C_a)\).
2. For every predeclared \((K,\delta)\in\mathcal J\), evaluate the sound
   interval rule \(\operatorname{Cert}_{K,\delta}(I_a)\).
3. Compute

   \[
   \widehat R_{K,\delta}^{\mathrm{mass}}
   =
   \frac1A\sum_{a=1}^A
   \operatorname{Cert}_{K,\delta}(I_a).
   \]

4. Compute

   \[
   \varepsilon_G
   =
   \sqrt{\frac{\log(|\mathcal J|/\eta_G)}{2A}},
   \]

   and

   \[
   \underline R_{K,\delta}^{\mathrm{mass}}
   =
   \max\left\{
   0,
   \widehat R_{K,\delta}^{\mathrm{mass}}
   -\xi_E-\varepsilon_G
   \right\}.
   \]

5. Apply the preregistered selection rule to the simultaneous lower bounds.
6. Certify only if the selected bound is at least \(1-\beta\).

The familywise certificate may be reported as a more conservative secondary
analysis. A finite-catalogue analysis may additionally report exact
hypergeometric panel-census errors, but it answers a different question.

Changing \(\mathcal J\), \(\xi_E\), population weights, exclusion rules, or the
selection rule after inspecting the matrix invalidates the primary
multiplicity accounting.

## 8. Secondary and diagnostic analyses

Report at least:

- acceptance means and interval endpoints by resolution;
- certified fraction and lower coverage for every candidate;
- selected \(K\), or an explicit no-candidate-certified result;
- exact finite-census panel errors when meaningful;
- mass near the threshold and near each certification boundary;
- generator- and evaluator-family stratification;
- sensitivity to preregistered population weights;
- self-evaluation sensitivity if applicable;
- malformed-output, timeout, retry, and provider-error rates;
- time-window or model-version drift;
- cost and latency per generated resolution and evaluated cell.

If external labels exist, report semantic accuracy, false acceptance, false
rejection, and cases of high-confidence population error separately. Do not
fold external truth into the definition of resolvability.

## 9. Data and provenance contract

Every artifact must be tied to one immutable run identifier and one
preregistration hash.

Minimum logical tables:

- **problems:** problem identifier, canonical input hash, component schema,
  rule identifier, and workload weight;
- **generator_draws:** generator configuration, draw weight, seed, timestamps,
  provider metadata, and resolution hash;
- **resolutions:** immutable canonical resolution and parse status;
- **evaluator_rows:** evaluator configuration, row seed, provider metadata,
  and eligibility;
- **component_verdicts:** problem, resolution, evaluator row, component,
  binary verdict, raw-output reference, status, and timestamps;
- **global_verdicts:** component-vector hash and deterministic global vote;
- **analysis_manifest:** all statistical parameters, source commit, dependency
  lock, data hashes, and output hashes.

Secrets and provider credentials must never enter Git. Large or restricted raw
outputs may live in immutable object storage, but the repository must contain
content hashes, retrieval instructions, schemas, and a public derived matrix
whenever licensing and privacy permit.

## 10. Required companion-repository interface

The standalone repository should expose these commands:

~~~text
make validate
make analyze
make paper-artifacts
make test
~~~

- **make validate** checks the preregistration hash, schemas, matrix
  completeness, deterministic aggregation, and artifact hashes.
- **make analyze** computes all primary and secondary outputs from frozen data.
- **make paper-artifacts** writes publication-ready figures, tables, CSVs, and
  a provenance manifest.
- **make test** runs parser, aggregation, interval, quota, and certificate
  regression tests without calling providers.

Recommended layout:

~~~text
ads-llm-resolvability-study/
  README.md
  PROTOCOL.md
  Makefile
  pyproject.toml
  uv.lock
  preregistration/
    confirmatory.yaml
  configs/
    problems/
    generators/
    evaluators/
  schemas/
  src/
  tests/
  data/
    README.md
    manifests/
  results/
    pilot/
    confirmatory/
    paper/
      summary.csv
      coverage_curves.csv
      operational_coverage.png
      operational_coverage_table.tex
      provenance.json
~~~

The analysis implementation carries an exact, tested copy of Part 1's reference
certificate. The released reference implementation is
[operational_certificate.py](theory/part1/operational_certificate.py).

## 11. Manuscript integration

For every manuscript release:

1. Add it at **external/ads-llm-resolvability-study**.
2. Pin the exact release commit in this repository.
3. Add a root build target that runs its validation and imports only
   **results/paper/**.
4. Record the submodule commit in the Part 1 appendix and empirical-results
   caption.
5. Keep the manuscript build usable without provider credentials and without
   rerunning model calls.
6. Package copied paper artifacts together with their provenance manifest for
   journal submission.

The private manuscript workspace integrates this repository as a submodule:

~~~text
git submodule add \
  git@github.com:genlayerlabs/ads-reproducibility.git \
  external/ads-llm-resolvability-study
~~~

The submodule commit, preregistration hash, data-manifest hash, and analysis
commit must all appear in the generated provenance file.

## 12. Acceptance gates

The empirical section may be injected into Part 1 only if:

1. The preregistration predates every confirmatory model call.
2. Generator and evaluator sampling match the declared populations.
3. The frozen matrix passes completeness and deterministic-aggregation checks.
4. All primary parameters match the preregistration exactly.
5. The analysis is reproducible from the pinned submodule commit.
6. Failures and exclusions are accounted for under the predeclared rule.
7. The paper reports a failed certificate as inconclusive rather than
   re-tuning the design on the same data.
8. Claims about external correctness are supported by independent labels.

Passing these gates makes the empirical result auditable. It does not turn a
declared LLM population into a universal population, and it does not make
coordination equivalent to truth.
