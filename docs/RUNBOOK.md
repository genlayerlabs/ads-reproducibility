# Router execution runbook

This runbook executes the 50-item diagnostic pilot against the OpenAI-compatible
Unhardcoded endpoint at `https://router.ygr.ai/v1`. It does not authorize a
confirmatory claim; that requires the independent workload and frozen manifest
described in [PROTOCOL.md](../PROTOCOL.md).

## 1. Obtain the correct credential

Use a dedicated **active router consumer key** with access to every `family:`
route in `preregistration/pilot.yaml`. A provider credential such as an
OpenRouter key is not a substitute: the public router authenticates its own
consumer bearer tokens before selecting a provider.

Put the value in an ignored local file. Both `LLM_API_KEY` and
`UNHARDCODED_API_KEY` are accepted:

```bash
cp .env.example .env.local
# Edit only .env.local and set one consumer-key variable.
```

Alternatively export `LLM_API_KEY` in the invoking shell. Never paste the key
into YAML, a command argument, a run identifier, Git, an issue, or a paper
artifact.

## 2. Install and validate offline

```bash
nix-shell --run 'uv sync --frozen'
nix-shell --run 'make validate test lint'
```

These commands make no model calls.

## 3. Probe the router without inference

```bash
nix-shell --run 'make probe'
```

Proceed only if:

- health and authenticated `/v1/models` checks succeed;
- every configured family is available;
- the endpoint in the report is exactly `https://router.ygr.ai/v1`.

`caller_inactive` means the token is recognized but its router consumer is
disabled. `caller_auth` means it is not a valid consumer token. In either case,
activate or issue a dedicated consumer key in the router control plane; do not
fall back to a provider key.

## 4. Run staged live matrices

The pilot manifest permits at most three concurrent calls globally, at most one
active call per evaluator family, and has no artificial inter-call delay. Use
unique immutable identifiers and inspect each stage before continuing:

```bash
STAGE1_RUN_ID=stage1-YYYYMMDD-HHMM nix-shell --run 'make smoke-one'
RUN_ID=stage1-YYYYMMDD-HHMM nix-shell --run 'make validate analyze'

STAGE2_RUN_ID=stage2-YYYYMMDD-HHMM nix-shell --run 'make smoke-two'
RUN_ID=stage2-YYYYMMDD-HHMM nix-shell --run 'make validate analyze'

SMOKE_RUN_ID=smoke-YYYYMMDD-HHMM nix-shell --run 'make smoke'
RUN_ID=smoke-YYYYMMDD-HHMM nix-shell --run 'make validate analyze'

FAMILY_SMOKE_RUN_ID=family-smoke-YYYYMMDD-HHMM \
nix-shell --run 'make smoke-families'
RUN_ID=family-smoke-YYYYMMDD-HHMM nix-shell --run 'make validate analyze'

STRESS_RUN_ID=stress-YYYYMMDD-HHMM nix-shell --run 'make stress-families'
RUN_ID=stress-YYYYMMDD-HHMM nix-shell --run 'make validate analyze'
```

The stages contain one cell, two questions with one evaluator row, and finally
two questions by two evaluator rows (four cells). The last stage sends one
question to one deterministic diagnostic row from each of the twenty-one declared
evaluator families. These diagnostic rows are outside the primary IID sample.
The coverage stage makes 21 calls; the load gate then makes three calls per
family, 63 total, with global concurrency capped at three and same-family
requests serialized. Use repeated `--configuration-id` options for smaller
reviewable batches when adding or repairing routes.
The live runner repeats the
authenticated catalog preflight before every stage. HTTP authentication and
request-configuration failures stop the run; timeouts and exhausted retryable
provider failures follow the preregistered conservative `REJECT` mapping. The
pilot allows three transport attempts and 4,096 completion tokens; the latter
prevents reasoning-model hidden tokens from truncating the required JSON object.

Check `data/runs/<run-id>/run.json` and the generated provider summary for:

- the expected number of completed cells and no malformed/missing cells;
- a populated, sanitized `x_router` decision trace;
- the intended served model families;
- plausible latency, token counts, and reported cost;
- no secret-like strings (`make validate` scans the artifacts).

The raw run directory is intentionally ignored by Git.

If a family-specific format problem appears, freeze the corrective prompt first
and retest only the affected routes with repeated `--configuration-id` options
together with `--one-row-per-configuration`. Do not continue to the full pilot
until every declared family has produced a parseable response under the frozen
request contract.

## 5. Run the complete diagnostic pilot

The matrix contains 50 items by 40 evaluator rows: 2,000 live calls. With three
workers and no artificial delay, wall time is governed by provider latency and
retry tails; use the completed 66-call load gate to estimate it. Review its cost,
error rate, and per-provider concurrency before continuing. Then run:

```bash
RUN_ID=pilot-YYYYMMDD \
CONFIRM_LIVE_PILOT=YES \
nix-shell --run 'make pilot'
```

Both `--confirm-spend` and `CONFIRM_LIVE_PILOT=YES` are enforced. Runs are
resumable by cell; reusing the same identifier with a changed manifest or
request payload is rejected.

Validate and generate results:

```bash
RUN_ID=pilot-YYYYMMDD nix-shell --run 'make validate analyze paper-artifacts'
```

Do not release `results/paper/` until the run has been reviewed and its
provenance file travels with every copied figure and table.

## 6. Confirmatory study

Do not promote the pilot matrix to confirmatory evidence. First create the
larger independently adjudicated workload, complete
`preregistration/confirmatory.template.yaml`, run the power analysis for
generator count `A` and evaluator rows `M`, hash and commit the manifest, and
create an immutable tag before the first provider call. A failed frozen
certificate is reported as inconclusive; it is never tuned on the same calls.
Confirmatory execution additionally requires a clean committed worktree and
`CONFIRM_LIVE_CONFIRMATORY=YES`.
