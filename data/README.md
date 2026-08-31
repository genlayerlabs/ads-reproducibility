# Run data

`data/runs/<run-id>/` is created by the live runner and is gitignored. A run
contains its resolved manifest, evaluator-row design, request ledger, response
records, deterministic global votes, and hashes. Raw responses remain local or
move to immutable object storage; publication releases expose a derived matrix
and content hashes under the protocol's data contract.

Never place API keys, authorization headers, provider credentials, or `.env`
files here.

Repository-wide validation reports but skips explicitly non-complete runs;
validate one directly with `ads-study validate --run-id <id>`. Reusing the same
identifier resumes cells only when its manifest and request hashes still match.
