# Evaluator model population

The diagnostic pilot uses a frozen, uniform catalogue of 21 live-eligible
router family routes. Each route has population weight
`0.047619047619047616` (exactly `1/21` conceptually); the primary 40 evaluator
rows remain IID draws with replacement from that catalogue. Consequently, a
realized primary matrix need not contain every route. The separate
configuration-coverage smoke is diagnostic only.

The catalogue was checked against the authenticated
`https://router.ygr.ai/v1/models` endpoint on 2026-08-03. Live eligibility was
then established by the smoke, load-gate, and targeted retest runs
`family-smoke-32-20260803-01`, `family-retest-16-20260803-01`,
`format-retest-1024-20260803-01`, `glm-5-1-final-20260803-01`,
`stress-22x3-20260803-01`, and `stability-qwen-kimi-2x5-20260803-01`.
A retained route produced at least one parseable verdict with exact
`model_family`, a coherent `served_model_id`, and policy fingerprint
`887242138-1382858753`.

## Frozen live-eligible routes

- OpenAI: `gpt-5.5`, `gpt-5.6-luna`, and `gpt-5.6-terra`.
- Anthropic: `claude-opus-4-8`, `claude-opus-5`, and
  `claude-sonnet-4-6`.
- Google: `gemini-3-flash-preview` and `gemini-3.1-pro-preview`.
- xAI: `grok-4.5`.
- Moonshot: `kimi-k2.6` and `kimi-k3`.
- Alibaba: `qwen3.6-plus`, `qwen3.7-max`, and `qwen3.7-plus`.
- Zhipu: `glm-4.6`, `glm-5.1`, and `glm-5.2`.
- DeepSeek: `deepseek-v4-flash` and `deepseek-v4-pro`.
- Other: `muse-spark-1.1` and `mercury-2`.

## Catalogue-present routes excluded by the live screen

These routes remain visible in `/v1/models` but either did not execute a valid
model response in staged testing or failed the subsequent availability gate,
so catalogue presence alone is not treated as eligibility:

- `qwen3.5-397b-a17b`: only 3/8 successful calls across the load gate and
  targeted stability retest; repeated bad gateway 502 responses would map
  infrastructure failures to `REJECT` in the primary matrix.
- `gpt-5.5-pro`: repeated timeout/rate-limit 429.
- `gpt-5.6-sol`: repeated `no_candidates` 503.
- `claude-fable-5`: repeated gateway timeout 504.
- `claude-opus-4-5`: `no_candidates`/gateway timeout.
- `claude-opus-4-6`: repeated timeout/rate-limit 429.
- `claude-opus-4-7`: repeated timeout/rate-limit 429.
- `gemini-3.1-pro`: repeated gateway timeout 504.
- `grok-4.1-fast`: repeated timeout/rate-limit 429.
- `hy3`: repeated bad gateway 502.
- `nemotron-3-nano-30b-a3b`: repeated bad gateway 502.

## Requested names not silently substituted

- Claude Mythos Preview: no matching family route.
- Gemini 3 Pro: no matching text family; only image variants are listed.
- Seed 2.0 Pro: no Pro route; only `seed-2.0-mini` and `seed-2.0-lite` are
  listed.
- Grok-4 Heavy: no matching family route. Standard and multi-agent Grok routes
  are not assumed equivalent to the requested product name.
- DeepSeek V4.1 Pro: no matching route. It is not conflated with
  `deepseek-v4-pro`, which represents the separately requested V4 Pro Max
  entry.

Any later substitution changes the declared evaluator law and therefore
requires a new manifest hash and a fresh run identifier.
