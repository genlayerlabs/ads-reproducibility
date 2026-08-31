# Equivalence Principles Dataset — LLM Resolvability Study (seed set)

Seed catalogue of decision problems for the `ads-llm-resolvability-study`
experiment: realistic equivalence principles of the kind an intelligent-contract
validator panel would be asked to decide. Its labels were assigned by
construction and are independent of panel consensus, but they are not
independently adjudicated external truth.

50 items across 13 use-case families, stratified by expected clarity.

## Schema (one JSON object per line, `dataset.jsonl`)

| Field | Meaning |
|---|---|
| `id` | Unique item id, `{family-prefix}-{nn}` |
| `family` | Use-case family (workload population component) |
| `family_weight` | Suggested mixture weight of the family in the workload distribution (weights sum to 1.0 across families) |
| `use_case` | Human-readable product context |
| `clarity_stratum` | Design stratum. The frozen schema value `ambiguous_adversarial` is displayed in the paper as `input-presentation stress`; it is not a claim that those units are ambiguous or that evaluators are Byzantine. |
| `decision_question` | What is being decided (for the reader; the validator prompt should use the principle) |
| `equivalence_principle` | The binary rule G, phrased as "ACCEPT iff ..." |
| `evidence` | Self-contained evidence available to the validator (no live retrieval needed) |
| `gold_label` | Construction label: `accept`, `reject`, or `unresolvable` |
| `defensible_labels` | Labels a competent honest judge could defend; equals `[gold_label]` for clear items, `["accept","reject"]` for unresolvable ones |
| `ambiguity_axis` | Named source of difficulty (null for clear items) |
| `notes` | Labeling rationale |

## Design decisions

**Binary rule (G).** Every principle is normalized to "ACCEPT iff ⟨condition⟩", so a validator's raw output maps to {ACCEPT, REJECT} with a trivial parser. Items never require a third output. An abstention-enabled arm would define a different mechanism and estimand and must be preregistered as a separate study.

**Clarity strata vs. construction labels.** Strata are a *design* attribute (what the item was built to be); labels are also construction metadata, not external measurements. They deliberately do not coincide for the schema stratum `ambiguous_adversarial`: 8 of its 12 items have a determinate construction label. The paper therefore displays this stratum as *input-presentation stress*. It contains injection, spoofing, tampering, rhetoric, and one crafted boundary case; it is not an evaluator-corruption model.

**Adversarial models.** Two families, matching the two threat models in the paper:
1. *Evidence-channel adversary* — injected instructions, fake authority claims, spoofed/tampered/fabricated sources inside the evidence (items: sla-04, pay-04, pms-04, pmo-04, ins-04, ref-04, hlt-03, grt-04, air-04, mod-04, bug-04).
2. *Boundary-crafting adversary* — instances optimized against the letter of the rule so that honest judges split (items: mkt-04, and the benign-ambiguous stratum read as the adversary's target zone).

**Construction labels.** `gold_label` is assigned by construction and documented in `notes`. Independent semantic-accuracy claims require a separate adjudication protocol; for example, a preregistered multi-annotator design with an explicit rule for disagreement. The current labels may describe construction agreement, but must not be called external truth.

**Fictional entities.** All names (Arvenia, Vardal FC, Ostemark, Belvara, GenProtocol, M-12b, ...) are fictional, and all evidence is self-contained. This removes dependence on training-data world knowledge and on live retrieval: the experiment measures rule application, not recall. If you later want a retrieval-in-the-loop arm, fork the objective families (pmo, ins) into real-entity versions.

**Family weights.** Suggested workload mixture reflecting expected on-chain volume for a GenLayer-like deployment (dispute/SLA/payment heavy, referee/halt/fingerprinting light). Treat them as a prior; the paper's per-α safe-workload percentages should be reported both under these weights and under uniform weights as a robustness check.

## Composition

- Families (weight): sla_enforcement (.13), payment_dispute (.13), marketplace_dispute (.12), prediction_market_objective (.10), parametric_insurance (.10), airdrop_task_eval (.08), prediction_market_subjective (.07), grant_scoring (.06), bug_bounty_severity (.06), content_moderation (.06), sports_referee (.04), model_fingerprinting (.03), emergency_halt (.02)
- Gold labels: 15 accept / 22 reject / 13 unresolvable
- Strata: 13 clear_accept / 13 clear_reject / 12 ambiguous_benign / 12 ambiguous_adversarial

## Recipe for scaling the population

The confirmatory matrix needs a larger N. To extend without diluting realism:

1. Keep the four-stratum structure per family; write clear items first, then perturb them into ambiguous ones by *removing exactly one specification* (rounding, source hierarchy, duration bound, aggregation rule, partial-performance clause). Each `ambiguity_axis` in this seed set is such a single removed specification — reuse the axis vocabulary.
2. Generate adversarial variants mechanically: take a clear item and inject one of the evidence-channel attack templates (validator-directed instruction, fake authority, lookalike source, reverted-but-cited transaction, fabricated track record, homoglyph obfuscation).
3. Keep evidence self-contained and numbers pre-digested but re-derivable, so wrong decisions are attributable to rule application rather than arithmetic or retrieval.
4. Hold out the seed set from any prompt-engineering iterations; pilot on generated variants only, per the pre-registration.
