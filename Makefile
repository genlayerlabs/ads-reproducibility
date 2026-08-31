UV ?= uv
RUN_ID ?= pilot-current
STAGE1_RUN_ID ?= stage1-current
STAGE2_RUN_ID ?= stage2-current
SMOKE_RUN_ID ?= smoke-current
FAMILY_SMOKE_RUN_ID ?= family-smoke-current
STRESS_RUN_ID ?= stress-current
MANIFEST ?= preregistration/pilot.yaml

.PHONY: sync validate test test-empirical test-theory lint probe smoke-one smoke-two smoke smoke-families stress-families pilot analyze paper-artifacts refresh-release manifest-hash theory-figures theory-monte-carlo theory-security-curves

sync:
	$(UV) sync --frozen

validate:
	$(UV) run --frozen ads-study validate --manifest $(MANIFEST)

test: test-empirical test-theory

test-empirical:
	$(UV) run --frozen pytest

test-theory:
	nix develop ./theory/part1 --command env PYTHONPATH=theory/part1 \
		python -m unittest discover -s theory/part1 -p 'test_*.py'

lint:
	ruff format --check src tests
	ruff check src tests

probe:
	$(UV) run --frozen ads-study probe --manifest $(MANIFEST)

smoke-one:
	$(UV) run --frozen ads-study run --manifest $(MANIFEST) \
		--run-id $(STAGE1_RUN_ID) --limit-items 1 --limit-rows 1 --confirm-spend

smoke-two:
	$(UV) run --frozen ads-study run --manifest $(MANIFEST) \
		--run-id $(STAGE2_RUN_ID) --limit-items 2 --limit-rows 1 --confirm-spend

smoke:
	$(UV) run --frozen ads-study run --manifest $(MANIFEST) \
		--run-id $(SMOKE_RUN_ID) --limit-items 2 --limit-rows 2 --confirm-spend

smoke-families:
	$(UV) run --frozen ads-study run --manifest $(MANIFEST) \
		--run-id $(FAMILY_SMOKE_RUN_ID) --limit-items 1 \
		--one-row-per-configuration --confirm-spend

stress-families:
	$(UV) run --frozen ads-study run --manifest $(MANIFEST) \
		--run-id $(STRESS_RUN_ID) --limit-items 3 \
		--one-row-per-configuration --confirm-spend

pilot:
	@test "$(CONFIRM_LIVE_PILOT)" = "YES" || \
		(echo "Refusing 2,000 live calls: set CONFIRM_LIVE_PILOT=YES"; exit 2)
	$(UV) run --frozen ads-study run --manifest $(MANIFEST) \
		--run-id $(RUN_ID) --confirm-spend

analyze:
	$(UV) run --frozen ads-study analyze --manifest $(MANIFEST) --run-id $(RUN_ID)

paper-artifacts:
	$(UV) run --frozen ads-study paper-artifacts --manifest $(MANIFEST) --run-id $(RUN_ID)

refresh-release:
	$(UV) run --frozen ads-study refresh-release --manifest $(MANIFEST)

manifest-hash:
	$(UV) run --frozen ads-study hash-manifest --manifest $(MANIFEST)

theory-figures:
	nix develop ./theory/part1 --command python \
		theory/part1/convergence_validation.py
	nix develop ./theory/part1 --command python \
		theory/part1/adversarial_security_profiles.py

theory-monte-carlo:
	nix develop ./theory/part1 --command python \
		theory/part1/operational_monte_carlo.py \
		--paper-study --repetitions 2000

theory-security-curves:
	nix develop ./theory/part1 --command python \
		theory/part1/adversarial_security_profiles.py
