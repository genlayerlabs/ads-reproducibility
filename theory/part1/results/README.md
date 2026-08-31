# Published operational Monte Carlo results

These files are generated artifacts underlying Table 1 and Figure 4 of Part 1.
They must not be edited by hand.

- **operational_monte_carlo_summary.csv** contains one row per matrix-size and
  dependence scenario, including every statistical parameter, seed,
  simultaneous-violation count, target-certification rate, and the lower-bound
  summary at the largest candidate panel.
- **operational_monte_carlo_curves.csv** contains the true coverage and the
  mean, 5th percentile, and 95th percentile of the certified lower bound for
  every candidate panel size.
- **adversarial_security_margin_floors.csv** contains the exact post-corruption
  clarity required by every implemented GenLayer panel size at pointwise error
  tolerances $10^{-2}$, $10^{-3}$, and $10^{-6}$. It is generated from exact
  binomial tails, not Monte Carlo.

Regenerate the CSV files and publication figure from the repository root with:

~~~text
nix-shell --run 'make theory-monte-carlo'
~~~

The paper study uses 2,000 repetitions per scenario and base seed 5510.
Scenario-specific seeds are recorded in the summary CSV. The simulation is a
calibration and power diagnostic under a known synthetic population; it is not
an empirical claim about real LLMs.

Regenerate the non-adaptive corruption figure and margin table with:

~~~text
nix-shell --run 'make theory-security-curves'
~~~

Those curves use the large-population i.i.d. benchmark and the panel schedule
pinned in the source script. A deployed-network claim additionally requires an
active-set and stake snapshot plus the actual without-replacement selection law.
