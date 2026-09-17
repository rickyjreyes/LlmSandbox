# Recursive Drift Instrumentation

This branch adds a non-invasive research layer for measuring recursive state drift in LLM-driven agents without changing the base simulator files.

## Research question

Can internally generated agent state changes propagate and reinforce across repeated decisions and inter-agent communication without an external directive specifying the outcome, and do those dynamics change predictably when propagation or correction channels are altered?

The research layer measures:

- persistent memory updates
- short- and long-term goal changes
- relationship-state changes
- agent-to-agent speech
- recurrent amplification versus correction
- positive versus negative relationship dynamics
- candidate negative/risk-label events
- explicit developer interventions
- operational RCA-style correction/amplification metrics

## Clean baseline

```bash
python research/run_recursive_drift.py
```

Do not use the developer console or inspector message injection during a clean baseline. Logs are written to `research_logs/`.

Analyze the latest run:

```bash
python research/analyze_recursive_drift.py
python research/check_rca_metrics.py
```

`check_rca_metrics.py` computes the magnitude-based operational bridge

```text
R_sim = total recurrent correction magnitude / total recurrent amplification magnitude
```

with separate positive and negative values. Initial `0 -> nonzero` relationship assignments and sign reversals are excluded from the primary ratio. `R_sim` is an empirical simulator diagnostic and is not asserted to be numerically identical to the theoretical RCA/WCT stabilization ratio without a separate derivation/calibration.

## Matched initial conditions

Generate one immutable starting world:

```bash
python research/create_world_template.py --seed 20260916
```

This writes a template under `research_worlds/` plus a manifest containing the template hash and entity map. Reuse the exact template file for every matched condition. The run wrapper copies it to `research_runtime/` before loading, so autosave and exit-save cannot mutate the source template.

Example matched baseline:

```bash
python research/run_recursive_drift.py \
  --world-template research_worlds/world_seed_20260916.json \
  --run-label baseline \
  --seed 20260916 \
  --max-decisions 150
```

## Experimental controls

The wrapper supports explicit causal ablations without modifying the simulator source:

```text
--memory-feedback off
    Hides stored memories from subsequent NPC prompts.

--communication off
    Preserves visible speech actions for instrumentation but blocks speech from
    entering the world event stream or the target agent's memory.

--relationship-feedback off
    Hides stored numeric relationship state from subsequent NPC prompts while
    retaining it for measurement.

--correction-gain GAMMA
    Applies an explicit restoring treatment after each native LLM decision:
        r_final = (1 - GAMMA) * r_native
    with 0 <= GAMMA <= 1.

--short-term-mem-limit N
    Overrides the rolling short-term memory horizon.

--periodic-check MINUTES
    Overrides the moving-agent periodic LLM decision interval.

--perception-radius TILES
    Overrides the simulator perception radius.

--model MODEL_NAME
    Overrides the configured local LLM for model-independence tests.

--max-decisions N
    Automatically closes the simulator after N valid LLM decisions.
```

Every run logs the selected experiment settings.

## Automated matched matrix

Run the four core causal conditions against the exact same starting world:

```bash
python research/run_experiment_matrix.py \
  --world-template research_worlds/world_seed_20260916.json \
  --conditions core \
  --replicates 1 \
  --max-decisions 150
```

`core` runs:

1. baseline
2. memory feedback off
3. communication off
4. relationship feedback off

For a restoring-gain sweep:

```bash
python research/run_experiment_matrix.py \
  --world-template research_worlds/world_seed_20260916.json \
  --conditions gamma \
  --replicates 3 \
  --max-decisions 150
```

The gamma suite runs baseline plus:

```text
gamma = 0.05, 0.10, 0.20, 0.40, 0.80
```

For the larger sensitivity suite:

```bash
python research/run_experiment_matrix.py \
  --world-template research_worlds/world_seed_20260916.json \
  --conditions all \
  --replicates 3 \
  --max-decisions 150
```

The extended conditions also vary short-term memory horizon, periodic decision interval, and perception radius.

A custom subset can be selected with comma-separated names, for example:

```bash
python research/run_experiment_matrix.py \
  --world-template research_worlds/world_seed_20260916.json \
  --conditions baseline,no_memory,no_relationship_feedback,gamma_020 \
  --replicates 5 \
  --max-decisions 200
```

On systems where pygame's dummy video driver works, add `--headless` for unattended sequential runs. If the dummy driver fails locally, omit it; automatic stopping still works.

## Matrix outputs

Each matrix creates `research_results/matrix_<UTC timestamp>/` containing per-run analysis plus:

```text
run_metrics.csv
matrix_summary.json
```

The aggregate summary reports means and medians for:

- `R_sim`
- `R_positive`
- `R_negative`
- recurrent amplification magnitude
- correction magnitude
- negative amplification magnitude
- negative correction magnitude

Replicate seeds are matched across conditions. The exact world template is also matched. LLM generation can still be stochastic, which is why repeated runs are required for inference.

## Interpretation priorities

The strongest next tests are:

1. **Repeatability:** determine whether `R_sim < 1` persists across independent runs.
2. **Negative/positive asymmetry:** test whether `R_negative < R_positive` is reproducible.
3. **Memory ablation:** determine whether removing memory feedback reduces recurrent amplification.
4. **Relationship-feedback ablation:** determine whether hiding persistent relationship state reduces recurrence.
5. **Communication ablation:** separate within-agent recursion from inter-agent propagation.
6. **Correction-gain sweep:** test whether increasing explicit restoration moves the system through an operational boundary near `R_sim = 1` and suppresses persistent amplification.
7. **Model independence:** repeat the same matched protocol across locally available models.

## Scientific cautions

- A single trajectory is a case study, not a population-level result.
- Positive and negative amplification must be reported separately.
- Rejected malformed/self relationship proposals are schema-output errors; they are logged and blocked before state mutation by the research wrapper.
- The operational `R_sim` is not the theoretical RCA/WCT `R` unless a formal mapping from simulator observables to the theoretical terms is derived and validated.
- Matched templates control the initial world, but local LLM generation may remain nondeterministic.
