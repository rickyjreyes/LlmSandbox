# Recursive Drift Instrumentation

This branch adds a non-invasive research layer for measuring recursive state drift in LLM-driven agents without changing the base simulator's decision logic.

## Research question

Can internally generated agent state changes propagate and reinforce across repeated decisions and inter-agent communication without an external directive specifying the outcome?

The instrumentation focuses on:

- persistent memory updates
- short- and long-term goal changes
- relationship-state changes
- agent-to-agent speech
- repeated amplification or correction of relationship states
- possible negative/risk label propagation
- explicit developer interventions, which are logged so contaminated runs can be identified

## Why wrapper instrumentation

`run_recursive_drift.py` monkey-patches the scheduler at runtime. The original simulator files remain unchanged. This makes it easier to compare the research run against the upstream behavior and reduces the risk of coding the expected result into the simulator.

## Run a clean baseline

From the repository root:

```bash
python research/run_recursive_drift.py
```

For a baseline run:

1. Start a fresh world if possible.
2. Do not type developer-console commands.
3. Do not inject messages through the inspector.
4. Let agents interact normally.
5. Close the simulator after the desired observation period.

Logs are written to:

```text
research_logs/recursive_drift_<UTC timestamp>.jsonl
```

Every decision record contains pre/post snapshots of:

- mood
- goals
- relationship values
- memory counts and recent memories
- position
- the validated LLM result
- applied relationship changes

Speech and attack actions are logged separately. Developer commands and inspector interruptions are logged as `external_intervention` events.

## Analyze a run

Analyze the latest run:

```bash
python research/analyze_recursive_drift.py
```

Or provide a specific JSONL file:

```bash
python research/analyze_recursive_drift.py research_logs/recursive_drift_YYYYMMDDTHHMMSSZ.jsonl
```

Outputs are written by default to:

```text
research_results/latest/
```

Files:

- `summary.json`
- `relationship_trajectory.csv`
- `label_events.csv`

## Descriptive metrics

Each relationship update is classified as:

- `amplification`: absolute relationship magnitude moves farther from neutral
- `correction`: absolute relationship magnitude moves toward neutral
- `lateral`: magnitude is effectively unchanged

The analyzer reports:

```text
R_emp = correction_count / amplification_count
```

This is only a descriptive empirical diagnostic. It is **not** asserted to be numerically equivalent to any theoretical RCA/WCT stabilization ratio.

## Candidate label events

The first-pass analyzer flags memory updates and speech containing terms such as:

- dangerous
- threat
- hostile
- suspicious
- untrustworthy
- enemy
- attack
- kill
- fear
- deception / lying / betrayal

This is intentionally simple. It is meant to locate candidate episodes for detailed review, not to determine semantic truth automatically.

## Strong evidence pattern

A particularly useful event chain would be:

```text
initial weak inference
  -> persistent memory
  -> relationship-state change
  -> communication to another agent
  -> receiving agent stores the information
  -> receiving agent changes state or behavior
  -> target reacts to the changed behavior
  -> subsequent agents treat the reaction as confirming evidence
  -> original state/label strengthens
```

The strongest baseline evidence is a chain like this occurring with `external_intervention_count == 0`.

## Controls to add next

For causal tests, compare repeated runs under matched initial conditions:

1. normal persistent memory
2. persistent memory disabled
3. agent-to-agent communication disabled
4. relationship feedback disabled
5. reduced memory horizon
6. verification/correction step before memory persistence

The central experimental question is whether recursive drift changes systematically when the propagation or correction pathway is altered.
