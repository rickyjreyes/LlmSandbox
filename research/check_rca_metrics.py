#!/usr/bin/env python3
"""Compute operational propagation/correction metrics from LlmSandbox trajectories.

This script provides an empirical bridge to RCA-style propagation-vs-correction
calculations. It does NOT assert numerical identity with any theoretical RCA/WCT
parameter. Initial 0->nonzero relationship assignments and sign reversals are
reported but excluded from the primary recurrent amplification/correction ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORY = ROOT / "research_results" / "latest" / "relationship_trajectory.csv"
DEFAULT_OUTPUT = ROOT / "research_results" / "latest" / "rca_metrics.json"


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python research/analyze_recursive_drift.py"
        )
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def magnitude_change(before: float, after: float) -> float:
    return abs(after) - abs(before)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = p.parse_args()

    rows = load_rows(args.trajectory)

    amplification_magnitude = 0.0
    correction_magnitude = 0.0
    positive_amplification_magnitude = 0.0
    negative_amplification_magnitude = 0.0
    correction_from_positive_magnitude = 0.0
    correction_from_negative_magnitude = 0.0

    amplification_events = 0
    correction_events = 0
    sign_reversals = 0
    initial_assignments = 0

    times = []
    pair_metrics = defaultdict(lambda: {
        "amplification_magnitude": 0.0,
        "correction_magnitude": 0.0,
        "amplification_events": 0,
        "correction_events": 0,
    })

    for row in rows:
        cls = row.get("classification", "")
        before = float(row["before"])
        after = float(row["after"])
        t = row.get("game_time")
        if t not in (None, ""):
            times.append(float(t))

        if cls.startswith("initial_"):
            initial_assignments += 1
            continue
        if cls == "sign_reversal":
            sign_reversals += 1
            continue

        dmag = magnitude_change(before, after)
        pair = f'{row.get("observer_name")}->{row.get("target_id")}'

        if cls in ("amplification_positive", "amplification_negative"):
            mag = max(0.0, dmag)
            amplification_events += 1
            amplification_magnitude += mag
            pair_metrics[pair]["amplification_magnitude"] += mag
            pair_metrics[pair]["amplification_events"] += 1

            if cls == "amplification_positive":
                positive_amplification_magnitude += mag
            else:
                negative_amplification_magnitude += mag

        elif cls == "correction_toward_neutral":
            mag = max(0.0, -dmag)
            correction_events += 1
            correction_magnitude += mag
            pair_metrics[pair]["correction_magnitude"] += mag
            pair_metrics[pair]["correction_events"] += 1

            if before > 0:
                correction_from_positive_magnitude += mag
            elif before < 0:
                correction_from_negative_magnitude += mag

    r_count = (
        correction_events / amplification_events
        if amplification_events else None
    )
    r_magnitude = (
        correction_magnitude / amplification_magnitude
        if amplification_magnitude > 0 else None
    )
    r_negative = (
        correction_from_negative_magnitude / negative_amplification_magnitude
        if negative_amplification_magnitude > 0 else None
    )
    r_positive = (
        correction_from_positive_magnitude / positive_amplification_magnitude
        if positive_amplification_magnitude > 0 else None
    )

    span = (max(times) - min(times)) if len(times) >= 2 else None
    amp_rate = (
        amplification_magnitude / span
        if span is not None and span > 0 else None
    )
    corr_rate = (
        correction_magnitude / span
        if span is not None and span > 0 else None
    )

    pair_out = {}
    for pair, m in sorted(pair_metrics.items()):
        a = m["amplification_magnitude"]
        c = m["correction_magnitude"]
        m = dict(m)
        m["R_sim_magnitude"] = (c / a) if a > 0 else None
        pair_out[pair] = m

    summary = {
        "source_trajectory": str(args.trajectory),
        "initial_assignment_count_excluded": initial_assignments,
        "sign_reversal_count_excluded": sign_reversals,
        "recurrent_amplification_event_count": amplification_events,
        "correction_event_count": correction_events,
        "R_event_count_correction_to_amplification": r_count,
        "recurrent_amplification_magnitude": amplification_magnitude,
        "correction_magnitude": correction_magnitude,
        "R_sim_magnitude_correction_to_amplification": r_magnitude,
        "positive_amplification_magnitude": positive_amplification_magnitude,
        "negative_amplification_magnitude": negative_amplification_magnitude,
        "correction_from_positive_magnitude": correction_from_positive_magnitude,
        "correction_from_negative_magnitude": correction_from_negative_magnitude,
        "R_sim_positive_correction_to_amplification": r_positive,
        "R_sim_negative_correction_to_amplification": r_negative,
        "observation_span_game_minutes": span,
        "amplification_magnitude_rate_per_game_minute": amp_rate,
        "correction_magnitude_rate_per_game_minute": corr_rate,
        "pair_metrics": pair_out,
        "interpretation_note": (
            "R_sim_magnitude is an operational simulator diagnostic: total recurrent "
            "relationship-magnitude correction divided by total recurrent relationship-"
            "magnitude amplification. Initial assignments and sign reversals are excluded. "
            "It is not numerically identified with the theoretical RCA/WCT stabilization ratio "
            "without an explicit calibration/derivation mapping simulator observables to the "
            "theoretical terms."
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote RCA bridge metrics to: {args.output}")


if __name__ == "__main__":
    main()
