#!/usr/bin/env python3
"""Run matched recursive-drift controls and aggregate RCA bridge metrics.

This orchestrator launches the existing pygame simulator sequentially. Each run
loads the same immutable world template, stops automatically after a requested
number of valid decisions, then runs the standard analyzer and RCA bridge.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
LOG_DIR = ROOT / "research_logs"
RESULTS_ROOT = ROOT / "research_results"

CORE_CONDITIONS = {
    "baseline": [],
    "no_memory": ["--memory-feedback", "off"],
    "no_communication": ["--communication", "off"],
    "no_relationship_feedback": ["--relationship-feedback", "off"],
}

GAMMA_CONDITIONS = {
    "gamma_005": ["--correction-gain", "0.05"],
    "gamma_010": ["--correction-gain", "0.10"],
    "gamma_020": ["--correction-gain", "0.20"],
    "gamma_040": ["--correction-gain", "0.40"],
    "gamma_080": ["--correction-gain", "0.80"],
}

EXTENDED_CONDITIONS = {
    "memory_horizon_1": ["--short-term-mem-limit", "1"],
    "memory_horizon_5": ["--short-term-mem-limit", "5"],
    "periodic_2": ["--periodic-check", "2"],
    "periodic_10": ["--periodic-check", "10"],
    "periodic_20": ["--periodic-check", "20"],
    "perception_5": ["--perception-radius", "5"],
    "perception_10": ["--perception-radius", "10"],
    "perception_40": ["--perception-radius", "40"],
}


def condition_catalog() -> dict[str, list[str]]:
    return {**CORE_CONDITIONS, **GAMMA_CONDITIONS, **EXTENDED_CONDITIONS}


def select_conditions(spec: str) -> list[str]:
    catalog = condition_catalog()
    if spec == "core":
        return list(CORE_CONDITIONS)
    if spec == "gamma":
        return ["baseline", *GAMMA_CONDITIONS]
    if spec == "all":
        return list(catalog)

    names = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [x for x in names if x not in catalog]
    if unknown:
        raise ValueError(
            f"Unknown conditions: {unknown}. Available: {', '.join(catalog)}"
        )
    return names


def latest_log_for_label(label: str) -> Path:
    matches = list(LOG_DIR.glob(f"recursive_drift_*_{label}.jsonl"))
    if not matches:
        raise FileNotFoundError(f"No log found for run label {label}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_or_none(values):
    vals = [float(v) for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def median_or_none(values):
    vals = [float(v) for v in values if v is not None]
    return statistics.median(vals) if vals else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--world-template", type=Path, required=True)
    p.add_argument("--conditions", default="core",
                   help="core, gamma, all, or comma-separated condition names")
    p.add_argument("--replicates", type=int, default=1)
    p.add_argument("--max-decisions", type=int, default=150)
    p.add_argument("--seed", type=int, default=20260916,
                   help="Replicate r uses seed + r; matched across conditions")
    p.add_argument("--model", default=None)
    p.add_argument("--headless", action="store_true",
                   help="Set SDL_VIDEODRIVER=dummy for unattended pygame runs")
    args = p.parse_args()

    if args.replicates <= 0:
        p.error("--replicates must be > 0")
    if args.max_decisions <= 0:
        p.error("--max-decisions must be > 0")

    template = args.world_template.resolve()
    if not template.exists():
        p.error(f"world template not found: {template}")

    try:
        conditions = select_conditions(args.conditions)
    except ValueError as ex:
        p.error(str(ex))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    matrix_dir = RESULTS_ROOT / f"matrix_{stamp}"
    matrix_dir.mkdir(parents=True, exist_ok=True)

    catalog = condition_catalog()
    rows = []
    total_runs = len(conditions) * args.replicates
    run_no = 0

    for rep in range(args.replicates):
        replicate_seed = args.seed + rep
        for condition in conditions:
            run_no += 1
            label = f"{condition}_rep{rep + 1:02d}_{stamp}"
            out_dir = matrix_dir / label
            out_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(RESEARCH / "run_recursive_drift.py"),
                "--run-label", label,
                "--world-template", str(template),
                "--max-decisions", str(args.max_decisions),
                "--seed", str(replicate_seed),
                *catalog[condition],
            ]
            if args.model:
                cmd.extend(["--model", args.model])

            env = os.environ.copy()
            if args.headless:
                env["SDL_VIDEODRIVER"] = "dummy"

            print(f"\n[{run_no}/{total_runs}] {condition}, replicate {rep + 1}")
            print(" ".join(cmd))
            proc = subprocess.run(cmd, cwd=ROOT, env=env)
            if proc.returncode != 0:
                raise SystemExit(
                    f"Run failed ({proc.returncode}): {condition}, replicate {rep + 1}"
                )

            log_path = latest_log_for_label(label)
            summary_path = out_dir / "summary.json"
            trajectory_path = out_dir / "relationship_trajectory.csv"
            rca_path = out_dir / "rca_metrics.json"

            subprocess.run([
                sys.executable,
                str(RESEARCH / "analyze_recursive_drift.py"),
                str(log_path),
                "--output", str(out_dir),
            ], cwd=ROOT, check=True)

            subprocess.run([
                sys.executable,
                str(RESEARCH / "check_rca_metrics.py"),
                "--trajectory", str(trajectory_path),
                "--output", str(rca_path),
            ], cwd=ROOT, check=True)

            summary = load_json(summary_path)
            rca = load_json(rca_path)
            rows.append({
                "condition": condition,
                "replicate": rep + 1,
                "seed": replicate_seed,
                "run_label": label,
                "log": str(log_path),
                "behaviorally_interpretable": summary.get("behaviorally_interpretable"),
                "decision_success_rate": summary.get("decision_success_rate"),
                "valid_decisions": summary.get("valid_decision_count"),
                "R_sim": rca.get("R_sim_magnitude_correction_to_amplification"),
                "R_positive": rca.get("R_sim_positive_correction_to_amplification"),
                "R_negative": rca.get("R_sim_negative_correction_to_amplification"),
                "amplification_magnitude": rca.get("recurrent_amplification_magnitude"),
                "correction_magnitude": rca.get("correction_magnitude"),
                "negative_amplification_magnitude": rca.get("negative_amplification_magnitude"),
                "negative_correction_magnitude": rca.get("correction_from_negative_magnitude"),
                "amplification_rate": rca.get("amplification_magnitude_rate_per_game_minute"),
                "correction_rate": rca.get("correction_magnitude_rate_per_game_minute"),
            })

    csv_path = matrix_dir / "run_metrics.csv"
    fields = list(rows[0]) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    aggregate = {}
    for condition in conditions:
        rs = [r for r in rows if r["condition"] == condition]
        aggregate[condition] = {
            "n": len(rs),
            "interpretable_n": sum(bool(r["behaviorally_interpretable"]) for r in rs),
            "R_sim_mean": mean_or_none(r["R_sim"] for r in rs),
            "R_sim_median": median_or_none(r["R_sim"] for r in rs),
            "R_positive_mean": mean_or_none(r["R_positive"] for r in rs),
            "R_positive_median": median_or_none(r["R_positive"] for r in rs),
            "R_negative_mean": mean_or_none(r["R_negative"] for r in rs),
            "R_negative_median": median_or_none(r["R_negative"] for r in rs),
            "amplification_magnitude_mean": mean_or_none(
                r["amplification_magnitude"] for r in rs
            ),
            "correction_magnitude_mean": mean_or_none(
                r["correction_magnitude"] for r in rs
            ),
            "negative_amplification_magnitude_mean": mean_or_none(
                r["negative_amplification_magnitude"] for r in rs
            ),
            "negative_correction_magnitude_mean": mean_or_none(
                r["negative_correction_magnitude"] for r in rs
            ),
        }

    matrix_summary = {
        "world_template": str(template),
        "conditions": conditions,
        "replicates": args.replicates,
        "max_decisions": args.max_decisions,
        "base_seed": args.seed,
        "model_override": args.model,
        "aggregate": aggregate,
        "interpretation_note": (
            "Matched conditions reuse the exact same world template. Replicate seeds are "
            "matched across conditions, but LLM generation may remain stochastic. R_sim is "
            "the operational relationship-magnitude correction/amplification diagnostic, "
            "not a numerically calibrated theoretical RCA/WCT parameter."
        ),
    }
    summary_out = matrix_dir / "matrix_summary.json"
    summary_out.write_text(json.dumps(matrix_summary, indent=2), encoding="utf-8")

    print("\n=== MATRIX COMPLETE ===")
    print(json.dumps(matrix_summary, indent=2))
    print(f"\nRun metrics:    {csv_path}")
    print(f"Matrix summary: {summary_out}")


if __name__ == "__main__":
    main()
