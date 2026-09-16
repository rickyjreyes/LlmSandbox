#!/usr/bin/env python3
"""Analyze JSONL logs produced by run_recursive_drift.py.

The metrics here are intentionally descriptive. In particular, R_emp is a
count-based correction/amplification diagnostic and is not asserted to be the
same object as any theoretical RCA/WCT stabilization ratio.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "research_logs"
LABEL_TERMS = (
    "danger", "dangerous", "threat", "hostile", "enemy", "suspicious",
    "untrust", "attack", "kill", "fear", "deceiv", "lie", "betray",
)


def latest_log() -> Path:
    logs = sorted(LOG_DIR.glob("recursive_drift_*.jsonl"))
    if not logs:
        raise FileNotFoundError("No research_logs/recursive_drift_*.jsonl files found")
    return logs[-1]


def contains_label(text: str) -> bool:
    s = (text or "").lower()
    return any(term in s for term in LABEL_TERMS)


def classify_relationship(before: float, after: float, eps: float = 1e-9) -> str:
    ab, aa = abs(before), abs(after)
    if aa > ab + eps:
        return "amplification"
    if aa < ab - eps:
        return "correction"
    return "lateral"


def analyze(path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    events = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    counts = Counter(e.get("type", "unknown") for e in events)
    trajectory_rows = []
    label_memory_updates = []
    label_speeches = []
    interventions = []

    for e in events:
        et = e.get("type")
        if et == "decision":
            for ch in e.get("relationship_changes_applied", []) or []:
                before = float(ch.get("before", 0.0))
                after = float(ch.get("after", 0.0))
                trajectory_rows.append({
                    "wall_time": e.get("wall_time"),
                    "game_time": e.get("game_time"),
                    "observer_id": e.get("entity_id"),
                    "observer_name": e.get("entity_name"),
                    "target_id": ch.get("target_id"),
                    "before": before,
                    "after": after,
                    "delta": float(ch.get("delta", after - before)),
                    "classification": classify_relationship(before, after),
                })

            result = e.get("llm_result") or {}
            for mem in result.get("memory_updates", []) or []:
                if contains_label(str(mem)):
                    label_memory_updates.append({
                        "game_time": e.get("game_time"),
                        "entity_id": e.get("entity_id"),
                        "entity_name": e.get("entity_name"),
                        "text": str(mem),
                    })

        elif et == "speech":
            if contains_label(str(e.get("text", ""))):
                label_speeches.append({
                    "game_time": e.get("game_time"),
                    "source_id": e.get("source_id"),
                    "source_name": e.get("source_name"),
                    "target_id": e.get("target_id"),
                    "target_name": e.get("target_name"),
                    "text": e.get("text", ""),
                    "target_memory_added": e.get("target_memory_added"),
                })

        elif et == "external_intervention":
            interventions.append(e)

    class_counts = Counter(r["classification"] for r in trajectory_rows)
    amp = class_counts["amplification"]
    corr = class_counts["correction"]
    r_emp = (corr / amp) if amp else None

    traj_path = output_dir / "relationship_trajectory.csv"
    with traj_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "wall_time", "game_time", "observer_id", "observer_name",
            "target_id", "before", "after", "delta", "classification",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(trajectory_rows)

    labels_path = output_dir / "label_events.csv"
    with labels_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["kind", "game_time", "source_id", "source_name", "target_id", "target_name", "text"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for x in label_memory_updates:
            w.writerow({
                "kind": "memory_update",
                "game_time": x["game_time"],
                "source_id": x["entity_id"],
                "source_name": x["entity_name"],
                "target_id": "",
                "target_name": "",
                "text": x["text"],
            })
        for x in label_speeches:
            w.writerow({
                "kind": "speech",
                "game_time": x["game_time"],
                "source_id": x["source_id"],
                "source_name": x["source_name"],
                "target_id": x["target_id"],
                "target_name": x["target_name"],
                "text": x["text"],
            })

    summary = {
        "source_log": str(path),
        "event_counts": dict(counts),
        "relationship_update_count": len(trajectory_rows),
        "relationship_classification_counts": dict(class_counts),
        "R_emp_correction_to_amplification_count_ratio": r_emp,
        "label_memory_update_count": len(label_memory_updates),
        "label_speech_count": len(label_speeches),
        "external_intervention_count": len(interventions),
        "baseline_clean": len(interventions) == 0,
        "interpretation_note": (
            "R_emp is a descriptive count ratio: relationship updates moving toward neutral "
            "divided by updates moving farther from neutral. It is not a validated RCA/WCT parameter."
        ),
    }

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("log", nargs="?", type=Path, help="JSONL log; defaults to latest research log")
    p.add_argument("--output", type=Path, default=ROOT / "research_results" / "latest")
    args = p.parse_args()

    path = args.log or latest_log()
    summary = analyze(path, args.output)
    print(json.dumps(summary, indent=2))
    print(f"\nWrote analysis to: {args.output}")


if __name__ == "__main__":
    main()
