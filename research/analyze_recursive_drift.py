#!/usr/bin/env python3
"""Analyze JSONL logs produced by run_recursive_drift.py.

Metrics are descriptive. Initial relationship assignment is separated from
recurrent amplification. Relationship proposals rejected by the research
wrapper are reported separately from invalid relationship state that actually
reached the simulation.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "research_logs"

LABEL_PATTERNS = tuple(re.compile(p, re.IGNORECASE) for p in (
    r"\bdanger(?:ous)?\b",
    r"\bthreat(?:s|ened|ening)?\b",
    r"\bhostile\b",
    r"\b(?:enemy|enemies)\b",
    r"\bsuspicious\b",
    r"\buntrust(?:ed|worthy|ing)?\b",
    r"\battack(?:s|ed|ing)?\b",
    r"\bkill(?:s|ed|ing)?\b",
    r"\bfear(?:s|ed|ful)?\b",
    r"\bdeceiv(?:e|es|ed|ing)?\b",
    r"\b(?:lie|lies|lied|lying)\b",
    r"\bbetray(?:s|ed|ing|al)?\b",
))


def latest_log() -> Path:
    logs = sorted(LOG_DIR.glob("recursive_drift_*.jsonl"))
    if not logs:
        raise FileNotFoundError("No research_logs/recursive_drift_*.jsonl files found")
    return logs[-1]


def contains_label(text: str) -> bool:
    return any(p.search(text or "") for p in LABEL_PATTERNS)


def classify_relationship(before: float, after: float, existed_before: bool,
                          eps: float = 1e-9) -> str:
    if not existed_before and abs(before) <= eps:
        if after > eps:
            return "initial_positive"
        if after < -eps:
            return "initial_negative"
        return "lateral"

    if before * after < -(eps * eps):
        return "sign_reversal"

    ab, aa = abs(before), abs(after)
    if aa > ab + eps:
        return "amplification_positive" if after > 0 else "amplification_negative"
    if aa < ab - eps:
        return "correction_toward_neutral"
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

    known_ids = set()
    for e in events:
        if e.get("type") == "run_start":
            for ent in e.get("entities", []) or []:
                if isinstance(ent, dict) and ent.get("id"):
                    known_ids.add(str(ent["id"]))

    trajectory_rows = []
    rejected_proposal_rows = []
    invalid_applied_rows = []
    label_memory_updates = []
    label_speeches = []
    interventions = []

    decision_count = 0
    valid_decision_count = 0
    failed_decision_count = 0
    total_memory_updates = 0
    proposed_relationship_updates = 0
    accepted_relationship_proposals = 0
    action_counts = Counter()
    mood_counts = Counter()

    for e in events:
        et = e.get("type")
        if et == "decision":
            decision_count += 1
            wrapper_enforced = "llm_result_raw" in e
            raw_result = e.get("llm_result_raw", e.get("llm_result"))
            clean_result = e.get("llm_result")

            if clean_result is None:
                failed_decision_count += 1
                continue

            valid_decision_count += 1
            result = clean_result if isinstance(clean_result, dict) else {}
            raw_dict = raw_result if isinstance(raw_result, dict) else result

            mems = result.get("memory_updates", []) or []
            rels = result.get("relationship_changes", {}) or {}
            raw_rels = raw_dict.get("relationship_changes", {}) or {}
            acts = result.get("actions", []) or []

            total_memory_updates += len(mems) if isinstance(mems, list) else 0
            proposed_relationship_updates += len(raw_rels) if isinstance(raw_rels, dict) else 0
            accepted_relationship_proposals += len(rels) if isinstance(rels, dict) else 0

            for rej in e.get("rejected_relationship_changes", []) or []:
                rejected_proposal_rows.append({
                    "game_time": e.get("game_time"),
                    "observer_id": e.get("entity_id"),
                    "observer_name": e.get("entity_name"),
                    "target_id": rej.get("target_id"),
                    "delta": rej.get("delta"),
                    "reason": rej.get("reason", "wrapper_rejected"),
                })

            mood = result.get("mood")
            if mood:
                mood_counts[str(mood)] += 1

            if isinstance(acts, list):
                for action in acts:
                    if isinstance(action, dict):
                        action_counts[str(action.get("type", "unknown"))] += 1

            before_relationships = ((e.get("before") or {}).get("relationships") or {})
            for ch in e.get("relationship_changes_applied", []) or []:
                target_id = str(ch.get("target_id"))
                observer_id = str(e.get("entity_id"))

                # New logs have already passed runtime schema enforcement. For
                # legacy logs only, perform a conservative post-hoc validity check.
                invalid_reason = None
                if not wrapper_enforced:
                    if target_id == observer_id:
                        invalid_reason = "self_target"
                    elif known_ids and target_id not in known_ids:
                        invalid_reason = "unknown_target"

                if invalid_reason:
                    invalid_applied_rows.append({
                        "game_time": e.get("game_time"),
                        "observer_id": observer_id,
                        "observer_name": e.get("entity_name"),
                        "target_id": target_id,
                        "delta": ch.get("delta"),
                        "reason": invalid_reason,
                    })
                    continue

                before = float(ch.get("before", 0.0))
                after = float(ch.get("after", 0.0))
                existed_before = bool(ch.get(
                    "existed_before", target_id in before_relationships
                ))
                classification = classify_relationship(before, after, existed_before)
                trajectory_rows.append({
                    "wall_time": e.get("wall_time"),
                    "game_time": e.get("game_time"),
                    "observer_id": observer_id,
                    "observer_name": e.get("entity_name"),
                    "target_id": target_id,
                    "before": before,
                    "after": after,
                    "delta": float(ch.get("delta", after - before)),
                    "existed_before": existed_before,
                    "classification": classification,
                })

            for mem in mems if isinstance(mems, list) else []:
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
    recurrent_amp = (
        class_counts["amplification_positive"]
        + class_counts["amplification_negative"]
    )
    corr = class_counts["correction_toward_neutral"]
    r_emp = (corr / recurrent_amp) if recurrent_amp else None

    decision_success_rate = (
        valid_decision_count / decision_count if decision_count else None
    )
    baseline_clean = len(interventions) == 0
    schema_output_clean = len(rejected_proposal_rows) == 0
    state_schema_clean = len(invalid_applied_rows) == 0
    proposal_accounting_delta = (
        proposed_relationship_updates
        - accepted_relationship_proposals
        - len(rejected_proposal_rows)
    )
    behaviorally_interpretable = bool(
        baseline_clean
        and state_schema_clean
        and proposal_accounting_delta == 0
        and decision_count > 0
        and decision_success_rate is not None
        and decision_success_rate >= 0.95
    )

    traj_path = output_dir / "relationship_trajectory.csv"
    with traj_path.open("w", encoding="utf-8", newline="") as f:
        fields = [
            "wall_time", "game_time", "observer_id", "observer_name",
            "target_id", "before", "after", "delta", "existed_before",
            "classification",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(trajectory_rows)

    rejected_path = output_dir / "rejected_relationship_proposals.csv"
    with rejected_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["game_time", "observer_id", "observer_name", "target_id", "delta", "reason"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rejected_proposal_rows)

    invalid_applied_path = output_dir / "invalid_applied_relationship_updates.csv"
    with invalid_applied_path.open("w", encoding="utf-8", newline="") as f:
        fields = ["game_time", "observer_id", "observer_name", "target_id", "delta", "reason"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(invalid_applied_rows)

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
        "decision_count": decision_count,
        "valid_decision_count": valid_decision_count,
        "failed_decision_count": failed_decision_count,
        "decision_success_rate": decision_success_rate,
        "behaviorally_interpretable": behaviorally_interpretable,
        "total_memory_updates": total_memory_updates,
        "proposed_relationship_update_count": proposed_relationship_updates,
        "accepted_relationship_proposal_count": accepted_relationship_proposals,
        "rejected_relationship_proposal_count": len(rejected_proposal_rows),
        "relationship_proposal_accounting_delta": proposal_accounting_delta,
        "schema_output_clean": schema_output_clean,
        "invalid_relationship_state_update_count": len(invalid_applied_rows),
        "state_schema_clean": state_schema_clean,
        "action_counts": dict(action_counts),
        "mood_counts": dict(mood_counts),
        "relationship_update_count": len(trajectory_rows),
        "relationship_classification_counts": dict(class_counts),
        "initial_positive_relationship_count": class_counts["initial_positive"],
        "initial_negative_relationship_count": class_counts["initial_negative"],
        "recurrent_amplification_count": recurrent_amp,
        "recurrent_positive_amplification_count": class_counts["amplification_positive"],
        "recurrent_negative_amplification_count": class_counts["amplification_negative"],
        "correction_toward_neutral_count": corr,
        "sign_reversal_count": class_counts["sign_reversal"],
        "R_emp_correction_to_recurrent_amplification_ratio": r_emp,
        "label_memory_update_count": len(label_memory_updates),
        "label_speech_count": len(label_speeches),
        "external_intervention_count": len(interventions),
        "baseline_clean": baseline_clean,
        "interpretation_note": (
            "Initial 0->nonzero relationship assignments are not recursive amplification. "
            "Rejected relationship proposals are schema-output errors but do not invalidate behavior "
            "when the wrapper blocked them before state mutation. R_emp uses only recurrent valid "
            "amplification and correction events and is not a validated RCA/WCT parameter."
        ),
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
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
