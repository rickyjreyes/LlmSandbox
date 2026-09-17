#!/usr/bin/env python3
"""Run LlmSandbox with non-invasive recursive-drift instrumentation and ablations.

The base simulator remains unchanged. This wrapper records decision-level state,
enforces declared relationship target IDs, hardens research-run schema validation,
and can selectively remove feedback channels or apply an explicit restoring gain
for matched causal experiments.
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import shutil
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import llm as llm_module
import scheduler as scheduler_module


def _parse_args():
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--run-label", default="baseline")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--world-template", type=Path, default=None)
    p.add_argument("--max-decisions", type=int, default=None,
                   help="Automatically close after this many valid LLM decisions")
    p.add_argument("--max-failed-decisions", type=int, default=10,
                   help="Abort a research run after this many invalid/failed LLM decisions")
    p.add_argument("--memory-feedback", choices=("on", "off"), default="on")
    p.add_argument("--communication", choices=("on", "off"), default="on")
    p.add_argument("--relationship-feedback", choices=("on", "off"), default="on")
    p.add_argument("--correction-gain", type=float, default=0.0,
                   help="Per-decision damping r <- (1-gamma) r; 0 <= gamma <= 1")
    p.add_argument("--periodic-check", type=float, default=None)
    p.add_argument("--perception-radius", type=float, default=None)
    p.add_argument("--short-term-mem-limit", type=int, default=None)
    p.add_argument("--model", default=None)
    args, _unknown = p.parse_known_args()
    if not 0.0 <= args.correction_gain <= 1.0:
        p.error("--correction-gain must be between 0 and 1")
    if args.max_decisions is not None and args.max_decisions <= 0:
        p.error("--max-decisions must be > 0")
    if args.max_failed_decisions < 0:
        p.error("--max-failed-decisions must be >= 0")
    return args


EXP = _parse_args()
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
SAFE_LABEL = re.sub(r"[^A-Za-z0-9_.-]+", "_", EXP.run_label).strip("_") or "run"

# Apply explicit research overrides before the simulator starts.
if EXP.seed is not None:
    random.seed(EXP.seed)
if EXP.periodic_check is not None:
    config.PERIODIC_CHECK_INTERVAL = EXP.periodic_check
if EXP.perception_radius is not None:
    config.PERCEPTION_RADIUS = EXP.perception_radius
if EXP.short_term_mem_limit is not None:
    config.SHORT_TERM_MEM_LIMIT = EXP.short_term_mem_limit
if EXP.model:
    config.LLM_MODEL = EXP.model

# An immutable template is copied to a run-local save so the simulator's normal
# autosave/exit-save behavior cannot mutate the matched starting condition.
RUNTIME_WORLD = None
if EXP.world_template is not None:
    template = EXP.world_template.resolve()
    if not template.exists():
        raise FileNotFoundError(f"World template not found: {template}")
    runtime_dir = ROOT / "research_runtime" / f"{STAMP}_{SAFE_LABEL}"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    RUNTIME_WORLD = runtime_dir / "world_save.json"
    shutil.copy2(template, RUNTIME_WORLD)
    config.SAVE_FILE = str(RUNTIME_WORLD)

LOG_DIR = ROOT / "research_logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = LOG_DIR / f"recursive_drift_{STAMP}_{SAFE_LABEL}.jsonl"
_LOCK = threading.Lock()
_VALID_DECISIONS = 0
_FAILED_DECISIONS = 0
_SCHEMA_NORMALIZATIONS = 0


def _experiment_metadata() -> dict:
    return {
        "run_label": EXP.run_label,
        "seed": EXP.seed,
        "world_template": str(EXP.world_template) if EXP.world_template else None,
        "runtime_world": str(RUNTIME_WORLD) if RUNTIME_WORLD else None,
        "max_decisions": EXP.max_decisions,
        "max_failed_decisions": EXP.max_failed_decisions,
        "memory_feedback": EXP.memory_feedback,
        "communication": EXP.communication,
        "relationship_feedback": EXP.relationship_feedback,
        "correction_gain": EXP.correction_gain,
        "periodic_check_interval": config.PERIODIC_CHECK_INTERVAL,
        "perception_radius": config.PERCEPTION_RADIUS,
        "short_term_mem_limit": config.SHORT_TERM_MEM_LIMIT,
        "model": getattr(config, "LLM_MODEL", None),
    }


def _write(event: dict) -> None:
    event = dict(event)
    event.setdefault("wall_time", time.time())
    with _LOCK:
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _snapshot(e) -> dict:
    return {
        "id": e.id,
        "name": e.name,
        "mood": e.mood,
        "status": e.status,
        "short_term_goals": list(e.short_term_goals),
        "long_term_goals": list(e.long_term_goals),
        "relationships": dict(e.relationships),
        "memory_count": len(e.memories),
        "recent_memories": [m.get("text", "") for m in e.memories[-15:]],
        "position": dict(e.position),
    }


def _relationship_diff(before: dict, after: dict) -> list[dict]:
    out = []
    keys = set(before) | set(after)
    for target_id in sorted(keys):
        b = float(before.get(target_id, 0.0))
        a = float(after.get(target_id, 0.0))
        if abs(a - b) > 1e-12:
            out.append({
                "target_id": target_id,
                "before": b,
                "after": a,
                "delta": a - b,
                "abs_before": abs(b),
                "abs_after": abs(a),
                "existed_before": target_id in before,
                "exists_after": target_id in after,
            })
    return out


def _sanitize_relationship_changes(world, entity_id: str, result):
    """Reject self, nonexistent, and nonnumeric relationship targets."""
    if not isinstance(result, dict):
        return result, []

    clean_result = copy.deepcopy(result)
    rels = clean_result.get("relationship_changes", {})
    if not isinstance(rels, dict):
        clean_result["relationship_changes"] = {}
        return clean_result, []

    accepted = {}
    rejected = []
    for target_id, delta in rels.items():
        target_id = str(target_id)
        reason = None
        if target_id == entity_id:
            reason = "self_target"
        elif target_id not in world.entities:
            reason = "unknown_target"
        else:
            try:
                delta = float(delta)
            except (TypeError, ValueError):
                reason = "nonnumeric_delta"

        if reason:
            rejected.append({"target_id": target_id, "delta": delta, "reason": reason})
        else:
            accepted[target_id] = delta

    clean_result["relationship_changes"] = accepted
    return clean_result, rejected


def _post_quit_event() -> None:
    try:
        import pygame
        pygame.event.post(pygame.event.Event(pygame.QUIT))
    except Exception as ex:
        _write({"type": "research_warning", "warning": f"auto-stop failed: {ex}"})


# Research-side schema hardening. The upstream validator assumes action fields
# such as priority/duration/target IDs already have scalar JSON types. Local LLMs
# occasionally return valid JSON with the wrong type (for example priority=[1]).
# Normalize or drop only schema-invalid action fields before calling the original
# validator so malformed output cannot crash an experimental condition.
_ORIG_VALIDATE_NPC_RESPONSE = llm_module.validate_npc_response


def _safe_int(value, default: int) -> int:
    if isinstance(value, bool) or isinstance(value, (list, dict, tuple, set)):
        return default
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _safe_float(value, default: float) -> float:
    if isinstance(value, bool) or isinstance(value, (list, dict, tuple, set)):
        return default
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _research_validate_npc_response(data: dict) -> dict:
    global _SCHEMA_NORMALIZATIONS

    if not isinstance(data, dict):
        data = {}

    clean = copy.deepcopy(data)
    raw_actions = clean.get("actions", [])
    if not isinstance(raw_actions, list):
        raw_actions = []
        _SCHEMA_NORMALIZATIONS += 1

    normalized_actions = []
    for action in raw_actions:
        if not isinstance(action, dict):
            _SCHEMA_NORMALIZATIONS += 1
            continue

        a = dict(action)
        atype = a.get("type")
        if not isinstance(atype, str) or atype not in config.VALID_ACTIONS:
            _SCHEMA_NORMALIZATIONS += 1
            continue

        original_priority = a.get("priority", 5)
        normalized_priority = _safe_int(original_priority, 5)
        if normalized_priority != original_priority:
            _SCHEMA_NORMALIZATIONS += 1
        a["priority"] = normalized_priority

        if atype == config.ACTION_MOVE:
            if not isinstance(a.get("to"), dict):
                _SCHEMA_NORMALIZATIONS += 1
                continue

        elif atype == config.ACTION_SAY:
            target = a.get("target")
            if target is not None and not isinstance(target, str):
                a["target"] = None
                _SCHEMA_NORMALIZATIONS += 1

        elif atype == config.ACTION_ATTACK:
            if not isinstance(a.get("target"), str) or not a.get("target"):
                _SCHEMA_NORMALIZATIONS += 1
                continue

        elif atype in (config.ACTION_PICK_UP, config.ACTION_USE, config.ACTION_DROP):
            if not isinstance(a.get("item_id"), str) or not a.get("item_id"):
                _SCHEMA_NORMALIZATIONS += 1
                continue

        elif atype == config.ACTION_GIVE:
            if (not isinstance(a.get("item_id"), str) or not a.get("item_id") or
                    not isinstance(a.get("target"), str) or not a.get("target")):
                _SCHEMA_NORMALIZATIONS += 1
                continue

        elif atype == config.ACTION_WAIT:
            original_duration = a.get("duration", 5)
            normalized_duration = _safe_float(original_duration, 5.0)
            if normalized_duration != original_duration:
                _SCHEMA_NORMALIZATIONS += 1
            a["duration"] = normalized_duration

        normalized_actions.append(a)

    clean["actions"] = normalized_actions
    return _ORIG_VALIDATE_NPC_RESPONSE(clean)


# Prompt-level ablations. They hide a feedback channel from the next LLM prompt
# while leaving the underlying simulator state available for instrumentation.
_ORIG_BUILD_NPC_PROMPT = llm_module.build_npc_prompt


def _instrumented_build_npc_prompt(entity, world, injected_message: str = None):
    saved_memories = entity.memories
    saved_short_memories = entity.short_term_memories
    saved_relationships = entity.relationships
    try:
        if EXP.memory_feedback == "off":
            entity.memories = []
            entity.short_term_memories = []
        if EXP.relationship_feedback == "off":
            entity.relationships = {}
        return _ORIG_BUILD_NPC_PROMPT(entity, world, injected_message)
    finally:
        entity.memories = saved_memories
        entity.short_term_memories = saved_short_memories
        entity.relationships = saved_relationships


_ORIG_INIT = scheduler_module.Scheduler.__init__
_ORIG_RESPONSE = scheduler_module.Scheduler._on_npc_response
_ORIG_EXECUTE = scheduler_module.Scheduler._execute_action
_ORIG_DISPATCH = scheduler_module.Scheduler._dispatch_global_command
_ORIG_INTERRUPT = scheduler_module.Scheduler.interrupt_entity


def _instrumented_init(self, world):
    _ORIG_INIT(self, world)
    _write({
        "type": "run_start",
        "game_time": getattr(world, "game_time", 0.0),
        "model": getattr(config, "LLM_MODEL", None),
        "npc_count": len(world.entities),
        "world_seed": getattr(world, "seed", None),
        "experiment": _experiment_metadata(),
        "entities": [_snapshot(e) for e in world.entities.values()],
    })


def _instrumented_response(self, entity_id, result):
    global _VALID_DECISIONS, _FAILED_DECISIONS

    e = self.world.entities.get(entity_id)
    before = _snapshot(e) if e else None

    raw_result = copy.deepcopy(result) if isinstance(result, dict) else result
    clean_result, rejected_relationships = _sanitize_relationship_changes(
        self.world, entity_id, result
    )

    _ORIG_RESPONSE(self, entity_id, clean_result)

    e2 = self.world.entities.get(entity_id)
    post_llm_relationships = dict(e2.relationships) if e2 else {}
    correction_changes = []

    # Explicit experimental restoring term. gamma=0 is exactly the baseline.
    # For gamma>0, every currently stored relationship is damped toward neutral
    # after the native LLM update: r_final = (1-gamma) * r_native.
    if e2 and EXP.correction_gain > 0.0:
        gamma = EXP.correction_gain
        for target_id, value in list(e2.relationships.items()):
            native = float(value)
            final = (1.0 - gamma) * native
            if abs(final) < 1e-12:
                final = 0.0
            if abs(final - native) > 1e-12:
                e2.relationships[target_id] = final
                correction_changes.append({
                    "target_id": target_id,
                    "native_after": native,
                    "final_after": final,
                    "treatment_delta": final - native,
                    "gamma": gamma,
                })

    after = _snapshot(e2) if e2 else None
    event = {
        "type": "decision",
        "game_time": getattr(self.world, "game_time", None),
        "entity_id": entity_id,
        "entity_name": e2.name if e2 else (e.name if e else None),
        "model": getattr(config, "LLM_MODEL", None),
        "experiment": _experiment_metadata(),
        "before": before,
        "after": after,
        "post_llm_relationships_before_correction": post_llm_relationships,
        "correction_treatment_changes": correction_changes,
        "llm_result": clean_result,
        "llm_result_raw": raw_result,
        "rejected_relationship_changes": rejected_relationships,
        "schema_normalization_count_total": _SCHEMA_NORMALIZATIONS,
        "external_directive_present": bool(
            getattr(e2 or e, "pending_scheduler_message", None)
        ) if (e2 or e) else False,
    }
    if before and after:
        event["relationship_changes_applied"] = _relationship_diff(
            before.get("relationships", {}), after.get("relationships", {})
        )
        added = max(0, after["memory_count"] - before["memory_count"])
        event["new_memories"] = after["recent_memories"][-added:] if added else []
    _write(event)

    if clean_result is not None:
        _VALID_DECISIONS += 1
        if EXP.max_decisions is not None and _VALID_DECISIONS >= EXP.max_decisions:
            _write({
                "type": "auto_stop",
                "game_time": getattr(self.world, "game_time", None),
                "valid_decisions": _VALID_DECISIONS,
                "failed_decisions": _FAILED_DECISIONS,
                "reason": "max_decisions_reached",
            })
            _post_quit_event()
    else:
        _FAILED_DECISIONS += 1
        if _FAILED_DECISIONS > EXP.max_failed_decisions:
            _write({
                "type": "auto_stop",
                "game_time": getattr(self.world, "game_time", None),
                "valid_decisions": _VALID_DECISIONS,
                "failed_decisions": _FAILED_DECISIONS,
                "reason": "max_failed_decisions_exceeded",
            })
            _post_quit_event()


def _instrumented_execute(self, e, action):
    atype = action.get("type") if isinstance(action, dict) else None
    target_id = action.get("target") if isinstance(action, dict) else None
    target = self.world.entities.get(target_id) if isinstance(target_id, str) else None
    target_mem_before = len(target.memories) if target else None
    communication_blocked = False

    if atype == getattr(config, "ACTION_SAY", "say") and EXP.communication == "off":
        # Preserve the source agent's visible speech action but do not place the
        # utterance in the world event stream or the target's memory. This
        # removes the explicit agent-to-agent text propagation pathway.
        text = action.get("text", "")
        e.say(str(text), time.time())
        communication_blocked = True
    else:
        _ORIG_EXECUTE(self, e, action)

    if atype == getattr(config, "ACTION_SAY", "say"):
        _write({
            "type": "speech",
            "game_time": getattr(self.world, "game_time", None),
            "source_id": e.id,
            "source_name": e.name,
            "target_id": target_id,
            "target_name": target.name if target else None,
            "text": action.get("text", ""),
            "communication_blocked": communication_blocked,
            "target_memory_added": (
                len(target.memories) > target_mem_before
                if target is not None and target_mem_before is not None else False
            ),
        })
    elif atype == getattr(config, "ACTION_ATTACK", "attack"):
        _write({
            "type": "attack_action",
            "game_time": getattr(self.world, "game_time", None),
            "source_id": e.id,
            "source_name": e.name,
            "target_id": target_id,
            "target_name": target.name if target else None,
        })


def _instrumented_dispatch(self, command):
    _write({
        "type": "external_intervention",
        "game_time": getattr(self.world, "game_time", None),
        "mode": "global_command",
        "command": command,
    })
    return _ORIG_DISPATCH(self, command)


def _instrumented_interrupt(self, entity_id, message):
    _write({
        "type": "external_intervention",
        "game_time": getattr(self.world, "game_time", None),
        "mode": "entity_interrupt",
        "entity_id": entity_id,
        "message": message,
    })
    return _ORIG_INTERRUPT(self, entity_id, message)


def install_instrumentation() -> None:
    llm_module.validate_npc_response = _research_validate_npc_response
    llm_module.build_npc_prompt = _instrumented_build_npc_prompt
    scheduler_module.Scheduler.__init__ = _instrumented_init
    scheduler_module.Scheduler._on_npc_response = _instrumented_response
    scheduler_module.Scheduler._execute_action = _instrumented_execute
    scheduler_module.Scheduler._dispatch_global_command = _instrumented_dispatch
    scheduler_module.Scheduler.interrupt_entity = _instrumented_interrupt


if __name__ == "__main__":
    install_instrumentation()
    _write({
        "type": "instrumentation",
        "log_path": str(LOG_PATH),
        "argv": sys.argv,
        "experiment": _experiment_metadata(),
        "note": (
            "Base simulator files are unmodified. Research wrapper logs state, "
            "hardens schema-invalid local-model output, enforces relationship target IDs, "
            "and applies only explicitly selected experimental feedback ablations/treatments."
        ),
    })
    print(f"[research] recursive-drift log: {LOG_PATH}")
    print("[research] experiment:", json.dumps(_experiment_metadata(), sort_keys=True))

    import main as sandbox_main
    sandbox_main.main()
