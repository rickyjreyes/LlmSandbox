#!/usr/bin/env python3
"""Run LlmSandbox with non-invasive recursive-drift instrumentation.

The base simulator remains unchanged. This wrapper records decision-level state
and enforces the simulator's declared relationship schema during research runs:
relationship_changes may only target another NPC that actually exists.
"""

from __future__ import annotations

import copy
import json
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
import scheduler as scheduler_module

LOG_DIR = ROOT / "research_logs"
LOG_DIR.mkdir(exist_ok=True)
STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
LOG_PATH = LOG_DIR / f"recursive_drift_{STAMP}.jsonl"
_LOCK = threading.Lock()


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
    """Reject self, nonexistent, and nonnumeric relationship targets.

    This is schema enforcement, not a behavioral intervention: the base prompt
    already requires relationship_changes keys to be valid NPC IDs.
    """
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
            rejected.append({
                "target_id": target_id,
                "delta": delta,
                "reason": reason,
            })
        else:
            accepted[target_id] = delta

    clean_result["relationship_changes"] = accepted
    return clean_result, rejected


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
        "entities": [_snapshot(e) for e in world.entities.values()],
    })


def _instrumented_response(self, entity_id, result):
    e = self.world.entities.get(entity_id)
    before = _snapshot(e) if e else None

    raw_result = copy.deepcopy(result) if isinstance(result, dict) else result
    clean_result, rejected_relationships = _sanitize_relationship_changes(
        self.world, entity_id, result
    )

    _ORIG_RESPONSE(self, entity_id, clean_result)

    e2 = self.world.entities.get(entity_id)
    after = _snapshot(e2) if e2 else None
    event = {
        "type": "decision",
        "game_time": getattr(self.world, "game_time", None),
        "entity_id": entity_id,
        "entity_name": e2.name if e2 else (e.name if e else None),
        "model": getattr(config, "LLM_MODEL", None),
        "before": before,
        "after": after,
        "llm_result": clean_result,
        "llm_result_raw": raw_result,
        "rejected_relationship_changes": rejected_relationships,
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


def _instrumented_execute(self, e, action):
    atype = action.get("type") if isinstance(action, dict) else None
    target_id = action.get("target") if isinstance(action, dict) else None
    target = self.world.entities.get(target_id) if target_id else None
    target_mem_before = len(target.memories) if target else None

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
        "note": (
            "Base simulator code is unmodified; research wrapper logs state and "
            "enforces declared relationship target IDs."
        ),
    })
    print(f"[research] recursive-drift log: {LOG_PATH}")

    import main as sandbox_main
    sandbox_main.main()
