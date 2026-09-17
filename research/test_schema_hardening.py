#!/usr/bin/env python3
"""Regression test for research-side LLM schema hardening.

No model call is made. This reproduces malformed-but-valid JSON shapes that
previously could raise inside llm.validate_npc_response().
"""

from run_recursive_drift import _research_validate_npc_response


def main() -> None:
    malformed = {
        "actions": [
            {"type": "say", "target": None, "text": "hello", "priority": [1]},
            {"type": "wait", "duration": {"bad": 1}, "priority": "2"},
            {"type": "move", "to": [10, 20], "priority": 1},
            {"type": "attack", "target": ["npc_bad"], "priority": 1},
            {"type": "say", "target": ["npc_bad"], "text": "test", "priority": 3},
        ],
        "mood": "curious",
        "memory_updates": [],
        "long_term_goals": [],
        "short_term_goals": [],
        "relationship_changes": {},
        "metadata": {"reasoning": "schema-hardening regression"},
    }

    out = _research_validate_npc_response(malformed)
    actions = out["actions"]

    assert len(actions) == 3, actions
    assert actions[0]["type"] == "say"
    assert actions[0]["priority"] == 5
    assert actions[1]["type"] == "wait"
    assert actions[1]["priority"] == 2
    assert actions[1]["duration"] == 5.0
    assert actions[2]["type"] == "say"
    assert actions[2]["target"] is None

    print("PASS: malformed action fields were normalized/dropped without exception")
    print(actions)


if __name__ == "__main__":
    main()
