# llm.py — LLM integration: prompt building, response parsing, API calls
#
# Uses the OpenAI-compatible API so this works with both Ollama and LM Studio
# (and anything else that exposes a /v1 endpoint).
# Set LLM_BASE_URL and LLM_MODEL in config.py to point at your local server.

import json
import re
import threading
import logging
from typing import Optional, Callable
from openai import OpenAI
import config

logger = logging.getLogger(__name__)

_client = OpenAI(
    base_url=config.LLM_BASE_URL,
    api_key="ollama",
)
_semaphore = threading.Semaphore(config.MAX_LLM_CONCURRENT)

NPC_SYSTEM_PROMPT = """You are the decision-making AI for an NPC in a real-time 2D RPG sandbox.
Your role: given the NPC's internal state, surroundings, and recent events, decide what they
should do next. Stay in character based on their goals, mood, and personality.

IMPORTANT RULES:
- Respond ONLY with a single valid JSON object. No prose, markdown fences, or JSON comments.
- Keep speech/say actions concise (15 words or fewer).
- Actions must use valid coordinates within the map (0-63 on each axis).
- Do not hallucinate entity IDs; only reference IDs present in the context.
- Multiple actions are allowed but must not contradict each other.
- Be creative and emergent; NPCs should feel alive and goal-driven.

RESPONSE SCHEMA (all fields required):
{
  "actions": [
    {"type":"move","to":{"x":0,"y":0},"priority":1},
    {"type":"say","target":null,"text":"...","priority":2},
    {"type":"attack","target":"npc_id","priority":1},
    {"type":"pick_up","item_id":"item_id","priority":2},
    {"type":"drop","item_id":"item_id","priority":3},
    {"type":"use","item_id":"item_id","priority":2},
    {"type":"give","item_id":"item_id","target":"npc_id","priority":2},
    {"type":"wait","duration":5,"priority":5},
    {"type":"sleep","priority":3}
  ],
  "mood": "calm",
  "memory_updates": ["short string describing notable events to remember"],
  "long_term_goals": ["updated long-term goal"],
  "short_term_goals": ["updated short-term goal"],
  "relationship_changes": {"npc_id": 0.0},
  "metadata": {"reasoning": "brief internal reasoning"}
}

Use only the action objects that are appropriate; the example above shows allowed shapes, not a requirement to emit every action.
Mood must be one of: calm, happy, sad, angry, fearful, curious, bored, excited, tired, hungry.
"""

SCHEDULER_SYSTEM_PROMPT = """You are the world-director AI for an RPG sandbox simulation.
Your job: translate high-level developer commands or world events into targeted messages
for individual NPCs injected into their next decision prompts.

Respond ONLY with a single valid JSON object. No prose, markdown fences, or JSON comments.

RESPONSE SCHEMA:
{
  "messages": [
    {"npc_id": "npc_xxxxx", "message": "You should now ..."}
  ],
  "world_effects": [
    {"type": "set_status", "npc_id": "npc_xxxxx", "status": "idle"},
    {"type": "set_health", "npc_id": "npc_xxxxx", "health": 50},
    {"type": "set_mood", "npc_id": "npc_xxxxx", "mood": "angry"},
    {"type": "teleport", "npc_id": "npc_xxxxx", "x": 10, "y": 10},
    {"type": "spawn_item", "name": "apple", "x": 20, "y": 20},
    {"type": "log_event", "text": "Something happened in the world"}
  ],
  "reasoning": "brief explanation of decisions"
}

Use only the message and world-effect objects that are appropriate.
"""


def build_npc_prompt(entity, world, injected_message: str = None) -> str:
    e = entity
    inventory_names = []
    for item_id in e.inventory:
        it = world.items.get(item_id)
        if it:
            inventory_names.append(it.name)

    perceptions = world.get_perceptions(e)
    recent_evts = [ev["text"] for ev in world.recent_events(30)][-8:]
    st_mems = [m["text"] for m in e.short_term_memories[-8:]]
    lt_mems = [m["text"] for m in e.memories[-200:] if m not in e.short_term_memories][-5:]

    relationships_summary = {
        world.entities[eid].name if eid in world.entities else eid: round(v, 2)
        for eid, v in e.relationships.items()
    }

    context = {
        "world_time": world.clock_str(),
        "time_of_day": world.time_of_day(),
        "npc": {
            "id": e.id,
            "name": e.name,
            "role": e.sprite_type,
            "position": {"x": round(e.position["x"], 1), "y": round(e.position["y"], 1)},
            "health": round(e.health, 1),
            "max_health": round(e.max_health, 1),
            "hunger": f"{e.hunger:.0%}",
            "energy": f"{e.energy:.0%}",
            "mood": e.mood,
            "weapon": e.weapon_type,
            "inventory": inventory_names,
            "short_term_goals": e.short_term_goals,
            "long_term_goals": e.long_term_goals,
            "relationships": relationships_summary,
        },
        "recent_memories": st_mems,
        "long_term_notes": lt_mems,
        "visible_entities_and_items": perceptions,
        "recent_world_events": recent_evts,
    }

    if injected_message:
        context["URGENT_DIRECTIVE"] = injected_message

    return json.dumps(context, indent=2)


def build_scheduler_prompt(command: str, world) -> str:
    npc_summaries = []
    for e in world.entities.values():
        if not e.is_alive():
            continue
        npc_summaries.append({
            "id": e.id,
            "name": e.name,
            "role": e.sprite_type,
            "pos": {"x": round(e.position["x"], 1), "y": round(e.position["y"], 1)},
            "status": e.status,
            "mood": e.mood,
            "goals": e.short_term_goals[:2],
        })
    context = {
        "world_time": world.clock_str(),
        "developer_command": command,
        "all_npcs": npc_summaries,
    }
    return json.dumps(context, indent=2)


def _extract_json(text: str) -> Optional[dict]:
    """Parse a JSON object, retaining fence stripping as a compatibility fallback."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group())
                return data if isinstance(data, dict) else None
            except Exception:
                pass
    return None


def validate_npc_response(data: dict) -> dict:
    """Coerce and validate the NPC JSON response."""
    result = {
        "actions": [],
        "mood": data.get("mood", "calm"),
        "memory_updates": data.get("memory_updates", []),
        "long_term_goals": data.get("long_term_goals", []),
        "short_term_goals": data.get("short_term_goals", []),
        "relationship_changes": data.get("relationship_changes", {}),
        "metadata": data.get("metadata", {}),
    }
    if result["mood"] not in config.MOODS:
        result["mood"] = "calm"

    for action in data.get("actions", []):
        if not isinstance(action, dict):
            continue
        atype = action.get("type")
        if atype not in config.VALID_ACTIONS:
            continue
        clean = {"type": atype, "priority": int(action.get("priority", 5))}

        if atype == config.ACTION_MOVE:
            to = action.get("to", {})
            try:
                clean["to"] = {
                    "x": max(0.0, min(float(to.get("x", 0)), config.MAP_WIDTH - 1)),
                    "y": max(0.0, min(float(to.get("y", 0)), config.MAP_HEIGHT - 1)),
                }
            except (TypeError, ValueError):
                continue
        elif atype == config.ACTION_SAY:
            clean["target"] = action.get("target")
            clean["text"] = str(action.get("text", ""))[:100]
        elif atype == config.ACTION_ATTACK:
            if not action.get("target"):
                continue
            clean["target"] = action["target"]
        elif atype in (config.ACTION_PICK_UP, config.ACTION_USE, config.ACTION_DROP):
            if not action.get("item_id"):
                continue
            clean["item_id"] = action["item_id"]
        elif atype == config.ACTION_GIVE:
            if not action.get("item_id") or not action.get("target"):
                continue
            clean["item_id"] = action["item_id"]
            clean["target"] = action["target"]
        elif atype == config.ACTION_WAIT:
            clean["duration"] = float(action.get("duration", 5))

        result["actions"].append(clean)

    if not isinstance(result["memory_updates"], list):
        result["memory_updates"] = []
    if not isinstance(result["long_term_goals"], list):
        result["long_term_goals"] = []
    if not isinstance(result["short_term_goals"], list):
        result["short_term_goals"] = []
    if not isinstance(result["relationship_changes"], dict):
        result["relationship_changes"] = {}
    if not isinstance(result["metadata"], dict):
        result["metadata"] = {}

    return result


def validate_scheduler_response(data: dict) -> dict:
    return {
        "messages": data.get("messages", []),
        "world_effects": data.get("world_effects", []),
        "reasoning": data.get("reasoning", ""),
    }


def call_npc_llm_async(entity, world, injected_message: str = None,
                       callback: Callable = None):
    """Fire an async LLM call for an NPC decision. callback(entity_id, result_dict | None)."""
    entity.pending_llm_call = True
    prompt = build_npc_prompt(entity, world, injected_message)
    eid = entity.id

    def _run():
        with _semaphore:
            try:
                resp = _client.chat.completions.create(
                    model=config.LLM_MODEL,
                    max_tokens=config.LLM_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": NPC_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = resp.choices[0].message.content or ""
                data = _extract_json(raw)
                if data:
                    validated = validate_npc_response(data)
                    if callback:
                        callback(eid, validated)
                else:
                    finish = getattr(resp.choices[0], "finish_reason", None)
                    logger.warning(
                        f"NPC {eid}: could not parse JSON response "
                        f"(finish_reason={finish}, chars={len(raw)}):\n{raw[:500]}"
                    )
                    if callback:
                        callback(eid, None)
            except Exception as ex:
                logger.error(f"LLM call failed for {eid}: {type(ex).__name__}: {ex}")
                if callback:
                    callback(eid, None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()


def call_scheduler_llm_async(command: str, world,
                              callback: Callable = None):
    """Translate a developer command into NPC messages / world effects."""
    prompt = build_scheduler_prompt(command, world)

    def _run():
        with _semaphore:
            try:
                resp = _client.chat.completions.create(
                    model=config.LLM_MODEL,
                    max_tokens=config.LLM_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SCHEDULER_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = resp.choices[0].message.content or ""
                data = _extract_json(raw)
                if data:
                    validated = validate_scheduler_response(data)
                    if callback:
                        callback(validated)
                else:
                    finish = getattr(resp.choices[0], "finish_reason", None)
                    logger.warning(
                        f"Scheduler: could not parse JSON response "
                        f"(finish_reason={finish}, chars={len(raw)}):\n{raw[:500]}"
                    )
                    if callback:
                        callback(None)
            except Exception as ex:
                logger.error(f"Scheduler LLM call failed: {type(ex).__name__}: {ex}")
                if callback:
                    callback(None)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
