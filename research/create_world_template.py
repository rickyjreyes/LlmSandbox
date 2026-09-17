#!/usr/bin/env python3
"""Create an immutable LlmSandbox world template for matched experiments.

Generate once, then pass the resulting JSON to run_recursive_drift.py with
--world-template. Each run copies the template before loading it, so normal
simulator save/autosave behavior cannot modify the original starting world.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main as sandbox_main
from persistence import save_world


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--npcs", type=int, default=6)
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: research_worlds/world_seed_<seed>.json",
    )
    args = p.parse_args()

    if args.npcs <= 0:
        p.error("--npcs must be > 0")

    output = args.output or ROOT / "research_worlds" / f"world_seed_{args.seed}.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    world = sandbox_main.create_new_world(num_npcs=args.npcs)

    if not save_world(world, str(output)):
        raise RuntimeError(f"Could not save world template: {output}")

    manifest = {
        "template": str(output),
        "sha256": sha256(output),
        "requested_random_seed": args.seed,
        "world_seed": getattr(world, "seed", None),
        "npc_count": len(world.entities),
        "entities": [
            {
                "id": e.id,
                "name": e.name,
                "sprite_type": e.sprite_type,
                "position": dict(e.position),
            }
            for e in world.entities.values()
        ],
        "note": (
            "Use this exact file as --world-template for every matched condition. "
            "The template, not regeneration from seed, is the authoritative matched start."
        ),
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest, indent=2))
    print(f"\nWorld template: {output}")
    print(f"Manifest:      {manifest_path}")


if __name__ == "__main__":
    main()
