#!/usr/bin/env python3
"""Create deterministic development/held-out game manifests without model feedback."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from phase1.archive import atomic_write_json, sha256_file

TASK_CONTEXT = {
    "pick_and_place_simple": "pick_and_place",
    "look_at_obj_in_light": "look_at_obj_in_light",
    "pick_clean_then_place_in_recep": "clean",
    "pick_heat_then_place_in_recep": "heat",
    "pick_cool_then_place_in_recep": "cool",
    "pick_two_obj_and_place": "pick_two",
}


def collect(split_root: Path) -> dict[str, list[str]]:
    result = {context: [] for context in TASK_CONTEXT.values()}
    for trajectory_path in sorted(split_root.rglob("traj_data.json")):
        data = json.loads(trajectory_path.read_text(encoding="utf-8"))
        context = TASK_CONTEXT.get(data.get("task_type"))
        game_path = trajectory_path.with_name("game.tw-pddl")
        if not context or not game_path.exists():
            continue
        game_data = json.loads(game_path.read_text(encoding="utf-8"))
        if game_data.get("solvable", False):
            result[context].append(str(game_path.resolve()))
    return result


def write_list(path: Path, items: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in items), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alfworld-data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--games-per-context", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--contexts", nargs="+", choices=sorted(TASK_CONTEXT.values()))
    parser.add_argument("--all-games", action="store_true")
    parser.add_argument("--include-train", action="store_true")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    manifest = {
        "schema_version": "phase1.game_manifest.v1",
        "selection_seed": args.seed,
        "games_per_context": None if args.all_games else args.games_per_context,
        "splits": {},
    }
    source_splits = {
        "development": collect(args.alfworld_data / "json_2.1.1" / "valid_train"),
        "valid_seen": collect(args.alfworld_data / "json_2.1.1" / "valid_seen"),
    }
    if args.include_train:
        source_splits["train"] = collect(args.alfworld_data / "json_2.1.1" / "train")
    for split_name, grouped_games in source_splits.items():
        for context, games in grouped_games.items():
            if args.contexts and context not in args.contexts:
                continue
            rng.shuffle(games)
            if not args.all_games and len(games) < args.games_per_context:
                raise ValueError(
                    f"{context} has {len(games)} games in {split_name}; "
                    f"need {args.games_per_context}"
                )
            selected = games if args.all_games else games[:args.games_per_context]
            path = args.output_dir / split_name / f"{context}.txt"
            write_list(path, selected)
            manifest["splits"].setdefault(split_name, {})[context] = {
                "path": str(path), "count": len(selected), "sha256": sha256_file(path),
            }

    unseen = collect(args.alfworld_data / "json_2.1.1" / "valid_unseen")
    for context, games in unseen.items():
        if args.contexts and context not in args.contexts:
            continue
        rng.shuffle(games)
        if not args.all_games and len(games) < args.games_per_context:
            raise ValueError(f"{context} has {len(games)} valid_unseen games; need {args.games_per_context}")
        selected = games if args.all_games else games[:args.games_per_context]
        path = args.output_dir / "valid_unseen" / f"{context}.txt"
        write_list(path, selected)
        manifest["splits"].setdefault("valid_unseen", {})[context] = {
            "path": str(path), "count": len(selected), "sha256": sha256_file(path),
        }
    atomic_write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
