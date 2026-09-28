"""One paid synthetic-state router request followed by a no-call cache replay.

Run as a module from the code root. The key is read from the dedicated runtime
environment variable or an echo-disabled prompt, never from a command argument.
No policy, benchmark environment, or historical trajectory is loaded.
"""

import argparse
import getpass
import json
import os
from pathlib import Path

from agent_system.memory.external_skill_router import API_KEY_ENV, ExternalRouterError
from agent_system.memory.router_cache import RouterCacheError
from agent_system.memory.skillnet_runtime import DEFAULT_ROUTER_PROFILE, create_skillnet37_runtime


SYNTHETIC_STATE = {
    "task_description": "put a clean mug in cabinet 1",
    "current_observation": "You are at sinkbasin 1. You are holding mug 1, which has not yet been cleaned.",
    "admissible_actions": ["clean mug 1 with sinkbasin 1", "go to cabinet 1", "inventory"],
    "history": [
        {"observation": "On countertop 1, you see mug 1.", "action": "take mug 1 from countertop 1"},
        {"observation": "You pick up mug 1 from countertop 1.", "action": "go to sinkbasin 1"},
    ],
    "step_index": 2,
}


def run_smoke(memory, router):
    candidates = memory.retrieve(SYNTHETIC_STATE["task_description"])
    first = router.route(candidates, **SYNTHETIC_STATE)
    second = router.route(candidates, **SYNTHETIC_STATE)
    assert first["selected_skill_id"] == second["selected_skill_id"]
    assert second["skill_router_api"]["cache_hit"] is True
    assert second["skill_router_api"]["api_calls_this_step"] == 0
    return {
        "status": "synthetic_router_and_cache_replay_passed",
        "benchmark_experiment": False,
        "candidate_count": len(first["candidate_skill_ids"]),
        "selected_skill_id": first["selected_skill_id"],
        "payload_utf8_bytes": len(memory.format_for_prompt(first).encode("utf-8")),
        "bank_manifest_sha256": memory.bank.manifest_sha256,
        "first_call": first["skill_router_api"],
        "replay_cache_hit": second["skill_router_api"]["cache_hit"],
        "live_api_calls_this_invocation": first["skill_router_api"]["api_calls_this_step"],
        "cache_stats": router.cache.stats(),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_ROUTER_PROFILE)
    parser.add_argument("--cache", type=Path, required=True, help="New SQLite cache path for this authorized smoke test")
    parser.add_argument("--prompt-api-key", action="store_true", help="Read key without terminal echo; never save it")
    args = parser.parse_args(argv)
    previous = os.environ.get(API_KEY_ENV)
    router = None
    try:
        if args.prompt_api_key:
            os.environ[API_KEY_ENV] = getpass.getpass("Router API key (hidden): ")
        memory, router = create_skillnet37_runtime(profile_path=args.profile, cache_path=args.cache, max_api_calls=1)
        report = run_smoke(memory, router)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except (ExternalRouterError, RouterCacheError) as error:
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "detail": str(error),
                          "cache_stats": router.cache.stats() if router else None}))
        return 1
    except Exception as error:
        # Do not print arbitrary SDK exception strings, response bodies or headers.
        print(json.dumps({"status": "failed", "error_type": type(error).__name__, "detail": "Unexpected failure; no raw exception logged"}))
        return 1
    finally:
        if router is not None:
            router.close()
        if args.prompt_api_key:
            if previous is None:
                os.environ.pop(API_KEY_ENV, None)
            else:
                os.environ[API_KEY_ENV] = previous


if __name__ == "__main__":
    raise SystemExit(main())
