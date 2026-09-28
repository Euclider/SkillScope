"""Prediction -> proposal -> paired development gate -> immutable commit."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .common import digest, finite, require, strict_json, write_new
from .editor import choose_evidence, propose
from .readout import select


def paired_effect(before, after):
    """Game-equal effect; seeds are repeats, not independent training runs."""
    def index(rows):
        out = {}
        for row in rows:
            key = (row["game_id"], row["eval_seed"])
            require(key not in out and type(row["success"]) is bool, "Duplicate/invalid evaluation result")
            out[key] = row["success"]
        require(out, "Empty evaluation")
        return out
    a, b = index(before), index(after)
    require(a.keys() == b.keys(), "Unpaired evaluation game/seed sets")
    games = sorted({key[0] for key in a})
    means = []
    for game in games:
        keys = [key for key in a if key[0] == game]
        means.append(sum(int(b[key]) - int(a[key]) for key in keys) / len(keys))
    successes = sum(a.values())
    failures = len(a) - successes
    return {"games": len(games), "episodes": len(a), "delta_success_rate": sum(means) / len(means),
            "before_success_rate": sum(a.values()) / len(a), "after_success_rate": sum(b.values()) / len(b),
            "regressions": sum(a[key] and not b[key] for key in a),
            "repairs": sum(not a[key] and b[key] for key in a),
            "regression_rate": sum(a[key] and not b[key] for key in a) / successes if successes else None,
            "repair_rate": sum(not a[key] and b[key] for key in a) / failures if failures else None,
            "per_game_deltas": means}


def evolve(bank, *, output, event_id, selector, api, episodes, evidence_games, gate_games,
           max_evidence_trajectories, tolerance_pp, evaluate, readout_bundle=None, identity=None,
           payload_validator=None):
    """evaluate(bank) must evaluate the fixed current checkpoint on gate games.

    Final held-out tests are never passed to this function. Completed events
    can be replayed only with identical inputs; incomplete events fail closed.
    """
    output = Path(output)
    evidence_games, gate_games = set(evidence_games), set(gate_games)
    require(evidence_games and gate_games and not evidence_games & gate_games, "Evidence/gate games must be disjoint")
    tolerance_pp = finite(tolerance_pp, "gate tolerance")
    require(0 <= tolerance_pp <= 100, "Invalid development tolerance")
    require(identity is not None and identity.branch_id == bank.branch_id
            and identity.bank_sha256 == bank.manifest_sha256, "Missing/foreign endpoint identity")
    require(episodes and all(item.get("global_update") == identity.start + 1 and item.get("split") == "train"
                             and item.get("game_id") in evidence_games for item in episodes),
            "Editing requires only old-policy initial-batch training episodes")
    source = {"event_id": event_id, "bank_sha256": bank.manifest_sha256, "selector": selector,
              "evidence_game_ids": sorted(evidence_games), "gate_game_ids": sorted(gate_games),
              "evidence_sha256": digest(episodes), "readout_sha256": digest(readout_bundle),
              "endpoint_identity": asdict(identity),
              "evidence_protocol": "same_old_policy_batch_as_readout_v13",
              "evidence_global_update": identity.start + 1, "readout_batch_update": identity.start + 1,
              "sampling_policy_update": identity.start, "evidence_split": "train",
              "post_update_validation_outcomes_read": False,
              "trajectory_cap": None,
              "historical_configured_trajectory_cap": max_evidence_trajectories,
              "tolerance_pp": tolerance_pp}
    for episode in episodes:
        require(episode.get("bank_sha256") == bank.manifest_sha256, "Evidence belongs to a different bank version")
        for step in episode["steps"]:
            sid = step.get("selected_skill_id")
            require(sid in bank.active_versions and step.get("skill_version_sha256") == bank.active_versions[sid],
                    "Stale/unknown evidence skill version")
    if (output / "complete.json").exists():
        record = strict_json((output / "complete.json").read_text())
        require(record["source_sha256"] == digest(source), "Event resume inputs changed")
        if record.get("editor_evidence_sha256") is not None:
            excerpt = strict_json((output / "editor_evidence-v13.json").read_text())
            require(excerpt["editor_evidence_sha256"] == record["editor_evidence_sha256"]
                    == digest(excerpt["editor_evidence"]), "Changed sealed editor evidence")
        from .bank import Bank
        selected = Bank.load(output / "banks" / f"{record['selected_bank_sha256']}.json", record["selected_bank_sha256"])
        return selected, record
    write_new(output / "source-v13.json", source)
    failure = selector == "failure_driven"
    old_batch_failed = choose_evidence(episodes, selector="failure_driven", priority_ids=(),
                                       source_update=identity.start + 1, allowed_games=evidence_games)
    if not failure:
        require(readout_bundle is not None, "Missing readout bundle")
        invoked = {step["selected_skill_id"] for episode in old_batch_failed for step in episode["steps"]}
        selection = select(readout_bundle, expected=identity, active_versions=bank.active_versions,
                           selector=selector, eligible_skill_ids=invoked)
        priority = [row["skill_id"] for row in selection["selected"]]
        evidence = choose_evidence(episodes, selector=selector, priority_ids=priority,
                                   source_update=identity.start + 1, allowed_games=evidence_games)
    else:
        priority = []
        evidence = old_batch_failed
        selection = {"schema_version": "skillrl.phase3.failure-selection.v5",
                     "selector": "failure_driven", "candidate_budget": len(bank.skills),
                     "selected": [], "visible_skill_ids": list(bank.skill_ids),
                     "ranked_candidates": False, "abstain": not evidence,
                     "evidence_rule": "all_old_policy_initial_batch_training_failures",
                     "target_gold_read": False}
    selection["old_batch_failed_evidence_sha256"] = digest(old_batch_failed)
    selection["old_batch_failed_trajectory_count"] = len(old_batch_failed)
    selection["evidence_global_update"] = identity.start + 1
    selection["sampling_policy_update"] = identity.start
    write_new(output / "selection-v13.json", selection)
    write_new(output / "evidence-v13.json", evidence)
    if evidence:
        if not failure:
            require(priority, "Editor evidence without selected candidate skills")
            require(all(not item["success"] for item in evidence), "Readout editor evidence must be failed trajectories")
        editor_evidence = evidence
        candidate_calls = {sid: sum(step["selected_skill_id"] == sid for item in evidence
                                    for step in item["steps"]) for sid in priority}
        write_new(output / "editor_evidence-v13.json", {"source_evidence_sha256": digest(evidence),
            "editor_evidence": editor_evidence, "editor_evidence_sha256": digest(editor_evidence),
            "candidate_skill_ids": priority, "full_bank_retained": failure,
            "bank_titles_retained": failure,
            "selection_sha256": digest(selection), "trajectory_count": len(evidence),
            "old_batch_failed_trajectory_count": len(old_batch_failed),
            "step_count": sum(len(item["steps"]) for item in evidence),
            "candidate_call_counts": candidate_calls,
            "evidence_selection_rule": ("all_old_policy_initial_batch_training_failures" if failure else
                "all_old_policy_initial_batch_failures_invoking_selected_top5"),
            "trajectory_excerpting": False, "local_editor_input_cap_enforced": False})
    else:
        editor_evidence = evidence
    record = {"source_sha256": digest(source), "selector": selector, "event_id": event_id,
              "source_artifact": "source-v13.json", "editor_protocol": "same_old_policy_batch_as_readout_v13",
              "before_bank_sha256": bank.manifest_sha256, "rollback_count": 0,
              "candidate_count": len(bank.skills) if failure else len(priority),
              "old_batch_failed_trajectory_count": len(old_batch_failed),
              "editor_trajectory_count": len(evidence),
              "proposed": False, "accepted": False, "proposal_rejected": False,
              "editor_evidence_sha256": digest(editor_evidence) if evidence else None,
              "editor_input_scope": {"selected_skills_only": not failure,
                                     "full_bank_body_visible": failure,
                                     "complete_trajectories": True,
                                     "local_input_cap_enforced": False}}
    selected = bank
    if (selection is not None and selection["abstain"]) or not evidence:
        record["outcome"] = "abstain_no_supported_candidates_or_evidence"
    else:
        if (output / "proposal.json").exists():
            result = strict_json((output / "proposal.json").read_text())
            candidate, changes = bank.apply(result["patch"], event_id=event_id,
                evidence_ids={item["evidence_id"] for item in evidence}, payload_validator=payload_validator)
            require(changes == result["changes"] and result["input_sha256"] == digest(result["input"]), "Changed saved proposal")
            if failure:
                require(result["input"]["evidence"] == editor_evidence
                        and len(result["input"]["bank"]) == len(bank.skills)
                        and "candidate_skills" not in result["input"], "Changed failure proposal evidence")
            else:
                require(result["input"]["evidence"] == editor_evidence and result["input"]["priority_targets"] == priority,
                        "Changed proposal evidence")
        else:
            result = propose(bank, api=api, event_id=event_id, evidence=editor_evidence, priority_ids=priority,
                             selector=selector, payload_validator=payload_validator)
            candidate = result.pop("candidate")
            write_new(output / "proposal.json", result)
        candidate.save(output / "banks")
        record.update(proposed=True, changes=result["changes"], editor_cost=result["accounting"],
                      candidate_bank_sha256=candidate.manifest_sha256)
        if result["changes"]["noop"]:
            record["outcome"] = "editor_noop"
        else:
            before, after = evaluate(bank), evaluate(candidate)
            require({row["game_id"] for row in before} == gate_games == {row["game_id"] for row in after}, "Incomplete/foreign gate evaluation")
            policies = {row["policy_sha256"] for row in before + after}
            require(len(policies) == 1, "Gate compared different policies")
            if identity is not None:
                require(policies == {identity.new_policy_sha256}, "Gate is not at the readout endpoint")
            require(all(row["bank_sha256"] == bank.manifest_sha256 for row in before)
                    and all(row["bank_sha256"] == candidate.manifest_sha256 for row in after), "Wrong evaluated bank")
            effect = paired_effect(before, after)
            write_new(output / "gate.json", {"before": before, "after": after, "effect": effect})
            accepted = 100 * effect["delta_success_rate"] >= -tolerance_pp
            selected = candidate if accepted else bank
            record.update(accepted=accepted, proposal_rejected=not accepted, gate_effect=effect,
                          outcome="accepted" if accepted else "candidate_rejected_not_rollback")
    selected.save(output / "banks")
    record["selected_bank_sha256"] = selected.manifest_sha256
    write_new(output / "complete.json", record)
    return selected, record
