"""Portable cumulative Phase3 runner with immutable U20 interim pause."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from .api import APIConfig, JSONClient
from .bank import Bank
from .common import digest, positive_int, require, strict_json, write_new
from .prepare import ARMS, ROOT, load
from .readout import WindowIdentity
from .routing import create_provider, router_backend


def stop_at(runtime, requested):
    stop = runtime['initial_stop_update'] if requested is None else requested
    positive_int(stop, 'requested stop update')
    require(stop % 5 == 0 and runtime['initial_stop_update'] <= stop <= runtime['optimizer_horizon_updates'],
            'Stop must be a five-update boundary between U20 and the frozen optimizer horizon')
    return stop


def base_plan(manifest, runtime, root, branch):
    return {'preparation_sha256': digest(manifest), 'branch': branch, 'selector': ARMS[branch],
        'root': str(Path(root).resolve()), 'seed': runtime['rl_seed'],
        'optimizer_horizon_updates': runtime['optimizer_horizon_updates'],
        'initial_stop_update': runtime['initial_stop_update'], 'window_updates': 5,
        'gpu_ids': runtime['gpu_ids'], 'router_api_cap': runtime['router'].get('max_api_calls', 0),
        'router_backend': router_backend(runtime['router']), 'router_local_cap': runtime['router'].get('max_local_calls', 0),
        'editor_api_cap': runtime['editor']['max_api_calls'], 'runtime_sha256': digest(runtime),
        'initial_bank_sha256': manifest['initial_banks'][branch], 'execution_started': False,
        'test_gold_for_editing': False, 'independent_cumulative_RL': True,
        'checkpoint_retention': 'latest_native_and_latest_HF_export_after_seal'}


def plan(preparation, root, branch, stop_update=None):
    manifest, runtime = load(preparation)
    require(branch in ARMS, 'Unknown branch')
    stop = stop_at(runtime, stop_update)
    return {**base_plan(manifest, runtime, root, branch), 'requested_stop_update': stop,
            'windows': [[start, start+5] for start in range(0, stop, 5)],
            'interim_unless_optimizer_horizon': stop < runtime['optimizer_horizon_updates']}


def model_identity(path):
    from skillnet_cohort.assets import model_inventory
    return digest(model_inventory(path)['files'])


def verify_same_training_batch(batch_path, episodes, *, branch, bank_sha256, update):
    """Prove editor trajectories and fixed-state readout share source rollouts."""
    import torch
    batch = torch.load(batch_path, map_location='cpu', weights_only=True, mmap=True)
    require(batch['schema_version'] == 'skillrl.phase3.direction_batch.v1'
            and batch['branch_id'] == branch and batch['bank_sha256'] == bank_sha256
            and batch['global_update'] == update, 'Foreign readout direction batch')
    direction_ids = {str(item['trajectory_id']) for item in batch['metadata']}
    episode_ids = {item['trajectory_id'] for item in episodes}
    require(len(episode_ids) == len(episodes) == 128 and direction_ids == episode_ids,
            'Editor episodes do not match readout source trajectories')


def command(args, log):
    require(not log.exists(), 'A prior subprocess log exists; reconcile the failed block rather than silently rerunning it')
    log.parent.mkdir(parents=True, exist_ok=True)
    # No API keys in argv, configuration, or this log header.
    with log.open('x') as stream:
        result = subprocess.run([sys.executable, '-B', '-m', *map(str, args)], cwd=ROOT,
                                stdout=stream, stderr=subprocess.STDOUT, check=False)
    require(result.returncode == 0, f'Phase3 subprocess stopped (exit {result.returncode}); inspect {log}')


def prediction_artifacts(root, event, start, end, *, repair=False):
    """Resolve an explicit, non-overwriting retry after a failed prediction."""
    label = f'u{start:04d}-u{end:04d}'
    original = root / 'predictions' / label
    original_log = root / 'logs' / f'predict-u{end:04d}.log'
    receipt_path = event / 'prediction_recovery.json'
    recoveries = {
        'matched-offline-BF16-SDPA-full-vocabulary-v1': (
            label + '-fp32-autocast-v2', f'predict-u{end:04d}-fp32-autocast-v2.log',
            'live actor uses FP32 weights with BF16 autocast; original BF16-weight parity failed'),
        'matched-offline-FP32-weights-BF16-autocast-SDPA-full-vocabulary-v2': (
            label + '-bf16-weights-v3', f'predict-u{end:04d}-bf16-weights-v3.log',
            'resumed native FSDP forward uses BF16 parameters; FP32-weight parity failed'),
    }
    source_path = original / 'source.json'
    backend = strict_json(source_path.read_text()).get('backend') if source_path.exists() else None
    if receipt_path.exists():
        receipt = strict_json(receipt_path.read_text())
        require(backend in recoveries, 'Foreign prediction recovery backend')
        replacement, log_name, reason = recoveries[backend]
        require(receipt == {'original': label, 'replacement': replacement,
                            'failed_log': original_log.name, 'reason': reason}, 'Foreign prediction recovery receipt')
        return (root / 'predictions' / receipt['replacement'],
                root / 'logs' / log_name)
    if repair and source_path.exists() and not (original / 'complete.json').exists():
        require(original_log.is_file(), 'Cannot repair a prediction without its failed log')
        require(backend in recoveries and (backend != 'matched-offline-FP32-weights-BF16-autocast-SDPA-full-vocabulary-v2'
                or start > 0) and 'Offline/live chosen-token parity exceeds the predeclared tolerance' in original_log.read_text(),
                'Prediction recovery is only for a recorded endpoint precision parity failure')
        replacement, _, reason = recoveries[backend]
        write_new(receipt_path, {'original': label, 'replacement': replacement,
            'failed_log': original_log.name, 'reason': reason})
        return prediction_artifacts(root, event, start, end)
    require(not (original / 'source.json').exists() or (original / 'complete.json').exists(),
            'Incomplete prediction requires an explicit non-overwriting recovery')
    return original, original_log


def retire_previous_endpoint(root, event, start, old_hash, new_hash, current_model, current_native, world_size):
    """Reclaim only the preceding generated endpoint after sealing its successor.

    A persisted intent makes an interruption between the two exact deletions
    distinguishable from unexplained missing data. Initial B0 is never touched.
    """
    if start == 0:
        return
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    root, event = Path(root).resolve(), Path(event).resolve()
    require(root not in {ROOT, ROOT.parent, ROOT.parent.parent, Path('/')},
            'Refusing checkpoint retention in a broad workspace/source directory')
    plan_record = strict_json((root / 'plan.json').read_text())
    require(plan_record['root'] == str(root)
            and plan_record['checkpoint_retention'] == 'latest_native_and_latest_HF_export_after_seal',
            'Run plan did not authorize narrow endpoint retention')
    require(start >= 5 and start % 5 == 0 and event == root / 'events' / f'u{start+5:04d}',
            'Invalid retention boundary')
    record = strict_json((event / 'complete.json').read_text())
    identity = strict_json((event / 'identity.json').read_text())
    require((event / 'running_metrics.json').is_file() and record['selected_bank_sha256'],
            'Cannot retire an unsealed evolution event')
    require(identity['old_policy_sha256'] == old_hash and identity['new_policy_sha256'] == new_hash,
            'Retention identity mismatch')
    prediction, _ = prediction_artifacts(root, event, start, start+5)
    readout = strict_json((prediction / 'readout.json').read_text())
    require(strict_json((prediction / 'complete.json').read_text())['readout_sha256'] == digest(readout)
            and readout['identity'] == identity, 'Cannot retire endpoint without a sealed readout')
    shadow = strict_json((event / 'shadow_action_bias.json').read_text())
    require(shadow['readout_sha256'] == digest(readout) and shadow['threshold_decision_applied'] is False,
            'Cannot retire endpoint without the passive action-bias record')
    require(model_identity(current_model) == new_hash, 'Current export changed before retention')
    require(validate_full_checkpoint(current_native)['world_size'] == world_size,
            'Current native checkpoint incomplete before retention')
    targets = [root / 'checkpoints' / f'global_step_{start}', root / 'models' / f'u{start:04d}']
    intent = {'event_complete_sha256': digest(record), 'old_policy_sha256': old_hash,
              'new_policy_sha256': new_hash, 'targets': [str(path.relative_to(root)) for path in targets],
              'scope': 'previous_generated_endpoint_only; episodes_batches_banks_reports_retained'}
    intent_path = event / 'retention_intent.json'
    if not intent_path.exists():
        require(all(path.is_dir() and not path.is_symlink() for path in targets),
                'Unexplained missing previous endpoint; retention has not been authorized by an intent')
    write_new(intent_path, intent)
    for path in targets:
        require(path.parent == root / ('checkpoints' if path.name.startswith('global_step_') else 'models')
                and not path.is_symlink(), 'Retention target escaped the run root')
        if path.exists():
            require(path.is_dir(), 'Retention target is not a directory')
            if path.name.startswith('global_step_'):
                require(validate_full_checkpoint(path)['world_size'] == world_size,
                        'Previous native checkpoint is incomplete')
            else:
                require(model_identity(path) == old_hash, 'Previous export changed before retention')
            shutil.rmtree(path)
    write_new(event / 'retention_complete.json', {**intent, 'targets_absent': True})


def retire_penultimate_checkpoint(root, event, end, current_native, world_size):
    """Keep U(end-1) until the U(end) event is sealed, then reclaim it."""
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    root, event = Path(root).resolve(), Path(event).resolve()
    require(end >= 5 and end % 5 == 0 and event == root / 'events' / f'u{end:04d}',
            'Invalid penultimate checkpoint boundary')
    require((event / 'complete.json').is_file() and (event / 'running_metrics.json').is_file(),
            'Cannot retire a recovery checkpoint before event sealing')
    target = root / 'checkpoints' / f'global_step_{end-1}'
    if not target.exists():
        return
    require(validate_full_checkpoint(current_native)['world_size'] == world_size,
            'Current endpoint checkpoint is incomplete')
    require(target.parent == root / 'checkpoints' and target.is_dir() and not target.is_symlink(),
            'Invalid penultimate checkpoint target')
    require(validate_full_checkpoint(target)['world_size'] == world_size,
            'Penultimate checkpoint is incomplete')
    intent = {'target': str(target.relative_to(root)), 'event_complete_sha256':
              digest(strict_json((event / 'complete.json').read_text())),
              'scope': 'penultimate_native_checkpoint_only_after_sealed_successor'}
    write_new(event / 'penultimate_retention_intent.json', intent)
    shutil.rmtree(target)
    write_new(event / 'penultimate_retention_complete.json', {**intent, 'target_absent': True})


def execute(preparation, root, branch, stop_update=None, resume_update=None, repair_prediction=False,
            retry_editor_request=None):
    from skillnet_cohort.common import exclusive_writer
    from skillnet_cohort.runtime import disk_gate
    from skillnet_cohort.assets import LocalTokenizer, neutral_control
    from phase1.watch_qwen35_checkpoints import validate_full_checkpoint
    from .evaluate import evaluate_bank
    from .evolution import evolve
    manifest, runtime = load(preparation)
    require(branch in ARMS, 'Unknown branch')
    stop = stop_at(runtime, stop_update)
    root, assets = Path(root).resolve(), Path(preparation).resolve().parent
    if stop > runtime['initial_stop_update']:
        first_milestone = root / 'milestones' / f"u{runtime['initial_stop_update']:04d}"
        require((first_milestone / 'complete.json').is_file(),
                'Complete and inspect the U20 interim milestone before requesting continuation')
        first_record = strict_json((first_milestone / 'complete.json').read_text())
        require(first_record['endpoint'] == runtime['initial_stop_update'] and first_record['branch'] == branch
                and first_record['preparation_sha256'] == digest(manifest)
                and all((first_milestone / split / 'complete.json').is_file() for split in ('valid_seen', 'valid_unseen')),
                'Foreign or incomplete U20 continuation milestone')
    setting = strict_json((assets / 'setting.json').read_text())
    require(setting['schema_version'] in ('skillrl.phase3.setting.embedding.v3', 'skillrl.phase3.setting.embedding.v4')
            and setting['initialization']['execution_order'] == list(ARMS)
            and setting['training']['optimizer_horizon_updates'] == runtime['optimizer_horizon_updates']
            and setting['training']['first_execution_stop_update'] == runtime['initial_stop_update']
            and setting['readout']['skill_level_edit_threshold'] is None
            and setting['execution']['approved'] is True,
            'Phase3 setting awaits launch approval and preflight; execution is blocked')
    if setting['editor']['model'] != 'gpt-5.5':
        amendment_path = root.parents[1] / 'editor_model_switch.json'
        require(amendment_path.is_file(), 'Frozen o3 preparation requires an explicit editor-model amendment')
        amendment = strict_json(amendment_path.read_text())
        require(amendment['preparation_sha256'] == digest(manifest)
                and amendment['from_model'] == setting['editor']['model']
                and amendment['to_model'] == 'gpt-5.5'
                and amendment['first_affected_branch'] == 'readout_d'
                and amendment['first_affected_update'] == 5,
                'Editor-model amendment does not match the frozen preparation')
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(map(str, runtime['gpu_ids'])), ALFWORLD_DATA=runtime['data_root'],
                      PYTHONDONTWRITEBYTECODE='1', TOKENIZERS_PARALLELISM='false')
    credentials = ['SKILLRL_PHASE3_EDITOR_API_KEY']
    if router_backend(runtime['router']) == 'external_llm':
        credentials.append('SKILLNET_ROUTER_API_KEY')
    for variable in credentials:
        require(bool(os.environ.get(variable)), f'Missing dedicated {variable}')
    split = strict_json((assets / 'split.json').read_text())
    games = strict_json((assets / 'games.json').read_text())
    initial_model = strict_json((assets / 'model.json').read_text())
    require(model_identity(runtime['model_path']) == initial_model['identity_sha256'], 'B0 weights/tokenizer changed')
    plan_record = base_plan(manifest, runtime, root, branch)
    with exclusive_writer(root):
        write_new(root / 'plan.json', plan_record)
        write_new(root / 'requests' / f'u{stop:04d}.json', {'stop_update': stop,
            'optimizer_horizon_updates': runtime['optimizer_horizon_updates'],
            'plan_sha256': digest(plan_record), 'interim': stop < runtime['optimizer_horizon_updates']})
        bank = Bank.load(assets / 'banks' / f"{manifest['initial_banks'][branch]}.json", manifest['initial_banks'][branch])
        bank.save(root / 'banks')
        router = create_provider(runtime['router'], root, branch)
        editor = JSONClient(APIConfig(stage='editor', model='gpt-5.5', **runtime['editor']), root / 'editor.sqlite3', allow_live=True)
        if retry_editor_request is not None:
            receipt = editor.authorize_timeout_retry(retry_editor_request, timeout_seconds=600)
            write_new(root / 'recovery' / f'editor-retry-{retry_editor_request}.json', receipt)
        tokenizer = LocalTokenizer(runtime['model_path'])
        old_path, old_hash = Path(runtime['model_path']), initial_model['identity_sha256']
        try:
            for start in range(0, stop, 5):
                end = start + 5
                event = root / 'events' / f'u{end:04d}'
                native = root / 'checkpoints' / f'global_step_{end}'
                new_path = root / 'models' / f'u{end:04d}'
                if (event / 'complete.json').exists():
                    record = strict_json((event / 'complete.json').read_text())
                    require(record['before_bank_sha256'] == bank.manifest_sha256, 'Broken bank lineage on resume')
                    source_artifact = record.get('source_artifact', 'source.json')
                    require(source_artifact in ('source.json', 'source-v11.json', 'source-v12.json', 'source-v13.json'),
                            'Unknown event source artifact')
                    require(record['source_sha256'] == digest(strict_json((event / source_artifact).read_text())),
                            'Changed completed event source')
                    bank = Bank.load(event / 'banks' / f"{record['selected_bank_sha256']}.json", record['selected_bank_sha256'])
                    identity = strict_json((event / 'identity.json').read_text())
                    require(identity['old_policy_sha256'] == old_hash and identity['bank_sha256'] == record['before_bank_sha256']
                            and identity['branch_id'] == branch and identity['start'] == start and identity['end'] == end,
                            'Broken endpoint identity chain on resume')
                    prediction, _ = prediction_artifacts(root, event, start, end)
                    readout = strict_json((prediction / 'readout.json').read_text())
                    require(readout['identity'] == identity
                            and strict_json((prediction / 'complete.json').read_text())['readout_sha256'] == digest(readout),
                            'Changed sealed prediction on resume')
                    shadow = strict_json((event / 'shadow_action_bias.json').read_text())
                    require(shadow['readout_sha256'] == digest(readout) and shadow['threshold_decision_applied'] is False,
                            'Changed shadow action-bias record on resume')
                    new_hash = identity['new_policy_sha256']
                    if new_path.exists():
                        require(model_identity(new_path) == new_hash, 'Changed retained endpoint weights on resume')
                    else:
                        receipt = root / 'events' / f'u{end+5:04d}' / 'retention_complete.json'
                        require(receipt.is_file() and f'models/u{end:04d}' in strict_json(receipt.read_text())['targets'],
                                'Unexplained missing endpoint export on resume')
                    if not (event / 'running_metrics.json').exists():
                        from .report import summarize
                        write_new(event / 'running_metrics.json', summarize(root))
                    retire_penultimate_checkpoint(root, event, end, root / 'checkpoints' / f'global_step_{end}',
                                                  len(runtime['gpu_ids']))
                    if start and not (event / 'retention_complete.json').exists():
                        retire_previous_endpoint(root, event, start, old_hash, new_hash, new_path, native, len(runtime['gpu_ids']))
                    old_path, old_hash = new_path, new_hash
                    continue
                disk_gate(root, runtime['storage']['checkpoint_reserve_bytes'], **{
                    name: runtime['storage'][name] for name in ('minimum_free_bytes', 'maximum_run_bytes')})
                bank_path = bank.save(root / 'banks')
                if not native.exists():
                    recovery_here = resume_update is not None and start < resume_update < end
                    require(resume_update is None or recovery_here,
                            'Recovery checkpoint is not inside the first incomplete window')
                    train_args = ['phase3.training', '--preparation', preparation, '--root', root, '--branch', branch,
                                  '--bank-path', bank_path, '--bank-sha256', bank.manifest_sha256,
                                  '--start', start, '--execute']
                    log_name = f'train-u{start:04d}-u{end:04d}'
                    if recovery_here:
                        train_args += ['--resume-update', resume_update]
                        log_name += f'-resume-u{resume_update:04d}'
                    command(train_args, root / 'logs' / f'{log_name}.log')
                    resume_update = None
                checkpoint = validate_full_checkpoint(native)
                require(checkpoint['world_size'] == len(runtime['gpu_ids']), 'Wrong checkpoint world size')
                require((root / 'metrics' / f'u{end:04d}.json').is_file(), 'Block checkpoint exists without completed training metrics')
                if not new_path.exists():
                    command(['phase2.export_model', '--checkpoint', native, '--target', new_path], root / 'logs' / f'export-u{end:04d}.log')
                require((new_path / 'phase2_export.json').exists(), 'Incomplete endpoint export')
                new_hash = model_identity(new_path)
                identity = WindowIdentity(branch, bank.manifest_sha256, old_hash, new_hash, start, end)
                write_new(event / 'identity.json', asdict(identity))
                target, predict_log = prediction_artifacts(root, event, start, end, repair=repair_prediction)
                if not (target / 'complete.json').exists():
                    command(['phase3.predict', '--bank', bank_path, '--bank-sha256', bank.manifest_sha256,
                             '--old-path', old_path, '--new-path', new_path, '--batch', root / 'direction_batches' / f'u{start+1:04d}.pt',
                             '--output', target, '--calibration', root / 'calibration-fp32-autocast-v2.json',
                             '--identity', event / 'identity.json',
                             '--parity-atol', runtime['readout_parity_atol']], predict_log)
                readout = strict_json((target / 'readout.json').read_text())
                require(strict_json((target / 'complete.json').read_text())['readout_sha256'] == digest(readout), 'Changed readout result')
                write_new(event / 'shadow_action_bias.json', {'readout_sha256': digest(readout),
                    'metric': 'M_delta_centered', 'skill_scores': {row['skill_id']: row['M_delta_centered']
                        for row in readout['rows'] if row['supported']},
                    'edit_threshold': None, 'threshold_decision_applied': False,
                    'metric_drives_ranking': ARMS[branch] == 'centered_magnitude',
                    'shadow_record_is_edit_gate': False})
                # Editor failures and readout use the same first training batch
                # sampled by the window-start policy. U(end) Seen validation
                # occurs after updating and must never enter this selector.
                episodes = [strict_json(path.read_text()) for path in
                            sorted((root / 'episodes' / f'u{start+1:04d}' / 'train').glob('*.json'))]
                require(len(episodes) == 128, 'Incomplete old-policy initial training batch for editor evidence')
                verify_same_training_batch(root / 'direction_batches' / f'u{start+1:04d}.pt', episodes,
                    branch=branch, bank_sha256=bank.manifest_sha256, update=start+1)
                allowed = {row['game_id'] for row in games['splits']['train']['games']}

                def gate(candidate):
                    return evaluate_bank(bank=candidate, checkpoint=new_path, policy_sha256=new_hash,
                        games=split['gate'], seeds=runtime['eval_seeds'], data_root=runtime['data_root'],
                        output=event / 'evaluations' / candidate.manifest_sha256, router_api=router,
                        inference_profile=runtime['inference_profile'], gpu_ids=runtime['gpu_ids'])

                bank, _ = evolve(bank, output=event, event_id=f'u{end:04d}', selector=ARMS[branch], api=editor,
                    episodes=episodes, evidence_games=allowed, gate_games={row['game_id'] for row in split['gate']},
                    max_evidence_trajectories=runtime['max_evidence_trajectories'], tolerance_pp=runtime['gate_tolerance_pp'],
                    evaluate=gate, readout_bundle=readout if ARMS[branch] != 'failure_driven' else None, identity=identity,
                    payload_validator=lambda skill: neutral_control(tokenizer, skill.payload))
                bank.save(root / 'banks')
                from .report import summarize
                write_new(event / 'running_metrics.json', summarize(root))
                retire_penultimate_checkpoint(root, event, end, native, len(runtime['gpu_ids']))
                retire_previous_endpoint(root, event, start, old_hash, new_hash, new_path, native, len(runtime['gpu_ids']))
                old_path, old_hash = new_path, new_hash
            # Evaluate the requested milestone, never the best-on-test checkpoint.
            require(old_path.is_dir() and model_identity(old_path) == old_hash,
                    'Latest endpoint export missing or changed before milestone evaluation')
            milestone = root / 'milestones' / f'u{stop:04d}'
            for name in ('valid_seen', 'valid_unseen'):
                evaluate_bank(bank=bank, checkpoint=old_path, policy_sha256=old_hash, games=games['splits'][name]['games'],
                    seeds=runtime['eval_seeds'], data_root=runtime['data_root'], output=milestone / name,
                    router_api=router, inference_profile=runtime['inference_profile'], gpu_ids=runtime['gpu_ids'])
            milestone_record = {'branch': branch, 'endpoint': stop, 'policy_sha256': old_hash,
                'bank_sha256': bank.manifest_sha256, 'preparation_sha256': digest(manifest),
                'evaluated_splits': ['valid_seen', 'valid_unseen'],
                'status': 'interim' if stop < runtime['optimizer_horizon_updates'] else 'horizon_complete'}
            write_new(milestone / 'complete.json', milestone_record)
            if stop == runtime['optimizer_horizon_updates']:
                write_new(root / 'complete.json', milestone_record)
            from .report import summarize
            write_new(milestone / 'running_metrics.json', summarize(root))
        finally:
            router.close()
            editor.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--branch', choices=ARMS, required=True)
    p.add_argument('--stop-update', type=int, help='U20 initially; later five-update milestones use the same frozen preparation')
    p.add_argument('--resume-update', type=int, help='Explicit intra-window native checkpoint recovery; never automatic')
    p.add_argument('--repair-prediction', action='store_true',
                   help='Explicit, non-overwriting precision recovery after a recorded parity failure')
    p.add_argument('--retry-editor-request',
                   help='Original SHA-256 of one recorded editor timeout explicitly authorized for a single retry')
    p.add_argument('--execute', action='store_true')
    a = p.parse_args()
    if a.execute:
        execute(a.preparation, a.root, a.branch, a.stop_update, a.resume_update, a.repair_prediction,
                a.retry_editor_request)
    else:
        from .common import canonical
        print(canonical(plan(a.preparation, a.root, a.branch, a.stop_update)))


if __name__ == '__main__':
    main()
