"""Window schedules, current-bank prompts and native single-step training resume."""
from pathlib import Path

from .common import digest, require, strict_json, write_new
from .logicbench import validate_split
from skillnet_cohort.common import file_hash

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/logicbench/sra19/aug_split_v2'


def make_schedule(setting):
    from scripts.prepare_sra_logicbench_phase12 import select_questions
    require(setting['updates'] == 50 and setting['window_updates'] == 5
            and setting['optimizer_horizon'] == 150, 'Wrong approved update schedule')
    manifest = strict_json((DATA / 'manifest.json').read_text())
    for split in ('train', 'dev'):
        require(file_hash(DATA / f'{split}.json') == manifest[f'{split}_sha256'], 'Changed Aug split')
    train = strict_json((DATA / 'train.json').read_text())
    dev = strict_json((DATA / 'dev.json').read_text())
    excluded = {q['context_id'] for q in dev[:32]}
    heldout = select_questions([q for q in dev if q['context_id'] not in excluded],
        seed=setting['split_seed'], count=setting['gate_questions'] + setting['monitor_questions'])
    gate = heldout[:setting['gate_questions']]
    monitor = heldout[setting['gate_questions']:]
    validate_split(train, heldout)
    validate_split(gate, monitor)
    windows = []
    for start in range(0, setting['updates'], setting['window_updates']):
        stream_seed = setting['seed'] + 16 * (start // setting['window_updates'])
        rows = select_questions(train, seed=stream_seed,
                                count=setting['questions_per_update'] * setting['window_updates'])
        windows.append({'start': start, 'end': start + setting['window_updates'],
                        'stream_seed': stream_seed, 'questions': rows})
    return {'schema_version': 'skillrl.phase3.logicbench.schedule.v1', 'windows': windows,
            'gate': gate, 'monitor': monitor, 'source_sha256': {split: manifest[f'{split}_sha256'] for split in ('train', 'dev')},
            'within_window_unique_contexts': True, 'across_window_context_reuse_allowed': True}


def prepare_window(setting, root, bank, window, monitor, provider):
    """Window-frozen question-only retrieval can be done once before its RL steps."""
    import pandas as pd
    from scripts.prepare_sra_logicbench_phase12 import _parquet_rows
    target = Path(root) / 'datasets' / f"u{window['start']:04d}-u{window['end']:04d}"
    identity = {'bank_sha256': bank.manifest_sha256, 'window_sha256': digest(window),
                'monitor_sha256': digest(monitor), 'setting_sha256': digest(setting)}
    write_new(target / 'source.json', identity)
    if (target / 'complete.json').exists():
        prior = strict_json((target / 'complete.json').read_text())
        require(prior['source_sha256'] == digest(identity), 'Changed window dataset identity')
        require(all(file_hash(target / name) == checksum for name, checksum in prior['files'].items()), 'Changed routed dataset')
        return target
    router = provider.for_bank(bank)
    files = {}
    for name, rows in (('train.parquet', window['questions']), ('dev.parquet', monitor)):
        routes = router.route_many([{'candidate_bundle': router.memory.retrieve(''), 'question': q['question']} for q in rows])
        data = _parquet_rows(rows, routes, bank)
        for row in data:
            row['bank_sha256'] = bank.manifest_sha256
            row['skill_version_sha256'] = bank.get(row['selected_skill_id']).version_sha256
        path = target / name
        # Incomplete preparation may resume from deterministic cached routing;
        # no training reads the parquet until complete.json is sealed.
        if path.exists():
            require(pd.read_parquet(path).to_json(orient='records') == pd.DataFrame(data).to_json(orient='records'),
                    'Changed partial window dataset')
        else:
            pd.DataFrame(data).to_parquet(path, index=False)
        files[name] = file_hash(path)
    write_new(target / 'complete.json', {'source_sha256': digest(identity), 'files': files,
        'router_protocol_hash': router.protocol_hash, 'bank_sha256': bank.manifest_sha256})
    return target


def resume_data_action(phase3, global_step):
    require(phase3.domain == 'logicbench', 'Wrong resume adapter')
    require(global_step == phase3.resume_update and phase3.segment_start <= global_step < phase3.segment_end,
            'Native checkpoint does not match the requested resume position')
    return 'new_window' if global_step == phase3.segment_start else 'restore'


def training_configuration(setting, root, bank, bank_path, *, start, gpu_ids, resume_update=None):
    from omegaconf import OmegaConf
    from scripts.inspect_skillrl_alignment import load_config
    require(len(gpu_ids) == setting['gpu_count'] and len(set(gpu_ids)) == len(gpu_ids), 'Wrong GPU world size')
    require(start % 5 == 0 and 0 <= start < setting['updates'], 'Invalid training window')
    resume_update = start if resume_update is None else resume_update
    require(start <= resume_update < start + 5, 'Resume must stay inside the current window')
    overlay = OmegaConf.load(ROOT / 'verl/trainer/config/sra_logicbench19_phase12_v1.yaml')
    del overlay['defaults']
    cfg = OmegaConf.merge(load_config(), overlay)
    root = Path(root).resolve()
    dataset = root / 'datasets' / f'u{start:04d}-u{start+5:04d}'
    cfg.logicbench_run = {'seed': setting['seed'], 'train_files': str(dataset / 'train.parquet'),
                         'val_files': str(dataset / 'dev.parquet'), 'run_root': str(root)}
    cfg.alignment_run = {'seed': setting['seed'], 'model_path': setting['model_path'],
        'train_files': str(dataset / 'train.parquet'), 'val_files': str(dataset / 'dev.parquet'),
        'run_id': f"logicbench-phase3-{bank.branch_id}-s{setting['seed']}",
        'output_dir': str(root / 'checkpoints'), 'archive_dir': str(root / 'unused-phase1'),
        'router_cache': str(root / 'router-local.sqlite3'), 'ray_temp_dir': f'/tmp/lb3-{digest([str(root), start])[:12]}'}
    cfg.phase3 = {'enabled': True, 'domain': 'logicbench', 'branch_id': bank.branch_id,
        'root': str(root), 'bank_path': str(Path(bank_path).resolve()), 'bank_sha256': bank.manifest_sha256,
        'segment_start': start, 'segment_end': start + 5, 'resume_update': resume_update,
        'questions_per_update': setting['questions_per_update'], 'repeats': setting['repeats'],
        'penultimate_recovery_checkpoint': True}
    cfg.phase2 = {'enabled': False}
    cfg.env.phase1_archive.enabled = False
    cfg.data.train_batch_size = setting['questions_per_update']
    cfg.data.seed = setting['seed']
    cfg.data.shuffle = False
    cfg.data.max_prompt_length = setting['max_prompt_tokens']
    cfg.data.max_response_length = setting['max_new_tokens']
    cfg.logicbench_phase12.repeats = setting['repeats']
    cfg.env.rollout.n = setting['repeats']
    cfg.actor_rollout_ref.rollout.temperature = setting['training_temperature']
    cfg.actor_rollout_ref.cohort_seed = setting['seed']
    cfg.actor_rollout_ref.rollout.seed = setting['seed']
    cfg.actor_rollout_ref.actor.optim.total_training_steps = setting['optimizer_horizon']
    cfg.trainer.total_training_steps = setting['optimizer_horizon']
    cfg.trainer.total_epochs = setting['optimizer_horizon']
    cfg.trainer.n_gpus_per_node = len(gpu_ids)
    cfg.trainer.project_name = 'logicbench_phase3_v13'
    cfg.trainer.default_local_dir = str(root / 'checkpoints')
    cfg.trainer.max_actor_ckpt_to_keep = 2
    cfg.trainer.max_critic_ckpt_to_keep = 2
    cfg.trainer.resume_mode = 'resume_path' if resume_update else 'disable'
    cfg.trainer.resume_from_path = str(root / 'checkpoints' / f'global_step_{resume_update}') if resume_update else None
    cfg.ray_init.num_cpus = 32
    cfg.ray_init.object_store_memory = 8 * 2**30
    cfg.ray_init.include_dashboard = False
    OmegaConf.resolve(cfg)
    return cfg
