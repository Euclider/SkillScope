"""Versioned inference identity; no GPU/model initialization during planning."""
from copy import deepcopy
from importlib.metadata import version
from pathlib import Path

from .common import REPO, file_hash, read_json

PROFILE = REPO / 'configs/phase12_vllm_v1.json'
PHASE3_PROFILE = REPO / 'configs/phase3_vllm_v1.json'
PHASE3_MEMORY_PROFILE = REPO / 'configs/phase3_vllm_memory_v2.json'
PHASE3_MEMORY_PROFILE_V3 = REPO / 'configs/phase3_vllm_memory_v3.json'


def registration(path):
    settings = read_json(path)
    registered = (PROFILE, PHASE3_PROFILE, PHASE3_MEMORY_PROFILE, PHASE3_MEMORY_PROFILE_V3)
    if not any(Path(path).resolve() == candidate.resolve() and settings == read_json(candidate)
               for candidate in registered) or settings['execution_authorized']:
        raise ValueError('Unregistered inference profile; no silent backend fallback')
    return {'path': str(Path(path).resolve()), 'sha256': file_hash(path), 'settings': settings}


def validate(settings):
    binding = settings.get('inference_profile')
    if binding is not None and binding != registration(binding['path']):
        raise ValueError('Frozen inference profile changed')
    return binding


def apply_training(cfg, binding):
    validate({'inference_profile': binding})
    cfg.actor_rollout_ref.rollout.name = 'vllm_v1'
    cfg.actor_rollout_ref.rollout.inference_profile = deepcopy(binding)
    cfg.actor_rollout_ref.rollout.tensor_model_parallel_size = 1
    # The older scope profile's microbatch=2 was an HF generate memory bound.
    # V1 schedules up to 16 concurrent sequences, chunking prefill at 8192 tokens.
    cfg.actor_rollout_ref.rollout.micro_batch_size = binding['settings']['max_num_seqs']
    for key in ('dtype', 'max_model_len', 'max_num_batched_tokens', 'max_num_seqs',
                'gpu_memory_utilization', 'enforce_eager', 'enable_prefix_caching', 'enable_chunked_prefill'):
        cfg.actor_rollout_ref.rollout[key] = binding['settings'][key]
    cfg.actor_rollout_ref.rollout.load_format = 'dummy'
    cfg.actor_rollout_ref.rollout.free_cache_engine = False
    # Native transformer-layer FSDP wrapping is restored when HF generate is
    # no longer called on the actor. Optimizer/math/capture are not replaced.


def require_versions(settings):
    for package in ('vllm', 'torch', 'transformers'):
        if version(package).split('+')[0] != settings[package + '_version']:
            raise RuntimeError(f'{package} differs from the frozen vLLM environment')


def make_policy(checkpoint, settings, max_prompt_tokens=4096):
    from .runtime import BoundedPolicy
    binding = validate(settings)
    if binding is None:  # Historical manifests only; not an exception fallback.
        from phase1.eval_skill_margin import TransformersPolicy
        return BoundedPolicy(TransformersPolicy(str(checkpoint)), max_prompt_tokens)
    from .vllm_backend import VLLMPolicy
    return BoundedPolicy(VLLMPolicy(str(checkpoint), binding), max_prompt_tokens)
