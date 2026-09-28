"""FP32 master delta from reconstructible B0, never upcast a lossy U0 export."""
from contextlib import ExitStack
import math
from pathlib import Path

import torch
from safetensors import safe_open

from .common import file_hash, load_preparation, read_json


def compare_state(old, target):
    with ExitStack() as stack:
        index = {}
        for path in sorted(Path(target).glob('*.safetensors')):
            handle = stack.enter_context(safe_open(path, framework='pt', device='cpu'))
            for name in handle.keys():
                if name in index:
                    raise ValueError('Duplicate endpoint parameter')
                index[name] = handle
        if set(index) != set(old):
            raise ValueError('Reconstructed native / endpoint parameter names differ')
        norms = []
        # Native Qwen3.5 has one shared input/output embedding, even if both
        # state-dict names were serialized independently by the FSDP merger.
        for name in sorted(index):
            a, b = old[name], index[name].get_tensor(name)
            if a.dtype != torch.float32 or b.dtype != torch.float32 or a.shape != b.shape:
                raise ValueError('FP32 native parameter mismatch: '+name)
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError('Nonfinite native parameter')
            if name == 'lm_head.weight':
                embed = 'model.embed_tokens.weight'
                if (not torch.equal(a, old[embed])
                        or not torch.equal(b, index[embed].get_tensor(embed))):
                    raise ValueError('Registered tied embeddings disagree')
                continue
            diff = b-a
            norms.append({'parameter': name, 'elements': a.numel(),
                'delta_squared': float(diff.square().sum(dtype=torch.float64)),
                'old_squared': float(a.square().sum(dtype=torch.float64))})
    ds = sum(x['delta_squared'] for x in norms)
    os = sum(x['old_squared'] for x in norms)
    return {'delta_l2': math.sqrt(ds), 'relative_delta_l2': math.sqrt(ds/os),
            'unique_parameters': sum(x['elements'] for x in norms), 'tensors': norms}


def registered_initial_delta(config, target):
    runtime = config.get('runtime', {})
    if (runtime.get('kind') != 'skillnet37' or config.get('parent_update') != 0
            or config.get('capture_scope') != 'window_start_old_only_v1'):
        raise ValueError('B0 reconstruction only applies to this registered independent window')
    prep = Path(runtime['preparation'])
    load_preparation(prep)
    spec = read_json(prep.parent/'spec.json')
    inventory = read_json(prep.parent/'model.json')
    source = Path(spec['model_path']).resolve()
    if str(source) != inventory['model_path']:
        raise ValueError('Changed registered B0 source')
    for item in inventory['files']:
        path = source/item['path']
        if not path.resolve().is_relative_to(source) or file_hash(path) != item['sha256']:
            raise ValueError('Registered pretrained initialization changed')
    from transformers import AutoConfig, AutoModelForCausalLM
    native_config = AutoConfig.from_pretrained(source, local_files_only=True)
    # Match the actual actor loader, including its legacy torch_dtype keyword.
    model, info = AutoModelForCausalLM.from_pretrained(source, config=native_config,
        torch_dtype=torch.float32, local_files_only=True, output_loading_info=True)
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise ValueError('B0 reconstruction has missing/mismatched weights')
    model.to(torch.float32)
    if model.config.model_type != 'qwen3_5_text' or not model.config.tie_word_embeddings:
        raise ValueError('Unexpected registered native model architecture')
    value = compare_state(model.state_dict(), target)
    value.update(status='PASS', precision='FP32_native_master_parameters',
        start_source='CPU reconstruction of registered pretrained B0 with the native actor initialization recipe',
        source_model_manifest_sha256=file_hash(prep.parent/'model.json'),
        source_model_path=str(source), old_HF_snapshot_used_as_FP32_master=False,
        additional_environment_rollouts=0, optimizer_updates=0,
        initialization_recipe='AutoConfig + AutoModelForCausalLM.from_pretrained(torch_dtype=float32); then model.to(float32)')
    return value
