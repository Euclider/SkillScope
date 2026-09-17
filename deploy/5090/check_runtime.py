#!/usr/bin/env python3
"""Target-only import and small CUDA checks; no model loading or experiments."""
import argparse
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root/'SkillRL'))
    modules = [
        'torch', 'torchvision', 'transformers', 'ray', 'alfworld', 'textworld',
        'fla', 'causal_conv1d', 'verl', 'verl.workers.fsdp_workers',
        'agent_system.environments.env_package.alfworld.alfworld.agents.environment.alfred_tw_env',
        'transformers.models.qwen3_5.modeling_qwen3_5',
    ]
    for name in modules:
        importlib.import_module(name)
        print('IMPORT_OK', name, flush=True)
    import torch
    if torch.__version__.split('+')[0] != '2.10.0' or torch.version.cuda != '12.8':
        raise RuntimeError('Expected torch 2.10.0 with CUDA 12.8')
    result = {'python': sys.version, 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'visible_gpus': torch.cuda.device_count(), 'model_forward_tested': False,
              'training_tested': False, 'gpu_checks': []}
    if args.gpu:
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        from causal_conv1d import causal_conv1d_fn
        for index in range(torch.cuda.device_count()):
            with torch.cuda.device(index), torch.no_grad():
                a = torch.ones((64, 64), device='cuda', dtype=torch.bfloat16)
                if not torch.all(a @ a == 64).item():
                    raise RuntimeError('BF16 matmul mismatch')
                x = torch.randn((1, 8, 16), device='cuda', dtype=torch.bfloat16)
                w = torch.randn((8, 3), device='cuda', dtype=torch.bfloat16)
                actual = causal_conv1d_fn(x, w, activation='silu')
                expected = torch.nn.functional.silu(torch.nn.functional.conv1d(
                    x.float(), w.float().unsqueeze(1), padding=2, groups=8)[..., :16])
                torch.testing.assert_close(actual.float(), expected, atol=0.04, rtol=0.04)
                torch.cuda.synchronize()
                result['gpu_checks'].append({'index': index, 'name': torch.cuda.get_device_name(index),
                                              'capability': torch.cuda.get_device_capability(index),
                                              'bf16_matmul': 'pass', 'causal_conv1d': 'pass'})
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
