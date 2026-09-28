"""Frozen Qwen3.5 TP=1 V1 batch rollout; samples are not readout distributions."""
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.workers.rollout.base import BaseRollout


def pack_outputs(prompts, outputs, *, response_length, pad_token_id):
    ids = prompts.batch['input_ids']
    mask = prompts.batch['attention_mask']
    if len(outputs) != len(ids):
        raise ValueError('vLLM output count differs from the rollout batch')
    responses = torch.full((len(ids), response_length), pad_token_id, dtype=ids.dtype, device=ids.device)
    response_mask = torch.zeros_like(responses)
    logprobs = torch.zeros_like(responses, dtype=torch.float32)
    for row, output in enumerate(outputs):
        if len(output.outputs) != 1:
            raise ValueError('GRPO group replication is external; vLLM n must be 1')
        sample = output.outputs[0]
        tokens = list(sample.token_ids)
        if not 0 < len(tokens) <= response_length or sample.logprobs is None or len(sample.logprobs) != len(tokens):
            raise ValueError('Missing/invalid generated tokens or sampling log probabilities')
        responses[row, :len(tokens)] = torch.tensor(tokens, device=ids.device)
        # Lengths, not PAD/EOS equality: valid EOS remains scored even if PAD=EOS.
        response_mask[row, :len(tokens)] = 1
        logprobs[row, :len(tokens)] = torch.tensor(
            [lp[token].logprob for token, lp in zip(tokens, sample.logprobs)], device=ids.device)
    positions = prompts.batch['position_ids']
    if positions.ndim != 2:
        raise ValueError('Registered text-only rollout requires 2D position ids')
    extra_positions = positions[:, -1:] + torch.arange(1, response_length + 1, device=ids.device)[None]
    return DataProto(TensorDict({'prompts': ids, 'responses': responses,
        'input_ids': torch.cat([ids, responses], dim=-1),
        'attention_mask': torch.cat([mask, response_mask], dim=-1),
        'position_ids': torch.cat([positions, extra_positions], dim=-1),
        'rollout_log_probs': logprobs}, batch_size=[len(ids)]))


class VLLMV1Rollout(BaseRollout):
    def __init__(self, model_path, config, tokenizer):
        from omegaconf import OmegaConf
        from skillnet_cohort.vllm_backend import build_engine
        if config.tensor_model_parallel_size != 1:
            raise ValueError('Only eight TP=1 replicas are registered')
        self.config, self.tokenizer = config, tokenizer
        binding = OmegaConf.to_container(config.inference_profile, resolve=True)
        rank = torch.distributed.get_rank()
        self.inference_engine = build_engine(model_path, binding, training=True,
                                            seed=int(config.get('seed', binding['settings']['seed'])) + rank)

    @torch.no_grad()
    def generate_sequences(self, prompts):
        from vllm import SamplingParams
        ids, masks = prompts.batch['input_ids'], prompts.batch['attention_mask']
        inputs = [{'prompt_token_ids': row[mask.bool()].tolist()} for row, mask in zip(ids, masks)]
        if any(not p['prompt_token_ids'] or len(p['prompt_token_ids']) > self.config.prompt_length for p in inputs):
            raise ValueError('Empty/oversized rollout prompt; no truncation')
        response_length = int(prompts.meta_info.get('response_length', self.config.response_length))
        if not 0 < response_length <= self.config.response_length:
            raise ValueError('Invalid response cap')
        sampling = self.config.val_kwargs if prompts.meta_info.get('validate', False) else self.config
        temperature = float(sampling.get('temperature', 1.0)) if prompts.meta_info.get('do_sample', True) else 0.0
        eos = prompts.meta_info['eos_token_id']
        eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]
        params = SamplingParams(n=1, temperature=temperature, top_p=float(sampling.get('top_p', 1.0)),
            top_k=int(sampling.get('top_k', -1)) or -1, max_tokens=response_length,
            stop_token_ids=eos, logprobs=0, detokenize=False)
        outputs = self.inference_engine.generate(inputs, params, use_tqdm=False)
        pad = prompts.meta_info.get('pad_token_id')
        result = pack_outputs(prompts, outputs, response_length=int(self.config.response_length),
                              pad_token_id=self.tokenizer.eos_token_id if pad is None else pad)
        result.meta_info['inference_backend'] = 'vllm_v1_0.22.0'
        return result
