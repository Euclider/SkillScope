"""Native vLLM V1 inference shared by rollout, performance, anchors and O/P/N."""
from .inference import require_versions, validate


def normalized_text_weights(weights):
    for name, value in weights:
        if name.startswith(('model.visual.', 'visual.', 'mtp.')):
            continue
        if name.startswith('model.language_model.'):
            name = 'model.' + name[len('model.language_model.'):]
        elif name.startswith('language_model.'):
            name = name[len('language_model.'):]
        yield name, value


def text_config(config):
    # vLLM first calls an override with a deliberately empty dummy config to
    # probe model-type overrides, before loading the actual checkpoint config.
    if getattr(config, 'model_type', '').startswith('dummy_'):
        return config
    text = getattr(config, 'text_config', config)
    if text.model_type != 'qwen3_5_text':
        raise ValueError('This registered backend supports dense Qwen3.5 text only')
    from copy import deepcopy
    text = deepcopy(text)
    text.architectures = ['SkillScopeQwen35Text']
    return text


def build_engine(model_path, binding, *, training=False, seed=None):
    validate({'inference_profile': binding})
    settings = binding['settings']
    require_versions(settings)
    import os
    from vllm import LLM, ModelRegistry
    ModelRegistry.register_model('SkillScopeQwen35Text',
                                'skillnet_cohort.vllm_qwen35:SkillScopeQwen35Text')
    options = {key: settings[key] for key in (
        'dtype', 'max_model_len', 'max_num_batched_tokens', 'max_num_seqs',
        'gpu_memory_utilization', 'enforce_eager', 'enable_prefix_caching', 'enable_chunked_prefill')}
    if training:
        # apply_model then runs locally, not via pickled GPU tensors. Existing
        # torchrun/Ray ranks own the devices; no nested vLLM GPU worker pool.
        os.environ['VLLM_ENABLE_V1_MULTIPROCESSING'] = '0'
    return LLM(model=str(model_path), tokenizer=str(model_path),
               tensor_parallel_size=1, distributed_executor_backend='external_launcher' if training else 'uni',
               hf_overrides=text_config, load_format='dummy' if training else 'auto',
               enable_sleep_mode=training, seed=settings['seed'] if seed is None else seed,
               generation_config='vllm', disable_log_stats=True, **options)


class VLLMPolicy:
    def __init__(self, checkpoint, binding):
        self.engine = build_engine(checkpoint, binding)
        self.tokenizer = self.engine.get_tokenizer()

    def generate_batch(self, requests):
        from vllm import SamplingParams
        prompts, params = [], []
        for request in requests:
            rendered = self.tokenizer.apply_chat_template(
                [{'role': 'user', 'content': request['prompt']}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            if len(ids) > 4096 or not 0 < request['max_new_tokens'] <= 512:
                raise ValueError('Request exceeds frozen 4096+512 boundary; no truncation')
            prompts.append({'prompt_token_ids': ids})
            params.append(SamplingParams(n=1, seed=int(request['seed']),
                temperature=float(request['temperature']), top_p=float(request['top_p']),
                max_tokens=int(request['max_new_tokens']), detokenize=True))
        outputs = self.engine.generate(prompts, params, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise RuntimeError('vLLM dropped a request')
        return [(o.outputs[0].text, len(o.prompt_token_ids), len(o.outputs[0].token_ids)) for o in outputs]

    def generate(self, prompt, seed, temperature, top_p, max_new_tokens):
        return self.generate_batch([dict(prompt=prompt, seed=seed, temperature=temperature,
                                       top_p=top_p, max_new_tokens=max_new_tokens)])[0]
