"""Idle single-GPU forward smoke; never substitutes for eight-rank acceptance."""
import argparse
from pathlib import Path


def main():
    import torch
    from omegaconf import OmegaConf
    from transformers import AutoModelForCausalLM
    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from phase3.common import write_new
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=Path, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    payload = torch.load(a.batch, map_location='cpu', weights_only=True, mmap=True)
    tensors = payload['tensors']
    lengths = tensors['actual_loss_mask'].sum(-1)
    indices = sorted({int(lengths.argmin()), int(lengths.argmax()), 0, len(lengths)//2})
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                  attn_implementation='sdpa', local_files_only=True).cuda().eval()
    actor = DataParallelPPOActor(OmegaConf.create(dict(use_torch_compile=False,
                  ulysses_sequence_parallel_size=1)), model)
    rows = []
    for i in indices:
        data = {k: tensors[k][i:i+1].cuda() for k in
                ('input_ids', 'attention_mask', 'position_ids', 'responses')}
        valid = tensors['actual_loss_mask'][i:i+1].bool().cuda()
        if not valid.any():
            continue
        with torch.no_grad():
            actor.response_logits_only = False
            e0, p0 = actor._forward_micro_batch(data, payload['temperature'], True)
            actor.response_logits_only = True
            e1, p1 = actor._forward_micro_batch(data, payload['temperature'], True)
        rows.append(dict(row=i, length=int(lengths[i]),
            max_logprob_error=(p0-p1)[valid].abs().max().item(),
            max_entropy_error=(e0-e1)[valid].abs().max().item()))
    write_new(a.output, dict(status='single_gpu_forward_only_not_distributed_acceptance', rows=rows,
            peak_gib=torch.cuda.max_memory_allocated()/2**30, model=a.model,
            batch=str(a.batch), masks_and_position_ids='unaltered'))
    assert rows and all(x['max_logprob_error'] <= 1e-5 and x['max_entropy_error'] <= 1e-5 for x in rows)
    print('SINGLE_GPU_FORWARD_PASS', rows, flush=True)


if __name__ == '__main__':
    main()
