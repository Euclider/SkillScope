import json
from dataclasses import replace

import pytest
import torch

from phase3.common import ProtocolError
from phase3.readout import CompactReadout, WindowIdentity


def inputs(n=35, vocab=101):
    g = torch.Generator().manual_seed(41)
    vectors = [torch.randn(n, vocab, generator=g).log_softmax(-1) for _ in range(4)]
    return (*vectors, torch.arange(n) % vocab, torch.linspace(-2, 2, n))


def test_fast_signals_are_bitwise_identical_for_all_consumed_fields():
    from phase2.stable_direction import token_signals
    from phase3.fast_direction import token_signals as fast
    args = inputs()
    original = token_signals(*args)
    actual = fast(*args)
    for key in actual:
        assert torch.equal(actual[key], original[key]), key
    args = list(args)
    args[0][0] = -100.
    args[0][0, 0] = 0.
    for key, value in fast(*args).items():
        assert torch.equal(value, token_signals(*args)[key]), key


@pytest.mark.parametrize('bad', [float('nan'), float('inf')])
def test_fast_signals_preserve_input_validation(bad):
    from phase3.fast_direction import token_signals
    args = list(inputs(3))
    args[0][0, 0] = bad
    with pytest.raises(ValueError):
        token_signals(*args)
    with pytest.raises(ValueError):
        token_signals(*inputs(3), epsilon=0)


def accumulator():
    return CompactReadout(WindowIdentity('readout_d', 'a'*64, 'b'*64, 'c'*64, 0, 5),
                          {'s': 'd'*64, 'unused': 'e'*64})


def add(acc, i):
    a, b, c, d, actions, advantages = inputs(3)
    acc.add(skill_id='s', skill_version_sha256='d'*64, decision_id=f'{i:04d}',
            game_id=f'g{i%2}', trajectory_id=f't{i%4}', old_original=a,
            new_original=b, old_placebo=c, new_placebo=d,
            actions=actions, advantages=advantages * (i % 3 - 1))


def test_contiguous_shards_reproduce_serial_game_equal_aggregation_exactly():
    original, merged = accumulator(), accumulator()
    for i in range(17):
        add(original, i)
    from phase3.parallel_predict import partition
    ranges = [partition(17, i, 4) for i in range(4)]
    assert ranges == [(0, 4), (4, 8), (8, 12), (12, 17)]
    for lo, hi in ranges:
        shard = accumulator()
        for i in range(lo, hi):
            add(shard, i)
        merged.merge_state(json.loads(json.dumps(shard.export_state())))
    assert merged.bundle() == original.bundle()
    with pytest.raises(ProtocolError, match='Duplicate'):
        merged.merge_state(shard.export_state())


def test_merge_rejects_changed_window_or_missing_decision_tokens():
    shard = accumulator()
    add(shard, 0)
    bad = shard.export_state()
    bad['identity']['end'] = 10
    with pytest.raises(ProtocolError):
        accumulator().merge_state(bad)
    bad = shard.export_state()
    bad['seen'].append('foreign')
    with pytest.raises(ProtocolError):
        accumulator().merge_state(bad)


def test_partition_covers_every_decision_once_and_rejects_empty_shards():
    from phase3.parallel_predict import partition
    for n in (4, 5, 17, 5351):
        ranges = [partition(n, i, 4) for i in range(4)]
        assert [j for lo, hi in ranges for j in range(lo, hi)] == list(range(n))
    with pytest.raises(ProtocolError):
        partition(3, 0, 4)


def test_completed_parallel_cache_rejects_a_different_batch(tmp_path):
    from types import SimpleNamespace
    from phase3.bank import Bank, Skill
    from phase3.common import digest, write_new
    from phase3.parallel_predict import run_parallel
    from skillnet_cohort.common import file_hash
    bank = Bank('test', [Skill('s', 'name', 'description', '### name\n\nbody')], source='0'*64)
    bank_path = bank.save(tmp_path / 'bank')
    identity = WindowIdentity('test', bank.manifest_sha256, 'b'*64, 'c'*64, 0, 5)
    from dataclasses import asdict
    identity_path = write_new(tmp_path / 'identity.json', asdict(identity))
    batch = tmp_path / 'batch.pt'
    batch.write_bytes(b'changed input which must never reach a torch loader')
    write_new(batch.with_suffix('.json'), {'sha256': file_hash(batch)})
    output = tmp_path / 'prediction'
    result = CompactReadout(identity, bank.active_versions).bundle()
    write_new(output / 'readout.json', result)
    write_new(output / 'source.json', {'batch_sha256': 'a'*64})
    write_new(output / 'complete.json', {'readout_sha256': digest(result)})
    args = SimpleNamespace(bank=bank_path, bank_sha256=bank.manifest_sha256,
                           identity=identity_path, output=output, batch=batch, parity_atol=.03)
    with pytest.raises(ProtocolError, match='source'):
        run_parallel(args, [0, 1])


def test_activation_only_at_registered_boundary_with_accepted_sources(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from phase3.common import write_new
    from phase3 import parallel_predict as module
    from skillnet_cohort.common import file_hash
    identity = write_new(tmp_path / 'identity.json', {'branch_id': 'skillrl_failure', 'start': 10})
    args = SimpleNamespace(output=tmp_path / 'runs/skillrl_failure/predictions/u0010-u0015',
                           identity=identity)
    assert module.maybe_dispatch(args) is False
    receipt = write_new(tmp_path / 'acceptance.json', {'passed': True, 'source_hashes': module.source_hashes()})
    write_new(tmp_path / 'readout-speed-next-window-v1.json', {
        'schema': 'phase3.readout.speed.v1', 'run_root': str(tmp_path),
        'start_updates': {'skillrl_failure': 15}, 'gpu_ids': list(range(8)),
        'acceptance_receipt': str(receipt), 'acceptance_sha256': file_hash(receipt),
        'source_hashes': module.source_hashes()})
    assert module.maybe_dispatch(args) is False
    args.identity = write_new(tmp_path / 'next.json', {'branch_id': 'skillrl_failure', 'start': 15})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0,1,2,3,4,5,6,7')
    calls = []
    monkeypatch.setattr(module, 'run_parallel', lambda a, ids: calls.append(ids))
    assert module.maybe_dispatch(args) is True
    assert calls == [list(range(8))]
    monkeypatch.setattr(module, 'source_hashes', lambda: {'changed': 'code'})
    with pytest.raises(ProtocolError, match='not accepted'):
        module.maybe_dispatch(args)
