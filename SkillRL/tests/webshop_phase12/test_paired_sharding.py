import copy
import json

import pytest


def fixtures():
    anchors = [{'task_id': i, 'skill_id': 'gen_001'} for i in range(5)]
    rows = []
    for i, anchor in enumerate(anchors):
        conditions = {f'{m}_{a}': {'success': float(a == 'skill'), 'task_score': 1., 'generated_tokens': 3}
                      for m in ('old', 'new') for a in ('skill', 'control')}
        rows.append({'anchor_id': i, **anchor, 'paired': [{'seed': s, 'conditions': copy.deepcopy(conditions),
                     'm_old': 1., 'm_new': 1., 'delta_m': 0.} for s in range(8)],
                     'm_old': 1., 'm_new': 1., 'delta_m': 0.})
    return anchors, rows


def test_shards_preserve_every_global_anchor_and_merge_in_original_order(tmp_path):
    from webshop_phase12.pairing import anchor_partition, merge_paired_rows
    anchors, rows = fixtures()
    shards = []
    for i in range(2):
        assignments = anchor_partition(anchors, i, 2)
        assert [a for _, a in assignments] == anchors[i::2]
        p = tmp_path/f'shard{i}.jsonl'
        p.write_text(''.join(json.dumps(rows[j])+'\n' for j, _ in assignments))
        shards.append(p)
    assert merge_paired_rows(shards[::-1], anchors, tuple(range(8))) == rows


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'seed', 'condition', 'identity', 'utility'])
def test_merge_rejects_incomplete_or_mismatched_paired_evidence(tmp_path, mutation):
    from webshop_phase12.pairing import merge_paired_rows
    anchors, rows = fixtures()
    if mutation == 'missing': rows.pop()
    if mutation == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    if mutation == 'seed': rows[0]['paired'][0]['seed'] = 99
    if mutation == 'condition': rows[0]['paired'][0]['conditions'].pop('old_skill')
    if mutation == 'identity': rows[0]['task_id'] = 99
    if mutation == 'utility': rows[0]['m_new'] = .5
    p = tmp_path/'shard.jsonl';p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError): merge_paired_rows([p], anchors, tuple(range(8)))
