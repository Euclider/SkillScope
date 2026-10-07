"""Deterministic anchor partitioning and strict paired-evidence merge."""
import json
import math


CONDITIONS = {'old_skill', 'old_control', 'new_skill', 'new_control'}


def anchor_partition(anchors, shard, shards):
    if shards < 1 or not 0 <= shard < shards:
        raise ValueError('Invalid anchor shard')
    return [(i, anchors[i]) for i in range(shard, len(anchors), shards)]


def merge_paired_rows(paths, anchors, seeds):
    found = {}
    def same(a, b):
        return math.isfinite(float(a)) and abs(float(a)-float(b)) <= 1e-12
    for path in paths:
        for line in path.read_text().splitlines():
            row = json.loads(line); index = row['anchor_id']
            if index in found or not 0 <= index < len(anchors):
                raise ValueError('Duplicated or invalid anchor ID')
            anchor = anchors[index]
            if any(row[key] != anchor[key] for key in ('task_id', 'skill_id')):
                raise ValueError('Anchor task/skill identity differs')
            pairs = row['paired']
            if [p['seed'] for p in pairs] != list(seeds):
                raise ValueError('Continuation seed coverage differs')
            old = []; new = []
            for pair in pairs:
                conditions = pair['conditions']
                if set(conditions) != CONDITIONS:
                    raise ValueError('Incomplete four-condition pairing')
                if any(c['success'] not in (0., 1.) or not math.isfinite(c['task_score'])
                       or c['generated_tokens'] < 0 for c in conditions.values()):
                    raise ValueError('Invalid continuation outcome')
                mo = conditions['old_skill']['success']-conditions['old_control']['success']
                mn = conditions['new_skill']['success']-conditions['new_control']['success']
                if not all(same(pair[k], v) for k, v in [('m_old', mo), ('m_new', mn), ('delta_m', mn-mo)]):
                    raise ValueError('Paired utility differs from raw outcomes')
                old.append(mo); new.append(mn)
            mo = sum(old)/len(old); mn = sum(new)/len(new)
            if not all(same(row[k], v) for k, v in [('m_old', mo), ('m_new', mn), ('delta_m', mn-mo)]):
                raise ValueError('Mean utility differs from paired outcomes')
            found[index] = row
    if set(found) != set(range(len(anchors))):
        raise ValueError('Missing anchor evidence')
    return [found[i] for i in range(len(anchors))]
