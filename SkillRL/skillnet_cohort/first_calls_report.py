"""Game-clustered all-first-call analysis and recoverable report publication."""
from pathlib import Path
import json
import os
import tempfile

import numpy as np
import pandas as pd

from .common import file_hash, read_json, write_new_bytes, write_new_json
from .reports import passport, ranking_diagnostics
from phase2.utilities import read_evaluations, margins
from phase2.ranking import evaluate_snapshot


def clustered_units(frame, anchors, repetitions=10000, tau=.05):
    """No invocation/repeat is treated as an independent game or RL seed."""
    anchor_frame = pd.DataFrame(anchors)
    if anchor_frame.duplicated(['skill_id', 'source_trajectory_id']).any():
        raise ValueError('Only one first-call anchor per skill/source trajectory is allowed')
    m = margins(frame); keys = ['skill_id', 'anchor_id', 'game_id', 'context_id', 'phase', 'trigger_step', 'continuation_seed']
    old = m[(m['update'] == 0) & (m.purpose == 'gold')]
    new = m[(m['update'] == 5) & (m.purpose == 'gold')]
    paired = old.merge(new, on=keys, how='outer', suffixes=('_old', '_new'), validate='one_to_one', indicator=True)
    if not paired._merge.eq('both').all():
        raise ValueError('All anchors and continuation seeds require matched endpoint arms')
    paired['source_trajectory_id'] = paired.anchor_id.map({a['anchor_id']: a['source_trajectory_id'] for a in anchors})
    if paired.source_trajectory_id.isna().any():
        raise ValueError('Unknown U0 source trajectory')
    units, games = [], []; rng = np.random.default_rng(20260909)
    for control in ('placebo', 'null'):
        for (skill, context), group in paired.groupby(['skill_id', 'context_id']):
            for phase in ('all', 'initial', 'early', 'middle', 'late'):
                q = group.copy() if phase == 'all' else group[group.phase == phase].copy()
                if q.empty:
                    continue
                q['old'] = q[f'M_{control}_old']; q['new'] = q[f'M_{control}_new']; q['delta'] = q['new']-q['old']
                cols = ['old', 'new', 'delta', 'original_old', 'original_new', f'{control}_old', f'{control}_new']
                # Means over paired repeats/anchors, then trajectories, then games.
                trajectory = q.groupby(['game_id', 'source_trajectory_id'])[cols].mean()
                game = trajectory.groupby('game_id')[cols].mean().reset_index()
                vals = game.delta.to_numpy()
                if len(game) >= 2:
                    from phase2.utilities import paired_bootstrap
                    draws = paired_bootstrap(q, rng, repetitions)
                    low, high = map(float, np.quantile(draws, [.025, .975]))
                    direction = 'positive' if low > tau else 'negative' if high < -tau else 'stable' if low >= -tau and high <= tau else 'uncertain'
                    status = 'paired_game_cluster_bootstrap'
                else:
                    low = high = float('nan'); direction = 'uncertain'; status = 'single_game_interval_undefined'
                row = {'start_update': 0, 'global_update': 5, 'window_horizon': 5, 'window_role': 'test',
                    'control': control, 'skill_id': skill, 'context_id': context, 'phase': phase,
                    'anchor_count': q.anchor_id.nunique(), 'game_count': len(game),
                    'source_trajectory_count': q.source_trajectory_id.nunique(),
                    'continuation_repeats': q.continuation_seed.nunique(), 'paired_anchor_repeats': len(q),
                    'utility_old': float(game.old.mean()), 'utility_new': float(game.new.mean()),
                    'delta_utility': float(vals.mean()), 'ci_low': low, 'ci_high': high,
                    'direction': direction, 'interval_status': status, 'quantity_filter_applied': False,
                    'negative_point_label': bool(vals.mean() < -tau)}
                for endpoint in ('old', 'new'):
                    row[f'original_{endpoint}'] = float(game[f'original_{endpoint}'].mean())
                    row[f'control_{endpoint}'] = float(game[f'{control}_{endpoint}'].mean())
                row['delta_original'] = row['original_new']-row['original_old']
                row['delta_control'] = row['control_new']-row['control_old']
                units.append(row)
                games.extend([{**{k: row[k] for k in ('start_update', 'global_update', 'control', 'skill_id', 'context_id', 'phase')}, **item}
                              for item in game.to_dict('records')])
    return pd.DataFrame(units), pd.DataFrame(games), m


def report(root):
    from phase2.protocol import anchor_sets, expected_trajectory_ids
    from .common import REPO
    root = Path(root).resolve(); plan = read_json(root.parent/'plan.json'); config = read_json(root/'protocol.json')
    frame = read_evaluations(root)
    expected = sum((expected_trajectory_ids(config, REPO, u) for u in (0, 5)), [])
    if len(frame) != len(expected) or set(frame.trajectory_id) != set(expected):
        raise ValueError('Do not publish until every first-call endpoint/arm/repeat is complete')
    anchors = [a for s in anchor_sets(config, REPO) for a in s['anchors']]
    units, games, m = clustered_units(frame, anchors, config['evaluation']['bootstrap_repetitions'], config['evaluation']['tau_M'])
    directory = root/'window_metrics'; output = root/'reports'; seed = plan['jobs'][0]['seed']
    for name, table in (('utility_units', units), ('utility_games', games), ('anchor_margins', m)):
        write_new_bytes(directory/(name+'.parquet'), table.to_parquet(index=False))
    frozen = read_json(root/'window_signals/u0000-u0005/prediction.json')
    metrics, pools, scores = evaluate_snapshot(frozen['ranking_scores'], units[units.control == 'placebo'], config['ranking'])
    # Empty complete-case pools are explicit abstentions, not schema errors.
    if metrics.empty:
        metrics = pd.DataFrame(columns=['context_id', 'phase', 'score', 'target', 'threshold', 'k', 'n_candidates'])
    diagnostics = ranking_diagnostics(scores, {'candidate_pools': pools}, config['ranking'])
    if diagnostics.empty:
        diagnostics = pd.DataFrame(columns=['context_id', 'phase', 'score', 'threshold', 'candidates', 'declines', 'average_precision', 'spearman'])
    coverage = pd.read_csv(root/'support/coverage.csv')
    features = pd.read_parquet(root/'window_signals/u0000-u0005/skill_context_features.parquet')
    keys = ['skill_id', 'context_id', 'phase']
    complete = coverage.merge(features[features.control == 'placebo'][keys+['supported', 'P_int', 'D_contribution', 'C_upd', 'direction_coverage', 'gate_coverage']],
        on=keys, how='left', validate='one_to_one').merge(units[units.control == 'placebo'].drop(columns=['quantity_filter_applied']),
        on=keys, how='left', validate='one_to_one')
    for name, table in (('ranking_metrics', metrics), ('locked_scores_and_gold', scores), ('coverage_and_effects', complete)):
        write_new_bytes(directory/(name+'.csv'), table.to_csv(index=False).encode())
    write_new_json(directory/'ranking_support.json', {'candidate_pools': pools, 'quantity_thresholds_applied': False})
    training_root = Path(plan['jobs'][0]['training_root'])
    performance = pd.read_csv(training_root/'reports/performance.csv')
    tables = {'performance': performance, 'utility_units': units.assign(seed=seed),
        'ranking_diagnostics': diagnostics.assign(seed=seed), 'ranking_budgets': metrics.assign(seed=seed)}
    for name, table in tables.items():
        content = (training_root/'reports/performance.csv').read_bytes() if name == 'performance' else table.to_csv(index=False).encode()
        write_new_bytes(output/(name+'.csv'), content)
    head = passport('all_u0_first_calls_v1')
    boundary = ('\n本版本由用户在seed404旧结果已产生后授权修改覆盖规则，并在seed505仍训练时冻结统一规则。seed404属于事后覆盖扩展，'
        '不是原预登记分析的重新命名。技能内容、policy、router、C/P/D公式及其方向不变。'
        '取消的是数量门槛；缺少真实训练token的分数为NA，不填零。零advantage时按公式得到零分，'
        '同时明确记录reward_direction_observed=False，不据此断言效用稳定。\n\n'
        '每条U0轨迹对每个skill只取首次自然调用，全部首调用锚点均评估；后续调用仅统计次数，不新增锚点。固定原动作前缀回放，'
        '从锚点起对目标skill的本次及后续自然调用施加O/P/N，其他技能不变。'
        '这些是阶段起的剩余效用，不是单次调用的孤立因果效应。\n\n'
        '实际训练batch上的C/P/D不改为首调用读出，不修改advantage；本次去重只作用于效用锚点。\n\n'
        '点估计依次等权平均continuation、首调用源轨迹、game；区间按game聚类重采样，'
        '沿用配对game/continuation重采样，所有arm和端点配对不拆开。单game只给点估计、不伪造跨game置信区间；'
        '阶段分层、anchors、continuation seeds都不是独立RL seeds。排名指标为描述性比较，'
        '不据多重分层选择赢家或声称跨seed显著性；D=0不等于无效用变化。\n\n')
    boundary += ('原训练恢复历史与U0导出的BF16/FP32元数据边界全部继承，未因重算报告而消失；'
        '完整历史说明见本版本plan.json绑定的archived-reports及原训练目录，原批次、OLD概率与检查点保留。\n\n')
    all_rows = complete[complete.phase == 'all']
    observed = all_rows[all_rows.utility_evaluable]
    p1 = head+f'# Phase1：seed {seed}，U0→U5，全部首调用锚点评估\n\n'+boundary
    p1 += f'自然出现技能 {len(observed)}/37；锚点 {len(anchors)}；两端点续跑 {len(frame)}。训练和完整seen/unseen性能不重跑。\n\n'
    p1 += performance[performance.task == 'all'].to_markdown(index=False)+'\n\n'
    cols = ['skill_id', 'source_calls', 'source_games', 'train_decisions', 'train_games', 'train_nonzero_games',
            'anchor_count', 'utility_old', 'utility_new', 'delta_utility', 'ci_low', 'ci_high', 'interval_status']
    p1 += observed[cols].to_markdown(index=False)+'\n'
    p2 = head+f'# Phase2：seed {seed}，无数量门槛的预测与排序\n\n'+boundary
    p2 += f'全部 {len(observed)} 种有自然首调用锚点的技能均计算效用；只有数学上没有实际batch读出或没有匹配效用的行无法进行预测比较，这些行仍完整列于coverage表。\n\n'
    q = diagnostics[(diagnostics.phase == 'all') & (diagnostics.threshold == 0)]
    p2 += q.to_markdown(index=False)+'\n\n'
    p2 += '分阶段、NULL对照、0/5pp阈值、同池top-k、低支持数量和NA原因见同目录CSV及版本化window_metrics。旧报告归档，不删除旧轨迹/评估。\n'
    write_new_bytes(output/'phase1-results.md', p1.encode())
    write_new_bytes(output/'phase2-results.md', p2.encode())
    write_new_json(output/'provenance.json', {'verification_status': 'UNVERIFIED', 'seed': seed,
        'amendment': config['coverage_amendment'], 'plan_sha256': file_hash(root.parent/'plan.json'),
        'source_window': config['coverage_amendment']['legacy_window'], 'result_root': str(root),
        'protocol_sha256': file_hash(root/'protocol.json'), 'readout_labels_used_for_scoring': False,
        'raw_training_reexecuted': False, 'all_registered_first_invocations_evaluated': True,
        'recorded_skills_including_unobserved': 37})
    write_new_json(root/'analysis-complete.json', {'status': 'complete', 'anchors': len(anchors),
        'continuations': len(frame), 'skills_with_utility': len(observed), 'reports': str(output)})
    write_new_json(directory/'summary.json', {'status': 'complete', 'protocol_sha256': file_hash(root/'protocol.json'),
        'anchors': len(anchors), 'continuations': len(frame), 'skills_with_utility': len(observed),
        'coverage_amendment': config['coverage_amendment'], 'scientific_verification': 'UNVERIFIED'})


def publish(root):
    """User-authorized replacement of report views only, after immutable backup."""
    root = Path(root).resolve(); plan = read_json(root.parent/'plan.json')
    if read_json(root/'analysis-complete.json')['status'] != 'complete':
        raise ValueError('No replacement before complete all-first-call analysis')
    seal = read_json(root/'sealed.json')
    if (seal['start'], seal['end']) != (0, 5):
        raise ValueError('Publication requires the completed U0-U5 seal')
    hashes = {r['path']: r['sha256'] for r in seal['files']}
    for item in plan['legacy_reports']:
        name = 'reports/'+Path(item['path']).name
        if hashes.get(name) != file_hash(root/name):
            raise ValueError('Report must match the completed evidence seal')
    replace_reports(root, plan['legacy_reports'], Path(plan['jobs'][0]['training_root'])/'reports',
        root.parent/'archived-reports')


def replace_reports(root, records, target_directory, archive_directory):
    replacements = []
    for item in records:
        target, archive = Path(item['path']), Path(item['archive'])
        new = root/'reports'/target.name
        if (file_hash(target) != item['sha256'] or file_hash(archive) != item['sha256'] or not new.is_file()
                or target.parent != target_directory):
            raise ValueError('Original report changed or replacement/backup is missing; no clobber')
        replacements.append((target, new, item))
    write_new_json(root/'report-publication-intent.json', {'replacements': [item for _, _, item in replacements],
        'new_files': [{'path': str(new), 'sha256': file_hash(new)} for _, new, _ in replacements]})
    for target, new, item in replacements:
        fd, temporary = tempfile.mkstemp(prefix='.first-calls-report-', dir=target.parent)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(new.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            if file_hash(target) != item['sha256']:
                raise ValueError('Report changed during publication')
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
    write_new_json(root/'report-publication.json', {'status': 'published', 'old_reports_recoverable': True,
        'archive_directory': str(archive_directory),
        'files': [{'path': str(t), 'sha256': file_hash(t)} for t, _, _ in replacements]})


def publish_cohort(output):
    """Replace only this cohort's summary, after every seed has the same rule."""
    output = Path(output); plan = read_json(output/'plan.json'); cohort = Path(plan['cohort_root'])
    stage = output/'cohort-summary'; records = []
    for seed in (404, 505, 606):
        variant = (output if seed == 404 else output/f'followup-s{seed}')/f'seed-{seed}'
        if read_json(variant/'complete.json').get('reports_published') is not True:
            raise ValueError('Do not combine mixed or incomplete coverage versions')
        stage.mkdir(parents=True, exist_ok=True)
        (stage/f'seed-{seed}').symlink_to(cohort/f'seed-{seed}', target_is_directory=True)
    for target in sorted((cohort/'reports').glob('*')):
        if target.is_file():
            archive = output/'archived-reports/cohort'/target.name
            write_new_bytes(archive, target.read_bytes())
            records.append({'path': str(target), 'sha256': file_hash(target), 'archive': str(archive)})
    for name in ('performance', 'utility_units', 'ranking_diagnostics', 'ranking_budgets'):
        frame = pd.concat([pd.read_csv(cohort/f'seed-{s}/reports'/(name+'.csv')) for s in (404, 505, 606)], ignore_index=True)
        write_new_bytes(stage/'reports'/(name+'-all-completed-seeds.csv'), frame.to_csv(index=False).encode())
    text = passport('all_u0_first_calls_v1')+'# Phase1–2：同协议三seed首调用评估\n\n'
    text += ('404/505/606分别从同一B0独立训练五轮；完整37技能库冻结。每条来源轨迹每个skill只取首次调用，'
        '所有首调用锚点均评估，数量仅作支持描述、不作门槛。seed404为旧结果已可见后的覆盖扩展；'
        '505/606沿用在505训练时锁定的同一规则。缺真实读出为NA，不补零；一个game不构造跨game置信区间。\n\n'
        '各seed的held-out效用仍配对U0/U5、O/P/N与continuation seeds；来源game与评估随机数可能共用，'
        '不能把skill×phase×anchor×seed行当作独立RL重复。先报告每个seed同池排序与效应，再描述跨seed一致性；'
        '本汇总不自动声称显著或idea已验证。详细历史恢复和精度边界见各seed报告及归档。\n\n')
    diagnostics = pd.read_csv(stage/'reports/ranking_diagnostics-all-completed-seeds.csv')
    q = diagnostics[(diagnostics.phase == 'all') & (diagnostics.threshold == 0)]
    text += q[['seed', 'score', 'candidates', 'declines', 'average_precision', 'spearman']].to_markdown(index=False)+'\n'
    write_new_bytes(stage/'reports/phase12-cohort-summary.md', text.encode())
    write_new_json(stage/'coverage-amendment.json', {'plan_sha256': file_hash(output/'plan.json'),
        'all_seed_reports_version': 'all_u0_first_calls_v1', 'seed404_posthoc_expansion_disclosed': True})
    replace_reports(stage, records, cohort/'reports', output/'archived-reports/cohort')
