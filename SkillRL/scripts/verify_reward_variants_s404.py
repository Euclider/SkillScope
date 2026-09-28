"""Independent sklearn/scipy check of the completed, frozen CPU analysis.

Does not regenerate scores, change formulas, evaluate another seed, or replace
reports. Adds a separate verification receipt and direction confusion counts.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from skillnet_cohort.common import file_hash, read_json, write_new_bytes, write_new_json
from skillnet_cohort.reward_variant_analysis import DEFAULT_OUTPUT, SOURCE, check_plan, verify_records


def read(path):
    return pd.read_csv(path, keep_default_na=False, na_values=[''], float_precision='round_trip')


def verify(output):
    output = Path(output).resolve()
    if (output/'independent-verification.json').exists():
        raise FileExistsError('Verification receipt already exists')
    complete = read_json(output/'complete.json')
    if complete['status'] != 'complete' or complete['provenance_sha256'] != file_hash(output/'provenance.json'):
        raise ValueError('Incomplete or changed result')
    plan = check_plan(output)
    provenance = read_json(output/'provenance.json')
    verify_records([{'path': str(output/r['path']), 'sha256': r['sha256']} for r in provenance['files']])
    diagnostics = read(output/'ranking_diagnostics.csv')
    scores = read(output/'skill_scores.csv')
    pools = read_json(SOURCE/'ranking-snapshots.json')['candidate_pools']
    units = pd.read_parquet(SOURCE/'reused-labels/utility_units.parquet')
    max_error = {}; checked = 0; confusions = []
    def check(got, expected, name):
        if not np.isclose(got, expected, atol=2e-12, rtol=2e-12, equal_nan=True):
            raise ValueError(f'Independent mismatch {name}: {got} != {expected}')
        if np.isfinite(got) and np.isfinite(expected):
            max_error[name] = max(max_error.get(name, 0.), abs(got-expected))
    for control in ('placebo', 'null'):
        for pool in pools[control+'/stable_raw']:
            identities = pool['shared_skill_ids']
            if not identities:
                continue
            part = scores[(scores.control == control)&(scores.phase == pool['phase'])&(scores.context_id == pool['context_id'])]
            matrix = part.pivot(index='score', columns='skill_id', values='value').reindex(columns=identities)
            u = units[(units.control == control)&(units.phase == pool['phase'])&(units.context_id == pool['context_id'])].set_index('skill_id').reindex(identities)
            target = -u.delta_utility.to_numpy(float)
            expected_rows = diagnostics[(diagnostics.control == control)&(diagnostics.phase == pool['phase'])&(diagnostics.context_id == pool['context_id'])]
            for row in expected_rows.itertuples():
                v = matrix.loc[row.score].to_numpy(float)
                positive, negative = target > row.threshold+1e-12, target < -row.threshold-1e-12
                label = positive | negative
                ap = average_precision_score(positive, v) if positive.any() else np.nan
                auc = roc_auc_score(positive, v) if positive.any() and not positive.all() else np.nan
                auc_dir = roc_auc_score(positive[label], v[label]) if positive.any() and negative.any() else np.nan
                variable = np.unique(v).size > 1 and np.unique(target).size > 1
                rho = spearmanr(v, target).statistic if variable else np.nan
                tau = kendalltau(v, target).statistic if variable else np.nan
                for key, expected in [('average_precision', ap), ('auroc_decline_vs_rest', auc),
                    ('auroc_decline_vs_increase', auc_dir), ('spearman', rho), ('kendall', tau)]:
                    check(getattr(row, key), expected, key)
                checked += 1
                if pool['phase'] == 'all' and row.signed and row.mode in ('reward', 'unsigned'):
                    pred_down, pred_up = v > 1e-12, v < -1e-12
                    correct_down = int((pred_down & positive).sum()); correct_up = int((pred_up & negative).sum())
                    abstentions = int((label & ~(pred_down | pred_up)).sum())
                    n = int(label.sum()); ndown = int(positive.sum()); nup = int(negative.sum())
                    accuracy = (correct_down+correct_up)/n if n else np.nan
                    check(row.sign_accuracy_abstention_wrong, accuracy, 'sign_accuracy_abstention_wrong')
                    confusions.append({'control': control, 'threshold': row.threshold, 'score': row.score,
                        'nonzero_change_skills': n, 'decline_skills': ndown, 'increase_skills': nup,
                        'correct_declines': correct_down, 'correct_increases': correct_up,
                        'abstentions': abstentions, 'accuracy_abstention_wrong': accuracy,
                        'balanced_accuracy_abstention_wrong': .5*(correct_down/ndown+correct_up/nup) if ndown and nup else np.nan,
                        'always_increase_accuracy': nup/n if n else np.nan,
                        'always_decline_accuracy': ndown/n if n else np.nan})
            print(f'checked {control}/{pool["phase"]}: {len(expected_rows)} metric rows', flush=True)
    mapping = {
        'D_original::token::reward': ('D_contribution', 'stable_raw'),
        'D_ungated::token::reward': ('D_ungated_contribution', 'stable_raw'),
        'D_signed::token::reward': ('P_int', 'stable_raw'),
        'C_raw::token::reward': ('C_upd', 'stable_raw'),
        'C_centered::token::reward': ('C_upd', 'stable_centered_gate'),
        'M_delta_raw::token::magnitude': ('delta_norm', 'stable_raw'),
        'M_delta_centered::token::magnitude': ('delta_centered_norm', 'stable_raw'),
        'M_kl::token::magnitude': ('forward_kl_original', 'stable_raw'),
        'M_js::token::magnitude': ('js_original', 'stable_raw'),
    }
    old = read(SOURCE/'reports/ranking_diagnostics.csv'); comparisons = []
    keys = ['control', 'context_id', 'phase', 'threshold']
    for name, (old_name, variant) in mapping.items():
        q = diagnostics[diagnostics.score == name].merge(old[(old.score == old_name)&(old.variant == variant)],
            on=keys, how='outer', validate='one_to_one', suffixes=('_new', '_old'), indicator=True)
        if not q._merge.eq('both').all():
            raise ValueError('Old reporting pool changed')
        for metric in ['candidates', 'declines', 'average_precision', 'auroc_decline_vs_rest', 'spearman', 'kendall']:
            for got, expected in zip(q[metric+'_new'], q[metric+'_old']):
                check(got, expected, 'legacy_'+metric)
        comparisons.append({'score': name, 'matched_old_rows': len(q), 'matched': True})
    confusion_path = output/'direction-confusions.csv'
    write_new_bytes(confusion_path, pd.DataFrame(confusions).to_csv(index=False).encode())
    check_plan(output)
    receipt = {'status': 'PASS', 'verified_unix': time.time(), 'primary_and_stratified_metric_rows': checked,
        'independent_engines': ['sklearn.average_precision_score', 'sklearn.roc_auc_score',
            'scipy.spearmanr', 'scipy.kendalltau', 'manual_direction_confusion'],
        'max_abs_errors': max_error, 'legacy_report_comparisons': comparisons,
        'input_and_runtime_hashes_unchanged': True, 'new_formula_selection_or_score_computation': False,
        'same_skill_pools': True, 'scientific_status': 'ANALYZED_single_seed_posthoc_not_confirmatory',
        'analysis_complete_sha256': file_hash(output/'complete.json'),
        'verification_source_sha256': file_hash(Path(__file__)),
        'additional_files': [{'path': str(confusion_path), 'sha256': file_hash(confusion_path)}],
        'tests': {'path': str(output.parent.parent.parent.parent/'code_checks/reward-variants-s404-20260922-v1/regression-tests.xml'),
                  'note': 'Test report location is also recorded in the research handoff; this verifier checks numerical outputs independently.'}}
    # Do not guess a relative test path in the receipt.
    receipt.pop('tests')
    write_new_json(output/'independent-verification.json', receipt)
    print(f'PASS: {checked} independent metric rows, {len(comparisons)} original methods', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    verify(parser.parse_args().output)
