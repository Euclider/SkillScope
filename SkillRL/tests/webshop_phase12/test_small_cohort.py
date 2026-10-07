import pytest

from webshop_phase12.assets import make_schedule


def test_small_subset_is_deterministic_distinct_and_split_isolated():
    plan = make_schedule(12087, tasks_per_update=16, eval_tasks=50, eval_seed=20261007)
    assert plan == make_schedule(12087, tasks_per_update=16, eval_tasks=50, eval_seed=20261007)
    assert plan['tasks_per_update'] == 16 and plan['repeats'] == 8 and plan['updates'] == 5
    assert len(plan['eval_ids']) == len(set(plan['eval_ids'])) == 50
    assert set(plan['eval_ids']) <= set(range(500))
    for ids in plan['seeds'].values():
        assert len(ids) == len(set(ids)) == 80
        assert min(ids) >= 1500
        assert not set(ids) & set(plan['eval_ids'])
    assert plan['eval_sampling_seed'] == 20261007


@pytest.mark.parametrize('count', [0, 501])
def test_subset_rejects_invalid_eval_budget(count):
    with pytest.raises(ValueError):
        make_schedule(12087, tasks_per_update=16, eval_tasks=count)


def test_small_config_composes_the_registered_optimization_and_task_budget(monkeypatch):
    from hydra import compose, initialize_config_dir
    from webshop_phase12.assets import ROOT
    from webshop_phase12.accelerated import forward_contract
    monkeypatch.setenv('WEBSHOP_BASE_MODEL', '/model/qwen')
    with initialize_config_dir(config_dir=str(ROOT/'verl/trainer/config'), version_base=None):
        cfg = compose(config_name='webshop54_phase12_small_v4', overrides=[
            'webshop_run.seed=404', 'webshop_run.prepared=/prepared', 'webshop_run.run_root=/run'])
    assert cfg.webshop_run.tasks_per_update == 16 and cfg.env.rollout.n == 8
    assert cfg.webshop_run.updates == 5
    assert cfg.actor_rollout_ref.actor.accumulate_no_sync
    assert cfg.actor_rollout_ref.ref.fsdp_config.wrap_policy.disable
    assert cfg.actor_rollout_ref.actor.active_response_logits_only
    assert forward_contract(cfg) == 'hf-bucket256-active-v4'
