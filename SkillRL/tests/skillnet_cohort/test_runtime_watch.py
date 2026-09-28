import json
from types import SimpleNamespace

from skillnet_cohort.common import write_new_json
from skillnet_cohort import runtime_watch
from skillnet_cohort.recover_again import consumed_seconds


def test_watch_records_terminal_exit_without_retry(tmp_path, monkeypatch):
    root, audit = tmp_path/'run', tmp_path/'attempt'
    root.mkdir()
    write_new_json(audit/'forward_progress/rank-0.json', {'role': 'reference', 'completed_microbatches': 10,
                   'total_microbatches': 20, 'global_update': 1})
    monkeypatch.setattr(runtime_watch, 'gpu_snapshot', lambda: [])
    child = SimpleNamespace(pid=123, poll=lambda: 1)
    watch = runtime_watch.StageWatch(root, audit, 'train', 10000)
    watch.tick([(child, None, 'train')], event='stopped', force=True)
    value = json.loads((audit/'runtime-status.json').read_text())
    assert value['event'] == 'stopped' and value['children'][0]['exit_code'] == 1
    assert value['automatic_retry'] is False
    assert len((audit/'runtime-history.jsonl').read_text().splitlines()) == 1
    watch.last_progress -= 1000
    value = watch.snapshot([(child, None, 'train')], runtime_watch.time.time(), 'running')
    assert value['advisory_stall'] is True
    assert value['stall_action'] == 'warn_only_no_retry_no_kill'


def test_cumulative_runtime_does_not_double_count_original_attempt(tmp_path):
    prior = tmp_path/'recovery-v1'
    write_new_json(prior/'seed-404-exit.json', {'elapsed_seconds': 7000, 'exit_code': 1, 'completed': False})
    write_new_json(prior/'seed-404/supervisor_launch.json', {'started_unix': 100.25})
    write_new_json(prior/'queue_launch.json', {'actual_attempt_started_unix': 100., 'previous_active_seconds': 4000})
    assert consumed_seconds(tmp_path) == 7000.25


def test_watch_tracks_evaluation_without_reading_outcomes(tmp_path, monkeypatch):
    root, audit = tmp_path/'run', tmp_path/'attempt'
    root.mkdir()
    monkeypatch.setattr(runtime_watch, 'gpu_snapshot', lambda: [])
    watch = runtime_watch.StageWatch(root, audit, 'evaluation', None)
    before = runtime_watch.time.time()
    assert watch.snapshot([], before, 'running')['evaluation_completed_files'] == {}
    write_new_json(root/'evaluations/u0000-valid_seen-performance/shards/00/results/episode.json', {})
    write_new_json(root/'windows/u0000-u0005-valid_unseen/evaluations/u0000/trajectories/skill/episode.json', {})
    value = watch.snapshot([], before+600, 'running')
    assert sum(value['evaluation_completed_files'].values()) == 2
    assert not value['advisory_stall']
