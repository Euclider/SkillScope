import json
import sys
from pathlib import Path

import pytest


def module():
    import scripts.watch_logicbench_phase3_resume as queue
    return queue


def snapshot(busy=False):
    return [dict(index=i, uuid=f'gpu{i}', used_mib=100,
                 total_mib=80000, utilization=0,
                 compute_pids=[123] if busy and i == 3 else [])
            for i in range(8)]


def test_proxy_cleanup_preserves_credentials_without_mutating_parent():
    queue = module()
    original = {'HTTP_PROXY': 'bad', 'https_proxy': 'bad',
                'SKILLRL_PHASE3_EDITOR_API_KEY': 'test-only', 'PATH': '/bin'}
    clean = queue.launch_environment(original, Path('/cache'))
    assert 'HTTP_PROXY' not in clean and 'https_proxy' not in clean
    assert clean['TIKTOKEN_CACHE_DIR'] == '/cache'
    assert clean['SKILLRL_PHASE3_EDITOR_API_KEY'] == 'test-only'
    assert original['HTTP_PROXY'] == 'bad'


@pytest.mark.parametrize('returncode', [0, 7])
def test_waits_for_original_cards_and_stability_then_launches_only_once(tmp_path, monkeypatch, returncode):
    queue = module()
    samples = iter([snapshot(True), snapshot(), snapshot(True),
                    snapshot(), snapshot(), snapshot()])
    monkeypatch.setattr(queue, 'gpu_snapshot', lambda: next(samples))
    sleeps = []
    monkeypatch.setattr(queue.time, 'sleep', lambda seconds: sleeps.append(seconds))
    counter = tmp_path / 'calls'
    command = [sys.executable, '-c',
               'from pathlib import Path; import sys; '
               'p=Path(sys.argv[1]); p.write_text(p.read_text()+"x" if p.exists() else "x"); '
               'sys.exit(int(sys.argv[2]))', str(counter), str(returncode)]
    status = queue.wait_and_launch(tmp_path, [1, 2, 3, 4], command,
                                   dict(queue.os.environ), lambda: {}, poll_seconds=1)
    assert counter.read_text() == 'x'
    assert status['exit_code'] == returncode
    assert status['state'] == ('finished' if returncode == 0 else 'failed')
    assert len(sleeps) == 4
    history = [json.loads(line) for line in (tmp_path / 'history.jsonl').read_text().splitlines()]
    assert history[0]['blocked_gpu_ids'] == [3]
    assert history[0]['state'] == 'waiting'


def test_failed_probe_resets_stability(tmp_path, monkeypatch):
    queue = module()
    samples = iter([snapshot(), OSError('probe failed'), snapshot(), snapshot(), snapshot()])
    def probe():
        value = next(samples)
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(queue, 'gpu_snapshot', probe)
    sleeps = []
    monkeypatch.setattr(queue.time, 'sleep', lambda seconds: sleeps.append(seconds))
    result = queue.wait_and_launch(tmp_path, [1, 2, 3, 4],
                                   [sys.executable, '-c', 'pass'], dict(queue.os.environ),
                                   lambda: {}, poll_seconds=1)
    assert result['exit_code'] == 0
    assert len(sleeps) == 3


def test_disk_pressure_prevents_launch_until_recovered(tmp_path, monkeypatch):
    queue = module()
    monkeypatch.setattr(queue, 'gpu_snapshot', snapshot)
    checks = iter([False, True, True, True])
    def disk():
        if not next(checks):
            raise OSError('Storage budget would be exceeded')
        return {'free_bytes': 100}
    sleeps = []
    monkeypatch.setattr(queue.time, 'sleep', lambda seconds: sleeps.append(seconds))
    result = queue.wait_and_launch(tmp_path, [1, 2, 3, 4],
                                   [sys.executable, '-c', 'pass'], dict(queue.os.environ),
                                   disk, poll_seconds=1)
    assert result['exit_code'] == 0
    assert len(sleeps) == 2
    history = [json.loads(line) for line in (tmp_path / 'history.jsonl').read_text().splitlines()]
    assert 'Storage budget' in history[0]['blocker']


def test_duplicate_queue_cannot_launch(tmp_path):
    queue = module()
    with (tmp_path / 'watch.lock').open('a+') as lock:
        queue.fcntl.flock(lock, queue.fcntl.LOCK_EX | queue.fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            queue.wait_and_launch(tmp_path, [1, 2, 3, 4], ['false'], {}, lambda: {})
