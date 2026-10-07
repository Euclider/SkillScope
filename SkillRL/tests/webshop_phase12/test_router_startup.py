import json
import subprocess
import sys


def test_shared_router_finishes_initialization_and_sleeps_before_constructor_returns(monkeypatch, tmp_path):
    from webshop_phase12.router import RouterClient
    events = tmp_path/'commands.jsonl'
    program = '''import json,sys
for line in sys.stdin:
    request=json.loads(line)
    if request.get('close'):break
    command=request['command']
    with open(sys.argv[1],'a') as out:out.write(json.dumps(command)+'\\n')
    print(json.dumps({'ack':command}),flush=True)
'''
    real_popen = subprocess.Popen
    def start(*args, **kwargs):
        return real_popen([sys.executable, '-B', '-u', '-c', program, str(events)], **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', start)
    monkeypatch.setenv('WEBSHOP_ROUTER_GPU', '7')
    monkeypatch.setenv('WEBSHOP_ROUTER_CACHE', str(tmp_path/'router.sqlite3'))
    monkeypatch.setenv('WEBSHOP_ROUTER_SHARED', '1')
    client = RouterClient('startup-test')
    try:
        commands = [json.loads(s) for s in events.read_text().splitlines()] if events.exists() else []
        assert commands == ['ready', 'sleep']
        assert client.sleeping
    finally:
        client.close()
