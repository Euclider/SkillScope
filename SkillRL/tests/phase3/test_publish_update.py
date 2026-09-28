"""No network: verified fast-forward publication and concurrent-change gates."""
import hashlib
from pathlib import Path

import pytest

from phase3.common import ProtocolError, digest, write_new
from skillnet_cohort.common import file_hash, write_new_bytes


def test_update_is_verified_before_nonforce_publication(tmp_path, monkeypatch):
    from phase3 import publish as module
    directory = tmp_path / 'export'
    write_new_bytes(directory / 'README.md', b'new portable instructions\n')
    rows = [{'path': 'README.md', 'sha256':file_hash(directory/'README.md'), 'bytes':len((directory/'README.md').read_bytes())}]
    write_new(directory/'RELEASE_MANIFEST.json', {'files':rows,'inventory_sha256':digest(rows)})
    parent, commit, calls = 'a'*40, 'b'*40, []
    def api(endpoint, payload=None, method=None, optional=False):
        calls.append((endpoint, payload, method))
        if endpoint == f'repos/{module.REPOSITORY}':
            return {'full_name':module.REPOSITORY, 'permissions':{'push':True}, 'size':1}
        if endpoint.endswith('/git/ref/heads/main'):
            return {'object': {'sha':commit if any(c[2]=='PATCH' for c in calls) else parent}}
        if endpoint.endswith('/git/commits/'+parent):
            return {'tree':{'sha':'old-tree'}}
        if endpoint.endswith('/git/trees/old-tree?recursive=1'):
            return {'truncated':False, 'tree':[{'path':'README.md','type':'blob','sha':'old'}]}
        if endpoint.endswith('/git/trees'):
            return {'sha':'new-tree'}
        if endpoint.endswith('/git/trees/new-tree?recursive=1'):
            return {'truncated':False,'tree':[{'path':p.name,'type':'blob','sha':hashlib.sha1(b'blob '+str(p.stat().st_size).encode()+b'\0'+p.read_bytes()).hexdigest()} for p in directory.iterdir()]}
        if endpoint.endswith('/git/commits'):
            assert payload['parents'] == [parent]
            return {'sha':commit}
        if endpoint.endswith('/git/refs/heads/main'):
            assert payload == {'sha':commit,'force':False}
            return {}
        raise AssertionError(endpoint)
    monkeypatch.setattr(module, 'api', api)
    with pytest.raises(ProtocolError, match='exact reviewed'):
        module.publish(directory, tmp_path/'wrong.json', 'c'*40)
    assert all(c[2] is None for c in calls)
    calls.clear()
    receipt = module.publish(directory, tmp_path/'receipt.json', parent)
    assert receipt['all_remote_blob_hashes_match'] and not receipt['force_push']
    tree_check = next(i for i,c in enumerate(calls) if c[0].endswith('new-tree?recursive=1'))
    update = next(i for i,c in enumerate(calls) if c[2]=='PATCH')
    assert tree_check < update
