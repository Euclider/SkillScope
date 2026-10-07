import importlib.util
import json
import hashlib
from pathlib import Path
import pytest


def test_published_manifest_detects_changed_content(tmp_path):
    repo=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('webshop_handoff_verifier',repo/'deploy/webshop/verify_handoff.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    folder=tmp_path/'deploy/webshop';folder.mkdir(parents=True)
    (tmp_path/'example.py').write_bytes(b'original')
    (folder/'handoff-manifest.json').write_text(json.dumps({'base_commit':'test','files':[
        {'path':'example.py','sha256':hashlib.sha256(b'original').hexdigest()}]}))
    assert module.verify(tmp_path)['files']==1
    (tmp_path/'example.py').write_bytes(b'changed')
    with pytest.raises(ValueError,match='content mismatch'):module.verify(tmp_path)
