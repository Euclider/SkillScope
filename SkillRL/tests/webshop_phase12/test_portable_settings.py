import json
import os
import subprocess
import sys
from pathlib import Path
import hashlib


def test_model_snapshot_binds_contents_and_local_file_identity(tmp_path):
    from webshop_phase12.snapshot import model_receipt
    model=tmp_path/'model';model.mkdir()
    (model/'config.json').write_text('{"model_type":"test"}')
    (model/'model.safetensors').write_bytes(b'test-weights')
    receipt=model_receipt(model)
    weights=next(row for row in receipt['files'] if row['path']=='model.safetensors')
    assert weights['sha256']==hashlib.sha256(b'test-weights').hexdigest()
    assert weights['size']==12
    assert weights['mtime_ns']==(model/'model.safetensors').stat().st_mtime_ns


def test_other_server_paths_are_selected_from_environment(tmp_path):
    source=Path(__file__).resolve().parents[2]
    script='from webshop_phase12.assets import RUN_ROOT,BASE_MODEL; import json; print(json.dumps([str(RUN_ROOT),str(BASE_MODEL)]))'
    env={**os.environ,'PYTHONPATH':str(source),'PYTHONDONTWRITEBYTECODE':'1',
        'WEBSHOP_RUN_ROOT':str(tmp_path/'output'),'WEBSHOP_BASE_MODEL':str(tmp_path/'model')}
    result=subprocess.run([sys.executable,'-B','-c',script],env=env,capture_output=True,text=True,check=True)
    assert json.loads(result.stdout)==[str(tmp_path/'output'),str(tmp_path/'model')]
