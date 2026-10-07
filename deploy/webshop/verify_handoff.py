#!/usr/bin/env python3
"""Verify the WebShop handoff overlay without models, JVM or GPU."""
import hashlib
import json
from pathlib import Path


def verify(root):
    root=Path(root).resolve()
    manifest=json.loads((root/'deploy/webshop/handoff-manifest.json').read_text())
    for row in manifest['files']:
        relative=Path(row['path'])
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe handoff path')
        path=root/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('WebShop handoff content mismatch: '+str(relative))
    return {'status':'webshop_handoff_verified','files':len(manifest['files']),
        'base_commit':manifest['base_commit'],'model_data_gpu_execution_checked':False}


if __name__=='__main__':print(json.dumps(verify(Path(__file__).resolve().parents[2]),indent=2))
