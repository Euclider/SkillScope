"""Lossless compression of newly produced WebShop training evidence."""
import gzip
import hashlib
import json
import shutil
from pathlib import Path

import torch
from phase1.archive import atomic_write_json,sha256_file


def save_gzip_training_archive(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.partial')
    digest=hashlib.sha256();size=0
    class HashedWriter:
        def __init__(self,stream):self.stream=stream
        def write(self,data):
            nonlocal size
            digest.update(data);size+=len(data)
            return self.stream.write(data)
        def flush(self):return self.stream.flush()
    with gzip.open(temporary,'wb',compresslevel=1) as stream:
        torch.save(value,HashedWriter(stream))
    check=hashlib.sha256()
    with gzip.open(temporary,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):check.update(block)
    if check.hexdigest()!=digest.hexdigest():raise ValueError('Compressed training bytes do not match native serialization')
    temporary.replace(path)
    return {'batch_file':path.name,'batch_compression':'gzip-lossless-level1',
            'uncompressed_batch_sha256':digest.hexdigest(),'uncompressed_bytes':size,
            'compressed_bytes':path.stat().st_size}


def compress_training_archive(directory):
    directory=Path(directory);original=directory/'training_batch.pt'
    target=directory/'training_batch.pt.gz';temporary=directory/'training_batch.pt.gz.partial'
    if target.exists():raise ValueError('Compressed training evidence already exists')
    manifest=json.loads((directory/'manifest.json').read_text())
    if sha256_file(original)!=manifest['batch_sha256']:
        raise ValueError('Uncompressed training archive hash differs from native manifest')
    with original.open('rb') as source,gzip.open(temporary,'wb',compresslevel=1) as output:
        shutil.copyfileobj(source,output,4*1024**2)
    check=hashlib.sha256()
    with gzip.open(temporary,'rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):check.update(block)
    if check.hexdigest()!=manifest['batch_sha256']:
        raise ValueError('Compressed training evidence is not byte-lossless')
    temporary.replace(target)
    manifest.update(batch_file=target.name,batch_compression='gzip-lossless-level1',
                    uncompressed_batch_sha256=manifest['batch_sha256'],batch_sha256=sha256_file(target),
                    uncompressed_bytes=original.stat().st_size,compressed_bytes=target.stat().st_size)
    atomic_write_json(directory/'manifest.json',manifest)
    # This is only the duplicate just produced by the current capture call.
    # Its complete bytes have been verified in the committed gzip artifact.
    original.unlink()


def load_training_archive(directory):
    directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
    name=manifest.get('batch_file','training_batch.pt')
    if name not in ('training_batch.pt','training_batch.pt.gz'):
        raise ValueError('Unknown training evidence storage format')
    path=directory/name
    if manifest.get('batch_sha256') and sha256_file(path)!=manifest['batch_sha256']:
        raise ValueError('Training evidence hash differs from registered manifest')
    if name.endswith('.gz'):
        with gzip.open(path,'rb') as stream:
            return torch.load(stream,map_location='cpu',weights_only=False)
    return torch.load(path,map_location='cpu',weights_only=False)
