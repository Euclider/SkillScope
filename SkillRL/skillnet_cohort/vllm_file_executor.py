"""Single-rank vLLM rendezvous without the probe-close-bind TCP port race.

Only the rendezvous transport changes. All worker initialization, model loading,
NCCL/Gloo groups, scheduling and sampling use the installed UniProcExecutor.
Each engine gets a fresh local directory and a never-before-used FileStore path.
No previous rendezvous file is reused or deleted by this adapter.
"""
import os
from pathlib import Path
import tempfile

from vllm.v1.executor.uniproc_executor import UniProcExecutor

from .common import write_new_json

RECORD_ENV = 'SKILLSCOPE_EVAL_RENDEZVOUS_RECORD'


def file_rendezvous_args(config, record):
    parallel = config.parallel_config
    for field in ('world_size', 'world_size_across_dp', 'tensor_parallel_size',
                  'pipeline_parallel_size', 'data_parallel_size', 'nnodes'):
        if getattr(parallel, field) != 1:
            raise ValueError('File rendezvous is restricted to one local evaluation rank')
    device = str(config.device_config.device).split(':')
    if device[0] != 'cuda':
        raise ValueError('This executor is scoped to the registered CUDA evaluator')
    local_rank = int(device[1]) if len(device) == 2 else 0
    if local_rank != 0:
        raise ValueError('Each shard must expose exactly one GPU as local rank zero')
    record = Path(record)
    if (not record.is_absolute() or record.is_symlink() or record.exists()
            or not record.parent.is_dir() or record.parent.is_symlink()):
        raise FileExistsError('Fresh explicit rendezvous audit path required; no retry')
    # /tmp is outside the evidence tree: FileStore may remove its own live file
    # during normal teardown, and that must not race the evidence disk scanner.
    directory = Path(tempfile.mkdtemp(prefix='skillscope-eval-rdzv-', dir='/tmp'))
    path = directory/'store'
    if path.exists():
        raise FileExistsError('Rendezvous path must never have existed')
    method = path.as_uri()
    write_new_json(record, {'schema_version': 'skillnet.file_rendezvous.v1',
        'engine_pid': os.getpid(), 'parent_pid': os.getppid(), 'init_method': method,
        'rank': 0, 'local_rank': local_rank, 'world_size': 1,
        'transport': 'FileStore', 'tcp_rendezvous_port': None,
        'directory': str(directory), 'path_initially_absent': True,
        'automatic_retry': False, 'old_files_deleted': False})
    return method, 0, local_rank


class FileStoreUniProcExecutor(UniProcExecutor):
    def _distributed_args(self):
        return file_rendezvous_args(self.vllm_config, os.environ[RECORD_ENV])
