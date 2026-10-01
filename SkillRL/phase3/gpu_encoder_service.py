"""Local GPU embedding sidecar for Ray CPU-only Phase3 coordinators.

The coordinator keeps Ray's CUDA resource mask untouched. Only this child
process sees its explicitly selected physical GPU. Pipes carry NumPy vectors;
the service never receives the editor credential or writes policy outputs.
"""
from __future__ import annotations

import argparse
import atexit
import os
import pickle
import struct
import subprocess
import sys
import tempfile
import threading
from pathlib import Path


MAX_MESSAGE_BYTES = 64 * 1024 * 1024
HEADER = struct.Struct("!Q")
_SHARED_ENCODERS = {}
_SHARED_ENCODERS_LOCK = threading.RLock()


def _read_exact(stream, size):
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise EOFError("GPU encoder pipe closed")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def send(stream, value):
    payload = pickle.dumps(value, protocol=5)
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("GPU encoder message exceeds fixed cap")
    stream.write(HEADER.pack(len(payload)))
    stream.write(payload)
    stream.flush()


def receive(stream):
    header = stream.read(HEADER.size)
    if not header:
        raise EOFError("GPU encoder pipe closed")
    if len(header) != HEADER.size:
        header += _read_exact(stream, HEADER.size - len(header))
    length = HEADER.unpack(header)[0]
    if not 0 < length <= MAX_MESSAGE_BYTES:
        raise ValueError("Invalid GPU encoder message size")
    return pickle.loads(_read_exact(stream, length))


def physical_gpu(shared_gpu_physical_id):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    ids = visible.split(",") if visible else []
    if len(ids) == 1 and ids[0].isdigit():
        return int(ids[0])  # Evaluator shard: exactly its assigned physical GPU.
    return shared_gpu_physical_id  # Ray CPU-only TaskRunner: explicitly shared GPU.


def encoder_environment(device, shared_gpu_physical_id=None):
    permitted = ('PATH', 'LD_LIBRARY_PATH', 'HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR',
                 'HF_HUB_OFFLINE', 'TOKENIZERS_PARALLELISM', 'OMP_NUM_THREADS',
                 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')
    environment = {key: os.environ[key] for key in permitted if key in os.environ}
    environment.update(CUDA_VISIBLE_DEVICES='' if device == 'cpu' else str(physical_gpu(shared_gpu_physical_id)),
                       PYTHONPATH=str(Path(__file__).resolve().parents[1]), PYTHONDONTWRITEBYTECODE='1')
    return environment


class GPUEncoderProxy:
    def __init__(self, *, model_path, profile_sha256, intra_op_threads,
                 shared_gpu_physical_id, ledger_path, forward_microbatch_size=None, python_executable=None,
                 device='cuda:0', runtime_variant=None):
        self.device, self.runtime_variant = device, runtime_variant
        self.python_executable = python_executable or sys.executable
        self.model_path = str(model_path)
        self.profile_sha256 = profile_sha256
        self.intra_op_threads = intra_op_threads
        self.forward_microbatch_size = forward_microbatch_size
        self.shared_gpu_physical_id = shared_gpu_physical_id
        self.ledger_path = Path(ledger_path)
        self.process = None
        self.lock = threading.RLock()
        self.log_path = None
        atexit.register(self.close)

    def _start(self):
        if self.process is not None:
            return
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, path = tempfile.mkstemp(prefix="router-gpu-sidecar-", suffix=".log",
                                          dir=self.ledger_path.parent)
        self.log_path = path
        environment = encoder_environment(self.device, self.shared_gpu_physical_id)
        args = [self.python_executable, "-B", "-m", "phase3.gpu_encoder_service", "--serve",
                "--model-path", self.model_path, "--profile-sha256", self.profile_sha256,
                "--intra-op-threads", str(self.intra_op_threads)]
        if self.forward_microbatch_size is not None:
            args += ["--forward-microbatch-size", str(self.forward_microbatch_size)]
        if self.device == 'cpu':
            args += ['--device', 'cpu']
        if self.runtime_variant:
            args += ['--runtime-variant', self.runtime_variant]
        with os.fdopen(descriptor, "wb") as log:
            self.process = subprocess.Popen(args, cwd=Path(__file__).resolve().parents[1],
                                            env=environment, stdin=subprocess.PIPE,
                                            stdout=subprocess.PIPE, stderr=log, bufsize=0)

    def encode(self, texts):
        with self.lock:
            self._start()
            if self.process.poll() is not None:
                raise RuntimeError(f"GPU encoder sidecar exited; inspect {self.log_path}")
            try:
                send(self.process.stdin, {"operation": "encode", "texts": list(texts)})
                response = receive(self.process.stdout)
            except (BrokenPipeError, EOFError) as error:
                raise RuntimeError(f"GPU encoder sidecar disconnected; inspect {self.log_path}") from error
            if response.get("status") != "ok":
                raise RuntimeError(f"GPU encoder sidecar failed ({response.get('error_type')}); inspect {self.log_path}")
            return response["vectors"], response["accounting"]

    def close(self):
        with self.lock:
            process, self.process = self.process, None
            if process is None:
                return
            if process.poll() is None:
                try:
                    send(process.stdin, {"operation": "close"})
                    process.wait(timeout=5)
                except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
            process.stdin.close()
            process.stdout.close()


def shared_gpu_encoder(*, model_path, profile_sha256, intra_op_threads,
                       shared_gpu_physical_id, ledger_path, forward_microbatch_size=None, python_executable=None,
                       device='cuda:0', runtime_variant=None):
    """Reuse one frozen encoder sidecar across train and native validation.

    Both environment managers live in the same Ray coordinator process and
    use the same branch ledger. Distinct branch/model/device settings never
    share a process. The proxy's encode lock serializes concurrent callers.
    """
    key = (os.getpid(), str(Path(model_path).resolve()), profile_sha256,
           intra_op_threads, shared_gpu_physical_id, str(Path(ledger_path).resolve()),
           forward_microbatch_size, python_executable or sys.executable, device, runtime_variant)
    with _SHARED_ENCODERS_LOCK:
        proxy = _SHARED_ENCODERS.get(key)
        if proxy is None:
            proxy = GPUEncoderProxy(model_path=model_path, profile_sha256=profile_sha256,
                intra_op_threads=intra_op_threads, shared_gpu_physical_id=shared_gpu_physical_id,
                ledger_path=ledger_path, forward_microbatch_size=forward_microbatch_size,
                python_executable=python_executable, device=device, runtime_variant=runtime_variant)
            _SHARED_ENCODERS[key] = proxy
        return proxy


def serve(model_path, profile_sha256, intra_op_threads, forward_microbatch_size=None, device='cuda:0', runtime_variant=None):
    from agent_system.memory.skillnet_runtime import DEFAULT_EMBEDDING_ROUTER_PROFILE
    from agent_system.memory.skillrl_embedding_batch_router import BatchedSentenceEncoder
    from agent_system.memory.skillrl_embedding_router import load_profile
    from skillnet_cohort.common import file_hash
    if file_hash(DEFAULT_EMBEDDING_ROUTER_PROFILE) != profile_sha256:
        raise ValueError("GPU sidecar embedding profile changed")
    config, files = load_profile(DEFAULT_EMBEDDING_ROUTER_PROFILE)
    from .embedding_routing import runtime_profile
    config = runtime_profile(config, device, runtime_variant)
    encoder = BatchedSentenceEncoder(config, model_path, files, device,
                                     intra_op_threads=intra_op_threads,
                                     forward_microbatch_size=forward_microbatch_size)
    import torch
    print(f"Encoder sidecar device={device} visible={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          file=sys.stderr, flush=True)
    try:
        while True:
            try:
                request = receive(sys.stdin.buffer)
            except EOFError:
                break
            if request == {"operation": "close"}:
                break
            if (not isinstance(request, dict) or set(request) != {"operation", "texts"}
                    or request["operation"] != "encode" or not isinstance(request["texts"], list)
                    or not request["texts"] or not all(isinstance(text, str) for text in request["texts"])):
                send(sys.stdout.buffer, {"status": "error", "error_type": "InvalidRequest"})
                break
            try:
                vectors, accounting = encoder.encode(request["texts"])
                if device != 'cpu':
                    torch.cuda.empty_cache()
                send(sys.stdout.buffer, {"status": "ok", "vectors": vectors, "accounting": accounting})
            except Exception as error:
                print(f"GPU encoder forward failed: {type(error).__name__}: {error}",
                      file=sys.stderr, flush=True)
                send(sys.stdout.buffer, {"status": "error", "error_type": type(error).__name__})
                break
    finally:
        encoder.close()
        if device != 'cpu':
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--profile-sha256", required=True)
    parser.add_argument("--intra-op-threads", type=int, required=True)
    parser.add_argument("--forward-microbatch-size", type=int)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--runtime-variant', choices=('cpu_torch_2_11_0_v1',))
    args = parser.parse_args()
    serve(args.model_path, args.profile_sha256, args.intra_op_threads,
          args.forward_microbatch_size, args.device, args.runtime_variant)


if __name__ == "__main__":
    main()
