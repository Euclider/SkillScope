#!/usr/bin/env bash
set -euo pipefail

if [[ ${CONDA_DEFAULT_ENV:-} != "skill-RL" ]]; then
  echo "Activate the environment first: conda activate skill-RL" >&2
  exit 2
fi

MODEL_ROOT=/home/wangyifan/model
MODEL_IDS=(
  Qwen/Qwen2.5-0.5B-Instruct
  Qwen/Qwen2.5-1.5B-Instruct
)

if [[ ! -d "$MODEL_ROOT" || ! -w "$MODEL_ROOT" ]]; then
  echo "$MODEL_ROOT must exist and be writable by $(id -un)" >&2
  exit 2
fi

for MODEL_ID in "${MODEL_IDS[@]}"; do
  MODEL_DIR="$MODEL_ROOT/${MODEL_ID##*/}"
  huggingface-cli download "$MODEL_ID" \
    --local-dir "$MODEL_DIR"

  python - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import json
import sys
from pathlib import Path

model_id = sys.argv[1]
model_dir = Path(sys.argv[2])
required = ["config.json", "tokenizer_config.json"]
missing = [name for name in required if not (model_dir / name).exists()]
weight_files = list(model_dir.glob("*.safetensors"))
if missing or not weight_files:
    raise SystemExit(f"Incomplete model: missing={missing}, safetensors={len(weight_files)}")
print(json.dumps({
    "model_id": model_id,
    "model_dir": str(model_dir),
    "weight_files": [path.name for path in sorted(weight_files)],
    "size_bytes": sum(path.stat().st_size for path in model_dir.rglob("*") if path.is_file()),
}, indent=2))
PY
done
