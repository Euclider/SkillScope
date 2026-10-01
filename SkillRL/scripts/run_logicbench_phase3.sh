#!/usr/bin/env bash
set -euo pipefail
TASK_PYTHON=/home/wangyifan/miniconda3/envs/skill-RL/bin/python
TASK_LIB=/home/wangyifan/miniconda3/envs/skill-RL/lib
export LD_LIBRARY_PATH="$TASK_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONDONTWRITEBYTECODE=1
cd /home/wangyifan/skill-RL/SkillRL
exec "$TASK_PYTHON" -B -m phase3.logicbench_loop "$@"
