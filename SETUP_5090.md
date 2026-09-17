# Skill-RL：5090 下载依赖并重新部署

本包只带项目源码、配置、测试、冻结 Skill bank、报告和用于核对结论的统计证据。基础模型、ALFWorld 原始数据和 Python/CUDA 依赖在 5090 自行下载。没有搬运 RL checkpoint、optimizer、旧模型权重或已安装环境。

本说明由 91 的实际环境与官方下载入口整理。**尚未在 5090 执行安装或验证，不能视为已经通过的训练环境。** 当前阶段只部署与检查；新训练需另定配置与新输出目录。

## 1. 目录和证据

机器为 Ubuntu 22.04、x86_64，8 × RTX 5090、每卡约 32 GB。用户目录是 `/mnt/workspace/users/wangyifan`，不用 `/home/wangyifan`。

仓库地址为 `https://github.com/Euclider/SkillScope.git`，仓库根对应本目录。clone 到用户目录下的 `skill-RL`：

```bash
git clone https://github.com/Euclider/SkillScope.git /mnt/workspace/users/wangyifan/skill-RL
```

如果该目标目录已经存在，先检查内容，不覆盖或清理现有目录。

```bash
export SKILLRL_ROOT=/mnt/workspace/users/wangyifan
export SKILLRL_PROJECT="$SKILLRL_ROOT/skill-RL"
export SKILLRL_DEPLOY="$SKILLRL_PROJECT/deploy/5090"
cd "$SKILLRL_PROJECT"
python3 "$SKILLRL_DEPLOY/verify_project.py"
```

本次导出逐文件保存了现有源码，包括原工作区 19 项已跟踪修改和 17 个顶层未跟踪条目中的项目内容。原 `.git` 没有导入新快照；源 HEAD、原 Git 状态和 tracked patch 位于 `deploy/5090/`。没有对 91 执行提交、回滚或清理。

结论入口仍以已核验的 `2026-09-14-phase2-complete-analysis.md` synthesis-v2 为准。保留 Phase1/Phase2 协议、统计 CSV/Parquet、预测摘要、报告修订输入及完成记录。未带完整逐轨迹 JSON、token 张量和 checkpoint，因此可核对报告与统计输入，不能在此包上完整重放旧实验。

另有 `phase2-complete-analysis.md` 和 `reward-directed-family-v3` 修订记录。当前非日期报告 SHA-256 与 v3 记录值不同，原因未在本次迁移中确认；两者原样保留，验证脚本会明确报告 `false`，不会更改历史哈希。v1 的报告哈希指向早期报告，也不等于当前 synthesis-v2 文本；其输入文件仍可验证。

## 2. 安装 Python 环境，无需 sudo

先读 `deploy/5090/source-environment-packages.json` 可查 91 的完整安装清单。旧 `SkillRL/environment/phase1-constraints.txt` 和早期 `requirements-lock.txt` 属于较早环境，不用于本次重建；`environment.yml` 虽已有部分新版本，仍未固定 wheel、CUDA 变体和安装顺序。

关键基线：Python 3.10、PyTorch 2.10.0 + cu128、torchvision 0.25.0、Transformers 5.10.4、Ray 2.43.0、fla-core 0.5.2、causal-conv1d 1.7.0。先使用历史 Phase2 的 HF live-policy / SDPA 文本路线。vLLM 0.19.1 是 91 已安装项，最小 HF 路线暂不需要；本说明不安装旧 flash-attn，也不启动 vLLM 服务。

已知驱动为 595.58.03。`nvidia-smi` 的 CUDA 13.2 表示驱动支持能力，不要求将 PyTorch 改装成 CUDA 13.2；这里明确下载 cu128 wheel。安装后的实际 GPU 执行仍须检查。

若已有可用的 Conda，可复用其管理器，创建下面的新环境；若没有，用户目录安装 Miniforge：

```bash
export SKILLRL_ROOT=/mnt/workspace/users/wangyifan
test ! -e "$SKILLRL_ROOT/miniforge3" || { echo 'miniforge3 已存在，请先检查并复用'; exit 1; }
SKILLRL_INSTALLER_DIR=$(mktemp -d "$SKILLRL_ROOT/miniforge-installer.XXXXXX")
cd "$SKILLRL_INSTALLER_DIR"
curl -fL --retry 3 -o Miniforge3-Linux-x86_64.sh \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
curl -fL --retry 3 -o Miniforge3-Linux-x86_64.sh.sha256 \
  https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh.sha256
sha256sum -c Miniforge3-Linux-x86_64.sh.sha256
bash Miniforge3-Linux-x86_64.sh -b -p "$SKILLRL_ROOT/miniforge3"
```

安装脚本本身的版本由当次下载的官方 checksum 核验；不要把 `latest` 当作固定 Miniforge 版本。保存该目录即可追溯 bootstrap 文件。

```bash
source "$SKILLRL_ROOT/miniforge3/etc/profile.d/conda.sh"
test ! -e "$SKILLRL_ROOT/envs/skill-RL" || { echo '目标环境已存在，请先检查'; exit 1; }
conda create -y --override-channels -c conda-forge \
  --prefix "$SKILLRL_ROOT/envs/skill-RL" python=3.10 pip=25.2
conda activate "$SKILLRL_ROOT/envs/skill-RL"
export CONDA_DEFAULT_ENV=skill-RL
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export SKILLRL_PROJECT="$SKILLRL_ROOT/skill-RL"
export SKILLRL_DEPLOY="$SKILLRL_PROJECT/deploy/5090"
python -m pip install 'setuptools==84.0.0' wheel
python -m pip install 'torch==2.10.0+cu128' 'torchvision==0.25.0+cu128' \
  'torchaudio==2.10.0+cu128' --index-url https://download.pytorch.org/whl/cu128
python -m pip install -c "$SKILLRL_DEPLOY/constraints-core.txt" \
  -r "$SKILLRL_DEPLOY/requirements-core.txt"
python -m pip install --no-build-isolation \
  -c "$SKILLRL_DEPLOY/constraints-core.txt" -e "$SKILLRL_PROJECT/SkillRL"
```

Conda 按路径激活时，`CONDA_DEFAULT_ENV` 可能是完整路径；显式设为 `skill-RL` 是为了匹配已有 launcher 的检查。后续每次使用环境也设置下面第 3 节的路径变量。

`causal-conv1d` 使用 91 元数据记录的 Python 3.10 / torch 2.10 / CXX11 ABI wheel，不让 pip 临时挑选源码编译路线：

```bash
python - <<'PY'
import torch
assert torch.__version__.split('+')[0] == '2.10.0'
assert torch.version.cuda == '12.8'
assert torch._C._GLIBCXX_USE_CXX11_ABI is True
print(torch.__version__, torch.version.cuda, 'CXX11_ABI=True')
PY
python -m pip install --no-deps \
  'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp310-cp310-linux_x86_64.whl#sha256=f9a51d1fbc44735142905168914c5e74864fd68d81e9ddd65e85f38148905ac2'
python -m pip check
```

若 wheel 链接、版本解析或 import 失败，保留错误与版本输出，不自动升级 torch/transformers，不把 91 的 CUDA 扩展文件复制来覆盖。仅运行这些预编译 wheel 通常不需要系统 nvcc；后续若确需编译，再单独匹配 CUDA toolkit 和架构。

## 3. 下载原始基础模型与文本数据

```bash
export ALFWORLD_DATA="$SKILLRL_PROJECT/data/alfworld"
export PHASE1_MODEL_PATH="$SKILLRL_ROOT/model/Qwen3.5-4B"
export PHASE1_RAY_TEMP_DIR="/tmp/skillrl-5090-$(id -un)"
python "$SKILLRL_DEPLOY/download_assets.py" model --root "$SKILLRL_ROOT"
python "$SKILLRL_DEPLOY/download_assets.py" data --root "$SKILLRL_ROOT"
```

模型固定为 `Qwen/Qwen3.5-4B`，revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`，来自 91 的 Hugging Face 下载元数据。脚本下载后按 91 已记录的 SHA-256 校验模型与 tokenizer 等文件，不下载 Qwen2.5 早期 smoke 模型。

数据脚本使用 ALFWorld 0.4.2 的官方 `alfworld-download` 中三个文本数据 zip 入口，复制对应 logic 文件。跳过视觉 detector 和 BUTLER 预训练模型；完整安装器的 `--extra` 不适用于当前文本路线。下载后按 91 的文本数据文件清单逐项校验；如果上游数据或本地派生文件不一致，输出具体差异，不能将“不一致”改记为“完全复现”。已经存在的文本数据文件不会被脚本覆盖。

`SkillRL/artifacts/datasets/verl-agent/text/{train,test}.parquet` 是项目已有的两个很小的训练入口表，随项目保留；冻结 bank、clean 划分和配置也保留。它们与外部 ALFWorld 游戏文件是不同层次。

## 4. 只做部署检查

先看 GPU 是否空闲；有其他任务时只运行 import 检查。GPU 检查仅做小矩阵与 causal-conv1d 算子，**不加载模型、不生成 rollout、不执行训练更新**。

```bash
cd "$SKILLRL_PROJECT/SkillRL"
python "$SKILLRL_DEPLOY/check_runtime.py"
nvidia-smi
# 确认准备使用的 GPU 空闲后执行；也可用 CUDA_VISIBLE_DEVICES 指定空闲卡。
python "$SKILLRL_DEPLOY/check_runtime.py" --gpu
python -m pip check
```

记录本次新环境的实际版本，可写入用户目录下一个新日志目录：

```bash
SKILLRL_SETUP_LOG=$(mktemp -d "$SKILLRL_ROOT/skillrl-setup-log.XXXXXX")
python -m pip freeze > "$SKILLRL_SETUP_LOG/pip-freeze.txt"
python -m pip check > "$SKILLRL_SETUP_LOG/pip-check.txt" 2>&1
python "$SKILLRL_DEPLOY/check_runtime.py" > "$SKILLRL_SETUP_LOG/import-check.txt" 2>&1
```

即使这些检查通过，也只证明相关 import 与小算子可运行；FLA 实际模型内核、模型前向、8 卡 NCCL/FSDP 和 32 GB 每卡训练内存仍需在后续新配置中验证。这里不复跑历史 233 项测试，也不执行历史模型评估。

## 5. 新训练之前

旧启动脚本含 `/home/wangyifan`、旧 checkpoint 和旧输出目录等路径，有些会覆盖环境变量。不要直接运行旧 formal/phase2 launcher，也不要全局替换报告和冻结协议中的路径。

后续另建 5090 配置：指定本说明的模型/数据路径、新 run ID 和输出目录；根据 32 GB 每卡重新确定 FSDP、optimizer offload、batch、sequence 长度与 checkpoint 预算。重新从原始 Qwen3.5-4B 训练，不恢复旧 optimizer。协议、seed、支持范围和预算待用户确认后再启动。

## 6. GitHub 与本包的关系

本快照对应用户指定的 `Euclider/SkillScope` 仓库。发布工作在独立副本中进行，原 91 工作区保持不动。仓库根就是本目录，clone 到 `$SKILLRL_ROOT/skill-RL`。源码与历史证据保持原字节；发布时增加了仓库首页与部署入口。

原项目的 `SkillRL/.gitignore` 会忽略部分 artifacts；发布时必须依据本包文件清单显式包含已挑选的证据，不能只 `git add .` 后假定报告依赖已经上传。使用 LF 并保留文件原字节，否则报告哈希会变化。不要上传原迁移暂存目录整体，其中还有工具、废弃大包与 SSH 专用密钥。

Windows 中转仅用于传这个小项目包也是可选路径；环境、模型和数据均在 5090 下载，不需要保持之前两个隧道。

## 官方入口与版本依据

- PyTorch cu128 wheel 索引：<https://download.pytorch.org/whl/cu128/torch/>。
- Qwen 固定 revision：<https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a>。
- CUDA 扩展 release：<https://github.com/Dao-AILab/causal-conv1d/releases/tag/v1.7.0>；具体 wheel URL 与 hash 来自 91 的安装元数据。
- ALFWorld：<https://github.com/alfworld/alfworld>；数据 URL 来自 91 的 0.4.2 `alfworld-download` 脚本。
- Miniforge 用户目录安装：<https://github.com/conda-forge/miniforge>。
