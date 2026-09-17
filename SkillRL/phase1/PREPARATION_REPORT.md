# Phase-I 实验前准备报告

更新时间：2026-08-25T16:50:29+08:00

## 当前结论

代码、Conda 环境、两个模型、ALFWorld 数据、冻结 Skill Bank、固定状态
action-flip 探针、matched rollout evaluator、负对照、统计和分层归档均已准备
并通过本地测试。最终 preflight 通过；尚未开始模型推理、rollout 或 RL 实验。

根分区剩余约 362 GiB，低于实验规格建议的 2 TiB，preflight 将其记录为
warning，但它不阻止 smoke validation。

## 固定资产

- 上游仓库：`aiming-lab/SkillRL`
- 固定 commit：`8e66726ed866a4e0a7f053586a41022798192e6c`
- 工作分支：`phase1-prep`
- Conda 环境：`/home/wangyifan/miniconda3/envs/skill-RL`
- ALFWorld 数据：`/home/wangyifan/skill-RL/data/alfworld`（2.3 GiB）
- Smoke 模型：`/home/wangyifan/model/Qwen2.5-0.5B-Instruct`（约 1.00 GB）
- Fallback 模型：`/home/wangyifan/model/Qwen2.5-1.5B-Instruct`（约 3.10 GB）
- 机器：8 × NVIDIA A800 80GB PCIe

内容哈希：

- `environment/requirements-lock.txt`: `d29c3cb28bbcd75e9079ff749fd7caacb68a3ff4e0625b2893398ddae0039f4d`
- `phase1/config/phase1_protocol.json`: `820276951627999323dfb65e852ae4c4025d36b72f18ee662cda57dd99cc342f`
- `phase1/config/model_manifest.json`: `e7eed743ba5fc28986584a331fbe3c2df568b5f311065b6d2f207ead1d1376c0`
- `phase1/config/frozen_alfworld_skills.json`: `b9e4b7a3302bf14a7356a006b0cc9c4485331b1cec829ef68eb9b44e78fdc8c1`
- `phase1/config/game_ids/manifest.json`: `4de858f9e41b1091eb2502a7a682477ffe649590a08377cfc01bb895147c3a47`

## 实验协议与实现

- 固定 Skill Bank 共 7 个唯一 Skill：1 个 general、6 个 context-specific；
  template retrieval，dynamic update 关闭。
- `pick_two` 已从上游 `pick_and_place` 路由中拆出，六个 context 独立。
- 固定状态探针对同一 `probe_id + skill_id + condition` 强制校验 prompt hash
  和 admissible action set 跨 checkpoint 不变；记录 greedy action flip、action
  admissibility 和 constrained-action JS divergence。
- episode 主指标是同 checkpoint/game/seed 下
  `success(FULL_BANK) - success(MINUS_SKILL)`；`NO_SKILL` 为补充基线。
- development、valid_seen、valid_unseen 各 context 8 个独立 game，共 144 个
  固定 game ID；三组无交集，所有文件均存在。
- checkpoint、RL/eval seed、CI-based strict flip、能力门槛及 Phase-I go/no-go
  阈值已写入不可变协议。
- 负对照包括 zero-update、group 内 shuffled reward、matched-norm random
  parameter perturbation。

## 归档保证

- 每个 run 先创建不可变 manifest，包含 repo commit、Skill Bank hash、protocol
  hash、模型 config hash、Conda/pip/GPU 清单、seed 与 update type。
- 每条 episode 单独保存完整 JSON：prompt、observation、admissible actions、raw
  completion、projected action、reward、下一 observation、每步 retrieved/injected/
  disabled Skill IDs、token 数及完整 action sequence。
- trajectory 顶层记录去重后的 Skill ID 集合与 `unique_skill_count`。
- 每个 RL step 另存 reward/return/advantage 的逐轨迹摘要、actor grad norm 和完整
  trainer metrics，并用 run/step/trajectory ID 对齐。
- 相邻已保存 checkpoint 的精确参数 L2 delta 由独立脚本计算并归档；随机参数
  对照复用同一 delta norm。
- JSONL 追加带文件锁和唯一键，重复 evaluator/归档不会产生重复记录。

## 已完成验证

- `ruff check phase1 tests/phase1`：通过。
- `pytest -q tests/phase1`：16 passed。
- 新增及改动 Python 文件：`compileall` 通过。
- launcher/download 脚本：`bash -n` 通过。
- Hydra：`phase1_control` 与 `env.phase1_archive` 配置组合通过，未启动训练。
- PyTorch CUDA：8 张 GPU 可见；BF16 FlashAttention CUDA kernel 实算通过。
- ALFWorld：单 game environment initialize/reset/close 通过，reset 返回 14 个
  admissible actions；没有调用模型或生成 action。
- 两个模型的 config、tokenizer、safetensors header 与完整参数均可在 offline
  模式读取；参数量分别为 494,032,768 和 1,543,714,304，权重文件分别包含
  290 和 338 个 tensor。
- 0.5B 权重 SHA-256：`fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`。
- 1.5B 权重 SHA-256：`dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee`。
- 最终 Preflight：全部检查通过；存储容量为 warning。
