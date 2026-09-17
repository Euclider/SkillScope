# Qwen3.5 clean RL 中断与恢复记录（2026-09-01）

## 结论

2026-09-01 08:36 CST 检查时，seed 101 与 seed 202 的训练进程均已停止。两条训练链按“只从完整预注册 checkpoint 恢复”的规则处理，没有根据 success、Skill utility 或 action flip 决定恢复点。

- seed 101 尚未产生 C1，因此从公共 B0 重新开始完整的 30 updates。
- seed 202 已产生完整 C1（update 10），因此从 C1 恢复并继续到 update 30。
- 固定评估 checkpoint 仍为 update 10/20/30；中断不改变 checkpoint 选择规则。

## 中断时状态

### Seed 101 原尝试

- run ID：`qwen35-clean-formal-seed101-u30-20260831`
- 已完整归档的 training-step provenance：update 1–5。
- 已归档 trajectory：192 条，即 6 批 × 32；其中最后一批 rollout 没有对应的已完成参数更新。
- 完整 FSDP checkpoint：无。
- 处理：该尝试整体标记为 `abandoned_before_C1`，不进入正式 seed 101 主分析，也不与重启轨迹拼接。

正式替代 run：

```text
qwen35-clean-formal-seed101-u30-r2-20260901
```

该 run 从同一个公共 B0、同一 RL seed 101 和同一冻结 Skill Bank/Router 开始，完整运行 update 1–30。

## Seed 202 原尝试

- run ID：`qwen35-clean-formal-seed202-u30-20260831`
- 已完整归档的 training-step provenance：update 1–13。
- 已归档 trajectory：475 条，包括 update 1–13 的训练 rollout、update 10 的 27 条 validation，以及中断前已经生成但尚未完成 update 14 的 32 条 rollout。
- 完整 checkpoint：`global_step_10`。
- checkpoint 完整性：4 个 model shards、4 个 optimizer shards、4 个 extra-state shards 均存在且非空；tracker 值为 10。
- update 11–13 及未完成的 update 14 rollout：保留作故障记录，但从正式主分析排除，因为没有预注册 checkpoint 可精确恢复这些 optimizer 状态。

C1 已额外合并并验证为独立 HF model-only 权重：

```text
/home/wangyifan/skill-RL/SkillRL/artifacts/model_only/qwen35-clean-formal-seed202-u30-c1-update10
```

其中 `model.safetensors` 约 9.1 GB，共 427 tensors；配置可由 `AutoConfig` 识别为 `Qwen3_5TextConfig`。

恢复 run：

```text
qwen35-clean-formal-seed202-u30-resume10-r2-20260901
```

启动日志显示 `Training Progress 10/30`，确认恢复了 global step 10。后续 C2/C3 将分别对应累计 update 20/30，而不是恢复后的局部 update 20/30。

## 第一次恢复时计划的 lineage（后被第二次 OOM 修订）

| RL seed | B0 → C1 | C1 → C2/C3 | 排除部分 |
|---:|---|---|---|
| 101 | seed101 r2 update 1–10 | seed101 r2 update 11–30 | 原尝试全部 |
| 202 | 原尝试 update 1–10 | resume10 r2 update 11–30 | 原尝试 update 11–13 与未完成 update 14 rollout |

所有旧日志、trajectory 和 training-step provenance 均保留，避免静默删除失败路径。后续构建 anchor、训练统计和 checkpoint lineage 时必须按本表过滤 run ID/global step。

## 恢复后的归档入口

- seed 101 r2 日志：`SkillRL/artifacts/logs/qwen35-clean-formal-seed101-u30-r2-20260901.log`
- seed 101 r2 manifest：`SkillRL/artifacts/manifests/qwen35-clean-formal-seed101-u30-r2-20260901.json`
- seed 202 r2 日志：`SkillRL/artifacts/logs/qwen35-clean-formal-seed202-u30-resume10-r2-20260901.log`
- seed 202 r2 manifest：`SkillRL/artifacts/manifests/qwen35-clean-formal-seed202-u30-resume10-r2-20260901.json`

## 第二次运行中断与资源优先级

2026-09-01 15:15 CST 检查发现，恢复后的两条进程均因 CUDA OOM 退出。日志和 GPU process audit 表明，训练启动后其他用户任务占用了物理 GPU 1–3（每卡约 28–39 GB），与本实验 FSDP rank 发生显存竞争；未终止或修改任何其他用户进程。

- seed 101 r2：完整完成 update 1–3，update 4 rollout 已归档但 backward OOM；仍未到 C1，因此标记为 `abandoned_before_C1_due_external_gpu_contention`。
- seed 202 resume10 r2：从 C1 完整执行 update 11–15，update 16 backward OOM；因为没有新的完整分析 milestone，正式 lineage 仍停留在 C1=10。
- 资源决策：按用户指示优先完成 seed 101，seed 202 暂停在可恢复 C1。
- seed 101 新 run：`qwen35-clean-formal-seed101-u30-r3-20260901`，使用当时完全空闲的 GPU 4–7，从公共 B0 重跑。
- 操作性恢复 checkpoint 改为每 5 updates 保存、最多保留最近 2 个；正式分析 checkpoint 仍严格限定为累计 update 10/20/30。

## 恢复时资源状态

- seed 101 r2：GPU 0–3。
- seed 202 resume10 r2：GPU 4–7。
- 恢复后两组进程均存活，GPU 0–7 均有训练负载。
- 恢复时磁盘可用空间约 384 GB。

## Seed 101 逐 update 恢复保存修订

2026-09-01 15:47 CST，根据共享 GPU 环境已发生两次外部显存竞争 OOM 的事实，预注册协议增加纯操作性恢复修订。该修订不读取 validation performance、Skill utility 或 action flip，也不改变 optimizer、训练数据、Skill Bank、Router、RL seed 及正式分析 checkpoint：

- 当前 seed 101 r3 先继续以 `save_freq=5` 跑到第一个完整恢复点 `global_step_5`；
- tracker 提交 update 5 后立即冻结当前进程组，验证 4 个 model shards、4 个 optimizer shards、4 个 extra-state shards、`data.pt` 及总大小；
- 仅在验证通过后停止旧进程，并从同一 `global_step_5` 恢复到新的 continuation run `qwen35-clean-formal-seed101-u30-resume5-s1-20260901`；新 run ID 用于保存变更后的不可变配置 manifest，科学 lineage 仍与 r3 连续；
- 恢复后采用 `save_freq=1`，始终保留最近 2 个完整恢复 checkpoint；
- continuation 的 update 6 完整提交并验证后，才删除作为恢复源的旧段 update 5；因此切换过程始终有可恢复副本，同时完整 checkpoint 峰值保持在约两个槽位；
- update 10/20/30 检测到完整提交后，先在临时目录转换为 model-only HF 权重，验证模型大小、tensor 数和 `model_type`，再通过原子 rename 永久归档；
- 非 milestone 的约 50GB 完整 checkpoint 滚动清理，trajectory、metrics 和 provenance 继续全部保留。

当前自动切换监督器已经启动，等待 update 5：

```text
SkillRL/phase1/switch_seed101_to_per_update.sh
SkillRL/artifacts/logs/qwen35-clean-formal-seed101-u30-r3-20260901-per-update-switch.log
```

正式 checkpoint 归档监视器：

```text
SkillRL/phase1/watch_qwen35_checkpoints.py
```

截至本修订，最新有效 lineage 为：

| RL seed | 当前正式训练链 | 当前可恢复点 | 状态 |
|---:|---|---:|---|
| 101 | r3 update 1–5 → `qwen35-clean-formal-seed101-u30-resume5-s1-20260901` update 6–30 | 尚未产生；目标 update 5 后切换 | GPU 4–7 运行中，优先完成 |
| 202 | 原始 run 的 update 1–10 | update 10 | 暂停；等待 seed 101 完成或独占资源可用 |

## Seed 101 → Seed 202 自动交接（2026-09-02）

2026-09-02 10:03 CST，Seed 101 continuation 已完整提交至 update 29，C1/C2 model-only 均已验证。自动交接监督器已启动，采用以下不读取 performance/Skill utility/action flip 的确定性门槛：

1. Seed 101 tracker 必须提交 update 30；
2. `global_step_30` 必须通过完整 FSDP checkpoint 校验；
3. C3 model-only 必须通过模型大小、tensor 数和 `model_type` 校验；
4. Seed 101 launcher 与 checkpoint watcher 必须正常退出；
5. GPU 4–7 每张显存占用必须低于 5GB，磁盘可用空间必须不少于约 160GB；
6. Seed 202 的原始 update 10 FSDP checkpoint 与 C1 model-only 必须再次通过校验。

全部满足后：

- 删除已完成 Seed 101 的两个滚动 optimizer checkpoint，并生成删除清单，保留 C1/C2/C3 model-only、全部轨迹、metrics 和 provenance；
- 在 GPU 4–7 启动 `qwen35-clean-formal-seed202-u30-resume10-s1-20260902`；
- 从原始 Seed 202 的 `global_step_10` 恢复，之前 OOM run 的 update 11–15 仍不纳入正式 lineage；
- Seed 202 继续采用 `save_freq=1`、保留最近 2 个完整恢复点、update 20/30 自动转换 model-only 的规则。

监督入口与日志：

```text
SkillRL/phase1/start_seed202_after_seed101.sh
SkillRL/artifacts/logs/seed101-to-seed202-handoff-20260902.log
```

## Seed 202 完成与 Seed 303 八卡启动（2026-09-02）

截至 2026-09-02 23:09 CST，Seed 101 与 Seed 202 均已完成累计 30 updates；两者的 update 10/20/30 model-only 权重均已验证为约 9.1GB、427 tensors、`qwen3_5_text`。Seed 202 的 update 29/30 完整 optimizer checkpoint 在 C1/C2/C3 验证后删除，释放约 99GB；删除清单为：

```text
SkillRL/artifacts/manifests/qwen35-clean-formal-seed202-u30-resume10-s1-20260902-completed-full-checkpoint-cleanup.json
```

2026-09-02 23:33 CST，按用户指示使用全部 8 张空闲 A800 启动 Seed 303：

```text
run_id: qwen35-clean-formal-seed303-u30-8gpu-s1-20260902
CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
FSDP world size: 8
global train batch: 32
PPO mini-batch: 32
micro-batch per GPU: 1
save_freq: 1
full checkpoints retained: 2
analysis milestones: update 10/20/30
```

启动校验确认 8 个 GPU worker 均已建立，训练进入 `0/30`，每卡初始稳定占用约 34GB。为支持 8-way FSDP，checkpoint watcher 已从固定 4 shards 改为根据 shard 文件名自动推断并校验 world size。

需要在最终统计中披露：Seed 303 使用 8-way FSDP，而 Seed 101/202 使用 4-way FSDP。全局 batch 和优化超参数未变，但并行切分及浮点规约顺序不同，因此三者可用于当前三-seed 现象验证，不能严格表述成仅改变随机 seed 的纯复制。

运行日志：

```text
SkillRL/artifacts/logs/qwen35-clean-formal-seed303-u30-8gpu-s1-20260902.log
SkillRL/artifacts/logs/qwen35-clean-formal-seed303-u30-8gpu-s1-20260902-checkpoint-watcher.log
```

## Seed 303 暂停用于全面 Skill 评估（2026-09-03）

2026-09-03 09:01 CST，按用户指示暂停尚未完成的 Seed 303，并先基于已完成的 Seed 101/202 开展全面首次调用效用评估。暂停前已校验：

- 最新提交点为 update 12；
- update 11/12 各有 8 个 model、optimizer 和 extra-state shards，总大小各约 50GB；
- update 10 的 C1 model-only 为约 9.1GB、427 tensors、`qwen3_5_text`；
- 8 张 GPU 已全部释放；
- update 13 若已产生部分 rollout，只保留为中断 provenance，不视为完成 update。

恢复时必须从 `global_step_12`、使用新的 continuation run ID 和相同 8-way FSDP 配置继续。暂停清单：

```text
SkillRL/artifacts/manifests/qwen35-clean-formal-seed303-u30-8gpu-s1-20260902-paused-update12.json
```

## Seed 303 从 update 12 恢复（2026-09-03）

2026-09-03 17:49 CST，在再次验证源 checkpoint 完整性后，使用全部 8 张空闲 A800 启动 continuation：

```text
continuation run_id: qwen35-clean-formal-seed303-u30-resume12-s2-20260903
resume source: SkillRL/artifacts/checkpoints/qwen35-clean-formal-seed303-u30-8gpu-s1-20260902/global_step_12
source tracker: 12
source FSDP world size: 8
source full checkpoint size: 53,033,829,306 bytes
source shards: 8 model + 8 optimizer + 8 extra-state
CUDA_VISIBLE_DEVICES: 0,1,2,3,4,5,6,7
continuation FSDP world size: 8
save_freq: 1
full checkpoints retained per continuation run: 2
analysis milestones: update 10/20/30
```

U10 model-only 在恢复前再次验证为 9,682,950,504 bytes、427 tensors、`qwen3_5_text`。启动时磁盘可用空间约 375GB。checkpoint watcher 会在 continuation 的首个新 checkpoint 完整提交并校验后才删除旧 U12；因此加载或首次保存失败时仍可回退到旧 U12。旧 U11 暂时保留为第二恢复副本，待新 lineage 建立足够的已校验恢复点后再清理。

启动后 TaskRunner 明确记录 `Setting global step to 12`，8 个 FSDP ranks 均分别读取了 U12 的 model、optimizer 与 extra-state shard；随后 8 个 actor workers 进入 `actor_rollout_generate_sequences`，即 continuation 已开始生成 update 13 的训练 rollout。此时每卡约占用 50.8GB，未发现 OOM、Traceback 或 Ray task exception。

本次 continuation 的归档入口：

```text
SkillRL/artifacts/manifests/qwen35-clean-formal-seed303-u30-resume12-s2-20260903.json
SkillRL/artifacts/manifests/qwen35-clean-formal-seed303-u30-resume12-s2-20260903-session.pid
SkillRL/artifacts/logs/qwen35-clean-formal-seed303-u30-resume12-s2-20260903.log
SkillRL/artifacts/logs/qwen35-clean-formal-seed303-u30-resume12-s2-20260903-checkpoint-watcher.log
```
