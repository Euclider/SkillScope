# SkillRL / Qwen3.5-4B 日报进度（2026-08-31）

## 当前结论

Qwen3.5-4B 已完成正式多 seed RL 前的单 update 可行性闭环：模型能够进行 clean-only ALFWorld rollout、真实 GRPO 参数更新、checkpoint 保存与恢复、HF 权重合并，以及 ORIGINAL/PLACEBO/NULL 首次调用点干预。首轮正式范围已按用户指示改为 3 seeds × 30 updates；seed 101 于 2026-08-31 21:05 CST 启动，101/202/303 均属于预先指定的首轮范围。

## 今日已完成

- 模型与环境：Qwen3.5-4B 已下载至 `/home/wangyifan/model/Qwen3.5-4B`，Qwen3.5 text-only、Transformers、FSDP 和 HF rollout 适配完成。
- B0 能力门槛：在 valid_seen clean 的 54 个 episode 上，success 为 `32/54 = 59.26%`，action parse 为 `100%`，admissible action 为 `99.46%`；因此直接冻结原始 post-trained Qwen3.5-4B 为公共 B0，无需额外 warm-up。
- Router 覆盖：B0 rollout 已自然选择 `cle_003 / cle_004 / cle_006 / gen_002`，包含非 step-0 的 Skill 首次调用位置，能够支持后续中途 prefix intervention。
- 真实 RL gate：seed 101 已完成 1 个 global update（32 条训练 rollout），训练成功率 `19/32 = 59.4%`，valid action ratio `99.7%`，随后 validation success 为 `62.5%`。
- 参数更新核验：48.41 亿参数元素中约 45.81 亿发生变化；相对参数 L2 位移为 `3.67e-4`，全部参数有限，证明不是伪更新。
- Checkpoint 闭环：4-GPU FSDP checkpoint 已完整保存并成功恢复；恢复后 validation success 仍为 `62.5%`。checkpoint 已成功合并为约 9.1 GB 的 HF model-only 权重。
- 三臂干预 gate：在同一个 `cle_003` step-20 自然锚点上，三臂 prefix 均严格重放成功。ORIGINAL 首动作转向水槽并执行 clean，PLACEBO/NULL 首动作转向抽屉，证明 payload 替换可以触发 action flip 和 suffix divergence。该单锚点三臂 reward 都为 0，仅用于验证评测链路，不作为 Skill 效用结论。
- 归档：训练轨迹、validation 轨迹、模型权重、参数差分、恢复日志和三臂干预轨迹均已保存在 `SkillRL/artifacts/` 下。
- 正式启动：`qwen35-clean-formal-seed101-u30-20260831` 已在 GPU 0–3 启动。运行配置已核验为 30 updates、每 update 8 games × 4 rollouts、27 条 valid_seen validation、固定 save/test cadence 10。
- 并行启动：`qwen35-clean-formal-seed202-u30-20260831` 已在 2026-08-31 21:35 CST 使用 GPU 4–7 启动；两个 seed 使用独立 run ID、checkpoint 目录和 Ray 临时目录，公共 JSONL 索引通过文件锁并发追加。

## 正式实验配置

- 任务：clean，多 concrete games。
- 公共起点：同一个冻结 B0。
- 首轮 RL seeds：`101 / 202 / 303`。原计划中的 `404 / 505` 暂缓到扩展轮，不因训练性能或 Skill utility 被剔除。
- 每个 seed：30 global updates；每 update 为 `8 games × 4 rollouts = 32` 条轨迹。
- 固定 checkpoint：update `10 / 20 / 30`，不根据性能或 Skill flip 事后筛选。
- Skill Bank 与 state Router：训练期间完全冻结。
- Gold Skill utility：训练结束后才在 valid_unseen 上按 ORIGINAL/PLACEBO/NULL 首次自然调用协议评估。
- 预注册协议：`SkillRL/phase1/config/qwen35_clean_seed_generalization_protocol.json`，schema `v1.3`，SHA256 `45aa5b596335aa971562688da5b4d6309567e66b33e0a4a6b174e5dfd1711003`；seed 范围变更保留在 `seed_scope_amendment` 中，明确未读取 Skill utility/action flip。
- 正式单-seed launcher：`SkillRL/examples/grpo_trainer/run_qwen35_clean_seed_formal.sh`。
- seed 101 日志：`SkillRL/artifacts/logs/qwen35-clean-formal-seed101-u30-20260831.log`。
- seed 101 immutable manifest：`SkillRL/artifacts/manifests/qwen35-clean-formal-seed101-u30-20260831.json`。
- seed 202 日志：`SkillRL/artifacts/logs/qwen35-clean-formal-seed202-u30-20260831.log`。
- seed 202 immutable manifest：`SkillRL/artifacts/manifests/qwen35-clean-formal-seed202-u30-20260831.json`。

## 当前待办与预估

1. seed 101 运行至 update 10 后，立即合并并校验 C1 model-only checkpoint；随后继续 update 20/30。
2. 运行其余 2 个 seed（202/303）：当前单 update 实测约 49 分钟；validation 只在 milestone 执行，预计每 seed 约 20–24 小时。
3. 首轮总训练量为 90 updates、约 2,880 条训练 rollout；8 张 A800 可同时运行两个 4-GPU seed。
4. 全 Skill anchor 构建及三臂评测：预计在训练完成后再根据自然 Router coverage 决定规模，约 1–3 天。
5. 汇总分层 bootstrap、action flip、suffix divergence 与全局性能解耦统计：约 0.5–1 天。

## 风险与资源

- 当前磁盘剩余约 251 GB，不能同时长期保留全部 FSDP optimizer checkpoint。正式运行需在每个 milestone 验证 HF model-only 合并后清理中间完整训练态，仅保留轨迹、指标、三个 model-only milestone 和必要的活动恢复点。
- 启动前已删除已知不完整且无法恢复的 a7 FSDP 训练态，释放约 49 GB；a7 的 model-only 权重、差分、日志和轨迹继续保留。启动时可用空间约 299 GB。
- seed 202 启动前进一步清理约 149 GB 可替代训练态：旧 1.5B pilot 的五组 FSDP shards（五个 HF merged checkpoints 均保留）、Qwen3.5 a8 feasibility 的完整 FSDP 恢复态（恢复闭环已验证且 a8 merged model 保留）、以及被 a8 取代的 a7 merged 副本。上述删除不可恢复，但报告、metrics、日志、轨迹和所需 merged 权重均保留；清理后可用空间约 448 GB。0.5B 记录仅约 4 MB、模型约 954 MB，因释放收益很小而未删除。
- 两个并行 seed 若暂时保留全部六个正式 FSDP milestone，预计占用约 300 GB，仍留约 147 GB。实际执行中将在新 milestone 完成并验证 model-only 合并后清理被取代的旧恢复态，避免触底。

## 2026-09-01 中断恢复补记

- seed 101 原进程在 C1 前中断：完成 update 1–5、另有一批只完成 rollout，因无完整 checkpoint 而从公共 B0 以新 run ID 全量重跑。
- seed 202 原进程在完成 update 13 后中断：C1=update 10 完整可恢复，因此舍弃无 checkpoint 的 update 11–13 状态，从 C1 精确恢复。
- seed 202 C1 已合并并验证为 9.1 GB HF model-only 权重。
- 恢复后 seed 101 r2 显示 `0/30`，seed 202 r2 显示 `10/30`；两组已重新并行运行。
- 完整纳入/排除与 checkpoint lineage 见 `/home/wangyifan/skill-RL/2026-09-01-rl-interruption-and-resume-record.md`。
- 单 update 的主要耗时在 actor backward/update；HF rollout 已通过 prompt 上限从 4096 调整到 2048 明显加速，且该调整只依据 prompt 长度，不读取 Skill utility 或 action flip。
- 目前只能声称验证不同随机训练路径下的 seed 泛化，不能声称不同 RL 超参数设置的泛化。
