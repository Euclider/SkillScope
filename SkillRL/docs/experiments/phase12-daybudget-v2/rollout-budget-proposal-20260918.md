# Rollout 数量缩减：待确认提案，不改冻结配置

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-18
- Verification Status: UNVERIFIED
- Version Label: phase12_rollout_reduction_proposal_v1

## 问题和结论边界

用户询问：在还能验证 idea 的前提下，合理降低 rollout 数是否可以提高训练和评估速度。答案是可以，但训练采样、完整性能评估、效用配对续跑是三个独立预算；只减少训练 rollout 不会自动减少后两者，也不会使 checkpoint/模型初始化等固定开销减半。

本项目 Phase1–2 检验的是：真实 RL 更新后，在读取目标 gold 前固定的 readout 排序是否与技能效用变化相关。它不要求先把 policy 训练至收敛，但需要足够的有效更新、自然 skill 支持、可辨别的效用变化和足够的技能排序单元。没有先验保证某个更小样本量一定能验证 idea；支持不足或置信区间过宽应报告 inconclusive，不能当作 idea 被证伪，也不能按已经看到的 gold 加样本追求显著。

## 优先提案 A：先减训练，保留效用标签精度

| 项目 | 已确认 v2 | 待确认 A |
|---|---:|---:|
| games/iteration | 16 | 8 |
| trajectories/game 的 GRPO group size | 8 | 8 |
| 每轮训练 trajectories | 128 | 64 |
| RL iterations / 固定窗口 | 5 / U0→U5 | 不变 |
| 总训练 trajectories | 640 | 320 |
| U0/U5 全量 seen+unseen performance | 548 | 不变 |
| unseen anchor-source episodes | 134 | 不变 |
| 最多自然支持 skills | 12 | 不变 |
| anchors/skill 上限 | 12 | 不变 |
| evidence/gold seeds | 1 / 2 | 不变 |
| O/P/N 续跑上界 | 2592 | 不变 |

优先保留每个 GRPO group 的 8 条轨迹；组内 reward 全相同时，当前均值中心化 GRPO advantage 为零。减少组内轨迹可能更容易丢掉组内对比，但实际影响需看 reward 分布，不能保证少任务组一定优于更多任务组/更少重复。A 的代价是任务和技能自然覆盖减少；保留现有支持门槛，不为了报告更多技能而降低门槛。

若采用 A，应另建 v3 或后续版本，核对实际多步训练 transition batch、PPO minibatch、梯度累积和八卡分片的可整除/归一化；不得仅修改 train_batch_size 而静默丢样本、拆坏 GRPO group 或沿用不匹配的更新账本。不能再将每轮128轨迹称为完全照搬官方脚本；应注明同一GRPO recipe的预算适配。未来baseline必须共用同一采样与实际更新预算。

按此前保守监测上界128计，整个流程 episode/continuation 总上界由4042降至3722，数量仅下降约7.9%。这不代表只省7.9%的时间：训练还包含优化器和全词表记录，单位成本不同；但更不能承诺整体快一倍。主要速度改进仍需高吞吐后端、八卡并发及 router 批量化配合。

## 更紧的备选 B：再减 anchor 数（不默认采用）

在 A 基础上将每技能 anchors 上限12→8，继续保留最多12个skill和1 evidence+2 gold seeds，O/P/N续跑上界2592→1728（少约1/3），全流程数量上界2858。代价是单技能效用差异/变化估计更粗；重复seed共享anchor，不能把所有续跑当独立样本，分析仍需按anchor/game结构处理相关性。不能在无效应量/方差估计时承诺有足够统计功效。

不优先削减技能排序单元数、丢掉PLACEBO/NULL、只留1个gold seed、或把全量seen/unseen改为挑游戏。固定5轮窗口、完整37技能路由、既定最大长度、读出记录粒度继续保留。

## 执行状态

A/B均为待确认提案，没有写入active profile/preparation，没有新训练/评估。优先建议A结合verl+vLLM迁移，先验证真实batch与recording资源，再冻结最终预算；若时间仍不足，B必须在目标gold之前另行确认登记，不能运行到一半按效果裁减。当前v2的失败预检和NOT_ADMITTED状态不变。
