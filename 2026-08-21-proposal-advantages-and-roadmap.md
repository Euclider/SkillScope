# Proposal 优势、推进计划与资源预估

> 

## 1. 核心问题与主贡献

一次 RL update 完成后，旧 Skill 在新 Policy 上的边际效用可能发生变化，甚至从有益转为有害。传统方法通常要等待新 Policy 产生足够多的 post-update trajectories，再依据 reward、失败轨迹或 LLM summary 事后发现并修改 Skill，因而存在反馈滞后、归因混杂和 rollout 成本高的问题。

本工作的核心问题是：

> **在不执行完整 post-update rollout evaluation 的情况下，能否利用本次 Policy update 已产生的低成本参数与功能 readout，提前预测哪些旧 `(Skill, context)` 的边际效用将下降，尤其是发生由正到负的 harmful sign flip？**

主贡献不是声称参数变化就是 Skill 失效的机制，而是验证：

> **更新后的 Policy 本身是否携带对旧 Skill 边际效用变化具有增量预测能力的信号。**

## 2. 相比轨迹驱动方法的预期优势

### 2.1 归因更早

诊断发生在一次 RL update 之后、完整 post-update evaluation 之前。若信号有效，无需等待大量新失败轨迹积累，就能优先定位高风险 `(Skill, context)`，缩短从 Policy 改变到 Skill 审计、刷新或淘汰之间的反馈延迟。

### 2.2 归因对象更准确

方法同时比较 Skill-conditioned 与 Skill-free 的 Policy change，并在离线评估阶段用 Skill/no-Skill matched rollouts 构造边际效用标签。这有助于区分：

- Policy 整体能力退化；
- Policy 已内化部分 Skill 能力；
- Skill 与新 Policy 产生有害交互；
- Skill 与新 Policy 仍保持正向协同。

因此，它预测的是 **Skill-specific marginal-utility change**，而不是把所有任务失败都归因于 Skill。

### 2.3 Policy update 具有可检验的预测价值

参数投影、activation interaction shift、action-distribution shift、reward-directed interaction signal 等都只作为候选 readout。通过跨 update 的数据隔离和 held-out matched evaluation，可以检验这些信号是否在旧 Skill margin、update magnitude、原轨迹 reward 等强基线之外，仍能提高 harmful sign-flip 的预测性能。

若这一点成立，即可说明 Policy update 不只是训练过程中的中间产物，还可以作为 Skill 演化的早期诊断信息源。

### 2.4 可转化为更低成本、更少回滚的 Skill 演化

预测器先对 `(Skill, context)` 排序，只对高风险项投入 post-update audit 或编辑预算。相较于“全面 rollout 后再 summary”的旧流程，预期形成更优的 accuracy–cost Pareto frontier：

- 在相同 harmful-Skill recall 下，减少 post-update rollout 数和生成 token 数；
- 在相同 audit/edit budget 下，提高 Precision@K、AUPRC 和有效修改比例；
- 减少误改正常 Skill 导致的 false edit 与后续 Skill rollback；
- 在不降低最终任务成功率的前提下，提高单位 rollout 或单位 token 找到并修复有害 Skill 的数量。

这里的“明显改进”应同时满足统计显著和实际有用，不能只报告单一准确率上涨。建议至少联合报告：

| 目标 | 建议指标 |
|---|---|
| 预测准确性 | AUPRC、Precision@K、Recall@K、Brier/ECE、置信区间 |
| rollout 效率 | 达到固定 recall 所需 rollouts、每发现一个 harmful pair 的 rollouts |
| token 效率 | 每个有效 Skill 修改消耗的生成 token、相对旧方法的 token 节省比例 |
| 修改质量 | beneficial-edit rate、false-edit rate、修改后的 held-out utility gain |
| 稳定性 | Skill rollback rate、跨 seed/update/context 的预测一致性 |
| 最终性能 | 环境成功率、平均 return，以及性能—成本 Pareto 曲线 |

## 3. 精简推进计划：存在 → 预测 → 增量价值

### 阶段一：证明现象真实、稳定且足够频繁

固定 Skill Bank，在连续 RL checkpoints 上选取 update 前已确认有益的 `(Skill, context)`，用独立 matched Skill/no-Skill evaluation 测量 update 前后的边际效用。

需要回答：

1. RL update 后，Skill 边际效用变化和 harmful sign flip 是否自然发生；
2. 事件比例是否足以支持预测研究，而不是极少数偶然样本；
3. 同类 shift 在不同 minibatch、seed、update 和相似 context 上是否可重复；
4. 置信区间明确的变化是否显著强于随机方向、打乱 reward 等负对照。

主要输出为：边际效用变化分布、harmful sign-flip 发生率、稳定事件比例、不同 Skill/context 的异质性，以及低支持或标签模糊样本的 abstention 比例。

**推进条件：** 若可靠 sign flip 过少，应先将主任务降级为连续边际效用变化预测或事件频率研究，不能通过事后挑选 checkpoint 人为制造正例。

### 阶段二：证明 Policy-update signal 与效用变化有关且具有预测能力

在不使用 post-update held-out rollout label 的前提下，从 update 中提取参数投影、activation、action distribution、Skill-free/Skill-conditioned interaction 等候选信号，对 `(Skill, context)` 进行风险排序。随后才运行 held-out matched evaluation 构造 gold label。

训练与测试必须按 **update 级别** 隔离，防止同一次更新的相似状态同时出现在训练集和测试集。重点比较：

- 旧 Skill margin；
- 全局 update magnitude；
- update batch 的 trajectory reward/advantage 与 LLM summary；
- 各类 Policy-update readout；
- 小预算 post-update behavioral probe；
- 完整 matched evaluation oracle。

本阶段的关键结论不是相关系数本身，而是：在控制旧 margin、整体 Policy regression 和原轨迹反馈后，Policy-update readout 是否仍能在 held-out updates 上带来稳定的 AUPRC、Precision@K 或 calibration 增益。

### 阶段三：证明在没有完整 post-update evaluation 时具有增量价值

冻结新 Policy 和固定 Skill 编辑器，仅改变“审计/修改哪些 Skill”的选择策略，比较：

1. trajectory-summary-guided；
2. old-margin-guided；
3. random audit；
4. budget-matched behavioral probe；
5. Policy-update-signal-guided；
6. full-evaluation oracle。

在相同 audit/edit budget 下报告：

- 高风险 Skill 的捕获比例和漏检比例；
- 可以省去的完整 post-update evaluation 比例；
- 达到固定 recall 所需的 rollout/token 数；
- 有效修改比例、false-edit rate 与 rollback rate；
- 编辑后的 held-out utility 和最终任务成功率。

由此验证完整链路：

`Policy update → 低成本风险预测 → 定向 audit/edit → 更少 rollout/token 与回滚 → 保持或提高任务性能`

## 4. Benchmark 与模型选择

### 最小可行版本

- **主 Benchmark：ALFWorld。** 成功条件清晰，任务类型可分簇，便于重建相同 context 并执行 Skill/no-Skill matched evaluation，适合作为主结论环境。
- **Policy：单一 7B/8B instruct 模型。** 可优先选择 Qwen2.5-7B-Instruct、Qwen3-8B 或同规模开放模型中的一种，避免最小实验同时引入模型家族变量。
- **训练：GRPO 或同类 group-based agentic RL。** 第一阶段冻结 Skill Bank，仅更新 Policy，以隔离 Policy update 对旧 Skill 效用的影响。
- **快速管线验证：** 可先使用同家族 1.5B/3B 模型验证 checkpoint、probe、matched evaluation 和标签构造流程，但论文主结果应在 7B/8B 上完成。

### 外部复验

- **ScienceWorld：** 检验更长程、组合式科学任务中的可迁移性；
- **WebShop：** 检验搜索、选择和稠密/半稠密 reward 场景；
- 仅在 ALFWorld 上确认“现象存在、可预测、具有成本优势”后再扩展，避免前期资源被多环境工程分散。

## 5. 资源与周期粗估

以下按 **8 × A100-80GB**、7B/8B 模型估算；实际消耗高度依赖环境吞吐、平均轨迹长度、并行框架、checkpoint 密度和是否采用 LoRA/全参数更新。

| 阶段 | 建议规模 | 粗略计算预算 |
|---|---|---|
| 管线 pilot | 1 个小模型或少量 8B updates，1 seed | 8 卡连续 2–4 天，约 384–768 A100 GPU-hours |
| ALFWorld 主实验 | 8B，多个连续 updates，至少 3 seeds，完整基线与消融 | 8 卡连续 7–14 天，约 1,344–2,688 A100 GPU-hours |
| 外部复验 | ScienceWorld/WebShop 中至少一个，减少消融 | 额外约 1,000–3,000 A100 GPU-hours |
| 完整论文预算 | 主实验、复验、失败重跑及编辑后验证 | 建议预留约 4,000–8,000 A100 GPU-hours |

存储方面，8B BF16 权重单份约为十余 GB；若保存 20–40 个连续快照，仅模型权重就可能需要约 0.3–0.7 TB。建议只对少数关键 checkpoint 保存完整 optimizer state，其余保存模型权重、LoRA/delta、update metadata 和 rollout 日志。整体建议配置：

- 2–4 TB NVMe；
- 256 GB 左右主机内存；
- 32–64 个 CPU cores 用于环境 workers 与评估；
- 完整 experiment tracking，包括 update id、seed、Skill/context id、probe 版本、rollout budget 和 token budget。
