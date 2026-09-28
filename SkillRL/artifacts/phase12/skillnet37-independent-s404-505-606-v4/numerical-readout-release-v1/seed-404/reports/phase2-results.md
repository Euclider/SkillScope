## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21T14:38:11.605147+00:00
- Verification Status: UNVERIFIED
- Version Label: numerical_readout_v1

# Phase2：seed 404，原版与稳定数值实现对照

这是已见旧结果后获批的数值实现修正，不追溯改写为新的预登记实验。同一训练批次、U0/U5、动作token、advantage、完整37技能库、router、首调用锚点、阈值、候选池与聚合权重保持。模型前向仍为原BF16；只对读出算术使用FP64。旧版源码/报告和所有RL/效用轨迹保留；没有重新采样RL、性能或O/P/N效用。

legacy_recorded为原报告；stable_raw为规范化OLD概率和稳定零和投影的FP64修正；stable_centered_gate使用同一稳定P，只将C及D门控切换为中心化量。P_centered仅作恒等式检查，不当作新预测器。原版在同次前向上的所有标量读出逐值一致，原版PLACEBO排序报告也已复现；因此不是把后端前向变化误称为数值改进。KL/JS及activation基线保留原定义，不拟合阈值或挑选赢家。

所有自然出现技能与全部首次调用锚点保留；未出现/没有训练读出的NA仍明确记录。有效方向比例改变可能来自数值纠错，不能解释为新轨迹支持。点估计、配对game/continuation区间和历史U0权重精度/恢复边界全部继承原报告。

## 数值一致性

token×control行数：155638；P中心化最大绝对误差：1.77636e-15；原始/中心化稳定门控差异：0；稳定版相对旧版门控差异：0；有效方向差异：0。

## 相同候选池的下降排序（0pp点估计阈值）

| control   | variant              | score                  |   candidates |   declines |   average_precision |   auroc_decline_vs_rest |   spearman |
|:----------|:---------------------|:-----------------------|-------------:|-----------:|--------------------:|------------------------:|-----------:|
| placebo   | legacy_recorded      | C_upd                  |           18 |          5 |            0.449048 |                0.707692 |  0.107217  |
| placebo   | legacy_recorded      | D_contribution         |           18 |          5 |            0.50965  |                0.630769 | -0.147161  |
| placebo   | legacy_recorded      | D_ungated_contribution |           18 |          5 |            0.563333 |                0.738462 |  0.0136649 |
| placebo   | legacy_recorded      | P_int                  |           18 |          5 |            0.42381  |                0.461538 | -0.0641201 |
| placebo   | stable_raw           | C_upd                  |           18 |          5 |            0.449048 |                0.707692 |  0.127189  |
| placebo   | stable_raw           | D_contribution         |           18 |          5 |            0.50965  |                0.630769 | -0.147161  |
| placebo   | stable_raw           | D_ungated_contribution |           18 |          5 |            0.552222 |                0.723077 | -0.0115626 |
| placebo   | stable_raw           | P_int                  |           18 |          5 |            0.42381  |                0.461538 | -0.0641201 |
| placebo   | stable_centered_gate | C_upd                  |           18 |          5 |            0.688095 |                0.723077 |  0.0231253 |
| placebo   | stable_centered_gate | D_contribution         |           18 |          5 |            0.50965  |                0.630769 | -0.147161  |
| placebo   | stable_centered_gate | D_ungated_contribution |           18 |          5 |            0.552222 |                0.723077 | -0.0115626 |
| placebo   | stable_centered_gate | P_int                  |           18 |          5 |            0.42381  |                0.461538 | -0.0641201 |
| null      | legacy_recorded      | C_upd                  |           18 |          5 |            0.325714 |                0.538462 |  0.0651713 |
| null      | legacy_recorded      | D_contribution         |           18 |          5 |            0.406726 |                0.4      | -0.435176  |
| null      | legacy_recorded      | D_ungated_contribution |           18 |          5 |            0.371551 |                0.538462 | -0.163979  |
| null      | legacy_recorded      | P_int                  |           18 |          5 |            0.36     |                0.538462 |  0.063069  |
| null      | stable_raw           | C_upd                  |           18 |          5 |            0.325714 |                0.538462 |  0.0809385 |
| null      | stable_raw           | D_contribution         |           18 |          5 |            0.406726 |                0.4      | -0.435176  |
| null      | stable_raw           | D_ungated_contribution |           18 |          5 |            0.371551 |                0.538462 | -0.163979  |
| null      | stable_raw           | P_int                  |           18 |          5 |            0.36     |                0.538462 |  0.063069  |
| null      | stable_centered_gate | C_upd                  |           18 |          5 |            0.574188 |                0.553846 | -0.153468  |
| null      | stable_centered_gate | D_contribution         |           18 |          5 |            0.406726 |                0.4      | -0.435176  |
| null      | stable_centered_gate | D_ungated_contribution |           18 |          5 |            0.371551 |                0.538462 | -0.163979  |
| null      | stable_centered_gate | P_int                  |           18 |          5 |            0.36     |                0.538462 |  0.063069  |

NULL/PLACEBO、所有阶段、0/5pp阈值、全部基线及同池预算见CSV。差异是描述性数值敏感性，不意味着显著改善；单seed不能证明跨seed泛化。D是单侧风险，D=0不能判断稳定/改善；P符号准确率与排序相关性分别记录。

## 统计误用检查

覆盖11/11类（适用性与证据边界）：

- Simpson：control/phase分层保留，不用合并方向代替分层。
- 生态谬误：分析单位为skill窗口，不能下推单token或外推独立seed。
- Berkson：自然调用与可计算性限定候选池，完整coverage保留NA。
- Collider：无新增按目标结果筛选或协变量控制。
- 基率忽略：每组明确candidates与declines，零事件指标NA。
- 均值回归：不按极端效用选择技能或宣布改善。
- 幸存者偏差：完整效用标签是验收条件，未观测技能仍在coverage。
- 多重检验：全部预先登记变体/层次展示，不据此宣布显著赢家。
- 分析路径：事后数值修正明确披露，原版与全部固定变体保留。
- 相关不等于因果：读出关联不证明技能编辑收益。
- 反向因果：读出公式不使用U5效用，但历史标签已可见，不宣称新前瞻研究。
