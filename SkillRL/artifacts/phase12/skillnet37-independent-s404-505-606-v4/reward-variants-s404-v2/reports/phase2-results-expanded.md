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


---

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-22T08:05:53.388811+00:00
- Verification Status: ANALYZED
- Version Label: reward_variants_seed404_exploratory_v1

# seed404：reward-directed 读出变式扩展分析

共 39 个奖励相关公式 × 3 种聚合 = 117 个 reward 分数，另有相应去奖励对照、幅度和旧基线，共 261 列。主池 18 技能，0pp 点估计下降 5 种；不是独立重复实验的数量。所有公式和方向在本次实际计算前登记，但历史标签已见，属于事后探索。

目标是检验 reward 是否提供增量信息，而不是必须找出赢家。最高 AP 仅为描述性最优，不能把同一 seed 上大量尝试后的改善写成已证实泛化或 Phase3 编辑收益。原 C/中心化 C 本身含 reward。

## 1. 相同候选池：各家族探索性最高 AP（不是确认性选型）

| score                                     |   average_precision |   auroc_decline_vs_rest |   auroc_decline_vs_increase |   spearman |   sign_accuracy_called |   sign_coverage |
|:------------------------------------------|--------------------:|------------------------:|----------------------------:|-----------:|-----------------------:|----------------:|
| C_centered::decision::reward              |             0.69359 |                 0.73846 |                     0.68571 |    0.04205 |              nan       |       nan       |
| D_negative_fraction::game::reward         |             0.64722 |                 0.73846 |                     0.62857 |    0.12088 |              nan       |       nan       |
| D_tauC_q25::game::reward                  |             0.59222 |                 0.60000 |                     0.62857 |    0.37736 |                0.75000 |         1.00000 |
| D_soft_negative::game::reward             |             0.57583 |                 0.55385 |                     0.51429 |    0.01577 |              nan       |       nan       |
| D_cos_centered_gate::game::reward         |             0.57043 |                 0.61538 |                     0.68571 |    0.36054 |                0.66667 |         1.00000 |
| D_adv::decision::reward                   |             0.57038 |                 0.53846 |                     0.57143 |    0.20708 |                0.58333 |         1.00000 |
| D_adv_negative_fraction::decision::reward |             0.55091 |                 0.72308 |                     0.54286 |   -0.01578 |              nan       |       nan       |
| D_tanh::game::reward                      |             0.49083 |                 0.55385 |                     0.57143 |    0.16818 |                0.58333 |         1.00000 |
| D_ratio_adv::decision::reward             |             0.43703 |                 0.47692 |                     0.60000 |    0.23020 |                0.66667 |         1.00000 |
| D_policy_action_adv::game::reward         |             0.29631 |                 0.46154 |                     0.57143 |    0.04520 |                0.58333 |         1.00000 |

所有家族、聚合和负对照均在完整 CSV；这里的最佳值包含公式/聚合选择偏差。

## 2. 无 reward 幅度与原有参考

| score                                 |   average_precision |   auroc_decline_vs_rest |   auroc_decline_vs_increase |   spearman |
|:--------------------------------------|--------------------:|------------------------:|----------------------------:|-----------:|
| M_delta_raw::token::magnitude         |             0.36154 |                 0.55385 |                     0.57143 |    0.06412 |
| M_delta_centered::token::magnitude    |             0.61365 |                 0.73846 |                     0.74286 |    0.34898 |
| M_original_norm::token::magnitude     |             0.50317 |                 0.67692 |                     0.74286 |    0.50771 |
| M_control_norm::token::magnitude      |             0.35216 |                 0.55385 |                     0.82857 |    0.52032 |
| M_kl::token::magnitude                |             0.26556 |                 0.33846 |                     0.22857 |   -0.42992 |
| M_js::token::magnitude                |             0.23175 |                 0.27692 |                     0.17143 |   -0.46040 |
| M_action_abs::token::magnitude        |             0.46345 |                 0.69231 |                     0.65714 |    0.01261 |
| M_delta_raw::decision::magnitude      |             0.40189 |                 0.58462 |                     0.57143 |    0.02943 |
| M_delta_centered::decision::magnitude |             0.64222 |                 0.80000 |                     0.77143 |    0.40680 |
| M_original_norm::decision::magnitude  |             0.51216 |                 0.70769 |                     0.74286 |    0.49825 |
| M_control_norm::decision::magnitude   |             0.31645 |                 0.50769 |                     0.77143 |    0.36790 |
| M_kl::decision::magnitude             |             0.25788 |                 0.35385 |                     0.22857 |   -0.40680 |
| M_js::decision::magnitude             |             0.22119 |                 0.24615 |                     0.11429 |   -0.47722 |
| M_action_abs::decision::magnitude     |             0.41806 |                 0.64615 |                     0.57143 |   -0.08830 |
| M_delta_raw::game::magnitude          |             0.41298 |                 0.53846 |                     0.48571 |   -0.11668 |
| M_delta_centered::game::magnitude     |             0.72476 |                 0.80000 |                     0.80000 |    0.36265 |
| M_original_norm::game::magnitude      |             0.51216 |                 0.69231 |                     0.80000 |    0.45725 |
| M_control_norm::game::magnitude       |             0.38996 |                 0.66154 |                     0.82857 |    0.44569 |
| M_kl::game::magnitude                 |             0.29171 |                 0.32308 |                     0.22857 |   -0.46461 |
| M_js::game::magnitude                 |             0.23522 |                 0.29231 |                     0.17143 |   -0.43202 |
| M_action_abs::game::magnitude         |             0.29417 |                 0.46154 |                     0.34286 |   -0.24597 |
| B_activation_l8_norm                  |             0.74333 |                 0.87692 |                     0.82857 |    0.31009 |
| B_activation_l16_norm                 |             0.59111 |                 0.78462 |                     0.71429 |    0.18816 |
| B_activation_l24_norm                 |             0.65556 |                 0.78462 |                     0.71429 |    0.20287 |
| B_activation_l32_norm                 |             0.65333 |                 0.81538 |                     0.80000 |    0.40259 |
| B_old_margin                          |             0.43583 |                 0.63077 |                     0.54286 |   -0.10177 |
| B_random_expected                     |             0.27778 |                 0.50000 |                     0.50000 |  nan       |

## 3. 全部奖励公式：固定 token 等权

| score                                  |   average_precision |   auroc_decline_vs_rest |   auroc_decline_vs_increase |   spearman |   sign_accuracy_called |   sign_coverage |
|:---------------------------------------|--------------------:|------------------------:|----------------------------:|-----------:|-----------------------:|----------------:|
| D_original::token::reward              |             0.50965 |                 0.63077 |                     0.42857 |   -0.14716 |              nan       |       nan       |
| D_centered_gate::token::reward         |             0.50965 |                 0.63077 |                     0.42857 |   -0.14716 |              nan       |       nan       |
| D_ungated::token::reward               |             0.55222 |                 0.72308 |                     0.54286 |   -0.01156 |              nan       |       nan       |
| D_signed::token::reward                |             0.42381 |                 0.46154 |                     0.45714 |   -0.06412 |                0.41667 |         1.00000 |
| D_signed_gate::token::reward           |             0.41600 |                 0.41538 |                     0.42857 |   -0.06622 |                0.41667 |         1.00000 |
| D_sign_balance::token::reward          |             0.47333 |                 0.56923 |                     0.51429 |    0.02102 |                0.41667 |         1.00000 |
| D_negative_fraction::token::reward     |             0.56551 |                 0.73846 |                     0.62857 |    0.11668 |              nan       |       nan       |
| D_adv::token::reward                   |             0.56580 |                 0.52308 |                     0.54286 |    0.16398 |                0.50000 |         1.00000 |
| D_adv_gate::token::reward              |             0.42552 |                 0.43077 |                     0.45714 |    0.09776 |                0.58333 |         1.00000 |
| D_sqrtadv::token::reward               |             0.52154 |                 0.58462 |                     0.57143 |    0.15977 |                0.41667 |         1.00000 |
| D_work::token::reward                  |             0.25583 |                 0.36923 |                     0.51429 |    0.07148 |                0.50000 |         1.00000 |
| D_work_gate::token::reward             |             0.41583 |                 0.43077 |                     0.57143 |    0.27120 |                0.66667 |         1.00000 |
| D_cos_raw::token::reward               |             0.44011 |                 0.50769 |                     0.48571 |    0.00736 |                0.41667 |         1.00000 |
| D_cos_centered::token::reward          |             0.43297 |                 0.49231 |                     0.45714 |   -0.02943 |                0.41667 |         1.00000 |
| D_cos_centered_gate::token::reward     |             0.45164 |                 0.49231 |                     0.45714 |   -0.01051 |                0.41667 |         1.00000 |
| D_cos_adv::token::reward               |             0.56580 |                 0.52308 |                     0.54286 |    0.22074 |                0.58333 |         1.00000 |
| D_cos_negative::token::reward          |             0.40465 |                 0.63077 |                     0.48571 |   -0.10722 |              nan       |       nan       |
| D_soft::token::reward                  |             0.49913 |                 0.50769 |                     0.54286 |    0.10722 |                0.41667 |         1.00000 |
| D_soft_negative::token::reward         |             0.51630 |                 0.56923 |                     0.45714 |   -0.15242 |              nan       |       nan       |
| D_soft_cos::token::reward              |             0.46580 |                 0.49231 |                     0.54286 |    0.06517 |                0.41667 |         1.00000 |
| D_sigmoid::token::reward               |             0.44071 |                 0.49231 |                     0.48571 |   -0.01787 |                0.41667 |         1.00000 |
| D_tauC_q25::token::reward              |             0.41545 |                 0.41538 |                     0.45714 |   -0.04310 |                0.41667 |         1.00000 |
| D_tauC_q50::token::reward              |             0.43175 |                 0.41538 |                     0.45714 |   -0.02418 |                0.33333 |         1.00000 |
| D_tauC_q75::token::reward              |             0.46580 |                 0.49231 |                     0.54286 |    0.12929 |                0.33333 |         1.00000 |
| D_delta_q25::token::reward             |             0.41996 |                 0.44615 |                     0.45714 |   -0.05781 |                0.41667 |         1.00000 |
| D_delta_q50::token::reward             |             0.27265 |                 0.41538 |                     0.37143 |   -0.11247 |                0.33333 |         1.00000 |
| D_delta_q75::token::reward             |             0.29219 |                 0.32308 |                     0.34286 |   -0.14926 |                0.33333 |         1.00000 |
| D_winsor::token::reward                |             0.42821 |                 0.47692 |                     0.45714 |   -0.05046 |                0.41667 |         1.00000 |
| D_tanh::token::reward                  |             0.43535 |                 0.49231 |                     0.45714 |   -0.03889 |                0.41667 |         1.00000 |
| D_action_sign::token::reward           |             0.31389 |                 0.49231 |                     0.60000 |    0.23230 |                0.58333 |         1.00000 |
| D_action_adv::token::reward            |             0.40220 |                 0.38462 |                     0.54286 |    0.24912 |                0.66667 |         1.00000 |
| D_action_adv_clip::token::reward       |             0.29848 |                 0.35385 |                     0.51429 |    0.22284 |                0.66667 |         1.00000 |
| D_ratio_adv::token::reward             |             0.40583 |                 0.40000 |                     0.51429 |    0.16188 |                0.66667 |         1.00000 |
| D_policy_action_adv::token::reward     |             0.24831 |                 0.33846 |                     0.57143 |    0.20813 |                0.58333 |         1.00000 |
| D_reward_only::token::reward           |             0.49222 |                 0.47692 |                     0.37143 |   -0.26292 |                0.25000 |         1.00000 |
| D_adv_negative_fraction::token::reward |             0.55091 |                 0.72308 |                     0.54286 |   -0.06731 |              nan       |       nan       |
| C_raw::token::reward                   |             0.44905 |                 0.70769 |                     0.68571 |    0.12719 |              nan       |       nan       |
| C_centered::token::reward              |             0.68810 |                 0.72308 |                     0.65714 |    0.02313 |              nan       |       nan       |
| C_adv_centered::token::reward          |             0.35882 |                 0.44615 |                     0.31429 |   -0.36054 |              nan       |       nan       |

## 4. reward 的配对增益，而非仅看最高 AP

| score                                     | reference_kind                      |   point_difference |      low |    high |   valid_draws |
|:------------------------------------------|:------------------------------------|-------------------:|---------:|--------:|--------------:|
| D_adv::decision::reward                   | original_D                          |            0.06073 | -0.33405 | 0.33664 |          1671 |
| D_adv::decision::reward                   | same_aggregation_centered_magnitude |           -0.07184 | -0.36366 | 0.20380 |          1671 |
| D_adv::decision::reward                   | matched_unsigned                    |            0.12920 | -0.19809 | 0.23995 |          1671 |
| D_ratio_adv::decision::reward             | original_D                          |           -0.07262 | -0.46071 | 0.38338 |          1671 |
| D_ratio_adv::decision::reward             | same_aggregation_centered_magnitude |           -0.20519 | -0.48639 | 0.29652 |          1671 |
| D_ratio_adv::decision::reward             | matched_unsigned                    |            0.11216 | -0.22446 | 0.34316 |          1671 |
| D_adv_negative_fraction::decision::reward | original_D                          |            0.04126 | -0.06342 | 0.09656 |          1671 |
| D_adv_negative_fraction::decision::reward | same_aggregation_centered_magnitude |           -0.09131 | -0.26052 | 0.15111 |          1671 |
| D_adv_negative_fraction::decision::reward | matched_unsigned                    |            0.27313 | -0.01232 | 0.47222 |          1671 |
| C_centered::decision::reward              | original_D                          |            0.18394 | -0.10469 | 0.24596 |          1671 |
| C_centered::decision::reward              | same_aggregation_centered_magnitude |            0.05137 | -0.18735 | 0.16667 |          1671 |
| C_centered::decision::reward              | matched_unsigned                    |            0.48286 | -0.32483 | 0.58968 |          1671 |
| D_negative_fraction::game::reward         | original_D                          |            0.13757 | -0.06195 | 0.15503 |          1671 |
| D_negative_fraction::game::reward         | same_aggregation_centered_magnitude |           -0.07754 | -0.33750 | 0.27371 |          1671 |
| D_negative_fraction::game::reward         | matched_unsigned                    |            0.07675 | -0.36677 | 0.47222 |          1671 |
| D_cos_centered_gate::game::reward         | original_D                          |            0.06078 | -0.14131 | 0.32875 |          1671 |
| D_cos_centered_gate::game::reward         | same_aggregation_centered_magnitude |           -0.15433 | -0.35342 | 0.34299 |          1671 |
| D_cos_centered_gate::game::reward         | matched_unsigned                    |           -0.04322 | -0.27648 | 0.33009 |          1671 |
| D_soft_negative::game::reward             | original_D                          |            0.06618 | -0.10054 | 0.27781 |          1671 |
| D_soft_negative::game::reward             | same_aggregation_centered_magnitude |           -0.14893 | -0.36512 | 0.33883 |          1671 |
| D_soft_negative::game::reward             | matched_unsigned                    |            0.08000 | -0.13750 | 0.25972 |          1671 |
| D_tauC_q25::game::reward                  | original_D                          |            0.08257 | -0.13464 | 0.37697 |          1671 |
| D_tauC_q25::game::reward                  | same_aggregation_centered_magnitude |           -0.13254 | -0.33611 | 0.39497 |          1671 |
| D_tauC_q25::game::reward                  | matched_unsigned                    |            0.03007 | -0.19956 | 0.35644 |          1671 |
| D_tanh::game::reward                      | original_D                          |           -0.01882 | -0.17160 | 0.25099 |          1671 |
| D_tanh::game::reward                      | same_aggregation_centered_magnitude |           -0.23393 | -0.35513 | 0.27024 |          1671 |
| D_tanh::game::reward                      | matched_unsigned                    |            0.02746 | -0.42560 | 0.53402 |          1671 |
| D_policy_action_adv::game::reward         | original_D                          |           -0.21334 | -0.41205 | 0.22976 |          1671 |
| D_policy_action_adv::game::reward         | same_aggregation_centered_magnitude |           -0.42845 | -0.54016 | 0.21511 |          1671 |
| D_policy_action_adv::game::reward         | matched_unsigned                    |            0.05294 | -0.13016 | 0.25048 |          1671 |

差值是 reward 版本 AP 减去参考 AP；区间为固定读出下配对 gold-label bootstrap，未校正候选选择，也未包含 RL seed/训练读出不确定性。P/C 几何去奖励版沿用原 A!=0 支持，只移除该支持上的符号/强度，不能作为全集完全 reward-free 的证据。采样动作类 A=+1 对照覆盖全部 token。

## 5. trajectory-block reward 符号负对照

| score                                     |   observed |    mean |     low |    high |   reference_fraction_ge_observed |   family_max_fraction_ge_observed_AP |
|:------------------------------------------|-----------:|--------:|--------:|--------:|---------------------------------:|-------------------------------------:|
| D_adv::decision::reward                   |    0.57038 | 0.43994 | 0.21380 | 0.73631 |                          0.17969 |                              0.99219 |
| D_ratio_adv::decision::reward             |    0.43703 | 0.46818 | 0.22319 | 0.73889 |                          0.56445 |                              1.00000 |
| D_adv_negative_fraction::decision::reward |    0.55091 | 0.46020 | 0.24111 | 0.84358 |                          0.25000 |                              0.99414 |
| C_centered::decision::reward              |    0.69359 | 0.42573 | 0.20227 | 0.82630 |                          0.08594 |                              0.77734 |
| D_negative_fraction::game::reward         |    0.64722 | 0.52482 | 0.27265 | 0.77192 |                          0.18555 |                              0.89844 |
| D_cos_centered_gate::game::reward         |    0.57043 | 0.48192 | 0.30265 | 0.69889 |                          0.23438 |                              0.99219 |
| D_soft_negative::game::reward             |    0.57583 | 0.56404 | 0.38244 | 0.75606 |                          0.42969 |                              0.99219 |
| D_tauC_q25::game::reward                  |    0.59222 | 0.56273 | 0.44872 | 0.73041 |                          0.31445 |                              0.98633 |
| D_tanh::game::reward                      |    0.49083 | 0.43993 | 0.21476 | 0.72448 |                          0.33984 |                              1.00000 |
| D_policy_action_adv::game::reward         |    0.29631 | 0.34689 | 0.21309 | 0.64558 |                          0.62109 |                              1.00000 |

512 次随机化保留轨迹内相关性与 |A|，重新计算 gate；最后一列用每次随机化的所有 reward 候选最高 AP 做参照，用于显示 look-elsewhere 风险。比例不是校准 p 值，不能据此宣称显著。

## 6. 方向分类、编辑预算与 NULL 稳健性

| score                                     |   k |   precision_at_k |   recall_at_k |   captured_decline_mass |
|:------------------------------------------|----:|-----------------:|--------------:|------------------------:|
| D_adv::decision::reward                   |   1 |          1.00000 |       0.20000 |                 0.25415 |
| D_ratio_adv::decision::reward             |   1 |          1.00000 |       0.20000 |                 0.25415 |
| D_adv_negative_fraction::decision::reward |   1 |          1.00000 |       0.20000 |                 0.45747 |
| C_centered::decision::reward              |   1 |          1.00000 |       0.20000 |                 0.45747 |
| D_negative_fraction::game::reward         |   1 |          1.00000 |       0.20000 |                 0.45747 |
| D_cos_centered_gate::game::reward         |   1 |          1.00000 |       0.20000 |                 0.45747 |
| D_soft_negative::game::reward             |   1 |          1.00000 |       0.20000 |                 0.45747 |
| D_tauC_q25::game::reward                  |   1 |          1.00000 |       0.20000 |                 0.45747 |
| D_tanh::game::reward                      |   1 |          1.00000 |       0.20000 |                 0.45747 |
| D_policy_action_adv::game::reward         |   1 |          0.00000 |       0.00000 |                 0.00000 |
| D_adv::decision::reward                   |   2 |          1.00000 |       0.40000 |                 0.71162 |
| D_ratio_adv::decision::reward             |   2 |          0.50000 |       0.20000 |                 0.25415 |
| D_adv_negative_fraction::decision::reward |   2 |          0.50000 |       0.20000 |                 0.45747 |
| C_centered::decision::reward              |   2 |          1.00000 |       0.40000 |                 0.71162 |
| D_negative_fraction::game::reward         |   2 |          1.00000 |       0.40000 |                 0.52895 |
| D_cos_centered_gate::game::reward         |   2 |          0.50000 |       0.20000 |                 0.45747 |
| D_soft_negative::game::reward             |   2 |          1.00000 |       0.40000 |                 0.52895 |
| D_tauC_q25::game::reward                  |   2 |          0.50000 |       0.20000 |                 0.45747 |
| D_tanh::game::reward                      |   2 |          0.50000 |       0.20000 |                 0.45747 |
| D_policy_action_adv::game::reward         |   2 |          0.00000 |       0.00000 |                 0.00000 |
| D_adv::decision::reward                   |   5 |          0.40000 |       0.40000 |                 0.71162 |
| D_ratio_adv::decision::reward             |   5 |          0.20000 |       0.20000 |                 0.25415 |
| D_adv_negative_fraction::decision::reward |   5 |          0.40000 |       0.40000 |                 0.52895 |
| C_centered::decision::reward              |   5 |          0.60000 |       0.60000 |                 0.78310 |
| D_negative_fraction::game::reward         |   5 |          0.40000 |       0.40000 |                 0.52895 |
| D_cos_centered_gate::game::reward         |   5 |          0.60000 |       0.60000 |                 0.68403 |
| D_soft_negative::game::reward             |   5 |          0.40000 |       0.40000 |                 0.52895 |
| D_tauC_q25::game::reward                  |   5 |          0.60000 |       0.60000 |                 0.68403 |
| D_tanh::game::reward                      |   5 |          0.40000 |       0.40000 |                 0.52895 |
| D_policy_action_adv::game::reward         |   5 |          0.20000 |       0.20000 |                 0.07148 |
| D_adv::decision::reward                   |   9 |          0.22222 |       0.40000 |                 0.71162 |
| D_ratio_adv::decision::reward             |   9 |          0.22222 |       0.40000 |                 0.40923 |
| D_adv_negative_fraction::decision::reward |   9 |          0.33333 |       0.60000 |                 0.78310 |
| C_centered::decision::reward              |   9 |          0.33333 |       0.60000 |                 0.78310 |
| D_negative_fraction::game::reward         |   9 |          0.44444 |       0.80000 |                 0.74585 |
| D_cos_centered_gate::game::reward         |   9 |          0.33333 |       0.60000 |                 0.68403 |
| D_soft_negative::game::reward             |   9 |          0.22222 |       0.40000 |                 0.52895 |
| D_tauC_q25::game::reward                  |   9 |          0.33333 |       0.60000 |                 0.68403 |
| D_tanh::game::reward                      |   9 |          0.33333 |       0.60000 |                 0.68403 |
| D_policy_action_adv::game::reward         |   9 |          0.22222 |       0.40000 |                 0.52895 |

| score                                     |   candidates |   declines |   average_precision |   auroc_decline_vs_rest |   auroc_decline_vs_increase |   spearman |   sign_accuracy_called |   sign_coverage |
|:------------------------------------------|-------------:|-----------:|--------------------:|------------------------:|----------------------------:|-----------:|-----------------------:|----------------:|
| D_adv::decision::reward                   |           18 |          5 |             0.63939 |                 0.69231 |                     0.71429 |    0.28907 |                0.66667 |         1.00000 |
| D_ratio_adv::decision::reward             |           18 |          5 |             0.54771 |                 0.66154 |                     0.74286 |    0.25753 |                0.66667 |         1.00000 |
| D_adv_negative_fraction::decision::reward |           18 |          5 |             0.45963 |                 0.55385 |                     0.37143 |   -0.26187 |              nan       |       nan       |
| C_centered::decision::reward              |           18 |          5 |             0.58371 |                 0.58462 |                     0.51429 |   -0.12088 |              nan       |       nan       |
| D_negative_fraction::game::reward         |           18 |          5 |             0.31583 |                 0.40000 |                     0.42857 |   -0.06097 |              nan       |       nan       |
| D_cos_centered_gate::game::reward         |           18 |          5 |             0.57164 |                 0.53846 |                     0.57143 |    0.31534 |                0.66667 |         1.00000 |
| D_soft_negative::game::reward             |           18 |          5 |             0.41863 |                 0.44615 |                     0.25714 |   -0.26699 |              nan       |       nan       |
| D_tauC_q25::game::reward                  |           18 |          5 |             0.57709 |                 0.55385 |                     0.60000 |    0.28907 |                0.75000 |         1.00000 |
| D_tanh::game::reward                      |           18 |          5 |             0.41556 |                 0.53846 |                     0.54286 |    0.21443 |                0.58333 |         1.00000 |
| D_policy_action_adv::game::reward         |           18 |          5 |             0.38917 |                 0.61538 |                     0.80000 |    0.44359 |                0.75000 |         1.00000 |

带符号方法记录非零 ΔM 的符号准确率、弃权和覆盖；完整 CSV 同时给出弃权计错的准确率。非负 D 与作为风险分数使用的 C 不提供双向分类准确率。AP/AUROC 衡量排序，不等于零阈值方向判断。0/5pp 标签阈值、全部 phase 及所有方法的预算均保留；phase 不是额外独立窗口。

## 7. 原指标复现、样本与不确定性边界

| name             |   matched_rows |   max_abs_error | passed   |
|:-----------------|---------------:|----------------:|:---------|
| D_original       |            140 |     5.82867e-16 | True     |
| D_centered_gate  |            140 |     5.82867e-16 | True     |
| D_ungated        |            140 |     1.9984e-15  | True     |
| D_signed         |            140 |     8.32667e-16 | True     |
| C_raw            |            140 |     1.46367e-18 | True     |
| C_centered       |            140 |     6.28837e-18 | True     |
| M_delta_raw      |            140 |     1.36424e-12 | True     |
| M_delta_centered |            140 |     5.68434e-13 | True     |
| M_original_norm  |            140 |     4.09273e-12 | True     |
| M_control_norm   |            140 |     4.09273e-12 | True     |
| M_kl             |            140 |     4.71845e-16 | True     |
| M_js             |            140 |     1.07553e-16 | True     |

| control   |   requested_draws |   complete_pool_draws |   missing_any_skill_draws |   distinct_source_games | continuation_seeds   |   rng_seed | scope                                                                                          |
|:----------|------------------:|----------------------:|--------------------------:|------------------------:|:---------------------|-----------:|:-----------------------------------------------------------------------------------------------|
| placebo   |              2000 |                  1674 |                       326 |                     134 | [63011, 63021]       |   20260922 | conditional gold-label uncertainty only; not training/seed uncertainty; not selection-adjusted |
| null      |              2000 |                  1674 |                       326 |                     134 | [63011, 63021]       |   20260922 | conditional gold-label uncertainty only; not training/seed uncertainty; not selection-adjusted |

全部效用标签复用已保存的 U0/U5 配对结果；旧区间和报告原文不改。新增 bootstrap 在任一技能缺少抽中 game 时整池记缺失，不为不同方法挑选不同技能。置信区间条件于当前自然出现技能和训练读出，不支持跨 seed 泛化。无额外 RL、模型前向、环境 rollout 或 API；505/606 未进行本轮变式分析。

## 8. 统计误用核查：11/11

- Simpson：control/phase 分层，不用合并结果替代分层。
- 生态谬误：skill-window 单位，不把 token/anchor/重复当独立 seed。
- Berkson：旧共同池保持，完整自然支持及缺失可查。
- Collider：未按结果新增支持筛选或拟合控制变量。
- 基率：每表保留候选/下降/上升数，单类或零事件指标 NA。
- 均值回归：不按极端 ΔM 选技能、检查点或重采样。
- 幸存者：全部候选和 NA 记录，不仅报告赢家。
- 多重检验：完整枚举与 family-max 符号负对照；不作未校正显著性声明。
- 分析路径：已见标签的事后探索，代码/输入/候选先锁定，未伪装前瞻验证。
- 相关非因果：预测关联不证明编辑收益。
- 反向因果/泄漏：评分函数不读效用标签；已有标签知识仍限制确认性解释。

## 9. 文件与后续

`registry.csv` 列出公式与全部口径；`skill_scores.csv` 为逐技能原分数；`ranking_diagnostics.csv`、`ranking_budgets.csv` 是全分层结果；`paired_reward_gains.csv`、`bootstrap_summary.csv`、`sign_null_summary.csv` 为配对/不确定性对照。不要从本次多个赢家中逐 seed 切换指标；如继续验证，需先冻结有限候选，再用于新的独立窗口/seed。
