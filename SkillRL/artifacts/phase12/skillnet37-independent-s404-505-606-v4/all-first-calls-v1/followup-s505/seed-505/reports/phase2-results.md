## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21T23:25:07.321430+00:00
- Verification Status: UNVERIFIED
- Version Label: all_u0_first_calls_v1

# Phase2：seed 505，无数量门槛的预测与排序


本版本由用户在seed404旧结果已产生后授权修改覆盖规则，并在seed505仍训练时冻结统一规则。seed404属于事后覆盖扩展，不是原预登记分析的重新命名。技能内容、policy、router、C/P/D公式及其方向不变。取消的是数量门槛；缺少真实训练token的分数为NA，不填零。零advantage时按公式得到零分，同时明确记录reward_direction_observed=False，不据此断言效用稳定。

每条U0轨迹对每个skill只取首次自然调用，全部首调用锚点均评估；后续调用仅统计次数，不新增锚点。固定原动作前缀回放，从锚点起对目标skill的本次及后续自然调用施加O/P/N，其他技能不变。这些是阶段起的剩余效用，不是单次调用的孤立因果效应。

实际训练batch上的C/P/D不改为首调用读出，不修改advantage；本次去重只作用于效用锚点。

点估计依次等权平均continuation、首调用源轨迹、game；区间按game聚类重采样，沿用配对game/continuation重采样，所有arm和端点配对不拆开。单game只给点估计、不伪造跨game置信区间；阶段分层、anchors、continuation seeds都不是独立RL seeds。排名指标为描述性比较，不据多重分层选择赢家或声称跨seed显著性；D=0不等于无效用变化。

原训练恢复历史与U0导出的BF16/FP32元数据边界全部继承，未因重算报告而消失；完整历史说明见本版本plan.json绑定的archived-reports及原训练目录，原批次、OLD概率与检查点保留。

全部 25 种有自然首调用锚点的技能均计算效用；只有数学上没有实际batch读出或没有匹配效用的行无法进行预测比较，这些行仍完整列于coverage表。

| context_id   | phase   | score                  |   threshold |   candidates |   declines |   average_precision |   auroc_decline_vs_rest |     spearman |     kendall |   P_sign_scored_units |   P_sign_agreement_on_nonzero_point_delta | tie_handling_ap               | zero_event_is_undefined   |
|:-------------|:--------|:-----------------------|------------:|-------------:|-----------:|--------------------:|------------------------:|-------------:|------------:|----------------------:|------------------------------------------:|:------------------------------|:--------------------------|
| all_alfworld | all     | C_upd                  |           0 |           19 |          1 |           0.1       |                0.5      |  -0.0629696  |  -0.0499512 |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | D_contribution         |           0 |           19 |          1 |           0.0625    |                0.166667 |  -0.0521748  |  -0.0124878 |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | D_ungated_contribution |           0 |           19 |          1 |           0.0625    |                0.166667 |  -0.106149   |  -0.0624391 |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | P_int                  |           0 |           19 |          1 |           0.0769231 |                0.333333 |  -0.0215896  |  -0.0124878 |                    11 |                                  0.181818 | threshold-grouped standard AP | False                     |
| all_alfworld | all     | activation_l16_norm    |           0 |           19 |          1 |           0.0769231 |                0.333333 |  -0.179913   |  -0.137366  |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | activation_l24_norm    |           0 |           19 |          1 |           0.2       |                0.777778 |  -0.00359826 |  -0.0249756 |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | activation_l32_norm    |           0 |           19 |          1 |           0.142857  |                0.666667 |  -0.0179913  |  -0.0124878 |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | activation_l8_norm     |           0 |           19 |          1 |           0.0625    |                0.166667 |  -0.298656   |  -0.212293  |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | delta_centered_norm    |           0 |           19 |          1 |           0.166667  |                0.722222 |   0.199704   |   0.137366  |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | delta_norm             |           0 |           19 |          1 |           0.333333  |                0.888889 |   0.115144   |   0.0624391 |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | forward_kl_original    |           0 |           19 |          1 |           0.0909091 |                0.444444 |  -0.327442   |  -0.237268  |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | js_original            |           0 |           19 |          1 |           0.1       |                0.5      |  -0.221293   |  -0.137366  |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | old_margin             |           0 |           19 |          1 |           0.111111  |                0.555556 |  -0.274112   |  -0.193981  |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | u_control_norm         |           0 |           19 |          1 |           0.2       |                0.777778 |   0.584718   |   0.449561  |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | u_original_norm        |           0 |           19 |          1 |           0.333333  |                0.888889 |   0.456979   |   0.374634  |                   nan |                                nan        | threshold-grouped standard AP | False                     |
| all_alfworld | all     | random_expected        |           0 |           19 |          1 |           0.0526316 |                0.5      | nan          | nan         |                   nan |                                nan        | threshold-grouped standard AP | False                     |

分阶段、NULL对照、0/5pp阈值、同池top-k、低支持数量和NA原因见同目录CSV及版本化window_metrics。旧报告归档，不删除旧轨迹/评估。
