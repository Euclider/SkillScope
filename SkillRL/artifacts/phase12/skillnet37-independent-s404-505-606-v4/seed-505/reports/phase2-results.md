## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21T03:59:46.319272+00:00
- Verification Status: UNVERIFIED
- Version Label: phase12_readout_seed_v4

# Phase2：seed 505，U0→U5 预测性

实际第一轮训练 batch 提供固定状态、动作 token、mask 和 advantage；U0 与 U5 在相同输入上读出。gated D 为主，−P、C 及同源幅度指标为预登记诊断；不按 gold 选方向或窗口。readout 无额外环境 rollout，但验证标签来自独立 O/P/N 续跑，两者成本不混淆。

| context_id   | phase   | score                  |   threshold |   candidates |   declines |   average_precision |   auroc_decline_vs_rest |   spearman |    kendall |   P_sign_scored_units |   P_sign_agreement_on_nonzero_point_delta |   seed |
|:-------------|:--------|:-----------------------|------------:|-------------:|-----------:|--------------------:|------------------------:|-----------:|-----------:|----------------------:|------------------------------------------:|-------:|
| all_alfworld | all     | C_upd                  |           0 |            3 |          2 |            0.833333 |                     0.5 |       -0.5 |  -0.333333 |                   nan |                                nan        |    505 |
| all_alfworld | all     | D_contribution         |           0 |            3 |          2 |            0.833333 |                     0.5 |       -0.5 |  -0.333333 |                   nan |                                nan        |    505 |
| all_alfworld | all     | D_ungated_contribution |           0 |            3 |          2 |            0.583333 |                     0   |       -1   |  -1        |                   nan |                                nan        |    505 |
| all_alfworld | all     | P_int                  |           0 |            3 |          2 |            0.583333 |                     0   |       -0.5 |  -0.333333 |                     3 |                                  0.666667 |    505 |
| all_alfworld | all     | activation_l16_norm    |           0 |            3 |          2 |            0.583333 |                     0   |       -0.5 |  -0.333333 |                   nan |                                nan        |    505 |
| all_alfworld | all     | activation_l24_norm    |           0 |            3 |          2 |            0.583333 |                     0   |       -0.5 |  -0.333333 |                   nan |                                nan        |    505 |
| all_alfworld | all     | activation_l32_norm    |           0 |            3 |          2 |            0.583333 |                     0   |       -0.5 |  -0.333333 |                   nan |                                nan        |    505 |
| all_alfworld | all     | activation_l8_norm     |           0 |            3 |          2 |            0.583333 |                     0   |       -0.5 |  -0.333333 |                   nan |                                nan        |    505 |
| all_alfworld | all     | delta_centered_norm    |           0 |            3 |          2 |            0.583333 |                     0   |       -0.5 |  -0.333333 |                   nan |                                nan        |    505 |
| all_alfworld | all     | delta_norm             |           0 |            3 |          2 |            0.583333 |                     0   |       -0.5 |  -0.333333 |                   nan |                                nan        |    505 |
| all_alfworld | all     | forward_kl_original    |           0 |            3 |          2 |            0.583333 |                     0   |       -0.5 |  -0.333333 |                   nan |                                nan        |    505 |
| all_alfworld | all     | js_original            |           0 |            3 |          2 |            0.583333 |                     0   |       -0.5 |  -0.333333 |                   nan |                                nan        |    505 |
| all_alfworld | all     | old_margin             |           0 |            3 |          2 |            0.583333 |                     0   |       -0.5 |  -0.333333 |                   nan |                                nan        |    505 |
| all_alfworld | all     | u_control_norm         |           0 |            3 |          2 |            1        |                     1   |        1   |   1        |                   nan |                                nan        |    505 |
| all_alfworld | all     | u_original_norm        |           0 |            3 |          2 |            1        |                     1   |        1   |   1        |                   nan |                                nan        |    505 |
| all_alfworld | all     | random_expected        |           0 |            3 |          2 |            0.666667 |                     0.5 |      nan   | nan        |                   nan |                                nan        |    505 |

各分数使用相同的自然支持候选池；缺支持为弃权，零下降事件时 AP/Recall 不定义。D 是单侧下降风险，不能把 D=0 解释为稳定或上升；−P 的排序关联也不等于方向分类准确率。完整 top-k/比例预算、5pp 阈值、NULL、分阶段敏感性见 CSV/window 报告。一个 seed 不支持跨 seed 泛化结论，所有阶段分层和 continuation repeats 不是独立 RL seed。

运行时限修订：用户在2026-09-20明确取消三个seed累计时间上限；以上恢复说明中的30小时为历史规则，不再约束本次接续。磁盘保护、科学设置、404→505→606顺序不变，仍记录累计实际运行时间。
