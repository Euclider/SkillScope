## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21T03:59:59.261211+00:00
- Verification Status: UNVERIFIED
- Version Label: phase12_independent_cohort_v4

# Phase1–2 独立窗口汇总

预登记 seeds：[404, 505, 606]；完成：[404, 505]；未完成：[606]。

顺序由登记决定，不按 outcome 筛 seed。所有 seed 从共同 B0 独立训练五轮，技能内容冻结。评估 seeds 为共同随机数，因此不将重复的 U0/评估样本当成额外独立重复。跨 seed 比较先看各 seed 的同池排名和效应；不把所有 skill×seed 或阶段行视为独立样本。

|   seed | score                  |   candidates |   declines |   average_precision |   spearman |
|-------:|:-----------------------|-------------:|-----------:|--------------------:|-----------:|
|    404 | C_upd                  |            3 |          1 |            1        |        0.5 |
|    404 | D_contribution         |            3 |          1 |            0.333333 |       -1   |
|    404 | D_ungated_contribution |            3 |          1 |            0.333333 |       -1   |
|    404 | P_int                  |            3 |          1 |            0.333333 |       -0.5 |
|    404 | activation_l16_norm    |            3 |          1 |            1        |        0.5 |
|    404 | activation_l24_norm    |            3 |          1 |            1        |        0.5 |
|    404 | activation_l32_norm    |            3 |          1 |            1        |        0.5 |
|    404 | activation_l8_norm     |            3 |          1 |            1        |        0.5 |
|    404 | delta_centered_norm    |            3 |          1 |            0.5      |       -0.5 |
|    404 | delta_norm             |            3 |          1 |            0.5      |       -0.5 |
|    404 | forward_kl_original    |            3 |          1 |            1        |        1   |
|    404 | js_original            |            3 |          1 |            1        |        0.5 |
|    404 | old_margin             |            3 |          1 |            0.5      |       -0.5 |
|    404 | u_control_norm         |            3 |          1 |            0.333333 |       -0.5 |
|    404 | u_original_norm        |            3 |          1 |            0.333333 |       -1   |
|    404 | random_expected        |            3 |          1 |            0.333333 |      nan   |
|    505 | C_upd                  |            3 |          2 |            0.833333 |       -0.5 |
|    505 | D_contribution         |            3 |          2 |            0.833333 |       -0.5 |
|    505 | D_ungated_contribution |            3 |          2 |            0.583333 |       -1   |
|    505 | P_int                  |            3 |          2 |            0.583333 |       -0.5 |
|    505 | activation_l16_norm    |            3 |          2 |            0.583333 |       -0.5 |
|    505 | activation_l24_norm    |            3 |          2 |            0.583333 |       -0.5 |
|    505 | activation_l32_norm    |            3 |          2 |            0.583333 |       -0.5 |
|    505 | activation_l8_norm     |            3 |          2 |            0.583333 |       -0.5 |
|    505 | delta_centered_norm    |            3 |          2 |            0.583333 |       -0.5 |
|    505 | delta_norm             |            3 |          2 |            0.583333 |       -0.5 |
|    505 | forward_kl_original    |            3 |          2 |            0.583333 |       -0.5 |
|    505 | js_original            |            3 |          2 |            0.583333 |       -0.5 |
|    505 | old_margin             |            3 |          2 |            0.583333 |       -0.5 |
|    505 | u_control_norm         |            3 |          2 |            1        |        1   |
|    505 | u_original_norm        |            3 |          2 |            1        |        1   |
|    505 | random_expected        |            3 |          2 |            0.666667 |      nan   |

本文件为运行结果汇总，不预先断言 idea 成立；预算停止、支持不足或无下降事件必须一并报告。
