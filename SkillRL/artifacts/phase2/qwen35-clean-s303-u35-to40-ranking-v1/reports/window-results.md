# Extended Phase2：按冻结窗口汇总

路径：`seed303-continued-u35-to40-ranking-v1`；更新时间：2026-09-14T00:44:17.495714+00:00。

窗口方向为起始 batch 的 reward direction 在窗口端点 interaction 上的投影，不是逐 update D 的相加。all 与各阶段单独分析，不将重复分层当成独立样本。

## 语义效用变化（pp）

|   start_update |   global_update | window_role   | skill_id   | context_id   |   anchor_count |   continuation_repeats |   delta_utility |   ci_low |   ci_high |
|---------------:|----------------:|:--------------|:-----------|:-------------|---------------:|-----------------------:|----------------:|---------:|----------:|
|             35 |              40 | test          | cle_003    | clean        |             50 |                      4 |         15.7258 |   2.4194 |   29.0323 |
|             35 |              40 | test          | cle_004    | clean        |             50 |                      4 |         -6.4516 | -14.9194 |    1.2097 |
|             35 |              40 | test          | cle_006    | clean        |             50 |                      4 |         -1.6129 |  -5.6452 |    2.0161 |
|             35 |              40 | test          | gen_002    | clean        |             50 |                      4 |         -3.0172 |  -7.7586 |    0.0000 |

## 主分析：同一窗口内的 Skill top-k 排序

分数不依赖拟合回归器；正负方向和预算先于目标 gold 固定。所有方法使用同一支持候选池；并列按随机打破的期望计分。下表下降是 gold 点估计，不代表每个 Skill 的下降已统计确证。

|   start_update |   global_update | context_id   | score                  |   n_candidates |   events |   k |   precision_at_k |   recall_at_k |   captured_change_mass |   spearman |
|---------------:|----------------:|:-------------|:-----------------------|---------------:|---------:|----:|-----------------:|--------------:|-----------------------:|-----------:|
|             35 |              40 | clean        | D_contribution         |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | D_contribution         |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | D_ungated_contribution |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | D_ungated_contribution |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | P_int                  |              3 |        2 |   1 |           1.0000 |        0.5000 |                 0.6814 |     1.0000 |
|             35 |              40 | clean        | P_int                  |              3 |        2 |   2 |           1.0000 |        1.0000 |                 1.0000 |     1.0000 |
|             35 |              40 | clean        | activation_l16_norm    |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | activation_l16_norm    |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | activation_l24_norm    |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | activation_l24_norm    |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | activation_l32_norm    |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | activation_l32_norm    |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | activation_l8_norm     |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | activation_l8_norm     |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | delta_centered_norm    |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | delta_centered_norm    |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | delta_norm             |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | delta_norm             |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | forward_kl_original    |              3 |        2 |   1 |           1.0000 |        0.5000 |                 0.6814 |     1.0000 |
|             35 |              40 | clean        | forward_kl_original    |              3 |        2 |   2 |           1.0000 |        1.0000 |                 1.0000 |     1.0000 |
|             35 |              40 | clean        | js_original            |              3 |        2 |   1 |           1.0000 |        0.5000 |                 0.6814 |     1.0000 |
|             35 |              40 | clean        | js_original            |              3 |        2 |   2 |           1.0000 |        1.0000 |                 1.0000 |     1.0000 |
|             35 |              40 | clean        | old_margin             |              3 |        2 |   1 |           1.0000 |        0.5000 |                 0.3186 |     0.5000 |
|             35 |              40 | clean        | old_margin             |              3 |        2 |   2 |           1.0000 |        1.0000 |                 1.0000 |     0.5000 |
|             35 |              40 | clean        | u_control_norm         |              3 |        2 |   1 |           1.0000 |        0.5000 |                 0.6814 |     1.0000 |
|             35 |              40 | clean        | u_control_norm         |              3 |        2 |   2 |           1.0000 |        1.0000 |                 1.0000 |     1.0000 |
|             35 |              40 | clean        | u_original_norm        |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | u_original_norm        |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | random_expected        |              3 |        2 |   1 |           0.6667 |        0.3333 |                 0.3333 |   nan      |
|             35 |              40 | clean        | random_expected        |              3 |        2 |   2 |           0.6667 |        0.6667 |                 0.6667 |   nan      |

下降事件为零时 Recall/下降量覆盖率未定义。完整 CSV 同时包含 >5 pp 阈值、|ΔM| 审计及各阶段；这些不是额外独立窗口。首个单窗口排序只作描述，不声称跨窗口/seed 泛化或排序指标显著性。

## 辅助：数值预测（非主要成功标准）

| phase   | model                |   units |   windows |   mae_pp |
|:--------|:---------------------|--------:|----------:|---------:|
| all     | predicted_delta_zero |       3 |         1 |   8.3982 |
| early   | predicted_delta_zero |       1 |         1 |   6.4516 |
| late    | predicted_delta_zero |       1 |         1 |   0.0000 |
| middle  | predicted_delta_zero |       2 |         1 |  11.0628 |
