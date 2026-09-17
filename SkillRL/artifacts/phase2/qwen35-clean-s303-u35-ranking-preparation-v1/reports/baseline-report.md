# U35 新支持集覆盖与多重复效用基线

这是一份 pre-update 准备报告，不包含新 RL 窗口、ΔM 或排序预测结论；旧实验保持不变。

自然轨迹 124 条，覆盖 31 个 game；平均每条调用 3.726 种 Skill。

## 自然支持审计

| context_id   |   early |   games |   initial |   late |   middle |   natural_occurrences | placebo_ready   |   selected_anchors |   selected_games | skill_id   | supported   |
|:-------------|--------:|--------:|----------:|-------:|---------:|----------------------:|:----------------|-------------------:|-----------------:|:-----------|:------------|
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_001    | False       |
| clean        |      26 |      29 |         0 |      1 |       23 |                    84 | True            |                 50 |               29 | gen_002    | True        |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_003    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_004    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_005    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_006    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_007    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_008    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_009    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_010    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_011    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_012    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | cle_001    | False       |
| clean        |       0 |      11 |         0 |      0 |        0 |                    18 | False           |                  0 |                0 | cle_002    | False       |
| clean        |      20 |      31 |         0 |      0 |       30 |                   111 | True            |                 50 |               31 | cle_003    | True        |
| clean        |      50 |      31 |         0 |      0 |        0 |                   114 | True            |                 50 |               31 | cle_004    | True        |
| clean        |       0 |       7 |         0 |      0 |        0 |                    11 | False           |                  0 |                0 | cle_005    | False       |
| clean        |       0 |      31 |        50 |      0 |        0 |                   124 | True            |                 50 |               31 | cle_006    | True        |

## 独立 gold 基线（成功率及效用均为百分点）

| purpose   | skill_id   | context_id   | phase   | control   |   anchor_count |   game_count |   continuation_repeats |   original_success |   control_success |   semantic_or_total_margin |   ci_low |   ci_high |
|:----------|:-----------|:-------------|:--------|:----------|---------------:|-------------:|-----------------------:|-------------------:|------------------:|---------------------------:|---------:|----------:|
| gold      | cle_003    | clean        | all     | placebo   |             50 |           31 |                      4 |             53.226 |             6.855 |                     46.371 |   31.048 |    61.290 |
| gold      | cle_004    | clean        | all     | placebo   |             50 |           31 |                      4 |             33.871 |            27.419 |                      6.452 |   -1.210 |    14.919 |
| gold      | cle_006    | clean        | all     | placebo   |             50 |           31 |                      4 |             45.968 |            44.355 |                      1.613 |   -2.016 |     5.645 |
| gold      | gen_002    | clean        | all     | placebo   |             50 |           29 |                      4 |             60.776 |            57.759 |                      3.017 |    0.000 |     7.759 |

全部 evidence/gold、ORIGINAL−PLACEBO / ORIGINAL−NULL 和各阶段数值见 baseline_margins.csv。多个 continuation repeats 在 anchor/game 内配对，不作为独立 games 或独立更新窗口。

后续主分析为同一更新窗口内的 Skill top-k 排序；不以回归 MAE 作为必要成功条件。新窗口预算、完整训练状态和存储策略仍需单独冻结，当前队列不会自动训练。
