## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-24T19:26:49.662841+00:00
- Verification Status: UNVERIFIED
- Version Label: utility_precision_seed505_v1

# Phase2：seed505 全部285冻结读出与效用变化预测

主口径PLACEBO/all/0pp；方向目标`−m=−ΔM`，幅度目标`|m|=|ΔM|`。没有依据505新增gold调整公式、阈值、候选池或聚合方向。

## 方向主表

| score                              |   candidates |   declines |   increases |   average_precision |   auroc_decline_vs_increase |   spearman |
|:-----------------------------------|-------------:|-----------:|------------:|--------------------:|----------------------------:|-----------:|
| D_original::token::reward          |           19 |          3 |          14 |              0.2116 |                      0.4048 |     0.0018 |
| D_signed::token::reward            |           19 |          3 |          14 |              0.2269 |                      0.5714 |     0.1757 |
| M_delta_centered::token::magnitude |           19 |          3 |          14 |              0.5000 |                      0.6190 |    -0.0571 |
| D_real::token::reward              |           19 |          3 |          14 |              0.2298 |                      0.6190 |     0.3830 |
| D_factor::token::reward            |           19 |          3 |          14 |              0.5214 |                      0.7143 |     0.2679 |

## 幅度主表

| score                              |   candidates |   large_change_gt_5pp |   spearman |   kendall |   average_precision |   auroc_large_change_vs_rest |   top_quartile_lift |
|:-----------------------------------|-------------:|----------------------:|-----------:|----------:|--------------------:|-----------------------------:|--------------------:|
| D_original::token::reward          |           19 |                    11 |     0.0299 |    0.0178 |              0.5374 |                       0.4432 |              0.8057 |
| D_signed::token::reward            |           19 |                    11 |    -0.0413 |    0.0059 |              0.5219 |                       0.4091 |              0.7238 |
| M_delta_centered::token::magnitude |           19 |                    11 |     0.2522 |    0.1598 |              0.6932 |                       0.5114 |              1.0993 |
| D_real::token::reward              |           19 |                    11 |    -0.2838 |   -0.2190 |              0.5356 |                       0.3750 |              0.4918 |
| D_factor::token::reward            |           19 |                    11 |     0.0466 |    0.0059 |              0.6879 |                       0.5795 |              0.9155 |

全部285指标×PLACEBO/NULL×5阶段×0/5pp方向和raw/abs(score)幅度见CSV；其中`all-285-primary-direction-and-magnitude.csv`逐行合并主口径的两类目标；配对game×continuation区间及所有D的匹配无reward对照一并归档。8组重复是标签精度扩展，不是8个独立RL seed。505原两组标签在部分公式开发时已可见，这是独立RL种子的复核，不是完全盲法的前瞻确认。统计误用核查11/11：Simpson、生态谬误、Berkson、collider、基率、均值回归、幸存者、多重比较、分析路径、相关非因果、反向因果/泄漏。
