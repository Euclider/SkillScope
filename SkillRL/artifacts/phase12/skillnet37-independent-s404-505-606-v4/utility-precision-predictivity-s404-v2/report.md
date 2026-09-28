## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Verification Status: ANALYZED
- Version Label: utility_precision_predictivity_seed404_v1

# Seed404：更新效用标签下的方向与幅度预测

原始精度扩展目录：`/mnt/workspace/users/wangyifan/skill-RL/SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s404-v1`；本分析目录：`/mnt/workspace/users/wangyifan/skill-RL/SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-predictivity-s404-v2`。冻结37技能库、U0→U5、404个首次调用锚点、每个arm/endpoint 8组配对续跑；不重跑训练、模型前向或环境轨迹。285列冻结readout全部进入同一分析，方向表与原精度报告逐项复算一致（5700行）。

目标严格区分：`m=ΔM=M_U5−M_U0`，下降排序是`−m`；变化幅度是`|m|`，不是`|M_U0|`。`raw`表示原始读出值，`absolute_readout`表示`|读出值|`；后者为事后诊断变换，不是独立冻结方法。预测分数越大，预期变化越大。大变化事件固定定义为`|m|>0.05`，top-k为共同池前25%（18技能时k=5），ties等权；升降方向不混入幅度标签。

## 主口径：placebo、all、18技能

| score                              | transform        |   candidates |   large_change_gt_5pp |   spearman_decline |   spearman_magnitude |   kendall |   auroc_large_change_vs_rest |   average_precision_magnitude |   top_quartile_lift |
|:-----------------------------------|:-----------------|-------------:|----------------------:|-------------------:|---------------------:|----------:|-----------------------------:|------------------------------:|--------------------:|
| D_original::token::reward          | raw              |           18 |                     5 |             0.0031 |               0.5571 |    0.4601 |                       0.7692 |                        0.5833 |              1.4897 |
| D_signed::token::reward            | raw              |           18 |                     5 |            -0.0809 |               0.0290 |    0.0200 |                       0.4923 |                        0.3416 |              1.2019 |
| C_centered::token::reward          | raw              |           18 |                     5 |            -0.0062 |               0.2759 |    0.2334 |                       0.7077 |                        0.4525 |              1.3678 |
| M_delta_raw::token::magnitude      | raw              |           18 |                     5 |             0.1629 |              -0.3496 |   -0.2734 |                       0.2769 |                        0.2321 |              0.4603 |
| M_delta_centered::token::magnitude | raw              |           18 |                     5 |             0.3537 |              -0.4471 |   -0.2867 |                       0.2154 |                        0.2150 |              0.4969 |
| D_real::token::reward              | raw              |           18 |                     5 |             0.0363 |              -0.3662 |   -0.2601 |                       0.4308 |                        0.3212 |              0.5992 |
| D_factor::token::reward            | raw              |           18 |                     5 |            -0.0643 |              -0.1286 |   -0.1000 |                       0.4769 |                        0.2898 |              0.5078 |
| D_original::token::reward          | absolute_readout |           18 |                     5 |             0.0031 |               0.5571 |    0.4601 |                       0.7692 |                        0.5833 |              1.4897 |
| D_signed::token::reward            | absolute_readout |           18 |                     5 |            -0.0809 |               0.0425 |    0.0200 |                       0.4154 |                        0.3350 |              1.6766 |
| C_centered::token::reward          | absolute_readout |           18 |                     5 |            -0.0062 |              -0.2542 |   -0.1800 |                       0.4154 |                        0.2690 |              0.2702 |
| M_delta_raw::token::magnitude      | absolute_readout |           18 |                     5 |             0.1629 |              -0.3496 |   -0.2734 |                       0.2769 |                        0.2321 |              0.4603 |
| M_delta_centered::token::magnitude | absolute_readout |           18 |                     5 |             0.3537 |              -0.4471 |   -0.2867 |                       0.2154 |                        0.2150 |              0.4969 |
| D_real::token::reward              | absolute_readout |           18 |                     5 |             0.0363 |               0.3817 |    0.2867 |                       0.6000 |                        0.4735 |              1.1903 |
| D_factor::token::reward            | absolute_readout |           18 |                     5 |            -0.0643 |              -0.5612 |   -0.4067 |                       0.2923 |                        0.2331 |              0.2194 |

其中下降Spearman对`−m`，幅度Spearman/Kendall对`|m|`；AP/AUROC对应`|m|>5pp`，top-k lift是选中技能平均`|m|`除以全池平均`|m|`。方向任务的全部AP/AUROC/相关系数见`direction-recomputed.csv`，幅度任务的全部raw/absolute、placebo/null和阶段结果见`magnitude-points.csv`。

## 幅度相关的配对标签不确定性

| score                              | transform        |   point |     low |   high |   valid_draws |
|:-----------------------------------|:-----------------|--------:|--------:|-------:|--------------:|
| D_original::token::reward          | raw              |  0.5571 | -0.2066 | 0.5906 |          1717 |
| D_signed::token::reward            | raw              |  0.0290 | -0.2801 | 0.4335 |          1717 |
| C_centered::token::reward          | raw              |  0.2759 | -0.2347 | 0.5479 |          1717 |
| M_delta_raw::token::magnitude      | raw              | -0.3496 | -0.4716 | 0.2338 |          1717 |
| M_delta_centered::token::magnitude | raw              | -0.4471 | -0.5926 | 0.1754 |          1717 |
| D_real::token::reward              | raw              | -0.3662 | -0.6041 | 0.2420 |          1717 |
| D_factor::token::reward            | raw              | -0.1286 | -0.3861 | 0.3790 |          1717 |
| D_original::token::reward          | absolute_readout |  0.5571 | -0.2066 | 0.5906 |          1717 |
| D_signed::token::reward            | absolute_readout |  0.0425 | -0.4560 | 0.3441 |          1717 |
| C_centered::token::reward          | absolute_readout | -0.2542 | -0.5304 | 0.2390 |          1717 |
| M_delta_raw::token::magnitude      | absolute_readout | -0.3496 | -0.4716 | 0.2338 |          1717 |
| M_delta_centered::token::magnitude | absolute_readout | -0.4471 | -0.5926 | 0.1754 |          1717 |
| D_real::token::reward              | absolute_readout |  0.3817 | -0.1024 | 0.6272 |          1717 |
| D_factor::token::reward            | absolute_readout | -0.5612 | -0.6559 | 0.1284 |          1717 |

区间为复用原2000组game×continuation配对抽样的2.5%–97.5%描述分位数；仅完整保留18技能的1717组进入主口径。全部幅度指标区间见`magnitude-bootstrap.csv`。这些区间只覆盖固定读出下的gold标签抽样，不覆盖RL种子、训练批次、任务域或285种公式搜索。

## 匹配对照与重复数敏感性

| score                     | reference                               |   point_difference |     low |   high |   valid_draws |
|:--------------------------|:----------------------------------------|-------------------:|--------:|-------:|--------------:|
| D_original::token::reward | D_original::token::unsigned             |             0.4025 | -0.0633 | 0.5027 |          1717 |
| D_original::token::reward | M_delta_centered::token::magnitude      |             1.0042 | -0.0726 | 0.8487 |          1717 |
| D_real::token::reward     | D_real_absA::token::reward_sign_removed |            -0.9492 | -1.0879 | 0.1680 |          1717 |

差值顺序为左侧分数减去reference；每组都在相同技能、相同gold抽样上计算，区间未经285公式搜索校正。

| panel       | score                              |   large_change_gt_5pp |   spearman |
|:------------|:-----------------------------------|----------------------:|-----------:|
| gold2       | D_original::token::reward          |                     8 |     0.6205 |
| gold2       | M_delta_centered::token::magnitude |                     8 |     0.0852 |
| gold2       | D_original::token::unsigned        |                     8 |     0.4386 |
| gold4       | D_original::token::reward          |                     5 |     0.3760 |
| gold4       | M_delta_centered::token::magnitude |                     5 |    -0.4401 |
| gold4       | D_original::token::unsigned        |                     5 |    -0.0579 |
| gold8       | D_original::token::reward          |                     5 |     0.5571 |
| gold8       | M_delta_centered::token::magnitude |                     5 |    -0.4471 |
| gold8       | D_original::token::unsigned        |                     5 |     0.1546 |
| added6_only | D_original::token::reward          |                     5 |     0.2998 |
| added6_only | M_delta_centered::token::magnitude |                     5 |    -0.4098 |
| added6_only | D_original::token::unsigned        |                     5 |     0.1566 |

`gold2/gold4/gold8/added6_only`共用冻结读出，仅续跑标签精度变化；重复组不是独立RL seed，不可用最有利的子集替换8组主结果。全部对照和精度表见`paired-magnitude-contrasts.csv`与`precision-curve-magnitude.csv`。

## 解释边界

这次是已经观察过seed404/505旧标签后的事后分析，不能从285列中挑最大相关值作确认性结论。研究单位是同一U0→U5窗口内的技能，不是token或独立训练seed；18技能共同池不能代表完整37技能库。稀少anchor使单技能标签区间仍宽；大变化事件的基率和阈值敏感，不能把相关性当作编辑收益或逐步因果归因。统计误用检查11/11：Simpson、生态谬误、Berkson/支持集、collider、基率、均值回归、幸存者、多重比较、分析路径、相关非因果、反向因果/泄漏。读出分数先于新增效用续跑冻结，但公式族与原标签已有交互，属于探索性证据。
