# Seed404：中心化幅度 × reward 定向因子验证

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Verification Status: ANALYZED（单seed事后探索）
- Version Label: factorized_centered_reward_v1

## 1. 冻结公式和范围

`b=A*delta(a); B=Agg(||Hdelta||); R=Agg(b)/(Agg(abs(b))+1e-12); D_factor=-B*R`。`-R`单独列出；A1在全部原token上置A=1，不继承零A屏蔽。主口径PLACEBO/all/token/0pp。没有新训练、模型前向、效用rollout或API；505/606未启动；旧报告不覆盖。

R只是正负证据净比例，不是置信度；B>0时新D符号与D_action_adv相同。重排不等于改善方向判别。范数预测的是策略响应变化量，不能直接认定已预测效用变化绝对值。

## 2. 主口径结果

| name             | aggregation   |   candidates |   declines |   increases |   auroc_decline_vs_increase |   spearman |   average_precision |   sign_accuracy_called |   balanced_accuracy_abstention_wrong |
|:-----------------|:--------------|-------------:|-----------:|------------:|----------------------------:|-----------:|--------------------:|-----------------------:|-------------------------------------:|
| D_original       | token         |           18 |          5 |           7 |                    0.428571 |  -0.147161 |            0.509650 |             nan        |                           nan        |
| D_signed         | token         |           18 |          5 |           7 |                    0.457143 |  -0.064120 |            0.423810 |               0.416667 |                             0.414286 |
| D_action_adv     | token         |           18 |          5 |           7 |                    0.542857 |   0.249123 |            0.402197 |               0.666667 |                             0.600000 |
| C_centered       | token         |           18 |          5 |           7 |                    0.657143 |   0.023125 |            0.688095 |             nan        |                           nan        |
| M_delta_centered | token         |           18 |          5 |           7 |                    0.742857 |   0.348982 |            0.613651 |             nan        |                           nan        |
| D_real           | token         |           18 |          5 |           7 |                    0.485714 |   0.131394 |            0.228008 |               0.583333 |                             0.500000 |
| D_factor         | token         |           18 |          5 |           7 |                    0.600000 |   0.101962 |            0.294949 |               0.666667 |                             0.600000 |
| D_orientation    | token         |           18 |          5 |           7 |                    0.600000 |   0.097757 |            0.294949 |               0.666667 |                             0.600000 |
| D_factor_A1      | token         |           18 |          5 |           7 |                    0.428571 |  -0.026279 |            0.435014 |             nan        |                           nan        |
| D_orientation_A1 | token         |           18 |          5 |           7 |                    0.342857 |  -0.110371 |            0.396125 |             nan        |                           nan        |

无符号基线不解释为正负方向；方向AUC仅下降vs上升，AP包含全部固定技能池。

## 3. 主口径配对差值与95%描述区间

| reference                          | metric                             |   point_difference |       low |     high |   valid_draws |
|:-----------------------------------|:-----------------------------------|-------------------:|----------:|---------:|--------------:|
| M_delta_centered::token::magnitude | auroc_decline_vs_increase          |          -0.142857 | -0.700000 | 0.439236 |          1671 |
| M_delta_centered::token::magnitude | spearman                           |          -0.247020 | -0.842129 | 0.454834 |          1674 |
| D_action_adv::token::reward        | auroc_decline_vs_increase          |           0.057143 | -0.333333 | 0.277778 |          1671 |
| D_action_adv::token::reward        | spearman                           |          -0.147161 | -0.392500 | 0.215136 |          1674 |
| D_action_adv::token::reward        | balanced_accuracy_abstention_wrong |           0.000000 |  0.000000 | 0.000000 |          1671 |
| D_factor_A1::token::reward_free    | auroc_decline_vs_increase          |           0.171429 | -0.566667 | 0.583333 |          1671 |
| D_factor_A1::token::reward_free    | spearman                           |           0.128240 | -0.550837 | 0.616879 |          1674 |
| D_orientation::token::reward       | auroc_decline_vs_increase          |           0.000000 | -0.037778 | 0.047619 |          1671 |
| D_orientation::token::reward       | spearman                           |           0.004205 | -0.022189 | 0.027330 |          1674 |
| D_orientation::token::reward       | balanced_accuracy_abstention_wrong |           0.000000 |  0.000000 | 0.000000 |          1671 |
| D_original::token::reward          | auroc_decline_vs_increase          |           0.171429 | -0.416667 | 0.600000 |          1671 |
| D_original::token::reward          | spearman                           |           0.249123 | -0.319465 | 0.679579 |          1674 |
| D_signed::token::reward            | auroc_decline_vs_increase          |           0.142857 | -0.428571 | 0.650000 |          1671 |
| D_signed::token::reward            | spearman                           |           0.166082 | -0.389661 | 0.701723 |          1674 |
| D_signed::token::reward            | balanced_accuracy_abstention_wrong |           0.185714 | -0.300000 | 0.529167 |          1671 |
| C_centered::token::reward          | auroc_decline_vs_increase          |          -0.057143 | -0.545455 | 0.615434 |          1671 |
| C_centered::token::reward          | spearman                           |           0.078836 | -0.484992 | 0.646821 |          1674 |
| D_policy_action_adv::token::reward | auroc_decline_vs_increase          |           0.028571 | -0.500000 | 0.500000 |          1671 |
| D_policy_action_adv::token::reward | spearman                           |          -0.106166 | -0.530552 | 0.397753 |          1674 |
| D_policy_action_adv::token::reward | balanced_accuracy_abstention_wrong |           0.100000 | -0.196429 | 0.267857 |          1671 |
| D_real::token::reward              | auroc_decline_vs_increase          |           0.114286 | -0.450000 | 0.500000 |          1671 |
| D_real::token::reward              | spearman                           |          -0.029432 | -0.471446 | 0.421333 |          1674 |
| D_real::token::reward              | balanced_accuracy_abstention_wrong |           0.100000 | -0.166667 | 0.222222 |          1671 |

复用2000组game×continuation抽样；完整/缺技能draw数见bootstrap_summary.csv。部分指标还因无类别而NA。仅涵盖固定训练读出下的gold-label不确定性，不包含训练seed/读出抽样方差、不校正历史探索。

## 4. 轨迹块reward符号对照

| score                        | metric                             |   observed |   reference_fraction_ge_observed |   three_aggregation_max_fraction_ge_observed |
|:-----------------------------|:-----------------------------------|-----------:|---------------------------------:|---------------------------------------------:|
| D_factor::token::reward      | auroc_decline_vs_increase          |   0.600000 |                         0.316406 |                                     0.445312 |
| D_orientation::token::reward | auroc_decline_vs_increase          |   0.600000 |                         0.310547 |                                     0.453125 |
| D_factor::token::reward      | spearman                           |   0.101962 |                         0.345703 |                                     0.441406 |
| D_orientation::token::reward | spearman                           |   0.097757 |                         0.343750 |                                     0.460938 |
| D_factor::token::reward      | average_precision                  |   0.294949 |                         0.697266 |                                     0.773438 |
| D_orientation::token::reward | average_precision                  |   0.294949 |                         0.695312 |                                     0.765625 |
| D_factor::token::reward      | balanced_accuracy_abstention_wrong |   0.600000 |                         0.218750 |                                     0.363281 |
| D_orientation::token::reward | balanced_accuracy_abstention_wrong |   0.600000 |                         0.218750 |                                     0.363281 |

512组共同轨迹块翻号，保留A幅度与局内A变化。参考比例不是校准p值，不校正此前多公式搜索。

## 5. 完整敏感性（不择优替代主分析）

| control   | name             | aggregation   |   candidates |   declines |   increases |   auroc_decline_vs_increase |   spearman |   average_precision |   sign_accuracy_called |   balanced_accuracy_abstention_wrong |
|:----------|:-----------------|:--------------|-------------:|-----------:|------------:|----------------------------:|-----------:|--------------------:|-----------------------:|-------------------------------------:|
| placebo   | D_original       | token         |           18 |          5 |           7 |                    0.428571 |  -0.147161 |            0.509650 |             nan        |                           nan        |
| placebo   | D_signed         | token         |           18 |          5 |           7 |                    0.457143 |  -0.064120 |            0.423810 |               0.416667 |                             0.414286 |
| placebo   | D_action_adv     | token         |           18 |          5 |           7 |                    0.542857 |   0.249123 |            0.402197 |               0.666667 |                             0.600000 |
| placebo   | C_centered       | token         |           18 |          5 |           7 |                    0.657143 |   0.023125 |            0.688095 |             nan        |                           nan        |
| placebo   | M_delta_centered | token         |           18 |          5 |           7 |                    0.742857 |   0.348982 |            0.613651 |             nan        |                           nan        |
| placebo   | D_original       | decision      |           18 |          5 |           7 |                    0.457143 |  -0.119831 |            0.542984 |             nan        |                           nan        |
| placebo   | D_signed         | decision      |           18 |          5 |           7 |                    0.457143 |  -0.048353 |            0.423810 |               0.416667 |                             0.414286 |
| placebo   | D_action_adv     | decision      |           18 |          5 |           7 |                    0.485714 |   0.171337 |            0.394674 |               0.666667 |                             0.600000 |
| placebo   | C_centered       | decision      |           18 |          5 |           7 |                    0.685714 |   0.042046 |            0.693590 |             nan        |                           nan        |
| placebo   | M_delta_centered | decision      |           18 |          5 |           7 |                    0.771429 |   0.406795 |            0.642222 |             nan        |                           nan        |
| placebo   | D_original       | game          |           18 |          5 |           7 |                    0.485714 |  -0.016818 |            0.490227 |             nan        |                           nan        |
| placebo   | D_signed         | game          |           18 |          5 |           7 |                    0.571429 |   0.176593 |            0.333824 |               0.666667 |                             0.628571 |
| placebo   | D_action_adv     | game          |           18 |          5 |           7 |                    0.600000 |   0.071478 |            0.332157 |               0.583333 |                             0.528571 |
| placebo   | C_centered       | game          |           18 |          5 |           7 |                    0.542857 |   0.104064 |            0.584038 |             nan        |                           nan        |
| placebo   | M_delta_centered | game          |           18 |          5 |           7 |                    0.800000 |   0.362647 |            0.724762 |             nan        |                           nan        |
| placebo   | D_real           | token         |           18 |          5 |           7 |                    0.485714 |   0.131394 |            0.228008 |               0.583333 |                             0.500000 |
| placebo   | D_real           | decision      |           18 |          5 |           7 |                    0.514286 |   0.146110 |            0.231817 |               0.583333 |                             0.500000 |
| placebo   | D_real           | game          |           18 |          5 |           7 |                    0.628571 |   0.174491 |            0.325362 |               0.500000 |                             0.428571 |
| placebo   | D_factor         | token         |           18 |          5 |           7 |                    0.600000 |   0.101962 |            0.294949 |               0.666667 |                             0.600000 |
| placebo   | D_orientation    | token         |           18 |          5 |           7 |                    0.600000 |   0.097757 |            0.294949 |               0.666667 |                             0.600000 |
| placebo   | D_factor_A1      | token         |           18 |          5 |           7 |                    0.428571 |  -0.026279 |            0.435014 |             nan        |                           nan        |
| placebo   | D_orientation_A1 | token         |           18 |          5 |           7 |                    0.342857 |  -0.110371 |            0.396125 |             nan        |                           nan        |
| placebo   | D_factor         | decision      |           18 |          5 |           7 |                    0.542857 |   0.053609 |            0.282222 |               0.666667 |                             0.600000 |
| placebo   | D_orientation    | decision      |           18 |          5 |           7 |                    0.542857 |   0.062018 |            0.282222 |               0.666667 |                             0.600000 |
| placebo   | D_factor_A1      | decision      |           18 |          5 |           7 |                    0.457143 |   0.031534 |            0.445966 |             nan        |                           nan        |
| placebo   | D_orientation_A1 | decision      |           18 |          5 |           7 |                    0.400000 |  -0.046251 |            0.415454 |             nan        |                           nan        |
| placebo   | D_factor         | game          |           18 |          5 |           7 |                    0.571429 |  -0.027330 |            0.349475 |               0.583333 |                             0.528571 |
| placebo   | D_orientation    | game          |           18 |          5 |           7 |                    0.542857 |  -0.078836 |            0.332808 |               0.583333 |                             0.528571 |
| placebo   | D_factor_A1      | game          |           18 |          5 |           7 |                    0.371429 |  -0.081990 |            0.409643 |             nan        |                           nan        |
| placebo   | D_orientation_A1 | game          |           18 |          5 |           7 |                    0.314286 |  -0.119831 |            0.402197 |             nan        |                           nan        |
| null      | D_original       | token         |           18 |          5 |           7 |                    0.200000 |  -0.435176 |            0.406726 |             nan        |                           nan        |
| null      | D_signed         | token         |           18 |          5 |           7 |                    0.514286 |   0.063069 |            0.360000 |               0.416667 |                             0.414286 |
| null      | D_action_adv     | token         |           18 |          5 |           7 |                    0.628571 |   0.190258 |            0.330476 |               0.666667 |                             0.600000 |
| null      | C_centered       | token         |           18 |          5 |           7 |                    0.485714 |  -0.153468 |            0.574188 |             nan        |                           nan        |
| null      | M_delta_centered | token         |           18 |          5 |           7 |                    0.628571 |   0.087245 |            0.516667 |             nan        |                           nan        |
| null      | D_original       | decision      |           18 |          5 |           7 |                    0.200000 |  -0.430971 |            0.402330 |             nan        |                           nan        |
| null      | D_signed         | decision      |           18 |          5 |           7 |                    0.514286 |   0.045199 |            0.360000 |               0.416667 |                             0.414286 |
| null      | D_action_adv     | decision      |           18 |          5 |           7 |                    0.542857 |   0.179747 |            0.305490 |               0.666667 |                             0.600000 |
| null      | C_centered       | decision      |           18 |          5 |           7 |                    0.514286 |  -0.120882 |            0.583712 |             nan        |                           nan        |
| null      | M_delta_centered | decision      |           18 |          5 |           7 |                    0.628571 |   0.095655 |            0.602976 |             nan        |                           nan        |
| null      | D_original       | game          |           18 |          5 |           7 |                    0.257143 |  -0.340573 |            0.412120 |             nan        |                           nan        |
| null      | D_signed         | game          |           18 |          5 |           7 |                    0.628571 |   0.244918 |            0.413997 |               0.583333 |                             0.557143 |
| null      | D_action_adv     | game          |           18 |          5 |           7 |                    0.485714 |   0.100910 |            0.253845 |               0.500000 |                             0.457143 |
| null      | C_centered       | game          |           18 |          5 |           7 |                    0.542857 |  -0.007358 |            0.604167 |             nan        |                           nan        |
| null      | M_delta_centered | game          |           18 |          5 |           7 |                    0.628571 |   0.098808 |            0.607372 |             nan        |                           nan        |
| null      | D_real           | token         |           18 |          5 |           7 |                    0.771429 |   0.385772 |            0.406061 |               0.750000 |                             0.700000 |
| null      | D_real           | decision      |           18 |          5 |           7 |                    0.714286 |   0.350033 |            0.392157 |               0.750000 |                             0.700000 |
| null      | D_real           | game          |           18 |          5 |           7 |                    0.800000 |   0.495092 |            0.391204 |               0.666667 |                             0.600000 |
| null      | D_factor         | token         |           18 |          5 |           7 |                    0.400000 |  -0.109320 |            0.276424 |               0.666667 |                             0.600000 |
| null      | D_orientation    | token         |           18 |          5 |           7 |                    0.400000 |  -0.099859 |            0.280969 |               0.666667 |                             0.600000 |
| null      | D_factor_A1      | token         |           18 |          5 |           7 |                    0.628571 |   0.129291 |            0.564884 |             nan        |                           nan        |
| null      | D_orientation_A1 | token         |           18 |          5 |           7 |                    0.657143 |   0.173440 |            0.464884 |             nan        |                           nan        |
| null      | D_factor         | decision      |           18 |          5 |           7 |                    0.371429 |  -0.081990 |            0.268271 |               0.666667 |                             0.600000 |
| null      | D_orientation    | decision      |           18 |          5 |           7 |                    0.428571 |  -0.029432 |            0.280969 |               0.666667 |                             0.600000 |
| null      | D_factor_A1      | decision      |           18 |          5 |           7 |                    0.571429 |   0.069376 |            0.538824 |             nan        |                           nan        |
| null      | D_orientation_A1 | decision      |           18 |          5 |           7 |                    0.628571 |   0.159775 |            0.457871 |             nan        |                           nan        |
| null      | D_factor         | game          |           18 |          5 |           7 |                    0.457143 |  -0.006307 |            0.264303 |               0.500000 |                             0.457143 |
| null      | D_orientation    | game          |           18 |          5 |           7 |                    0.457143 |  -0.004205 |            0.264303 |               0.500000 |                             0.457143 |
| null      | D_factor_A1      | game          |           18 |          5 |           7 |                    0.628571 |   0.149263 |            0.557871 |             nan        |                           nan        |
| null      | D_orientation_A1 | game          |           18 |          5 |           7 |                    0.571429 |   0.122985 |            0.439884 |             nan        |                           nan        |

阶段/5pp结果、top-k、所有285列见ranking_diagnostics.csv与ranking_budgets.csv。

## 6. 幅度与效用变化大小的核对

| control   | aggregation   |   skills |   rho_B_vs_absolute_delta_utility |   rho_B_vs_signed_decline |
|:----------|:--------------|---------:|----------------------------------:|--------------------------:|
| placebo   | token         |       18 |                          0.085189 |                  0.348982 |
| placebo   | decision      |       18 |                          0.112533 |                  0.406795 |
| placebo   | game          |       18 |                          0.218756 |                  0.362647 |
| null      | token         |       18 |                          0.328134 |                  0.087245 |
| null      | decision      |       18 |                          0.257670 |                  0.095655 |
| null      | game          |       18 |                          0.272394 |                  0.098808 |

## 7. 工程验收及证据边界

全部155638个token×control行、5511个决策复用；新因子420个汇总单元独立fsum核验。符号恒等通过；epsilon弃权差异0个单元。本轮CPU统计耗时约44.9秒（不含开发测试）。

273个旧指标列和5460条旧分层指标保留并复现；全部新旧指标另用sklearn/scipy、方向混淆手算核验。这是数值实现交叉核验，不是独立RL重复实验或跨seed可复现性证明；Verification Status保持ANALYZED。

## 8. 统计解释风险检查

Coverage: 11/11 checked（不是11项显著性检验）。

| risk                         | status   | boundary                                                                        |
|:-----------------------------|:---------|:--------------------------------------------------------------------------------|
| Simpson / 分层反转           | CAUTION  | 保留全部阶段、control、聚合；不以单个有利分层替代主分析，稀疏分层不做总体外推。 |
| Ecological / 层级外推        | CAUTION  | 分析单位为skill-window，token不是独立验证样本；不能推出逐token因果credit。      |
| Berkson / 支持集选择         | CAUTION  | 固定18技能共同池；37技能coverage另存，不外推未自然支持的技能。                  |
| Collider / 条件选择          | CAUTION  | 方向AUC只含非零点标签；另报包含零标签的AP/AUROC，不按结果加筛选。               |
| Base-rate neglect            | CHECKED  | 列出下降/上升数量、恒预测类别基线和平衡准确率。                                 |
| Regression to mean           | CAUTION  | 不按极端效用或新score筛选样本；配对标签仍存在抽样误差。                         |
| Survivorship                 | CAUTION  | 零A、失败轨迹及不支持技能不隐去；bootstrap缺任何技能则整池NA并报告。            |
| Look-elsewhere               | CAUTION  | 历史273列已看，新组合仍是事后探索；无确认性p值。                                |
| Forking paths                | CAUTION  | 本次运行前冻结一个组合、两种对照分解；不改变符号或择优聚合。                    |
| Correlation vs causation     | CAUTION  | R是reward交互关联，不是动作真实贡献；B不是校准后的效用变化大小。                |
| Reverse causality / 标签泄漏 | CAUTION  | 新分数函数不接收gold且先commit；历史标签已知，不能宣称前瞻验证。                |
