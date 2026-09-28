# Seed404：reward 校准的实际更新读出（新增探索）

Material Passport — Origin Skill: academic-research-suite / experiment-agent; Mode: run/validate; Version: realized_reward_secant_v1; Status: ANALYZED，单 seed 事后探索。

## 1. 公式与研究边界

`u=log p5(skill)-log p0(skill); v=Hu; xi=Hdelta; q=A*u(a)`；`D_real=-Agg[q*dot(v,xi)/(||v||²+1e-12)]`。H为逐位置词表中心化。保留全部正负贡献与零优势分母，不额外门控。q使用未中心化的动作log概率差。

q是固定优势局部log-likelihood surrogate的变化，不是实际PPO/GRPO总目标变化；该方向是rank-one/secant近似，不是纯reward梯度，更不是逐步因果credit。u同时出现在delta内，存在机械几何关联，必须对比A=1、abs(A)、仅投影系数和轨迹块符号随机化。

主口径预先固定PLACEBO/all/token/0pp。decision与game聚合、NULL、阶段分层和5pp仅敏感性分析。旧404/505标签已见，先锁代码再计算也不能称为独立确认性检验。

## 2. 方向预测主表

下降风险固定为分数越高越大；原始delta是向量，幅度对照是||delta||与||Hdelta||。幅度或去reward对照没有校准的正负方向，不把其数值符号当预测。

| score                                        |   candidates |   declines |   increases |   auroc_decline_vs_increase |   spearman |   average_precision |   auroc_decline_vs_rest |   sign_accuracy_called |   sign_coverage |
|:---------------------------------------------|-------------:|-----------:|------------:|----------------------------:|-----------:|--------------------:|------------------------:|-----------------------:|----------------:|
| D_original::token::reward                    |           18 |          5 |           7 |                    0.428571 |  -0.147161 |            0.509650 |                0.630769 |             nan        |      nan        |
| D_original::token::unsigned                  |           18 |          5 |           7 |                    0.714286 |   0.187105 |            0.625195 |                0.800000 |             nan        |      nan        |
| D_signed::token::reward                      |           18 |          5 |           7 |                    0.457143 |  -0.064120 |            0.423810 |                0.461538 |               0.416667 |        1.000000 |
| D_signed::token::unsigned                    |           18 |          5 |           7 |                    0.485714 |   0.006307 |            0.443333 |                0.523077 |               0.500000 |        1.000000 |
| D_action_adv::token::reward                  |           18 |          5 |           7 |                    0.542857 |   0.249123 |            0.402197 |                0.384615 |               0.666667 |        1.000000 |
| D_action_adv::token::unsigned                |           18 |          5 |           7 |                    0.428571 |  -0.129291 |            0.363369 |                0.461538 |               0.416667 |        1.000000 |
| D_policy_action_adv::token::reward           |           18 |          5 |           7 |                    0.571429 |   0.208128 |            0.248311 |                0.338462 |               0.583333 |        1.000000 |
| D_policy_action_adv::token::unsigned         |           18 |          5 |           7 |                    0.371429 |  -0.334266 |            0.315490 |                0.446154 |               0.333333 |        1.000000 |
| C_centered::token::reward                    |           18 |          5 |           7 |                    0.657143 |   0.023125 |            0.688095 |                0.723077 |             nan        |      nan        |
| C_centered::token::unsigned                  |           18 |          5 |           7 |                    0.057143 |  -0.537138 |            0.199617 |                0.138462 |             nan        |      nan        |
| M_delta_raw::token::magnitude                |           18 |          5 |           7 |                    0.571429 |   0.064120 |            0.361538 |                0.553846 |             nan        |      nan        |
| M_delta_centered::token::magnitude           |           18 |          5 |           7 |                    0.742857 |   0.348982 |            0.613651 |                0.738462 |             nan        |      nan        |
| D_real::token::reward                        |           18 |          5 |           7 |                    0.485714 |   0.131394 |            0.228008 |                0.276923 |               0.583333 |        1.000000 |
| D_real_A1::token::reward_free                |           18 |          5 |           7 |                    0.428571 |  -0.228100 |            0.335556 |                0.476923 |             nan        |      nan        |
| D_real_absA::token::reward_sign_removed      |           18 |          5 |           7 |                    0.485714 |  -0.229151 |            0.450427 |                0.584615 |             nan        |      nan        |
| D_real_projection_only::token::geometry_only |           18 |          5 |           7 |                    0.257143 |  -0.263839 |            0.207157 |                0.184615 |             nan        |      nan        |

完整三聚合表见 `reports/primary-all-aggregations.csv`；方向混淆、平衡准确率、恒预测上升/下降的类别基率对照见 `direction-confusions.csv`。

## 3. 配对增益及不确定性

| control   | phase   |   threshold | score                 | reference                                    | metric                    |   point_difference |       low |     high |      mean |   valid_draws |   draw_fraction_positive | not_a_confirmatory_p_value   |
|:----------|:--------|------------:|:----------------------|:---------------------------------------------|:--------------------------|-------------------:|----------:|---------:|----------:|--------------:|-------------------------:|:-----------------------------|
| placebo   | all     |    0.000000 | D_real::token::reward | D_real_A1::token::reward_free                | auroc_decline_vs_increase |           0.057143 | -0.503571 | 0.557292 |  0.060540 |          1671 |                 0.590664 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_real_A1::token::reward_free                | spearman                  |           0.359493 | -0.328066 | 0.809254 |  0.267035 |          1674 |                 0.829152 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_real_absA::token::reward_sign_removed      | auroc_decline_vs_increase |           0.000000 | -0.563542 | 0.646429 |  0.067404 |          1671 |                 0.593058 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_real_absA::token::reward_sign_removed      | spearman                  |           0.360544 | -0.358973 | 0.946497 |  0.306329 |          1674 |                 0.826165 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_real_projection_only::token::geometry_only | auroc_decline_vs_increase |           0.228571 | -0.219618 | 0.416667 |  0.133712 |          1671 |                 0.791143 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_real_projection_only::token::geometry_only | spearman                  |           0.395232 | -0.154692 | 0.600763 |  0.273277 |          1674 |                 0.911589 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_original::token::reward                    | auroc_decline_vs_increase |           0.057143 | -0.312946 | 0.444444 |  0.050531 |          1671 |                 0.559545 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_original::token::reward                    | spearman                  |           0.278555 | -0.349729 | 0.788728 |  0.248802 |          1674 |                 0.805854 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_signed::token::reward                      | auroc_decline_vs_increase |           0.028571 | -0.209821 | 0.333333 |  0.066466 |          1671 |                 0.622382 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_signed::token::reward                      | spearman                  |           0.195514 | -0.189207 | 0.550741 |  0.198891 |          1674 |                 0.848865 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | C_centered::token::reward                    | auroc_decline_vs_increase |          -0.171429 | -0.515993 | 0.445833 | -0.045436 |          1671 |                 0.400359 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | C_centered::token::reward                    | spearman                  |           0.108268 | -0.469985 | 0.684366 |  0.129714 |          1674 |                 0.673238 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | M_delta_raw::token::magnitude                | auroc_decline_vs_increase |          -0.085714 | -0.585938 | 0.458242 | -0.077239 |          1671 |                 0.366248 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | M_delta_raw::token::magnitude                | spearman                  |           0.067274 | -0.631882 | 0.629457 |  0.014727 |          1674 |                 0.523297 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | M_delta_centered::token::magnitude           | auroc_decline_vs_increase |          -0.257143 | -0.625000 | 0.300000 | -0.183641 |          1671 |                 0.214243 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | M_delta_centered::token::magnitude           | spearman                  |          -0.217588 | -0.779497 | 0.458659 | -0.141919 |          1674 |                 0.353047 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_policy_action_adv::token::reward           | auroc_decline_vs_increase |          -0.085714 | -0.166667 | 0.071429 | -0.043528 |          1671 |                 0.159785 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_policy_action_adv::token::reward           | spearman                  |          -0.076734 | -0.178794 | 0.086125 | -0.046477 |          1674 |                 0.281362 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_action_adv::token::reward                  | auroc_decline_vs_increase |          -0.057143 | -0.363636 | 0.250000 | -0.054564 |          1671 |                 0.276481 | True                         |
| placebo   | all     |    0.000000 | D_real::token::reward | D_action_adv::token::reward                  | spearman                  |          -0.117729 | -0.342369 | 0.184099 | -0.070291 |          1674 |                 0.314217 | True                         |

与v2共用2000组game×continuation配对标签bootstrap；任何技能缺失则整池draw失效，不逐方法删样本。区间仅反映固定readout条件下的标签不确定性，不是训练seed泛化区间。

## 4. Reward符号负对照

| control   | phase   |   threshold | score                 | metric                    |   observed |       low |     high |     mean |   valid_draws |   reference_fraction_ge_observed |   three_aggregation_max_fraction_ge_observed | not_a_confirmatory_p_value   | does_not_adjust_historical_candidate_search   |
|:----------|:--------|------------:|:----------------------|:--------------------------|-----------:|----------:|---------:|---------:|--------------:|---------------------------------:|---------------------------------------------:|:-----------------------------|:----------------------------------------------|
| placebo   | all     |    0.000000 | D_real::token::reward | auroc_decline_vs_increase |   0.485714 |  0.171429 | 0.885714 | 0.502734 |           512 |                         0.560547 |                                     0.697266 | True                         | True                                          |
| placebo   | all     |    0.000000 | D_real::token::reward | spearman                  |   0.131394 | -0.536559 | 0.553641 | 0.006779 |           512 |                         0.314453 |                                     0.451172 | True                         | True                                          |
| placebo   | all     |    0.000000 | D_real::token::reward | average_precision         |   0.228008 |  0.204508 | 0.776817 | 0.438257 |           512 |                         0.931641 |                                     0.972656 | True                         | True                                          |
| placebo   | all     |    0.000000 | D_real::token::reward | auroc_decline_vs_rest     |   0.276923 |  0.165769 | 0.861538 | 0.503606 |           512 |                         0.902344 |                                     0.968750 | True                         | True                                          |

共用512组trajectory-block ±1符号；同轨迹所有token一起翻转，保留优势绝对值与内部差异。这不是交换性已证明的随机化试验，不报告成确认性p值。三聚合max不校正此前39公式的历史搜索。

## 5. 覆盖、数值和成本

复用seed404 U0/U5、原U0实际训练批次、所有155638 token×control行；逐决策复现旧稳定分数。效用标签仍来自每轨迹每技能首次自然调用；25种效用技能、18种全阶段共同可比较技能。零优势和稀疏支持不额外过滤，覆盖/频次/game数以及极小更新范数单列记录。

仅新增模型前向与标量读出；无新RL、环境rollout、API、505/606执行、阈值拟合或Phase3换指标。保留历史报告、125份原运行源码及既有v1/v2；所有旧比较指标复核，完整指标使用独立sklearn/scipy检查。

## 6. 统计误用检查（11/11）

1.非显著不等于等效；2.效应量与区间并报；3.技能不是独立token样本；4.类别基率及弃权透明；5.平均值不掩盖阶段/聚合差异；6.稀疏技能支持量不隐去；7.配对及共享game依赖保持；8.不从多重尝试择优宣称确认；9.已见标签探索如实标注；10.预测相关不等于编辑因果收益；11.实际更新共项可能导致机械相关，负对照不能自动证明无泄漏。

这次目标是检验方向性增益，而不是保证新公式获胜。即使单seed更优，仍需先冻结少数指标再用于独立seed/窗口，不能自动替换Phase3主指标。
