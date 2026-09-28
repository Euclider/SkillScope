## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21T23:27:37.979502+00:00
- Verification Status: UNVERIFIED
- Version Label: numerical_readout_v1

# Phase1：seed 505，效用标签复用

这是已见旧结果后获批的数值实现修正，不追溯改写为新的预登记实验。同一训练批次、U0/U5、动作token、advantage、完整37技能库、router、首调用锚点、阈值、候选池与聚合权重保持。模型前向仍为原BF16；只对读出算术使用FP64。旧版源码/报告和所有RL/效用轨迹保留；没有重新采样RL、性能或O/P/N效用。

legacy_recorded为原报告；stable_raw为规范化OLD概率和稳定零和投影的FP64修正；stable_centered_gate使用同一稳定P，只将C及D门控切换为中心化量。P_centered仅作恒等式检查，不当作新预测器。原版在同次前向上的所有标量读出逐值一致，原版PLACEBO排序报告也已复现；因此不是把后端前向变化误称为数值改进。KL/JS及activation基线保留原定义，不拟合阈值或挑选赢家。

所有自然出现技能与全部首次调用锚点保留；未出现/没有训练读出的NA仍明确记录。有效方向比例改变可能来自数值纠错，不能解释为新轨迹支持。点估计、配对game/continuation区间和历史U0权重精度/恢复边界全部继承原报告。

|   seed |   update | split        | task   |   episodes |   success_rate |
|-------:|---------:|:-------------|:-------|-----------:|---------------:|
|    505 |        0 | valid_seen   | all    |        140 |       0.235714 |
|    505 |        0 | valid_unseen | all    |        134 |       0.238806 |
|    505 |        5 | valid_seen   | all    |        140 |       0.257143 |
|    505 |        5 | valid_unseen | all    |        134 |       0.365672 |

| skill_id                                     |   source_calls |   source_games |   train_decisions |   anchor_count |   utility_old |   utility_new |   delta_utility |      ci_low |     ci_high | interval_status                |
|:---------------------------------------------|---------------:|---------------:|------------------:|---------------:|--------------:|--------------:|----------------:|------------:|------------:|:-------------------------------|
| skillnet:alfworld-appliance-navigator        |            498 |             62 |               275 |             62 |    -0.0564516 |     0.016129  |       0.0725806 |  -0.0322581 |   0.185484  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-appliance-preparer         |             89 |              8 |                21 |              8 |     0         |     0         |       0         |   0         |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-clean-object               |           1270 |             37 |               937 |             37 |    -0.0945946 |    -0.0540541 |       0.0405405 |  -0.108108  |   0.189189  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-device-operator            |            643 |             59 |               676 |             59 |     0.059322  |     0.144068  |       0.0847458 |  -0.0423729 |   0.220339  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-heat-object-with-appliance |            452 |             32 |               365 |             32 |     0.15625   |     0.234375  |       0.078125  |  -0.046875  |   0.21875   | paired_game_cluster_bootstrap  |
| skillnet:alfworld-inventory-management       |              2 |              2 |               583 |              2 |     0         |     0         |       0         |   0         |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-locate-target-object       |              2 |              2 |                23 |              2 |     0         |     0         |       0         |   0         |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-cooler              |            260 |             21 |                36 |             21 |     0.0238095 |     0.0952381 |       0.0714286 |  -0.0952381 |   0.261905  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-disposer            |              3 |              2 |                 0 |              2 |     0         |     0         |       0         |   0         |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-heater              |             23 |              9 |                23 |              9 |     0         |     0         |       0         |  -0.277778  |   0.277778  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-picker              |             29 |              7 |                75 |              7 |     0         |     0         |       0         |  -0.214286  |   0.214286  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-placer              |            403 |             23 |               529 |             23 |    -0.152174  |    -0.0869565 |       0.0652174 |  -0.108696  |   0.23913   | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-retriever           |             16 |             15 |                21 |             15 |    -0.0333333 |     0         |       0.0333333 |  -0.1       |   0.2       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-state-inspector     |            684 |             18 |               804 |             18 |     0.0555556 |     0.25      |       0.194444  |  -0.0833333 |   0.5       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-storer              |            267 |             18 |                 1 |             18 |    -0.0555556 |     0         |       0.0555556 |   0         |   0.166667  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-transporter         |              1 |              1 |                99 |              1 |     0         |     0         |       0         | nan         | nan         | single_game_interval_undefined |
| skillnet:alfworld-receptacle-closer          |             10 |              5 |                 0 |              5 |     0.1       |    -0.1       |      -0.2       |  -0.5       |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-finder          |            155 |             10 |               219 |             10 |    -0.2       |     0.1       |       0.3       |   0.05      |   0.6       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-navigator       |              2 |              2 |                 3 |              2 |     0         |     0         |       0         |   0         |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-opener          |             26 |              8 |                 0 |              8 |     0         |    -0.0625    |      -0.0625    |  -0.25      |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-operator        |             24 |              7 |                 0 |              7 |     0         |    -0.142857  |      -0.142857  |  -0.428571  |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-searcher        |              2 |              2 |                 0 |              2 |     0.25      |     0         |      -0.25      |  -0.75      |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-storage-explorer           |            657 |             44 |               899 |             44 |    -0.0681818 |    -0.113636  |      -0.0454545 |  -0.159091  |   0.0681818 | paired_game_cluster_bootstrap  |
| skillnet:alfworld-temperature-regulator      |             36 |              6 |                 0 |              6 |     0.0833333 |     0         |      -0.0833333 |  -0.333333  |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-tool-locator               |              4 |              4 |                 8 |              4 |    -0.125     |     0         |       0.125     |   0         |   0.5       | paired_game_cluster_bootstrap  |
