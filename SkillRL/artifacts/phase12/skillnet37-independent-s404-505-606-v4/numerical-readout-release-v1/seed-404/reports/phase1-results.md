## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21T14:38:11.567619+00:00
- Verification Status: UNVERIFIED
- Version Label: numerical_readout_v1

# Phase1：seed 404，效用标签复用

这是已见旧结果后获批的数值实现修正，不追溯改写为新的预登记实验。同一训练批次、U0/U5、动作token、advantage、完整37技能库、router、首调用锚点、阈值、候选池与聚合权重保持。模型前向仍为原BF16；只对读出算术使用FP64。旧版源码/报告和所有RL/效用轨迹保留；没有重新采样RL、性能或O/P/N效用。

legacy_recorded为原报告；stable_raw为规范化OLD概率和稳定零和投影的FP64修正；stable_centered_gate使用同一稳定P，只将C及D门控切换为中心化量。P_centered仅作恒等式检查，不当作新预测器。原版在同次前向上的所有标量读出逐值一致，原版PLACEBO排序报告也已复现；因此不是把后端前向变化误称为数值改进。KL/JS及activation基线保留原定义，不拟合阈值或挑选赢家。

所有自然出现技能与全部首次调用锚点保留；未出现/没有训练读出的NA仍明确记录。有效方向比例改变可能来自数值纠错，不能解释为新轨迹支持。点估计、配对game/continuation区间和历史U0权重精度/恢复边界全部继承原报告。

|   seed |   update | split        | task   |   episodes |   success_rate |
|-------:|---------:|:-------------|:-------|-----------:|---------------:|
|    404 |        0 | valid_seen   | all    |        140 |       0.235714 |
|    404 |        0 | valid_unseen | all    |        134 |       0.238806 |
|    404 |        5 | valid_seen   | all    |        140 |       0.2      |
|    404 |        5 | valid_unseen | all    |        134 |       0.179104 |

| skill_id                                     |   source_calls |   source_games |   train_decisions |   anchor_count |   utility_old |   utility_new |   delta_utility |       ci_low |     ci_high | interval_status                |
|:---------------------------------------------|---------------:|---------------:|------------------:|---------------:|--------------:|--------------:|----------------:|-------------:|------------:|:-------------------------------|
| skillnet:alfworld-appliance-navigator        |            498 |             62 |               675 |             62 |    -0.0564516 |     0.0241935 |       0.0806452 |   0.00806452 |   0.169355  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-appliance-preparer         |             89 |              8 |                17 |              8 |     0         |     0         |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-clean-object               |           1270 |             37 |               294 |             37 |    -0.0945946 |    -0.108108  |      -0.0135135 |  -0.162162   |   0.135135  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-device-operator            |            643 |             59 |               966 |             59 |     0.059322  |     0.0254237 |      -0.0338983 |  -0.152542   |   0.0847458 | paired_game_cluster_bootstrap  |
| skillnet:alfworld-heat-object-with-appliance |            452 |             32 |               146 |             32 |     0.15625   |     0.140625  |      -0.015625  |  -0.125      |   0.09375   | paired_game_cluster_bootstrap  |
| skillnet:alfworld-inventory-management       |              2 |              2 |                 0 |              2 |     0         |     0         |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-locate-target-object       |              2 |              2 |                 1 |              2 |     0         |     0         |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-cooler              |            260 |             21 |               625 |             21 |     0.0238095 |     0.0952381 |       0.0714286 |  -0.0952381  |   0.285714  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-disposer            |              3 |              2 |                 0 |              2 |     0         |     0         |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-heater              |             23 |              9 |                10 |              9 |     0         |    -0.0555556 |      -0.0555556 |  -0.222222   |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-picker              |             29 |              7 |                25 |              7 |     0         |     0         |       0         |  -0.214286   |   0.214286  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-placer              |            403 |             23 |              1136 |             23 |    -0.152174  |     0         |       0.152174  |  -0.0434783  |   0.347826  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-retriever           |             16 |             15 |                 0 |             15 |    -0.0333333 |    -0.0333333 |       0         |  -0.166667   |   0.2       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-state-inspector     |            684 |             18 |               399 |             18 |     0.0555556 |     0.194444  |       0.138889  |  -0.111111   |   0.388889  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-storer              |            267 |             18 |                97 |             18 |    -0.0555556 |    -0.0555556 |       0         |  -0.166667   |   0.166667  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-transporter         |              1 |              1 |                 0 |              1 |     0         |     0         |       0         | nan          | nan         | single_game_interval_undefined |
| skillnet:alfworld-receptacle-closer          |             10 |              5 |                 1 |              5 |     0.1       |     0         |      -0.1       |  -0.4        |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-finder          |            155 |             10 |               291 |             10 |    -0.2       |    -0.1       |       0.1       |  -0.1        |   0.35      | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-navigator       |              2 |              2 |                 0 |              2 |     0         |     0         |       0         |  -0.5        |   0.5       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-opener          |             26 |              8 |                13 |              8 |     0         |     0         |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-operator        |             24 |              7 |                25 |              7 |     0         |     0         |       0         |  -0.285714   |   0.216071  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-searcher        |              2 |              2 |                 0 |              2 |     0.25      |     0         |      -0.25      |  -0.75       |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-storage-explorer           |            657 |             44 |               786 |             44 |    -0.0681818 |    -0.0340909 |       0.0340909 |  -0.0909091  |   0.159091  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-temperature-regulator      |             36 |              6 |                 4 |              6 |     0.0833333 |     0.166667  |       0.0833333 |  -0.25       |   0.5       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-tool-locator               |              4 |              4 |                 0 |              4 |    -0.125     |    -0.125     |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
