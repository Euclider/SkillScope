## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21T23:25:07.321430+00:00
- Verification Status: UNVERIFIED
- Version Label: all_u0_first_calls_v1

# Phase1：seed 505，U0→U5，全部首调用锚点评估


本版本由用户在seed404旧结果已产生后授权修改覆盖规则，并在seed505仍训练时冻结统一规则。seed404属于事后覆盖扩展，不是原预登记分析的重新命名。技能内容、policy、router、C/P/D公式及其方向不变。取消的是数量门槛；缺少真实训练token的分数为NA，不填零。零advantage时按公式得到零分，同时明确记录reward_direction_observed=False，不据此断言效用稳定。

每条U0轨迹对每个skill只取首次自然调用，全部首调用锚点均评估；后续调用仅统计次数，不新增锚点。固定原动作前缀回放，从锚点起对目标skill的本次及后续自然调用施加O/P/N，其他技能不变。这些是阶段起的剩余效用，不是单次调用的孤立因果效应。

实际训练batch上的C/P/D不改为首调用读出，不修改advantage；本次去重只作用于效用锚点。

点估计依次等权平均continuation、首调用源轨迹、game；区间按game聚类重采样，沿用配对game/continuation重采样，所有arm和端点配对不拆开。单game只给点估计、不伪造跨game置信区间；阶段分层、anchors、continuation seeds都不是独立RL seeds。排名指标为描述性比较，不据多重分层选择赢家或声称跨seed显著性；D=0不等于无效用变化。

原训练恢复历史与U0导出的BF16/FP32元数据边界全部继承，未因重算报告而消失；完整历史说明见本版本plan.json绑定的archived-reports及原训练目录，原批次、OLD概率与检查点保留。

自然出现技能 25/37；锚点 404；两端点续跑 7272。训练和完整seen/unseen性能不重跑。

|   seed |   update | split        | task   |   episodes |   success_rate |
|-------:|---------:|:-------------|:-------|-----------:|---------------:|
|    505 |        0 | valid_seen   | all    |        140 |       0.235714 |
|    505 |        0 | valid_unseen | all    |        134 |       0.238806 |
|    505 |        5 | valid_seen   | all    |        140 |       0.257143 |
|    505 |        5 | valid_unseen | all    |        134 |       0.365672 |

| skill_id                                     |   source_calls |   source_games |   train_decisions |   train_games |   train_nonzero_games |   anchor_count |   utility_old |   utility_new |   delta_utility |      ci_low |     ci_high | interval_status                |
|:---------------------------------------------|---------------:|---------------:|------------------:|--------------:|----------------------:|---------------:|--------------:|--------------:|----------------:|------------:|------------:|:-------------------------------|
| skillnet:alfworld-appliance-navigator        |            498 |             62 |               275 |             6 |                     4 |             62 |    -0.0564516 |     0.016129  |       0.0725806 |  -0.0322581 |   0.185484  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-appliance-preparer         |             89 |              8 |                21 |             1 |                     1 |              8 |     0         |     0         |       0         |   0         |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-clean-object               |           1270 |             37 |               937 |             6 |                     5 |             37 |    -0.0945946 |    -0.0540541 |       0.0405405 |  -0.108108  |   0.189189  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-device-operator            |            643 |             59 |               676 |             4 |                     2 |             59 |     0.059322  |     0.144068  |       0.0847458 |  -0.0423729 |   0.220339  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-heat-object-with-appliance |            452 |             32 |               365 |             4 |                     2 |             32 |     0.15625   |     0.234375  |       0.078125  |  -0.046875  |   0.21875   | paired_game_cluster_bootstrap  |
| skillnet:alfworld-inventory-management       |              2 |              2 |               583 |             3 |                     1 |              2 |     0         |     0         |       0         |   0         |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-locate-target-object       |              2 |              2 |                23 |             4 |                     4 |              2 |     0         |     0         |       0         |   0         |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-cooler              |            260 |             21 |                36 |             1 |                     0 |             21 |     0.0238095 |     0.0952381 |       0.0714286 |  -0.0952381 |   0.261905  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-disposer            |              3 |              2 |                 0 |             0 |                     0 |              2 |     0         |     0         |       0         |   0         |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-heater              |             23 |              9 |                23 |             2 |                     1 |              9 |     0         |     0         |       0         |  -0.277778  |   0.277778  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-picker              |             29 |              7 |                75 |             7 |                     4 |              7 |     0         |     0         |       0         |  -0.214286  |   0.214286  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-placer              |            403 |             23 |               529 |             7 |                     4 |             23 |    -0.152174  |    -0.0869565 |       0.0652174 |  -0.108696  |   0.23913   | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-retriever           |             16 |             15 |                21 |             5 |                     3 |             15 |    -0.0333333 |     0         |       0.0333333 |  -0.1       |   0.2       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-state-inspector     |            684 |             18 |               804 |             3 |                     2 |             18 |     0.0555556 |     0.25      |       0.194444  |  -0.0833333 |   0.5       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-storer              |            267 |             18 |                 1 |             1 |                     1 |             18 |    -0.0555556 |     0         |       0.0555556 |   0         |   0.166667  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-transporter         |              1 |              1 |                99 |             3 |                     1 |              1 |     0         |     0         |       0         | nan         | nan         | single_game_interval_undefined |
| skillnet:alfworld-receptacle-closer          |             10 |              5 |                 0 |             0 |                     0 |              5 |     0.1       |    -0.1       |      -0.2       |  -0.5       |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-finder          |            155 |             10 |               219 |             4 |                     2 |             10 |    -0.2       |     0.1       |       0.3       |   0.05      |   0.6       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-navigator       |              2 |              2 |                 3 |             1 |                     1 |              2 |     0         |     0         |       0         |   0         |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-opener          |             26 |              8 |                 0 |             0 |                     0 |              8 |     0         |    -0.0625    |      -0.0625    |  -0.25      |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-operator        |             24 |              7 |                 0 |             0 |                     0 |              7 |     0         |    -0.142857  |      -0.142857  |  -0.428571  |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-searcher        |              2 |              2 |                 0 |             0 |                     0 |              2 |     0.25      |     0         |      -0.25      |  -0.75      |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-storage-explorer           |            657 |             44 |               899 |             8 |                     5 |             44 |    -0.0681818 |    -0.113636  |      -0.0454545 |  -0.159091  |   0.0681818 | paired_game_cluster_bootstrap  |
| skillnet:alfworld-temperature-regulator      |             36 |              6 |                 0 |             0 |                     0 |              6 |     0.0833333 |     0         |      -0.0833333 |  -0.333333  |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-tool-locator               |              4 |              4 |                 8 |             1 |                     1 |              4 |    -0.125     |     0         |       0.125     |   0         |   0.5       | paired_game_cluster_bootstrap  |
