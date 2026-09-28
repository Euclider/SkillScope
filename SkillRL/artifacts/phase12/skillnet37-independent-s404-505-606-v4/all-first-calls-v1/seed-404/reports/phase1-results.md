## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21T10:35:57.635974+00:00
- Verification Status: UNVERIFIED
- Version Label: all_u0_first_calls_v1

# Phase1：seed 404，U0→U5，全部首调用锚点评估


本版本由用户在seed404旧结果已产生后授权修改覆盖规则，并在seed505仍训练时冻结统一规则。seed404属于事后覆盖扩展，不是原预登记分析的重新命名。技能内容、policy、router、C/P/D公式及其方向不变。取消的是数量门槛；缺少真实训练token的分数为NA，不填零。零advantage时按公式得到零分，同时明确记录reward_direction_observed=False，不据此断言效用稳定。

每条U0轨迹对每个skill只取首次自然调用，全部首调用锚点均评估；后续调用仅统计次数，不新增锚点。固定原动作前缀回放，从锚点起对目标skill的本次及后续自然调用施加O/P/N，其他技能不变。这些是阶段起的剩余效用，不是单次调用的孤立因果效应。

实际训练batch上的C/P/D不改为首调用读出，不修改advantage；本次去重只作用于效用锚点。

点估计依次等权平均continuation、首调用源轨迹、game；区间按game聚类重采样，沿用配对game/continuation重采样，所有arm和端点配对不拆开。单game只给点估计、不伪造跨game置信区间；阶段分层、anchors、continuation seeds都不是独立RL seeds。排名指标为描述性比较，不据多重分层选择赢家或声称跨seed显著性；D=0不等于无效用变化。

原训练恢复历史与U0导出的BF16/FP32元数据边界全部继承，未因重算报告而消失；完整历史说明见本版本plan.json绑定的archived-reports及原训练目录，原批次、OLD概率与检查点保留。

自然出现技能 25/37；锚点 404；两端点续跑 7272。训练和完整seen/unseen性能不重跑。

|   seed |   update | split        | task   |   episodes |   success_rate |
|-------:|---------:|:-------------|:-------|-----------:|---------------:|
|    404 |        0 | valid_seen   | all    |        140 |       0.235714 |
|    404 |        0 | valid_unseen | all    |        134 |       0.238806 |
|    404 |        5 | valid_seen   | all    |        140 |       0.2      |
|    404 |        5 | valid_unseen | all    |        134 |       0.179104 |

| skill_id                                     |   source_calls |   source_games |   train_decisions |   train_games |   train_nonzero_games |   anchor_count |   utility_old |   utility_new |   delta_utility |       ci_low |     ci_high | interval_status                |
|:---------------------------------------------|---------------:|---------------:|------------------:|--------------:|----------------------:|---------------:|--------------:|--------------:|----------------:|-------------:|------------:|:-------------------------------|
| skillnet:alfworld-appliance-navigator        |            498 |             62 |               675 |             8 |                     7 |             62 |    -0.0564516 |     0.0241935 |       0.0806452 |   0.00806452 |   0.169355  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-appliance-preparer         |             89 |              8 |                17 |             1 |                     1 |              8 |     0         |     0         |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-clean-object               |           1270 |             37 |               294 |             3 |                     3 |             37 |    -0.0945946 |    -0.108108  |      -0.0135135 |  -0.162162   |   0.135135  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-device-operator            |            643 |             59 |               966 |             8 |                     7 |             59 |     0.059322  |     0.0254237 |      -0.0338983 |  -0.152542   |   0.0847458 | paired_game_cluster_bootstrap  |
| skillnet:alfworld-heat-object-with-appliance |            452 |             32 |               146 |             3 |                     3 |             32 |     0.15625   |     0.140625  |      -0.015625  |  -0.125      |   0.09375   | paired_game_cluster_bootstrap  |
| skillnet:alfworld-inventory-management       |              2 |              2 |                 0 |             0 |                     0 |              2 |     0         |     0         |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-locate-target-object       |              2 |              2 |                 1 |             1 |                     1 |              2 |     0         |     0         |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-cooler              |            260 |             21 |               625 |             4 |                     4 |             21 |     0.0238095 |     0.0952381 |       0.0714286 |  -0.0952381  |   0.285714  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-disposer            |              3 |              2 |                 0 |             0 |                     0 |              2 |     0         |     0         |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-heater              |             23 |              9 |                10 |             1 |                     1 |              9 |     0         |    -0.0555556 |      -0.0555556 |  -0.222222   |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-picker              |             29 |              7 |                25 |             5 |                     5 |              7 |     0         |     0         |       0         |  -0.214286   |   0.214286  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-placer              |            403 |             23 |              1136 |             7 |                     7 |             23 |    -0.152174  |     0         |       0.152174  |  -0.0434783  |   0.347826  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-retriever           |             16 |             15 |                 0 |             0 |                     0 |             15 |    -0.0333333 |    -0.0333333 |       0         |  -0.166667   |   0.2       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-state-inspector     |            684 |             18 |               399 |             1 |                     1 |             18 |     0.0555556 |     0.194444  |       0.138889  |  -0.111111   |   0.388889  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-storer              |            267 |             18 |                97 |             4 |                     3 |             18 |    -0.0555556 |    -0.0555556 |       0         |  -0.166667   |   0.166667  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-object-transporter         |              1 |              1 |                 0 |             0 |                     0 |              1 |     0         |     0         |       0         | nan          | nan         | single_game_interval_undefined |
| skillnet:alfworld-receptacle-closer          |             10 |              5 |                 1 |             1 |                     1 |              5 |     0.1       |     0         |      -0.1       |  -0.4        |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-finder          |            155 |             10 |               291 |             5 |                     4 |             10 |    -0.2       |    -0.1       |       0.1       |  -0.1        |   0.35      | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-navigator       |              2 |              2 |                 0 |             0 |                     0 |              2 |     0         |     0         |       0         |  -0.5        |   0.5       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-opener          |             26 |              8 |                13 |             2 |                     1 |              8 |     0         |     0         |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-operator        |             24 |              7 |                25 |             2 |                     1 |              7 |     0         |     0         |       0         |  -0.285714   |   0.216071  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-receptacle-searcher        |              2 |              2 |                 0 |             0 |                     0 |              2 |     0.25      |     0         |      -0.25      |  -0.75       |   0         | paired_game_cluster_bootstrap  |
| skillnet:alfworld-storage-explorer           |            657 |             44 |               786 |             9 |                     8 |             44 |    -0.0681818 |    -0.0340909 |       0.0340909 |  -0.0909091  |   0.159091  | paired_game_cluster_bootstrap  |
| skillnet:alfworld-temperature-regulator      |             36 |              6 |                 4 |             2 |                     1 |              6 |     0.0833333 |     0.166667  |       0.0833333 |  -0.25       |   0.5       | paired_game_cluster_bootstrap  |
| skillnet:alfworld-tool-locator               |              4 |              4 |                 0 |             0 |                     0 |              4 |    -0.125     |    -0.125     |       0         |   0          |   0         | paired_game_cluster_bootstrap  |
