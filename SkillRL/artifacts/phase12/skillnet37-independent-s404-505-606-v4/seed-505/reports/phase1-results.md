## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21T03:59:46.277809+00:00
- Verification Status: UNVERIFIED
- Version Label: phase12_independent_seed_v4

# Phase1：seed 505，冻结 SkillNet-37，U0→U5

训练为五次 GRPO 迭代，不是五次 optimizer.step。所有任务共用完整 37 技能库及独立冻结 router。

|   seed |   update | split        | task   |   episodes |   success_rate |
|-------:|---------:|:-------------|:-------|-----------:|---------------:|
|    505 |        0 | valid_seen   | all    |        140 |       0.235714 |
|    505 |        0 | valid_unseen | all    |        134 |       0.238806 |
|    505 |        5 | valid_seen   | all    |        140 |       0.257143 |
|    505 |        5 | valid_unseen | all    |        134 |       0.365672 |

## 同锚点、同 continuation seed 的边际效用变化

| skill_id                                     |   anchor_count |   game_count |   utility_old |   utility_new |   delta_utility |    ci_low |   ci_high |
|:---------------------------------------------|---------------:|-------------:|--------------:|--------------:|----------------:|----------:|----------:|
| skillnet:alfworld-appliance-navigator        |             12 |           12 |     0         |    -0.0833333 |      -0.0833333 | -0.25     | 0         |
| skillnet:alfworld-clean-object               |             12 |           12 |    -0.0833333 |     0         |       0.0833333 | -0.166667 | 0.334375  |
| skillnet:alfworld-device-operator            |             12 |           12 |     0.166667  |     0.0416667 |      -0.125     | -0.375    | 0.0833333 |
| skillnet:alfworld-heat-object-with-appliance |             12 |           12 |     0         |     0.0416667 |       0.0416667 |  0        | 0.166667  |
| skillnet:alfworld-storage-explorer           |             12 |           12 |    -0.125     |    -0.333333  |      -0.208333  | -0.5      | 0.0833333 |

M=success_ORIGINAL−success_PLACEBO，ΔM=M_U5−M_U0；NULL 为次要对照。CI 使用配对 game/continuation bootstrap。点估计有变化不等于已统计确证；不以正负翻转为必要成功标准。原始配对结果、动作序列和逐技能支持记录保留于各 window。

运行时限修订：用户在2026-09-20明确取消三个seed累计时间上限；以上恢复说明中的30小时为历史规则，不再约束本次接续。磁盘保护、科学设置、404→505→606顺序不变，仍记录累计实际运行时间。
