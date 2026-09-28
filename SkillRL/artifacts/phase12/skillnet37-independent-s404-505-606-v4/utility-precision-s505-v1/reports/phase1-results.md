## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-24T19:26:49.614290+00:00
- Verification Status: UNVERIFIED
- Version Label: utility_precision_seed505_v1

# Phase1：seed505 U0→U5 独立RL窗口，8组gold配对效用

固定37技能库、逐状态top1路由、404首调用锚点；复用原2组，新增6组。同一anchor、arm和endpoint共用续跑seed；不重训、不重采样旧轨迹。

| skill_id                                     |   source_games |   anchor_count |   delta_utility |      ci_low |     ci_high |
|:---------------------------------------------|---------------:|---------------:|----------------:|------------:|------------:|
| skillnet:alfworld-appliance-navigator        |             62 |             62 |       0.0524194 |  -0.0141129 |   0.120968  |
| skillnet:alfworld-appliance-preparer         |              8 |              8 |       0         |   0         |   0         |
| skillnet:alfworld-clean-object               |             37 |             37 |       0.114865  |  -0.0168919 |   0.253378  |
| skillnet:alfworld-device-operator            |             59 |             59 |       0.0423729 |  -0.0550847 |   0.141949  |
| skillnet:alfworld-heat-object-with-appliance |             32 |             32 |       0.09375   |   0.0078125 |   0.191406  |
| skillnet:alfworld-inventory-management       |              2 |              2 |       0.0625    |   0         |   0.25      |
| skillnet:alfworld-locate-target-object       |              2 |              2 |       0         |   0         |   0         |
| skillnet:alfworld-object-cooler              |             21 |             21 |       0.0416667 |  -0.113095  |   0.196429  |
| skillnet:alfworld-object-disposer            |              2 |              2 |       0.0625    |   0         |   0.25      |
| skillnet:alfworld-object-heater              |              9 |              9 |       0.0416667 |  -0.0833333 |   0.180556  |
| skillnet:alfworld-object-picker              |              7 |              7 |      -0.0535714 |  -0.232143  |   0.0892857 |
| skillnet:alfworld-object-placer              |             23 |             23 |       0.0923913 |  -0.0271739 |   0.217391  |
| skillnet:alfworld-object-retriever           |             15 |             15 |       0.05      |  -0.116667  |   0.25      |
| skillnet:alfworld-object-state-inspector     |             18 |             18 |       0.347222  |   0.1875    |   0.513889  |
| skillnet:alfworld-object-storer              |             18 |             18 |       0.0208333 |  -0.0625    |   0.111111  |
| skillnet:alfworld-object-transporter         |              1 |              1 |       0.125     | nan         | nan         |
| skillnet:alfworld-receptacle-closer          |              5 |              5 |      -0.05      |  -0.15      |   0         |
| skillnet:alfworld-receptacle-finder          |             10 |             10 |       0.0625    |  -0.0375    |   0.1875    |
| skillnet:alfworld-receptacle-navigator       |              2 |              2 |      -0.125     |  -0.375     |   0         |
| skillnet:alfworld-receptacle-opener          |              8 |              8 |      -0.03125   |  -0.109375  |   0         |
| skillnet:alfworld-receptacle-operator        |              7 |              7 |      -0.142857  |  -0.357143  |   0         |
| skillnet:alfworld-receptacle-searcher        |              2 |              2 |      -0.0625    |  -0.375     |   0.25      |
| skillnet:alfworld-storage-explorer           |             44 |             44 |      -0.03125   |  -0.116477  |   0.0482955 |
| skillnet:alfworld-temperature-regulator      |              6 |              6 |      -0.0625    |  -0.1875    |   0         |
| skillnet:alfworld-tool-locator               |              4 |              4 |       0.15625   |   0         |   0.5       |

全部自然技能、未支持原因、原始成功率与精度曲线分别见CSV。
