## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-20T10:39:04.464271+00:00
- Verification Status: UNVERIFIED
- Version Label: phase12_independent_seed_v4

# Phase1：seed 404，冻结 SkillNet-37，U0→U5

训练为五次 GRPO 迭代，不是五次 optimizer.step。所有任务共用完整 37 技能库及独立冻结 router。

|   seed |   update | split        | task   |   episodes |   success_rate |
|-------:|---------:|:-------------|:-------|-----------:|---------------:|
|    404 |        0 | valid_seen   | all    |        140 |       0.235714 |
|    404 |        0 | valid_unseen | all    |        134 |       0.238806 |
|    404 |        5 | valid_seen   | all    |        140 |       0.2      |
|    404 |        5 | valid_unseen | all    |        134 |       0.179104 |

## 同锚点、同 continuation seed 的边际效用变化

| skill_id                                     |   anchor_count |   game_count |   utility_old |   utility_new |   delta_utility |    ci_low |   ci_high |
|:---------------------------------------------|---------------:|-------------:|--------------:|--------------:|----------------:|----------:|----------:|
| skillnet:alfworld-appliance-navigator        |             12 |           12 |     0         |     0         |       0         |  0        |  0        |
| skillnet:alfworld-clean-object               |             12 |           12 |    -0.0833333 |     0.0833333 |       0.166667  |  0        |  0.416667 |
| skillnet:alfworld-device-operator            |             12 |           12 |     0.166667  |     0.125     |      -0.0416667 | -0.291667 |  0.208333 |
| skillnet:alfworld-heat-object-with-appliance |             12 |           12 |     0         |     0         |       0         |  0        |  0        |
| skillnet:alfworld-storage-explorer           |             12 |           12 |    -0.125     |    -0.0416667 |       0.0833333 | -0.166667 |  0.333333 |

M=success_ORIGINAL−success_PLACEBO，ΔM=M_U5−M_U0；NULL 为次要对照。CI 使用配对 game/continuation bootstrap。点估计有变化不等于已统计确证；不以正负翻转为必要成功标准。原始配对结果、动作序列和逐技能支持记录保留于各 window。


恢复说明：seed404 在首轮 rollout 完成、首次 optimizer.step 之前发生进度文件扫描竞态。原128条轨迹及动作token被复用，未重新采样；已有OLD概率与恢复后的前向结果逐bit核对。vLLM/worker随机状态在登记seed上重新初始化，不声称与不中断运行逐bit等价。首轮遗失的vLLM采样logprob仅用于后端差异诊断，未伪造；训练使用重新完成的native OLD。累计30小时包含首次失败运行，扣除故障停机；详见recovery-v1及其保全证据。

第二次恢复补充：首次恢复已完成5512行native OLD和reference前向，随后Python bool/NumPy标量兼容错误在首次优化前中断。本次复用全部已存的实际trainer-chosen OLD概率；每rank一个原状态经native前向精确核验，并验证所有有效token在恢复native dtype后保持精确。reference因未持久化而重算。首轮OLD entropy日志缺失，未伪造；优化器内entropy正则及其余RL参数完全保留。U1没有重新采样环境轨迹；累计30小时计入此前两个attempt。详见recovery-v2。

第三次恢复补充：五轮RL、640条训练轨迹及每rank的204次Adam更新已完成，仅训练后导出子进程因本地verl导入路径中断。以模块入口修复并直接导出原U5，没有重跑训练、重采样训练轨迹或改变原生检查点。新导出在发布前与八rank原生FP32分片逐张量、逐bit核验；旧失败目录及记录保留。累计30小时计入此前所有实际运行，故障停机不计；详见recovery-v3。

运行时限修订：用户在2026-09-20明确取消三个seed累计时间上限；以上恢复说明中的30小时为历史规则，不再约束本次接续。磁盘保护、科学设置、404→505→606顺序不变，仍记录累计实际运行时间。
第四次恢复仅修复窗口汇总对逐轮NEW概率的过时依赖，复用原5轮RL、U5导出、U0全量评估/anchors/540条续跑及8个读出分片。窗口协议和C/P/D定义未改；起点OLD为实际训练概率，终点为相同输入的U5前向。U0 HF快照物理权重为BF16，不能把元数据的FP32当成原生master精度；参数范数诊断使用已登记原模型按原生加载步骤在CPU重建的FP32 B0，不将有舍入的U0导出上转后冒充原生master。无额外RL或环境rollout。原U0 BF16运行值核对为PASS，FP32导出一致性检查为FAIL，两者证据均保留。
