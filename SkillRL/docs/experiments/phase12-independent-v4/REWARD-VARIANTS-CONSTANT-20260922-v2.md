## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-22
- Verification Status: ANALYZED (v1 的恒等式问题已定位；v2执行结果另见回执)
- Version Label: reward_variants_constant_preserving_v2

# 常数负对照的有限精度修正

用户授权的本轮 seed404 探索在 v1 计算完成后发现：A=+1 的 reward-only 负对照理论上恒为 -1，
但浮点加权求和因各 skill 的 token/decision/game 数不同产生约机器精度的差异，导致伪排序。
同一个输入分数的 sklearn 独立核验会复现这种伪排序，不能替代数学恒等式检查。
例如 v1 的 decision 等权该常数对照出现 AP=.670909，这是无意义的数值产物，不作科研证据。

v1 源码、plan、全部输出、初版解释文字均保留，并新增 NUMERICAL-CONSTANT-WARNING.md 标记。
不改变任何候选、符号、门槛、聚合定义、效用标签或数据池；不新增公式以追求更好结果。

唯一修正：对每个 skill/context/phase/control 的每个逐 token 信号列，若全部输入值严格相同，
聚合直接返回该常数；否则完全保留 v1 的运算。三个聚合均适用。
对轨迹块符号负对照，根据每条轨迹正/反向 token 信号的 min/max，判断抽样后全体输入是否严格为常数，
仅在严格为常数时作相同修正。不是对相近分数做舍入、阈值截断或合并；不是改写任何真实效用。

新增独立 adapter `skillnet_cohort/reward_variant_constant_fix.py` 及专项测试，原 v1 所有源码冻结不改。
新输出 `reward-variants-s404-v2` 绑定 v1 plan/complete/provenance/分数/指标及原输入，
重新计算同一批 CPU 标量统计；无 RL、模型前向、环境 rollout、API 或其他 seed。
这不是故障后自动重启 GPU 实验，而是本次分析验收中已公开说明的数值修正。

必须验收：单位 advantage reward-only 在每个 skill/聚合严格为 -1，其 AP 为事件比例、AUROC=.5、
Spearman 未定义；所有非恒定列保持；其他原指标与旧报告仍一致；记录受修正的全部单元及排序变化。
不将修正后的候选表现称作新的理论预测信息。最终报告引用 v2，同时保留 v1 和缺陷说明。
