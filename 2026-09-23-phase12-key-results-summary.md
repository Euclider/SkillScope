## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent / validate
- Origin Date：2026-09-23；Version Label：phase12_key_results_v1
- Verification Status：ANALYZED（核对既有报告与 CSV，未重跑实验）；Overall Confidence：CAUTION

# Phase1–2 关键结果汇报

> 一句话结论：已有证据支持“策略更新可以改变未修改技能的边际效用”，也有固定状态读出包含预测信息的线索；但目前尚未证明 reward-directed 读出相对幅度读出具有稳定、可泛化的方向预测增益。新实验的效用标签精度不足，是当前重要限制；Phase3 编辑收益尚无实验结果。

数据截止：2026-09-23 00:56 UTC。下列新 cohort 数字使用**已完整结束的两组 gold 续跑版本**，不混入正在进行的 seed404 八组 gold 补评。

## 1. 实验口径与完成范围

- 效用 `M(policy, skill) = E[Success_ORIGINAL − Success_PLACEBO]`；变化 `ΔM = M_new − M_old`。负值表示相对对照的贡献下降，不等于技能已经有害。NULL 为辅助对照。
- 两端 policy 在同一锚点、相同前缀上配对续跑。每条起点轨迹、每个技能只取首次自然调用；从该处开始，目标技能后续再次调用也持续使用同一干预。因此测量的是“从锚点开始的剩余贡献”，不是一次调用的孤立因果效应。
- 新 cohort：Qwen3.5-4B、冻结 SkillNet-37、冻结 Qwen3-Embedding-0.6B 逐状态 top-1；ALFWorld 六类任务。404/505 各独立 GRPO 5 轮，16×8 轨迹/轮、lr=1e-6，共各 640 条训练轨迹。606 未运行，不能称三个新 seed 已完成。
- 全量性能：140 Seen、134 Unseen。效用：同一 U0 来源的 404 个首调用锚点、25/37 个自然调用技能；取消数量筛选，但缺少起点训练读出的技能仍为 NA。可比较池为 seed404 的 18 技能、seed505 的 19 技能。
- 读出使用窗口首轮实际训练批次，在固定状态、动作 token、技能内容下前向比较 U0/U5；gold 续跑只用于验证，不是读出输入。两 seed 的 U0 与效用锚点共享，不将其当成额外独立重复。

## 2. Phase1：效用变化现象的主要证据

**历史 clean 小规模实验**（旧 SkillRL 库、seeds 101/202/303，不与新库合并）：`cle_003` 在 U20 相对 B0 的三 seed 汇总效用变化为 **+40.56 pp，95% 区间 [+28.89, +52.78]**；中途调用锚点也有变化。它支持“冻结技能的效用依赖 policy”，不支持所有技能都变化或必然由有益转为有害。[历史 Phase1](2026-09-04-qwen35-clean-all-skill-three-seed-utility-results.md)

历史动作交互信号 `S_int` 加入小型 logistic probe 后，leave-one-seed-out 的“是否变化”AUPRC 从 **0.192 → 0.274**；但“是否下降”AUPRC 为 **0.317 → 0.314**，未改善。前者是动作信号＋拟合 probe 的历史结果，不能改写为当前无拟合 C/P/D 的跨 seed 方向验证。

新 cohort 的完整任务成功率如下；这是 policy+冻结库的整体表现，**不是技能边际效用，也不是编辑收益**：

| Policy | Seen 成功率（成功数/140） | Unseen 成功率（成功数/134） |
|---|---:|---:|
| 共同 U0 | 23.57%（33） | 23.88%（32） |
| seed404 U5 | 20.00%（28） | 17.91%（24） |
| seed505 U5 | 25.71%（36） | 36.57%（49） |

两条训练路径的整体变化不同；只训练了 5 轮，不能据此推断收敛性能。[404 性能/效用](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/numerical-readout-release-v1/seed-404/reports/phase1-results.md)、[505 性能/效用](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/numerical-readout-release-v1/seed-505/reports/phase1-results.md)

新库的效用也有正向变化个例：404 的 appliance-navigator 为 **+8.06 pp [0.81, 16.94]**；505 的 receptacle-finder 为 **+30.00 pp [5.00, 60.00]**。这是逐技能的描述性 95% 区间，未作跨技能多重比较校正，不能据个例外推全库。

## 3. Phase2：保留正面案例，也保留未复现结果

历史 seed303 的 5 个相邻单步窗口、19 个技能×窗口单元：gated D 的下降 AP 为 **0.799**，中心化交互范数为 **0.667**；Spearman(score, −ΔM) 分别为 **0.417 / 0.062**。但后续 U35→U40 的 3 技能池里，两者 AP 均为 **0.583**，−P 与 KL 等为 **1.000**。这是小样本、重复技能/相邻窗口的描述性正面案例，未确立一个稳定胜出的 reward 构造。[历史 Phase2](phase2-complete-analysis.md)

新 SkillNet-37 实验的关键比较：PLACEBO、all phase、token 等权，下降标签为 `ΔM` 点估计 < 0；统一使用稳定数值版本。AP 越高表示下降技能更靠前，ρ 为与连续 `−ΔM` 的 Spearman。

中心化范数为 `mean(||Hδ||)`；H 表示每个 token 位置的 δ 减去其词表维度均值。它是策略响应变化的几何幅度，不自带效用增减的符号。

| 风险读出 | 404 AP | 404 ρ | 505 AP | 505 ρ |
|---|---:|---:|---:|---:|
| gated D，负部聚合 | 0.510 | −0.147 | 0.063 | −0.052 |
| −P，有符号投影 | 0.424 | −0.064 | 0.077 | −0.022 |
| 原始 +C_upd | 0.449 | +0.127 | 0.100 | −0.063 |
| 中心化 +C_upd | 0.688 | +0.023 | 0.083 | −0.148 |
| 中心化交互范数，无 reward | 0.614 | +0.349 | 0.167 | +0.200 |

404 池为 18 技能、5 个下降；505 为 19 技能、仅 1 个下降。对应常数分数 AP 为 5/18=0.278 和 1/19=0.053；不同基率的 AP 不直接跨 seed 比大小。中心化 C 在 404 的 AP 较高，但连续排序相关很弱、505 未复现；范数在这两池优于原 D，但**不是所有指标、所有历史实验上的统一最优**。数值稳定修正及中心化门控没有改变这两 seed 的 gated D 排名。[404 读出](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/numerical-readout-release-v1/seed-404/reports/phase2-results.md)、[505 读出](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/numerical-readout-release-v1/seed-505/reports/phase2-results.md)

seed404 的扩展探索目前累计保留 285 个分数列（含不同聚合/对照，不是 285 次独立实验）。两个近期 reward 方案均未改善主口径：

| 固定 token 聚合 | 方向 AUROC：下降 vs 上升 | 下降 AP |
|---|---:|---:|
| 中心化交互范数 | 0.743 | 0.614 |
| reward 校准实际更新投影 D_real | 0.486 | 0.228 |
| 中心化幅度 × reward 定向因子 D_factor | 0.600 | 0.295 |

方向 AUROC 只在 5 降＋7 升的 12 技能上计算；AP 使用完整 18 技能。D_factor 相对范数的方向 AUROC 差为 −0.143，95% 配对描述区间 **[−0.700, +0.439]**。区间宽，不能宣布显著优劣或方法等效；也不能据单 seed 大量尝试后的赢家宣称 reward 增益。[实际更新方案](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/realized-reward-s404-v2/reports/realized-reward-analysis.md)、[幅度×方向方案](2026-09-22-seed404-factorized-reward-validation.md)

## 4. 汇报时必须说清的限制

1. **边际效用评测目前精度不足，方向标签尚不稳健。** 现有标签每锚点只有 2 组 gold。两 seed 各自 25 个效用技能中均有 6 个下降点估计，但这些下降的 95% 区间全部触及或跨 0。可比较池中的 404 五个、505 唯一个下降也都未在区间上与 0 分离。例：404 的 clean-object 为 −1.35 pp，区间 [−16.22, +13.51]。点估计的负号不能当作可靠真值；零点估计/退化零区间也不能证明真实效用不变。
2. **覆盖与有效样本量有限。** 37 个库技能不等于 37 个可排序样本；505 的 6 个下降点估计中有 5 个没有该 seed 的起点训练读出。token、阶段、续跑 seed 都不是独立 RL 重复；game/续跑配对区间也不包含训练 seed 方差。
3. **不能把 reward 未占优全部归因于标签噪声。** 轨迹级 advantage 不等于逐动作因果 credit；训练状态与 Unseen 效用锚点也有分布差异。噪声、信号定义和覆盖不足均可能影响结果，当前不能区分各自贡献。
4. **大量变式属于已见标签后的探索。** 支持范围、数值实现和 reward 公式经历扩展；旧/新 cohort、不同聚合、不同 control 不混池择优。读出算术的 FP64 修正不代表模型前向是 FP64，也不能替代独立 seed 验证。
5. **尚无 Phase3 闭环收益或节省成本的实证。** “读出不额外采集更新后 rollout”描述的是诊断计算；本项目的 gold 验证确实花费大量续跑，不能写成整个实验没有额外 rollout 成本。

## 5. 正在进行、尚不能纳入结论的工作

seed404 的 gold 从 2 组扩至 8 组：保持模型、37 库、404 锚点及全部 285 列读出不变；同锚点各臂/两端配对续跑。主结果固定为 8 组，并报告 2/4/8 精度曲线及新增 6 组单独敏感性。两端合计新增 14,544 条续跑；**导出时尚未完成，本文没有使用其中部分结果**。增加续跑数不等于新增 RL seed，续跑 seed 数字等于 404 也不自动提高可信度。[冻结协议](SkillRL/docs/experiments/phase12-independent-v4/UTILITY-PRECISION-SEED404-20260922-v1.md)

**适合口头收束的表述：**“效用随 policy 改变的现象已有证据；方向预测有局部线索，但 reward 的稳定增量仍待验证。当前先提高效用标签精度，再判断读出优劣，之后才检验编辑闭环的实际收益。”

统计误用检查覆盖 11/11：分层反转、生态外推、自然支持选择、条件化、事件基率、均值回归、缺失/幸存、多重搜索、分析路径、相关与因果、时间顺序。主要风险已在上文披露；本次仅导出摘要，不新增检验、不改旧报告或运行设置。表中数值核对源为两 seed 的 `reports/ranking_diagnostics.csv`、`performance.csv`、`coverage_and_effects.csv`。
