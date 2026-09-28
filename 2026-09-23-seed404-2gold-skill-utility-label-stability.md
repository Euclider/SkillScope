## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: validate
- Origin Date: 2026-09-23
- Verification Status: ANALYZED（只读复核已保存的统计表和配对续跑记录；未重跑实验）
- Version Label: seed404_2gold_utility_label_stability_v1

# Seed404 的 2-gold 技能效用：逐技能统计与标签精度边界

本文只整理已完成的 seed404、U0→U5、**2 个 gold continuation seed** 的结果；未完成的 8-gold 补评不纳入。技能库在该窗口内冻结。以下数值是描述性验证数据，不把单个 RL seed 的结果当作跨 seed 结论。

## 口径与数据来源

对每种在 U0 的 unseen 参考轨迹中自然出现的技能，每条轨迹只取该技能的首次调用作为锚点。U0 和 U5 在相同锚点、相同 gold continuation seed 下分别从技能条件与 placebo 对照条件续跑。定义边际效用 (M_u(s)=\mathrm{Success}_{u,\mathrm{skill}}-\mathrm{Success}_{u,\mathrm{placebo}})，效用变化为 \(\Delta M(s)=M_5(s)-M_0(s)\)。点估计按 continuation、来源轨迹、game 等权汇总；区间为原报告的配对 game/continuation 聚类 bootstrap 的第 2.5% 和 97.5% 分位数。一个 game 时不生成跨 game 区间。

表中全部效用和变化都以**百分点（pp）**表示；各列独立四舍五入至 0.1 pp，因而显示值之间可能相差 0.1 pp。最后两列分别只使用 seed 63011 和 63021 的配对 U0/U5 续跑计算 \(\Delta M\)，用于展示测量重复间的方向差异。这两个续跑 seed **不是独立 RL 训练 seed**。

原始来源：

- [逐技能汇总和区间](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/numerical-readout-release-v1/seed-404/reports/coverage_and_effects.csv)：取 `phase=all, control=placebo`，包含 `utility_old`、`utility_new`、`delta_utility`、`ci_low`、`ci_high`、两端原始/对照成功率及覆盖度。
- [逐锚点配对续跑](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/numerical-readout-release-v1/seed-404/reused-labels/anchor_margins.parquet)：取 `purpose=gold`、`continuation_seed∈{63011,63021}`，按同一 `skill_id/anchor_id/continuation_seed` 配对 update 0 与 5。
- [评估协议说明](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1/seed-404/reports/phase2-results.md)：首调用锚点、等权聚合、区间与解释边界。

## 全部有锚点技能的效用统计

| Skill（共同前缀 `alfworld-` 省略） | 锚点数 | U0 效用 | U5 效用 | Δ效用 | 95% 区间 | 63011 Δ | 63021 Δ |
|---|---:|---:|---:|---:|---:|---:|---:|
| receptacle-searcher | 2 | +25.0 | 0.0 | −25.0 | [−75.0, 0.0] | 0.0 | −50.0 |
| receptacle-closer | 5 | +10.0 | 0.0 | −10.0 | [−40.0, 0.0] | 0.0 | −20.0 |
| object-heater | 9 | 0.0 | −5.6 | −5.6 | [−22.2, 0.0] | −11.1 | 0.0 |
| device-operator | 59 | +5.9 | +2.5 | −3.4 | [−15.3, +8.5] | −3.4 | −3.4 |
| heat-object-with-appliance | 32 | +15.6 | +14.1 | −1.6 | [−12.5, +9.4] | −6.2 | +3.1 |
| clean-object | 37 | −9.5 | −10.8 | −1.4 | [−16.2, +13.5] | 0.0 | −2.7 |
| locate-target-object | 2 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 |
| inventory-management | 2 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 |
| object-transporter | 1 | 0.0 | 0.0 | 0.0 | 无区间 | 0.0 | 0.0 |
| object-storer | 18 | −5.6 | −5.6 | 0.0 | [−16.7, +16.7] | 0.0 | 0.0 |
| object-picker | 7 | 0.0 | 0.0 | 0.0 | [−21.4, +21.4] | +14.3 | −14.3 |
| object-retriever | 15 | −3.3 | −3.3 | 0.0 | [−16.7, +20.0] | +6.7 | −6.7 |
| receptacle-navigator | 2 | 0.0 | 0.0 | 0.0 | [−50.0, +50.0] | +50.0 | −50.0 |
| receptacle-opener | 8 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 |
| object-disposer | 2 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 |
| appliance-preparer | 8 | 0.0 | 0.0 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 |
| receptacle-operator | 7 | 0.0 | 0.0 | 0.0 | [−28.6, +21.6] | 0.0 | 0.0 |
| tool-locator | 4 | −12.5 | −12.5 | 0.0 | [0.0, 0.0] | 0.0 | 0.0 |
| storage-explorer | 44 | −6.8 | −3.4 | +3.4 | [−9.1, +15.9] | +11.4 | −4.5 |
| object-cooler | 21 | +2.4 | +9.5 | +7.1 | [−9.5, +28.6] | +14.3 | 0.0 |
| appliance-navigator | 62 | −5.6 | +2.4 | +8.1 | [+0.8, +16.9] | +11.3 | +4.8 |
| temperature-regulator | 6 | +8.3 | +16.7 | +8.3 | [−25.0, +50.0] | 0.0 | +16.7 |
| receptacle-finder | 10 | −20.0 | −10.0 | +10.0 | [−10.0, +35.0] | +30.0 | −10.0 |
| object-state-inspector | 18 | +5.6 | +19.4 | +13.9 | [−11.1, +38.9] | +22.2 | +5.6 |
| object-placer | 23 | −15.2 | 0.0 | +15.2 | [−4.3, +34.8] | +13.0 | +17.4 |

完整 SkillNet-37 库中，以上 **25 种技能**在本次 U0 unseen 参考轨迹中有自然首次调用锚点，合计 **404 个锚点**。另 **12 种技能没有本次可评估锚点**，其 \(M_0/M_5/\Delta M\) 均应记为缺失，不能填零：environment-scanner、goal-interpreter、location-navigator、navigation-planner、object-locator、object-state-modifier、open-receptacle、receptacle-preparer、search-pattern-executor、search-verifier、task-verifier、tool-user。这不等于它们在训练中从未被调用。

## 标签精度与重复一致性

| 核查项 | 2-gold 结果 | 可作出的解释 |
|---|---:|---|
| 25 个有标签技能的合并点估计 | 下降 6、上升 7、恰为零 12 | 仅为当前锚点与两次续跑下的点标签。 |
| 两次 gold 的变化方向 | 相反且均非零 6；一零一非零 6；同向且均非零 4；均为零 9 | 两次续跑的方向存在明显不一致，尤其不能把合并零值自动解释为真实稳定。 |
| 原报告的区间 | 24 个可计算；23 个包含零；1 个严格为正；0 个严格为负 | 本次没有区间支持“确定下降”的单项结果；这也不是“没有真实下降”的证明。 |
| 区间宽度 | 可计算区间的中位数 31.53 pp | 典型单技能标签精度有限。 |
| 退化区间 | 6 个为 [0,0]；另 1 个只有一个 game，无法给出区间 | 离散的少量观测和有限重采样不能证明这些技能真实效用不变。 |

方向翻转的 6 个技能是：heat-object-with-appliance（−6.25/+3.13 pp）、object-picker（+14.29/−14.29 pp）、object-retriever（+6.67/−6.67 pp）、receptacle-navigator（+50/−50 pp）、storage-explorer（+11.36/−4.55 pp）、receptacle-finder（+30/−10 pp）。其中 `object-picker`、`object-retriever`、`receptacle-navigator` 的两次结果相互抵消，使合并点估计恰为零；它们尤其不能作为“效用稳定”的强证据。两次 gold 同向并不保证精确，例如 `device-operator` 两次均为 −3.4 pp，但其区间仍跨零。

这些观察说明的是**当前下游效用变化标签的测量不确定性**：同一 RL seed、相同锚点，续跑随机性和有限 game 支持就足以让若干技能的点标签改变方向。区间跨零是精度不足，不等于不存在 policy-update 效应；区间严格为正的 `appliance-navigator` 也只是未经跨技能多重比较校正的单项结果。不能据这批数据推断所有 37 种技能、其他 RL seed，或独立重采样状态分布下的方向准确率。

## 统计解释边界与复核状态

- 分析单位是**技能 × U0→U5 窗口**。锚点、game、两次 continuation 都不作为独立 RL seed；25 个效用标签中，只有 18 个还具备本窗口可比较的训练读出，不能将效用覆盖率等同于读出比较样本量。
- 区间条件于固定的 U0 锚点、U0/U5 模型及本次评估协议；不包含 RL 训练随机性、不同锚点分布或技能版本变化。多技能区间未作 family-wise/FDR 校正。
- 自然首次调用决定哪些技能进入表格，低支持技能的离散变化幅度很大。无锚点与无区间均保持缺失，不按零处理。
- 仅重新汇总已保存的 CSV 与逐锚点记录：两次 gold 的逐技能 \(\Delta M\) 均值与原汇总 `delta_utility` 一致（最大绝对差低于 \(10^{-15}\)）；没有重新执行 ALFWorld 或 bootstrap。因此状态为 `ANALYZED`，不是独立实验复现的 `VERIFIED`。
- 统计误用检查覆盖 **11/11 类**：Simpson 分层混淆（保留 game/阶段边界）、生态谬误（不下推至 token）、Berkson/选择偏差（自然调用限定样本）、collider（未新增按结果控制）、基率忽略（报告完整上升/下降/零计数）、均值回归（未按极端标签选技能）、幸存者偏差（12 个无锚点技能保留缺失）、look-elsewhere 与分析路径（不挑选有利技能或变式）、相关不等于因果（不从标签精度推断编辑收益）、反向因果（不从该统计表推断机制方向）。主要警戒是选择覆盖与有限重复，非通过统计检验宣布某方法优胜。
