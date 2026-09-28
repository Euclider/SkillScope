## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-22
- Verification Status: ANALYZED
- Version Label: seed404_reward_variants_exploratory_v2_constant_preserving

# seed404：reward-directed 读出扩展分析

## 1. 结果摘要

本轮完成 39 个奖励相关公式 × 3 种聚合，共 117 个 reward 分数；另有 117 个对应去奖励版本、
21 个幅度分数、6 个原有参考，总计 261 列。既包含 P 的变式，也包含直接使用
advantage × 采样动作 logprob 变化、截断概率比差、reward-only、C 等不要求依赖 P 的信号。
全部候选与方向在此次实际计算前锁定；但 seed404 历史结果已经可见，因此明确属于事后探索。

主要发现不是“找到了已确证更好的 D”，而是：

1. 直接去掉正部截断未改善 token 等权的下降排序；保留门控的 signed D，AP 从原 D 的 0.509650 降为 0.415995。
2. 奖励相关信号中最高 AP 是 decision 等权的中心化 C：0.693590；只比原 token 等权中心化 C 的 0.688095 高 0.005495。
3. 其相对同聚合中心化交互幅度的 AP 差为 +0.051368，但配对标签 bootstrap 的 95% 区间为 [-0.187348, 0.166667]。
4. 更换聚合同样改善无奖励基线：中心化交互幅度 AP 从 token 的 0.613651 提高到 game 的 0.724762，超过所有本次 reward 分数；原有 activation-l8 基线 AP=0.743333。
5. 带符号方法最高方向准确率为 9/12（75%），不是 18/18；对比恒预测上升为 7/12（58.33%）。这仍只是同 seed、多候选探索后的点估计。

因此，本轮发现了值得保留的奖励相关候选，但**尚不能据此确认“加入 reward 有稳健增益”**，
也不能声称中心化 C 对所有指标、预算或 control 均占优。没有追加新公式直到找到正面结果。

## 2. 数据与比较口径

- 固定 seed404、U0→U5、SkillNet-37、原 router、原 U0 训练 batch。
- 稳定读出为 155,638 条 token×control 记录，即每 control 77,819 个实际 loss token、5,511 个决策。
- 134 条 U0 unseen 来源轨迹；25 种技能有首调用效用，404 个首调用锚点。没有训练读出的技能继续记 NA。
- 主共同池为 18 技能；PLACEBO 0pp 点估计标签为下降 5、上升 7、不变 6。NULL 单独报告，不混池。
- 5 个下降标签的原配对置信区间均触及或跨越 0；它们是有不确定性的点估计标签，不是已确证下降。
- 只做 CPU 标量分析：初版约 42 秒、常数保护修正版约 43 秒；没有新训练、模型前向、ALFWorld 续跑或 API，也没有在 505/606 上试这些变式。

统计单位为 skill-window。token、同局反复调用、continuation repeats、phase 和 bootstrap draws 都不是额外独立 RL seed。
同一指标的 token/decision/game 版本不是独立实验；D_centered_gate 与原 D 等重复/高度相关候选仍保留，不将其当独立证据累加。

## 3. 关键对照（PLACEBO，all phase，0pp）

| 信号 | 聚合 | AP | 下降/其余 AUROC | 与连续下降量的 Spearman |
|---|---|---:|---:|---:|
| 原 D：g[-P]+ | token | 0.509650 | 0.630769 | -0.147161 |
| signed D：-gP | token | 0.415995 | 0.415385 | -0.066222 |
| 不带门控的 -P | token | 0.423810 | 0.461538 | -0.064120 |
| 中心化交互余弦 D | token | 0.432967 | 0.492308 | -0.029432 |
| advantage 强度加权 -P | token | 0.565797 | 0.523077 | 0.163979 |
| -A × 采样动作交互变化 | token | 0.402197 | 0.384615 | 0.249123 |
| 中心化 C | token | 0.688095 | 0.723077 | 0.023125 |
| 中心化 C | decision | 0.693590 | 0.738462 | 0.042046 |
| 负投影比例 | game | 0.647222 | 0.738462 | 0.120882 |
| signed D，中心化 C 绝对值 Q25 门控 | game | 0.592222 | 0.600000 | 0.377363 |
| 无奖励中心化交互幅度 | token | 0.613651 | 0.738462 | 0.348982 |
| 无奖励中心化交互幅度 | decision | 0.642222 | 0.800000 | 0.406795 |
| 无奖励中心化交互幅度 | game | 0.724762 | 0.800000 | 0.362647 |
| 原 activation-l8 范数参考 | 原口径 | 0.743333 | 0.876923 | 0.310089 |

这是解释性摘录，不是只保留赢家。完整 39 公式 × 3 聚合、所有去奖励版本、control/phase/阈值见机器表。
C 以原策略更新 u_O 为基础，而交互幅度以 delta=u_O-u_control 为基础；两者的幅度比较不能冒称完全同源的 reward 消融。
本轮另列每个公式的单位 advantage 对照和 trajectory-block 符号负对照，正是为了区分这个问题。

## 4. reward 本身带来了多少信息？

中心化 C、decision 聚合：

| AP 对比 | 点估计差 | 配对标签 bootstrap 95% 区间 |
|---|---:|---:|
| 相对原 D | +0.183939 | [-0.104695, 0.245959] |
| 相对同聚合中心化交互幅度 | +0.051368 | [-0.187348, 0.166667] |
| 相对同公式、固定原方向支持、A=+1 | +0.482861 | [-0.324829, 0.589677] |

去奖励后的中心化 C AP=0.210729，而 reward 版为 0.693590；这是本 seed 有价值的描述性差异。
但 P/C 的 A=0 行没有保存足够的无奖励几何，故这类对照固定原非零方向支持，
只移除该支持上的 reward 符号/强度，不能声称完全去除了 reward 所决定的支持结构。
直接采样动作类 A=+1 对照则可以覆盖全部原 token。

512 次 trajectory-block 符号随机化中，中心化 C/decision 的 AP 均值约 0.425735，
约 8.59% 的随机化达到其实际 AP；如果每次允许在全部 117 个 reward 分数中取最高 AP，
约 77.73% 的随机化最高值达到该水平。后一个数字直接说明大规模试公式的择优风险。
二者都是条件负对照比例，不是有交换性保证的因果/确认性 p 值。

新增 2,000 次全局 game/continuation 配对 bootstrap，两种 control 各有 1,674 次保留完整共同池；
326 次因稀有技能没有抽中 game 而整体记缺失，不为不同方法缩小池子。0pp AP 有 1,671 次可计算，另 3 次没有下降事件。
区间条件于当前训练读出、完整池覆盖和本次技能集合，不包含跨训练 seed 或模型更新不确定性，也未校正候选选择。

## 5. 排序不等于方向分类，也不等于所有预算占优

符号判断只对 12 种非零点估计变化技能计算；零分弃权计错，同时单列覆盖率。

| 带符号读出 | 正确下降/5 | 正确上升/7 | 总准确率 | 平衡准确率 |
|---|---:|---:|---:|---:|
| signed D，token | 2 | 3 | 5/12 = 41.67% | 41.43% |
| -A × 动作交互变化，token | 1 | 7 | 8/12 = 66.67% | 60.00% |
| sqrt(abs(A)) 加权 -P，game | 3 | 6 | 9/12 = 75.00% | 72.86% |
| Q25 中心化 C 门控 signed D，game | 3 | 6 | 9/12 = 75.00% | 72.86% |
| winsorized -P，game | 3 | 6 | 9/12 = 75.00% | 72.86% |

这些方法的同一组混淆计数不是三次独立复现。原非负 D 和作为风险分数使用的 C 不拿正负号充当双向分类。

中心化 C/decision 的 Top-2 命中 2/2，覆盖 71.16% 点估计下降量；
但中心化交互幅度/game 同样做到 2/2 和 71.16%。Top-5 时 activation-l8 命中 4/5、覆盖 92.85%，
高于中心化 C/decision 的 3/5、78.31%；Top-9 时中心化幅度/game 覆盖 92.85%，也高于中心化 C 的 78.31%。

5pp 阈值下只剩 2 个下降技能，中心化 C/decision 和中心化幅度/game 的 AP 都为 1。
这不是 reward 独有收益，不能选择 5pp 而隐藏 0pp 或很少的事件数。
NULL 下中心化 C/decision AP=0.583712、Spearman=-0.120882；game Q25-gated signed D AP=0.577094。
同一公式在 control 间的优势并不完全稳定。

## 6. 具体错误与后续边界

中心化 C/decision 将 receptacle-closer、object-heater 排在前两位，其 ΔM 点估计分别为 -10pp、-5.56pp；
但第 3 位 object-placer 实际点估计为 +15.22pp，属于下降排序中的假阳性；
clean-object（-1.35pp）和 device-operator（-3.39pp）分别在第 12、13 位。
这解释了 AP 尚可而连续下降量 Spearman 仅 0.042 的并存，不能只展示前两名。

后续若要验证 reward 增量，应先冻结少量候选、聚合及参考，再在独立窗口/seed 检查，
不能逐 seed 从这里选不同赢家。当前无需重跑 RL 或效用；本轮没有自动改写 Phase3 的主排序指标。

## 7. 文件、验证与归档

最终结果使用 v2。v1 成功完成后的负对照检查发现：理论上恒为 −1 的单位 advantage / reward-only 对照，
经过加权聚合后有末位浮点差异，造成伪排序。因此另建 v2，仅对所有输入严格相等的聚合列保留精确常数；
所有非恒定列、候选公式、标签、阈值与评估范围不变。共修正 366 个标量单元；主分析 117 个 reward 分数的 AP 均未改变。
主分析中三种聚合的该无奖励常数对照现为 AP=5/18=0.277778、AUROC=0.5、Spearman=NA。
v1 全部输出及初版解释稿保留，不能再用其中常数负对照的伪排序。详细修正规则见
[常数保护登记](SkillRL/docs/experiments/phase12-independent-v4/REWARD-VARIANTS-CONSTANT-20260922-v2.md)。

- [完整扩展报告 v2](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/reports/phase2-results-expanded-v2.md)：保留旧数值报告正文并附本次全家族分析及修正说明。
- [公式登记与聚合规则](SkillRL/docs/experiments/phase12-independent-v4/REWARD-VARIANTS-SEED404-20260922-v1.md)、[候选全集](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/registry.csv)。
- [全部分层指标](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/ranking_diagnostics.csv)、[全部编辑预算](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/ranking_budgets.csv)。
- [逐技能分数与效用](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/skill_scores_and_gold.csv)、[原完整覆盖与区间](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/coverage_and_effects.csv)。
- [配对 reward 增益](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/paired_reward_gains.csv)、[方向混淆计数](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/direction-confusions.csv)、[符号负对照](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/sign_null_summary.csv)。
- [独立验收](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/independent-verification.json)：5,220 行 AP/AUROC/方向 AUROC/Spearman/Kendall 独立核对，9 个原指标各 20 行与旧报告一致，最大误差约 3.33e-16。
- [287 项 CPU 测试](SkillRL/artifacts/code_checks/reward-variants-s404-20260922-v1/constant-fixed-regression-tests.xml)。开发中 3 次合成测试失败记录保留；初版与数值修正版各成功执行一次，没有失败后自动重试。
- [常数修正逐单元对照](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/constant-correction-cells.csv)、[最终发布验收](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2/publication-verification.json)。

统计误用检查覆盖 11/11：重点为自然支持的选择边界、事件基率、小样本、多公式择优、阶段依赖、
相关不等于编辑因果收益，以及已见标签后的探索不能包装成前瞻验证。
原 125 个冻结运行源码及 19 个输入文件逐哈希保留；原数值报告和历史实验结果未覆盖。
