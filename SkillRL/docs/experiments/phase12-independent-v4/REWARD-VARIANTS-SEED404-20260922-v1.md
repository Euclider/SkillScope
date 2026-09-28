## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan / run / validate
- Origin Date: 2026-09-22
- Verification Status: UNVERIFIED (本文件是执行前登记，结果状态另见完成回执)
- Version Label: reward_variants_seed404_exploratory_v1

# seed404：reward-directed 读出扩展（事后探索）

用户授权扩展 D 的变式，不要求依赖 P，在 seed404 评估后扩充分析报告。
本轮只增加独立 CPU 分析入口和输出，不改已冻结的 D、训练/评估代码、旧报告或 Phase3 配置。
不启动 RL、模型前向、ALFWorld rollout、外部 API、505/606 分析或 Git 发布。
先完成代码与合成测试，登记源码/输入哈希、全部候选，再一次计算实际结果；不是在结果出现后继续加公式。
404 和 505 的历史结果均已可见，因此这是探索，不声称前瞻预登记或“证明了增益”。

## 1. 问题与不变项

问题：同一 U0→U5 窗口、固定状态与技能下，奖励相关的读出能否提供优于无奖励幅度或去奖励对照的效用变化方向/下降排序信息？
输入为 `numerical-readout-release-v1/seed-404` 的稳定 FP64 标量、已存在的 game/continuation 配对效用标签。
U0/U5、37 技能、冻结路由、训练 batch、动作、advantage、两种 control、首调用锚点及实际效用均不改变。
主标签为 PLACEBO 的 ΔM；NULL 是单独报告的敏感性分析。0 和 5pp 点估计阈值均保留。
主比较池严格复用上一版每个 control/context/phase 的共同池；无新数量筛选，不填补未观测技能。
18 个主池技能不是 77,819 个独立统计样本；阶段、token、continuation 和 bootstrap draws 不是独立 RL seed。

## 2. 候选全集

`skillnet_cohort/reward_variants.py:RECIPES` 为唯一候选枚举。
包括负部/带符号/符号平衡，advantage 绝对值和平方根强度，未归一化方向内积，
原始/中心化交互余弦，连续 C 权重/sigmoid，固定分位数门槛，winsor/tanh，
采样动作 logprob 的 reward 符号/强度/截断强度，截断概率比差，原策略单臂读出，以及 reward-only 弱对照。
原始 C、中心化 C、advantage 加权 C 是奖励相关参考，绝不是无奖励 baseline。
所有符号在执行前固定，不根据真实 ΔM 翻转。

每种候选均计算 token 等权、decision 等权、game 等权三种聚合。
game 聚合为 token→decision→trajectory→game 的逐级平均，技能与阶段内独立归一化；
未过门控和 A=0 的 token 仍留在分母。不是累加调用次数，不把实际 readout 改成首调用。

数值参数：epsilon=1e-12、原始 norm guard=1e-8；C 分位门槛为 pooled |C_centered| 的 25/50/75%，
交互幅度门槛为 pooled ||delta_centered|| 的 25/50/75%；按 control、原非零方向行计算并固定，
不按技能/阶段/效用标签拟合。sigmoid 温度=median|C_centered|，tanh 尺度=median|P|，winsor cap=Q95|P|。
这些是明确登记的探索变式，不冒称数值噪声校准。advantage 截断为 [-2,2]；概率比截断 [.8,1.2]。
后者为单步概率比差代理，不等同 PPO 损失或真实技能效用的无偏估计。

## 3. reward 增益对照

1. 相同聚合的 delta 原始/中心化范数、O/对照更新范数、KL、JS、采样动作变化绝对值；旧 activation 和 old-margin 参考也保留。
2. 每个公式的 A=+1 配对版本。采样动作类能在全部 token 真正移除 reward；
   P/C 几何类只可在原 A!=0 支持上从标量恢复单位优势向量，A=0 没有保存足够几何信息。
   这类称“固定原支持、去 reward 符号/幅度对照”，不谎称完全无 reward 的全集对照。
3. 512 次 trajectory-block Rademacher 符号随机化：每条轨迹的所有决策/token、两种 control 同乘 ±1，
   保留 |A|、局内形状、原支持和几何，重新计算 P/C/门控。所有候选共用同一次随机化。
   不逐 token 打散，不假设 advantage 在轨迹内恒定。随机化分布和跨候选最大 AP 均记录；
   这是条件负对照诊断，不具有随机分配实验的交换性保证，不报告成因果 p 值。
   分位数尺度因只依赖绝对值，符号随机化时固定。

## 4. 评价与不确定性

完整报告 AP、下降对其余 AUROC、下降对上升 AUROC、风险方向 Spearman/Kendall、
Top-1/2/25%/50% precision/recall/captured decline mass；并列使用期望选择，不用真实标签破平。
带符号方法另报告非零变化上的符号准确率、弃权数、覆盖率与将弃权计错的准确率；
非负 D/C 风险排序不冒充双向分类器。零下降或单类 AUROC 保持 NA。

追加 2,000 次全局 game-cluster/continuation 配对 bootstrap，仅考察 gold-label 抽样不确定性，固定训练读出。
同一次 draw 对各技能共享 game 抽样、对每个抽中的 game 共享 continuation 抽样；所有 arm/endpoint 对比先配对。
若某个技能在 draw 中没有任何 game，整次共同池比较记缺失，不缩小候选池；报告有效 draw 比例。
指标及相对原 D、同聚合中心化范数和同公式去奖励版的配对差异给出 percentile 区间。
区间不包含训练 seed/读出估计误差，不对多公式择优校正，不作为确认性显著性声明。
完整旧标签和原有逐技能区间不被替换。

## 5. 产物与验收

新根：`artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v1`。
输出 registry/plan/scales/prediction snapshot、完整逐技能分数、全分层/预算指标、配对增益、
bootstrap/符号负对照、覆盖表、扩展 Markdown 和 provenance/complete 回执。
入口为 `python -B -m skillnet_cohort.reward_variant_analysis prepare|run --output ...`；
仅 CPU，CUDA_VISIBLE_DEVICES 为空，BLAS/OMP 1线程；无时间预算扩张或 GPU 任务。
进程存活与 stdout 阶段进度由执行工具监测，异常保留，无自动重试。

验收：所有候选有限且同池；原 D/−P/C/KL/JS/范数与稳定版一致；手工公式、聚合权重、
原/负对照/符号随机化、并列/零事件、配对 bootstrap 与 no-clobber 单测通过。
所有输入/旧源码/旧报告哈希执行前后不变；新输出可独立重新核算。
原数值报告保持封存，另建扩展版 `phase2-results-expanded.md` 包含旧正文和新分析；
研究根新增入口文档并在 HANDOFF 中追加，便于后续 summary，不破坏旧 provenance 链。

## 6. 统计误用检查（11/11）

Simpson：control/phase 分开；生态谬误：skill-window 单位不下推 token；Berkson：同池且全 coverage；
Collider：不以结果筛支持；基率：每表事件数/候选数；均值回归：不按极端 ΔM 选技能；
幸存者：完整输入/缺失保留；多重检验：候选全集与 max-null；分析路径：事后探索及固定枚举；
相关非因果：排序不等于编辑收益；反向因果/泄漏：公式不读 gold，但历史标签已见，不伪装前瞻验证。
