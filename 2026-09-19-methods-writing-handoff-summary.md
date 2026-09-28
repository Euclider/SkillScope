# 方法部分写作交接：定义、实验设置、证据边界与文件索引

## 0. 材料身份与使用范围

本文件是交给写作 agent 的材料汇总，不是论文正文，也不是新增实验报告。依据用户最新讨论、现有文档、冻结配置及关键实现整理；不启动/重跑实验，不改现有代码、配置、报告或 Git 状态。正文使用中文说明，保留代码字段和必要公式以减少转述歧义。

Material Passport：

- `origin_skill`: academic-research-suite / academic-paper
- `origin_mode`: materials-handoff；只做材料接收、来源整理和证据分层，不执行论文起草流程。
- `origin_date`: 2026-09-19 UTC
- `version_label`: methods_handoff_v1
- `verification_status`: UNVERIFIED。已做文件/配置/公式的只读核对；不表示科学结论、引用准确性或新实验全流程通过独立验证。
- `repro_lock`: null。本文件不是对正在运行的工作区创建的不可变全仓库快照。
- 新实验完成标记核对时间：2026-09-19 05:35:24 UTC。后续主线程可能继续更新；本文件不承担实时监控。

路径约定：

| 简写 | 当前本机路径 |
|---|---|
| R：研究根 | `/mnt/workspace/users/wangyifan/skill-RL` |
| C：代码根 | `/mnt/workspace/users/wangyifan/skill-RL/SkillRL` |
| Q：当前 Phase1–2 队列 | `C/artifacts/phase12/skillnet37-independent-s404-505-606-v4` |
| L：历史 Phase2 单步 cohort | `C/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1` |
| B：历史 U35 新锚点准备 | `C/artifacts/phase2/qwen35-clean-s303-u35-ranking-preparation-v1` |
| A：历史已完成 U35→U40 cohort | `C/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1` |

这些是文档缩写，不是 shell 环境变量。历史文件内的 `/home/wangyifan/skill-RL` 通常需要映射为当前 R；不要照抄旧环境激活或启动命令。本文不收录凭据；第三方 API 密钥、聊天内出现过的密钥均不得进入论文或公开包。

### 0.1 写作 agent 的最短阅读顺序

1. 本文 §1–5：当前研究命题、效用定义、C/P/D 和窗口协议。
2. [当前独立窗口协议](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/experiments/phase12-independent-v4/README.md)及本文 §6–8：实际新实验设置，不用早期 150 轮方案代替。
3. [Phase3 SETTING](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/phase3/SETTING.md)及本文 §9：已确认但尚无正式闭环结果的设计。
4. [历史 Phase1 三 seed 报告](/mnt/workspace/users/wangyifan/skill-RL/2026-09-04-qwen35-clean-all-skill-three-seed-utility-results.md)、[Phase2 synthesis-v2 完整数据报告](/mnt/workspace/users/wangyifan/skill-RL/2026-09-14-phase2-complete-analysis.md)、[后续无日期综合稿](/mnt/workspace/users/wangyifan/skill-RL/phase2-complete-analysis.md)及本文 §10：哪些事实已经测过，哪些尚未测过；最后两份不是同一版本。
5. [原始 idea](/mnt/workspace/users/wangyifan/skill-RL/2026-08-19-policy-update-skill-sign-flip-forecasting-design.md)：用于追溯动机和公式，不把其早期全部设想当作当前方法或已执行实验。

来源使用规则：方法意图优先遵循用户最新确认；“实际怎样计算/运行”以对应 cohort 的冻结配置、实现及产物为准。两者不一致时明确标注，不让旧文档、模板默认值或未执行方案覆盖新设置。

## 1. 当前研究问题，以及不再要求什么

### 1.1 用户最新摘要/intro 所对应的主线

已有 skill evolution 方法通常依据执行轨迹及成功/失败反馈筛选、修订技能。但即使技能文本不变，policy 更新也可能改变技能对行为与边际效用的影响；旧轨迹归因可能不再代表新 policy 下的贡献。轨迹失败还有多重来源，不能直接把失败等同于某个被调用技能有害。

本项目要检验的是：**真实 policy 更新产生的信号，是否包含技能边际效用变化的预测信息；把已有训练 batch 的 advantage 方向加入读出，是否比同源的纯幅度读出更能优先定位效用下降技能。**

- “预测”的时间点是 policy 已更新、目标 post-update utility gold 尚未读取；不是更新发生前预测未来参数。
- 读出重用训练时的状态、动作 token、mask、advantage，进行新旧 policy 的固定输入前向评分。
- **读出本身不需要新的 post-update 环境 rollout**；用于证明预测准确性的 O/P/N 效用标签仍需续跑，Phase3 编辑验收也需环境评估。这三种成本不能混称为零。
- 方法首先是预测/诊断，不是技能失效机制的证明，也不是完整因果 credit assignment。

最新讨论已将早期“必须正效用→负效用翻转”的目标放宽为效用变化、方向信息和下降优先级；不要求每个技能发生翻转、每个 seed 同向、每个窗口 Top-1 正确，也不要求拟合效用数值回归器。

### 1.2 三层命题要分别检验

| 层次 | 对应比较 | 不能偷换成 |
|---|---|---|
| 现象 | 同技能、同锚点的旧/新 policy 边际效用是否变化 | 全局成功率改变就证明每个技能效用改变 |
| reward-directed 信号族 | C 对同源 `u_original_norm`；P/D 对同源 `delta_norm` 等 | 只有 D 全面胜过 C/P 才算有预测信息 |
| 更具体的构造增量 | interaction 差分、投影、负部、门控分别是否有价值 | C 表现好就证明 interaction 或 D 的独特价值 |
| 闭环实用性 | 同编辑器与资源上限下，不同选择/证据策略导致的最终性能 | 离线排序好就已证明编辑收益 |

用户聊天内提供的英文摘要/intro 是研究叙事与待验证主张来源；本次未定位到与该段完整对应的独立 `.tex`/`.bib` 稿件。不要把其 “Our experiments show” 或红字 Phase3 待验证句直接当作新 cohort 的已完成结果。

## 2. 实验对象、分析单元与两组固定状态

Policy 为 Qwen3.5-4B。技能是外部可复用文本指导，不是可执行宏动作，也不是 policy 参数。对一个窗口记旧/新 policy 为 `π⁻=π_q`、`π⁺=π_(q+h)`，当前 `h=5`；窗口内技能内容版本固定。

分析中有两类状态，必须区分：

| 状态来源 | 用途 | 是否额外环境交互 | 新旧 policy 如何比较 |
|---|---|---|---|
| 窗口起点生成的第一轮实际训练 batch，记 `B_(q+1)` | C/P/D 等读出 | 复用已有 RL rollout | 相同 observation/history、所选技能、实际动作及 teacher-forced token 前缀 |
| 窗口起点 policy 在指定 evaluation split 的自然调用轨迹所构造的首次调用 anchors | 独立 O/P/N utility gold | 是，包含锚点采集和配对续跑 | 同 game、同调用前 prefix、同 continuation seed，在两端点分别续跑 |

二者并非逐状态一一对应；当前预测评价主要在 `skill × window × context × phase` 层面关联训练读出和评估效用。新设置的主 context 是 `all_alfworld`，所有 task/game 使用同一个库；任务类型仍用于覆盖和成功率分层，不用于候选库硬过滤。

环境 decision 是一次文本动作；模型读出中的基本 `j` 是该动作响应中实际参与 loss 的 **token**。全词表打分不是在环境 admissible actions 列表上直接做有限动作分类。padding/分布式补齐重复行须剔除；词表维度为 248,320，非 top-k 概率近似。

## 3. 语义边际效用与配对干预

### 3.1 首次自然调用锚点

从起点 policy 自然到达的轨迹中，对每个目标技能取该轨迹的第一次调用；保存调用**前**的 game、动作 prefix 和相关状态。评估时先重放相同 prefix，再进入目标条件自由续跑，不强迫整个后缀动作相同。

| 条件 | 目标 skill 的 payload | 不变的部分 |
|---|---|---|
| ORIGINAL，O | 原始完整技能正文及其附属文本包 | 候选 ID/描述、router 规则、非目标技能、其余环境/解码协议 |
| PLACEBO，P | 任务无关中性文本，匹配规定的外层包装及 token 长度 | 同上；不是替换为另一条技能再检索 |
| NULL，N | 目标 payload 为空 | 目标候选 ID 仍存在；不是删除候选或关闭全库 |

目标技能若在自由后缀中再次被路由到，持续使用同一个干预臂。路由选择先于目标正文替换；轨迹分叉后到达不同状态，冻结 router 仍可能自然选择不同技能。**固定 router 函数不等于固定整条轨迹的未来调用序列。**

当前 SkillNet PLACEBO 的精确定义：保留 `### Frozen Skill: <ID>` 首行包装及匹配的尾部空白，用关于 typography 的固定无关文本填充；使用实际 Qwen tokenizer 校验独立 payload 及两种真实 prompt 后缀边界的 token 数。**不保证复制原技能内部 Markdown 布局**，也不能笼统声称已去除 ID/包装携带的一切语义。文件：[assets.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/skillnet_cohort/assets.py)，每技能内容见当前 preparation 的 `placebos.json`。

### 3.2 目标量

令 `Y_π^O(a,ξ)` 等为同一 anchor `a`、continuation seed `ξ` 下的二值最终成功结果，则：

\[
M_{\pi}^{\rm sem}(s)=\mathbb E_{a,\xi}[Y_\pi^O(a,\xi)-Y_\pi^P(a,\xi)],
\qquad
\Delta M^{\rm sem}(s)=M_{\pi^+}^{\rm sem}(s)-M_{\pi^-}^{\rm sem}(s).
\]

`O−NULL` 是次要对照，不因结果好坏切换主 estimand。建议同时保留 `Δsuccess_O` 与 `Δsuccess_control`，因为：

\[
\Delta M=\Delta\mathrm{success}_O-\Delta\mathrm{success}_{control}.
\]

因此 ORIGINAL 成功率提高时，相对效用仍可能下降；这不等于 policy 绝对退步。此处的 control 是**目标 payload 控制**，不是整个系统的 skill-free policy。早期 idea 的 `B/W/M` 可以辅助理解，但不能把当前 O−P 直接改写成“全库有/无”的收益。

统计规则：先在每个 game 内平均其 anchors/continuation repeats，再 games 等权；报告 `100×M`、`100×ΔM` 为百分点 pp。环境成功奖励 10、失败 0，纯终局 return 差为 success 差的 10 倍；不要把训练的 invalid-action 惩罚也当作 gold success 定义。

95% CI 用 10,000 次配对 game/continuation bootstrap：先形成 O/P、旧/新成对差，再按 game 重采样；多 continuation seeds 时在 anchor 内成对重采样重复。不是把所有 suffix 当独立训练重复。实现：[utilities.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase2/utilities.py)。

## 4. Reward-directed readout：以实现为准的公式

核心来源：[direction.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase2/direction.py)、[measure.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase2/measure.py)、[aggregate.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase2/aggregate.py)。

### 4.1 四个对齐分布与 interaction

对训练 batch 的有效响应 token `j`，固定原始状态与实际已生成 response prefix，分别对旧/新 policy、ORIGINAL/control 进行 teacher-forced 前向。`ℓ` 是整个词表上的 log-softmax 概率向量，按实际训练 temperature 评分；本次为 1。

\[
u_j^O=\ell_j^{+,O}-\ell_j^{-,O},\qquad
u_j^{ctl}=\ell_j^{+,ctl}-\ell_j^{-,ctl},\qquad
\delta_j=u_j^O-u_j^{ctl}.
\]

主 control 为 token-matched PLACEBO；Phase1–2 同时测 NULL 敏感性。各条件重放**同一原始 action token 序列**，不是先分别生成动作再比较不同 token 位置。

`δ` 是函数空间的差分之差，扣除该控制条件下的共同 policy 变化；它是候选预测特征，不自动等于真实长期 utility change。

### 4.2 从实际 advantage 构造局部方向

\[
p_j^{-,O}=\exp(\ell_j^{-,O}),\qquad
d_j=\hat A_j\bigl(e_{a_j}-p_j^{-,O}\bigr).
\]

`a_j` 是实际训练 token，`Â_j` 必须来自真实优化 batch；不使用后来评估的 success 或 gold 标签替换。内积权重 `W=I`；全词表数据以 FP32 log-prob 运算，范数/点积等归约使用 FP64。

`d` 是 outcome-consistent 的局部 logit 方向，不是完整的参数梯度；未包含 Adam、clipping、KL、entropy 等对真实参数更新的全部影响，也不是 PRM/逐步正确性标签。

复现注意：当前 GRPO 实现由 `ray_trainer.compute_advantage()` 调用 `core_algos.compute_grpo_outcome_advantage()`，沿用 `compute_mean_std_cross_steps=True` 默认值；组内均值/标准差在展开的 step rows 上计算，而非另行重写为只对 8 个 episode 标量的去重平均。最终 token advantage 由 response mask 广播/约束。写方法时优先说“使用实际 GRPO batch advantage”，若展开 RL 公式须与此实现一致。

### 4.3 C、P、D

取 `ε=10⁻¹²`。在 `||d_j||>ε` 且 `||u_j^O||>ε` 时：

\[
C_j^{upd}=\frac{\langle d_j,u_j^O\rangle}{\|d_j\|\|u_j^O\|},
\qquad
P_j^{int}=\frac{\langle d_j,\delta_j\rangle}{\|d_j\|+\epsilon}.
\]

C 衡量实际 ORIGINAL 更新与 reward direction 的一致性；P 的正/负表示 interaction 沿该局部方向增强/削弱。数值实现另对 C 的分母乘积设置 `clamp_min(ε)`。无效 C 在 token 统计中记 0，并另存有效覆盖率；这不等于技能层面支持不足可以填 0。

主 D 为：

\[
D(s)=\frac{1}{N_s}\sum_{j\in J_s}
\mathbf1(\|d_j\|>\epsilon,\|u_j^O\|>\epsilon)
\mathbf1(C_j^{upd}\ge0)
\mathbf1(\|\delta_j\|\ge\tau_\delta)
[-P_j^{int}]_+.
\]

- `J_s` 是所选 skill/context/phase 的全部实际 response loss tokens，`N_s=|J_s|`。
- **零 advantage、未过门控的 token 仍在分母中**，相应 D 贡献为 0；不是只对负向或通过门控样本求条件均值。
- `τ_C=0`；`τ_δ=max(10⁻⁸,10×同 checkpoint 重复前向 L2 噪声 p95)`，在目标 gold 前校准并冻结。历史两个 cohort 实际均为 `10⁻⁸`，新 cohort 不能预先当作已测到同样噪声。
- 主 skill 聚合为 token 等权平均。另有 decision 等权 D 敏感性，不替代主分母。
- 不另设技能级 `D>d₀` 风险门槛；支持充分者按连续分数排序。

无门控版本：

\[
D_{ungated}(s)=\frac1{N_s}\sum_{j\in J_s}\mathbf1(\|d_j\|>\epsilon)[-P_j^{int}]_+.
\]

它移除 C/norm gate，仍保留方向有效性、负部截断、相同分母；**不是 `−mean(P)`**。`mean(P)>0` 与 `D>0` 可以同时成立。D 是非负单侧风险，不是 signed ΔM 估计，D=0 不证明稳定或提升。

### 4.4 比较分数与解释

| 字段/分数 | 定义或作用 | 当前下降风险方向 |
|---|---|---|
| `D_contribution` | gated D，主方法 | 越大越优先 |
| `D_ungated_contribution` | 无 C/norm gate 的负部投影 | 越大越优先 |
| `P_int` | 有符号 interaction 投影的 token 均值 | 使用 **−P** |
| `C_upd` | 原始 `cos(d,u_O)` 均值；不含 control interaction | 当前新实验预登记 **+C** |
| `C_upd_centered` | 对 `u_O−mean_vocab(u_O)` 计算 cosine | 历史辅助审计；不是 Phase3 的 C 分支 |
| `delta_norm` | token 级 `||δ||₂` 的均值 | + |
| `delta_centered_norm` | token 级 `||δ−mean_vocab(δ)||₂` 的均值 | + |
| `u_original_norm` / `u_control_norm` | 各自更新向量范数均值 | + |
| `forward_kl_original` / `js_original` | O 条件旧/新分布的 forward KL / JS | + |
| `activation_l{8,16,24,32}_norm` | 对预定层 hidden states 做同类 interaction 差分，取范数 | + |
| `old_margin` | 起点 evidence seeds 得到的旧语义效用 | 当前低效用优先，即 −old_margin |
| `random_expected` | 无差别排序的期望命中 | 全部同分 |

公平的“同源幅度”比较：C 对 `u_original_norm`；P/D 对 `delta_norm`，centered interaction norm 是中心化敏感性。**C_centered 不能拿 centered interaction norm 冒充它自己的同源范数**；当前汇总没有保存 `norm(centered u_original)` 作为独立字段。

`+C` 是预先固定的比较方向，不表示“更新越符合 reward，技能越有害”的理论定理。C 属于 reward-directed 信号，不是 magnitude-only baseline。历史 C 的若干排序结果来自事后审计；不能追溯升级为当时预注册的主比较。

## 5. 窗口、支持门槛、锁分与排序评价

### 5.1 相邻窗口，不固定永久 U0

一般协议：

```text
π_q + 固定库 K_q
  ├─ 第一轮真实训练 batch B_(q+1)：状态、动作 token、advantage
  ├─ 正常进行 q+1 ... q+5 的 RL
  └─ π_(q+5) 在同一 B_(q+1) 上评分
          → C/P/D → 锁定风险排序 → 后续 utility gold 或 Phase3 编辑
```

`U0→U5` 是五次 rollout/update **迭代**、两个比较端点，不是五个快照间隔共 25 轮，也不等于五次 `optimizer.step()`。若运行 100 迭代、每 5 轮一个窗口，则有 20 个相邻窗口、包括 U0 在内的 21 个边界；U5→U10 使用 batch U6，U10→U15 使用 batch U11。终点总位移投影到起点 batch 的方向，**不是五个单步 D 求和**，不假定中间 reward direction 不变。

当前 Phase1–2 实际确认的是 404/505/606 三条**独立 U0→U5**，不是一条 15 轮或 100 轮路径。Phase3 才是各分支持续 RL、按相邻窗口滚动编辑。

### 5.2 自然支持和 abstention

读出资格要求同一 skill/context/phase 至少：20 个非零方向支持 decisions、4 个非零支持训练 games、8 条非零支持轨迹。实现以 `direction_valid` 检查而不是仅靠被调用次数。记录 token/decision/game/trajectory 数、direction coverage、gate coverage、排除原因。

另有 **gold anchor 资格**：当前先要求 ≥30 个轨迹级首次自然调用记录、≥10 games，再按预登记规则选择最多 12 个技能，每技能最多 12 anchors。它与训练方向支持是两道独立门槛；有 gold 不代表有可用读出。库中未自然支持或未入预算的技能不得写成“效用为零/低风险”。

`initial=step0`，`early=1–4`，`middle=5–14`，`late≥15`，另有 `all`。主比较为 all；阶段切片与 all 重叠，不能当独立样本相加。

### 5.3 锁分与统计口径

- 目标端点 gold 打开前，锁定窗口、score 符号、支持池、风险列表、预算；新实验不拟合 predictor，不根据 gold 改用 C/P/D 中表现最好的一个。
- 当前比较每个 window/context/phase 内相同自然支持、所有所比较字段与标签齐全的候选池；报告被排除 skill 和原因。
- 下降点估计标签：`−ΔM > τ`，`τ∈{0,0.05}`，计算容差 `10⁻¹²`；第二个阈值是 5 pp，不是 5%。另报告 `|ΔM|` 的 any-change 审计，不混入下降排名。
- 指标：AP（标准 average precision）、AUROC（decline vs rest）、Spearman/Kendall 与 `−ΔM` 的相关；`P@k/R@k`，`k=1,2` 及候选数的 25%/50% 向上取整。
- 下降量覆盖率为 `Σ_selected max(0,−ΔM) / Σ_pool max(0,−ΔM)`；它不同于下降事件召回。
- Phase1–2 Top-k 同分按“同分组内均匀随机排列的期望选择权重”处理，不看 gold 打破平局。Phase3 必须实际选 ID，改用分数降序后 skill ID 升序的确定性规则。
- 没有下降时 AP/Recall/下降量覆盖不定义；AUROC 要有两类，相关要有变化。报告候选数/事件数，不把缺失记成性能 0 或 1。
- `direction` 的保守 CI 字段沿用 `τ_M=.05`：要求整个 CI 超过 ±5 pp 才标方向；“CI 排除 0”“点估计下降”“至少下降 5 pp”是三个不同口径。
- 新报告另记录 P 的非零点估计符号一致率，但不把相关系数直接叫方向分类准确率。
- 配对效用 CI 的抽样单位是 game/continuation；新跨 seed 汇总先逐 seed 展示同池结果。当前汇总器不自动提供经过独立 RL seed 层级推断的普遍显著性结论。

排序实现：[ranking.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase2/ranking.py)；前瞻锁分：[window_forecast.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase2/window_forecast.py)；新 AP/AUROC 汇总：[reports.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/skillnet_cohort/reports.py)。

## 6. 冻结初始库与独立 router

### 6.1 SkillNet-37：复用资源，不声称复现整个 SkillNet

| 项目 | 当前设置 |
|---|---|
| 来源 | SkillNet 的 ALFWorld benchmark 目录，不是 SkillNet 平台总技能量 |
| 上游 commit | `5c472b36d2a435001fdae3bc8439886d8050645a` |
| bank ID | `skillnet-alfworld-37-5c472b36d2a4` |
| 内容 | 37 个主技能文件、45 个附属文件、1 个 MIT LICENSE；83 个原文件逐字节保留 |
| manifest SHA-256 | `0767ff7578b1e997119a36b5f636fec6ebc1d0ac600ffb40b1e599065b7fe514` |
| 稳定 ID | `skillnet:<上游技能目录名>` |
| 候选集合 | 所有 task/game 均可检索完整 37 项；无 task 专属子库 |
| actor 可见内容 | 每步选一个 ID，注入该技能完整 SKILL.md 和按路径排序的附属文本；不注入 37 项的全部正文 |
| 技能粒度 | 混合原子动作、短流程、搜索/子目标和多步指导；只作文本建议，不执行技能包文件 |
| Phase1–2 | 库内容及版本冻结，不新增、合并、删除、改写或禁用 |
| Phase3 | 初始库不可变；每分支另外维护可增长的版本化活动库 |

当前实际 Qwen tokenizer 审计的 payload 范围为 **402–1967 tokens**，只是单技能载荷，不是完整状态 prompt。完整 prompt 仍需满足 4096 上限。

没有将旧 SkillRL bank 合并进主库。旧 bank 是 12 general +32 task-specific=44 技能，另有 11 条 common mistakes 不计入这 44；旧 clean 任务实际候选 18 项。曾讨论 37+44=81 的来源保留式 union，但未作为本轮 setting 实施；81 也不表示 81 种互不重复能力。

上游 SkillNet 从 ETO 专家轨迹构造 benchmark 技能及“不与 seen/unseen 测试重叠”的说法，属于已归档来源中的作者声明；本项目未逐技能溯源到原轨迹并独立核验精确 game ID 无重叠。不得写成我们已证明无泄漏。上游原 runner 的检索/过程生成也不是本项目的每状态 top-1。

资料：[冻结构建记录](/mnt/workspace/users/wangyifan/skill-RL/2026-09-17-skillnet37-frozen-bank-setting.md)、[完整 manifest](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/memory_data/alfworld/skillnet37/manifest.json)、[库说明](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/memory_data/alfworld/skillnet37/README.md)、[只读加载器](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/agent_system/memory/frozen_skill_bank.py)。

### 6.2 当前 router

采用 **SkillRL 官方 embedding 机制的状态感知适配版**：

- 模型：`Qwen/Qwen3-Embedding-0.6B`，revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；冻结、eval、独立于正在训练的 policy。
- 上游 SkillRL 参照 commit：`8e66726ed866a4e0a7f053586a41022798192e6c`。
- query：canonical JSON，含 task、当前 observation、admissible actions、最近两步 observation/action history、step index。无未来结果、reward、readout、gold、干预臂或 policy 权重；不按 game task 类别硬筛选。
- skill 编码沿用官方 `_skill_to_text` 机制，将 SkillNet name 映射 title、description 映射 principle；这只是检索表示，不改写原始 payload。
- 使用归一化 FP32 向量的 cosine/dot-product，全候选排名，逐状态 top-1；同分按规范 skill ID 顺序。
- 当前 Phase1–2 使用 CPU FP32、8 threads、encode batch 8；技能索引缓存，状态请求去重/批量编码，**不是每次重编码全库**。
- 每个 router 数值协议/库版本有独立缓存身份。Phase3 ADD/MODIFY/DELETE/MERGE 后为活动库重建索引，跨分支不混用版本；相同 query 在不同库上不保证相同选择。
- 外部 router API 调用为 0。GPT-5.4-mini 是历史配置；冻结 Qwen3.5-4B 生成式 router 是被搁置草稿；o3 当前只属于 Phase3 editor。

必须披露的适配差异：官方原 embedding 模式只用任务描述，在 reset 检索后整局复用 top-k，并分 general/task-specific 排名；我们采用完整统一库、状态 query、每步 top-1。不要称为“完全复现 SkillRL 原生 router”。也没有实验证据能据此宣称所有 37 条必然有足够自然支持、检索精度更好或已消除全部路由影响。

当前 Phase1–2 批量 profile SHA：`091b514862df2358b06cae7e5db71ce5233d134d8ca2dfda2f220490ba28fb4b`；Phase3 当前文档/profile 仍是非批量 `skillrl_embedding_state`，SHA `d1fc18a18537e65d4503d65d46187b8b63d21d9a884f6332262dbbb2dda778c8`。共享方法规则不等于两个执行后端逐 bit 等价。

资料：[选型沿革](/mnt/workspace/users/wangyifan/skill-RL/2026-09-18-router-selection-skillrl-embedding.md)、[详细路由协议](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/experiments/skillrl-embedding-router-v1/README.md)、[基础 profile](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/configs/skillnet37_router_qwen3_embedding_v1.json)、[批量 profile](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/configs/skillnet37_router_qwen3_embedding_batch_v1.json)。

## 7. 当前 Phase1–2：三 seed 独立 U0→U5

权威配置入口：[v4 profile](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/configs/phase12_independent_windows_v4.json)、[prepared-v2 队列](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/experiments/phase12-independent-v4/prepared-v2/cohort.json)、[seed404 spec](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/experiments/phase12-independent-v4/prepared-v2/preparation-s404-8gpu/spec.json)、[seed404 resolved 配置](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/experiments/phase12-independent-v4/prepared-v2/resolved-s404.json)。505/606 有对应文件。

注意：配置模板中的 `approved=false`、零预算等是静态防误启动字段；实际执行另有绑定 preparation hash 的授权与 recovery amendment。不能只看模板字段判断进程有没有运行，也不能把旧授权当成新实验依据。

### 7.1 训练参数

| 项目 | 本轮冻结值/含义 |
|---|---|
| policy 初始化 | 本地共同 post-trained Qwen3.5-4B B0；无额外 SFT，无另一个 warm-up B0；各 seed 独立 optimizer |
| RL seeds | 404、505、606，预登记顺序执行，不按结果筛选 |
| 训练框架 | verl，FSDP1；全参数更新，无 LoRA；不是换成了纯 vLLM 训练 |
| 算法 | GRPO，group advantage 标准差归一化；PPO epoch 1，token-mean loss |
| 每轮 rollout | 16 个 game groups ×每组 8 条=128 条；组内同 game |
| 迭代数 | 每 seed 5 次；640 条训练轨迹/seed，三 seed 完成时共 1920 条 |
| 学习率/优化器 | AdamW，lr `1e-6`，β=(.9,.999)，weight decay .01，constant、0 warmup |
| 优化 minibatch | 全局 128 个展开的状态/动作 rows；microbatch=1/GPU；不是 128 条完整 episode |
| 八卡 full minibatch | 每 rank 16 rows，对应 16 次 micro 前向累积；尾批以实际记录为准，不宣称固定等于论文的 4 accumulation steps |
| PPO/梯度 | clip low/high .2；dual-clip 3；grad clip 1；无动态 batch、无 actor shuffle |
| 正则 | `low_var_kl` actor loss，系数 .01；entropy .001；`use_kl_in_reward=false` |
| reward | 终局成功 10，失败 0；训练 invalid-action penalty 系数 .1；无 learned reward model/PRM |
| token/环境限制 | prompt≤4096，response≤512，最多 50 环境步，history 2，action-only，thinking=false |
| 解码 | train temperature 1；eval .4；top-p 1，采样开启；top-k=0 对应不作 top-k 截断 |
| prompt 溢出 | 实际完整 prompt 超限显式失败；不静默截断、不删长 game 来维持完成率 |
| 监控/保存 | 每 5 迭代 Seen monitor 64 episodes；`val_before_train=false`；当前端点为 U0/U5 |
| 硬件 | 本机 8×RTX 5090（32 GiB/card 适配） |
| 数值/显存 | 原生 FP32 master、BF16 mixed；decoder-layer FSDP、CPU shard 初始化、原生 optimizer-state CPU offload |
| optimizer offload 含义 | offload optimizer 状态；不是换 CPU Adam 或更改优化算法 |

“与 SkillRL 对齐”主要指其 README-linked `examples/grpo_trainer/run_alfworld_skills.sh` 的公开代码配方，而非把论文正文/附录不一致的 batch 数拼成新配方。基础脚本是 150 迭代上限、每 5 轮验证；**本轮用户确认覆盖为每 seed 5 轮**。模型、统一 SkillNet 库、逐状态 router、8 卡/microbatch、thinking、capture、推理后端等均有研究/工程适配，不能宣称所有设置原封不动。

配方沿革：[SkillRL 对齐留档](/mnt/workspace/users/wangyifan/skill-RL/2026-09-17-skillrl-rl-alignment-and-phase3-setting.md)、[基础对齐清单](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/configs/skillrl_public_alignment_v1.json)。清单里的 mini router、HF generation、4 卡、150 轮等旧字段，必须由当前 spec/resolved 覆盖解释。

### 7.2 推理与评分后端

- 当前生成：vLLM 0.22.0，Torch 2.11.0+cu130、Transformers 5.10.4；TP=1，每卡一个 replica，共 8 卡。
- `max_model_len=4608`，`max_num_seqs=16`，prefill budget 8192，memory utilization .45，eager=true，chunked prefill=true，prefix caching=false。
- 模型权重使用登记的原生/FSDP→vLLM 同步接口；生成无静默 HF fallback。
- 配置组合后 rollout `micro_batch_size=16`，不要把 budget profile 中兼容字段 `rollout_microbatch_per_gpu=2` 当作最终引擎生成 batch 上限。
- 训练/精确概率评分仍使用原生 PyTorch/Transformers forward；“不用 HF”只对应停止以 HF generate 作为该新 cohort 的生成路径，不是移除 tokenizer、teacher forcing 或训练依赖。
- Phase1–2 的 OLD ORIGINAL 来自实际训练现场完整 FP32 log-prob；终点 ORIGINAL 及对照在相同起点输入上 replay，另记录 live/offline parity 和 matched-backend 敏感性。
- Phase3 当前独立实现采用 matched offline BF16/SDPA 四条件 forward，并与起点实际 chosen-token log-prob 做容差校验。两阶段的公式相同，但不可把数值采集路径写成完全一致。

配置：[phase12_vllm_v1.json](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/configs/phase12_vllm_v1.json)。不依据未经完整实测的估时宣称加速倍数；当前日志中不支持 Qwen3.5 的 MFU 估算值 0 也不是实测 GPU 利用率。

### 7.3 ALFWorld split 与 game 清单

采用已有 train / valid_seen / valid_unseen，不重新按 task 类别划一个 train/test；六类均参与。运行清单对应 `json_2.1.1` 下有效 solvable games，按实现排除 movable/Sliced 等非当前任务条目，并保存逐 game 文件 hash。

| task_type | train | seen | unseen |
|---|---:|---:|---:|
| `pick_and_place_simple` | 790 | 35 | 24 |
| `pick_clean_then_place_in_recep` | 650 | 27 | 31 |
| `pick_heat_then_place_in_recep` | 459 | 16 | 23 |
| `pick_cool_then_place_in_recep` | 533 | 25 | 21 |
| `pick_two_obj_and_place` | 813 | 24 | 17 |
| `look_at_obj_in_light` | 308 | 13 | 18 |
| 合计 | 3553 | 140 | 134 |

上表是已冻结 inventory 数，不是宣布所有 game 已完成 runtime 评估。训练按 SkillRL 方式从全部 train 池采样，**不要求每轮遍历 3553 games**。全量成功率指每个端点逐项走完 seen140/unseen134 的 manifest，并校验唯一 game 覆盖；64 monitor episodes 不是全 seen，也未必等于 64 个 unique games。

Seen 用于开发/监控，Unseen 单独报告，不与 Seen 合成一个无差别总分。历史 clean seen27/unseen31 已被项目分析过，后续全 unseen134 不应称为项目历史上完全未触及的数据。当前新窗口仍需先锁 readout 后读目标 gold，不能用 unseen 选择指标或修改技能。

跨 RL seed 确实改变训练采样/RNG 流：离线预核对首批 unique games 分别 16/16/15，三者首批重叠为 0，均覆盖六类；五轮 unique games 分别 79/78/79。该预核对依赖当前文件枚举顺序，必须以真实 rollout 记录最终核验，不保证每个技能充分支持。评估 game manifests 与 continuation seeds 则保持共同，以便配对比较；不为了“不同 seed”更换测试集。

来源：[games.json](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/experiments/phase12-independent-v4/prepared-v2/preparation-s404-8gpu/games.json)、[sampling-precheck.json](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/experiments/phase12-independent-v4/prepared-v2/sampling-precheck.json)。

### 7.4 评估与 gold 工作量

| 项目 | 每 seed 设置 |
|---|---|
| 性能评估 | U0/U5 各 140 seen +134 unseen，共 548 full episodes；performance seed 61001 |
| 自然 anchor 来源 | U0 policy，unseen134 全量，source seed 61011 |
| gold skill pool | 先自然支持，再用固定 hash 规则 `sha256_daybudget_s404_natural_support_v1` 选择最多 12 项；不是按效用或风险挑技能 |
| anchors | 每技能最多 12，用 game round-robin 选择；无强制技能调用来制造 coverage |
| old evidence seed | 62011，用于 old-margin 等旧信息；与 gold repeat 分开 |
| gold seeds | 63011、63021；同 anchors/seed 跨 O/P/N 和两端点配对 |
| O/P/N 最大工作量 | `12 skills×12 anchors×3 seeds×3 arms×2 endpoints=2592` 条 suffix；实际依自然支持可更少 |
| Utility split | 当前只在 unseen 上，不是同时为 seen/unseen 做全技能 utility |
| 评分 | gated D 主，−P、+C 和预登记幅度/旧效用等比较，不拟合 predictor |

RL seeds 与评估 seeds 是不同角色。共同 B0 和共同评估随机流带来的重复起点评估，不构成额外独立 RL 复现。全面日志/全 split 性能 ≠ 所有 37 技能都有 gold ≠ 训练收敛。

## 8. 记录、存储与当前完成边界

### 8.1 窗口起点证据而非五轮全词表归档

当前 `window_start_old_only_v1` 保存第一轮实际 batch 的完整状态、action IDs、masks、advantage、skill/version metadata 与 live OLD 全词表概率。后四轮仍照常 rollout/优化，但主要保留轻量 token/reward 记录、game/skill 计数、reward/advantage 统计及 optimizer 行/step/lr/gradnorm/metrics，不再额外计算和存每轮 NEW 全词表，也不保存所有中间轮完整状态 batch。

U5 在同一 U1 batch 上重新前向；控制条件及读出可逐行算后保留紧凑结果。U0/U5 模型、U5 原生恢复 checkpoint、起点证据、评估轨迹、router 账本、支持/弃权/分数及报告需留存。**中间完整轨迹不是本窗口 C/P/D 必需条件；Phase3 编辑器却需要窗口内轨迹证据，不能机械照搬本轮精简策略。**

大张量采用 `torch_shuffle4_lzma_v1` 无损封装，保留完整 FP32 位模式并逐行 round-trip 核验；不是量化/只存 top-k。全部紧凑结果封存后，仅可回收该新实验中有独立逐行 bitwise 再生证明的临时行；无证明则保留。不能承诺自动删除全部旧模型，也不能说最终只留 Markdown 就能完整重算。

实现：[capture_scope.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/skillnet_cohort/capture_scope.py)、[lossless_tensor.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/skillnet_cohort/lossless_tensor.py)、[window_storage.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/skillnet_cohort/window_storage.py)。

### 8.2 时间与恢复记录

三 seed 合计软目标 12h、累计实际运行硬上限 30h；不是每 seed 30h。后续 seed 按登记顺序及剩余时间/容量准入，不按结果选。故障停机时间经用户确认扣除，首次失败运行约 69 分钟仍计入总额。共同磁盘上限 760 GiB、空闲保护线 100 GiB、checkpoint reserve 80 GiB。时限是停止规则，不是已达成的实验耗时。

Seed404 首次尝试在 U1 OLD 前向期间因容量扫描短寿命文件的竞态停止，**不是 OOM，也尚无 optimizer 更新**。恢复复用原 128 条 rollout（5511 active decisions、77819 response tokens），不重新采样；5512-row 原生 batch 含一个 padding 重复。已有 2050 行 OLD 的位置/token/rank 被对齐，恢复评分要求保留行逐 bit 相等后复用。

必须披露恢复边界：worker/vLLM RNG 从登记 seed 重新初始化，不能声称与不中断运行逐 bit 等价；缺失的原 vLLM sampling log-prob 仅用于后端诊断，不伪造，GRPO 使用 native OLD。以后新增 pre-forward batch 落盘保障，不改变 rollout 数或损失。来源：[RECOVERY-20260919.md](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/experiments/phase12-independent-v4/RECOVERY-20260919.md)。

截至本文完成标记核对时：404/505/606 均无本轮 `complete.json`，404 无 U5 metric/已封存窗口标记；因此**新三 seed Phase1–2 结果尚不能引用**。原 `queue_stopped.json` 属于历史失败；恢复尝试有独立 `recovery-v1/`，不能据旧停止标记判断新尝试状态。本文未中断、重启或修复主线程任务。

未来产物位置（仅列约定，未生成不作已完成证据）：

- `Q/seed-<seed>/reports/phase1-results.md`、`phase2-results.md`。
- 同目录 `performance.csv`、`utility_units.csv`、`ranking_diagnostics.csv`、`ranking_budgets.csv`、`provenance.json`。
- `Q/seed-<seed>/windows/u0000-u0005-valid_unseen/` 下协议、readout commitment/prediction、O/P/N evaluations、`window_metrics/` 与 `sealed.json`。
- `Q/reports/phase12-cohort-summary.md` 和跨已完成 seeds 的 CSV；报告必须列出未完成 seed，不只展示成功完成的部分。

## 9. Phase3：已确认的闭环实验设计，尚无正式结果

主设置：[phase3_setting_embedding_v1.json](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/configs/phase3_setting_embedding_v1.json)、[SETTING.md](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/phase3/SETTING.md)。远端目标为 `https://github.com/Euclider/SkillScope-phase3`；本文不重新推送或核验远端同步，也不假定本机后续修改已全部出现在远端。

### 9.1 四条独立持续 RL 分支

| 分支 | 选择/证据逻辑 | 角色 |
|---|---|---|
| `skillrl_failure` | 失败轨迹驱动，编辑器依据失败证据审查库 | SkillRL-style 主 baseline |
| `readout_d` | gated D 降序，优先审查 top-k 技能的实际调用轨迹 | 主实验 |
| `readout_p` | −P 降序，独立训练/编辑演化 | 完整闭环消融 |
| `readout_c` | 原始 +C_upd 降序，不是 centered C | 完整闭环消融 |

共同初始 Qwen3.5-4B、SkillNet-37、seed404、router 规则和训练配方；但 optimizer、轨迹、bank lineage、缓存、费用独立。库变化会影响未来 routing、rollout 和 RL，所以不能只训练一次 policy 来声称测了四条累计演化路径。

登记上限 150 迭代，每 5 轮一个编辑机会（最多 30 个），U0 不编辑；主终点 U150，而非 unseen 最佳 checkpoint。各窗口内同一 bank 内容固定；在 `π_q→π_(q+5)` 完成后评分/编辑/验收，接受的新 bank 与当前 policy 进入下一窗口。native checkpoint 连续恢复 policy/optimizer，窗口环境流使用预定调度；不是每窗口重新从 B0 训练。

没有独立 norm-only、random、no-evolution 或 oracle 闭环分支。那些出现在旧 idea 中的建议不等于已确认运行条件。**当前四分支不能支持“Phase3 已优于 magnitude-only baseline”这一实验主张。**

当前 Phase3 `training.py` 仍组合基础 HF/SDPA generation 配方；本机 Phase1–2 的新 vLLM profile 并未由该入口自动加载。不得把本机 vLLM 吞吐/验收直接当作 Phase3 远端环境事实。

### 9.2 同一 o3 编辑器：输入与输出

共同编辑器：`o3`，reasoning effort `medium`，兼容 API base URL `https://api.zhizengzeng.com/v1`。无自动模型 fallback、无失败后自动重试；凭据只由安全环境配置，不入库。

输入契约：

- 全部当前活动技能的 ID、版本 hash、name、description、完整 body；各分支读权限相同。
- failure 分支：按固定 trajectory ID 顺序取失败 episodes，上限由 runtime 冻结。
- readout 分支：支持池内最多 top-3 优先 ID，轮转收集这些技能实际出现的 episodes，成功和失败都可进入；去重并受相同轨迹数/token 上限约束。
- evidence 包含 task、game ID、success、逐步 observation/action/next observation、调用 skill/version、动作合法性。
- 当前 `editor.propose()` 传的是优先 ID 与实际 evidence，**不传 C/P/D 数值、分支方法名或 utility gold**；优先 ID 是审查建议，不是“必须修改/已经有害”的标签。
- 没有额外免费、不计预算的 LLM summary 阶段。证据整理是确定性选择，内容分析/编辑由同一次共享 editor 完成。

输出为严格 JSON patch：`op / targets(ID+version) / skill(name,description,body) / rationale / evidence_ids`；候选变动先做 schema、版本、预算、payload 控制校验，再做环境 gate。

| 操作 | 含义 | mutation units |
|---|---|---:|
| ADD | 新建独立 ID/版本 | 1 |
| MODIFY | 改已有技能，生成新内容版本/沿袭 lineage | 1 |
| DELETE | tombstone 停用，不删除历史正文 | 1 |
| MERGE | 停用 n 个目标，创建一个新技能 | n+1；两条合一用完 3 units |
| NOOP | 不编辑，须作为唯一操作 | 0 |

每次最多 **3 mutation units**，而非必改三个技能。top-k=3 是候选优先级上限；编辑器可 NOOP、合并、新增，且当前 prompt 对全库有读权，并未将所有可修改 target 硬限制为这三个 ID。应从 patch 记录报告实际 touched IDs、操作数、units 和库增长，不用 top-k 代替编辑次数。

新加/改写/合并版本不得继承旧版本分数；只在有本版本自然训练支持后参与下一次 readout 排名。unsupported 是 abstain；不同分支实际调用/编辑次数可因支持或 NOOP 不同，公平性主要是共享机会、规则及上限，不保证事后实际花费严格相同。

实现：[editor.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase3/editor.py)、[bank.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase3/bank.py)、[readout.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase3/readout.py)。

### 9.3 Baseline 的准确标签

应写 **SkillRL-style failure-driven evolution with a shared editor**，不是原生 SkillRL 完整复现。其原公开更新器主要新增技能，且有 success-rate<.4 等原生触发逻辑；当前受控对比使用共同每 5 轮机会、共享 o3、统一 ADD/MODIFY/DELETE/MERGE/NOOP 空间与 gate。现有 `evolution.py` 的 failure 分支按失败证据提出编辑，不能把原生 .4 触发器当作这里实际执行的规则。

当前可归因的比较因素是选择/证据来源策略。Phase3 总效应还包括编辑后库、路由、训练分布及后续 policy 的连锁变化，不是 Phase1–2 固定 bank 的 O/P/N semantic utility estimand。

### 9.4 验收、拒绝与 rollback

- Seen 分为编辑 evidence/monitor games 与 gate games，集合不重叠；train 轨迹也可作为 evidence。Unseen 不用于编辑、门控、选阈值或选赢家。
- 在**同一个当前 policy** 下，对旧 bank 与候选 bank 使用同 gate games×eval seeds 配对评估。
- 当前接受规则是 game-equal 的成功率差 `100×ΔSR ≥ −tolerance_pp`。这是预定点估计容忍规则，不是自动完成了统计非劣效检验。
- 不接受候选叫 `proposal_rejected`，不是已提交后的 rollback。实现记录这两类字段；现有常规 gate 路径将 rollback_count 记 0，不能据字段存在宣称已实现某种自动后续回滚策略。
- 若发生独立的 bank rollback，作用域不能扩成回退 policy/optimizer；所有 bank 版本及 tombstones 保留。
- 最终报告全 seen140 / unseen134 和六类任务细分；Seen 含已用于开发/gate 的 games，不能称它是完全 held-out test。

实现：[evolution.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase3/evolution.py)、[run.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase3/run.py)。

### 9.5 评价与记录要求

| 类别 | 必须区分/保留的量 |
|---|---|
| 性能 | seen/unseen SR、六 task、逐 game/seed 成败；配对 ΔSR、repair/regression 数和比率；中途监控与最终结果分开 |
| 选择 | 支持/弃权、top-k、每次 candidate 数、三分数原值/排名、gate coverage、bank/policy/window 身份 |
| 编辑 | proposed/accepted 操作分别计数、实际 touched IDs、mutation units、NOOP、候选拒绝、rollback、库大小/增长和版本谱系 |
| Actor 成本 | training/seen monitor/paired gate/final 各阶段 prompt/completion tokens、环境步、无效动作、episodes |
| Router 成本 | 本地查询/编码 tokens、命中/新调用、索引重建、延迟；不是外部 API token 花费 |
| Editor 成本 | 请求/成功/失败/未对账数，返回的 input/output/cached/reasoning 等 usage 字段、延迟；服务未返回用量则 unknown，不填 0 |
| Readout 成本 | forward calls/input tokens、walltime；与 actor 生成及付费 LLM 成本分开 |
| 完整性 | 实际完成迭代数、停止原因、缺失阶段/未完成 rollout；不能将不同训练终点当作匹配 U150 比较 |

实现 [report.py](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/phase3/report.py) 汇总上述主要记录；其 actor totals 只计已持久化完整 episodes，未完成工作不能当作零。API `provider_cost` 仍为 null，不能只拿历史 mini 单价推算 o3 账单。

该文件提供 2000 次 paired game bootstrap 的 CI helper，区别于 Phase1–2 的 10000 次；默认 summary CLI 并不自动完成四分支间所有配对统计。发表前应核对实际跨分支分析产物及调用设置，不把 helper 存在写成已生成 CI。单 RL seed/arm 的 game 或 decoding repeats 不代表训练 seed 稳健性。

Phase3 使用 compact prediction/edit provenance，不保存 Phase1–2 全词表/hidden activations，不做 Phase2 utility gold。它仍保留窗口内编辑所需的训练/监控轨迹和固定起点 direction batch。

### 9.6 尚未冻结/完成的边界

[runtime template](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/configs/phase3_runtime_template.json) 仍需新服务器 preparation 明确：editor 总调用上限、router 本地上限、gate 每 task 数量/容忍 pp、eval seed 列表、live/offline parity 容差、GPU 与存储预算。模板中的 max input 100000、completion 8192、evidence 10 是模板值，**不是已执行正式实验的实测预算**。

[VALIDATION.md](/mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/phase3/VALIDATION.md) 记录的是离线契约/组件/GPU 验收；没有四条正式 Phase3 RL 成功率结果，o3 真实编辑请求/完整闭环也未由这些 fake-client 测试验证。不能把已打包/已发布当作科学实验完成。

## 10. 历史结果：与新实验分开引用

### 10.1 Cohort 对照表

| cohort | 真实训练/窗口 | 库、router 与范围 | 标签/样本边界 |
|---|---|---|---|
| 历史 Phase1 | seeds101/202/303 各从 B0 训练30轮，固定 U10/U20/U30；8 games×4=32 rollout/轮 | 旧 SkillRL bank，clean 候选18；冻结词匹配/阶段规则 router，非当前 embedding | B0 自然200 anchors，4技能各50；clean unseen31 games；报告累计 B0→里程碑效用，也分析相邻10轮的 S_int 变化 |
| 历史 Phase2 L | seed303 从 U30 连续恢复到 U35；五个单步窗口 | 同旧库/clean/旧router | 旧 B0 的200 anchors；1 evidence+1 gold；六端点×1200=7200 suffix；19支持单元，其中后段7是子集 |
| 历史 Phase2 A | 同 seed303 从 U35→U40，5 RL迭代、160 rollout、70 Adam steps（611→681） | 同旧库/clean；HF generation/SDPA，非本轮8卡vLLM | U35重新自然采集200 anchors；2 evidence+4 gold；两端点各3600=7200 suffix；4 gold技能、3共同支持技能 |
| 当前 Phase1–2 Q | 404/505/606 独立 U0→U5，每轮16×8 | SkillNet37、全六任务、embedding状态top1、vLLM | 全 seen/unseen性能；unseen ≤12技能×12 anchors；尚无完成的本轮结果 |
| 当前 Phase3 设计 | seed404，四独立持续路径，5轮滚动，150轮上限 | 同初始SkillNet37，分支库随后演化 | o3共享编辑器、Seen gate、最终seen/unseen；正式闭环未完成 |

历史 L/A 的训练限制为 prompt2048、response64、max_steps30、decision minibatch32；当前 Q 为4096/512/50/128，不能把它们作为同一配置的跨 seed 直接合并。历史 Phase1 seed101/202 为4-way FSDP，303为8-way；历史 Phase2 分片随资源调整，最后 U40 native world_size=1，不是全程固定八卡。

L 名称含 `u30-to36`，但旧计划 U36 未执行；不要据名字补跑。A 的 U36–39 有训练/读出归档而无新增中间 gold；窗口的 reward direction 来自 U36 batch 在 U35 的起点状态，不是5轮 advantage 平均。

### 10.2 已有现象和预测线索

以下是**历史报告/归档值**，本次未重跑统计，更不是新 SkillNet-37 cohort 的结果。

| 证据 | 关键数值 | 能支持/不能支持 |
|---|---|---|
| 历史 Phase1 的无方向 `S_int` | `S_int=1/0` 时效用变化率34.90%/8.41%，RR4.15；leave-one-seed-out AUPRC .192→.274 | 支持 action interaction 与是否变化关联；不是 C/P/D 的跨seed结果 |
| 同一 Phase1 的下降方向 | 加 S_int 前/后下降 AUPRC .317/.314 | “是否变化”收益不能偷换为稳定的下降方向收益 |
| 旧 cle_004 U34→U35 | Δ(O−P)=−14.29 pp，95% CI[−28.57,−1.79] | 支持固定 skill 的相对效用下降；不直接证明 D 预测成功 |
| 旧案例敏感性 | Δ(O−N)=−14.29 pp，CI[−28.57,−3.57]；early anchors语义−15.38 pp，CI[−30.77,−1.92] | 支持该案例非初始状态/跨对照可见；共享anchors不是独立复现 |
| 旧19支持单元，D vs centered interaction norm | AP .799/.667；ρ(score,−ΔM) +.417/+.062 | 描述性 pooled 优势，不是跨seed显著性；后段7已包含其中 |
| v3同源对照，旧19单元 | C vs `u_original_norm` AP .377/.288，ρ+.336/+.132；D vs `delta_norm` AP .799/.558 | 对 reward-directed 信号族有探索性支持；不能隔离reward、归一化、负部和门控各自的因果作用 |
| 新历史窗口 U35→U40 | gen_002 −3.02 pp；cle_003 +15.73 pp；cle_004 −6.45 pp | 只有 cle_003 的CI完全高于0；下降CI均含或触及0，不能称所有下降已统计确证 |
| 同窗口三技能排序 | D/centered norm AP均.583、ρ−.5；−P/KL/JS/control norm AP均1、ρ+1 | −P有排序线索，但非独特增量；D旧优势未在该窗复现 |

历史 `S_int` 定义为：O 的新旧第一步动作是否改变，与 P 的新旧第一步动作是否改变，两者做 XOR；它不是连续的 full-vocabulary `δ`，不使用 advantage。其1800行来自3 seeds×3相邻窗口×200 anchors，不能当1800个独立RL更新。小型L2 logistic probe仅用于该历史分析；当前新窗口不拟合该probe。

旧后段7单元中的 conditional AUROC（仅2降/2升）不同于 decline-vs-rest（2降/其余5）；前者 D/norm/KL=1/.75/.5，后者=1/.9/.7。不要把表中不同分母的 AUROC 混用。旧19+子集7+新3不能相加成29；全部仍出自一条 Phase2 seed303 延续路径。

两批历史 U35 的 anchors/continuation seeds 不同，不把其 utility 点直接接成未控制的连续曲线。14,400条归档 suffix 是工作量，不是独立预测样本数。所有正面/负面比较都应保留，不按窗口选择最佳 score 后汇总成一个方法。

### 10.3 两份 Phase2 文档与三个修订版本

- [带日期完整报告](/mnt/workspace/users/wangyifan/skill-RL/2026-09-14-phase2-complete-analysis.md)的当前 hash 为 `a697119808ba394cf608a910ae3142257721d802d9e8baa072734fe6c051292e`，与 synthesis-v2 revision 一致；适合查完整效用、NULL、阶段、预算和数值来源。
- [无日期综合稿](/mnt/workspace/users/wangyifan/skill-RL/phase2-complete-analysis.md)不是上述文件的链接或字节副本。它更明确按 C/P/D **信号族**组织命题，并收录同源幅度对照；当前 hash 为 `b13d3960b808af3c96b20bc5618c4e764da283146f75350e04704dc711ca88a4`。
- `A/reports/2026-09-14-reward-directed-family-v3/revision.json` 记录后续同源比较，明确没有新RL、rollout、model forward或拟合predictor。其 report hash `472218b28c542872f9583e4c83a48fae5bbf7128b33469988d4196253626c9a0` 对应历史版本，**不等于目前压缩稿的hash**；无日期稿末尾也说明这一点。不要声称该 revision 已校验当前压缩稿全部字节。
- 早期 HANDOFF 提到 synthesis-v2 的停点与上述后续材料并存；本次仅整理，不覆盖任何一个版本。方法叙事参考最新讨论/无日期稿，具体历史数字用原表、v2/v3各自归档核对。

## 11. 文件索引：按写作问题定位

下面列的是写 Methods/Experimental Setup/附录最相关的材料，不把环境依赖、全部技能正文、所有失败日志逐个复制到本文。未特别说明的路径均为当前 C/R 下的本地文件；目录/模式用于定位大量同类产物。

### 11.1 研究定义与讨论沿革

| 文件（相对 R） | 用途与边界 |
|---|---|
| `HANDOFF.md`、`HANDOFF-DETAILS.md` | 停点、旧新实验路径、证据边界；含历史路径/时间点，不能取代最新runtime |
| `2026-08-19-policy-update-skill-sign-flip-forecasting-design.md` | 原始 u/δ/d/C/P/D、post-update pre-evaluation、预测与机制区分；sign-flip必需、8B/A100、JVP/许多baseline是早期设计 |
| `2026-08-21-proposal-advantages-and-roadmap.md` | 研究动机/路线，作为历史设计材料 |
| `2026-08-24-phase1-minimal-validation-repository-and-experiment-spec.md` | 早期最小验证协议，不代表当前全六任务设置 |
| `2026-08-25-step-routed-skillrl-rl-validation-protocol.md` | 逐状态技能暴露与早期RL协议 |
| `2026-08-26-fixed-state-and-matched-evaluation-results.md` | fixed-state/matched测量的早期验证与局限 |
| `2026-08-29-first-invocation-skill-utility-evaluation-results.md` | 首次调用干预与中途状态评估演进 |
| `2026-09-08-phase2-reward-directed-skill-utility-direction-validation-plan.md` | Phase2方向读出计划；未实现部分须与后续代码区别 |
| `2026-09-09-phase2-semantic-direction-fast-execution.md` | 单步方向测量的执行/采集协议 |
| `2026-09-12-phase2-expanded-validation-code-and-observation-appendix.md` | 扩展测量、数学实现、观察附录 |
| `2026-09-17-skillnet37-frozen-bank-setting.md` | 库来源/完整原文/不合并旧库/粒度和泄漏边界 |
| `2026-09-17-skillrl-rl-alignment-and-phase3-setting.md` | 官方脚本与论文参数差异、受控SkillRL-style baseline的缘由 |
| `2026-09-18-phase3-confirmed-setting-and-readout-plan.md` | editor o3、四分支和读出选择讨论留档 |
| `2026-09-18-router-selection-skillrl-embedding.md` | mini→放弃4B生成式→0.6B状态top1的确认顺序 |
| `2026-09-18-embedding-phase3-publication-and-phase12-launch.md` | embedding迁移/历史发布记录；不把该150轮旧launch当当前v4 |

### 11.2 当前 Phase1–2 的机器可读配置、实现与记录

| 文件/目录（相对 C） | 写作用途 |
|---|---|
| `docs/experiments/phase12-independent-v4/README.md` | 当前总协议和限制，优先阅读 |
| `docs/experiments/phase12-independent-v4/RECOVERY-20260919.md` | 恢复时RNG/概率记录边界及累计时间规则 |
| `configs/phase12_independent_windows_v4.json` | seeds、5轮、anchor/gold预算、capture scope、+C预登记 |
| `configs/phase12_vllm_v1.json` | 实际generation后端版本及engine参数 |
| `docs/experiments/phase12-independent-v4/prepared-v2/cohort.json` | 三seed队列与共享预算注册 |
| `.../prepared-v2/preparation-s{404,505,606}-8gpu/{manifest,spec,model,games,placebos}.json` | 每seed冻结资源、B0/tokenizer文件身份、game名单和37条控制payload |
| `.../prepared-v2/resolved-s{404,505,606}.json` | 组合后的最终参数；不是只看基础Hydra YAML |
| `.../prepared-v2/sampling-precheck.json` | 跨seed采样与六类覆盖预核对 |
| `skillnet_cohort/{prepare,assets,independent_preparation}.py` | 资源inventory、token匹配控制、离线注册 |
| `skillnet_cohort/{training,segmented_training}.py` | 真实参数组合、五轮block、连续恢复规则 |
| `skillnet_cohort/{inference,vllm_backend,vllm_qwen35}.py` | vLLM适配及Qwen3.5文本生成/权重同步 |
| `skillnet_cohort/{evaluate,support}.py` | 全game遍历、自然首次调用、gold pool与新窗口协议注册 |
| `skillnet_cohort/{run,seed_queue,recover_queue}.py` | 顺序/停止/锁分时点，非method数学本体 |
| `skillnet_cohort/{capture_scope,lossless_tensor,window_storage,rollout_recovery}.py` | 证据精简、精确存储/回收、恢复声明 |
| `skillnet_cohort/reports.py` | 当前Phase1/2分seed和cohort报告、AP/AUROC/符号统计 |
| `verl/trainer/ppo/{core_algos,ray_trainer}.py` | 实际GRPO advantage、reward penalty、训练batch与capture hooks |
| `verl/workers/actor/dp_actor.py`、`phase2/capture.py` | 实际optimizer更新、live概率/有效token采集 |
| `Q/seed-404/recovery-v1/segments/u0000-u0005.json` | 本次恢复尝试的实际组合参数，Q见§0；勿改动 |

表内 `...` 专指 `docs/experiments/phase12-independent-v4`，不是任意搜索路径。`prepared/` 是保留的未完成早期准备；**当前是 prepared-v2**。

### 11.3 数学、干预与分析代码

| 文件（相对 C） | 作用 |
|---|---|
| `phase1/first_invocation.py` | 构建anchor、prefix replay、目标payload O/P/N、自由续跑 |
| `phase1/build_all_first_invocation_anchors.py` | 首次自然调用与game round-robin anchor选择 |
| `phase1/eval_first_invocation_utility.py`、`eval_all_first_invocation_utility.py` | 历史效用评估入口及policy适配 |
| `phase1/compute_all_first_invocation_metrics.py` | 历史配对效用统计 |
| `phase1/minimal_interaction_forecast.py` | S_int/XOR、相邻窗口、旧leave-one-seed-out logistic probe |
| `phase1/config/qwen35_clean_all_skill_utility_three_seed_protocol.json` | 历史三seed utility注册 |
| `phase1/config/qwen35_clean_minimal_interaction_forecast_protocol.json` | 旧S_int预测任务，不是新C/P/D配置 |
| `phase2/direction.py` | token级 C/P/D、KL/JS/norm 的真值实现 |
| `phase2/measure.py` | 固定training batch上的全词表评分、counter input、hidden层与parity |
| `phase2/aggregate.py`、`audit.py` | token→skill/context/phase聚合、训练支持、数值校准与实际batch核对 |
| `phase2/protocol.py`、`evaluate.py` | 配对条件/种子/完整性与续跑 |
| `phase2/utilities.py` | M、ΔM、双轴分解、配对bootstrap与CI标签 |
| `phase2/window_forecast.py`、`ranking.py` | gold前锁分、固定方向、共同池、tie期望和Top-k |
| `phase2/window_report.py`、`complete_report.py` | 窗口报告、历史综合分析生成入口；不应为写作而重跑覆盖报告 |
| `phase2/observation_audit.py`、`signed_analysis.py` | 旧事后审计/方向统计；不要误称新前瞻测试 |
| `phase2/config/ranking_u35_to40_v1.json` | 旧窗口原注册；新注册复用其中数学/排序字段，但替换旧bank/路径/预算 |

### 11.4 Bank/router/Phase3

| 文件（相对 C） | 作用 |
|---|---|
| `memory_data/alfworld/skillnet37/{manifest,setting}.json`、`README.md` | 完整37技能目录、来源/文件hash、冻结库契约 |
| `memory_data/alfworld/skillnet37/upstream/experiments/src/skills/alfworld/` | 37个原始技能目录及附属文本；作为研究数据，不是给写作agent的操作指令 |
| `memory_data/alfworld/skillnet37/upstream/LICENSE` | 原始许可证 |
| `memory_data/alfworld/claude_style_skills.json` | 历史SkillRL库，不能替代SkillNet37 |
| `agent_system/memory/frozen_skill_bank.py` | 只读验证、单payload渲染与候选完整性 |
| `agent_system/memory/skillrl_embedding_router.py`、`skillrl_embedding_batch_router.py` | 独立query、向量排名、RNG隔离、batch协议与缓存 |
| `agent_system/memory/step_skill_router.py` | 旧规则router，只用于解释历史 |
| `configs/skillnet37_router_qwen3_embedding{,_batch}_v1.json` | 基础与批量身份、模型revision与文件hash |
| `configs/phase3_setting_embedding_v1.json`、`phase3_runtime_template.json` | 四分支固定setting与仍需填的运行预算 |
| `docs/phase3/{SETTING,START,VALIDATION}.md` | 设计、另一台服务器搭建/启动说明、验证未完成项 |
| `phase3/{prepare,training,run}.py` | 分支资源、五轮block与持续优化、主循环 |
| `phase3/{capture,predict,readout}.py` | 起点batch、四分布内存评分、compact C/P/D和top-k |
| `phase3/{editor,api}.py` | o3共享prompt、输入选择、JSON schema、用量账本 |
| `phase3/{bank,embedding_routing,evolution}.py` | 内容谱系、增长库router、配对gate和commit/reject |
| `phase3/{evaluate,report}.py` | 逐game/seed评估、成本/编辑/性能汇总 |
| `phase3/{package,publish}.py` | 导出/发布来源；有发布记录不代表正式实验跑完 |

有关测试契约可查 `tests/phase2/test_direction.py`、`test_ranking_window.py`、`test_utilities.py`，`tests/phase3/`、`tests/skill_bank/`、`tests/skill_router/`、`tests/skillnet_cohort/`。本文没有运行它们；历史通过项数只是工程证据。

### 11.5 历史报告与原始表

优先使用的根目录报告：

- [Phase1三seed主报告](/mnt/workspace/users/wangyifan/skill-RL/2026-09-04-qwen35-clean-all-skill-three-seed-utility-results.md)。
- [Phase2旧单步方向分析](/mnt/workspace/users/wangyifan/skill-RL/2026-09-12-phase2-skill-utility-direction-analysis.md)。
- [Phase2扩大窗口执行记录](/mnt/workspace/users/wangyifan/skill-RL/2026-09-13-phase2-u35-to40-ranking-execution.md)。
- [Phase2带日期完整报告](/mnt/workspace/users/wangyifan/skill-RL/2026-09-14-phase2-complete-analysis.md)和[当前无日期综合稿](/mnt/workspace/users/wangyifan/skill-RL/phase2-complete-analysis.md)，版本差异见§10.3。

原始表入口：

| 位置（简写见§0） | 内容 |
|---|---|
| `L/reports/2026-09-12-observation-audit-v1/{all_supported,heldout,comparisons}.csv` | 旧19、其中后段7、全候选比较 |
| `L/metrics/{utility_units,anchor_margins}.parquet` | 旧配对效用/anchor分解 |
| `A/protocol.json`、`A/window_signals/u0035-u0040/prediction.json` | 旧新窗口协议、gold前锁分（2026-09-13T13:46:46.281540+00:00） |
| `A/window_metrics/raw_features_and_semantic_utility.csv` | raw P/C/D与全部phase效用；P列不是−P |
| `A/window_metrics/locked_scores_and_gold.csv` | 已按固定符号变换的风险分数；不要再对P重复取负 |
| `A/window_metrics/{ranking_metrics.csv,ranking_support.json}` | 各预算、阈值、同池支持 |
| `A/window_metrics/{utility_units,utility_games,anchor_margins}.parquet` | paired效用、game层和anchor层统计 |
| `A/evaluations/u0035/`、`u0040/` | 索引、完成标记、逐条suffix轨迹 |
| `A/{evaluation_completion,full_report_completion}.json` | 对应历史评估/报告版本的完成证明 |
| `A/reports/2026-09-14-ranking-interpretation-v1/` | overall/within-window排名审计及v1修订 |
| `A/reports/2026-09-14-phase2-synthesis-v2/revision.json` | 带日期主报告的综合补全与数据hash |
| `A/reports/2026-09-14-reward-directed-family-v3/` | same_basis_overall.csv、same_basis_within_window.csv及后续信号族修订 |

历史模型位于 `C/artifacts/model_only/qwen35-clean-formal-seed{101,202,303}-u30-c{1,2,3}-update{10,20,30}`（c1↔U10、c2↔U20、c3↔U30，非所有组合）；旧 Phase2 `models/batches/old_logprobs/new_logprobs/optimizer_steps` 见 L/A。本文未加载模型或重算这些大张量；方法写作通常不需要读取权重。

### 11.6 仅作历史，不用于覆盖当前配置

`R/2026-09-17-skillnet37-external-router-setting.md`、`C/configs/skillnet37_router_gpt54mini_v1.json`、旧 `phase3_setting_v1.json`：mini版本；本轮不再作为router。

`C/configs/phase12_day_budget_proposal_v1.json`、`phase12_30h_budget_v1/v2.json`、`phase12_single_window_v3.json`，以及 `docs/experiments/phase12-daybudget-v1/v2/`、`phase12-vllm-v1/`：可解释规模/后端演进，但当前主设置为 independent-v4。早期100/150轮、4B生成式router、小模型smoke、旧累计窗口等均不得拼到当前一张参数表。

## 12. 写作前必须保留的未决事项与风险

### 12.1 不可由现有材料直接宣称

- 新 SkillNet37 三seed已完成、全部37技能都有充分训练/gold支持、短U0→U5证明全训练阶段均成立。
- C/P/D跨独立seed稳定优于所有无方向指标，或 gated D 始终最优、门控总是有益。
- P的符号数学保证长期ΔM方向，D是校准概率/utility估计，unsupported等于低风险。
- 全部流程无需post-update rollout、forward读出没有算力/token成本、已测得普遍固定加速比。
- Phase3已改善最终held-out性能，或四个已确认分支中存在norm-only实验。
- 同库同editor就保证全部实际花费完全一致；不同结束轮数可以直接当同训练预算比较。
- SkillNet来源无泄漏已由我们独立证明；seen是从未用于开发的test；旧unseen clean子集从未被项目接触。
- JVP/线性化误差、reward shuffle、等范数随机/正交更新、PRM、activation patching等早期idea负对照已完成。当前材料没有这些完成证据。
- 模型恢复逐bit校验等同于中断前后整条采样随机路径逐bit一致。

### 12.2 当前摘要需要保留的条件性

已有旧数据支持部分效用变化案例和描述性排序线索。对“reward-directed 比 magnitude-only 更好”的叙述，应绑定比较对象、cohort、候选池与指标，不能合并不同窗口的赢家形成事后最优方法。当前主 D 与两条完整C/P消融已经预登记；不要根据新结果在方法部分改写主实验身份。

若正式论文决定只将旧分析作为pilot而以新三seed为主，需等新结果完成再填样本数/性能/耗时；也不能让 Methods 的“实际执行”时态掩盖未完成的seed。Phase3相关收益在结果完成前保留 pending，不由写作agent补出数值。

### 12.3 本次只读核对发现的实现状态提示

新窗口的 mathematical protocol 已明确，但“代码中有入口”不是新流程已经端到端验收。一个具体静态风险：读取时 `phase2/aggregate.py` 的提交检查仍遍历 `old/new_logprobs` 两类起点行，而 `skillnet_cohort/support.py` 对 `window_start_old_only_v1` 只链接 OLD、不链接 NEW；两者与旧双份归档约定的兼容需主线程后续核验。**这是代码阅读发现的接口风险，不是本次观察到新的运行失败；本文件未修复或运行验证。** 不应据此修改论文数学定义，更不能因此重采样旧轨迹。

Phase3的真实o3网关、runtime预算、完整长跑/跨分支统计和vLLM迁移状态也需以之后的新服务器preparation/运行产物为准。写作agent只负责整理表述，不替主线程启动或调整实验。

## 13. 版本锚点、引用与交接要求

### 13.1 本次核对的小文件哈希

| 文件 | SHA-256 |
|---|---|
| 带日期 Phase2 synthesis-v2 报告 | `a697119808ba394cf608a910ae3142257721d802d9e8baa072734fe6c051292e` |
| 当前无日期 Phase2 综合稿 | `b13d3960b808af3c96b20bc5618c4e764da283146f75350e04704dc711ca88a4` |
| SkillNet37 manifest | `0767ff7578b1e997119a36b5f636fec6ebc1d0ac600ffb40b1e599065b7fe514` |
| phase12_independent_windows_v4.json | `0e59c324f85d4ea8479de6b5fa3ff2c987c9a8b39cb446a4312c9f0f669a9d1b` |
| phase3_setting_embedding_v1.json | `6fd32e42fc553f3cf3eedbfde4686f0ea1ec60568f3e93a78b7e6c2b41a67aa3` |

这些定位材料版本，不是对全部运行状态/历史权重做新的完整审计。修改 Methods 叙述不意味着可以覆盖这些证据文件。

### 13.2 可追溯公开来源与待核验引用

本次只整理本地归档，未重新进行文献检索/引文核验。以下 URL 是已经在项目来源记录中出现的定位信息，不将其当前内容状态、引用量、star数或撤稿状态当作本次新验证：

- [SkillNet 固定 commit 的 ALFWorld 资源](https://github.com/zjunlp/SkillNet/tree/5c472b36d2a435001fdae3bc8439886d8050645a/experiments/src/skills/alfworld)。
- [SkillRL 固定 commit](https://github.com/aiming-lab/SkillRL/tree/8e66726ed866a4e0a7f053586a41022798192e6c)，RL脚本为 `examples/grpo_trainer/run_alfworld_skills.sh`，检索实现为 `agent_system/memory/skills_only_memory.py`。
- [Qwen3-Embedding-0.6B 模型](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)，应同时保留本文固定 revision。

用户intro的引用key包括 `jin2025searchr1trainingllmsreason`、`zheng-etal-2025-deepresearcher`、`liang2026skillnetcreateevaluateconnect`、`xia2026skillrlevolvingagentsrecursive`、`tang2026wikiskillcompilingagentexperience`、`zheng2025skillweaverwebagentsselfimprove`、`ni2026trace2skilldistilltrajectorylocallessons`、`he2026reskillreconcilingskillcreation`、`ding2026agentskillevaluationevolution`。这里仅保存key线索，没有核验题名/作者/版本/DOI或自动生成BibTeX。原idea里其他方法/论文链接同样属于待核验引用线索，不能直接从设计稿照搬为已读文献。

### 13.3 对接手写作 agent 的约束

交付范围是根据本文和原文件撰写 Methods/Experiment Setting，不替换主线程继续跑实验。优先把“定义与算法”“实验实例化”“历史/新结果边界”分开；不补造缺失超参数，不根据已有gold调整指标，不把proposal写成executed。必要参数未冻结时标为待确认，并指出具体配置字段。

保留所有已有未提交改动、历史失败和报告版本；不重跑旧实验、不覆盖旧报告、不提交或回滚。本文是本次唯一新增的整理文件，没有复制密钥或大体积实验数据。
