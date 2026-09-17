# Forecasting When Old Skills Turn Harmful After Policy Updates

> **Policy 更新后旧 Skill 有害边际效用翻转的低成本预测**  （不一定是翻转？可能是由低成本观测信号对边际效用改变的预测？）
> 一次 RL update 已经完成、但尚未运行完整 post-update evaluation 时，能否利用低成本的参数与功能偏移信号，预测哪些原本有效的 `(Skill, context)` 会在新 Policy 上从“有益”变为“有害”？

1. “有益”变为“有害”存在
2. 与老方法优势



归因早

update 的policy是否有预测能力



先证明RL训练后有change产生（稳定 频繁）-》有关联（预测能力）-》有增量价值（without post update rollout；比例；rollout）

## 0. 相对 2026-08-18 版本的更新摘要

| 更新 | 新版处理 | 更新理由 |
|---|---|---|
| 收窄主问题 | 从 generic compatibility drift 收敛为 **harmful marginal-utility sign flip forecasting** | ReSkill 已讨论 evolving Policy 下证据失效，SLIM 已做边际贡献与 lifecycle，泛化的 drift/lifecycle 难以作为主贡献 |
| 修改预测标签 | 主标签由任意 utility drop 改为原本有益的 Skill 从正边际效用翻为负边际效用 | 正效用下降不等于有害；`0.8→0.4`、`0.4→0` 与 `0.3→-0.2` 的含义不同 |
| 增加双轴分解 | 同时测量 Skill-free base performance 与 marginal Skill utility | 用于区分 Policy 整体 regression、internalization-consistent redundancy、Skill-specific harmful flip 与 positive synergy |
| 明确信号边界 | 延续旧版“parameter、activation、action shift 与 `D_t` 均为 candidate signals”的设定，并将该边界前置 | 这不是把旧版的机制解释降级；真正变化是将信号的验证目标从 generic utility drop 收窄为独立 evaluation 上的 harmful sign flip |
| 引入旧效用边界 | 预测器必须显式使用 update 前的边际效用及其置信度 | 相同强度的负向 shift 对“接近零”和“强正效用”的 Skill 造成符号翻转的概率不同 |
| 生命周期降级 | retain/refresh/retire 与 Skill editing 仅作为可选 downstream utility test | 生命周期操作本身已有充分先例；本文先回答“能否提前发现将变有害的 Skill-context pair” |
| 明确比较对象 | 与 raw parameter、activation、action、trajectory 和小预算 behavioral probe 做统一成本比较 | 核心不是设计更复杂的 predictor，而是判断 Policy update 中哪类低成本 readout 真有增量信息 |
| 区分两类泛化 | immediate held-out prediction 为主，later-update persistence 为次 | later updates 还受到后续 Policy 更新影响，不能与一次 update 后的即时预测混为同一因果问题 |



## 1. 核心研究主张

现有方法通常通过 post-update rollouts 直接观察 Skill 是否有用，或者根据失败轨迹事后总结 Skill 是否需要修改。本文研究更早的诊断时点：

$$
\theta_t
\xrightarrow{\text{one real RL update}}
\theta_{t+1}
\xrightarrow{\text{no full post-update evaluation}}
\widehat{\Pr}\!\left(\text{old Skill turns harmful}\right).
$$

不把“Policy 和 Skill 会共同变化”作为创新，也不把“Skill 有边际效用”作为新定义。核心贡献候选是：

> **使用一次真实 Policy update 产生的低成本参数/功能 readout，在完整 post-update rollout evaluation 之前，预测旧 `(Skill, context)` 的边际效用是否会发生有害符号翻转。**

现有工作已经覆盖 evolving Policy 下的 Skill lifecycle、paired marginal utility 和 counterfactual edit utility。因此，本文聚焦以下差异：

> **已有工作主要通过新的 rollout/validation 观察当前 Skill utility；本文把该 utility 作为离线 gold label，研究能否在不获得这些 post-update labels 时，从 Policy update 本身的低成本 readout 预测其符号翻转。**

## 2. Motivation

### 2.1 轨迹反馈在时间上滞后

一次 RL update 已经改变了 Policy 对 Skill 的解析与使用方式，但 trajectory-summary 方法只有在新 Policy 继续运行、积累足够成功/失败轨迹后才能发现问题。在长程环境中，等待完整失败证据的代价较高，并可能让有害 Skill 持续进入后续 rollout 和 Policy update。

本文希望使用 update 后立即可获得的信号，在完整 downstream evaluation 之前完成风险排序。这里的“提前”严格表示 **post-update、pre-evaluation**，不是在 update 发生前预测未来参数变化。

### 2.2 轨迹失败混合了多种来源

一条失败轨迹同时受以下因素影响：

- Policy 本身的能力变化；
- Skill 是否被正确检索；
- Skill 在当前状态是否适用；
- Policy 是否正确理解并执行 Skill；
- 长程 credit assignment；
- 环境与解码随机性。

因此，失败轨迹不能直接回答“是 Skill 失效，还是 Policy 整体退化”。同样，一个 Skill 的平均收益下降也不能自动被称为 incompatibility。

### 2.3 Utility drop 不等于 Skill 变得有害

假设某 Skill 的边际效用发生以下变化：

| 变化 | 可能含义 |
|---|---|
| `0.8 → 0.4` | Skill 变弱但仍然有益 |
| `0.4 → 0` | Policy 可能已内化相应能力，Skill 变得冗余 |
| `0.3 → -0.2` | 原本有益的 Skill 在新 Policy 上开始造成伤害 |
| `-0.1 → 0.3` | 新 Policy 与 Skill 形成正向协同 |

本文将第三类 **positive-to-negative sign flip** 作为主要预测目标；连续 utility change 只作为辅助分析。

## 3. Related Work 边界

| 类别 | 代表工作与已覆盖内容 | 本文区别 |
|---|---|---|
| Evolving Policy 与 lifecycle | [ReSkill](https://arxiv.org/html/2606.01619) 处理旧评估证据失效并在线比较 Skill 版本；[SLIM](https://arxiv.org/html/2605.10923) 用 leave-one-skill-out utility 做 retain/retire/expand；[PATS](https://arxiv.org/html/2607.21419) 测量 scaffold 收益的跨 checkpoint 变化 | 这些方法通过训练 rollout、validation 或 replay **观察** Skill utility；本文在这些 post-update labels 出现前预测细粒度 harmful sign flip |
| Paired Skill utility | [UCOB](https://arxiv.org/html/2606.29502) 在 anchor state 比较 Skill/no-Skill local return；[SkillC](https://arxiv.org/html/2605.27899) 用 paired contrast 测量 internalization gap | Paired utility 在本文中主要用于构造 gold label，不是创新；本文预测的是 $M_t>0\rightarrow M_{t+1}<0$ 的时间转移 |
| Skill edit utility | [SkillMaster](https://arxiv.org/html/2605.08693) 用 original/modified Skill Bank 的 probe utility 评价编辑 | 本文先预测哪些旧 Skill-context 值得进入昂贵的 audit/edit，再用统一编辑器做反向验证 |

## 4. 问题定义

### 4.1 分析单元与时间边界

一次真实 RL update 将旧 Policy 更新为新 Policy：

$$
\pi_{\theta_t}
\longrightarrow
\pi_{\theta_{t+1}},
\qquad
\Delta\theta_t=\theta_{t+1}-\theta_t.
$$

基本分析单元为：

$$
(t,s,c),
$$

其中 $t$ 是 update index，$s$ 是更新期间未改动的旧 Skill，$c$ 是一组语义和决策结构相近的 context/state cluster。主分析不研究 Skill retrieval error，假设候选 Skill 已给定，以隔离“Policy 更新后该 Skill 是否仍有益”。

$c$ 必须由环境状态 schema、任务子类型或只使用旧 Policy 数据构造的 anchor-state abstraction 预先确定，不能根据 post-update utility label 聚类。对 reward-directed signal，进一步记录本次 update batch 的有效支持量：

$$
n_t(s,c)
=
\sum_j
\mathbb I
\left[s_j=s,\ z_j\in c\right].
$$

只有 $n_t(s,c)\ge n_{\min}$ 时，才把 $P_t^{\mathrm{int}}$ 与 $D_t$ 视为可用预测量。低支持样本必须输出 `unsupported/abstain` 或只使用不依赖 on-batch reward attribution 的 functional-shift features，不能把缺少证据编码成零风险。实验同时报告该约束下的样本覆盖率。

### 4.2 Skill-free base、Skill-conditioned return 与边际效用

对相同 evaluation 实例 $x\in c$、环境状态、随机种子 $\xi$ 和解码配置，定义 Skill-free base performance：

$$
B_t(c)
=
\mathbb E_{x,\xi}
\left[
G_{\theta_t}^{\varnothing}(x,\xi)
\right],
$$

Skill-conditioned performance：

$$
W_t(s,c)
=
\mathbb E_{x,\xi}
\left[
G_{\theta_t}^{s}(x,\xi)
\right],
$$

以及 Skill 的边际效用：

$$
M_t(s,c)
=
W_t(s,c)-B_t(c).
$$

对应的时间变化为：

$$
\Delta B_t(c)=B_{t+1}(c)-B_t(c),
$$

$$
\Delta W_t(s,c)=W_{t+1}(s,c)-W_t(s,c),
$$

$$
\Delta M_t(s,c)
=
M_{t+1}(s,c)-M_t(s,c)
=
\Delta W_t(s,c)-\Delta B_t(c).
$$

$\Delta B_t$ 描述新 Policy 自身的变化；$\Delta M_t$ 描述 Skill 相对于新 Policy baseline 的额外价值变化。二者必须同时报告。

### 4.3 主要 gold label：harmful marginal-utility sign flip

点估计层面的定义为：

$$
H_t(s,c)
=
\mathbb I
\left[
M_t(s,c)>\tau_+
\ \land\
M_{t+1}(s,c)<-\tau_-
\right].
$$

实验中使用 paired bootstrap 或分层 bootstrap 构造置信区间。保守 gold label 定义为：

$$
H_t^{\mathrm{gold}}(s,c)
=
\mathbb I
\left[
\operatorname{LCB}\!\left(M_t(s,c)\right)>\tau_+
\ \land\
\operatorname{UCB}\!\left(M_{t+1}(s,c)\right)<-\tau_-
\right].
$$

置信区间跨越中性区间 $[-\tau_-,\tau_+]$ 的样本标记为 ambiguous，不强行赋予正/负标签。主结果同时报告保守标签的覆盖率，并在包含 ambiguous 样本的软标签/敏感性分析中复验，避免通过删除难样本夸大性能。

### 4.4 现象分桶采用双轴而非单一“兼容/不兼容”标签

边际效用转移使用三状态：

$$
\operatorname{sign}_{\tau}(M)
\in
\{+,0,-\}.
$$

核心转移是 $+\rightarrow-$。其余结果按以下现象报告：

| 现象标签 | 操作定义 | 谨慎解释 |
|---|---|---|
| Stable helpful | $M_t>\tau_+$ 且 $M_{t+1}>\tau_+$ | Skill 继续提供正边际价值 |
| Weakened but useful | 新旧均为正但 $\Delta M_t<0$ | Skill 价值下降，但不应淘汰 |
| Internalization-consistent | $+\rightarrow0$ 且 $\Delta B_t>\tau_B$ | 与 Policy 内化/Skill 冗余一致，但不是单凭该标签证明内化机制 |
| Harmful sign flip | $+\rightarrow-$ | 原本有效的 Skill 在新 Policy 上产生负边际效用；本文主目标 |
| Positive synergy | $0/-\rightarrow+$ 或 $\Delta M_t>\tau_M$ | 新 Policy 更能利用该 Skill |
| Pre-existing harmful | $-\rightarrow-$ | Skill 原本就有害，不应归因于当前 update |
| Global regression modifier | $\Delta B_t<-\tau_B$ | Policy baseline 退化；它是正交修饰标签，可与上述任一转移共存 |

“Global regression”不与其他类型强制互斥。例如，新 Policy 可能整体退化，同时某个 Skill 又从正效用翻为负效用。主结果应报告 `base change × marginal transition` 的联合矩阵，而不是把所有样本硬分成互斥原因。

## 5. Policy update 的候选低成本信号

### 5.1 Skill-conditioned functional shift

在固定 probe state $z$ 上，令 $a$ 表示可比较的动作/token 支持，定义：

$$
u_t^s(z,a)
=
\log\pi_{\theta_{t+1}}(a\mid z,s)
-
\log\pi_{\theta_t}(a\mid z,s).
$$

它描述真实 Policy update 在给定 Skill context 下如何改变动作偏好。

### 5.2 Skill-free functional shift

$$
u_t^{\varnothing}(z,a)
=
\log\pi_{\theta_{t+1}}(a\mid z,\varnothing)
-
\log\pi_{\theta_t}(a\mid z,\varnothing).
$$

它是 Policy baseline change 的无 rollout readout。它不能单独代表真实 RL 学习方向，但可帮助识别一般性 Policy regression 或能力提升。

它反映 Policy 基础行为的变化，主要用于判断：

- 更新是否独立于 Skill 发生；
- 能力是否已被 Policy 内化；
- Skill-conditioned 变化是否只是一般性 Policy drift。

### 5.3 Policy–Skill interaction shift

$$
\delta_t^s(z)
=
u_t^s(z)-u_t^{\varnothing}(z).
$$

展开为：

$$
\delta_t^s(z)
=
\left[
\log\pi_{\theta_{t+1}}(\cdot\mid z,s)
-
\log\pi_{\theta_t}(\cdot\mid z,s)
\right]
-
\left[
\log\pi_{\theta_{t+1}}(\cdot\mid z,\varnothing)
-
\log\pi_{\theta_t}(\cdot\mid z,\varnothing)
\right].
$$

$\delta_t^s$ 是 function space 的 difference-in-differences：它扣除了 update 对无 Skill 行为的共同影响，只保留 update 对 Skill 使用方式造成的额外改变。它是本文最重要的 candidate feature，但不是 utility label，也不是 causal mechanism。

### 5.4 Parameter projection

全局参数范数对同一次 update 的所有 `(s,c)` 相同，不能进行 update 内排序。参数信号必须投影到 state–Skill sensitivity：

$$
\delta_t^s(z)
\approx
\left[
J_{\theta_t}^{s}(z)
-
J_{\theta_t}^{\varnothing}(z)
\right]
\Delta\theta_t.
$$

右侧 Jacobian-vector product 是 parameter-projection baseline；左侧由新旧 Policy 的前向计算直接得到。一阶近似误差也应作为信号可靠性的诊断量，而不是默认忽略。

* Skill interaction 的变化 ≈ 参数更新 × 这些参数对“有 Skill / 无 Skill”两种行为的影响差异。

### 5.5 Activation interaction shift

对预注册层 $\ell$ 与 decision-token pooling，定义：

$$
\delta_{t,\ell}^{h,s}(z)
=
\left[
h_{\theta_{t+1},\ell}(z,s)-h_{\theta_t,\ell}(z,s)
\right]
-
\left[
h_{\theta_{t+1},\ell}(z,\varnothing)-h_{\theta_t,\ell}(z,\varnothing)
\right].
$$

层选择、token pooling、标准化与降维方式只能预注册或在 development split 上确定，不能根据最终 evaluation label 选择。

## 6. Reward-directed local signal

### 6.1 Update fidelity

对于生成 update batch 的旧 Policy、decision step $j$、实际动作 $a_j$ 和 advantage $\hat A_j$，定义 logit-space 的局部优化方向：

$$
d_t(z_j,s_j)
=
\hat A_j
\left[
\mathbf e_{a_j}
-
\pi_{\theta_t}(\cdot\mid z_j,s_j)
\right].
$$

它对应旧 Policy 上 policy-gradient 对 logits 的局部方向。首先验证真实 Skill-conditioned function shift 是否实现了该 reward-directed update：

$$
C_{t,j}^{\mathrm{upd}}
=
\cos_W
\left(
d_t(z_j,s_j),
u_t^{s_j}(z_j)
\right).
$$

$C_{t,j}^{\mathrm{upd}}$ 低表示局部 readout 未忠实反映预期的 reward-directed update，此时不宜继续做 Skill-specific 归因。

### 6.2 Skill-specific reward opposition

在 update fidelity 基本成立后，计算 interaction shift 沿 reward direction 的带符号投影：

$$
P_{t,j}^{\mathrm{int}}
=
\frac{
\left\langle
d_t(z_j,s_j),
\delta_t^{s_j}(z_j)
\right\rangle_W
}{
\left\|d_t(z_j,s_j)\right\|_W+\epsilon
}.
$$

- $P_{t,j}^{\mathrm{int}}>0$：Skill-specific change 相对 Skill-free baseline 强化 reward direction；
- $P_{t,j}^{\mathrm{int}}<0$：Skill-specific change 削弱 reward direction，是 harmful flip 的候选线索。

聚合后的 reward-opposing interaction score 为：

$$
D_t(s,c)
=
\mathbb E_{j:\,s_j=s,\,z_j\in c}
\left[
\mathbb I
\left(C_{t,j}^{\mathrm{upd}}\ge\tau_C\right)
\mathbb I
\left(\left\|\delta_t^{s_j}(z_j)\right\|_W\ge\tau_\delta\right)
\left[-P_{t,j}^{\mathrm{int}}\right]_+
\right].
$$

新版中 $D_t$ 的解释被严格限制为：

> **在可信 update steps 上，Policy 更新对 Skill 使用方式造成的额外改变，有多大程度与 rollout reward direction 相反。**

$D_t$ 高不等于已经发生 sign flip。它必须与旧边际效用 $M_t$、Skill-free shift、其他 readout 一起预测独立 gold label。

预测器中使用的簇级方向统计量由 step-level 量在预注册聚合规则下得到，例如：

$$
C_t^{\mathrm{upd}}(s,c)
=
\mathbb E_{j:\,s_j=s,\,z_j\in c}
\left[C_{t,j}^{\mathrm{upd}}\right],
$$

$$
P_t^{\mathrm{int}}(s,c)
=
\mathbb E_{j:\,s_j=s,\,z_j\in c}
\left[P_{t,j}^{\mathrm{int}}\right].
$$

除均值外的分位数、方差或正负 step 比例只能在 development split 中预先确定，不能根据最终 evaluation label 临时挑选。

因此，本节是两阶段测量：

$$
d_t\leftrightarrow u_t^{s_j}
\quad\text{检验 update fidelity},
\qquad
d_t\leftrightarrow\delta_t^{s_j}
\quad\text{诊断 Skill-specific effect}.
$$

### 6.3 信号可信度与负对照

至少保留以下验证：

1. 从同一 $\theta_t$ 出发，用不同 rollout minibatch/seed 独立更新，检验 $\delta_t^s(c)$ 与 $D_t(s,c)$ 的跨更新重复性；
2. 打乱 reward/advantage 后重新更新；
3. 用等范数随机或正交参数方向替换真实 $\Delta\theta_t$；
4. 在相同 state 输入语义无关的错误 Skill，排除一般 prompt sensitivity；
5. 同时报告不过滤样本与按预注册 fidelity/magnitude gate 过滤后的结果，避免只展示“看起来可信”的子集。

如果只有 trajectory-level advantage，$d_t$ 只能称为 outcome-consistent direction，不能解释为严格 step correctness。若可获得 process reward/PRM，则作为补充实验复验。

## 7. 从 candidate signal 到 sign-flip forecast

### 7.1 预测时可用的信息

在部署时点，允许使用：

- update 前已经积累的 Skill utility 估计 $\widehat M_t(s,c)$ 及其不确定性；
- 真实 Policy update 的 $\Delta\theta_t$；
- 固定 replay/probe states 上的新旧 Policy 前向结果；
- update batch 已有的 actions、advantages、rewards 和 trajectories；
- 可选的小预算 post-update behavioral probes。

禁止使用：

- 完整 post-update held-out rollout evaluation；
- 由最终 evaluation split 选择的阈值、层或特征；
- 未来 update 的数据。

### 7.2 为什么必须加入旧 Skill margin

同样的负向 interaction shift 对两个 Skill 的意义不同：若一个 Skill 原先只有微弱正效用，它更容易跨过零点；若原先效用显著为正，则需要更强的负向变化才会翻转。因此预测输入至少包含：

$$
x_t(s,c)
=
\left[
\widehat M_t,
\operatorname{SE}(\widehat M_t),
u_t^{\varnothing},
\delta_t^s,
C_t^{\mathrm{upd}},
P_t^{\mathrm{int}},
D_t
\right].
$$

主预测任务为：

$$
\widehat p_t^{\mathrm{flip}}(s,c)
=
f_\phi\!\left(x_t(s,c)\right)
\approx
\Pr\!\left(H_t^{\mathrm{gold}}(s,c)=1\right).
$$

为了不把贡献变成复杂 classifier，采用两条评估轨道：

1. **Direct-score track：** 直接用 $D_t$、$P_t^{\mathrm{int}}$、$\|\delta_t^s\|$ 排序，并与 old-margin-only baseline 比较；
2. **Equal-capacity probe track：** 对每类信号使用相同容量、相同正则化的 logistic/linear probe，在 earlier updates/development contexts 上拟合，在 later updates/held-out contexts 上测试。

预测器只是测量不同表示包含多少可泛化信息，不是主要算法贡献。

### 7.3 两阶段验证闭环

**第一阶段验证诊断准确性：** RL update 后先计算候选信号并对 Skill-context pairs 排序；随后才在独立 held-out instances 上运行 matched evaluation，构造真实 sign-flip label，检验 AUPRC、Precision@K 与 calibration。

**第二阶段验证进化指导价值：** 冻结新 Policy，按风险排序选择相同数量的 Skill，交给统一编辑器修改；再与 trajectory-summary、随机选择和 behavioral probe 等方法比较修改后的 held-out utility 与错误修改率。

新增约束不改变“信号排序 → 修改 Skill → 性能后验证”的原始逻辑，只用于保证第一阶段标签和比较可信：

- **旧 Skill margin：** 表示 Skill 距离效用零点有多远；
- **置信区间与 ambiguous label：** 避免把 rollout 波动误判为正负翻转；
- **低支持样本 abstention：** 避免把缺少 `(s,c)` update evidence 误当成低风险；
- **按 update 划分数据：** 避免共享同一 $\Delta\theta_t$ 的样本同时进入训练集和测试集。

### 7.4 候选信号组

| 信号组 | 内容 | 新环境 rollout 成本 |
|---|---|---:|
| Old evidence | $\widehat M_t$、置信区间、历史使用频率 | 0 |
| Raw update | 全局/逐层 $\|\Delta\theta_t\|$、KL、训练 loss | 0 |
| Parameter projection | state–Skill JVP、线性化误差 | 0；需要额外反向/JVP |
| Activation | $\delta_{t,\ell}^{h,s}$ | 0；需要新旧 × 有/无 Skill 前向 |
| Action distribution | $u^s$、$u^{\varnothing}$、$\delta^s$、$P^{\mathrm{int}}$、$D_t$ | 0；需要新旧 × 有/无 Skill 前向 |
| Existing trajectory | update batch reward、advantage、failure type、LLM summary | 0 新 rollout |
| Budget-matched post-update trajectory | 新 Policy 的 $k$ 条普通 trajectories 及其 reward/summary | $k$ 条 rollout |
| Lightweight behavioral probe | 新 Policy 在 $k$ 个 matched states 上的 Skill/no-Skill 短程 continuation gap | $2k$ 条短程 continuation |
| Full evaluation | 大规模 matched Skill/no-Skill rollout | gold label/oracle，不作为在线信号 |

Lightweight behavioral probe 的作用不是生成完整任务轨迹，而是在少量代表状态上直接测量新 Policy 的局部行为差异。改变 $k$ 和 continuation horizon 可得到 accuracy–cost curve。

## 8. Research Questions 与假设

### 8.1 Research Questions

**RQ1：Harmful sign flip 是否可被提前预测？**  
一次真实 RL update 后，无新环境 rollout 的 Policy-shift readout 能否预测原本有益的 `(Skill, context)` 在新 Policy held-out instances 上发生 $+\rightarrow-$ 边际效用翻转？

**RQ2：哪类 readout 提供最佳精度—成本权衡？**  
Parameter projection、activation interaction、action-distribution interaction、existing trajectory signal 与 $k$-shot behavioral probe，谁在相同成本报告协议下具有最高 AUPRC、Precision@K 与 calibration？

**RQ3：Skill-specific interaction 是否具有增量预测价值？**  
在控制旧 Skill margin、Policy baseline shift、update magnitude 和 trajectory reward 后，$\delta_t^s$、$P_t^{\mathrm{int}}$ 或 $D_t$ 是否仍能改善 held-out sign-flip prediction？

**RQ4：双轴分解能否减少错误归因？**  
联合 $\Delta B_t$ 与 $M_t\rightarrow M_{t+1}$，是否能刻画 harmful flip、internalization-consistent redundancy 与 positive synergy，并将 global regression 作为正交修饰因素标出，避免把所有 utility drop 都标为 incompatibility？

**次要 RQ：预测是否跨后续 updates 持续？**  
即时 sign-flip 风险能否预测 Skill 在 $\theta_{t+k}$ 上仍保持负边际效用？这只作为 temporal persistence，不解释为一次 update 的直接因果效应。

### 8.2 假设

- **H1：** 自然 RL updates 中存在数量非零且跨 Skill/context 异质的 $+\rightarrow-$ sign flips。
- **H2：** 旧边际效用是强 baseline，但 Skill-specific functional interaction 对 sign flip 具有额外预测价值。
- **H3：** Raw parameter norm 的 update 内区分能力有限；经过 state–Skill sensitivity 投影的 parameter/activation/action readout 更有效。
- **H4：** 最佳 no-new-rollout signal 在显著更低 rollout 成本下接近小预算 behavioral probe，并优于 trajectory-summary-only attribution。
- **H5：** 将 base regression 与 marginal transition 分开后，错误地把 Policy regression/internalization 判为 Skill harm 的比例下降。

## 9. Gold label 与数据划分

### 9.1 Matched evaluation

对每个 `(t,s,c)`，离线构造 gold labels 时运行旧/新 Policy × Skill/no-Skill 四个条件：

$$
\left\{
(\theta_t,s),
(\theta_t,\varnothing),
(\theta_{t+1},s),
(\theta_{t+1},\varnothing)
\right\}.
$$

四个条件从相同保存状态开始，并匹配任务实例、环境随机种子、候选动作集合、temperature、top-p、最大步数和 history truncation。主标签使用 binary success/continuation return，shaped return 与 action efficiency 作为次要指标。

部署时不需要重新估计完整 $M_t$；它来自历史 evidence。四条件 evaluation 仅用于研究阶段构造无泄漏 gold labels。

### 9.2 三类数据集合

1. **Signal probe bank：** 固定 states，只用于计算 parameter/activation/action readout，不运行完整环境任务；
2. **Development split：** 用于阈值、层选择、特征标准化和等容量线性 probe；
3. **Held-out evaluation split：** 只用于构造最终 $B$、$W$、$M$ 与 sign-flip labels。

必须按 update 分组切分，不能把同一次 $\Delta\theta_t$ 下的不同 Skill-context pairs 随机分到 train 与 test。主 temporal split 使用 earlier updates 训练/开发、later updates 测试；另设 held-out task instances，条件允许时增加 held-out Skill family。

### 9.3 Immediate held-out 与 later-update persistence 分开

主要标签是同一次 update 后立即测得的：

$$
H_{t\rightarrow t+1}^{\mathrm{gold}}(s,c).
$$

次要 persistence 标签为：

$$
H_{t\rightarrow t+k}^{\mathrm{persist}}(s,c)
=
\mathbb I
\left[
M_t(s,c)>\tau_+
\land
M_{t+k}(s,c)<-\tau_-
\right].
$$

测量 persistence 的期间冻结 Skill Bank，不执行 Skill edits；否则未来标签会受到诊断系统自身行为影响。即使如此，$H^{\mathrm{persist}}$ 也只表示预测稳定性，不用于宣称一次 update 对远期结果的直接因果效应。

## 10. 最小实验设计

### 10.1 环境与训练配置

- 主环境：ALFWorld，原因是任务成功可验证、存在明确子任务类型，并可保存或重建 matched decision contexts；
- Policy：8B instruct model；
- RL：GRPO 或相同 group-based agentic RL；
- 资源：8 × A100；
- Skill Bank：在 Phase I 全程冻结内容，仅允许 Policy 更新；
- checkpoint：保存连续 update 前后参数、optimizer metadata、rollout actions、advantages、rewards 与 probe states。

ScienceWorld/WebShop 只在 ALFWorld 主结果成立后用于外部复验，不进入最小可行版本。

### 10.2 样本构造

1. 在 $\theta_t$ 上从历史 evidence 中选择确认有正边际效用的 `(s,c)`；
2. 执行一次正常 RL update 得到 $\theta_{t+1}$；
3. 在固定 probe states 上计算所有 no-new-rollout signals；
4. 在独立 held-out instances 上执行 matched evaluation，仅用于构造 $H_t^{\mathrm{gold}}$；
5. 跨多个连续 updates 和独立 seeds 重复。

不得只筛选表现出大 shift 的样本再构造 labels。自然 sign flip 的总体发生率必须报告。若主数据中可靠正例数量不足以训练/评估 predictor，应将结论降级为“事件频率与检测可行性研究”；可以预注册更长 checkpoint interval 作为次要标签，但不能事后扩大间隔制造翻转。

### 10.3 Baselines

1. Random ranking；
2. Old-margin-only：仅使用 $\widehat M_t$ 与不确定性；
3. Update-level statistics：KL、loss、global/row-wise parameter norm；
4. Trajectory reward/advantage statistics；
5. Trajectory + LLM failure summary；
6. Budget-matched post-update trajectory reward/summary；
7. Parameter projection；
8. Activation interaction shift；
9. Action-distribution shift；
10. $P_t^{\mathrm{int}}$ 与 $D_t$；
11. Old margin + best interaction signal；
12. 不同 $k$ 和 horizon 的 lightweight behavioral probe；
13. Full matched evaluation，作为 oracle upper bound，不计入可部署方法排名。

如果需要与现有 lifecycle 方法做实现级比较，应将 ReSkill/SLIM 式 rollout audit 按相同 post-update rollout budget 实现，而不是把论文报告数字直接并排比较。

### 10.4 主要指标

由于 harmful sign flip 可能稀有，主指标为：

- AUPRC；
- Precision@K / Recall@K，其中 $K$ 对应可承担的 audit/edit budget；
- Brier score 与 Expected Calibration Error；
- 相同 rollout/GPU budget 下的 recall；
- risk–coverage curve 与 abstention rate，检验低支持样本不输出判断时的代价；
- false alarm rate，特别是把 Stable helpful 或 Internalization-consistent 样本误判为 harmful flip 的比例。

AUROC、Spearman 与连续 $\Delta M_t$ 回归误差只作为次要指标。置信区间按 update 聚类 bootstrap，避免把同一次 Policy update 下的多个 `(s,c)` 当作独立样本。

### 10.5 关键消融

- 去掉 old margin；
- 去掉 Skill-free branch；
- 只用 $\|\delta_t^s\|$，去掉 reward-directed sign；
- 去掉 update fidelity gate；
- 用随机/正交等范数 update 替代真实 update；
- 用错误 Skill 替代真实 Skill；
- 不同层的 activation readout；
- trajectory-level advantage 与 process reward；
- immediate held-out 与 later-update split。

## 11. 第二阶段：Skill 修改的反向验证

本章保留旧版“按候选信号排序后修改 Skill，再用性能收益反向验证”的逻辑。它不是主预测结论成立的前提；只有第一阶段表明某个 no-new-rollout signal 对 sign flip 有稳定增量价值后，才进行该预算受限验证。

在固定新 Policy $\theta_{t+1}$、固定编辑器和相同候选数下比较：

1. No Evolution；
2. Random audit/edit；
3. Trajectory-summary-guided audit/edit；
4. Old-margin-only guided audit/edit；
5. Sign-flip-risk-guided audit/edit；
6. $k$-shot behavioral-probe-guided audit/edit；
7. Oracle sign-flip label guided audit/edit。

主要比较谁能在相同 audit/edit budget 下获得更高的 harmful-Skill removal/repair rate、更低的 false edit rate，以及更大的 held-out performance gain。具体采用 retain、refresh 还是 retire 由共享编辑器决定，不作为本文的方法创新。

最关键对比是组 3 与组 5：它直接回答 Policy-update shift guidance 是否比纯失败轨迹归因更适合选择“值得检查/修改”的 Skill-context pairs。

## 12. 机制结论边界

本研究首先是 predictive/diagnostic study。即使 parameter projection、activation shift 或 $D_t$ 能预测 sign flip，也只能说明该 readout 包含与未来边际效用相关的信息，不能说明它是 Skill 失效的生成机制。

只有在主预测结果成立后，才考虑以下探索性干预：

$$
\theta_t(\lambda)
=
\theta_t+\lambda\Delta\theta_t,
\qquad
\lambda\in\{0,0.25,0.5,0.75,1\},
$$

以及 block/LoRA update injection、activation patching。若干预同时按剂量改变 interaction signal 与 matched marginal utility，且显著强于随机等范数方向，才可提出受限的作用路径解释。否则结论停留在“低成本 readout 可以预测”。

现象分桶也不是机制证明：“Internalization-consistent”只表示观测模式与内化解释一致；harmful sign flip 可与 Skill–Policy interference 解释一致，但不能单凭符号翻转证明 interference 的作用机制。

## 13. 预期贡献

1. 将研究对象从宽泛的 compatibility drift 收敛为 **Policy-update-induced harmful marginal-utility sign flip**，并给出带不确定性的可检验标签；
2. 提出 Skill-free base change 与 marginal Skill transition 的双轴分解，避免把 Policy regression、冗余化和有害翻转混为同一问题；
3. 在统一 held-out labels、相同 predictor capacity 与显式计算预算下，系统比较 parameter、activation、action-distribution、trajectory 与 behavioral readouts；
4. 检验 reward-aligned Policy–Skill interaction shift 是否在旧 Skill margin 和一般 Policy shift 之外提供增量预测信息；
5. 通过预算受限的 audit/edit 后验证，检验提前预测能否比 trajectory-summary attribution 更有效地指导 Skill evolution。

本文不声称以下内容是创新：定义 Skill marginal utility、发现 Skill 可能有害、提出 Skill lifecycle、设计 retain/refresh/retire 操作、用 paired rollout 直接测量 Skill utility，或证明 parameter shift 是因果机制。

## 14. 论文成立所需的证据链

1. 自然 RL updates 中存在足够数量、置信区间明确的 $+\rightarrow-$ harmful sign flips；
2. 该事件不能仅由旧 Skill margin、Policy 全局 regression 或 update magnitude 解释；
3. 至少一种 no-new-rollout readout 在跨 update、跨 seed、held-out context 上稳定提高 AUPRC、Precision@K 或 calibration；
4. 该增量在加入 old margin、Skill-free baseline 与 trajectory statistics 后仍然存在；
5. 其 accuracy–cost 位置优于 trajectory-summary，并相对小预算 behavioral probe 形成有意义的 Pareto trade-off；
6. 双轴分解显著减少把 internalization-consistent/global-regression 样本误判为 harmful Skill 的比例；
7. 在固定 Policy 与统一编辑预算下，risk-guided audit/edit 比 trajectory-guided 方法带来更高收益或更少错误修改；
8. 若第 3–7 项不成立，应将结论限制为“Policy update readout 未能替代 rollout-based utility audit”，而不是继续主张 synchronous Skill evolution；
9. 若没有干预证据，不作 parameter/activation causal mechanism 声称。

## 15. 最小版本的停止条件

Phase I 只回答三个问题：

1. 有害 sign flip 是否真实且可稳定测量；
2. 无新 rollout 的 Policy-update signal 是否能预测它；
3. 该信号相对 trajectory 和小预算 behavioral probe 是否具有成本优势。

只有这三点成立，才进入 Skill editing、online co-evolution 和机制干预。这样可以避免在核心预测命题尚未成立前，将工作扩张为难以归因的全流程系统。





![image-20260819210252684](C:\Users\WangYiFan\AppData\Roaming\Typora\typora-user-images\image-20260819210252684.png)
