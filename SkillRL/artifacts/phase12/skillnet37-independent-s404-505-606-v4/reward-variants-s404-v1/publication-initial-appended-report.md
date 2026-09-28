# Phase2 完整实验分析：Policy update 是否包含 Skill 效用变化的预测信息



## 1. 结论摘要

* **phase2 目标**

  核心研究目标是：真实 policy update 中，加入 reward direction 的局部读出，能否相比只测变化幅度的读出，更好地排序 Skill 边际效用的变化方向/下降风险，进而定位优先 audit/edit 的 Skill。C_upd、C_upd_centered、P_int、gated/ungated D 都是这一信号族的候选；不要求 D 优于另外几个 reward-directed 指标，不要求训练预测器或精确回归效用数值，也不以正→负 flip 为必要条件。增量价值不应只用 Top-1 判定：应评估整体前置能力、排序单调性与预算效益。

    - 下降单元平均名次：真正下降的 Skill 平均排在第几名。
    - AP（平均精确率）：综合整条排序，衡量下降对象是否集中在前端。
    - Spearman：比较风险分数与连续 (-\Delta M) 的整体名次是否一致。
    - Kendall：逐对比较两个 Skill：预测谁风险更高，实际是否也是谁下降更多。顺序一致的比例越高，指标越接近 +1。
    - 多个预算下的 Top-k 命中：只允许检查／更新前 k 个 Skill 时，能找到多少下降对象。通常报告 Precision@k＝命中的下降数/k，以及 Recall@k＝命中的下降数/全部下降数。
      
    - 下降量覆盖：选中的 Skill 覆盖了全部下降损失的多少：
      $$
      \frac{\sum_{\text{选中}}\max(0,-\Delta M)}
      {\sum_{\text{全部}}\max(0,-\Delta M)}
      $$
      越高越好；没有下降时未定义。

* **效用变化现象**：
  * 旧 U34→U35 的 cle_004 下降 −14.29 pp，95% CI [−28.57,−1.79]；
  * 新 U35→U40 的 cle_003 上升 +15.73 pp，95% CI [+2.42,+29.03]。

* **旧 U30→U35 的全部 19 个支持单元中存在 D 的整体排序优势**
  
  * 19 个支持单元中，**gated D** 的下降平均名次为 4.00、AP=0.799、Spearman(D,−ΔM)=+0.417
  * **centered norm（衡量变化强弱，无变化具体方向）** 分别为 4.50、0.667、+0.062。
  * **优势边界**：
    * 旧后段 7 单元在 2 降/2 升中计算的条件 AUROC 为 D=1.000、centered norm=0.750、KL=0.500；但对唯一超过 5 pp 的下降事件，AP 为 D=0.500、norm=0.333、KL=1.000。C_upd/C_upd_centered （扣除 u 在整个词表上的共同偏移，再计算夹角）的后段任意下降 AP 也均为 1.000，后者的下降 Spearman=+0.667 高于 D 的 +0.556。但 C 同样使用 reward-directed local signal，因此它的良好表现也是信号族预测价值的正面证据。

* **引入 reward-directed local signal的预测优势**

  同一基础向量的补充对照加强了这一解释：旧全部 19 单元中，C_upd 对下降的 Spearman/AP 为 +0.336/0.377，对应 u_original_norm 为 +0.132/0.288；D 对应 +0.417/0.799，同源的 delta_norm 为 −0.004/0.558。

  U34→U35 单个窗口内，C_upd 的 Spearman=+0.949，高于 u_original_norm 的 +0.738。这些不需要拟合预测器，是 reward-directed 读出相比对应幅度读出的描述性优势。

* **是否加入门控**

  门控与无门控分别评价：旧 19 单元中，ungated D 的下降平均名次进一步改善至 3.25、AP=0.854，但连续下降排序相关 +0.284 低于 gated D 的 +0.417。在 U31→U32 窗口内，ungated D 将唯一下降 Skill 排第 1，centered norm 排第 2，gated D 排第 3；因此不能概括为“所有 D 版本在所有排序口径下均无优势”，也不能说门控已被证明有益。

* **P的Spearman稳定性**

  P 的窗口内线索不能被 pooled 相关概括：五个旧窗口中四个、加上新窗口，共六个窗口中五个的 Spearman(P,ΔM) 为正；但每窗只有 3–4 个 Skill，多个标签为零。

综合结论：

> Phase2 的初步结果支持 reward-directed policy-update 读出这一信号族的预测价值：旧 C/D 在部分整体及窗口内方向排序上优于对应幅度读出，P 在新窗口和多数窗口的相对排序上也有线索。

## 2. 两个批次的证据范围与可比性

| 项目 | 旧单步 pilot | 新扩大窗口 |
|---|---|---|
| 同一 Seed303 延续路径 | U30→U35，5 个相邻窗口 | U35→U40，1 个五步窗口 |
| 评估端点 | U30–U35，共 6 个 | U35/U40，共 2 个 |
| anchors | 旧 B0 的 200 个首次调用 anchors | U35 自然 rollout 新采集的 200 个 anchors |
| 每 anchor–arm–端点 | 1 evidence + 1 gold | 2 evidence + 4 gold |
| 三臂 suffix 总数 | 7,200 | 7,200（U35 导入 3,600；U40 新增 3,600） |
| 主要比较 | 时间外 readout/等容量 probe | 预锁定同窗口直接 top-k 排序 |

Phase1 的多 seed 结果只作为研究动机：支持效用会变化及 S_int 与变化的关联。Phase2 当前仍是一条 seed303 路径，不能将 Phase1 的跨 seed 泛化直接转写为 P/D 的跨 seed 泛化。

## 3. 实际 RL 更新和测量协议

Qwen3.5-4B、ALFWorld clean、GRPO；8 games × 4 rollouts=32 条/update，LR=1e-6，KL=0.01；decision minibatch=32，microbatch=1，prompt/response=2048/64，最多 30 环境步。model、Adam 状态和 scheduler 连续恢复；global update 不等于一次 Adam step。硬件/分片数随资源改变，不声称固定并行布局的逐 bit 随机路径复现。

| cohort   |   update |   rollouts |   Adam_steps |   Adam_before |   Adam_after |   parameter_L2 |   relative_L2_percent |
|:---------|---------:|-----------:|-------------:|--------------:|-------------:|---------------:|----------------------:|
| pilot    |       31 |         32 |           24 |           495 |          519 |       0.24903  |             0.0325582 |
| pilot    |       32 |         32 |           22 |           519 |          541 |       0.266719 |             0.0348708 |
| pilot    |       33 |         32 |           27 |           541 |          568 |       0.278805 |             0.036451  |
| pilot    |       34 |         32 |           22 |           568 |          590 |       0.199623 |             0.0260986 |
| pilot    |       35 |         32 |           21 |           590 |          611 |       0.178982 |             0.0234001 |
| expanded |       36 |         32 |           18 |           611 |          629 |       0.214879 |             0.0280933 |
| expanded |       37 |         32 |           12 |           629 |          641 |       0.134042 |             0.0175246 |
| expanded |       38 |         32 |           12 |           641 |          653 |       0.142957 |             0.0186902 |
| expanded |       39 |         32 |           14 |           653 |          667 |       0.184265 |             0.0240907 |
| expanded |       40 |         32 |           14 |           667 |          681 |       0.165251 |             0.0216048 |

参数 L2 是去除 tied-head 重复计数后的 FP32 差分；累计净位移不是各单步 L2 相加。新窗口共 160 条训练 rollout、70 次 Adam steps（611→681），U35→U40 净参数 L2=0.517594766。所有 intermediate FP32 policy、实际 batch、advantage/mask 和 old/new 全词表概率保留。

Bank 冻结为 12 general + 32 task-specific；clean 候选为 12+6。新 anchor 覆盖：

| skill   |   anchors |   games |   first_step_min |   first_step_max |   initial |   early |   middle |   late |
|:--------|----------:|--------:|-----------------:|-----------------:|----------:|--------:|---------:|-------:|
| gen_002 |        50 |      29 |                2 |               29 |         0 |      26 |       23 |      1 |
| cle_003 |        50 |      31 |                3 |               11 |         0 |      20 |       30 |      0 |
| cle_004 |        50 |      31 |                1 |                1 |         0 |      50 |        0 |      0 |
| cle_006 |        50 |      31 |                0 |                0 |        50 |       0 |        0 |      0 |



## 5. 旧单步结果：保留但不混入新测试



### 5.1 总体关联

* **范围**

  - 全部 19 单元：合并 U30→31、31→32、32→33、33→34、34→35 五个窗口。每个窗口评估 4 个 Skill，原本 20 个单元，其中 1 个支持不足被排除。

* **各 signal** 

    - P_int：interaction shift 沿 reward direction 的有符号投影；负值表示反向，因此使用 −P 排下降风险。

    - D_contribution：通过门控后，P 的负向部分的平均值。

    - delta_norm：interaction shift 的平均原始幅度。

    - delta_centered_norm：去除词表共同偏移后的平均 interaction 幅度。

    - old_margin：更新前的 Skill 语义边际效用。

    - C_upd_centered：去均值后的 Skill-conditioned 更新与 reward direction 的平均对齐程度。


* **结果分析**

    - Centered norm，全部 19 单元：幅度相关 0.665，下降相关只有 0.062。说明它与“变化有多大”关联较强，但对“是否朝下降方向变化”的排序较
      弱。


    - D，全部 19 单元：下降相关 0.417、AP 0.799，说明它在这组数据中更能把下降单元排到前面。


  这些是小样本描述性结果。AP 的“下降”指 ΔM 点估计为负，不要求置信区间排除 0；19 个与 7 个也不是两组独立证据。

| signal              | risk_orientation | units | updates | rho_raw_magnitude（变化幅度相关系数） | rho_risk_decline（风险分数与 −ΔM 的 Spearman 相关） | ap_any_decline（将所有单元按风险排序后，实际下降单元是否集中在前面） |
| :------------------ | :--------------- | ----: | ------: | ------------------------------------: | --------------------------------------------------: | -----------------------------------------------------------: |
| P_int               | -value           |    19 |       5 |                              0.350334 |                                          -0.0488495 |                                                     0.189552 |
| D_contribution      | +value           |    19 |       5 |                              0.301405 |                                        **0.417174** |                                                 **0.798611** |
| delta_norm          | +value           |    19 |       5 |                              0.522566 |                                         -0.00390796 |                                                     0.558333 |
| delta_centered_norm | +value           |    19 |       5 |                               0.66544 |                                           0.0615503 |                                                     0.666667 |
| old_margin          | +value           |    19 |       5 |                          **0.797527** |                                          -0.0424515 |                                                     0.583333 |
| C_upd_centered      | +value           |    19 |       5 |                           -0.00195718 |                                            0.319476 |                                                     0.360417 |



### 5.2 旧后段子集的四口径比较（历史时间外划分）

以下来自旧 U34/U35 两个后段窗口，共 7 个有支持单元，是旧 19 单元的子集，当前作为分段敏感性分析而非额外独立样本。D 指主 gated D；三个分数均按原值越高风险越大排序。信号曾在目标 gold 前锁定，但本比较口径为事后审计，不是新增的确认性测试。

| 测试口径 | D | Centered interaction norm | KL |
|---|---:|---:|---:|
| 与连续下降 −ΔM 的 Spearman；全部 7 单元 | **0.556** | 0.185 | 0.037 |
| 任意下降 AP；2 个下降对其余 5 个单元 | **1.000** | 0.833 | 0.700 |
| 条件方向 AUROC；仅 2 个下降对 2 个上升 | **1.000** | 0.750 | 0.500 |
| 下降超过 5 pp 的 AP；1 个事件对其余 6 个单元 | 0.500 | 0.333 | 1.000 |



### 5.3 Reward-directed 信号族：C、P、D 的贡献层次

- **核心问题**：reward-directed 读出是否比不使用 reward direction 的幅度读出包含更多效用方向/下降风险排序信息。C、C_centered、P、D 的优势都可提供正面证据。
- **次级问题**：在 reward-aligned 的一般 Skill-conditioned 更新 u_original 之外，引入 ORIGINAL−control 的 interaction 差分 δ_int 是否还有额外价值。C 与 P/D 的比较涉及这一层。
- **构造选择问题**：D 的负部、门控和聚合是否优于 P/C 或其他构造。它决定最终使用什么读出，不是核心命题成立的先决条件。

| 范围 | 风险读出 | ρ(score,−ΔM) | 任意下降 AP | 条件下降/上升 AUROC | 超过 5 pp 下降 AP |
|---|---|---:|---:|---:|---:|
| 19 单元，pooled | D | **+0.417** | **0.799** | **0.938** | 0.250 |
| 同上 | C_upd | +0.336 | 0.377 | 0.812 | 0.333 |
| 同上 | C_upd_centered | +0.319 | 0.360 | 0.812 | 0.200 |



### 5.4 补充分析：同一基础向量的 reward-directed / 幅度对照

为了更直接地观察“加入 reward direction 的读出是否比只看幅度更有方向信息”，应尽量保持基础向量一致：

| 基础偏移 | Reward-directed 读出 | 同源幅度基线 | 当前可验证范围 |
|---|---|---|---|
| u_original：ORIGINAL 条件的新旧 log-prob 变化 | C_upd=cos(d,u_original) | u_original_norm | 已归档，可直接比较 |
| centered u_original | C_upd_centered | norm(centered u_original) | 该范数不在当前汇总特征中；不以 interaction norm 冒充 |
| δ_int=u_original−u_control | P_int、gated/ungated D | delta_norm；centered interaction norm 为中心化敏感性对照 | 已归档，可直接比较 |

#### 5.4.1 同源对照的总体与后段结果

ρ 为风险分数与连续 −ΔM 的 Spearman；AP/平均名次以点估计任意下降定义。以下每个单元格先列 reward-directed 读出，后列幅度基线；只能在同一行比较，不能将不同候选数的名次或不同下降比例的 AP 直接混比。

| 基础向量 | reward-directed 读出 / 幅度基线 | 范围 | 平均下降名次：读出 / 基线 ↓ | AP：读出 / 基线 ↑ | ρ：读出 / 基线 ↑ | 优势 |
|---|---|---|---:|---:|---:|----|
| u_original | **C_upd** / u_original_norm | 旧全部 19 | **6.50** / 9.00 | **0.377** / 0.288 | **+0.336** / +0.132 | C_upd |
| u_original | **C_upd** / u_original_norm | 旧后段 7 | **1.50** / 2.00 | **1.000** / 0.833 | **+0.519** / +0.296 | C_upd |
| u_original | C_upd / u_original_norm | 新窗口 3 | **2.00** / 2.50 | **0.833** / 0.583 | −0.500 / −0.500 | C_upd |
| δ_int | D / delta_norm | 旧全部 19 | **4.00** / 6.00 | **0.799** / 0.558 | **+0.417** / −0.004 | D |
| δ_int | D / delta_norm | 旧后段 7 | **1.50** / 2.00 | **1.000** / 0.833 | **+0.556** / +0.296 | D |
| δ_int | D / delta_norm | 新窗口 3 | 2.50 / 2.50 | 0.583 / 0.583 | −0.500 / −0.500 | 持平 |
| δ_int | −P / delta_norm | 旧全部 19 | 12.75 / **6.00** | 0.190 / **0.558** | −0.049 / **−0.004** | delta_norm |
| δ_int | −P / delta_norm | 旧后段 7 | 6.50 / **2.00** | 0.226 / **0.833** | −0.408 / **+0.296** | delta_norm |
| δ_int | −P / delta_norm | 新窗口 3 | **1.50** / 2.50 | **1.000** / 0.583 | **+1.000** / −0.500 | −P |

* 旧全部样本中，C 与 D 均在同源幅度基线之上呈现更好的下降 AP/相关；新窗口则是 −P 相比 delta_norm 更好。
* 旧 P 的 pooled 排名明显弱于 delta_norm，也完整保留。



#### 5.4.3 对“reward 增量”的解释边界

同一基础向量对照比直接拿 C 与 interaction norm 相比更接近所要检验的问题，但仍不是只改变一个因素的机制实验：C 使用角度归一化及无效方向零填充，P 使用方向投影，D 还包含负部截断与门控。当前比较显示这些 reward-directed 构造具有部分排序收益，不能把每项差异都严格归因于 reward 内容本身。

更强的验证可在预先固定方案后，保留相同 u/δ、支持样本和聚合方式，对比真实 reward direction 与预定的打乱/随机方向，并在新窗口/seed 上评价；这里仅列为后续验证，未执行，也不在当前 gold 上调参。C 的正面结果可支撑核心信号族命题；是否还需要 interaction 差分、负部或特定门控，是进一步的问题。





## 6. 新 U35→U40 窗口：对旧结果的关键补充

本节只保留新窗口对方向排序结论的增量信息。该窗口沿同一 seed303 从 U35 继续训练 5 个 updates；不是另一 seed 的复现。新旧批次的 anchors 和 continuation seeds 不同，因此分开比较，不把旧 19 单元与新 3 单元混池，也不把差异仅归因于窗口变大。完整预算、对照、轨迹和 phase 表见第 9 节归档。

### 6.1 预测目标与核心读出

主目标继续使用语义边际效用 M=ORIGINAL−PLACEBO，ΔM=M_U40−M_U35。每个 Skill 有 50 个固定首次调用 anchors，每个 anchor–arm–端点有 4 次 gold 续跑；效用按 game 等权汇总。效用变化和 CI 单位为 pp，P/D/norm 保留原始读出尺度。

| Skill | 信号支持 | games | ΔM，pp | 配对 95% CI，pp | 原始 P | gated D | centered interaction norm |
|---|---|---:|---:|---|---:|---:|---:|
| gen_002 | 是 | 29 | −3.02 | [−7.76, 0.00] | +0.059212 | 0.056023 | 113.505 |
| cle_003 | 是 | 31 | +15.73 | [+2.42, +29.03] | +0.091699 | 0.087943 | 223.899 |
| cle_004 | 是 | 31 | −6.45 | [−14.92, +1.21] | −0.012389 | 0.076906 | 153.036 |
| cle_006 | 否，不参与排序 | 31 | −1.61 | [−5.65, +2.02] | +0.040337 | 0.021579 | 76.773 |

主排序池仅 3 个有支持单元：2 个下降点估计、1 个上升。两项下降 CI 均包含或触及 0；仅上升的 CI 排除 0。下文 AP/命中针对点估计标签，不等于识别了统计确证的下降。cle_006 有 gold 估计但训练信号支持不足，不能因其 D 小而视为低风险。P 列是原始有符号投影，下降风险排序使用 −P。

### 6.2 新窗口改变了哪些判断？

下表是同一窗口内的比较，已排除 pooled 排序仅区分 update 身份的可能。平均名次和 AP 使用 ΔM<0；ρ、τ 分别是风险分数与连续 −ΔM 的 Spearman、Kendall。共同支持池和风险方向在目标 gold 前固定；完整 AP/平均名次复核属于事后描述性审计，不升级为预注册检验。

| 风险读出 | 下降平均名次 ↓ | AP ↑ | ρ ↑ | τ ↑ |
|---|---:|---:|---:|---:|
| gated D / ungated D | 2.50 | 0.583 | −0.500 | −0.333 |
| delta_norm / centered interaction norm | 2.50 | 0.583 | −0.500 | −0.333 |
| −P | 1.50 | 1.000 | +1.000 | +1.000 |
| KL / JS / u_control_norm | 1.50 | 1.000 | +1.000 | +1.000 |

- **有符号投影提供了 norm 没有的方向排序信息。** −P 按 cle_004 → gen_002 → cle_003 排序，与下降点估计顺序一致；旧 pooled 中 −P 较弱，不代表它在窗口内始终没有信息。这是相对同源幅度基线的正面案例，不代表 P 的正负号逐项预测正确。
- **旧 D 的优势未在新窗口延续，负部截取/门控不能默认有益。** 两版 D 与 norm 排序相同，均把上升最多的 cle_003 排第一；它们与 |ΔM| 的 Spearman 均为 +1，却与 −ΔM 为 −0.5。在这个窗口，D 更符合变化幅度而非下降方向；取消门控也未改变这一点。
- **仍未证明相对所有强基线的独特增量。** KL、JS、u_control_norm 与 −P 得到相同排序。C_upd 相比 u_original_norm 的 AP 为 0.833 对 0.583，但两者下降相关均为 −0.5（同源比较见第 5.4 节），说明“下降靠前”与“连续下降单调”不能互相替代。
- **预算结果与上述排序一致。** 新窗口 D 的 Top-1/Top-2 分别覆盖 0%/68.1% 的下降量；−P 和 KL 等为 68.1%/100%。采用超过 5 pp 标签时，唯一事件仍是 cle_004：D 排第 2，−P/KL 排第 1。它仍是不确定下降，不能因超过 5 pp 的点估计而改称确证事件。

新窗口只提供 3 个候选的描述性检验，ρ/AP=1 不等于统计显著或跨 seed 泛化。主结论也限定于 O−PLACEBO：例如 cle_004 在 NULL 对照下仅变化 −0.81 pp，CI [−7.66,+6.05]，不宣称所有效用变化均与对照无关。

## 7. 跨窗口综合：保留整体排序、窗口内定位与门控差异

### 7.1 旧总体结果与新窗口的对应关系

旧全部 19 个支持单元中有 4 个下降点估计。下表保留第 5 节未完整展开的平均名次、Kendall 和无门控版本；旧后段 7 单元是其子集，不重复作为独立证据。各指标按固定风险方向排序，没有拟合预测器。

| 旧 19 单元的风险读出 | 下降平均名次 ↓ | AP ↑ | ρ(score,−ΔM) ↑ | Kendall ↑ |
|---|---:|---:|---:|---:|
| gated D | 4.00 | 0.799 | +0.417 | +0.307 |
| ungated D | 3.25 | 0.854 | +0.284 | +0.164 |
| centered interaction norm | 4.50 | 0.667 | +0.062 | +0.036 |
| −P | 12.75 | 0.190 | −0.049 | −0.036 |
| KL | 7.50 | 0.551 | +0.207 | +0.150 |

旧 pooled 中 ungated D 的下降前置能力更好，gated D 的连续下降相关更好；新窗口则是 −P 优于 norm，D 不占优。两批结果共同支持 reward-directed 读出的潜在价值，但尚未选出稳定的固定构造。不能把“旧窗口选 D、新窗口选 −P”的事后择优当成已经验证的部署规则；不同候选数/下降比例下的 AP 和平均名次也不直接比较高低。

阈值敏感性仍保留：若只找旧 19 单元中唯一超过 5 pp 的下降，gated D、ungated D、centered norm、KL 分别把它排第 4、6、6、1；因此旧 D 的任意下降排序优势不等于对较大下降也优于 KL。

### 7.2 同一 update 内能否选对 Skill？

以下保留 6 个窗口的完整相关，避免 pooled 结果掩盖窗口内关系。ρ(P,ΔM)=ρ(−P,−ΔM)，各列正值均表示风险顺序与连续下降程度一致。

| 窗口 | 候选数 | ρ(P,ΔM) | ρ(D,−ΔM) | ρ(ungated D,−ΔM) | ρ(centered norm,−ΔM) |
|---|---:|---:|---:|---:|---:|
| U30→31 | 4 | +0.775 | −0.775 | −0.775 | −0.775 |
| U31→32 | 4 | +0.632 | −0.632 | +0.316 | −0.316 |
| U32→33 | 4 | +0.258 | +0.775 | +0.775 | +0.775 |
| U33→34 | 3 | +0.500 | −0.500 | −0.500 | −0.500 |
| U34→35 | 4 | −0.738 | +0.738 | +0.738 | +0.738 |
| U35→40 | 3 | +1.000 | −0.500 | −0.500 | −0.500 |

P 在 5/6 个窗口呈正秩相关，提示应区分跨 update 的分数尺度与窗口内相对排序；但每窗仅 3–4 个 Skill，包含零标签、相邻共享端点和相同 Skill 的重复测量，不能称为 5 次独立复现。旧后段下降都来自 U35、上升都来自 U34，因此 pooled 优势仍需结合本表判断。

下降定位的补充事实：

- U31→32：ungated D 将唯一下降 Skill 排第 1，centered norm 排第 2，gated D 排第 3；KL 也排第 1。这是旧窗口内无门控版本优于 norm 的案例。
- U34→35：C_upd 的下降相关 +0.949，高于同源 u_original_norm 的 +0.738；D 与 centered norm 均为 +0.738。D 把轻微下降的 cle_003 排在较大下降的 cle_004 前，Top-1 只覆盖 10.4% 的下降量，但 Top-2 找到两个下降对象。
- 旧三个有下降窗口中，−P 的 Top-1 均未命中下降；新窗口则命中。因此正相关不自动等于小预算编辑有效，也不能仅用 Top-1 否定整体排序信息。U30→31、U33→34 无下降，下降平均名次、AP、召回及下降量覆盖记为未定义，不计作“成功”。

### 7.3 D 的门控与平均方式

当前计算顺序是先在同一位置上计算向量 d、δ 与标量 C、P，再按 Skill/context 聚合，不跨状态先累加向量：

    gated D   = mean_j[valid_j × I(C_j ≥ 0) × I(||δ_j|| ≥ τδ) × max(−P_j, 0)]
    ungated D = mean_j[I(||d_j|| > ε) × max(−P_j, 0)]
    −平均 P   = −mean_j(P_j)

无门控 D 仍保留负部截取，因此不等于 −平均 P；前者不让正向投影抵消负向贡献。主聚合是全部实际 response loss tokens 等权，零 advantage 和未通过门控的 token 贡献为零但仍计入分母；不是只对负向或通过门控的位置求平均。

实际 τ_C=0、τδ=max(10⁻⁸,10×重复前向噪声 p95)=10⁻⁸，ε=10⁻¹²；两批噪声 p95 均为 0。fidelity 有效还要求 ||u_original||>ε。支持门槛另为至少 20 个非零 advantage decisions，覆盖 4 个 games、8 条 trajectories；未支持则 abstain。没有 Skill 级“D>d₀”阈值，直接对连续分数排序。token 门控、Skill 支持、gold 的 0/5 pp 标签和 Top-k 预算是四个不同口径，不能混淆。

对应实现：[局部信号](/home/wangyifan/skill-RL/SkillRL/phase2/direction.py:7)、[聚合与支持](/home/wangyifan/skill-RL/SkillRL/phase2/aggregate.py:66)、[噪声校准](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/signals/calibration.json)。

## 8. 综合判断与下一步

> Phase2 初步发现，reward-directed policy-update 读出在部分总体及窗口内比较中，比对应幅度读出更能排序 Skill 效用变化方向。旧 C/D 与新 −P 提供不同形式的正面案例；但具体构造的优势随窗口变化，尚未建立固定指标相对强基线的稳定跨窗口/seed 增量。

这一定义不要求精确回归 ΔM、不要求每次 Top-1 都正确，也不要求各 Skill/seed 同向变化。当前全部来自一条 seed303 路径；19 个旧单元包含后段 7 个，新窗口另有 3 个，不能重复计数或当成 22 次独立实验。同源幅度对照支持预测价值的探索，但未隔离 reward 内容、归一化、负部和门控各自的因果贡献。

下一步仅列计划，不在本次修订中执行：

- 增加独立更新窗口/seed 与自然有支持的 Skill；事先固定读出、聚合、风险方向和标签，检验总体排序、窗口内排序及预算收益，不根据 gold 临时切换赢家。
- 在同一 u/δ 和支持集合上比较真实 reward direction 与预定打乱/随机方向，进一步检验 reward 信息的增量，而非只比较不同公式。
- Phase3 用相同编辑器和 Top-k 预算评估实际 Skill 更新收益，并与 norm、KL、old-margin、随机及预算匹配的新 rollout/probe 比较。

新窗口信号在 U40 gold 续跑结果前锁定，符合 without post-update gold rollout 的读出边界；仍使用已有训练 rollout、新旧模型前向及旧效用 evidence。gold 是离线验证成本，不是在线输入。尚未验证节省多少 rollout、总算力或编辑后的实际收益。

## 9. 详细数据与复核入口

本次仅压缩报告展示，不新增统计检验、不重跑训练/评估、不修改原始数据。原章节的完整双轴分解、NULL、各预算、轨迹与 phase 表可查 [带日期的历史完整报告](/home/wangyifan/skill-RL/2026-09-14-phase2-complete-analysis.md:211)；该报告用于展开数据和溯源，当前综合结论以本文为准。

- 新窗口：[全部原始读出与分层效用](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/window_metrics/raw_features_and_semantic_utility.csv)、[锁定分数与 gold](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/window_metrics/locked_scores_and_gold.csv)、[完整 Top-k/阈值/幅度分析](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/window_metrics/ranking_metrics.csv)、[效用分解与 CI](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/window_metrics/utility_units.parquet)、[逐轨迹存档](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/evaluations/u0040/trajectories)。
- 旧窗口：[全部 19 单元](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/reports/2026-09-12-observation-audit-v1/all_supported.csv)、[历史后段 7 单元](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/reports/2026-09-12-observation-audit-v1/heldout.csv)、[多指标比较](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/reports/2026-09-12-observation-audit-v1/comparisons.csv)。
- 完整排序审计：[跨 update/新窗口统计，42 行](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-ranking-interpretation-v1/overall_ranking_audit.csv)、[逐窗口统计，84 行](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-ranking-interpretation-v1/within_window_ranking_audit.csv)、[同源总体对照](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-reward-directed-family-v3/same_basis_overall.csv)、[同源窗口内对照](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-reward-directed-family-v3/same_basis_within_window.csv)。
- 科学协议 SHA-256：0ca78a05b4d58584be39ecfaa9335458b41dfd50ea6e34f80e43be797acbe47a；[端点完成与轨迹校验](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/evaluation_completion.json)。
- 历史修订记录：[排序审计 v1](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-ranking-interpretation-v1/revision.json)、[综合补全 v2](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-phase2-synthesis-v2/revision.json)、[信号族分析 v3](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-reward-directed-family-v3/revision.json)。其中报告哈希对应各自历史版本，不代表本次压缩后的文件；历史记录不覆盖。

## 10. 2026-09-22 追加：SkillNet-37 / seed404 的 reward-directed 变式探索

本节是新 cohort 的追加分析，不改写第 1–9 节的旧 seed303 / U35→U40 证据，也不把两个实验池合并。
范围为已完成的 seed404 U0→U5、冻结 SkillNet-37、所有自然首调用效用；在已见历史结果后，
用户授权广泛比较 reward 相关读出，不要求它们依赖 P。**科学状态：ANALYZED，事后探索。**

共登记并计算 39 个 reward 公式 × token/decision/game 三种聚合（117 个分数），
以及 117 个去奖励版本、21 个幅度分数和 6 个原有参考，共 261 列。
涵盖去截断、advantage 强度、未归一化内积、中心化交互余弦、软/分位数门控、稳健累加、
采样动作 logprob/概率比、reward-only 与 C 参考。原始/中心化 C 本身属于 reward 信号族，不是无奖励基线。
本轮只复用已存标量与标签，未重跑 RL、模型前向或 ALFWorld；没有在 505/606 上试这些变式。

### 10.1 同池结果与关键反例

PLACEBO、all phase、0pp：18 技能，5 个下降、7 个上升、6 个不变点估计。

| 读出 | 聚合 | AP | 下降/其余 AUROC | Spearman |
|---|---|---:|---:|---:|
| 原 D | token | 0.509650 | 0.630769 | -0.147161 |
| 去截断 signed D，保留门控 | token | 0.415995 | 0.415385 | -0.066222 |
| 不带门控的 -P | token | 0.423810 | 0.461538 | -0.064120 |
| 中心化 C | token | 0.688095 | 0.723077 | 0.023125 |
| 中心化 C（本次 reward 最高 AP） | decision | 0.693590 | 0.738462 | 0.042046 |
| 负投影比例 | game | 0.647222 | 0.738462 | 0.120882 |
| 中心化交互幅度，无 reward | decision | 0.642222 | 0.800000 | 0.406795 |
| 中心化交互幅度，无 reward | game | 0.724762 | 0.800000 | 0.362647 |
| 原 activation-l8 范数参考 | 原口径 | 0.743333 | 0.876923 | 0.310089 |

直接移除截断未改善 token 等权排序；最好的 reward AP 仍来自中心化 C，且仅比其原 token 版本高约 0.0055。
game 等权也明显改善无奖励幅度，因此不能把聚合变化的收益都归因于 reward。
中心化 C/decision 在 Top-2 命中 2/2、覆盖约 71.16% 下降量，但幅度/game 也同样做到；
Top-5 则 activation-l8 命中 4/5，高于该 C 的 3/5。5pp 时仅两例下降，C/decision 与幅度/game 的 AP 均为 1。

### 10.2 奖励增量、方向与不确定性

中心化 C/decision 相对同聚合中心化交互幅度的 AP 差为 +0.051368，
固定读出下配对 game/continuation bootstrap 的 95% 区间为 [-0.187348, 0.166667]。
它与固定原非零方向支持上的单位 advantage 版本相比 AP 增加 +0.482861，区间仍包含 0。
前一个对照的基础向量不完全相同（C 用 u_O，交互幅度用 delta）；后一个对照不能恢复 A=0 行的无奖励几何，
这两个边界均明确保留。512 次轨迹块符号随机化约 8.59% 达到该 C 的实际 AP；允许每次从全部 reward 候选选最高值时，
约 77.93% 达到该水平。这些是负对照比例，不是确认性 p 值。

方向判断不能用 AP 替代：最佳若干带符号变式在 12 个非零变化技能上正确 9 个（75%，平衡准确率 72.86%），
恒预测上升则为 7/12（58.33%）。它们是同 seed 多候选探索后的点估计，并非独立复现。
中心化 C 的高 AP 也不等于连续方向整体更准：其 Spearman 仅 0.042；object-placer 排第 3，但实际 ΔM 点估计为 +15.22pp。
5 个下降标签的原区间均触及或跨 0，不是无噪声真值。

**当前结论：保留了若干 reward 候选及局部优势，但尚不能确认稳健的 reward 增量；不能据本次择优结果写成 idea 已证明。**
后续应先冻结有限候选再做独立窗口/seed 检查，不逐 seed 换赢家。本轮未改 Phase3 主指标。

### 10.3 完整材料与验收

[解释性分析、公式类别和失败案例](2026-09-22-seed404-reward-readout-variants-analysis.md)；
[保留旧数值报告正文的完整扩展版](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v1/reports/phase2-results-expanded.md)；
[候选全集](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v1/registry.csv)；
[所有分层指标](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v1/ranking_diagnostics.csv)；
[逐技能分数与效用](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v1/skill_scores_and_gold.csv)。

283 项 CPU 测试通过，实际分析约 42 秒、一次完成；5,220 行统计以 sklearn/scipy 独立复核，9 个原有指标与旧报告一致，
最大误差约 3.33e-16。2,000 次 bootstrap 各有 1,674 次完整池 draw；缺稀有技能时整池记缺失，未临时缩池。
统计误用检查覆盖 11/11，特别标记了多公式择优、事件基率和单 seed 泛化边界。
旧数值报告、原 19 个输入和 125 个冻结源码逐哈希未改。
本文第 1–9 节逐字节保留，追加前的[全文备份](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v1/publication-original-phase2-complete-analysis.md)
SHA-256 为 `b13d3960b808af3c96b20bc5618c4e764da283146f75350e04704dc711ca88a4`。
