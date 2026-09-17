# Phase2 完整实验分析：Policy update 是否包含 Skill 效用变化的预测信息

生成时间（UTC）：2026-09-14T00:44:24.306368+00:00。本报告整合已完成的旧单步 pilot 与新 U35→U40 窗口；新批次全部轨迹及预测时间隔离已通过核验。

2026-09-14 首次讨论修订：更新第 1、9 节结论，并追加第 11 节的整体排序、窗口内比较及 D 门控说明；该历史版本及统计归档记录于[首次修订清单](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-ranking-interpretation-v1/revision.json)。

2026-09-14 综合补全：在第 5.2–5.4 节补齐旧测试四口径对照、NULL/中途锚点稳健性证据及 C_upd 等竞争指标的跨批次解读，并同步修订结论，使本文件作为已完成 Phase2 的统一分析入口。仅使用已归档数据，未新增训练/rollout、修改 gold、重选 checkpoint 或改动预锁定排序；补充比较为事后描述性分析。原始完成证明与首次修订清单保持不变，当前版本见[综合补全修订清单](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-phase2-synthesis-v2/revision.json)。

## 1. 结论摘要

研究目标是以实际 policy update 的读出对优先 audit/edit 的 Skill 排序；不要求准确拟合效用数值，也不以正→负 flip 作为必要条件。增量价值不应只用 Top-1 判定：下降单元平均名次、AP、与连续 −ΔM 的 Spearman/Kendall，以及多个预算下的 Top-k 命中和下降量覆盖，分别描述整体前置能力、排序单调性与预算效益。D 是非负下降风险分数，P 是有符号投影，幅度指标回答的是变化程度。

效用变化现象得到进一步支持：旧 U34→U35 的 cle_004 下降 −14.29 pp，95% CI [−28.57,−1.79]；新 U35→U40 的 cle_003 上升 +15.73 pp，95% CI [+2.42,+29.03]。新窗口四个 Skill 的点估计为 3 个下降、1 个上升；下降 ΔM 的配对 95% CI 均包含或触及 0。方向无需跨 Skill 一致，但点估计下降不等于统计确证的下降。

旧 cle_004 的下降不只依赖一种对照或初始调用：O−NULL 同次变化为 −14.29 pp [−28.57,−3.57]，early anchors（step 1–4）上的语义变化为 −15.38 pp [−30.77,−1.92]；同时 ORIGINAL 成功率仍提高 +5.36 pp，PLACEBO 提高更多（+19.64 pp）。这支持“中途调用的相对效用可在 policy 行为改善时下降”，但这些共享 anchors 的检验不是独立复现，也不证明 D 已能预测该下降。

旧数据中存在 D 的整体排序优势，不能因 Top-1 未选中下降最多的 Skill 而否定：19 个支持单元中，gated D 的下降平均名次为 4.00、AP=0.799、Spearman(D,−ΔM)=+0.417，centered norm 分别为 4.50、0.667、+0.062。旧 7 个时间外单元中，D 将两个点估计下降单元排在第 1、2 名，AP=1.000，优于 centered norm 的第 1、3 名及 AP=0.833。这些是小样本、跨 update 合并的描述性优势，不是显著性或跨 seed 泛化证明；7 个测试单元包含在 19 个单元中。

这一优势有明确比较边界：旧测试在 2 降/2 升中计算的条件 AUROC 为 D=1.000、centered norm=0.750、KL=0.500；但对唯一超过 5 pp 的下降事件，AP 为 D=0.500、norm=0.333、KL=1.000。C_upd/C_upd_centered 的旧测试任意下降 AP 也均为 1.000，后者的下降 Spearman=+0.667 高于 D 的 +0.556。因此“D 在部分整体排序口径优于 norm/KL”有证据，“D 优于全部候选读出”或“Skill-specific reward opposition 提供了独特增量”尚无证据。
门控与无门控应分别评价：旧 19 单元中，ungated D 的下降平均名次进一步改善至 3.25、AP=0.854，但连续下降排序相关 +0.284 低于 gated D 的 +0.417。在 U31→U32 窗口内，ungated D 将唯一下降 Skill 排第 1，centered norm 排第 2，gated D 排第 3；因此不能概括为“所有 D 版本在所有排序口径下均无优势”，也不能说门控已被证明有益。

这些优势尚未在新窗口复现。新主共同支持池只有 gen_002、cle_003、cle_004，cle_006 因训练方向支持不足不进入统一比较。gated/ungated D 与 centered norm 的下降平均名次均为 2.50、AP=0.583、Spearman=−0.500；−P、KL、JS 和 control-update norm 均为 1.50、1.000、+1.000。按 −P 的 Top-1/Top-2 排序可覆盖 68.1%/100% 的点估计下降量，但这些强基线也达到相同结果，因此 reward projection 的独特增量尚未建立。第 7 节保留原有全部 Top-k 结果。

P 的窗口内线索不能被 pooled 相关概括：五个旧窗口中四个、加上新窗口，共六个窗口中五个的 Spearman(P,ΔM) 为正；但每窗只有 3–4 个 Skill，多个标签为零，旧有下降的三个窗口中 −P 的 Top-1 均未命中下降 Skill。因此“多数窗口有相对排序线索”不等于“已能稳定选对优先更新对象”，更不是六次独立成功复现。

主 D 使用预定 token 级门控 C_upd≥0、interaction norm≥10⁻⁸，再聚合负向投影；ungated D 已并行归档。没有设置 Skill 级“D 大于某阈值才算风险”的截断，排序使用连续分数。该门控、效用标签阈值（0/5 pp）和训练支持门槛是三类不同设置，详见第 11.5 节。

综合结论：

> Phase2 已建立真实更新信号与独立语义效用标签的测量链路；旧中途锚点和两种对照支持效用下降现象，D 在旧样本的部分整体下降排序口径优于 norm/KL，P 在多数窗口呈相对排序线索。但 C_upd 等候选读出可匹配部分优势，新窗口亦未复现 D 的优势，尚未建立独特、跨窗口/seed 可重复的 reward-directed 增量。当前支持“有正面证据的预测可行性线索”，而不是“完全无优势”或“稳定预测已成立”。

不要求指标在每一个窗口都获胜，也不要求不同 Skill/seed 的效用方向一致；需要在预先固定的规则下，以更多独立窗口/seed 检验总体排序收益及不确定性。旧等容量回归的 MAE 未改善作为辅助结果保留，不再作为直接排序命题的否决标准。不能事后按窗口择优使用 gated D、ungated D 或 −P，再把该 oracle 选择当作已验证预测器。

## 2. 两个批次的证据范围与可比性

| 项目 | 旧单步 pilot | 新扩大窗口 |
|---|---|---|
| 同一 Seed303 延续路径 | U30→U35，5 个相邻窗口 | U35→U40，1 个五步窗口 |
| 评估端点 | U30–U35，共 6 个 | U35/U40，共 2 个 |
| anchors | 旧 B0 的 200 个首次调用 anchors | U35 自然 rollout 新采集的 200 个 anchors |
| 每 anchor–arm–端点 | 1 evidence + 1 gold | 2 evidence + 4 gold |
| 三臂 suffix 总数 | 7,200 | 7,200（U35 导入 3,600；U40 新增 3,600） |
| 主要比较 | 时间外 readout/等容量 probe | 预锁定同窗口直接 top-k 排序 |

两个批次合计 14,400 条独立归档的实验 suffix，但不是 14,400 个独立预测单元。新旧 U35 的支持 state 和 continuation seeds 不同，不能拼接成一条未经控制的效用时间曲线，也不能把差异全部归因于窗口变大。旧 pilot 原计划的 U36 未执行；新批次 U36–U39 仅归档单步信号，没有新设中间 gold。

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

Bank 冻结为 12 general + 32 task-specific；clean 候选为 12+6。每步按 state 路由一个 Skill，不是一次把整个 bundle 喂给 policy。本轮评估固定 4 个自然路由 Skill，不宣称全 18/44 条 Skill 已评估。新 anchor 覆盖：

| skill   |   anchors |   games |   first_step_min |   first_step_max |   initial |   early |   middle |   late |
|:--------|----------:|--------:|-----------------:|-----------------:|----------:|--------:|---------:|-------:|
| gen_002 |        50 |      29 |                2 |               29 |         0 |      26 |       23 |      1 |
| cle_003 |        50 |      31 |                3 |               11 |         0 |      20 |       30 |      0 |
| cle_004 |        50 |      31 |                1 |                1 |         0 |      50 |        0 |      0 |
| cle_006 |        50 |      31 |                0 |                0 |        50 |       0 |        0 |      0 |

initial=0、early=1–4、middle=5–14、late≥15。信号支持另要求至少 20 个非零 advantage decisions、4 个非零支持训练 games、8 条非零支持轨迹。gold anchors 多不代表训练方向支持充分；unsupported 不编码为零风险。

## 4. 干预、效用和指标的精确定义

在目标 Skill 第一次调用前精确重放同一 prefix，随后 ORIGINAL/模板与 token 长度匹配的 PLACEBO/目标 payload 为空的 NULL 自由续跑。后续每次路由到目标 Skill 都继续相同干预；其他 Skill、候选 ID/描述和 Router 不变。NULL 不是删除该条目重新检索，也不是禁用整个 Skill Bank。

主效用 M_sem=E[success_O−success_P]，ΔM_sem=M_new−M_old；O−NULL 为次要对照。每个 game 内平均 anchors/repeats，再对 games 等权；95% CI 使用 10,000 次配对 game/continuation bootstrap。效用表单位 pp；环境终局成功奖励为 10，return 差为 success 差乘 10。

设 u_O=logπ_new(·|z,s)−logπ_old(·|z,s)，u_ctl 为相同 teacher-forced response 前缀下的对照变化，δ_int=u_O−u_ctl；d=A(e_a−π_old)，C_upd=cos(d,u_O)，P_int=〈d,δ_int〉/(||d||+ε)，D=E[有效方向及 C_upd≥0、||δ||≥τδ 时的 max(0,−P)]。词表全量 248,320，W=I，ε=1e−12，τδ=1e−8；零 advantage/未通过 gate 的 token 仍在 D 聚合分母。mean(P)>0 与 D>0 可同时成立。

新窗口将 U36 旧 batch 的实际 state/token/advantage 固定，在 U35 和 U40 端点重放。它是累计 interaction 对起始 reward direction 的投影，不是五个 D 相加，也不假定中间 reward direction 不变。d 是 outcome-consistent 局部方向，不是包含 KL/Adam/clipping 的完整参数梯度。

P>0 不数学保证长期 ΔM>0；D≥0，无符号，小 D 不证明效用上升。centered/raw norm、KL/JS 和 activation norm 是幅度读出。全局参数范数同一窗口内对所有 Skill 相同，不能据此排序。JVP 和线性化误差未实现，不能用参数范数冒充。

## 5. 旧单步结果：保留但不混入新测试

### 5.1 总体关联与辅助回归

| scope           | signal              | risk_orientation   |   units |   updates |   rho_raw_magnitude |   rho_risk_decline |   ap_any_decline |
|:----------------|:--------------------|:-------------------|--------:|----------:|--------------------:|-------------------:|-----------------:|
| all_exploratory | P_int               | -value             |      19 |         5 |          0.350334   |        -0.0488495  |         0.189552 |
| all_exploratory | D_contribution      | +value             |      19 |         5 |          0.301405   |         0.417174   |         0.798611 |
| all_exploratory | delta_norm          | +value             |      19 |         5 |          0.522566   |        -0.00390796 |         0.558333 |
| all_exploratory | delta_centered_norm | +value             |      19 |         5 |          0.66544    |         0.0615503  |         0.666667 |
| all_exploratory | old_margin          | +value             |      19 |         5 |          0.797527   |        -0.0424515  |         0.583333 |
| all_exploratory | C_upd_centered      | +value             |      19 |         5 |         -0.00195718 |         0.319476   |         0.360417 |
| heldout         | P_int               | -value             |       7 |         2 |          0.630062   |        -0.407687   |         0.22619  |
| heldout         | D_contribution      | +value             |       7 |         2 |          0.407687   |         0.555937   |         1        |
| heldout         | delta_norm          | +value             |       7 |         2 |          0.481812   |         0.2965     |         0.833333 |
| heldout         | delta_centered_norm | +value             |       7 |         2 |          0.630062   |         0.185312   |         0.833333 |
| heldout         | old_margin          | +value             |       7 |         2 |          0.640807   |        -0.11651    |         0.5      |
| heldout         | C_upd_centered      | +value             |       7 |         2 |          0.259437   |         0.667124   |         1        |

旧时间外非零变化按 update 分布如下。若不同 update 自身的标签方向不同，pooled 相关可能来自区分 update，不能据此证明同一 update 内能选对 Skill：

|   global_update |   positive |   negative |
|----------------:|-----------:|-----------:|
|              34 |          2 |          0 |
|              35 |          0 |          2 |

旧回归结果仅作为辅助，不再以预测具体 ΔM 的 MAE 作为当前主要成功标准：

| model      |   test_units |   test_updates |   signed_MAE |   conditional_sign_accuracy |
|:-----------|-------------:|---------------:|-------------:|----------------------------:|
| zero       |            7 |              2 |      3.77551 |                       nan   |
| dev_mean   |            7 |              2 |      4.03061 |                         0.5 |
| old_margin |            7 |              2 |      4.07609 |                         0.5 |
| unsigned   |            7 |              2 |      6.20656 |                         0.5 |
| activation |            7 |              2 |      7.09238 |                         0.5 |
| signed     |            7 |              2 |      7.87121 |                         0.5 |
| opposition |            7 |              2 |      7.85252 |                         0.5 |

上表 signed_MAE 已转为 pp。旧 `cle_004` U34→U35 的 ΔM=−14.29 pp、95% CI [−28.57,−1.79]，属于明确的变化样本；点估计由正到负不等价于独立 calibration 确认的 harmful sign flip。19 个支持单元及 7 个测试单元的全部原始值仍在旧 observation-audit-v1 的 CSV。

### 5.2 旧时间外测试的四口径比较

以下均来自旧 U34/U35 两个时间外窗口，共 7 个有支持单元，是旧 19 单元的子集。D 指主 gated D；三个分数均按原值越高风险越大排序。信号曾在目标 gold 前锁定，但本比较口径为事后审计，不是新增的确认性测试。

| 测试口径 | D | Centered interaction norm | KL |
|---|---:|---:|---:|
| 与连续下降 −ΔM 的 Spearman；全部 7 单元 | 0.556 | 0.185 | 0.037 |
| 任意下降 AP；2 个下降对其余 5 个单元 | 1.000 | 0.833 | 0.700 |
| 条件方向 AUROC；仅 2 个下降对 2 个上升 | 1.000 | 0.750 | 0.500 |
| 下降超过 5 pp 的 AP；1 个事件对其余 6 个单元 | 0.500 | 0.333 | 1.000 |

表的前三项支持 D 在该样本中对下降对象的整体前置和方向排序优于这两个基线；第四项则支持 KL 更早定位唯一下降较大的 cle_004@U35。D 的两个下降单元排第 1、2 名，但先排的是下降 −1.67 pp 的 cle_003，随后才是 −14.29 pp 的 cle_004，故“整体下降对象靠前”与“下降最严重者优先”需要分别评价。

这里的条件 AUROC 明确排除了 3 个点估计不变单元，只有 2×2=4 对下降/上升比较。它不是“2 个下降对其余 5 个单元”的 AUROC；后一口径对应 D=1.000、centered norm=0.900、KL=0.700。第 11 节审计 CSV 中的 AUROC 使用后者，不能与本表 0.750/0.500 混用。AP 使用完整 7 单元，并未排除不变单元。

两个下降均来自 U35、两个上升均来自 U34；因此条件 AUROC=1.000 可能部分反映 update 间分离，不能直接证明同一次 update 内的独立方向区分。这个限制不抹去表中的描述性优势，但限制其泛化解释。以上标签均由效用变化点估计定义，不等价于所有方向已经置信区间确证。

### 5.3 旧 cle_004 案例：两种对照与非初始锚点

U34→U35 的固定首次调用 anchors 上，ORIGINAL 成功率提高，但对照提高更多：

| 量 | U34 | U35 | 变化 |
|---|---:|---:|---:|
| ORIGINAL 成功率 | 21.43% | 26.79% | +5.36 pp |
| PLACEBO 成功率 | 14.29% | 33.93% | +19.64 pp |
| 语义效用 ORIGINAL−PLACEBO | +7.14 pp | −7.14 pp | −14.29 pp |

该分解表明下降并非 ORIGINAL 任务能力绝对退步，而是 Skill 相对增益被对照的更大改善抵消。它与第 6 节新窗口中“ORIGINAL 改善、某些 Skill 的 ΔM 却下降”的现象相呼应，但两批 anchors 不同，不能拼成未控制的连续曲线；这些成功率也不是从 reset 开始的独立全局 validation。

| 对照 / 锚点范围 | anchors | games | ΔM，pp | 配对 95% CI，pp |
|---|---:|---:|---:|---|
| PLACEBO / all | 50 | 28 | −14.29 | [−28.57, −1.79] |
| NULL / all | 50 | 28 | −14.29 | [−28.57, −3.57] |
| PLACEBO / early，step 1–4 | 46 | 26 | −15.38 | [−30.77, −1.92] |
| NULL / early，step 1–4 | 46 | 26 | −15.38 | [−30.77, −3.85] |
| PLACEBO / middle，step 5–14 | 4 | 2 | 0.00 | [0.00, 0.00] |

该案例的价值有三点：

- 下降在 O−NULL 下仍成立，因此不是只有 token-matched PLACEBO 才出现的结果。主分析仍为 O−PLACEBO，不能切换对照挑选有利结论。
- 支持明确下降的 early 子集从非初始 state 干预，体现了 prefix replay 后的自然中途调用效用变化；不是只改变开场 prompt。middle 仅 4 anchors/2 games，不能凭其零值宣称中后期普遍不变，也不能宣称 late 阶段已全面验证。
- 50 个 anchors 中有 8 个下降、1 个上升、41 个不变；平均效用下降不要求每个 state 同向。此计数不取代 game 等权的主效用估计。

这些对照和 phase 子集共享轨迹/anchors，区间为单项、未作多重比较校正，不能视为四次独立复现。旧分类字段要求整个 CI 位于 −5 pp 以下才标“可靠至少下降 5 pp”；本例虽然排除了 0，但上界未低于 −5 pp，因此原始 direction 仍为 uncertain，不能混淆这两个标准。退化 [0,0] 也只描述当前采样结果。

对照稳健性不是所有 Skill/窗口都具备：旧 cle_003 同次 Δ(O−PLACEBO)=−1.67 pp，而 Δ(O−NULL)=+1.67 pp，均不确定；新 cle_004 的语义下降为 −6.45 pp，但 NULL 下只有 −0.81 pp，CI [−7.66,+6.05]。新 cle_003 的两种对照均为正向点估计（语义 +15.73 pp，NULL +12.50 pp），但只有语义区间排除 0。这些结果共同说明，应区分“某个效用变化案例具有对照稳健性”和“所有下降/上升都不依赖对照”。

原始支持：[旧效用单元与分解](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/metrics/utility_units.parquet)、[旧逐 anchor 效用](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/metrics/anchor_margins.parquet)。这些证据支持效用变化现象，不自动构成 D 的预测准确性或因果机制证据。

### 5.4 C_upd 等竞争读出：D 的优势属于哪个层次？

C_upd 是 reward direction d 与实际 Skill-conditioned 更新 u_original 的 cosine；C_upd_centered 是 d 与去除词表均值后的 u_original 的 cosine。两者都不包含 D/P 所使用的 ORIGINAL−control 更新交互差 δ_int；C_upd_centered 也不是 centered interaction norm。

它们可用于检验“reward 对齐的更新读出是否有信息”，但不是 reward-free 基线，与 D 也并非独立构造。因此 C 表现好既不否定广义 policy-update readout 命题，也不能被当作 D 的独立增量证据。下表沿用旧审计的 +C 风险排序方向，仅作统一方向的描述性比较；不赋予“C 越大必然更坏”的机制含义。新窗口的 C 排序重算并非原 Top-k 主比较，不重新标记为预注册结果。

| 范围 | 风险读出 | ρ(score,−ΔM) | 任意下降 AP | 条件下降/上升 AUROC | 超过 5 pp 下降 AP |
|---|---|---:|---:|---:|---:|
| 旧 19 单元，pooled | D | +0.417 | 0.799 | 0.938 | 0.250 |
| 同上 | C_upd | +0.336 | 0.377 | 0.812 | 0.333 |
| 同上 | C_upd_centered | +0.319 | 0.360 | 0.812 | 0.200 |
| 旧测试 7 单元，pooled | D | +0.556 | 1.000 | 1.000 | 0.500 |
| 同上 | C_upd | +0.519 | 1.000 | 1.000 | 1.000 |
| 同上 | C_upd_centered | +0.667 | 1.000 | 1.000 | 0.500 |
| 新窗口 3 单元 | D | −0.500 | 0.583 | 0.000 | 0.500 |
| 同上 | C_upd | −0.500 | 0.833 | 0.500 | 0.333 |
| 同上 | C_upd_centered | −0.500 | 0.833 | 0.500 | 0.333 |

只在同一范围内比较方法：旧全样本的 D 在任意下降 AP/相关上更强；旧测试的 C_upd/C_upd_centered 已匹配 D 的 AP 和条件 AUROC，其中 C_upd 将唯一下降超过 5 pp 的事件排第 1，C_upd_centered 的连续下降相关更高。新窗口中 C 虽在“任意下降前置”的 AP 上优于 D，但单调性仍为负，且较大下降事件排得更后，不能据此宣布 C 是稳定赢家。不同事件比例、候选数和窗口协议下，不直接比较 AP 的跨批次绝对大小。

其他已测对照也必须保留在共同解释中：

- KL/JS 在旧测试的较大下降事件上优于 D，在新窗口又与 −P 排序一致。因此 D 对旧“任意下降”的优势不能泛化成优于所有无符号分布变化指标。
- 旧训练成功率与下降的 pooled Spearman 达 +0.749，但同一 update 内对所有 Skill 相同，不能用于该 update 内的 Skill 排序；全局参数范数也有同样限制。该结果提醒需要控制 update-level 差异，而不是证明训练成功率可以替代 Skill 读出。
- 旧 centered interaction norm 和 activation norms 与变化幅度有线索，但 old-margin 的幅度关联也很强：全样本 ρ(old-margin,|ΔM|)=+0.798，高于 centered norm 的 +0.665。这不等于它能预测下降方向，也说明泛化预测需与便宜的旧效用基线比较。旧事后比较使用 +old-margin，新预锁定风险排序使用低 margin 优先，跨表比较时不能忽略方向定义差异。
- 实际 FP32 参数差分、live/offline logits 对齐、token/decision 等权 D 等结果证明测量链路可审计；它们不是 JVP、线性化准确性、节省 rollout 成本或编辑收益的替代证据。

综合两个批次，当前最稳妥的分层判断是：效用变化案例得到支持；部分 policy-update 读出包含描述性排序信息；D 对旧 norm/KL 的部分优势值得保留；D 相对 C 等 reward-aligned 读出的独特增量，以及任一指标在独立窗口/seed 上的稳定优越性，仍未建立。无需要求每个窗口都获胜，但也不能按观察结果为每个窗口挑一个赢家并合并声称成功。

数据来源：[旧全候选比较](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/reports/2026-09-12-observation-audit-v1/comparisons.csv)、[旧逐单元读出](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/reports/2026-09-12-observation-audit-v1/all_supported.csv)、[新窗口原始读出](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/window_metrics/raw_features_and_semantic_utility.csv)。本节三批次 D/C 的重算值及验证记录随[综合补全修订清单](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-phase2-synthesis-v2/revision.json)归档。

## 6. 新窗口的边际效用变化与双轴分解

| skill_id   | supported   |   anchor_count |   game_count |   utility_old |   utility_new |   delta_utility |    ci_low |   ci_high |
|:-----------|:------------|---------------:|-------------:|--------------:|--------------:|----------------:|----------:|----------:|
| gen_002    | True        |             50 |           29 |       3.01724 |        0      |        -3.01724 |  -7.75862 |   0       |
| cle_003    | True        |             50 |           31 |      46.371   |       62.0968 |        15.7258  |   2.41935 |  29.0323  |
| cle_004    | True        |             50 |           31 |       6.45161 |        0      |        -6.45161 | -14.9194  |   1.20968 |
| cle_006    | False       |             50 |           31 |       1.6129  |        0      |        -1.6129  |  -5.64516 |   2.01613 |

方向不需要跨 Skill 一致。CI 跨 0 表示不确定，不等于必须让所有 Skill 效用统一变正或变负。有限生成重复下的退化 [0,0] CI 也不证明真实期望恒定。

| skill_id   |   original_old |   original_new |   control_old |   control_new |   delta_original |   delta_control |   delta_utility |
|:-----------|---------------:|---------------:|--------------:|--------------:|-----------------:|----------------:|----------------:|
| gen_002    |        60.7759 |        76.7241 |      57.7586  |      76.7241  |          15.9483 |        18.9655  |        -3.01724 |
| cle_003    |        53.2258 |        67.7419 |       6.85484 |       5.64516 |          14.5161 |        -1.20968 |        15.7258  |
| cle_004    |        33.871  |        70.9677 |      27.4194  |      70.9677  |          37.0968 |        43.5484  |        -6.45161 |
| cle_006    |        45.9677 |        73.3871 |      44.3548  |      73.3871  |          27.4194 |        29.0323  |        -1.6129  |

ΔM=ΔORIGINAL−ΔPLACEBO。ORIGINAL 改善但 ΔM 下降可表示相对增益缩小；ORIGINAL 和 PLACEBO 都退化则需考虑一般性 policy 变化。这些是固定 prefix 的锚点续跑成功率，不是全局从 reset 开始的 policy validation；PLACEBO/NULL 也不是全库 skill-free baseline。

NULL 次要对照（pp）：

| skill_id   |   utility_old |   utility_new |   delta_utility |   ci_low |   ci_high |
|:-----------|--------------:|--------------:|----------------:|---------:|----------:|
| cle_003    |     47.5806   |     60.0806   |       12.5      | -1.6129  |  26.2097  |
| cle_004    |      0        |     -0.806452 |       -0.806452 | -7.66129 |   6.04839 |
| cle_006    |      0.806452 |      0        |       -0.806452 | -4.03226 |   2.41935 |
| gen_002    |      2.58621  |      3.01724  |        0.431034 | -1.72414 |   3.44828 |

## 7. D 相比幅度读出是否有下降排序优势？

以下风险方向、共同支持池、k=1/2 与 25%/50% 预算均在目标 gold 前锁定；重合预算去重。D、−P、+norm 与旧 margin 低优先、通用对照更新、随机期望比较。并列使用随机选取的期望，不按 gold 破并列。Precision/Recall 标签阈值分别为 0 与 5 pp；下降量始终累计 max(0,−ΔM)，不随标签阈值修改。

本节是新窗口的预锁定预算分析，不能单独代表 D 在所有排序目标上的表现。跨批次的下降平均名次、AP、单调性与 gated/ungated 比较追加于第 11 节；这些事后描述性统计不修改原有预注册结果。

### 7.1 下降事件阈值：0 pp

| score                  |   n_candidates |   events |   k |   precision_at_k |   recall_at_k |   captured_change_mass |   spearman |    kendall |
|:-----------------------|---------------:|---------:|----:|-----------------:|--------------:|-----------------------:|-----------:|-----------:|
| D_contribution         |              3 |        2 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| D_contribution         |              3 |        2 |   2 |         0.5      |      0.5      |               0.681351 |       -0.5 |  -0.333333 |
| D_ungated_contribution |              3 |        2 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| D_ungated_contribution |              3 |        2 |   2 |         0.5      |      0.5      |               0.681351 |       -0.5 |  -0.333333 |
| P_int                  |              3 |        2 |   1 |         1        |      0.5      |               0.681351 |        1   |   1        |
| P_int                  |              3 |        2 |   2 |         1        |      1        |               1        |        1   |   1        |
| activation_l16_norm    |              3 |        2 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| activation_l16_norm    |              3 |        2 |   2 |         0.5      |      0.5      |               0.681351 |       -0.5 |  -0.333333 |
| activation_l24_norm    |              3 |        2 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| activation_l24_norm    |              3 |        2 |   2 |         0.5      |      0.5      |               0.681351 |       -0.5 |  -0.333333 |
| activation_l32_norm    |              3 |        2 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| activation_l32_norm    |              3 |        2 |   2 |         0.5      |      0.5      |               0.681351 |       -0.5 |  -0.333333 |
| activation_l8_norm     |              3 |        2 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| activation_l8_norm     |              3 |        2 |   2 |         0.5      |      0.5      |               0.681351 |       -0.5 |  -0.333333 |
| delta_centered_norm    |              3 |        2 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| delta_centered_norm    |              3 |        2 |   2 |         0.5      |      0.5      |               0.681351 |       -0.5 |  -0.333333 |
| delta_norm             |              3 |        2 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| delta_norm             |              3 |        2 |   2 |         0.5      |      0.5      |               0.681351 |       -0.5 |  -0.333333 |
| forward_kl_original    |              3 |        2 |   1 |         1        |      0.5      |               0.681351 |        1   |   1        |
| forward_kl_original    |              3 |        2 |   2 |         1        |      1        |               1        |        1   |   1        |
| js_original            |              3 |        2 |   1 |         1        |      0.5      |               0.681351 |        1   |   1        |
| js_original            |              3 |        2 |   2 |         1        |      1        |               1        |        1   |   1        |
| old_margin             |              3 |        2 |   1 |         1        |      0.5      |               0.318649 |        0.5 |   0.333333 |
| old_margin             |              3 |        2 |   2 |         1        |      1        |               1        |        0.5 |   0.333333 |
| u_control_norm         |              3 |        2 |   1 |         1        |      0.5      |               0.681351 |        1   |   1        |
| u_control_norm         |              3 |        2 |   2 |         1        |      1        |               1        |        1   |   1        |
| u_original_norm        |              3 |        2 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| u_original_norm        |              3 |        2 |   2 |         0.5      |      0.5      |               0.681351 |       -0.5 |  -0.333333 |
| random_expected        |              3 |        2 |   1 |         0.666667 |      0.333333 |               0.333333 |      nan   | nan        |
| random_expected        |              3 |        2 |   2 |         0.666667 |      0.666667 |               0.666667 |      nan   | nan        |

### 7.2 下降事件阈值：5 pp

| score                  |   n_candidates |   events |   k |   precision_at_k |   recall_at_k |   captured_change_mass |   spearman |    kendall |
|:-----------------------|---------------:|---------:|----:|-----------------:|--------------:|-----------------------:|-----------:|-----------:|
| D_contribution         |              3 |        1 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| D_contribution         |              3 |        1 |   2 |         0.5      |      1        |               0.681351 |       -0.5 |  -0.333333 |
| D_ungated_contribution |              3 |        1 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| D_ungated_contribution |              3 |        1 |   2 |         0.5      |      1        |               0.681351 |       -0.5 |  -0.333333 |
| P_int                  |              3 |        1 |   1 |         1        |      1        |               0.681351 |        1   |   1        |
| P_int                  |              3 |        1 |   2 |         0.5      |      1        |               1        |        1   |   1        |
| activation_l16_norm    |              3 |        1 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| activation_l16_norm    |              3 |        1 |   2 |         0.5      |      1        |               0.681351 |       -0.5 |  -0.333333 |
| activation_l24_norm    |              3 |        1 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| activation_l24_norm    |              3 |        1 |   2 |         0.5      |      1        |               0.681351 |       -0.5 |  -0.333333 |
| activation_l32_norm    |              3 |        1 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| activation_l32_norm    |              3 |        1 |   2 |         0.5      |      1        |               0.681351 |       -0.5 |  -0.333333 |
| activation_l8_norm     |              3 |        1 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| activation_l8_norm     |              3 |        1 |   2 |         0.5      |      1        |               0.681351 |       -0.5 |  -0.333333 |
| delta_centered_norm    |              3 |        1 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| delta_centered_norm    |              3 |        1 |   2 |         0.5      |      1        |               0.681351 |       -0.5 |  -0.333333 |
| delta_norm             |              3 |        1 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| delta_norm             |              3 |        1 |   2 |         0.5      |      1        |               0.681351 |       -0.5 |  -0.333333 |
| forward_kl_original    |              3 |        1 |   1 |         1        |      1        |               0.681351 |        1   |   1        |
| forward_kl_original    |              3 |        1 |   2 |         0.5      |      1        |               1        |        1   |   1        |
| js_original            |              3 |        1 |   1 |         1        |      1        |               0.681351 |        1   |   1        |
| js_original            |              3 |        1 |   2 |         0.5      |      1        |               1        |        1   |   1        |
| old_margin             |              3 |        1 |   1 |         0        |      0        |               0.318649 |        0.5 |   0.333333 |
| old_margin             |              3 |        1 |   2 |         0.5      |      1        |               1        |        0.5 |   0.333333 |
| u_control_norm         |              3 |        1 |   1 |         1        |      1        |               0.681351 |        1   |   1        |
| u_control_norm         |              3 |        1 |   2 |         0.5      |      1        |               1        |        1   |   1        |
| u_original_norm        |              3 |        1 |   1 |         0        |      0        |               0        |       -0.5 |  -0.333333 |
| u_original_norm        |              3 |        1 |   2 |         0.5      |      1        |               0.681351 |       -0.5 |  -0.333333 |
| random_expected        |              3 |        1 |   1 |         0.333333 |      0.333333 |               0.333333 |      nan   | nan        |
| random_expected        |              3 |        1 |   2 |         0.333333 |      0.666667 |               0.666667 |      nan   | nan        |

无下降事件时 Recall/下降量覆盖率未定义；没有 score/gold 变异时相关未定义。主共同支持池较小时排序分辨率低，不能把表中数值当作统计显著性。同一窗口的所有候选共享 update 身份，因此该表直接检验窗口内定位，但仍不足以估计跨独立更新/seed 的稳定性。

### 7.3 原始读出与 signed ΔM

| skill_id   | supported   |   delta_utility |      P_int |   D_contribution |   D_ungated_contribution |   delta_centered_norm |   delta_norm |       C_upd |   gate_coverage |
|:-----------|:------------|----------------:|-----------:|-----------------:|-------------------------:|----------------------:|-------------:|------------:|----------------:|
| gen_002    | True        |        -3.01724 |  0.0592123 |        0.0560228 |                0.101476  |              113.505  |      218.266 | 0.000518489 |        0.542037 |
| cle_003    | True        |        15.7258  |  0.0916988 |        0.0879431 |                0.224182  |              223.899  |      562.073 | 0.000463877 |        0.458469 |
| cle_004    | True        |        -6.45161 | -0.0123887 |        0.0769062 |                0.164984  |              153.036  |      248.072 | 0.000334997 |        0.495007 |
| cle_006    | False       |        -1.6129  |  0.0403366 |        0.0215787 |                0.0456613 |               76.7731 |      193.687 | 0.000117223 |        0.279487 |

P 列为原始 P，不是风险排序使用的 −P。Reward projection 的优势须体现在统一池中的命中/覆盖或顺序，而非仅因 D 有符号语义就认定其更优。这里 D 本身无符号；有符号趋势应结合原始 P 和连续 ΔM 读出，不做事后反号或重新拟合。

### 7.4 幅度目标的次要审计

| score               |   n_candidates |   events |   k |   precision_at_k |   recall_at_k |   captured_change_mass |   spearman |    kendall |
|:--------------------|---------------:|---------:|----:|-----------------:|--------------:|-----------------------:|-----------:|-----------:|
| D_contribution      |              3 |        3 |   1 |                1 |      0.333333 |               0.624172 |        1   |   1        |
| D_contribution      |              3 |        3 |   2 |                1 |      0.666667 |               0.880243 |        1   |   1        |
| P_int               |              3 |        3 |   1 |                1 |      0.333333 |               0.256071 |       -0.5 |  -0.333333 |
| P_int               |              3 |        3 |   2 |                1 |      0.666667 |               0.375828 |       -0.5 |  -0.333333 |
| delta_centered_norm |              3 |        3 |   1 |                1 |      0.333333 |               0.624172 |        1   |   1        |
| delta_centered_norm |              3 |        3 |   2 |                1 |      0.666667 |               0.880243 |        1   |   1        |
| delta_norm          |              3 |        3 |   1 |                1 |      0.333333 |               0.624172 |        1   |   1        |
| delta_norm          |              3 |        3 |   2 |                1 |      0.666667 |               0.880243 |        1   |   1        |
| random_expected     |              3 |        3 |   1 |                1 |      0.333333 |               0.333333 |      nan   | nan        |
| random_expected     |              3 |        3 |   2 |                1 |      0.666667 |               0.666667 |      nan   | nan        |

|ΔM| 定位与下降定位是两个不同目标。各原预定 score 的方向没有为幅度目标重新调参，该表不能替代下降主分析。

## 8. 中途调用、轨迹变化与支持不足

|   update | control   |   paired_anchor_repeats |   first_action_flip |   suffix_action_divergence |   reward_disagreement |   original_suffix_unique_skills |   original_suffix_steps |
|---------:|:----------|------------------------:|--------------------:|---------------------------:|----------------------:|--------------------------------:|------------------------:|
|       35 | placebo   |                     800 |              29.75  |                     50.5   |                14.625 |                          2.7325 |                 18.6962 |
|       35 | null      |                     800 |              28.75  |                     50.875 |                13.625 |                          2.7325 |                 18.6962 |
|       40 | placebo   |                     800 |              23.875 |                     40.125 |                15     |                          2.7275 |                 13.4612 |
|       40 | null      |                     800 |              22.625 |                     39.75  |                15.75  |                          2.7275 |                 13.4612 |

上述比例单位 %，是各端点 ORIGINAL 与对照臂的行为差异，不是跨 checkpoint 的 S_int；按 matched anchor/repeat 描述，不按 game 等权。distinct Skill 数和长度仅统计自由 suffix，不含重放 prefix。行为 divergence 不必产生 reward 差，更不等价于 harmful flip。

| skill_id   | phase   | supported   | gold_evaluation_available   |   delta_utility |        P_int |   D_contribution |   delta_centered_norm |
|:-----------|:--------|:------------|:----------------------------|----------------:|-------------:|-----------------:|----------------------:|
| gen_002    | all     | True        | True                        |        -3.01724 |   0.0592123  |        0.0560228 |              113.505  |
| gen_002    | initial | False       | False                       |       nan       | nan          |      nan         |              nan      |
| gen_002    | early   | False       | True                        |         0       |   0.0387788  |        0.0316254 |              104.368  |
| gen_002    | middle  | True        | True                        |        -7.35294 |   0.0260676  |        0.0593328 |              117.379  |
| gen_002    | late    | True        | True                        |         0       |   0.0906984  |        0.0660105 |              115.55   |
| cle_003    | all     | True        | True                        |        15.7258  |   0.0916988  |        0.0879431 |              223.899  |
| cle_003    | initial | False       | False                       |       nan       | nan          |      nan         |              nan      |
| cle_003    | early   | False       | True                        |        20.8333  |   0.110163   |        0.0203956 |              232.512  |
| cle_003    | middle  | True        | True                        |        14.7727  |   0.0604912  |        0.0842498 |              218.806  |
| cle_003    | late    | True        | False                       |       nan       |   0.129586   |        0.129108  |              227.137  |
| cle_004    | all     | True        | True                        |        -6.45161 |  -0.0123887  |        0.0769062 |              153.036  |
| cle_004    | initial | False       | False                       |       nan       | nan          |      nan         |              nan      |
| cle_004    | early   | True        | True                        |        -6.45161 |   0.00117886 |        0.0479429 |               94.1106 |
| cle_004    | middle  | True        | False                       |       nan       |  -0.0389784  |        0.0721909 |              152.358  |
| cle_004    | late    | True        | False                       |       nan       |  -0.00214604 |        0.0898967 |              174.411  |
| cle_006    | all     | False       | True                        |        -1.6129  |   0.0403366  |        0.0215787 |               76.7731 |
| cle_006    | initial | False       | True                        |        -1.6129  |   0.0403366  |        0.0215787 |               76.7731 |
| cle_006    | early   | False       | False                       |       nan       | nan          |      nan         |              nan      |
| cle_006    | middle  | False       | False                       |       nan       | nan          |      nan         |              nan      |
| cle_006    | late    | False       | False                       |       nan       | nan          |      nan         |              nan      |

训练信号支持和自然 gold anchor 支持分开记录；没有 anchor 的阶段标签留空，不能伪造零变化。相同 anchors/repeats 的不同 phase 和 all 不是独立窗口。

## 9. 结论边界与下一阶段的增量价值

本批完成的是测量链路、预锁定直接排序检验及明确标记的事后排序审计。旧样本中 D 相比 centered norm 的平均下降名次、AP 和 pooled 单调性确有描述性优势；一个旧窗口中 ungated D 也更好。这些证据应保留，不能仅因 Top-1 或新窗口不理想而抹去。但新窗口没有复现 D 的优势，P 的良好排序亦被 KL/JS 等简单基线匹配，故不能宣布 reward projection 的稳定增量已经成立。

第 5.2–5.4 节进一步区分两类证据：旧 cle_004 在 NULL 和非初始 early anchors 上也下降，增强的是效用变化现象的可信度；旧 D 的部分排序优势则不能扩展为优于 C_upd/C_upd_centered 等全部候选读出。C 与 D 都利用 reward direction，C 的良好表现可支持更宽的 reward-aligned readout 候选方向，但不证明 Skill-specific opposition 的独特收益。当前没有跨 seed 的 Phase2 预测验证，也没有因果归因或编辑后收益验证。

预测价值不要求精准回归，不要求每次 Top-1 都正确，也不要求 Skill/seed 同向变化；应在固定规则下比较总体排名、单调性和不同编辑预算的收益，并按窗口/seed 报告不确定性，而不是根据各窗口 gold 临时择优换指标。当前只有一条 seed303 延续路径、4 个评估 Skill；旧 19 个支持单元中的 7 个为时间外测试子集，新窗口有 3 个共同支持单元，不能加成 29 个样本，也不能把 22 条支持记录当作 22 个独立 Skill/更新实验。相邻窗口共享端点，旧新 cohort 的 anchors 又不同，不能无控制混池或归因于窗口长度。

下一步优先增加独立更新窗口/seed 与自然有支持的 Skill，固定 gated D、ungated D、−P、norm、KL/JS、old-margin 等比较及标签阈值；对有/无下降窗口分别报告适用统计。AP、平均名次与相关描述同一份排序的不同侧面，不是互相独立的多份证据。门控门槛如需改动，应在开发数据上确定后用新数据检验，不能在当前 gold 上选最优值并作为确认性结果。训练 batch 与 unseen anchors 的 state 分布不同，局部投影到长期 reward 仍是经验问题。

预测锁定前没有使用 U40 held-out 续跑结果，符合 without post-update gold rollout 的边界；仍使用已有训练 rollout、完整新旧模型前向和旧效用 evidence。离线 gold 是验证成本，不是在线输入。尚未做 equal-budget 新 trajectory/短 behavioral probe 对比，不能宣称已验证节省多少 rollout 或端到端 FLOPs/时间优势。

Phase3 应预注册相同 top-k 编辑预算、相同编辑器和独立编辑后评估，比较 D/−P/norm、随机、old-margin、已有 trajectory-summary，以及预算匹配的新 rollout/probe。同时扫描所需编辑比例、rollout 预算和提升/错误修改率，形成成本—收益曲线。不能用当前观测性排序直接宣称它是失效机制或 Skill 修改已有效。

## 10. 归档与可复核性

科学协议 SHA-256：`0ca78a05b4d58584be39ecfaa9335458b41dfd50ea6e34f80e43be797acbe47a`；新端点完整性和逐轨迹 SHA-256：[evaluation_completion.json](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/evaluation_completion.json)。

原始指标/全部分层：[CSV](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/window_metrics/raw_features_and_semantic_utility.csv)；预锁定排序与实际标签：[CSV](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/window_metrics/locked_scores_and_gold.csv)；top-k 全结果：[CSV](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/window_metrics/ranking_metrics.csv)；轨迹目录：`/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/evaluations/u0040/trajectories`。

旧 19 单元：[CSV](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/reports/2026-09-12-observation-audit-v1/all_supported.csv)；旧 7 测试单元：[CSV](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/reports/2026-09-12-observation-audit-v1/heldout.csv)；旧多指标比较：[CSV](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/reports/2026-09-12-observation-audit-v1/comparisons.csv)。

报告只描述完成的实验与统计边界，不混入中断/启动失败作为实验样本。旧报告和 Phase1 结果保持不变；执行资源修订与日志独立归档。

## 11. 追加分析：整体下降排名、排序单调性与 D 的门控

追加日期：2026-09-14。依据用户讨论，对已归档数值进行只读重算，未运行新的 RL 或环境评估。本节的新增平均名次、完整 AP/相关对照及窗口内复核属于事后描述性分析，不将其重新标记为预注册测试；原预锁定的信号方向与支持集合均保持不变。

### 11.1 评估目标与统计口径

此处“负效应”严格指 ΔM_sem<0，即 ORIGINAL−PLACEBO 边际效用下降；不是 M_new<0，也不是经独立 calibration 确认的 harmful sign flip。主表采用点估计任意下降（排除绝对值≤10⁻¹²的数值零误差）；另按既有 5 pp 阈值作敏感性比较。CI 包含或触及 0 的下降仍是不确定标签，不因为纳入排名而升级为可靠下降。

- 下降单元平均名次：分数越高风险越大，降序第 1 名为最优先；并列取平均名次。仅在有下降单元时定义，值越小表示下降单元整体越靠前。不同候选池大小的原始名次不直接跨组比较。
- Average precision（AP）：衡量下降单元在整个风险列表前端的集中程度，不限定一个 k。不同事件比例下的 AP 不直接比较绝对优劣；没有下降的窗口本审计记为未定义。
- Spearman/Kendall：风险分数与连续 −ΔM 的秩相关，正值表示更高风险总体对应更大下降；它不同于仅区分“下降/非下降”的 AP。
- 所有方向保持固定：D、ungated D、centered norm、KL/JS 和 control-update norm 用原值，P 用 −P；因此 Spearman(−P,−ΔM)=Spearman(P,ΔM)。不依据当前 gold 反号。
- 全部比较只使用 placebo、phase=all 的共同支持单元。旧 19 单元包含开发/隔离/测试，旧 7 单元是其中测试子集；新窗口仅 3 单元。分层 phase 与 all 不作为独立样本重复计数。

### 11.2 整体下降排名与单调性

以下为任意点估计下降。旧 19 单元有 4 个下降，旧 7 测试单元有 2 个下降，新 3 单元有 2 个下降。行间方法比较必须限定在同一数据范围。

| 数据范围 | 风险排序 | 下降平均名次 ↓ | AP ↑ | Spearman ↑ | Kendall ↑ |
|---|---|---:|---:|---:|---:|
| 旧 19 单元，pooled | D，带门控 | 4.00 | 0.799 | +0.417 | +0.307 |
| 同上 | D，无门控 | 3.25 | 0.854 | +0.284 | +0.164 |
| 同上 | centered norm | 4.50 | 0.667 | +0.062 | +0.036 |
| 同上 | −P | 12.75 | 0.190 | −0.049 | −0.036 |
| 同上 | KL | 7.50 | 0.551 | +0.207 | +0.150 |
| 旧测试 7 单元，pooled | D，带门控 | 1.50 | 1.000 | +0.556 | +0.411 |
| 同上 | D，无门控 | 1.50 | 1.000 | +0.482 | +0.309 |
| 同上 | centered norm | 2.00 | 0.833 | +0.185 | +0.103 |
| 同上 | −P | 6.50 | 0.226 | −0.408 | −0.206 |
| 同上 | KL | 3.00 | 0.700 | +0.037 | 0.000 |
| 新窗口 3 单元 | D，带门控 | 2.50 | 0.583 | −0.500 | −0.333 |
| 同上 | D，无门控 | 2.50 | 0.583 | −0.500 | −0.333 |
| 同上 | centered norm | 2.50 | 0.583 | −0.500 | −0.333 |
| 同上 | −P | 1.50 | 1.000 | +1.000 | +1.000 |
| 同上 | KL | 1.50 | 1.000 | +1.000 | +1.000 |

旧数据中 D 的整体排序优势是真实的描述性结果：例如旧测试 D 的两个下降单元位于第 1、2 名，而 centered norm 位于第 1、3 名；这不是 Top-1 是否命中的重复表述。ungated D 在旧全样本的 AP/平均下降名次更好，gated D 的连续下降单调性更好，说明“下降是否靠前”与“下降程度是否单调”并非同一个目标，门控也不是各项指标都改善。

这些旧 pooled 优势尚未在新窗口复现。新窗口中两版 D 与 centered norm 顺序一致，均将上升最多的 cle_003 放在最前；−P、KL、JS、control-update norm 则得到相同的正确点估计顺序。因此当前不能把旧数据的优势推广为稳定的 reward-directed 增量，也不能反过来用新窗口覆盖旧正面证据。

旧测试下降均来自 U35、上升均来自 U34，故 pooled AP/相关可能部分反映 update 间分离；但不能据此断言全部关联都只是混杂。应结合下一节的窗口内比较判断。同一排序的平均名次、AP、Spearman/Kendall 并不是多个独立复现实验。

### 11.3 同一更新窗口内的复核

先看整体顺序，正值表示风险排序符合连续下降程度。每行只有 3–4 个 Skill，零标签较多，不计算把这些相关当作独立成功次数的显著性。

| 窗口 | 候选数 | ρ(P,ΔM) | ρ(D,−ΔM) | ρ(ungated D,−ΔM) | ρ(centered norm,−ΔM) |
|---|---:|---:|---:|---:|---:|
| U30→31 | 4 | +0.775 | −0.775 | −0.775 | −0.775 |
| U31→32 | 4 | +0.632 | −0.632 | +0.316 | −0.316 |
| U32→33 | 4 | +0.258 | +0.775 | +0.775 | +0.775 |
| U33→34 | 3 | +0.500 | −0.500 | −0.500 | −0.500 |
| U34→35 | 4 | −0.738 | +0.738 | +0.738 | +0.738 |
| U35→40 | 3 | +1.000 | −0.500 | −0.500 | −0.500 |

P 在 5/6 个窗口呈正秩相关，是窗口内排序线索；旧 pooled 相关弱不能直接等同于每个窗口都无信息。但这包含开发数据、共享端点及相同 Skill 的重复测量，不能称为跨 seed 复现，也不代表 P 的正负号逐项正确。

再看下降单元是否整体靠前。仅对存在点估计下降的窗口列出平均名次；U30→31、U33→34 无下降，因此不以零或“成功”填补。

| 窗口 | 候选数 / 下降数 | D，带门控 ↓ | D，无门控 ↓ | centered norm ↓ | −P ↓ | KL ↓ |
|---|---|---:|---:|---:|---:|---:|
| U31→32 | 4 / 1 | 3.00 | 1.00 | 2.00 | 2.00 | 1.00 |
| U32→33 | 4 / 1 | 1.00 | 1.00 | 1.00 | 2.00 | 3.00 |
| U34→35 | 4 / 2 | 1.50 | 1.50 | 1.50 | 3.50 | 2.00 |
| U35→40 | 3 / 2 | 2.50 | 2.50 | 2.50 | 1.50 | 1.50 |

U31→32 的 ungated D 确实有窗口内优势：将唯一下降的 cle_004 排第 1，centered norm 排第 2，gated D 排第 3，KL 也排第 1。其他三个有下降窗口中两版 D 的平均下降名次与 centered norm 相同。因此“gated D 未体现稳定窗口内优势”和“ungated D 在一个旧窗口更好”应同时报告。

Top-1 与整体排序仍需互补：旧三个有下降窗口中，−P 的 Top-1 均未命中下降，尽管其中两个窗口的 Spearman 为正；U34→35 的 D 将 −1.67 pp 的 cle_003 放在 −14.29 pp 的 cle_004 前面，Top-1 仅覆盖 10.4% 的下降量，但 Top-2 命中两个下降单元。因此不能只用整体正相关宣布优先编辑有效，也不能只用 Top-1 否定整体排名收益。

### 11.4 下降标签阈值的敏感性

上述主表使用 ΔM<0。沿用既有“下降超过 5 pp”口径后，旧 19 单元、旧 7 单元和新 3 单元各只有一个事件。此时 AP 主要反映唯一事件的位置：

| 数据范围 | 事件 | D 排名 | ungated D 排名 | centered norm 排名 | KL 排名 |
|---|---|---:|---:|---:|---:|
| 旧 19 单元 | cle_004@U35 | 4 | 6 | 6 | 1 |
| 旧测试 7 单元 | cle_004@U35 | 2 | 2 | 3 | 1 |
| 新窗口 3 单元 | cle_004@U40 | 2 | 2 | 2 | 1 |

阈值改变的是“哪些 gold 单元算下降事件”，不改变 D 数值、排序方向或连续 −ΔM 相关；亦不改变下降量覆盖所用的 max(0,−ΔM)。旧测试 D 优于 centered norm，但 KL 在该较大下降事件上更好；因此不能只选择有利标签阈值概括全部优势。新 cle_004 的 CI 仍跨 0，不因下降点估计超过 5 pp 而成为确证事件。

### 11.5 D 的阈值、无门控版本与聚合分母

此前报告的主 D 是 D_contribution，即 idea 中带 token 级门控的版本，不是无门控分数：

\[
D_{\mathrm{gated}}(s,c)=\frac{1}{N}\sum_j
\mathbf 1(\mathrm{direction/fidelity\ valid}_j)
\mathbf 1(C_j^{\mathrm{upd}}\ge 0)
\mathbf 1(\|\delta_j^{\mathrm{int}}\|_2\ge 10^{-8})
[-P_j^{\mathrm{int}}]_+ .
\]

- 更新 fidelity 门槛 τ_C=0：方向有效且 C≥0 才计入负向投影。其含义是“不与局部方向反向”，不是高强度 fidelity 已经验证。
- interaction 幅度门槛 τ_δ=max(10⁻⁸,10×同 checkpoint 重复前向噪声 p95)。旧/新批次测量噪声 p95 均为 0，实际 τ_δ 均为 10⁻⁸，在读取目标 gold 前固定。它主要是数值下限，不是事后调整的效果阈值。
- 方向有效性以 ||d||>ε 检查；fidelity 还要求 ||u_original||>ε，ε=10⁻¹²。[-P]₊=max(0,−P) 是 reward-opposing 定义本身的负部截断。
- N 是该 Skill/context/phase 的全部实际 response loss tokens，包含零 advantage 与未通过 gate 的 token；失败项贡献为零，而不是从分母中删掉。分数主聚合是 token 等权，不是仅对负向 token 条件平均。
- 训练支持另要求至少 20 个非零 advantage decisions、4 个非零支持 games、8 条非零支持 trajectories。不满足时 abstain；不会因 D 很小而被当成低风险。

已并行计算并归档的 D_ungated_contribution 为：

\[
D_{\mathrm{ungated}}(s,c)=\frac{1}{N}\sum_j
\mathbf 1(\|d_j\|_2>\epsilon)[-P_j^{\mathrm{int}}]_+ .
\]

它移除 C_upd 与 interaction norm 门控，仍保留 reward direction 有效性、负部截断及相同聚合分母/Skill 支持标准。“无门控”不等于去掉负部截断后直接使用 −mean(P)；两者不同。

没有设置 Skill 级“D>d₀ 才算风险”的阈值：直接对连续 D 降序排序。必须区分 token 级信号门控、Skill 支持门槛、gold 的 0/5 pp 下降标签阈值，以及 Top-k 的预算截点。不能把其中一类阈值的效果归因给另一类。

对应实现：[token 级信号](/home/wangyifan/skill-RL/SkillRL/phase2/direction.py:28)、[聚合与支持](/home/wangyifan/skill-RL/SkillRL/phase2/aggregate.py:66)、[新批次噪声校准](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/signals/calibration.json)。门控会实质改变排序，当前观察不支持其总是改善，也不支持事后择优切换主版本。

### 11.6 修订后的判断与复核归档

> 整体排名与单调性是合理的主评估维度，不需要以精准回归或单个 Top-1 作为必要条件。D 在旧样本中相对 norm/KL 的部分整体下降排序有优势，无门控版本在一个旧窗口也有优势；P 则在多数窗口呈相对排序线索。但旧测试中 C_upd 等读出匹配部分优势，新窗口未复现 D 的优势，P 的良好排序也被通用偏移基线匹配。因此 Phase2 已有正面的预测可行性线索，独特、稳定的 reward-directed 增量和跨 seed 泛化仍待检验；NULL/中途锚点的稳健性支持效用变化现象，而不直接证明预测成功。

应保留正面和负面证据，并以固定规则检验总体收益，而非要求每个窗口都获胜或在当前 gold 上挑选最优版本。当前追加统计没有改变原预注册结果或解锁任何新训练任务。

本次完整数值包括 7 种固定方向分数、0/5 pp 两种标签口径：

- [跨 update 合并及新窗口统计 CSV](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-ranking-interpretation-v1/overall_ranking_audit.csv)：42 行，含平均下降名次、AP、下降对其余单元 AUROC、Spearman、Kendall。
- [逐窗口统计 CSV](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-ranking-interpretation-v1/within_window_ranking_audit.csv)：84 行；无下降时平均下降名次/AP 留空，不作为零分。
- [首次修订清单与文件哈希](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-ranking-interpretation-v1/revision.json)：记录第 11 节首次追加时的文档版本及上述审计 CSV；该历史记录不覆盖。
- [当前综合补全修订清单](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1/reports/2026-09-14-phase2-synthesis-v2/revision.json)：记录本次报告哈希、旧/新输入哈希、补充读出及稳健性核验。原始 full_report_completion.json 和首次修订清单各自对应历史文档版本；实验完成证明与原始轨迹保持不变。
