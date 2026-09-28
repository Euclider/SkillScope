# Phase1–2 核心结果与 Phase3 设置（汇报版）

> 状态：基于已完成报告与冻结配置汇总，未重跑实验；Phase3 尚无已封存的闭环收益结论。

## Material Passport

- academic-research-suite / experiment-agent；2026-09-26；ANALYZED，结论置信度 CAUTION。

## 核心结论

1. **冻结技能的边际效用会随 policy 更新改变。** Qwen3.5-4B 在 ALFWorld 上独立训练 seed 404、505，各完成 5 次 GRPO 更新；SkillNet-37 和冻结逐状态 top-1 router 均不变。两端在同一首次调用锚点、同一续跑 seed 下做 8 组配对效用评估。404 的 `object-placer` 效用变化为 **+13.6 个百分点**（95% 配对区间 **+1.6 至 +26.6**）；505 的 `object-state-inspector` 为 **+34.7 个百分点**（**+18.8 至 +51.4**）。这支持“固定技能的贡献具有 policy 依赖性”；目前没有单个下降技能的区间明确低于零，不能宣称已稳健识别具体退化技能。
2. **固定状态读出有下降排序信号，但 reward 增益仍属探索性。** 在统一的 PLACEBO 对照、game 等权聚合和 8 组 gold 标签下，Phase3 选用的 reward-directed `D_sign_balance` 对下降技能的 AP/AUROC 如下；`‖Hδ‖` 是不含 reward 的中心化交互幅度基线。读出只使用窗口起点训练批次及两端模型的固定状态前向，不使用效用续跑结果。

   | 独立 RL seed | U0→U5 Unseen 成功率 | 可比较技能／下降点估计 | `D_sign_balance` AP / AUROC | `‖Hδ‖` AP / AUROC |
   | --- | ---: | ---: | ---: | ---: |
   | 404 | 23.9% → 17.9% | 18／7 | 0.734 / 0.753 | 0.540 / 0.597 |
   | 505 | 23.9% → 36.6% | 19／3 | 0.567 / 0.792 | 0.483 / 0.625 |

   两个 seed 上 `D_sign_balance` 的描述性排序均优于该幅度基线，但技能数和下降标签少，且指标选择已参考这两组标签；**不能据此宣称 reward 读出已有确认性、跨 seed 的稳定增益**，也不能把读出排序当成编辑后成功率收益。两 seed 的下降基率不同，AP 不宜直接横向比较。

## Phase3 当前实验设置

- **目的与六臂：**比较读出引导的实际技能演化收益。依次独立训练 `D_sign_balance` 主方法、SkillRL-style 失败轨迹基线、纯幅度 `‖Hδ‖`、旧负部 gated D、`−P`、中心化 `+C`；各臂共享初始 Qwen3.5-4B、SkillNet-37、RL seed **707**、router、编辑器与预算规则，但 policy、优化器和后续 skill bank 独立。SkillRL 臂是共用编辑器的适配基线，**不是官方原样复现**。
- **训练与窗口：**8×RTX 5090，vLLM 生成＋原生 FSDP/GRPO；学习率 `1e-6`，每次更新 **16 个 game×8 条轨迹**。每 **5 次更新**比较相邻快照，在窗口起点训练批次上用稳定 FP64、game 等权口径算读出，再提供一次编辑机会；先统一运行到 **U20（4 个窗口）**作为中期点，优化器规划上限仍为 U150，不把 U20 当最终收敛结果。
- **路由与编辑：**冻结 Qwen3-Embedding-0.6B 对完整活动库逐状态 top-1 检索；当前本机用 GPU 微批量实现。共同 `o3` 编辑器可 `ADD/MODIFY/DELETE/MERGE/NOOP`；每窗最多 3 个候选、8 条证据轨迹、3 个 mutation units。没有按动作偏置阈值跳过编辑的门控，编辑器可 NOOP。候选只用 Seen gate 配对验收（每类 2 个 game、5 pp 容差、2 个评估 seed）；Unseen 不参与选择或编辑。
- **评估与记录：**每臂报告 140 Seen、134 Unseen 的任务成功率，以及编辑／拒绝／rollback、库规模、actor/router/readout/editor 的 token 或计算成本与耗时。Phase3 **不做 Phase2 的逐技能 gold 效用续跑**；只有六臂闭环完成后，才能检验读出优先级是否改善下游表现。U20 若用于决定续跑，同一 Unseen 集后续结果须标为探索性。

数据依据：[404 八组效用](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s404-v1/reports/phase1-results.md)、[505 八组效用](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s505-v1/reports/phase1-results.md)、[404 排序](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s404-v1/precision/gold8-ranking.csv)、[505 排序](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s505-v1/precision/gold8-ranking.csv)、[Phase3 冻结设置](SkillRL/docs/phase3/PHASE2_ALIGNMENT_V3.md)。

## Phase1–2 补充结果

**测量范围。** ALFWorld 六类任务共用冻结 SkillNet-37；404/505 各独立运行 5 次 GRPO 更新，每次 16 个 game×8 条轨迹，即各 640 条训练轨迹。U0 自然调用覆盖 **25/37** 个技能，效用评价保留每条来源轨迹中每个技能的首次调用锚点，不再按调用次数或 game 数筛除；有起点训练读出的可比较池为 **18/19** 个技能（404/505）。gold 在同一锚点、同一续跑 seed 下配对，8 组重复提高效用估计精度，但不是 8 个独立 RL seed；606 尚无完成结果。边际效用 `M = SR(ORIGINAL) − SR(token 匹配 PLACEBO)`，`ΔM = M(U5) − M(U0)`；NULL 是辅助对照。锚点后对目标技能的干预持续到续跑结束，因此不是单次调用的孤立效应。

| Policy | Seen 成功率 | Unseen 成功率 |
| --- | ---: | ---: |
| 共同 U0 | 33/140 = **23.6%** | 32/134 = **23.9%** |
| seed404 U5 | 28/140 = **20.0%** | 24/134 = **17.9%** |
| seed505 U5 | 36/140 = **25.7%** | 49/134 = **36.6%** |

这些是 policy＋冻结库的**整体成功率**，不是技能边际效用。除上面的正向个例，505 的 `heat-object-with-appliance` 效用变化也为 **+9.4 pp**（95% 配对区间 **+0.8 至 +19.1**）。逐技能区间未做跨技能多重比较校正。两 seed 均有负的 `ΔM` 点估计，但 8 组 gold 下**没有单个下降技能的区间明确低于零**；点估计负号不能当作已确认退化。

**历史先导结果单列，不与新库混池。** 旧 SkillRL 技能库、Clean 任务、训练 seeds 101/202/303 中，`cle_003` 在 U20 相对 B0 的三 seed 汇总效用变化为 **+40.56 pp [28.89, 52.78]**；动作交互 `S_int` 加入小型 probe 后，leave-one-seed-out 的“是否变化”AUPRC **0.192→0.274**，但“是否下降”AUPRC **0.317→0.314**。旧 seed303 的 19 个技能×相邻窗口单元，gated D/中心化交互范数的下降 AP 为 **0.799/0.667**；后续 U35→U40 仅 3 技能时两者均为 **0.583**。这些是不同库、不同规模的先导证据，不是当前指标的额外独立复现。[历史 Phase1](2026-09-04-qwen35-clean-all-skill-three-seed-utility-results.md)；[历史 Phase2](phase2-complete-analysis.md)。

### Phase2：下降方向、审查预算和变化幅度

下表补足主表以外的读出：统一 PLACEBO、all phase、game 等权、8 组 gold，下降定义为 `ΔM<0`。AP 衡量下降技能靠前的程度；AUROC 为下降对其余技能的区分度；ρ 是风险分数与连续 `−ΔM` 的 Spearman。404 的 18 个可比较技能中 **7 降/7 升/4 零或小变化**，505 的 19 个中 **3 降/14 升/2 零或小变化**；对应 AP 基率约 **0.389/0.158**，不能直接跨 seed 比 AP 大小。

| 读出（高分优先审查） | 404 AP / AUROC / ρ | 505 AP / AUROC / ρ |
| --- | ---: | ---: |
| reward `D_sign_balance` | **0.734 / 0.753 / +0.255** | **0.567 / 0.792 / +0.416** |
| 无 reward 的 `‖Hδ‖` | 0.540 / 0.597 / +0.282 | 0.483 / 0.625 / −0.032 |
| 旧负部 gated D | 0.717 / 0.779 / +0.303 | 0.365 / 0.792 / +0.458 |
| `−P_int` | 0.646 / 0.714 / +0.279 | 0.288 / 0.646 / +0.314 |
| 中心化 `+C_upd` | 0.537 / 0.545 / +0.062 | 0.152 / 0.313 / +0.024 |

按固定审查预算，`D_sign_balance` 的 Top-2 命中在 404/505 为 **2/2、1/2**，纯幅度为 **1/2、1/2**；Top-5 分别是 D 的 **3/5、2/5**，纯幅度的 **2/5、1/5**。这是候选排序表现，**不是编辑后的 repair 或任务收益**。[404 排序预算](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s404-v1/reports/ranking_budgets.csv)；[505 排序预算](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s505-v1/reports/ranking_budgets.csv)。

两个补充目标也已评估：若只把**下降超过 5 pp**算事件，每 seed 仅 **2** 个下降，D 的 AP 为 **0.258/0.700**，纯幅度为 **0.134/0.567**（404/505）；样本太少，不能独立作强结论。对变化**绝对幅度** `|ΔM|`，纯幅度 `‖Hδ‖` 的 Spearman 在 404 为 **−0.373**、505 为 **+0.272**，未呈一致关系；动作变化大不等于技能效用一定变化大。其他 reward 变式并未统一占优：`D_real` 的下降 AP 为 **0.493/0.170**，`D_factor` 为 **0.330/0.479**。完整公式族共归档 285 个分数列，属于探索性指标比较，不能从中择优后把同一 404/505 标签上的成绩当独立验证。[404 幅度结果](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s404-v1/reports/magnitude-absolute-utility.csv)；[505 幅度结果](SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s505-v1/reports/magnitude-points.csv)。

## Phase3 设置补充：公平性、编辑决策与成本

六臂从共同 B0、SkillNet-37 和 RL seed **707** 出发，之后 policy、Adam、轨迹、当前技能库和缓存各自独立，在同一组 8 卡上**串行**运行；不能用一次 RL 兼作六条闭环。共同训练配方为 GRPO、`lr=1e-6`、16×8 轨迹/update、全局决策 minibatch=128、每 GPU microbatch=1；沿用 SkillRL 对齐参数 KL=0.01、entropy=0.001、clip=0.2、weight decay=0.01、invalid-action penalty=0.1。vLLM 用于生成及整局评估；固定状态全词表前向使用 BF16 HF/SDPA，读出算术统一为稳定 FP64、game 等权。六臂都先停在 **U20 中期点**，保留 U150 优化器规划上限；四个相邻窗口为 `U0→5、5→10、10→15、15→20`，不是始终对 U0 比较。每窗用**起点第一轮真实优化 batch**与两端 policy 读出，技能内容在窗内固定，编辑通过后才影响下一窗。支持规则为当前版本技能至少有一个自然训练 loss token，不沿用旧 20/4/8 门槛。[冻结六臂配置](SkillRL/configs/phase3_setting_embedding_v3.json)。

**统一编辑接口。** 冻结 Qwen3-Embedding-0.6B 对完整活动库逐状态 top-1 选技，与 policy 不共享权重；这是 SkillRL embedding 的状态级适配，不是其官方 task-only、整局 top-k 原样复现。readout 臂给共同 `o3` 编辑器 top-k 技能 ID 与其实际出现的轨迹（可成功或失败）；SkillRL-style 臂给失败轨迹；两者均可见完整当前库，**不传 Phase2 gold 效用**。允许 `ADD/MODIFY/DELETE/MERGE/NOOP`；每窗最多 **3 个候选、8 条证据轨迹、3 个 mutation units**，可 NOOP 或因缺支持而弃权。固定机会和预算上限，不强迫各臂实际编辑数一致。没有“动作偏置低于阈值便跳过本窗编辑”的 gate；幅度只在幅度臂用于候选排序，其他臂仅作影子记录。`o3` 用兼容 `/v1` 网关、medium reasoning；运行模板上限为单次输入 64k tokens、输出 8192 tokens、每臂 30 次 API 请求（U20 至多 4 个窗口），API 凭据不进入实验报告。

**候选验收。** 当前 policy 不变，在相同 Seen gate game×解码 seed 下配对比较旧库与候选库；每类 2 个 gate game、seeds `1707/2707`、观测成功率容差 **−5 pp**。gate games 与编辑证据 Seen games 不交叉；Unseen 不参与候选选择、编辑、gate 或调阈值。提交前拒绝计 **rejection**，不计 **rollback**；当前无接受后自动 rollback。这个容差是工程验收规则，**不是统计非劣证明**。若看过 U20 Unseen 后据此决定续训，同一 Unseen 的后续分析须标为探索性序贯分析。[运行模板](SkillRL/configs/phase3_runtime_template_v3.json)。

**评价与归档。** 每臂按 140 Seen、134 Unseen 及六类任务报告成功率，并记录相同 game/seed 下的分支差值；逐轮记录训练 reward、有效动作、episode 步数，逐窗记录候选/实际修改技能数、ADD/MODIFY/DELETE/MERGE/NOOP、mutation units、接受/拒绝/rollback、bank 大小及版本、repair/regression。成本分栏报告 actor 输入/输出 tokens、本地 router 调用/编码量/时间、readout 前向 tokens/时间、o3 请求/usage/延迟、wall-clock 和可核算费用；缺失 usage 不填零。SkillRL-style 臂的只读影子读出不影响选择，其计算成本与基线算法必要成本分列。Phase3 不做 Phase2 的 O/P/N gold 效用续跑，也不保存全词表概率张量；每窗封存后只保留最新可恢复 checkpoint 和必要证据。多个 game、解码 seed 或窗口不是独立 RL seed，U20 若有差异也不能直接宣称跨 seed 的确认性优势。[完整启动口径](SkillRL/docs/phase3/PHASE2_ALIGNMENT_V3.md)。

源报告按 11/11 类常见统计误用检查；本摘要最重要的边界是自然支持选择、下降低基率、多变式择优、把续跑当训练 seed，以及把排序相关误写为编辑因果收益。本轮仅扩充汇报，没有启动训练、续跑或新检验。
