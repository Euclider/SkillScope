## Material Passport

- Origin Skill / Mode：academic-research-suite / experiment-agent / validate（既有设置核对）
- Origin Date：2026-09-23；Version Label：phase3_setting_summary_v1
- Verification Status：ANALYZED；正式四分支闭环实验尚未运行

# Phase3 简洁设置摘要

目标：在共享初始化、路由器、编辑器和预算上限下，比较不同“候选优先级／轨迹证据选择”方案带来的累计技能演化收益。本文记录**现有实现与待确认项**，不是新实验启动许可，也不是已完成的性能结果。

## 1. 分支、训练与滑动窗口

| 项目 | 当前登记设置 |
|---|---|
| 四条独立 RL 分支 | SkillRL-style failure-driven；gated D 主实验；−P；原始 +C_upd |
| 共同初始化 | Qwen3.5-4B B0；SkillNet-37，不混入旧 SkillRL 库；RL seed=404，各分支独立优化器 |
| ALFWorld | 六类任务共用完整当前库；3553 train games；140 Seen、134 Unseen |
| RL | GRPO，lr=1e-6；每轮 16 games×8 trajectories；最多 150 轮 |
| 优化批次 | 全局 PPO minibatch=128 个决策行，microbatch=1/GPU，PPO epoch=1；不是每 128 条完整轨迹才一次 Adam 更新 |
| 其他参数 | KL=0.01/low_var_kl，entropy=0.001，clip=0.2，weight decay=0.01，invalid-action penalty=0.1；不新增 SFT/LoRA |
| 生成限制 | prompt≤4096、response≤512、最多 50 环境步、history=2；训练温度 1.0，评估 0.4，top-p=1 |
| 验证／编辑机会 | 每 5 轮一次，最多 30 次；预定最终端点 U150，不按 Unseen 挑最佳 checkpoint |

四分支技能库会分化并影响后续 RL，必须独立训练，不能用同一次 RL 冒充四个闭环。

窗口依次为 **U0→U5、U5→U10、…、U145→U150**。每窗使用该窗起点策略产生的首轮真实优化 batch（含实际 advantage），在同一批状态/动作上比较两端 policy；窗口内 skill 内容固定，编辑通过后才影响下一窗。不是一直用 U0，也不是把五个逐轮 D 相加。每个 block 恢复 policy/optimizer/scheduler 等原生状态；环境流按 `404+16×block_index` 重新创建，属于共同 block 适配，不声称与原生 SkillRL 连续运行逐 bit 相同。

## 2. Router、读出与编辑器

**Router：**冻结 Qwen3-Embedding-0.6B，与 policy 不共享权重、不训练。输入仅含任务、当前观察、可用动作、最近两步可见历史与 step index；对完整 active bank 做余弦 top-1，无 task 子库。库版本变化即重建索引/缓存。称为 **SkillRL embedding 的逐状态适配版**，不是官方 task-only／整局 top-k 的原样复现；不调用付费 router API。

**当前读出：**真实技能对 token 匹配 PLACEBO；逐 token 算 C/P/D，再按技能平均。D 仍为 `mean(gate × max(−P, 0))`；P 臂按 −P 排序，C 臂按**未中心化**的 +C_upd 排序。`τC=0`，`τδ=max(1e-8, 10×首窗重复前向噪声p95)`，ε=1e-12；零优势和未过门控 token 计入分母。每窗记录三种分数，不根据结果换主指标。

当前内容版本须至少 20 个非零方向决策、4 个 game、8 条轨迹才进入排序；top-k≤3，不足则弃权，不填零或强迫路由。新/改写/合并版本不继承旧分数。

**共享编辑器：**`o3`、medium reasoning，网关 `https://api.zhizengzeng.com/v1`；凭据仅环境注入，不进入导出文件。同一 prompt、输出 schema、证据/调用/token 上限，无额外 summary LLM。

| 输入或决策 | 约定 |
|---|---|
| 共同输入 | 完整当前 skillbank、选定的真实观察/动作/调用记录及成功标记 |
| SkillRL-style 证据 | 当前窗口的失败轨迹；失败不被直接认定为某 skill 导致 |
| Readout 证据 | top-k 技能 ID＋这些技能实际出现过的轨迹，可成功或失败；不传 gold 效用标签 |
| 证据上限 | 现模板最多 10 条；输入 100,000 tokens、输出 8,192 tokens，属于待运行清单确认的工程默认值 |
| 输出操作 | ADD / MODIFY / DELETE / MERGE / NOOP；带版本引用、理由和证据 ID |
| 编辑量 | 每次≤3 mutation units；add/modify/delete 各 1，N→1 merge 算 N+1，故 2→1 合并耗 3 |

Top-k 是审查建议，**不是必须修改 3 个，也不是硬性只能改 top-k**；编辑器可读完整库，实际操作受统一 mutation budget 约束。允许库增长；删除保留 tombstone，历史版本不可覆盖。

## 3. 变量控制与接受规则

- **主要自变量：**失败驱动与各读出驱动的候选／证据选择规则。共用编辑器，但实际证据和后续训练轨迹不同；不能说只换一个数值而其余观测完全相同。
- **共同控制：**B0、初始库、RL 配方与机会频率、数据划分/seed 规则、router 模型及查询、editor 及操作空间、各类预算上限、评估 games/seeds、停止规则、硬件/推理配置。预算上限一致不等于实际开销相同，必须报告利用量与弃权。
- **Seen gate：**gate games 与 evidence Seen games 不交叉。候选库与旧库在同一个当前 policy、同 game×eval-seed 下配对评估；观测 ΔSR≥−预登记容差才接受。这个阈值不自动构成统计非劣检验。
- **Unseen 隔离：**不用于编辑、门控、选分支或调阈值；只用于预声明最终评估。完整 Seen 包含开发见过的 games，不能称全部 held-out。
- **拒绝与回滚分开：**提交前拒绝候选计 rejection，不计 rollback。当前没有接受后自动 rollback；手动恢复只允许 bank，不回滚 policy/optimizer，并须单列事件。
- baseline 正式名称为 **SkillRL-style with a shared editor**；统一编辑器、操作空间和逐状态路由后，不是原生 SkillRL 的严格复现。

## 4. 待评测／记录的关键指标

| 类别 | 核心记录 |
|---|---|
| 最终任务性能 | U150 全 Seen/Unseen SR、相对 SkillRL-style 的 ΔSR、六类 task SR；各臂全部报告 |
| 学习与单次编辑 | 每 5 轮 Seen monitor；同当前 policy 的 gate 前后 ΔSR、repair / regression 数量与比例、接受率 |
| 编辑行为 | 机会数、提案/接受的 ADD/MODIFY/DELETE/MERGE 次数；每次候选数、实际触及/新增/删除技能数、mutation units、NOOP、低支持弃权、库大小/版本谱系 |
| 稳定性 | rejection 与 rollback 分列；失败、未完成阶段、invalid-action rate、episode 步数 |
| 成本 | actor 输入/输出 tokens；本地 router 编码量/调用/缓存/时间；o3 输入/输出/可得 reasoning tokens 与 API 次数/延迟；readout 前向 tokens/次数/时间；总 wall-clock 与可核算费用 |
| 配对统计 | 相同评估 game/seed 下的分支差值及 game-cluster bootstrap 区间；单 RL seed 的限制单列，不把解码重复当独立训练 seed |

缺失 usage、失败/未对账请求不能按零计费；本地计算不是 API 费用，但也不是零成本。总 wall-clock、跨臂配对区间等应从保存的时间/逐 game 记录统一汇总，不能把字段存在写成已有实测结果。

Phase3 只保留预测、必要轨迹、编辑/门控、版本与成本记录；不做 Phase2 的 O/P/N gold 效用续跑，不落盘全词表张量或 hidden activations。因此这轮直接验证的是**下游闭环收益**，不是重新估计逐技能 ΔM。

## 5. 开跑前尚未闭合的事项（本次不修改）

1. **近期 Phase1–2 改动未自动迁移。** 当前 Phase3 仍调用原 `phase2.direction.token_signals`、采用负部 D/原始 C 和 20/4/8 支持门槛；与近期稳定数值实现、全首调用评估及新增 reward 变式不同。须另行冻结最终指标、数值版本与支持规则，不能按 seed404 赢家静默替换。
2. **尚无纯幅度闭环臂。** 现有 D、−P、+C 都属于 reward 信号族。四臂可比较失败驱动与 readout 驱动，但不足以支持“Phase3 相对 magnitude-only 有增益”；若要该结论，需要另行确认并实际运行同预算的 `||Hδ||` 等幅度臂。
3. **计算后端尚有差距。** 当前 Phase3 training/evaluate 路径仍为 HF/SDPA，不能把本机 Phase1–2 的 vLLM 八卡验收当作 Phase3 已迁移。若统一改为 vLLM，需四臂一致并重新做对应验收；本摘要不承诺运行耗时。
4. **运行清单仍需冻结：**gate 每 task 的 game 数、容差、eval seeds、readout parity 容差、每臂 editor/API 与本地 router 总限额、实际 token/证据 cap、GPU/磁盘限额。当前模板有空值，不能直接作为正式实验已就绪证明。
5. **证据边界：**本地仅有工程测试/历史发布记录，尚无四分支 Phase3 结果；o3 网关真实编辑兼容性及新服务器完整闭环仍需验收。当前四臂只登记一个 RL seed=404，不能宣称跨训练 seed 稳健收益。

## 来源与启动入口

- [完整设置](SkillRL/docs/phase3/SETTING.md)、[启动说明](SkillRL/docs/phase3/START.md)、[工程验收边界](SkillRL/docs/phase3/VALIDATION.md)。
- [冻结比较配置](SkillRL/configs/phase3_setting_embedding_v1.json)、[待填写运行模板](SkillRL/configs/phase3_runtime_template.json)。
- 本地实际核对：[相邻窗口调度](SkillRL/phase3/run.py)、[训练](SkillRL/phase3/training.py)、[读出/支持](SkillRL/phase3/readout.py)、[编辑输入](SkillRL/phase3/editor.py)、[配对门控](SkillRL/phase3/evolution.py)、[记录汇总](SkillRL/phase3/report.py)。
- 历史目标仓库：[Euclider/SkillScope-phase3](https://github.com/Euclider/SkillScope-phase3)。本次没有联网核对远端最新内容、上传、启动实验或修改实现；以本地核对日期为设置快照。
