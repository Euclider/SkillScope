## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run (explicit user-authorized code/protocol amendment)
- Origin Date: 2026-09-20 UTC
- Verification Status: UNVERIFIED (scientific findings); CPU engineering checks recorded separately
- Version Label: all_u0_first_calls_v1

# Phase1–2：全技能、全部首调用锚点评估

## 最终授权与顺序

用户取消自然支持/训练支持的数量门槛，要求补评 seed404 并替换现有分析报告，
但必须等 seed505 的五轮 RL 正常完成，不中途打断。
讨论中的“每次调用均作锚点”**未运行**；用户最终改为每条轨迹、每个 skill 只取第一次调用。
本版本执行最终口径。后续 505/606 使用同一规则，不根据 404 结果另行调整。

当前旧队列 105 个 runtime 源码已冻结，不能直接热改，否则后续 seed 准入失败。
因此以新增 `skillnet_cohort/first_calls_*.py` 实现独立评估覆盖扩展，不改旧 RL 入口。
执行顺序：

1. 等 seed505 RL 子进程正常退出，核对五轮 metrics、八 rank 完整 U5 检查点及成功退出记录。
2. 只暂缓旧队列和 seed505 的两个调度进程。若训练退出与轮询间隙已经启动导出/评估子进程，等它自然结束，绝不信号中断。
3. 八卡空闲后，优先补齐 seed404，完整验证后发布新报告；无论成功或出错，恢复旧调度进程。
4. 旧队列继续完成 505/606 原流程；其原生评估和读出可复用。队列正常结束后，按相同规则补齐 505、606，最后更新当前 cohort 汇总。

这是同一批已登记 seed 的评估扩展，不新开 RL、不重跑已有 RL/完整性能评估/锚点来源轨迹。
中间可能同时存在已扩展的 404 和旧口径的 505/606 报告；**不得将这个中间状态解释为同协议跨 seed 对比**。
以 `deferred-complete.json` 的全三 seed 完成记录为同口径汇总完成标志。

## 锚点、覆盖与缺失值

来源范围维持原协议：U0、`valid_unseen` 全 134 个 game 的已登记 anchor-source 轨迹。
对每个 `(source_trajectory_id, skill_id)` 只保留首次自然选中处；完整保留所有这样的锚点。
不再要求 30 条轨迹、10 个 game，不设技能上限，也不再每技能抽 12 个锚点。
同轨迹后续调用计入原始调用频次及阶段统计，不重复产生效用样本。

seed404 实际来源：25/37 个 skill、5,558 次自然调用、404 个首调用锚点；其余 5,154 次为后续调用。
固定两端点 U0/U5 × O/P/N × (1 evidence seed + 2 gold seed)，共 7,272 条续跑。
旧 1,080 条续跑在原模型、router、payload、placebo、锚点状态、prefix、采样参数和 continuation seed
全部匹配、原 seal/hash 验证通过后复制引用并留 provenance；**实际只需新增 6,192 条**。
保留每条续跑的完整 action/observation/skill 记录，不将只写汇总当作全评估。

`support/coverage.csv` 为完整 37 × 5 phase 记录：原始调用数、源轨迹数、game 数、首调用锚点数、
后续调用数、实际训练决策/token 数、非零 advantage 数及其 game/轨迹覆盖。
覆盖少不再排除；未自然出现的技能不强行注入。没有 unseen 首调用锚点时效用为 NA，
没有真实 on-batch 读出输入时分数为 NA，均记录原因，不能填成零。
实际404来源的25种技能中，18种有U0训练batch读出输入，另7种仍评效用、读出标NA。
有真实输入但 advantage 全零时，按公式得到零的量保留零，同时标记没有 reward-direction 支持。

## 数学与统计不混淆

本次首调用去重**只用于效用评估锚点**。C/P/D 仍基于第一轮实际训练 batch 的全部真实决策、
动作 token、loss mask、advantage；仅去除原存档 padding 的重复 decision ID，不删真实的后续决策。
不重复累加多个效用锚点的 advantage，也不把 readout 改成只看第一调用。
移除 20 非零决策 / 4 game / 8 轨迹的候选资格门槛；公式、固定排序方向、fidelity/direction gate、
噪声阈值、完整性检查不变。旧分片的精确信号复用，新覆盖技能只补缺失前向；分片首 witness
必要时重做单步前向并与旧数值逐值核对，不发生新环境采样。

干预仍是：回放 U0 固定动作前缀，在首调用处以及之后自然路由到该目标 skill 时应用 O/P/N。
它估计“从首调用起的目标 skill 剩余轨迹贡献”，不是只替换一次调用的孤立因果效应。
O/P/N 与 U0/U5 对每个锚点使用相同 continuation seed，预算/最大总步数/上下文长度均保持原值。

点估计先平均 paired continuation，再平均 source trajectory/game；区间沿用配对 game/continuation
bootstrap（10,000 次），端点和 arms 不拆配。只有一个 game 仍给点估计，但跨 game 区间为 NA，
方向记 uncertain。阶段、anchor、continuation seed、skill 都不是独立 RL seed。
同池 AP、AUROC、Spearman、Kendall、top-k/比例预算、0/5pp 阈值与 NULL 敏感性保持显式记录。
零下降事件 AP/Recall 未定义；D=0 不是稳定；排名关联不能替代方向分类或跨 seed 泛化结论。

seed404 的旧效用标签已经可见，因此明确标注为事后覆盖扩展，不能冒称新预登记确认性实验。
新评分阶段不读取新目录的 U5 gold，固定后才导入旧 U5 / 补评新 U5；这不抹去旧标签已存在这一事实。
505/606 沿用本轮提前冻结的同一覆盖规则，不根据其结果筛选技能、锚点或分数方向。

## 文件与保护

队列根为 `artifacts/phase12/skillnet37-independent-s404-505-606-v4`（Q）。
新入口为 `Q/all-first-calls-v1/plan.json`；404 新结果在其 `seed-404/`，后续在 `followup-s505/`、`followup-s606/`。
旧报告先按 SHA256 归档在 `archived-reports/`；只有完整评估、配对核验、报告、seal 均完成后，
才替换对应 seed 的 `reports/` 分析视图，之后统一替换 Q/reports/ 汇总。发布前若发现人工修改则停止，绝不覆盖。
旧 windows/sealed.json、轨迹、概率、模型、检查点、optimizer 记录、失败日志不改不删。
不覆盖历史 101/202/303 的根目录论文分析文档，不提交、不回滚、不推送 Git。

本版本无运行时间上限（沿用用户最新授权），不是新的完工时间承诺。仍保留 100 GiB 最低空闲、
80 GiB 后续检查点预留、760 GiB cohort 上限。404 按每条续跑 4 MiB + 8 GiB 辅助记录做容量准入，
共约 36.4 GiB；运行中每分钟重查磁盘，不能自动删旧文件或缩减锚点来绕过保护。
router 仍为冻结 SkillRL 0.6B embedding 逐状态 top-1，无外部 API 消耗。

## 只读状态与故障处理

- `deferred-status.json`：等待 RL / 排空已启动子任务 / 404 评估 / 等原队列 / 后续补评 / 完成。
- `deferred.log`、`assessment.log`、各 seed 新目录 `logs/`：失败原因保留，不自动重试。
- `paused-coordinators.json`、`handoff.json`、`coordinators-resumed.json`：交接与恢复审计。
- `seed-404/runtime-status.json`、`complete.json`、`report-publication.json`：实际计算与发布状态。

正常退出、SIGTERM 和 Python 异常都会恢复本轮暂缓的旧调度进程。若系统强杀 deferred supervisor
（SIGKILL/掉电不执行 finally），先依据 `paused-coordinators.json` 对照 `/proc/PID/stat` 的 start_ticks
和 cmdline hash 确认还是同一进程；确认新 assessment 已退出并无占用后，才对这两个**单独 PID**
发 SIGCONT，绝不能对训练 worker 或整个进程组发信号。此为故障人工处理说明，不自动重启失败计算。

## 工程验收

新增 CPU 测试涵盖首调用去重、原锚点语义、零/缺 advantage、真实 404 全覆盖、旧 1,080 续跑身份与
八分片重分配、无数量门槛但保留数学完整性、配对 bootstrap、可恢复报告发布、PID 复用防护、
训练未退出不得 signal、无 RL/export 的评估阶段顺序、C/P/D 与原公式逐值一致、旧105源码冻结。
实际 PASS 数及启动快照在后续工程验收记录中填写；测试通过不表示新实验科学结论已经验证。

## 服务器 UTC 16:10 验收／交接记录

- 专项初检31项PASS；完整七组740项PASS/145.04秒；最终定向32项PASS/37.55秒。
  记录为 `artifacts/engineering/first-calls-20260920-full-v2.xml` 和 `first-calls-20260920-final-v3.xml`。
- 首次完整回归739项通过、1项旧CPU测试因CUDA隐藏后显存日志接口不存在而失败，原XML保留。
  后续仅以 `tests.skillnet_cohort.first_calls_cpu_plugin` 在该测试中隔离显存日志；真实CPU optimizer/
  梯度/loss及全部断言保持执行，不修改冻结的训练源码，不使用GPU。
- 实际plan/protocol/授权、旧seal所有文件、episode语义、磁盘准入均CPU核验PASS。
  当前约404.2GiB空闲，新评估容量上界36.40625GiB，加80GiB检查点预留，仍满足100GiB最低空闲。
- plan SHA256：`e3521e2e66a02a5cc006905e8432d35e52a8c5b05b68b8610b8ea14e7cadfeb9`。
  原105源码不改，新增7源码另行冻结，共112项；旧7份seed404报告与各自归档SHA完全一致。
- 后台deferred supervisor PID **1259771**，已登记为 `WAITING_SEED505_RL`。
  原queue/supervisor/training PIDs **1201615/1207419/1207476** 均存活，未发暂停信号。
  seed505已落盘U1、正在U2优化阶段；新404评估未开始、旧报告未替换。
- 此后以动态 `deferred-status.json` 和阶段完成/发布记录为准，不拿本快照冒充后续实验完成。
  Agent结束主动盯守；后台交接、遥测、磁盘保护继续，无失败实验自动重试。
