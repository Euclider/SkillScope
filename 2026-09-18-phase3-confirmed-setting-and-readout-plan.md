# Phase3 打包与本机 Phase1–2 接续记录

本文件是新增留档，不覆盖 HANDOFF、技术附录、synthesis-v2 或历史报告。设置来自本轮用户确认；工程验收不等于研究效果验证。

## Phase3 冻结比较

四条独立累积RL分支：SkillRL-style failure-driven + shared editor、gated D主实验、−P完整消融、原始+C_upd完整消融。不能在一次同策略RL中冒充四条已分化skillbank的闭环比较。

共同Qwen3.5-4B B0、SkillNet-37初始库、RL seed404、全六类ALFWorld任务共享完整当前库；不合并旧SkillRL bank。GRPO、lr1e-6、16×8轨迹/轮、PPO minibatch128决策行、microbatch1、150轮上限、每5轮验证/编辑。独立gpt-5.4-mini router；o3 medium编辑器；用户指定网关 `/v1`，凭据仅环境注入。

固定5轮读出窗口：首轮真实优势/批次，start→end endpoint projection；不累加五个逐轮D。k≤3；当前内容版本支持至少20非零决策、4games、8trajectories；不足则弃权，不填零、不强行路由。新版本不继承旧分数。

编辑器输入完整当前库：失败臂使用失败轨迹；readout臂使用top-k及这些skill实际出现的观察/动作轨迹，可成功或失败。共用证据数上限10、token/call预算及输出schema，无额外summary LLM。ADD/MODIFY/DELETE/MERGE/NOOP；≤3 mutation units，2→1 merge=3。不可变snapshot、tombstone与版本绑定。

配对Seen gate与Seen证据games分离；同当前policy、同game×seed比较候选/旧库。候选拒绝不是rollback；没有自动接受后rollback，默认真实计数0。仅bank可恢复，不回滚policy/optimizer。Unseen只做预声明最终评估；U150报告完整140Seen/134Unseen与六任务拆分。Seen含开发使用过的games，不称全held-out。

指标包括SR、任务/episode/step/invalid-action、actor/router/editor tokens与未知usage、读出forward token/call/time、提案/接受操作、每次实际触及skill数、mutation units、库大小/版本、NOOP/弃权、gate ΔSR/repair/regression、rejection与rollback分别统计。各分支单RL seed404；game bootstrap不能替代多训练seed结论。

完整可迁移说明：`SkillRL/docs/phase3/START.md`；附录设置：`SETTING.md`；证据边界：`VALIDATION.md`。新服务器必须先冻结gate数量/容差、eval seeds、parity容差、API上限、GPU和磁盘限额。四分支本机不启动。

## 工程验证与发布边界

8×RTX5090原生合成验收全部PASS：4096+512前后向、Adam、参数/优化器/RNG逐值精确恢复、恢复后HF生成；峰值allocated29.442434GiB。只测每rank一个microbatch，不是完整128行批次、Ray+ALFWorld或真实o3连接证明。最新离线回归541通过。

32GiB适配为CPU分片初始化及逐参数native恢复，不降低权重/optimizer精度。TextWorld未序列化：每5轮block环境seed=404+16*i；policy/Adam/scheduler/RNG/dataloader原生恢复，所有分支一致。需披露block适配，不是连续原生SkillRL逐bit复现。

已获授权发布至唯一目标 `Euclider/SkillScope-phase3`。导出allowlist、逐文件hash、secret扫描、独立新导出目录；不操作原工作区Git index，不提交/回滚原文件，不含权重/运行数据/凭据。发布完成以独立receipt与远端逐blob核对为准，本文不是上传成功凭据。

## 本机 Phase1–2 新cohort

必须在Phase3上传验证成功后才启动。本机8卡、冻结37库、seed404、固定5轮窗口；沿用assets-v2，原manifest不改，新增执行permit绑定manifest SHA-256 `afc003d9619d12ac71d24d47c4a1dc2dee35d35a63e7199428efe8c8bca1baf5`。

费用解释：用户给出输入$0.75/M、输出$4.5/M及约100元人民币停止。采用8元/USD的保守**记账系数而非汇率行情**，总额度$12.50。并发请求先预留，返回usage后结算，未知usage继续占用预留；无自动重试或预算扩展。真实账单受网关计费影响，不伪称已核验实付。100元很可能无法完成首个5轮窗口；预算中断不当成全量完成。

窗口调度：B0导出→5轮训练/native checkpoint→端点导出→两split起点自然anchor与效用→匹配读出/预测锁定→端点O/P/N与完整Seen/Unseen性能→报告封存→下一窗口。两split预测都锁定后才打开对应target gold；native Seen训练监测仍按SkillRL每5轮执行。无自然支持则显式弃权；不能为覆盖率强迫skill调用。所有窗口和评价seed预先登记，不按结果挑选。

用户允许仅回收**这个新run**可再生的全词表临时张量。实现采取fail-closed：先完整配对效用/紧凑报告封存与hash验证，再要求每个待回收row具有保留batch/model/native-source的hash及逐值精确再生证明。没有证明就保留；每5轮保存并不自动证明中间策略可再生，禁止据此删除中间raw rows。当前没有自动构造逐row再生证明的工具，故默认保留这些rows，可能先触发磁盘保护停机。这是明确的存储限制，不声称已实现整段全词表自动无损回收。

保留全部原始轨迹、训练批次、optimizer/native checkpoint、报告及历史文件；不重跑旧实验。遇到budget、磁盘、上下文、服务或子进程失败时保存stopped记录，人工对账后再决定下一次attempt；不自动重开已经产生证据的RL block。
