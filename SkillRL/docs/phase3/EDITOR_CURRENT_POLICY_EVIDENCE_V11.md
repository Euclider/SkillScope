# Phase3 终点策略编辑证据协议（2026-09-27 UTC）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: execution protocol amendment
- Origin Date: 2026-09-27
- Verification Status: offline code tests and U5 input audit passed; external editor/gate pending
- Version Label: phase3_terminal_failed_current_policy_v11

## 动机与证据边界

原实现把窗口 U1–U5 的所有训练及验证轨迹（首窗共 704 条）作为 U5 编辑
证据来源。其中 U1–U5 训练轨迹由各次更新前的策略产生，不能都当成 U5
策略的当前失败。SkillRL 官方默认在每次验证时基于**当次验证结果**收集
失败轨迹；本实验适配版的编辑证据也改为当前端点的当次 Seen 验证失败。

**不要混淆 readout 与编辑证据：** U0→U5 readout 仍只使用 U1 更新前、
由 U0 策略采样的真实优化批次，在固定状态上比较 U0/U5 的 action
preferences；U2–U5 轨迹不参与 readout。U5 编辑只消费 U5 更新之后
生成的当次 `valid_seen` 失败轨迹。未来 U5→U10 窗口类推：U6 首批作
readout 固定状态，U10 当次失败验证作编辑证据。

## 六臂统一规则

1. 保持 37 技能初始库与逐状态本地 router；运行时 router 始终检索当前
   **完整活动库**。每窗最多一次 gpt-5.5 编辑机会，3 mutation units，
   ADD/MODIFY/DELETE/MERGE/NOOP 与配对 Seen gate 不变。
2. 定义编辑轨迹池为该窗口**终点更新号**的 `valid_seen` 完整失败轨迹，
   game 必须在预先划定的 evidence Seen 集，bank 版本与窗口一致；不使用
   train、旧更新、Unseen、gate 或编辑后的轨迹。不重新采集环境 rollout。
3. `skillrl_failure` 适配基线不筛技能，编辑器看到完整活动库正文及该池
   **全部**完整失败轨迹。它不是官方 SkillRL 的逐字复现：官方编辑器主要
   见全库标题与少量截短失败轨迹、主要新增技能；此处为共享编辑器的
   全库/失败驱动适配版。
4. 五个 readout 臂使用各自预登记分数，但排序池取“本窗有自然 readout
   支持”与“终点失败轨迹实际调用过”两者交集；各自取 top‑5。选好技能后，
   反选该失败池中调用过至少一个 top‑5 技能的**全部**完整轨迹，不按
   调用次数排序或施加 8 条上限。编辑器只看到本臂 top‑5 正文及这些轨迹；
   其他活动技能正文不暴露。未在终点失败轨迹中出现的高分技能记录排除
   原因，不悄悄改分数或冒充有编辑证据。
5. 本地 editor 输入 token 拦截和轨迹截断均取消；仍记录本地估算、
   网关实际 usage、API 次数、输出上限、请求 hash、证据/候选/库版本、
   gate 结果。网关可能有独立上下文限制；失败按原账本 fail-closed，
   不自动重试。旧 runtime `max_evidence_trajectories=8` 保留为历史冻结
   配置字段但在 v11 编辑证据中不生效；新 `source-v11.json` 明确记载
   `trajectory_cap=null`。

## U5 启动前只读核验

- U1 首批方向张量回执：5,352 个决策行、71,148 loss tokens；正式
  readout 5,351 个有效决策。不是 704 条累积轨迹一起算 readout。
- 704 条累积轨迹里，U5 当次 Seen 验证 64 条，失败 31 条；失败轨迹
  涉及 18 个技能，其中 16 个有自然 readout 支持。
- `readout_d` 在这 16 个技能中选出 top‑5，反选 21 条完整失败轨迹、
  1,050 个环境步；本地 gpt-5.5 输入估算 164,581 tokens。
- 同一首臂数据若按适配失败基线输入规则，31 条完整失败轨迹、
  1,550 个环境步及 37 个技能正文，估算 276,627 tokens。**这不是
  `skillrl_failure` 独立 RL 臂的实测费用**；后续费用比较必须用各臂
  自己的 API usage，并说明输入范围不同是预设机制的一部分。
- 旧 U5 `source.json`、top‑3 `selection.json`、`evidence.json`、
  U4/U5 checkpoint、readout、失败日志都不覆盖。v11 写独立
  `source-v11.json`、`selection-v11.json`、`evidence-v11.json` 与
  `editor_evidence-v11.json`；截至本协议写入时编辑账本为 0 次尝试。

此次变更是基于策略时效性与用户明确的协议确认，在任何编辑请求或
Phase3 闭环结果出现前冻结；应在论文附录披露，不把工程修正写成
事前最优参数搜索。
