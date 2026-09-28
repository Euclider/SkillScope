# Phase3 最近更新前训练证据协议 v12（2026-09-27 UTC）

> 历史提案，未用于正式编辑：用户随后选择
> [v13 同源旧策略批次协议](EDITOR_OLD_POLICY_BATCH_EVIDENCE_V13.md)。
> v12 接续器在触发前停止；保留此页作为决策轨迹，不视作执行设置。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: execution protocol amendment and mixed-protocol audit
- Origin Date: 2026-09-27
- Verification Status: offline tests passed; resumed live U10 editing and six-arm outcomes pending
- Version Label: phase3_latest_preoptimizer_training_failures_v12

## 冻结规则与时间线

这是 v11 的前瞻性修订，不改写 v11 事件或历史证据。每个五更新窗口的
readout 始终用窗口**首批**训练数据的固定状态和优势：U0→U5 用 U1
更新前、U0 策略采样的批次；U5→U10 用 U6 更新前、U5 策略采样的
批次。两端策略在这批状态上重新前向评分，窗口内技能库固定。

编辑器则使用窗口**末次更新前**最近的完整失败训练轨迹：U5 的训练
轨迹由 U4 策略采集、在 U5 优化前封存；U10 的训练轨迹由 U9 策略采集、
在 U10 优化前封存。它们不是 readout 的首批固定状态，亦不是由窗口终点
U5/U10 策略采集的 post-update utility 标签。读取这些轨迹不增加 rollout。
两类证据的策略代次不同，论文必须分别说明，不能称编辑证据也严格来自
窗口起点策略。

训练内置的 U5/U10 `valid_seen` 是**末次优化之后**的定期 Seen 性能
监测（`test_freq=5`）；不参与梯度，也不作为 readout、候选池或编辑器
输入。之后 U6/U11 的训练轨迹才分别由 U5/U10 策略采集；本协议不为
当前窗口等待或使用它们。独立 paired Seen gate 仅验证编辑候选是否接受，
Unseen 只用于预设里程碑评价，均不回流到本窗 readout 排序。

## 六臂共同证据规则

1. 仅加载 `episodes/u{end}/train`，要求 `global_update=end`、训练 game、
   当前 bank 及技能版本一致。训练成功轨迹不入编辑失败池。代码即使收到
   同号 `valid_seen` 或较早训练轨迹也拒绝作为 `evolve` 输入。
2. `skillrl_failure` 适配基线把当前活动库的**全部正文**和该批**全部完整
   失败训练轨迹**交给共享编辑器；不做技能候选排序。仍须注明这是 SkillRL-
   style 适配基线，不等同官方原样实现。
3. 五个 readout 臂在“有自然 readout 支持”与“本批失败轨迹实际调用过”
   的交集中，按各自预登记分数取 top-5；再反选调用至少一个所选技能的
   **全部完整失败轨迹**。编辑器只见 top-5 对应技能的完整正文及这些轨迹。
   不施加历史 8 条轨迹或本地 token 截断；保留网关实际 usage、请求账本
   与 API 上限。每窗最多 3 mutation units；ADD/MODIFY/DELETE/MERGE/
   NOOP、共同 Seen paired gate 与拒绝语义均不变。
4. `source-v12.json` 明示证据更新号、采样策略更新号、来源 split 及
   `post_update_validation_outcomes_read=false`；`selection-v12.json`、
   `evidence-v12.json`、`editor_evidence-v12.json` 和事件 `complete.json`
   按版本独立封存，不覆盖 v11 文件或报告。

## 已运行数据的边界

当前 v7 队列首臂 `readout_d` 的 U5 编辑已按 v11（U5 更新后 Seen 验证
失败）执行一次；候选经 gate 拒绝，库未改变。用户确认**保留该进度**，
故不追溯重做 U5。首臂 U5→U10 起采用 v12，其 U20 汇总必须标
`mixed_editor_protocols=true` 并列出各窗版本。其他五臂若从头执行，
使用 v12 全程；因此现有六臂结果不能宣称所有编辑机会证据协议完全一致。
若未来需要严格的同协议确认性比较，须另开全部六臂 v12 的独立 cohort；
不得将首臂 U5 v11 悄悄解释成 v12，也不得依据阶段结果补选规则。

本修订由用户根据论文方法的时间顺序提出；v11 U5 gate 结果已知，
因此它不是完全事前冻结的首窗比较。报告应标注这一选择时间和首臂
混合协议，避免将后续差异归因于单一 readout 排序机制。
