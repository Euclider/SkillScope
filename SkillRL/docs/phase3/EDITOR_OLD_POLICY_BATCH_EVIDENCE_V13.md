# Phase3 同源旧策略批次编辑证据协议 v13（2026-09-27 UTC）

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: user-confirmed execution protocol amendment
- Origin Date: 2026-09-27
- Verification Status: offline tests passed; resumed live editing and six-arm outcomes pending
- Version Label: phase3_same_old_policy_batch_as_readout_v13

## 核心时间线

用户最终确认：每个五更新窗口的编辑失败轨迹和 readout 从**同一首批
旧策略训练数据**取得。U0→U5 使用 U1 更新前由 U0 策略采样的批次；
U5→U10 使用 U6 更新前由 U5 策略采样的批次，后续类推。
readout 对该批次保留的固定状态、动作 token 和优势做两端策略前向评分；
编辑器仅看该批次中失败轨迹的完整环境记录。U2–U5 / U7–U10
轨迹、终点 `valid_seen` 和新策略后续训练轨迹都不进入当前窗口的
候选池或编辑器输入。窗口内技能库固定，readout 的两端策略和独立
paired utility 测量定义不变。

这满足论文方法中“诊断与编辑证据复用旧策略训练批次”的时间顺序：
虽已训练到 U5/U10，编辑输入不依赖终点策略产生的新增 rollout。
U5/U10 的训练内置 `valid_seen` 仍在优化后按每五步进行 Seen 性能
监测，不参与梯度、readout、候选筛选或编辑器。独立 paired Seen
gate 只用于接受/拒绝编辑候选；Unseen 只用于预设里程碑评估。

## 六臂统一选择与封存

1. 只读取 `episodes/u{start+1}/train` 的完整 128 条首批轨迹，
   校验 `global_update=start+1`、训练 game、bank 和技能版本。
   在编辑前还校验这 128 个 trajectory ID 与 readout 方向批次元数据
   的 trajectory ID 集合完全一致；当前首臂 U1 和 U6 已分别核验通过。
   其中 `success=false` 的轨迹形成共同编辑失败池；成功轨迹仍在
   readout 训练批次中，但不作为失败驱动编辑证据。不得用后一轮
   更新的训练轨迹或验证轨迹填补当前窗口。
2. 失败驱动适配基线向共享 gpt-5.5 编辑器提供完整当前 bank 正文
   和该失败池的全部完整轨迹；不做 skill 候选排序。它是 SkillRL-style
   适配基线，非官方逐字复现。
3. 五个 readout 臂各按预登记分数，在自然 readout 支持与这批失败
   轨迹实际调用过的技能交集中取 top-5，然后反选调用至少一个候选
   的全部完整失败轨迹。编辑器只见候选技能正文和反选轨迹。无历史
   8 条上限和本地 token 截断；每窗仍最多 3 mutation units，保留
   API 调用/usage、NOOP、配对 gate 与拒绝账本。
4. 新事件写 `source-v13.json`、`selection-v13.json`、`evidence-v13.json`、
   `editor_evidence-v13.json`，明确 `readout_batch_update` 与
   `evidence_global_update` 相同、`sampling_policy_update=start`、
   `post_update_validation_outcomes_read=false`。旧 v11/v12 文件和
   训练/报告不覆盖；v12 提案尚未用于正式编辑。

## 已运行 cohort 的比较边界

v7 首臂 `readout_d` 的 U5 编辑已经按 v11 使用 U5 终点 Seen 验证
失败轨迹；候选经配对 gate 拒绝，bank 未改变。用户明确选择保留进度，
不回滚、不重跑 U5。U10 起及其它尚未启动的臂改用 v13。
因此首臂的 U20 是混合编辑证据协议；`running_metrics.json` 显式
列出逐版本窗口数并置 `mixed_editor_protocols=true`。跨六臂对比可作
探索性结果，但不能声称六臂**每次**编辑机会的输入协议完全一致，
尤其不能把首臂 U5 的差异纯归因于 readout 排序。严格同协议的
确认性比较须另设全部六臂 v13 cohort，且不据当前结果调参。

v13 是在已见首臂 U5 gate 结果后由用户确认的时间顺序修订，
其生效点和混合协议必须在论文附录披露。
