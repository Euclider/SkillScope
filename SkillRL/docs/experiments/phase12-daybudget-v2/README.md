# Phase1–2：15–30 小时预算版（seed 404）

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan/run
- Origin Date: 2026-09-18
- Verification Status: UNVERIFIED
- Version Label: phase12_daybudget_v2_optimizer_offload

## 当前状态与授权

这是新预算版设置及工程验证记录，不是实验结果论文稿，也不是正式执行许可。用户确认停止旧 150 更新任务并保留证据，确认 5 更新单窗口/全量 seen-unseen 性能/缩小 Phase2 效用规模，随后将时间目标调整为 **15–30 小时**。本方案采用 30 小时硬上限，准入预测上界不超过 27 小时，留 3 小时余量；达到时限仅停止并标记未完成，不宣称跑完整流程。

旧任务于 2026-09-18 06:04:49 UTC 确认停止，完成更新 **0/150**，完整 rollout batch 为 0；757 个本地新路由决策、265 次缓存命中保留。停止回执在 `artifacts/phase12/skillnet37-qwen35-embed06-s404-8gpu-v1/user-stop-receipt-20260918.json`。不重启该 root，不删除旧日志/缓存/初始 HF 模型。

v1 的 batch-2 八卡预检在长 prompt 生成阶段 OOM。用户获知失败后，另行批准开启 optimizer-state CPU offload 并做**一次**新预检，不启动正式 RL。v1 配置、preparation、失败日志、约 50 GiB 合成 checkpoint 保留；v2 新增 offload，不覆盖 v1。最终预检状态见本文追加验收段及独立输出目录。

早先 `configs/phase12_day_budget_proposal_v1.json` 是未执行的 24 小时/128 输出 token 提案；它已被用户确认的 15–30 小时、**512 输出 token**方案取代，不可用作正式设置。

## 冻结的研究设置

| 项目 | 本预算版 |
|---|---|
| Policy / seed | 本地 Qwen3.5-4B / 404 |
| 训练算法与采样 | GRPO、lr=1e-6、16 games × 8 trajectories/更新；完整六类 train pool 采样，不是穷举全部训练游戏 |
| 更新与窗口 | 5 次 RL 更新；固定 U0→U5；每 5 更新验证/保存；不按结果挑窗口 |
| 初始 bank | 冻结 SkillNet-37，全程完整候选库，不按 game/task 类别过滤 |
| Router | 冻结 Qwen3-Embedding-0.6B、逐状态 global top-1；独立于训练 policy；不训练 router、无付费 API |
| 轨迹长度 | 最多 50 环境步；prompt 上限 4096、每步输出上限 512；不静默截断/丢游戏 |
| 优化器 batch | PPO minibatch 128；optimizer microbatch/GPU=1；八卡原生 GPU Adam |
| 生成与内存候选配置 | rollout microbatch/GPU=2；v2 在更新之间将 optimizer states 卸载到 CPU；工程验收不等于正式准入 |
| 完整性能评估 | U0、U5 均遍历 140 seen + 134 unseen games；performance seed=61001；seen/unseen 分别报告 |
| Phase2 源轨迹 | 仅 valid_unseen，134 games，anchor source seed=61011 |
| Phase2 技能/锚点 | 最多 12 个自然支持技能、每技能最多 12 anchors；不足不补强制调用 |
| 效用重复 | 1 evidence seed=62011；2 gold seeds=63011,63021；U0/U5 × ORIGINAL/PLACEBO/NULL |
| 预测与分数 | 保留 gated D 主分数与已有 C/P/对照记录；窗口预测先锁定，再开启目标端点 gold；不按标签改方向 |
| 声明范围 | 单训练 seed、单固定窗口、预算受限验证；不是 150 更新收敛结果，不证明跨 seed 稳定性 |

自然支持门槛仍为 anchor occurrence≥30、distinct games≥10；训练 readout 门槛仍为非零优势决策≥20、games≥4、trajectories≥8。超过 12 个 eligible 技能时，按 `SHA256('daybudget-s404:' + skill_id)` 排序选前 12，tie 用 ID；不按 D、reward、gold 或显著性择优。`budget_not_selected` 与 `insufficient_natural_support` 分开记录。全库路由不等于每个技能都有充分支持或因果效用标签。

## 工程变更与可追溯性

- 新 router 版本 `skillrl-embedding-state-batch-top1-v1`：同一批当前状态去重并批量编码，CPU 8 intra-op threads，仍 FP32，候选文本、权重、revision、状态输入及逐 query dot scoring 保持原定义；线程/RNG 状态恢复。批次并行可能有浮点差异，因此使用新协议、新缓存，不宣称所有状态逐 bit 等价。
- 八卡 performance/anchor job 分片，按确定性分区执行，合并时核验所有 job 的精确并集，缺片/重复不得报告全量完成。仍是 HF 推理，不是 vLLM，未引入新推理依赖。Phase2 O/P/N 和 measurement 沿用八分片。
- 可选 `rollout_progress/uNNNN/step-NNNN.json` 在一批环境步完成后落盘，记录真实动作、reward、response token 和关联 ID；完整 GRPO group 前 advantage 标为 pending，不编造零值。完整轨迹、优化器 batch、advantage、更新前后概率等原有记录继续启用。
- 所有 budget 执行入口检查相同准入证明及绝对截止时间。只有初始 supervisor 准入要求剩余时间覆盖整个预计流程；后续阶段不会错误地重复预留整次流程时间。超时/失败终止所属子进程并保留未完成记录，不自动重试。
- `optimizer_offload=true` 使用框架原生 GPU Adam：更新前将 states 回载 GPU，更新后/恢复后放回 CPU；没有启用 legacy `PHASE2_CPU_ADAM`。Reference 参数 offload 原来已经开启；actor 参数 offload 仍关闭。

主要代码：`agent_system/memory/skillrl_embedding_batch_router.py`、`router_cache.py`、`skillnet_runtime.py`、`env_manager.py`、`skillnet_cohort/day_budget.py`、`prepare.py`、`common.py`、`runtime.py`、`training.py`、`segmented_training.py`、`evaluate.py`、`support.py`、`run.py`、`gpu_preflight.py`、`phase2/capture.py`、`multi_turn_rollout/rollout_loop.py`。

这些是本地未提交改动。本轮没有提交、回滚、清理或推送 GitHub。已发布 Phase3 main 仍对应之前的 `674dd36a54a83c5061af43f0f80b2ee3da924468`；不能声称此预算版/批量 router 已发布，Phase3 编辑器仍为 o3。

## 规模、容量和准入条件

| 工作量 | 上界/固定量 |
|---|---:|
| RL 训练轨迹 | 5×16×8 = 640 |
| 原生 seen 监测轨迹保守上界 | 128 |
| U0/U5 全量性能 episodes | 2×(140+134) = 548 |
| unseen anchor-source episodes | 134 |
| Phase2 O/P/N 续跑上界 | 12×12×2 endpoints×3 arms×3 seeds = 2592 |
| 按每条最多 50 步计本地 router 调用上界 | 202100 |

以上是任务数量，不是实测时长。只允许新 cohort 固定 5 更新窗口计算并封存紧凑结果后，回收有逐行可再生证明的全词表临时张量。原始轨迹、batch、报告、checkpoint、旧实验及失败预检证据不清理。

FP32 old+new 全词表记录每个 response token 至少 `248320×4×2=1,986,560` bytes；10 万 response tokens 约 **185 GiB**，尚未加 metadata/复制峰值。只改 5 次更新或加硬超时，不能证明容量够用。v1 失败预检后可用磁盘约 779 GiB（06:37 UTC），还要保留至少 100 GiB 空闲、检查点 reserve；760 GiB run cap 不是额外可用空间。v2 预检会再占用一个新合成 checkpoint，准入前必须重新检查实际空间。

正式启动需要绑定本 preparation 的独立 execution permit 和实测准入证明，至少包括八卡 microbatch 验收、exact capture、真实八卡评估、完整流程时间上界≤27h、峰值存储适配实际可用空间。当前 preparation 的 `execution.approved=false`，没有有效预算准入证书；正式 run root 尚未创建/启动。禁止复用旧 150 更新 permit，禁止仅修改 JSON 的 approved 字段绕过验收。

## 本次证据

1. [旧状态 replay 的 router 标定](../../../../daybudget-router-calibration-20260918-v1/report.json)：32/32 选技相同，32 query 总耗时 19.6797 s（0.6150 s/query），这些旧状态账本耗时合计 66.4127 s，观察比值 3.37×。冷启动/建索引另为 17.979 s。这是短样本路由测量，既非全流程加速保证，也非准确率/覆盖率提升证据。只读旧 SQLite，SHA 未变。
2. [v1 GPU 失败审计](../../../../daybudget-gpu-preflight-20260918-v1/failure-audit.json)：8/8 完成合成前后向和 optimizer 阶段，但 batch-2、4096 prompt 的生成 prefill 全部 OOM；没有 complete/rank PASS。不能用此前短 prompt 的旧预检代替此边界验收。
3. [v2 用户批准的新预检计划](../../../../daybudget-gpu-preflight-20260918-v2/execution-plan.md)：新目录、最长 1800 s、生成上限 512、native offload 和回载验证，无真实 ALFWorld 或 API 调用；结果另追加。
4. 第一轮广泛离线回归 434 passed / 78.38 s，[XML](../phase12-daybudget-v1/offline-tests-v1.xml)；v2/offload/截止保护相关回归 151 passed / 74.97 s，[XML](../phase12-daybudget-v1/offload-offline-tests-v2.xml)。后者包括受影响 cohort/router/Phase3 embedding 用例，不冒称硬件验收。

离线命令曾因指定不存在的 test 文件而退出 4、未运行测试；空结果 XML `offload-unit-tests-v2.xml` 保留，修正为实际测试目录后得到上述 151 PASS。这不算 GPU 预检重跑，也不隐去失败命令。

## 冻结身份

| 文件/对象 | SHA-256 |
|---|---|
| v2 budget profile | `c2f1ecf1b7c22dc06db91a339209c36c3380d3b1e12935f576545c90f510ae6d` |
| v2 preparation manifest | `174839baeeab0b5270e699daaf4072107a682c94447d9dce895ba4aa5d7b767b` |
| v2 spec | `fb67c10f973577999af408b2484d27fe3416c4f776a1e6d7a41b620cb1bae6d7` |
| batch router profile | `091b514862df2358b06cae7e5db71ce5233d134d8ca2dfda2f220490ba28fb4b` |
| batch router CPU protocol | `fd36c1ad1988a05d0bb26a0a5a18ea298f7a19add42c70d3b20f2182e053d308` |
| SkillNet-37 bank manifest | `0767ff7578b1e997119a36b5f636fec6ebc1d0ac600ffb40b1e599065b7fe514` |
| 旧已停止 cohort router SQLite | `9214ce320e1ba1b29752a9c771fa3f1d69c48d13a0efb3cbb60190e5b4fde7ee` |

准备入口为本目录的 `preparation-s404-8gpu/manifest.json`；配置为 `configs/phase12_30h_budget_v2.json`。v1 preparation 仍完整保留在相邻 `phase12-daybudget-v1/`。

## 06:50 UTC 追加：v2 预检失败，正式执行仍锁定

本次用户批准的 v2 预检只运行一次，启动 06:43:27 UTC，06:46:17 UTC 失败，父命令退出 1；最后日志写入距启动约 173.89 s。完整 [失败审计](../../../../daybudget-gpu-preflight-20260918-v2/failure-audit.json)、`run.log`、各 rank 阶段记录和新合成 checkpoint 均保留。没有 v2 `complete.json` 或 `rank-*.json` PASS。

| 阶段 | 实际结果 |
|---|---|
| actor/reference 初始化、4096+512 概率前向 | 8/8 通过 |
| 第一次合成 backward/GPU Adam，states 卸载 | 8/8 通过，states 确认为 CPU |
| native 参数/optimizer/RNG 精确恢复 | 8/8 通过，恢复后 states 确认为 CPU |
| batch-2、4096 prompt、512 输出上限生成 | 8/8 通过；16 条响应中 15 条生成到 512，一条在 57 EOS |
| 生成后第二次合成更新 | 8/8 在 backward OOM；并未完成第二次 optimizer step |

生成阶段每 rank 19.87–20.90 s，合计实际 7737 tokens，以最慢 rank 的阶段耗时计算约 370 tokens/s（八卡合计）。这是合成长输入的组件吞吐，不是 ALFWorld episode 吞吐，不能据此承诺 15–30h 整体完成。阶段记录在生成结束、释放临时缓存后采样，不能把其中约 2 GiB allocated 当生成峰值显存。

OOM 日志为 `dp_actor.py:485 loss.backward()` 申请约 7.83 GiB，当时各卡可用约 5.49–5.91 GiB。只读代码确认 `fsdp_workers.py:update_actor` 在整个前后向**之前**回载 Adam states；HF actor 还因既有兼容分支使用 `auto_wrap_policy=None`，不能简单假定已逐 transformer block 细粒度分片。第一步 states 尚未初始化，与第二步显存条件不同，因此旧“一次更新 PASS”不足以保证持续训练。

建议下一步（**未实现、未重测**）：评估将 states 回载延后至 `optimizer.step()` 前、每次 step 后卸载，避开 backward 的峰值；保留 GPU Adam 和既定长度/批次。不直接切 CPU Adam、不缩短长度、不静默退回 microbatch=1，也不把生成通过改写为完整预检 PASS。修复仍需用户复核及新的连续更新验收，结论暂为假设。

06:46:56 UTC 八卡均已释放；约 730 GiB 可用磁盘。没有新正式 Phase1–2/Phase3 运行，无付费调用，无自动重试。最终广泛离线回归 **435 passed / 81.49 s**，见 [XML](offline-tests-v2.xml)，SHA256 `c787772d9aed083ae364d74bfeb8a73d397a27cf0806b18682f6c21a707a8835`。当前 [准入审计](admission-not-ready-v2.json) 明确 `NOT_ADMITTED`；不是有效的 PASS 准入文件。

## 后续用户追加：高吞吐 RL 框架优先

用户随后要求核查 vLLM 并尽量选择最快的 RL 框架。本地事实是 verl/FSDP1/HF，无 vLLM 或 SGLang 安装。已完成只读/官方资料选型，建议优先考虑 **verl + FSDP2 + vLLM**，先验收 TP1×8 的批量 rollout/评估，保持记录协议；此结论是优先候选，不是本机最快的实测证明。详见 [后端选型与迁移边界](backend-selection-20260918.md)。尚未迁移/安装或追加预检，下一步不能只照上段建议自动继续旧 HF 修复。

用户又询问在验证idea前提下降低rollout数量。新增 [缩量提案](rollout-budget-proposal-20260918.md)：优先建议训练16×8→8×8，保留5轮窗口与原评估规模；anchor12→8作为更紧但效用估计更不确定的备选。两者均未确认/实施，当前frozen profile仍为16×8、12 anchors，不得混称新设置已生效。
