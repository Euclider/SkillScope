# Phase2：U35→U40 首个 5-update 窗口与 Skill 排序评估

日期：2026-09-13。用户已确认启动。此为新实验，不覆盖 Phase1、旧 Phase2 或 U35 新支持基线。

2026-09-13 06:02（北京时间）后台流水线已启动，supervisor PID 为 `3413749`。U35 三卡恢复转换的 3 个 ranks 均已完成参数/Adam 校验；实际训练阶段与进度以 `status.json` 和 `logs/train-u36.log` 为准。

06:07 启动核查：GPU 4/5/6 已进入 U36 训练迭代；3 个 live restore audits 均 `checksums_match=true`，Adam step 从 `611` 恢复，未重置 optimizer。`Training Progress: 35/36` 表示进入 U36，不能解释为 U36 已完成。新 root 不含旧 U35 live-logit batch，因此启动时未额外声称完成 parent full-vocabulary parity；本次通过的是参数与 Adam 状态哈希校验。详见 `launch_verified.json`。

## 固定实验范围

- 从 Seed303 最新完整 U35 checkpoint 继续至 U40，共 5 个 global updates（U36–U40），不是重新初始化 optimizer，也不是新增独立 seed。
- 每次 8 个 clean train games × 4 rollouts = 32 条训练轨迹，合计计划 160 条。LR=1e-6、KL coefficient=0.01、optimizer global decision minibatch=32、microbatch=1；prompt 2048、response 64、episode 最多 30 步，沿用 GRPO/HF text-only 路径。
- 训练 sampler seeds 为 30336–30340；Bank、Router 与 payload 内容冻结，不按效用结果选择 checkpoint 或停止位置。
- 本批主窗口固定为 U35→U40，直接在同一窗口内对 Skill 排序；不需要拟合效用数值回归器。单步 U35→36、…、U39→40 的 P/C/D 等信号仍全部计算归档，但本批不为中间 U36–U39 额外生成 gold rollout。
- 窗口读出使用 U36 实际旧 policy batch 的 state、响应 tokens、advantage，与 U35/U40 的 ORIGINAL/PLACEBO endpoint log-probs 对齐。它是“累计交互变化对起始 reward direction 的投影”，不是各步 D 的相加，也不假定五次更新的 reward direction 始终不变。

## 基线与 gold

复用已完成的 [U35 新支持基线](/home/wangyifan/skill-RL/2026-09-12-phase2-ranking-evaluation-queue.md)：31 个 clean unseen games 的 124 条自然轨迹、4 个支持 Skill、每 Skill 50 anchors，共 200 anchors。Skill 为 `cle_003`、`cle_004`、`cle_006`、`gen_002`，包含非初始调用。

每个端点固定 ORIGINAL / token-template-matched PLACEBO / target-payload NULL 三臂；精确 replay 首次调用前 prefix，后续自由 rollout，重复选中目标 Skill 时持续应用干预。evidence seeds=51011/51021，gold seeds=52011/52021/52031/52041；多次 continuation 不作为独立 game/update 样本。

U35 的 3,600 条 suffix 以校验后的新索引和明确的 source provenance 导入，未重新运行环境；U40 计划新增 3,600 条 suffix。源轨迹 SHA-256、原始 ID、来源路径保存在 `baseline_import.json` 和每条导入记录中。模型通过只读路径别名复用，不复制已有 U35 FP32 权重。

执行顺序：每次真实训练与原始证据保存 → 每步 signals → U35→U40 window signals commitment → 锁定各指标直接排序分数 → U40 三臂 gold → 配对效用及排序分析。预测分数不得读取 U40 gold。

## 主要分析：排序，而非精确回归 ΔM

主效用为 `M_sem = success_ORIGINAL − success_PLACEBO`，主目标为效用下降 `−ΔM_sem`；无需正→负 sign flip。

- 比较 D、未门控 D、−P、centered/raw interaction norm、KL/JS、四层 activation norm，以及 ORIGINAL/PLACEBO 通用更新范数、旧效用低优先、随机期望排序。
- 同一窗口/context/phase 内，所有方法用相同的自然支持、完整观测候选池。unsupported 单元单独报告，不设为零效用或零风险。
- 固定预算 k=1/2，以及候选池前 25%/50%；重合预算合并为同一个实际 k，明确展示候选池大小。
- 主要统计为 Precision@k、Recall@k、下降量覆盖率；辅助为窗口内 Spearman/Kendall。不要求任何模型精确预测效用值，不将 MAE 作为主要成功标准。
- 并列分数使用并列组内随机选取的期望，不根据 gold 打破并列；无下降事件时 Recall 和覆盖率未定义，保留其计数。
- 点估计下降与 >5 pp 下降分别报告；`|ΔM|` 的变化定位为独立的次要目标。C/upd fidelity、gate coverage、原始 P 等读出一起锁定保存，便于解释。
- 阈值只改变 Precision/Recall 的事件标签；下降量覆盖率始终累计候选池内全部 `max(0, −ΔM)`，不会因阈值变动而事后改变 gain 定义。
- 各 Skill ΔM 保留 10,000 次 paired game/continuation bootstrap CI。首个窗口的 top-k 统计是描述性结果，不宣称排序指标显著性、多个更新窗口或跨 seed 泛化。全阶段 all 与 initial/early/middle/late 分开，不拼成独立样本。

## 存储与恢复

已按本次授权删除 6 个可重建的旧 `elastic_resume` 副本，约释放 282 GiB：`u0031-w4`、`u0033-w1/w2/w5/w6`、`u0034-w1`。删除前核实全部原始 FSDP checkpoint、FP32 模型及转换清单，并归档元数据、文件列表和删除回执。原始数据不变；旧转换副本可重新生成。

新窗口采用：

1. 每 update 保存完整 checkpoint；最新提交后校验，再滚动保留最近两份恢复状态。写入期间可短暂同时存在三份，不能按“瞬时最多两份”估算。
2. 每个 U36–U40 的完整 FP32 policy 永久保留；所有训练轨迹、batch、mask/advantage 对齐、old/new 全词表 log-probs、optimizer step 记录及 P/C/D/activation 信号保留。
3. 旧更新的完整恢复 checkpoint 只有在其 FP32 policy 和单步 signals 已归档，且最新两个完整恢复点均有效时才移除。被移除的旧 Adam 状态不再保证可恢复，但相应 policy 权重、实际更新证据和效用分析材料都保留。
4. GPU 数量改变时进行模型与 Adam 的无损重分片。新 native checkpoint 提交后，释放该次临时转换副本；不删除旧实验的原始恢复点。
5. update 边界继续保留 250 GiB 磁盘保护；启动前还检查下一次 update 的预估写入（至少 100 GiB，需重分片时另加 50 GiB），预估瞬时剩余不低于 200 GiB。共享磁盘增长或轨迹较长时可能停在安全边界，不保证整个窗口的磁盘需求恒定。

## GPU 与进度

启动请求时 GPU 4、5、6 空闲。先将 U35 的单卡恢复状态转换为三卡格式；每个 update 在边界选择当时空闲卡数，更新内部不改变 world size。异构资源续训保留 Adam 数值与计数，但 rank RNG 重设，因此不宣称与固定 8 卡运行 bitwise 相同。

初始 CPU 回归：208 项测试通过（包含新排序、并列处理、基线一致性、清理范围和恢复检查）。预注册与完整基线导入已完成；实际 GPU forward/backward 是否通过以运行日志为准。

归档根目录：[qwen35-clean-s303-u35-to40-ranking-v1](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1)。

- `protocol.json` / `manifest.json`：冻结窗口、来源、哈希与参数。
- `status.json` / `supervisor.log` / `logs/`：实时任务与日志。
- `allocations/` / `gpu_watch_latest.json`：实际物理卡分配与等待状态。
- `storage_audit/`：旧副本删除及新恢复 checkpoint 轮换回执。
- `window_signals/u0035-u0040/prediction.json`：目标 gold 前冻结的分数与排序规则。
- `window_metrics/ranking_metrics.csv` / `locked_scores_and_gold.csv`：窗口内 top-k 统计与全部分数/ΔM。
- `reports/window-results.md`：完成后的结果分析；将自动追加到本文末尾。

恢复入口（`conda activate skill-RL` 后，在仓库根目录）：

```bash
python -m phase2.run_extended --root /home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1 --detach
```

原始 4-Skill 支持限制仍在；本窗口不冒充已经扩展到更多 Skill 或多个独立 RL 路径。

## 2026-09-13 下午续跑：U39 → U40

截至 14:50（北京时间），U36–U39 均已完成训练、FP32 导出及单步信号归档；最新两份完整恢复点为 U38/U39。U40 尚未训练，U40 gold 尚未打开。本次只继续原先冻结的窗口，不更改学习率、采样 seed、训练 batch、Skill/anchor 集、指标或排序规则。

续跑时 GPU 0–4 空闲，GPU 5–7 被其他任务占用。U39 的 native checkpoint 为七卡格式，计划无损重分片后使用五卡。临时转换副本放在独立、带归属标记的 `/dev/shm/skillrl-phase2-u40-37z1sgne` 内存盘目录；它可由磁盘上的 native U39 和 FP32 U39 重建。所有新旧 checkpoint、实际训练 batch、old/new 全词表 log-probs 和评测轨迹仍写入持久磁盘。新 native U40 提交并校验后才释放该临时转换副本。

此次只修订资源准入的峰值核算，未降低保护线：保留 250 GiB 边界保护、200 GiB 预估瞬时剩余，以及至少 100 GiB 总写入预算。现有执行顺序是先提交 U40、验证最新两份恢复点，再移除已经完成归档的 U38，最后导出 U40 FP32。因此约 18.04 GiB 的 FP32 导出发生在约 49.39 GiB 的旧恢复点轮转之后；100 GiB 总写入预算对应的净增长峰值约为 81.96 GiB，而非 100 GiB。14:50 实测磁盘空闲约 292.96 GiB，预估最低剩余约 211.00 GiB，满足原保护线。

实际 rollout batch 生成后、OLD log-probs 捕获及 optimizer 更新前，另按全部 attention-valid response token 数 × 248,320 词表 × FP32 × OLD/NEW 两份，重新计算本批存储需求，并加入 native checkpoint、导出及额外写入余量。不满足原 200 GiB 保护线则停止，不缩减证据、不偷偷降低精度。共享磁盘仍可能被其他任务消耗，不能将准入估计视为容量保证。

执行修订归档于 `staged_storage_amendment.json`；各阶段存储测算位于 `storage_audit/admission-u0040-*.json`。冻结科学协议 SHA-256 仍为 `0ca78a05b4d58584be39ecfaa9335458b41dfd50ea6e34f80e43be797acbe47a`。本次 CPU 回归共 216 项通过，包含峰值测算、实际 token 预算、协议哈希检查、临时副本清理范围及 native checkpoint 保护测试。实际恢复校验、GPU 进度及最终结果以下续运行记录为准。

15:03 执行状态：五个转换分片均已完成序列化及数值哈希校验，后台 supervisor PID 为 `646919`。训练加载期间其他任务占用了原先空闲的 GPU 0–4，U40 在初始化阶段退出，尚无新 optimizer update 或训练证据；因此未损失已完成更新。启动诊断只归档在 `attempts/`，不计入实验结果。当前八张 GPU 均被其他任务使用，流程处于 `waiting_for_any_free_training_gpu`，每 30 秒检查一次；满足资源条件后自动重新分配空闲卡，从 native U39 继续。实际 live restore 的 log-prob parity 和 U40 完成状态仍待后续确认，不能把转换成功视为 U40 训练完成。

## 2026-09-13 晚间：完成剩余评估的资源调度

U40 已完成真实训练并提交 native checkpoint；21:05 完成完整 FP32 policy 导出。此后不再需要训练，剩余阶段为 U39→U40 单步信号、U35→U40 窗口信号、目标 gold 前锁定排序分数、U40 的 3,600 条三臂 suffix，以及最终效用/排序分析。源训练证据、已有 U35 基线和冻结协议不变。

用户要求提高有效 GPU 占用、完成剩余评估。本次复查 GPU 2/4 空闲，因此启用独立评估进程同卡并行：measurement 每卡最多 3 个进程、每进程预算 23 GiB；evaluation 每卡最多 4 个进程、每进程预算 14 GiB；均额外预留 8 GiB 显存。调度会计入尚在加载的进程预算，不把其暂时较小的 CUDA 占用误判为剩余容量。只向空闲 GPU 或完全由本调度器工作进程使用的 GPU 增派任务；不空填显存、不终止其他任务，也不能保证其他用户不在运行中启动竞争进程。

八个逻辑分片和各 trajectory 的 seed/ID、模型精度、推理方式、prompt、anchor、arms、gold 锁定顺序均保持不变。同一进程每步推理仍采用原有固定 seed，进程并发不共享 RNG。只对只读指标计算/评估的 GPU OOM 自动归档并重试；完整 trajectory 和已提交分片复用，不完整分片从缺失任务恢复。遇到非 OOM 的对齐、协议或计算错误则停止，不掩盖错误。

磁盘约剩 244 GiB。250 GiB 的**新训练**准入线保持不变；对已提交模型的剩余评估，采用原 200 GiB 安全余量加 20 GiB 剩余写入预算，即准入要求至少 220 GiB。每轮调度检查容量，不再错误要求另一个完整训练 checkpoint 的空间。

执行策略归档于 `evaluation_resource_amendment.json`，逐次分配记录在 `evaluation_allocation_history.jsonl`，重试诊断在 `evaluation_attempts/`。代码/测试快照由 `source_snapshots/` 留存；224 项 CPU 回归通过。结果仍须等待完整 gold 和最终校验，不能把并行启动视为评估完成。

21:33 已实际观测 GPU 2/4 各运行 3 个 measurement 进程，显存各约 55 GiB、利用率各 100%。21:36 八个 U39→U40 measurement 分片全部提交，进入单步汇总。新 supervisor PID 为 `1712430`，随后自动执行窗口信号、预测锁定与目标端点评估。

另启用 `phase2.finalize_evaluation` 完成核验：仅在主流程完整结束后读取全部注册端点，核验 trajectory ID/seed/分片完整性、新轨迹文件内容与 SHA-256，以及目标轨迹生成时间严格晚于 prediction 锁定。核验成功后生成 `evaluation_completion.json`（状态 `complete_and_verified`），追加完整原始指标对照表，并保存 `window_metrics/raw_features_and_semantic_utility.csv`。其中 P 是**未经排序方向变换的原始 P**；下降排序使用 −P，二者明确区分。完成核验进度在 `completion_watch_status.json`。包含核验新增测试后，共 226 项 CPU 回归通过。

21:47 状态更新：U39→U40 单步信号和 U35→U40 窗口信号均已提交，预测分数已先于目标 gold 锁定；U40 的 8 个评估分片全部启动，GPU 2/4 各 4 个进程，观测利用率约 99%。每个分片固定 450 条 suffix，共 3,600 条，不能把启动计为完成。主流程 PID 为 `1712430`，独立完成核验 PID 为 `1732571`。增加端到端核验/原始 P 符号保留测试后，227 项 CPU 回归全部通过。本轮完成范围仍是当前冻结的 U35→U40 窗口；不额外训练旧 pilot 的 U36，也不增设 U36–U39 的中间 gold。

## 2026-09-14：取消评估预算限制，续跑并形成完整分析报告

用户明确授权“不用管磁盘预算”，因此将原 200+20 GiB 评估预留线单独豁免，科学协议和原资源策略文件保留。新授权记录为 `evaluation_disk_waiver_20260914.json`，绑定原策略与科学协议 SHA-256；只保留 256 MiB 的实际近写满保护，不再因为 200/220 GiB 的长期预算停止评估。没有删除 checkpoint 或原始证据，也没有因此启动新训练。

续跑前已有 674/3,600 条完整轨迹，逐行 JSON、唯一 ID 和轨迹文件存在性检查均通过；约占 65 MiB，剩余轨迹预估仅数百 MiB。当时磁盘实际空闲约 212 GiB。主流程 PID 为 `4063983`，继续 GPU 2/4 各 4 个进程，只补缺失轨迹，锁定的 prediction 不变。恢复后已观察到 920/3,600 条记录和双卡约 99% 利用率。

本次新增最终综合报告计划 `complete_report_plan_20260914.json`：完成核验之后，由 `phase2.complete_report` 自动整合旧 U30–U35 单步 pilot 与新 U35→U40 窗口，生成 `/home/wangyifan/skill-RL/2026-09-14-phase2-complete-analysis.md`。包括更新/Adam/参数差分、支持集差异、三臂语义效用及双轴分解、D/−P/幅度指标直接排序、全部分层原始值、行为与 Skill 调用统计、外推限制与 Phase3 成本验证边界。旧 report、Phase1 和 idea 文件均哈希绑定且不覆盖；不同 anchor 支持集不混为一条效用时间序列。

最终报告只有在 `evaluation_completion.json` 为 `complete_and_verified` 且全部结果文件校验通过时生成；成功标记为 `full_report_completion.json`，附输入和报告 SHA-256。故障过程仅保留在执行归档，不作为研究结果混入最终分析。

最终综合报告生成器已通过端到端测试（包含原始 P、缺失阶段、配对行为统计、固定排序池及报告/输入哈希），整套 CPU 回归为 233 项通过。当前完成核验及报告进程 PID 为 `4088408`；等待评估结束后自动执行，无需再次手动启动汇总。


<!-- completed-u35-to40-ranking-window -->

# Extended Phase2：按冻结窗口汇总

路径：`seed303-continued-u35-to40-ranking-v1`；更新时间：2026-09-14T00:43:55.927353+00:00。

窗口方向为起始 batch 的 reward direction 在窗口端点 interaction 上的投影，不是逐 update D 的相加。all 与各阶段单独分析，不将重复分层当成独立样本。

## 语义效用变化（pp）

|   start_update |   global_update | window_role   | skill_id   | context_id   |   anchor_count |   continuation_repeats |   delta_utility |   ci_low |   ci_high |
|---------------:|----------------:|:--------------|:-----------|:-------------|---------------:|-----------------------:|----------------:|---------:|----------:|
|             35 |              40 | test          | cle_003    | clean        |             50 |                      4 |         15.7258 |   2.4194 |   29.0323 |
|             35 |              40 | test          | cle_004    | clean        |             50 |                      4 |         -6.4516 | -14.9194 |    1.2097 |
|             35 |              40 | test          | cle_006    | clean        |             50 |                      4 |         -1.6129 |  -5.6452 |    2.0161 |
|             35 |              40 | test          | gen_002    | clean        |             50 |                      4 |         -3.0172 |  -7.7586 |    0.0000 |

## 主分析：同一窗口内的 Skill top-k 排序

分数不依赖拟合回归器；正负方向和预算先于目标 gold 固定。所有方法使用同一支持候选池；并列按随机打破的期望计分。下表下降是 gold 点估计，不代表每个 Skill 的下降已统计确证。

|   start_update |   global_update | context_id   | score                  |   n_candidates |   events |   k |   precision_at_k |   recall_at_k |   captured_change_mass |   spearman |
|---------------:|----------------:|:-------------|:-----------------------|---------------:|---------:|----:|-----------------:|--------------:|-----------------------:|-----------:|
|             35 |              40 | clean        | D_contribution         |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | D_contribution         |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | D_ungated_contribution |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | D_ungated_contribution |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | P_int                  |              3 |        2 |   1 |           1.0000 |        0.5000 |                 0.6814 |     1.0000 |
|             35 |              40 | clean        | P_int                  |              3 |        2 |   2 |           1.0000 |        1.0000 |                 1.0000 |     1.0000 |
|             35 |              40 | clean        | activation_l16_norm    |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | activation_l16_norm    |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | activation_l24_norm    |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | activation_l24_norm    |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | activation_l32_norm    |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | activation_l32_norm    |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | activation_l8_norm     |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | activation_l8_norm     |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | delta_centered_norm    |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | delta_centered_norm    |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | delta_norm             |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | delta_norm             |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | forward_kl_original    |              3 |        2 |   1 |           1.0000 |        0.5000 |                 0.6814 |     1.0000 |
|             35 |              40 | clean        | forward_kl_original    |              3 |        2 |   2 |           1.0000 |        1.0000 |                 1.0000 |     1.0000 |
|             35 |              40 | clean        | js_original            |              3 |        2 |   1 |           1.0000 |        0.5000 |                 0.6814 |     1.0000 |
|             35 |              40 | clean        | js_original            |              3 |        2 |   2 |           1.0000 |        1.0000 |                 1.0000 |     1.0000 |
|             35 |              40 | clean        | old_margin             |              3 |        2 |   1 |           1.0000 |        0.5000 |                 0.3186 |     0.5000 |
|             35 |              40 | clean        | old_margin             |              3 |        2 |   2 |           1.0000 |        1.0000 |                 1.0000 |     0.5000 |
|             35 |              40 | clean        | u_control_norm         |              3 |        2 |   1 |           1.0000 |        0.5000 |                 0.6814 |     1.0000 |
|             35 |              40 | clean        | u_control_norm         |              3 |        2 |   2 |           1.0000 |        1.0000 |                 1.0000 |     1.0000 |
|             35 |              40 | clean        | u_original_norm        |              3 |        2 |   1 |           0.0000 |        0.0000 |                 0.0000 |    -0.5000 |
|             35 |              40 | clean        | u_original_norm        |              3 |        2 |   2 |           0.5000 |        0.5000 |                 0.6814 |    -0.5000 |
|             35 |              40 | clean        | random_expected        |              3 |        2 |   1 |           0.6667 |        0.3333 |                 0.3333 |   nan      |
|             35 |              40 | clean        | random_expected        |              3 |        2 |   2 |           0.6667 |        0.6667 |                 0.6667 |   nan      |

下降事件为零时 Recall/下降量覆盖率未定义。完整 CSV 同时包含 >5 pp 阈值、|ΔM| 审计及各阶段；这些不是额外独立窗口。首个单窗口排序只作描述，不声称跨窗口/seed 泛化或排序指标显著性。

## 辅助：数值预测（非主要成功标准）

| phase   | model                |   units |   windows |   mae_pp |
|:--------|:---------------------|--------:|----------:|---------:|
| all     | predicted_delta_zero |       3 |         1 |   8.3982 |
| early   | predicted_delta_zero |       1 |         1 |   6.4516 |
| late    | predicted_delta_zero |       1 |         1 |   0.0000 |
| middle  | predicted_delta_zero |       2 |         1 |  11.0628 |


<!-- phase2-evaluation-completion-audit-v1 -->

## 完整评估校验及原始指标对照

全部注册端点评估通过 ID/seed/分片覆盖检查；新端点 3,600 条完整轨迹均已核对索引并留存 SHA-256。目标轨迹生成均晚于对应 prediction 锁定。

下表效用与 CI 单位为 pp；D、P、norm 等是原始量，未作排序方向变换。正式下降排序使用 D、−P、+norm；不能把排序 CSV 中的 −P 误读为原始 P。

|   start_update |   global_update | skill_id   | supported   |   utility_old |   utility_new |   delta_utility |    ci_low |   ci_high |   D_contribution |      P_int |   delta_centered_norm |   delta_norm |       C_upd |   gate_coverage |
|---------------:|----------------:|:-----------|:------------|--------------:|--------------:|----------------:|----------:|----------:|-----------------:|-----------:|----------------------:|-------------:|------------:|----------------:|
|             35 |              40 | gen_002    | True        |       3.01724 |        0      |        -3.01724 |  -7.75862 |   0       |        0.0560228 |  0.0592123 |              113.505  |      218.266 | 0.000518489 |        0.542037 |
|             35 |              40 | cle_003    | True        |      46.371   |       62.0968 |        15.7258  |   2.41935 |  29.0323  |        0.0879431 |  0.0916988 |              223.899  |      562.073 | 0.000463877 |        0.458469 |
|             35 |              40 | cle_004    | True        |       6.45161 |        0      |        -6.45161 | -14.9194  |   1.20968 |        0.0769062 | -0.0123887 |              153.036  |      248.072 | 0.000334997 |        0.495007 |
|             35 |              40 | cle_006    | False       |       1.6129  |        0      |        -1.6129  |  -5.64516 |   2.01613 |        0.0215787 |  0.0403366 |               76.7731 |      193.687 | 0.000117223 |        0.279487 |

包含 initial/early/middle/late 在内的全部信号支持/不支持单元及原始观测量详见 `window_metrics/raw_features_and_semantic_utility.csv`。没有自然评估 anchor 的阶段保留为空标签，并标注 `gold_evaluation_available=false`，不伪造零效用。正式方法比较仍以预先锁定的共同支持池、top-k 下降事件命中及下降量覆盖率为准，不要求精确回归 ΔM。

本次仅一个 seed303 延续路径、一个 U35→U40 窗口、原有 4 个 Skill；分层和 continuation 重复不增加独立训练窗口数。点估计方向、CI 与排序表现需分别解读；不能据单窗口宣称跨 seed 泛化或 D 已普遍优于幅度指标。
