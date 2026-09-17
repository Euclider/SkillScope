# Phase2：语义边际效用方向验证（首批滚动报告）

更新时间（UTC）：2026-09-11T18:28:56.377120+00:00。汇报目标：2026-09-11 上午（北京时间）。

运行：`qwen35-clean-s303-u30-to36-semantic-direction-fast-v1`。当前阶段：`stopped_on_error`。

## 1. 主目标与实验边界

主分析为 `M_sem = success(ORIGINAL) − success(PLACEBO)`，以连续 signed `ΔM_sem` 为目标，重点检验 `P_intᴾ` 的方向预测增量。`Dᴾ` 对下降风险的预测为次要分析；NULL 是同步记录的次要对照。上升、下降均有研究价值，harmful sign flip 不是成立条件。

本轮从选定的 Seed303 U30 完整恢复，固定采集 U31–U36。U31/U32 为开发，U33 为边界隔离，U34–U36 为时间外测试。首批只有一个父 seed、3个测试 update；事件不足或区间跨0时报告不确定。

为赶汇报，先采用每 anchor 两次独立 continuation：41011 估计旧 margin，42011 构造 matched old/new gold。使用全部200个旧 B0 anchors、4个 supported Skills，覆盖中途调用。时间外测试复用固定 game/state support，不声称新的 game 或 seed 泛化。

## 2. 实际进度与完整性

已保存恢复 checkpoint：[31, 32, 33, 34, 35]；已锁定信号：[31, 32, 33, 34, 35]；完整 gold/evidence suffix：7,200 / 8,400。

旧/新 ORIGINAL 全词表概率直接来自训练 actor 的相同前向路径。PLACEBO/NULL 用原始精度端点重建并 teacher-force 同一旧 response。advantage 保存实际 actor loss tensor，decision ID 随重排携带；补齐重复行在统计中去重，并保留实际 optimizer 使用记录。

当前需处理的运行错误：`Free disk below the preregistered 250GiB reserve; stopped at a recovery boundary`。

9月11日00:43的U34单卡尝试在训练前被旧preflight的两卡下限拦下，未发生新更新。05:23后已适配显式卡数检查（Phase1默认不变），从U33重新按空闲资源续训；失败记录见 `attempts/u0034-failed-1789058617473162626/README.md`。

9月11日06:10的U34五卡尝试在首次backward发生共享显存OOM，未完成Adam step或保存U34端点。13:10人工审计后，将32条rollout、750行旧概率及batch等移入 `attempts/u0034-failed-1789078232762546641/evidence/`，失败数据不进入主分析；原文件与轨迹索引备份均可恢复。从U33重新采样续训，不声称逐bit续接失败五卡随机流；具体原因见该尝试的 `README.md`。

9月11日13:31的U34原生四卡尝试也在首次backward因另一进程占约39GiB显存而OOM，未完成Adam step。15:35将32条轨迹、676行batch/旧概率及前向进度归入 `attempts/u0034-failed-1789104704603119618/evidence/`，未删除或计入正式结果。按既有资源协议准备从U33改用当前空闲卡数续训，双卡时采用CPU AdamW；原生四卡没有elastic审计，不将其误记为丢失恢复记录。

最近一次等待资源检查（UTC）`2026-09-11T12:58:31.539434+00:00`：即时空闲候选 `[3]`，连续两次确认空闲 `[3]`。此为调度边界快照，不表示运行中GPU的实时空闲状态。

监控每30秒检查，显存占用<1500MiB且利用率<10%、连续两次满足才续训；使用所有确认空闲卡。U34首次初始化OOM记录已保存于 `attempts/u0034-attempt1-startup-oom/`。只在尚无任何新训练证据的初始化OOM时自动等待重试；其他异常仍停止。`gpu_watch.jsonl`保留检查记录，`status.json`为当前阶段。

9月11日15:43起，另有独立只读监控每15秒归档物理GPU显存/利用率、compute PID与磁盘余量，见 `runtime_resources/pipeline-2142632.jsonl`（后续运行按pipeline PID另存）。该监控不预留显存或终止其他任务；不能保证共享GPU独占。

### 按可用资源续训（用户于9月10日授权）

下表为各update最近一次启动尝试的分配，不代表当前仍占用这些GPU；完整分配历史保存在 `allocation_history.jsonl` 与失败尝试目录。9月11日起，每次启动的preflight与不可变manifest分别保存在 `launch_manifests/`，路径由allocation记录；旧manifest不覆盖，逻辑训练ID不变。

|   global_update | physical_gpus   |   world_size |   source_world_size | optimizer_backend   |   training_rollouts |
|----------------:|:----------------|-------------:|--------------------:|:--------------------|--------------------:|
|              32 | [0, 1, 2, 3]    |            4 |                   8 | gpu_adamw           |                  32 |
|              33 | [0, 1, 2, 3]    |            4 |                   4 | gpu_adamw           |                  32 |
|              34 | [3, 6]          |            2 |                   4 | cpu_adamw           |                  32 |
|              35 | [3]             |            1 |                   2 | cpu_adamw           |                  32 |

LR=1e-6、KL=0.01保持不变。卡数只在 update 边界调整；参数、Adam moments/counter 与 scheduler 保留。换 world size 后使用预定的独立 rank RNG，不能声称逐 bit 延续8卡轨迹。1–2卡采用CPU AdamW；其他卡数保留GPU AdamW。3/5/6/7卡通过零权重同步槽保持每个完整 optimizer minibatch 的32个有效 decision，而不是悄悄改变有效batch。

资源占用已经导致先前12–18小时排期失效；首批报告按实际完成端点发布，不以汇报截止时间冒充实验完成。恢复逐rank校验见 `elastic_restore_audits/`，资源修订见 `resource_policy.json`。

|   update |   audited_ranks |   world_size |   Adam_at_restore | all_checksums_match   |   parity_max_abs |
|---------:|----------------:|-------------:|------------------:|:----------------------|-----------------:|
|       32 |               4 |            4 |               519 | True                  |           0.0000 |
|       34 |               2 |            2 |               568 | True                  |           0.0000 |
|       35 |               1 |            1 |               590 | True                  |           0.0000 |

该表只证明加载状态校验通过，不代表对应 update 已训练完成。U32 首次四卡尝试在恢复后 rollout 入口遇到 FSDP inference-buffer 错误，未进行新更新；已改用 no_grad 校验并通过真实双卡生命周期测试，失败日志完整保留在 `attempts/u0032-attempt1-inference-cache/`。

|   update |   batch_rows |   unique_decisions |   loss_tokens | alignment_audit   |   old_full_vocab_rows |   new_full_vocab_rows |   recorded_Adam_steps |
|---------:|-------------:|-------------------:|--------------:|:------------------|----------------------:|----------------------:|----------------------:|
|       31 |          744 |                740 |         15297 | True              |                   744 |                   744 |                    24 |
|       32 |          696 |                695 |         13586 | True              |                   696 |                   696 |                    22 |
|       33 |          864 |                861 |         17653 | True              |                   864 |                   864 |                    27 |
|       34 |          680 |                679 |         12439 | True              |                   680 |                   680 |                    22 |
|       35 |          654 |                654 |         13608 | True              |                   654 |                   654 |                    21 |

此表可含尚在进行的 update；只有 checkpoint 与 signals committed 后才视为端点测量完成。

## 3. Reward-directed signals（读取新 gold 前锁定）

|   global_update | skill_id   | supported   |   nonzero_advantage_decisions |   nonzero_advantage_games |   P_int |   D_contribution |   gate_coverage |
|----------------:|:-----------|:------------|------------------------------:|--------------------------:|--------:|-----------------:|----------------:|
|              31 | cle_003    | True        |                           120 |                         5 |  0.0264 |           0.0278 |          0.4071 |
|              31 | cle_004    | True        |                           294 |                         5 |  0.0198 |           0.0239 |          0.5246 |
|              31 | cle_006    | True        |                            20 |                         5 | -0.0098 |           0.0188 |          0.3236 |
|              31 | gen_002    | True        |                            64 |                         4 |  0.0130 |           0.0189 |          0.3590 |
|              32 | cle_003    | True        |                           145 |                         7 |  0.1110 |           0.0324 |          0.6498 |
|              32 | cle_004    | True        |                           305 |                         7 |  0.0153 |           0.0291 |          0.5550 |
|              32 | cle_006    | True        |                            28 |                         7 |  0.0372 |           0.0164 |          0.6212 |
|              32 | gen_002    | True        |                           130 |                         7 |  0.0098 |           0.0311 |          0.6944 |
|              33 | cle_003    | True        |                           188 |                         7 |  0.0085 |           0.0635 |          0.5599 |
|              33 | cle_004    | True        |                           368 |                         7 |  0.0210 |           0.0304 |          0.5345 |
|              33 | cle_006    | True        |                            28 |                         7 | -0.0007 |           0.0374 |          0.5855 |
|              33 | gen_002    | True        |                           120 |                         7 |  0.0242 |           0.0425 |          0.7186 |
|              34 | cle_003    | True        |                            68 |                         4 |  0.0083 |           0.0127 |          0.1714 |
|              34 | cle_004    | True        |                           167 |                         4 |  0.0122 |           0.0273 |          0.4525 |
|              34 | cle_006    | False       |                            16 |                         4 | -0.0003 |           0.0056 |          0.2119 |
|              34 | gen_002    | True        |                           100 |                         4 |  0.0121 |           0.0150 |          0.2940 |
|              35 | cle_003    | True        |                           108 |                         6 |  0.0533 |           0.0639 |          0.5477 |
|              35 | cle_004    | True        |                           257 |                         6 |  0.0307 |           0.0398 |          0.6777 |
|              35 | cle_006    | True        |                            24 |                         6 |  0.0052 |           0.0238 |          0.3880 |
|              35 | gen_002    | True        |                            69 |                         6 | -0.0015 |           0.0204 |          0.3563 |

`D_contribution` 的 token 均值即 D；分母包括零 advantage 与未通过 gate 的 token。supported=false 的组不参与主方向预测，不能把它们解释成低风险。

`d=A(e_a−π_old)` 是训练 outcome-consistent direction，包含实际 GRPO normalization 与 invalid penalty。它不是逐动作正确性；C/P/D 原式、中心化坐标及 matched-backend sensitivity 全部归档。

|   update |   Adam_steps |   Adam_before |   Adam_after |   parameter_L2 |   live_rows |
|---------:|-------------:|--------------:|-------------:|---------------:|------------:|
|  31.0000 |      24.0000 |      495.0000 |     519.0000 |         0.2490 |    744.0000 |
|  32.0000 |      22.0000 |      519.0000 |     541.0000 |         0.2667 |    696.0000 |
|  33.0000 |      27.0000 |      541.0000 |     568.0000 |         0.2788 |    864.0000 |
|  34.0000 |      22.0000 |      568.0000 |     590.0000 |         0.1996 |    680.0000 |
|  35.0000 |      21.0000 |      590.0000 |     611.0000 |         0.1790 |    654.0000 |

### 指标覆盖与解释

已计算逐token（保留decision ID）的 `P_int`、`C_upd`、门控/未门控D、`u_original/u_control/delta` 的范数、KL/JS，以及8/16/24/32层activation interaction范数；原始FP32参数差分另行归档。正文P是预定token等权聚合，不是每个环境step独立gold。可由token归档重建decision等权读出，但不临时选择更有利的聚合。

P保留正负方向；D是负向部分的门控聚合，不能以D低推断效用上升。C检查update fidelity，不是效用方向标签。action/activation/parameter范数没有符号；同一次update的全局参数范数对所有Skill相同。JVP与线性化误差尚未完成，不能用参数范数冒充。

|   global_update | skill_id   |   C_upd |   direction_coverage |   gate_coverage |
|----------------:|:-----------|--------:|---------------------:|----------------:|
|              31 | cle_003    |  0.0004 |               0.6568 |          0.4071 |
|              31 | cle_004    |  0.0005 |               0.8370 |          0.5246 |
|              31 | cle_006    |  0.0002 |               0.6204 |          0.3236 |
|              31 | gen_002    |  0.0002 |               0.6026 |          0.3590 |
|              32 | cle_003    |  0.0007 |               0.9140 |          0.6498 |
|              32 | cle_004    |  0.0006 |               0.9317 |          0.5550 |
|              32 | cle_006    |  0.0005 |               0.8468 |          0.6212 |
|              32 | gen_002    |  0.0008 |               0.9716 |          0.6944 |
|              33 | cle_003    |  0.0005 |               0.8821 |          0.5599 |
|              33 | cle_004    |  0.0006 |               0.8405 |          0.5345 |
|              33 | cle_006    |  0.0004 |               0.8696 |          0.5855 |
|              33 | gen_002    |  0.0010 |               1.0000 |          0.7186 |
|              34 | cle_003    |  0.0001 |               0.4297 |          0.1714 |
|              34 | cle_004    |  0.0004 |               0.7742 |          0.4525 |
|              34 | cle_006    | -0.0000 |               0.4925 |          0.2119 |
|              34 | gen_002    |  0.0003 |               0.5218 |          0.2940 |
|              35 | cle_003    |  0.0005 |               0.7680 |          0.5477 |
|              35 | cle_004    |  0.0007 |               0.8936 |          0.6777 |
|              35 | cle_006    |  0.0000 |               0.7678 |          0.3880 |
|              35 | gen_002    |  0.0003 |               0.5379 |          0.3563 |

## 4. 三臂效用与变化方向

数值使用 success-rate 单位；例如0.05为5个百分点。CI按相同 game 的 old/new paired difference bootstrap 10,000次；预定方向阈值为±5pp。

|   global_update | skill_id   |   utility_old |   utility_new |   delta_utility |   ci_low |   ci_high | direction   |
|----------------:|:-----------|--------------:|--------------:|----------------:|---------:|----------:|:------------|
|              31 | cle_003    |        0.4833 |        0.5333 |          0.0500 |  -0.0667 |    0.1667 | uncertain   |
|              31 | cle_004    |        0.0357 |        0.0357 |          0.0000 |   0.0000 |    0.0000 | stable      |
|              31 | cle_006    |        0.0000 |        0.0000 |          0.0000 |   0.0000 |    0.0000 | stable      |
|              31 | gen_002    |        0.0000 |        0.0000 |          0.0000 |   0.0000 |    0.0000 | stable      |
|              32 | cle_003    |        0.5333 |        0.5667 |          0.0333 |   0.0000 |    0.1000 | uncertain   |
|              32 | cle_004    |        0.0357 |        0.0000 |         -0.0357 |  -0.1071 |    0.0000 | uncertain   |
|              32 | cle_006    |        0.0000 |        0.0000 |          0.0000 |   0.0000 |    0.0000 | stable      |
|              32 | gen_002    |        0.0000 |        0.0000 |          0.0000 |   0.0000 |    0.0000 | stable      |
|              33 | cle_003    |        0.5667 |        0.5333 |         -0.0333 |  -0.1000 |    0.0000 | uncertain   |
|              33 | cle_004    |        0.0000 |        0.0000 |          0.0000 |   0.0000 |    0.0000 | stable      |
|              33 | cle_006    |        0.0000 |        0.0000 |          0.0000 |   0.0000 |    0.0000 | stable      |
|              33 | gen_002    |        0.0000 |        0.0000 |          0.0000 |   0.0000 |    0.0000 | stable      |
|              34 | cle_003    |        0.5333 |        0.5667 |          0.0333 |   0.0000 |    0.1000 | uncertain   |
|              34 | cle_004    |        0.0000 |        0.0714 |          0.0714 |   0.0000 |    0.1786 | uncertain   |
|              34 | cle_006    |        0.0000 |        0.0000 |          0.0000 |   0.0000 |    0.0000 | stable      |
|              34 | gen_002    |        0.0000 |        0.0000 |          0.0000 |   0.0000 |    0.0000 | stable      |
|              35 | cle_003    |        0.5667 |        0.5500 |         -0.0167 |  -0.1333 |    0.1000 | uncertain   |
|              35 | cle_004    |        0.0714 |       -0.0714 |         -0.1429 |  -0.2857 |   -0.0179 | uncertain   |
|              35 | cle_006    |        0.0000 |        0.0000 |          0.0000 |   0.0000 |    0.0000 | stable      |
|              35 | gen_002    |        0.0000 |        0.0000 |          0.0000 |   0.0000 |    0.0000 | stable      |

所有 initial/early/middle/late 分层、O−NULL 与 O−PLACEBO 对照保存在机器可读表，不能把多条 anchor 当作独立 update。

这里 stable 是本批固定 continuation seed 下的 game bootstrap 分类；若所有观测差值均为0，重采样CI会退化为[0,0]，这不证明对其他生成随机种子或真实期望效用也完全不变。

### Policy 与对照臂的双轴变化

|   global_update | skill_id   |   original_old |   original_new |   control_old |   control_new |   delta_original |   delta_control |
|----------------:|:-----------|---------------:|---------------:|--------------:|--------------:|-----------------:|----------------:|
|              31 | cle_003    |         0.6000 |         0.5667 |        0.1167 |        0.0333 |          -0.0333 |         -0.0833 |
|              31 | cle_004    |         0.1786 |         0.1429 |        0.1429 |        0.1071 |          -0.0357 |         -0.0357 |
|              31 | cle_006    |         0.3226 |         0.2581 |        0.3226 |        0.2581 |          -0.0645 |         -0.0645 |
|              31 | gen_002    |         0.5345 |         0.5345 |        0.5345 |        0.5345 |           0.0000 |          0.0000 |
|              32 | cle_003    |         0.5667 |         0.5667 |        0.0333 |        0.0000 |           0.0000 |         -0.0333 |
|              32 | cle_004    |         0.1429 |         0.1071 |        0.1071 |        0.1071 |          -0.0357 |          0.0000 |
|              32 | cle_006    |         0.2581 |         0.2581 |        0.2581 |        0.2581 |           0.0000 |          0.0000 |
|              32 | gen_002    |         0.5345 |         0.5345 |        0.5345 |        0.5345 |           0.0000 |          0.0000 |
|              33 | cle_003    |         0.5667 |         0.5667 |        0.0000 |        0.0333 |           0.0000 |          0.0333 |
|              33 | cle_004    |         0.1071 |         0.1071 |        0.1071 |        0.1071 |           0.0000 |          0.0000 |
|              33 | cle_006    |         0.2581 |         0.2581 |        0.2581 |        0.2581 |           0.0000 |          0.0000 |
|              33 | gen_002    |         0.5345 |         0.5345 |        0.5345 |        0.5345 |           0.0000 |          0.0000 |
|              34 | cle_003    |         0.5667 |         0.6000 |        0.0333 |        0.0333 |           0.0333 |          0.0000 |
|              34 | cle_004    |         0.1071 |         0.2143 |        0.1071 |        0.1429 |           0.1071 |          0.0357 |
|              34 | cle_006    |         0.2581 |         0.1613 |        0.2581 |        0.1613 |          -0.0968 |         -0.0968 |
|              34 | gen_002    |         0.5345 |         0.5345 |        0.5345 |        0.5345 |           0.0000 |          0.0000 |
|              35 | cle_003    |         0.6000 |         0.6167 |        0.0333 |        0.0667 |           0.0167 |          0.0333 |
|              35 | cle_004    |         0.2143 |         0.2679 |        0.1429 |        0.3393 |           0.0536 |          0.1964 |
|              35 | cle_006    |         0.1613 |         0.4516 |        0.1613 |        0.4516 |           0.2903 |          0.2903 |
|              35 | gen_002    |         0.5345 |         0.5345 |        0.5345 |        0.5345 |           0.0000 |          0.0000 |

这里 control 指 PLACEBO，不是完全 skill-free Policy。NULL 也只移除目标 Skill 的 payload，其他 Skill 仍保留；两者均不能冒充全库 no-skill baseline。`ΔM_sem=ΔORIGINAL−ΔPLACEBO`。

## 5. 关联与前瞻预测

| scope                   |   units |   updates |   P_int_vs_signed_delta_spearman |   D_vs_negative_delta_spearman |   negative_point_count |   negative_prevalence_AP_baseline |   direct_D_negative_AUPRC |
|:------------------------|--------:|----------:|---------------------------------:|-------------------------------:|-----------------------:|----------------------------------:|--------------------------:|
| all_updates_exploratory |      19 |         5 |                          -0.0488 |                         0.4172 |                      1 |                            0.0526 |                    0.2500 |
| heldout_updates         |       7 |         2 |                          -0.4077 |                         0.5559 |                      1 |                            0.1429 |                    0.5000 |

上表是探索性点估计，不能仅凭相关系数方向正确就宣称预测成立。跨 update 稳定性及增量预测需要结合后续表、覆盖率和极小的测试 update 数判断。

### 5.1 连续方向分析补充（9月10日经用户确认）

本补充在已观察U31–U33、尚未取得U34–U36测试gold时归档：`analysis_amendment_v2.json`。既有公式、门控、Skill/anchor集合、预测模型特征、开发/隔离/测试划分不变。新增全部已采集指标的描述性比较和简单基线，不按这里的相关性选择特征。8月19日idea文件仍含早期harmful-flip主目标文字；本报告沿用后续已确认的连续语义效用目标，不声称验证了旧版全部强主张。

局部训练reward方向上的P与独立长期效用并无逐例同号保证。排序关联、逐例同号、时间外预测是三个不同层次；尤其不能把有正相关系数写成方向已预测成功。

| scope                   | signal                 |   units |   updates |   rho_signed_delta |   rho_absolute_delta |
|:------------------------|:-----------------------|--------:|----------:|-------------------:|---------------------:|
| all_updates_exploratory | P_int                  |      19 |         5 |            -0.0488 |               0.3503 |
| all_updates_exploratory | D_contribution         |      19 |         5 |            -0.4172 |               0.3014 |
| all_updates_exploratory | D_ungated_contribution |      19 |         5 |            -0.2843 |               0.5774 |
| all_updates_exploratory | C_upd                  |      19 |         5 |            -0.3361 |               0.1644 |
| all_updates_exploratory | u_original_norm        |      19 |         5 |            -0.1319 |              -0.1155 |
| all_updates_exploratory | u_control_norm         |      19 |         5 |            -0.1251 |              -0.0528 |
| all_updates_exploratory | delta_norm             |      19 |         5 |             0.0039 |               0.5226 |
| all_updates_exploratory | delta_centered_norm    |      19 |         5 |            -0.0616 |               0.6654 |
| all_updates_exploratory | forward_kl_original    |      19 |         5 |            -0.2071 |               0.2681 |
| all_updates_exploratory | js_original            |      19 |         5 |            -0.1426 |               0.3445 |
| all_updates_exploratory | activation_l8_norm     |      19 |         5 |             0.0616 |               0.2838 |
| all_updates_exploratory | activation_l16_norm    |      19 |         5 |             0.0664 |               0.5598 |
| all_updates_exploratory | activation_l24_norm    |      19 |         5 |             0.0557 |               0.5950 |
| all_updates_exploratory | activation_l32_norm    |      19 |         5 |            -0.0635 |               0.6224 |
| all_updates_exploratory | raw_parameter_delta_l2 |      19 |         5 |            -0.0224 |              -0.2096 |
| all_updates_exploratory | old_margin             |      19 |         5 |             0.0425 |               0.7975 |
| all_updates_exploratory | old_margin_se          |      19 |         5 |             0.2002 |               0.7739 |
| all_updates_exploratory | train_success          |      19 |         5 |            -0.1361 |               0.2160 |
| all_updates_exploratory | advantage              |      19 |         5 |             0.0889 |              -0.4952 |
| heldout_updates         | P_int                  |       7 |         2 |            -0.4077 |               0.6301 |
| heldout_updates         | D_contribution         |       7 |         2 |            -0.5559 |               0.4077 |
| heldout_updates         | D_ungated_contribution |       7 |         2 |            -0.4818 |               0.5559 |
| heldout_updates         | C_upd                  |       7 |         2 |            -0.5189 |               0.6301 |
| heldout_updates         | u_original_norm        |       7 |         2 |            -0.2965 |               0.4818 |
| heldout_updates         | u_control_norm         |       7 |         2 |            -0.2965 |               0.4818 |
| heldout_updates         | delta_norm             |       7 |         2 |            -0.2965 |               0.4818 |
| heldout_updates         | delta_centered_norm    |       7 |         2 |            -0.1853 |               0.6301 |
| heldout_updates         | forward_kl_original    |       7 |         2 |            -0.0371 |               0.7783 |
| heldout_updates         | js_original            |       7 |         2 |             0.0371 |               0.7042 |
| heldout_updates         | activation_l8_norm     |       7 |         2 |             0.4447 |               0.7783 |
| heldout_updates         | activation_l16_norm    |       7 |         2 |            -0.0371 |               0.6671 |
| heldout_updates         | activation_l24_norm    |       7 |         2 |            -0.0371 |               0.6671 |
| heldout_updates         | activation_l32_norm    |       7 |         2 |            -0.1853 |               0.6301 |
| heldout_updates         | raw_parameter_delta_l2 |       7 |         2 |             0.7489 |               0.1498 |
| heldout_updates         | old_margin             |       7 |         2 |             0.1165 |               0.6408 |
| heldout_updates         | old_margin_se          |       7 |         2 |             0.1923 |               0.6538 |
| heldout_updates         | train_success          |       7 |         2 |            -0.7489 |              -0.1498 |
| heldout_updates         | advantage              |       7 |         2 |            -0.1112 |              -0.8154 |

上述所有相关系数仅为探索性点估计。`rho_signed_delta`均对应ΔM；D的下降方向相关系数应取相反数。范数与signed ΔM的相关性本身不赋予范数正负方向。多指标、小样本、同一Skill重复观测及相邻差分共享端点均限制解释；不计算把12个单元当12个独立updates的显著性。

| scope                   |   units |   positive_points |   negative_points |   zero_points |   nonzero_point_coverage |   conditional_sign_accuracy |   always_positive_conditional_accuracy |   ci_excludes_zero_points |
|:------------------------|--------:|------------------:|------------------:|--------------:|-------------------------:|----------------------------:|---------------------------------------:|--------------------------:|
| all_updates_exploratory |      19 |                 4 |                 4 |            11 |                   0.4211 |                      0.5000 |                                 0.5000 |                         1 |
| heldout_updates         |       7 |                 2 |                 2 |             3 |                   0.5714 |                      0.5000 |                                 0.5000 |                         1 |

P同号率只对非零效用点估计且P非零的单元计算，必须结合覆盖率；不是可靠方向标签。CI排除0数量只是额外可信度描述，不取代预定±5pp标签。零变化、CI不确定样本全部保留在主连续MAE中，不能删掉或临时更换无变化Skill。

|   update |   units |   P_vs_signed_delta |   D_vs_decline |   positive_points |   negative_points |   zero_points |   conditional_sign_accuracy |
|---------:|--------:|--------------------:|---------------:|------------------:|------------------:|--------------:|----------------------------:|
|  31.0000 |  4.0000 |              0.7746 |        -0.7746 |            1.0000 |            0.0000 |        3.0000 |                      1.0000 |
|  32.0000 |  4.0000 |              0.6325 |        -0.6325 |            1.0000 |            1.0000 |        2.0000 |                      0.5000 |
|  33.0000 |  4.0000 |              0.2582 |         0.7746 |            0.0000 |            1.0000 |        3.0000 |                      0.0000 |
|  34.0000 |  3.0000 |              0.5000 |        -0.5000 |            2.0000 |            0.0000 |        1.0000 |                      1.0000 |
|  35.0000 |  4.0000 |             -0.7379 |         0.7379 |            0.0000 |            2.0000 |        2.0000 |                      0.0000 |

逐update表用于检查关联是否被单次更新主导，4个Skill的相关性仍极不稳定。四个冻结Skill可用于最小可行性验证，不能据此声称全面Skill或跨seed泛化。

### 5.2 时间外预测比较

开发集多数方向（仅U31/U32非零变化、平局abstain）的时间外方向基线：{"units": 7, "updates": 2, "positive_points": 2, "negative_points": 2, "zero_points": 3, "nonzero_points": 4, "nonzero_point_coverage": 0.5714285714285714, "prediction_coverage": 1.0, "evaluated_nonzero_points": 4, "sign_agree": 2, "conditional_sign_accuracy": 0.5, "always_positive_conditional_accuracy": 0.5, "ci_excludes_zero_points": 1, "ci_excludes_zero_sign_accuracy": 0.0}。该±1方向标签不是效用幅度预测，不参与MAE。

| model      |   test_units |   test_updates |   signed_MAE |   rho_signed_delta |   prediction_coverage |   conditional_sign_accuracy |   ci_excludes_zero_sign_accuracy |
|:-----------|-------------:|---------------:|-------------:|-------------------:|----------------------:|----------------------------:|---------------------------------:|
| zero       |            7 |              2 |       0.0378 |           nan      |                0.0000 |                    nan      |                         nan      |
| dev_mean   |            7 |              2 |       0.0403 |           nan      |                1.0000 |                      0.5000 |                           0.0000 |
| old_margin |            7 |              2 |       0.0408 |             0.1165 |                1.0000 |                      0.5000 |                           0.0000 |
| unsigned   |            7 |              2 |       0.0621 |            -0.2965 |                1.0000 |                      0.5000 |                           0.0000 |
| activation |            7 |              2 |       0.0709 |            -0.5189 |                1.0000 |                      0.5000 |                           0.0000 |
| signed     |            7 |              2 |       0.0787 |            -0.4077 |                1.0000 |                      0.5000 |                           0.0000 |
| opposition |            7 |              2 |       0.0785 |            -0.4077 |                1.0000 |                      0.5000 |                           0.0000 |

方向分类首先报告上述 CI 支持的 positive/negative/stable/uncertain。AUPRC/Brier 的负事件仅指点估计 ΔM<−0.05，并不把这些事件称作可靠下降；continuous signed MAE 是不依赖二值标签的主要预测误差。无下降事件时不计算无意义的下降分类AUPRC。新增zero-delta、development-mean基线只作参照，既有模型不重选或调参。

拟合器只使用 U31/U32，最多8个 Skill-update 单元；统一 StandardScaler、Ridge α=1 / Logistic C=1，不调参。支持不足6个时只保留 direct scores，不拟合。测试只有3个 updates，不能将最多12个 Skill-update 单元当成12次独立训练重复。


![方向关联](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/plots/direction_associations.png)

## 6. 尚不能据此声称的结论

JVP 的可微实现、额外随机/打乱 reward 更新对照、更多训练 seed 和完整预算匹配基线仍需分别验证；未运行的项目不会列成已完成。第一批以精确 P/D、参数和 activation、三臂 gold 及时间外方向分析优先。

## 7. 归档

运行根目录：`/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1`。执行协议：`protocol.json`；训练数据：`batches/`；真实新旧概率：`old_logprobs/`、`new_logprobs/`；信号：`signals/`；预测锁定记录：`predictions/`；完整三臂轨迹：`evaluations/`；统计：`metrics/`。

Phase1 原三-seed 报告和权重未修改。当前报告按完整 endpoint 自动更新，缺失结果保持待完成。
