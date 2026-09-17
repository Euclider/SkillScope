# Phase2 扩展验证：代码改动与观测量追加分析

日期：2026-09-12。状态：已修改代码并完成 CPU 回归检查；没有启动新的 RL、GPU signal measurement 或环境 rollout。旧实验数据、checkpoint、冻结预测和 Phase1 报告未修改。

## 1. 本次追加的结果表

[Phase2 分析报告](/home/wangyifan/skill-RL/2026-09-12-phase2-skill-utility-direction-analysis.md) 末尾新增第 11 节，原第 1–10 节保留。

包含：

- 全部 19 个有支持 Skill–update 单元的完整观测量；
- 单独列出其中 7 个测试单元，二者不是相加得到 26 个独立样本；
- P、门控/未门控 D、C 及 centered C、原始/centered interaction norm、ORIGINAL/PLACEBO 更新范数、KL/JS、四个层的 activation norm、参数差分、旧效用/标准误、训练成功率、平均 advantage、训练支持量和 gate 覆盖；
- 对每个观测量统一计算幅度相关、下降相关、下降/上升条件 AUROC、任意下降 AP，以及原来的超过 5 pp 下降 AP；
- 完整精度 CSV、输入 SHA-256、生成脚本与生成时的关键源文件副本。

这属于看到本批结果后的探索性追加比较，没有重新拟合原预测器。P 的下降风险分数固定取 −P；D/幅度取原值，不在测试集上择优翻转符号。

当前值得注意的对照：

| 测试口径 | D | Centered interaction norm | KL |
|---|---:|---:|---:|
| 与下降 `−ΔM` 的 Spearman 相关 | 0.556 | 0.185 | 0.037 |
| 任意下降对其余 7 单元的 AP | 1.000 | 0.833 | 0.700 |
| 只在 2 降/2 升中区分方向的 AUROC | 1.000 | 0.750 | 0.500 |
| 唯一超过 5 pp 下降事件的 AP | 0.500 | 0.333 | 1.000 |

因此 D 的负向排序存在比部分幅度读出更好的线索，但不是所有指标都优于 KL/JS。两次测试更新中，下降全部来自 U35，上升全部来自 U34，无法据此证明 update 内 Skill 方向区分或稳定增量；原来学习模型未优于基线的结论不变。

归档目录：[observation-audit-v1](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1/reports/2026-09-12-observation-audit-v1)。`all_supported.csv`、`heldout.csv` 保留完整数值；`source/` 保存生成时的关键代码。

## 2. 已实现的代码支持

| 讨论项 | 实现 | 与旧实验的隔离 |
|---|---|---|
| 多次 gold continuation | 每个 anchor/arm 可配置多个 seeds；按 seed 配对三臂和 old/new | 旧配置仍为 1 evidence + 1 gold |
| 动态 Skill/anchor 数量 | 由冻结 anchor 文件生成完整 job 集，不再硬编码每端点 1,200 条 | 旧 identity/trajectory ID 不变 |
| 多个 context / 更多 Skill | 显式 `anchor_sets`，按 `(skill_id, context_id, phase)` 汇总训练信号与效用 | general Skill 在不同 task 不会直接合并 |
| 自然支持覆盖审计 | 从 pre-update coverage/all-anchors 构造新支持建议，按不同 games 轮转选样 | 不读取 post-update utility，不改 Router/Skill 内容 |
| 大窗口 endpoint 测量 | 支持 `--start-update` 与 `--update`，另存 `window_signals/` | 不替换原单次 update 的 P/D |
| 时间隔离 | 验证主窗口不重叠、开发/测试不共享端点；保留 boundary window | 敏感性窗口不作为开发主样本 |
| 简单预测器 | 同一 baseline/categorical controls，每次只加入一个候选量 | 不重新训练旧 fast-v1 模型 |
| 阶段分析 | all、initial、early、middle、late 分别拟合/报告 | 不将 all 与其组成阶段拼成独立样本 |
| 多训练路径 | 独立 root、`rl_path_id`、run prefix、sampler seed、数据文件和短 Ray 临时路径 | 防止不同分支覆盖训练与 dataset 产物 |
| 扩展训练 batch | 新协议显式 opt-in 可增加环境 games/rollouts；optimizer minibatch 仍固定 32 decisions | 旧 32-rollout 训练检查保持原行为 |
| 运行顺序 | 每 update 保存训练证据；窗口 signals→prediction→目标端点 gold | 草案协议不可启动；只评测已注册端点 |

没有在本次更换模型、修改学习率、扩大 Bank 内容或重选已经发表在 Phase2 报告中的样本。

## 3. 多重复效用估计的具体含义

配对键现在包含 `continuation_seed`；缺少任一 arm 或 old/new 配对会报错，不能 inner join 后悄悄丢弃。

顺序为：同一 anchor、同一 seed 的 ORIGINAL−PLACEBO → 同 seed 的 old/new difference → anchor/repeats 在 game 内平均 → games 等权平均。增加重复不会将一个 anchor 计为四个独立 anchors。

当有多个 continuation seeds 时，CI 使用 paired game bootstrap，并在选中的 game 内对每个 anchor 的配对 continuation 重采样。所有臂和端点先配对再抽样；不拆开重采样。此为新协议的重复评测估计器，仍需报告有限 games、共同支持集和随机生成噪声的边界。

只有一个 seed 时保留旧 game-bootstrap 的计算与随机数流。实际复核：旧 130 行效用统计与 95% CI 完全一致；U31–U35 共 200 行信号汇总表也逐值一致。

## 4. 大窗口的 reward direction：明确命名，不冒充原单步公式

窗口从 `T` 到 `T+K`。代码保留每个单步 update 的原始 P/C/D，同时新增窗口读出：

\[
\delta_{T\to T+K}^{s}(z)=
[\log\pi_{T+K}^{O}-\log\pi_T^{O}]
-[\log\pi_{T+K}^{P}-\log\pi_T^{P}].
\]

本次实现的窗口方向模式显式命名为 `start_batch_endpoint_projection`：

- 使用第一个更新 batch（`T+1`）实际记录的 state、response tokens、`π_T` 与 advantage；
- old ORIGINAL 使用该 batch 的 live log-probs；新端点 `π_{T+K}` 在同一旧 response 上重新 teacher-force；
- O/P 四条件按同一旧 token prefix 对齐，再代入 P/C/D 的投影与 gate 结构；
- 不拿窗口最后一个 batch 的 token 或 advantage 与起点 batch 错配；
- 不把 `D_T + ... + D_{T+K−1}` 当作窗口 D；
- window commit 明确记载 end ORIGINAL 来自 endpoint replay，而非末次训练 batch 的 live 前向。

它解释的是“累计策略变化相对窗口起始 reward direction 的交互投影”，不是整个窗口所有局部梯度的统一方向。窗口内后续 learning direction 可能改变，因此这是明确标记的扩展读出，不能省略与原 idea 单次 update 公式的区别。正式新实验冻结协议时需要一并确认这一解释。

## 5. 新实验模板与尚未冻结的内容

模板：[extended_direction_v2.template.json](/home/wangyifan/skill-RL/SkillRL/phase2/config/extended_direction_v2.template.json)。其 `status=draft`，不能直接启动。

当前模板中的可调整建议：gold 4 个独立 seeds、evidence 2 个独立 seeds，主窗口间隔 5，短窗口 1/3 为可选敏感性；LR/KL 先保持不变。预测使用旧效用、旧效用标准误和对照更新范数，再逐一添加 centered norm/P/D/KL；同时控制 Skill/context/phase 身份。每个阶段独立要求至少 4 个开发窗口和 24 个有支持开发单元，否则只保存 direct scores，不勉强拟合。

这些数值是工程模板建议，不是宣称已经通过校准证明充分，也不是已启动的正式实验参数。以下字段故意留空：

1. 新分支共同起点与完整 optimizer 恢复来源；不能把只有 model-only 的旧 seed 当作具备完整续训状态。
2. 实际需要的独立训练分支、开发/测试窗口数和固定停止预算。
3. 新 pre-update 覆盖审计后确定的 task types、Skill/context 清单、anchor 文件与哈希。
4. 正式 gold/evidence 重复数及窗口方向解释的确认。

增加训练/评测规模会明显增加磁盘、全词表前向与环境交互成本；当前 runner 保留逐 update 证据和既有 250 GiB 磁盘保护线，不会自动删除旧 checkpoint。长程正式采集前仍需检查归档预算。跨训练路径的汇总/留出结论也必须等实际路径数据齐备后单独验证，本次没有虚报跨 seed 结果。

## 6. 使用入口

在 `conda activate skill-RL` 后进入 `/home/wangyifan/skill-RL/SkillRL`。

自然覆盖建议入口：

```bash
python -m phase2.coverage_audit --help
```

输入是 `phase1.build_all_first_invocation_anchors` 从选定 pre-update policy 的自然轨迹产生的 coverage 文件；可同时输入多个 contexts。该工具不自行运行模型。缺少新 Skill 的 placebo 时标记 `placebo_ready=false`，可用现有 `phase1.create_payload_placebo` 创建后再冻结。

协议准备/训练入口（仅在另行完成并冻结新配置后）：

```bash
python -m phase2.prepare --config /absolute/path/to/frozen-new-protocol.json
python -m phase2.launch_training --protocol /absolute/path/to/new-root/protocol.json --update UPDATE --dry-run
python -m phase2.run_extended --root /absolute/path/to/new-root
```

单独测量窗口时，`phase2.measure`、`phase2.aggregate`、`phase2.window_forecast` 均接受 `--start-update T --update END`。测量与预测必须发生在 END 的 gold 打开之前。

`phase2.window_report` 将新结果写入新 root 的 `window_metrics/` 与 `reports/window-results.md`，不写入旧 fast-v1 报告。原 `phase2.run_fast` 继续服务旧协议，新协议必须使用独立 runner。

## 7. 验证与限制

已完成 CPU 检查：

- 189 项 Phase1/Phase2 单元与回归测试通过（其中新增 22 项多重复、窗口、支持、预测、调度和观测量比较测试）；
- 旧 130 行效用/CI 逐值复算一致，旧 200 行信号汇总逐值复算一致；
- 全部 Phase2 Python 文件语法检查及修改的 shell launcher 语法检查通过；
- 原始评测/预测/信号文件未重写，新增分析仅写入独立归档与报告追加章节。

尚未执行新配置下的真实 GPU forward/backward 或更大 batch 的 RL smoke；代码测试通过不等于新实验已经运行。当前完成的是实现、旧结果复核和可讨论的观测量表格。

验证记录及代码 SHA-256：[verification.json](/home/wangyifan/skill-RL/SkillRL/artifacts/code_checks/phase2-expanded-v2-20260912/verification.json)。
