# Phase2 排序验证：GPU 等待队列与 U35 起点评估

日期：2026-09-12。本文记录新队列，不替换 Phase1 或已有 Phase2 结果。

2026-09-12 13:12（北京时间）已启动后台 supervisor，PID 为 `1255047`。启动时 8 卡均被占用，状态为 `waiting_for_idle_gpu`；已有 198 项回归测试通过，并完成 31 个 game 文件及单 game reset 检查。此行是启动记录，实时进展见下方状态路径。

## 当前范围

本次授权执行 GPU 监控及扩展评估。队列先采集新的 pre-update 自然调用支持集与多重复三臂基线，不会自动启动尚未冻结预算的新 RL 窗口，也不会把旧测试数据当作新的前瞻性测试。

起点为最新已完整保存的 Seed303 U35，不按 Skill 效用或预测指标挑选。新锚点仅用于 U35 起点及之后的新窗口；不能反过来重选 U30–U35 已有测试样本。

## 已排定工作

1. 使用原冻结 Bank 与 state Router，在全部 31 个 `valid_unseen clean` games 上各采样 4 条 full-bank 自然轨迹，共 124 条。采样 seeds 为 61011、61021、61031、61041；温度 0.4、top-p 1、最多 30 步、每动作 64 tokens、history length 2。
2. 审计 clean 全部 18 个候选 Skill（12 general + 6 specific）的自然首次调用位置，包括 initial/early/middle/late。支持要求至少 30 次自然出现、15 个不同 games；每个支持 Skill 最多选 50 anchors，按 game 确定性轮转。未达到阈值的 Skill 原样报告 unsupported，不强制调用，不根据 reward 筛选。
3. 生成保持模板/分词长度的 PLACEBO，冻结锚点与 payload 哈希。在同一 U35 上执行 ORIGINAL / PLACEBO / NULL；prefix 精确 replay，后续自由 rollout，目标 Skill 再次被路由时持续施加同一干预。
4. 每个 anchor/arm 使用 evidence seeds 51011、51021 与独立 gold seeds 52011、52021、52031、52041。每个 anchor 共 18 个 suffix；具体总数在自然支持审计后确定。如仍为 4 Skills × 50 anchors，则为 3,600 个 suffix，而不是声称重复次数增加了独立 game/update 数。
5. 自动归档完整步骤、prompt/response、路由结果、首次调用位置、reward、唯一 Skill 数量、index、分片日志及覆盖/基线报告。完成后停在“等待新窗口协议”，不把仅有基线的结果解释成 ΔM 或预测成功。

`phase1.eval_skill_margin` 的 `--skill-id cle_006` 仅是该旧 CLI 要求的归档标签；本次 condition 只有 `full_bank` 且开启 state routing，不强制每步调用 cle_006。三臂评估则按每个自然支持 Skill 分别干预。

新起点/多次采样可能提高自然 Skill 覆盖，但不保证一定出现更多支持 Skill。当前阶段仍是 clean 的 31 个不同 unseen games，不是已扩展到其他 task types 或增加了独立更新窗口。

## 排序是主要验证目标

后续关注同一更新窗口内哪些 Skill 应优先检查/更新，不要求精确预测效用值，也不要求先拟合回归器。

- 主目标：语义效用下降，gain = `max(0, −ΔM_sem)`；不要求正→负 sign flip。
- 直接排序：D、未门控 D、−P、centered/raw interaction norm、KL/JS；与随机期望、旧效用低优先和通用对照更新幅度比较。
- 预算：k=1/2 及候选池前 25%/50%；各方法使用同一支持候选池。
- 主要统计：Precision@k、Recall@k、下降量覆盖率；辅助为窗口内 Spearman/Kendall。精确数值的 MAE 不再作为必要成功条件。
- 并列分数按并列组内随机选择的期望计分，不读取 gold 打破并列。无下降事件的窗口单独统计，Recall/下降量覆盖率记为未定义，不悄悄计为零。
- 点估计下降和超过 5 pp 下降分别报告，保留配对估计噪声/CI 的限制；`|ΔM|` 的变化审计作为另一目标单独报告。
- 新 post-update rollout 只用于离线检验排序；分数和预算必须先于目标端点 gold 固定。跨窗口/训练路径汇总需要之后实际采集，不能将所有窗口混池排名冒充当前 policy 的 Skill 选择能力。

以上排序口径已记录到本队列，完整窗口排序分析要等新增端点及信号后实施；本次基线报告不虚报这部分结果。

## GPU / 磁盘监控

每 30 秒采样 GPU 显存与利用率。只有连续两次满足显存占用 <1,500 MiB、利用率 <10% 的卡才可派发；启动前再次检查。每张卡最多一个本队列 worker，使用跨队列 advisory lock。该锁只能协调采用它的本项目进程，不能阻止其他用户临时启动任务，因此仍保留共享 GPU 竞争的风险。

8 个逻辑分片与物理卡数解耦：空出一张即可运行，后续更多卡空闲时增加并行。利用率 0% 但占有大量显存的卡不算空闲；不停止、驱逐或修改其他用户任务。

磁盘低于 250 GiB 时不再派发新分片，已有分片完成后继续等待；不自动删除任何旧 checkpoint。新队列通过只读模型路径别名复用 U35 的已有权重，不额外复制约 19 GB FP32 模型。训练和新增全词表 logits 的长期存储预算仍需另行确定。

异常会保留已完成轨迹和日志，停止而不是无限重试未知错误。受控停止只终止本队列创建的 worker 进程组。恢复时检查冻结输入、代码与索引身份。

## 路径与恢复

- 队列归档：[ranking-preparation-v1](/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-ranking-preparation-v1)
- 实时状态：该目录 `status.json`、`gpu_watch_latest.json`
- 历史采样：`gpu_watch.jsonl`；派发记录：`allocations.jsonl`
- 后台主日志：`supervisor.log`；worker 日志：`logs/`
- 冻结配置：`queue.json`；代码与输入哈希：`queue_manifest.json`；源代码副本：`source.zip`
- 完成后：`support/coverage.csv`、`reports/baseline-report.md`、`reports/baseline_margins.csv`、`reports/paired_anchor_margins.parquet`，并在本文末尾追加基线结果。

当前是否已派发/完成应以实时状态文件为准，不以本文的静态描述判断。

在 `conda activate skill-RL` 后进入 `SkillRL`，恢复入口为：

```bash
python -m phase2.queued_preparation --root /home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-ranking-preparation-v1 --detach
```

后台持有唯一 supervisor lock，重复启动不会并行执行同一个队列。若源代码或冻结输入发生变化，会拒绝无记录地续跑，需要先归档变更。

## 后续仍需确定

增加独立更新窗口/路径才会补足 update 层面的样本；本次增加 continuation repeats 不能替代它。更大窗口的具体数量、独立路径及 checkpoint/logits 保留预算仍待冻结。目前窗口 D 的已实现定义是“累计 endpoint interaction 对起始 batch reward direction 的投影”，应与原单步 D 区分；不能把各步 D 简单相加。


<!-- completed-preupdate-baseline -->

## 已完成：U35 新支持集覆盖与多重复效用基线

这是一份 pre-update 准备报告，不包含新 RL 窗口、ΔM 或排序预测结论；旧实验保持不变。

自然轨迹 124 条，覆盖 31 个 game；平均每条调用 3.726 种 Skill。

## 自然支持审计

| context_id   |   early |   games |   initial |   late |   middle |   natural_occurrences | placebo_ready   |   selected_anchors |   selected_games | skill_id   | supported   |
|:-------------|--------:|--------:|----------:|-------:|---------:|----------------------:|:----------------|-------------------:|-----------------:|:-----------|:------------|
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_001    | False       |
| clean        |      26 |      29 |         0 |      1 |       23 |                    84 | True            |                 50 |               29 | gen_002    | True        |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_003    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_004    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_005    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_006    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_007    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_008    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_009    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_010    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_011    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | gen_012    | False       |
| clean        |       0 |       0 |         0 |      0 |        0 |                     0 | False           |                  0 |                0 | cle_001    | False       |
| clean        |       0 |      11 |         0 |      0 |        0 |                    18 | False           |                  0 |                0 | cle_002    | False       |
| clean        |      20 |      31 |         0 |      0 |       30 |                   111 | True            |                 50 |               31 | cle_003    | True        |
| clean        |      50 |      31 |         0 |      0 |        0 |                   114 | True            |                 50 |               31 | cle_004    | True        |
| clean        |       0 |       7 |         0 |      0 |        0 |                    11 | False           |                  0 |                0 | cle_005    | False       |
| clean        |       0 |      31 |        50 |      0 |        0 |                   124 | True            |                 50 |               31 | cle_006    | True        |

## 独立 gold 基线（成功率及效用均为百分点）

| purpose   | skill_id   | context_id   | phase   | control   |   anchor_count |   game_count |   continuation_repeats |   original_success |   control_success |   semantic_or_total_margin |   ci_low |   ci_high |
|:----------|:-----------|:-------------|:--------|:----------|---------------:|-------------:|-----------------------:|-------------------:|------------------:|---------------------------:|---------:|----------:|
| gold      | cle_003    | clean        | all     | placebo   |             50 |           31 |                      4 |             53.226 |             6.855 |                     46.371 |   31.048 |    61.290 |
| gold      | cle_004    | clean        | all     | placebo   |             50 |           31 |                      4 |             33.871 |            27.419 |                      6.452 |   -1.210 |    14.919 |
| gold      | cle_006    | clean        | all     | placebo   |             50 |           31 |                      4 |             45.968 |            44.355 |                      1.613 |   -2.016 |     5.645 |
| gold      | gen_002    | clean        | all     | placebo   |             50 |           29 |                      4 |             60.776 |            57.759 |                      3.017 |    0.000 |     7.759 |

全部 evidence/gold、ORIGINAL−PLACEBO / ORIGINAL−NULL 和各阶段数值见 baseline_margins.csv。多个 continuation repeats 在 anchor/game 内配对，不作为独立 games 或独立更新窗口。

后续主分析为同一更新窗口内的 Skill top-k 排序；不以回归 MAE 作为必要成功条件。新窗口预算、完整训练状态和存储策略仍需单独冻结，当前队列不会自动训练。
