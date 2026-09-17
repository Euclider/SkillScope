# Qwen3.5 全面首次调用 Skill 效用评测：设置改进、已有结果与运行进度

> 日期：2026-09-03  
> 状态：正式三臂评测运行中，本文不包含尚未完成的最终效用统计  
> 对照报告：`2026-08-29-first-invocation-skill-utility-evaluation-results.md`  
> 正式评测 run：`qwen35-clean-all-skill-seeds101-202-v1`

## 1. 阶段性结论

上一版实验已经证明了首次调用点三臂干预链路可行：固定调用前 prefix 后，仅替换目标 Skill 的 prompt payload，能够观察 action flip 和累计 reward contrast，并将 Skill 语义效用与 prompt nuisance 分开。但上一版只有一个 `pic_001`、一个 RL seed、8 个 games，而且全部 anchor 都在 step 0，因此只能作为机制验证。

本版将验证范围扩展到 Qwen3.5-4B、两个独立 RL seeds、固定的 update 10/20/30、31 个 `valid_unseen` clean games，以及所有 18 个 clean 候选 Skill 的自然调用覆盖审计。根据预注册阈值，4 个 Skill 获得充分支持并进入三臂评测，其中 3 个包含 step > 0 的真实中途 anchor。正式效用结果仍在生成，目前不能提前判断边际效用变化方向或 harmful sign flip 数量。

## 2. 相比 2026-08-29 版本的主要改进

| 维度 | 2026-08-29 版本 | 当前版本 | 改进意义 |
|---|---|---|---|
| Policy | Qwen2.5-1.5B-Instruct | Qwen3.5-4B | Base 已具有更稳定的 ALFWorld 行为能力，减少“只是在学合法动作格式”的混杂 |
| Task type | pick-and-place | clean | 同时包含搜索、获取、清洗、状态推进和交付阶段 |
| Skill 范围 | 仅 `pic_001` | 审计 12 general + 6 clean-specific，共 18 个 | 从单 Skill 可行性转向 Bank 内自然可路由 Skill 的全面覆盖 |
| 可估计 Skill | 1 个 | 4 个 supported、14 个 unsupported | 未自然调用的 Skill 不被强制插入，避免不自然 state 干预 |
| Anchor 来源 | checkpoint-0 `valid_seen` | 公共 B0 的 `valid_unseen` | 使用独立未见 games 检验 game-level 泛化 |
| Game 数 | 8 | 31 | game-cluster 统计支持显著增加 |
| B0 源轨迹 | 24（8 games × 3 seeds） | 93（31 games × 3 seeds） | Router coverage 和 state 支持更充分 |
| 固定评测 anchors | 24 | 每个 supported Skill 50，共 200 | 先按 game round-robin，防止少数 game 的重复采样主导 |
| 调用阶段 | 24 个全部 step 0 | step 0–26，包含 initial/early/middle/late | 真正验证 prefix replay 和中途 intervention |
| RL seed | 仅 seed 101 | seed 101、202 | 初步检验随机训练路径上的方向复现与异质性 |
| Checkpoint | update 0/1/2/3/4/5 | B0 与每个 seed 的 update 10/20/30 | 参数更新累计尺度从最多 160 条训练 rollout 扩展到 320/640/960 条 |
| Checkpoint 选择 | 连续 1–5 updates | 预先固定 10/20/30 | 不依据 Skill flip 或性能曲线事后挑点 |
| 模型数 | 6（共同单 seed 训练链） | 7（共同 B0 + 两个 seed 各 3 个） | 可比较同一 update milestone 的跨 seed 表现 |
| PLACEBO | 等 token 无关文本，但 section header 硬编码为 Pick And Place | 保留原 Skill 的 section header 与 bullet 模板，仅替换为等 token 无关内容 | 更严格控制模板、文本存在和 token 长度效应 |
| 统计层级 | game cluster bootstrap | seed 内 game cluster；跨 seed 按 `RL seed → game → generation seed` | 避免将同一 game 的重复 rollout 当作独立样本 |
| 计划总量 | 432 条三臂分支轨迹 | 4,200 条三臂分支轨迹 | 支持 per-Skill、per-update、per-seed 和调用阶段分析 |

## 3. 保持不变的核心因果协议

当前版本没有改变上一版最关键的归因逻辑：

1. Anchor 来自公共 B0 在冻结 full Bank 下的自然 rollout；
2. 对每条源轨迹、每个 Skill，只取 Router 第一次自然选择该 Skill 的位置；
3. 触发点之前的环境 action prefix 完整重放，并校验 observation、admissible actions 和 task；
4. Router descriptor bank 在 ORIGINAL、PLACEBO、NULL 三臂中完全相同；
5. ORIGINAL 注入原 Skill，PLACEBO 注入同模板、同 token 数的任务无关内容，NULL 保留路由事件但不给 policy Skill payload；
6. 从触发位置开始自由推理，之后若 Router 再次选择目标 Skill，继续执行同一 arm 的 payload 干预；
7. 三臂使用相同 generation seed schedule；
8. 累计 suffix reward 决定 action change 是有益、无效还是有害，action flip 本身不承担方向标签。

主要估计量仍为：

- `U_sem = R(ORIGINAL) - R(PLACEBO)`：Skill 任务相关语义的边际效用；
- `U_total = R(ORIGINAL) - R(NULL)`：Skill 内容相对空 payload 的总效用；
- `U_prompt = R(PLACEBO) - R(NULL)`：模板中存在等长无关文本的 nuisance；
- `Delta U_sem(seed, update) = U_sem(seed, update) - U_sem(B0)`：相同 B0 anchor 支持集上的纵向语义效用变化。

可靠 harmful sign flip 的判定也保持严格：只有 B0 的 game-cluster 95% CI 严格为正，且 post-RL checkpoint 的 CI 严格为负，才标记为可靠 harmful flip。点估计变号、action flip 或 CI 跨 0 都不能单独满足该定义。

## 4. 已完成的训练与 checkpoint 基础

Seed 101 和 Seed 202 均从同一个 Qwen3.5-4B B0 出发，使用相同训练数据、冻结 Skill Bank/Router、GRPO 超参数和 4-way FSDP。每个 update 使用 8 games × 4 rollouts，即 32 条训练轨迹：

| Checkpoint | 每个 seed 的累计 update | 每个 seed 的累计训练 rollout | 当前状态 |
|---|---:|---:|---|
| B0 | 0 | 0 | 公共原始 Qwen3.5-4B |
| C1 | 10 | 320 | Seed 101/202 均已归档并验证 |
| C2 | 20 | 640 | Seed 101/202 均已归档并验证 |
| C3 | 30 | 960 | Seed 101/202 均已归档并验证 |

六个 post-RL model-only checkpoint 均约 9.1GB，包含 427 tensors，配置识别为 `qwen3_5_text`。评测只使用预先固定的 C1/C2/C3，不读取 Skill utility 后再挑选 checkpoint。

Seed 303 暂不进入本轮统计：它使用 8-way FSDP，目前安全暂停在完整 update 12，保留 update 11/12 两个可恢复 optimizer checkpoint和 update 10 model-only C1。将其排除是因为尚无完整 C2/C3，而不是因为看到了它的 Skill utility。

## 5. 已有结果一：B0 在 valid_unseen 的正常 rollout 能力

正式 anchor 来源为 31 个 clean `valid_unseen` games，每个 game 使用 generation seeds 1101、2202、3303，共 93 条 full-bank B0 轨迹。这里尚未做 Skill payload 对照，因此只能描述基础 policy 与 Router 覆盖，不能解释为 Skill 边际效用：

- episode success：52/93，即 55.91%；
- 平均轨迹长度：19.03 steps；
- 每条轨迹平均自然调用 3.77 种不同 Skill；
- `cle_006` 在 93/93 条轨迹中出现，并且第一次调用全部位于 step 0；
- `cle_003`、`gen_002`、`cle_004` 分别覆盖 89、85、83 条轨迹；
- 所有 93 条轨迹的第一个 Router 选择均为 `cle_006`。

55.91% 是 full-bank B0 的来源轨迹成功率，不是 ORIGINAL/PLACEBO/NULL 的效用差，也不能与旧版 Qwen2.5-1.5B 的三臂 success 直接比较。

## 6. 已有结果二：全 Skill Router coverage audit

Skill 是否进入效用评测只依赖自然调用次数和不同 game 覆盖，不读取 reward。预注册门槛为至少 30 个自然首次调用 occurrence、至少 10 个不同 games；通过后按 game round-robin 固定选取最多 50 个 anchors。

| Skill | 含义 | 自然 occurrence | Games | 不同 states | 首次调用 step | 固定 anchors | 阶段分布 initial/early/middle/late |
|---|---|---:|---:|---:|---:|---:|---:|
| `cle_006` | Use Location Priors | 93 | 31 | 31 | 0 | 50 | 50 / 0 / 0 / 0 |
| `cle_004` | Systematic Container Sweep | 83 | 28 | 35 | 1–5 | 50 | 0 / 46 / 4 / 0 |
| `gen_002` | Immediate Acquisition | 85 | 29 | 49 | 1–25 | 50 | 0 / 28 / 21 / 1 |
| `cle_003` | Sink First for Cleaning | 89 | 30 | 60 | 2–26 | 50 | 0 / 29 / 18 / 3 |
| `cle_002` | Pick Before You Wander | 1 | 1 | 1 | 12 | 0 | unsupported |
| 其余 13 个候选 | 其他 general/clean-specific | 0 | 0 | 0 | 未调用 | 0 | unsupported |

因此，“全面 Skill 评估”在本轮的准确含义是：对 18 个候选全部做自然路由支持审计，对其中 4 个支持充分的 Skill 做三臂因果评估，对另外 14 个报告 unsupported。它不是人为把 18 个 Skill 逐个塞进不匹配的 state。

相较上一版，最关键的实质扩展是 `cle_004`、`gen_002`、`cle_003`：它们的 anchor 位于 step 1–26，需要真实重放非空 prefix，能够检验中途 Skill 内容变化对剩余累计 reward 的影响。

## 7. 已有结果三：三臂 preflight

正式启动前对一个 step 1 的 `cle_004` anchor 跑通三臂：三条轨迹均通过 prefix replay 和触发 Router 校验；ORIGINAL/PLACEBO 的 Skill payload token 数相同，NULL 的目标 payload 注入次数为 0。

该单 anchor 上 ORIGINAL suffix return 为 10，PLACEBO 和 NULL 均为 0；ORIGINAL 与 NULL 的首动作相同，而 PLACEBO 首动作不同。这只能证明新 runner 能处理非 step-0 prefix、持续干预和三臂分叉。由于样本数为 1，不把 `+10` 作为 `cle_004` 有益或发生效用漂移的统计证据。

## 8. 与上一版已有结论的关系

上一版 `pic_001` 实验得到的关键结果包括：

- ORIGINAL 相对 PLACEBO 的首动作 flip rate 为 41.67%–70.83%；
- `U_sem` 随 update 1–5 非单调变化；
- Base → checkpoint 2 的 `Delta U_sem` 为 +12.50 pp，game-cluster 95% CI 为 `[4.17, 25.00]`；
- Base → checkpoint 5 的聚合 `Delta U_sem` 为 0，但内部同时存在正负 anchor turnover；
- RL 后 invalid action 大幅减少，但这不等价于 Skill 语义效用单调改善；
- 尚未发现“B0 可靠为正、post-RL 可靠为负”的 harmful sign flip。

当前实验不是简单重复这些数字，而是在检验它们能否扩展到：更强模型、非初始 state、多个 Skill、更大累计 update、更多 unseen games，以及两个独立训练 seed。尤其需要区分三种可能结果：

1. 两个 seed 在同一 Skill/update 上出现同方向 `Delta U_sem`，支持随机训练路径下的初步复现；
2. 两个 seed 方向不同，说明 Skill-policy 交互具有显著训练路径异质性；
3. 聚合变化接近 0，但 anchor/game 层面存在正负 turnover，说明均值掩盖了 context-specific 漂移。

## 9. 当前正在运行的正式评测

正式矩阵为：

```text
7 models × 4 supported Skills × 50 anchors × 3 payload arms = 4,200 branch trajectories
```

七个模型为公共 B0，以及 Seed 101/202 各自的 update 10/20/30。每个 checkpoint 单独占用一张 GPU 并行运行；第八张 GPU 不属于正式七模型矩阵。截至 2026-09-03 09:17 CST：

- 7 个 checkpoint evaluator 均在运行；
- 已写入 43/4,200 条 index records；
- 当前日志未发现 traceback 或 CUDA OOM；
- 每个输出均可断点续跑，完整 trajectory 按 checkpoint/Skill/anchor/arm 独立归档；
- 预计在无抢卡和 OOM 情况下，正式 rollout、统计与报告总计还需约 14–20 小时。

当前 43 条是运行进度，不构成中间效用结论。评测结束前不根据早期 reward 调整 Skill、anchor、checkpoint 或停止规则。

## 10. 最终将报告的统计量

评测完成后将生成：

1. 每个 Skill × checkpoint 的 ORIGINAL/PLACEBO/NULL success 与 return；
2. `U_sem`、`U_total`、`U_prompt` 及 game-cluster 95% CI；
3. Seed 101/202 在 update 10/20/30 相对 B0 的 `Delta U`；
4. positive/negative/zero anchor 比例和 point-level turnover；
5. initial/early/middle/late 分层结果；
6. 首动作 flip、完整 suffix divergence 和首次分歧位置；
7. 两个 RL seed 的方向一致率、差异范围和分层 bootstrap；
8. 可靠 harmful sign flip、仅点估计变号和不确定样本的分离统计；
9. supported、unsupported、uncertain Skill 比例；
10. Skill utility 与全局 policy success/invalid-action 改善是否解耦。

## 11. 当前限制

- 只有 clean 一个 task type，尚不能外推至 pick-and-place、heat、cool 等任务；
- 18 个候选中只有 4 个被当前确定性 Router 充分调用，说明主要瓶颈是 Router 可路由性，而不是 Bank 总条目数；
- 两个 RL seeds 只能提供初步复制证据，无法精确估计 seed-level population variance；
- 固定 B0 anchor 回答的是相同 state 支持集上的效用漂移，不等价于每个 checkpoint 自己的 on-policy Skill utility；
- terminal reward 仍较稀疏，很多 Skill/checkpoint 的区间可能跨 0；这应报告为 uncertain，而不能解释成“没有 state-level 变化”；
- Seed 303 尚未完成，当前报告不使用其结果。

## 12. 归档索引

- 上一版结果：`/home/wangyifan/skill-RL/2026-08-29-first-invocation-skill-utility-evaluation-results.md`
- 当前预注册协议：`SkillRL/phase1/config/qwen35_clean_all_skill_utility_protocol.json`
- B0 valid-unseen 来源轨迹：`SkillRL/artifacts/evaluations/qwen35-b0-clean-valid-unseen-anchor-v1/`
- 全 Skill coverage 与 anchors：`SkillRL/artifacts/anchors/qwen35-clean-all-skill-seeds101-202-v1/`
- 等 token PLACEBO：`SkillRL/artifacts/controls/qwen35-clean-all-skill-seeds101-202-v1/`
- 正式评测与完整轨迹：`SkillRL/artifacts/evaluations/qwen35-clean-all-skill-seeds101-202-v1/`
- Seed 303 暂停清单：`SkillRL/artifacts/manifests/qwen35-clean-formal-seed303-u30-8gpu-s1-20260902-paused-update12.json`

协议文件 SHA-256：

```text
b554a98e26e2b0b8d22eecf89cc18b7b222c8461d95464fdb3ce370dda9254a7
```
