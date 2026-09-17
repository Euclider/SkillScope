# SkillRL 逐状态 Skill 路由改动与 RL 最小验证协议

> 日期：2026-08-25  
> 状态：代码与环境准备完成，RL 尚未启动  
> 本文只记录逻辑、参数、测试步骤、归档和预期统计量，不包含任何 RL 结果。

## 1. 本轮验证要回答的问题

在 Skill Bank 和 Skill Router 都冻结不变时，仅通过真实 RL 更新 policy model 参数
`θ`，同一个 state `s` 与同一个被选 Skill `z` 的策略行为及 Skill 边际效用是否会发生
可测量的偏移。

需要同时观察两个层次：

1. 固定状态层：在完全相同的 state、admissible actions 和 Skill 条件下，RL 前后
   action、action distribution 是否变化，例如 action flip 和 JS divergence。
2. 完整轨迹层：一个 Skill 在 full-bank rollout 中被逐步选择并调用后，相对于移除该
   Skill 的 matched rollout，其 episode success margin 是否随 RL checkpoint 变化。

本实验不会把“某个 Skill 在任意 state 下都必须有用”作为假设。只有该 Skill 在
full-bank 条件下被 Router 实际选中的 state/trajectory，才进入该 Skill 的归因集合。

## 2. 原始仓库逻辑与本实验改动

### 2.1 原始 SkillRL 的 ALFWorld Skill 使用方式

原始仓库不是把全局 55 个条目全部塞给每个 game，但其粒度确实是 **game/episode
级 bundle**：

1. environment reset 后，根据 game 的 task description 做一次任务类型识别和 Skill
   retrieval。
2. 原始示例配置取前 6 个 general Skills、当前任务类型的全部 task-specific Skills，
   以及前 5 个 common mistakes。
3. 检索结果保存为该 episode 的 `retrieved_memories`。
4. 初始 step 不注入 memory；从后续 step 开始，每一步重复向 policy prompt 注入同一
   份 bundle。
5. 原始训练脚本可以根据验证失败动态更新 Skill Bank。

因此，同一 game 内不同 state 没有明确的 `selected_skill_id`。即使 prompt 中存在多个
Skill，也无法知道 policy 在某一步究竟使用了哪一个 Skill，不能形成可靠的步骤级 Skill
效用归因。

原始默认 bundle 大小如下：

| 任务类型 | general | task-specific | common mistakes | 每步注入条目总数 |
|---|---:|---:|---:|---:|
| pick_and_place | 6 | 5 | 5 | 16 |
| look_at_obj_in_light | 6 | 5 | 5 | 16 |
| clean | 6 | 6 | 5 | 17 |
| heat | 6 | 5 | 5 | 16 |
| cool | 6 | 5 | 5 | 16 |
| examine | 6 | 6 | 5 | 17 |

### 2.2 本实验的改动：task bundle + frozen state router

本实验保留 task-filtered bundle，但在 policy action 前增加一个冻结的步骤级 Router：

```text
完整冻结 Skill Bank
        ↓
根据 task 形成候选 bundle
  = 全部 general + 当前 task 类型的全部 task-specific
        ↓
Frozen Router(state, task, admissible actions, history, Skill descriptions)
        ↓
selected_skill_id：每一步恰好选择 1 个 Skill
        ↓
只把该 Skill 注入 policy prompt
        ↓
policy model 生成本步 action
```

具体变化：

- Router 从第 0 步开始工作，因此初始 action 也有明确的 selected Skill。
- 每执行一个 action 并获得新 observation 后，下一步重新路由，不再整局重复同一个
  Skill bundle。
- Router 是确定性的规则路由器，没有可训练参数；RL 期间保持完全冻结。
- Router 只使用 agent 可见信息：task description、当前 observation、admissible
  actions、过去的 action/observation history 和 step index。
- Router 不读取 reward、`won`、未来 observation 或其他 success 信息。
- Router 使用 Skill 的 `title`、`principle`、`when_to_apply` 描述与可观察 state phase
  做匹配并记录完整分数；稳定的文档顺序只用于最终 tie-break。
- Policy prompt 只看到最终选中的一个 Skill，不看到候选 bundle 和路由分数。
- Skill Bank 的动态更新关闭，以免把 Bank 漂移与 policy 参数漂移混在一起。

Router 当前识别的 observable phases 包括 initial、visible target、holding target、
process available、processed recently、ready to place、closed container、search、loop、
lamp available 和 multiple targets。

### 2.3 为什么采用独立冻结 Router，而不是让 RL policy 自己选择 Skill

如果 Skill 选择也由正在被 RL 更新的 policy 完成，checkpoint 间观察到的变化会同时
包含两部分：

1. policy 对同一 Skill 的使用方式改变；
2. policy 的 Skill 选择机制改变。

这会使“固定 state 和 Skill 后，policy 更新是否改变 Skill 效用”难以识别。当前版本
把路由器冻结，使实验的主要变化来源限制为 policy 参数 `θ`。Router 自身是否应该学习，
可作为后续独立实验，而不进入本次最小验证。

## 3. Skill Bank 与每步候选数量

使用原始完整 Bank：

`/home/wangyifan/skill-RL/SkillRL/memory_data/alfworld/claude_style_skills.json`

Bank 内条目如下：

| 类别 | 数量 |
|---|---:|
| General Skills | 12 |
| pick_and_place | 5 |
| look_at_obj_in_light | 5 |
| clean | 6 |
| heat | 5 |
| cool | 5 |
| examine | 6 |
| Task-specific Skills 合计 | 32 |
| Skill ID 合计 | 44 |
| Common mistakes | 11 |
| Bank 总条目 | 55 |

本次 Router 的候选范围是 **全部 12 个 general Skills + 当前 task context 的全部
task-specific Skills**：

| Context | 每步候选 Skill 数 | 每步实际注入 Skill 数 |
|---|---:|---:|
| pick_and_place | 17 | 1 |
| pick_two | 17（复用 pick_and_place 的 5 个 Skills） | 1 |
| look_at_obj_in_light | 17 | 1 |
| clean | 18 | 1 |
| heat | 17 | 1 |
| cool | 17 | 1 |
| examine | 18 | 1 |

11 个 common mistakes 仍保留在完整 Bank 和 Bank hash 中，但当前版本不作为可路由
Skill；不会与正向 Skill 混为同一归因单位。

## 4. 冻结项、更新项与因果对照

### 4.1 RL 期间冻结

- 完整 Skill Bank 文件与 SHA-256；
- Router 代码、版本和打分逻辑；
- task → candidate bundle 规则；
- prompt template；
- ALFWorld action projection；
- 每个 matched evaluation 的 game、environment seed、evaluation seed、解码参数；
- Skill 条件构造规则。

### 4.2 唯一主要更新项

- Qwen2.5-1.5B-Instruct policy model 参数 `θ`，通过真实 GRPO/RL update 更新。

### 4.3 评估条件

- `full_bank`：使用冻结完整 Bank，Router 正常选择一个 Skill。
- `minus_skill`：只从候选集合移除目标 Skill，并在完全相同 state 上由同一 Router
  确定性选择 fallback Skill。
- `no_skill`：所有 Skill 禁用，prompt 不注入 Skill。

目标 Skill 只有在 matched `full_bank` trajectory 中至少被选择一次时才具备 episode
层归因资格。未被选择不能解释成“该 Skill 无效”。

## 5. 第一批真实 RL pilot 参数

第一批只跑一个短 pilot，验证真实 policy update、逐状态路由和归档链路，再决定是否进入
30-epoch、3-seed 正式实验。

| 项目 | Pilot 设置 |
|---|---|
| 模型 | `/home/wangyifan/model/Qwen2.5-1.5B-Instruct` |
| 环境 | ALFWorld `AlfredTWEnv` |
| RL 算法 | GRPO |
| update type | `real_rl` |
| RL seed | 101 |
| train games | 8 |
| validation games | 8 |
| rollout 数 | 每个 game 4 条 |
| epoch 数 | 5 |
| 最大环境步数 | 30 |
| history length | 2（仓库配置默认值） |
| train batch size | 8 games |
| validation batch size | 8 games |
| 最大 prompt 长度 | 4096 tokens |
| 最大 response 长度 | 512 tokens/step |
| policy learning rate | `1e-6` |
| PPO mini-batch | 32 trajectories |
| PPO micro-batch/GPU | 4 trajectories |
| train rollout sampling | temperature 1.0、top-p 1.0、do_sample=true |
| validation sampling | temperature 0.4、top-p 1.0、do_sample=true |
| KL loss | 开启，coefficient `0.01`，`low_var_kl` |
| KL in reward | 关闭 |
| invalid action penalty | 开启，coefficient `0.1` |
| gradient checkpointing | 开启 |
| rollout engine | vLLM |
| tensor parallel size | 1 |
| GPU memory utilization | 0.5 |
| GPU 数 | 2 |
| checkpoint cadence | 每 epoch 保存一次 |
| validation cadence | 每 epoch 一次 |
| validation before train | 关闭 |
| Skill dynamic update | 关闭 |

Pilot 的逻辑 checkpoint 序列为 base model/checkpoint 0，以及 epoch 1–5 的 checkpoint。
base model 本身作为 checkpoint 0，不需要复制一份模型后才能评估。

## 6. RL 验证执行步骤

### Step 0：启动前冻结与预检

1. 激活 `conda activate skill-RL`。
2. 验证 1.5B 模型、ALFWorld 数据、完整 Bank 数量和 hash、两张以上 GPU、依赖版本。
3. 用确定性样例验证 Router 能从 clean context 的 18 个候选中选中正确阶段 Skill。
4. 创建 immutable run manifest，记录仓库 commit、dirty worktree fingerprint、模型配置
   hash、Bank hash、protocol hash、Conda/Pip 和 GPU 信息。
5. 每个新实验必须使用新的 `PHASE1_RUN_ID`，禁止用相同 run ID 覆盖不同配置。

### Step 1：运行 5-epoch real-RL pilot

启动脚本：

`/home/wangyifan/skill-RL/SkillRL/examples/grpo_trainer/run_alfworld_phase1_step_router.sh`

预定命令：

```bash
cd /home/wangyifan/skill-RL/SkillRL
conda activate skill-RL
export ALFWORLD_DATA=/home/wangyifan/skill-RL/data/alfworld
export PHASE1_RUN_ID=phase1-step-router-pilot-seed-101
bash examples/grpo_trainer/run_alfworld_phase1_step_router.sh vllm
```

### Step 2：确认真实 update 确实发生

至少检查：

- 每个 training step 的 reward、return、advantage 汇总；
- `actor/grad_norm` 是否存在并非始终为零；
- optimizer/update 是否无 NaN/Inf；
- base model 与 checkpoint 的参数差值是否非零；
- checkpoint 之间的 L2、relative L2、max absolute delta；
- 不能把 zero-update 或只做 rollout 的运行误标为真实 RL。

### Step 3：审计逐状态路由与轨迹归档

对每个已归档 trajectory 检查：

- 每个正常 full-bank step 都恰好存在一个 `selected_skill_id`；
- `injected_skill_ids == [selected_skill_id]`；
- selected Skill 位于当步 `candidate_skill_ids`；
- Router version 在所有 checkpoint 中一致；
- Skill Bank hash 在所有 checkpoint 中一致；
- 多步轨迹中至少有一部分使用两个或更多不同 Skill，证明不是退化为 game-level 固定 Skill；
- prompt 中只包含当步选中的 Skill，不包含整个 bundle。

### Step 4：生成 pilot 路由与能力统计

运行 `phase1.summarize_step_routing`，形成 routing summary。若路由不变量失败、Bank/Router
发生漂移、没有任何多 Skill 轨迹，先停止并修正，不进入正式效用分析。

同时检查 policy 基础能力：success rate、invalid action rate、trajectory length、是否存在
退化循环，以及不同 context 的覆盖情况。

### Step 5：固定状态 action probe

1. 从 full-bank trajectory 中截取目标 Skill 被选中的 state。
2. 固定 state、task、history、admissible actions 和三种 Skill condition。
3. 在 base model 和 RL checkpoints 上分别计算：
   - greedy projected action；
   - action 是否有效、是否属于 admissible set；
   - 所有 admissible actions 的 constrained log-probability distribution。
4. 比较 checkpoint 转移中的 action flip 和 action distribution JS divergence。

这一层直接回答：固定 `s` 和 Skill 条件后，RL update 是否改变 policy action。

### Step 6：matched episode-level Skill margin

对目标 Skill、相同 game 与相同 seeds 分别运行：

```text
FULL_BANK
MINUS_SKILL
NO_SKILL
```

主要 episode 指标：

```text
Skill margin = success(FULL_BANK) - success(MINUS_SKILL)
```

只保留目标 Skill 在 full-bank arm 中实际被 Router 选中过的 eligible pairs。统计按
`game_id` 聚类 bootstrap，evaluation seed 不是独立样本单位。

### Step 7：Pilot gate 后再扩展正式实验

若 pilot 证明 update、路由、归档和基础能力均有效，再扩展到：

- RL seeds：101、202、303；
- 30 epochs；
- checkpoint：0、5、10、15、20、25、30；
- evaluation seeds：11、22、33；
- 至少 8 个独立 game clusters/context；
- negative controls：zero update、shuffled reward、matched-norm random parameter。

## 7. 每条轨迹的归档内容

轨迹采用 `phase1.trajectory.v2`，一个 trajectory 一个 JSON，并写入幂等 index。

### 7.1 每一步记录

- `step_index`；
- 完整 `prompt_text`；
- `task_description`；
- 当前 `observation`；
- `admissible_actions`；
- `raw_model_output`；
- `projected_action`；
- `is_action_valid`；
- `reward` 与 `next_observation`；
- `retrieved_skill_ids`：mask 前与 task 匹配的条目；
- `candidate_skill_ids`：当步 Router 可以实际选择的 Skills；
- `selected_skill_id`；
- `injected_skill_ids`；
- `disabled_skill_ids`；
- `skill_router_version`；
- `skill_router_scores` 与分项 score details；
- `skill_router_state_flags` 与 selection reason；
- prompt/completion token 数。

### 7.2 轨迹顶层记录

- run、trajectory、game、context 和 global step 标识；
- episode return、trajectory length；
- 整条轨迹出现过的 retrieved/candidate/injected/selected Skill 集合；
- `unique_skill_count`；
- `skill_selection_counts`：每个 Skill 在本轨迹被调用多少步；
- Router version 集合；
- 完整 step sequence。

### 7.3 每个 RL training step 记录

- 与每条 trajectory 对齐的 reward/return/advantage sum、mean、min、max；
- optimizer/actor metrics，包括 gradient norm；
- update type 和 global step；
- checkpoint 及 run manifest 的可追溯关系。

## 8. 预期输出的统计量

### 8.1 基础能力与训练健康度

- overall success rate；
- success rate by context/checkpoint/RL seed；
- episode return 分布；
- trajectory length 均值和分布；
- invalid action count/rate；
- prompt/completion token 数；
- gradient norm、reward、return、advantage；
- checkpoint parameter delta。

### 8.2 Router 与 Skill 使用统计

- 每个 Skill 总 selection/invocation 次数；
- 每个 context 的 Skill selection 分布；
- 每条 trajectory 使用的不同 Skill 数量；
- 至少使用两个不同 Skill 的 trajectory 数和比例；
- 相邻 step 的 Skill switch 次数；
- 每个 state phase 的 Skill selection；
- candidate → selected 的频次；
- Router invariant violations；
- 各 Skill 的 eligible game/trajectory/state 数。

这些统计用于判断 Router 是否真正产生 state-dependent Skill 使用，并防止只由少量
频繁 Skill 支撑所有结论。

### 8.3 固定状态策略变化统计

- greedy action flip rate；
- action admissibility 和 validity；
- constrained action JS divergence；
- full/minus/no-skill 条件下的 checkpoint transition；
- 相同 Skill、相似 state phase 的 action flip 一致性。

### 8.4 Skill 边际效用与更新后偏移

- checkpoint、Skill、context 级 episode success margin；
- 按 game cluster bootstrap 的 estimate、LCB、UCB；
- 相邻 checkpoint 的 `delta_margin`；
- stable positive、stable negative、beneficial sign flip、harmful sign flip、ambiguous；
- Skill 步骤级调用的累计归因支持量；
- 跨 RL seed 的方向一致率和 Spearman correlation；
- real RL 与 zero/shuffled/random-parameter controls 的效应量比较。

## 9. Pilot 继续/停止条件

Pilot 至少需要同时满足：

- 真实非零 policy update；
- trajectory v2 文件和幂等 index 完整；
- 每步 selected/injected/candidate 不变量无错误；
- Router 与 Skill Bank hash 全程不变；
- 存在使用两个或更多不同 Skills 的多步轨迹；
- 1.5B policy 能产生基本有效 action，invalid action rate 未严重退化；
- 存在足够的 Skill selection coverage，可形成后续 fixed-state probes 和 matched pairs。

若不满足，优先诊断 Router coverage、prompt、模型 action protocol、训练 update 或归档链路，
不直接扩大到 30 epochs × 3 seeds。

## 10. 代码与协议位置

- Proposal：`/home/wangyifan/skill-RL/2026-08-21-proposal-advantages-and-roadmap.md`
- 最小验证原始 spec：`/home/wangyifan/skill-RL/2026-08-24-phase1-minimal-validation-repository-and-experiment-spec.md`
- 本文：`/home/wangyifan/skill-RL/2026-08-25-step-routed-skillrl-rl-validation-protocol.md`
- v2 machine-readable protocol：`/home/wangyifan/skill-RL/SkillRL/phase1/config/phase1_step_routing_protocol.json`
- Frozen Router：`/home/wangyifan/skill-RL/SkillRL/agent_system/memory/step_skill_router.py`
- ALFWorld manager 接线：`/home/wangyifan/skill-RL/SkillRL/agent_system/environments/env_manager.py`
- v2 trajectory archive：`/home/wangyifan/skill-RL/SkillRL/phase1/archive.py`
- matched evaluator：`/home/wangyifan/skill-RL/SkillRL/phase1/eval_skill_margin.py`
- fixed-state probe：`/home/wangyifan/skill-RL/SkillRL/phase1/action_probe.py`
- routing summary：`/home/wangyifan/skill-RL/SkillRL/phase1/summarize_step_routing.py`
- RL launcher：`/home/wangyifan/skill-RL/SkillRL/examples/grpo_trainer/run_alfworld_phase1_step_router.sh`
- preflight 记录：`/home/wangyifan/skill-RL/SkillRL/artifacts/preflight-step-router-code-ready.json`

## 11. 当前状态

- 逐状态 Frozen Router、训练接线、评估、probe、归档、统计脚本和 launcher 已准备。
- 启动前测试与 preflight 已完成。
- 本文不包含 RL success rate、Skill margin、action flip 或其他结果；这些结果必须在
  RL pilot 实际完成并归档后单独汇总。
