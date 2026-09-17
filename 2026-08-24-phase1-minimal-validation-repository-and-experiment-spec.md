# Phase I 最小验证：仓库选型与实验实施规范

## 1. 文档用途

本文档作为服务器 Agent 的工程输入，规定第一阶段最小验证的代码底座、实验变量、代码改动、数据格式、统计口径和验收条件。

本阶段只回答以下问题：

1. 固定 Skill Bank 后，真实 RL update 是否引起 Skill 边际效用变化；
2. harmful sign flip 是否自然发生，发生率是否足以支撑后续预测研究；
3. 同类变化在不同 RL seed、minibatch、update 和相似 context 上是否重复出现；
4. 真实 RL update 产生的变化是否显著强于评估噪声、shuffled-reward update 和随机参数扰动。

本阶段不训练 sign-flip predictor，不修改 Skill 内容，不执行 retain、refresh、retire、merge 或 create 操作，不研究 Skill retrieval error，不实现在线 Skill evolution。

---

## 2. GitHub 调研结论

检索日期：2026-08-24。Star 数量为检索时的 GitHub API 快照。

| 项目 | Star | RL checkpoint | Skill Bank | Skill/no-Skill 边际评估 | 工程定位 |
|---|---:|---|---|---|---|
| [aiming-lab/SkillRL](https://github.com/aiming-lab/SkillRL) | 945 | 已实现 | 已实现，支持冻结 | 未实现单 Skill matched evaluator | 主仓库 |
| [langfengQ/verl-agent](https://github.com/langfengQ/verl-agent) | 2,247 | 已实现 | 未实现 | 未实现 | SkillRL 的 RL 底座来源 |
| [microsoft/SkillLens](https://github.com/microsoft/SkillLens) | 157 | 不含 RL 训练 | 已实现 | 支持 Skill-set/无 Skill 推理 | 独立评估框架参考 |
| [amazon-science/reskill](https://github.com/amazon-science/reskill) | 26 | 已实现 | 已实现，支持版本化 | A/B 比较新旧 Skill 版本 | Policy–Skill 联合训练参考 |
| [ejhshen/SLIM](https://github.com/ejhshen/SLIM) | 22 | 已实现 | 已实现 | 已实现 leave-one-skill-out | 边际效用评估代码参考 |

### 2.1 主仓库

主仓库固定为：

```text
https://github.com/aiming-lab/SkillRL
```

采用 SkillRL 的以下现有组件：

- ALFWorld 环境及 action projection；
- veRL/verl-agent GRPO 训练链路；
- `SkillsOnlyMemory`；
- template retrieval；
- JSON Skill Bank；
- validation rollout；
- 周期性 checkpoint 保存；
- rollout reward、success、action 和 trajectory 日志。

SkillRL 的 ALFWorld 启动脚本：

```text
examples/grpo_trainer/run_alfworld_skills.sh
```

Skill memory 实现：

```text
agent_system/memory/skills_only_memory.py
```

### 2.2 边际评估参考实现

SLIM 已实现以下关键逻辑：

```python
perf_with = self._perf(routed_records)
perf_without = float(audit_callback(skill_id, dataset_indices))
utility = perf_with - perf_without
```

以及：

```python
with memory.temporarily_disabled(skill_id), memory.suspend_usage_tracking():
    rollout_result = self._run_validation_rollout(test_data, self.val_envs)
```

参考文件：

- [SLIM lifecycle.py](https://github.com/ejhshen/SLIM/blob/main/slim_method/lifecycle.py)
- [SLIM trainer.py](https://github.com/ejhshen/SLIM/blob/main/slim_method/trainer.py)

只移植只读评估逻辑，不移植 retain、retire、expand 和 Skill Creator。

---

## 3. 模型与环境

### 3.1 Smoke-test 模型

```text
Qwen/Qwen2.5-0.5B-Instruct
```

该模型参数量为 0.49B，支持 Transformers、vLLM 和 SGLang。模型来源：

```text
https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct
```

0.5B 实验用于验证：

- 模型加载；
- ALFWorld 环境交互；
- action format；
- Skill 注入与屏蔽；
- GRPO update；
- checkpoint 保存和恢复；
- matched evaluation；
- 指标计算与结果落盘。

### 3.2 正式最小模型切换条件

以下任一条件成立时，正式 Phase I 切换到：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

切换条件：

- invalid action rate 高于 30%；
- no-Skill success rate 接近零，无法形成有效能力区间；
- update 前无法获得至少 4 个正边际效用 `(Skill, context)`；
- 真实 RL update 后 validation reward 没有可测变化；
- 0.5B 输出无法稳定遵守 ALFWorld action protocol。

0.5B 未通过上述门槛时，结果只用于验证工程链路，不用于否定研究假设。

### 3.3 环境

主环境固定为 ALFWorld text-based environment。

数据划分：

- train：RL rollout；
- development：update 前正边际 Skill-context 筛选及工程调试；
- held-out evaluation：跨 checkpoint 的 matched Skill/no-Skill gold measurement；
- valid_seen 与 valid_unseen 分开统计，不混合调参。

---

## 4. 全局实验约束

1. Skill Bank 在全部 RL updates 中保持内容、数量、顺序和 trigger 规则不变；
2. 设置 `enable_dynamic_update=False`；
3. 使用 template retrieval，不加载 embedding retriever；
4. retrieval 结果必须记录具体 `skill_id`；
5. 训练 rollout 与 held-out evaluation instances 完全隔离；
6. evaluation 不产生梯度，不修改 optimizer、Skill Bank、retrieval statistics 和 usage counter；
7. 每个 RL checkpoint 使用同一组 held-out game IDs；
8. paired conditions 使用相同初始游戏、环境 seed、sampling seed、temperature、top-p、最大步数和 action projection；
9. update 前的 Skill-context eligibility 不得利用 update 后结果；
10. 不得按 shift 大小筛选样本后再计算事件发生率；
11. 全部低支持和标签不确定样本保留在结果中，以 `abstain_reason` 标记；
12. 代码、配置、依赖、随机种子、checkpoint 和输出 schema 全部进入版本控制或实验清单。

---

## 5. Skill Bank 与 context 定义

### 5.1 精简 Skill Bank

从 SkillRL 的 ALFWorld Skill Bank 构建固定子集：

- 1 个 general exploration Skill；
- 每个纳入实验的 ALFWorld task type 保留 1 个 task-specific Skill；
- 总规模控制在 4–7 个 Skill；
- 每个 task-specific Skill 具有唯一 `skill_id`；
- `when_to_apply` 与 task type 显式对应；
- 不包含动态生成 Skill。

精简 Skill Bank 以独立 JSON 文件保存，不覆盖上游原始 Skill Bank。

### 5.2 最小 context 粒度

Phase I 的 context 采用 episode-level task-type cluster：

```text
c = ALFWorld task type
```

候选类型包括：

```text
pick_and_place
clean
heat
cool
look/examine
pick_two
```

本阶段不把 episode-level matched evaluation 解释为 step-level state attribution。两个 matched trajectories 在执行第一个不同 action 后允许自然分叉。

---

## 6. 代码改动

### 6.1 Skill 屏蔽接口

修改：

```text
agent_system/memory/skills_only_memory.py
```

新增状态：

```python
self.disabled_skill_ids: set[str] = set()
```

新增接口：

```python
from contextlib import contextmanager
from collections.abc import Iterator

def set_disabled_skill_ids(self, skill_ids: set[str]) -> None:
    """Replace the current read-only evaluation mask."""

@contextmanager
def temporarily_disabled(self, skill_id: str) -> Iterator[None]:
    """Disable exactly one skill and restore the previous mask on exit."""

@contextmanager
def temporarily_disabled_all(self) -> Iterator[None]:
    """Disable every skill and restore the previous mask on exit."""
```

`retrieve()` 在格式化 prompt 前过滤 `disabled_skill_ids`。训练模式下该集合始终为空。

### 6.2 Evaluation condition

实现三种互斥 condition：

```text
FULL_BANK
MINUS_SKILL
NO_SKILL
```

- `FULL_BANK`：固定 Skill Bank 正常运行；
- `MINUS_SKILL`：只删除目标 `skill_id`，其余 Skill 保持不变；
- `NO_SKILL`：完全关闭 Skill 注入。

`FULL_BANK` 与 `MINUS_SKILL` 构成单 Skill leave-one-out 主对照；`NO_SKILL` 用于测量整个 Skill Bank 的总增益和 Policy baseline。

### 6.3 独立 evaluator

创建：

```text
phase1/eval_skill_margin.py
```

命令行参数：

```text
--checkpoint
--skill-bank
--skill-id
--context-id
--game-ids-file
--eval-seeds
--conditions
--temperature
--top-p
--max-steps
--output
```

evaluator 按 `(checkpoint, skill, context, game, eval_seed, condition)` 生成完整记录，不覆盖已有结果。重复运行时按唯一键跳过已经完成的记录。

### 6.4 训练启动脚本

复制并修改：

```text
examples/grpo_trainer/run_alfworld_skills.sh
```

创建：

```text
examples/grpo_trainer/run_alfworld_phase1.sh
```

不得直接改写上游复现脚本。

关键配置：

```bash
actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct
actor_rollout_ref.rollout.tensor_model_parallel_size=1
trainer.n_gpus_per_node=2
trainer.save_freq=5
trainer.test_freq=5
trainer.total_epochs=30
+env.skills_only_memory.enable_dynamic_update=False
+env.skills_only_memory.retrieval_mode=template
```

### 6.5 统计脚本

创建：

```text
phase1/compute_phase1_metrics.py
```

输入为 evaluator 产生的 JSONL/Parquet，输出：

```text
phase1_summary.json
skill_context_metrics.parquet
checkpoint_transition_metrics.parquet
bootstrap_intervals.parquet
abstention_report.json
figures/
```

### 6.6 测试

创建：

```text
tests/phase1/test_skill_mask.py
tests/phase1/test_matched_pair_keys.py
tests/phase1/test_margin_metrics.py
tests/phase1/test_sign_flip_labels.py
tests/phase1/test_abstention_rules.py
tests/phase1/test_resume_idempotence.py
```

测试覆盖：

- context manager 退出后恢复原始 Skill mask；
- `MINUS_SKILL` 只删除目标 Skill；
- `NO_SKILL` 不注入任何 Skill 文本；
- matched pair 的 game、seed 和生成参数完全一致；
- margin、delta margin 和 sign-flip label 使用固定公式；
- CI 跨零时输出 ambiguous；
- 重复执行 evaluator 不产生重复记录。

---

## 7. 核心定义

对 checkpoint $t$、Skill $s$、context $c$，定义：

$$
R_t^{\mathrm{full}}(s,c)
=
\mathbb{E}[G \mid \theta_t, c, \mathrm{FULL\_BANK}]
$$

$$
R_t^{-s}(s,c)
=
\mathbb{E}[G \mid \theta_t, c, \mathrm{MINUS\_SKILL}_s]
$$

Skill 在固定 Bank 中的条件边际效用为：

$$
M_t(s,c)
=
R_t^{\mathrm{full}}(s,c)
-
R_t^{-s}(s,c)
$$

相邻 checkpoint 的边际效用变化为：

$$
\Delta M_t(s,c)
=
M_{t+1}(s,c)-M_t(s,c)
$$

Skill Bank 总增益为：

$$
M_t^{\mathrm{bank}}(c)
=
R_t^{\mathrm{full}}(c)
-
R_t^{\mathrm{no\text{-}skill}}(c)
$$

严格 harmful sign flip 标签满足：

$$
\mathrm{LCB}\big(M_t(s,c)\big)>0
$$

且：

$$
\mathrm{UCB}\big(M_{t+1}(s,c)\big)<0
$$

不满足严格正、严格负或稳定零标签的样本记为 ambiguous，不进入二分类主指标。

---

## 8. 最小实验配置

| 配置项 | 数值 |
|---|---|
| Environment | ALFWorld text-based |
| Smoke model | Qwen2.5-0.5B-Instruct |
| Formal fallback model | Qwen2.5-1.5B-Instruct |
| RL algorithm | GRPO |
| Skill retrieval | template |
| Skill Bank | frozen, 4–7 Skills |
| Train games per update | 8 |
| Group size | 4 |
| RL updates | 30 |
| Saved checkpoints | 0, 5, 10, 15, 20, 25, 30 |
| RL seeds | 3 |
| Evaluation seeds | 3 |
| Development games per context | 8 |
| Held-out games per context | 8 |
| Max environment steps | 30 |
| Primary return | episode success；保留原始 reward |
| Primary context | task type |
| Bootstrap resamples | 2,000，按 game cluster 重采样 |

八张 A100 的执行分配：

- 每个 RL seed 使用 2 张 GPU；
- 三个 RL seeds 并行使用 6 张 GPU；
- 剩余 2 张 GPU 执行 checkpoint evaluation 或保留给调度；
- `tensor_model_parallel_size=1`；
- ALFWorld environment workers 使用独立 CPU cores。

---

## 9. 执行流程

### 9.1 环境和 action smoke test

1. 下载 ALFWorld 数据；
2. 加载 0.5B 模型；
3. 在无 RL、无 Skill 条件运行固定 development games；
4. 检查 action format、invalid action rate、环境终止条件和日志字段；
5. 在同一 games 上运行 `FULL_BANK`；
6. 验证 prompt 中出现目标 Skill 文本；
7. 运行 `MINUS_SKILL` 和 `NO_SKILL`；
8. 验证 Skill mask 与输出记录。

### 9.2 Update 前 positive-pair 筛选

在 checkpoint 0 上，对每个 `(Skill, context)` 执行 `FULL_BANK` 和 `MINUS_SKILL` matched rollouts。

eligibility 条件：

- Skill 在该 context 中被实际检索；
- 每个 context 至少包含 8 个独立 development games；
- $M_0(s,c)$ 点估计为正；
- 使用 2,000 次 game-cluster bootstrap 计算 95% CI，且置信区间下界大于 0；
- 不使用 checkpoint 1–30 的任何数据完成筛选。

至少保留 4 个 eligible pairs。数量不足时切换到 1.5B 模型，并从 smoke test 重新执行。

### 9.3 真实 RL updates

1. 固定 eligible pairs 和 held-out games；
2. 固定 Skill Bank 文件 hash；
3. 从同一初始模型启动 3 个独立 RL seeds；
4. 每 5 updates 保存模型、optimizer metadata、训练配置和 update 日志；
5. 全程禁止 Skill Bank 更新；
6. 记录每次 update 使用的 train game IDs、rollout seeds、rewards、advantages 和参数 update norm。

### 9.4 跨 checkpoint matched evaluation

对每个 checkpoint、eligible pair 和 held-out game 执行：

```text
FULL_BANK
MINUS_SKILL
NO_SKILL
```

同一 matched block 内共享：

```text
game_id
environment_seed
evaluation_seed
temperature
top_p
max_steps
action_projection_version
prompt_template_version
```

### 9.5 负对照

#### Zero-update control

对同一 checkpoint 重复 matched evaluation，以此估计 sampling noise 和 environment noise。

#### Shuffled-reward update

使用与真实 update 相同的 rollout batch、reward multiset、optimizer、learning rate 和 update step 数，在 group 内打乱 reward/advantage 对应关系后执行 update。

#### Matched-norm random parameter perturbation

构造与真实参数变化 $\Delta\theta_t$ 范数一致的随机方向扰动，保持模型架构和 Skill Bank 不变。该对照单独记录，不与 shuffled-reward update 合并。

---

## 10. 输出数据 schema

每条 rollout 写入 JSONL 或 Parquet，字段固定为：

```text
run_id
repo_commit
config_hash
skill_bank_hash
model_id
checkpoint_id
update_id
update_type
rl_seed
eval_seed
environment_seed
game_id
split
context_id
skill_id
skill_condition
retrieved_skill_ids
disabled_skill_ids
success
episode_return
invalid_action_count
trajectory_length
prompt_tokens
completion_tokens
action_sequence
trajectory_path
```

`update_type` 取值：

```text
real_rl
zero_update
shuffled_reward
random_parameter
```

`skill_condition` 取值：

```text
full_bank
minus_skill
no_skill
```

每个 `(checkpoint, update_type, rl_seed, eval_seed, game_id, skill_id, skill_condition)` 组成唯一键。

---

## 11. 统计分析

### 11.1 置信区间

以 game instance 为 cluster 进行 bootstrap。相同 game 的多个 evaluation seeds 不视为完全独立样本。

输出：

- $M_t(s,c)$ 点估计与 95% CI；
- $\Delta M_t(s,c)$ 点估计与 95% CI；
- Skill Bank 总增益；
- 每个 Skill、context、checkpoint 和 RL seed 的分层结果。

### 11.2 事件频率

harmful sign-flip rate 定义为：

$$
\mathrm{FlipRate}
=
\frac{
\#\{\text{reliable positive-to-negative flips}\}
}{
\#\{\text{eligible pre-update positive pairs}\}
}
$$

同时输出：

- raw sign-flip rate；
- reliable sign-flip rate；
- significant negative-shift rate；
- arbitrary significant-change rate；
- ambiguous rate；
- low-support abstention rate。

### 11.3 稳定性

输出：

- 三个 RL seeds 的 $\Delta M$ 符号一致率；
- 跨 seed Spearman correlation；
- pooled effect 与分 seed effect；
- 至少 2/3 seeds 同方向的事件比例；
- 相邻 update interval 的方向持续性；
- 相同 Skill 在不同 context 的异质性。

### 11.4 真实 update 与负对照

比较以下分布：

```text
|ΔM|
negative-shift rate
reliable sign-flip rate
cross-seed sign agreement
between-context heterogeneity
```

真实 update 的效应同时报告 effect size、置信区间和置换检验结果，不以单一 p-value 作为结论依据。

---

## 12. Abstention 规则

以下情况不产生硬标签：

- update 前 $M_t$ 的 CI 跨越 0；
- update 后 $M_{t+1}$ 的 CI 跨越 0；
- 独立 game cluster 数量低于 8；
- Skill 在目标 context 中没有实际触发；
- matched conditions 的 game 或 seed 不一致；
- evaluator 发生异常退出或部分缺失；
- prompt template、Skill Bank hash 或 action projection 在 pair 内不一致。

`abstain_reason` 使用固定枚举：

```text
pre_margin_ambiguous
post_margin_ambiguous
low_support
skill_not_triggered
pair_mismatch
incomplete_rollout
configuration_mismatch
```

---

## 13. Go/No-Go 验收条件

进入后续 sign-flip prediction 阶段必须同时满足：

1. update 前存在至少 4 个稳定正边际 `(Skill, context)`；
2. 真实 RL update 的 median $|\Delta M|$ 至少为 zero-update control 的 1.5 倍，且 game-cluster bootstrap 的差值 95% CI 不包含 0；
3. 至少 60% 的非 ambiguous 事件在 3 个 RL seeds 中达到 2/3 同方向，跨 seed 平均 Spearman correlation 不低于 0.3；
4. strict harmful sign flips 不少于 12 个且占 eligible transitions 的比例不低于 5%；strict flips 不足但显著负向 shift 不少于 30 个时，只进入 continuous utility-change prediction；
5. 真实 RL update 的 median $|\Delta M|$ 至少为 shuffled-reward control 的 1.25 倍，cluster permutation test 的 $p<0.05$；
6. paired evaluation 缺失率不高于 5%，configuration mismatch 数量为 0；
7. 全部结论在 held-out games 上计算。

停止条件：

- 0.5B 失败但 1.5B 通过能力门槛：使用 1.5B 完成 Phase I；
- 1.5B 仍无法形成稳定正边际 pairs：停止 sign-flip 研究，记录 Skill utilization floor；
- 连续 $\Delta M$ 存在但严格 sign flip 稀少：Phase I 结论限定为边际效用变化存在，后续目标改为连续 utility-change forecasting；
- 真实 update 与负对照无差异：停止 Policy-update-based forecasting；
- 事件存在但跨 seed 不稳定：扩大重复实验，不进入 predictor 训练。

---

## 14. 资源预算

0.5B smoke test：

```text
2 × A100 per RL seed
3 RL seeds in parallel
20–30 updates
总计约 24–60 A100 GPU-hours
```

matched evaluation：

```text
7 checkpoints
4–7 skills
3 evaluation seeds
3 conditions
约 10–40 A100 GPU-hours
```

完整 0.5B Phase I 预留：

```text
40–100 A100 GPU-hours
```

实际墙钟时间主要受 ALFWorld environment workers、轨迹长度和 evaluation rollout 数量影响。GPU-hours、rollout 数和 token 数必须写入最终实验报告。

---

## 15. 交付物

服务器 Agent 完成以下交付物后结束 Phase I：

1. 固定 commit 的 SkillRL fork；
2. 环境与依赖锁定文件；
3. 精简且冻结的 ALFWorld Skill Bank；
4. Skill mask 接口及单元测试；
5. `run_alfworld_phase1.sh`；
6. `eval_skill_margin.py`；
7. `compute_phase1_metrics.py`；
8. 全部 checkpoint 和实验配置 manifest；
9. rollout-level JSONL/Parquet；
10. margin、delta margin、sign flip、stability、abstention 和负对照结果；
11. 资源消耗统计；
12. `phase1_report.md`，明确给出 Go/No-Go 结论和证据。

---

## 16. 来源

- [SkillRL repository](https://github.com/aiming-lab/SkillRL)
- [SkillRL README](https://github.com/aiming-lab/SkillRL/blob/main/README.md)
- [SkillRL ALFWorld GRPO launcher](https://github.com/aiming-lab/SkillRL/blob/main/examples/grpo_trainer/run_alfworld_skills.sh)
- [SkillRL SkillsOnlyMemory](https://github.com/aiming-lab/SkillRL/blob/main/agent_system/memory/skills_only_memory.py)
- [SLIM repository](https://github.com/ejhshen/SLIM)
- [SLIM lifecycle evaluator](https://github.com/ejhshen/SLIM/blob/main/slim_method/lifecycle.py)
- [SLIM counterfactual trainer evaluation](https://github.com/ejhshen/SLIM/blob/main/slim_method/trainer.py)
- [ReSkill repository](https://github.com/amazon-science/reskill)
- [SkillLens repository](https://github.com/microsoft/SkillLens)
- [verl-agent repository](https://github.com/langfengQ/verl-agent)
- [Qwen2.5-0.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
