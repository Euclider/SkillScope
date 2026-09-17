# SkillRL 固定状态 Action Probe 与 Matched Skill Margin 结果

> 日期：2026-08-26  
> 来源 RL run：`phase1-step-router-pilot-seed-101-r2`  
> 目标：`pic_001 / pick_and_place`  
> 状态：固定状态 probe 与 base/checkpoint 1/checkpoint 5 的 matched episode 评估均完成并归档

## 1. 核心结论与本轮定位

这轮最小验证已经回答了当前最重要的问题：**真实 RL update 后，在固定相同 state 和固定
Skill 条件下，policy action 是否频繁改变。**答案是肯定的。在 8 个固定状态上，base 到
checkpoint 5 的 FULL_BANK（这些状态下 Router 选择 `pic_001`）greedy action 有 7/8
发生 flip；MINUS_SKILL 为 5/8，NO_SKILL 为 4/8。连续 checkpoint 的相邻比较中，
FULL_BANK 有 20/40，即 50.0% 的 action flip。

这里的 **action flip 只表示功能行为发生变化，不自带“好/坏”方向**。本轮 action probe
的目标是确认这种变化是否存在且是否足够频繁，从而判断后续研究参数更新信号是否有意义；
它并不尝试仅凭某个 action 的文本判断 Skill 变好或变坏。

本轮同时把 matched episode evaluation 链路跑通，但它在这里属于方向标签的低支持试跑，
不是要求所有 Skill 的效用随更新统一变正或统一变负。当前 bootstrap 结论只针对
`pic_001 / pick_and_place` 这一个 `(Skill, context)` 聚合项：其 margin 方向尚不能可靠确定。
不同 Skill/context 出现不同方向的效用变化正是 proposal 预期研究的异质性。

## 2. 为什么选择 pic_001

`pic_001` 是任务特定的 **Systematic First-Pass Search** Skill：在获取目标物体前，对可见
表面与关闭容器做系统的一次遍历。它在 RL pilot 中有较高且可解释的支持：

- 475 次步骤级调用；
- 59 条触发轨迹；
- 22 个 game；
- 出现在 14 条成功轨迹中。

相较跨多个 context 复用的 general Skill，先选择这个 task-specific Skill 能减少归因歧义。

## 3. Checkpoint 1–5 的关系与参数更新量

### 3.1 它们是同一条连续训练链，不是五次独立训练

`checkpoint-0` 是 `/home/wangyifan/model/Qwen2.5-1.5B-Instruct` 原始模型；
`checkpoint-1` 到 `checkpoint-5` 是同一个 RL seed 101、同一条连续 GRPO run 在每次
global update 完成后保存的累计快照。后一 checkpoint 继承前一 checkpoint 的参数和优化器
状态，不是分别从 base 初始化。

每个 global update 的实际训练量为：

- 8 个训练 game prompts；
- 每个 game 采样 4 条 rollout，因此每次 update 有 32 条训练轨迹；
- 每条轨迹最多 30 个环境步骤；
- GRPO，`ppo_epochs=1`，global PPO mini-batch size 为 32；
- actor learning rate 为 `1e-6`，KL loss coefficient 为 `0.01`；
- 每次 actor update 后立即保存 checkpoint；每次 update 的 8 条 validation rollout 不参与梯度。

| Checkpoint | 累计 GRPO updates | 本 checkpoint 前累计训练 rollouts | 含义 |
|---|---:|---:|---|
| Base / checkpoint 0 | 0 | 0 | 原始 instruct model |
| checkpoint 1 | 1 | 32 | 第一次真实 RL update 后 |
| checkpoint 2 | 2 | 64 | 在 checkpoint 1 上再更新一次 |
| checkpoint 3 | 3 | 96 | 累计第三次更新 |
| checkpoint 4 | 4 | 128 | 累计第四次更新 |
| checkpoint 5 | 5 | 160 | 累计第五次更新 |

因此，文中的“checkpoint 1 → 5”表示在 checkpoint 1 的基础上又发生了 **4 次连续
GRPO update、128 条新训练 rollout**；base → checkpoint 5 则是累计 5 次 update、160 条
训练 rollout。

### 3.2 参数变化有多大

FSDP checkpoint 相对原始 HF base 的精确参数差分为：

| 参数区间 | Delta L2 | 相对 base 参数 L2 | 最大单元素绝对变化 | 发生非零变化的参数元素 |
|---|---:|---:|---:|---:|
| Base → checkpoint 1 | 0.231851 | 0.013740% | 2.9007e-5 | 1,770,695,160 / 1,777,088,000（99.64%） |
| Base → checkpoint 5 | 0.409177 | 0.024249% | 8.5197e-5 | 1,774,219,878 / 1,777,088,000（99.84%） |

另外，使用合并后的 HF 权重直接比较 checkpoint 1 → 5，distinct named parameter 的
delta L2 为 **0.206170**（按 HF tied-weight 去重后的 1,543,714,304 个参数元素统计）。
该数值与上表采用不同的 tied-weight 计数口径，适合证明 checkpoint 1 到 5 之间仍发生了
实质更新，不应直接拿绝对值与上表逐项相减。

“99% 以上元素变化”只表示几乎所有可训练元素都出现了非零数值差，不表示每个参数都发生
大幅改变；相对 L2 显示这仍是一段幅度较小、但真实且可测的全参数更新。

## 4. 实际的候选检索与 state Router 逻辑

本实现不是把一个 game 的全部 Skill 一次性塞进每一步 prompt，也没有训练另一个神经网络
Retriever。它由两个确定性阶段组成，并在整个 RL 过程中冻结：

1. **Episode candidate retrieval。** 根据 task description 的关键词确定 ALFWorld context。
   `pick_and_place` 下取 Skill Bank 中前 12 个 general Skills 和该 context 的全部 5 个
   task-specific Skills。common mistakes 虽保留在 retrieved 审计字段中，但本 run 设置
   `include_common_mistakes=False`，不进入 Router 候选。因此 FULL_BANK 实际有 17 个可选
   Skill。
2. **每一步 state routing。** 每个环境步骤重新读取 task、当前 observation、当前 admissible
   actions、最近 action/observation history 和 step index，构造以下可观察状态 flags：
   `initial`、`visible_target`、`holding_target`、`process_available`、
   `processed_recently`、`ready_to_place`、`closed_container`、`search`、`loop`、
   `lamp_available`、`multiple_targets`。
3. **确定性打分。** 每个候选 Skill 的分数由三部分相加：state/task 与 Skill description 的
   token overlap（每个重合 token 0.08，最高 1.5）、激活 phase flag 与 Skill
   `title/principle/when_to_apply` 的 phase-marker 匹配分，以及 task-specific 的 +0.35
   prior；最后用极小的文档顺序项确定性打破平分。
4. **只注入一个 Skill。** 取最高分 Skill，将该 Skill 的文本放入当前步骤 prompt；下一步
   根据新 state 重新选择。Router 无可训练参数，版本固定为
   `alfworld-observable-phase-router-v1`。

所以这里应区分三个归档字段：`retrieved_skill_ids` 是 mask 前取回、用于审计的 Bank 条目；
`candidate_skill_ids` 是该 condition 下 Router 真正可以选择的条目；`injected_skill_ids`
是这一状态最终放进 prompt 的条目，本实验通常恰好为 1 个。

## 5. 三种 condition 到底改变了什么

三种 condition 都固定相同 checkpoint、game、环境 seed、evaluation seed 和路由代码，只改变
目标 Skill 的可用性。它们作用在 **Router 选择前**，而不是 rollout 结束后改标签。

| Condition | Router 可选集合（pick_and_place） | 每一步实际注入 | 精确定义 |
|---|---:|---:|---|
| FULL_BANK | 12 general + 5 task-specific = 17 | 最高分的 1 个 | 不屏蔽任何 Skill；`pic_001` 只是其中一个候选 |
| MINUS_SKILL(`pic_001`) | 12 general + 其余 4 task-specific = 16 | 重新打分后的 1 个 | **只屏蔽 `pic_001`**，不是屏蔽全部任务特定 Skill |
| NO_SKILL | 0 | 0 | 屏蔽全部 Skill，prompt 不含 Skill 文本 |

因此 MINUS_SKILL 并不是“仍把全部全局 Skill 喂进 prompt、只不喂任务 Skill”。准确说法是：
其余 general 和 task-specific Skills 仍在候选池中，但每个 state 仍然只注入 Router 选中的
一个。对于本轮 8 个固定 probe state，FULL_BANK 均选择 `pic_001`；屏蔽它后，Router 均
确定性 reroute 到 `gen_010`。

“FULL_BANK + `pic_001`”在两类评估中的含义也不同：

- **固定状态 action probe：** state 样本本来就是从训练归档中筛出 Router 在该步骤实际选择
  `pic_001` 的状态；重新评估 FULL_BANK 时，这 8 个 state 仍都选择 `pic_001`。因此这里
  是严格的“相同 state + 相同 Skill”。
- **matched full-episode evaluation：** FULL_BANK 是正常运行完整 Bank；它不要求每一步都
  使用 `pic_001`。计算 `pic_001` margin 时才做 eligibility conditioning：只有对应
  FULL_BANK 轨迹至少一个步骤实际选择过 `pic_001`，该 full/minus matched pair 才进入
  该 Skill 的统计。本轮 72 条 FULL_BANK 轨迹恰好全部满足，但各轨迹仍会在不同步骤切换
  多个 Skill。

### 5.1 一条轨迹实际使用多少种 Skill

在 RL pilot 归档的 200 条训练加 validation 轨迹中：

- 因为 FULL_BANK 每一步只选择并注入 1 个 Skill，所以平均每条轨迹有 28.38 次步骤级
  Skill invocation（与平均轨迹长度相同）；
- 平均每条轨迹选择 **3.57 种不同 Skill**，范围为 1–6；
- 199/200（99.5%）条轨迹使用了至少 2 种不同 Skill；
- 平均每条轨迹发生 **8.425 次 Skill 切换**；
- 200 条轨迹共覆盖 22 个不同 Skill ID，平均轨迹长度为 28.38 步。

不同 Skill 种类数的轨迹分布为：1 种 1 条、2 种 16 条、3 种 83 条、4 种 72 条、5 种
24 条、6 种 4 条。这直接验证了实际运行不是“一条 game 全程只用一个 Skill”。

matched evaluation 的 72 条 FULL_BANK 轨迹平均使用 2.94 种 Skill（范围 1–4）；72 条
MINUS_SKILL 轨迹平均使用 3.44 种（范围 2–5）；NO_SKILL 为 0。MINUS 条件种类数略高，
是因为去掉 `pic_001` 后，state 演化和重新路由会让其他 Skill 补位，并不意味着同时把更多
Skill 塞进同一步 prompt。

## 6. 固定状态 Action Probe

### 6.1 设置

- 从已归档 full-bank 训练轨迹中抽取 8 个不同 game 的 state；
- 这些 state 都由冻结 Router 实际选择了 `pic_001`；
- 每个 state 固定 task、observation、history 和 admissible actions；
- 比较 base、checkpoint 1–5；
- 每个 checkpoint 比较 `full_bank / minus_skill / no_skill`；
- full-bank 在这些目标状态注入 `pic_001`；minus-skill 屏蔽且只屏蔽 `pic_001`，然后
  确定性 reroute 到 `gen_010`；no-skill 不注入任何 Skill；
- 对每个 state 同时记录 greedy projected action 与 admissible actions 上的 constrained
  log-probability distribution。

本轮 8 个 state 均为 step 0。这适合验证 `pic_001` 的 initial-search 使用场景，但不能代表
中后期所有 state phase。

### 6.2 相邻 checkpoint 结果

| Condition | Action flip rate | Mean JS divergence |
|---|---:|---:|
| full_bank + `pic_001` | 50.0% | 0.002590 |
| minus_skill | 40.0% | 0.001277 |
| no_skill | 27.5% | 0.001592 |

逐段来看：

| Checkpoint transition | FULL_BANK | MINUS_SKILL | NO_SKILL |
|---|---:|---:|---:|
| Base → checkpoint 1 | 5/8（62.5%） | 5/8（62.5%） | 4/8（50.0%） |
| checkpoint 1 → 2 | 4/8（50.0%） | 5/8（62.5%） | 3/8（37.5%） |
| checkpoint 2 → 3 | 3/8（37.5%） | 3/8（37.5%） | 1/8（12.5%） |
| checkpoint 3 → 4 | 4/8（50.0%） | 0/8（0.0%） | 2/8（25.0%） |
| checkpoint 4 → 5 | 4/8（50.0%） | 3/8（37.5%） | 1/8（12.5%） |

这里共有 8 states × 5 相邻 checkpoint transitions = 每个 condition 40 个比较。
full-bank 发生 20/40 flips，minus-skill 16/40，no-skill 11/40；三种 condition 合计
47/120，即 39.17%。full-bank flip rate 比 minus-skill 高 10 个百分点，比 no-skill 高
22.5 个百分点。

### 6.3 Base → checkpoint 5 端点结果

| Condition | Action flips | Flip rate | Mean JS divergence |
|---|---:|---:|---:|
| full_bank + `pic_001` | 7/8 | 87.5% | 0.005255 |
| minus_skill | 5/8 | 62.5% | 0.005775 |
| no_skill | 4/8 | 50.0% | 0.004573 |

- full-bank 比 minus-skill 多 25 个百分点的 action flip；
- 2/8 states 出现“full-bank flip，但 minus-skill 不 flip”；
- full-bank 比 no-skill 多 37.5 个百分点的 action flip；
- 3/8 states 出现“full-bank flip，但 no-skill 不 flip”。

这说明存在 **Skill-condition 与 policy update 的交互候选信号**。但 full-bank 的 endpoint
mean JS 并没有高于 minus-skill，因此 greedy flip 与整体分布变化并非完全一致；8 个 state
也不足以估计稳定效应量。

### 6.4 Flip 的“好/坏”不由本表判断

action flip、JS divergence 和 FULL/MINUS interaction 都是 **policy-update functional
readout**，证明固定输入下 policy 的响应变了，并提供 Skill-specific change 的候选早期信号。
但它们不直接给出 reward valence：变到新 action 可能更好，也可能更坏。

按照 proposal，后续应把本次 RL update 的参数/梯度方向以及 update batch 的 reward 或
advantage 组合成 reward-directed interaction signal，在看不到 post-update held-out label
时预测哪些 `(Skill, context)` 的边际效用会下降。真正用于监督或最终核验的方向标签仍应是
独立 held-out matched rollout 的
`m_t(k,c)=E[R|FULL_BANK]-E[R|MINUS_k]`，再比较
`Delta m=m_post-m_pre`，而不是凭 action 文本或仅凭参数方向主观命名 good/bad flip。

## 7. Matched Episode-Level Skill Margin（方向标签链路试跑）

### 7.1 设置与完整性

- held-out split：`valid_seen`；
- 8 个独立 game clusters；
- evaluation seeds：11、22、33；
- checkpoint：base、checkpoint 1、checkpoint 5；
- 每个 checkpoint：8 games × 3 seeds × 3 conditions = 72 rollouts；
- 总计 216 rollouts 与 216 个完整 trajectory JSON；
- 每个 checkpoint/condition 的分母为 24；
- 所有 72 条 full-bank trajectory 都实际选择过 `pic_001`；
- duplicate matched keys、minus-skill 目标泄漏、no-skill 注入泄漏和路由不变量违例均为 0。

### 7.2 Success 与 margin

| Checkpoint | FULL_BANK | MINUS_SKILL | NO_SKILL | FULL − MINUS margin | Game-cluster bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| Base | 0/24 | 1/24 | 0/24 | -4.17% | [-12.50%, 0.00%] |
| Checkpoint 1 | 3/24 | 2/24 | 0/24 | +4.17% | [-8.33%, +16.67%] |
| Checkpoint 5 | 1/24 | 1/24 | 2/24 | 0.00% | [0.00%, 0.00%] |

checkpoint 5 的 full/minus matched outcomes 在每个 pair 上相同，因此 margin bootstrap 区间
精确为 0；这不表示模型性能确定，而只表示本次匹配样本中目标 Skill 的 success contrast
为零。

这里的 game-cluster bootstrap 是先对同一个 `pic_001 / pick_and_place / checkpoint` 下的
matched success difference 按 game 聚合，再对 8 个 game clusters 重采样。区间跨过或接触
0，只表示 **当前这个 `(Skill, context)` 在该 checkpoint 的 margin 方向无法可靠与 0 分开**。
它不要求不同 Skills 的效用统一为正或统一为负，也不否定不同 Skill 在同一次 update 后朝
相反方向变化。要研究这种异质性，必须扩大到多个 Skill/context 和多个独立 RL seed，而
不是把它们预先汇总成一个共同方向。

### 7.3 Checkpoint transitions

| Transition | Margin delta | Strict label |
|---|---:|---|
| Base → checkpoint 1 | +8.33 pp | ambiguous |
| Checkpoint 1 → checkpoint 5 | -4.17 pp | ambiguous |
| Base → checkpoint 5 | +4.17 pp | ambiguous |

没有检测到可可靠标注的 harmful sign flip；但这不是“证明不存在”，也不是本轮 action
flip 验证失败，而是当前仅一个 Skill/context、8 个 game clusters 的 gold-label 试跑不足以
把该项的正负号与 0 分离，所以协议要求对这个 pair abstain。

### 7.4 额外能力诊断

三种 condition 汇总后的 evaluator `invalid_action_count` 平均值：

| Checkpoint | Mean invalid actions / trajectory |
|---|---:|
| Base | 12.153 |
| Checkpoint 1 | 0.111 |
| Checkpoint 5 | 0.000 |

这表明 RL 很快改善了输出/动作投影有效性。成功率没有同步单调增长：checkpoint 1 的
full-bank success 为 12.5%，checkpoint 5 降到 4.17%。因此继续训练不一定持续改善长程
任务表现，也不能把 action-format 改善直接等同于 Skill utility 改善。

## 8. 当前能说与不能说的结论

可以说：

- 真实 RL 后，在相同 state 与相同 `pic_001` 条件下，policy action 明确发生变化；
- 连续 checkpoint 的 FULL_BANK action flip 为 20/40，base → checkpoint 5 为 7/8，
  频率足以支持继续提取和研究 policy-update readout；
- full-bank 的 greedy flip 多于 minus/no-skill，出现了 Skill-conditioned interaction 候选；
- RL 显著减少 evaluator invalid actions；
- state routing、Skill masking、matched seeds 和轨迹归档链路全部通过审计。

不能说：

- action flip 本身是 beneficial 还是 harmful；
- `pic_001` 的 episode success utility 已可靠变正或变负；
- 已观察到统计可靠的 harmful sign flip；
- 8 个初始状态和一个 Skill/context 可以代表完整 Skill Bank；
- 单个 RL seed 的结果具有跨 seed 稳定性。

下一轮应把“频繁 functional shift”作为已通过的管线门槛，扩大目标 Skill/context 数与
独立 RL seeds，并在每个 update 上计算参数投影、action interaction 和 reward-directed
signal，再用独立 matched margin change 作为异质的方向标签。若要提高单个标签的置信度，
还需要增加独立 game clusters；仅增加同一 8 个 game 的 evaluation seed 数不能替代它。

## 9. 归档位置

- 总机器可读汇总：`SkillRL/artifacts/metrics/phase1-step-router-pilot-seed-101-r2/evaluation-summary.json`
- 固定状态样本：`SkillRL/artifacts/probes/phase1-step-router-pilot-seed-101-r2/pic_001-states.jsonl`
- Action probe 原始记录：`SkillRL/artifacts/evaluations/phase1-step-router-pilot-seed-101-r2/action-probe-pic_001.jsonl`
- Action probe 指标：`SkillRL/artifacts/metrics/phase1-step-router-pilot-seed-101-r2/action-probe-pic_001/`
- Matched evaluator 索引：`SkillRL/artifacts/evaluations/phase1-step-router-pilot-seed-101-r2/matched-pic_001/{base,checkpoint-1,checkpoint-5}.jsonl`
- 216 条完整 matched trajectories：`SkillRL/artifacts/evaluations/phase1-step-router-pilot-seed-101-r2/matched-pic_001/trajectories/`
- Matched margin、transition、bootstrap 与 abstention：`SkillRL/artifacts/metrics/phase1-step-router-pilot-seed-101-r2/matched-pic_001/`
- Checkpoint 1 → 5 直接参数差分：`SkillRL/artifacts/metrics/phase1-step-router-pilot-seed-101-r2/checkpoint-1-to-5-delta.json`
- 推理用 merged checkpoints：`SkillRL/artifacts/merged_checkpoints/phase1-step-router-pilot-seed-101-r2/`
- 完成状态：`SkillRL/artifacts/run_status/phase1-fixed-state-and-matched-pic_001-seed-101.json`
