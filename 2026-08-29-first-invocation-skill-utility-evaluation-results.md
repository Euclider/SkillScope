# SkillRL 首次调用点三臂 Skill 效用评测结果

> 日期：2026-08-29  
> 评测 run：`phase1-first-invocation-pic_001-seed-101-v1`  
> 来源 RL run：`phase1-step-router-pilot-seed-101-r2`  
> 目标：`pic_001 / pick_and_place`  
> 状态：base 与 checkpoint 1–5 全部完成，共 432 条分支轨迹

## 1. 本轮修正了什么

此前的 `FULL_BANK / MINUS_SKILL / NO_SKILL` full-episode 对照会在 episode 开始前改变
Router 候选集合。这样不仅改变目标 Skill 的内容，还可能提前改变路由、动作和后续 state，
因此不能把结果严格归因于“在实际第一次调用 `pic_001` 的 state 上，Skill 内容产生了多少
效用”。

本轮改为首次实际调用点干预：先从 checkpoint 0 的正常 FULL_BANK rollout 中找出第一次
选择 `pic_001` 的位置，将此前的环境前缀固定，在这个位置才把目标 Skill 的 prompt
payload 分成三臂。分支之后，各臂按自己的新 state 自由生成动作、自由逐步路由，直到成功
或耗尽原有 30 步预算。

三臂始终保留同一份冻结 Router descriptor bank。因此 Router 在相同 state 上看到的 Skill
描述与分数不变，改变的只是 policy prompt 中目标 Skill 的内容：

| 干预臂 | Router 中的 `pic_001` | policy prompt 的 Skill slot | 用途 |
|---|---|---|---|
| ORIGINAL | 原 descriptor，正常参与路由 | 原始 `pic_001` 内容 | 真实 Skill |
| PLACEBO | 与 ORIGINAL 完全相同 | 相同模板、相同 tokenizer token 数的任务无关内容 | 控制文本长度、模板与“有一段文字”效应 |
| NULL | 与 ORIGINAL 完全相同 | 保留 Skill slot，但 payload 为空 | 测量完全去除内容的总影响 |

这不是旧版 `MINUS_SKILL` 的重新命名：本轮没有从候选 bank 删除 `pic_001`，也没有因此
reroute 到替代 Skill。即使在 NULL 臂，Router 仍可选择 `pic_001`；只是 policy 收不到它
的内容。

## 2. 估计量与解释

ALFWorld 本轮 terminal reward 为成功 `10`、失败 `0`，所以 return 差除以 10 后等价于
success probability 差。三项配对估计量为：

- **语义效用**：`U_sem = R(ORIGINAL) - R(PLACEBO)`。这是主指标，尽量隔离目标 Skill
  内容本身的任务相关语义。
- **总效用**：`U_total = R(ORIGINAL) - R(NULL)`。它包含语义效应与 prompt 结构/文本存在
  效应，不能单独解释成纯语义效用。
- **prompt nuisance**：`U_prompt = R(PLACEBO) - R(NULL)`。用于判断同长度无关文本相对
  空 payload 是否也改变结果；恒等式为 `U_total = U_sem + U_prompt`。
- **纵向变化**：`Delta U_k = U(checkpoint-k) - U(checkpoint-0)`。所有 checkpoint 使用同一
  批 checkpoint-0 anchor，因此它表示固定支持集上的效用漂移，而不是不同 on-policy
  state 分布之间的混合差异。

ORIGINAL 与 PLACEBO 在触发点使用相同生成 seed；NULL 也使用相同 seed schedule。由于
prompt token 序列不同，这不是 token-by-token 强制耦合，但控制了采样随机数设置。置信区间
采用 8 个 game 为 cluster 的 10,000 次 bootstrap，seed 为 `20260828`。

## 3. Anchor、模型与采样规模

### 3.1 固定 anchor 支持集

anchor 来自 checkpoint 0 在 `valid_seen` 上已经归档的 FULL_BANK rollout：8 个 game，
每个 game 使用 evaluation seed 11、22、33，共 24 个首次调用 occurrence、8 个不同 state。
每个 anchor 保存 game、environment seed、源轨迹、首次调用前的 action/history/reward 前缀、
触发 observation、admissible actions、Router 分数和 state hash。

本次 24 个 anchor 的首次 `pic_001` 调用全部发生在 **step 0**，因此固定前缀为空。代码仍按
一般的“重放前缀并核验触发 state”协议实现，但这批结果实际是 8 个不同初始 state、每个
重复 3 个采样 seed 的初始调用点评测。它不能外推到中后期 state。

### 3.2 Checkpoint 关系

base 是 `/home/wangyifan/model/Qwen2.5-1.5B-Instruct`；checkpoint 1–5 来自同一个 seed 101
的连续 GRPO 训练链。每个 global update 使用 8 个训练 game × 每个 4 条 rollout，即 32 条
训练轨迹，`ppo_epochs=1`、actor learning rate `1e-6`、KL coefficient `0.01`。因此：

| 模型 | 累计 update | 累计用于更新的 rollout |
|---|---:|---:|
| Base / checkpoint 0 | 0 | 0 |
| checkpoint 1 | 1 | 32 |
| checkpoint 2 | 2 | 64 |
| checkpoint 3 | 3 | 96 |
| checkpoint 4 | 4 | 128 |
| checkpoint 5 | 5 | 160 |

它们不是五次独立训练。已有参数审计显示 base → checkpoint 1 的 delta L2 为 `0.231851`，
base → checkpoint 5 为 `0.409177`；相对 base 参数 L2 分别为 `0.013740%` 与
`0.024249%`。这是小幅但真实的全参数连续更新。

### 3.3 PLACEBO 控制

原始 payload 与 PLACEBO 均由 Qwen2.5-1.5B tokenizer 编码为 **58 tokens**。PLACEBO 保持
原模板和标题结构，但内容改成与 ALFWorld 任务无关的 typography 描述；其文本 SHA-256 为
`8a85c421a2efd35f5daa930587660cf1955a76750484566edf132d3820e54cab`。

### 3.4 总规模

评测包含 6 个模型 × 24 个 anchor × 3 个 payload arm = **432 条 rollout**。每个模型恰有
72 条，每个 arm 跨模型恰有 144 条；432 个完整 trajectory JSON 均已归档。

## 4. 横向结果：每个 checkpoint 内的三臂比较

下表中 success 写作 `ORIGINAL / PLACEBO / NULL`，分母均为 24；所有效用与区间已经换算成
success percentage points（pp）。

| Checkpoint | Success O/P/N | 语义效用 O−P（95% CI） | 总效用 O−N（95% CI） | Prompt nuisance P−N（95% CI） | 首动作 O/P flip |
|---|---:|---:|---:|---:|---:|
| Base | 0 / 1 / 1 | -4.17 pp [-12.50, 0.00] | -4.17 pp [-12.50, 0.00] | 0.00 pp [-12.50, 12.50] | 41.67% |
| Checkpoint 1 | 3 / 2 / 0 | +4.17 pp [-8.33, 16.67] | +12.50 pp [4.17, 25.00] | +8.33 pp [0.00, 25.00] | 58.33% |
| Checkpoint 2 | 3 / 1 / 3 | +8.33 pp [0.00, 20.83] | 0.00 pp [-12.50, 12.50] | -8.33 pp [-20.83, 0.00] | 45.83% |
| Checkpoint 3 | 1 / 0 / 2 | +4.17 pp [0.00, 12.50] | -4.17 pp [-16.67, 8.33] | -8.33 pp [-20.83, 0.00] | 70.83% |
| Checkpoint 4 | 1 / 2 / 0 | -4.17 pp [-12.50, 0.00] | +4.17 pp [0.00, 12.50] | +8.33 pp [0.00, 20.83] | 50.00% |
| Checkpoint 5 | 1 / 2 / 0 | -4.17 pp [-12.50, 0.00] | +4.17 pp [0.00, 12.50] | +8.33 pp [0.00, 20.83] | 50.00% |

点估计呈现明确的非单调轨迹：`U_sem` 从 base 的负值，在 checkpoint 1–3 转为正值，再在
checkpoint 4–5 回到负值。但每个 checkpoint 只有 8 个 game clusters，reward 又极稀疏，
这些 checkpoint 内区间均跨过或接触 0；因此不能声称每一个符号都已被高置信度确定。

三臂控制确实必要。例如 checkpoint 1 中 ORIGINAL−NULL 为 +12.50 pp，但其中 PLACEBO−NULL
也有 +8.33 pp，不能把 +12.50 pp 全部归于 Skill 语义；checkpoint 2 中 ORIGINAL 与 NULL
同为 3/24，导致总效用为 0，但 ORIGINAL 比 PLACEBO 多成功 2 条，主语义效用却为
+8.33 pp。旧版二臂 FULL/MINUS 无法分解这两种现象。

## 5. 纵向结果：相对 checkpoint 0 的效用变化

| 对比 | Delta 语义效用（95% CI） | 正/负/零 anchor | Delta 总效用（95% CI） |
|---|---:|---:|---:|
| Base → checkpoint 1 | +8.33 pp [0.00, 20.83] | 2 / 0 / 22 | +16.67 pp [4.17, 33.33] |
| Base → checkpoint 2 | **+12.50 pp [4.17, 25.00]** | 3 / 0 / 21 | +4.17 pp [0.00, 12.50] |
| Base → checkpoint 3 | +8.33 pp [0.00, 20.83] | 2 / 0 / 22 | 0.00 pp [-12.50, 12.50] |
| Base → checkpoint 4 | 0.00 pp [-12.50, 12.50] | 1 / 1 / 22 | +8.33 pp [0.00, 20.83] |
| Base → checkpoint 5 | 0.00 pp [-12.50, 12.50] | 1 / 2 / 21 | +8.33 pp [0.00, 20.83] |

在当前 bootstrap 口径下，只有 **base → checkpoint 2 的语义效用变化区间严格高于 0**。
这说明真实 RL update 后，同一批 `(state, Skill)` anchor 上的 Skill 语义效用可以发生可测
变化。它不是“所有 checkpoint 都持续改善”：checkpoint 4 和 5 的聚合语义变化回到 0。

更重要的是，聚合 0 会掩盖 state/game 层面的 turnover。base → checkpoint 5 中：

- `GarbageCan / seed 22` 的语义 contrast 从 -10 变成 +10，纵向变化为 +20 return；
- `Tomato / seed 33` 纵向变化为 -10；
- 另一个 `GarbageCan / seed 11` 纵向变化为 -10；
- 其余 21 个 anchor 为 0，最终平均正好为 0。

因此“CI 跨 0”不应被解释成所有 Skill 效用必须统一变正或统一变负。这里研究的正是
`(Skill, state/context)` 的异质变化；一个聚合均值可以为 0，同时其内部有方向相反的变化。

各 checkpoint 内非零语义 anchor 也很稀疏且方向会变化：base 为正/负/零 `0/1/23`，
checkpoint 1 为 `2/1/21`，checkpoint 2 为 `2/0/22`，checkpoint 3 为 `1/0/23`，
checkpoint 4 为 `0/1/23`，checkpoint 5 为 `1/2/21`。这支持后续按 state/context 建模，
而不是给整个 Skill 预设一个全局固定符号。

## 6. 行为变化与 reward 变化的关系

ORIGINAL 相对 PLACEBO 的触发点首动作 flip rate 为 41.67%–70.83%；ORIGINAL 相对 NULL
为 45.83%–83.33%。Skill 内容会频繁改变当步 action，但大多数 anchor 的 terminal reward
contrast 仍为 0。这说明 action flip 是灵敏的功能读数，却不是效用方向标签：它可能是好
flip、坏 flip，或者最终被后续动作抵消。

base 三臂平均 invalid action 次数较高：ORIGINAL `10.67`、PLACEBO `16.92`、NULL
`15.88`。checkpoint 1 后分别降到 `0.21/0.13/0.13`，checkpoint 2–5 各臂均不超过
`0.125`，checkpoint 5 三臂均为 0。RL 明显改善了输出格式/动作有效性，但这同样不等于
`pic_001` 语义效用单调上升。

本轮也记录了分支后 Router 再次选择目标 Skill 的次数。ORIGINAL/PLACEBO 中每次选择都会
注入相应 payload；NULL 中 Router 仍会选择目标 Skill，但注入计数严格为 0。这保证后续自由
rollout 中干预语义保持一致，而不是只替换触发点一次后又恢复原 Skill。

## 7. 完整性与复现实验

自动完整性检查结果：

- 432/432 trajectory 成功写入，失败 0；
- 432 个 trajectory path 全部唯一；
- `(checkpoint, anchor, arm)` 重复 0；
- ORIGINAL/PLACEBO 触发 prompt token 数不一致 0；
- 三臂触发 Router scores 不一致 0；
- payload 干预违例 0；
- 每个 checkpoint 72 条、每个 arm 跨 checkpoint 144 条。

另外将新协议 ORIGINAL 臂与旧归档 FULL_BANK 按 checkpoint、game、evaluation seed 匹配，
在 base、checkpoint 1、checkpoint 5 共检查 72 对：**action sequence 或 success 不一致为
0**。这证明新 runner 的 ORIGINAL 分支精确复现了原正常路径，而不是评估重构引入了新的
行为差异。

代码测试结果为 `31 passed`，新增评测代码通过 Ruff 检查。

## 8. 当前可以与不可以得出的结论

当前可以得出：

1. 首次实际调用点三臂评测链路已经跑通，能把 Skill 语义、prompt nuisance 和完全去除内容
   的总效应分开。
2. 固定 checkpoint-0 state 支持集后，`pic_001` 的语义效用随真实 GRPO updates 发生了
   非单调变化；base → checkpoint 2 的纵向变化在本次 cluster bootstrap 下严格为正。
3. state/game 层面存在方向相反的 utility turnover；聚合均值为 0 不代表每个 state 都没变。
4. Skill payload 会频繁导致 action flip，但 flip 本身不能判断好坏，必须由后续累计 reward
   contrast 给方向。

当前不能得出：

1. 不能证明一个“初始可靠为正”的 Skill 已被 RL 更新成可靠为负；base 的 `pic_001` 语义
   效用本来就没有严格正证据。
2. 不能把结论外推到所有 Skill、所有 context 或中后期 state；本轮仅一个 Skill，且 24 个
   occurrence 全在 step 0。
3. 不能据此估计 population-level 因果效应；只有 8 个 game、一个 RL seed、一个 PLACEBO
   文本，terminal reward 很稀疏。
4. 不能把固定 checkpoint-0 支持集结果等同于各 checkpoint 自己的 on-policy 效用。固定
   支持适合纵向归因，on-policy 分布变化需要另行报告。

因此，这一轮支持 proposal 的核心可行性命题——policy 更新会让固定 `(state, Skill)` 的
行为响应和累计效用发生变化，而且变化具有 state 异质性——但仍属于机制验证，不是最终
统计结论。

## 9. 下一步建议

下一轮应保持本协议不变，优先扩展统计支持：

1. 对多个调用量足够的 task-specific 与 general Skill 各自构造首次调用 anchor；
2. 特别加入首次调用发生在 step > 0 的 Skill/state，实际验证 prefix replay 与中途分支；
3. 增加 held-out game、evaluation seeds 与独立 RL seeds；
4. 为每个 Skill 使用多个 token-matched、任务无关 PLACEBO，估计 placebo 文本方差；
5. 同时保留固定 base-support 与各 checkpoint on-policy support，分别回答效用漂移和分布
   漂移；
6. 在不读取 post-update reward 标签的前提下，计算 proposal 中基于参数/更新方向的预测
   分数，再用本三臂 `Delta U_sem` 作为事后 gold label 检验排序和符号预测。

## 10. 归档索引

- 协议配置：`SkillRL/phase1/config/first_invocation_payload_protocol.json`
- Anchor：`SkillRL/artifacts/anchors/phase1-first-invocation-pic_001-seed-101-v1/`
- PLACEBO 控制：`SkillRL/artifacts/controls/phase1-first-invocation-pic_001-seed-101-v1/`
- 逐条评测记录：`SkillRL/artifacts/evaluations/phase1-first-invocation-pic_001-seed-101-v1/`
- 432 条完整轨迹：上述目录的 `trajectories/` 子目录
- 日志：`SkillRL/artifacts/logs/phase1-first-invocation-pic_001-seed-101-v1/`
- 指标与完整性审计：`SkillRL/artifacts/metrics/phase1-first-invocation-pic_001-seed-101-v1/`
- Run manifest：`SkillRL/artifacts/manifests/phase1-first-invocation-pic_001-seed-101-v1.json`

