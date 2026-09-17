# Skill–Policy Interaction：Phase I 最小验证汇报

> 文档状态：实验前汇报稿  
> 更新日期：2026-08-25  
> 当前阶段：工程与数据准备完成，Phase I rollout、RL update 和实验统计结果待运行

## 1. 项目目标

本项目研究的问题是：一次 RL update 后，固定 Skill Bank 中原本有益的 Skill，
在新 Policy 上的边际效用是否会下降，甚至从“有益”翻转为“有害”。

Phase I 暂不训练预测器，也不修改、创建或淘汰 Skill，只验证以下基础事实：

1. 固定 Skill Bank 后，真实 RL update 是否会引起 Skill 边际效用变化；
2. harmful sign flip 是否会自然发生，数量是否足以支持后续预测研究；
3. 变化能否在不同 RL seed、update 和相似 context 上重复；
4. 真实 RL update 的效应是否明显强于评估噪声、打乱 reward 和随机参数扰动。

Phase I 的 episode-level 主指标为目标 Skill 的 leave-one-out 边际效用：

\[
M_t(s,c)=P(\text{success}\mid\text{FULL\_BANK})-
P(\text{success}\mid\text{MINUS\_SKILL}_s)
\]

严格 harmful sign flip 要求 update 前边际效用的置信区间严格为正，update 后
置信区间严格为负，而不只是点估计发生符号变化。

## 2. 代码底座与仓库来源

Phase I 基于开源仓库
[aiming-lab/SkillRL](https://github.com/aiming-lab/SkillRL) 开发：

- 本地项目：`/home/wangyifan/skill-RL/SkillRL`
- 固定上游 commit：`8e66726ed866a4e0a7f053586a41022798192e6c`
- 工作分支：`phase1-prep`

采用的上游能力包括：

- veRL/verl-agent 的 GRPO 训练链路；
- ALFWorld 环境与 action projection；
- `SkillsOnlyMemory`、template retrieval 和 JSON Skill Bank；
- validation rollout 与周期性 checkpoint 保存。

在此基础上补充了单 Skill 屏蔽、固定状态 action probe、matched evaluator、
负对照、统计分析以及完整实验归档。边际效用评估的设计还参考了 SLIM 的
leave-one-skill-out 思路，但没有引入其 Skill retain、retire 或 create 流程。

## 3. Benchmark 与任务

### 3.1 Phase I 主 Benchmark：ALFWorld

Phase I 只使用 ALFWorld text-based environment。ALFWorld 具有明确的任务成功
条件、离散可执行 action 和可按任务类型划分的 context，适合进行
Skill/no-Skill matched evaluation。

本阶段覆盖六类 context：

| Context | ALFWorld 原始任务类型 | 任务含义 |
|---|---|---|
| `pick_and_place` | `pick_and_place_simple` | 找到指定物体并放到目标位置 |
| `look_at_obj_in_light` | `look_at_obj_in_light` | 将指定物体置于灯光下查看 |
| `clean` | `pick_clean_then_place_in_recep` | 清洁物体后放置 |
| `heat` | `pick_heat_then_place_in_recep` | 加热物体后放置 |
| `cool` | `pick_cool_then_place_in_recep` | 冷却物体后放置 |
| `pick_two` | `pick_two_obj_and_place` | 找到并放置两个同类物体 |

数据用途分为：

- train：RL rollout；
- development：工程调试和 checkpoint 0 的正边际 pair 筛选；
- valid_seen：同分布 held-out matched evaluation；
- valid_unseen：跨场景 held-out evaluation，单独报告，不与 valid_seen 混合调参。

ScienceWorld 和 WebShop 仅作为 proposal 中的后续外部复验候选，不属于当前
Phase I 最小验证范围。

### 3.2 ALFWorld 数据样例

数据目录：`/home/wangyifan/skill-RL/data/alfworld`，当前约 2.3 GiB。

一个 development/clean 样例为：

```yaml
game_file: /home/wangyifan/skill-RL/data/alfworld/json_2.1.1/valid_train/
  pick_clean_then_place_in_recep-Cloth-None-Drawer-416/
  trial_T20190908_040722_151303/game.tw-pddl
task_id: trial_T20190908_040722_151303
task_type: pick_clean_then_place_in_recep
instruction: "Put a cleaned rag in a drawer under the right sink."
abstract_plan:
  - GotoLocation(countertop)
  - PickupObject(cloth)
  - CleanObject(cloth, sink)
  - PlaceObject(cloth, drawer)
```

实际 rollout 中，Policy 每一步接收当前 observation、任务历史、可执行 action
列表和按条件注入的 Skill 文本，并输出带 action 标签的动作。归档会保存完整
prompt、原始模型输出、投影后的 action、reward、下一 observation 和本步涉及的
Skill IDs。

当前固定数据清单使用 seed `20260825` 预先选定：development、valid_seen、
valid_unseen 各有 6 个 context × 8 个 game，共 144 个 game；三个 split 互不
重叠。清单位于 `SkillRL/phase1/config/game_ids/manifest.json`。

## 4. 初始冻结 Skill Bank

### 4.1 来源与构建方式

初始 Skill Bank 不是由本次 RL 实验动态生成的，而是从 SkillRL 上游文件
`memory_data/alfworld/claude_style_skills.json` 中按预先确定的规则抽取：

- 保留 1 个通用探索 Skill；
- 六个 ALFWorld context 各保留 1 个 task-specific Skill；
- 不保留 common mistakes；
- 为 `pick_two` 单独配置路由，不再复用 `pick_and_place`；
- 固定内容、顺序、触发规则和文件 hash；
- 使用 template retrieval，`task_specific_top_k=1`；
- 全部 RL update 中设置 `enable_dynamic_update=False`。

冻结文件为：
`SkillRL/phase1/config/frozen_alfworld_skills.json`。

### 4.2 Skill Bank 内容

| Skill ID | 类型/context | 核心作用 |
|---|---|---|
| `gen_001` | general | 系统探索所有尚未检查的表面和容器，减少重复搜索 |
| `pic_001` | pick_and_place | 首轮建立位置检查清单，找到目标前不重复访问 |
| `loo_001` | look_at_obj_in_light | 优先搜索通常放置台灯的桌面和边桌 |
| `cle_001` | clean | 固定执行“寻找→拿取→清洗→移动→放置”的阶段顺序 |
| `hea_001` | heat | 使用微波炉前先确认并拿到任务要求的准确目标物体 |
| `coo_001` | cool | 系统搜索未访问位置，找到目标后再执行冷却和放置 |
| `pic_005` | pick_two | 显式维护所需、已拿取和已放置物体的数量 |

总计 7 个唯一 Skill：1 个 general Skill 和 6 个 task-specific Skill。

## 5. 实验前准备状态

### 5.1 环境与硬件

- Conda 环境：`skill-RL`
- Python：3.10
- 核心软件：PyTorch 2.6.0、Transformers 4.51.1、vLLM 0.8.4、
  Ray 2.43.0、ALFWorld 0.4.2、FlashAttention 2.7.4.post1
- GPU：8 × NVIDIA A800 80GB PCIe
- CUDA：PyTorch CUDA 12.4；BF16 FlashAttention kernel 已完成工程级实算检查
- 依赖锁：`SkillRL/environment/requirements-lock.txt`

### 5.2 模型

| 角色 | 模型 | 本地路径 | 参数量 |
|---|---|---|---:|
| Smoke | Qwen2.5-0.5B-Instruct | `/home/wangyifan/model/Qwen2.5-0.5B-Instruct` | 494,032,768 |
| 能力不足时的 fallback | Qwen2.5-1.5B-Instruct | `/home/wangyifan/model/Qwen2.5-1.5B-Instruct` | 1,543,714,304 |

两个模型均已下载，并完成 config、tokenizer、safetensors 和离线整模加载检查。
模型 hash 记录在 `SkillRL/phase1/config/model_manifest.json`。

0.5B 主要用于先验证完整工程链路；如果 invalid action rate 高于 30%、无法形成
至少 4 个稳定正边际 pair，或不能稳定遵守 ALFWorld action protocol，则从 smoke
test 开始切换到 1.5B。0.5B 的能力失败不能直接作为研究假设失败的证据。

### 5.3 数据与固定配置

- ALFWorld 数据已下载；
- Skill Bank、game lists、模型 hash、seed、checkpoint cadence 和判定阈值已冻结；
- RL seeds：`101, 202, 303`；
- evaluation seeds：`11, 22, 33`；
- 计划执行 30 个 RL updates；
- checkpoint：`0, 5, 10, 15, 20, 25, 30`；
- 每次 update 使用 8 个 train games，每个 game 生成 4 条 rollout；
- episode 最大步数：30。

### 5.4 代码实现概况

现有准备代码主要覆盖：

- 只读 Skill 屏蔽：`FULL_BANK`、`MINUS_SKILL`、`NO_SKILL`；
- 同 state、同 Skill 条件的跨 checkpoint action-flip probe；
- episode-level matched Skill margin evaluator；
- zero-update、shuffled-reward 和 matched-norm random-parameter 三类负对照；
- game-cluster bootstrap、严格 sign-flip 标签、跨 seed 稳定性与 abstention；
- 不可变 run manifest、可恢复 JSONL 和每条 rollout 的完整轨迹归档。

代码入口与使用说明位于 `SkillRL/phase1/README.md`。

## 6. Phase I 实验验证步骤

### Step 0：冻结协议并完成 preflight

固定仓库 commit、模型、Skill Bank、game IDs、RL/eval seeds、prompt template 和
action projection。运行 preflight，确认环境、模型、GPU 和数据可用。任何协议
变更都必须使用新的 protocol version 和 run ID。

### Step 1：环境与 action smoke test

在固定 development games 上先运行无 RL、无 Skill 的 0.5B Policy，再依次运行
`FULL_BANK`、`MINUS_SKILL` 和 `NO_SKILL`：

- 检查模型能否遵守 ALFWorld action format；
- 统计 invalid action rate、episode termination 和基础 success rate；
- 检查目标 Skill 是否出现在 prompt 中；
- 检查单 Skill 屏蔽只删除目标 Skill，NO_SKILL 不注入任何 Skill；
- 验证轨迹和实验 manifest 是否完整落盘。

若 0.5B 不满足能力门槛，则使用 1.5B 从本步骤重新开始。

### Step 2：checkpoint 0 的 positive-pair 筛选

在 development games 上，对每个 `(Skill, context)` 运行 matched
`FULL_BANK` 与 `MINUS_SKILL` trajectories，计算 checkpoint 0 的边际效用。

只有同时满足以下条件的 pair 才进入后续实验：Skill 确实被检索、至少有 8 个
独立 game、边际效用点估计为正，并且 2,000 次 game-cluster bootstrap 得到的
95% CI 下界大于 0。筛选过程不得查看任何 post-update 结果，目标至少保留 4 个
eligible pairs。

### Step 3：执行真实 RL updates

从同一初始模型启动 3 个独立 RL seeds，采用 GRPO 更新 Policy，Skill Bank 始终
冻结。共执行 30 个 updates，每 5 个 updates 保存 checkpoint，并归档：

- train game IDs 和 rollout seeds；
- 完整 trajectories；
- rewards、returns 和 advantages；
- actor grad norm、trainer metrics；
- 相邻保存 checkpoint 的精确参数 L2 delta。

### Step 4：固定状态 action-flip 最小验证

从已归档的 FULL_BANK trajectories 中提取确实注入目标 Skill 的中间状态。对每个
固定状态，在 update 前后 checkpoint 上保持完全相同的 prompt、observation、
admissible action set 和 Skill condition，以 greedy decoding 比较：

- projected action 是否发生 flip；
- action 是否仍为 admissible；
- admissible-action 约束分布的 JS divergence。

该步骤是低成本行为信号验证，用于回答“RL 后同 state、同 Skill 下 Policy 动作
是否改变”，但 action flip 本身不等价于 Skill 从有益变为有害。

### Step 5：跨 checkpoint held-out matched evaluation

对每个 checkpoint、eligible `(Skill, context)` 和 held-out game，分别运行：

- `FULL_BANK`：完整固定 Skill Bank；
- `MINUS_SKILL`：只移除目标 Skill；
- `NO_SKILL`：完全关闭 Skill 注入。

matched conditions 共享 game、environment seed、evaluation seed、temperature、
top-p、最大步数、prompt template 和 action projection。轨迹在产生第一个不同
action 后可以自然分叉。该 episode-level 结果是 Skill 边际效用的 gold
measurement。

### Step 6：负对照

运行三类对照以排除评估噪声或普通参数扰动：

1. zero-update：不更新模型，重复 matched evaluation；
2. shuffled-reward：使用相同 rollout batch 和 reward multiset，但在 GRPO group
   内打乱 reward/advantage 与 trajectory 的对应关系；
3. random-parameter：沿随机方向扰动参数，扰动 L2 norm 与真实 update 匹配。

### Step 7：统计、稳定性和 Go/No-Go 判断

以 game instance 为 cluster 执行 bootstrap，计算每个 checkpoint、Skill、context
和 RL seed 的边际效用及 95% CI，并报告：

- 连续边际效用变化 `ΔM`；
- raw/reliable harmful sign-flip rate；
- ambiguous 与 low-support abstention rate；
- 3 个 RL seeds 的方向一致率与 Spearman correlation；
- 真实 update 与三类负对照的 effect size、CI 和置换检验。

只有现象数量、效应强度、跨 seed 稳定性和数据完整性同时达到冻结协议中的门槛，
才进入 Phase II 的 sign-flip prediction。若严格 flip 太少但连续负向变化充分，
后续任务调整为 continuous utility-change forecasting；若真实 update 与负对照无
差异，则停止 Policy-update-based forecasting 路线。

## 7. 实验记录与轨迹归档

每个 run 在执行前生成不可变 manifest，记录 repo commit、模型和 Skill Bank
hash、protocol hash、配置、seed、GPU 与依赖环境。

每条 rollout 单独保存完整 JSON，至少包括：

- game/context/checkpoint/update/seed/condition；
- 每步 prompt、observation 和 admissible actions；
- raw model output 与 projected action；
- reward、success、invalid action 和完整 action sequence；
- 每步 retrieved/injected/disabled Skill IDs；
- 该轨迹使用过的不同 Skill 集合和 `unique_skill_count`；
- prompt/completion token 数。

JSONL 使用唯一键和文件锁，重复运行时跳过已完成记录，避免覆盖或重复结果。

## 8. Phase I 测试与实验结果

Phase I 实验尚未运行，以下结果均待实验完成后补充：

| 项目 | 当前状态 | 后续汇报内容 |
|---|---|---|
| 0.5B action smoke test | 待定 | invalid action rate、success rate、协议遵循率 |
| 1.5B fallback 是否启用 | 待定 | 触发原因及重新 smoke-test 结果 |
| checkpoint 0 eligible pairs | 待定 | pair 数量、Skill/context 和初始 margin CI |
| 固定状态 action flip | 待定 | 各 condition 的 flip rate、admissibility、JS divergence |
| 跨 checkpoint Skill margin | 待定 | `M_t`、`ΔM`、置信区间和分层结果 |
| harmful sign flip | 待定 | raw/reliable flip 数量与比例 |
| 跨 seed 稳定性 | 待定 | 2/3 同方向比例、Spearman correlation |
| 负对照比较 | 待定 | real/zero/shuffled/random 的 effect size 与显著性 |
| Phase I Go/No-Go | 待定 | 是否进入 sign-flip 或 continuous-change prediction |

当前已完成的仅是工程就绪检查，包括环境导入、CUDA/FlashAttention、模型离线
加载、ALFWorld reset、配置组合和归档模块的单元检查；这些检查不构成 Phase I
实验结果。

## 9. 存储警告

当前根分区总容量约 7.0 TiB，已使用约 95%，剩余约 362 GiB。原始 proposal 对
完整实验建议预留 2–4 TiB NVMe，因此当前容量明显低于正式实验建议值。

0.5B smoke validation 可以在现有空间下开始，但在并行运行 3 个 seeds、保存 7
个 checkpoint、完整 rollout JSON、负对照 checkpoint 和 matched evaluation 前，
需要持续监控空间。建议优先控制 optimizer state、重复 checkpoint、临时 Ray/
vLLM 缓存和失败 run 的中间文件，并在扩展到 7B/8B 或外部 benchmark 前增加
存储容量。任何情况下都不应为了节省空间删除 run manifest、冻结配置、最终
checkpoint、trajectory index 或用于主结论的原始轨迹。

## 10. 相关文件

- 原始研究 proposal：`/home/wangyifan/skill-RL/2026-08-21-proposal-advantages-and-roadmap.md`
- Phase I 工程规范：`/home/wangyifan/skill-RL/2026-08-24-phase1-minimal-validation-repository-and-experiment-spec.md`
- 实验前准备报告：`/home/wangyifan/skill-RL/SkillRL/phase1/PREPARATION_REPORT.md`
- Phase I 使用说明：`/home/wangyifan/skill-RL/SkillRL/phase1/README.md`
- 冻结实验协议：`/home/wangyifan/skill-RL/SkillRL/phase1/config/phase1_protocol.json`
- 冻结 Skill Bank：`/home/wangyifan/skill-RL/SkillRL/phase1/config/frozen_alfworld_skills.json`
- 模型资产清单：`/home/wangyifan/skill-RL/SkillRL/phase1/config/model_manifest.json`
- 固定数据清单：`/home/wangyifan/skill-RL/SkillRL/phase1/config/game_ids/manifest.json`
