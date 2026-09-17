# Qwen3.5-4B Clean 全 Skill、三 RL Seed 边际效用分析

状态：Seed101/202/303 均训练至 update 30；完成 6,000 条三臂 suffix rollout，并通过完整性检查。

## 1. 核心结论

1. **RL 会改变固定 `(state, Skill)` 的边际效用。** 最稳定的结果来自 `cle_003`：U20 相对 B0 的三-seed pooled 变化为 `+40.56 pp [28.89, 52.78]`，U30 为 `+27.78 pp [8.33, 50.00]`。
2. **这种变化不只发生在 step 0。** `cle_003` 在 early 和 middle anchors 上也出现了可靠变化，说明结果不是初始 prompt 效应。
3. **Skill 会频繁改变模型行为，但动作变化不等于最终收益变化。** ORIGINAL 与 PLACEBO 的首动作有 26.95% 不同，后续轨迹有 41.70% 不同，最终 reward 有 16.15% 不同。
4. **不同 RL seed 可以让同一 Skill 朝不同方向变化。** 这反映不同训练轨迹产生的不同参数更新方向，不表示现象没有跨 seed 复现。
5. **全局任务成功率不能代替逐 Skill 效用。** 模型整体表现较好时，单个 Skill 仍可能变差；反之亦然。
6. **低成本动作交互信号 `S_int` 能跨 seed 预测 Skill 效用是否变化。** 加入 `S_int` 后，leave-one-seed-out AUPRC 从 0.192 提高到 0.274。
7. **当前不能用 `S_int` 稳定预测变化方向，也没有确认严格 harmful sign flip。** 方向预测留给后续 `P_int/D_t` 实验。

总之，当前结果支持：

> RL 会改变模型使用冻结 Skill 的方式；这种动作交互变化能够提前提示哪些 `(state, Skill)` 的效用可能发生变化。但目前不能据此判断效用一定变好或变坏。

## 2. 实验设置

### 2.1 模型与 checkpoint

三个 RL seed 从同一个 B0 `/home/wangyifan/model/Qwen3.5-4B` 出发，optimizer 独立初始化。每个 update 使用 8 games × 4 rollouts，共 32 条训练轨迹。checkpoint 预先固定为 update 10/20/30。

| Checkpoint | RL seed | 累计 update | 单 seed 累计训练 rollout |
|---|---:|---:|---:|
| B0 | — | 0 | 0 |
| U10 | 101 / 202 / 303 | 10 | 320 |
| U20 | 101 / 202 / 303 | 20 | 640 |
| U30 | 101 / 202 / 303 | 30 | 960 |

Seed101/202 使用 4-way FSDP，Seed303 使用 8-way FSDP；其余训练设置不变。

### 2.2 Skill 与 anchors

B0 在 Clean `valid_unseen` 的 31 个 games 上生成 93 条 FULL_BANK 轨迹，成功率为 55.91%，每条轨迹平均自然调用 3.77 种 Skill。

只有至少自然出现 30 次、覆盖至少 10 个 games 的 Skill 进入评测。最终评估 4 个 Skills，每个 Skill 选择 50 个首次调用 anchor：

| Skill | 含义 | 自然首次调用位置 | 入选 anchors / games | anchor 阶段 |
|---|---|---:|---:|---|
| `cle_003` | Sink First for Cleaning | step 2–26 | 50 / 30 | early 29；middle 18；late 3 |
| `cle_004` | Systematic Container Sweep | step 1–5 | 50 / 28 | early 46；middle 4 |
| `cle_006` | Use Location Priors | step 0 | 50 / 31 | initial 50 |
| `gen_002` | Immediate Acquisition | step 1–25 | 50 / 29 | early 28；middle 21；late 1 |

200 个 anchors 中，150 个不在 step 0，47 个位于 step ≥5，因此评测包含真实的中途 Skill 调用。

### 2.3 三臂评测

每个 anchor 先重放目标 Skill 第一次调用前的相同 action prefix，再从同一 state 分成三组：

| Arm | 输入内容 |
|---|---|
| ORIGINAL | 原始 Skill 内容 |
| PLACEBO | 模板和 token 数相同，但内容与任务无关 |
| NULL | 不输入 Skill 内容 |

分支后自由 rollout。若后续再次调用目标 Skill，继续使用同一组的设置。

主要指标为：

- `U_sem = ORIGINAL reward − PLACEBO reward`：Skill 语义的边际效用；
- `ΔU_sem = checkpoint U_sem − B0 U_sem`：RL 后 Skill 效用的变化。

成功记为 10、失败记为 0；报告统一换算为成功率百分点（pp）。置信区间采用 10,000 次 game-cluster bootstrap；三 seed 汇总按 `RL seed → game` 分层抽样。

## 3. Skill 边际效用如何变化

### 3.1 B0 的 Skill 效用

| Skill | B0 `U_sem`（95% CI） | 判断 |
|---|---:|---|
| `cle_003` | **+21.67 [+8.33, +36.67]** | 可靠为正 |
| `cle_004` | +7.14 [+0.00, +17.86] | 倾向为正，但区间接触 0 |
| `cle_006` | -3.23 [-8.06, +0.00] | 未可靠偏离 0 |
| `gen_002` | +0.00 [+0.00, +0.00] | 无效用差异 |

### 3.2 相对 B0 的变化

| Update | Skill | Seed101 Δ | Seed202 Δ | Seed303 Δ | 三-seed pooled Δ（95% CI） |
|---:|---|---:|---:|---:|---:|
| 10 | `cle_003` | +21.67 | +33.33 | -3.33 | +17.22 [-4.44, +35.56] |
| 10 | `cle_004` | -32.14 | +0.00 | -7.14 | -13.10 [-32.14, +2.38] |
| 10 | `cle_006` | +3.23 | +4.84 | -1.61 | +2.15 [-2.69, +6.99] |
| 10 | `gen_002` | +1.72 | +0.00 | +0.00 | +0.57 [+0.00, +2.30] |
| 20 | `cle_003` | +48.33 | +33.33 | +40.00 | **+40.56 [+28.89, +52.78]** |
| 20 | `cle_004` | -3.57 | -5.36 | -7.14 | -5.36 [-12.50, +1.79] |
| 20 | `cle_006` | +3.23 | +6.45 | +0.00 | +3.23 [-1.08, +7.53] |
| 20 | `gen_002` | +1.72 | -1.72 | +1.72 | +0.57 [-3.45, +3.45] |
| 30 | `cle_003` | +50.00 | +10.00 | +23.33 | **+27.78 [+8.33, +50.00]** |
| 30 | `cle_004` | -3.57 | +5.36 | +0.00 | +0.60 [-8.33, +10.12] |
| 30 | `cle_006` | +3.23 | +0.00 | +3.23 | +2.15 [-2.15, +5.38] |
| 30 | `gen_002` | +0.00 | +3.45 | +0.00 | +1.15 [-0.57, +4.60] |

核心结果：

- `cle_003@U20/U30` 是唯一跨三个 seed 都可靠为正的效用变化。
- 其他 Skill 的变化较小、区间跨 0，或随 seed 改变方向。
- 没有 Skill 满足“B0 可靠为正、RL 后可靠为负”的严格 harmful sign flip 条件。
- Seed101 U10 的 `cle_004` 明显变负，但其 B0 区间接触 0，因此只能算 harmful candidate，不能算 confirmed flip。

## 4. 中途调用是否也会变化

`cle_003` 的首次调用全部发生在 step 2 之后：

| 调用阶段 | anchors / games | B0 `U_sem` | pooled ΔU10 | pooled ΔU20 | pooled ΔU30 |
|---|---:|---:|---:|---:|---:|
| early | 29 / 22 | +25.00 [+9.09, +43.18] | +16.67 [-9.85, +42.42] | **+34.85 [+21.21, +49.24]** | **+26.52 [+7.58, +45.45]** |
| middle | 18 / 15 | +26.67 [+6.67, +53.33] | +6.67 [-16.67, +25.56] | **+33.33 [+17.78, +51.11]** | +15.56 [-3.33, +37.78] |
| late | 3 / 3 | +33.33 [+0.00, +100.00] | +0.00 | +22.22 [+0.00, +66.67] | +11.11 [+0.00, +44.44] |

U20 的 early 和 middle 结果都可靠为正，U30 的 early 结果也可靠为正。因此，Skill 效用变化不仅出现在初始 prompt，也出现在已有多步轨迹前缀的中间 state。

## 5. Skill 是否改变模型行为

这里比较相同 anchor 下 ORIGINAL 与 PLACEBO：

- `first-action flip`：第一步动作不同；
- `suffix divergence`：后续动作序列不同；
- `reward contrast`：最终一个成功、一个失败；
- `good/bad`：ORIGINAL 相比 PLACEBO 最终更好/更差。

全部 2,000 个 checkpoint-anchor pairs 的结果为：

| 指标 | 结果 |
|---|---:|
| first-action flip | 539 / 2,000 = **26.95%** |
| suffix divergence | 834 / 2,000 = **41.70%** |
| reward contrast | 323 / 2,000 = **16.15%** |
| good / bad / zero | 286 / 37 / 1,677 |
| `P(reward contrast | first-action flip)` | 285 / 539 = **52.88%** |

结论：真实 Skill 经常改变第一步动作和后续轨迹，但动作不同不一定改变最终成败。约一半 first-action flip 最终没有产生 reward contrast，因此不能把 action flip 直接解释为“变好”或“变坏”。

## 6. 不同 RL seed 的结果是否一致

下表比较相同 anchor 在三个 seed 中的 `ΔU_sem` 方向：

| Update / Skill | 至少一个 seed 非零 | 三 seed全正 | 三 seed全负 | 正负冲突 | 三 seed全零 |
|---|---:|---:|---:|---:|---:|
| U10 / `cle_003` | 30 | 4 | 0 | 1 | 20 |
| U10 / `cle_004` | 18 | 0 | 3 | 4 | 32 |
| U20 / `cle_003` | 27 | **11** | 0 | **0** | 23 |
| U20 / `cle_004` | 12 | 0 | 3 | 2 | 38 |
| U30 / `cle_003` | 30 | 4 | 0 | 1 | 20 |
| U30 / `cle_004` | 16 | 0 | 3 | 0 | 34 |

`cle_003@U20` 最一致：11 个 anchors 在三个 seed 中都正向变化，没有正负冲突。其他组合更依赖 seed。

这里不要求同一 Skill 在三个 seed 中都朝相同方向变化。不同 seed 使用不同训练采样并走向不同参数更新方向，出现不同效用方向是合理的。跨 seed 要验证的是“RL 后会出现 Skill 效用变化”以及“动作交互信号能否预测各 seed 自己的变化”。

## 7. 全局成功率不能替代 Skill 效用

| RL seed / update | train success | valid_seen success | train valid-action ratio |
|---|---:|---:|---:|
| 101 / 10 | 84.38% | 81.48% | 98.03% |
| 101 / 20 | 100.00% | 88.89% | 100.00% |
| 101 / 30 | 100.00% | 81.48% | 100.00% |
| 202 / 10 | 84.38% | 85.19% | 99.53% |
| 202 / 20 | 84.38% | 77.78% | 99.34% |
| 202 / 30 | 87.50% | 85.19% | 100.00% |
| 303 / 10 | 68.75% | 55.56% | 97.24% |
| 303 / 20 | 96.88% | 66.67% | 100.00% |
| 303 / 30 | 40.62% | 66.67% | 89.29% |

全局成功率和单个 Skill 的边际效用并不同步。例如：

- Seed101 U10 的 `valid_seen` 成功率为 81.48%，但 `cle_004` 的效用明显为负；
- Seed303 U30 的 train success 只有 40.62%，但 `cle_004` 在该 checkpoint 内仍有正效用。

因此，全局成功率只能说明 policy 整体表现，不能判断某个 Skill 对当前 policy 是否有帮助。

## 8. 低成本预测：动作交互变化能否预测效用变化

### 8.1 `S_int` 是什么

对同一个 anchor，分别比较 RL 更新前后 ORIGINAL 和 PLACEBO 的第一步动作：

- 如果只有一组的动作发生变化，记为 `S_int=1`；
- 否则记为 `S_int=0`。

`S_int=1` 表示 RL 对真实 Skill 的响应变化不同于一般 prompt 对照，即出现了 Skill-specific action interaction shift。

预测标签是 `utility_changed`：同一 anchor 的 `U_sem` 在相邻 checkpoint 间是否变化。共分析 3 seeds × 3 update windows × 200 anchors = 1,800 个样本。

### 8.2 直接关联

| RL seed | `S_int=1` 样本数 | P(utility change｜`S_int=1`) | P(utility change｜`S_int=0`) | 差值（95% CI） | risk ratio |
|---:|---:|---:|---:|---:|---:|
| 101 | 132 | 27.54% | 6.58% | **+20.97 pp [+12.23, +29.97]** | 4.19 |
| 202 | 130 | 39.60% | 10.08% | **+29.52 pp [+18.72, +40.26]** | 3.93 |
| 303 | 97 | 38.33% | 8.55% | **+29.78 pp [+19.50, +40.01]** | 4.48 |
| 三-seed pooled | 359 | 34.90% | 8.41% | **+26.49 pp [+19.06, +34.01]** | 4.15 |

三个 seed 中结果一致：出现 `S_int` 时，Skill 效用发生变化的概率约为未出现时的 4 倍。

### 8.3 Leave-one-seed-out 预测

这里训练的不是 Qwen，而是一个小型 L2 Logistic Regression：用两个 seed 的样本学习“哪些信号对应效用变化”，再测试完全没参与训练的第三个 seed。三个 seed 轮流作为测试集。

比较四组输入：

| 模型 | 使用的信息 |
|---|---|
| prevalence | 只猜训练集平均变化率 |
| old-margin | 更新前 Skill 效用 `M_t` |
| old+generic | `M_t` + PLACEBO 动作是否变化 |
| old+interaction | 上述信息 + `S_int` |

AUPRC 越高表示越能找出真正发生效用变化的 anchors：

| Held-out seed | Prevalence | Old-margin | Old+generic | Old+interaction |
|---:|---:|---:|---:|---:|
| 101 | 0.112 | 0.098 | 0.141 | **0.202** |
| 202 | 0.167 | 0.180 | 0.225 | **0.335** |
| 303 | 0.135 | 0.218 | 0.210 | **0.283** |
| 三折宏平均 | 0.138 | 0.165 | 0.192 | **0.274** |

加入 `S_int` 后，三个 held-out seed 的 AUPRC 都提高，宏平均从 0.192 提高到 0.274，增幅约 42%。这说明模型学到的不是“某个 Skill 固定会变好或变坏”，而是可迁移到新 RL seed 的关系：

> 当 RL 特别改变了模型对真实 Skill 的动作响应时，该 Skill 在这个 state 上的效用更可能发生变化。

### 8.4 能否预测效用下降方向

`negative_delta` 表示 Skill 效用相对前一个 checkpoint 下降，不等于已经发生 harmful flip。

| RL seed | P(negative｜`S_int=1`) | P(negative｜`S_int=0`) | 差值（95% CI） |
|---:|---:|---:|---:|
| 101 | 4.70% | 3.07% | +1.64 pp [-1.20, +4.64] |
| 202 | 16.32% | 5.03% | **+11.28 pp [+4.61, +17.56]** |
| 303 | 16.55% | 3.43% | **+13.12 pp [+5.98, +20.67]** |
| 三-seed pooled | 12.18% | 3.83% | **+8.35 pp [+1.89, +15.24]** |

`S_int=1` 时效用下降整体更常见，但跨 seed 方向预测没有稳定提高：

- 不加入 `S_int` 的 `old+generic` 宏平均 AUPRC：0.317；
- 加入 `S_int` 的 `old+interaction` 宏平均 AUPRC：0.314。

因此，`S_int` 能预测“效用是否变化”，但目前不能稳定预测“效用会上升还是下降”。这不是本轮主假设失败，因为 `S_int` 本身是无方向的动作变化信号。

### 8.5 成本与后续方向验证

完整三臂评测包含 81,630 次 suffix action decisions。若更新前 probe 已缓存，9 个 post-RL checkpoints 的 ORIGINAL+PLACEBO 单步 probe 只需 3,600 次 action decisions，相当于完整评测的 4.41%。

后续将使用真实单次 RL update 计算：

- `P_int`：Skill-specific action shift 与 reward direction 对齐还是相反；
- `D_t`：reward-opposing interaction 的风险强度。

再用独立三臂 rollout 检验它们能否预测 `ΔU_sem` 的方向。高 `D_t` 只是风险信号，不等于一定发生 harmful flip。

## 9. 最终结论边界

当前实验支持：

- RL 会改变固定 state 上 Skill 的动作影响和累计边际效用；
- 这种变化覆盖 step 0 以外的 early/middle states；
- `S_int` 与效用变化显著相关，并能提高未见 RL seed 的预测结果；
- 不同 seed 中同一 Skill 的变化方向可以不同。

当前实验不支持：

- RL 会普遍让旧 Skill 变坏；
- `S_int` 能稳定预测效用变化方向或 harmful sign flip；
- 当前结果已经泛化到其他 task type、Skill family 或新的 state support。
