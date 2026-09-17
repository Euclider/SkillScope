# Qwen3.5-4B Clean 全 Skill、跨 RL Seed 首次调用边际效用评测

> 日期：2026-09-03  
> 评测 run：`qwen35-clean-all-skill-seeds101-202-v1`  
> 状态：已完成并通过完整性审计，共 4,200 条三臂 suffix rollout

## 1. 研究问题

本轮沿用 `2026-08-29-first-invocation-skill-utility-evaluation-results.md` 的首次实际调用点三臂协议，但把单一 `pic_001`、单一 RL seed、全部 step 0 anchor 扩展为：

- Qwen3.5-4B 公共 B0，以及 Seed 101/202 各自 update 10/20/30；
- Clean context 中所有被冻结 Router 自然调用且达到预注册支持阈值的 Skill；
- `valid_unseen` 的 31 个 concrete games，每个 game 三个 generation seeds；
- step 0、step 1–4、step 5–14、step ≥15 四种首次调用阶段；
- 固定同一批 B0 anchor，对不同 checkpoint 和 RL seed 做配对纵向比较。

需要回答的是：真实 RL update 后，同一固定 `(state, Skill)` 上 Skill 内容相对等长无关文本的累计边际效用是否发生变化；这种变化在不同 Skill、调用阶段和 RL seed 间是否复现或呈现异质性。

## 2. 模型、训练链与 checkpoint

公共 B0 是 `/home/wangyifan/model/Qwen3.5-4B`。Seed 101 与 Seed 202 使用相同训练配置、相同 B0，分别初始化随机训练路径；每个 global update 使用 8 games × 4 rollouts = 32 条训练轨迹。checkpoint 由预先固定的累计 update 里程碑决定，没有查看 Skill utility 或 validation performance 后再挑点。

| 模型 | RL seed | 累计 update | 单 seed 累计训练 rollout |
|---|---:|---:|---:|
| B0 | — | 0 | 0 |
| C1 | 101 / 202 | 10 | 320 |
| C2 | 101 / 202 | 20 | 640 |
| C3 | 101 / 202 | 30 | 960 |

同一 seed 内 C1→C2→C3 是连续参数更新，不是三个独立模型；两个 seed 之间才是从共同 B0 分叉的独立随机训练路径。Seed 303 在 update 12 暂停并保留恢复点，因为还没有完整 C2/C3，本报告不把它混入已预注册的 0/10/20/30 纵向比较。

## 3. 固定 anchor 支持集与 Skill 覆盖

B0 在 Clean `valid_unseen` 的 31 个 games 上以 generation seed `1101/2202/3303` 生成 93 条 FULL_BANK 来源轨迹。来源轨迹成功 52/93（55.91%），平均每条轨迹自然调用 3.77 种不同 Skill。Skill 是否进入评测只看自然 Router 调用次数和 game 覆盖，不读取 reward。

预注册支持阈值为至少 30 个自然 occurrence、至少 10 个不同 game；达到阈值后，用按 game 轮转的确定性规则最多选 50 个 anchor，避免同一 game 的多个 generation seed 垄断样本。

| Skill | 含义 | 自然 occurrence / games / states | 自然首次调用 step | 入选 anchors / games / states | 入选阶段分布 |
|---|---|---:|---:|---:|---|
| `cle_003` | Sink First for Cleaning | 89 / 30 / 60 | 2–26，均值 5.75 | 50 / 30 / 42 | early 29；middle 18；late 3 |
| `cle_004` | Systematic Container Sweep | 83 / 28 / 35 | 1–5，均值 1.47 | 50 / 28 / 32 | early 46；middle 4 |
| `cle_006` | Use Location Priors | 93 / 31 / 31 | 全部 step 0 | 50 / 31 / 31 | initial 50 |
| `gen_002` | Immediate Acquisition | 85 / 29 / 49 | 1–25，均值 4.33 | 50 / 29 / 38 | early 28；middle 21；late 1 |

总支持集为 200 个 anchor。其中 150 个首次调用不在 step 0，47 个在 step ≥5；入选 anchor 的最晚首次调用为 `cle_003` step 22 和 `gen_002` step 21。因此本轮确实包含固定非空 prefix 后的中途干预，而不只是旧实验的初始状态干预。

18 个候选中有 14 个按预注册规则 abstain：`cle_002` 只有 1 occurrence/1 game（step 12），其余 `gen_001`、`gen_003`–`gen_012`、`cle_001`、`cle_005` 为零自然调用。它们均记录为 unsupported，没有人为强制 Router 调用。

## 4. 三臂干预与估计量

每个 anchor 固定重放首次自然调用前的环境 action prefix，并核验到达相同 state。从首次调用位置开始：

| Arm | 冻结 Router 中目标 Skill | policy prompt 中 payload |
|---|---|---|
| ORIGINAL | 保留，正常参与同一确定性路由 | 原 Skill 内容 |
| PLACEBO | 与 ORIGINAL 完全相同 | 保留原 section/bullet 模板、token 数严格相同的任务无关文字 |
| NULL | 与 ORIGINAL 完全相同 | payload 为空 |

分支后 policy 和 Router 均自由运行；后续若再次选择目标 Skill，持续使用该 arm 的 payload。因而估计的是从第一次自然调用开始对后续累计 reward 的影响，而不是仅比较触发点单步 action。

- 主指标：`U_sem = R(ORIGINAL) − R(PLACEBO)`；
- 总效应：`U_total = R(ORIGINAL) − R(NULL)`；
- prompt nuisance：`U_prompt = R(PLACEBO) − R(NULL)`；
- 纵向变化：`ΔU_sem(k) = U_sem(checkpoint k) − U_sem(B0)`。

ALFWorld 本轮仍为成功 10、失败 0，return 差乘以 10 即 success percentage points（pp）。置信区间为 10,000 次 bootstrap：单 seed 内按 game 聚类；跨 seed 先抽 RL seed、再在 seed 内抽 game。多个 generation seed 不作为独立 game 计数。

`harmful sign flip` 只在 B0 的 Skill-level `U_sem` 95% CI 严格大于 0、post checkpoint 的 95% CI 严格小于 0 时成立。一般 action flip、单 anchor 正负转换和均值 CI 跨 0 都不能替代这一判据。

## 5. 完整结果

以下所有效用均换算为 success percentage points。区间为 game-cluster 95% bootstrap CI；区间端点恰好为 0 不视为严格排除 0。

### 5.1 各 checkpoint 的横向三臂结果

每个单元格为 `ORIGINAL/PLACEBO/NULL 成功数；U_sem [95% CI]`，每个 Skill 分母均为 50。

| Checkpoint | `cle_003` | `cle_004` | `cle_006` | `gen_002` |
|---|---|---|---|---|
| B0 | 27/14/10；+21.67 [+8.33, +36.67] | 25/22/23；+7.14 [+0.00, +17.86] | 26/28/27；-3.23 [-8.06, +0.00] | 28/28/26；+0.00 [+0.00, +0.00] |
| Seed101 U10 | 31/9/6；+43.33 [+26.67, +60.00] | 21/34/33；-25.00 [-37.50, -12.50] | 29/29/29；+0.00 [+0.00, +0.00] | 37/36/36；+1.72 [+0.00, +5.17] |
| Seed101 U20 | 35/0/0；+70.00 [+53.33, +86.67] | 42/40/40；+3.57 [+0.00, +10.71] | 36/36/36；+0.00 [+0.00, +0.00] | 41/40/40；+1.72 [+0.00, +5.17] |
| Seed101 U30 | 34/0/0；+71.67 [+55.00, +86.67] | 42/40/39；+3.57 [+0.00, +10.71] | 40/40/40；+0.00 [+0.00, +0.00] | 44/44/44；+0.00 [+0.00, +0.00] |
| Seed202 U10 | 35/7/6；+55.00 [+38.33, +71.67] | 34/29/29；+7.14 [-5.36, +19.64] | 37/37/37；+1.61 [-6.45, +11.29] | 32/32/33；+0.00 [+0.00, +0.00] |
| Seed202 U20 | 28/1/0；+55.00 [+36.67, +73.33] | 37/36/34；+1.79 [-3.57, +8.93] | 37/35/35；+3.23 [+0.00, +8.06] | 37/37/38；-1.72 [-10.34, +5.17] |
| Seed202 U30 | 35/16/10；+31.67 [+13.33, +50.00] | 39/32/34；+12.50 [+0.00, +26.79] | 38/40/41；-3.23 [-9.68, +0.00] | 42/40/41；+3.45 [-3.45, +10.34] |

横向看，`cle_003` 是唯一在 B0 已可靠为正的 Skill，并在所有 post-RL checkpoint 保持可靠正效用。`cle_004` 的 Seed101 U10 则可靠为负，但在 U20/U30 恢复到接近零的小正值。`cle_006` 和 `gen_002` 的效应主要集中在少数 anchor，checkpoint 内区间多接触或跨过 0。

三臂控制并非多余。以 Seed101 U10 的 `cle_004` 为例，`U_sem=-25.00 pp`，`U_total=-23.21 pp [-35.71,-12.50]`，而 `U_prompt=+1.79 pp [+0.00,+5.36]`；大的负效应来自原 Skill 语义相对等长无关文本，而不是简单的“prompt 中多了一段文字”。`cle_003` 的部分 checkpoint 存在正的 prompt nuisance，例如 Seed202 U30 为 +13.33 pp [+0.00,+26.67]，所以其 ORIGINAL−NULL 总效应不能全部归因于语义；主结论始终使用 ORIGINAL−PLACEBO。

### 5.2 同一 anchor 上相对 B0 的纵向变化

| Skill | B0 `U_sem` | Seed101 ΔU10 | Seed101 ΔU20 | Seed101 ΔU30 | Seed202 ΔU10 | Seed202 ΔU20 | Seed202 ΔU30 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cle_003` | +21.67 [+8.33, +36.67] | +21.67 [+6.67, +38.33] | +48.33 [+31.67, +66.67] | +50.00 [+33.33, +66.67] | +33.33 [+16.67, +51.67] | +33.33 [+18.33, +50.00] | +10.00 [-6.67, +26.67] |
| `cle_004` | +7.14 [+0.00, +17.86] | -32.14 [-50.00, -17.86] | -3.57 [-14.29, +7.14] | -3.57 [-17.86, +7.14] | +0.00 [-16.07, +16.07] | -5.36 [-17.86, +5.36] | +5.36 [-12.50, +21.43] |
| `cle_006` | -3.23 [-8.06, +0.00] | +3.23 [+0.00, +8.06] | +3.23 [+0.00, +8.06] | +3.23 [+0.00, +8.06] | +4.84 [+0.00, +12.90] | +6.45 [+1.61, +12.90] | +0.00 [-8.06, +6.45] |
| `gen_002` | +0.00 [+0.00, +0.00] | +1.72 [+0.00, +5.17] | +1.72 [+0.00, +5.17] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -1.72 [-10.34, +5.17] | +3.45 [-3.45, +10.34] |

这张表直接回答固定 `(state, Skill)` 的效用漂移。`cle_003` 在 Seed101 三个点和 Seed202 U10/U20 的 Δ 区间均严格大于 0；Seed202 U30 仍为正点估计，但区间跨 0。`cle_004` 的巨大负向变化只出现在 Seed101 U10，随后恢复，并未在 Seed202 复现。`cle_006` 的 Seed202 U20 是单 seed 下严格正的改善，其余多为接触 0 的小变化。

### 5.3 跨 RL seed 的层级汇总

跨 seed point estimate 是先对每个 seed 的 game-level mean 等权，再在 `RL seed → game` 两层 bootstrap；不是把 100 个 seed-anchor 行当独立样本。

| Update | Skill | Seed101 Δ | Seed202 Δ | 跨 seed pooled Δ（95% CI） | 两 seed 非零同号 |
|---:|---|---:|---:|---:|---|
| 10 | `cle_003` | +21.67 | +33.33 | +27.50 [+14.17, +42.50] | 是 |
| 10 | `cle_004` | -32.14 | +0.00 | -16.07 [-40.18, +7.14] | 否 |
| 10 | `cle_006` | +3.23 | +4.84 | +4.03 [+0.81, +8.87] | 是 |
| 10 | `gen_002` | +1.72 | +0.00 | +0.86 [+0.00, +3.45] | 否 |
| 20 | `cle_003` | +48.33 | +33.33 | +40.83 [+25.83, +56.67] | 是 |
| 20 | `cle_004` | -3.57 | -5.36 | -4.46 [-13.39, +3.57] | 是，但不确定 |
| 20 | `cle_006` | +3.23 | +6.45 | +4.84 [+0.81, +9.68] | 是 |
| 20 | `gen_002` | +1.72 | -1.72 | +0.00 [-5.17, +3.45] | 否 |
| 30 | `cle_003` | +50.00 | +10.00 | +30.00 [+2.50, +58.33] | 是 |
| 30 | `cle_004` | -3.57 | +5.36 | +0.89 [-10.71, +13.39] | 否 |
| 30 | `cle_006` | +3.23 | +0.00 | +1.61 [-4.03, +6.45] | 否 |
| 30 | `gen_002` | +0.00 | +3.45 | +1.72 [-0.86, +6.90] | 否 |

跨 seed 可复现的主结果是：`cle_003` 在 U10/U20/U30 的 pooled Δ 均严格为正；`cle_006` 在 U10/U20 有小但严格为正的 pooled Δ。其余 Skill/update 要么 seed 方向不同，要么区间跨/接触 0。尤其 `cle_004` U10 的 pooled CI 跨 0，说明不能把 Seed101 的负效应声称为随机训练路径下的普遍规律。

`cle_003` U30 虽然两 seed 都为正，但 Seed101 为 +50 pp、Seed202 只有 +10 pp，between-seed range 达 40 pp；两 seed 能说明方向的初步复制，却不足以精确估计效应方差。

### 5.4 非初始调用阶段

| Skill / phase | anchors / games | B0 `U_sem` | pooled ΔU10 | pooled ΔU20 | pooled ΔU30 |
|---|---:|---:|---:|---:|---:|
| `cle_003` / early | 29/22 | +25.00 [+9.09, +43.18] | +29.55 [+10.23, +50.00] | +35.23 [+18.18, +52.27] | +27.27 [+1.14, +52.27] |
| `cle_003` / middle | 18/15 | +26.67 [+6.67, +53.33] | +15.00 [-1.67, +31.67] | +36.67 [+16.67, +60.00] | +20.00 [-5.00, +48.33] |
| `cle_003` / late | 3/3 | +33.33 [+0.00, +100.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +16.67 [+0.00, +50.00] |
| `cle_004` / early | 46/26 | +7.69 [+0.00, +19.23] | -17.31 [-42.31, +7.69] | -4.81 [-14.42, +3.85] | +0.96 [-11.54, +14.42] |
| `cle_004` / middle | 4/2 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |
| `cle_006` / initial | 50/31 | -3.23 [-8.06, +0.00] | +4.03 [+0.81, +8.87] | +4.84 [+0.81, +9.68] | +1.61 [-4.03, +5.65] |
| `gen_002` / early | 28/18 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | -1.39 [-9.72, +4.17] | +4.17 [+0.00, +12.50] |
| `gen_002` / middle | 21/14 | +0.00 [+0.00, +0.00] | +1.79 [+0.00, +7.14] | +1.79 [+0.00, +7.14] | -1.79 [-7.14, +0.00] |
| `gen_002` / late | 1/1 | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] | +0.00 [+0.00, +0.00] |

最有统计支持的非初始结果是 `cle_003`：early 层在三个 update 的跨 seed Δ 均严格为正；middle 层在 U20 严格为正，U10/U30 同号但区间跨 0。这说明效用变化不是由 step-0 prompt alone 驱动，而能在已执行若干动作、持有目标物体后的固定中间状态出现。late 的 3 个 `cle_003` anchor 和 1 个 `gen_002` anchor 只作为链路验证，不作泛化结论。`cle_004` 的有效支持几乎全部在 early；4 个 middle anchor 仅来自 2 games，不能解释为真正的“零效应”。

### 5.5 Action flip、suffix divergence 与 reward

下表跨四个 Skill 汇总，每个 checkpoint 为 200 个配对 anchor。`good/bad/zero` 指 `U_sem` 在单 anchor 上为正/负/零。

| Checkpoint | first-action O/P flip | suffix O/P divergence | 非零 reward contrast | good / bad / zero | P(非零 reward｜first flip) |
|---|---:|---:|---:|---:|---:|
| B0 | 29.5% | 55.5% | 9.0% | 16 / 2 / 182 | 28.8% |
| Seed101 U10 | 34.0% | 46.5% | 18.0% | 23 / 13 / 164 | 51.5% |
| Seed101 U20 | 26.5% | 34.0% | 19.0% | 38 / 0 / 162 | 66.0% |
| Seed101 U30 | 24.0% | 31.5% | 18.0% | 36 / 0 / 164 | 70.8% |
| Seed202 U10 | 30.0% | 44.0% | 19.5% | 36 / 3 / 161 | 55.0% |
| Seed202 U20 | 29.5% | 40.5% | 17.0% | 32 / 2 / 166 | 55.9% |
| Seed202 U30 | 29.0% | 42.0% | 18.0% | 31 / 5 / 164 | 51.7% |

全部 1,400 个 checkpoint-anchor pair 中，first-action flip 为 28.93%，完整 suffix divergence 为 42.00%，但 terminal reward contrast 非零只有 16.93%。即使已经 first-action flip，也只有 53.58% 最终产生非零 reward 差。因此 action flip 是高灵敏度的 policy–Skill 交互读数，但不能自身提供“好/坏”方向；累计 reward 仍是方向标签。

不同 Skill 的灵敏度差异很大。`cle_003` 在 Seed101 U20/U30 的 first-action flip 为 96%/94%、suffix divergence 均为 100%，同时 `U_sem` 为 +70.00/+71.67 pp；相反，B0 `gen_002` 虽有 34% suffix divergence，`U_sem` 却在全部 50 个 anchor 上恰为 0。

### 5.6 全局 policy performance 并不能替代 Skill utility

checkpoint 的常规 monitor 来自独立 `valid_seen` Clean games，不参与 checkpoint 选择：

| RL seed / update | train success | valid_seen success | train valid-action ratio |
|---|---:|---:|---:|
| 101 / 10 | 84.38% | 81.48% | 98.03% |
| 101 / 20 | 100.00% | 88.89% | 100.00% |
| 101 / 30 | 100.00% | 81.48% | 100.00% |
| 202 / 10 | 84.38% | 85.19% | 99.53% |
| 202 / 20 | 84.38% | 77.78% | 99.34% |
| 202 / 30 | 87.50% | 85.19% | 100.00% |

这些数值本身震荡，而且与单个 Skill 的效用轨迹并不一一对应。例如 Seed101 U10 的全局 valid_seen success 为 81.48%，但 `cle_004` 在固定 valid_unseen anchor 上可靠为负；到 U20 全局性能升高且该 Skill 恢复。Seed101 U30 的全局性能又回落，`cle_003` 的固定支持效用却仍很高。由此不能用 episode success 的一个全局标量推断每个旧 Skill 的边际效用方向。

### 5.7 完整性

- 7 checkpoints × 4 Skills × 50 anchors × 3 arms = 4,200/4,200 条；
- `(checkpoint, Skill, anchor, arm)` 重复 0，trajectory path 4,200 个唯一；
- prefix replay/payload 干预失败 0；
- ORIGINAL/PLACEBO 触发 prompt token mismatch 0；
- 三臂触发 Router score mismatch 0；
- B0 ORIGINAL 与 200 个来源 suffix 的 action sequence 或 success mismatch 0；
- 4,200 个完整 trajectory JSON、28 个执行 index 和 28 个日志均已归档；
- phase1 测试 `33 passed`，新增代码 Ruff 检查通过。

## 6. 结果解释与边界

### 6.1 核心结论

本轮对 proposal 的核心机制给出更强支持：真实 RL 参数更新后，同一批固定 `(state, Skill)` 上的累计语义边际效用会显著变化，而且变化既可能跨随机训练路径复现，也可能只在某个 seed 的某个阶段短暂出现。

最稳健的正例是 `cle_003`。B0 ORIGINAL 在 47/50 anchor 的第一步直接去 `sinkbasin 1`；等长 PLACEBO 只有 18/50。RL 后这种语义响应进一步增强：Seed101 U10 的 ORIGINAL 50/50 去 sink，而 U20/U30 的高频 O/P action divergence 最终转化为大幅成功差。两个 seed、三个累计 update、early 和 middle state 的结果共同表明，这不是单一初始 prompt 或单一训练随机性的假象。

最重要的负向案例是 `cle_004` Seed101 U10：

- checkpoint 内 `U_sem=-25.00 pp [-37.50,-12.50]`；
- 相对 B0 `ΔU_sem=-32.14 pp [-50.00,-17.86]`；
- 13 个负 anchor 分布在 11 个 games，首次调用均在 step 1–2；
- 13 个负 anchor 中 12 个当步发生 O/P action flip；
- ORIGINAL 平均 invalid/projection count 为 6.08，PLACEBO 为 0，NULL 为 0.38；
- ORIGINAL 常转向 `examine coffeemachine/shelf` 并形成检查循环，PLACEBO 在 12/13 个负 anchor 首步为 `go to countertop 2`，平均 suffix 长度分别为 28.69 与 7.00。

这说明负效应有可解释的行为路径：更新后的 Seed101 U10 policy 对“Systematic Container Sweep”内容产生过强或失配响应，偏向反复 examine，而等长无关文本反而让它继续更有效的搜索路径。但该现象在 Seed202 U10 不复现，并在 Seed101 U20/U30 消失，因此应称为 **seed-specific transient harmful interaction**，不能称为普遍的 RL-induced Skill degradation。

### 6.2 严格 harmful sign flip 判据

本轮 **没有** 任何 pair 满足预注册的可靠 harmful sign flip：B0 可靠为正的只有 `cle_003`，而它在所有 post checkpoint 仍可靠为正。`cle_004` 的 B0 点估计为 +7.14 pp，但 CI 下界恰为 0，不满足“B0 CI 严格大于 0”；因此 B0→Seed101 U10 虽然是明显的点估计符号反转、post 可靠为负且 Δ 可靠为负，仍只能列为高价值候选，而不能升级成预注册的 confirmed harmful sign flip。

anchor 层的符号 turnover 同样异质。Seed101 U10 的 `cle_004` 有 1 个 `positive→negative`、2 个 `positive→zero` 和 12 个 `zero→negative`；聚合负效应不是所有 state 同时翻转。其他 Skill/update 也主要由少量 `zero↔positive/negative` anchor 驱动。这再次说明 CI 跨 0 或均值为 0 不代表每个 state 的效用都没有变化。

### 6.3 对“跨 seed 泛化”的回答

当前证据可分三级：

1. **初步跨 seed 复现且 pooled CI 排除 0**：`cle_003` U10/U20/U30；`cle_006` U10/U20。
2. **两个 seed 点估计同方向但 pooled CI 跨 0**：例如 `cle_004` U20，说明方向一致但证据不足。
3. **seed-specific 或方向冲突**：`cle_004` U10/U30、`gen_002` U20 等，不能泛化。

anchor 层也显示这种差别。`cle_003` U20 的两个 seed 在 22 个至少一边非零的 anchor 中，有 14 个为同向非零、0 个反向；到 U30，30 个非零 union 中只有 6 个同向，并出现 1 个反向，说明同一个 Skill 的整体正方向可以复现，但具体受益 state 仍随 RL seed 改变。

### 6.4 不能外推的部分

1. 这里只有两个完成 seed；层级 bootstrap 能防止伪独立，却不能稳定估计 seed population variance。Seed303 完成 U20/U30 后必须用完全相同的固定 anchors 和协议补入。
2. 当前只覆盖 Clean task type。结果不能直接外推到 pick-and-place、heat/cool、look 或 multi-object。
3. 18 个候选只有 4 个达到自然调用支持，说明跨 Skill 结论仍受 Router coverage 限制；unsupported 不是“效用为零”。
4. phase 分层中，`cle_003` late 只有 3 games、`gen_002` late 只有 1 game，不能用其宽区间或零结果作统计结论。
5. 固定 B0 支持集回答的是同一 state support 上的 policy–Skill 兼容性漂移，不等同于各 checkpoint 自己 on-policy state distribution 下的平均效用。
6. 每个 Skill 目前只有一个 token-matched PLACEBO；虽已隔离明显的长度/模板效应，尚未估计不同无关文本带来的 placebo 方差。
7. reward 是稀疏 terminal success，且本报告对多个 Skill/update/phase 给出未做 multiplicity correction 的区间；phase 与 anchor 转移分析应视为机制分析。

### 6.5 最终判断

与 2026-08-29 的单 Skill、单 seed、step-0 pilot 相比，本轮已经证明：

- 效用漂移可发生在真实非空 prefix 的中途状态；
- 某些 Skill 的正向漂移能跨 RL seed 和累计 update 复现；
- 另一些明显负向漂移高度依赖 RL seed 和训练阶段；
- action flip 很频繁，但只有约一半 first-action flip 会改变 terminal reward，必须继续用累计 reward 定好坏；
- 全局 episode performance 与单 Skill utility 可以解耦。

因此当前最准确的结论不是“RL 普遍让旧 Skill 变坏”，而是：**RL 会系统性改变固定 state 上 policy 对 Skill 语义的利用方式；变化具有 Skill、state、update 和 RL-seed 异质性。** 已观察到一个强烈但未跨 seed 复现、也未满足严格 baseline-positive 条件的 harmful candidate；尚未观察到预注册意义上的 confirmed harmful sign flip。

## 7. 归档索引

- 预注册协议：`SkillRL/phase1/config/qwen35_clean_all_skill_utility_protocol.json`
- B0 来源 rollout：`SkillRL/artifacts/evaluations/qwen35-b0-clean-valid-unseen-anchor-v1/`
- coverage 与固定 anchors：`SkillRL/artifacts/anchors/qwen35-clean-all-skill-seeds101-202-v1/`
- token-matched PLACEBO：`SkillRL/artifacts/controls/qwen35-clean-all-skill-seeds101-202-v1/`
- 三臂 index、完整 suffix trajectories 与日志：`SkillRL/artifacts/evaluations/qwen35-clean-all-skill-seeds101-202-v1/`
- 指标、bootstrap 与完整性审计：`SkillRL/artifacts/metrics/qwen35-clean-all-skill-seeds101-202-v1/`
- 完成 manifest：`SkillRL/artifacts/manifests/qwen35-clean-all-skill-seeds101-202-v1.json`
- Seed303 暂停清单：`SkillRL/artifacts/manifests/qwen35-clean-formal-seed303-u30-8gpu-s1-20260902-paused-update12.json`
