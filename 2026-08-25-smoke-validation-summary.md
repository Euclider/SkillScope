# Phase I：0.5B / 1.5B Skill 条件 Smoke Validation 简要总结

> 日期：2026-08-25  
> 状态：smoke validation 已完成；尚未执行任何 RL update

## 1. 本轮验证目的

本轮只验证无 RL checkpoint 下的基础能力与工程链路：

- Qwen2.5-0.5B-Instruct 和 Qwen2.5-1.5B-Instruct 能否完成 ALFWorld rollout；
- 固定 Skill Bank 的 `FULL_BANK`、`MINUS_SKILL`、`NO_SKILL` 三种条件能否正确注入和屏蔽 Skill；
- matched 条件、完整轨迹、Skill 使用情况和实验元数据能否可靠归档；
- 当前模型是否具备通过 episode success 估计 Skill 边际效用的能力基础。

本轮不是研究假设的正式验证，因为 Policy 没有发生真实 RL update，不能测量 RL 引起的 action flip、Skill margin shift 或 harmful sign flip。

## 2. 实验规模与配置

- 模型：Qwen2.5-0.5B-Instruct、Qwen2.5-1.5B-Instruct；
- 环境：ALFWorld text-based；
- context：6 类；
- game：每类 2 个，共 12 个独立 development games；
- condition：`FULL_BANK`、`MINUS_SKILL`、`NO_SKILL`；
- 每个模型共 36 episodes、1,080 environment steps；
- temperature：0.4；top-p：1.0；最大 30 步；history length：2；
- update 类型：`zero_update`。

## 3. 主要结果

| 指标 | 0.5B | 1.5B |
|---|---:|---:|
| Episodes | 36 | 36 |
| Success | 0/36 | 0/36 |
| 到达 30 步上限 | 36/36 | 36/36 |
| 标签格式 invalid-action rate | 32.87% | 30.74% |
| 标签格式遵循率 | 67.13% | 69.26% |
| 平均每步 completion tokens | 120.23 | 69.05 |

1.5B 输出更简洁，格式遵循略好，但两个模型都没有产生成功 episode，因此所有基于 success 的初始 Skill margin 均为 0，无法筛选可靠的正边际 `(Skill, context)`。

Skill 条件对格式有明显影响：

| 模型 | FULL_BANK 格式遵循率 | MINUS_SKILL | NO_SKILL |
|---|---:|---:|---:|
| 0.5B | 74.44% | 77.22% | 49.72% |
| 1.5B | 79.44% | 78.33% | 50.00% |

这里的 `invalid_action_count` 只检查 `<think>` / `<action>` 标签等输出格式，不检查动作是否真正属于当前 admissible action 集。对完整轨迹补充诊断后，真正的 admissible-action 命中率如下：

| 模型/条件 | Admissible-action 命中率 | 下一 observation 为 `Nothing happens.` |
|---|---:|---:|
| 0.5B FULL_BANK | 43.3% | 56.7% |
| 0.5B NO_SKILL | 17.8% | 82.2% |
| 1.5B FULL_BANK | 65.6% | 34.4% |
| 1.5B NO_SKILL | 48.6% | 51.4% |

1.5B 已能完成部分局部动作，但没有完成完整长程任务。例如其 FULL_BANK 轨迹中出现成功拾取、冷却、清洁和开灯动作，但没有形成最终成功的完整任务序列。

## 4. 已验证与未验证的结论

已验证：

- 模型加载、ALFWorld 交互、三种 Skill 条件和恢复执行链路可用；
- 每条 rollout 的完整轨迹、动作、observation、Skill IDs、token 数和校验信息均已归档；
- 固定状态 probe states 已成功提取；
- 1.5B 相比 0.5B 改善了简洁性、格式遵循和动作落地，但仍存在 episode-success floor。

尚未验证：

- 真实 RL update 是否会改变相同 state、相同 Skill 下的 action；
- Skill 的 episode-level 边际效用是否随 Policy update 改变；
- harmful sign flip 是否存在及其发生率；
- Policy-update signal 是否强于 zero/shuffled/random 等负对照。

因此，本轮结果只说明无 update 的小模型 smoke 无法直接验证研究目标，不能据此否定研究假设。下一步需要执行固定 Skill Bank 下的真实 RL updates，并在保存的 checkpoints 之间进行固定状态 probe 和 matched rollout evaluation。

## 5. 归档位置

- 0.5B 完整报告：`SkillRL/artifacts/metrics/smoke-0p5b-20260825/REPORT.md`
- 1.5B 完整报告：`SkillRL/artifacts/metrics/smoke-1p5b-20260825/REPORT.md`
- 模型对比：`SkillRL/artifacts/metrics/smoke-model-comparison-20260825/REPORT.md`
- 0.5B rollouts：`SkillRL/artifacts/evaluations/smoke-0p5b-20260825/`
- 1.5B rollouts：`SkillRL/artifacts/evaluations/smoke-1p5b-20260825/`
- 固定状态样本：`SkillRL/artifacts/probes/smoke-0p5b-20260825/`、`SkillRL/artifacts/probes/smoke-1p5b-20260825/`

