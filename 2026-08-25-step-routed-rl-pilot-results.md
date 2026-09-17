# SkillRL 逐状态 Skill 路由：真实 RL Pilot 结果

> 日期：2026-08-25  
> Run ID：`phase1-step-router-pilot-seed-101-r2`  
> 状态：5/5 个真实 GRPO update 已完成，checkpoint、轨迹和统计均已归档

## 1. 结论

本轮 pilot 已验证真实 RL 训练链路和逐状态 Skill 路由归档链路可用：Qwen2.5-1.5B-Instruct
完成了 5 次非零 policy update；200 条 train/validation 轨迹全部按步骤保存；每步选择并注入
一个 Skill 的路由不变量没有出现违例；199/200 条轨迹在同一 game 内实际调用了两个或更多
不同 Skill。因此当前实现没有退化成“一个 game 始终使用同一个 Skill bundle”。

这还不是研究假设的最终结论。逐步 validation 只有 8 条随机采样轨迹，成功率不单调；本轮
只能说明可以进入固定状态 action probe 和 matched `full_bank / minus_skill / no_skill`
效用评估，不能据此声称 RL 已提高 ALFWorld 成功率或已经观察到 Skill 效用偏移。

## 2. 实验设置

| 项目 | 设置 |
|---|---|
| 模型 | `/home/wangyifan/model/Qwen2.5-1.5B-Instruct` |
| 环境 / 算法 | ALFWorld `AlfredTWEnv` / GRPO |
| RL seed | 101 |
| train / validation games | 8 / 8 |
| train rollout | 每个 game 4 条，即每个 update 32 条 |
| update 数 | 5 |
| 最大环境步数 | 30 |
| learning rate | `1e-6` |
| PPO mini-batch / micro-batch | 32 / 每 GPU 4 |
| KL loss | 开启，coefficient `0.01`，`low_var_kl` |
| invalid action penalty | 开启，coefficient `0.1` |
| validation sampling | temperature `0.4`，每个 update 8 条 |
| GPU | 2 张 A800 80GB（物理 GPU 1、2） |
| Skill Bank / Router | 全程冻结；动态 Skill 更新关闭 |
| 每步 Skill | task-filtered 候选为 17 或 18 个；Router 选 1 个注入 prompt |
| checkpoint / validation | 每个 update 各执行一次 |

运行 manifest 固定了 repo commit、dirty worktree fingerprint、模型配置 hash、protocol hash、
Skill Bank hash、Conda/Pip 和 GPU 信息。Skill Bank hash 为
`e8a953beac1809591fadf0d3509db5dea6e66b0fb56ddb573cf30e6d8879e909`。

## 3. 每步训练与验证统计

| Update | Train success | Validation success | Train mean reward | Valid action ratio | Grad norm | PG loss | KL loss |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3/32（9.38%） | 0/8（0%） | 0.9375 | 72.03% | 4.4974 | 0.00727 | 0.03697 |
| 2 | 1/32（3.12%） | 0/8（0%） | 0.3125 | 97.19% | 4.0303 | 0.03227 | 0.01755 |
| 3 | 6/32（18.75%） | 1/8（12.5%） | 1.8750 | 98.98% | 3.6988 | 0.02938 | 0.00518 |
| 4 | 7/32（21.88%） | 1/8（12.5%） | 2.1875 | 99.76% | 2.3132 | -0.00335 | 0.05632 |
| 5 | 6/32（18.75%） | 0/8（0%） | 1.8750 | 99.56% | 2.0614 | 0.07551 | 0.00685 |

汇总后，训练 rollout 成功 23/160（14.38%），逐步 validation 成功 2/40（5.0%）。这些
训练成功率来自不同 update 下的 on-policy 随机轨迹；不能把 14.38% 当作最终 checkpoint
在独立测试集上的 Pass@1，也不能用它与先前论文的 agent baseline 直接对比。

## 4. 真实参数更新证据

五个 update 的 `actor/grad_norm` 均为 finite 且非零。另对原始 Hugging Face base 参数与
FSDP checkpoint 分片进行了逐参数精确比较：

| 比较 | 参数 L2 delta | Relative L2 | Max absolute delta | Finite |
|---|---:|---:|---:|---|
| base → checkpoint 1 | 0.231851 | 1.3740e-4 | 2.9007e-5 | 是 |
| base → checkpoint 5 | 0.409177 | 2.4249e-4 | 8.5197e-5 | 是 |

比较覆盖 1,777,088,000 个参数。由此可确认保存的模型不是 base copy，也不是 zero-update。

## 5. 步骤级 Skill 路由与轨迹审计

| 指标 | 结果 |
|---|---:|
| 归档轨迹 | 200（train 160，validation 40） |
| 归档步骤 / Skill 调用 | 5,676 |
| 平均轨迹长度 | 28.38 |
| 使用至少 2 个不同 Skill 的轨迹 | 199/200 |
| 每条轨迹平均不同 Skill 数 | 3.57 |
| 单条轨迹最多不同 Skill 数 | 6 |
| 平均路由切换次数 | 8.425 |
| 实际被选中的不同 Skill ID | 22 |
| 成功轨迹 | 25/200（train 与 validation 合计） |
| 路由不变量违例 | 0 |

本轮采样覆盖 `clean`、`cool`、`heat`、`look_at_obj_in_light`、`pick_and_place` 和
`pick_two` 六种 context。所有归档步骤均满足：

- selected Skill 属于当前 task-filtered candidate bundle；
- `injected_skill_ids == [selected_skill_id]`；
- Router version 恒为 `alfworld-observable-phase-router-v1`；
- policy prompt 仅注入当步选中的一个 Skill；
- Skill Bank 和 Router 在 RL 期间保持冻结。

## 6. Pilot gate

预设 pilot gate 全部通过：真实非零 policy update、`phase1.trajectory.v2` 轨迹归档、
selected/injected 一致、Router version 恒定、Skill Bank hash 恒定，以及多 Skill 轨迹存在。

因此下一阶段应在 base 和 checkpoint 1–5 上执行：

1. 固定相同 state、history、admissible actions 与 selected Skill 的 action distribution
   probe，统计 action flip、JS divergence 和 admissible-action probability 变化；
2. 对 full-bank 中确实被选择的 Skill 执行 matched `full_bank / minus_skill / no_skill`
   rollout，计算 checkpoint 间 Skill success margin；
3. 按 game 聚类 bootstrap，并单独报告 selection-conditioned eligible pairs，避免把“未被
   Router 选择”错误解释成“Skill 无效”。

## 7. 归档位置

- 完整控制台日志：`SkillRL/artifacts/logs/phase1-step-router-pilot-seed-101-r2.log`
- immutable manifest：`SkillRL/artifacts/manifests/phase1-step-router-pilot-seed-101-r2.json`
- 200 条逐轨迹 JSON：`SkillRL/artifacts/trajectories/phase1-step-router-pilot-seed-101-r2/`
- 全局幂等索引：`SkillRL/artifacts/trajectories/index.jsonl`
- 五步 update 指标与 token-level update 摘要：`SkillRL/artifacts/training_steps/phase1-step-router-pilot-seed-101-r2/`
- 五个 FSDP checkpoints：`SkillRL/artifacts/checkpoints/phase1-step-router-pilot-seed-101-r2/`
- RL 机器可读汇总：`SkillRL/artifacts/metrics/phase1-step-router-pilot-seed-101-r2/rl-summary.json`
- 路由汇总：`SkillRL/artifacts/metrics/phase1-step-router-pilot-seed-101-r2/routing-summary.json`
- base→checkpoint 参数差值：`SkillRL/artifacts/metrics/phase1-step-router-pilot-seed-101-r2/checkpoint-0-to-{1,5}-delta.json`
- 完成状态：`SkillRL/artifacts/run_status/phase1-step-router-pilot-seed-101-r2.json`

两次启动前失败也分别保留了 manifest、日志和 status：首次因误连共享的 stale Ray cluster，
第二次因 Ray UNIX socket 路径超过 107 bytes；二者均在 rollout 和 RL update 前终止，未混入
成功 run 的轨迹或指标。最终运行使用显式 local Ray 和短路径
`/home/wangyifan/ray-skillrl-101`。

运行期间 Ray 持续提示根分区使用率超过 95%；任务结束时仍有约 274 GB 可用，未发生 object
spill failure、OOM、NaN 或 worker crash。五个 FSDP checkpoint 当前合计约 91 GB。
