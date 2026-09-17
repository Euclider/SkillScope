# Phase2：Reward-directed interaction 与 Skill 边际效用变化方向验证

日期：2026-09-08。

2026-09-09 执行修订：用户已确认 **O−PLACEBO语义效用为主分析**，并要求赶在9月11日上午汇报前优先产出。首批采用6个global updates，具体参数、独立重复和暂缓项目见 `2026-09-09-phase2-semantic-direction-fast-execution.md` 及 `SkillRL/phase2/config/semantic_direction_fast_v1.json`。以下保留9月8日草案供追溯，主对照及首批规模以执行修订为准。

状态：根据现有代码、checkpoint 和 Phase1 结果整理的实验方案草案；尚未启动 Phase2 RL，也未产生 Phase2 指标或 gold 结果。本文中的待确认定义应在正式采集前落实为版本化协议。

设计依据：`2026-08-19-policy-update-skill-sign-flip-forecasting-design.md` 第 4–10 节；研究目标采用后续讨论确认的“预测边际效用变化”，不再要求必须发生正→负 harmful flip。模型和环境沿用已经验证的 Qwen3.5-4B / ALFWorld Clean。

## 1. 本轮要回答的问题

Phase1 已支持：RL 会改变固定 `(state, Skill)` 的边际效用，且无符号 `S_int` 对“效用是否发生变化”有跨训练路径的增量预测信息。

Phase2 检验两个尚未验证的假设：

1. `P_int` 是否与独立评测的有符号 `ΔM` 存在可迁移到后续 updates 的正向关系；
2. `D_t` 是否能在旧效用、一般 policy shift、无符号 interaction magnitude 和训练 reward 信息之外，提高对 `ΔM<0` 的风险排序。

`P_int` 保留 reward alignment 的符号，`D_t` 只聚合 reward-opposing 分量。二者是预测特征，不是效用方向的定义。高 `D_t` 不保证 `ΔM<0`，`D_t=0` 也不证明正向变化；严格 harmful flip 继续作为次要事件审计。

本轮先做一个训练分支上的前瞻验证。历史三个 seed 的权重、轨迹、指标及报告保持原样。新实验的单 seed 结果不能单独证明方向预测跨 RL seed 泛化。

## 2. 推荐起点：Seed303 U30

### 2.1 选择依据与选择偏差

Seed303 同时具备明显效用漂移和完整续训状态：

- `cle_003` 相对 B0 的 `ΔU_sem` 在 U10/U20/U30 分别为 −3.33 / +40.00 / +23.33 pp，存在上升与回落。
- Phase1 中，`S_int=1/0` 对应的效用变化率为 38.33% / 8.55%，差值 +29.78 pp；负向变化率为 16.55% / 3.43%。
- U30 的 Clean train success 为 40.62%，valid_seen success 为 66.67%；有待提升空间，也需要同时监测 invalid actions 和 policy 整体退化。
- Seed101/202 当前仅保留 model-only 分析权重；Seed303 仍保留 U29/U30 的完整 model、optimizer、scheduler 和 rank RNG 文件。

这不是声称 Seed303 在所有指标上“最好”。它是基于历史 Phase1 数据及恢复条件选择的开发起点。因此，新实验解释为“在选定起点上的前瞻可预测性验证”，不用于无偏估计任意 seed 的效应大小或事件率。

父 checkpoint：

```text
/home/wangyifan/skill-RL/SkillRL/artifacts/checkpoints/qwen35-clean-formal-seed303-u30-resume12-s2-20260903/global_step_30
```

训练初始化必须恢复上述完整 checkpoint；KL reference 继续采用原始公共 B0，不能随着 actor 起点一起改成 U30。

### 2.2 已核实的恢复细节

2026-09-08 读取 checkpoint 得到：

| 项目 | 实测 |
|---|---:|
| U29 Adam 累计 step | 472 |
| U30 Adam 累计 step | 495 |
| U29→U30 Adam 更新次数 | 23 |
| U30 scheduler `last_epoch` | 30 |
| U30 learning rate | 1e-6 |
| model shards 总大小 | 19,367,672,664 bytes |
| optimizer shards 总大小 | 33,646,031,896 bytes |
| 旧 BF16 model-only 大小 | 9,682,950,504 bytes |

当前训练 actor 默认持有 FP32 参数，forward 使用 BF16 mixed precision。现有 `scripts/model_merger.py` 会把导出权重转换为 BF16。Phase2 的参数差分必须保留训练原始精度，不能只保存旧式 BF16 导出。

当前 checkpoint 有 actor rank RNG 和 dataloader 状态，但现有 ALFWorld worker 接口没有环境 sampler 的 save/restore。恢复模型和 optimizer 不等于恢复未中断运行的全部环境随机序列。实施时应验证能否重建 sampler 游标；不能时，在新分支中显式登记新的 continuation sampler seed（建议固定为 30331），并从此完整存档 game schedule / RNG，不声称逐 bit 延续旧 sampler。

## 3. 训练时间轴与最小规模

### 3.1 update 的定义

沿用现有训练语义：一次 **global update** 为“旧 policy 生成 8 games × 4 rollouts，完成该 batch 的 actor optimization”。一个 global update 会把轨迹拆成多个 decision rows，并执行多次 Adam step。

主分析以完整 global update 为 `t→t+1`，因此是真实的一轮 RL 更新前后精确端点比较；不写成“只进行一次 Adam.step”。每轮额外存档 optimizer step 的起止编号、minibatch 顺序、实际 loss 权重、LR、clip/KL 与梯度统计。若要将 `t` 严格改为单次 Adam step，必须额外保存内部端点；那属于另一种测量粒度，不能通过改 batch/learning rate 悄悄替换。

### 3.2 建议固定 U30→U42

| 阶段 | transition（按 post-update 编号） | 用途 |
|---|---|---|
| 工程闭环 | U30→U31 | 对齐、数值、归档、恢复测试；通过且测量实现未变时计入开发段 |
| 开发段 | U31 / U32 / U33 / U34 | 4 次相邻转移；只用 development games 拟合、标准化或定阈值 |
| 边界隔离 | U34→U35 | 完整采集，但不参与主 predictor 拟合或测试指标 |
| 时间外测试 | U36 / U37 / U38 / U39 / U40 / U41 / U42 | 7 次相邻转移；第一个测试为 U35→U36 |

边界隔离避免开发标签和测试标签直接共享 U34 的效用测量误差。所有 12 次转移都保留记录，不能看到 `D_t` 或 gold 结果后挑出“最明显”的几轮。

新增训练为 12 × 32 = 384 条 rollout，保存 U30–U42 的完整精度 policy 端点。1–3 次 update 适合验证测量闭环；仅凭这些 update 不足以评估稳定的跨 update 预测。12 次仍是小样本探索性方案，不能预先保证显著性。

训练 LR 保持 1e-6，KL coefficient 0.01，8 卡 FSDP、HF rollout、每轮 8×4 条轨迹、max prompt 2048、max response 64、episode horizon 30。Skill Bank 和 Router 冻结，Clean 候选集合仍为 12 general + 6 task-specific Skills。

不以提高 LR 放大信号作为默认步骤。预先增加 U30→U33、U33→U36、U36→U39、U39→U42 四个不重叠的 3-global-update endpoint sensitivity windows；只能称为 window signal。`D_window` 不能由逐 update 的 `D_t` 简单求和，也不能把混合旧 policy 的 advantage 伪装成同一个 `d_t`。

如单 update 的非零 advantage、Skill 支持或效用方向事件过少，先报告 unsupported / inconclusive。是否增加新的训练段另作带日期的协议修订，保留当前预定段的完整结果，不以“跑到显著”为停止条件。

## 4. 三臂对照与要预测的效用

继续使用自然首次调用干预。目标为 Skill A 时，原始 Router descriptor bank 在所有条件下不变；只在实际选到 A 时替换 policy payload。

| 条件 | 首次及后续再次选到 A 时的 payload |
|---|---|
| O / ORIGINAL | A 原内容 |
| P / PLACEBO | token/template matched 的无关内容 |
| N / TARGET_NULL | 空 payload |

前缀完全重放，首次调用位置的 state、history、admissible actions 一致；分支后自由行动和路由。选到其他 Skill 时保留其原内容。

分别定义：

$$
M_t^N(s,c)=E[G_t^O-G_t^N],\qquad
M_t^P(s,c)=E[G_t^O-G_t^P],
$$

$$
\Delta M_t^q=M_{t+1}^q-M_t^q,\qquad q\in\{N,P\}.
$$

建议把 idea 的 `∅` 具体化为“当前目标 Skill 的空 payload、其余 scaffold 固定”，主检验 `P_int^N/D_t^N → ΔM^N`。为承接 Phase1 的语义效用，同时计算显式命名的 `P_int^P/D_t^P → ΔM^P`，以及 nuisance `G^P-G^N`。

这是将单 Skill 公式应用到步骤级多 Skill bank 时必须说明的 estimand。TARGET_NULL 不等于整个 suffix 永远关闭所有 Skills；后者测的是整套路由 Skill scaffold 的差异，不能直接当作单个 A 的边际效用。若要求独立测量“全程 Skill-free policy”，应另加 GLOBAL_NO_SKILL 臂，作为全局解释量。

每种对照都报告 `ΔW`、`ΔB^q`、`ΔM^q=ΔW−ΔB^q`。不能用 O−N 的 signal、O−P 的 label，却省略这层区别。

## 5. 按 idea 计算各类指标

### 5.1 共同支持、teacher forcing 与精度

采用 design 明确允许的 **token support**：完整词表，大小 248,320。环境 decision 为 `j`，其生成的有效 response token 为 `k`，原始 token 为 `a_{j,k}`。训练观测以 `(j,k)` 为原子单位，不能把一个多 token ALFWorld action 当成一个真实 categorical action。

新旧 policy × O/N/P 都在同一旧 policy 生成的 response prefix `a_{j,<k}` 上 teacher-force，只替换 Skill payload；独立采样的新 action 不能替代这个对齐。训练采用的 temperature、position IDs、attention/loss masks、tokenizer、chat template 和截断方式全部固定并归档。

按训练实际有效 token mask 计算，包含实际参与 loss 的 EOS 等 token，排除 padding / 已结束 episode 的补齐行。另存 action 文本对应的 token span，可做预定 action-span sensitivity，不因观测方向筛选 token。

旧 policy 的完整词表 log-probability 用 FP32 `log_softmax` 记录；各 branch 采用相同 forward 精度规范。chosen-token log-probability 应与 trainer 实际使用的 `old_log_probs` 逐项核对。原始精度 checkpoint 与数值重算是可复现的基础，不能用新 policy 的 argmax 代替旧动作。

### 5.2 Functional shift：design §5.1–5.3

对 branch `q∈{O,N,P}`，完整词表向量为：

$$
u_t^q(z_j,k)=\log\pi_{\theta_{t+1}}^q(\cdot\mid z_j,a_{j,<k})-
\log\pi_{\theta_t}^q(\cdot\mid z_j,a_{j,<k}).
$$

$$
\delta_t^{s,N}=u_t^O-u_t^N,\qquad
\delta_t^{s,P}=u_t^O-u_t^P.
$$

保存 norm、完整支持的 forward KL / JS、旧动作 log-probability shift、O/N/P 首动作及相邻 update 的 `S_int`。后几项是基线或解释量，不替代完整向量 `δ`。

### 5.3 Reward-directed direction：design §6.1

$$
d_{t,j,k}=\hat A_{t,j,k}\,[e_{a_{j,k}}-\pi_{\theta_t}^O(\cdot\mid z_j,a_{j,<k})].
$$

`A` 必须来自本轮真实传入 actor loss 的 advantage tensor。当前实现将 episode outcome 分配到 decision rows，还包含 invalid-action penalty，并默认跨同组 decision rows 计算 GRPO mean/std；不能离线改成“只对 4 条 trajectory 标准化”的另一个 advantage。

这属于 **outcome-consistent local direction**，不是逐动作正确性的标注。记录 terminal success、invalid penalty、最终 token reward、group normalization、advantage 的各层来源。另存 clip、reference KL、Adam 状态和 minibatch 顺序，说明实际更新还受到这些因素影响；`d` 不是完整 Adam 参数更新的精确梯度。

固定 `W=I` 作为默认内积；这是对 design 未指定 `W` 的显式实例化。`A=0` 时 `d=0`，`C_upd` 的方向不可识别；记 `direction_valid=false`，令其门控贡献为零。零 advantage 比例和非零方向支持单独报告；整个 `(s,c)` 无足够非零支持时输出 unsupported，不因贡献为零就解释为低风险。

### 5.4 Fidelity、signed projection 和 D：design §6.1–6.2

$$
C_{t,j,k}^{upd}=\frac{\langle d_{t,j,k},u_t^O\rangle_W}
{\|d_{t,j,k}\|_W\|u_t^O\|_W},
$$

$$
P_{t,j,k}^{int,q}=\frac{\langle d_{t,j,k},\delta_t^{s,q}\rangle_W}
{\|d_{t,j,k}\|_W+\epsilon},\qquad q\in\{N,P\}.
$$

令 `J_t(s,c)` 为该 `(s,c)` 中所有实际参与 loss、非重复的 token 集合，定义：

$$
D_t^q(s,c)=\frac{1}{|J_t(s,c)|}
\sum_{(j,k)\in J_t(s,c)}
I(C_{t,j,k}^{upd}\ge\tau_C)
I(\|\delta_t^{s,q}\|_W\ge\tau_\delta)
[-P_{t,j,k}^{int,q}]_+.
$$

主聚合是 token 等权，符合所选 token 原子支持。另报告先按 environment decision 的 token 均值、再按 decision 等权的敏感性结果，检查输出长度权重的影响。

分母包含零 advantage 和未通过 fidelity/magnitude gate 的有效 token；这些 token 的被门控贡献为零。不能只对非零 advantage 或通过 gate 的 token 重新取均值，那会得到条件化后的不同指标。非零 advantage fraction、gate pass rate 和有效样本数同时报告；不足支持时主 D 标为 unavailable。

阈值建议：`epsilon=1e-12`，`tau_C=0`；`tau_delta` 在读取 Phase2 gold 前，由同一 checkpoint 重复 forward 的数值噪声校准，固定为 `max(1e-8, 10 × noise_norm 的第95百分位)`。完整记录校准分布，不能用效用标签调出最佳 gate。

同时输出 `P_int` 的有符号均值、正负比例、未门控 D、门控 D 及 risk–coverage 曲线。direction risk 的预期为 `P_int` 与 `ΔM` 正相关、`D_t` 与负向变化风险正相关；检验可以不成立。

### 5.5 公式需要注意的坐标问题

design 的 `d` 在 logit space 定义，`u` 是 log-probability difference。采用 `W=I` 时，因为 `sum(d)=0`，`d` 与 `u` 的点积不受 log-softmax 归一化引入的常数偏移影响；但 `||u||` 和 `C_upd` 的分母会受影响。

因此主结果保留 design 原式，同时预先增加 vocab-centered `u/δ` 的 fidelity/norm sensitivity，单独命名，不能静默把主公式换掉。若两种坐标使 fidelity 覆盖率或结论明显相反，应回到指标定义讨论，而不是挑更显著的版本。

### 5.6 Parameter 与 activation：design §5.4–5.5

| 信号组 | 计算和归档 |
|---|---|
| Raw parameter update | FP32 master 参数的全局/逐层 L2、relative L2；确认 tied weights 不重复计数 |
| State–Skill parameter projection | `(J_t^O−J_t^q)Δθ_t`，`J` 为固定 token 支持上 log-probability 对参数的 Jacobian |
| 线性化误差 | `‖δ−δ_JVP‖/(‖δ‖+epsilon)`，在相同 precision/reference function 下比较 |
| Activation interaction | 层 8/16/24/32 的 `(h_new^O−h_old^O)−(h_new^q−h_old^q)`；按 teacher-forced response prefix 的最后一个有效位置池化 |
| Training statistics | LR、KL、loss、gradient norm、Adam step、return、advantage、invalid-action rate、group success 和 Skill 使用频率 |

Parameter/activation 的 probe bank 从 U30 的 development 数据自然调用状态中冻结，不使用 post-update gold 选择层、state 或 token。JVP 先在预定 32 个自然 probe decisions 上实现可复现计算，清楚报告其子集覆盖；不能把 sampled-token 的一维导数称为完整 action-distribution JVP。

Qwen3.5 含混合注意力，当前代码尚没有这套 JVP 路径。工程 gate 必须验证自动微分支持以及 padding/teacher-force 对齐。BF16 cast 也使严格数值函数不光滑：JVP 的数学核验采用 FP32 smooth reference function，并同时保存该 reference 的 endpoint `δ`、部署 BF16 endpoint `δ` 和两者差异。不能把不同精度函数的差直接称为一阶近似误差。

有限差分只用于检查 JVP；若必须用有限差分替代，须明确登记近似与误差。JVP 若不可计算，报告 blocked/unsupported 并讨论实现，不以参数范数冒充。`P_int/D_t` 的完整端点定义不依赖 JVP 是否通过。

## 6. 对齐数据与归档规范

新增不可变 run root，建议 `SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to42-direction-v1/`。父 Phase1 目录只读引用。

每条记录使用显式联合键：

```text
run_id / global_update / optimizer_step / group_id / trajectory_id /
game_id / environment_step / selected_skill_id / response_token_offset
```

必须在 rollout 创建时赋 `decision_id`，在 flatten、padding、batch balance、FSDP dispatch、minibatch slicing 后原样携带。仅凭重复 prompt 或 trajectory_id+row index 事后猜测 step 不合格。

| 归档 | 内容 |
|---|---|
| `protocol/` | 配置、代码 fingerprint、模型/tokenizer/skill/router hash、sampling schedule、阈值、split、环境版本 |
| `checkpoints/` | 每轮更新前后完整 policy 参数；原始精度不可损失；恢复所需 optimizer/scheduler/RNG/data 状态另保存 |
| `batches/uXXXX/` | pre-optimizer DataProto 的 token tensors、mask、IDs、reward、advantage、old/ref log-probs、所有重排与复制来源 |
| `logprobs/uXXXX/` | 旧 ORIGINAL 的全词表有效 response-token log-probs；新旧 O/N/P chosen-token log-probs；完整 probe 向量证据与数值规范 |
| `signals/uXXXX/` | token/step/Skill-context 级 `u/δ/d/C/P/D` 的精确可重算统计、gate/support、parameter/activation/JVP 结果 |
| `anchors/` | 固定 prefix、首次调用位置、state/history/admissible actions、game/source seed/phase、Router 决策和 payload |
| `evaluations/` | 每条三臂 suffix 的完整动作、观测、reward、再次调用、Skill 种类、invalid/loop、token 和时间开销 |
| `predictions/` | gold 打开前写出的 feature 表、冻结模型、预测排序、时间戳和 hash |
| `metrics/` | paired utility、CI、方向标签、预测误差、消融、覆盖率、成本、完整性审计 |

六种全词表 branch 张量逐 chunk 计算，不能在 GPU 同时堆积所有 logits。旧 O 完整 log-probs 永久留存；其余完整分布可在精度一致的 policy checkpoint 和输入上重算，并永久保留一组预定 witness tokens 的完整六分布。压缩 chosen log-prob 或 top-k 不是计算完整 `d/δ/P/D` 的替代。

保存后先验证齐全、hash、参数 dtype、chosen-token log-prob parity，再标记 committed。确保原始 FP32 policy 已无损归档后，才允许在 **Phase2 新目录内部**滚动保留最近两份完整 optimizer 恢复点；Phase1 父 checkpoint 不参与清理。

每轮必须在运行下一轮训练、或读取该轮 post-update gold 之前，写出并锁定 features 和已可用的预测结果。研究阶段构造 gold 的成本单独计账；不能把完整 post-update return 混入所谓无需新 rollout 的特征。

## 7. Signal support、development 与 gold

### 7.1 三类数据各自的用途

- **训练 signal 数据**：Clean train 650 games；每轮实际 update batch 用于 `A/d/P/D`。训练独立于 gold。
- **Development / fixed functional probes**：Clean `valid_train` 的 37 games，先审计与所有实际训练 game 的不重合；用于 U30 自然路由 probe bank、校准和早期 update 的开发标签。
- **Gold**：保留原 B0 的 200 个 fixed anchors、4 Skills、31 个 valid_unseen games；用新 continuation seeds 重新评估 U30–U42。

原 valid_seen 27 games 的全局 monitor 只作解释，不选 checkpoint。Phase1 已分析过 valid_unseen 这些 games，因此本轮是未来 update labels 的时间外验证，不声称这些 game 对研究者从未可见。

开发 predictor 只读开发段 + development games；最终报告读取测试段 + valid_unseen。主时间外评测不把同一个 update 的 anchors 随机拆进 train/test。

### 7.2 分析单位和支持阈值

主分析为 `(global update, skill, clean)`；phase 分层为预定的 initial=0、early=1–4、middle=5–14、late≥15。phase 是旧 state 的属性，不由新结果聚类。训练中的同 Skill、同 phase 支持聚合后预测 gold 的同组效用；不能把训练 state 的 advantage 直接贴到一个 held-out anchor 上。

建议 reward-directed 可用支持为至少 20 个非重复、非零 advantage 的 **environment decisions**、覆盖至少 4 个训练 games 和 8 条 trajectories。token 数不能冒充 step/game 支持。预测前能检查的支持条件才允许作为主 eligible mask；零 advantage 或无支持时输出 abstain。

Gold 保留现有 4 个 supported Skills 全覆盖，包括中途 anchors。新增自然可路由 Skill 若满足 Phase1 的 ≥30 occurrences、≥10 games，只进入单独登记的扩展 cohort。未达到阈值的 Skill 不强制调用。

Phase 分层的精度单独报告；建议至少 15 anchors、10 games 才报告独立方向判断，低支持 late 等仍保留轨迹和描述性统计。主 Skill-level 均值不能只保留表现显著的 phase。

每个 update 只有 4 个主 Skill-level 信号单元；phase 单元是相关的补充。200 anchors、重复 seeds 或 token 行增加测量精度，不能增加独立 RL update 数。

### 7.3 重新估计效用和避免旧 margin 的机械相关

Gold 每个 anchor × arm 使用 4 个新的 continuation seeds，建议固定为 41011、41012、42011、42012。前两个仅估计部署时可用的 old margin / SE；后两个用于成对 old/new gold return difference。相同用途内，old/new/各 arm 采用共同随机种子；两个用途之间保持独立。

如此避免把同一个带噪声的 `M_t` 同时当特征和 `M_{t+1}−M_t` 的负项，制造 regression-to-the-mean 关联。每个 endpoint 的 O/N/P return 可在相邻转移中缓存复用；其统计依赖仍需保留。

Gold 保持 temperature=0.4、top_p=1、max_new_tokens=64、总 horizon=30、history_length=2。训练信号仍按实际训练 temperature 计算，另外报告匹配 gold temperature 的 functional sensitivity，不能混用概率。

## 8. 方向标签、预测比较与判据

### 8.1 标签

主连续目标为 game-equal `ΔM^N`；`ΔM^P` 为承接 Phase1 的同步报告目标。统一转成 success-rate 单位，同时保存原始 0/10 return。

方向实用阈值建议提前固定为 `tau_M=0.05`（5 pp）。按 paired game bootstrap 的 95% CI 给出：

- positive：`LCB(ΔM)>+tau_M`；
- negative：`UCB(ΔM)<−tau_M`；
- stable：整个 CI 位于 `[−tau_M,+tau_M]`；
- uncertain：其余。

连续目标保留全部有 signal 支持的组，不因 CI 跨 0 删除。二元点标签 `I(ΔM<−tau_M)` 用于负向排序，同时报告 label uncertainty；只对可靠方向 subset 的 balanced accuracy 必须连同覆盖率报告。不得把 uncertain 当成 stable 或正例。

### 8.2 预定比较

1. Direct score：`P_int` 对 signed `ΔM`；`D_t` 对负向变化；`||δ||` 对变化幅度；逐 update 和 Skill 内均报告，避免只看 pooled correlation。
2. 基础预测器：旧 `M_t/SE` + 一般 policy shift + update magnitude + 训练 reward/advantage 摘要。
3. 等容量增量：基础输入加入无符号 `||δ||/S_int`，再加入 `C_upd/P_int/D_t`；parameter/JVP/activation 在相同 split、正则化和成本口径下比较。
4. 用固定正则化的 ridge 回归预测 signed `ΔM`；logistic 预测负向点标签。标准化只在开发段拟合。开发样本少时以 direct-score 为主，不增加非线性模型或标签驱动的特征搜索。
5. 增加 Skill-ID 基线/Skill 固定效应及逐 Skill 时间变化分析，防止只学到“cle_003 总体比其他 Skill 高”。同 update 的样本不能随机分折。

报告 signed MAE/RMSE、Spearman、negative AUPRC、Brier、Precision@K（固定 audit budget）和 coverage/cost。Pearson、各层 activation 最好值或不同阈值最好值不得事后选成唯一主结果。

统计上 games 跨 update 重复，相邻 `ΔM` 共享 endpoint。bootstrap 要保留这一结构：game ID 在所有 update 一致重采样；update 采用预先固定长度 2 的连续 block 抽样，并报告长度 3 敏感性。不能把 `(update, game, anchor)` 独立重采样，也不计算只有一个 seed 却声称反映 seed 方差的 CI。7 个测试 update 的区间仍需谨慎解释。

本轮覆盖 design §5–6 的核心数学指标。§7/§10 的完整 LLM failure-summary、budget-matched 新轨迹和多种短程 behavioral probe 成本曲线属于后续扩大比较；本轮不能据此声称已经优于这些尚未运行的基线。

### 8.3 何种结果构成最小支持

- 数据端：所有用于主分析的 token–advantage–checkpoint 对齐审计通过；支持不足的组可识别。
- 信号端：在后续测试 updates 中，signed `P_int` 与 `ΔM` 的关系符合预定方向，并优于仅含无符号信号的对照；或 `D_t` 稳定改善 negative-risk 排序。
- 增量端：改善在控制旧 margin、一般 shift、训练 reward 和 Skill identity 后仍存在，并且不是由单个 update 独占。
- 证据强度：报告跨 update 的变化、区间和删除单个 update 的敏感性。若区间跨 0，结论为方向信息尚不确定，不把点估计改善写成可靠预测。
- 数据充分性：若可靠方向只出现于少于 3 个测试 updates，或几乎全为同一方向，方向分类只作为描述性结果；保持完整连续结果，不通过补采直到显著来满足条件。

方向预测不成立时仍能保留 Phase1 的无符号变化风险结论。若 Phase2 成立，再以完全冻结协议在其他父 seed 分支复验；当前选择 Seed303 不能替代这一步。

## 9. design §6.3 的负对照与实现顺序

原 design 包含真实更新负对照；不是把已算出的 D 随机打乱就完成：

| 对照 | 实施要求 |
|---|---|
| 同一旧 θ、不同 batch/rollout seed | 从 U30 同一 optimizer 状态生成独立 batch，做一轮 side-branch update |
| shuffled reward/advantage update | 同一 U30、同一训练 batch；在 game group 内打乱轨迹 outcome 分配，保留映射，再按原代码重算 A 并真实更新 |
| 等范数随机/正交参数方向 | 预定 U30→U31 的 Δθ 范数，生成独立参数方向；不覆盖主分支，报告全局/逐层范数匹配方式 |
| 无关 Skill payload | 保留 O/P/N 对照，另将错误 Skill 明确标为 payload 负对照，不改 Router 选择 |
| 无 gate 与有 gate | 主 D 及 ungated D、覆盖率同时输出 |

独立 batch 和 shuffled-reward 分支需要自己的 post-update gold；这些诊断分支不混入主 U30→U42 的时间外预测样本。打乱训练 reward 后，分别保留相对实际打乱 A 的信号和相对原始 A 的诊断信号，不能混称同一个 reward direction。

最先完成测量代码与 U30 无更新重复 forward，然后完成 U30→U31 对齐闭环；确认 full-vocabulary、原始精度和恢复流程后继续主时间轴。Side branches 在 U30 起点和 batch 已归档后独立执行，优先保证主链可恢复。

JVP/kernel 或坐标定义有实质不一致时，应先讨论；不更换成近似信号后沿用原名称。

## 10. 资源、时长与存储预算

2026-09-08 只读检查：8 × A800 80GB 均空闲；文件系统约剩 1.3TB。因此不需要清理现有三个 seed 的结果才能准备 Phase2。

历史 Seed303 U21–U30 单轮总耗时约 0.52–1.17 h；不能继续沿用“小量新增 probe 几小时”的旧估计来覆盖本轮完整方案。

Gold 主集规模：13 endpoints × 200 anchors × 3 arms × 4 seeds = **31,200 条 suffix**。开发集若按最多 100 anchors、5 endpoints、3 arms、4 seeds，另有最多 6,000 条；side-branch gold 另计。

按此前约 1,800 条 / 72 min 的记录粗估，主 gold 约 21 h，开发 gold 约 4 h；这是历史吞吐外推，实施 gate 后按实际 token/轨迹长度修正。12 次训练约 7–14 h；完整分布、activation、JVP 和对照另有显著开销。完整单 seed 方案建议预留 **约 2–4 天**，包含测量适配、审计和报告，不能当作承诺完成时间。单次闭环可先测出可靠吞吐再更新预算。

全词表日志不能按普通 JSON 体量估算：20,000 有效 tokens × 248,320 vocab × FP32 ≈ 18.5 GiB，仅是一种 policy/branch。六种分布全部永久保存，12 轮可能超过 1.3TB。

建议存储策略：

- 全部 12 个新 endpoint 无损保留原始精度 model（约 19GB/端点）；不能只留 BF16。
- Phase2 最近两个完整恢复点滚动保留（约 106GB），父 Phase1 恢复点不清理。
- 全部 batch/实际 advantage/chosen log-probs 永久保留，旧 O 的 full-vocab log-probs 单独分块压缩归档。
- 六分布中的反事实分布逐轮流式计算并保存精确统计与预定 witness；完整分布可由原始精度 endpoint + token inputs 重算。
- 单独预留一轮 logits 临时空间；根据实际有效 token 数预测下一轮占用。空闲低于 250GiB 或预计下一轮会侵入保留空间时，先停在已 committed 的 checkpoint。

按近期典型长度，主链新增约 0.5–0.8TB；极端长输出可能更高。实施时先实测压缩率和有效 token 数，不能以理论 30-step/64-token 上限内永不超额为假设。任何 Phase1 文件删除都不属于本方案。

## 11. 实施前需讨论并冻结的三处具体化

1. **更新粒度**：建议主 t 为一轮 global update，内部 Adam step 完整记账；若要求一次 Adam.step，就额外展开内部端点，不改变训练 minibatch 来凑定义。
2. **对照目标**：建议主 O−TARGET_NULL 按单 Skill 公式实例化，同时完整报告 O−PLACEBO 语义效用；全程关闭所有 Skills 另作不同 estimand。
3. **数值定义**：采用完整 token 词表、`W=I`、实际训练 advantage；design 的原式与 centered 坐标敏感性分别报告，FP32 JVP 与 BF16 部署行为的差异显式记录。

这些选择影响“严格按照 idea”具体意味着什么。冻结后才能把草案升级为执行协议；不能看到 Phase2 方向结果后再修改。

## 12. 来源与本次修改范围

本次仅新增本文，未训练、未修改原实验结果或训练代码。

主要本地依据：

- `2026-08-19-policy-update-skill-sign-flip-forecasting-design.md`：§4–10 的公式、支持、划分及对照。
- `2026-09-04-qwen35-clean-all-skill-three-seed-utility-results.md`：当前 Phase1 结果与方向验证边界。
- `SkillRL/verl/trainer/ppo/core_algos.py`：真实 GRPO advantage normalization。
- `SkillRL/agent_system/reward_manager/episode.py`：episode outcome 到 decision row 的 reward。
- `SkillRL/verl/workers/actor/dp_actor.py`：temperature、token loss、minibatch 与 Adam 更新。
- `SkillRL/verl/workers/fsdp_workers.py`：FP32 actor、BF16 mixed precision、scheduler。
- `SkillRL/agent_system/multi_turn_rollout/rollout_loop.py`：step 展平、batch 组装。
- `SkillRL/phase1/archive.py`：旧归档只保留 advantage 的 sum/mean/min/max，缺少完整对齐 tensor。
- `SkillRL/phase1/first_invocation.py`：TARGET_NULL/PLACEBO 的持续 payload 干预。
- `SkillRL/scripts/model_merger.py`：旧导出会转换为 BF16。

只读检查时的源文档 SHA256：

```text
idea:    73927fcb500620cc41497d50c7a49c0de90ffee3d10854f41713d60f69e91886
phase1:  e080b2b4dcd58bc44bd7d4e97cbd3a044abee372791bc818e9cce5b5c09d23a7
```
