# Seed404 实际更新 / reward 校准读出：计算前登记 v1

Material Passport

- Origin Skill: academic-research-suite / experiment-agent（inline，无子agent或外部模型）。
- Mode: code experiment run / validate。
- Date: 2026-09-22；Version: realized_reward_secant_v1。
- Status: UNVERIFIED（公式与工程协议已登记；结果以新目录 complete.json 为准）。
- User authority: 按已讨论方案扩充并启动全量评估，检验相对原始 delta 的方向预测增益。
- Scientific status: seed404 已见标签的事后探索，不是独立前瞻预登记。

## 1. 唯一新增 reward 主公式

固定 U0 实际训练批次中的状态、动作 token、优势 A、技能和控制条件。

```text
u = log p_U5(skill) - log p_U0(skill)
delta = u - [log p_U5(control) - log p_U0(control)]
H x = x - mean_vocabulary(x)
v = H u, xi = H delta
q = A * u[a]
r_real = q * v / (||v||² + 1e-12)
D_real(skill) = -Agg[ dot(r_real, xi) ]
             = -Agg[ A*u[a]*dot(v,xi)/(||v||²+1e-12) ]
```

正分为下降风险、负分为上升方向；方向不按效用结果翻转。无 ReLU，无 C/P 门控。
所有零优势行保留在聚合分母；u为零时数值自然为零，不另做标签驱动筛选。
极小 `||v||²<=epsilon` 只记录，不能临时删行/截断/改epsilon。
q必须用**未中心化**的动作log概率变化。中心化只用于v、xi；不是技能间/样本间中心化。

q是固定优势的局部log-likelihood surrogate改变量，不是总PPO/GRPO目标、真实局部回报或逐动作credit。
该方向为rank-one/secant代理，不是精确梯度、Adam更新或JVP，不承诺解决轨迹奖励分配。
u也出现在delta中，必须量化机械相关。没有额外反向传播、训练或标签拟合。

## 2. 固定对照（3种聚合共12个新增分数）

| 名称 | token贡献 | 解释 |
|---|---|---|
| D_real | `-A*u[a]*coefficient` | 唯一新增reward主公式 |
| D_real_A1 | `-u[a]*coefficient` | 全部原行设A=+1，真正去reward；不继承A!=0掩码 |
| D_real_absA | `-abs(A)*u[a]*coefficient` | 去reward符号但保留强度；不能称完全去reward |
| D_real_projection_only | `-coefficient` | 仅更新与交互的几何投影系数 |

`coefficient=dot(Hu,Hdelta)/(||Hu||²+epsilon)`。
全词表FP64计算，16 token分块；模型仍为原BF16/SDPA执行，不更换后端或精度语义。
复用v2全部261列旧分数，包括原D、−P、centered C、原始/中心化delta范数、KL/JS、
activation、`-A*u[a]`和`-A*delta[a]`。旧比较全量复现；总计273列，不重新选旧指标。
旧几何A=1重构受A=0缺失限制，保留其原有标记；本次D_real_A1直接从向量算，无该限制。

## 3. 主比较与完整统计口径

主口径固定为 PLACEBO / all / token / 0pp；主指标：下降vs上升条件AUROC及Spearman(-ΔM)。
decision等权、token→decision→trajectory→game等权作为敏感性，不能事后换主口径。
NULL、initial/early/middle/late、5pp阈值全部输出，与原来共同池相同，不能按新分数删技能。
辅助指标：下降vs其余AP/AUROC、带符号准确率、弃权覆盖率、平衡准确率、
恒预测上升/下降及多数类准确率；top1/top2/25%/50%预算的precision/recall/下降质量覆盖。
非方向校准的幅度/去reward分数只作排名对照，不把正数当成必然下降预测。

配对差值：D_real对同聚合的A1、absA、投影系数、原D、−P、centered C、
原始/中心化delta范数、q-only与sampled-action交互。共用v2的2000组game×continuation
标签bootstrap，不按方法改变共同池；缺失任何技能时整池draw作废并报数量。
区间仅为固定readout下标签不确定性，不覆盖RL seed方差、选择偏差或历史公式搜索。

共用512组trajectory-block reward符号随机化。整条轨迹统一乘±1，保留内部优势差异和零值，
统计新D_real三种聚合的条件AUROC/Spearman/AP/AUROCrest以及三聚合max。
不是交换性已证明的检验，其超过观察值的比例不是确认性p值；max不校正历史39公式搜索。
所有聚合与符号对照保留严格常数，不让末位误差产生伪排序。

## 4. 数据边界与资源

- Scope: seed404 U0→U5，冻结SkillNet-37与逐状态top1 router均不变。
- 全部155638个token×control行（每control77819）、5511个决策；原U0 batch的128轨迹/16game。
- 效用复用已有404首调用anchors及U0/U5各3636条续跑；不增加/重跑环境rollout。
- 原效用25种自然技能；all共有18种可比技能；各阶段池完全沿用快照。
- 不运行RL、API、505/606，不修改Phase3或任何既有队列。
- 新补算所需Hu范数和Hu·Hdelta；旧标量不足以精确重建，不凭范数猜均值符号。
- U0现场OLD全词表概率保持原始归档；U0/U5在原始及PLACEBO/NULL同输入下前向。
- 8个逻辑分片，4个并行worker各2张GPU（U0/U5各1张），两波完成；最长实际动作先验收。
- 不新增全词表常驻大文件，只保存标量和每分片最多16 token核验样本。
- 原磁盘minimum free100GiB、cohort maximum760GiB及checkpoint reserve80GiB保留，另预留8GiB新结果。
- 用户已取消时间上限，不加30分钟默认超时；后台每30秒记录进程/日志/磁盘，90秒静默仅告警。
- 无自动重试。已启动的同波worker不因另一个失败而被无故杀掉；该波结束后失败即不启动后续阶段。
- 所有原125源码与旧变式源码/输入均SHA绑定，模型文件比对旧清单；OLD逐行对照原压缩账本SHA。

## 5. 验收与输出

新增代码独立成模块，不修改已封存源文件。先过CPU合成/回归，再prepare生成不可覆盖plan，
然后launch。前向逐决策精确复现旧稳定C/P/D等标量，否则报错停点保留，不自动重试。
最终273列的全部分层统计使用独立sklearn/scipy复核，5220条既有比较指标另行回归。

新运行目录：`artifacts/phase12/skillnet37-independent-s404-505-606-v4/realized-reward-s404-v2/`。
v1目录仅完成prepare、未启动任何GPU前向；启动前封存审查补充排除并发日志/临时文件的保护，
原plan及当时源码快照保留于v1/prepared-source-snapshot，另建v2启动。数学公式版本仍为v1，
候选、聚合、阈值、随机化和预算不变，不属于失败实验的重跑。
完成后自动产生：

- `plan.json`, `registry.csv`, `score-commit.json`：计算协议/公式及先评分后读标签的收据。
- `token_signals.parquet`, `skill_scores.csv`, `skill_scores_and_gold.csv`：原值和逐技能分数。
- `ranking_diagnostics.csv`, `direction-confusions.csv`, `ranking_budgets.csv`：全分层指标。
- `paired_direction_gains.csv`, `bootstrap_summary.csv`, `sign_null_*`：配对差值和负对照。
- `geometry-and-support-audit.csv`, `coverage_and_effects.csv`：频次、game、零优势、极小范数等。
- `reports/realized-reward-analysis.md`, `reports/phase2-results-expanded.md`：新分析与旧报告拼接的新副本。
- `independent-verification.json`, `provenance.json`, `complete.json`：验收与封存。

原报告不覆盖。用户目标是检验加入reward后的方向性预测增益，不是以必须优于delta为停止条件。
即使404有所改善，仍不能直接写成跨seed结论或编辑收益；需先冻结少数候选，再独立seed验证。

运行入口（仅新目录首次启动；禁止对已attempt目录重复执行）：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B \
  -m skillnet_cohort.realized_reward_run prepare --tests <passing-regression.xml>
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B \
  -m skillnet_cohort.realized_reward_run launch
```

若失败，保留failed.json/分片/日志/账本，先分析原因并获得显式续算授权，不重跑旧RL或效用。
