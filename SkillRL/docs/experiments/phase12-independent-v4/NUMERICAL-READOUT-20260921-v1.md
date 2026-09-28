## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + validate
- Origin Date: 2026-09-21
- Verification Status: UNVERIFIED（工程检查与科学结论分别验收）
- Version Label: numerical_readout_v1

# C/P/D 数值稳定性修正与原版对照

## 授权与范围

用户已确认“数值稳定性修正＋原版对照”，先按修正版重新评估404，完成后再启动505及其相关修正。
这不是重新训练、重新采样性能或效用轨迹，也不改变技能库、router、锚点、游戏、续跑随机数。
仅重做同一实际训练batch上的端点/对照前向与读出计算，然后复用既有效用标签分析。
没有费用API、模型下载、Git提交/回滚/push、旧证据删除、原报告覆盖或606启动。
新版本用于Phase1–2；此次不改Phase3 GitHub包或已冻结的Phase3运行入口。

原版 `phase2/direction.py`、汇总器和旧118份运行源码保持不变。新版本独立入口：

- `phase2/stable_direction.py`：稳定算术及逐token恒等式诊断。
- `skillnet_cohort/numerical_readout.py`：准备、八逻辑分片重算、汇总与源码/输入绑定。
- `skillnet_cohort/numerical_readout_report.py`：固定变体、同池对照与新报告。
- `skillnet_cohort/numerical_readout_run.py`：404→505修正→恢复505的显式顺序与监控。

## 数学定义与修正边界

沿用每个实际action loss token上的原定义：

\[
u=\log\pi_5^O-\log\pi_0^O,\quad
\delta=u-(\log\pi_5^B-\log\pi_0^B),\quad
d=\hat A(e_a-\pi_0^O),\quad
P=d^\top\delta/(\|d\|+\epsilon).
\]

原实现先在输入精度（此处FP32）进行exp、差分、乘积、平方，只在sum处指定FP64。
新版把读出差分与投影的前置运算也提升到FP64；模型前向仍采用既有BF16模型、SDPA及FP32 log-softmax。
不把原U0物理BF16权重或上转后的数值冒充原生FP32 master权重。

1. 使用FP64 softmax重新规范化**用于方向的OLD概率**；四份已记录的log-probability向量本身不重新规范化，
   因此u、delta的原定义及共同分量仍可审计。
2. 非动作位置保持`d_i=-A*p_i`；动作位置用`A*sum(p_i, i!=a)`，避免动作概率接近1时`1-p_a`灾难性相消。
3. 投影分子用`A*sum_{i!=a} p_i*(x_a-x_i)`，数学上等于`d^T x`，在乘法之前消去共同分量。
4. raw C与centered C使用同一个稳定分子，分母分别用原u范数与逐token词表中心化u范数。
5. P_centered独立计算并检查与P一致；这不是给各skill汇总P减均值，也不是新增预测信息。
6. epsilon=1e-12、tau_C=0、原label-free校准tau_delta（404/505均1e-8）保持。
   valid/fidelity_valid的数学定义、C分母的clamp、P分母的epsilon、token等权均值均不改。
7. 原始门控D与中心化门控D共用同一稳定P；centered gate使用centered C、centered delta范数及对应有效性。
   不把更改阈值/归一化分母/有效性定义混称为数值修复。
8. KL/JS保持原实现；activation沿用原forward及计算。没有原生训练readout的skill仍为NA，不补零。

动作多达512 token时，FP64临时向量按16 token分块，**每个token仍使用完整248,320词表**。
原版比较器仍按原完整token行形状调用，避免误把改变reduction形状引入的误差当成算法差异。

## 固定对照与判定

不按404 gold选择实现、阈值或赢家。固定报告三个版本：

| 版本 | P | C / D门控 |
|---|---|---|
| legacy_recorded | 原FP32实现 | 原版 |
| stable_raw | 稳定FP64投影 | raw C及raw范数 |
| stable_centered_gate | 与stable_raw同一个P | centered C及centered范数 |

每次重新前向还计算原版全部标量，要求与保存的原版逐值相同（NaN等价），并检查动作/状态/advantage身份。
不相同即保留证据并停止，不能放宽阈值自动继续。原PLACEBO排序报告需再次按同一公式复现。
P中心化一致性准入为`abs_error <= 1e-9 + 1e-11*abs(P)`；这是数值核验阈值，不是预测性筛选。
任何valid/gate变化均计数，不据此删除token；候选池、token数量与decision数量须与原版一致。

报告保留PLACEBO/NULL、all/各阶段、0/5pp效用事件阈值、top-k/比例预算，
AP、AUROC、Spearman、Kendall、方向一致率、gate/direction覆盖及逐skill支持明细。
原先配对game/continuation bootstrap标签直接按字节复用，不重新抽样区间。
阶段/anchor/continuation不是独立RL seeds，不据单seed或多个变体的最好分数宣布显著改善。
404已有全部目标标签，505也有旧覆盖协议的标签；数值修正是事后版本，不追溯伪装为新预登记实验。

## 运行与显存安排

暂停现场：505新口径U0索引810/3636，17进程SIGSTOP，未终止；每GPU约15217MiB。
新读出每worker分别将U0/U5放到两张GPU，四worker并行，原八逻辑分片分两波：
0/1/2/3对应GPU(0,1)/(2,3)/(4,5)/(6,7)，4/5/6/7复用同一四对GPU。
模型forward输入、计算dtype、hidden层、temperature均沿用原实现，只把输出搬回该worker的cuda:0计算投影。
每分片优先算最长实际action以尽早检验内存，落盘恢复原分片内行序，不改变样本或聚合顺序。
每张GPU启动时至少15,000MiB空闲，不驱逐或重启暂停的505引擎。

执行顺序：404八片→404汇总→404完整对照报告与SHA验收→505同规则八片→505读出汇总锁定→
身份核验后SIGCONT恢复原505引擎/评估器/调度器→等待既有505剩余效用完成→505修正版报告。
恢复前还核对505暂停前八份索引未变、原源码仍匹配、数值worker已释放GPU。
恢复只针对已核验PID/start_ticks/command SHA并通过pidfd发信号，不发新环境评估命令、不授权新的router续算。
任一数值阶段失败不自动重试，且不提前恢复505。恢复后若505失败，也只记录，不自动补发查询或重启。

同步vLLM输出客户端的当前本地实现使用无timeout的队列等待；这降低了长挂起后立即超时的风险，
但不是保证。恢复后仍监测原恢复器stopped/complete和进程身份，保留失败证据。
遥测每30秒，磁盘沿用100GiB底线、80GiB保留、760GiB cohort上限，另给新数值产物8GiB保留。
无时间上限；挂起时长单独登记，不能计为实际模型计算时间。606原磁盘准入没有被豁免。

## 路径与启动

工作目录：`/mnt/workspace/users/wangyifan/skill-RL/SkillRL`。
Python：`/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python`。
设Q为`artifacts/phase12/skillnet37-independent-s404-505-606-v4`，新N为`Q/numerical-readout-v1`。
原数据404为`Q/all-first-calls-v1/seed-404`；505为其`followup-s505/seed-505`。
N必须为新目录；同一attempt拒绝隐式重跑。命令示意（实际执行另记现场）：

```bash
python -B -m skillnet_cohort.numerical_readout --prepare artifacts/phase12/skillnet37-independent-s404-505-606-v4/numerical-readout-v1 --test-report artifacts/engineering/numerical-readout-20260921-full-final-v1.xml
python -u -B -m skillnet_cohort.numerical_readout_run --plan artifacts/phase12/skillnet37-independent-s404-505-606-v4/numerical-readout-v1/plan.json --execute --detach
```

主要输出：

- N/plan.json：授权、固定变体/阈值、源码与旧输入SHA、显存/磁盘约束、暂停现场绑定。
- N/seed-S/token_signals.parquet：稳定读出、原版同次前向读出、数值误差、各门控及完整身份。
- N/seed-S/skill_context_features.parquet、skill_score_comparison.csv：逐skill/phase支持与新旧分数。
- N/seed-S/reports/phase1-results.md、phase2-results.md及明细CSV：修正版与原版对照；原报告不覆盖。
- N/seed-S/reused-labels：原效用紧凑标签的逐字节副本。
- N/seed-S/committed.json、report-provenance.json、complete.json：分阶段验收，不能把启动当完成。
- N/workflow.log、runtime-status.json、logs：当前执行；stopped.json表示失败，不自动重试。
- N/seed505-resume-intent.json、seed505-resumed.json：只有404报告验收及505读出锁定后才允许出现。

## 启动前核验记录

首批数学/原版单元测试16 PASS；首轮集成专项40 PASS；全回归初版895 PASS。
补充了暂停索引不可推进、最长action优先和无覆盖witness发布测试，最终回归另记。
只读核对原404报告路径：原版ranking keys一致，AP/AUROC/Spearman/Kendall差异仅CSV往返的约1e-16。
八份既有witness（222个token×control）CPU稳定算术核验全部通过，P中心化最大误差2.220446049250313e-16。
这不是完整404重算结果，不能据此宣布D排序改善；完整产物和实际GPU启动验收另行追加。

最终代码回归：专项43 PASS/5.21s，完整898 PASS/162.01s，16条既有依赖告警。
最终完整XML `artifacts/engineering/numerical-readout-20260921-full-final-v1.xml`，SHA
`21219188250536eb48e90465fe6525bc6e7b0f0ef8ef0be43498dc44e67e2c86`；
专项XML `numerical-readout-20260921-target-final-v1.xml`，SHA
`411f13549ab16dcc76a20b7612f451e55eb6b7d606458f52d65a65c5a70d8944`。
原118源码SHA仍匹配；用户再次明确要求启动完整seed404重算。准备/真实启动状态以N及后续验收为准。

## 首次实际启动与失败记录（2026-09-21 12:44 UTC）

已实际启动，随后失败停止，不能标记为完整重算通过。plan SHA
`ee6ec2cdee6333ed8ecf7589367b94f58a2e66ae808a097d6c660cfeef02dfa0`；
122源码与111旧文件已绑定；启动前free326.441GiB，需保留88GiB。
launch为12:36:29 UTC、PID1342117；首波四个双卡worker已加载模型并进入实际评分。
12:39:55 UTC停止：shard1/2/3在512-token原版比较器中CUDA OOM，
`direction.py:41`需申请970MiB而仅余约839MiB。暂停505仍占每卡15217MiB；
原版比较器未分块（FP64修正版已分块），造成旧版全形状临时向量峰值超限。
启动前15,000MiB显存阈值不充分，本次不据CPU测试或模型加载成功宣称GPU验收通过。

shard0通过19个短动作原版逐值/稳定P恒等式核验；其首个row1008仅18 token。
未产生任何完整分片、新报告或新性能结果。505仍17进程T，U0八份索引SHA不变，
保持810/3636；未发恢复信号。全部失败记录保留，原RL、效用、报告和122源码不变。
现场见N/active-monitor-handoff.json；按academic-research-suite规则没有自动重试。

建议后续修复将原版对照亦按token分块、保留全词表和原版逐值一致性检查；
在独立恢复目录先验证512-token GPU峰值，再执行全量，不丢弃长动作或改评估定义。
这一修复和再次启动等待用户确认，尚未执行。
