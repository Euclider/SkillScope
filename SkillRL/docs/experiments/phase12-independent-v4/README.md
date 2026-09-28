## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-18
- Verification Status: UNVERIFIED
- Version Label: phase12_independent_windows_v4

# 冻结 SkillNet-37：三个独立 U0→U5 窗口

## 用户确认与研究问题

用户确认修改代码并开始 Phase1–2；随后确认 seeds **404/505/606**，依序执行，共用 **12h 目标、30h 硬上限**。
后续 seed 只按剩余时间/磁盘及已完成 seed 的时长决定是否启动，不按 reward、效用、排名或显著性决定。
三条都从共同 Qwen3.5-4B B0 初始化，不能把 505 当作 404 的延续。
当前摘要/intro 中的 “Our experiments show” 是待检验论文表述，不是这批尚未完成实验的结论。

RQ1：技能内容及 router 冻结时，RL 更新是否伴随技能语义边际效用变化？
RQ2：复用更新 batch 的 reward-directed readout 能否预测变化/优先定位下降，比同源 magnitude-only readout 有增量？
不要求每个技能正负翻转、不要求每个 seed 同向，不按结果挑窗口/符号/赢家。
短独立窗口支持初始策略附近的局部现象和跨 RL seed 重复性，**不直接证明整个训练阶段都成立**。
重复的 continuation seeds、锚点及技能/阶段分层不是额外独立 RL seeds。

Phase3 不在本机开跑：现有实现已经以相邻 policy 快照计算 readout，冻结该窗口内的 skill 版本，
编辑/验收后将新 policy 和接受的库作为下一窗口起点；不是永远固定 U0。
当前确认四分支为 SkillRL-style failure / gated D / −P / +C，**没有单独 norm-only 闭环分支**。
摘要若宣称 Phase3 优于 magnitude-only，须另行确认并实际运行对应分支；本轮不擅自新增。

## 参数与工作量

| 项目 | 冻结设置 |
|---|---|
| Policy / bank | Qwen3.5-4B / SkillNet-37，完整同一库、无 task 专属子库 |
| Router | 官方 Qwen3-Embedding-0.6B，冻结 CPU FP32、8线程、batch8；逐状态 top1 适配版；0 API |
| RL | verl/FSDP1，GRPO，lr=1e-6，16 games×8 trajectories/迭代，5迭代/seed |
| Optimizer | AdamW .9/.999，wd .01，constant，无 warmup；PPO epoch1；global mini-batch128状态行，micro1/GPU |
| 其他训练 | clip .2、dual clip3、grad clip1、KL low_var .01、entropy .001；reward成功10、失败0、invalid action .1 |
| 数值/显存 | 原生 FP32 master / BF16 mixed；按层 FSDP、CPU shard初始化、原生 optimizer-state CPU offload |
| Generation | vLLM0.22.0、TP1×8、每卡 max_num_seqs16、prefill8192、ctx4608、memory .45、eager；无HF generate回退 |
| Limits | prompt4096、response512、50环境步、history2，train T1、eval T.4，top_p1；不截断溢出 prompt |
| Checkpoint | 每5迭代保存/Seen monitor64；无验证前置；本次端点仅 U0/U5 |
| Performance | 每seed U0/U5 各140 seen+134 unseen，共548完整episodes |
| Utility | unseen134全任务采集自然调用锚点；最多12自然支持技能、每技能最多12 anchors；1 evidence+2 gold seeds |
| O/P/N | Original、token-matched Placebo、NULL；两个端点同锚点/同 seed，自调用前固定prefix；最多2592续跑/seed |
| Prediction | 固定状态/动作token；先锁 readout 排名，再打开 U5 gold；不拟合新预测器 |

16×8=128 条 rollout/迭代，五轮为640条/seed；**128 optimizer minibatch 是状态行而非128完整轨迹**。
当前用户询问适度增量，但兼顾三seed合计12h，未增加训练量/anchor量。
实际采样池为3553 solvable train games、全部六类。相同group内8条同game，group使用seed+group_index的TextWorld洗牌。
只读预核对前三seed的首轮分别16/16/15 unique games，均覆盖六类，首轮跨seed game overlap=0；
五轮分别79/78/79 unique games。预核对依赖当前文件枚举顺序，真实轨迹仍要核验。
全任务覆盖不保证每个skill有足够支持；首轮每类仅1–6games，而读出须每skill≥4games/8trajectories/20非零优势decisions。
支持不足必须弃权，不强制选skill、不用gold补采样。

## 起点采集协议及保留边界

新 `window_start_old_only_v1`：仅窗口起点 policy 生成的**第一轮实际训练 batch**完整归档，
包括状态/动作token、mask、真实GRPO advantage、skill/version metadata、live OLD全词表FP32概率。
U0→U5对应batch u0001；未来U5→U10应对应batch u0006，不是一直用u0001。
本次独立seeds各有自己的u0001。其余四轮仍正常生成/优化，保留轻量逐步token/reward、
逐轨迹game/skill计数、reward/advantage统计、实际optimizer行索引/step/lr/gradnorm及metrics；不存完整中间batch/状态。
不再额外前向/保存每轮NEW全词表。U5在**同一份起点输入**上重放，NULL/Placebo对照及C/P/D数学不变。

唯一无损序列化为 `torch_shuffle4_lzma_v1`：保存全部FP32位模式、mask和metadata，每行编码后完整解码逐值核验，
再无覆盖发布并校验文件SHA。新reader兼容历史torch文件；不能直接对新封装调用torch.load。
自然状态24行压缩比范围约.412–.449，不能用合成行约.257冒充真实比例。
新容量准入按每token650000bytes+每行65536bytes上限及首批token估计1.5倍留量，
超界直接停，不减精度、不丢token、不改预算。

保留 U0/U5 FP32模型、U5完整原生checkpoint、起点batch/轨迹、压缩live OLD行、router账本、所有评估轨迹、
compact per-token/decision/skill结果、支持/弃权、封存manifest和报告。窗口全部结果封存后才进入回收检查；
现有安全回收仍要求独立逐行 bitwise 再生证明，无证明则保留。
**本轮未授权任意删除模型/历史记录；不会承诺仅保留Markdown仍可完全重算。**
Phase3的编辑器需要窗口内失败/成功轨迹，不能照搬Phase1–2的中间状态不归档策略。

## 时间、容量与执行

已通过且不重跑：v4八卡原生恢复/vLLM连续更新预检、16条真实ALFWorld八分片测时、128状态行精确采集、
32自然状态router校准、40行无损压缩核验。它们是工程证据，不是这批科学结果。
组件外推的条件保守情景单seed约26.18h，**不是保证或统计置信上界**：假定训练均50步、无router缓存，
完整12技能评估，readout另留1s/forward+1s/row scoring以及2400s初始化/导出/报告余量。
未实测完整Ray RL/全readout/seed耗时，不能保证三seed12h甚至30h内全完成。
完成404后，用完整seed耗时的1.25倍重估下一条；时间不足则不启动，不将未完成seed隐藏。
存储共同上限760GiB、空闲线100GiB、下一checkpoint reserve80GiB；总队列统一计量。

新目录及启动命令（只能执行一次，失败不自动重试）：

```bash
cd /mnt/workspace/users/wangyifan/skill-RL/SkillRL
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -u -B -m skillnet_cohort.seed_queue \
  --plan docs/experiments/phase12-independent-v4/prepared-v2/cohort.json --execute --detach
```

Root：`artifacts/phase12/skillnet37-independent-s404-505-606-v4`，每seed子目录独立。
看 `queue_launch.json`/`queue.log`、`seed-404/supervisor.log`、`seed-404/logs/train-u0000-u0005.log`、
`rollout_progress/`、`forward_progress/`、`optimizer_steps/`、`metrics/`。
`queue_finished.json`区分complete/budget_stop；任何`queue_stopped.json`/seed `stopped.json`都不是完成。
本地只核对已上传GitHub main，无新推送/提交/回滚，不运行Phase3。

## 报告与结论边界

每seed新生成 `reports/phase1-results.md`、`phase2-results.md`，另有performance/task、utility/CI、
AP/AUROC/相关和多预算CSV、输入哈希。共同队列最终生成新 `reports/phase12-cohort-summary.md`，
明确全部登记seed及未完成seed，不覆盖历史Phase1/Phase2报告。
Phase1主指标Δ(success_O−success_P)，NULL为次要；配对game/continuation bootstrap，保留不确定性。
Phase2主gated D，−P、同源幅度/centered norm/KL/JS/activation及random/old utility对照；
同一支持池比较AP、Spearman/Kendall、P@k/R@k、下降量捕获率；0/5pp阈值、分阶段单独报告。
D为单侧下降风险，D=0不能解释为稳定/上升；不把方向相关偷换为准确预测每个skill符号。
readout本身无需post-update rollout；独立金标签验证仍需要且完整计入环境成本。
短窗口/三个RL seeds不能证明全部训练阶段、其他benchmark或Phase3闭环收益。

## 启动后首次核对

2026-09-18 11:09:54 UTC队列已启动，PID1016912；seed404 supervisor PID1016953。
11:14:58 UTC，首步128条真实训练记录已落盘，16个game与离线预核对集合一致；当前仍0/5更新，
505/606未启动，无停止标记。首批未结束，尚无完整迭代/seed实测ETA。
verl日志中Qwen3.5的MFU估算未受支持，因此MFU=0不可作为实测算力利用率用于论文；
应使用实际walltime、token/环境步吞吐及GPU监测，保留该测量缺口。
