## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-22
- Verification Status: UNVERIFIED (before new continuations finish)
- Version Label: utility_precision_seed404_v1

# seed404：冻结读出、增加效用标签重复数

用户授权：根据效用标签不稳定的讨论修改，并重新评估seed404。只执行本次
效用精度扩展；不重训RL、不做读出前向、不启动505/606、Phase3或外部API。
不覆盖旧报告，不删除、提交或回滚任何历史文件或未提交修改。

## 科学设置（在新增标签产生前固定）

- 固定原U0/U5权重、完整SkillNet-37、冻结Qwen3-Embedding-0.6B逐状态top1。
- 全部25种自然出现技能、404个首次调用锚点保持不变；每条轨迹每技能只取首次。
- 环境seed、game、历史前缀、技能/control内容和50步上限不变。
- O/P/N×U0/U5按同一个anchor、同一个continuation seed配对。没有改变现有干预：
  从锚点开始，后续目标技能再次被选中时继续同一payload干预。
- gold seeds从[63011,63021]扩至
  **[63011,63021,404,404100,404200,404300,404400,404500]**。
  新增6组，全部锚点使用同一组列表，不看结果增减重复或筛选技能。
- 404是训练seed，也是其中一个新增续跑base；数值相同本身不保证测量更准。
  其余5组为404*1000+100*r，r=1..5。当前解码逐步使用base+absolute_step，
  新base之间及新旧base之间至少间隔100/50，避免新重复的50步seed区间重叠。
  旧63011/63021的区间有交叠，保留历史实现，不将数字不同等同严格独立随机流。
- 独立evidence seed62011不变，不转成gold。旧每端点3,636条全部验证后复用；
  新增每端点7,272条，共14,544条；完成时两端点共21,816条。
- 285列已有读出及其候选池、方向、阈值、聚合方式全部冻结；只重算它们与新标签的比较。
  不拟合新指标、不按新gold挑赢家。旧标签已经可见，属于事后精度扩展而非新确认性研究。

## 统计输出

主结果固定为8组gold；完整保留2/4/8组前缀精度曲线，以及新增6组单独的敏感性结果。
这些不是4个可择优主实验。原2组效用点估计、区间和全部285列排名必须先复现。

每个skill保留调用/轨迹/game/anchor数量、M0/M5/DeltaM、配对区间、逐续跑seed效用、
重复间符号分歧、bootstrap符号频率。全部37技能coverage保留未调用/缺readout的NA；
区间跨0标为方向尚未分辨，但不从原始点标签排名中事后删除。
两组无变化的退化零宽区间不证明真实误差为零；单game不伪造跨game区间。

点估计仍等权game。逐技能区间沿用10,000次配对game/continuation bootstrap；
读出比较沿用2,000次共享game/continuation bootstrap，在同一次draw上算方法差。
缺任一共同池技能则整个draw为NA，不按方法临时换池。bootstrap频率不是后验概率，
区间不包含训练随机性、训练状态到unseen锚点的分布差异，也没有校正既往公式搜索。
更多续跑主要减小条件于锚点的蒙特卡洛误差，不增加独立game或RL seed数量。

## 工程范围及启动

新目录：`artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s404-v1`。
全部原输入、模型清单、141份历史源码及新增实现均SHA绑定；旧数据只读。
复用轨迹复制到新分片，旧文件不移动或改写。新router缓存独立，API额度严格为0。
vLLM TP=1、8个worker，每卡一片；沿用FileStore rendezvous避免此前TCP端口争用。
U0补齐后U5补齐，随后CPU自动计算全部报告并封存。失败保留、不自动重试。

沿用用户取消总运行预算后的无限总时长；不恢复30分钟硬超时。
保留磁盘门槛：至少100GiB空闲、cohort最多760GiB、80GiB检查点保留量；
新增记录按每条4MiB加8GiB杂项作准入预留，不自动删除旧数据。
30秒进程/日志/显存心跳；停滞只告警。子进程失败或磁盘保护仅停止本次创建的进程组。

以下为新目录的一次性入口；首次启动后不要重复执行：

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B \
  -m skillnet_cohort.utility_precision_run prepare --tests <passing-regression.xml>
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B \
  -m skillnet_cohort.utility_precision_run launch
```

cwd为SkillRL仓库。完成标记是新目录`complete.json`，报告为
`reports/phase1-results.md`、`reports/phase2-results.md`；旧报告不会被覆盖。
当前文件仅登记设置，不宣称新增评估已完成或标签已经可靠。
