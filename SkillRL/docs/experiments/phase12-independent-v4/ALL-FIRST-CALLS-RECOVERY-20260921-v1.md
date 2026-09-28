# 全部首调用补评：2026-09-21 显式恢复

**最新停点：04:17:07 UTC 本次恢复也已停止，不在运行。** 原磁盘扫描竞态未再出现；
新原因是旧中断留下的本地router未完成reservation触发禁止自动重试。具体审计见末节。
本文件命令已执行过一次，不能再次原样执行以重试；下一attempt须先复核并经用户确认。

## 失败与新发现

用户授权修复并复用结果恢复；不重跑 RL，不中断现有任务，不覆盖失败证据。
本次遵循 academic-research-suite 的失败复核流程，工程测试不视为科学验证。

`all-first-calls-v1/assessment.log` 记录 00:51:42 UTC：容量扫描列出
`seed-404/router.sqlite3-journal` 后，SQLite 提交正常移除了该文件，随后的 `lstat`
触发 `FileNotFoundError`。不是模型/RL失败或该次补评容量不足。

修复前实际可复用：seed404 的八个 readout 分片和聚合；U0 的 1,871/3,636 条续跑
（540 条旧协议复用，1,331 条新补充）。索引与完整轨迹文件一一对应，未发现孤立文件。
U5 新目录尚未打开，预测尚未锁定；旧报告没有替换。

核查过程中旧队列自然结束 seed505，然后在 seed606 准入处停止：
`recovery-v4/not-started-606.json` 的原因是 `remaining_shared_disk_budget`。
`queue_finished.json` 为 `budget_stop`、completed=[404,505]，不是三 seed 完成，
也不是已取消的时间上限重新生效。606 从未启动。
当时空闲约 332.1 GiB；606 峰值新增估计 176.859 GiB + checkpoint reserve 80 GiB
+ 最小空闲 100 GiB = 356.859 GiB，缺约 24.8 GiB。760 GiB cohort 上限保持不变。

## 修复范围

- 新增 `skillnet_cohort.first_calls_storage`，不改冻结的旧 runtime。
  仅忽略扫描期间消失的 `router.sqlite3-{journal,wal,shm}`，且其永久数据库必须仍为
  非符号链接的普通文件。仍存在的 sidecar 正常计入容量。
  已知原子发布的进度、续跑 JSON 和完成收据临时文件也按严格名称/目录条件处理；
  后两类要求永久替代文件已存在。普通证据丢失、权限错误仍 fail-closed。
- 新增 `skillnet_cohort.first_calls_recovery`，独立 attempt 和日志，不热改旧队列105个、
  覆盖修订112个冻结源码。恢复 plan 额外冻结这两个模块。
- seed404 只续 U0 缺失1,765条；逐条校验已有身份、arm/seed/anchor、prefix replay、
  实际续跑、分片归属和文件 SHA；保留索引原前缀，不覆盖、不重跑已完成记录。
  完整分片不再次初始化 policy。孤立轨迹、截断索引或错误完成收据拒绝自动覆盖。
- 跳过所有已完成 readout/聚合，U0补齐后锁定原固定评分，U5仍复用旧540条，
  再补其余3096条；报告、封存、旧报告归档核验均完成后，才替换当前seed报告视图。
  每轨迹每skill仅首调用、全自然技能/全首调用、原C/P/D公式/优势权重均不改。
- 不给任何旧进程发信号；旧队列已退出，不再需要暂停调度器。
  对成功完成的505进行相同冻结口径补评。旧全队列完成门槛改为逐seed真实完成验证，
  **不伪造全三seed完成收据**。606保留磁盘阻塞记录；不在补评入口启动RL、降低保护或删除文件。
- 只有三seed新口径均完成才发布跨seed总报告。当前计划可完成404/505补评，
  随后留下 pending606；不是默默缩减科学计划或把两个seed写成三个。
- 无自动重试。新失败保留在独立恢复目录，需再次复核，不重用旧attempt日志。

## 路径与命令

仓库：`/mnt/workspace/users/wangyifan/skill-RL/SkillRL`。
Python：`/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python`。
F：`artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1`。
恢复目录：`F/recovery-v1`。

CPU回归使用 `CUDA_VISIBLE_DEVICES=''` 和 OMP/OPENBLAS/MKL 1线程。
旧CPU optimizer测试仅通过现有 `tests.skillnet_cohort.first_calls_cpu_plugin` 隔离GPU内存日志。
初次定向回归71 PASS/1 FAIL是测试fixture导入绑定污染，修复测试隔离后73 PASS；
失败XML保留。最终完整回归 **786 PASS / 156.93s / 0 failures**，XML为
`artifacts/engineering/first-calls-recovery-20260921-full-v1.xml`。
包含真实SQLite提交竞态、永久证据/权限错误fail-closed、容量保护、严格续跑校验、
跳过完整分片、404恢复顺序、505逐seed完成门槛、606容量阻塞及禁止重训/导出/性能重采。
没有新的真实ALFWorld/GPU工程预检或科学实验重跑；执行验收另行追加。

准备（只运行一次；不能把旧失败目录当新attempt）：

```bash
python -B -m skillnet_cohort.first_calls_recovery \
  --plan artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1/plan.json \
  --prepare artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1/recovery-v1 \
  --test-report artifacts/engineering/first-calls-recovery-20260921-full-v1.xml
```

准备PASS后显式启动一次：

```bash
python -u -B -m skillnet_cohort.first_calls_recovery \
  --plan artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1/recovery-v1/plan.json \
  --execute --detach
```

只读状态入口：`F/recovery-v1/deferred-status.json`、`recovery.log`、`supervisor.json`；
各seed的恢复日志在 `F/recovery-v1/seed-N/logs/`；原始新口径产物继续写已有variant，
即404在`F/seed-404`，505在`F/followup-s505/seed-505`。
`pending.json` 表示未启动的606仍被原磁盘门槛阻挡，不能视作错误自动重试或全部完成。

磁盘不足时优先增加容量或由用户指定可回收目标。原逐行概率回收仍要求完整封存和
每行独立的逐bit再生证明；当前不删旧概率、检查点、轨迹或失败目录。
本轮无Git提交、回滚、推送、付费API或旧RL/完整性能重跑。

## 04:18 UTC 失败复核：本地router未完成reservation

新plan SHA=`5847e38b1ab7e94a987a50a7fd943390afdeb4d614558a8df89043196a7d5dfd`，114源码。
04:14:42启动PID1295037，04:15:51启动8个效用分片。04:16:56成功新增1条完整续跑，
U0变为1872/3636，所有原1871条和索引前缀未变；八分片读出没有重算。
04:17:03 shard5报 `RouterCacheError: Prior failed/incomplete local query; no automatic retry`，
04:17:07恢复器保护性退出，其他7个自己的子会话被停止；八卡随后均空闲。

SQLite只读quick_check=ok；30851条成功attempt与30851条decision一一对应，成功记录没有损坏。
有5条started且没有decision：4条时间为旧停机00:51:42，1条为本次保护性停止04:17:06。
不是embedding模型/API故障，外部API=0。原缓存代码为避免悄悄重复调用，拒绝同key的再次reservation；
本次恢复准入未事先检查这一状态，因此是在GPU加载后才暴露的独立恢复缺口。

`A/active-monitor-handoff.json`保存确切5个id/key/time、SQLite hash、失败log hash及新的1872计数。
原报告、原日志、1989保留文件和114源码hash仍验证通过，U5及505新口径补评均未开始。
新错误不推翻786项离线测试，但说明它们没有覆盖带孤立router reservation的真实恢复启动。

下一步建议：在不删除/覆盖原attempt账本、不丢弃成功缓存的前提下，只对这5个未完成key登记
一次显式本地续算许可；新恢复入口应在任何GPU启动前检查全部未完成reservation及允许列表。
必须另建恢复attempt并绑定新代码/当前缓存快照，不绕过付费API重试保护，不重跑完整轨迹/RL。
按academic-research-suite的失败不自动重试规则，本轮停在复核/确认处；已向用户发出该范围的询问。
606仍另被原磁盘准入阻挡，两个问题不可混写成一次修复已全流程成功。
