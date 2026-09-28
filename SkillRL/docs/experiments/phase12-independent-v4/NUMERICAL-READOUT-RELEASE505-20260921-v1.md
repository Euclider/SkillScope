## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run + engineering validation
- Origin Date: 2026-09-21
- Verification Status: UNVERIFIED（进程释放已验收；完整数值/科学结果另行验收）
- Version Label: numerical_readout_released505_v1

# 释放505后独占GPU重算404，再新进程接续505

## 授权与实际释放

用户明确要求不再保留暂停505的内存现场，先release，404重算完成后重新启动505评估。
这替代上一版“始终SIGSTOP，最后SIGCONT”安排；不是删除已落盘实验数据的授权。
仅对先前暂停收据绑定的17个PID重新核验start_ticks/command SHA/父子关系，
使用pidfd定向SIGKILL。因原进程均为SIGSTOP，直接终止避免SIGCONT后意外继续采样。
无进程组广播、无无关任务信号。原17个内存状态已不可恢复；磁盘模型、批次、轨迹、缓存和报告保留。

释放收据：新N=`artifacts/phase12/skillnet37-independent-s404-505-606-v4/numerical-readout-release-v1`下
`release-intent.json`、`release-505.json`。八卡从各15217MiB降至各约2MiB。
505八个U0索引SHA不变，仍810/3636；完整审计810条均满足身份、分片、回放和原anchor一致性，
无孤立轨迹文件。释放前数据库字节副本位于`release-cache-snapshot/router.sqlite3`，
SHA=`e7194869e3b202dcc34f4b4f2f73a20fbc07b08bfb4d4827d6ecb24496bc1b0e`。

## 数学与样本不变

上一版规范见[数值修正登记](NUMERICAL-READOUT-20260921-v1.md)。
保留legacy_recorded、stable_raw、stable_centered_gate全部三版；centered D已实际接入。
本次不改其四个冻结数值模块或原118个运行源码，也不改原版比较器的完整token形状。
释放505后，每个新数值worker仍用两GPU分别放原U0/U5 BF16模型，四worker并行、八逻辑分片分两波。
启动前每卡需至少28000MiB空闲；此前15000MiB准入已由真实OOM证明不足。
同一batch、action token、advantage、完整词表、controls、阈值和聚合权重保持。
每分片最长动作优先，原版逐值一致性和固定P中心化恒等式阈值不放宽。
这是显存占用/执行调度修正，不是新的指标定义，也不预设性能改善。

## 后台顺序与505磁盘接续

1. 404八分片完整重算、汇总、三版报告及所有报告文件SHA验收。
2. 505同规则八分片重算、汇总并锁定稳定版读出。
3. 新建505 U0/U5效用进程，仅补缺失的轨迹，保留全部成功缓存和已完成810条U0。
4. 505完整效用标签形成后，生成原版对照分析和数值三版新报告。

404完成前不创建505 GPU worker；505稳定分数锁定前不开始505效用。
不唤醒旧PID，不重跑RL、模型导出、seen/unseen成功率采样或已完成效用轨迹；606不启动。
505仍用同一局内路由/首调用anchor/O/P/N/continuation seed和vLLM采样参数。
复用已修复的FileStore单rank executor，各引擎新建独立rendezvous记录，不复用旧端口或旧FileStore。
U5旧协议已完成的540条继续按原复用规则导入；不是要求重新计算它们。

冻结505数据库有5831次attempt、5826条成功decision，5条在11:57:08暂停时尚未完成。
本次只对ID5826/5828/5829/5830/5831的精确key/input/原行SHA登记各一次显式续算，
依据用户“释放后重新启动评估”授权恢复被丢弃的局内状态。不是之前seed404的旧5条query授权。
原started行、成功decision及历史cache-hit不删除、不改写；只append新attempt并标记本次授权。
成功结果继续走缓存。其他未完成/失败查询以及已用过一次的许可均拒绝自动重试。
仅本地冻结Qwen3-Embedding-0.6B router，外部API调用数为0。

原报告不覆盖。505原版全首调用分析写到其原未完成目录（此前没有该报告），
不调用原publish函数覆盖训练目录的旧报告；修正版报告统一写到N/seed-404和N/seed-505。
所有科学边界继承上一版：事后数值修正、同候选池、完整NA覆盖、全部变体，不能依据最好版本宣称改善。

## 入口、测试与失败策略

新增三个模块：`release_paused_505.py`、`numerical_released_run.py`、`released_505_evaluation.py`。
前者已完成一次显式释放，后两者实现新目录调度与缺失505评估。
释放专项8 PASS；数值/流程/缓存综合专项111 PASS/12.95s。
完整CPU回归、实际plan SHA、GPU长动作验收与启动PID随后追加，不能把准备状态当作运行中。

工作目录为`/mnt/workspace/users/wangyifan/skill-RL/SkillRL`，使用既有
`/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python`。

```bash
python -u -B -m skillnet_cohort.numerical_released_run --prepare --test-report artifacts/engineering/numerical-released505-20260921-full-v1.xml
python -u -B -m skillnet_cohort.numerical_released_run --plan artifacts/phase12/skillnet37-independent-s404-505-606-v4/numerical-readout-release-v1/plan.json --execute --detach
```

状态看N/runtime-status.json、workflow.log、logs、seed505-utility，不能看旧A3的过期running字段。
新N的launch/attempt只允许一次，失败保留stopped和所有日志；不自动重试。
每30秒监测新进程和日志大小，并维持磁盘保护；用户已取消时间上限，不恢复默认30分钟上限。
磁盘保护仍为100GiB空闲底线、80GiB检查点保留、760GiB cohort上限，数值产物另预留8GiB。
没有删除原证据、清理历史张量、Git提交/回滚/push、API或606磁盘豁免。

## 最终工程回归

完整回归935 PASS/173.19s，16条既有依赖告警。XML
`artifacts/engineering/numerical-released505-20260921-full-v1.xml`，SHA
`e08bc430b1c79562b053b37a06695b843b56b25219e3b4cb73df721fd1041845`。
综合专项111 PASS/12.95s，XML SHA
`bbaab02edba844520c14aacb96d8010bcc2cc902d40385874bb75ffd78b474d5`。
这仅是工程验收；完整404新数值报告与GPU长动作验收仍以实际执行产物为准。

## 实际启动与GPU验收

N/plan.json SHA=`4edd50b1afcbeb829a8e98721e8ab440e7ffdfd6ca9af713df16d5ef064380cf`，
125源码、118输入绑定，启动准入free326.418GiB、reserved88GiB。
launch为13:11:47.626301 UTC，父PID1348866，首波四worker PID1349405–1349408。
13:15:29四片实际完成22/11/10/11个decision，全部原版标量逐值一致。
shard1/2/3首个row17/1378/339均有512实际token/control，三者真实评分已通过，未再OOM；
P中心化日志最大绝对误差8.88e-16。shard0首个row1008为18 token，单独注明不外推。
GPU显存观测最高约20197MiB。原505进程均已退出，索引810/3636与125源码SHA再次核验通过。
这只是完整重算的启动验收；404报告尚未完成，505仍待前置条件，不据此宣称排序改善。
