## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21
- Verification Status: UNVERIFIED
- Version Label: seed505_port_recovery_v3

# seed505：端口竞态修复和显式续评

## 授权与真实停点

用户明确要求修复505第7分片启动vLLM的端口冲突后续跑505。本次只执行既有
all_u0_first_calls_v1效用补评，不重跑RL、导出、完整seen/unseen性能或已完成readout。
不复算/替换seed404已发布的新报告；不启动606，不降低存储门槛，不删除证据。
全部旧失败、未提交修改、router账本及成功缓存保留，无Git提交、回滚或推送。

路径约定（均在SkillRL目录下）：

- Q = artifacts/phase12/skillnet37-independent-s404-505-606-v4
- F = Q/all-first-calls-v1
- 旧attempt A2 = F/recovery-v2
- 新attempt A3 = F/recovery-v3
- 保留的505结果目录 = F/followup-s505/seed-505

A2于2026-09-21 11:04:43 UTC停止。505 shard7在模型加载/首条续跑之前出现
`torch.distributed.DistNetworkError: port: 40669 ... EADDRINUSE`。
其余分片被原调度器的子任务错误保护终止，并非八次独立RL失败。
505的八个readout分片及聚合完整；U0只有540条严格匹配复用结果，新增0条，
预期3636条，缺3096条；U5未打开。505本地router缓存完整且attempt/decision均0，
因此本次无需、也不新增任何中断查询续算许可。

404已于10:35 UTC生成新报告，完成U0/U5各3636条、25种自然出现技能、404个
每轨迹每技能首次调用锚点。404的完整seal及正式报告在新恢复计划中作为只读证据绑定。
606仍因旧队列物理磁盘准入不足而未启动，不伪造三seed完成。

## 根因与窄范围修复

本机vLLM 0.22.0的UniProcExecutor先调用get_open_port()，该函数bind一个socket
后立即关闭，之后才用返回端口调用torch.distributed初始化。检查与占用不是原子操作。
旧日志八个rendezvous端口彼此不同，故不能把占用者断言为另一评估分片；日志没有
保存占用者身份。能够确认的是40669在TCPStore bind时不可用，以及这条实现路径存在竞态。

新增两个模块，不编辑原116份冻结源码或已安装依赖：

1. `skillnet_cohort.vllm_file_executor.FileStoreUniProcExecutor`继承安装版UniProcExecutor，
   只覆盖`_distributed_args`，将TCP rendezvous改为FileStore；其worker初始化、加载模型、
   NCCL/Gloo组、execute_model、sample_tokens等全部继承。严格限制单机、TP/PP/DP及world size
   均1、单张可见CUDA卡。每个引擎通过mkdtemp在/tmp下创建独立0700目录，store路径从未存在；
   不复用旧文件，不搜索“下一个端口”，不重试失败引擎。
2. `skillnet_cohort.first_calls_port_recovery`为新attempt准备/运行入口，以及仅用于其505
   评估子进程的policy适配入口。和原build_engine参数逐项比较，只有executor类路径不同；
   继承原VLLMPolicy.generate/generate_batch，逐请求seed、温度、top-p、token边界等均不变。

采用文件初始化的依据：PyTorch正式文档支持file:// rendezvous，要求支持fcntl锁且每次使用
新路径。此任务每个通信组只有一台机器上的一个rank，因此本机/tmp足够；未改变跨机训练。
见[PyTorch文件初始化说明](https://docs.pytorch.org/docs/2.12/distributed.html#shared-file-system-initialization)。
每次引擎的文件URI、PID、rank和初始化完成receipt记录在A3/seed-505/rendezvous。
FileStore可能自行清理临时通信文件，因此把它放在证据树外，避免与已有磁盘扫描发生新的删除竞态。
该修复消除这条TCPStore探测/释放/再次bind路径，并不声称所有网络或显存故障都不再可能。

## 续跑流程与证据保护

准备阶段核对特定旧失败、GPU空闲、旧supervisor退出、原116源码、505已完成readout、
保留模型与旧seal、540条逐一身份/replay/steps/哈希、原索引前缀、本地缓存完整性和存储准入。
同时绑定404完整seal、已发布7份报告、所有旧attempt日志/计划与报告归档。
新计划冻结118份运行源码及相关安装版vLLM/PyTorch源码哈希，科学协议JSON不变。

执行顺序固定：

1. 505 U0仅补缺失的3096条，完整分片不启动policy，已有trajectory ID全部跳过。
2. 锁定既有readout直接评分；然后严格匹配复用U5旧540条，补齐其余续跑。
3. 按相同首调用协议统计、封存；核对保留证据后才替换505正式报告视图，旧报告可恢复。
4. 标记404/505补评完成、606仍pending；不发布“三seed完成”的汇总、不自动启动606训练。

子进程日志、launch/stop/status/遥测全部写入A3，A2失败记录不覆盖。每阶段GPU启动前检查
router是否存在未知未完成query，遇到则停在准入处，不自动放宽原no-retry保护。
每60秒执行原容量保护，每30秒状态记录；无新时间上限。失败只停止本attempt启动的
子进程组，不重试、不信号其他实验进程。原100GiB空闲底线、80GiB预留、760GiB队列上限不变。

## 工程测试与执行命令

2026-09-21专项回归42 PASS/15.62s，报告：
`artifacts/engineering/seed505-port-recovery-20260921-target-v1.xml`。
包括真实八CPU进程FileStore/Gloo初始化、指定TCP端口被占用、路径不复用、单rank限制、
vLLM自定义executor解析与继承、全部生成参数一致、冻结方案/依赖/测试哈希、防越权命令、
未知cache中断在GPU前失败、子进程异常不自动重试、仅505的恢复顺序和policy类异常后复原。
完整回归及实际GPU接续验收结果在本文件末尾和A3中追加，准备/启动不等于科学结论已验证。

```bash
# cwd: /mnt/workspace/users/wangyifan/skill-RL/SkillRL
# 以下两个命令本轮由agent按用户明确授权执行；不是要求再次运行。
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B \
  -m skillnet_cohort.first_calls_port_recovery \
  --plan artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1/recovery-v2/plan.json \
  --prepare artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1/recovery-v3 \
  --test-report artifacts/engineering/seed505-port-recovery-20260921-full-v1.xml

/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -u -B \
  -m skillnet_cohort.first_calls_port_recovery \
  --plan artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1/recovery-v3/plan.json \
  --execute --detach
```

### 完整离线验收

`seed505-port-recovery-20260921-full-v1.xml`：859 PASS、0 FAIL，172.97秒，16条既有依赖/
语法告警。测试覆盖phase1、phase2、phase3、skill_bank、skill_router、skillnet_cohort和
experiment_settings，CUDA_VISIBLE_DEVICES为空。沿用既有first_calls_cpu_plugin，仅隔离旧
CPU优化器测试中的GPU日志调用，不跳过其优化器或数学断言。没有额外真实RL或GPU采样预检。

### 准入与启动

prepare成功：118份运行源码、6份安装依赖源码、8081份保留文件及8份索引原前缀。
全部404 seal/报告与505现有readout/540轨迹通过验证，505空router缓存PASS。
原容量准入PASS：空闲326.699GiB、cohort231.005GiB、本次required116.406GiB。
plan SHA为`9d319b2c7e9a4fb2d2025f2ec1537f272544b152dee6ecfcc609e959f92b2e16`。
显式恢复进程PID1325542启动一次；这是启动记录，实际GPU生成验收在下文另记。

### 真实八卡接续验收（11:41 UTC）

八个引擎均使用不同file:// URI，通过NCCL初始化与模型加载，8份ENGINE_READY记录与
plan/rendezvous SHA匹配。评估器1326115–1326122和supervisor1325542均存活，无新stop。
每卡约15217MiB，磁盘326.6367GiB；最初有Triton JIT预热提示，随后实际生成成功。

505 U0为563/3636，原540条保留、新增23条，八片新增2/2/3/3/3/3/3/4。
逐一核验新增identity、分片、随机seed、原锚点语义、prefix replay及完整steps PASS；
8份旧索引前缀原字节不变。11:38:18的全量保留核验为118源码/8081文件/8索引PASS，
404完整seal与已发布报告、505旧报告及归档均未改。

U5及505新报告尚未开始，606仍pending；不能将恢复验收解读为实验全部完成。
完整机器记录在A3/active-monitor-handoff.json。按用户此前要求，实际稳定接续后结束
agent主动检查，后台评估和原容量/异常保护继续，不进行新失败的自动重试。
