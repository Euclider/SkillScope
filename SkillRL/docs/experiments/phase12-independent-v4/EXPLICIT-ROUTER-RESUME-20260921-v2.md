# 五条未完成本地选技：显式续算恢复 v2

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-21
- Verification Status: UNVERIFIED（工程验证不构成科学结论）
- Version Label: explicit_router_resume_v2

## 授权与不变量

用户已明确批准“允许修复这5条未完成查询后继续”。仅恢复已登记404/505全首调用补评，
不调用外部API，不重跑RL、已完成轨迹、已完成读出/聚合或完整性能评估。
每轨迹每skill仅首调用、全自然技能/全首调用锚点、C/P/D数学与实际训练advantage、
冻结SkillNet37和逐状态top-1的Qwen3-Embedding-0.6B router均保持不变。

原故障：00:51磁盘扫描竞态使四条本地选技reservation未写回；其修复后的首个恢复attempt
于04:17碰到旧reservation no-retry保护并退出，保护性停止另留下1条，共5条。
准确id为30851、30852、30853、30854、30856，key/time见旧恢复
`recovery-v1/active-monitor-handoff.json`。本轮开始时U0已有1872/3636完整轨迹。

这些是本地CPU检索，不是付费API或新的环境采样。旧`started`行保持原样，
**不删除、清空、改key或把旧失败伪装成成功**。仅给这5个key各一次显式续算许可。

## 实现和审计

- 新模块`explicit_router_resume.py`：只读SQLite完整性/协议/成功记录digest校验，
  严格5-key授权；精确绑定原reservation的id、时间、status及完整row SHA。
  新请求输入必须与原输入、原cache key逐项一致。
- 原router SQLite在新attempt中按原始字节备份，记录SHA；每次阶段前后用只读SQL
  比较四张表中原有的全部行（protocol/decisions/attempts/cache_hits），允许追加，禁止改旧行。
- 在`BEGIN IMMEDIATE`事务内检查完整batch、未消费的许可及原调用预算，再append新attempt。
  新attempt/新decision携带授权SHA与原attempt id。多进程竞争至多一个能占用许可；
  若这一新增尝试再次中断/失败，不可再次续算。
- 原5条`started`行会永久保留；不能仅凭raw started计数认定仍有5个活动任务。
  应按key关联decision及`explicit_local_resume`来源判断历史中断是否已有结果。
- 未登记的中断仍拒绝；GPU启动前的完整cache preflight拦截，不再等加载模型后才发现。
  成功记录始终走cache hit，不重复encoding；原尝试与新增尝试都计入已有本地调用上限。
- 适配器只在显式seed404评估子进程中替换batched local router的cache factory；
  不修改原源文件/默认RouterCache，不给API路由加重试，不改变encoder、向量、排序或数值配置。
- 新`first_calls_router_recovery.py`复用上一版本已测试的续跑/发布顺序，仅在本进程切换
  assessment类并在退出时恢复；源码文件和原失败日志均不修改。
  原114源码仍冻结，增加2文件后新plan共116源码。

旧的30851成功decision和全部attempt/cache-hit账本必须保留。所有1872条现有完整轨迹、
8个读出分片/聚合、OLD概率、batch、模型/checkpoint和旧报告也均保留。
新日志/状态/授权在`all-first-calls-v1/recovery-v2`，不重用v1日志或启动命令。
404 U0缺1764条；补齐后沿原固定评分锁定→复用U5旧540条→补缺→分析/封存/发布。
随后505按原提前固定覆盖规则补评。

606保持单独磁盘阻塞：原队列completed=[404,505]且606未启动；准入需约356.859GiB空闲，
本轮开始约332GiB。补评不放宽该条件、不删除证据、不偷开606训练；不把两seed写成三seed完成。

## 验证与启动

专项测试36 PASS/9.93s；覆盖真实冻结router的CPU fake-encoder路径、5条续算后所有旧行逐项不变、
成功cache hit、事务原子性、8线程竞争单许可、进程重新创建仍不能重复续算、再失败仍拒绝、
未知中断启动前拦截、错输入/错模型/预算/API/路径/快照/计划拒绝、404/505命令接线、
工作流类退出恢复及原始evaluate参数完整透传。未加载GPU policy或改变实验结果。
初次专项30PASS/1FAIL是非法root先读文件而非先拒绝权限；调整校验顺序后通过，失败XML保留。
完整回归 **817 PASS / 154.81s / 0 failures**，报告为
`artifacts/engineering/explicit-router-resume-20260921-full-v1.xml`。
16条依赖弃用/语法告警保留，不作为失败隐藏；实际启动验收另行追加，不预写运行成功。

在仓库`/mnt/workspace/users/wangyifan/skill-RL/SkillRL`使用环境
`/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python`。
准备一次（CPU，无GPU/环境交互）：

```bash
python -B -m skillnet_cohort.first_calls_router_recovery \
  --plan artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1/recovery-v1/plan.json \
  --prepare artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1/recovery-v2 \
  --test-report artifacts/engineering/explicit-router-resume-20260921-full-v1.xml
```

准备通过后，按本次明确授权启动一次：

```bash
python -u -B -m skillnet_cohort.first_calls_router_recovery \
  --plan artifacts/phase12/skillnet37-independent-s404-505-606-v4/all-first-calls-v1/recovery-v2/plan.json \
  --execute --detach
```

观察新目录`deferred-status.json`、`recovery.log`、`seed-404/runtime-status.json`和`seed-404/logs/`。
`cache-admission/`记录每阶段GPU启动前检查，`cache-after/`记录成功阶段后的完整原账本保留检查。
实际产物仍在已有全首调用variant中；报告只在完整分析/封存/旧归档hash检查后授权替换。
仍无自动失败重试，无时间上限，原磁盘保护继续有效；稳定验收后停止agent主动盯守，后台状态记录继续。

## 04:40 UTC 已准备并启动

新plan SHA=`18eaaddd388dff9a2028be797f13834d700d97cb5e557b1e81daaf7e62fc54fe`，
116源码、2009份既有文件/备份的SHA绑定；旧114源码没有修改。
本地续算授权SHA=`4e04becb9c8f712ddc813ae0d62351a4f862bc5951b54ce5e27e415b956dd56c`。
数据库备份306270208bytes，SHA与原账本同为
`cda4c3db81f9dc870f28de5080873b3d2b96e0a9d854941a9e48dd500f191f58`。
启动前cache audit PASS：30856原attempt，30851成功decision，恰好5条已授权未完成查询。
磁盘可用331.855GiB；本次补评reserve116.40625GiB，保留100GiB底线，准入通过。
606更高的356.859GiB准入未满足，仍无清理/放宽/启动。
04:40:08 UTC新launch，后台PID1302727，入口为`first_calls_router_recovery`。
这是恢复启动记录，尚不代表全部5条已经请求续算或任何端点补评已完成；后续以实际验收为准。

## 04:44 UTC 实际接续验收通过

五条查询全部各成功续算一次，旧id→新id为30851→30878、30852→30860、30853→30858、
30854→30900、30856→30857；新decision都携带相同授权SHA，旧started行原样保留。
只读事务将live库与封存快照按四张表逐行比对，旧protocol/decisions/attempts/cache_hits改变行数均0。
2009份既有文件及索引前缀、116源码hash仍全部PASS，旧7份seed404报告未变。

截至04:44:17 UTC，U0为1895/3636（本attempt新增23条），八分片241/238/232/237/236/235/238/238，
各自新增3/3/2/3/3/2/4/3条。新增记录身份、prefix replay及完整steps校验通过。
PID1302727存活，无新stopped；八个评估子进程持续运行。现场记录为
`recovery-v2/active-monitor-handoff.json`。未开启U5或505补评，未替换最终报告，不能声称完整补评已完成。
此次工程恢复已经越过原五条阻塞，agent主动验收结束；后台继续404→505，606磁盘阻塞不绕过。
