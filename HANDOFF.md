# Skill-RL 任务交接

核对时间：2026-09-16 21:15 起（北京时间）。研究根目录 /home/wangyifan/skill-RL，Git 仓库为其下 SkillRL/。仅新增交接文档，未改代码或启停实验。命令、路径与验证详见 [附录](HANDOFF-DETAILS.md)。

## 1. 目标与约束

最终目标：真实 RL 更新后、读取更新后效用标签前，从策略更新读出判断哪些固定 Skill 的语义边际效用变化，并排序选择优先检查/编辑对象；再检验相同编辑/轨迹预算下的增量价值。

验收不是精准回归 ΔM，也不限定正→负翻转、同一 Skill 跨 seed 同向或每次 Top-1 正确；重点是固定分数在独立窗口/seed 上的整体排名、AP、单调性及多预算收益，相比强基线是否可重复。必须归档配置、权重、轨迹、逐步 Skill 调用及统计原值。

保留 Phase1 与现有三个 seed 的结果，冻结 bank/Router，不按效用标签挑 checkpoint、改分数方向或临时切换赢家。共享 GPU 不得终止他人任务或用虚假占显存；历史“忽略磁盘预算”授权仅适用于已完成的那轮评估，不是删除数据或新训练授权。当前请求仅交接，下一轮实验尚未确认。

## 2. 已确认决策、事实与假设

已确定：Qwen3.5-4B，conda 环境 skill-RL，模型在 /home/wangyifan/model。Phase1 从共同 B0 跑 seeds 101/202/303，固定 U10/20/30，不按性能挑点。clean train 用于 RL、27 个 valid_seen 用于监测、31 个 valid_unseen 用于效用标签。库含 12 通用+32 专用 Skill；clean 候选 18 个，每步由冻结的确定性状态路由器选一个。

主效用 M=E[success_ORIGINAL−success_PLACEBO]，ΔM=M_new−M_old；O−NULL 是次要对照。首次自然调用前固定 prefix，从调用位置自由续跑，后续目标 Skill 再次调用继续同一 payload 干预。PLACEBO 模板/token 长度匹配；NULL 仅置空目标 payload，不删除 ID 重路由，也不关闭全库。

已验证：Phase1 的 S_int 与是否变化关联，跨 seed AUPRC 0.192→0.274，不等于 D 的跨 seed 验证。旧 19 单元 D 的下降 AP=0.799，高于中心化范数的 0.667；测试 7 单元是其子集。新三单元 D/范数均为 0.583，−P/KL/JS 均为 1.000。旧优势不能抹去，但未稳定复现；C_upd 也可匹配部分优势，尚无 D 独特增量。平均名次与单调性见报告第 11 节。

主 D 带 token 门控 C≥0、||δ||≥10⁻⁸，聚合 [-P]₊，分母含零优势/未过门控的 token。无门控版已归档。无 Skill 级 D 截断；0/5 pp 是标签阈值。支持门槛为 20 个非零优势决策、4 个游戏、8 条轨迹；不足弃权，不填零风险。

当前假设：负部聚合、门控和训练状态→评估锚点的分布差异可能影响预测，尚未验证原因。待确认：新窗口、seed、支持范围和预算。已否定/搁置：只做无 RL 冒烟验证、整轨迹技能包归因、性能挑点、扩库、仅用 MAE/Top-1 判成败；不得事后按窗口择优用 D/−P。

## 3. 当前进度与停点

旧 Phase2 完成 U30–35 六端点，各 1,200 条续跑；名字虽含 to36，旧 U36 未完成，不要补跑。新 U35→40 五次更新、单步信号、两端点各 3,600 条续跑均完成，八分片齐全；U36–39 无中间标签。训练、评估及报告于 9 月 14 日完成。

停在报告完善：第 11 节排序/门控与后续 v2 第 5.2–5.4 节 NULL/中途锚点/C 对照均已归档并核验。无待续跑任务；未完成的是跨 seed/独立窗口预测增量、JVP/负对照、Phase3 编辑与成本验证，不是当前评估欠账。

## 4. 文件与工作区

入口：[最新完整报告](2026-09-14-phase2-complete-analysis.md)，先读第 1、5、9、11 节；[设计](2026-08-19-policy-update-skill-sign-flip-forecasting-design.md)的早期 harmful-flip 表述须结合最新约定；[Phase1](2026-09-04-qwen35-clean-all-skill-three-seed-utility-results.md)。

分支 phase1-prep，HEAD 8e66726；19 个已跟踪文件修改、17 个未跟踪条目，无暂存。phase1/、phase2/、tests/phase2/、artifacts/ 均未跟踪；禁止 reset/clean/自动 add。

代码定位：phase2/direction.py、aggregate.py 管信号；evaluate.py 管三臂；ranking.py 管排序。结果根 A/L、权重和轨迹索引见附录。勿重跑 complete_report/finalize_evaluation 覆盖手工修订。

## 5. 验证与边界

本次只读检查 Git、协议、代码、环境、进程、分片数量及哈希：最新 synthesis-v2 报告及 11 个输入哈希通过；两端点各 8×450 行，旧六端点各 1,200 行。v1/首次生成报告哈希与现文不同是正常版本演进，不是损坏。

历史记录为 233 项 CPU 回归通过；本次未重跑测试、模型前向或训练，未重新哈希大权重/全部轨迹。核对与历史启动命令见附录，不将历史测试当本次测试。

## 6. 当前进程与资源

本次未发现匹配项目训练/评估进程；历史 PID 4063983、4088408 均已不存在，完成标记有效。GPU 0/1/5 当时约 13 MiB；2/3/4/6/7 被占用，部分进程无权限核查归属，禁止操作。磁盘约余 264 GiB、97% 使用率；这些只是瞬时快照，下一次必须重查。日志中的磁盘预算失败为已恢复的历史事件。

## 7. 下一步（依序，尚非启动授权）

1. 读报告与附录，做只读 v2 哈希/端点核对。成功标准：文件匹配 v2、U35/U40 各 3,600 行；不重跑报告生成器。
2. 用已有 CSV 复核共同池的整体排名及窗口差异；区分下降/上升与下降/其余 AUROC。成功标准：明确标签、分数版本和支持集，不混池、不覆盖冻结结果。
3. 与用户确认独立窗口/seed、自然 Skill 支持、固定门控/分数及预算。成功标准：新协议与验收规则获确认；已看标签不能冒充新测试。
4. 获授权才另建实验目录，复测恢复、方向对齐、支持和资源；先锁预测再评估并归档。Phase3 同预算编辑仍是后续计划，不能宣称已有效。

## 8. 2026-09-18 追加交接：SkillRL embedding 适配路由

以上第 1–7 节是 9 月 16 日历史快照，不覆盖之后的用户确认。新 setting 为完整冻结 SkillNet-37、policy Qwen3.5-4B、seed=404、GRPO 16×8、lr=1e-6、150 更新上限、每 5 更新验证及固定 5 更新读出窗口。新 preparation 登记八卡 policy。

最新用户确认：复用官方 Qwen3-Embedding-0.6B，保留逐状态全库 top-1；论文注明 SkillRL embedding 的 state-aware adaptation，不称为原生 task-only/整局 top-k 复现。检索器冻结独立于 policy，不新增 router 训练。

当前实际停点：本机 Phase1–2 的训练、performance、anchor 和 O/P/N 工厂已接入 `skillrl_embedding_state`；398 项离线回归通过，两个实权重 CPU 合成状态的调用/缓存/RNG 隔离检查通过。第一次 RNG 隔离检查失败的记录保留。两状态都选中清洗技能，不能据此声称更高选技准确率或覆盖率。新 preparation 静态验收通过但执行锁仍关闭，没有新 RL/真实 ALFWorld 运行，没有付费调用、提交、回滚或推送。旧 mini cohort、账本、报告和 synthesis-v2 结果未替换。

入口：[新 setting、操作说明与完整证据边界](SkillRL/docs/experiments/skillrl-embedding-router-v1/README.md)，[选型决策](2026-09-18-router-selection-skillrl-embedding.md)。新数据准备位于 `SkillRL/docs/experiments/skillrl-embedding-router-v1/preparation-s404-8gpu/manifest.json`；CPU router 不占 policy GPU。不要直接重启旧 mini manifest，不能跨协议复用旧缓存。

后续：核对新的 scoped execution permit、本地调用/存储限额，并验收新 router 与八卡训练/恢复的实际配合，再决定启动。Phase3 编辑器仍为 o3；现有增长库 router 及 GitHub 发布包尚未迁移，四分支开跑前需共同改装 embedding 后端并重建版本化索引。本次的固定 37 库 factory 不能直接用于增长库。没有依据称 Phase3 已采用新 router。

## 9. 2026-09-18 05:25 UTC 追加：Phase3 已发布，新 ALFWorld Phase1–2 已启动

用户已授权“先迁移至 GitHub，再直接开始本机 Phase1–2”。第 8 节的待迁移/待启动是之前停点，不再代表当前状态。Phase3 四分支已接入冻结 0.6B 逐状态 top-1，按增长库 manifest 重建索引/缓存，共享分支级本地额度；o3 编辑器不变。本机不启动 Phase3。

GitHub main 已发布为 `674dd36a54a83c5061af43f0f80b2ee3da924468`，877 个远程文件核验通过，非强制更新且无远程删除。本地研究仓库未提交、回滚或清理。发布包 v4、上传回执、测试范围与命令见 [本次启动记录](2026-09-18-embedding-phase3-publication-and-phase12-launch.md)。

本轮新增八卡合成前后向/优化器/检查点恢复与 CPU router 共存检查 8/8 PASS，未调用 API；这是工程预检，不是 ALFWorld 性能实验。正式 ALFWorld supervisor 于 **05:21:02 UTC** 启动，PID **903045**，新根目录为 `SkillRL/artifacts/phase12/skillnet37-qwen35-embed06-s404-8gpu-v1`。保留旧 mini cohort，不能混用两者的协议/缓存。

05:25:04 UTC 只读快照：初始模型导出完成，真实 ALFWorld 环境和八卡 worker 已初始化，进入首个 rollout 的状态路由，已有 2 个成功本地决策、14 次 cache hit，1 个在途调用；没有 stopped marker，完成更新为 **0/150**。router 使用 CPU、付费调用为 0。启动不是完成，也不保证 150 更新在现有容量内完成。

新 permit 单独绑定原 preparation 哈希，授权 training/evaluation/readout/exports；原 `approved=false` preparation 快照保持不变。760 GiB run cap、100 GiB 空闲线、80 GiB 下一检查点 reserve 继续有效。只清理由已封存窗口证明可再生的本轮全词表临时行；轨迹、batch、报告、检查点及历史证据不动。运行失败保留记录，不自动重跑或扩预算。查看新 root 的 `supervisor.log`、`logs/train-u0000-u0005.log`、`router.sqlite3`、`metrics/`，不要再次执行 new-run-only 启动命令。

## 10. 2026-09-18 06:50 UTC 追加：旧任务已停，15–30h 预算版未准入

用户确认停止第 9 节的旧 150 更新任务并保留证据。06:04:49 UTC 已核实 supervisor/Ray 及所属训练进程退出，八卡释放；旧任务完成更新 0/150、完整 rollout batch=0，757 个本地 router 决策和 265 cache hits 保留。旧 root 的 `user-stop-request-20260918.json`、`user-stop-receipt-20260918.json`、`stopped.json` 为停止证据。不要复用旧 permit 重启。

用户确认预算范围：5 次 RL 更新、固定 U0→U5；U0/U5 均做 140 seen+134 unseen 全量性能；Phase2 只在 unseen，对按预登记 hash 规则选出的最多 12 个自然支持技能、每技能最多 12 anchors、1 evidence+2 gold seeds 做效用验证。路由仍全库 37、逐状态 top1。用户随后给出时间目标“15–30 h”；上限登记 30h，非旧 24h 草案。50 环境步/4096 prompt/512 输出及 GRPO lr=1e-6、16×8 不变，结果只能称单窗口预算受限验证。

本地已接入 CPU 8 线程批量/去重 embedding router、八卡 performance/anchor 分片、逐步 rollout journal、绝对时限及实测准入门槛。32 个旧可见状态 router replay 32/32 选技一致，观察到约 3.37× 组件时长比；不是准确率提升或全流程保证。新的数值协议使用新缓存，不复用旧缓存。最新离线回归 435 PASS。

工程预检实际停点：v1 batch-2 长 prompt 生成 OOM；用户知情后批准 optimizer-state CPU offload + 一次新 v2 预检。v2 8/8 完成第一步更新、原生恢复和长 prompt 生成（15/16 响应到 512），但生成后的第二步 backward 仍 OOM，整体失败，无完整 PASS。两次日志/checkpoint 全部保留。06:46:56 UTC 八卡已释放；**没有新正式 RL**，没有 API 消耗，没有自动重试/提交/回滚/推送。

当前入口：[完整设置、命令与证据边界](SkillRL/docs/experiments/phase12-daybudget-v2/README.md)，[未准入审计](SkillRL/docs/experiments/phase12-daybudget-v2/admission-not-ready-v2.json)，[v2 失败审计](daybudget-gpu-preflight-20260918-v2/failure-audit.json)。v2 preparation 为 `SkillRL/docs/experiments/phase12-daybudget-v2/preparation-s404-8gpu/manifest.json`，仍 approved=false；v1 与 24h 草案保留但不可当执行配置。

建议下一步须先复核：当前 GPU Adam states 在前后向之前回载，考虑改为 optimizer.step 前才回载、step 后卸载；这是待验证假设，尚未实现。修复后需新的连续更新验收，以及真实 batch/全词表记录容量/八卡评估耗时准入。不能因为合成生成通过就开长跑，不能声称已可在 30h 完成。学术实验审计技能的失败复核门槛导致本轮停在此处，等待用户指示。

随后用户追加要求优先采用高速 RL 框架。只读核实当前是 verl/FSDP1/HF，并非 vLLM；本机没有 vLLM/SGLang/FLA/causal-conv1d。官方资料选型优先候选为 verl+FSDP2+vLLM、TP1×8 批量 rollout/评估，详见 [选型追加](SkillRL/docs/experiments/phase12-daybudget-v2/backend-selection-20260918.md)。尚未安装、迁移、追加 GPU 预检；下一步先确认高速后端迁移/验收，不能自动按旧 HF 修复方向继续。无任何最快/30h完成的实测结论。

用户再问能否合理降低rollout数量以提速。已归档 [待确认缩量提案](SkillRL/docs/experiments/phase12-daybudget-v2/rollout-budget-proposal-20260918.md)：推荐训练8×8/轮、5轮窗口不变，完整seen/unseen和12 skills×12 anchors×1 evidence+2 gold不减；更紧备选为anchors12→8。均未实施，active v2仍16×8且NOT_ADMITTED。缩量不保证统计功效，支持不足/效用估计不确定必须如实报告。

## 11. 2026-09-18 07:18 UTC 追加：vLLM 代码迁移完成，等待参数确认／新 GPU 验收

最新用户要求不再使用 HF generation，改为 vLLM；随后明确要求“改完之后给我发一下目前的RL训练参数我确认下”。已完成新后端代码与独立环境安装，**未启动新的 GPU 预检或正式 RL**。旧 HF v1/v2 失败记录、旧 router SQLite 的哈希再次核验未变；不重跑旧任务，不提交、回滚、清理或推送。

新入口：[迁移、当前参数、证据边界和待确认预检命令](SkillRL/docs/experiments/phase12-vllm-v1/README.md)，[实际解析参数](SkillRL/docs/experiments/phase12-vllm-v1/resolved-training-review.json)。preparation 在同目录 `preparation-s404-8gpu/manifest.json`，SHA `e9f895ef159cd2f47db99d06764439193d7cb2ab7b68028e40db02b6ac128078`。没有新的 execution permit，不能使用旧授权开跑。

当前待确认参数仍为 seed404、**5 个 rollout/update 迭代**（不是5个optimizer.step）、16个game×group8、lr1e-6、每5迭代验证/保存、4096prompt/512response/50环境步。用户举例的20–30、50–100updates和group4/8并未改入配置。新文档按实际单窗口范围描述，不再使用用户不希望的“预算受限验证”标签；旧冻结文档保留历史表述，不能据此声称已完成150迭代或多seed稳健性。

实际实现保留 verl/FSDP1 的训练和全词表 readout，**没有切到 FSDP2**；仅 generation 使用 vLLM0.22.0，独立 Python 环境为 `/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918`（torch2.11.0+cu130、transformers5.10.4）。训练侧恢复原生 decoder-layer FSDP wrap；GPU Adam + CPU optimizer-state offload 保留。八卡TP1副本，每卡最大16个并发推理请求、prefill8192、engine memory utilization .45；eager、无prefix cache。生成批量调度数16不是GRPO group大小，不改变每轮128条轨迹。

新V1桥接使用公开 apply_model + in-process external launcher 流式加载FSDP分片；actor更新／恢复后重新同步，连续环境步复用权重；native actor/ref计算及检查点前休眠vLLM。performance、anchor、O/P/N共用新policy工厂；旧协议保留历史路径，新协议报错不会fallback HF。评估现为八卡分片、每卡逐episode执行，尚未实现每卡多环境交错，不得宣称已最大化利用率。

本轮445项离线回归PASS（83.29s），最新后端针对性10项PASS（8.92s，与445范围重叠）；269个安装包依赖检查PASS。静态engine config识别原生Qwen3.5 text/hybrid成功，首次dummy-config探测不兼容已修复。没有加载vLLM policy权重或生成token，离线mock不能证明实卡权重同步、连续更新显存、全词表I/O或全流程时长。下一步先等用户确认参数，再复核新目录的一次8卡预检命令；遵循 academic-research-suite/experiment-agent 的失败实验不自动重试与执行命令确认要求。不得承诺15–30h跑完。

## 12. 2026-09-18 08:07 UTC 追加：用户确认先跑5轮估时，vLLM新预检初始化失败

用户已确认“先按现在的跑然后预估时间”，并提出后续通常希望50–100轮。当前批准范围仍是5次RL迭代、16×8轨迹、固定U0→U5，不是25轮或50/100轮。随后明确取消本次工程预检的30分钟硬上限；这不自动取消正式流程原有30h及存储保护。

新预检在 `vllm-gpu-preflight-20260918-v1` 于07:58:41 UTC启动，约08:02:33 UTC失败（约232秒）。八个rank均在 `ref.init_model()` 报 `Qwen3_5Config` 缺少 `vocab_size`；没有完成双模型初始化，没有生成rollout、synthetic optimizer step、checkpoint或正式ALFWorld RL。当前正式更新仍0，不能用这232秒外推每轮耗时。不是超时，也没有本次OOM证据。

只读CPU配置诊断已确认根因：vLLM初始化在全局AutoConfig中注册自己的Qwen3.5配置类，后加载的native ref取得该类；Transformers只在text_config类精确匹配时自动提取文本配置，因此把外层配置传给纯文本模型。尚未修复。完整证据、退出状态差异、清理说明与建议见 [本次失败结果](vllm-gpu-preflight-20260918-v1/result.md) 和 [失败审计](vllm-gpu-preflight-20260918-v1/failure-audit.json)。run.log SHA=`293b0f2bf576784630f28bff8317b3030e93e0758c4dedf8dd4d299fdbe4cfd9`。

用户取消时限后仅暂停外层timeout，worker继续运行并自然失败；worker退出码均1。确认全部worker退出后只清理暂停的timer wrapper，外层pipeline因此返回137，**不能把137误报为OOM/timeout或worker退出码**。08:07 UTC八卡均2MiB/0%，本次全部启动进程已退出。旧HF两次失败日志和旧router账本哈希未变，没有代码/config修改、API调用、旧实验重跑、文件删除、提交、回滚或推送。

已归档 [5/50/100轮测时与容量外推计划](SkillRL/docs/experiments/phase12-vllm-v1/timing-projection-plan.md)，只登记工作量与所需测量，不假造准入或ETA。下一步按academic-research-suite/experiment-agent失败不自动重试规则，先交用户确认配置隔离修复及另建目录的一次八卡预检；验收和真实吞吐/容量准入通过后才继续已批准的5轮。不得重用v1目录或自动扩到50/100轮。

## 13. 2026-09-18 09:00 UTC 追加：两处兼容修复完成，v2预检在恢复处失败

用户已回复“修复开跑吧”。本轮先实现native Qwen3.5配置隔离（`verl/utils/model.py`及`fsdp_workers.py`），14项CPU/meta回归通过后启动一次新八卡预检。`vllm-gpu-preflight-20260918-v2` 于08:51:42.60 UTC启动、08:54:06.87 UTC自然退出1，无30分钟timer。八个rank都完成双模型初始化、CPU router共存、4096+512概率前向及第一次synthetic optimizer更新，修复跨过旧配置错误。**尚无正式ALFWorld rollout/RL，完成正式迭代仍0**。

新失败是 `native_restore.py` 的旧单根FlatParameter限制，与迁移后按层FSDP布局不兼容，发生在保存后的恢复阶段，不是OOM或超时。原生checkpoint约49.4GiB及32个stage记录全部保留；完整 [v2结果](vllm-gpu-preflight-20260918-v2/result.md) 与 [失败审计](vllm-gpu-preflight-20260918-v2/failure-audit.json)。log SHA=`29052877f1bdc43b870792fa4f7bf04b1bcc488703252a549f23d06adef91995`。不要把每rank一次合成更新当实际训练轮次，也不能据7.58–7.75s的单microbatch更新推算真实每轮耗时。

随后在本次代码修复授权内，保留单根bounded恢复路径，对按层布局改用PyTorch原生strict sharded loader；不改变optimizer/scheduler/RNG文件格式、FSDP布局或RL参数。4项真实双进程CPU/Gloo恢复测试通过，涵盖单根/按层及ShardedTensor/DTensor；最后完整离线回归 **453 PASS / 92.16s**（包含这4项及配置测试）。入口：[恢复修复与验证](vllm-native-restore-repair-20260918-v1/README.md)、[代码/冻结配置审计](vllm-native-restore-repair-20260918-v1/repair-audit.json)。尚未在GPU验收这第二处修复。

当前按academic-research-suite/experiment-agent失败不自动重试规则停在新GPU确认处。已异步询问用户是否允许修复后另建目录再预检，通过后继续既定5轮；本节写入时尚无答复。下一条 [预检命令](vllm-native-restore-repair-20260918-v1/next-preflight.md) 使用未创建的v3目录。不能自行重启v2或把离线测试标成GPU PASS。配置/精确采集/分片评估/实测时间与容量准入仍须逐项有证据，不伪造permit或扩大到50/100轮。

08:58 UTC八卡均2MiB/0%，磁盘空闲约661.1GiB。旧HF两次失败日志、vLLM v1日志及旧router账本哈希未变；冻结preparation/inference/scope profile未变。新增/修改代码仅本轮两处兼容修复及其测试，全部未提交；无删除、回滚、历史重跑、付费API或GitHub更新。

## 14. 2026-09-18 09:40 UTC 追加：v3恢复实卡通过，M-RoPE接口修复中

用户批准第13节的新预检后，v3于09:31:39.991792→09:34:12.970806 UTC运行，152.979s自然退出1。八rank均通过native参数/Adam/RNG精确恢复，旧恢复问题已解决；首次vLLM生成在位置初始化报 `M-RoPE support is not implemented`。这是同步之后的请求位置接口错误，不是恢复或权重拷贝错误，没有生成完成、第二次update或正式RL。详见 [v3结果](vllm-gpu-preflight-20260918-v3/result.md) 和同目录failure-audit.json；40个stage与约49.4GiB新checkpoint均保留。

根因是自定义text adapter没有声明SupportsMRoPE。修复仅补齐无媒体三轴相同位置、delta=0，并拒绝媒体/空prompt，不改RoPE配置或RL参数；真实vLLM原生helper逐值对照测试和完整离线回归归档在 [M-RoPE修复目录](vllm-mrope-repair-20260918-v1/README.md)。此段写入时完整测试尚在运行，不能把它当作PASS。

用户得知本次失败后另行明确批准“允许修复后再预检并继续原定 5 轮”；下一次使用新v4目录，其 [计划](vllm-gpu-preflight-20260918-v4/execution-plan.md) 已登记。5迭代/16×8/seed404、原窗口与router均不变。正式RL仍0，无新permit，其余真实吞吐、全batch及全词表容量准入不可跳过；无自动扩到50/100轮、覆盖报告、删除、提交、回滚或推送。

## 15. 2026-09-18 11:10 UTC 追加：独立三 seed 窗口新队列已启动

第14节为历史停点。vLLM八卡v4完整预检已于09:48 UTC通过（双更新、原生参数/Adam/RNG恢复、生成/重同步）；
后续补齐冻结bank的只读`disabled_skill_ids`接口后，真实ALFWorld八分片16 episodes测时、128状态行精确采集、
32状态批router校准及40行无损压缩校验均完成。没有重复已通过的八卡恢复预检；工程PASS不等于科学结论。

用户最新确认：Phase1–2冻结完整SkillNet-37，**404/505/606各自从共同B0跑U0→U5，依序执行**；
三条合计12h为目标，原30h为硬上限。不是同seed训练到U15，也不是三个seed各30h。
维持每seed5迭代、16×8/迭代、完整seen/unseen性能、unseen最多12skills×12anchors×1 evidence+2gold。
用户提出考虑增加rollout以提高覆盖；综合12h目标，目前没有扩大128/轮或gold数量。
只读复现采样：三seed首轮16/16/15 unique games，均六类、跨seed首批overlap0；五轮79/78/79 games。
但首轮各任务只有1–6 games，≥4games的非零advantage技能支持仍可能不足，不保证37技能都可评估。

新采集协议`window_start_old_only_v1`仅保存第一轮实际batch/状态/advantage/mask和live OLD全FP32词表概率；
其余四轮正常优化但只归档轻量轨迹/game/skill计数、reward/advantage与optimizer日志，跳过多余的NEW全词表前向。
U5仍在同一U0 batch上读出，指标不变，+C排序作为新协议预登记对照；无损shuffle/LZMA每行逐bit往返验证。
中间轨迹不归档不是不训练；O/P/N gold rollout仍必须做来验证预测。Phase3仍使用相邻快照滑动窗口，且保留其编辑所需轨迹。

入口：[新setting/论文附录及命令](SkillRL/docs/experiments/phase12-independent-v4/README.md)，
[代码与666项离线回归审计](endpoint-capture-repair-20260918-v1/repair-audit.json)。
最终离线回归666 PASS /100.40s；旧账本/旧preparation及关键预检日志hash未变。
新prepared-v2中三份资产和cohort.json已冻结；此前prepared目录的离线路径清单错误保留为INCOMPLETE，未运行实验。

新root：`SkillRL/artifacts/phase12/skillnet37-independent-s404-505-606-v4`。
**11:09:54 UTC队列启动，PID1016912；seed404 supervisor PID1016953**。
11:10 UTC状态：U0导出完成，train-u0000-u0005进程初始化；正式完成更新仍0/5，505/606未启动，不能说全流程完成。
30h共同deadline为2026-09-19 17:09:54 UTC。组件条件保守外推约26.18h/seed，不是实测ETA，不能承诺3seed在12h或30h全部完成。
队列只在前seed完整成功且剩余预算足够时启动下一条，以完成seed时长×1.25重估，不读取结果决定seed。
失败不自动重试；时间/磁盘不足记录未开始项。无付费API、Phase3本机执行、新GitHub推送/本地提交/回滚/历史清理。

新报告写入各seed `reports/phase1-results.md`、`phase2-results.md`及CSV，最终共同汇总。
尚无新科学结果，不能把用户摘要中“Our experiments show”当本次已证实。
现有回收仍只删除有独立逐bit再生证明的本轮临时行；没有此证明就保留，不任意删除checkpoint或历史证据。
后续先看queue/seed存活及logs/rollout_progress/optimizer_steps/metrics，而不是再次执行new-run-only命令。

11:14:58 UTC只读追加：404已进入真实ALFWorld rollout，首个step的128条记录已落盘，16个实际game与采样预核对完全一致。
router已有123个决策、197次cache hit，另有在途调用；完成更新仍0/5，无stop marker，505/606尚未开始。
快照见 `endpoint-capture-repair-20260918-v1/running-snapshot-02.json`。这是首步执行/采集证据，不是完整首轮或科学结果。
日志提示Qwen3.5的MFU估算器不受支持、会打印0；该0不是实测GPU利用率，后续报告应标为不可用。

## 16. 2026-09-19：进度文件扫描竞态修复及显式恢复

第15节启动的旧attempt实际运行4114.38秒后，在U1 OLD概率采集阶段因`disk_gate`扫描一个
刚被原子重命名的`.rank-7.json.*`临时进度文件而抛出FileNotFoundError。不是OOM；
128条轨迹已完整采集、5511有效决策、77819动作token、2050行OLD已保存；optimizer更新为0。
用户授权“修复并继续跑”，并明确允许扣除故障停机时间，累计实际运行仍不超过30h。

原queue/seed的`stopped.json`保持原样，不表示新的恢复attempt停止。当前入口：
`C/artifacts/phase12/skillnet37-independent-s404-505-606-v4/recovery-v1/`。
恢复plan=`cohort-recovery.json`，SHA=`af2a94ccd8bc5667643b86b8bde2abc51d4d9b0cdd5cba3be1129633cb774709`。
2026-09-19 05:04:58 UTC已启动一次：queue PID1068291、seed404 supervisor PID1068370、
train PID1068412、Ray TaskRunner PID1070691（后续先查存活，不可盲用旧PID）。
新共同deadline=2026-09-20 09:56:24 UTC，含先前4114.38秒，不是重给30h。
05:06时仍在八卡初始化；没有新完成的RL更新，不能当作Phase1/2科学结果。

恢复复用原始U1日志和动作token；重建5512训练行（原生补齐1行）已逐行匹配2050条OLD的
token、position、rank。后续native前向与每个已有OLD payload逐bit比较，不同即停止，绝不覆盖。
首次128条环境轨迹不重采；只重置一次TextWorld game iterator以恢复下一轮任务位置。
worker/vLLM RNG按登记seed重启，不声称不中断运行的逐bit连续性；首轮遗失的采样logprob诊断
不伪造，GRPO仍用native OLD。后续start batch增加pre-forward持久化。
此边界会写入新结果报告；原失败日志、router账本快照和旧forward进度均保留。

18项针对性/673项完整离线回归通过（100.47s）；测试文件位于recovery-v1。
具体修复/恢复命令/证据边界：`C/docs/experiments/phase12-independent-v4/RECOVERY-20260919.md`
与HANDOFF-DETAILS第N节。不要重跑预检或旧采样；任何新失败停止并征求用户方向，不自动retry。

05:08:41 UTC追加验收：8rank均进入U1 OLD实际前向，各17/689微批；136个原OLD payload均与
恢复前向逐bit相同，原概率文件未覆盖。`pre_forward_batches/u0001.pt`已持久化，源码95个hash
仍匹配，原失败日志SHA不变，无新stop marker，optimizer更新仍0。见A/running-snapshot-01.json。
这证明已恢复执行和部分现场一致性，不是5轮完成，也不代表已越过原256微批的失败位置。

## 17. 2026-09-19 06:35 UTC：bool 兼容修复、完整 OLD 复用与 recovery-v2

第16节的 recovery-v1 于05:54 UTC在 `apply_invalid_action_penalty` 对 Python bool 调用
`.astype()` 时停止；OLD 5512行和reference前向已完成，**仍为0个真实optimizer更新**。
用户明确要求继续修复重启并加强监控。本轮修复动作有效性标量兼容及episode指标的标量提取，
不改变惩罚、GRPO、RL参数、bank、router、支持阈值或404/505/606顺序。

全部5512个OLD文件解压并核验行/rank/token/FP32 trainer-chosen，产生紧凑恢复cache；
完整真实batch（128轨迹、16组、77834 scored tokens，含原生补齐的一行）通过CPU reward、
与原惩罚逐值对照、GRPO advantage及metrics检查。692项离线回归PASS/101.67s，
含native optimizer接口连续/尾部不满mini-batch测试；不是科学结果或额外GPU实验。

新attempt：`C/artifacts/phase12/skillnet37-independent-s404-505-606-v4/recovery-v2`。
plan SHA=`d86afcea58f00e8457e1fae3820d3a0780ce1ca227a80180c5466d744bf70a6c`，冻结99个runtime源码。
06:35:27.983 UTC队列PID1107501启动，seed404 supervisor PID1107636；先查存活，不能盲重启。
新共同deadline=2026-09-20 10:37:35.261 UTC，计入此前7072.7218秒累计运行，
扣除停机而不重新给30h。06:35记录时仍初始化、正式更新0/5、505/606未启动。

U1不重采；每rank一个原状态的native OLD witness必须精确匹配，全部有效cache值必须能
无损恢复实际native dtype。首轮OLD entropy日志未持久化故明确缺失，optimizer内entropy正则
不变。reference输出未持久化，必须重算，不从OLD推断；今后每次完成chosen OLD/reference
都立即另存。旧压缩全词表/旧失败日志/旧stop markers不改、不删；v1 RNG恢复边界仍适用。

监督器每30秒记录 `recovery-v2/seed-404/runtime-status.json` 和append-only
`runtime-history.jsonl`，包括GPU、rank前向、Adam计数、迭代metrics、磁盘、预算、退出。
5分钟无进度仅告警，不误杀或自动retry；本轮人工验收到首个完整U1。新失败保全后停止。
方法、命令与证据边界见 [v2恢复说明](SkillRL/docs/experiments/phase12-independent-v4/RECOVERY-20260919-v2.md)
及HANDOFF-DETAILS第O节。所有未提交修改保留，无GitHub推送/提交/回滚/历史重跑。

07:21 UTC追加：用户明确要求“感觉后面没有问题了就停止监控吧”，因此结束本轮agent主动盯守，
**不停止训练，不关闭后台30秒状态监控或预算保护**；此前人工盯到完整U1的计划由此更新。
最后只读验收：八rank均已真实完成24/44个U1 Adam小批次，计数连续、所有已记录梯度有限，
99个源码哈希仍匹配；实际5512行训练batch的OLD/reference/mask精确对齐审计PASS。
GPU利用率72%–87%，显存27710–27936MiB，空闲约515.4GiB；无新stop marker。
**完整RL迭代仍0/5，U1正在进行，505/606未启动，科学指标尚未验证**。
交接证据为v2 `active-monitor-handoff.json` 与 `actual-native-batch-audit.json`。
以后用户询问状态再只读查看v2监控文件，不能把本段当作第一轮或全流程已完成。

## 18. 2026-09-20 02:32 UTC：U5导出修复完成，八卡正式评估已接续

第17节任务后来完成全部5轮RL、640条训练轨迹、每rank204次连续Adam更新，
于9月19日16:15 UTC在U5导出子进程报`ModuleNotFoundError: verl`后停止。
用户本轮明确要求分析错误并继续运行。根因是脚本路径调用把scripts/而非仓库根放入
导入路径；已改为`python -B -m scripts.model_merger`，无需重装环境或重跑训练。

新attempt：`C/artifacts/phase12/skillnet37-independent-s404-505-606-v4/recovery-v3`（记为A3）。
699项CPU回归PASS；完整U5原生检查点53,034,116,482bytes已保全；358文件hash留档。
新plan SHA=`cdacaaedfc0c01477b8f76cdcba4c6f827a5b2f8db27c10f98a35eb877fed507`，102源码冻结。
seed404恢复permit只有evaluation/readout/exports，无training，直接从U5导出接续。
不重采训练轨迹、不新增Adam更新、不覆盖旧导出暂存/旧报告，不改RL参数、bank或router。

02:21:46.443 UTC队列PID1173697启动，seed404 supervisor PID1174098。
此前累计41,864.653575秒已全部计入；新共同deadline为**2026-09-20 20:44:01.789 UTC**。
旧v2 deadline已被这次显式恢复的累计预算替代，不能拿旧stop marker误判当前状态。
02:26:16 U5成功导出：427个FP32张量与8rank原生分片逐bit相等且有限。
02:32:45 U0正式seen评估59/140条完成，八个分片均已有有效完整episode；
八卡各约15217MiB，无新stop。unseen、Phase2及505/606尚未开始，科学结论仍未验证。

额外边界：旧U0保存为BF16但元数据标FP32，24个norm相对FP32 master有舍入。
**FP32初始化权重比较FAIL，不能声称U0是无损FP32快照**；登记BF16运行dtype下，
全426项权重值及24个norm实际vLLM加载函数CPU核验PASS，不改变当前BF16实验。
两份结果均保留在A3的`u0-native-initialization-audit.json`、`u0-runtime-dtype-audit.json`。
不是全模型GPU前向等价证明；后续读出fidelity门槛照常生效，原OLD概率FP32归档不变。

agent已按用户此前要求结束主动盯守；任务继续运行，后台30秒状态记录、退出/预算保护不关。
实际交接快照：A3/`active-monitor-handoff.json`；只读状态入口：A3/seed-404/`runtime-status.json`。
新失败不自动重试。详情见[v3恢复说明](SkillRL/docs/experiments/phase12-independent-v4/RECOVERY-20260920-v3.md)
及附录P节。全部未提交修改保留，无GitHub推送、提交、回滚或历史实验重跑。

## 19. 2026-09-20 09:18 UTC：窗口汇总修复、取消时间上限、recovery-v4 接续

第18节的A3后来完成U0 seen140、unseen134、unseen anchors134、U0效用540及八个读出分片，
在汇总时报`Missing new live logit rows: 0/5512`停止。不是训练失败，5轮RL/U5导出仍完整。
用户明确要求修复后继续评估、顺序跑505/606；又明确取消三个seed累计运行时间预算。
**原12h目标和30h截止不再约束本次队列，磁盘/完整性/异常退出保护不变**。旧plan/profile不覆盖。

修复OLD-only窗口汇总的过时live NEW要求，保留单步模式的OLD+NEW校验。
真实八分片逐项核对PASS：5512起点batch行、2867去重候选决策、82362条control-token记录。
参数范数诊断以登记原模型按原生actor配方CPU重建FP32 B0，对比原U5；不冒用BF16 U0导出，
不改变C/P/D/支持阈值/rollout/RL参数。1082个已完成产物hash及router停机快照保全。
709项CPU回归PASS/116.38s；独立目录实际分片汇总→预测锁定→检查点绑定测试PASS，无U5 gold。
这只说明工程验收通过，不构成科学结论，材料状态UNVERIFIED。

新入口A4=`C/artifacts/phase12/skillnet37-independent-s404-505-606-v4/recovery-v4`。
plan SHA=`1f15f784f6b2df7ca25e64e59f91395e3da86a9a5c0662301dcbc4217e525970`，105个runtime源码冻结。
09:17:56.873 UTC新队列PID1201615启动，`deadline_unix=null`；此前48606.01990866661秒
仍记账，不再按剩余时间拦截505/606。seed404只有evaluation/readout权限，无training/exports。
从汇总接续，不重跑原5轮、U0评估、anchors、540条效用或8个读出分片。
旧窗口authorization_path不改，新授权只在原路径/SHA/preparation/run root匹配时供评估/读出使用。

此段记录时尚在启动核验，不把队列启动当成U5评估已完成。状态看A4而非旧stop marker；
seed404状态在A4/seed-404/runtime-status.json，505/606正常阶段状态在各自seed根目录，
队列与其supervisor日志在A4。后续以active-monitor-handoff及追加状态为准。新失败不自动重试。
详见[v4恢复说明](SkillRL/docs/experiments/phase12-independent-v4/RECOVERY-20260920-v4.md)
及附录Q节。所有未提交修改、旧失败日志、旧报告与检查点保留，未提交/回滚/推送。

09:24:50 UTC验收追加：09:21:11正式窗口汇总完成，09:21:16在U5 gold打开前锁定预测。
随后八卡U5效用评估已完成28/540条（各分片5/5/4/2/3/3/2/4），身份、原prefix回放及完整steps检查PASS。
正式两张信号parquet与独立CPU联调结果逐文件SHA相同；105源码hash仍匹配，旧失败日志SHA未变。
队列PID1201615、seed404 supervisor PID1201814存活，无新stop，deadline=null。
U5全量性能和最终报告尚未完成，505/606尚未启动；队列完成404后顺序接续，磁盘保护仍可阻止准入。
agent主动盯守到此结束，后台30秒状态记录/异常退出/磁盘保护继续运行。现场完整快照见
A4/active-monitor-handoff.json；不能把此状态当作seed404或三seed全流程完成。

## 20. 2026-09-20 16:10 UTC：全部首调用锚点评估已登记，等待505 RL后补评404

第19节旧队列已完成404全流程，505正在第二轮训练（落盘1/5），606尚未启动。
用户要求取消评估数量筛选、重算404并替换现有分析报告，但等505 RL结束，不中途打断。
最新最终口径是**每条轨迹每个skill只取第一次调用，所有首调用锚点全评估**；
此前短暂讨论的每次调用锚点方案没有运行。后续调用只记频次，不新增锚点。

取消自然支持30轨迹/10game、技能/锚点上限以及读出20非零决策/4game/8轨迹数量门槛。
不取消模型/数据/完整性/replay校验、有限性或D的数学gate；原C/P/D公式和训练batch决策不变。
源范围仍为登记U0 unseen134条轨迹。404共25技能、5558次调用、404首调用锚点；
U0/U5×O/P/N×3续跑seed=7272条，其中1080旧续跑严格核验复用，只新增6192条。
25技能全部评效用；18种有真实U0训练读出输入，其余7种读出NA，不是按支持数量筛掉。

为不破坏旧队列105源码冻结，新加7个`first_calls_*.py`独立入口，共112源码冻结。
Q=`C/artifacts/phase12/skillnet37-independent-s404-505-606-v4`；
F=`Q/all-first-calls-v1`，plan SHA=`e3521e2e66a02a5cc006905e8432d35e52a8c5b05b68b8610b8ea14e7cadfeb9`。
专项初检31项、完整740项、最终定向32项CPU检查PASS。schema/授权/原seal/语义/磁盘准入实际PASS。
完整回归首次1项失败是旧CPU测试的GPU日志接口，后用单测试日志隔离插件验证，未改训练运行时。

deferred supervisor PID1259771已在后台等待，状态`F/deferred-status.json=WAITING_SEED505_RL`。
原queue1201615/505supervisor1207419/training1207476存活，未暂停、未中断、未启动新GPU评估。
新调度先确认505训练成功退出、5份metrics和8rank U5 checkpoint，再只暂缓两个调度进程；
已启动的导出/评估子进程必须自然结束。空卡后优先补评404，成功才替换其reports，失败不retry。
不论新404成功或异常均恢复旧调度；原队列完整结束后，同规则补齐505/606并更新cohort汇总。
中间404新/505606旧口径的临时报告不能当作同协议跨seed结果。

旧7份404报告已按SHA归档于F/archived-reports/seed-404，当前原件**尚未替换**。
授权替换仅限当前cohort的reports视图；历史窗口seal/轨迹/OLD/模型/训练记录不改不删，
不覆盖历史101/202/303分析文档。seed404明确为旧标签已可见后的覆盖扩展，科学状态UNVERIFIED。
paired game/continuation统计；单game保留点估计、跨game CI为NA。相同训练batch不改为首调用读出。

无时间上限，磁盘仍为100GiB最低空闲、80GiB预留、760GiBcohort上限；新404按36.40625GiB准入。
此刻约404GiB可用。router保持冻结0.6B embedding逐状态top-1，无外部API。
详细设置及人工异常交接说明见
[首调用覆盖修订](SkillRL/docs/experiments/phase12-independent-v4/ALL-FIRST-CALLS-20260920-v1.md)
及附录R。用户未提交改动全部保留，无commit/rollback/push。Agent主动盯守结束，后台等待与保护继续。

## 21. 2026-09-21：首调用补评竞态修复，显式恢复；606另被磁盘准入阻挡

第20节后来运行的404补评在00:51 UTC因扫描过程中SQLite正常移除
`router.sqlite3-journal`而失败，旧调度已自动恢复。读出八分片/汇总完成，
U0完成1871/3636（540复用+1331新增），U5尚未开始，报告尚未替换。
用户本轮明确授权修复并复用结果恢复。所有失败证据及未提交修改保留。

核查期间505旧协议全流程于约03:59 UTC自然完成：5轮RL、U5导出、两端seen140/
unseen134、anchors134和旧效用均完整。旧队列未启动606，在磁盘准入处退出，
`A4/queue_finished.json`为budget_stop、completed=[404,505]；不是已取消的时间上限。
606需约356.859GiB空闲（176.859新增估计+80预留+100底线），现约332.1GiB，差24.8GiB。
不得把旧队列写成三seed完成，也不能擅自删除证据/放宽保护启动606。

新增独立`first_calls_storage.py`与`first_calls_recovery.py`，原105/112源码hash均保持。
SQLite临时sidecar竞态窄白名单处理，永久数据库/证据缺失及权限错误仍报错；
容量门槛不改。恢复校验每条既有续跑、SHA和索引原前缀，仅补缺失工作；
已完成读出、RL、导出、性能评估均不重跑。完整离线回归786 PASS/156.93s。

恢复目录A=`F/recovery-v1`，新plan SHA
`5847e38b1ab7e94a987a50a7fd943390afdeb4d614558a8df89043196a7d5dfd`，114源码、1989保留文件绑定。
04:14 UTC显式启动一次，PID1295037；04:15:51八个U0效用分片已启动，
此启动快照尚未出现新增episode，后续进展以A状态/验收记录为准。
先补404剩余1765条U0，然后锁定评分、补U5，再封存/发布；随后同口径补505。
606因原seed尚未准入保持pending，不伪造全队列完成，不在补评入口偷偷启动RL。
三seed都完成前不发布新的跨seed最终报告。无自动重试，无旧进程信号，无API。

入口：[竞态修复、执行命令及证据边界](SkillRL/docs/experiments/phase12-independent-v4/ALL-FIRST-CALLS-RECOVERY-20260921-v1.md)、
附录S；状态A/deferred-status.json、A/seed-404/runtime-status.json、A/recovery.log。
原F/deferred-status.json=STOPPED属于历史失败，不能当作本次恢复状态。
新增日志在A/seed-N/logs，产物仍在各首调用variant。未提交、回滚、推送或删除任何证据。

**04:18 UTC验收更正：A也已停止，当前不是运行中。** 04:16:56成功新增1条续跑，
U0为1872/3636；04:17:03 shard5遇到旧本地router未完成reservation，
报`Prior failed/incomplete local query; no automatic retry`，恢复器04:17:07退出，八卡空闲。
SQLite本身完整，30851条成功缓存无损；5条started无结果，4条来自旧00:51停机、1条来自本次停止。
磁盘扫描竞态没有复发，但初次恢复准入未检查孤立reservation这一缺口，不能声称已稳定恢复。
已保留新失败日志和所有新增/旧产物，未改缓存；114源码与1989原文件/索引前缀再次PASS。
U5和505新补评都未开始，旧报告仍未换。现场见A/active-monitor-handoff.json。
按academic-research-suite失败不自动重试规则，已询问用户是否允许保留原账本/成功缓存、
仅对这5条未完成本地选技登记一次续算许可，再建新attempt继续404/505；本轮未执行第二次重启。

## 22. 2026-09-21 04:44 UTC：获批的5条本地选技全部续算成功，404补评已继续

用户已明确确认“允许修复这5条未完成查询后继续”，第21节的待确认已解除。
新增`explicit_router_resume.py`和`first_calls_router_recovery.py`独立入口；
保留旧5条started行、原成功decision及cache-hit账本，只append各一次带授权来源的新attempt/decision。
其他中断仍拒绝；GPU启动前新增缓存完整性/原账本/白名单核对，付费API/default router保护不改。
原105/112/114源码均未热改，新plan共116源码。专项36 PASS，完整817 PASS/154.81s。

新恢复A2=`F/recovery-v2`，plan SHA=`18eaaddd388dff9a2028be797f13834d700d97cb5e557b1e81daaf7e62fc54fe`，
授权SHA=`4e04becb9c8f712ddc813ae0d62351a4f862bc5951b54ce5e27e415b956dd56c`。
完整原SQLite306270208bytes已按字节备份，绑定2009保留文件及索引前缀。
04:40:08显式启动PID1302727；04:41:25八个U0效用分片启动，未重跑RL/readout/完整性能。

04:44:17实际验收：全部5条各成功续算一次，原账本四张表的旧行改变数均0；
U0从1872增至1895/3636，八片各新增3/3/2/3/3/2/4/3，身份/replay/完整steps通过。
2009保留文件和116源码SHA仍匹配，旧报告尚未换，无新stopped，进程存活。
状态以A2/deferred-status.json、A2/seed-404/runtime-status.json及A2/active-monitor-handoff.json为准，
F与recovery-v1中的STOPPED是历史失败，不能误读成当前任务停止。

后台顺序404补齐U0→U5→封存/发布，再同口径补505。606仍未启动，原磁盘保护保持；
备份后空闲约331.86GiB，相对原356.86GiB准入仍缺约25GiB，未删除数据或放宽门槛。
Agent按此前要求结束稳定后的主动盯守，后台遥测/错误与磁盘保护继续。无API、Git提交/回滚/push。
详细说明：[五条本地选技显式续算](SkillRL/docs/experiments/phase12-independent-v4/EXPLICIT-ROUTER-RESUME-20260921-v2.md)、附录T。

## 23. 2026-09-21：404全首调用报告已发布，505端口冲突修复后显式接续

第22节A2先完成404全部补评：U0/U5各3636条、25种自然出现技能、404个首次调用锚点；
10:35 UTC新报告生成并封存，7份正式报告视图随后更新，旧7份报告归档完整。
404的Phase1/2报告入口仍为Q/seed-404/reports；新口径原产物在F/seed-404。
其中18技能有真实训练readout，7技能NA；全部37技能在coverage明细保留，无数量门槛筛选。

A2随后完成505新readout八分片及汇总、复用540条U0，但11:04:43 UTC因shard7
启动vLLM时TCPStore端口40669被占用而停止，未产生新的505效用轨迹。八卡空闲；
505的旧5轮RL/导出/完整性能结果均完整，不能重跑。606仍未因磁盘准入启动。

用户明确要求修复并续跑505。新增`vllm_file_executor.py`和`first_calls_port_recovery.py`，
不改原116冻结源码或安装依赖，只为单机单rank评估用继承UniProcExecutor的FileStore
rendezvous替代“探测空闲TCP端口后释放、再bind”的竞态。每引擎独立/tmp目录，
不复用旧路径；worker/model/generate/sampling保持。未识别占用40669的进程，不宣称占用者是谁。
专项42 PASS、完整859 PASS/172.97s；真实八CPU进程文件初始化和新旧生成参数一致性通过。

A3=`F/recovery-v3`，plan SHA=`9d319b2c7e9a4fb2d2025f2ec1537f272544b152dee6ecfcc609e959f92b2e16`。
准备绑定118源码、6份安装依赖源码、8081保留文件及索引原前缀，505空缓存检查PASS。
原404完整seal与已发布报告也绑定只读。空闲326.699GiB、cohort231.005GiB，
此次补评预留116.406GiB满足原保护；606更大的RL准入门槛未改、未绕过。

恢复进程PID1325542已显式启动一次，先做启动校验，随后只补505 U0缺失3096条，
锁定现有readout后复用U5旧540条并补齐，统计/封存通过才更新505报告。
不重新评估404、不重复读出或RL、不启动606、不发布三seed完成汇总，无自动重试/API。
这是启动记录，真实八卡生成验收另见后续追加和A3/active-monitor-handoff.json。
所有旧失败和未提交修改保留，无Git提交/回滚/推送、无证据删除。
详见[505端口恢复说明](SkillRL/docs/experiments/phase12-independent-v4/SEED505-PORT-RECOVERY-20260921-v3.md)、附录U。

11:41 UTC实际验收追加：505八卡均通过FileStore/NCCL初始化和生成，8个独立URI及
ENGINE_READY receipt核对PASS。U0已563/3636，比原540新增23条，八片分别新增
2/2/3/3/3/3/3/4条；新轨迹身份、continuation seed、原锚点/prefix replay、完整steps均通过。
118源码、8081保留文件及原索引前缀再次核验PASS，404 seal与报告未改。PID1325542及
八个评估器/引擎均存活，无A3/stopped；每卡约15217MiB，磁盘326.64GiB。
505 U5和新报告尚未开始，606仍未启动。Agent主动检查结束，后台按既定顺序及保护继续。
完整现场记录见A3/active-monitor-handoff.json，不能把此验收当作505全流程或三seed完成。

## 24. 2026-09-21 11:57 UTC：用户要求暂停505，待C/P/D数值一致性讨论

本节取代第23节末的“后台继续”运行状态。用户明确要求先暂停505评估，
讨论P的中心化不变性、FP32数值误差及D门控；没有授权本轮修改指标或自动恢复。
11:57:08 UTC对已核验PID/start_ticks/command SHA的supervisor1325542先发SIGSTOP，
随后挂起八个评估器与八个vLLM引擎，共17个自有进程；没有SIGTERM/SIGKILL。
11:57:49与11:58:50复核全部为T状态、八份索引SHA和大小不再变化。

505新口径U0已记录810/3636条（原540+新增270），八片103/102/99/101/101/99/103/102；
U5新口径尚未开始导入/补评，旧协议U5证据仍保留。404完成报告不动，606不启动。
八卡利用率0%，每卡15217MiB显存仍由暂停现场占用；这不是退出或释放显存。
新现场记录A3/pause-20260921T115708Z.json及pause-verification-20260921T115708Z.json；
原active-monitor-handoff、runtime-status/deferred-status仍是暂停前快照，不得据此自动续跑。

本轮只读确认direction.py的exp/差分/乘积沿用输入精度，float64仅指定在求和处；
这不是端到端FP64。P在归一化概率与零和方向下理论中心化不变，不能预设修正会改善性能。
用户引用的27个门控变化/排序不变及16-token数值检查本轮未重算，完整稳定版P/D尚未验证。
建议先做等价数值修正与恒等式测试，再在独立输出中比较旧/新readout；不改阈值、有效性
定义、聚合权重或候选池，不重跑RL/环境效用。该建议尚待确认，未实施。
所有代码、已发布报告、轨迹和缓存未由本轮修改；仅新增暂停记录与追加交接文档。
暂停中的正常in-flight router查询不能当作失败预约续算，恢复前须重新核验进程身份及
暂停时长对遥测/等待的影响。等待用户指示，不自动恢复，也不擅自终止以释放GPU。

## 25. 2026-09-21：获批数值稳定性修正＋原版对照，404先于505

用户已解除第24节的“修改待确认”，明确要求先修正并重算404，完成后再启动505及其修正。
新增独立stable_direction与numerical_readout/数值报告/顺序执行器，原118源码SHA核对未变。
完整归档规范见[数值修正登记](SkillRL/docs/experiments/phase12-independent-v4/NUMERICAL-READOUT-20260921-v1.md)。
固定三版：legacy_recorded、stable_raw、stable_centered_gate。新增D_centered_contribution；
两版稳定D使用同一P，centered P只作恒等式核验。概率方向规范化、前置运算/稳定零和投影
采用FP64；模型前向、epsilon/tau、有效性数学定义、候选池、token均值权重保持。
不将“中心化”预设为性能改善，不按404已知标签选最优版本。

新N=`Q/numerical-readout-v1`；只写新目录。404及505原报告、轨迹、RL、模型和缓存均保留。
404八逻辑读出分片四worker两GPU一组，分两波跑；每worker U0/U5分卡、模型前向仍原BF16，
FP64临时向量按16 token分块且不截词表。先算最长动作检验内存，落盘恢复原行序。
每个decision同次前向重算旧实现，必须与原版标量逐值一致；稳定P中心化误差须满足固定阈值。
最终报告复用原效用标签/区间，列出raw/centered C/P/D、AP/AUROC/预算及完整支持，不重跑环境。

执行器顺序固定：404重算/汇总/新报告SHA验收→505同规则重算/汇总锁定→仅对原17进程
定向SIGCONT恢复505效用→等旧505全部补齐→生成505数值对照报告。任何前置失败都不提前
恢复505，不自动重试，也不新增router续算授权。606仍未启动、原磁盘保护未豁免。
目前此段是实现与待启动登记；实际最终测试、plan SHA及GPU启动验收在后续追加。
旧505目前仍17进程T、U0索引810/3636；不能从新源码存在推断404已重算完成或505已恢复。

### 25.1 2026-09-21 12:44 UTC：首次完整重算因原版对照显存不足停止

已实际准备并显式启动N，不是仅登记。plan SHA
`ee6ec2cdee6333ed8ecf7589367b94f58a2e66ae808a097d6c660cfeef02dfa0`，
绑定122源码/111输入文件；完整898项CPU回归通过，但不能替代512-token GPU显存验收。
launch记录12:36:29 UTC、PID1342117；首波四worker分别使用GPU(0,1)/(2,3)/(4,5)/(6,7)。
12:39:55 UTC调度器已STOPPED，不是当前运行中。shard1/2/3在原版比较器
`phase2/direction.py:41`处理512-token完整词表时OOM：申请970MiB、当时仅约839MiB空闲。
暂停505每卡仍占15217MiB；新模型加载/前向通过，但原版比较器没有随FP64修正版分块，
此前每卡15,000MiB空闲准入不足以覆盖该峰值。此为实现/准入缺口，不是训练失败。
shard0日志19个短动作通过逐值legacy与P恒等式校验；首个row1008仅18 token，
不能把它当作512-token通过。无完整分片落盘、无新报告、505数值重算未开始。

新调度器及四worker已退出；旧505的17进程仍T、八份索引SHA不变，仍810/3636，
没有SIGCONT意图/完成记录。122源码SHA再次匹配，原模型/效用/报告未改，磁盘约327GiB空闲。
N/stopped.json、全部日志与attempt记录保留，详见N/active-monitor-handoff.json。
按academic-research-suite失败实验复核规则不自动重试。拟修复原版对照逐token分块、
保持原标量逐值一致性门禁，在新目录先验收512-token再全量；等待用户确认后执行。

## 26. 2026-09-21：用户授权释放505内存现场，再404→505新进程接续

用户最新明确要求“先不要保存505的状态，release；404对应重算跑完再重启505评估”。
这替代第25节的挂起/SIGCONT方案，不是删除磁盘实验数据。新增release_paused_505入口，
核验原17个PID/start_ticks/command SHA及唯一GPU所有权后，用pidfd定向SIGKILL，不SIGCONT。
释放已完成，旧505原17进程均退出，八卡显存各约2MiB；原索引810/3636与数据库SHA不变。
原内存现场不可恢复，磁盘证据均保留，另有释放前数据库字节副本。释放专项8 PASS。

新N2=`Q/numerical-readout-release-v1`；release-intent.json/release-505.json记录具体授权与PID。
505全部810条已落盘轨迹通过身份/分片/replay/anchor检查，无孤立JSON；
5831次router attempt中5826条成功，另5条在11:57:08暂停时未完成。
仅对本次5个精确ID5826/5828/5829/5830/5831登记各一次本地续算，原行和成功缓存不改；
不是复用seed404旧5条授权。无API、不重跑已完成轨迹或RL。

新增numerical_released_run/released_505_evaluation，原122源码与C/P/D保持；
每卡最低空闲提高为28000MiB，四双卡worker、八逻辑分片仍分两波。
顺序：404三版完整报告/SHA验收→505同规则读出commit→新进程补齐505 U0/U5效用→505三版报告。
不唤醒死PID，不调用旧publish覆盖训练目录旧报告，不启动606或豁免磁盘保护。
详细登记见[释放后恢复规范](SkillRL/docs/experiments/phase12-independent-v4/NUMERICAL-READOUT-RELEASE505-20260921-v1.md)。
111项综合专项通过，完整回归与实际启动验收在后续追加；此段不代表404已经开跑或完成。

### 26.1 2026-09-21 13:15 UTC：404完整重算已启动，512-token显存验收通过

最终完整回归935 PASS/173.19s、16条既有告警；原122源码不变，新plan绑定125源码与118输入。
N2/plan.json SHA=`4edd50b1afcbeb829a8e98721e8ab440e7ffdfd6ca9af713df16d5ef064380cf`。
磁盘准入free326.418GiB、保留88GiB；无时间上限，原容量保护未改。
实际launch 13:11:47.626301 UTC，supervisor PID1348866；四worker PID1349405–1349408。
启动前哈希/磁盘检查耗时数分钟，不是新失败；后台进程随后已进入真实GPU评分。

13:15:29 UTC首波四分片分别完成22/11/10/11个decision（各689个），均legacy_exact=True，
P中心化日志最大误差8.88e-16。shard1/2/3首个row17/1378/339均为每control512 token，
三者已完整通过；shard0首个row1008为18 token。此次不得再误称只验收了短动作。
新stopped不存在，125源码SHA再次PASS，505原17进程均已退出、原8索引SHA不变（810/3636），
没有seed505-restart-intent，505数值或效用均未提前启动。显存观测最大约20197MiB，未再OOM。

当前以N2/runtime-status.json、workflow.log、logs及active-monitor-handoff.json为准。
404报告尚未完成；完成并验收后，后台先锁定505同规则读出，再新进程补505效用并生成三版报告。
本轮到启动验收为止，不把部分评分通过当作完整统计结果或性能改善；不继续人工常驻监控。

## 27. 2026-09-22：seed404 的 reward-directed 变式扩展已完成（CPU、事后探索）

用户授权在404扩展所有合理reward相关D候选，不要求依赖P，并扩充当前分析报告。
新独立入口`reward_variants.py`、`reward_variant_statistics.py`、`reward_variant_analysis.py`；
不修改原125份冻结运行源码、原C/P/D、RL参数、skill/router或Phase3设置。

新根V=`Q/reward-variants-s404-v1`，plan SHA
`7a767ce6a82133c5a4b333f3fecda73c0404823cb022ab338a365f4a76793b6b`。
39 reward公式×3聚合=117，另117去奖励版本+21幅度+6旧参考，共261列；候选及方向先锁定，
但404/505旧结果已见，不能当作前瞻预登记。只在404运行，未扩到505/606，不新跑RL/模型前向/rollout/API。

283项CPU回归PASS/16.96s，含合成端到端、原指标、配对重采样、符号块负对照及no-clobber；
3次开发合成测试失败记录全部保留。实际分析于07:41:37左右→07:42:19 UTC一次成功、约42秒。
5,220行分层指标、19,836行预算、2,000次配对bootstrap及512次轨迹块符号随机化已生成。
独立sklearn/scipy核对5,220行PASS，9种原指标各20行复现，最大误差3.33e-16。
原19输入与125源码SHA均保持。完整结果和独立验收见V/complete.json、provenance.json、independent-verification.json。

主PLACEBO池18技能：5下降/7上升/6不变点估计。原D AP=.509650；token signed gated D=.415995，
去截断没有改善。最高reward AP是centered C/decision=.693590，但centered magnitude/game=.724762、
原activation-l8=.743333更高。C/decision对同聚合centered magnitude增益+.051368，配对标签CI[-.187348,.166667]。
最佳若干带符号变式为9/12方向正确；不能将117次候选择优当泛化或reward因果增益。Phase3主指标未改。

当前综合入口[phase2-complete-analysis.md](phase2-complete-analysis.md)只追加第10节，原29,970字节完整保留并备份；
新增[404变式解释性分析](2026-09-22-seed404-reward-readout-variants-analysis.md)，以及V/reports/phase2-results-expanded.md。
旧数值报告保持原hash，未重跑/覆盖旧报告生成器。详见附录Y与
[本次登记](SkillRL/docs/experiments/phase12-independent-v4/REWARD-VARIANTS-SEED404-20260922-v1.md)。
本轮已完成，不需启动后台恢复；后续是否冻结候选再迁移其他seed由用户确认。未提交、回滚、推送或删除任何历史证据。

### 27.1 最终采用 v2：严格常数聚合的数值修正（2026-09-22）

v1运行成功后负对照审计发现：单位A的reward-only分数理论恒为−1，但浮点加权聚合出现末位差异导致伪排序。
保留v1全部封存源码/输出/初版解释稿，另以独立适配器`reward_variant_constant_fix.py`修正严格常数列，
所有非恒定列和39公式/117 reward候选不变。V2=`Q/reward-variants-s404-v2`，plan SHA
`ae5e7074a4736a0b8dede198dbd6a313d89c03b6410d26849a8de3ca67316136`。

287项CPU回归PASS/14.22s；实际v2于08:05 UTC约43秒一次完成，5220行独立验收PASS，9个原指标复现。
共纠正366个标量单元，主PLACEBO/all/0pp的117 reward分数AP均不变；三种常数unsigned基线
AP均恢复5/18=.277778、AUROC=.5、Spearman=NA。轨迹块符号family-max比例修正为77.734375%，
不改变此前“尚不能确认稳健reward增量”的结论。原19输入/125运行源码仍未修改。

当前人工报告第10节、解释性分析和链接均指向V2；最终扩展入口为
`V2/reports/phase2-results-expanded-v2.md`，发布收据`V2/publication-verification.json`。
保留V2自动报告内部的v1公式族版本标签，并在新入口加v2数值修正说明，不覆盖已封存报告。
V1不能继续作为最终常数负对照的数值来源。详见技术附录Y1；无新RL、前向、rollout、API或其他seed变式实验。

## 28. 2026-09-22：reward 校准的实际更新读出，全量 seed404 前向扩展

用户确认扩充此前讨论的实际policy更新方向方案并启动全量评估，目标是检验是否比原始delta幅度
更能预测效用方向。范围仍为seed404 U0→U5；不启动/修改505/606、RL、效用rollout、API或Phase3。
原125运行源码和已完成reward-variants v1/v2源码、输入、结果均保留，不提交/回滚/推送。

新公式：`v=Hu, xi=Hdelta, q=A*u[a], D_real=-Agg[q*dot(v,xi)/(||v||²+1e-12)]`。
H为每token全词表中心化，q使用未中心化动作log概率差；不截断正负项、不增加门控。
该方向为reward校准的rank-one/secant代理，不是实际PPO/GRPO总梯度，也不解决逐步credit的识别问题。
新增A=1（含零A原行）、abs(A)和投影系数对照，三聚合共12列；保留旧261列，共273列。
主口径固定PLACEBO/all/token/0pp，主看下降vs上升条件AUROC及Spearman(-ΔM)，并记录AP、
方向混淆/平衡准确率、top-k、2000组配对标签bootstrap和512组轨迹块符号对照。
已有404标签可见，本次仍是事后探索，不能依据结果重新选聚合或宣称跨seed验证。

新模块 `realized_reward.py` / `realized_reward_measure.py` / `realized_reward_analysis.py` /
`realized_reward_run.py`；协议见
[REALIZED-REWARD-SEED404-20260922-v1.md](SkillRL/docs/experiments/phase12-independent-v4/REALIZED-REWARD-SEED404-20260922-v1.md)。
最终CPU回归244 PASS / 94.90s，另已有真实16-token、248320词表样本验证显式方向点积误差<5e-19。
无需新增环境rollout，但旧标量未保存Hu·Hdelta，因此必须补U0/U5同输入全词表前向。

运行根为 `Q/realized-reward-s404-v2`（Q同前）；v1只完成prepare、未启动GPU，启动前封存扫描
增加排除临时文件/活动日志的防竞态保护，v1原plan及源码快照保留。数学公式仍为v1，候选口径未改。
八个逻辑分片、四个双卡worker、两波；每分片先算最长动作，逐决策复现旧稳定C/P/D等标量。
5511决策、155638 token×control行全部覆盖；效用仍用原每轨迹每技能首次自然调用标签。
新报告独立写入v2/reports，不覆盖旧报告。启动与验收状态以28.1及新目录收据为准；本节本身不代表已完成。

### 28.1 已实际启动并通过首波前向验收（09:37 UTC）

`Q/realized-reward-s404-v2`于09:29:45 UTC启动，后台PID `1412373`，plan SHA
`dacb0ca075a11d743023582b858e49e5ff24ddba053bf68150f90aa3d62d3d6c`。
启动容量扫描与模型加载后，四个双卡worker均已实际完成计算；09:37:25 UTC已观察115/5511决策，
分片0/1/2/3分别37/27/26/25。每分片最长动作先执行，分片1–3各有512-token动作（两控制1024行），
都复现旧稳定标量，无异常；GPU约12–20GiB/卡。137份绑定源码（含原125份）保持hash。

收据`first-wave-acceptance.json`明确状态RUNNING而非COMPLETE；统计和新预测性能尚不可用。
自动顺序：分片0–3→4–7→CPU全统计/独立核验/报告。日志`workflow.log`、`logs/shard-*.log`，
成功看`complete.json`，失败看`failed.json`及日志，无自动retry。完成后不会接续505/606。
首波验收后结束本轮人工监控，后台保留心跳。旧报告、RL/效用证据和全部未提交修改均保留。

## 29. 2026-09-22：中心化幅度 × reward 定向因子验证已完成

此前28节的realized-reward-s404-v2已经于11:04:50 UTC完成，本轮核验其complete/provenance后复用，
不重跑旧前向。用户确认“按照上述方案开始验证”，仅新增独立CPU标量分析，不启动505/606或Phase3。

冻结公式 `b=A*delta(a); B=Agg(||Hdelta||); R=Agg(b)/(Agg(abs(b))+1e-12); D_factor=-B*R`。
保留−R和全行A=1两种对照；token主分析，decision/game敏感性，固定PLACEBO/all/0pp。
原273列保留、新增12列，共285列；原效用标签/首次调用anchor/候选池、2000组配对bootstrap、
512组轨迹块符号对照全部复用。已有404/505历史标签可见，仍为事后探索。

运行目录 `Q/factorized-reward-s404-v1`，85项CPU测试通过，正式分析一次成功，13:35:53→13:36:39 UTC，45.79秒。
5700条指标独立sklearn/scipy核验、3660条方向混淆手算、420个因子单元独立fsum核验通过。
原5460条指标复现，137份旧源码与76项绑定输入SHA保持；无新训练/模型前向/rollout/API。

主结果：中心化范数方向AUC=.742857、AP=.613651；新组合AUC=.600000、AP=.294949，
与单独−R的AUC/AP相同。准确率8/12，平衡准确率.6，5个下降只判对1个、7个上升全部判对。
新组合符号与D_action_adv相同，不能把幅度重排称为新增方向识别。
相对A1方向AUC+.171429，但95%配对区间[-.566667,.583333]；相对范数−.142857，区间[-.7,.439236]。
当前不支持该组合优于中心化范数，不代表所有reward方法无效；不自动继续搜索或接入Phase3。

人读总结：[2026-09-22-seed404-factorized-reward-validation.md](2026-09-22-seed404-factorized-reward-validation.md)。
新完整报告在Q/factorized-reward-s404-v1/reports；原报告不覆盖，全部未提交修改保留，无提交/回滚/push/删除。

## 30. 2026-09-22：seed404效用标签精度扩展（2→8组gold）

用户授权增加续跑组数、修改并重新评估404。仅扩展效用重复；不重训RL、不做读出前向，
不启动505/606、Phase3或API。新设置见
[UTILITY-PRECISION-SEED404-20260922-v1.md](SkillRL/docs/experiments/phase12-independent-v4/UTILITY-PRECISION-SEED404-20260922-v1.md)。

固定原U0/U5、完整37库、冻结0.6B逐状态top1、25种自然技能的404个首次调用锚点。
原gold63011/63021保留，新增404/404100/404200/404300/404400/404500；evidence62011不变。
同anchor的O/P/N与U0/U5共用同一续跑seed；环境seed和前缀不动。
数字等于RL seed本身不保证精度；新seed间隔避免base+step的50步区间交叠，旧两组不追溯更改。
每端点复用3636条、新增7272条；最终两端点21816条，其中新增14544条。

285列既有读出、共同池、符号和阈值冻结；主结果固定8组，另完整输出2/4/8精度曲线与新增6组单独敏感性。
保留全37技能coverage/缺失NA、逐续跑seed效用、方向分歧、配对区间；点标签与方向可分辨性分别记录。
新增采样发生在旧标签已见之后，不能称新独立RL重复或确认性研究，不按D是否获益挑seed/技能。

新目录`Q/utility-precision-s404-v1`；plan SHA
`81c7213909a3f884401d5aae016674d9af450b8a8219ac63d41fa195b45acc19`。
相关回归407 PASS，最终协议专项35 PASS；实际旧数据CPU预检复现176效用单元和5700指标，独立sklearn/scipy通过。
141份历史源码不改，新增3模块/1测试/1协议共146份绑定，74项核心输入及7304项原轨迹/索引哈希绑定。
准备时free319.086GiB，cohort238.531GiB，含80GiB检查点保留量的准入预留144.812GiB；磁盘限制未放宽。

本节写入时prepare已完成、launch处于容量准入，尚不是新增续跑完成证据；实际启动验收另追加。
沿用vLLM八卡单卡分片与FileStore，U0补齐→U5补齐→CPU分析封存；不覆盖旧报告。
无总时长限制，磁盘保护保留，失败不自动重试。结果看新目录complete.json；不得重复启动原目录。

### 30.1 八卡实际续跑启动验收（2026-09-22 16:03 UTC）

父进程PID1446810，启动时间15:54:44 UTC；八个U0 worker为1447876–1447883。
完成模型/容量/旧轨迹准入后，八卡vLLM均就绪，每卡15217MiB。
16:03:38 UTC已新增47条U0续跑，分片0–7分别5/5/6/6/7/6/6/6；U0新增目标7272条。
每片首条新轨迹都核验了登记seed、trajectory_id、环境前缀和原anchor不变；未见失败日志。
router新本地成功774次、现场2次正在计算、failed0；不涉及外部API。
146源码和74绑定输入再验通过，证据在新目录first-wave-acceptance.json。

这是RUNNING_NOT_COMPLETE，不是8组效用或新版报告已完成；U5新增续跑尚待U0结束。
之后由后台顺序推进U5→完整统计/独立核验/封存，不开启505/606。
启动验收后结束本轮人工监控，后台进程/磁盘保护保持；不自动重试或删除历史数据。

## 31. 2026-09-25：Phase3 六臂 U20 实现对齐，正式执行未启动

用户确认六臂串行顺序：reward `D_sign_balance`、SkillRL-style 失败驱动、中心化交互幅度
`‖Hδ‖`、旧负部 gated D、`−P`、`+C_centered`。每臂先停 U20（四个五轮滑动窗口）；
优化器/LR 日程仍固定 150 更新，后续仅在全六臂 U20 结果讨论后显式请求更远停点。
主实验不使用 skill 级动作偏置阈值跳过编辑；各臂机会/上限相同，允许自然弃权和 o3 NOOP；
中心化幅度在幅度臂驱动技能排序，但作为“是否编辑”的门控只作不影响决策的影子记录。
全部旧 v1/v2 设置和 Phase1–2 证据保持原样。

新入口：[完整设置、成本口径、端点回收与启动说明](SkillRL/docs/phase3/PHASE2_ALIGNMENT_V3.md)，
配置为 `SkillRL/configs/phase3_setting_embedding_v3.json` 和
`SkillRL/configs/phase3_runtime_template_v3.json`。新 preparation schema 与旧四臂隔离。
代码只在封存下一端点的读出、编辑/gate、训练摘要及 checkpoint 后，带 intent/receipt
删除**该新运行目录内**上一个 native checkpoint 和 HF 导出；当前最新 native/导出、B0、
轨迹、方向 batch、bank、API/router 账本和报告保留。U20 Seen/Unseen 是不可覆盖的中期
milestone，根目录最终完成标记只留给 150 更新 horizon。

本轮 Phase3 相关离线测试及相邻模块共 **339 PASS**，发布白名单 949 文件存在，
冻结 SkillNet-37 manifest 匹配。没有 Phase3 GPU/ALFWorld 端到端验收、o3 真请求或正式训练，
`execution.approved=false` 继续阻止启动；没有 GitHub 推送、提交、回滚或历史文件清理。
下一步须核准实际磁盘/API 额度，完成 Phase3 专项预检和新服务器闭环，再单独确认执行。

## 32. 2026-09-25：Phase3 路由加速迁移与 v4 首批真实 rollout

用户允许停止尚无更新的旧 Phase3 首臂，优化路由并从新目录重跑。原 CPU 单条路由首臂和六臂队列于 19:15 UTC 收到 SIGTERM，核验 0 次完成更新、无检查点，原轨迹、方向 batch、路由账本与日志均保留。精确停止收据见 `/data/disk1/wangyifan/skill-scope-phase3-v3-20260925/router-acceleration-stop-20260925.md`；不得与新运行拼接。

新路由保持冻结 Qwen3-Embedding-0.6B、完整动态 bank、逐状态 top-1、FP32/SDPA 与查询格式，改为批量最多 8 条、独立本地 GPU 编码子进程。Ray CPU TaskRunner 本身仍不可见 GPU；侧车在训练时共享物理 GPU0，评估分片各用其分配卡。新数值协议与独立缓存/账本防止混用。旧 5100 条自然查询在最终子进程实现上 top-1 为 5100/5100 一致，最大分数差 `4.172325134277344e-06`；不据此宣称未来状态逐位或端到端轨迹不变。完整证据与限制见 [Phase3 路由加速记录](SkillRL/docs/phase3/ROUTER_ACCELERATION_GPU_BATCH_V1.md)。

加速启动 v1 因 OmegaConf 类型校验失败，v2 因 Ray CPU actor CUDA 初始化不可见而失败，v3 因给 TaskRunner 暴露 CUDA 污染子 Ray GPU worker 分配而失败；三次都在参数更新前停止，目录、日志原样保留，无自动重试。最终使用的独立目录为 `/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v4-20260925`，preparation、六臂顺序、seed707、U20 停点、原 RL/编辑/评估设置保留，六臂串行脚本 `run_six_arms.sh` 已启动，失败即停。

截至 20:11 UTC，v4 首臂 `readout_d` 已完成 U1 的 128 条 rollout：4953 次新路由成功、失败 0，router 中位 10.48 ms/状态、累计 87.96 s，GPU0 与七张策略卡共驻未见 OOM。OLD/reference 前向和 PPO 更新尚未封存，完成更新仍为 0/20；不要把首批轨迹或路由速度当成 Phase3 结果，也不要把 U1 误称为检查点（本配置每 U5 保存）。所有旧报告、失败目录与未提交代码变动仍保留。

## 33. 2026-09-26：Phase3 v4 U1 后显存冲突，v5 独立重启

v4 实际完成 U1 更新并保存 `metrics/u0001.json`，但在 U2 的首次路由处 GPU0
侧车因显存不足而失败：策略worker占26.97GiB、侧车占4.34GiB、余11.56MiB，
再申请46MiB触发OOM。队列失败即停，其余五臂未启动。U1无native检查点，
不得把v4部分训练并入新运行；v4完整证据保持原样。

新v5 `/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v5-20260926`
保持FP32、相同冻结embedding模型、逐状态全库top-1、外层最多8状态，
仅将模型前向固定拆为每批2状态；数值协议与缓存版本独立。旧v4的4961条
真实状态全部top-1匹配，普通/模拟27GiB共驻压力下分别耗时146.39/149.97秒，
未观察相对旧batch-8同量级测试的明显速度下降。Phase3/路由专项128 PASS。
预检压力进程已释放，v5新准备已建立，六臂脚本于本轮启动，首臂进入训练子进程。
是否真正越过U1→U2并封存U5仍待日志验收；此节不等于Phase3结果。
详细记录见 [v5显存恢复说明](SkillRL/docs/phase3/ROUTER_GPU_MEMORY_MICROBATCH_V2.md)。

## 34. 2026-09-26：v5 U5 内置验证显存冲突，v6 新协议准备

v5 首臂完成 U1–U4 更新并落盘指标，U5 训练轨迹已采集，但 U5 内置Seen验证第一次
选技时 GPU0 仅余8.25MiB、侧车申请12MiB而OOM；U5指标和原生检查点均未形成。
队列失败即停，其余五臂未启动。v5完整证据保留，不能把未封存U1–U4拼入新实验。

v6 位于 `/data/disk1/wangyifan/skill-scope-phase3-batched-gpu-v6-20260926`：
注册独立vLLM profile，将显存占用上限0.45→0.38，训练侧FP32 router共驻卡改为
GPU7；其余RL/模型/路由语义和每5轮验证、保存不变。八卡合成4096+512、16序列、
两次更新与恢复预检全8 rank PASS；Phase3/路由129测试PASS。GPU7旧4961状态
top-1全一致，约27GiB共驻压力下亦全通过，路由耗时149.51/150.55秒。
这些是组件级验收，不等于真实U5已通过；完成U5验证并形成native checkpoint
仍是启动后的下一道关键门槛。详细证据及边界见
[v6 U5显存恢复说明](SkillRL/docs/phase3/ROUTER_GPU_MEMORY_VALIDATION_V3.md)。

### 34.1 v6 实际启动验收（2026-09-26 13:32 UTC）

v6六臂顺序脚本PID2289579，首臂runner PID2289607、training PID2289666存活。
八卡Qwen3.5-4B/vLLM已加载；GPU7共驻侧车实际选技，本地账本首16条全success、
失败0，非外部API。正式更新仍0/20，U5验证/检查点尚未验收。
本轮未覆写v5证据，不把组件压力测试或启动选技视为Phase3结论。
