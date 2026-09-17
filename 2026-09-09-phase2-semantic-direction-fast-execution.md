# Phase2 快速执行记录：语义边际效用为主分析

确认日期：2026-09-09；汇报目标：2026-09-11 上午（北京时间）。

用户确认以 Phase1 的语义边际效用 `ORIGINAL−PLACEBO` 作为 idea 对应的主分析，并要求尽快产生 Phase2 实验结果。本文件记录对9月8日草案的执行修订，修订发生在任何 Phase2 方向信号或 gold 结果出现之前。

## 1. 已固定的首批实验

- 父模型：Seed303 U30；恢复原 optimizer/scheduler/rank RNG，KL reference 仍为公共 B0。
- 主目标：`P_intᴾ/Dᴾ → ΔM_sem`，`M_sem=success(O)−success(P)`。
- 次要对照：O−TARGET_NULL，同步归档，不替换主语义目标。
- 固定六个 post checkpoints：U31/U32/U33/U34/U35/U36，逐 global update 保存。
- U31/U32 开发，U33 边界隔离，U34/U35/U36 时间外测试。
- 每个 global update 8 games×4 rollout、LR=1e-6、KL=0.01、8卡 FSDP；内部所有 Adam step 单独记账。
- 由于旧环境 worker 无 sampler 恢复接口，新分支的环境 seed 按 `30300+post_update` 预定；这延续父模型/optimizer，不声称逐 bit 延续旧环境采样流。
- 200个原 B0 anchors、4个 supported Skills，三臂保留自然首次调用、中途 prefix replay 和后续持续 payload 替换。
- 首批每 anchor 使用两个独立 continuation seed：41011仅用于旧效用特征，42011用于 paired old/new gold。共7个 endpoints×200 anchors×3 arms×2 seeds=8,400条 suffix。

首批使用相同固定 game/state support 的早期 updates 开发、后期 updates 测试；不额外声称新 game 泛化。仅三个测试 updates 的结果按探索性证据报告，连续效用和完整覆盖率保留。

## 2. 时间优先级与公式边界

首要顺序：严格采集 → 精确 full-vocabulary P/D → 三臂语义效用 → 方向关联/时间外预测 → 参数/activation 解释 → JVP工程验证。

保留 design 中 `d=A(e_a−π_old)`、`C_upd`、`P_int`、门控 `D_t` 原式，`W=I`。O/P/N共享旧 response prefix；advantage 直接取实际训练 tensor。零 advantage 与门控失败的 token 保留在均值分母，支持不足时 abstain。

新增强化：ORIGINAL 的旧/新全词表分布都直接从实际训练 actor 的同一条 compute-log-prob 路径采集；无损保留 FP32 master parameters，避免仅靠 BF16分析导出重建微小参数变化。PLACEBO/NULL进行相同精度的端点 teacher forcing，并记录 live/replay parity 与匹配 backend 的敏感性。

第一轮起点重复 forward 校准数值噪声，读取 gold 前锁定 gate。每个 checkpoint 的信号和预测文件先 committed，再打开该 checkpoint 的 gold。

12-update扩展、额外重复、shuffled-reward/independent-batch/random-direction训练对照和完整成本基线排在首批核心结果之后，不虚报已完成。JVP若遇到不可微 kernel 或数值问题，明确标记并讨论，不用参数范数冒充。

## 3. 已接入的代码

- `SkillRL/phase2/capture.py`：decision IDs、完整训练 batch、真实新旧全词表概率与 optimizer step。
- `SkillRL/phase2/direction.py`：完整 token 支持上的 C/P/D、门控和数值敏感性。
- `SkillRL/phase2/export_model.py`：无损 FP32 endpoint 导出。
- `SkillRL/phase2/measure.py`、`aggregate.py`：反事实 teacher forcing、activation、参数差分、支持审计。
- `SkillRL/phase2/evaluate.py`：带独立 evidence/gold seeds 的三臂完整轨迹。
- `SkillRL/phase2/forecast.py`：读取目标 gold 前锁定方向预测。
- `SkillRL/phase2/report.py`：按完整 endpoint 更新结果报告和图表。
- `SkillRL/phase2/run_fast.py`：逐 update 调度、失败停止、恢复边界和资源检查。

Phase2 训练钩子由 `phase2.enabled` 显式启用；普通 Phase1 路径默认不启用。旧3个seed的模型与实验结果不修改、不删除。

## 4. 归档和进度入口

运行根目录：

```text
/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1
```

执行协议：`protocol.json`；实时阶段：`status.json`；训练/评测日志：`logs/`；逐步完成记录：`completed_updates.jsonl`。

汇报用滚动结果：`/home/wangyifan/skill-RL/2026-09-10-phase2-semantic-direction-fast-results.md`。

已通过初始检查：33项 Phase1 测试、4项新增数值/采集测试；U30 FP32导出通过427个 tensors的 dtype/完整性检查。首轮已启动；此处不将启动等同于完成实验。

## 5. 首轮 gold 前锁定的分析实现细节

预测比较固定为 old-margin-only、unsigned update/action、activation（8/16/24/32层）、signed（加入 C_upd/P_int）、opposition（加入 C_upd/D）。统一 StandardScaler + Ridge α=1；负事件分类使用 Logistic C=1。仅用U31/U32开发，最多8个单元；有效支持不足6个时不拟合，保留direct scores。不根据后续结果挑特征或阈值。

方向预测的主误差为连续 signed ΔM 的 MAE；二值 AUPRC/Brier 使用 ΔM<−0.05 的**点估计**事件，并与 bootstrap CI 给出的可靠方向分开报告。ORIGINAL/PLACEBO 各自的性能变化同时存档。TARGET_NULL 仅清空目标 Skill，不能称为全库 skill-free baseline。

补齐分析依赖：scikit-learn 1.7.2、joblib 1.6.0、threadpoolctl 3.6.0、tabulate 0.9.0；未改动 torch、Transformers、numpy、scipy 训练栈版本。

## 6. 首轮实际采集检查（2026-09-10 00:30 左右）

U31 已开始真实 Adam 更新，恢复起点为 Adam step 495。完整 batch 为744行，其中740个独立 decision，4行为训练补齐重复；744行旧 actor 全词表概率已保存。1,422个 O→PLACEBO/NULL counterprompts 已逐一对齐实际训练 token，训练 temperature=1.0。检查中修正了 Transformers 5 的 chat-template 默认返回对象差异，未改动训练 prompt 或 rollout。

本 batch 的四个主评估 Skill 非零 advantage 决策支持分别为：cle_003 120个/5 games、cle_004 294个/5 games、cle_006 20个/5 games、gen_002 64个/4 games；最终 supported 判定还检查不同 trajectory 数。实际训练也记录了 cle_002/cle_005，但不在先前冻结的200-anchor主评测支持集中，不临时加入主分析。

41项测试已通过（33项 Phase1 + 8项 Phase2）。代码与依赖快照位于运行根目录 `source_snapshots/`；对齐审计为 `batches/u0031/alignment_audit.json`。此时完整 U31 checkpoint、post-update方向信号与 gold 尚未完成，不据此写出预测结论。后续训练调用另增加不改变模型计算的前向进度记录，便于区分耗时计算与停滞。

按当前更新速度与既有三臂评测吞吐粗估，首个方向对比仍需数小时，完整六-update首批约12–18小时；此为排期估计，并非已测得的全流程耗时，首轮完成后应校准。

## 7. 共享 GPU 调度恢复（2026-09-10 11:16）

U31已保存：24个真实Adam steps（495→519），参数更新L2=0.24903；信号在01:06（北京时间）锁定。U30的1,200条三臂evidence/gold评测已完成，U31预测在02:14锁定。此后调度器一直等待8张空闲GPU，U31 gold尚未启动，原12–18小时排期因此不能直接沿用。

恢复时GPU 0–3空闲，4–7被其他任务使用。仅调整独立推理任务的物理调度：保持8个逻辑分片及各分片样本/seed不变，在当前空闲卡上分批执行；已完成的分片直接跳过，不等待GPU。RL训练仍使用原定8卡与完整optimizer恢复，不混用其他任务的GPU，也不擅自改成4卡续训。等待训练资源时及评测运行中定期刷新结果报告，避免报告阶段长期过期。

## 8. 用户授权按空闲卡数续训（2026-09-10 16:15）

用户明确同意“不要空等，有几张卡就用几张”。本修订取代上一节等待8卡的训练调度规则，不改动既有U31结果、方向阈值、开发/测试划分及三臂gold协议。

每个update开始时使用当前空闲的1–8张卡，update内部不变更world size。维持8 games×4 rollout、LR=1e-6、KL=0.01及32个decision的完整optimizer minibatch。3/5/6/7卡使用零权重同步槽及相应梯度归一化，不改变有效batch。1–2卡启用CPU AdamW以降低显存，公式和超参数保持不变，CPU/GPU数值路径的差异作为资源变更记录，不声称逐bit等同。

当前优先将U31的8卡完整checkpoint转换成4卡恢复输入。转换严格保留单个FSDP FlatParameter的注册顺序、FP32值、Adam一阶/二阶moments、step=519及scheduler；新文件写入 `elastic_resume/`，原checkpoint不修改。保存后逐张量核对SHA256；实际加载到GPU后再次检查参数/optimizer校验和，并以U31旧训练batch的固定输入对齐原已归档的新actor全词表概率。任何检查失败即停止，不从空optimizer继续。

跨world size的rank RNG无法一一保持原映射，预定使用 `20260910 + 1009×parent_update + rank`，各rank不同。game sampler继续按 `30300+post_update`。因此仍是一条Seed303父模型的续训分支，但不是严格8卡随机流的延续。

资源策略：`resource_policy.json`；逐update实际卡数/后端/恢复路径：`allocations/`；逐rank加载校验：`elastic_restore_audits/`。转换与测试进行中时不写成已经通过真实多卡恢复。

### 8.1 四卡恢复及回归检查（2026-09-10 16:43）

U31→U32 的实际四卡恢复已完成逐rank检查：FP32参数、Adam moments与counter的SHA256全部匹配，Adam=519，固定输入全词表log-prob最大误差四卡均为0.0。首次启动随后在第一个rollout入口遇到FSDP复用inference tensor错误，尚未执行新的optimizer update；校验前向改为`no_grad`后，新增的真实双卡“恢复→校验前向→完整参数rollout上下文→backward→Adam”回归测试通过。失败尝试单独归档在`attempts/u0032-attempt1-inference-cache/`，第二次启动仍从同一U31状态恢复。

另完成133项单元/回归测试及三进程CPU/Gloo的全局梯度权重测试。非整除卡数下，原始trainer的整除限制只对经过配置检查的Phase2文本HF/GRPO路径放开；其他Phase1路径不变。推理请求沿用pad/unpad，实际环境轨迹仍为32条；训练decision补齐重复沿用既有记录与去重规则，不能把补齐行误算成新增独立支持。

目前按GPU 0–3续训，4–7为其他任务使用。第二次启动时源代码快照为`source_snapshots/1faf4d5807f0c47d/`；随后对未来非整除卡数启动校验与每次训练自动快照的补丁另行归档，不据此宣称完整六-update已完成。

## 9. 连续方向分析与 GPU 自动恢复补充（2026-09-10 晚）

用户确认主目标为连续 signed `ΔM_sem`，重点评估P_int的预测信息；D_t的下降风险为辅，不要求发生正→负翻转。U31–U33已观察，U34–U36尚未产生gold。本补充在测试gold前归档为 `analysis_amendment_v2.json`；不修改原公式、门控、模型特征、4个Skill/200 anchors或开发31/32、隔离33、测试34–36划分。

报告增加全部已有指标的signed/absolute ΔM描述性相关表、按update分解、P的点估计同号率及覆盖率，并明确排序关联不等于逐例方向正确。保留零变化、CI不确定样本参与主MAE。CI是否排除0仅作额外可信度描述，不替换原±5pp主标签。既有Ridge/Logistic不调参，另在未来测试预测锁定时保存zero-delta、development mean及development majority direction参照；多数方向仅使用31/32非零变化，平局abstain。描述性相关不用于挑指标。

U34第一次启动在FSDP模型初始化时发生显存OOM，无新训练证据；原尝试归档 `attempts/u0034-attempt1-startup-oom/`。完整U33有4份model/optimizer/extra shards，Adam=568。恢复后跳过已完成31–33，等待资源后从U33继续U34。

资源修订：`resource_watch_amendment_v2.json`。每30秒记录GPU显存/利用率，连续两次显存占用<1500 MiB且利用率<10%才选择该卡；使用所有确认空闲卡，不等待8卡。检查记录为 `gpu_watch.jsonl` / `gpu_watch_latest.json`。保留250GiB磁盘安全阈值，不终止其他任务。

如果再次发生模型初始化OOM，且本尝试尚未进入Training Progress、没有任何U34（或相应update）的轨迹、batch、optimizer记录、signals/evaluation或checkpoint数据，自动归档并返回资源监控。运行期OOM或其他异常仍停止，不能把部分更新盲目覆盖。共享GPU没有跨用户原子预留机制，因此这一规则降低损失但不保证不会再次竞争显存。

新增分析与监控回归检查后，146项Phase1/Phase2测试通过，包括初始化OOM归档后自动重试、原checkpoint不变的检查。旧Phase1报告及权重不变；剩余训练完成时间取决于空闲资源，报告只展示实际完成结果。

## 10. 9月11日恢复：启动检查卡数适配

监控于00:35选择物理GPU 5并完成U33→单卡转换，但00:43被旧Phase1 preflight硬编码“至少2张CUDA卡”拦下。没有U34训练数据或参数更新。失败尝试在 `attempts/u0034-failed-1789058617473162626/`，单卡转换文件保留不删除。

用户05:22要求继续。当时GPU 0/1/2/3/4/6六卡空闲，5/7仍被占用。修复preflight显式卡数参数，Phase1默认仍要求2卡；Phase2按照实际分配数检查1–8卡。通过156项单元/回归测试，六卡真实FSDP小模型的恢复→校验前向→rollout上下文→backward→Adam生命周期检查通过。这是工程检查，不是实验模型的效用评测。

后续继续从唯一完整U33、Adam step568恢复U34，必要时转换为六卡，并再次校验实际模型/optimizer SHA256与全词表log-prob。六卡采用既有零权重同步槽维持32个decision的optimizer minibatch；8 games×4rollouts、LR、KL、gold协议、预测划分均不变。完整实验模型六卡续训是否成功以实际加载审计、optimizer记录和checkpoint为准，不把小模型smoke通过当成正式结果。

05:33已完成全部六卡恢复文件的逐张量校验。随后首次六卡启动被run-manifest不可变配置检查拦下：同一run_id先前已记录四卡配置，不能直接覆写成六卡。修复采用每次启动独立的`launch_manifests/u0034-<attempt>/`目录保存preflight和manifest；原文件不覆盖，训练/轨迹/统计仍使用原`phase2-s303-fast-u34`逻辑ID。具体启动目录同时写入`allocations/`和`allocation_history.jsonl`，失败归档只复制对应尝试的manifest。测试增至157项通过，另有六进程梯度归一化与真实单卡preflight通过。该次manifest拦截仍发生在训练初始化前，不计为新update。

### 10.1 U34五卡实际恢复并开始rollout（2026-09-11 05:49，北京时间）

重启时GPU 0已被其他任务占用，因此重新按规则选择物理GPU 1/2/3/4/6。05:44完成U33四卡→五卡恢复文件转换与校验，原四卡checkpoint和此前单卡/六卡转换文件均保留。05:44:46启动U34，独立启动记录位于 `launch_manifests/u0034-1789076686208810698/`，已通过五卡preflight与manifest创建。

05:48:46五个实际训练rank全部通过恢复审计：Adam step=568，FP32参数与Adam moments/counter的SHA256一致，固定父端点输入的全词表log-prob最大绝对误差均为0.0。审计位于 `elastic_restore_audits/u0034/rank-0.json` 至 `rank-4.json`。05:49确认五个worker均进入 `actor_rollout_generate_sequences`，不再只是转换或模型初始化；此时尚未完成U34参数更新或gold评测。

本update固定五卡、每完整optimizer minibatch为32个有效decision，每rank至多7个同步槽，额外槽权重为0。新增空闲卡仅在后续调度边界使用，不为增加卡数中断当前update。共享环境仍可能在启动后有其他任务进入所选GPU，本记录不构成独占资源保证。

05:49磁盘余量约527GiB，继续执行update边界250GiB保留线检查。调度器完成本update后依次导出FP32模型、计算并锁定指标/预测、运行三臂gold、刷新结果报告，再按资源情况续跑U35/U36。U31–U33完整记录及原Phase1三seed报告保持不变。

## 11. U34运行期OOM后的显式恢复（2026-09-11 13:10，北京时间）

上一节记录的是实际rollout开始，而非完成更新。该五卡尝试于06:10在首次actor backward停止：FSDP为不能被五卡整除的全参数梯度进行padding时需要额外约15.67GiB；失败GPU上另一进程占约31.88GiB，本进程约43GiB，剩余约4.23GiB。OOM发生在运行期，调度器按保护规则停止，没有自动覆盖证据。

用户13:06要求继续。已检查该尝试没有任何成功的Adam-step记录、U34 checkpoint、新端点概率、signals、prediction或gold。留下的750行batch（746独立decision、14,678 loss tokens）、750行旧全词表概率、32条训练轨迹、五rank恢复审计和前向进度，全部归入 `attempts/u0034-failed-1789078232762546641/evidence/`；原文件未删除，U33未修改。轨迹共享索引在锁下备份后仅移出32条失败记录，避免重跑后将其混为正式训练数据。归档映射与计数见 `evidence/recovery.json`，完整原因见该尝试的README。

新增 `phase2.recover_failed_update` 只供明确要求续跑时处理经审计的“没有成功optimizer step的OOM”，不能自动处理已有参数更新、预测或gold的尝试。调度器的手动重启也会检查未归档证据，存在则拒绝覆盖。恢复/资源测试17项、全部Phase1/Phase2回归测试163项通过。

13:10重新启动流水线，当前空闲GPU为0/2/3/6，可直接加载U33原生四rankcheckpoint，无需新增约50GiB转换副本。此次重新采样U34，不把失败五卡batch接到四卡optimizer上；所有实验超参数、固定gold支持集和开发/测试划分保持不变。后续实际GPU分配以 `allocations/u0034.json` 为准。磁盘约513GiB，继续沿用250GiB边界保留线。

13:15确认四个实际worker均已进入 `actor_rollout_generate_sequences`，GPU 0/2/3/6各占约36GiB；当时无其他进程使用这四张卡。流水线PID为1718352，启动manifest为 `launch_manifests/u0034-1789103473341679201/`。这只确认新的四卡rollout已开始，不表示U34 optimizer更新、checkpoint或gold已完成。

## 12. 第二次运行期OOM后转入双卡CPU Adam（2026-09-11 15:33起）

上节四卡尝试于13:31首次backward时OOM：另一进程在运行中进入原本空闲GPU，占39.36GiB；本进程约33.86GiB，剩余5.90GiB，无法满足7.83GiB申请。没有成功的Adam-step记录或U34端点。本次失败不是五卡padding问题，不能把固定四卡当作共享资源的独占保证。

用户15:33要求继续。该尝试的32条轨迹、676行batch（673独立decision、11,991个loss tokens）、676行旧概率及前向进度于15:35归档至 `attempts/u0034-failed-1789104704603119618/evidence/`。完整索引先备份再持锁移出对应失败记录，其他记录不动。该次为原生同world-size恢复，没有elastic审计目录；恢复工具只在allocation和父checkpoint的world size/path均匹配时允许此情况，不能忽略跨world-size缺失审计。

当前GPU 3/6空闲。按照既有资源协议将U33无损转换成双卡恢复输入，Adam仍从568开始。1–2卡CPU AdamW路径已在9月10日注册：保持GPU上的参数/梯度，只将Adam moments和运算放到CPU，保持LR=1e-6、KL=0.01与32个decision的完整optimizer minibatch。此次不新增RL配置，不混入失败batch，后续CPU/GPU数值路径差异仍须披露。

已在真实GPU 3/6运行独立小模型FSDP恢复→校验前向→rollout上下文→backward→CPU Adam检查，两rank均从519更新到520，参数注册对象保留、moments位于CPU。输出为 `smoke/elastic-restore-cpu-adam-2gpu-20260911/`，这不是正式实验模型结果。全部Phase1/Phase2回归测试165项通过。

双卡转换文件约48GiB；转完后磁盘约453GiB。当前仍不删除任何实验数据。已向用户单独询问是否允许清理U33不再使用、可从原始checkpoint重建的单/五/六卡重分片副本（约144GiB），未得到确认前保留。该选择不阻塞当前U34恢复，但后续仍受250GiB边界保留线限制。

15:42两个正式恢复分片均回读校验通过，随后启动流水线PID 2142632；15:43分配GPU 3/6并通过preflight，独立启动目录 `launch_manifests/u0034-1789112583457258458/`。增加只读运行期资源监控PID 2152531，每15秒记录物理GPU显存、利用率、compute PID、流水线子进程及磁盘余量，输出 `runtime_resources/pipeline-2142632.jsonl`；它不干预任何任务，随所监控流水线退出而停止。新增两项物理GPU映射/空进程快照测试通过。此时仍在初始化，不据此声明U34已完成更新。

### 12.1 正式双卡恢复与首次真实CPU Adam更新已通过

15:47:47两rank正式恢复审计通过：Adam568、FP32参数及optimizer状态SHA256匹配、固定父端点输入全词表log-prob最大误差均为0.0。随后完成32条rollout及680行旧策略概率；16:02完成reference前向，保存680行batch（679独立decision、12,439 loss tokens）。

16:04:32–33两rank首次真实optimizer更新成功，均为Adam568→569、grad_norm=0.09470558911561966、LR=1e-6；每rank16个decision，共32个。凭据为 `optimizer_steps/u0034-rank0.jsonl` 与 `rank1.jsonl`。这次不再只是smoke或rollout启动，已越过此前两次OOM的首次backward/optimizer阶段，但尚未完成整个U34或保存新checkpoint。后续minibatch继续运行。

双卡反向传播观测显存峰值约74GiB/卡，CPU Adam阶段回落至约24GiB；不能因此保证共享GPU不会再次OOM。只读资源监控持续运行，磁盘约441GiB，仍执行原有边界安全检查。所有167项Phase1/Phase2回归测试通过。U34 gold和方向预测结果只有端点保存并完成后续流水线才发布。
