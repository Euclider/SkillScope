# Embedding router：Phase3 发布与本机 ALFWorld Phase1–2 启动

## Material Passport

- Origin Skill: experiment-agent (academic-research-suite)
- Origin Mode: run
- Origin Date: 2026-09-18T05:21:02Z
- Verification Status: UNVERIFIED
- Version Label: embedding_publication_launch_v1

本文是发布、启动和观测记录，不是性能结论。上述 UNVERIFIED 指正式实验尚未完成、结果尚未核验；发布文件和下述工程检查已分别核验。保留此前所有 mini router、synthesis-v2、组件失败检查及旧发布记录，不覆盖旧报告。

## 用户授权与正式任务

用户确认先迁移变动至 `Euclider/SkillScope-phase3`，再直接启动本机 Phase1–2。随后用户追问“不跑 ALFWorld”的含义，已明确：只有启动前的合成硬件检查不运行 ALFWorld；正式任务仍为真实 ALFWorld RL 训练和 Phase1–2 评估，没有改变 benchmark 或缩减为合成评估。

本轮不启动本机 Phase3、不调用 o3、不恢复已停止的付费 mini cohort，不提交本地研究仓库、不重置或回滚文件。允许的 GitHub 写入限于目标 Phase3 仓库发布。

## 已发布的 Phase3 迁移

- 仓库：<https://github.com/Euclider/SkillScope-phase3>
- 新 main commit：`674dd36a54a83c5061af43f0f80b2ee3da924468`
- 父 commit：`47e4435a9989c88157f2e11f374a28a5e7530670`
- tree：`e04a724474b81f51e10f3503f59890494285d02d`
- 发布前逐项核对 877 个远程 blob 与独立导出包一致，然后非强制更新 main；没有删除远程文件、没有修改本地 Git index。
- 上传回执：`/mnt/workspace/users/wangyifan/SkillScope-phase3-upload-20260918-v4.json`
- 回执 SHA256：`c8ad76348418e5358b0af986a03e78ebeda260228b80b10e209523339db489b3`
- 独立导出目录：`/mnt/workspace/users/wangyifan/SkillScope-phase3-export-20260918-v4`
- tar 包：`/mnt/workspace/users/wangyifan/SkillScope-phase3-20260918-v4.tar.gz`
- tar SHA256：`62b8027195b295a877451728951e4985042e5ee181e875a5b9b382d3c39f5fad`
- 发布 inventory SHA256：`15afd119fdc567594b9d38dca733a991b8417e3db8b056e8e2c21ef247b59f81`

四分支共用冻结 Qwen3-Embedding-0.6B、逐状态全库 top-1。Phase3 对每个实际 bank snapshot 重建索引、隔离缓存，以 bank manifest/protocol hash 绑定记录；分支级 SQLite 本地调用额度跨库版本、训练、gate 和最终评估累计。缓存重放不记新调用，失败或状态不明调用不能自动重试。允许库 add/modify/delete/merge 后增长或缩减，不沿用固定 37 库的错误候选列表。编辑器仍为 o3，外部 editor 预算与本地 router 额度分开。

新启动指南位于发布仓库根 `README.md`（源为 `SkillRL/docs/phase3/START.md`），配套 `SETTING.md`、`VALIDATION.md` 和 `configs/phase3_runtime_template.json`。新配置无 router API key；只在真正运行 Phase3 编辑时需要 o3 凭据。旧 mini setting 留作显式旧协议，不自动回退。

这是 **state-aware adaptation of SkillRL embedding retrieval**，不是原生 task-only、整局复用 top-k 的复现。没有增加路由器训练、候选模型比较、查询调参或人为强制 skill coverage。

### 验证边界

本轮较早的广泛回归为 416 passed；在最后增加 parity 检查和记录字段后，最终可移植测试集合在源码及独立导出目录各为 314 passed。两个数字对应不同范围/时间，不能合称最终版本重新通过了 416 项。独立导出导入路径、文件哈希、秘密扫描和 Python AST 检查通过；`uv --no-cache pip check` 检查 180 个依赖兼容。未运行真实 Phase3 RL、编辑或 API 测试。

启动后只读对照发布 manifest，本机 470 个对应 Python/config/依赖文件与发布字节一致；后续只追加本机启动/交接文档，不静默改动运行代码。

## 八卡与 CPU router 启动前验收

输出：`/mnt/workspace/users/wangyifan/skill-RL/embedding-gpu-preflight-20260918-v1/`。这是本轮新建的合成检查，不是历史实验重跑。使用 GNU timeout 的 1800 秒检查上限，未触发超时；退出码 0。

8/8 rank 为 PASS：全长 4096+512 前后向、Adam 更新、native 权重/优化器/RNG 保存恢复、恢复后 HF generation。rank 0 的 CPU router 在 GPU 前后向/恢复前后各完成一次新本地决策，外部 API 调用为 0。各 rank 最大 reserved GPU 内存约 30.234 GiB。

- `complete.json` SHA256：`29554caea616ea01550c4658502185aab7717d16702ca4b13192c6ca31add763`
- `embedding-router.json` SHA256：`39aeff20017cd5e11856f1db50a9b596fdf839978f8ff6ee3f3df0799f8aebb7`
- 其余证据：`rank-0.json` 至 `rank-7.json`、`run.log`、`synthetic-checkpoint/actor/`，全部保留。

这只证明合成单 microbatch/rank 的工程共存与恢复，不证明完整 128 轨迹 Ray+ALFWorld 训练、成功率、技能覆盖率或完整 150 更新可完成。公开包 VALIDATION 在发布时说这项新组合检查尚未执行，属于发布时真实状态；本文补充随后完成的本机记录。

## 正式 Phase1–2 注册与启动

- ID：`skillnet37-qwen35-embed06-s404-8gpu-v1`
- Type：training + evaluation
- Status：running / launched_not_complete；不是完成结果。
- 启动时间：2026-09-18 05:21:02 UTC（服务器 `ps` 默认显示 UTC+8，本文统一 UTC）。
- supervisor PID：`903045`
- Working Directory：`/mnt/workspace/users/wangyifan/skill-RL/SkillRL`
- Python：`/mnt/workspace/users/wangyifan/.venvs/skillnet-phase12-20260917/bin/python`
- Root：`/mnt/workspace/users/wangyifan/skill-RL/SkillRL/artifacts/phase12/skillnet37-qwen35-embed06-s404-8gpu-v1`
- Preparation：`SkillRL/docs/experiments/skillrl-embedding-router-v1/preparation-s404-8gpu/manifest.json`
- Preparation SHA256：`3b731d125adc0731b193f59821dea7f5dd0d54d6d94d17fb908deeafdb35c7fe`
- Authorization：新 Root 的 `authorization.json`；SHA256 `a871d5895bf54e05f853c847cce25a3acd06ad09a64421084a68c7ad1b71f2ca`
- 数据路径：`/mnt/workspace/users/wangyifan/skill-RL/data/alfworld`

prepared manifest 的 `execution.approved=false` 是保持不变的准备快照，实际授权由绑定其哈希的独立 permit 提供，包括 training/evaluation/readout/exports。不能为显示“已启动”而覆盖原 preparation。

### 冻结实验设置

Policy Qwen3.5-4B；完整初始 SkillNet-37；seed=404；GRPO lr=1e-6；每更新 16×8 条轨迹；PPO minibatch 128、microbatch/GPU=1；最多 150 更新；每 5 更新验证、保存及固定 5 更新读出。环境分段 seed 为 `404+16*(start_update//5)`。训练沿用 SkillRL 六类任务完整 train pool 的采样方式，不强行逐次穷举所有训练游戏。正式性能遍历 valid_seen 与 valid_unseen，分别报告，不把 seen 监测当最终 unseen 结果。

Router 为冻结 `Qwen/Qwen3-Embedding-0.6B`，revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`，运行在 CPU，不占 policy 的 GPU。输入为任务、当前 observation、合法动作、最近 2 步历史与 step；所有 37 个技能参与余弦 top-1，无 game 类别过滤。Profile SHA256 为 `d1fc18a18537e65d4503d65d46187b8b63d21d9a884f6332262dbbb2dda778c8`，CPU protocol SHA256 为 `2a5128c53f05c9e09f2cc951c0fa3694f271fa2289306098b2db5d58e87f7f72`。本地训练、性能、anchor 和 O/P/N 共用协议及新缓存，不复用 mini 的缓存。

先锁定窗口预测再开启目标 gold；自然支持不足明确 abstain，不为增加可报告技能强制调用。主分数与既有 readout 协议不变，不按结果挑窗口或改方向。

### 实际启动命令

在上述 Working Directory 执行：

```bash
env -u SKILLNET_ROUTER_API_KEY -u SKILLRL_PHASE3_EDITOR_API_KEY \
    -u OPENAI_API_KEY -u SKILLNET_ROUTER_COST_PROFILE \
    PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    TOKENIZERS_PARALLELISM=false \
    /mnt/workspace/users/wangyifan/.venvs/skillnet-phase12-20260917/bin/python -u -B \
    -m skillnet_cohort.run \
    --preparation /mnt/workspace/users/wangyifan/skill-RL/SkillRL/docs/experiments/skillrl-embedding-router-v1/preparation-s404-8gpu/manifest.json \
    --root /mnt/workspace/users/wangyifan/skill-RL/SkillRL/artifacts/phase12/skillnet37-qwen35-embed06-s404-8gpu-v1 \
    --authorization /mnt/workspace/users/wangyifan/skill-RL/SkillRL/artifacts/phase12/skillnet37-qwen35-embed06-s404-8gpu-v1/authorization.json \
    --execute --detach
```

该命令已执行，不要再次执行。detached 启动器退出码 0；正式 supervisor 尚未退出，不能把启动器的退出码写成训练完成。首个 block 为 u0000→u0005。

### 调用/磁盘边界与停机策略

外部 router API 调用预算为 0，启动环境移除 router/editor/API key 和旧费用 profile，不需要新付费额度。本地 router 调用上限 202,105,900 是现有 150 更新、两 split、最多 37 技能×50 anchors、固定 O/P/N 续跑设置的保守总上界，不是新加实验或预期实际调用量：训练 960,000，监测 99,200，性能 424,700，anchor 源 822,000，paired continuations 199,800,000。

继续执行已授权存储策略：本轮 root 上限 760 GiB、保留磁盘空闲至少 100 GiB、下一检查点预留 80 GiB。仅在固定 5 更新窗口封存且逐行可再生证明成立后，回收本轮可再生全词表临时张量；保留不能证明可再生的行。所有轨迹、真实优化器批次、报告、检查点和历史实验不删除。新检查后的磁盘空闲约 837 GiB；不保证容量足以保存全部 150 更新。

正式长跑遵守用户批准的更新/调用/存储停止条件，不擅自套用合成预检的 30 分钟超时。协议/额度/存储或子进程失败则停止本轮并保留证据，不自动重跑、不扩大预算、无 API fallback。以后恢复需要先审计，不重新执行本次 new-run-only 启动命令。

### 初始只读观测

2026-09-18 05:22:22 UTC：supervisor 存活，u0000 初始 HF 快照已导出，首个训练 block 已进入 Ray/ALFWorld 初始化。日志确认 train 3553 games、seen pool 140 games，两个 environment manager 均启用 `skillrl-embedding-state-top1-v1`。尚无更新指标文件（0/150），尚未以 router 决策账本证实进入真实 rollout；没有 stopped marker。后续进度另追加时间戳，不把初始化等同成功完成更新。

主要入口：新 Root 的 `supervisor_launch.json`、`launch.json`、`supervisor.log`、`logs/train-u0000-u0005.log`、`router.sqlite3`、`metrics/`、`completed_windows/`。只有最终 `complete.json` 和各窗口完整证据核验后，才可宣称全流程完成。

### 05:26:40 UTC 追加观测：真实首批 rollout

只读 SQLite 快照：16 个成功新决策、112 次 cache hit、0 个失败/在途记录，符合首批 16 个任务×8 条轨迹的初始状态路由复用。每条完整候选数为 37，初始状态共自然选中 7 个不同技能；这是瞬时调用覆盖，不是性能评价，也不是完整轨迹覆盖。

记录的 requested/response model 均为 `Qwen/Qwen3-Embedding-0.6B`、provider cost=0，protocol hash 与冻结值一致。最后一条对应真实任务 `put two pencil in drawer.`、step_index=0，选中 `skillnet:alfworld-object-placer`。8 张 GPU 同时约 28,000 MiB 显存、65–100% 利用率，正在首批动作生成。仍无 stopped marker，更新指标数量为 0/150；这些证据只支持“正式 ALFWorld rollout 已启动”，不支持“完成第一次参数更新”。

运行日志中有 Qwen3.5 未列入 MFU FLOPs 估算表的告警，MFU 数值将为 0，不能把它当真实硬件利用率；另有既有 FSDP CPU 初始化及缺少可选 fast path 的提示。目前未观测到 traceback 或子进程失败。GPU 活动以上述 `nvidia-smi` 实际采样为准。

### 辅助执行命令留档

发布时 Working Directory 为独立导出 v4，使用同一 Python：

```bash
/mnt/workspace/users/wangyifan/.venvs/skillnet-phase12-20260917/bin/python -B -m phase3.publish \
    --directory /mnt/workspace/users/wangyifan/SkillScope-phase3-export-20260918-v4 \
    --receipt /mnt/workspace/users/wangyifan/SkillScope-phase3-upload-20260918-v4.json \
    --expected-parent 47e4435a9989c88157f2e11f374a28a5e7530670 --execute
```

合成预检在源码 Working Directory 执行；环境去除上述 API 凭据，CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7，HF/Transformers offline，OMP/OpenBLAS/MKL 各 1 线程。GNU `timeout --signal=TERM --kill-after=10s 1800` 包围调用 `phase3.run.command` 的 Python 父进程，后者以 open-x 日志启动：

```bash
/mnt/workspace/users/wangyifan/.venvs/skillnet-phase12-20260917/bin/python -u -B \
    -m torch.distributed.run --standalone --nproc_per_node=8 \
    -m skillnet_cohort.gpu_preflight \
    --model /mnt/workspace/users/wangyifan/model/Qwen3.5-4B \
    --embedding-router-model /mnt/workspace/users/wangyifan/model/Qwen3-Embedding-0.6B \
    --output /mnt/workspace/users/wangyifan/skill-RL/embedding-gpu-preflight-20260918-v1 --execute
```

这些命令是已执行操作的记录，不是重新运行指令。实验执行审计要求促使本轮分别记录发布身份、合成预检边界、正式启动命令与实际进度，未据此修改研究假设或增设正式实验。
