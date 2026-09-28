# SkillRL embedding 逐状态适配版：实现与证据边界

日期：2026-09-18 UTC。状态：本机 Phase1–2 已接入并完成组件验收，未启动 RL。此目录是新协议记录；不是对已停止 mini cohort 的重跑、续跑或报告替换。

## 已确认的 setting

用户选择：复用官方 0.6B 检索器，保留逐状态 top-1，优先保持现有读出协议，论文明确注明适配版。

- 模型：`Qwen/Qwen3-Embedding-0.6B`；HF revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`。10 个运行文件的 SHA-256 在 [router profile](../../../configs/skillnet37_router_qwen3_embedding_v1.json)；初始化前验证完整文件集和字节哈希，之后检查文件状态漂移。
- SkillRL 上游 commit：`8e66726ed866a4e0a7f053586a41022798192e6c`。[官方检索代码](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/agent_system/memory/skills_only_memory.py) 使用 SentenceTransformer、归一化向量及余弦检索。
- 初始库：原始 SkillNet-37，所有 game/task 共用全部候选，不按类别、阶段或关键词预筛选。每个有效状态选一个 ID，再注入该技能的原始完整 payload。不生成额外建议，不强制轮换技能以提高覆盖率。
- 技能检索文本：复用官方 `_skill_to_text`；SkillNet `name → title`、`description → principle`，`when_to_apply` 为空。不改写源技能，不把 O/P/N 替换文本用于检索。
- 查询：固定 canonical JSON，字段为 task description、当前 observation、admissible actions、最近两条可见 observation/action、step index。不读取 reward、success、task-type/game 标签、policy checkpoint、未来轨迹、C/P/D 或 gold。没有额外 query prompt，与官方不传 prompt 参数的调用方式一致。
- 排序：FP32、L2 归一化向量点积、全库 top-1；精确平分时按冻结 bank 的 canonical 顺序取第一个。37 项完整分数逐步记录。
- 检索器参数冻结、eval/inference mode，不共享 policy 权重、不参与 GRPO，不在线训练。保护 Python/NumPy/Torch RNG，包括首次依赖导入，避免改变同步 policy 的采样流。
- ORIGINAL/PLACEBO/NULL 在选中 ID 后应用，既有归因单位和读出数学定义不变。相同可见输入在不同 checkpoint、O/P/N 分支共享同一协议缓存；状态因不同动作发生变化时重新评分。

论文必须区分：官方 Embedding 模式是 task-only 查询、每局检索并复用 general/task-specific top-k；官方 README 示例为 general top-6 + task top-5。本实现改成可见状态查询、统一 SkillNet-37、逐状态 top-1，应称为 **SkillRL embedding retrieval 的 state-aware adaptation**，不能称为原生 SkillRL router 的完整复现。官方检索器本身也不随 policy 联合训练；本次没有新增 ALFWorld router 训练。

## 接入位置与不变项

实现入口为 `agent_system/memory/skillrl_embedding_router.py`，运行后端名 `skillrl_embedding_state`。`skillnet_cohort` 的 training/segmented_training、seen/unseen performance、anchor collection 和 Phase2 O/P/N 工厂使用同一 profile，不会一边 embedding 一边 mini。

保留 GRPO、lr=1e-6、16×8 轨迹/更新、PPO minibatch=128、microbatch=1、150 更新上限、每 5 更新验证、seed=404、固定 5 更新读出窗口。八卡 policy 配置继续使用原有 CPU shard init；未改变 loss、sampling、controls 或 readout 公式。

新 preparation 为 [preparation-s404-8gpu/manifest.json](preparation-s404-8gpu/manifest.json)，SHA-256 `3b731d125adc0731b193f59821dea7f5dd0d54d6d94d17fb908deeafdb35c7fe`。它登记 policy=8 GPU、router=CPU（不抢占八卡 policy 显存）；这不是八卡共存验证。旧 preparation 默认仍是 mini，禁止把旧启动命令当成新协议使用。

新 profile SHA-256：`d1fc18a18537e65d4503d65d46187b8b63d21d9a884f6332262dbbb2dda778c8`。当前 CPU router protocol hash：`2a5128c53f05c9e09f2cc951c0fa3694f271fa2289306098b2db5d58e87f7f72`。协议包含模型、文件哈希、包版本、设备、文本映射及输入规则；不把可迁移的本地模型路径或 policy checkpoint 当成 router 身份。

## 验收与限制

- [preflight-v1.json](preflight-v1.json)：静态检查通过，`STATIC_READY_EXECUTION_LOCKED`。完整 game 文件清点为 train=3553、seen=140、unseen=134；这是静态 inventory，不是运行覆盖率。
- [smoke-cpu-20260918-v1/failure.json](smoke-cpu-20260918-v1/failure.json)：第一次组件检查失败，原因是首次导入 sentence-transformers 改变 Python RNG。该失败记录及两条本地决策全部保留，没有覆盖。
- [smoke-cpu-20260918-v2/report.json](smoke-cpu-20260918-v2/report.json)：补齐 RNG 隔离后，两个合成状态各做一次本地决策并各重放一次；trainable parameters=0；Python/NumPy/Torch CPU RNG 保持。CPU 测试中的 CUDA RNG 检查没有实际 CUDA 设备参与，不能解读为 GPU 测试。网络连接被测试脚本禁用。
- 第二次组件检查使用 CPU、4 Torch threads：第一次约 21.45 秒（含导入、文件校验、模型加载、37 技能建索引），查询前向约 0.337 秒；第二次约 0.285 秒，查询前向约 0.283 秒。每条 query=124 tokens，首次索引=2968 tokens。样本仅两条且无 policy 共存，不能外推实际 rollout 吞吐、完整状态长度或训练 wall time；正式 supervisor 当前使用 1 线程环境，亦不同于本组件测速。
- 两个合成状态都选中了 `skillnet:alfworld-clean-object`，包括已经清洁完成、可以放入柜子的第二个状态。完整余弦分数不同，但 top-1 未变。它证明接入和复用可运行，不证明下一步选技正确、优于 mini、自然覆盖更多技能或提高成功率。没有按该结果修改查询、加规则或调参。
- [offline-tests-v1.xml](offline-tests-v1.xml)：398 项通过（70.10 秒），涵盖离线路由、库、cohort、readout 和 Phase3 兼容回归；这些测试不进行真实 ALFWorld/RL/API 实验。测试命令及证据哈希见 [verification-v1.json](verification-v1.json)。

API router 消耗为零不等于本地计算免费。实际记录含 local attempt、冷启动索引/query token、latency、完整 scores、cache hit、源决策审计及 backend/model/revision。为了兼容已有轨迹，载体字段仍名为 `skill_router_api`，其中明确 `api_calls_this_step=0`。缓存数据库的历史列名 `api_attempts/max_api_calls` 对此协议表示本地查询尝试；对外使用 `router.stats()` 的 `local_attempts/max_local_calls`，不可误记成付费调用。缓存命中计入的本步 token/前向次数为零，源决策另存，避免双计。

## 本机安装与只读检查

本机 isolated venv 已新增 `sentence-transformers==6.0.1`，采用 `--no-deps`，没有升级既有 Torch/Transformers；profile 要求 torch=2.11.0、transformers=5.10.4、tokenizers=0.22.2。冻结模型在 `/mnt/workspace/users/wangyifan/model/Qwen3-Embedding-0.6B`。其他机器必须先匹配完整环境；不要仅凭一个 requirements 文件假定 GPU/RL 栈可用。

以下只读检查不会新建实验目录或发起路由请求（从 SkillRL 代码根目录运行）：

```bash
PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/mnt/workspace/users/wangyifan/.venvs/skillnet-phase12-20260917/bin/python -B \
-m skillnet_cohort.preflight \
--preparation docs/experiments/skillrl-embedding-router-v1/preparation-s404-8gpu/manifest.json \
--run-root artifacts/phase12/skillnet37-qwen35-embed06-s404-8gpu-v1
```

正式启动之前仍需新的 scoped authorization、明确的本地 router 总调用上限及既定存储限额，并验证新后端与八卡 forward/backward/checkpoint restore 的实际配合。当前 `approved=false`、`router_max_local_calls=0`、`router_max_api_calls=0`；没有注册已批准的新许可，也没有创建上述正式 run root。无 API key 要求；不得复制旧 mini 付费预算到本地调用预算。超限或错误停止，不退回 mini/关键词 router。

Phase3 的 o3 编辑器未改。本次只完成本机冻结 37 库 Phase1–2 接入；现有 Phase3 增长库 router 与已发布 GitHub 包仍是旧版 mini，尚未改装或重新上传。Phase3 开跑前必须另做同一 embedding 适配，按各分支当前 bank 快照重建索引，并确保 SkillRL、D、−P、+C 四条分支共用完全相同的检索规则。当前 factory 明确拒绝非原始 SkillNet-37 bank，不能把它直接假装用于增长库。

本轮未启动 RL、未跑真实 ALFWorld、未发起付费调用、未提交/回滚/推送、未删除旧证据。被用户否决的本地 Qwen3.5 生成式 router 草稿保留且禁用；不能当作本方案的可用后端。
