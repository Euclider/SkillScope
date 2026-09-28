# Phase3 验证状态

## 2026-09-25 Phase2 对齐 v2（当前代码）

- 当前新协议见 [PHASE2_ALIGNMENT_V2.md](PHASE2_ALIGNMENT_V2.md)；下文是历史 v1 验收，不可当作 v2 的 GPU/网关准入。
- 本机离线 `tests/phase3 tests/skill_router tests/skill_bank tests/experiment_settings tests/skillnet_cohort/test_vllm_backend.py`：设置 `ALFWORLD_DATA=/mnt/workspace/users/wangyifan/skill-RL/data/alfworld` 后 **336 passed**；只验证工程契约，不包含正式 RL、完整 vLLM rollout、API 编辑或跨服务器运行。
- 只读核查 404/505 已封存 scalar rows 上的新主指标和 game 聚合：分别 18/19 个技能，最大绝对误差 `1.67e-16` / `2.22e-16`。没有读取新效用标签作为在线读出输入；404/505 指标选择仍属探索性。
- 尚无 Phase3 v2 八卡原生/vLLM 专项通过回执、真实 ALFWorld gate/final 后端验收、chosen-token 跨后端 parity、o3 网关 schema/usage 真请求验收或正式四臂性能结果。运行配置中磁盘限额仍为 `null`，准备验证将拒绝执行。

本文件区分测试与实验，不声明 Phase3 闭环已有效。

- 历史 mini 版本相关离线回归：541 passed in 55.28s（Phase3、SkillNet cohort、router、setting、bank、Phase1/2测试）。另有先前423项及修改后190项子集通过；这些是工程测试，不是RL效果实验。
- 确认 GitHub 账号对用户指定目标仓库有写权限；发布是否完成以远端 commit 和本地发布回执为准。
- 首次8卡验收因独立测试误传Ray专用设备标志失败；未做RL更新。第二次揭示32GiB卡的完整FP32双拷贝初始化OOM。失败日志保留在本机验收目录，不混入训练结果。
- 2026-09-18 已通过8×RTX5090实际原生验收：8个rank均完成4096+512长度前后向、Adam更新、checkpoint保存、参数／Adam／RNG逐值精确恢复，以及恢复后的HF生成。峰值 allocated=29.442434 GiB、reserved=30.234375 GiB。
- 32GiB适配含 opt-in CPU参数分片初始化与逐参数恢复原生分片（避免默认恢复同时驻留两份完整FP32参数）；8卡测试核对全部权重／优化器张量及恢复后的RNG，不改变精度、原生checkpoint格式或优化配方。
- 该验收每rank仅一个合成microbatch，不是完整128行优化批次；没有Ray、ALFWorld rollout或真实API。第三次调试曾在默认原生恢复双份FP32峰值处OOM，修复后第四次全部通过。所有调试均保留日志，没有重跑历史实验。
- o3 第三方网关未在本次实现中做真实编辑请求；离线测试使用明确标注的fake client，不能作为服务连通性证据。
- 没有执行四条Phase3 RL，未生成成功率对比结果。完整闭环、真实状态4096-token覆盖、长时间运行吞吐仍需运行验收。
- 失败请求不自动重试。未完成子进程／不完整checkpoint需人工对账，尚未声称支持任意崩溃点全自动恢复。
- 导出仅支持已确认GRPO路径。本机已有的 `verl/workers/critic/dp_critic.py` 含未完成赋值语法，GRPO不启用该critic；为保留原工作区而不修改它，并在导出清单中排除该可选模块。不承诺此包可切换为PPO/critic训练。
- 本地541项回归中包含依赖本机历史/Phase1–2准备资产的测试；这些资产与对应测试不放入可迁移包。请使用README列出的Phase3/router/bank/setting离线测试集合，不把本机资产验证误称为另一台机器已验证。

## 2026-09-18 embedding 迁移版

- 当前可迁移测试集合 `tests/phase3 tests/skill_router tests/skill_bank tests/experiment_settings`：314 passed in 12.44s。此前同轮广泛本机回归416项通过；后来补充了 Phase1–2/Phase3 编码与排名一致性测试，已包含在314项中。
- 新测试覆盖四分支统一 query/encoder/top-1，ADD/MODIFY/DELETE/MERGE 后完整新库重建索引，版本缓存隔离，跨库/并发实例共用本地调用限额，失败不得自动重试，以及发布前验证 tree、非强制快进、拒绝陌生 parent。
- 既有两个 CPU 实权重合成状态已验证加载/冻结/缓存/RNG 隔离；都选择了清洗技能（含已清洁状态）。不据此宣称选技准确率、覆盖率或成功率提升。本次增长库路径用明确标记的 fake vectors 做离线契约测试，未调用 o3、未启动 Phase3 RL。
- 运行环境新增 SentenceTransformers 6.0.1，现有180包经 `uv --no-cache pip check` 检查兼容；没有更新既有 Torch/Transformers。新服务器仍需按 README 安装并自行验收。
- GPU preflight 新增可选 `--embedding-router-model`，在八卡 actor/ref/optimizer/restore 期间保留 rank 0 的 CPU 检索模型，并在前后做两条合成查询。发布时新组合验收尚未执行；不能把上文历史八卡验收冒充该组合已通过。本机发布后的启动验证另留实际回执。
- 新模板只请求 editor key；mini 仅作为历史显式后端保留，无自动 fallback。上传包不含任何凭据、模型权重、运行数据库或实验轨迹。
