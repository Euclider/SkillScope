# Router 选型调整：转向 SkillRL 官方检索实现

记录日期：2026-09-18 UTC。本文是决策与待确认项留档，不是实验启动或完成记录；不覆盖先前 mini router setting、报告和账本。

## 当前选择与事实边界

用户先选择本地冻结 Qwen3.5-4B，随后在代码准备途中明确改为“选用 SkillRL 当前官方实现，便于对比”。本地 4B 生成式 router 方案立即搁置，不再进行候选比较或模型前向。结合上一轮候选表，当前按 SkillRL 的 **Embedding 模式 / Qwen3-Embedding-0.6B** 理解；官方同时存在的默认 Template 模式不能与其混称。

2026-09-18 通过只读 GitHub API 确认，aiming-lab/SkillRL 的 main commit 为 `8e66726ed866a4e0a7f053586a41022798192e6c`。核对官方 README、`agent_system/memory/skills_only_memory.py`、ALFWorld 环境管理器及启动脚本：

- Template 是默认模式：任务类别关键词匹配，无 embedding 模型。
- Embedding 是官方提供的可选模式：`SentenceTransformer(Qwen/Qwen3-Embedding-0.6B)`，对向量归一化后按余弦相似度排序；技能文本拼接 `title`、`principle`、`when_to_apply`。
- 查询只用 `task_description`，不是当前 observation/admissible actions/history；ALFWorld reset 时检索，整局复用。
- 官方 README Embedding 示例为 general top-6 + task-specific top-5，二者分开排名；不是统一 SkillNet-37 库的全局 top-6，也不是逐步 top-1。
- 检索模型不随 policy 的 GRPO 联合训练；它的通用 embedding 训练不等于 SkillRL 另训了一个 ALFWorld 状态技能 router。尚无本机证据表明先前生成式 router 性能差，不能把用户的担忧写成结果。

一手来源：

- [官方 README 固定版本](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/README.md#embedding-mode)
- [官方检索代码固定版本](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/agent_system/memory/skills_only_memory.py)
- [官方环境管理器固定版本](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/agent_system/environments/env_manager.py)
- [Qwen3 Embedding 官方模型说明](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

## 需要确认的协议差异

已向用户询问：

1. 连同 task-only / 每局固定 top-k 一起采用，优先贴近原生 SkillRL。这将改变原来的逐状态单技能暴露，需要重新定义统一 37 库中的注入数量、prompt 预算及多技能归因接口。
2. 复用官方 0.6B embedding 模型和余弦检索，保留当前状态查询 / 全 37 候选 / 每步 top-1。这是 SkillRL 检索机制的状态感知适配版，不得称为原生路由复现。

在回答前不自行决定 top-k、general/task 层级映射、状态输入或修改正式实验协议。无论哪种，后续 baseline 与本方法应共用同一路由规则；Phase3 编辑器仍为 o3，不在本次选型中变更。

## 本轮实际写入与保全

用户转向前新增了 `SkillRL/agent_system/memory/local_skill_router.py` 草稿和 `SkillRL/configs/alfworld_skillnet37_local_router.yaml`，并在 `skillnet_runtime.py` / `env_manager.py` 新增显式 opt-in 接口。转向后保留草稿，标为 SHELVED / UNVALIDATED，overlay 关闭，公开 factory 显式报错阻止启用；没有建立 Qwen3.5 本地 router profile。

这些是未完成草稿，不是已验证的 router 实现。没有加载模型、调用外部 API、进行 ALFWorld rollout、启动 RL、提交或推送 GitHub。本轮只读取了本地 Qwen3.5 模型文件做哈希核对；这不是模型加载或性能测试。

旧 mini 配置、缓存、费用账本、历史数据和报告保持原状；正式 cohort 仍未切换到新 router。切换时必须另注册新协议和新缓存，不能将已停止的 mini cohort 静默改装后继续。

## 用户确认后的追加记录（同日）

前述小节记录的是确认前停点。用户随后明确选择第 2 种：**复用官方 0.6B 检索器，保留逐状态 top-1，优先保持现有读出协议，论文注明适配版**。不再等待路由粒度确认。

据此新增 `skillrl_embedding_router.py`、固定版本 profile、完整本地文件哈希和显式 opt-in 配置；本机 Phase1–2 的训练、全 split 评估、anchor 采集和 O/P/N 使用同一个后端。Qwen3.5 生成式草稿继续禁用。旧 mini 配置/代码/缓存、Phase3 编辑器均未替换。新增 `sentence-transformers==6.0.1`，没有升级既有训练依赖；本地仅下载和加载了用户所选 0.6B 检索器，没有再做候选模型比较。

组件检查第一版发现首次导入 sentence-transformers 改变 Python RNG；失败记录保留，补上包含依赖导入的 Python/NumPy/Torch 隔离后，第二版两个合成状态和两次 cache replay 通过。两个状态均选择清洗技能，含已完成清洗的状态；没有调提示词、加规则、强制 coverage 或把接通测试写成算法优越性结论。

398 项离线回归通过；新的 seed=404 / policy 八卡 / router CPU preparation 已独立生成，静态检查通过。仍 `approved=false`、本地和外部调用预算均为 0，没有启动正式新 cohort、旧实验重跑或付费请求。CPU 组件检查不等于八卡训练共存或成功率验收。

本机新方案具体协议、模型 revision、哈希、输入字段、输出/费用记录、运行检查命令以及 Phase3 尚未迁移的边界，见 [详细留档](SkillRL/docs/experiments/skillrl-embedding-router-v1/README.md)。本轮未更新 GitHub 发布包，未提交/回滚任何文件；不要以原发布包声称已启用 embedding。

## 后续授权执行追加（2026-09-18 05:25 UTC）

用户随后授权先迁移并发布 Phase3，再直接启动本机 Phase1–2。Phase3 增长库适配和启动文档现已发布到 `Euclider/SkillScope-phase3` 的 `674dd36a54a83c5061af43f0f80b2ee3da924468`，877 个文件逐项核验。四分支共用相同 embedding 机制，o3 编辑器不变；本机没有启动 Phase3。

八卡合成前后向/优化器恢复与 CPU router 共存检查通过后，新 ALFWorld Phase1–2 于 05:21:02 UTC 启动（PID 903045，独立 embed06 cohort），不是重启付费 mini cohort。05:25 UTC 仍是首个 rollout、0/150 已完成更新，尚无性能结论。完整路径、配置与授权哈希、发布回执、启动命令和资源停止边界见 [新的执行记录](2026-09-18-embedding-phase3-publication-and-phase12-launch.md)。前述“未发布/未运行”段落保留为之前时间点的记录。
