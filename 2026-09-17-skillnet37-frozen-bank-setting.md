# SkillNet-37 冻结初始库：设置决策、实现与证据边界

日期：2026-09-17（UTC）。这是新的 bank 构建记录，不是旧 Phase2 的修订报告，也不是新实验结果。后续 summary 请先读本节停点、§1 决策表和 §8 待办。

材料标识（Material Passport）：来源为本次用户授权与固定版本公开资源；工作流使用 `academic-research-suite / experiment-agent` 的可复现留档规范；模式为资源冻结、适配代码与离线测试；验证层级为 **source-byte / manifest / unit-contract verified**；ALFWorld 实证效果、LLM 路由质量、跨 seed 读出有效性 **未在本轮验证**。

## 0. 本轮实际停点

已完成：SkillNet-37 原始快照、外部固定 hash 的 manifest、只读加载器、单技能正文适配器、bank-only 设置文件、104 项新测试及 10 项旧接口回归测试。上游 83 个文件逐字节一致；现有研究目录的 20,164 个文件内容保持不变。

未完成：外部 API router、环境/训练入口切换、Qwen tokenizer/prompt 预算审核、全库 PLACEBO 等长校验、新 seed/全任务 split 与 game 清单/更新窗口/预算。没有启动模型、ALFWorld、训练、评估或真实 LLM API 调用。只有公开源码/论文查询使用了网络。

旧 Phase2 停点仍以 [HANDOFF.md](HANDOFF.md)、[技术附录](HANDOFF-DETAILS.md) 和 [synthesis-v2 对应报告](2026-09-14-phase2-complete-analysis.md) 为准；不存在需要补跑的旧评估。本轮没有覆盖这些文件、更新旧完成标记或重新生成报告。

当前实际路径为 `/mnt/workspace/users/wangyifan/skill-RL`，代码子目录 `SkillRL/`；历史文档的 `/home/wangyifan/skill-RL` 路径不能直接用于当前运行。当前校验解释器 `/usr/bin/python3` 为 3.12.13，不是技术附录中的旧 conda 解释器。当前没有 `git` 可执行文件，不能将旧分支/HEAD/dirty 数量当成本轮重新实查结果；保全采用内容哈希对比，未执行任何 git 写操作。

## 1. 从讨论到本轮确认的决策

| 事项 | 当前决定与理由 | 状态 |
|---|---|---|
| 研究对象 | 仍以 Qwen3.5-4B 的策略更新读出为目标；后续从 clean 最小验证扩展到 ALFWorld 全六类任务、新 seed | 目标确认，实验协议未锁定 |
| 初始库来源 | 优先复用公开现成库，不在本轮收集轨迹/调用 LLM 自生成；控制初始资源差异并便于引用 | 已确认并实现 |
| 主库 | 固定 SkillNet ALFWorld 的全部 37 个技能及其附属文件 | 已冻结 |
| task/game 区分 | 所有 task/game 共用同一完整候选库；不按类型硬过滤，但仍可记录类型作分层统计 | 已实现库接口 |
| 正文暴露 | 完整候选目录不等于完整正文注入；未来每个环境状态步选择至多一个 ID，再给出该技能完整文本包 | 载荷适配器已实现，router 待接入 |
| 库自进化 | 当前 readout 对照期间不增加、删除、改写或禁用技能，不合并新轨迹知识 | 已实现只读约束 |
| router 独立性 | 不由正在更新的 policy 同时决定路由；拟采用固定的独立外部 LLM | 设计方向确认，具体配置待定 |
| 与 SkillRL 合并 | 37+44 的来源保留式 union 技术上可做，但会改变库组成/重叠/路由竞争；不作为当前主设置 | 本轮不合并 |
| 实验授权 | 当前授权为构建与离线测试，不是启动新训练/评估/付费 API | 未启动 |

这里的“冻结”应覆盖初始库、router 函数及暴露协议，而不是要求不同 policy 的完整轨迹必然选择相同技能。不同策略到达不同状态，固定 router 可以自然给出不同 ID；同一可见输入的可重放选择才是缓存/协议要保证的对象。由此不能宣称消除了所有路由相关影响，只能说排除了 router 参数随 policy 联动更新这一混杂来源。

## 2. 来源、版本与文件身份

公开来源：[SkillNet 仓库固定 commit 的 ALFWorld 目录](https://github.com/zjunlp/SkillNet/tree/5c472b36d2a435001fdae3bc8439886d8050645a/experiments/src/skills/alfworld)。

| 字段 | 值 |
|---|---|
| Bank ID | `skillnet-alfworld-37-5c472b36d2a4` |
| 上游 commit | `5c472b36d2a435001fdae3bc8439886d8050645a` |
| 快照时间 | `2026-09-17T11:52:04Z` |
| 上游归档 | [固定 commit tar.gz](https://codeload.github.com/zjunlp/SkillNet/tar.gz/5c472b36d2a435001fdae3bc8439886d8050645a) |
| 归档 SHA-256 | `851f28606c4964acf18edd82a8d0ec70b7238242fd35d4310274d230a9657c91` |
| 归档大小 | 749,151 bytes |
| 原文件 | 37 个主文件 + 45 个附属文件 + 1 个仓库根 MIT LICENSE，共 83 个 |
| 原文件大小 | 152,902 UTF-8 bytes，含 LICENSE；不含 LICENSE 为 151,839 bytes |
| manifest SHA-256 | `0767ff7578b1e997119a36b5f636fec6ebc1d0ac600ffb40b1e599065b7fe514` |
| 文件清单内容 SHA-256 | `a3fdd5f265f4a9926533686aef315163a6fb6b6f6ba5fc3f2153a9240dc662ab` |

完整逐文件 hash、原始名称/描述、每个技能所有文件及 payload hash 见 [manifest.json](SkillRL/memory_data/alfworld/skillnet37/manifest.json)。文件清单内容 hash 使用所有原文件按相对路径排序后的 `"{sha256}  {path}\n"` UTF-8 串；不是归档 hash。manifest 自身的外部固定 hash 由加载器独立检查，防止仅同时改正文与 manifest 内部 hash 就通过“冻结”校验。

`upstream/` 中正文没有任何修订、去重、补写或截断。`alfworld-temperature-regulator/references/action_spec.md` 上游没有末尾换行，本地也保留该细节。根 [MIT LICENSE](SkillRL/memory_data/alfworld/skillnet37/upstream/LICENSE) 原样保存。下载归档只在内存读取，不将 tar 中的任意路径解压到工作区。

## 3. 本地加载与载荷契约

实现：[frozen_skill_bank.py](SkillRL/agent_system/memory/frozen_skill_bank.py)；使用入口：[库 README](SkillRL/memory_data/alfworld/skillnet37/README.md)；机器可读设置：[setting.json](SkillRL/memory_data/alfworld/skillnet37/setting.json)。

1. `load_skillnet37()` 要求固定 manifest hash，验证 schema、83 个文件的完整集合、大小/哈希、来源字段、YAML 元数据、37 个技能及各自完整附属文件、渲染顺序与 payload hash。拒绝多余/缺失文件、符号链接、越界相对路径、正文或描述不一致，不静默修复。
2. 稳定 ID 为 `skillnet:<上游目录名/name>`；候选按原始技能名排序，目录为原始 YAML 的 name/description。`retrieve(task)` 对六类任务和无关键词任务均返回同一组 37 个候选，拒绝 top-k 截断及额外过滤参数。
3. `FrozenSkillBankMemory` 只提供候选/正文传递，不选择 ID。`selected_bundle(id)` 包装外部已经选定的 ID；`format_for_prompt()` 只接受至多一个选择，拒绝全库正文注入。保留完整 candidate ID 列表与 selected/injected ID 的区别。
4. `skillrl.raw_skill_package.v1` 格式为 `### Frozen Skill: <ID>`，接原始 SKILL.md，再按相对路径排序接全部附属文本，每个附属文件有独立标题。只加边界标题，不改原文空白。MIT LICENSE 不注入正文。`general_skills` 是旧接口的传输字段，不意味着把上游所有技能重新定义为“通用技能”。
5. 原始文件内容、技能对象以不可变 bytes/string/dataclass 保存；更新、删除、store、禁用候选与覆盖保存接口拒绝操作。这里没有修改 filesystem 权限：磁盘文件若被外部改动，下次加载会失败；已加载内存副本仍是原始内容。
6. 与现有 ORIGINAL/PLACEBO/NULL 的单技能载荷接口兼容性已通过测试：先在原始候选目录选定 ID，随后仅干预目标正文；非目标正文不变。此处测试的是 transport，不是 token 等长 PLACEBO 或环境行为。`None` 的空载荷表示已支持，但 router 是否允许弃权尚未决定。

完整载荷为 1,619–7,653 UTF-8 bytes。**该数字不是 token 预算**，完整 prompt 还包含目标、观察、历史和动作格式；没有据此断言旧 2,048-token 限制可直接复用。库中有原子动作、短流程、搜索/子目标及多步任务指导，粒度并不统一；作为文本建议注入，不把多步技能自动执行为环境宏动作。

上游不完美之处保留：例如 `temperature-regulator` 的主步骤与示例采用的处理动作并不完全一致；`object-retriever` 的主文件末尾说明需结合附属文件才能看到完整动作示例。这也是不丢掉附属文件、也不默默“修好”上游库的原因。保留来源身份不等于保证每条指导都正确。

## 4. 与上游 SkillNet 方法的关系及数据来源边界

本轮冻结的是上游 **ALFWorld benchmark 目录的 37 个技能**，不是整个 SkillNet 平台的全量技能集合。上游为各 game 使用相同目录，并将 `args.model` 传给 SkillModule；其代码默认 `gpt-4o`，不是单独预置的冻结 router 型号。[上游 ALFWorld runner](https://github.com/zjunlp/SkillNet/blob/5c472b36d2a435001fdae3bc8439886d8050645a/experiments/alfworld_run.py#L223)、[SkillModule](https://github.com/zjunlp/SkillNet/blob/5c472b36d2a435001fdae3bc8439886d8050645a/experiments/src/skill.py#L71)。

固定 commit 中，runner 在初始观察处检索相关技能，随后生成 overall procedure/程序进行执行；检索 prompt 对数量是通常不超过 5 的软要求。不能据此描述为“原实现每个环境步都从 37 个中重新选择一个”。我们拟做的独立逐步 router 与 readout 干预是新的控制协议，不是直接照搬其整个运行方法。[初始检索入口](https://github.com/zjunlp/SkillNet/blob/5c472b36d2a435001fdae3bc8439886d8050645a/experiments/alfworld_run.py#L149)、[检索提示](https://github.com/zjunlp/SkillNet/blob/5c472b36d2a435001fdae3bc8439886d8050645a/experiments/src/prompt_generator.py#L8)。

SkillNet 论文说明 benchmark-specific 技能来自 ETO 专家轨迹，并声明这些经历不与 seen/unseen 测试划分重叠。这是**作者声明**；当前公开技能条目未在本轮被逐项反查到原轨迹与我们的精确 game IDs，不能写成“我们已独立验证无泄漏”。[SkillNet v3 §4.1](https://arxiv.org/html/2603.04448v3#S4.SS1)。

复用公开固定库有利于控制初始资源差异与复查，但“同库”本身并不完成公平性论证。后续 baselines 还需共享 router、可见输入、载荷协议、任务清单、token/rollout 预算及标签获取规则；router 的 API 成本应单独报告。不能把我们的运行结果直接当作复现上游表格。

## 5. 原 SkillRL bank 的合并评估结论

现有 [claude_style_skills.json](SkillRL/memory_data/alfworld/claude_style_skills.json) 内容 SHA-256 为 `e8a953beac1809591fadf0d3509db5dea6e66b0fb56ddb573cf30e6d8879e909`：12 general + 32 task-specific = 44 个技能；另有 11 个 common mistakes，不混入这 44 的计数。

旧 clean 协议只提供 12+6=18 个候选；旧 router 除描述词匹配外还有手工阶段分数、task-specific prior 和稳定 tie-break，不应称为“纯字符串匹配”。旧评估中自然支持技能较少，与候选硬过滤/路由机制均可能有关，但不能只凭条目数确认因果，也不能保证换成 LLM 后 37 个都有支持。

两库可以通过不同命名空间做来源保留式 union，名义上为 81 条；这不是已完成的语义去重结果，也不代表 81 种独立能力。粒度、重复指导、长度与生成来源不同，合并会额外改变候选竞争，难以只归因于 readout。现有库元数据记载来自 223 条轨迹/o3，精确 split 谱系同样需核对。本轮 **不合并、不改旧库**；未来如需 Union-81，应作为单独 bank ID/协议/敏感性分析，保留来源且重新做泄漏、重复度与长度审计。

迁移到其他 benchmark 的可复用方案是“固定来源与文件清单 → 确定稳定 ID/描述 → 完整候选目录 → 独立固定 selector → 确定性单技能正文 → 统一干预与日志”，不是把 ALFWorld 的这 37 条内容直接认定为跨 benchmark 通用。当前适配器的 ID/来源约束针对 SkillNet，尚未实现任意来源 bank 的一键导入器。

## 6. 外部固定 router 的文献依据与待落实契约

此前检索中用于讨论 router 的依据如下；它们只支持工程路线与模型候选讨论，不提供“该型号在本设置最优”的证据，也不是新实验结果：

| 工作 | 可引用的具体设置 | 不应外推的结论 |
|---|---|---|
| [AgentSkillOS §4.1](https://arxiv.org/html/2603.02176#S4.SS1) | Opus 4.5 做构树/检索/编排，Sonnet 4.5 执行 | 不是 ALFWorld 的逐状态、固定单 ID readout 路由验证 |
| [SkillsVote v2 §8.1](https://arxiv.org/html/2605.18401v2#S8.SS1) | 独立 routing study 测 GPT-5.5/5.4/5.4 mini，xhigh；返回 10 个文件，不执行下游任务 | 不能据此将 xhigh 或 top-10 原封不动视为我们的已确认配置；小型号在大库也会退化 |
| [SkillRouter v5 §4–5](https://arxiv.org/html/2603.22455v5#S4) | 主系统为训练后的 0.6B embedding + 0.6B reranker | 主系统不是 Chat API 逐步 router；文中另有 GPT-4o-mini/5.4-mini 的 judge-reranker 基线，不能说这些型号只参与数据生成 |

先前讨论推荐的 `gpt-5.4-mini-2026-03-17` 只作为待确认候选记在 setting 中，`model_snapshot` 和 `provider` 仍为 `null`。本轮没有锁定 provider、模型、reasoning effort、价格、预算或凭据，没有真实调用；不能把推荐误写为已运行事实。

下一阶段建议冻结并测试以下契约（本轮没有实现）：

- 独立 router 只看 task、当前 observation、合法动作和协议限定的可见历史；不看 policy logits/权重、checkpoint 身份、干预臂、未来结果或效用标签。
- 基于 37 个原始描述选择稳定 ID；结构化输出只传 ID 给 actor，不附带外部 LLM 现写的动作计划/推理，以免引入额外指导信号。是否允许 `null`、调用频率/历史长度需预先确认。
- 在 ORIGINAL/PLACEBO/NULL 替换前完成选择；缓存键由 bank hash、router 模型快照、prompt/decoder 版本、规范化可见输入构成，不以 checkpoint/arm 区分等价输入。轨迹仍需保留 checkpoint/arm 作为结果索引，不能把这两种用途混淆。
- 记录请求/返回 ID、输入 hash、cache hit、token 用量/成本、延迟/重试和模型响应标识。仅设 temperature=0 不是服务端确定性的证明；缓存和版本记录提供可重放证据。
- 无效 ID、格式错、超时或预算超限应显式失败/隔离，不静默退回旧 lexical router。具体可重试条件和上限仍待冻结。
- 自然测量选中频率、每技能的任务/轨迹/决策支持及不确定性；不强制轮换以制造全库覆盖，不把零支持写成零风险。

## 7. 本轮执行、验证与保全

最新离线验证记录见 [verification.json](SkillRL/docs/experiments/skillnet37/verification.json)，包括环境、命令、失败边界、来源核对、内容保全与实现文件 hashes。

在代码根执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m agent_system.memory.frozen_skill_bank
PYTHONDONTWRITEBYTECODE=1 python3 -B -m pytest -q -p no:cacheprovider tests/skill_bank/test_frozen_skill_bank.py tests/phase1/test_step_skill_router.py tests/phase1/test_skill_mask.py tests/phase1/test_step_routing_summary.py
```

结果：CLI 通过；最终 `114 passed in 0.64s`（104 项新 bank 测试 + 10 项旧接口回归，不是 114 次实验）。覆盖每个技能的完整原文、六类任务相同候选池、配置一致性、只读接口、全库注入拒绝、O/P/N 载荷传递，以及 manifest/正文/附属文件/许可证/路径/文件集合篡改的失败检测。全部是 CPU 单元测试，没有模型加载或环境 rollout。

首次测试命令额外包含历史 `tests/phase1/test_first_invocation.py`，在收集阶段因 `ModuleNotFoundError: No module named 'pandas'` 退出 2；没有安装依赖或改动环境。最终成功命令明确排除了该文件。新测试直接覆盖了现有 `intervention_payload` 的传递语义，但不替代该历史文件的 pandas 指标测试，更不等于历史 233 项全量回归本轮全部通过。较早的 101/111 项通过是本轮增补断言前的结果，与最终 114 重叠，不能相加。

2026-09-17T12:07:14Z 再取固定 commit 公开归档，在内存验证归档 hash、PAX commit、目录全部文件集合及本地 83 个文件逐字节一致。

保全基线覆盖研究根内所有既有普通文件/文件符号链接，排除 `.git`；本轮新增的五组路径单独排除。修改前后均为 **20,164 文件、2,189,614,759 内容 bytes**，树内容 SHA-256 均为 `bd345838f2be8e2795afda512d6a1cb88eceb05e4c5a218e085f3d215b6e4145`。算法按相对路径排序，逐文件求 SHA-256，再聚合 `path + NUL + digest + LF`；文件符号链接记录 link target。该校验证明既有文件内容未变，不是 git 状态、权限/mtime 或未挂载历史资产的完整审计。没有提交、回滚、删除历史数据或覆写报告。

本轮只新增：`SkillRL/memory_data/alfworld/skillnet37/`、`SkillRL/agent_system/memory/frozen_skill_bank.py`、`SkillRL/tests/skill_bank/`、`SkillRL/docs/experiments/skillnet37/` 和本文。没有修改现有环境入口/旧 router/旧 bank/HANDOFF。

## 8. 下一步：先接 router，再锁全量实验

1. 确认 API 服务/固定模型快照、解码与调用预算；实现无网络 mock 测试、严格 ID 校验、缓存及完整日志。之后获准才做真实 API 小规模连通性验证，不先跑整批任务。
2. 用本地实际 Qwen3.5 tokenizer 对全部技能、完整 actor prompt 和 router 目录测长度，明确上下文/截断策略；逐技能验证 PLACEBO 等 token 长度。不得为适应旧长度上限默默删附属文件或改正文，必要时修订新实验预算/渲染版本并重新冻结。
3. 冻结全六类任务的具体 train/monitor/gold split 与 game 清单、新 seed、checkpoint/更新窗口、rollout/anchor/support 规则及资源预算；先锁预测规则，再看新 gold。不能自动复用旧 seed303/clean 窗口或将已看标签重新称为独立测试。
4. 单独建立新 cohort 目录及来源清单；旧 Phase1/Phase2 和最新 synthesis-v2 保持原样。37 个候选不代表所有技能都有自然锚点或方向支持，须按既定门槛报告不足/弃权。

## 9. 论文附录可复用草稿（尚非完整实验方法）

已实现部分可写：

> 我们采用公开 SkillNet 仓库中固定版本的 ALFWorld 技能集合，包含 37 个技能及 45 个附属文件。所有原始文件按字节保留，使用来源 commit、逐文件 SHA-256 和外部固定的 manifest hash 确定资源身份，不进行语义去重、改写或在线更新。所有 ALFWorld 任务类型共享同一候选集合；候选元数据与技能正文分离。载荷适配器在得到单个有效技能 ID 后，按确定顺序呈现该技能的主文档与全部附属文本，不执行附属文件，也不将完整候选库正文一次性注入 policy。

尚待实现/确认，不能现在用完成时态写入结果部分：

> 计划使用独立冻结的外部 LLM 依据可见环境状态选择技能，冻结其模型版本、输入协议、解码和缓存规则，并在正文干预前保持路由输入不变。路由的固定不要求不同策略访问不同状态时产生相同调用序列。后续将分别报告自然技能支持、外部路由成本、上下文预算和来源谱系的核查范围。

英文 bank-only 草稿：

> We freeze the public ALFWorld collection from SkillNet at commit `5c472b36d2a435001fdae3bc8439886d8050645a`, comprising 37 skill documents and 45 supporting files. Source files are preserved byte-for-byte, with per-file SHA-256 checksums and an externally pinned manifest digest. All ALFWorld task types share the same candidate catalog, without task-specific filtering, semantic deduplication, instruction rewriting, or online bank updates. Candidate metadata are separated from policy-visible payloads: after a single valid skill identifier has been selected, the adapter renders its main document and all supporting text in a deterministic order. Bundled resources are not executed. This setup reuses SkillNet's public skill resources but does not reproduce its complete retrieval-and-execution pipeline.

引用时需加 SkillNet 论文与固定 commit；不要把本轮离线测试当作 ALFWorld 成功率、路由优越性、无数据泄漏或 readout 泛化的证据。
