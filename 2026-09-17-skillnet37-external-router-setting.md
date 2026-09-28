# SkillNet-37 外部 GPT-5.4 mini Router：实现与连通性留档

日期：2026-09-17 UTC。承接 [bank-only 设置记录](2026-09-17-skillnet37-frozen-bank-setting.md)。本轮用户确认模型为 **`gpt-5.4-mini`**、兼容端点为 `https://api.zhizengzeng.com/v1`，并授权一次小规模连通性测试。没有请求 `gpt-4o-mini`，没有训练或 ALFWorld rollout，没有重跑旧实验。

## 1. 实际停点

已实现独立外部逐步 router、严格单 ID 校验、完整 37 候选校验、持久化缓存、跨进程同输入互斥、共享 API 调用次数预算及脱敏审计。ALFWorld 环境管理器已支持显式选择 `external_llm` backend；训练轨迹和 first-invocation 续跑函数可保留外部路由元数据。旧默认路由及历史 Phase2 协议不变。

真实测试已成功：一次合成状态请求选中 `skillnet:alfworld-clean-object`，重复输入命中缓存。离线测试最终 **171 passed in 6.19s**。这不是 ALFWorld 成功率、router 准确率或全量评估结果。

仍未做：actor 全 prompt token 预算、37 个技能的 token 等长 PLACEBO 审核、新 seed/全任务 split/game 清单/训练窗口/资源预算与新 cohort 编排。因此本轮不直接启动全量评估。

## 2. 固定请求设置及 SDK

机器可读配置见 [router profile](SkillRL/configs/skillnet37_router_gpt54mini_v1.json)。

| 项目 | 本轮采用值 |
|---|---|
| 模型参数 | `gpt-5.4-mini`，不是 demo 的 `gpt-4o-mini` |
| 服务 | `https://api.zhizengzeng.com/v1`，用户指定的第三方兼容服务 |
| 接口 | `OpenAI(...).chat.completions.create(...)` |
| SDK | OpenAI Python `3.14.1`，本轮查询 PyPI 的最新稳定版本 |
| reasoning / sampling | `reasoning_effort="none"`、`temperature=0` |
| 输出预算 | `max_completion_tokens=128`、`n=1`、非流式、`store=False` |
| 输出结构 | strict JSON schema，唯一字段 `skill_id`，枚举完整 37 个 ID，不允许 null/额外字段 |
| 可见历史 | 最近 2 条 observation/action，支持训练历史的 `text_obs` 别名 |
| 输入上限 | system+user 正文 UTF-8 总大小不超过 65,536 bytes，超限报错、不截断；不是 tokenizer 预算 |
| 网络规则 | SDK `max_retries=0`、超时参数 30 秒、禁用 HTTP 重定向；没有自动重试或 fallback |
| bank manifest | `0767ff7578b1e997119a36b5f636fec6ebc1d0ac600ffb40b1e599065b7fe514` |
| system prompt SHA-256 | `51147edab1b4fb741b11e03be2db6ea5fdc25878b2fd2407af8ac2fa84323d11` |
| router protocol hash | `947890c5b48a04d3a28b3e0c02260f7e7db73cc4029a0d026ec4d4ac4d5bdffd` |

用户确认的是服务与模型；`none`/0、128 输出 tokens、2 步历史等是本轮明确落入 v1 的工程设置，不是文献证明的最优设置。正式实验可以在锁定前讨论，修改时必须产生新协议 hash，不能复用与其不匹配的旧 cache。

本轮使用 **OpenAI Docs** 核对请求字段。官方文档列出 GPT-5.4 mini 支持 Chat Completions、Structured Outputs 和 `none` 等 reasoning effort；Chat Completions 定义了 `max_completion_tokens` 和 JSON schema。官方说明不保证第三方服务逐项按相同语义实现，实际兼容性证据来自本轮请求与本地返回校验。[模型文档](https://developers.openai.com/api/docs/models/gpt-5.4-mini)、[Python Chat Completions 参数](https://developers.openai.com/api/reference/python/resources/chat/subresources/completions/methods/create)。

当前系统 SDK 原为 2.48.0；没有全局升级。新建独立环境：

```text
/mnt/workspace/users/wangyifan/.venvs/skillnet-router-20260917-sdk3141/bin/python
```

该环境为 Python 3.12.13、继承已有 system site-packages，安装 SDK 3.14.1 及其依赖，并为配置/CPU 接入测试安装 OmegaConf 2.3.0、PyYAML 6.0.3。新版 SDK 使用 httpx2；实现使用 SDK 提供的 `DefaultHttpxClient`，没有把旧版 httpx 客户端强塞给新 SDK。依赖入口见 [requirements-router.txt](SkillRL/requirements-router.txt)，实际版本清单见本轮 verification。未声称此环境已满足完整训练栈或 GPU 兼容性验收。

首次 uv 命令因默认 `/opt/uv/cache` 无权限失败，随后使用 `uv --no-cache` 完成安装；没有改目录权限或修改系统缓存。

## 3. 路由与缓存契约

- 每次都基于未修改的全部 37 个名称/描述，不以 task/game 硬过滤，不接受被禁用、被改写或已单选过的候选包。
- 发给外部服务的动态输入仅有 task、当前观察、合法动作、有界可见历史和 step index；candidate bundle 中额外出现的 checkpoint/arm/reward 字段不进入请求。历史中的额外 metadata 也被剥离。
- 外部返回仅作为 ID 使用。正文始终来自已校验的本地 SkillNet 包，不把外部 LLM 的 rationale、动作计划或新建议转交给 policy。原始 bank 文件、描述和 payload renderer 未改变。
- 候选确定和外部选择发生在 ORIGINAL/PLACEBO/NULL 干预前。相同可见输入共用缓存，缓存键不含 checkpoint 或干预臂；不同轨迹到达不同状态仍可得到不同 ID。
- 缓存键由完整请求协议 hash 和规范化可见输入 hash 构成。协议包括 model、endpoint、SDK/decoder、prompt、bank、完整候选和输出 schema。保留动作列表顺序，不把“语义上相近”的状态强行当作同一输入。
- SQLite 保存协议、可见输入、已验证 ID、响应标识、用量、延迟和成功/失败状态；每个输入的文件锁防止并发重复付费请求。不同输入可并行，API 次数在发送前以数据库事务原子预留，失败/进程中断也占用预算。同一 cohort 的 baseline/arm/checkpoint 应共享同一个本地 cache。
- 命中缓存时 `api_calls_this_step=0`、本步 API tokens/延迟为 0，原始响应的用量/延迟单列为 `source_decision_usage` / `source_decision_latency_ms`，不得重复计费。延迟字段不是整个路由函数的本地 wall time。
- 非法 ID、额外 JSON 字段、重复 JSON key、截断/refusal/tool call、网络错误、损坏 cache 和预算耗尽均失败，不降级到 lexical router，不临时换模型或放宽响应格式。重新手工调用消耗新的授权次数；程序没有隐藏重试。

缓存实现面向同一主机的本地文件系统；未验证 NFS/多机分布式锁语义。缓存 checksum 用于意外损坏检测，不是对能修改数据库和 checksum 的管理员进行密码学认证。请求参数固定加缓存重放，也不等于证明托管模型的服务端推理逐 bit 确定。

## 4. 代码接入与使用边界

新模块：[external_skill_router.py](SkillRL/agent_system/memory/external_skill_router.py)、[router_cache.py](SkillRL/agent_system/memory/router_cache.py)、[skillnet_runtime.py](SkillRL/agent_system/memory/skillnet_runtime.py)。

ALFWorld 管理器的 opt-in 覆盖配置见 [alfworld_skillnet37_external_router.yaml](SkillRL/configs/alfworld_skillnet37_external_router.yaml)。它不是完整训练配置：`cache_path` 必填；默认 `max_api_calls=0` 只允许缓存重放。启动实际任务之前需填入新 cohort 的 cache 路径并明确授权预算。旧配置未指定 backend 时仍使用原 `phase` 路由。

环境管理器在 initial 和每个仍活动的后续状态调用新 router，终止状态不额外请求；外部模式不沿用旧 top-k 截断参数。即便 actor history_length=0，也正确注入选中的技能；router 自己的两步历史窗口单独固定，正式实验需同时归档 actor 的历史规则。

通用读取入口（示意，不会在构造时请求 API）：

```python
from agent_system.memory.skillnet_runtime import create_skillnet37_runtime

memory, router = create_skillnet37_runtime(
    cache_path="/path/to/a/new-confirmed-cohort/router.sqlite3",
    max_api_calls=0,  # 无新调用授权；正式运行需换成明确确认的总次数上限。
)
candidates = memory.retrieve(task_description)
routed = router.route(
    candidates,
    task_description=task_description,
    current_observation=observation,
    admissible_actions=admissible_actions,
    history=visible_history,
    step_index=step_index,
)
payload = memory.format_for_prompt(routed)
```

已有 first-invocation `run_branch` 可接收上述 memory/router，须传 `router_general_top_k=37`，在选择后继续应用 O/P/N；本轮增加了其路由元数据归档。**历史 Phase1/Phase2 CLI 默认仍建立旧 memory/router**，没有暗中修改旧 cohort 的启动语义。新全量 cohort 的正式编排入口还需在冻结新协议时显式使用 factory；不要直接重启旧 `phase2.evaluate` 充当新实验。

旧 bank-only 校验器的 `router_implemented:false` 是该模块自身“不实现选择”的范围标签，不会探测本轮新增扩展；当前外部模块状态应看本文和新 profile/smoke 记录。原 bank-only 设置/verification 作为历史阶段证据保留原样。

## 5. 真实测试证据

入口：[scripts/test_skillnet_router.py](SkillRL/scripts/test_skillnet_router.py)。只发送公开技能描述和手工构造的状态，不读取旧轨迹/权重/环境。脚本每个 cache 最多 1 次真实 API 尝试，随后重放相同输入。

```bash
# 仅当另一次连通性测试已获授权时使用新 cache 路径；不要重建/覆写本轮证据。
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/users/wangyifan/.venvs/skillnet-router-20260917-sdk3141/bin/python -B -m scripts.test_skillnet_router --cache /path/to/new-smoke/router.sqlite3 --prompt-api-key
```

本轮实际 cache 在 `SkillRL/docs/experiments/skillnet37-router-v1/live-smoke/router.sqlite3`，响应时间戳 `2026-09-17T13:30:50.371802+00:00`。首次结果：[live-smoke-result.json](SkillRL/docs/experiments/skillnet37-router-v1/live-smoke-result.json)；后续跨进程无凭据缓存重放：[cache-replay-result.json](SkillRL/docs/experiments/skillnet37-router-v1/cache-replay-result.json)。

| 观测 | 值 |
|---|---|
| 真实 API 请求次数 | 1，失败 0，无自动重试 |
| 合成状态 | 目标放置干净杯子；当前持有尚未清洗的 mug 1、位于 sinkbasin 1，合法动作包含 clean |
| 候选 / 选择 | 37 / `skillnet:alfworld-clean-object` |
| 选择后的本地完整正文 | 4,040 UTF-8 bytes，不代表 actor tokens |
| 请求 / 返回 model 字段 | 均为 `gpt-5.4-mini` |
| 实际请求延迟 | 9,142.901 ms，单次观测，不是均值或吞吐测量 |
| 服务返回 token 用量 | input 4,269，output 19，total 4,288，reasoning 0 |
| 费用 | 服务响应未给金额，记 null；不拿官方 API 价目冒充第三方实际收费 |
| 后端 fingerprint | null |
| 最终缓存审计 | 1 个成功决策、1 次 API attempt、5 次 cache hit（含本次内和后续两次进程重放） |

输入 token 包括完整候选描述、schema 等请求内容；此处没有据单个示例预测全量总成本。此次输出符合这个简单状态的清洗需求，但只能证明连通性、本地协议校验和缓存路径工作，不能估计 router 的普遍选择质量。

特别边界：服务返回的是模型别名，没有日期快照和 system_fingerprint。本轮没有调用日期型模型 ID，也无法独立验证网关实际后端权重。论文应写“冻结请求协议与缓存选择，服务请求名为 gpt-5.4-mini”，不要写“已独立证实后端 GPT 权重快照全程固定”。正式全量比较前可向服务方确认固定快照/路由保证；模型名支持某接口不自动证明其身份/解码行为。

## 6. 密钥与文件保全

运行只读取专用 `SKILLNET_ROUTER_API_KEY`，不会自动借用共享 `OPENAI_API_KEY`，避免把另一个服务的凭据发到此网关。CLI 通过 getpass 无回显输入，仅存于该进程内存；没有写 `.env`、配置、源码、报告或请求正文。异常不记录 raw response body/headers，也不打印原始 SDK exception 文本。

这把密钥曾出现在用户聊天中；正式实验建议轮换，并通过运行环境或 secret manager 注入，不在附录或 shell 命令参数中粘贴。

本轮对既有源码只作三处局部接入：[env_manager.py](SkillRL/agent_system/environments/env_manager.py) 增加显式 backend，[archive.py](SkillRL/phase1/archive.py) 与 [eval_first_invocation_utility.py](SkillRL/phase1/eval_first_invocation_utility.py) 保留路由审计字段。没有替换原 FrozenStepSkillRouter、SkillRL bank、SkillNet 原始文件、旧报告/旧配置或 HANDOFF。未提交、回滚或删除数据。

三份允许修改的代码之外，保全基线为 20,251 个既有文件、2,189,820,796 bytes，内容树 SHA-256 为 `4a91e91726ed45412128c48c6149c0246f450c943a8249e81fff84d4102f39a8`；排除 `.git` 和本轮明确新增的路径，修改前后对比见 verification。没有 git 可执行程序，本轮不声称重新核验了 branch/HEAD/index，也不将未挂载历史大权重列为已验证。

## 7. 离线验收与论文附录补充

最终验证：[verification.json](SkillRL/docs/experiments/skillnet37-router-v1/verification.json)。171 项 = 55 项新 router/接入测试 + 104 项前一阶段 bank 测试 + 12 项旧接口/归档回归。覆盖 API 请求结构（最新版 SDK 的 mock transport）、有效/无效返回、完整候选与只读身份、缓存持久化/并发/预算、O/P/N 先选择后干预、训练/评估历史键归一化、环境管理器活跃步与终止状态、日志传递及旧默认行为。

全部 CPU 测试只用 mock client/fake environment；没有真实 ALFWorld 或 policy 前向。没有运行依赖 pandas 等完整实验栈的历史全量测试；171 与历史 233 不是同一测试集合。本轮离线命令：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/users/wangyifan/.venvs/skillnet-router-20260917-sdk3141/bin/python -B -m pytest -q -p no:cacheprovider tests/skill_router tests/skill_bank tests/phase1/test_step_skill_router.py tests/phase1/test_skill_mask.py tests/phase1/test_step_routing_summary.py tests/phase1/test_trajectory_archive.py tests/phase1/test_training_step_archive.py
```

可用于附录的实现描述，正式提交前补上全量实验协议及后端冻结证据范围：

> 我们将技能选择与被更新策略分离：独立外部 GPT-5.4 mini router 在固定 SkillNet-37 候选目录上，仅依据可见任务、当前状态、合法动作和最近两步历史返回一个技能 ID。策略接收该 ID 对应的本地原始技能正文，不接收外部 LLM 新生成的推理或计划。我们固定并归档 SDK、请求参数、prompt、bank 与输出 schema，将经严格校验的选择按请求协议和可见输入进行持久化缓存；O/P/N 正文干预发生在选择之后。不同策略访问不同状态时可以产生不同调用序列，而相同输入的选择通过共同缓存重放。第三方服务只返回模型别名，本轮未独立验证其后端权重快照。

> Skill selection is separated from the policy being updated. An external router requested as `gpt-5.4-mini` selects a single identifier from the frozen SkillNet-37 catalog using only the task, current observation, admissible actions, and the last two visible observation–action pairs. The policy receives the corresponding canonical local skill package, not newly generated router advice. Validated decisions are cached by the frozen request protocol and visible input, before payload interventions. The compatibility gateway reports a model alias but no backend fingerprint; the present validation establishes API compatibility and cache replay, not independent verification of a fixed backend weight snapshot or downstream evaluation quality.
