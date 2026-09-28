# Phase3 历史 v1 设置与附录留档

这是原始 HF/SDPA、seed404、gated D 协议记录。当前代码的拟议新协议见 [PHASE2_ALIGNMENT_V2.md](PHASE2_ALIGNMENT_V2.md)；本页不再是启动依据。

2026-09-18。用户确认四条独立累积闭环：SkillRL-style、gated D、−P、原始 +C_upd。初始库 SkillNet-37，全 task 共用完整库；后续允许新增、修改、合并、删除。相同 o3 编辑器，现统一采用冻结 Qwen3-Embedding-0.6B router，与 policy 独立。正式 Phase3 实验尚未执行，不能从工程测试推断效果。

## Router 迁移（替代 mini 的新实验协议）

固定官方 SkillRL commit `8e66726ed866a4e0a7f053586a41022798192e6c` 的 embedding 模型/归一化余弦检索机制，保留本研究逐状态 top-1，称为 state-aware adaptation，不是官方 task-only/每局 top-k 的原样复现。查询为 canonical JSON 的任务、当前观察、可用动作、最近两条可见 observation/action、step index；无 reward、C/P/D、game 分类或未来数据。技能文本使用官方 `_skill_to_text`，name 映射 title，description 映射 principle。不改写初始源 payload。

模型 revision `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`，profile SHA-256 `d1fc18a18537e65d4503d65d46187b8b63d21d9a884f6332262dbbb2dda778c8`。FP32、冻结参数、eval、RNG 隔离，不与 policy 共享权重或进行联合训练。全部当前 active skills 参与排名，同分按 canonical ID 顺序；新增/改写/删除/合并或 rollback 的新 bank manifest 创建新索引和新缓存，不继承旧决定。本地总调用限额跨所有库版本累计。四分支共用同一模型、query、技能文本映射和排序参数；缓存因分支/库内容不同而分开。

新设置归档为 `configs/phase3_setting_embedding_v1.json`。旧 `phase3_setting_v1.json` 与 mini 实现保留，只供旧 preparation 审计；不据结果在二者间切换。o3 编辑设置未改。本地路由的 token/前向时间不能记成付费 API 消耗，反之也不能把未完成本地请求的计算成本记成零。

## 设计与来源的区分

初始 37 个技能来自 SkillNet commit `5c472b36d2a435001fdae3bc8439886d8050645a`，本地 manifest SHA-256 `0767ff7578b1e997119a36b5f636fec6ebc1d0ac600ffb40b1e599065b7fe514`。保留原始正文、支持文件及上游许可证，不与旧 SkillRL bank 混合。库的 frozen 是初始条件，不限制 Phase3 的后续演化。

RL recipe 对齐 SkillRL 公开代码 `8e66726`：GRPO、lr=1e-6、16×8 轨迹/轮、150 轮上限、每5轮验证。Qwen3.5/HF/SDPA、microbatch=1、初始化与 block 执行是本研究适配，不声称论文原生配置逐 bit 重现。继承 entropy=.001、KL=.01/low_var_kl、invalid-action penalty=.1、weight decay=.01、clip=.2；没有 LoRA 或新 SFT。

上游原生更新器主要在失败轨迹上新增技能；本研究为了隔离归因来源，共用一个更一般的 o3 编辑器、操作集合和预算。它读取完整当前库，失败臂给失败轨迹，readout 臂给 top-k 技能实际出现的轨迹（可成功或失败）；失败不被断言为 skill 导致。相同证据条数、token cap、请求次数限制，没有额外不可计费的 summary LLM。故 baseline 名称应为 **SkillRL-style with a shared editor**。

参考入口：[SkillRL](https://github.com/aiming-lab/SkillRL)、[SkillNet](https://github.com/zjunlp/SkillNet)、[ALFWorld](https://github.com/alfworld/alfworld)、[o3 API 文档](https://developers.openai.com/api/docs/models/o3)。官方文档不保证用户指定第三方网关后端身份、别名版本或全部接口语义；响应模型、fingerprint、SDK、请求协议与 cache 应一起记录。

## Readout

固定5轮窗口，取首轮实际优化批次及实际优势 A，在同样 token/状态上计算 θ_start→θ_end 的原始／等 token PLACEBO 更新差，不累加五个逐轮 D。四次概率前向仅在内存；不取完整中间层、不做 Phase2 的后效用 gold。

与历史 Phase2 实现复用 `phase2.direction.token_signals` 的数学定义。D 为所有有效 response tokens 上 gated[-P]₊ 均值；零优势／gate失败仍进入分母。−P 和 +C 用同一支持池；C 指原始 C_upd，不是中心化 C。首次窗口按相同 checkpoint 重复前向的 L2 噪声 p95 冻结 τδ=max(1e-8,10×p95)，τC=0、ε=1e-12。记录离线/live chosen-logprob 最大误差；超预先登记容差就失败。

支持要求：当前 skill **内容版本**至少20个非零方向决策、4个game、8条trajectory。未支持技能不被填0参与排序；新增／改写／合并的版本不得继承旧分数。每窗记录三种分数和方向，主驱动分别固定，不能依据 gate/test 表现切换。

## 编辑与门控

Top-k=3 是候选上限，不强制修改3个。o3 medium reasoning；支持 ADD/MODIFY/DELETE/MERGE/NOOP。最大3 mutation units：add/modify/delete各1，N→1 merge为N+1，因此一次2→1合并已耗尽预算。目标绑定版本，删除是 tombstone，初始库与各快照不可覆盖。

固定 Seen gate 与 Seen evidence game 集不交叉；训练轨迹也可作编辑证据。候选与旧库在**同一当前策略、同一game×seed**配对评估，按预登记容差决定是否提交。候选拒绝不是 rollback；策略与优化器从不因 bank gate 退回。Seen是开发集；最终全140Seen的结果包含开发见过的games，不伪称完全held-out。134Unseen只用于预声明最终评估，不参与编辑、选分支、调阈值或rollback。

## 记录与统计

记录每次选了几个候选、实际触及／新增／删除几个技能、提案与接受后的各操作次数、mutation units、库大小、版本谱系、NOOP／支持不足弃权、候选拒绝／rollback分开、SR／repair／regression、逐episode/game/task与所有费用来源。API缺失usage和未对账请求不能按0计费；缓存只计真实新调用。Readout GPU前向token不是外部API token，应分栏报告。

主终点U150；四臂一致预算规则与停止规则，不能拿预算中断的早期endpoint和完整U150混比。条件允许时按game做配对差值及game级bootstrap，不把多个解码seed当作独立game或独立RLseed。单训练seed限制必须披露。

## 尚需在新服务器填写的运行数值

API总次数、gate每task样本数、gate容差、eval seeds、readout parity容差、GPU列表、磁盘限额均由准备清单冻结；模板保留空值并拒绝执行。它们不是已验证最优值。本文不替代运行时的manifest/hash或验收结果。
