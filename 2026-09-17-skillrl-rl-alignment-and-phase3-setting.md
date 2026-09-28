# SkillRL 公开代码 RL 参数对齐与 Phase3 共用编辑器设置

日期：2026-09-17 UTC。承接 [冻结初始库](2026-09-17-skillnet37-frozen-bank-setting.md)、[外部 router](2026-09-17-skillnet37-external-router-setting.md) 与 [历史交接](HANDOFF.md)。用户授权本轮对齐、归档 RL 配置；随后明确 Phase3 共用外部编辑 LLM，可以新增、删除或修改技能。本轮没有启动训练、ALFWorld rollout、效用评估或编辑 API 请求。

## 1. 实际停点与材料身份

- Origin Skill: academic-research-suite / experiment-agent。
- Origin Mode: plan；按用户明确授权实现独立配置与离线检查，不执行实验。
- Origin Date: 2026-09-17。
- Verification Status: UNVERIFIED（指实验效果）；来源已核对、配置及 CPU 测试已检查，不等于训练或性能验证。
- Version Label: skillrl-public-code-alignment-v1。

已新增 opt-in [Hydra 配置](SkillRL/verl/trainer/config/alfworld_skillnet37_skillrl_v1.yaml)、[机器可读设置](SkillRL/configs/skillrl_public_alignment_v1.json)、[只读检查器](SkillRL/scripts/inspect_skillrl_alignment.py) 和 [离线测试](SkillRL/tests/experiment_settings/test_skillrl_alignment.py)。旧配置、启动器、HANDOFF、报告和已有未提交文件保持原样。新 cohort 不沿用旧 resume 路径，未选新 seed，未另做 SFT。

Phase3 当前仅锁定设计方向：**共同冻结初始 SkillNet-37；两臂共用编辑器及增删改权限；主要比较归因／证据选择方式。** 可变库、编辑调度与新增技能的 readout 支持规则尚未实现，不应把设置记录当成运行完成记录。

当前挂载研究根为 `/mnt/workspace/users/wangyifan/skill-RL`；历史 `/home/wangyifan/...` 路径只是历史记录，不能直接复制执行。历史 Phase2 已完成 synthesis-v2，没有需要补跑的旧实验。

## 2. 对齐基线及来源边界

选择官方 README 指向的 `examples/grpo_trainer/run_alfworld_skills.sh`，即 validation-failure SkillBank 变体；不是无技能 `run_alfworld.sh`，也不是 `run_alfworld_skills_train_update.sh`。固定上游 commit：`8e66726ed866a4e0a7f053586a41022798192e6c`。[官方 README](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/README.md)、[固定版本运行脚本](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/examples/grpo_trainer/run_alfworld_skills.sh)。

本轮核对论文 v1 的 §3.3、§4.1、附录 B.1/Table 4，以及公开配置、训练器、actor、updater、reward 实现。九份上游文件的 URL、字节数、SHA-256 与部分代码定位见 [upstream_sources.json](SkillRL/docs/experiments/skillrl-alignment-v1/upstream_sources.json)。这是源码／论文证据，不是对其报告结果的复现；本地已有修改不能冒充未经修改的官方实现。[论文原文](https://arxiv.org/html/2602.08234v1)。

论文和可执行脚本不完全一致，因此本文用语是“**对齐 SkillRL 公开代码 recipe，另列研究适配**”，不是“所有超参数严格复现论文”。

| 项目 | 论文 | 官方 SkillBank 脚本／实现 | 本设置处理 |
|---|---|---|---|
| batch / accumulation | 正文 batch 16、group 8、accumulation 4；附录 RL batch 64 | 16 组×8 轨迹；PPO mini-batch 128 行；micro/GPU 4；4 GPU | 明确区分各单位，不宣称这些数字等价 |
| prompt / response 上限 | 6,000 / 1,024 | 4,096 / 512 | 采用公开脚本 4,096 / 512；仍需完整 prompt 预算审核 |
| 演化失败样本数 | 附录低／高成功率时最多 10／5 | 实现先判 success-rate <0.4，再取 failed[:10]，未发现对应 5 条路径 | 记录代码实际行为，不拼接一个“论文同款”实现 |
| 技能操作 | 新增技能 | 每次最多新增 3 个 | 用户确认共用编辑器可增删改；明确是受控变体 |

上述差异由 [论文 Table 4](https://arxiv.org/html/2602.08234v1) 与 [公开训练器](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/verl/trainer/ppo/ray_trainer.py)、[updater](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/agent_system/memory/skill_updater.py) 交叉核对。

## 3. 新 RL 配置与旧最小验证的区别

旧列指已完成 Qwen3.5 clean cohort 的实际协议／记录，不代表所有历史脚本的默认值；旧 Phase2 后期有弹性分片，不能将其实际硬件一概写成 4 GPU。

| 参数 | 旧最小验证 / clean cohort | 新公开代码对齐配置 |
|---|---|---|
| policy / 初始化 | Qwen3.5-4B，共同 post-trained B0 | 保留同模型族；新运行须重新登记 B0 身份、独立 optimizer |
| RL 算法 / LR | GRPO / 1e-6 | GRPO / 1e-6 |
| 每轮采样组数×每组轨迹 | 8×4=32 | 16×8=128 |
| PPO mini-batch | 32 个展平 state/action 行 | 128 行 |
| micro-batch / GPU | Qwen 路径为 1 | 1，明确偏离上游的 4 |
| KL loss / invalid-action penalty | 0.01 / 0.1 | 0.01 / 0.1 |
| prompt / response token 上限 | 最近 Phase2 为 2,048 / 64 | 4,096 / 512 |
| episode 最大步数 / history | 30 / 2 | 50 / 2 |
| rollout / validation temperature | 1.0 / 0.4 | 1.0 / 0.4，top_p=1，均采样 |
| global rollout/update 轮数 | Phase1 30；后续固定窗口延伸至 U40 | 明确上限 150 |
| 训练时验证机会 | clean 监测，历史窗口规则不变 | 每 5 轮，共 30 个机会；启动前不额外验证 |
| 训练时验证样本 | clean valid_seen 27 的历史协议 | sampled seen monitor batch 64，不是全量 140 games |
| bank / router | 旧 44 库，clean 候选 18，冻结 lexical/phase router | SkillNet-37 全库共享，独立 gpt-5.4-mini 每步选 1 个 |

保留公开默认：std-normalized GRPO advantage，gamma=1、lambda=1，PPO epoch=1、clip=0.2、grad clip=1、entropy=0.001、token-mean loss、low_var_kl、AdamW weight decay=0.01、无 LR warmup。终局环境 reward 成功为 10、失败为 0；penalty 与 KL 是另外的训练项，不将 reward=10 当成 success=1000%。来源是源码检查，未在本轮环境运行验收。[官方默认配置](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/verl/trainer/config/ppo_trainer.yaml)、[环境 reward](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/agent_system/environments/env_package/alfworld/envs.py)。

### 3.1 单位解释与研究适配

`env.rollout.n=8` 已产生每组 8 条轨迹；`actor_rollout_ref.rollout.n` 保持 1，不能再设成 8 而无意多乘一次。150 轮的名义训练轨迹数为 16×8×150=19,200；这不是独立 game 数，也不是实际已产生的轨迹数。

公开数据准备使用占位行驱动环境采样，在 16 行／batch 16 的设置下每 epoch 一次采样更新。150 epochs 在该入口下不是对 3,553 个 train games 做 150 个全遍历。新配置显式增加 `total_training_steps=150`，防止更换数据文件后仅凭 epoch 含义误延长训练。实际准备文件、游戏覆盖与 RNG 分配仍待验收。

PPO mini-batch 的单位是展平后的 state/action 训练行，不是完整轨迹。4 个 data-parallel rank、全局 mini-batch 128、micro=1 时，一个完整 mini-batch 对应每卡 32 次 micro forward/backward；公开脚本 micro=4 时是 8 次。尾 batch 可能更小；每一 global iteration 有多少 `optimizer.step()` 取决于采样轨迹长度。**不能声称本实现就是论文的“梯度累积 4”，也不能把 150 轮写成 150 次 Adam step。** 这是 [worker batch 换算](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/verl/workers/fsdp_workers.py) 与 [actor 更新循环](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/verl/workers/actor/dp_actor.py) 的静态推导，GPU 执行仍未验证。

明确保留的研究差异：

- 原工作 Qwen2.5-7B 经 SFT；本项目仍为 Qwen3.5-4B post-trained B0，本轮不追加 SFT，不宣称 policy 初始化完全一致。
- 保留现有 Qwen text-only、HF live-policy rollout、SDPA、enable_thinking=false、action-only prompt；不为模仿论文推理格式暗中切换 prompt。
- micro/GPU=1、logprob micro=1、actor 参数／optimizer 不 offload、ref 参数 offload、关闭 remove-padding／torch compile、LoRA=0；是现有兼容性路径，不是公开脚本完全相同的资源配置。
- `save_freq=5` 而非公开脚本的 10，以对齐可能的干预节点；不自动删除 checkpoint。存储预算因此需单独审批。
- `resume_mode=disable`；新数据、seed、B0、run ID、输出、archive、cache 和 Ray temp 路径必填。使用单独 `alignment_run` 必填组再插值，避免 OmegaConf 合并时 overlay 的 `???` 意外保留旧默认值。
- 当前训练参考路径仍冻结 bank：dynamic update 关闭，router API 预算为 0，Phase2 新 cohort readout capture 尚未启用。此配置不是 Phase3 编辑器，也不是现成的全量 readout launcher。

## 4. 数据划分、全量覆盖与标签边界

沿用官方 train / valid_seen / valid_unseen，而非把六种任务类型随机分作训练／测试类别。六类任务均在范围内，初始共享同一 37 技能候选库，不按 task/game 筛库。计划 train 3,553、seen 140、unseen 134；这些是 benchmark 标准数量，仍需实际安装版本的 game-ID 清单核对。[ALFWorld 数据划分](https://arxiv.org/html/2010.03768#S2)。

训练过程中采用公开 recipe 的 sampled seen monitor，batch 64、每 5 轮检查；worker 数或 episode 数不保证唯一 game 覆盖。最终性能应另做每个目标 game 恰当遍历的 Seen-140 / Unseen-134 报告，并保留六类 breakdown，不把两个 split 合并掩盖差异。**本轮未实现或验证该全量遍历器。**

Seen 若用于调参、触发或编辑证据，报告为 development performance，不称 untouched test。Unseen 不参与编辑／选点／调分数；但历史 clean 31 unseen 和 27 seen 已分析，新的 RNG seed 不能把同一批 game 变成从未接触的 holdout。正式论文须披露此历史暴露，也不能将新的 SkillNet cohort 与旧 clean cohort 的效用标签直接混池。

4,096 token 的上限只是对齐目标；正式启动前须审核全部 37 个技能正文与历史／动作拼接后的 actor prompt，以及 token 等长 PLACEBO。不能以静默截断、过滤超长任务或缩短部分技能来换取表面“全量”。若不满足，应显式修订协议、记录偏离，不暗改为论文的 6,000。

## 5. Phase3：用户确认的共用编辑器与可变库

### 5.1 已确认原则

主要对比变量为**归因／证据选择层**：SkillRL-style failure-driven 对比 readout-driven。后续由相同外部 LLM 决定新增、删除、修改或不操作，不能由实验者只允许 readout 臂修改、却要求另一臂只能新增。归因方法不同会自然产生不同编辑输出，不需要强制两臂编辑结果相同。

两臂须固定：初始 policy／optimizer 条件、初始库、RL 预算、可用 train/seen 证据池、编辑器模型、除归因／证据字段外的 prompt 模板、解码参数、单次输入输出预算、重试、验证与接受规则，以及干预机会。**GPT-5.4 mini 目前只确认作为 router；编辑 LLM 尚未定型，不能默认复用其身份或调用授权。** Readout 额外计算、证据采集及 editor/router 用量都应归档成本，不因“共用外部 LLM”而遗漏。

用户确认的 bank 是“冻结初始库”，不是所有未来轮次必须恰好 37 个。版本库中的活动技能数为 37 + 累计 accepted additions − accepted deletions，修改对应新内容版本。初始 SkillNet37 原件及 manifest 永远只读，不执行物理删除或覆盖。

这与用户的因果问题一致：只改变归因选择机制，允许其经编辑结果影响后续系统表现。但它不再是原生 SkillRL add-only 实现的逐项复现。论文应称 **SkillRL-style failure-driven baseline with a shared editor**。若将来另报原生基线，需要单独协议，不能混用名称。

### 5.2 时点、触发与预算

对齐的是每 5 个 global iterations 的检查机会：U5、U10、…、U150，共 30 个；不是每 5 轮必须做一次编辑，也不是每次 Adam step 都检查。原生代码以 success-rate <0.4 触发，循环包含 overall 等 success-rate 字段，不仅是任务类别；最多收集 10 条失败轨迹并请求最多 3 个新技能。[原生实现](https://github.com/aiming-lab/SkillRL/blob/8e66726ed866a4e0a7f053586a41022798192e6c/verl/trainer/ppo/ray_trainer.py)。

对于“归因带来的编辑增量”主比较，建议在共享 pre-edit 参考状态上确定成对干预事件与预算，再分支应用两种证据选择；预测须在读取 post-update utility gold 和分配编辑前锁定。若做长期在线共同演化，各分支 policy／bank 已分化，需预先确定共享触发日程、预算与资源记账规则，不能观察结果后补齐编辑次数。这是待冻结的执行细节，本轮未擅自选择其中一种运行方案。

暂将“每次最多接受 3 个操作”记作公共预算提案，**不是用户已批准的最终额度，也不表示一次删除／修改等价于原生新增一个技能**。最终需同时约束编辑调用数、证据轨迹／tokens、输出 tokens、accepted 操作及重试，并报告新增／删除／修改次数、活动库大小、正文长度。限额允许不编辑，不强制消耗预算。

### 5.3 分支版本、router 与新技能支持

- Add：分配不可重用的新 ID，并记录来源事件和父版本。
- Modify：保留 lineage，新增内容版本；描述若变化也记录 descriptor hash。
- Delete：分支中停用并记录 tombstone，不删除旧文件或证据。
- 每个 checkpoint／评估绑定活动 bank manifest；训练和评估必须读取该分支对应版本，不能训练用新库、评估误用旧库。
- Router 固定模型请求名、选择算法、prompt 模板及解码参数，但输出 schema 和候选目录必须来自该分支的完整活动库。库变化造成后续路由变化是编辑的下游效果，不是另行优化 router；不同分支不必强行选择同 ID。
- cache 身份必须包含目录／描述的 hash。目录不同的相同可见状态不能直接重放旧决策；仅正文变化而 ID／描述不变时可以讨论 ID 决策复用，但 payload 必须绑定当前内容版本，防止注入旧正文。旧 cache 与审计保留，不覆盖。
- 新增技能没有既往 readout/utility 标签，必须标记支持不足并预先规定 bootstrap／弃权规则，不能填成零风险。修改版本是否可继承旧统计同样要预先约定。

**当前实现仍只支持固定 37：** `FrozenSkillBankMemory` 拒绝修改，runtime/router 对完整原始 37 个 ID／描述作校验。因此不能简单把 `enable_dynamic_update` 打开充当 Phase3。未来需要独立的版本化分支适配器、schema/cache 绑定及新技能支持机制。本轮没有修改 frozen bank 的保护规则。

另一个原生实现陷阱：公开 updater 只对训练 memory 执行 `add_skills`，validation memory 不随之同步。主对比若评估编辑后效果，必须显式加载分支 bank，不能照搬该路径便声称测到了编辑收益。

Phase2 的固定库 ORIGINAL/PLACEBO/NULL 语义边际效用与 Phase3 的端到端编辑收益是不同 estimand。Phase3 可包含候选竞争、路由与后续 policy 变化；不能用 Phase3 提升倒推 Phase2 已证明精确归因，也不能为 Phase3 标签事后选择 readout 分数。

## 6. 验收、证据边界与下一步

本轮离线验收共 **210 passed**：39 项新配置／设计边界检查 + 171 项现有 bank/router/归档接口回归。所有环境与 API 都使用 mock；没有 ALFWorld reset、模型前向、训练或真实外部调用。没有重跑历史实验／报告生成器；也没有将历史 Phase2 的 233 项测试当作本轮测试数。

只读配置命令（在 `SkillRL/` 下）：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/users/wangyifan/.venvs/skillnet-router-20260917-sdk3141/bin/python -B -m scripts.inspect_skillrl_alignment
```

结果为 `CONFIGURATION_VALIDATED_NOT_RUN`；加 `--require-ready` 返回 2，明确仍有必填输入与启动门槛。检查器只用 OmegaConf 合并已知的 defaults 配置，并非启动 Hydra/Ray；本轮没有安装 Hydra 或验收完整训练环境。将来经批准的 runner 可选择配置名 `alfworld_skillnet37_skillrl_v1`，但本记录不是启动指令。

完整测试命令与结果、文件 hash、保全检查见 [verification.json](SkillRL/docs/experiments/skillrl-alignment-v1/verification.json)。本轮仅新增七个路径；其余 20,269 个既有文件、2,190,019,438 bytes 的内容树摘要在前后核对中相同（不含 `.git`）。保全 SHA-256：`7306e57bf68a49aa17b89423fb627e7e741811e66c8b462586e0320296df3753`。当前没有 git 可执行程序，未声称重新核对 index／branch，也未提交、回滚或清理任何内容。

后续启动前仍需：

1. 确认独立新 seed、B0 文件身份、训练／data-loader／actor／distributed RNG 生效范围，以及新的输出目录。
2. 锁定实际全 task/game manifests、全量 Seen/Unseen 遍历和 readout 预测／标签隔离规则。
3. 完成全 prompt／PLACEBO token 预算；确认新 cohort readout capture，不直接重启历史 launcher。
4. 验收当前训练依赖、Ray worker 凭据注入、4 GPU 上 batch/context 的 forward/backward 与 accumulation；单独确认 GPU、存储、时间、router API 总预算。旧资源豁免不迁移。
5. 若推进 Phase3，再确认 editor 模型与调用预算、成对触发规则、新技能支持及可变库实现。本轮允许操作类型的决定，不自动授权调用 editor。

本轮没有给出 ALFWorld 性能提升、readout 优越性、固定第三方后端权重或全量 game 覆盖的实验证据。

## 7. 论文附录备用表述（计划态，执行后补齐）

> 我们将 SkillRL 的公开 SkillBank 训练脚本作为 RL 超参数基线，固定上游代码版本并单列其与论文附录的差异。研究仍使用 Qwen3.5-4B、统一的 SkillNet-37 冻结初始库和独立外部 router。Phase3 的两种策略仅在归因／证据选择方式上构造差异；后续由具有相同提示模板、操作空间和预算的公共外部编辑器决定是否新增、删除或修改技能。技能库在独立分支中版本化，原始库保持只读。Router 规则保持固定，技能编辑引起的候选目录及后续路由变化作为干预的下游结果。该比较称为共用编辑器的 SkillRL-style failure-driven baseline，而非原生 add-only SkillRL 的严格复现。最终性能按 Seen 与 Unseen 分别报告，训练时的 sampled seen monitor 不作为全量覆盖证据。

> We use the pinned public SkillBank training recipe of SkillRL as the RL hyperparameter reference and disclose discrepancies between its code and paper. Our study retains Qwen3.5-4B, a shared frozen initial SkillNet-37 bank, and an external policy-independent router. In the planned Phase3 comparison, attribution/evidence selection differs between the failure-driven and readout-driven arms, while a shared external editor has the same prompt template, operation space, and budget. The editor may add, delete, modify, or leave skills unchanged. Banks are versioned per branch without overwriting the original bank. Router rules remain fixed; catalog and routing changes induced by editing are downstream intervention effects. This is a controlled SkillRL-style failure-driven baseline with a shared editor, not an exact reproduction of native add-only SkillRL. Seen and Unseen performance will be reported separately; sampled training-time monitoring does not establish exhaustive evaluation coverage.
