# SkillRL 主对话：用户消息原文记录

> 说明：仅保留当前上下文中可见的主线程用户消息，按出现顺序排列；不包含助手回答、系统消息、工具输出或 side conversation 消息。重复消息原样保留。由于这是从当前会话上下文重建，而不是平台原始聊天数据库导出，换行可能与 UI 显示略有差异，但文字内容按可见记录保留。

## 用户消息 001

test

## 用户消息 002

现在我需要针对idea开展我的最小验证实验，完整idea见skill-RL/2026-08-21-proposal-
advantages-and-roadmap.md，对于现在任务想先采取的最小验证细节见/home/wangyifan/skill-RL/2026-08-24-phase1-minimal-validation-repository-and-
experiment-spec.md。请在skill-RL的文件夹下完成项目创建 下载的模型放在/model，另外环境创建采取conda activate新的环境为 skill-RL，你先阅读下两个文
件，熟悉下任务需求，再把你要做的事先和我确认，不要直接开做

## 用户消息 003

解释下你说的主指标，我理解第一步验证应该是固定skill-bank，然后看RL后的model在相同state和skill下是否发生了action filp么？

## 用户消息 004

我觉得合理，每一步验证一定归档留存实验记录，另外对于rollout的轨迹也做一下存档，比如一次轨迹用到了多少个不同的skill，轨迹具体内容等

## 用户消息 005

好的，现在先开始环境配置，项目构建，先做完实验前的所有准备工作

## 用户消息 006

我说/model指放到/home/wangyifan/model/下，0.5b和1.5B都下载下吧

## 用户消息 007

你上述的准备有整理成文档吗，只需要回答有没有，有在哪个路径

## 用户消息 008

除了这个还有么，我记得你前面说了几个？

## 用户消息 009

ok现在环境准备好了，开始分布跑验证把，首先你准备跑什么，预计产出怎样的统计信息，我确认之后你再跑

## 用户消息 010

ok，先测试第一批，记得把测试结果归档便于我分析

## 用户消息 011

下一步使用 1.5B 重跑
相同 smoke

## 用户消息 012

但是你看下面这篇，为啥在不做RL下有成功的。Qwen2.5-0.5B-Instruct 在 ALFWorld，不做 RL 的 baseline：Pass@1 = 30.4%，Pass@8 = 76.5%。https://openreview.net/pdf?id=K8wCGMzeuY&utm_source=chatgpt.com模型/方法	ALFWorld Pass@1	Pass@8
Qwen2.5-0.5B-Instruct，Baseline（无 RL）	30.4%	76.5%
+ SPA	52.1%	93.3%

这里需要注意：30.4% 是 Pass@1，最接近你说的单次 rollout acc / success rate；76.5% 是同一个任务允许采样 8 次后“至少成功一次”的 Pass@8，不能直接当普通 acc。

所以如果你现在需要一个可以引用的结论，可以说：

0.5B 量级并不是完全跑不动 ALFWorld。已有实验中，Qwen2.5-0.5B-Instruct 在未经 RL 的 baseline 设置下，ALFWorld Pass@1 可以达到约 30%。

不过这个 30.4% 很依赖具体 agent scaffold / prompt / rollout 设置，不能理解成“裸 Qwen 0.5B 在所有 ALFWorld 设置都是 30%”。这也是为什么不同论文的 baseline 数字可能差很多。

## 用户消息 013

那实际上我们完成了初步的验证只跑“0.5B 模型能力与 Skill 条件 smoke validation”，暂不做 RL update在小模型下验证不了我们的目的，需要执行真实 RL updates，先吧目前得到的结果总结成简要的md文件放在/home/wangyifan/skill-RL/下，然后跟我确定一下下一步的验证步骤

## 用户消息 014

我这里有个疑问，因为我们idea设置应该是在一个轨迹中每个步骤都会选择合适的skill；目前的设置是一个通用的skill加上6个context特定的skill，是不是意味着对于每个game对应的轨迹，每一步调用的都是同一个skill，这与我们研究state和skill固定下的效用偏移吻合么，因为可能单个skill不适用于某个state并不意味着他没用，因为实际情况下应该是根据某个state去选择的最合适的skill？

## 用户消息 015

我想指导skillRL的原始项目实现是怎样的逻辑，一个game用一组特定skill而不加state区分么？

## 用户消息 016

请阅读我们skill的实现 /home/wangyifan/skill-RL/2026-08-21-proposal-advantages-and-roadmap.md，对于skill是否该更新，我们是评估的在rollout轨迹中被步骤级调用的累加效益的和，但是Skill bundle的方式感觉没法区分policy model在每一步到底用到的哪些skill（也就没法定位评估），因此我觉得是每一步policy先根据目前的state 和skill descrp选择最适用的skill放进prompt这样是不是ok，那这样的话相当于应该沿用完整的skill库（然后每一步加上一个基于bundle里skill选择的步骤？）

## 用户消息 017

是的请按照这个步骤修改代码，改完后我们准备开始RL验证

## 用户消息 018

ok，先把上述的逻辑改动（包括原始仓库可能一个game喂所有skill，我们做了state路由），包括RL的测试步骤（参数，统计量等），包含的skill数等整理成md 放在/home/wangyifan/skill-RL/ 下，目前还不用包括RL的结果等启动RL了做汇总

## 用户消息 019

开始执行RL，记得记录归档

## 用户消息 020

下一步应运行固定状
态 action probe 及 matched full_bank / minus_skill / no_skill 评估

## 用户消息 021

刚才超出额度停止了，请继续跑

## 用户消息 022

刚才超出额度停止了，请继续跑

## 用户消息 023

刚才超出额度停止了，请继续跑

## 用户消息 024

继续跑

## 用户消息 025

现在的/home/wangyifan/skill-RL/2026-08-26-fixed-state-and-matched-evaluation-results.md 我觉得有一些地方没有说清楚，请按照现在的实施补充说明下，首先checkpoint1到5的关联是什么，他们的转换经过了多大程度的参数更新；其次对于condition的设置上 fullbank+pic_001 minus_skill no_skill的设置点是什么，fullbank+pic_001是指筛选了只要轨迹中某一步调用了pic_001的轨迹？minus_skill还是喂全部的全局skill而每一步不喂任务特定skill么；其次检索器的具体逻辑是什么如何根据state找的合适的skill；最后轨迹中平均调用skill的种类，发生action filp的比例，在评估上由于本次验证只要说明有频繁flip，但是是好的filp还是坏的根据idea我们是要后续计算reward参数更新方向确定的，所以这里Game-cluster bootstrap 95% CI跨0并不意味着episode success utility 需要统一变正或者负，可能有些skill效益随更新变低有些变高。

## 用户消息 026

目前针对skill效用的评估是有问题的，minus_skill 应该设置为 针对checkpoint0 rollout出来的轨迹，我要评估某特定skill A的影响，首先我应该对 skill A进行三种改写，第一种保留skill A原内容、第二种保持喂入的skill模板不变 但其中的内容为与任务无关的内容以保持prompt数不变、第三种不喂入skill内容（并以此对bank进行三类替换），在评估时查找 rollout轨迹中第一次调用该skill A的所有轨迹的位置，在调用该skill的前面步骤保持不变，针对第一次调用位置进行上述三类替换，然后后续其自由按照替换后的bank进行重新推理，并以累计奖励变化评估替换该skill带来的在checkpoint0的效用变化；完成横向比较后，我们再分别在checkpoint1-5进行类似分析，纵向分析checkpoint0与 1-5相比该累计效益随着RL是否发生变化，发生了怎样的变化。具体idea参考/home/wangyifan/skill-RL/2026-08-21-proposal-advantages-and-roadmap.md 你觉得合理吗，先与我讨论确认方案。

## 用户消息 027

目前针对skill效用的评估是有问题的，minus_skill 应该设置为 针对checkpoint0 rollout出来的轨迹，我要评估某特定skill A的影响，首先我应该对 skill
A进行三种改写，第一种保留skill A原内容、第二种保持喂入的skill模板不变 但其中的内容为与任务无关的内容以保持prompt数不变、第三种不喂入skill内
容（并以此对bank进行三类替换），在评估时查找 rollout轨迹中第一次调用该skill A的所有轨迹的位置，在调用该skill的前面步骤保持不变，针对第一次调
用位置进行上述三类替换，然后后续其自由按照替换后的bank进行重新推理，并以累计奖励变化评估替换该skill带来的在checkpoint0的效用变化；完成横向比
较后，我们再分别在checkpoint1-5进行类似分析，纵向分析checkpoint0与 1-5相比该累计效益随着RL是否发生变化，发生了怎样的变化。具体idea参考/home/
wangyifan/skill-RL/2026-08-21-proposal-advantages-and-roadmap.md 你觉得合理吗，先与我讨论确认方案。

## 用户消息 028

可以，就按照你说的方案评测，评测的详细报告依旧存放于/home/wangyifan/skill-RL 下

## 用户消息 029

刚才中断了，请继续

## 用户消息 030

目前的评测在泛化性上还存在一定问题，1. 首先在RL上，目前是针对一个RL seed下的五个checkpoint，并没有探究在不同RL设置下出现现象的泛化性，我觉得可以设置为例如5个seed，每个seed保存3个chekpoint，其次在保存checkpoint上，目前观察下来性能存在震荡，我觉得更新步长可以选大一点，保留的checkpoint是性能稳定提升的，防止干扰filp观察。2. 在skill上，感觉目前的待选task-specific 的skill是不是有点少？，有什么可以扩充复用的吗，另外目前只针对pic001做了评估，我需要全面的skill评估（包括锚点不在轨迹的step1的一些调用）来验证泛化性。3. 在game选择上，可以现在一个有代表性的task上完成上述评估，然后扩充到多个game。  综上可以与我确认下不清楚的点，然后和我讨论下方案

## 用户消息 031

目前的评测在泛化性上还存在一定问题，1. 首先在RL上，目前是针对一个RL seed下的五个checkpoint，并没有探究在不同RL设置下出现现象的泛化
性，我觉得可以设置为例如5个seed，每个seed保存3个chekpoint，其次在保存checkpoint上，目前观察下来性能存在震荡，我觉得更新步长可以选大
一点，保留的checkpoint是性能稳定提升的，防止干扰filp观察。2. 在skill上，感觉目前的待选task-specific 的skill是不是有点少？，有什么可
以扩充复用的吗，另外目前只针对pic001做了评估，我需要全面的skill评估（包括锚点不在轨迹的step1的一些调用）来验证泛化性。3. 在game选择
上，可以现在一个有代表性的task上完成上述评估，然后扩充到多个game。  综上可以与我确认下不清楚的点，然后和我讨论下方案

## 用户消息 032

第一点纠正一下，我觉得可以先探究累计update大一点的checkpoint保存，性能控制的checkpoint暂时先不分析；第二层RL配置泛化就按你说的取不同seed先评估，其余设置可以先不考虑；第三个skill部分原skillRL设置是否只是用了比如每个task task-specific量级5-6个，如果是那也ok不扩充变动了；其他部分就按你说的实施，另外我觉得base是不是不要选原始模型（因为其指令遵循能力好像都很差），至少选一个action决策正确率高一点的做base checkpoint对比？最后评估下我模型换成Qwen-3.5-4B的可行性，我现在的算力能支持吗

## 用户消息 033

能就在当前的环境升级并作一些版本适配调整么？

## 用户消息 034

可以，开始吧

## 用户消息 035

下载吧开始

## 用户消息 036

继续跑

## 用户消息 037

继续

## 用户消息 038

可以按我们之前的方案开始RL的过程了，我把讨论的重点给你再remind下：整体方向合理，但需要避免两个新的偏差：

    1. 5 个 seed 只能验证“随机训练路径泛化”，还不能称为“不同 RL 设置泛化”。
    2. 不能看完 Skill flip 后再挑性能最好、最平滑的 checkpoint，否则会产生 checkpoint selection bias。checkpoint 选择规
       则必须提前固定，并且不能读取 Skill 效用结果。

  ## 一、RL 训练与 checkpoint 方案

  我建议把实验拆成“seed 泛化”和“RL 设置泛化”两层。

  ### 第一层：5 个 seed，同一 RL 配置

  例如：

  - seeds：101、202、303、404、505；
  - 每个 seed 训练约 30 个 global updates；
  - 每个 update 仍为 32 条训练 rollout；
  - 每个 seed 保留 3 个 post-RL checkpoint，加上共同的 base；
  - 推荐 checkpoint 间隔从目前的 1 update 增加到约 10 updates。

  也就是优先考虑：

   模型    累计 update    累计训练 rollout
  ━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━
   Base              0                   0
  ──────  ─────────────  ──────────────────
   C1               10                 320
  ──────  ─────────────  ──────────────────
   C2               20                 640
  ──────  ─────────────  ──────────────────
   C3               30                 960

  5 个 seed 共得到 15 个 post-RL checkpoint，base 只需保留一份。

  不过我不建议单纯假设 update 10/20/30 一定稳定。更稳妥的是采用双轨规则：

  - 主分析：固定 update milestone，例如 10/20/30，保证跨 seed 可比；
  - 性能控制分析：performance-aligned checkpoint，根据独立 validation monitor 预先定义的规则选择三个稳定阶段。

  performance-aligned checkpoint 只能根据全局 validation success 选择，不能看 Skill utility 或 action flip。例如：

  - 两个保留 checkpoint 至少间隔 8–10 updates；
  - validation 使用足够多的独立 game，不能继续只依赖 8 条 rollout；
  - rolling validation mean 连续多个评估窗口不下降；
  - 相比上一个阶段至少提高预设阈值；
  - 如果某个 seed 没达到规则，则记录“无合格 checkpoint”，不能事后选局部最高点替代。

  这样可以回答一个更强的问题：

  > 即使 policy 的整体任务性能处于稳定改善阶段，某些旧 Skill 的边际效用是否仍会下降或发生 harmful flip？

  同时，训练震荡本身也不应该被完全删除。它可以作为 sensitivity analysis，否则容易把真正存在的非单调 Skill 效用错误地当
  成噪声过滤掉。

  ### 第二层：RL 配置泛化

  如果要严格声称“不同 RL 设置下泛化”，后续至少还要改变一类超参数，例如：

  - learning rate；
  - KL coefficient；
  - rollout batch/group size；
  - train game distribution；
  - reward shaping。

  资源有限时，我建议先完成 5 seeds × 同一配置。确认现象存在后，再选择两套配置、每套至少 3 seeds。不要一开始用 5 个彼此
  不同的配置但每个只有一个 seed，因为无法区分超参数效应和随机 seed 效应。

  ## 二、Skill Bank 是否需要扩充

  当前 Bank 实际已经有：

  - 12 个 general Skills；
  - 32 个 task-specific Skills；
  - 每个 task type 有 5–6 个 task-specific Skills；
  - 单个 episode 的候选池通常是 12 general + 当前 context 的 5–6 个 specific，即17–18 个。

  真正的问题不是总数过少，而是可路由性和语义重叠。

  旧 pilot 中只有 22 个 Skill 曾被 Router 实际选择。以 clean 为例，6 个 task-specific Skills 中只有4个被实际调用：

   Skill      出现轨迹数           第一次调用位置
  ━━━━━━━━━  ━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━
   cle_006            38              全部 step 0
  ─────────  ────────────  ───────────────────────
   cle_004            31     step 1–18，均值 2.74
  ─────────  ────────────  ───────────────────────
   cle_003            25     step 2–27，均值 9.08
  ─────────  ────────────  ───────────────────────
   cle_005             8    step 7–28，均值 18.25
  ─────────  ────────────  ───────────────────────
   cle_001             0         未被 Router 选择
  ─────────  ────────────  ───────────────────────
   cle_002             0         未被 Router 选择

  所以直接继续增加 Skill，很可能只是增加更多永远不会被选中的条目。

  ### 建议的扩充方式

  我建议把 Bank 重构成“可复用阶段模板 + context binding”，而不是增加同义 Skill。例如：

    1. 初始目标分解；
    2. 基于位置先验选择搜索点；
    3. 系统搜索未访问表面/容器；
    4. 目标可见后立即获取；
    5. 携带目标前往处理设备；
    6. 执行 clean/heat/cool 操作；
    7. 验证状态转换；
    8. 运送并放置；
    9. 多目标计数；
    10. loop/invalid-action recovery。

  其中很多可以跨任务复用：

  - Acquire When Visible
  - Systematic Search
  - Navigate to Processor
  - Execute Transformation
  - Verify Transformed State
  - Deliver to Destination
  - Loop Recovery

  同一个模板在 clean、heat、cool 中绑定不同参数，最终评估单位仍然是 proposal 中的：

  (shared_skill_template, context)

  这比为每个任务生成大量近义文本更有研究价值，也能检验同一个 Skill 原理在不同 context 中是否出现不同效用漂移。

  ### Router 也必须一起调整

  新 Bank 应在 RL 前冻结，并用确定性的 phase eligibility gate 路由：

  - holding_target=false 才允许 search/acquire；
  - holding_target=true && transformed=false 才允许 processor/transform；
  - transformed=true 才允许 verify/deliver；
  - 出现重复动作或无进展才允许 recovery。

  然后再在当前 phase 的候选中确定性打分。这样能够保证不同 Skill 对应不同 state phase，减少现在 soft score 和 task-
  specific prior 导致的 Skill 垄断。

  正式训练前应运行一次 Router coverage audit。每个预期评估的 Skill至少应满足：

  - 被自然选择；
  - 覆盖足够多的不同 game；
  - 有明确的首次调用 state；
  - 不完全由 tie-break 触发；
  - first-step 分布与设计 phase 一致。

  达不到支持阈值的 Skill 应标记为 unsupported/abstain，不能强行计算效用。

  ## 三、代表性 task 与 game 选择

  这里需要区分：

  - task type/context：例如 clean；
  - game instance：例如清洗某个具体物体并放入某个具体 receptacle。

  我建议第一阶段选择 clean 作为代表性 task type，而不是继续选择 pick_and_place。原因是当前数据中：

  - clean 已有 9/38 成功，reward 不至于全为零；
  - 覆盖 11 个旧 pilot game；
  - 自然包含初始搜索、容器搜索、去水槽清洗、状态验证和最终放置；
  - 已有首次调用发生在 step 2–27、甚至 step 7–28 的 Skill；
  - 可以直接验证真正的 prefix replay 和中途 intervention。

  数据集中还有：

  - 650 个 train clean game/trial；
  - 27 个 valid_seen clean games；
  - 31 个 valid_unseen clean games。

  因此不建议先只跑一个 concrete game。单个 game 无法估计 game-level 泛化，也无法进行有效的 cluster bootstrap。建议：

  - train：clean train games；
  - checkpoint performance monitor：valid_seen clean games；
  - 最终 Skill utility gold evaluation：valid_unseen 的 31 个 clean games；
  - 每个 game 使用多个 generation seeds；
  - checkpoint 选择过程不能读取 valid_unseen Skill utility。

  完成 clean 后，再扩展到：

    1. pick-and-place；
    2. heat/cool；
    3. look-at-object-in-light；
    4. multi-object placement。

  这会同时扩展 task type、operation phase 和 game instance。

  ## 四、全面 Skill 评测协议

  对代表性 task 中每一个有支持的 Skill，都采用当前三臂首次调用协议：

  - ORIGINAL；
  - token-matched PLACEBO；
  - NULL；
  - 固定第一次实际调用前的 prefix；
  - 从调用位置开始自由 rollout；
  - 后续每次再次选择目标 Skill 时持续应用同一 payload 干预；
  - 所有 seed/checkpoint 使用同一 base anchor 支持集。

  anchor 应按调用阶段分层报告：

  - initial：step 0；
  - early：step 1–4；
  - middle：step 5–14；
  - late：step ≥15。

  每个 Skill 建议至少：

  - 30–50 个 anchor occurrences；
  - 覆盖至少 10–15 个不同 games；
  - 不能用同一 game 的大量 generation seeds冒充独立 game 支持。

  跨 seed 汇总时不能把全部 trajectory 当作独立样本。应采用层级结构：

  RL seed → checkpoint/update → game → rollout seed

  主要统计包括：

  - 各 Skill 的 U_sem = ORIGINAL − PLACEBO；
  - Delta U_sem；
  - harmful sign-flip rate；
  - positive/negative/zero anchor 比例；
  - 跨 seed 符号一致率；
  - game/seed 分层 bootstrap CI；
  - first-action flip 与 suffix-action divergence；
  - 全局 policy performance；
  - Skill utility 与全局 performance 的解耦；
  - unsupported/uncertain Skill 比例。

  只有 baseline utility 在独立 calibration 数据上可靠为正，之后变为可靠负的 pair，才能标记为 harmful sign flip。其他
  pair 仍作为连续 Delta U 样本保留。

  ## 五、需要你确认的四点

  我建议的默认选择如下：

    1. “3 个 checkpoint”理解为每个 seed 的 3 个 post-RL checkpoint，另加共同 base。
    2. 第一轮先做 5 seeds、同一 RL 配置；不同超参数配置放到第二轮。
    3. 代表性 task type 选择 clean，但从开始就覆盖多个 concrete games。
    4. “全面 Skill”定义为：评估 clean context 中所有自然可路由且支持充分的 task-specific 和 general Skills；未被调用的
       Skill报告为 unsupported，而不是人为强制调用。

  还需要你确认一个关键取舍：你更希望主 checkpoint 使用固定的 update 10/20/30，还是使用“由独立 validation performance
  预注册规则选出的三个稳定阶段”？我的建议是两者都保留：固定 milestone 作为主时间轴，performance-aligned 作为控制分析。

─ Worked for 3m 49s ──────────────────────────────────────────────────────────────────────────────────────────────────


› 第一点纠正一下，我觉得可以先探究累计update大一点的checkpoint保存，性能控制的checkpoint暂时先不分析；第二层RL配置泛
  化就按你说的取不同seed先评估，其余设置可以先不考虑；第三个skill部分原skillRL设置是否只是用了比如每个task task-
  specific量级5-6个，如果是那也ok不扩充变动了；其他部分就按你说的实施，另外我觉得base是不是不要选原始模型（因为其指令
  遵循能力好像都很差），至少选一个action决策正确率高一点的做base checkpoint对比？最后评估下我模型换成Qwen-3.5-4B的可
  行性，我现在的算力能支持吗



## 调整后的实验方案

  ### 1. RL 与 checkpoint

  采用同一 RL 配置、5 个独立 seed：

  101 / 202 / 303 / 404 / 505

  每个 seed 从同一个 base checkpoint 出发，重新初始化 optimizer，训练到30个 global updates，并保存：

   Checkpoint    累计 update    累计训练 rollout
  ━━━━━━━━━━━━  ━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━
   Base/B0                 0                   0
  ────────────  ─────────────  ──────────────────
   C1                     10                 320
  ────────────  ─────────────  ──────────────────
   C2                     20                 640
  ────────────  ─────────────  ──────────────────
   C3                     30                 960

  这里每次 update 仍按当前设置使用 8 games × 4 rollouts = 32 条训练轨迹。5 个 seed 总计约4,800条训练 rollout。

  暂时不根据性能挑选 checkpoint，也不丢弃震荡 checkpoint。validation success、return、invalid action rate仍完整记录，
  但只作为解释变量，不参与 checkpoint 选择。这样不会引入事后选择偏差。

  ### 2. Skill Bank 保持不变

  是的，当前原始 SkillRL Bank 的量级就是：

  - 每个 task type 有 5–6 个 task-specific Skills；
  - 6 个 task types 共32个 task-specific Skills；
  - 另外有12个 general Skills；
  - 单个 task 的候选集合通常为12个 general + 当前 task 的5–6个 specific。

  这个规模可以支撑阶段一，因此不扩充、不改写 Skill 内容，在所有 seed 和 checkpoint 间完全冻结。

  对于 clean 代表任务，全面评估定义为：

  - 评估所有被正常 Router 自然调用且支持充分的 task-specific/general Skills；
  - 每条源轨迹分别查找每个 Skill 的第一次调用；
  - 包括 step 0、early、middle 和 late anchors；
  - 未被 Router 调用的候选 Skill记录为 unsupported，不人为强制调用，避免制造不自然的干预 state。

  现有 clean 轨迹已经覆盖：

  - cle_006：step 0；
  - cle_004：step 1–18；
  - cle_003：step 2–27；
  - cle_005：step 7–28。

  因此能真正测试非初始步骤的 prefix replay 和中途干预。

  ### 3. 代表性任务与 game

  先选择 clean task type：

  - train：clean train games；
  - checkpoint 常规 validation：valid_seen clean games；
  - Skill 效用 gold evaluation：valid_unseen 的31个 clean games；
  - 每个 game 使用多个 generation seeds；
  - bootstrap 按 RL seed → game 分层。

  完成 clean 后，再扩展至 pick-and-place、heat/cool、look-at-object-in-light 和 multi-object。

  ## Base 不应继续使用能力过弱的原始模型

  我同意。研究目标不是测量“弱模型学会输出合法动作”，而是观察一个已有基本 agent 能力的 policy，在后续 RL update 中与固
  定 Skill 的交互如何变化。

  但“action 决策正确率”需要分成三项：

  - action format/parse 成功率；
  - 是否属于 admissible actions；
  - episode success rate。

  ALFWorld 同一 state 往往有多个合理动作，因此不存在天然唯一的逐步“正确 action”。Base 至少应满足：

  - action parse 和 admissible rate 接近98%或更高；
  - clean 上有非零且足够稳定的 success；
  - 能自然走到搜索、获取、清洗、验证、放置等中后期阶段；
  - 多个 Skill 有足够 anchor 支持。

  我建议定义一个公共 B0：

    1. 先测试 Qwen3.5-4B post-trained 模型；
    2. 如果其 clean baseline 已达到上述条件，直接作为 B0；
    3. 如果格式正确但 ALFWorld success 仍太低，则先在完全独立的 clean train games 上做一段统一 warm-up；
    4. warm-up 后冻结成唯一 B0；
    5. 5 个 RL seeds 都从相同 B0 出发，optimizer 各自重新初始化。

  B0 的选择只能依据独立 monitor 集上的全局 action validity 和 success，不能依据后续 Skill flip，以免选择偏差。

  ## Qwen3.5-4B 的算力可行性

  结论是：硬件足够，但当前软件环境不能直接运行，需要先迁移训练栈。

  当前机器有：

  - 8 × NVIDIA A800 80GB；
  - 约502 GiB主存；
  - 当前约398 GB磁盘余量；
  - 目前约5张 GPU 基本空闲。

  Qwen3.5-4B 是4B参数、32层、BF16、混合线性注意力/全注意力架构，并带 vision
  encoder；官方给出的原生上下文为262K。ALFWorld 只需要文本，应采用 language-model-only
  路径，避免加载或训练无用的视觉部分。Qwen3.5-4B 官方模型卡 (https://huggingface.co/Qwen/Qwen3.5-4B)

  以当前4096 prompt + 512 response、gradient checkpointing、FSDP 和 CPU offload 设置：

  - 单次4B全参数 GRPO 使用4张 A800-80GB 是比较稳妥的；
  - 2张卡理论上可能装得下，但 rollout/actor/ref 共存时吞吐和稳定性会较差；
  - 推荐5个 seed 顺序运行，或者在资源完全空闲时尝试两组并行。

  ### 当前真正的阻塞是软件版本

  现有环境为：

  - Transformers 4.51.1；
  - vLLM 0.8.4；
  - 当前 SkillRL fork 内没有 qwen3_5 适配代码。

  而 Qwen3.5 的架构标识是 Qwen3_5ForConditionalGeneration；当前官方 vLLM 文档已经支持，但这是较新的 vLLM
  版本能力。vLLM 支持模型列表 (https://docs.vllm.ai/en/v0.21.0/models/supported_models/)

  近期 verl 才合并 Qwen3.5/FSDP 相关适配，历史上也确实存在 Transformers 与 vLLM 版本不兼容问题。verl Qwen3.5 兼容问题
  (https://github.com/verl-project/verl/issues/5937)

  因此不能只把模型路径从 Qwen2.5 改成 Qwen3.5；需要：

    1. 创建独立的 Qwen3.5 环境，避免破坏当前已复现的 skill-RL 环境；
    2. 使用支持 Qwen3.5 的 Transformers、vLLM 和较新 verl 组合；
    3. 将现有 ALFWorld 环境、state Router、轨迹归档和三臂评测改动移植过去；
    4. 验证 text-only actor、reference model、FSDP backward、vLLM rollout 和 checkpoint merge；
    5. 先完成1次真实 GRPO update 的保存—恢复—评测闭环，再启动5 seeds。

  ### 存储需要改变策略

  当前1.5B的一个完整 FSDP checkpoint 约19GB。按参数量近似缩放，4B含 optimizer 的完整 checkpoint可能约45–55GB：

  - 15个完整 checkpoint 可能需要约675–825GB；
  - 当前398GB余量无法全部保存。

  解决方式是：

  - 每个 seed 只保留一个可恢复训练的 optimizer checkpoint；
  - C1/C2/C3 转换并保存为 model-only BF16/HF 权重；
  - 4B model-only checkpoint 预计约8–10GB；
  - 15个约120–150GB，可以容纳；
  - rollout、metrics 和 trajectory 的体积相比权重很小，继续完整保留。

  ## 建议的迁移门槛

  在正式5-seed实验前，先做 Qwen3.5-4B feasibility gate：

    1. text-only 加载与单步 action probe；
    2. Router Skill 覆盖和首次调用 step 分布；
    3. 1次完整 GRPO update；
    4. checkpoint保存、恢复和参数差分；
    5. ORIGINAL/PLACEBO/NULL 各跑少量中途 anchor。

  全部通过后才将 Qwen3.5-4B 或其统一 warm-up checkpoint确定为 B0。

  我的最终建议是：采用 Qwen3.5-4B + 必要时统一 warm-up B0 + 5 seeds + update 10/20/30 + clean 全 Skill首次调用点评测。
  硬件可行，最大的工作量在软件栈迁移和存储格式调整，而不是显存。现在只需要你确认是否接受创建独立 Qwen3.5环境并先跑上
  述 feasibility gate。

## 用户消息 039

更改为先跑3个seed的验证

## 用户消息 040

注意到现在其他四张卡是空闲的，能不能同步跑一下seed 202，同时防止磁盘满的风险，或许之前的0.5b的记录可以清理release下？

## 用户消息 041

我不小心把训练中断了，4卡seed101与4卡202

## 用户消息 042

根据现有gpu状况继续跑，资源不够优先把seed101跑完

## 用户消息 043

等第一个checkpoint跑完后按照下面逻辑修改下，保险点防止oom损失：• 建议改成每个 update 保存一次，比每 5 updates 更适合当前不稳定的共享 GPU 环境。

  实际数据：

  - 一个完整 checkpoint：约 50GB
  - 保存一次：约 40秒
  - 每 update 保存：30次约 20分钟
  - 每5 updates保存：6次约 4分钟
  - 总差异约 16分钟，相对约24小时训练时间只增加约 1%

  但可靠性差异很大：

  - 每5次保存：中断时最多损失约4个 updates，即约 3小时
  - 每次保存：最多损失约1个 update，即约 45–50分钟

  建议采用：

  - save_freq=1
  - 始终保留最近 2个完整 checkpoint，而不是只保留1个，避免最新 checkpoint 写入不完整时连上一个也被删除
  - update 10/20/30 立即转换成约9GB的 model-only 权重，永久归档
  - 其他 update 的50GB完整 checkpoint滚动清理
  - 最终只保留：
      - update 10/20/30 的 model-only 权重
      - 最新的一个或两个完整恢复 checkpoint

  这样瞬时最多占约100GB完整 checkpoint，当前364GB剩余空间能够承受，速度损失很小，明显更保险。需要注意，当前 Seed 101 已按每5 updates运
  行；修改为每次保存需要重启，因此至少应先等它保存出第一个 checkpoint。

## 用户消息 044

我看现在seed101已经快跑完了持续监控，跑完了切换seed202从10开始的训练

