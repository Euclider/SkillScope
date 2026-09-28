# Phase3 六臂 U20 设置与启动说明（2026-09-25）

> 2026-09-27 协议补记：下文的 v11 终点 Seen 验证编辑证据已由
> [v13 同源旧策略批次证据](EDITOR_OLD_POLICY_BATCH_EVIDENCE_V13.md)
> 前瞻性替代。中间的 [v12 提案](EDITOR_PREUPDATE_TRAIN_EVIDENCE_V12.md)
> 未用于正式编辑。首臂 U5 已按 v11 完成且保留，后续窗口按 v13
> 接续，因此首臂是混合协议；本页其它历史启动记录不追溯改写。

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-25
- Verification Status: UNVERIFIED（只有离线工程测试；无 Phase3 正式结果）
- Version Label: phase3_six_arm_u20_v3

状态：实验设计与本机启动已由用户确认。专项八卡合成预检和 o3 schema 真实调用均通过，`configs/phase3_setting_embedding_v3.json` 的 `execution.approved=true`；真实 ALFWorld 的整条 Phase3 闭环及新服务器端到端验收仍待完成。本页替代 v2 的拟议四臂/U50 启动设置，但不覆盖 v1/v2 文档、既有 Phase1–2 报告或实验数据。

2026-09-25 本机准入记录：八卡合成预检 `/data/disk1/wangyifan/phase3-v3-gpu-preflight-20260925-a/complete.json` 为 PASS，覆盖 vLLM 512-token 生成、一次 FSDP 优化、原生模型/优化器恢复和本地 0.6B router 共存；它明确**不**覆盖真实 ALFWorld 全路径。Phase3 专项离线测试 112/112 通过。更宽的历史回归为 683 通过、4 失败；失败都来自旧 Phase1–2 队列的冻结源码哈希与当前演进源码不一致，不能重写旧哈希让测试通过。o3 实际编辑 schema 调用成功，网关回报模型 `o3`、653 输入/129 输出 token、约 6.8 秒；合成测试响应为 NOOP，脱敏账本位于 `/data/disk1/wangyifan/skill-scope-phase3-v3-20260925/o3-smoke/ledger.sqlite3`。未把密钥写入代码、配置或账本。

首次离线登记的 `assets` 保留为未获准版本；已另建获准的 `assets-launch-v1`，不覆盖前者。`readout_d` 首臂于 2026-09-25 启动，运行根目录为 `/data/disk1/wangyifan/skill-scope-phase3-v3-20260925/runs/readout_d`；启动检查已见八卡 FSDP/vLLM worker 与真实 ALFWorld 的 3,553 个训练 game。后续五臂的失败即停顺序队列已启动并等待首臂 U20 封存，脚本及账本在同一目录的 `continue_six_arms.sh` 与 `queue.log`；它不重试失败臂、不跳过失败臂、不覆盖根目录。当前首臂的实际完整闭环尚未产生结果；不得把“进程已启动”写成“实验成功”。

本机新运行的磁盘保护值（只适用上述新目录）：每臂最多 450 GiB，始终至少保留 500 GiB 空闲，启动下一个 checkpoint 前按 80 GiB 预留。六臂顺序执行，三个值分别以 byte 写在仓库外 `runtime.json`；历史实验与报告不受其管理。

## 1. 冻结比较

六臂按以下顺序、在同一组 8 卡上**串行**运行。各臂从相同 Qwen3.5-4B B0、SkillNet-37 和 RL seed 707 出发，之后的 policy、Adam、轨迹、skill bank 与 router 缓存独立；不得用一臂的后续轨迹充当另一臂的闭环结果。

| 顺序 | branch | 唯一驱动编辑候选的选择器 |
| --- | --- | --- |
| 1 | `readout_d` | `D_sign_balance`，reward-directed 主实验 |
| 2 | `skillrl_failure` | SkillRL-style 失败轨迹证据基线 |
| 3 | `readout_magnitude` | 纯幅度 `‖Hδ‖`（代码字段 `M_delta_centered`） |
| 4 | `readout_gated_d` | 旧负部 gated D，即 gated `[-P_int]₊` |
| 5 | `readout_p` | `−P_int` |
| 6 | `readout_c` | 中心化 `+C_upd_centered` |

六臂使用同一稳定 `fp64_zero_sum_readout_v1` 数值实现与 game-equal 聚合；“旧 D”仅指**历史公式**，不是故意恢复旧 FP32 数值误差。404/505 的 Phase2 标签参与过指标探索，不能当作六臂闭环的独立验证。seed 707 仍只有一个独立 RL seed；多个 game、解码 seed 和窗口不是训练 seed。

选择顺序的已有证据边界：在 Phase2 的 8-gold、placebo、all_alfworld、阈值 0、game-equal 排序表中，下降检测 AP/AUROC 分别为 D_sign_balance：404 `0.734/0.753`、505 `0.567/0.792`；`‖Hδ‖`：404 `0.540/0.597`、505 `0.483/0.625`；旧负部 D：404 `0.717/0.779`、505 `0.365/0.792`。来源为 `artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s{404,505}-v1/precision/gold8-ranking.csv`，其余 −P/+C 及全部变式保留在同表。404 有 18 个可比较技能、7 个下降；505 有 19 个、仅 3 个下降，不能把 AP 差异解释为稳定的跨 seed 优势。

全部六臂先运行 U0→U20，再共同查看中期结果，不根据先运行的臂选择不同终点。四个相邻窗口为 U0→5、5→10、10→15、15→20，每窗最多一次编辑机会、最多 3 mutation units。GRPO 仍为 lr=1e-6、16 games×8 rollouts/update、全局决策 minibatch=128、每 5 次更新读出/编辑/验证。`optimizer_horizon_updates=150` 始终不变；`initial_stop_update=20` 只是执行暂停点，不把学习率调度缩成 20 步。U20 Seen/Unseen 全量指标是 **interim milestone**，不写分支最终完成标记；后续延长需显式 `--stop-update`，同一准备清单、同一运行目录和同一优化器/RNG 状态续训。

若查看 U20 Unseen 后依据效果决定是否继续，该 Unseen 集已经参与了实验决策；延长后的同一 Unseen 结果应标为探索性序贯分析，不能再称完全未触碰的独立最终测试。需要确认性闭环结论时，应事先冻结不依赖 U20 Unseen 的延长规则，或另用独立 RL seed/测试集验证。

没有 skill 级“动作偏置低于阈值就跳过编辑”的门控。每窗同等编辑机会，编辑器仍可 NOOP，缺自然支持/证据时仍可弃权。每窗保存各自然支持 skill 的 `M_delta_centered` 至 `events/uXXXX/shadow_action_bias.json`，`edit_threshold=null`、`threshold_decision_applied=false`；它在**幅度臂用于技能排序**，但在所有臂都不用于“本窗是否编辑”的门控，这不是事后挑阈值的正式消融。失败驱动臂也做**只读影子前向**用于共同记录，但该读出绝不传给其编辑选择器；汇总单列这部分诊断计算成本。若日后实验编辑门控，必须另行冻结公共量、阈值、预算和独立验证，不能改变本六臂设置后仍称同一比较。

## 2. 评价与成本

每臂当前 bank 由冻结本地 Qwen3-Embedding-0.6B 逐状态从**完整活动库** top-1 路由；候选在同一当前 policy×Seen gate game×解码 seed 下配对验证。此处原始 o3、top-3 和输入上限设置已被后续变更替代：编辑器统一改为 gpt-5.5；最终 v11 编辑证据仅取窗口终点当次 Seen 验证的失败轨迹，readout 各臂在当次失败调用过的技能中按本臂分数取 top-5，再反选所有相关完整失败轨迹；失败驱动适配基线看到完整活动库正文及当次全部完整失败轨迹，不设 8 条上限。本地 editor 输入 token 拦截取消，保留实际 token 账本、输出上限和 API 次数预算。详见 [编辑器模型切换](EDITOR_MODEL_SWITCH_2026-09-26.md)与 [终点策略证据 v11](EDITOR_CURRENT_POLICY_EVIDENCE_V11.md)。Unseen 不用于选择指标、候选、编辑或 gate。

记录每窗候选/自然支持、API 请求和 usage、编辑 NOOP/修改单位/技能数、gate 拒绝、bank 版本、轨迹和 actor tokens、router 计算、readout 前向时间/tokens、训练指标、rollback 与全部 Seen/Unseen 成功率。比较相同**机会和上限**下的性能及实际成本；不强迫各臂产生相同编辑次数。未提供的网关实付金额保持 `null`，不能填官方价格。每臂 API 请求上限模板为 30，对应 150 更新最多 30 个窗口；本次 U20 最多实际 4 次，不会因为上限为 30 自动运行到 U150。该上限和磁盘数值仍需启动前核准。

SkillRL-style 基线的影子读出不影响决策；报告须同时列出“实际基线算法成本（不含被动影子前向）”与“本次仪器化总计算成本（含影子前向）”，不能把被动审计开销算作它原生选择器的必要成本。

## 3. 只保留最新端点

每窗需要上一端点与当前端点完成固定状态读出。当前窗的读出、编辑/gate、运行摘要和原生 checkpoint 均封存后，才记录 `retention_intent.json`，并**仅移除本次新运行中上一端点的 native checkpoint 和 HF 导出**；`retention_complete.json` 记录完成。共同 B0、当前最新 native/导出、全部窗口轨迹、首批方向 batch、readout、银行版本、API/router 账本、事件和报告保留。若回收中断，恢复必须先根据 intent 对账；未解释的端点缺失直接停止。U20 暂停时仅需 U20 native/导出及其优化器/RNG；U15 不再是续训依赖。

这是用户确认的**新实验端点回收策略**，不应用于历史实验或其它目录。实际清理只会发生在未来显式启动的相应新运行目录内；本次代码对齐未删除任何既有 checkpoint。

## 4. 新服务器准备与只读计划

新服务器在独立目录安装 `requirements-phase3.txt` 和仓库包；该文件继承 vLLM 专用锁文件（OpenAI SDK 3.15.0、tiktoken 0.14.0），不能与旧 HF-only Phase1–2 依赖文件混装。获取固定模型、ALFWorld 文本 games、冻结 embedding router 与 SkillNet-37。vLLM 负责训练生成及 gate/整局评估；HF/SDPA BF16 仅用于两端固定状态全词表前向，必须通过 live/offline chosen-token parity。先运行离线测试及 Phase3 专项八卡/真实环境/o3 schema 准入，不得借用 Phase1–2 的通过回执。

复制 `configs/phase3_runtime_template_v3.json` 到**仓库外**填写模型、数据、router、GPU、磁盘保护值；不要把 API key 写入文件。`phase3.prepare` 只写新资产，不启动 RL/API。默认 `phase3.run` 不加 `--execute` 只打印 U20 计划：

```bash
python -B -m phase3.prepare --runtime /data/phase3-runtime-v3.json --output /data/phase3-assets-v3
python -B -m phase3.run --preparation /data/phase3-assets-v3/manifest.json \
  --root /data/phase3-runs/readout_d --branch readout_d
```

已获准的 `assets-launch-v1` 可执行 U20；**首臂正在运行并非已经完成**。正式运行必须六臂顺序、各自不同 root；不得同时争用八卡。完成 U20 且获新确认后，才可对**同一 branch/root/preparation**显式请求例如 `--stop-update 50`；代码拒绝没有 U20 封存 milestone 的直接越级启动。U20 结果位于 `milestones/u0020/{valid_seen,valid_unseen}`，最新摘要 `milestones/u0020/running_metrics.json`；U50 将另建 `milestones/u0050`，绝不覆盖 U20。只有达到预设优化器 horizon 才写根目录 `complete.json`。

如果需要对外打包，`phase3.package` 的发布 README 指向本文并保留历史 v2 文档。**本轮未推送 GitHub；已进行一次小额 o3 真实请求，并启动首臂真实 ALFWorld 训练。**
