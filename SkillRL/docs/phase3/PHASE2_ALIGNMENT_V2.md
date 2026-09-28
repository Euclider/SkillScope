# Phase3 × Phase2 数值协议对齐：v2 待确认设置

状态：**仅代码与离线契约测试；未批准正式 Phase3，不含四臂结果**。这是新协议，不改写历史 v1 设置、旧报告或 404/505 训练/效用证据。代码会检查 v2 setting 的 `execution.approved`；当前为 `false`，正式 `phase3.run --execute` 会直接拒绝。用户确认、八卡 Phase3 专项预检、o3 网关真请求与新服务器端到端预检之前，不能将其改为正式运行许可。

## 材料与证据边界

- 指标选择使用既有 seed404/505 的独立 U0→U5 RL 窗口及 8-gold 配对效用标签：`D_sign_balance::game::reward` 在 404 的下降检测 AP 0.734、AUROC 0.753，在 505 为 AP 0.567、AUROC 0.792（阈值 0、placebo、all_alfworld、all）。这是**看过两个标签后的探索性选择**；这些数不能充当 Phase3 闭环的独立结果，也不能证明该指标普遍优于中心化范数。新的 RL seed 707 必须作为独立验证，不按 gate 或最终 Unseen 结果再换指标/阈值。
- 原始审计源：`artifacts/phase12/skillnet37-independent-s404-505-606-v4/utility-precision-s{404,505}-v1/precision/gold8-ranking.csv`；公式源 `skillnet_cohort/reward_variants.py`，稳定数值源 `phase2/stable_direction.py`。Phase3 使用相同 `fp64_zero_sum_readout_v1`、相同 token→decision→trajectory→game 等权层级，保留 `P_int`、`C_upd_centered`、旧 rectified D、原始 C、中心化 delta 范数作对照/审计。
- 505 下降技能仅 3/19，404 为 7/18；独立训练 seed 的多窗口闭环不是对跨任务/跨 seed 因果效应的证明。Unseen 不进入选指标、编辑或 gate。

## 拟议单 seed 四臂设计（待用户确认）

| 项 | v2 提议 |
| --- | --- |
| 初始条件 | 四臂同一 Qwen3.5-4B B0、SkillNet-37、RL seed 707；四套独立 policy/optimizer/bank，按臂顺序跑，不共享后续轨迹 |
| 四臂 | `skillrl_failure`（SkillRL-style 失败证据）、`readout_d`（reward sign-balance）、`readout_p`（−P）、`readout_c`（中心化 +C）；共用同一编辑器/路由/证据与 gate 预算 |
| RL | GRPO，lr 1e-6，16 groups × 8 trajectories/update，128 决策行 minibatch，8 卡，建议 U50（10 个五轮窗口）；每轮保存训练指标，每窗保存原生 policy/Adam/RNG、导出模型和不可变窗口事件 |
| 读出 | 每窗首轮实际优化 batch 的 `A` 与固定状态，在 `U_t→U_{t+5}` 两端对当前不变 bank 的 ORIGINAL/等 token PLACEBO 做 BF16 HF/SDPA 全词表前向；只在内存保留词表向量，FP64 计算数值稳定的投影；没有额外环境 rollout；`D_sign_balance=-direction_valid×sign(P_int)`，分数越高越优先复核 |
| 支持 | 当前内容版本凡有至少一个自然损失 token 就进入排序；记录 token/非零 A 决策/轨迹/game 数，不再用 20/4/8 门槛；没出现的技能弃权，新/改写/合并版本不继承分数，平分按 skill ID 固定打破 |
| Router | 本地冻结 Qwen3-Embedding-0.6B，逐状态完整活动库 top-1；分支独立缓存和跨版本总计数；每臂建议上限 600,000 次本地新检索，不调用外部 router API |
| Editor | o3、用户指定 HTTPS `/v1` 网关、同 prompt、ADD/MODIFY/DELETE/MERGE/NOOP、每窗最多 3 mutation units；每臂 API 上限 12，输入估算 cap 64,000 tokens、输出 cap 8,192、证据最多 8 条轨迹；超限停止，不截断/偷偷摘要/自动重试 |
| Seen 开发 gate | 按 seed707 固定哈希，每类 task 取 2 个 Seen game（总 12）；其余 Seen 用作证据/监测。候选与旧库在同 policy×game×解码 seed 配对；建议容差 5 个百分点，解码 seed `[1707,2707]`，每个候选前后各 24 局 |
| 最终评价 | 固定 U50，140 Seen＋134 Unseen game，均用 `[1707,2707]`；六 task 分项、每 game 配对差、SR/repair/regression；不能以最优 Unseen 选择分支/终点。训练 seed 仅 1 个，解码重复不能当作训练重复 |
| 推理后端 | 四臂 GRPO generation 与整局 gate/final 全部 `vllm_v1`，注册文件 `configs/phase3_vllm_v1.json`（BF16，4608 context，8192 batched tokens，16 seq/GPU）；优化仍 native verl/FSDP。全词表 readout 单独用 HF/SDPA，必须过 chosen-token live/offline parity；建议事前容差 max abs log-prob 0.03 |
| 资源 | 相同 8×RTX 5090 GPU ID 列表（默认 0–7），四臂顺序运行；若新服务器初始空闲至少约 3 TiB，可拟定每臂 `maximum_run_bytes=751619276800`（700 GiB）、全盘 `minimum_free_bytes=214748364800`（200 GiB）、下一检查点 `checkpoint_reserve_bytes=64424509440`（60 GiB）。这是容量草案，不是实测准入值；必须按新服务器实际空闲空间核定后填入。无自动删除 checkpoint/报告。当前本机只剩约 306 GiB，不能据此批准四臂 U50。 |

四臂中没有“中心化范数”独立编辑臂，因此只能把它作为被记录的 magnitude-only 预测参考，**不能**由这四臂宣称 closed-loop 胜过 magnitude-driven 编辑。如论文第三项贡献要比较闭环，需另加独立第五臂或替换一条消融，先重新确认。

## 当前工程实现与缺口

- 当前 `phase3.readout` 已调用稳定 `phase2.stable_direction.token_signals`，按 Phase2 `aggregation_weights(..., game)` 计算 reward/−P/+C 和审计量。自然调用覆盖规则取代 20/4/8；同一窗口的 old/new 对照和 bank 版本绑定不变。选择器不读 post-update utility gold。
- `phase3.training` 已将四臂生成配置为相同的注册 vLLM V1；`phase3.evaluate` 的整局评估也从 TransformersPolicy 改为注册 vLLM，并按固定 game×seed 计划分给明确的 4/8 个 GPU 进程，成功结果不可覆盖，失败分片保留日志待人工对账；HF/SDPA 只作 readout 全词表前向。每轮指标及每窗 `running_metrics.json` 供途中查看，结束后才标最终完成。
- 已用封存的 placebo/all_alfworld token 标量**只读**重算 reward sign-balance/game score，与封存 skill score 对齐：404 的 18 个技能最大绝对差 `1.67e-16`，505 的 19 个技能为 `2.22e-16`。这只证明标量公式和聚合一致，不证明 Phase3 新模型前向或 rollout/readout chosen-token parity。
- **未完成验收**：Phase3 专项八卡 vLLM 参数同步/恢复＋真实 ALFWorld 生成、正式评估并发/吞吐、Phase3 前向对 404/505 封存全词表/标量的逐值 parity、o3 网关真实 schema/usage/响应兼容性、新服务器完整训练→读出→编辑→gate→续训 smoke test。因此本页是拟议设置，不是 ready-to-run 准入证明。
- 旧 `START.md` 与 `SETTING.md` 仅供 v1 追溯。若准备新服务器，先安装 `requirements-phase3.txt`（含 `vllm==0.22.0`）和运行离线测试；复制 `configs/phase3_runtime_template_v2.json` 到仓库外，填模型/数据/router 绝对路径、已确认数值及磁盘限额。`inference_profile` 指向新部署目录中的 `configs/phase3_vllm_v1.json`。改变任何冻结项须建新 preparation/run 目录，不覆盖旧记录。

确认并补齐硬件数值后，以下仅生成不可变输入与计划，**不启动训练或 API**：

```bash
python -B -m phase3.prepare --runtime /data/phase3-runtime-v2.json --output /data/phase3-assets-v2
python -B -m phase3.run --preparation /data/phase3-assets-v2/manifest.json \
  --root /data/phase3-runs/readout_d --branch readout_d
```

没有 `--execute` 时第二条只打印计划。四臂均须核对计划；当前 setting 的批准位为 `false`，即使手动加 `--execute` 仍会拒绝。

## 准入顺序（不在本次自动执行）

1. 用户确认本页的四臂、指标、U50 和预算数值，并提供新服务器 GPU/磁盘容量；准备清单仍保持不可运行的磁盘 `null`。
2. 各四臂相同注册 vLLM 配置完成真实八卡 Phase3 专项预检（含 native resume、ALFWorld、router、chosen-token parity、gate/final evaluator），不可挪用 Phase1–2 的通过回执。
3. 用专门密钥环境变量对 o3 网关做一次**计入预算的**真编辑 schema/usage 测试；无 fallback，失败保留脱敏账本。然后在新目录做一窗 smoke test，检查编辑、gate、继续训练和逐窗报告；这不等同于正式四臂结果。
4. 所有准入满足后，再由操作者显式执行四条独立正式臂；每次中断只对已完成不可变资产做对账，不自动重试付费请求或覆盖报告。
