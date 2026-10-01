# RL 计算优化 v1：跨服务器同步与独立验收

2026-10-01。本次同步供另一台服务器审核代码、重跑探针，并在自己的下一个完整窗口边界登记计算后端切换。**仓库中的原始 receipt 是本机既有运行的证据，不是另一台机器的验收凭证。**

## 文件入口

以下路径相对 `SkillRL/`：

| 用途 | 文件 |
|---|---|
| 核心前向和梯度累积 | `verl/workers/actor/dp_actor.py`、`verl/workers/actor/padded_forward.py` |
| Reference FSDP offload 开关 | `verl/workers/fsdp_workers.py` |
| 生效选项与完整性检查 | `logicbench_rl_speed/runtime.py` |
| 原运行入口与切换记录逻辑 | `logicbench_rl_speed/launch.py`、`handoff.py`、`scripts/run_logicbench_phase3_fast.sh` |
| 单卡同样本前向/梯度探针 | `scripts/benchmark_logicbench_rl_forward.py` |
| 四卡 reference、梯度累积和显存探针 | `scripts/benchmark_logicbench_rl_fsdp.py` |
| HF 生成与 reference/Adam 共存显存探针 | `scripts/benchmark_logicbench_rl_generation.py` |
| 方法、实测差异与启用/拒绝项目 | `docs/phase12/LOGICBENCH_PHASE3_RL_SPEED_V1.md` |
| 原始探针 JSON、日志、回归日志 | `artifacts/logicbench/rl-speed-v1/` |
| 本机 performance receipt、有效配置摘要 | `artifacts/logicbench/phase3-dsign-s707-u50-cpu-v2/performance/` |
| 原 setting、implementation、timeout receipt、U5 seal | 上述运行目录下对应原路径 |
| 文件校验清单 | `docs/phase12/RL_SPEED_V1_SHA256SUMS`（在仓库根目录执行 `sha256sum -c`） |

同时同步 runtime 所依赖的 LogicBench 单步训练、窗口恢复、CPU/GPU router 适配、冻结初始 bank、数据准备脚本和测试夹具。它们使独立 checkout 的导入与 CPU 测试完整；ALFWorld 路径仍按原配置分派。全词表张量、模型、optimizer checkpoint、编辑账本、凭据和本机缓存不随 Git 上传。SRA 技能来源与许可在 `memory_data/logicbench/sra19/manifest.json` 和 `LICENSE`；数据拆分来源/哈希在 `data/logicbench/sra19/aug_split_v2/manifest.json`。

## 有效优化与不可直接外推的结论

- Actor/reference 仅计算末尾 `response_length + 1` 个 logits，保留原上下文、mask、位置编码。
- Actor 同一 optimizer minibatch 的前若干 microbatch 用 `no_sync()`，最后一次同步；优化器和调度器更新边界不变。
- Reference 的 FSDP CPU offload 与 worker param offload 均关闭，并采用整模型 FSDP 包裹；需要重新验证目标卡的显存容量。
- **公共 padding 裁剪保持 False，actor/ref/生成 microbatch 保持 1/1/2。** 本机 Qwen3.5 的 MB1 padding 路径使裁剪或扩大 batch 的 chosen-token log-prob 偏差明显，未批准这些项目。
- 本机原训练每 RL 更新 128 题 × 8 回答，global PPO minibatch 128，8 次 optimizer 更新；不能将这里的 32 次/卡梯度累积数直接套到不同卡数或 ALF 决策行规模。
- 同样本回答 log-prob 和 entropy 的 suffix 探针最大差异为 0；BF16 梯度不是逐位等价。suffix 相对 L2 为 0.0125511；完整 32 次 no_sync 累积最大分片相对 L2 为 0.0239191，最小 cosine 为 0.9998476。
- 四卡 reference 探针 28.760→1.985 秒；actor 353.422→153.990 秒（对照已启用 suffix）。这些是局部探针，不能当作完整 RL 更新提速倍数。

原始完整验收环境：Torch 2.10.0、Transformers 5.10.4、Ray 2.43.0、TensorDict 0.10.0，四卡 80 GiB GPU。另一台服务器若使用 Torch 2.11、不同卡型、卡数、模型或数据，需按自己的环境独立验收。

## 先验文件校验与 CPU 测试

从仓库根目录执行：

```bash
sha256sum -c SkillRL/docs/phase12/RL_SPEED_V1_SHA256SUMS
cd SkillRL
CUDA_VISIBLE_DEVICES='' python -B -m pytest -q \
  tests/phase3 tests/phase1/test_watch_sra_logicbench_gpus.py \
  tests/phase1/test_logicbench_single_step.py \
  tests/phase1/test_logicbench_train_rollout.py \
  tests/phase1/test_prepare_sra_logicbench_aug.py \
  tests/skill_bank/test_sra_logicbench_bank.py \
  tests/skill_router/test_logicbench_embedding_router.py \
  tests/phase2/test_logicbench_fixed_state_scoring.py
```

使用已有训练环境，不要为通过 receipt 检查直接替换另一台服务器的依赖。此清单独立于历史 `PROJECT_SHA256SUMS`；后者记录旧发布快照，本次没有将它冒充新版本的全仓校验清单。

本次在独立发布 checkout 中执行以上 CPU 验证：**232 passed**，日志为 `artifacts/logicbench/rl-speed-v1/publish-regression-20261001.log`。本次未占用训练 GPU 重跑已有 GPU 探针，也不宣称全仓测试通过；历史广域回归的缺失 ALF 工件/环境失败仍保存在原 `regression.log`。原始日志保留字节（含 pytest 行尾空格），代码/说明文件的 diff whitespace 检查通过。

原 receipt 字节 SHA256：`3258ec2293231791a851bf91b40e13488b77c97026a8933d4811fce62963be6d`。其 `sources` 和 `validation` 引用文件均随本次同步按原路径提供，并逐项检查哈希。原 timeout receipt 中 `authorized_implementation` 引用的当前源码也一并提供。历史 `implementation.json` 是原运行的旧版本摘要，不要求它与后来已登记的新源码相等。

## 在目标机器重跑 GPU 探针

先准备该机器自己的真实方向批次（含 prompts、prompt_mask、responses、actual_loss_mask、advantages、old_log_probs）和封存端点的 HF export。下列参数必须由接收方填写；输出写入新的验收目录，不能覆盖仓库中原始探针结果。

```bash
TASK_BATCH=/path/to/direction_batches/u0001.pt
TASK_ACTOR=/path/to/sealed_endpoint/hf_export
TASK_REF=/path/to/initial_reference_model
TASK_OUT=/path/to/new_acceptance
mkdir -p "$TASK_OUT"

CUDA_VISIBLE_DEVICES=0 python -B -m scripts.benchmark_logicbench_rl_forward \
  --batch "$TASK_BATCH" --model "$TASK_ACTOR" \
  --rows 8 --suffix-only --force-fp32 --gradients \
  --output "$TASK_OUT/suffix-parity.json"

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  --module scripts.benchmark_logicbench_rl_fsdp \
  --batch "$TASK_BATCH" --actor-model "$TASK_ACTOR" \
  --reference-model "$TASK_REF" --rows-per-rank 32 \
  --skip-actor-baseline --output "$TASK_OUT/fsdp-32-parity.json"
```

`--skip-actor-baseline` 只跳过耗时完整 logits actor 对照，仍比较 suffix 同步和 suffix no_sync；先用单卡探针检查完整 logits 与 suffix。需匹配目标窗口实际每卡累积次数，不能只测 8 次却批准 32 次。

**生成探针仍是原始实验脚本**：reference 路径写死为 `/home/wangyifan/model/Qwen3.5-4B`，并读取 `--root` 下 `models/u0005` 和 `direction_batches/u0001.pt`，response 截取基于 prompt 4096、EOS/PAD 248044。接收方应在独立副本中改成实际路径/长度/token ID，记录副本 SHA256 后运行；不要把修改后的探针登记成原始字节。旧 fast launcher 同样固定本机 Python、库目录和 checkout 路径，不可在其他服务器直接执行。

探针不覆盖正式 checkpoint；FSDP/生成显存探针会在内存里分配真实 Adam 状态。除评分/梯度/峰值外，应在目标窗口确认 optimizer 次数、调度器步数、归一化、KL/entropy、生成参数和恢复的 RNG/数据进度。

## 下一窗口的切换登记

1. 完成独立验收，记录实际依赖版本、硬件、源码/探针/结果哈希和接受的数值误差。失败的优化仍保持关闭。
2. 等当前窗口的编辑、gate、monitor、技能库链与 seal 完整；核验 native checkpoint 的 policy、optimizer、RNG 和数据进度。
3. 为目标运行生成自己的计算 profile/receipt，绑定目标 root、setting、封存边界、源文件、验证结果和有效首轮编号，使用新版本号区分历史窗口。
4. 从下一窗口第一轮切换，只改变已验收的计算选项。恢复原 optimizer/scheduler horizon 与 RNG，不重置已完成更新，不重复编辑请求。
5. 记录首轮实际耗时、optimizer 次数及显存，再判断后续效果；按原五更新快照与封存后轮换协议保留恢复能力。

`logicbench_rl_speed/runtime.py` 的本机 profile 固定从 U5 边界、U6 首轮生效，并校验原 root、原 timeout receipt、包版本及 SQLite 中的迁移记录；`handoff.py` 还绑定当时暂停的 supervisor/gate PID。**它们是本机切换实现，不是接受任意目标窗口的通用授权器。** 不能仅替换 receipt 里的绝对路径、跳过检查或伪造 SQLite 记录来使它通过。若目标是 ALFWorld Phase3，应把核心开关接入该运行自己的窗口注册流程，重新记录实际源码和验收结果；本次没有远程切换另一台服务器的运行。
