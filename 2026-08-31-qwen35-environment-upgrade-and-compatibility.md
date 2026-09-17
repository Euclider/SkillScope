# Qwen3.5-4B 环境升级与 SkillRL 兼容性记录（2026-08-31）

## 1. 结论

当前 `/home/wangyifan/miniconda3/envs/skill-RL` 已原地升级到能够识别并训练 Qwen3.5 架构的依赖栈。现有 8 张 A800 80GB 的算力足以开展 Qwen3.5-4B 的两卡或多卡 FSDP 小规模 GRPO 验证。

目前已经证明的是：

- Transformers 能识别 `Qwen3_5Config` 与 text-only `Qwen3_5ForCausalLM`；
- Qwen3.5 的 Gated DeltaNet/全注意力混合层在 A800 上完成 BF16 forward + backward；
- 现有 SkillRL Phase 1 的 31 项回归测试全部通过；
- 更新后的 Hugging Face rollout 能用本地 Qwen2.5-1.5B 完成真实 GPU 生成；
- vLLM 0.19.1 能独立加载并推理本地 Qwen2.5-1.5B。

随后已将 Qwen3.5-4B 下载到 `/home/wangyifan/model/Qwen3.5-4B`，并完成完整权重的 text-only 加载、裸 Transformers 短生成和 SkillRL `HFRollout` 短生成。尚未执行的是一次端到端 RL update。Qwen 官方仓库：<https://huggingface.co/Qwen/Qwen3.5-4B>。

## 2. 回滚点

升级前环境已完整克隆到：

```text
/home/wangyifan/miniconda3/envs/skill-RL-backup-20260831
```

备份环境约 9.0GB，保留原栈：PyTorch 2.6.0+cu124、Transformers 4.51.1、vLLM 0.8.4。备份环境运行 `tests/phase1` 的结果同样是 31 passed。

当前升级后环境约 13GB。回滚时直接激活备份环境即可，不需要覆盖或删除当前环境：

```bash
conda activate /home/wangyifan/miniconda3/envs/skill-RL-backup-20260831
```

## 3. 当前版本

| 组件 | 版本 | 用途 |
|---|---:|---|
| Python | 3.10 | 保持当前 conda 环境解释器不变 |
| PyTorch | 2.10.0+cu128 | Qwen3.5 训练与 CUDA 12.8 runtime |
| Transformers | 5.10.4 | Qwen3.5 模型注册与 text-only 加载 |
| vLLM | 0.19.1 | Qwen3.5 可用的独立推理栈 |
| fla-core | 0.5.2 | Gated DeltaNet/线性注意力 fast path |
| causal-conv1d | 1.7.0 | Qwen3.5 线性注意力卷积内核 |
| protobuf | 5.29.6 | 满足当前 Ray/vLLM 依赖交集 |

没有升级到需要 CUDA 13/PyTorch 2.11 的 vLLM 0.22.1：下载阶段发现该组合会提高对当前 535 驱动的风险，已在任何包安装前中止，最终选择 CUDA 12.8 路线。

`fla-core` 在 Python 3.10 下会给出“推荐 Python 3.11”的提示，但本机的实际 CUDA forward/backward 已通过。因此它是提示，不是本轮阻塞项。为降低原地升级风险，没有同时替换 Python 主版本。

`pip check` 只报告升级前已经存在的 `decord 0.6.0` 和 `textworld 1.7.0` 平台元数据警告；核心升级包不存在版本冲突。

## 4. 代码适配

### 4.1 Qwen3.5 text-only policy

Qwen3.5-4B 官方 checkpoint 是统一视觉语言模型，但 ALFWorld 只需要文本 policy。worker 新增 `actor_rollout_ref.model.load_text_only=true`，通过 `AutoModelForCausalLM` 只加载语言模型并忽略视觉塔，避免无用显存和参数更新。

同时新增可配置的 `actor_rollout_ref.model.attn_implementation`；Qwen3.5 启动器使用 `sdpa` 处理全注意力层，Gated DeltaNet 层使用 FLA/causal-conv1d。

### 4.2 Transformers 5 API

Transformers 5 移除了 `AutoModelForVision2Seq`。worker、FSDP checkpoint manager 和 model merger 已兼容到 `AutoModelForImageTextToText`，同时保留 Transformers 4 的回退导入。

HF rollout 也改为只通过一个 `GenerationConfig` 传递生成参数，适配 Transformers 5 的 generate API。

### 4.3 rollout 后端边界

当前可用于首次 Qwen3.5 RL 验证的后端是：

```text
actor_rollout_ref.rollout.name=hf
```

它直接使用正在训练的 FSDP actor 生成下一批轨迹，因此一次 update 后的新参数天然会进入下一轮 rollout，满足本实验观察 policy 更新与 skill utility/action flip 的正确性要求。代价是吞吐低于 vLLM。

vLLM 0.19.1 的独立加载和生成已通过，但本 SkillRL fork 的 embedded vLLM sharding manager 使用 pre-V1 的 `model_executor.driver_worker.worker.model_runner` 内部路径；vLLM 0.19.1 已改为 EngineCore/collective RPC 架构。两者不能安全完成每次 RL update 后的在线权重同步。代码现在会对此组合 fail-fast，防止 rollout 悄悄使用旧 policy。后续若要恢复 vLLM 吞吐，需要单列任务迁移新版 verl 的 server-based weight transfer，不应把它与首轮科学验证混在一起。

## 5. 新增入口和复现实验

Qwen3.5 专用启动器（默认 `hf`）：

```text
/home/wangyifan/skill-RL/SkillRL/examples/grpo_trainer/run_alfworld_phase1_qwen35.sh
```

它复用现有 step-routed Skill Bank 方案，并增加以下保守设置：

- text-only model；
- SDPA + FLA；
- 关闭 remove-padding 与 torch compile；
- actor/ref/log-prob micro batch 降为 1；
- actor/optimizer 不 offload，避免 HF rollout 前后的参数驻留歧义；
- HF sampling 的 `top_k=0`。

环境预检入口：

```bash
cd /home/wangyifan/skill-RL/SkillRL
conda activate skill-RL
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
CUDA_VISIBLE_DEVICES=0 python -m phase1.qwen35_compat_preflight \
  --run-kernel-smoke \
  --output artifacts/preflight/qwen35-environment-20260831.json
```

本轮预检归档：

```text
/home/wangyifan/skill-RL/SkillRL/artifacts/preflight/qwen35-environment-20260831.json
```

首次环境预检中所有依赖、CUDA、模型注册、tiny Qwen3.5 forward/backward 均为 pass；当时 `local_model` 为 warning。模型下载后重新运行 `--require-model` 预检，`local_model` 已为 pass。

下载后的完整模型预检与加载/生成归档：

```text
/home/wangyifan/skill-RL/SkillRL/artifacts/preflight/qwen35-full-model-20260831.json
/home/wangyifan/skill-RL/SkillRL/artifacts/preflight/qwen35-full-model-load-generate-20260831.json
```

下载 revision 为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。索引中的 738 个 tensor 全部能在两个 safetensors 分片中找到。

Qwen3.5 默认会进入 thinking mode。ALFWorld 需要短而可解析的动作，因此专用启动器已固定 `data.apply_chat_template_kwargs.enable_thinking=false`。关闭后，完整模型对测试 observation 输出 `go to table 1`；同一输入通过 SkillRL `HFRollout` 也输出 `go to table 1`。

## 6. 已执行验证

1. 升级前备份环境：`tests/phase1`，31 passed。
2. 升级后当前环境：`tests/phase1`，31 passed（4.12s）。
3. A800 BF16 矩阵运算：pass。
4. tiny Qwen3.5（1 个 linear-attention layer + 1 个 full-attention layer）forward/backward：pass，归档 loss 为 4.887589。
5. 本地 Qwen2.5-1.5B 通过 vLLM 0.19.1 真实加载/生成：pass。
6. 本地 Qwen2.5-1.5B 通过修改后的 HF rollout 真实加载/生成：pass。
7. Transformers 5 worker、checkpoint manager、model merger 导入：pass。
8. 两个 launcher 的 bash 语法检查：pass。
9. 完整 Qwen3.5-4B text-only 加载：pass；类为 `Qwen3_5ForCausalLM`，参数量 4,205,751,296，BF16 CUDA allocated 约 7.94GiB。
10. 关闭 thinking 后的完整模型短生成：pass；SkillRL `HFRollout` 组合验证：pass。

## 7. 完整 RL 前仍需完成

1. 用 HF rollout 仅跑 1 个真实 update，验证：trajectory archive、非零参数 delta、checkpoint 保存/恢复、更新后 rollout 确实改变。
2. 1-update 通过后，再启动 5 seeds × 大累计 update checkpoint 的正式泛化实验。

模型下载后根分区剩余约 368GB。Qwen3.5-4B 本身没有空间问题，但 5 seeds × 3 checkpoints 若每个都保存 model + Adam optimizer + extra，可能超过剩余空间。正式运行前应确定 checkpoint 归档策略：用于横向/纵向评估的选定 checkpoint 保存 model 权重；只有确实需要续训的最近 checkpoint 保留 optimizer state。
