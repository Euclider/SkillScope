# WebShop 加速配置：hf-exact-length-v2

2026-10-07 用户要求停止当前 dense-v1 正式运行，优化后从原始 U0 重跑。旧运行停在 U1 rollout，尚无 optimizer update；轨迹、日志和旧预检结果保留。

## 输入与评分契约

使用 `webshop54_phase12_accel_v2`，保持 Qwen3.5-4B、同一冻结 54-skill bank、冻结 router、状态渲染、任务划分、128 task/update、8 trajectories/task、5 updates/seed、512 个 response slots 与全词表 D/reward 计算。

16,384 是 prompt 的无截断上限与训练证据存储宽度，不再是必须参与计算的前向宽度。

- HF generation 在每个 rank 内按**完全相同的真实 token 长度**分组，上限 16 条。每组去掉共同左侧 padding，生成后恢复原 16K 存储宽度和全局行顺序。
- FSDP 在整次生成请求前只聚合一次参数。各 rank 的长度组数量可以不同；不得在每个长度组内触发 collective。整个请求共享 autocast 参数缓存，避免逐 microbatch 清理显存。
- actor 与 reference 都启用既有 `trim_common_left_padding`。native actor training/log-prob microbatch=1，reference log-prob microbatch=1，保留原 position IDs、完整 response slots 和实际 mask。
- reference 在 GPU 上保留分片参数，并只生成 response logits，避免逐行 CPU offload 和完整 prompt 的词表投影。actor 的 FP32 主参数、optimizer offload 与训练梯度同步方式保留。
- readout 必须读取实际 U1 manifest 的配置，使用同样的 trim 函数。skill/control 都按各自的有效 prompt 去掉左侧 padding；不丢弃 masked response token，不改变实际 loss-token 选择。
- paired evaluation 使用相同完整事实包与 prompt 上限，以真实 prompt 长度生成。其 prediction lock 和训练 manifest 的 contract 必须一致。

这是显式的新前向契约。Qwen3.5 不保证不同 padding 形状的概率完全相同；不声明与 dense-v1 轨迹或概率相等，也不混合两个配置的评分与 outcomes。原 dense-v1 配置与默认行为保持可重现。

## 训练证据与存储

新 U1 证据直接将原生 `torch.save` 字节流写入 gzip level-1，避免未压缩大文件的写盘峰值。边写边记录未压缩字节的 SHA256，写完后验证 gzip 解压流的 SHA256 完全一致，再登记 `training_batch.pt.gz`。张量 dtype、输入、positions、mask、GRPO advantages 和 native old log-prob 都不改动。旧实验文件不清理。

读取使用 `webshop_phase12.training_storage.load_training_archive`；训练 manifest 提供 `batch_file / batch_compression / batch_sha256 / uncompressed_batch_sha256`。Phase3 消费者应使用该 helper，不假定证据只存在 `.pt` 文件。

仅 8-task 预检的 native checkpoint 保存 model；仍保留真实 optimizer-step 日志。正式两个 seed 保存 model/optimizer/RNG 全状态，native checkpoint 继续位于专属 `/dev/shm`，merged endpoint 与训练证据在持久盘。

预检使用明确的 `checkpoint.export_model_only=true`，并标记 `model_export_only.json` 为不可 resume；默认 FSDP checkpoint 仍强制保存 model/optimizer/extra。Ray object store 固定为 16GiB，避免预检自动预留约 140GB；本实验进程的 `local_fs_capacity_threshold=0.999`，启动时仍检查持久盘 ≥25GiB、tmpfs ≥55GiB，不修改其他作业配置。

## 验证与运行

首次粗略 GPU probe：相同 1,025-token prompt，greedy 16-token 响应，dense width=16,384 / batch=4 约 8.31s；有效长度 / batch=4 约 0.98s，batch=8 约 1.16s。首轮含 warmup/JIT 影响，不是正式吞吐结论。正式比较必须读取同规模 rollout 的 timing。

上线 gate：相关 CPU 回归通过；真实 GRPO U1 训练完成；完整 readout 与 native chosen-token log-prob 最大差异 ≤1e-3；配对重放/事实包重放与冻结 bank/router 验证通过。新的 source/model snapshot 与新 cache namespace 必须在开跑前固定。

```bash
python -m webshop_phase12.snapshot
python -m webshop_phase12.run --root "$WEBSHOP_RUN_ROOT/smoke-v2" \
  --prepared "$WEBSHOP_RUN_ROOT/prepared-smoke" --smoke \
  --gpus 1,2,4,5 --router-gpu 7 --evaluation-gpu 1 \
  --config webshop54_phase12_accel_v2
```

上述卡号只是本机示例；运行前检查空闲卡。正式训练需先通过新预检，使用新的 run root，并恢复正式 128-task × 8 trajectories、5 updates 与两个 seed 的准备文件。
