# WebShop 正式加速配置：hf-bucket256-v3

配置为 `webshop54_phase12_accel_v3`。沿用 [v2 的工程与存储改动](ACCELERATION_V2.md)，将完全相同长度分组改为 256-token 桶。真实预检发现 exact-length 分组会把 16 条输入拆成 12 个小批次，限制实际并行度。该预检尚无 optimizer update，原记录保留。

## 固定状态下的前向定义

对含完整所选 skill guidance 的 prompt，令真实 token 长度为 L，规范宽度为 `W=256*ceil(L/256)`，上限 16,384；不截断文本。

- 每个 rank 按 W 分组，每批最多 16 条。每条只计算其所属桶中的左侧 padding，最多 255 个；结果恢复原始行顺序与 16K 证据存储宽度。
- FSDP 仅在整次生成请求的边界聚合/释放完整参数，要求单一 root wrapper；桶内设置 `synced_gpus=False`。Transformers 会自动检测 FSDP 并逐 token all-reduce，必须显式关闭，否则不同 rank 的桶数不同会造成 collective 顺序不一致。
- actor/reference 的逐行前向使用同一 W，保留所有 512 个 response slots、原始 positions 与 mask。native actor/reference log-prob 和训练 microbatch 都为 1。
- 同一训练决策的 old/new × skill/control 四条件 **全部使用含 skill 的原始 prompt 所定义的 W**。control 只删除 guidance，保留 W，避免移除技能时顺带改变前向桶宽。
- paired continuation 的 skill/control 同样从当前完整 guidance prompt 定义 W；control 的实际输入移除目标 guidance 后补齐到该 W。U0 自然锚点、环境 prefix 与事实包重放仍保持原协议。

bank、router U0 权重、状态定义、任务 split、128 task/update、8 trajectories/task、5 updates/seed、响应上限、GRPO 参数与 D/reward 定义不变。此配置独立从 U0 开跑；不得与 dense-v1 或 exact-length-v2 的结果混合。

## 验证门槛

需重新完成真实 U1、完整 native chosen-logprob parity ≤1e-3、四条件 readout、固定 prefix/事实包重放和冻结 bank/router 检查。CPU 测试覆盖桶分组、行顺序、同宽 control、无截断上限、完整 response slots、无损 gzip 与预检 export 的不可 resume。

Phase3 接入时从实际训练 manifest 读取 `forward_contract`。使用 `training_storage.load_training_archive` 和上述 W 定义，不能假定 `training_batch.pt` 或固定 16K GPU 前向。

## 本机已完成的验证（2026-10-07）

- 287 项相关 CPU 回归通过；10 条依赖/CPU autocast 警告。
- 8-task × 8 trajectories、最多 8 步的真实 U1 完成，两个 rank 各 4 次 optimizer step；两张 A800，router 与其中一张共享。
- 全部 444 个训练决策、8,545 个实际 loss token 的 native old chosen-logprob 最大绝对误差为 **0**，预测已在独立 outcomes 前锁定。
- 原生训练序列化约 265.7MB，无损 gzip 约 8.0MB；解压字节 SHA256 与原生序列化流一致，所有 512 个 response slots 完整保留。

同规模原 dense-v1 与新 bucket-v3 的观测耗时如下。两者采样轨迹与物理卡分配不同；新预检仅导出 model，正式 checkpoint 仍保存完整 optimizer/RNG。以下不是正式 128-task batch 的吞吐保证。

| 阶段 | 原 dense-v1（秒） | 新 bucket-v3（秒） |
|---|---:|---:|
| generation + environment/router | 1005.582 | 178.166 |
| native old log-prob | 327.013 | 30.360 |
| reference log-prob | 1348.076 | 494.334 |
| actor update | 4208.257 | 2205.600 |
| 完整 U1（含保存） | 6953.636 | 2957.437 |

完整 U1 约快 2.35 倍；仅策略生成累计 wall-time 为 948.76s → 121.09s，约快 7.84 倍。router 占观测 rollout 的 18.76%，通过原 20% 门槛。正式两 seed 在完整配对重放预检通过后另行开跑。
