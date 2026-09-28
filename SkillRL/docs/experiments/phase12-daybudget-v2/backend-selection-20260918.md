# 后端选型追加：优先 verl + vLLM，尚未迁移

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: plan
- Origin Date: 2026-09-18
- Verification Status: UNVERIFIED
- Version Label: phase12_high_throughput_backend_selection_v1

## 用户新增要求与当前事实

用户在 v2 offload 预检结束后询问是否使用 vLLM，并要求尽量选择 RL 速度最快的框架。本轮做本地只读核查及官方资料检索；没有安装新依赖、切换运行配置、追加 GPU 预检或启动正式 RL。

当前实际路径是 vendored **verl + FSDP1 + Transformers/HF rollout**，`alfworld_skillnet37_skillrl_v1.yaml` 中 `actor.strategy=fsdp`、`rollout.name=hf`。环境安装信息：verl 0.3.1.dev0（本地已有大量修改，版本号不等于纯上游版本）、torch 2.11.0 / CUDA 13.0、transformers 5.10.4、Ray 2.43.0、sentence-transformers 6.0.1。vLLM、SGLang、flash-attn、fla-core、causal-conv1d 均未安装；`fla`/`causal_conv1d` 不可 import。

硬件实际核查为 8×RTX5090，driver 595.58.03。`nvidia-smi topo -m` 显示 GPU 之间为 PIX/PXB PCIe 连接，没有 NVLink 标记。此前“八卡”不表示采用 vLLM、连续批处理或八个高吞吐推理服务。

## 选型建议（不是已验证最快的结论）

优先候选：**verl 编排/GRPO + FSDP2 训练 + vLLM rollout/评估**；SGLang 保留为兼容性/吞吐备选。优先延续已有 ALFWorld、GRPO 和 readout 接口；不为框架替换同时改变研究目标、训练算法或库/路由设置。

Qwen3.5-4B 官方模型卡列出 vLLM、SGLang 部署方式，也提供 `--language-model-only` 的 vLLM 纯文本运行配置。因此可以将二者视为模型支持候选，但文档不是本机 RTX5090 上 RL 更新、权重热同步和 readout 的验收证据。[Qwen3.5-4B 官方模型卡](https://huggingface.co/Qwen/Qwen3.5-4B)

verl 官方安装文档将 FSDP/FSDP2 列为研究原型的训练后端，把 vLLM/SGLang 列为 rollout 后端；当前文档列 vLLM>=0.18.0。官方模型扩展文档也要求检查训练加载、推理架构和权重布局的共同支持。这些是迁移边界，不代表现有旧 fork 对所有新版本兼容。[verl 安装](https://verl.readthedocs.io/en/latest/start/install.html)、[FSDP 模型扩展](https://verl.readthedocs.io/en/latest/advance/fsdp_extension.html)

候选执行策略：

1. 八卡 rollout/评估先验收 **TP=1、八份副本、每卡批量生成/服务**。4B 可单卡推理，且本机互联为 PCIe，优先降低逐 token 跨卡通信；这是基于硬件和负载的工程推断，仍需吞吐/显存实测，不能说必定最优。
2. 用支持 Qwen3.5 的高效 attention/GatedDeltaNet kernel；核查其对 RTX5090/SM120 及当前 torch/CUDA 的支持。vLLM 官方 Qwen3.5 recipe 提供 Blackwell/CUDA13 路径，但其大模型 H200/MI300/GB200 示例不能直接当本机4B测速。[vLLM Qwen3.5 recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3.5.html)
3. 在同步 on-policy GRPO 边界内并行独立环境请求/批量推理；不引入跨更新采样或 policy staleness。训练和 rollout 分时复用八卡，验证 sleep/wake 与权重同步，不默认永久划出某张 policy GPU 做 router。
4. 沿用 BF16 policy 执行与 FP32 readout 分布记录。初次迁移不额外加入 FP8/INT4、speculative/MTP、改变采样温度或截断长度；这些可能引入额外数值/协议变量。Prefix caching/CUDA graph 只有在 Qwen3.5 状态缓存和显存边界通过测试后启用；recipe 明示部分 Mamba prefix-cache 模式仍属实验性。
5. 时间指标按 rollout、router、训练 update、checkpoint/weight sync、full-vocab capture、performance、O/P/N、报告分别记录。选择依据为全部必需阶段的壁钟时间和峰值容量，不仅是纯生成 tokens/s；当前不能宣布任何框架绝对最快或30h必定完成。

## 本地迁移边界

- 使用独立 venv/版本化适配目录，不覆盖当前研究环境、未提交源码或旧实验。是否需要整体升级 vendored verl，须在现有适配器核对后确定，不直接替换整个源码树。
- 本地 `setup.py` 的 vLLM extra 已写 `vllm==0.19.1`，SGLang extra 仍写 `sglang==0.5.5`/torch2.8，但实际均未安装；声明依赖不是实测通过的锁定组合。vLLM rollout/sharding manager 已有版本适配代码，仍需要新组合完整验收。
- vLLM 的采样 token/top-k logprobs 不能代替 Phase1–2 所需完整词表分布。真实训练 worker 的 before/after FP32 分布、实际 optimizer batch/advantage、step→skill→token 对齐必须保留；推理与训练引擎数值差异及 rollout logprob 另行留档。
- 必测 Qwen3.5 text-only 模型导出命名/权重映射、HF→vLLM 每次更新后同步、EOS/padding/action token 对齐、thinking=false、sampling seed、512输出/4096prompt边界、原生 checkpoint/optimizer/RNG 恢复，至少两个连续更新，再测实际 capture 容量。
- 若更换训练/生成执行协议，建立 v3 preparation/新缓存/新 run root，不能继续使用 v2 失败配置的哈希或旧150更新 permit。相同 seed 在不同后端不保证同轨迹；引擎版本、数值精度、并行度需冻结并在后续 baseline 共用。

## 当前停点

v2 的一次授权预检已经结束：第一步训练/恢复/长 prompt 生成通过，第二步 backward OOM，八卡释放，证据保留；见 [主记录](README.md)。本建议尚未落实为 vLLM/FSDP2 代码或环境，也没有新吞吐实验。接下来建议优先确认高吞吐后端迁移与新验收方案，再决定是否继续旧 HF offload 路径的局部修复；不得把这个建议文件当正式 RL 或无上限预检授权。
