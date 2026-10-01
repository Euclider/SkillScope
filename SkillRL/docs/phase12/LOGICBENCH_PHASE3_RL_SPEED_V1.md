# LogicBench Phase3：RL 计算优化 v1

2026-09-30。用户根据U1–U5的481.4分钟实测耗时授权优化。目标是在保持问题、奖励、回答长度上限、GRPO分组和优化器更新次数的前提下减少计算及通信。

## 实验边界

- 原运行：`artifacts/logicbench/phase3-dsign-s707-u50-cpu-v2`，seed707、50次RL更新。
- 每轮128题×8回答=1,024条样本，global PPO minibatch128，4卡时每卡32条；每轮8次optimizer更新、一次scheduler更新。
- 每卡actor/reference microbatch保持1，生成microbatch保持2；loss仍为token-mean。未改变原逐样本归一化。
- Prompt允许上限4096、response64、生成后端/采样/RNG策略保持；router、编辑与gate协议保持。
- U0–U5使用原计算路径。切换须等U5编辑、gate、monitor全部封存后，从U6开始；原native U5含optimizer/RNG状态，继续使用它恢复。
- 五更新快照轮换规则保留。旧配置与历史implementation不覆盖；新计算路径由独立入口与`performance/rl-speed-v1.json`记录。

## 已确认的模型兼容性边界

本机Transformers的Qwen3.5线性attention实现，仅在batch size大于1时将padding hidden states乘以mask。因而裁剪MB1的左padding，或者从MB1增大到MB4/8，会改变原路径的有效token计算。

8条真实U1训练样本的探针结果如下（仅比较有效回答token）：

| 变化 | 最大chosen-token log-prob差异 | 决策 |
|---|---:|---|
| 仅回答logits | 0 | 继续验证/采用 |
| MB1裁剪公共padding | 0.96044 | 不启用 |
| MB4、不裁剪 | 0.84683 | 不启用 |
| MB4并裁剪 | 1.07538 | 不启用 |
| MB8、不裁剪 | 0.96034 | 不启用 |

这不是loss归一化能够解决的问题。本次没有修改Transformers的mask实现，以免把模型行为修正混入性能优化。裁剪功能是默认关闭的候选开关，正式profile保持关闭。

## 回答位置输出层

Actor及reference设置`response_logits_only=true`，调用模型时传入`logits_to_keep=response_length+1`，随后沿用`[-response_length-1:-1]`切片。输入上下文、position IDs、mask和归档训练张量保持原样。

U5实际HF export、FP32参数+BF16 autocast、8条真实记录样本：chosen-token log-prob及entropy的最大差异均为0。单条样本完整梯度比较：余弦0.9999421、相对L2差异0.0125511。数学目标一致，但BF16矩阵乘及梯度求和随输出矩阵形状变化，不宣称逐位相同。单卡梯度首次运行耗时含kernel warmup，不能作为整体训练提速证据。

## Reference与梯度同步验收

`fsdp_workers.py`新增独立的`ref.fsdp_config.cpu_offload`开关，默认True，保留历史行为；显式False才取消FSDP构造时的CPUOffload。它与worker层的`param_offload`分别记录。

四卡初步打分测试（每卡8条、MB1）中，CPU offload路径28.17秒，驻留GPU并使用回答logits后22.06秒，最大log-prob差异0。继续检查reference按层FSDP通信与root FSDP的差别。

`accumulate_no_sync`默认False；可选路径将同一minibatch前面的forward/backward放在no_sync内，最后一次恢复同步。部署前必须测试正式的每卡32次累积，并比较裁剪前梯度。FSDP的BF16本地累积可能带来额外数值误差，不能仅用8次累积结果批准。

## 工件与验证

- `scripts/benchmark_logicbench_rl_forward.py`：同样本单卡前向/完整梯度对比。
- `scripts/benchmark_logicbench_rl_fsdp.py`：四卡reference、真实Adam状态、actor累积/梯度/显存对比；不更新正式checkpoint。
- 工件：`artifacts/logicbench/rl-speed-v1/`。首个8次累积探针在reference测试后被32次累积探针替代，不能把其未完成actor段算作验证通过。
- Phase3及新增针对性测试已通过；最终数量与部署结果在下节更新。
- 较广回归：956 passed、9 skipped、27 failed、22 errors。失败涉及已缺失的旧ALF准备/评估工件，以及历史vLLM环境锁与本机不匹配；完整测试名称及错误见`artifacts/logicbench/rl-speed-v1/regression.log`，未宣称全仓测试通过。

## 部署状态

北京时间10:59已登记性能profile并完成切换，新后台PID为3545131，GPU仍为1–4。旧U5已完整封存，候选技能库被接受：gate从93.5221%提升到94.2708%（+0.7487pp），monitor为90.4297%。新运行从原native U5模型/optimizer/RNG恢复，目标仍为50次RL更新。

生效选项：

| 选项 | 从U6起 |
|---|---|
| Actor/reference `response_logits_only` | True |
| Actor `accumulate_no_sync` | True，同一32次累积的前31次不做梯度同步 |
| Reference `fsdp_config.cpu_offload` | False |
| Reference `fsdp_config.param_offload` | False |
| Reference `fsdp_config.wrap_policy.disable` | True，整模型FSDP |
| 公共padding裁剪 | False |
| actor/ref/生成microbatch | 1 / 1 / 2，保持原值 |

最终四卡探针（GPU2–5）使用真实U1样本及U5权重；时长取最慢rank。正式续训仍用GPU1–4，不能把探针倍数直接当作整轮训练倍数。

| 子阶段 | 原/对照路径 | 新路径 | 结果 |
|---|---:|---:|---|
| Reference，每卡8条 | CPU+逐层FSDP 28.760秒 | GPU+整模型FSDP 1.985秒 | 约14.49倍，log-prob最大差异0 |
| Actor，每卡32条，一个global128 minibatch | 已启用回答logits但逐microbatch同步：353.422秒 | 回答logits+no_sync：153.990秒 | 约2.30倍；该对照已含回答logits优化 |
| Actor峰值，GPU ref和真实Adam同时驻留 | 49.401 GiB | 53.319 GiB | 80GiB卡内通过 |
| HF生成峰值，GPU ref和真实Adam同时驻留 | — | 38.242 GiB | 原microbatch2、response上限64，本次生成4 tokens |

no_sync比较的是**裁剪前**梯度：原norm1.523397，新norm1.546211；各分片relative L2为0.023919/0.017508/0.011781/0.006244，cosine为0.999912/0.999848/0.999931/0.999981。数学loss与optimizer边界保持，但FSDP的BF16累加顺序改变。梯度裁剪能抵消整体尺度部分，不能消除方向和逐参数差异。因此U0–U5和U6起明确标记为不同数值执行版本，不宣称bitwise或训练轨迹等价。

probe不更新正式policy；32条/卡覆盖一个完整optimizer minibatch，但不等于完整RL迭代的8个optimizer minibatch。实际整轮提速、8次optimizer计数与长时间峰值，以U6及后续训练指标为准。代码额外记录`actor/optimizer_steps`。

最终相关回归：**525 passed**、8条既有相关系数警告；新增优化测试包括真实loss/梯度比较、8次optimizer计数、handoff成功/失败分支。独立代码审查完成；新模块与探针的Ruff F检查通过。

性能receipt：运行目录下`performance/rl-speed-v1.json`，SHA256 `3258ec2293231791a851bf91b40e13488b77c97026a8933d4811fce62963be6d`。冻结源码、验证工件、Python包版本、原setting、超时恢复receipt及U5封存证据均绑定。有效配置摘要另见`performance/effective-u0005-u0010.json`。

切换时仅在U5 seal及native/HF验证后清理U4；U5仍保留，空闲磁盘回升至约159.5GiB。后续仍每五更新保存快照，并在下一窗口完整封存后轮换。

恢复入口：

```bash
scripts/run_logicbench_phase3_fast.sh \
  --setting artifacts/logicbench/phase3-dsign-s707-u50-cpu-v2/setting.json \
  --root artifacts/logicbench/phase3-dsign-s707-u50-cpu-v2 \
  --gpus 1,2,3,4 --execute
```

只有窗口内部中断才按原协议附加准确的`--resume-update`，例如保有U9且U10未完成时为9。不要把已开始的窗口重新当作全新窗口执行。本次handoff已完成，审计见`performance/handoff-complete.json`，主日志为`logs/rl-speed-handoff-20260930T025925Z.log`。
