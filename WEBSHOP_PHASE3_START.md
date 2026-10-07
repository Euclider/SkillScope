# WebShop 实现交接：Phase3 从这里开始

本目录交付的是经过本地 Phase1/2 小规模验证的 WebShop **状态、路由、动作生成、真实 GRPO 与固定状态评分**实现。Phase3 的技能编辑器、优先队列和版本更新循环需要在此基础上另行实现；这里没有把它们标记为已完成。

先读 [状态与路由协议](SkillRL/docs/webshop/PROTOCOL.md)，再读 [Phase3 接入与服务器配置](SkillRL/docs/webshop/PHASE3_HANDOFF.md)。源码入口为 [webshop_phase12](SkillRL/webshop_phase12/)。

| 需求 | 文件 |
|---|---|
| 目标、完整动作历史、近两轮反馈、当前页面、商品证据缓存 | [visible_state.py](SkillRL/webshop_phase12/visible_state.py) |
| 状态机械去重、执行提示、动作投影与 dense 输入 | [prompts.py](SkillRL/webshop_phase12/prompts.py) |
| 原生 WebShop 多 session、reset/replay、逐步状态更新 | [envs.py](SkillRL/webshop_phase12/envs.py) |
| 全部 54 个技能、受限单 token 选择、exact-state 缓存 | [llm_router.py](SkillRL/webshop_phase12/llm_router.py) |
| 固定 Qwen3.5-4B、本地 vLLM、并行副本、前缀缓存 | [llm_service.py](SkillRL/webshop_phase12/llm_service.py) |
| 冻结 bank、ID/正文/版本哈希 | [assets.py](SkillRL/webshop_phase12/assets.py) |
| native loss-mask/full-vocabulary readout | [dense_scoring.py](SkillRL/webshop_phase12/dense_scoring.py)、[readout.py](SkillRL/webshop_phase12/readout.py) |
| 首次自然调用锚点与四条件配对续跑 | [evaluate.py](SkillRL/webshop_phase12/evaluate.py) |
| verl/GRPO/FSDP 参数 | [webshop54_phase12_v1.yaml](SkillRL/verl/trainer/config/webshop54_phase12_v1.yaml) |
| CPU 回归测试和长菜单样例 | [tests/webshop_phase12](SkillRL/tests/webshop_phase12/) |

运行在本机副本和 GitHub 发布副本之间保持隔离。本次迁移对发布副本增加了路径环境变量；正在运行的原实验源码、模型、cache 和日志未改动。

发布范围是源码、原始小型技能 bank、合成/长菜单测试 fixture 与说明。模型权重、商品数据、搜索索引、已安装环境、训练 checkpoint、完整训练张量、真实轨迹和凭据由目标服务器独立配置。

## 已验证与未验证

- 原始 bank：54 个 skill，JSON SHA256 `79c6c60b6757b6b730e7471b537781936ce0e9cdbd8df6188b57c535893aec20`。
- 本地 A800 的真实冻结 router 检查：单条/批量/逆序、sleep/wake、固定目录前缀缓存通过。
- 本地 smoke：真实 GRPO U1 与 checkpoint 合并完成；观察到的路由 wall-time 占 rollout 3.25%。这是 8-task 预检数据，不是正式 128-task batch 的速度结论。
- 完整 450 个训练决策的 dense readout：已选 token 概率与 native 记录最大差异 **0**。
- 另一台服务器尚未执行 GPU 验证；不要把 A800 的依赖与速度直接当作 5090 验证结果。
- 正式 WebShop 两 seed 的结论及 Phase3 优化效果不在本次交付中声明。

## 验证发布文件

```bash
python3 deploy/webshop/verify_handoff.py
```

这个命令验证本次 WebShop 发布清单。`deploy/5090/verify_project.py` 及其旧清单是历史全项目快照验证器；旧清单不随这次新适配器被改写。
