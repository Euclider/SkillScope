# WebShop Phase3 接入契约与目标服务器配置

## 1. 当前数据流

`ShopWorld.reset_one/step_one` 提供原生页面和可见动作；`VisibleMemory.observe/transition` 维护 episode 内证据；`memory.state()` 生成确定的事实包。`FrozenLLMSkillRouter.route_many` 读取事实包和全 bank 目录，选择一个 ID。`build_state_prompt` 加入选中 skill 原文，verl 的 HF rollout 生成一个 `<action>...</action>`，`project_action` 投影后返回环境。每一步都按当前完整事实包重新路由；仅完全相同的输入共享 cached decision。

API 可在不加载模型或商品目录时检查：

```python
from webshop_phase12.visible_state import VisibleMemory
from webshop_phase12.assets import WebshopBank
from webshop_phase12.llm_router import build_router_prompt
from webshop_phase12.prompts import build_state_prompt

memory = VisibleMemory("Find a blue cotton shirt under $30", max_steps=50)
memory.observe("Search", ["search[<your query>]"], {"page_type": "index"})
memory.transition("search[blue cotton shirt]", True, "Results",
                  ["click[B000000001]"], {"page_type": "search_results"})
state = memory.state()
bank = WebshopBank()
router_prompt = build_router_prompt(bank, state)
execution_prompt = build_state_prompt(state, bank.get("gen_001").payload)
```

示例 ID 为合成值。实际 `visible_page` 来自 `public_page(browser.current_url)`，不把 session ID、目标属性、奖励或后台商品字段加入事实包。

## 2. 状态定义与更新

状态为 `task_description / action_history / recent_feedback / current_observation / admissible_actions / current_page / product_evidence / step_index / max_steps`。

- `task_description`：原始购物指令。
- `action_history`：所有已执行动作及其时间、可观察的合法性。
- `recent_feedback`：近两次动作后的页面反馈，包含当时合法动作列表；与当前页重复的文本在提示里用精确引用表示，归档保留原文。
- `current_observation`：原生 `text_rich`，保留选中/访问标记。
- `product_evidence`：已经访问商品的标题、显示价格、明确显示的选择标记，以及实际读过的 Description/Features/Reviews 原文和观察步数。相同事实去重，不做“所有约束已满足”等推断。
- `current_page`：公开地址中的页面类型、已可见 product ID、详情类型。

reset 建立新 memory，禁止跨 episode 混入旧证据。点击失败不能当作访问成功；点击选项不能代替页面中真实的选中标记。`replay_visible` 从相同 action prefix 重建环境和 memory；配对锚点同时验证 native digest 和事实包 digest。

## 3. 冻结 router 的边界

独立原始 Qwen3.5-4B U0 实例，temperature=0、thinking=False、max_tokens=1；54 个固定单 token 标签 A–Z/AA–AZ/BA/BB 映射到 bank 原顺序。没有 BM25、embedding 模型或可学习选择头。

`llm_service.py` 的副本池按长度分配查询并恢复原顺序；集中在 `FrozenLLMSkillRouter` 进行完整输入 hash 去重、缓存 admission 和原子发布。`mamba_cache_mode=align`，block 对齐预热 skill 目录；相同权重的多个 router 副本可以共用一份缓存协议。

缓存 key 包含模型、本地实现/提示/状态 renderer、bank、标签与解码协议 hash，以及完整事实包 hash。失败/未完成输入不能自动重复提交；不得静默选一个 fallback skill。

## 4. Phase3 要新增的接口

当前 `WebshopBank` 明确要求原始 JSON SHA，`Manager`、`FrozenLLMSkillRouter` 和 `LocalVLLM` 都从它构建初始 bank。Phase3 更新正文时，不能仅修改这个 JSON 然后绕过校验。

在另一服务器实现显式的 versioned-bank adapter，并把同一 bank 版本传给三个位置：

1. `Manager`：选中 ID 对应的执行 payload。
2. `FrozenLLMSkillRouter`：catalog、ID 集合、payload hash 与协议。
3. `LocalVLLM`：目录前缀及 KV 预热。

保留原始 ID 和顺序；记录 `bank_version / parent_version / content_sha256 / edited_skill_ids / payload_sha256 / edit_reason`。每个 Phase3 更新窗口内固定 bank，窗口边界提交下一版本。换版本时建立新的路由 cache 命名空间并刷新目录前缀/KV，冻结 router 的 U0 权重不更新。不要用新正文渲染旧版本已缓存的选择记录。

如果增删技能，54-label/one-token contract 必须显式重新验证；当前交付只支持固定 54 个 ID。

优先队列、D/reward 驱动选择与技能编辑器复用项目既有 Phase3 设计；本次不实现它们，也不把 WebShop Phase1/2 的 frozen-bank 路由自动改成可变 bank。

## 5. readout 与效用评估

使用实际训练的 `phase2_actual_loss_mask`、GRPO advantage、原始动作 token。`score_dense` 保留记录的完整 input IDs、attention mask、position IDs、16384 prompt 宽度与全部 512 响应位置；前向完成后才选真实 loss-token 位置。

control 只移除目标 guidance 段，保留响应、response mask 与原始 prompt 宽度，按新 prompt 的真实长度重建位置。四条件 old/new × skill/control 使用同一 HF BF16/SDPA 路径。已选 U0 概率与 native witness 最大误差必须 ≤1e-3；本地 450 决策验证为 0。禁止为了速度换成 unpadded forward：此前该路径最大误差为 6.8535。

paired evaluator 同样使用 16384 dense prompt 预算，固定重放 prefix，技能/control 不重选锚点技能；参考锚点来自 U0 的首次自然调用。在搜集独立 outcome 之前锁定预测。Phase3 的版本间效用比较需另外登记 bank/version/checkpoint estimand，不混合不同 bank 的预测文件。

## 6. 服务器路径与依赖

从仓库 `SkillRL` 目录运行：

```bash
cd SkillRL
export PYTHONPATH="$PWD"
export WEBSHOP_RUN_ROOT=/path/to/fresh/webshop-run
export WEBSHOP_BASE_MODEL=/path/to/Qwen3.5-4B
export WEBSHOP_ENV_ROOT="$PWD/agent_system/environments/env_package/webshop/webshop"
export WEBSHOP_ROUTER_PYTHON=/path/to/your/venv/bin/python
export WEBSHOP_JAVA_HOME=/path/to/jdk
export JAVA_HOME="$WEBSHOP_JAVA_HOME"
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
```

只在确有本机 ABI 需要时设置 `WEBSHOP_RUNTIME_LIBRARY_PATH / WEBSHOP_LD_PRELOAD`。输出默认是 `SkillRL/artifacts/webshop`，Python 默认是当前 `sys.executable`。固定模型 receipt 含本地路径/mtime；不要搬用本机 receipt，在目标机重新生成。

A800 实际测试版本见 [requirements-webshop-observed.txt](../../requirements-webshop-observed.txt)。这是观测清单，并不是兼容所有 GPU 的完整安装 lock。5090 环境应使用能支持该 GPU 的已确认 PyTorch/vLLM 组合，再跑 GPU gate；不能盲目覆盖为 A800 wheel。

原生环境还需 Java、spaCy `en_core_web_lg`、Pyserini 与 gym 0.26。商品数据和 `search_engine/indexes` 单独下载/建立。详情见原生 [WebShop README](../../agent_system/environments/env_package/webshop/webshop/README.md)。原始 JSON、搜索索引和模型权重未上传。

## 7. 上线前检查

```bash
# 不启动训练；先绑定本地模型和源文件。
python -m webshop_phase12.snapshot

# 与实验源码隔离的 CPU 检查。
CUDA_VISIBLE_DEVICES='' python -m pytest -q tests/webshop_phase12

# 给独立空闲卡；不会自动结束其他进程。
export WEBSHOP_ROUTER_GPU=0
export WEBSHOP_ROUTER_GPUS=0
export WEBSHOP_ROUTER_CACHE="$WEBSHOP_RUN_ROOT/router-gate.sqlite3"
CUDA_VISIBLE_DEVICES=0 python -m webshop_phase12.gpu_probe --out "$WEBSHOP_RUN_ROOT/gpu-gate"

# 需要模型、数据、已建好的索引和 Java。
python -m webshop_phase12.prepare --smoke
python -m webshop_phase12.prepare
python -m webshop_phase12.coordinate_llm --queue "$WEBSHOP_RUN_ROOT/new-cohort"
```

新的 queue 路径必须不存在。`prepare --smoke` 的两个任务只用于接口/重放检查；正式 Eval 使用 0–499、Dev 500–1499、Train ≥1500 的原始 split。

native checkpoint 默认保留在独立 `/dev/shm`，机器重启会丢失；若服务器需要持久重启恢复，应先修改自己的存储配置并登记。merged endpoint、训练证据和结果使用持久 run root，不依赖 GitHub 提供旧训练 checkpoint。

## 8. 必须保持的工程行为

- 在 `trainer.init_workers/fit` 的 finally 路径关闭 WebShop 自有环境/router。
- episode 结束后跳过策略生成，用无 response loss 的占位恢复原行位置，不缩减原有 8 trajectories/任务。
- global minibatch 128、每 GPU actor microbatch 1、HF generation microbatch 4；尾批仍需遵守上限。
- 共享 GPU 时，rollout 后 router level-1 sleep，下一轮前恢复；独立 router 卡保留缓存。
- 原有 ALFWorld/LogicBench 的 Phase3 speed audit 仍保留。本次只在 WebShop flag 下选择新的环境、捕获与清理逻辑。

集成源文件位于 `verl/trainer/main_ppo.py`、`verl/trainer/ppo/ray_trainer.py`、`agent_system/multi_turn_rollout/rollout_loop.py`、`verl/workers/rollout/hf_rollout.py`，cache bulk API 位于 `agent_system/memory/router_cache.py`。
