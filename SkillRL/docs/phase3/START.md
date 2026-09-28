# SkillScope Phase3：历史 v1 启动说明（已停用）

当前代码的 Phase2 对齐版请以 [PHASE2_ALIGNMENT_V2.md](PHASE2_ALIGNMENT_V2.md) 为准。本页仅作 2026-09-18 HF/SDPA、seed404、原始 gated D 协议留档；**不要用本页命令启动当前 Phase3**。

本包是可迁移的实现快照，不是已完成的 Phase3 实验结果。默认命令只做离线准备／计划；正式运行必须显式 `--execute`。先阅读 [SETTING.md](SETTING.md)。验收状态与已知限制见 [VALIDATION.md](VALIDATION.md)。

## 1. 固定比较

四条独立、累积演化的 RL 路径：`skillrl_failure`（失败驱动）、`readout_d`（gated D 主实验）、`readout_p`（−P）、`readout_c`（原始 +C_upd）。不能用一条 RL 轨迹同时冒充四条闭环实验。

共同 Qwen3.5-4B B0、SkillNet-37 初始库、seed=404；所有 ALFWorld task 共用完整当前库。GRPO，lr=1e-6，每轮 16×8 轨迹、全局 PPO minibatch=128 个决策行，150 轮上限，每 5 轮验证／编辑。HF rollout、microbatch=1。Router 已改为本地冻结 `Qwen/Qwen3-Embedding-0.6B`，逐状态完整当前库余弦 top-1；编辑器仍为 `o3`，使用 `https://api.zhizengzeng.com/v1`。新 router 不调用 API，没有模型或词匹配 fallback。

这是 SkillRL Embedding 的 **state-aware adaptation**：官方为 task-only 查询、每局复用 general/task-specific top-k；本研究保留当前 observation、admissible actions、最近两步历史与 step index 的单技能读出协议。四分支与本机 Phase1–2 共享相同查询/编码/排序规则；每次 bank snapshot 变化都建立独立索引和缓存。检索器不随 policy 训练。

这里只保存 Phase3 必需的预测、轨迹、编辑、门控、版本和成本记录；不产生 Phase2 O/P/N 效用 gold，不落盘全词表概率或 hidden activations。

## 2. 安装

在新的目录 clone；不要覆盖已有工作区。

```bash
git clone https://github.com/Euclider/SkillScope-phase3.git
cd SkillScope-phase3
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-phase3.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
python -B -m pytest -p no:cacheprovider tests/phase3 tests/skill_router tests/skill_bank tests/experiment_settings -q
```

固定核心版本：Python 3.12、torch 2.11.0、torchvision 0.26.0、Transformers 5.10.4、SentenceTransformers 6.0.1、tokenizers 0.22.2、Ray 2.43.0、OpenAI SDK 3.14.1、tiktoken 0.12.0。安装输出中的 CUDA wheel／驱动需记录；不要为了安装可选包静默改 torch。此 HF/SDPA 路线不安装旧版 vLLM、FlashAttention 或 THOR。缺 FLA/causal-conv1d 时 Qwen 使用较慢的 torch fallback，不能把 import 通过当作吞吐验证。

需要系统编译工具和 TextWorld/ALFWorld 所需运行库。若 `pip check` 或 ALFWorld 导入失败，先解决依赖，不能通过丢弃 game 来继续。不要复制另一台机器的 CUDA 扩展 `.so`。

## 3. 模型与数据

可以使用已下载的、校验一致的本地模型和数据。新下载示例：

```bash
python -B -m phase3.download model --output /data/models/Qwen3.5-4B
python -B -m phase3.download router --output /data/models/Qwen3-Embedding-0.6B
python -B -m phase3.download data --output /data/alfworld
export ALFWORLD_DATA=/data/alfworld
```

模型 revision 固定为 `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`。只下载公开基础模型和文本游戏，不需要旧实验权重。离线准备会核对 3553 train、140 Seen、134 Unseen 个可解文本 games，六类任务，并登记每个 game、tokenizer 与模型文件的 SHA-256。已有数据不会被下载器覆盖。

Router revision 固定为 `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`；下载和 prepare 都验证 `configs/skillnet37_router_qwen3_embedding_v1.json` 的10文件哈希。默认 CPU FP32，以保留八卡 policy 显存；不能未验收就改成共享 CUDA。模型加载是 local-files-only，不会在运行时自动下载或换模型。

## 4. 填运行配置并冻结

将 `configs/phase3_runtime_template.json` 复制到**仓库外**，填写真实绝对路径及所有 `null`，如 `/data/phase3-runtime.json`。四分支共用同一份准备清单。

尚需操作者明确的数值，不是已确认的论文结果：

- `gpu_ids`：单机 4 或 8 张空闲卡；各分支使用相同拓扑。不得占用或终止他人进程。
- `gate_games_per_task`：每个 Seen 任务抽取多少个 gate game。按固定哈希分层选择，剩余 Seen 用于监测／证据；两者不交叉。
- `gate_tolerance_pp`：相同当前策略下，候选库相对旧库允许下降的百分点；例如 `0` 表示不接受观测到的 SR 下降，不代表统计证明非劣。
- `eval_seeds`：非空、去重，所有分支／前后库一致；必须在看结果前登记。
- `router.model_path`、`router.device`：本地检索模型及明确放置；默认 `cpu`。`router.profile_sha256` 保持模板给定值，不按结果修改检索参数。
- `router.max_local_calls`、`editor.max_api_calls`：分别为本地路由和付费编辑的每分支累计上限，包括所有库版本、训练、gate 和最终评估，不随换库/窗口清零。每 5 轮至多一个编辑请求，150 轮至多 30 个机会；NOOP／弃权仍记录。
- `readout_parity_atol`：离线原始条件与实际 actor 的 chosen-token logprob 最大容差，需按新服务器预检确定，不能看效果后放宽。
- `storage`：以 bytes 计的运行目录上限、最小剩余空间、下一 checkpoint 预留。所有检查点默认保留，长跑可能需要每分支 TB 级空间；达到限制停止，**不会自动删除检查点**。

模板的输入／输出 token cap、最多 10 条证据轨迹是可见的工程默认值；编辑前完整当前库与证据若超限就停止，不静默截断或加一次未计费的摘要 LLM。

```bash
python -B -m phase3.prepare --runtime /data/phase3-runtime.json --output /data/phase3-assets-v1
python -B -m phase3.run --preparation /data/phase3-assets-v1/manifest.json \
  --root /data/phase3-runs/readout_d --branch readout_d
```

第二条只打印计划，不加载 GPU、不调用 API。准备文件冻结后不能直接修改；改变设置请创建新 assets/run 目录，不能把新设置混进旧结果。

## 5. GPU 与恢复验收

先检查 `nvidia-smi`；确认目标卡空闲后，在新输出目录执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -B -m torch.distributed.run \
  --standalone --nproc_per_node=8 -m skillnet_cohort.gpu_preflight \
  --model /data/models/Qwen3.5-4B --embedding-router-model /data/models/Qwen3-Embedding-0.6B \
  --output /data/phase3-gpu-preflight-v1 --execute
```

4 卡改为 4 个明确 GPU ID 和 `--nproc_per_node=4`。该测试使用实际原生 actor/ref/optimizer/checkpoint 路径、4096+512 长度的合成张量，无 ALFWorld RL 或 API。每 rank 只有一个合成 microbatch，**不是完整 128 行训练批次验收**；Ray 编排和真实状态 prompt 仍由后续运行检查。

32 GiB GPU 使用可选 CPU 参数分片初始化及逐参数原生恢复，以避开两份完整 FP32 权重的峰值；先逐 rank 校验相同初始参数，再搬运分片。权重／优化器精度、FSDP 分片拓扑与全局训练参数不因此改变。已在本机8×RTX5090通过上述验收（见VALIDATION）；新服务器仍须核验。必须以所有 rank 的 `PASS` 和 `complete.json` 为通过证据，不能忽略 OOM／恢复错误。

## 6. 凭据与启动

不要把 API key 写入 JSON、脚本、GitHub、命令行参数或 shell history。可由服务器 secret manager 注入；也可交互输入到当前 shell：

```bash
read -r -s -p 'Editor API key: ' SKILLRL_PHASE3_EDITOR_API_KEY
export SKILLRL_PHASE3_EDITOR_API_KEY
```

新 embedding 模板只需要 o3 编辑器凭据，router 无 key。代码不会从 `OPENAI_API_KEY` 偷用其他账号。编辑器凭据只用于指定网关，发送内容包括公开技能和观察动作证据，不上传权重或梯度。历史 mini 后端仍保留供旧 frozen preparation 使用，但不会自动回退到它；新实验不能沿用旧 mini 的 preparation/cache。

逐个运行四分支（以下以 D 为例；分支名和输出目录要一起改）：

```bash
python -B -m phase3.run --preparation /data/phase3-assets-v1/manifest.json \
  --root /data/phase3-runs/readout_d --branch readout_d --execute
```

其余名称为 `skillrl_failure`、`readout_p`、`readout_c`。不要让四个进程同时争用同一组 GPU。可以用 tmux 保持会话；本程序不会在失败后自动重试付费请求或重开一轮 RL。

每个 5 轮 block 保存原生 policy/optimizer/dataloader/RNG checkpoint 后退出训练子进程，释放 GPU，再做 endpoint prediction、o3 编辑和配对 Seen gate。接受的库影响下一 block；最后 U150 在固定策略与最终库上评估全部 Seen/Unseen。支持集不足则弃权，不把零风险填给从未自然调用的新技能。

重要实现边界：TextWorld worker 本身没有跨进程序列化，block `i` 的环境流按 `404 + 16*i` 预先登记并重新创建；四分支相同。Actor/Adam/scheduler/RNG/dataloader 则从原生检查点恢复。它不是不间断原生 SkillRL 的逐 bit 重现，论文需披露这项统一 block 适配。

## 7. 结果、费用与中断

每分支目录：`episodes/`（逐步观测、动作、技能版本、有效性与 actor tokens），`metrics/`（RL 指标），`direction_batches/`（每窗口首批真实 advantages/tokens/chosen logprobs），`predictions/`（C/P/D、support、校准与前向成本），`events/`（输入、top-k、JSON patch、门控、接受／拒绝），`banks/`（不可覆盖版本），`router-local.sqlite3`（跨版本本地尝试/总限额）、`router-local.banks/`（各库版本的决策/完整余弦分数缓存）、`editor.sqlite3`（o3 API 尝试），`final/`（Seen/Unseen 全 game×seed）。

本地 embedding tokens、index/query latency、失败/未对账本地调用单列，API router calls/cost 为零；不是本地计算费用为零。首次索引 token 计一次，cache replay 不重复记数。轨迹兼容字段仍叫 `skill_router_api`，其中 backend 明确为 `phase3_skillrl_embedding_state`。

```bash
python -B -m phase3.report --root /data/phase3-runs/readout_d \
  --output /data/phase3-readout-d-summary-v1.json
```

汇总保留 SR、六任务 SR、episode/step/invalid-action、actor/router/editor tokens、未知 usage 数、失败／未对账 API、编辑类型和每次 mutation units、active bank size、gate ΔSR／repair／regression、候选拒绝、rollback、readout forward token/call/time。缓存命中不会重复记为 API tokens。网关未提供金额时实际账单为 `null`，不能以官方价格冒充网关实付。

没有自动的接受后 rollback；目前安全机制是提交前 gate，拒绝候选计 rejection，**不是 rollback**。`Bank.rollback_to` 提供不可变 bank-only 恢复原语；如需人工恢复，须另登记事件和版本，不能修改旧 snapshot 或退回 policy/optimizer。默认运行中 rollback=0 是真实零计数，不是遗漏指标。

如果已完成 block 和事件，重复同一 `phase3.run` 命令会核验并继续，已有 episode 和 API 成功结果可复用。若存在未完成 API reservation／失败子进程日志／不完整 checkpoint，停止并人工对账；不要删 ledger、改 budget 或擦日志来绕过。临时预测失败应保留日志并使用显式新 attempt；不要把尚未实现自动恢复的情况写成支持无损自动重启。

只有 `complete.json` 加两份 final completion 才代表该分支完成。预算耗尽、低支持弃权、OOM、溢出和运行中断都不能被包装成全量成功。

## 8. 论文边界

比较是“SkillRL-style failure-driven + shared editor”，不是原生 SkillRL 的严格复现。D、−P、+C 的方向预先固定，C 不居中、不事后翻号。四臂必须都报告；单 RL seed=404，不把窗口、games 或重复解码 seeds 当作多个独立训练 seed。初始库及支持脚本只作为文本展示，不执行其中脚本。
