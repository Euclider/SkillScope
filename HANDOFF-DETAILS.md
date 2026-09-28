# Skill-RL 交接附录：定位、复核与操作边界

配合 [HANDOFF.md](HANDOFF.md)，记录 2026-09-16 21:15 起的只读核对。除这两份交接文档外，本轮不修改文件、不启动/终止实验、不提交/回滚。历史启动命令仅用于识别入口，不是立即执行指令。

## A. 根目录、环境与索引

- 研究根 R：/home/wangyifan/skill-RL（不是 Git 仓库；报告与 idea 在这里）。
- 代码根 C：/home/wangyifan/skill-RL/SkillRL。
- 旧 Phase2 根 L：C/artifacts/phase2/qwen35-clean-s303-u30-to36-semantic-direction-fast-v1。
- U35 准备根 B：C/artifacts/phase2/qwen35-clean-s303-u35-ranking-preparation-v1。
- 新完成根 A：C/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1。

R/C/L/B/A 是本文简写，不是已定义的 shell 变量。下文命令均使用明确路径。

环境激活（只在需要执行项目命令时使用）：

~~~bash
source /home/wangyifan/miniconda3/etc/profile.d/conda.sh
conda activate skill-RL
cd /home/wangyifan/skill-RL/SkillRL
~~~

当前实查：Python 3.10.21；解释器 /home/wangyifan/miniconda3/envs/skill-RL/bin/python；torch 2.10.0、transformers 5.10.4、vllm 0.19.1、ray 2.43.0、alfworld 0.4.2、textworld 1.7.0、pandas 2.3.3、scipy 1.15.3、scikit-learn 1.7.2、pytest 9.1.1。版本由安装元数据确认，本次没有重新加载大模型或做 GPU 兼容性测试。environment.yml、environment/ 为历史环境记录，不应假设与当前逐项一致。

模型目录实存于 /home/wangyifan/model/：Qwen2.5-0.5B-Instruct、Qwen2.5-1.5B-Instruct、Qwen3.5-4B。当前主要结论来自 4B；小模型早期 smoke 不足以验证真实更新命题。ALFWorld 数据路径为 /home/wangyifan/skill-RL/data/alfworld。

优先阅读：

1. R/2026-09-14-phase2-complete-analysis.md：当前统一入口，530 行；第 1/9 节结论，第 5.2 四口径对照，第 5.3 NULL/非初始锚点，第 5.4 C 对照，第 11 节整体排序及门控。
2. R/2026-08-19-policy-update-skill-sign-flip-forecasting-design.md：指标公式与预测边界；早期“必须 harmful sign flip”已被用户放宽，不能覆盖最新约定。
3. R/2026-08-21-proposal-advantages-and-roadmap.md、R/2026-08-24-phase1-minimal-validation-repository-and-experiment-spec.md：原始背景与最小验证。
4. R/2026-09-04-qwen35-clean-all-skill-three-seed-utility-results.md：Phase1 三 seed 与 S_int。
5. R/2026-09-12-phase2-skill-utility-direction-analysis.md：旧单步结果与完整 19/7 单元分析。
6. R/2026-09-13-phase2-u35-to40-ranking-execution.md：入口、资源修订、历史故障与测试；不是最新实时状态。

## B. 实验设置与结果边界

### B1. Phase1：三个 seeds，固定里程碑

seeds 101/202/303；共同 B0 为原始 post-trained /home/wangyifan/model/Qwen3.5-4B，报告没有另一个正式 warm-up B0。各 seed optimizer 独立初始化；30 global updates，8 games×4 rollouts=32/update；固定 U10/U20/U30。训练 clean train，validation clean valid_seen 27 games，gold valid_unseen 31 games。正式入口 C/examples/grpo_trainer/run_qwen35_clean_seed_formal.sh。

model-only 路径模式：

~~~text
C/artifacts/model_only/qwen35-clean-formal-seed{101|202|303}-u30-c1-update10
C/artifacts/model_only/qwen35-clean-formal-seed{101|202|303}-u30-c2-update20
C/artifacts/model_only/qwen35-clean-formal-seed{101|202|303}-u30-c3-update30
~~~

九个目录均列目录确认，未重新加载。Phase2 起点 native checkpoint：
C/artifacts/checkpoints/qwen35-clean-formal-seed303-u30-resume12-s2-20260903/global_step_30。

Phase1：S_int 有/无时效用变化率 34.90%/8.41%，risk ratio 4.15；leave-one-seed-out AUPRC 0.192→0.274。支持是否变化，不自动支持 P/D 方向预测或其跨 seed 泛化。保存策略后来改为每 update 保存、保留最近两个完整恢复点、U10/20/30 model-only 永久留存；不按性能挑点。

### B2. 旧 Phase2 cohort L

- 固定 seed303 延续路径，从 U30 连续恢复 optimizer，完成 U31–35。目录名 to36、协议及旧 status 仍涉及 U36，但只有 U30–35 完整模型/评估；不要自动续跑旧 U36。
- 模型 L/models/u0030 至 u0035；native L/checkpoints/global_step_31 至 35。
- 旧 B0 自然首次调用 anchors 共 200；cle_003、cle_004、cle_006、gen_002 各 50；每 anchor/arm/endpoint 为 evidence seed 41011、gold seed 42011。
- ORIGINAL/PLACEBO/NULL 三臂；六端点各 1,200 suffix，共 7,200。本次索引及每端点八个完成标记相符。
- 开发 U31/U32，隔离边界 U33，实际时间外测试 U34/U35。19 个支持单元包含 7 个测试单元，不能相加。
- cle_004 U34→35 语义 ΔM=−14.29 pp，CI [−28.57,−1.79]；NULL 同次 −14.29 pp [−28.57,−3.57]；early step 1–4 的语义 ΔM=−15.38 pp [−30.77,−1.92]。共享 anchors，不是独立重复。
- 原始表 L/reports/2026-09-12-observation-audit-v1/{all_supported.csv,heldout.csv,comparisons.csv}；L/metrics/{utility_units.parquet,anchor_margins.parquet}。

### B3. 新完成 cohort A

冻结协议 A/protocol.json；配置模板 C/phase2/config/ranking_u35_to40_v1.json。协议 SHA-256：

~~~text
0ca78a05b4d58584be39ecfaa9335458b41dfd50ea6e34f80e43be797acbe47a
~~~

起点 L/checkpoints/global_step_35；同一 seed303 路径连续恢复 optimizer，U36–40 共 160 训练 rollout、70 Adam steps（611→681）；LR=1e-6、KL=0.01、decision minibatch=32、microbatch=1、prompt/response=2048/64、episode≤30步。U35→40 FP32 净参数 L2=0.517594766。不是五个独立 seed，也不是五次 gold 测试。

训练用 text-only、HF live-policy rollout、SDPA、enable_thinking=false；不依赖旧 embedded-vLLM 权重同步。硬件分片随资源变动，最终 U40 native 为 world_size=1；不声称跨硬件逐 bit 复现。

已核对目录及元数据：

- A/checkpoints/global_step_39、global_step_40：最近两个完整恢复点。
- A/models/u0035 至 u0040：model-only；U40/phase2_export.json 标记 float32、427 tensors，权重约 19.37 GB，native 含 optimizer 约 53.03 GB。不要误算为早期约 9 GB 的 BF16 里程碑。
- A/batches、old_logprobs、new_logprobs、optimizer_steps：实际 batch、advantage/mask/token 对齐、live 概率和 optimizer 证据。
- A/signals/u0036 至 u0040：单步信号；A/window_signals/u0035-u0040：端点窗口信号。
- 窗口方向固定为 U36 旧 batch 的 state/token/advantage，在 U35/U40 两端重放；不是五个 D 相加，也不假定中间方向恒定。

U35 新自然 anchors 位于 B/support/selected/clean，每 Skill 50，共 200。29–31 games/Skill，gen_002 step2–29，cle_003 step3–11，cle_004 step1，cle_006 step0。evidence seeds 51011/51021，gold seeds 52011/52021/52031/52041，temperature=.4、top_p=1、history_length=2。U35 从 B 导入，U40 新跑；每端点 200×6×3=3,600 suffix，八分片各 450，本次均核对通过。

只有 U35/U40 gold，U36–39 无中间 gold。四个评估 Skill 中 cle_006 训练方向支持不足，主池只有三个。新语义 ΔM：gen_002 −3.02 pp、cle_003 +15.73 pp、cle_004 −6.45 pp；只有 cle_003 CI 完全高于 0，新下降 CI 均含或触及 0。

A/window_signals/u0035-u0040/prediction.json 于 2026-09-13T13:46:46.281540+00:00 锁定，target_gold_read=false。enable_fitted_models=false；主目标直接排序，不要求拟合回归。without post-update gold rollout 不等于没有旧训练轨迹或模型前向成本。

结果索引：

- A/window_metrics/raw_features_and_semantic_utility.csv：原始 P（不是 −P）、全部 phase、缺失标签明确保留。
- A/window_metrics/locked_scores_and_gold.csv、ranking_metrics.csv、ranking_support.json：固定风险方向与共同支持池。
- A/window_metrics/utility_units.parquet、utility_games.parquet、anchor_margins.parquet：效用与配对底层统计。
- A/evaluations/u0035 和 u0040：索引、完成标记、trajectories/ 下逐轨迹 JSON。
- A/reports/window-results.md：自动窗口结果，不替代根目录人工综合报告。

旧新 cohort anchors/repeats 不同，不把两批 U35 接成未控制的效用曲线。7,200+7,200 是归档 suffix 数，不是独立预测样本数。

### B4. 指标、干预与已否定解释

Bank 为 C/memory_data/alfworld/claude_style_skills.json：12 general+32 specific，clean 12+6 候选。FrozenStepSkillRouter 按可见 task/observation/admissible actions/history 计算描述词与阶段分数，有 task-specific prior 和文档顺序 tie-break；非 LLM 学习路由。曾讨论更强 phase eligibility gate/扩 bank，但不是现有协议，不能把建议当实现。

固定首次调用前 prefix；候选 ID/描述不变，目标 payload 按三臂替换且后续持续生效。每 game 内平均 anchors/repeats，再 games 等权；bootstrap 10,000 次配对 game/continuation。M 为 success 差，单位 pp；终局 reward=10，return 差需乘 10。

全词表 248,320，W=I。u_O=logπ_new−logπ_old，δ_int=u_O−u_control，d=A(e_a−π_old)，P=<d,δ>/(||d||+ε)。主 D 为全部 token 上 gated[-P]₊ 的均值，τ_C=0、τ_δ=10⁻⁸、ε=10⁻¹²；未过 gate/零 advantage 仍进分母。ungated 只移除 C/norm gate，不移除有效性/负部截断。mean(P)>0 与 D>0 可同时成立。

支持门槛：20 非零 advantage decisions、4 非零支持 games、8 非零支持 trajectories。零支持不是零风险。initial=step0、early=1–4、middle=5–14、late≥15；phase 与 all 不能重复当独立窗口。

固定风险方向：+D、+ungated D、−P、+norm、+KL/JS、+control-update norm；新预锁定 old-margin 为低 margin 优先，旧事后 audit 曾用 +old-margin，比较须标方向。+C 比较属于事后审计，不伪装成新预注册方法。

任意下降 AP 保留全部候选；条件方向 AUROC 仅下降对上升（旧测试 2 对 2）；下降对其余 AUROC 是 2 对 5。两者 norm/KL 数值不同，禁止混用。0/5 pp 标签阈值不是 D 门控。候选数/事件率不同时，不直接跨范围比较 AP 或原始平均名次。

被否定或暂缓的路线及原因：

- 不要求精准 ΔM 回归、每次 Top-1 正确或严格 harmful flip：实际目标是排序定位/有限编辑预算。
- 不看效用后挑 checkpoint、翻转分数或每窗口选赢家，否则 selection bias。
- 全局参数 norm/train success 在同 update 对所有 Skill 相同，不能当窗口内 Skill 读出。
- 旧 pooled D 优势不等于全由混杂产生，也不证明窗口内增量；新不理想不抹去旧正面证据。
- D 不是已确认失效机制；C 与 D 共享 reward direction，不能当 D 独立增量证据。
- JVP/线性化误差、shuffled-reward/random-direction/independent-batch 更新、跨 seed Phase2、Phase3 编辑及等预算成本曲线均未完成。
- 不换 bank/新增任务后与旧 frozen 结果直接混池；新方案另注册。

## C. 代码定位与 Git 状态

branch=phase1-prep；HEAD=8e66726（Fix _validate skill-update collecting no failed trajectories (#57)）。19 tracked 修改，diff --stat 为 560 insertions/70 deletions；diff --cached --stat 为空。未跟踪是 17 个顶层条目，不是只有 17 个文件。逐文件作者不能确认，全部按用户既有工作保护。

| 路径（相对 C） | 作用 |
|---|---|
| agent_system/memory/step_skill_router.py | state flags、描述评分、逐步路由日志 |
| agent_system/memory/skills_only_memory.py | 冻结 bank 候选池 |
| phase1/first_invocation.py、eval_first_invocation_utility.py | payload intervention、prefix replay、自由 suffix、TransformersPolicy |
| phase2/capture.py；verl/workers/actor/dp_actor.py | 训练 batch/advantage/token/概率采集 |
| phase2/direction.py、measure.py、aggregate.py | 全词表投影、D、支持及校准 |
| phase2/protocol.py、window_forecast.py | 协议/预测锁定、固定风险方向 |
| phase2/evaluate.py、utilities.py、ranking.py | 三臂评估、统计、排名及并列/零事件规则 |
| phase2/run_extended.py、launch_training.py、elastic_training.py | 新 cohort、资源适配、真实更新 |
| phase2/evaluation_resources.py、staged_storage.py | 评估并发/OOM 重试、完整 checkpoint 轮换 |
| phase2/finalize_evaluation.py、complete_report.py | 完成核验/自动初稿；重跑会写文件，可能覆盖人工补全 |
| tests/phase2/、tests/phase1/ | CPU 回归；本次未运行 |

tracked 修改完整清单：

~~~text
agent_system/environments/env_manager.py
agent_system/environments/env_package/alfworld/envs.py
agent_system/environments/env_package/alfworld/projection.py
agent_system/environments/prompts/alfworld.py
agent_system/memory/__init__.py
agent_system/memory/skills_only_memory.py
agent_system/multi_turn_rollout/rollout_loop.py
requirements.txt
scripts/model_merger.py
setup.py
verl/third_party/vllm/__init__.py
verl/trainer/config/ppo_trainer.yaml
verl/trainer/ppo/ray_trainer.py
verl/utils/checkpoint/fsdp_checkpoint_manager.py
verl/utils/fsdp_utils.py
verl/utils/vllm_utils.py
verl/workers/actor/dp_actor.py
verl/workers/fsdp_workers.py
verl/workers/rollout/hf_rollout.py
~~~

未跟踪条目：

~~~text
.gitignore
agent_system/memory/step_skill_router.py
artifacts/
environment.yml
environment/
examples/grpo_trainer/run_alfworld_phase1.sh
examples/grpo_trainer/run_alfworld_phase1_qwen35.sh
examples/grpo_trainer/run_alfworld_phase1_step_router.sh
examples/grpo_trainer/run_qwen35_clean_seed_formal.sh
outputs/
phase1/
phase2/
scripts/download_phase1_model.sh
scripts/run_phase1_smoke.sh
scripts/run_qwen35_b0_clean_monitor.sh
tests/phase1/
tests/phase2/
~~~

禁止 git clean/reset/checkout、自动提交或把 artifacts/权重批量 add。两份 HANDOFF 位于研究根，不在 Git 代码仓库内。本次未发现适用 AGENTS.md；新会话仍需检查是否新增。

## D. 报告版本与验证

### D1. 最新报告不是首次生成版本

版本链（历史证明不覆盖）：

| 版本 | 清单（相对 A） | 报告 SHA-256 |
|---|---|---|
| 初次生成 | full_report_completion.json | d2ad5fdfa197014c7ead2a48f70be67c070df54dca036d6a1273c99beff70f97 |
| 排名讨论 v1 | reports/2026-09-14-ranking-interpretation-v1/revision.json | a1c51eb37b15c2f1df8bce5df044e7f302ddbbd3905962e2e39667667a3f7f50 |
| 当前综合 v2 | reports/2026-09-14-phase2-synthesis-v2/revision.json | a697119808ba394cf608a910ae3142257721d802d9e8baa072734fe6c051292e |

v1：追加平均下降名次、AP、单调性、gated/ungated；overall_ranking_audit.csv 42 行、within_window_ranking_audit.csv 84 行，记录重算 126 行/核对 28 行 Markdown。

v2：磁盘实存且本次核验通过（清单创建时间为 2026-09-14 09:03:42 UTC）；补第 5.2–5.4 节旧四口径、NULL/中途锚点/C 对照。清单记录 21 行新 Markdown、15 个读出比较重算。其补全过程不在当前可见聊天逐步记录内，本次以文件/清单为证据，不猜测额外执行过程。

本次开始使用 v1 验现文触发 AssertionError，随后找到 v2 链并验证当前报告及 11 个 source_files 全部匹配。不是实验损坏；严禁“修复”为旧报告或覆盖历史哈希。

### D2. 历史测试与本次检查分开

历史文件证据：

- C/artifacts/code_checks/phase2-expanded-v2-20260912/verification.json：python -m pytest tests/phase1 tests/phase2 -q，189 passed / 8.35 s，22 新测试，未启动 rollout。
- R/2026-09-13-phase2-u35-to40-ranking-execution.md：后续 208、224、226、227、最终 233 项 CPU 回归通过，含端到端报告生成测试；是历史记录，不是本次重跑。
- A/evaluation_completion.json：complete_and_verified，U35/U40 各 3,600；新端点逐轨迹完整性与 SHA-256、先锁预测后生成 gold 的核验。
- A/completion_watch_status.json：complete_and_verified；A/status.json：extended_cohort_complete、completed=true、global_update=40。

本次实际只读命令/检查：

~~~bash
git -C /home/wangyifan/skill-RL/SkillRL status --short --branch
git -C /home/wangyifan/skill-RL/SkillRL log -1 --format='%h %s'
git -C /home/wangyifan/skill-RL/SkillRL diff --stat
git -C /home/wangyifan/skill-RL/SkillRL diff --cached --stat
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_gpu_memory --format=csv,noheader
df -h /home/wangyifan/skill-RL /home/wangyifan/model /dev/shm
~~~

另用 python -B 读取安装版本、校验 v2 全部哈希、计数六旧/两新端点、检查进程。未运行 pytest、GPU forward/backward、checkpoint restore、全部 bootstrap 或逐条重新哈希 14,400 轨迹。大权重仅查目录/导出元数据，本次未重验全部字节，不能冒称重现实验。

安全复核脚本（只读，不调用会写报告的 finalize/build）：

~~~bash
/home/wangyifan/miniconda3/envs/skill-RL/bin/python -B - <<'PY'
from pathlib import Path
import hashlib, json
root = Path('/home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1')
revision = json.loads((root/'reports/2026-09-14-phase2-synthesis-v2/revision.json').read_text())
for item in [revision['report'], *revision['source_files']]:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256'], item['path']
for update in (35, 40):
    folder = root/f'evaluations/u{update:04d}'
    counts = [sum(bool(line.strip()) for line in (folder/f'shard-{i}.jsonl').open()) for i in range(8)]
    assert counts == [450]*8, (update, counts)
    assert all((folder/f'shard-{i}-complete.json').exists() for i in range(8))
assert json.loads((root/'evaluation_completion.json').read_text())['status'] == 'complete_and_verified'
print('Latest report/source hashes and 2 x 3600 endpoint indexes verified.')
PY
~~~

### D3. 历史启动入口（当前不要执行）

U35→40 编排入口在执行记录中已确认使用过：

~~~bash
# 仅历史入口说明；当前 cohort 已完成，不要重启。
python -m phase2.run_extended --root /home/wangyifan/skill-RL/SkillRL/artifacts/phase2/qwen35-clean-s303-u35-to40-ranking-v1 --detach
~~~

逐次 GPU/恢复参数在 A/launch_manifests/、allocations/、evaluation_allocation_history.jsonl；U40 不止一次 attempt，不能从旧命令猜最终硬件。launcher 调用 run_alfworld_phase1_qwen35.sh hf，完整参数由冻结协议/当次资源清单拼接。不要复制旧 resume path 直接新开训练。

phase2.finalize_evaluation --root ... --watch --detach、phase2.complete_report --root ... 均是写操作；历史流程使用过，本轮没有运行。它们可能覆盖现有追加报告，不能当作“查看状态”。

新 cohort 用 phase2/launch_training.py、run_extended.py；旧 run_fast.py 服务旧协议。expanded v2 template 只是模板，不等于获准执行。

## E. 当前进程、资源与历史故障

核对时未发现匹配 phase2.run_*、phase2.measure/evaluate/finalize 或 formal-seed launcher 的进程；历史主 PID 4063983、完成检查 PID 4088408 均不在 /proc。A 完成时间为 2026-09-14T00:44:18/24 UTC（北京时间 08:44）。status 的 pipeline_pid 是历史值，不代表仍活着。

GPU 快照：8×A800 80GB；0/1/5 各约 13 MiB、0% utilization；2/3/4 各约 74–75 GB，6 约 62 GB，7 约 70 GB。占卡 PID 包括 2335364、2328300、2335377、4115540、3018723、564537、566156；读取部分 /proc 信息遇 PermissionError，归属/用途未确认，不得认领为空闲或终止。快照随时会变。

磁盘 / 使用率约 97%、可用 264 GiB；/dev/shm 252 GiB，当时仅用 84 KiB。不要从历史路径推断旧内存盘 checkpoint 副本仍存在；新任务需重查资源。

检查方式：

~~~bash
ps -p 4063983,4088408 -o pid,ppid,etime,comm
pgrep -af 'phase2[.](run_extended|run_fast|evaluate|measure|finalize_evaluation)'
nvidia-smi
~~~

pgrep 可匹配检查命令自身，须结合 PID/时间；本次 Python 检查过滤了自身。无匹配仅表示未找到这些项目入口，不代表整台服务器无任务。

日志：A/supervisor.log、A/logs/、A/logs/completion-watch.log；资源重试 A/evaluation_attempts/，训练 attempts/recovery 审计另存。

已恢复的历史问题：

- completion-watch.log 的 “Insufficient disk for remaining evaluation plus unchanged 200GiB reserve” / “Pipeline stopped before completion” 来自评估 200+20 GiB 预算线，不是实际磁盘满；用户授权豁免后已经完成。不能当当前失败。
- A/evaluation_disk_waiver_20260914.json 绑定协议/原资源文件，仅豁免该轮评估预算，保留 256 MiB 近写满保护；新训练 250 GiB 准入规则未取消。
- 共享 GPU/OOM 曾导致重试/弹性分片。evaluation_resources.py 仅自动重试只读 measurement/evaluation 的 GPU OOM，其他对齐/协议错误停止；完整 trajectory/committed shard 复用。
- A/evaluation_resource_amendment.json：measurement 最多 3 worker/GPU、23 GiB/worker；evaluation 最多 4 worker/GPU、14 GiB/worker；留 8 GiB headroom。只使用空闲或完全由本 scheduler 工作者占用的卡，不用 dummy tensors 占显存。

L/status.json 仍是 stopped_on_error/global_update=36（2026-09-11），无旧 U36 gold；独立 A cohort 后来完成 U36–40。两套协议/支持集不能混同，更不能为清除旧红色状态而重跑或改旧记录。

## F. 接手顺序与缺失信息

1. 读当前报告与附录，先做 D2 只读校验，不运行 historical launch/finalize。哈希若又变，先找新 revision，不回滚。
2. 保留旧 D/AP/排名优势、无门控 U31→32 案例及新失败；区分事实、假设、未验证跨 seed 增量。需要时仅用已有 CSV 作明确标记的描述性审计。
3. 与用户确认最小新实验：独立窗口/seed、自然支持 Skill、冻结分数/门控/预算、C/norm/KL/old-margin 对照、支持/不确定性标准。迁移请求不授权新开训练，同窗口加 rollout 不等于增加独立样本。
4. 获授权才建新 root/协议，检查 native 恢复/分片、FP32 证据、覆盖、GPU/磁盘；先锁 score 再 gold。允许单窗口失败，检验总体收益；随后才做同预算编辑/成本曲线。

缺失/未确认：下一轮具体 seed/窗口/预算；Phase2 跨 seed 稳定性；reward/parameter 负对照；JVP/线性化误差；当前占卡进程归属；修改代码逐文件作者；大模型权重本次字节校验。不得补猜。

## G. 2026-09-18 新 embedding cohort 的追加入口

前述 A–F 为历史核验记录，不覆盖之后用户确认及新的启动授权。当前 seed=404、5 更新窗口、八卡 policy / CPU 0.6B 冻结 router、完整 SkillNet-37 的新 cohort 已于 05:21:02 UTC 启动；Phase3 迁移先行发布至 GitHub commit `674dd36a54a83c5061af43f0f80b2ee3da924468`。详见 [完整发布/验证/启动审计](2026-09-18-embedding-phase3-publication-and-phase12-launch.md)，包含 preparation/permit/upload/preflight 哈希、实际命令、PID、停止边界及证据范围。

新 root：`/mnt/workspace/users/wangyifan/skill-RL/SkillRL/artifacts/phase12/skillnet37-qwen35-embed06-s404-8gpu-v1`。本次 supervisor PID=903045；05:25 UTC 仍处首个 rollout，更新完成数为 0，不是全流程完成。可只读检查该 root 的 supervisor/stage 日志、router.sqlite3 账本、metrics、completed_windows、stopped/complete 标记。不能因日志出现 Training Progress 0/150 就自动重启；先核对存活进程及路由账本是否增长。

历史 synthesis-v2、mini 停止记录、旧组件 RNG 失败记录、旧报告与测试哈希均保留。不重跑旧实验或历史报告生成器，不把新的工程预检 PASS 当算法有效性或完整 batch 保证。

## H. 2026-09-18 06:50 UTC：预算版重配置与失败停点

第 G 节“运行中”是 05:25 UTC 历史快照。用户已批准停止旧任务，确认 5 更新单窗口、完整 seen/unseen 性能、unseen 至多 12 技能×12 anchors×1 evidence+2 gold seeds，并将目标调整为 15–30h。范围、参数差异、工作量、测试命令及科学边界见 [预算版 v2 记录](SkillRL/docs/experiments/phase12-daybudget-v2/README.md)。

关键证据（相对研究根 R）：

| 对象 | 路径 / SHA-256 |
|---|---|
| 旧停止回执 | `SkillRL/artifacts/phase12/skillnet37-qwen35-embed06-s404-8gpu-v1/user-stop-receipt-20260918.json`；完成更新=0，完整 rollout batch=0 |
| 旧 SQLite 保持不变 | `9214ce320e1ba1b29752a9c771fa3f1d69c48d13a0efb3cbb60190e5b4fde7ee` |
| 新 router 短标定 | `daybudget-router-calibration-20260918-v1/report.json`；`c627c2916c76442436285c4df4333f4769c002ea6f081cdd55ac46279b26eeac` |
| v1 GPU 失败日志 | `daybudget-gpu-preflight-20260918-v1/run.log`；`5b9de81403054989894b6a3d688eeec4767f26d01bf36aec31cc5437ff37cc78` |
| v2 GPU 失败日志 | `daybudget-gpu-preflight-20260918-v2/run.log`；`cc33eb32af1f8cf7c65df824b98a350527bbf47288c4ba2605dfc3164f3d3a67` |
| v2 preparation | `SkillRL/docs/experiments/phase12-daybudget-v2/preparation-s404-8gpu/manifest.json`；`174839baeeab0b5270e699daaf4072107a682c94447d9dce895ba4aa5d7b767b` |
| v2 最终离线测试 | `SkillRL/docs/experiments/phase12-daybudget-v2/offline-tests-v2.xml`；435 PASS / 81.49s；`c787772d9aed083ae364d74bfeb8a73d397a27cf0806b18682f6c21a707a8835` |

v1 尝试：每卡 2 条 4096-token prompt，实际生成 cap=8（输出 padding 512），第一次 native update/restore 后生成 prefill OOM。用户知情后另行批准 v2 offload 尝试，执行计划在 `daybudget-gpu-preflight-20260918-v2/execution-plan.md`，实际 launch 和脚本/preparation SHA 见 `launch.json`。v2 06:43:27 启动，06:46:17 失败，总日志历时约 173.89s；完整 native restore 和 cap512 生成通过，第二次合成 backward OOM，没有 complete/rank PASS。v1/v2 都不是 ALFWorld RL 训练，不能计入完成更新。

只读诊断：`verl/workers/fsdp_workers.py` 在 `update_actor` 入口将 optimizer states 回载 GPU；`dp_actor.py:485 loss.backward()` 在第二次 synthetic update 申请 7.83 GiB 失败。HF actor 既有兼容路径关闭自动层级 wrap，而 ref 按层 wrap；未在本轮更改该行为。建议评估延迟 states 回载至 optimizer.step 前，仍需用户复核和新版本预检，不把推断当验证。

06:46:56 UTC 所有八卡均为 2 MiB、0% utilization；当前可用磁盘约 730 GiB，两次各约 50 GiB 的 synthetic checkpoints 保留。760 GiB run cap、100 GiB 空闲线和已授权的固定窗口临时概率回收不变，但**并未证明**完整 FP32 old/new 记录可以放下。每 response token 的 old+new vocabulary 基础量=1,986,560 bytes（100k tokens≈185 GiB）。

当前准入为 `NOT_ADMITTED`，无新正式 execution permit，无正在运行的新预算版 RL 或 Phase3，无本轮 GitHub push。后续先看 HANDOFF 第 10 节；不能从相邻 v1/v2 名称推断成功，不能复用旧 150 更新 permit，不自动重试、不删除失败证据。

## I. 2026-09-18 07:18 UTC：vLLM 接口迁移和参数复核停点

第H节为旧HF失败停点。用户随后要求切换vLLM并在修改完成后先确认RL参数。当前代码已接入vLLM V1，但无新实卡推理/工程预检/正式RL。完整参数、环境、命令、实现限制见 [新记录](SkillRL/docs/experiments/phase12-vllm-v1/README.md)。不把“安装/导入/配置可构造”当作“可开跑”。

新证据（研究根R下）：

| 对象 | 路径 / SHA-256 |
|---|---|
| preparation | `SkillRL/docs/experiments/phase12-vllm-v1/preparation-s404-8gpu/manifest.json`；`e9f895ef159cd2f47db99d06764439193d7cb2ab7b68028e40db02b6ac128078` |
| 实际start-segment参数（仅复核） | `SkillRL/docs/experiments/phase12-vllm-v1/resolved-training-review.json`；`990cf6eba5948f07e5f6def2b9a1c4b2bcd138b31b1ac1cb823839634b5bc3b7` |
| 445项离线回归 | `SkillRL/docs/experiments/phase12-vllm-v1/offline-tests-final.xml`；`d6f98e9b03bf9c5d424954874caa361a4d4993ebf1ea3ceaa5dcb40225c3f660` |
| 新环境全依赖锁 | `SkillRL/requirements-vllm-phase12.lock`；`c7c5d178dca623df5b8642ab785d0b91fd2f152ef5d6a95388039f4f4377395b` |

新增主要接口：`skillnet_cohort/inference.py`（冻结profile/factory）、`vllm_backend.py`、`vllm_qwen35.py`、`verl/workers/rollout/vllm_v1.py`、`verl/workers/sharding_manager/fsdp_vllm_v1.py`。训练和评估工厂显式按新profile选路；仅旧manifest保持HF历史行为。fsdp_workers对新后端接入sleep/wake、权重版本失效和原生计算前休眠；保留旧vLLM分支的版本guard，不通过删除guard冒充兼容。

参数：5迭代/16×8/8卡/seed404/lr1e-6/FSDP1（非FSDP2）；PPO全局128 state-action rows、每卡micro1，满mini-batch累积16；KL.01/entropy.001；完整37库+冻结0.6B state top1 CPU批路由。vLLM engine按inference profile并发16覆盖旧HF microbatch2，实际config已同步这一覆盖以及max_num_seqs16/max_model_len4608/gpu_memory_utilization.45，不更改统计采样量。

测试不涉及真实vLLM权重：172项相关回归、10项初始adapter单测、445项全回归、dummy-config兼容修复后最新10项adapter单测（包含在445测试范围中）。静态ModelConfig成功标识native text/hybrid，没有构建LLM engine。新环境依赖269项check通过，原环境未动。最后八卡均2MiB、0%，磁盘约711GiB可用；以后需重新核查。

下一步：将实际参数交用户确认。新预检计划命令在README，输出必须是尚不存在的 `R/vllm-gpu-preflight-20260918-v1`，不得覆盖HF失败目录；运行前另存launch/source哈希和日志，失败停止不自动重试。该合成预检还不覆盖128-row完整batch、Ray调度、真实ALFWorld、exact概率记录容量和全流程时长，这些准入仍未完成。没有本次GitHub更新。

## J. 2026-09-18 08:07 UTC：已获5轮测时确认，首个vLLM预检失败封存

第I节的“等待参数确认”是历史停点。用户已批准按当前5迭代配置先跑以估计50/100迭代，并在运行中取消本次30min工程预检上限。正式30h、存储保护、每5迭代窗口、16×8采样量不变；没有扩大正式horizon。

新root=`R/vllm-gpu-preflight-20260918-v1`，完整 [结果与CPU诊断命令](vllm-gpu-preflight-20260918-v1/result.md)、[launch](vllm-gpu-preflight-20260918-v1/launch-request.json)、[failure audit](vllm-gpu-preflight-20260918-v1/failure-audit.json)。进程07:58:41启动、08:02:33最终报错，约232s只代表初始化失败历时。stage记录、rank PASS、optimizer updates和真实rollout均为0。此次未生成synthetic checkpoint，native权重同步、连续更新、全词表归档、Ray与ALFWorld吞吐仍未验收。

根因：`gpu_preflight.py:86 ref.init_model` → `fsdp_workers.py:258 from_pretrained` → `modeling_qwen3_5.py:1143 config.vocab_size`。vLLM `transformers_utils/config.py:228` 注册自身Qwen3_5Config；Transformers `models/auto/auto_factory.py:394` 的精确类比较从true变false，无法自动提取text_config。已在独立CPU/offline进程只加载配置复现，未加载模型或重跑GPU：前配置类来自transformers，后来自vllm；模型期望类始终来自transformers。诊断值在 `config-dispatch-diagnostic.json`；修复尚未实施。

退出清理：torchrun报告8个worker退出码均1及ChildFailedError；僵尸torchrun可见proc stat字段52为0，记录差异、不当作PASS。用户取消timer后仅SIGSTOP wrapper PID958760，未暂停worker。全部worker自然退出后，仅SIGKILL该暂停wrapper清理；外层session27698返回137，不能当作实验worker exit code。此后shell/timeout/torchrun及worker均不再存在，八卡2MiB/0%。没有自动重试。

本次哈希：run.log `293b0f2bf576784630f28bff8317b3030e93e0758c4dedf8dd4d299fdbe4cfd9`（103562 bytes）；gpu_preflight.py `1c6157f7aa7604f64ee82340241516fe58bce8f8990fc9e5e1ce969a2d7169f9`；vllm_backend.py `165da93f0a1a2b5b59af1095bfc61fcb1fc0f5457f9ad371c75fa6fcbf14fd97`；fsdp_workers.py `f407e9fd23c90cd091c776e6fea8f7c1cb9e92efd97034d2cc28daf4b016ac4d`。preparation和migration-audit未修改，旧HF v1/v2日志和旧embedding router.sqlite3再次哈希一致；未重验全部历史模型/轨迹字节。

08:07:18磁盘空闲762860011520 bytes（约710.5GiB）。[外推计划](SkillRL/docs/experiments/phase12-vllm-v1/timing-projection-plan.md)分别列出RL-only与完整Phase1–2所需实测量；按每个5迭代窗口都执行当前协议时，5/50/100分别有640/6400/12800训练轨迹、548/3014/5754全量性能episode、至多2592/25920/51840条O/P/N续跑。纯工作量不是ETA。50/100还涉及检查点容量，无新的保留/删除授权或替代preparation。

当前仍NOT_ADMITTED；按experiment-agent失败不自动重试规则，等用户确认兼容性修复和新目录预检，再完成已批准5轮的其余准入。原始记录不覆盖，代码与RL配置不改，没有本轮付费请求或Git操作。

## K. 2026-09-18 09:00 UTC：v2恢复失败、配置/恢复修复及453项离线回归

第J节之后用户确认“修复开跑吧”。先修 `verl/utils/model.py::normalize_transformers_config`，仅将vLLM包的dense Qwen3.5配置转为Transformers原生类，保留attention和配置值、不修改vLLM registry；worker加载actor/ref时调用。真实vLLM parser顺序的CPU/meta测试包含旧错误复现及修复后构造成功。随后新v2预检跨过配置错误，但在旧checkpoint恢复函数拒绝多FlatParameter处失败。

v2根 `R/vllm-gpu-preflight-20260918-v2`：launch argv/env/source SHA、PID和实际returncode见 `launch.json`、`process-start.json`、`process-exit.json`；08:51:42.599601→08:54:06.867869 UTC，共144.27s；torchrun PID966454及8worker均自然退出，returncode1。没有timer wrapper。四个stage×八rank均完成：模型初始化82.46–85.44s、CPU router共存6.51–6.60s、actor/ref前向约4.154s、首个合成更新7.58–7.75s。保存checkpoint后、`gpu_preflight.py:132`恢复时报 `Bounded native restore requires one root FlatParameter`；还未生成policy token或做第二次更新。

`v2/failure-audit.json`归档全部stage/文件大小，checkpoint文件约53.03GB，未删除。原失败helper SHA=`4c922d9fe0f25298ac5e540c98751540d73cde47714b555cfc7ac077d6d92ce3`。错误从 `fsdp_checkpoint_manager.py:126`调用 `native_restore.py:18`；旧bounded restore假设全模型单根，而迁移vLLM时恢复按decoder layer包裹。不能用离线mock通过推断该实卡路径兼容。

后续修复仅在code/unit层继续：`native_restore.py`先校验live rank/world与FP32；单根原路径保留，按层使用原生strict SHARDED_STATE_DICT loader，保留未知/缺失keys错误及DTensor拓扑验证。checkpoint manager的Adam/scheduler/RNG逻辑未改。4个真正CPU/Gloo双rank测试覆盖root-flat/layer-wrapped × ShardedTensor/DTensor，逐值比对参数及Adam状态并继续第二次step；GPU显存和完整RNG恢复仍待新预检。

最终代码SHA与测试SHA在 `R/vllm-native-restore-repair-20260918-v1/repair-audit.json`；完整 **453 PASS /92.16s** 的XML及独立4项恢复测试（16.13s，与453重叠）在同目录。先前449项（78.53s，v2目录）对应第二处恢复修复之前，不能描述成修改后的完整回归。最终helper SHA=`b7a03d623fe77be51983ebfa66a97c99a7a5773441a5df0278b5009055edda69`；utils/model=`2b3aa5597498f4f31f59cdf0d7f412541915e782f6faae32ae3395b497fffc09`；fsdp_workers=`778463a489ea287e461829ccb2ec8537226ab50d01f363d60851dc18ca7e6469`。

冻结preparation SHA仍`e9f895ef159cd2f47db99d06764439193d7cb2ab7b68028e40db02b6ac128078`，科学参数/5迭代/16×8及router未改。formal root尚未创建，无新permit；当前NOT_ADMITTED。已异步询问下一次单独GPU预检授权，写入本节时尚未获得；[v3建议命令](vllm-native-restore-repair-20260918-v1/next-preflight.md)未执行。不能用这次synthetic step声称已完成RL、完整batch/捕获、sharded ALFWorld或50/100轮估时。

## L. 2026-09-18 09:40 UTC：v3恢复通过、生成接口失败及v4单独授权

v3=`R/vllm-gpu-preflight-20260918-v3`，用户批准后执行，无timer。launch/process-exit记载09:31:39.991792→09:34:12.970806 UTC，exit1；run.log SHA=`d31e0e9cf99afeba45b11cb2229006d0871082a51e38b8b35d374e536daf74f0`。5stage×8rank均完成，其中native_checkpoint_restore21.38–23.90s，包含保存、参数/Adam/RNG逐值恢复验证。checkpoint大小53,034,114,663bytes只做metadata统计，未重新哈希大张量。

失败 `gpu_preflight.py:152`→`vllm_v1.py:68`→`gpu_model_runner.py:1581::_init_mrope_positions`，assert supports_mrope(model)。桥接进入后第一次真实generate尚未完成；不能声称完整weight-sync数值等价或第二次更新通过。所有已知启动PID均退出，八卡2MiB/0%。PyTorch reserved_bytes含休眠vLLM记账，不当作物理显存峰值。原科学preparation/profile与先前配置/restore修复未改。

仅补 `vllm_qwen35.py` 的SupportsMRoPE和纯文本position方法，离线测试调用当前安装vLLM原生wrapper的无媒体helper比较，不构建engine/加载权重。完整结果另存 `R/vllm-mrope-repair-20260918-v1`。用户另行批准修复后新v4八卡预检及原5轮；此段写入时尚未启动v4，后续以v4 launch/process-exit/complete记录为准。无formal permit，不绕过剩余准入。

## M. 2026-09-18 11:10 UTC：独立起点窗口与共同预算队列

以HANDOFF第15节和 `C/docs/experiments/phase12-independent-v4/README.md` 为当前入口。
本轮使用academic-research-suite / experiment-agent；全部新科学结果仍UNVERIFIED，不把工程验收当统计复现。

已通过且不应重跑的工程证据（R下）：

| 证据 | 实际范围 |
|---|---|
| `vllm-gpu-preflight-20260918-v4/verification-audit.json` | 8rank，09:42:10→09:48:00，349.31s，双synthetic optimizer step、native恢复/vLLM连续生成；约53.03GB checkpoint保留 |
| `vllm-throughput-preflight-20260918-v2/timing-report.json` | 8train+8unseen共16 ALFWorld episodes/614steps，157.03s并行wall；训练mean571.875 output tokens/episode，max prompt2255、max response20 |
| `vllm-capture-preflight-20260918-v1/verification-audit.json` | global128状态行，micro1×16/rank，4096+512 padded但仅16 valid tokens；256 old/new文件全部SHA/FP32/chosen对齐，约4.07GB |
| `vllm-router-calibration-20260918-v1/result.json` | 32自然状态，4×batch8，32/32选技一致，.614476s/新状态（无cache）；不是准确率或覆盖率结论 |
| `lossless-capture-calibration-20260918-v1/complete.json` | 16合成+24自然重分词teacher-force行，全部无损逐bit核验；自然压缩比.412–.449；非原trainer token/live parity证明 |

新start-only采集将原677GiB“五轮×old/new”粗估降为约67.7GiB的一份raw OLD，再无损压缩；
这来自采集依赖修正，不是删vocab、降低精度、减rollout或改变C/P/D。
第一次实际训练batch在u0001，U0/U5对同state/token打分；其余更新batch本身仍生成用于GRPO但不完整落盘。
U0/U5模型、batch、起点轨迹、压缩OLD行、compact信号、所有paired gold、optimizer与router审计保留。
新`phase2.measure`对压缩行用`skillnet_cohort.lossless_tensor.load`，旧torch文件仍兼容。

本轮新增/修改的范围：capture_scope/capture_storage/lossless_tensor、phase2 capture/measure、ray_trainer和rollout_loop的可选采集钩子；
day_budget/training/support/runtime/run的协议绑定；新independent_preparation/seed_queue/reports及测试；
vLLM桥接seed=登记seed+rank（原来写死404+rank）；未更改GRPO数学、bank内容/router选择数或原生恢复机制。
固定+C_upd风险方向在新spec和预测锁定前登记，旧ranking模板/历史报告不改。
全部源码哈希在新cohort.json和repair-audit.json，队列各seed启动前核验，运行期间不要修改绑定源码。

新资产根 `C/docs/experiments/phase12-independent-v4/prepared-v2`，每seed的manifest/spec、admission、static/resolved配置齐全。
新cohort plan SHA=`059378537a4d1e2d48410322194eff7ddfc7c2a8711a267f378905b6e04cc470`。
首次离线`prepared/`在源码路径哈希清单处失败，留INCOMPLETE.md；没有环境、RL或API运行，不覆盖该目录。
最终666项CPU离线回归在 `R/endpoint-capture-repair-20260918-v1/offline-full-02.xml`（100.40s）；
之前同范围offline-full-01为101.32s，35项targeted与其重叠，不相加。
原router SQLite/旧preparation/v4/capture日志保全核验见同目录preservation-audit.json。

正式启动命令已执行一次（不要重启）：

```bash
cd /mnt/workspace/users/wangyifan/skill-RL/SkillRL
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -u -B -m skillnet_cohort.seed_queue \
  --plan docs/experiments/phase12-independent-v4/prepared-v2/cohort.json --execute --detach
```

Root=`C/artifacts/phase12/skillnet37-independent-s404-505-606-v4`；queue PID1016912、seed404 PID1016953；
11:09:54 UTC开始，absolute deadline=1789837794.7222931（2026-09-19 17:09:54 UTC）。
11:10快照仅U0已导出、第一段RL初始化，更新0；505/606尚未启动。
生成模型使用vLLM；原生FSDP训练和teacher-forced readout不是HF generation回退。
三seed每个640训练轨迹，最多548performance+134source+2592utility及Seen monitor。
每seed local router202100 calls cap，external API0；共同磁盘760GiB cap /100GiBfree /80GiBreserve。
单seed条件保守外推94250秒（26.18h）与176.86GiB的保守容量写入admission；包含未实测readout的显式allowance，
不是完整seed实测或可靠12h完工承诺。后seed仅按完整先前seed最大walltime×1.25和共同剩余预算准入，不看gold。

只读监测：queue.log、queue_launch/queue_finished/queue_stopped，seed404/supervisor.log和logs/train-u0000-u0005.log，
rollout_progress、forward_progress、optimizer_steps、metrics与complete/stopped。
任何失败停在新证据目录，不自动重试；读出不使用U5gold，但检验readout的配对gold仍有rollout开销。
Phase3目前四分支无独立norm-only分支，不得将未跑的magnitude-only闭环优势写成已验证。

## N. 2026-09-19：pre-optimizer恢复attempt

最新状态以HANDOFF第16节为准。旧queue实际只运行4114.376564741135秒；其seed404退出code1，
505/606未启动。失败链：`capture_old_logits -> capture_storage.save_row -> runtime.disk_gate`，
对一个刚rename完成的原子进度临时文件stat导致ENOENT。GPU已退出，磁盘约539GiB可用。
修复用一次lstat并仅对已知`.rank-N.json.<nonce>`/`.publish-<nonce>`暂存文件忽略ENOENT；
其他文件缺失/权限/容量异常仍fail closed，不放松原760GiB/100GiBfree/80GiBreserve。

用户本轮明确授权修复和继续，并确认“扣除故障停机时间，累计运行仍不超过30小时”。
原preparation、科学协议、超参数、bank和router不变；新的源码amendment列95个hash，
`recover_queue.py`只允许预列出的修复文件变更，不接受无关源文件变化。

运行根Q=`C/artifacts/phase12/skillnet37-independent-s404-505-606-v4`；新attempt A=`Q/recovery-v1`。
所有旧stop/launch/permit/log保持，不移动或覆盖。A/failure-snapshot保存原SQLite与8rank进度原字节。
模型U0文件、原日志、segment配置等SHA在A/cohort-recovery.json；journal和2050个OLD SHA在
A/reconstructed-u1/manifest.json，batch为876MiB左右的rollout_batch.pt。
批次5511有效状态/77819token，原生补齐5512行后16组/128轨迹不变；所有已有OLD token/rank布局吻合。

恢复代码：rollout_recovery.py从原journal重建精确采样token、原prompt/组/回报；
ray_trainer只在显式许可的U1跳过环境rollout和重复adjust/balance；
envs.envs.reset仅推进已完成首批的game iterator并核验128game，不做action/router/model generation。
新OLD前向仍完整执行，用原payload逐bit比较后保留已有行，只发布缺失行；参考/adv/optimizer照旧。
没有可恢复的原vLLM采样logprob或worker RNG：logprob诊断省略、RNG登记seed重启，明确不称bitwise uninterrupted。
GRPO actual OLD、原始action tokens/advantages所需输入不伪造；若native bitwise mismatch则本attempt停。
所有新的进度文件/log写A/seed-404，原source forward_progress不动；后续实际训练batch/optimizer/模型仍写Q/seed-404。
pre_forward_batches/u0001.pt新增为后续I/O故障的早期durable恢复点；没有额外采样/实验或损失改动。

离线证据：7项storage/capture初测，18项targeted，最终673项full/100.47s（测试有重叠，不能相加）。
一次离线构建曾因compute_position_id_with_mask import位置错误退出，修正后构建成功；没有环境/GPU实验重试。
保留工程预检v4/throughput/capture的PASS，不重跑。

唯一正式恢复命令已于05:04:58 UTC执行：

```bash
cd /mnt/workspace/users/wangyifan/skill-RL/SkillRL
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -u -B -m skillnet_cohort.seed_queue \
  --plan artifacts/phase12/skillnet37-independent-s404-505-606-v4/recovery-v1/cohort-recovery.json --execute --detach
```

plan SHA af2a94ccd8bc5667643b86b8bde2abc51d4d9b0cdd5cba3be1129633cb774709；
queue PID1068291，seed supervisor1068370，train1068412，TaskRunner1070691。
budget actual restart unix1789794298.671236，virtual start1789790184.2946713，deadline1789898184.2946713
（2026-09-20 09:56:24 UTC）。先前4114.38秒仍计入，后续三seed共享103885.62秒。
恢复首seed条件投影91581.57秒（原94250减完整首批已保留的2668.43秒，只用于工程准入）；
并非实测完整seed，更不能承诺三seed12h。后seed仍按完整先前seedwall×1.25+3hreserve准入。
必须读A/queue_stopped/queue_finished及A/seed-404的supervisor/logs/forward_progress；旧Q的stop marker不删除。
新失败不自动重试。恢复细节已加入seed404未来新报告的provenance/caveat；不改历史报告或GitHub。

05:08:41 UTC首个恢复现场快照A/running-snapshot-01.json：8rank全部old17/689；136个旧payload
完整逐bit一致，source hash全匹配，pre-forward batch已落盘，原失败日志未变，没有新stop。
此时仍0 optimizer更新，未越过原256微批位置；不可把“启动/部分前向通过”写成训练或科学验证完成。
## O. 2026-09-19 recovery-v2：完成 OLD 后的 bool 兼容失败

最新状态以HANDOFF第17节与 `C/artifacts/phase12/skillnet37-independent-s404-505-606-v4/recovery-v2`
为准。本节追加工程记录，不覆盖第N节或任何旧失败证据。

- 用户授权继续修复重启、改进监控；此前授权累计运行30h扣除故障停机时间仍有效。
- v1日志751行：Python bool没有`.astype()`；当时5512 OLD/reference已前向完成，optimizer为0。
- `ray_trainer.py`验证0/1元数据后向量化原invalid-action penalty；`metric_utils.py`兼容object数组
  中Python/NumPy标量。无训练公式/超参改动，真实5512行奖励惩罚逐值等于原NumPy语义。
- 新 `recovery_forward.py`逐文件decode/hash/token核验5512 OLD，保存trainer-chosen FP32小cache。
  cache SHA=`5069b62f4014a0d4abdabb90e982347bdab8c8696908deb75a274fba27dcf5fb`；
  manifest SHA=`c9b5b66ba6693b234f6543dd71f239617e0c029d27d71a680b0a3f554d0bb751`。
  下一步真实actor每rank witness精确值/dtype核验；所有有效chosen值精确dtype还原才允许optimizer。
- 全CPU真实batch检查：`actual-batch-cpu-audit.json` PASS，128轨迹/16组/5512行/77834tokens。
  末尾部分mini-batch、多次Adam和entropy/KL路径测试PASS；完整`offline-full-01.xml`692PASS/101.67s。
- 新chosen OLD/reference输出即时持久化；首轮缺失OLD entropy日志不伪造，entropy正则仍执行；
  reference没有存下故重算。无U1额外环境rollout，无额外GPU预检。v1已登记的RNG重启边界不变。
- `recover_again.py`显式v2计划绑定旧plan/日志/模型/原始batch/cache/回归，冻结99源码；
  `recover_queue.py`新增v2校验分派。新plan SHA=`d86afcea58f00e8457e1fae3820d3a0780ce1ca227a80180c5466d744bf70a6c`。
- 已消耗7072.721801秒：v1 exit的7072.600905本来已包括第一attempt4114秒，不能双计；
  另计v1队列至supervisor的0.120896秒。恢复时剩100927.278199秒。
  剩余首seed条件外推89815.295130秒（v1外推减已完成的U1 OLD/压缩分配，加60秒witness），
  并保留10800秒reserve；不是保证三seed全流程完成的实测ETA。
- 队列06:35:27.983UTC启动，PID1107501；seed404 supervisor PID1107636；
  deadline2026-09-20 10:37:35.261UTC。初始0/5，不把进程启动当科学完成。
- `runtime_watch.py`30秒持续状态/历史，进度停滞仅告警；原`run.py`子进程退出与硬deadline仍兜底。
  新status原子临时文件名也纳入storage_bytes特定ENOENT竞态测试，不能再因诊断临时文件消失丢训练。
- 当前agent承诺跟到首轮完整更新。看v2状态而非旧stop marker；禁止对同attempt重复执行启动命令。
  后续新失败不静默重试。报告会附v1/v2恢复边界，仍为UNVERIFIED的科学结果状态。

07:21UTC交接补充：用户要求运行稳定后结束主动监控，替代此前agent人工守到完整U1的计划。
最后验收8rank×24个连续Adam step、梯度全有限、lr1e-6、99源码hash匹配，正式batch的OLD/
reference/actual mask精确对齐。记录于v2 `active-monitor-handoff.json`、
`actual-native-batch-audit.json`。当前仍是0个完整RL迭代（U1共44小批次），未运行505/606。
训练继续后台执行，30秒runtime-watch和原时间/磁盘保护不变；没有终止任何训练进程。

## P. 2026-09-20 recovery-v3：已完成训练后的导出接续

最新读状态以HANDOFF第18节为准。Q=`C/artifacts/phase12/skillnet37-independent-s404-505-606-v4`，
S=`Q/seed-404`，A3=`Q/recovery-v3`。v2在五轮已完成后导出失败，不是bool错误复发。

- 原5轮/640轨迹/8rank×204Adam全部保留，U5原生检查点为53,034,116,482bytes。
- `phase2/export_model.py`改用本仓库模块入口；新`export_verify.py`从8rank DTensor逐项
  拼接，与导出safe-tensor的427个FP32张量逐bit核对且检查finite。只有PASS才原子发布。
- `post_training_recovery.py`审计已完成训练边界，保存358个文件（58.69GiB）的hash及
  停机router账本快照，单独绑定后训练接续；不挪用此前pre-optimizer恢复入口。
- `run.py`跳过已完成训练和U0导出；`checkpoints.py`仅允许绑定的原U5及新A3暂存路径；
  `seed_queue.py`移除恢复seed404的training权限。原S/models/u0005.partial为空且原样保留。
- `runtime_watch.py`增加完成episode文件计数，不读取outcome控制流程；无进展仅告警。
- CPU 699PASS/117.79s，包括真实两rank Gloo小Qwen3.5合并、值损坏拒绝、无PYTHONPATH
  模块导入、禁止重训/覆盖、累计预算和评估监控回归；这些不是新的环境实验。
- 新plan SHA=`cdacaaedfc0c01477b8f76cdcba4c6f827a5b2f8db27c10f98a35eb877fed507`；102源码。
  保全manifest SHA=`357f7696e6dc27fbf86259e7daa7a6db728e0ea3788e41c8449e35333e0eddad`。
- 旧v2 exit的41864.444859秒已含更早attempt，加队列至supervisor的0.208716秒，
  共41864.653575秒。不可再次叠加旧4114/7072秒。剩余66135.346425秒。
- 新队列02:21:46.443UTC，PID1173697，seed404 supervisor1174098；
  deadline2026-09-20 20:44:01.789UTC。预计剩余首seed38292秒仍只是条件外推。
  后续seed按实际完整seed耗时和共同预算决定是否准入，不承诺三seed都能完成。
- U5于02:26:16通过427张量全量FP32 parity；导出19,365,851,624bytes。
  02:32:45八卡正式seen59/140已完成，各分片身份/digest无重复检查PASS；无新失败，
  正式unseen、Phase2及505/606未启动。空闲约447.4GiB；剩余约18h11m。
- A3/active-monitor-handoff.json记录本次现场。agent主动监控已结束，但后台保护仍运行；
  查A3/seed-404/runtime-status.json而非v2。新失败不得静默重试。

U0额外精度审计：S/models/u0000原数据实际BF16/426张量，旧metadata误标float32。
24个linear-attention norm与原生初始化FP32 master不同（最大0.00390625），严格FP32
比较FAIL保留在A3/u0-native-initialization-audit.json。按冻结BF16推理dtype及vLLM显式
FP32 A_log核验，全426项的运行参数值相同；24个norm通过实际vLLM权重加载函数CPU检查，
结果见A3/u0-runtime-dtype-audit.json。此为参数值/加载源码检查，不是新的GPU前向实验证明。
因此当前BF16协议继续，不能把U0称为无损FP32 master快照；后续live/offline fidelity仍必须
经过既有门槛。旧metadata不覆盖，原OLD全词表FP32文件未改，运行中的源码不热更新。
未来需要FP32 master端点时应在新协议修正base exporter的实际保存dtype并验收。

本轮没有提交、回滚、清理历史、API费用或GitHub推送；只读核对远程仍为674dd36。
恢复和精度边界须保留于后续论文附录；工程PASS不等于科学结论，材料状态UNVERIFIED。

## Q. 2026-09-20 recovery-v4：OLD-only 汇总接续和运行时限显式撤销

Q=`C/artifacts/phase12/skillnet37-independent-s404-505-606-v4`，S=`Q/seed-404`，A4=`Q/recovery-v4`。
用户授权“修复，评估完后开始跑后续seed”，随后针对累计时限答复“不再设置预算限制，先跑”。
只撤销时间限制，不撤销磁盘保护，不扩大科学规模、不引入API、不清理历史。

- A3完成原训练/导出后：U0 seen140/unseen134/anchors134、U0 O/P/N续跑540、8个测量分片。
  汇总仍检查live NEW5512行，与已登记OLD-only起点batch/终点teacher-forced replay协议冲突。
- `phase2/aggregate.py`显式窗口只要求起点OLD，原单步仍要求OLD+NEW；先完整验证，再no-clobber发布。
  `window_evidence.py`核对实际batch SHA、全OLD文件名集合、去重/分片、两对照、完整actual loss mask、
  动作/advantage、元数据、噪声、端点replay witness；5512行/2867决策/82362control-token全部PASS。
- `parameter_delta.py`修复潜在后续诊断错误：登记B0的FP32 master用原加载配方CPU重建，
  源模型所有登记hash匹配、加载无缺参、U5全参数集合/dtype/finite/tied aliases一致；
  4,205,751,296独立参数，L2=0.7933540249608585，relative=0.0010372375762014014。
  原BF16 U0元数据不覆盖，C/P/D仍用已完成的原分片；没有新forward/训练/环境交互。
- `readout_recovery.py`检查原已完成训练/精确游戏集合/U0完整续跑/原协议SHA/导出parity；
  A4/retained-readout.json包含1082文件、21,260,881,502bytes的hash及账本快照。
  已完成工作严格复用；seed404权限仅evaluation/readout，跳过training/export/OLD评估/measure。
- `common.py`仅把明确绑定的旧authorization_path在子进程内转至新permit，检查原SHA、preparation、
  run root及操作白名单；不改旧protocol/manifest/permit。真实permit替换CPU检查PASS。
- `seed_queue.py`/`day_budget.py`/`run.py`允许显式用户授权的无时限plan/permit；所有证据/存储/模型/
  source hash和顺序仍强制。hard_limit_seconds/target_seconds/budget_deadline_unix为null；
  有限正工程估时和磁盘准入仍保留。旧profile的30h是历史登记，由新plan明确覆盖，不热改profile。
- `reports.py`只影响未来新报告，记载本次恢复、U0精度边界及撤销时限；旧报告不写。
- 实际分片在A4/offline-check独立目录汇总、锁定预测、校验preparation/U5绑定PASS；
  原窗口没有发布汇总且无U5 gold。CPU参数审计结果已实际计算，此联调复用其结果而未重复加载。
- 709项完整CPU回归PASS/116.38秒（A4/offline-full-01.xml）；小型真实两rank Gloo导出校验包含在内，
  不属于ALFWorld实验。新测试覆盖旧单步仍拒绝缺NEW、window成功、错误协议/advantage/hash/witness拒绝、
  禁止重训/重复OLD工作、旧permit字节不变、无时限仍绑定磁盘、累计时间不双计。
- 新plan SHA=`1f15f784f6b2df7ca25e64e59f91395e3da86a9a5c0662301dcbc4217e525970`；105源码。
  09:17:56.873UTC队列PID1201615启动。旧累计48606.01990866661秒继续记账，deadline=null。
  同一新attempt只启动一次，后续失败不自动retry；完成404后顺序505/606，但磁盘不足仍会保护性停。

只读入口A4/queue_launch.json、queue.log、queue_stopped.json、queue_finished.json；seed404的
A4/seed-404/runtime-status.json。505/606正常阶段的runtime-status在各自seed根，supervisor日志在A4。
实际是否越过汇总/U5首次有效episode，以后续active-monitor-handoff为准，不以本节准备/启动记录替代。
源协议、旧错误日志、原batch/OLD、模型、checkpoint、已完成评估不改。未提交/回滚/推送GitHub。

09:24:50UTC实际接续验收：正式commit09:21:11.454917，prediction09:21:16.304176，之后才开U5。
已完成28/540条U5续跑，八个分片均至少2条，实际identity属于冻结expected集合、无重复，
prefix replay均verified且trajectory有完整steps，continuation seed匹配。
正式skill_context_features/token_signals parquet与离线目录相同SHA；原参数诊断L2完全一致。
105源码hash和旧aggregate错误log（SHA b25b844fd66d0c3b3ed958b0ee7def57e986ed4bb99a54fe7f9821b144a4d96e）未变。
queue1201615/supervisor1201814存活，新permit权限仅evaluation/readout、deadline=null；
无新stop，505/606未启动。A4/active-monitor-handoff.json归档工程状态，scientific_verification_status仍UNVERIFIED。
用户此前要求稳定后结束agent主动监控，本轮已结束；后台遥测与队列/磁盘保护不关闭。

## R. 2026-09-20：全技能／全部首调用锚点覆盖修订（尚待GPU执行）

Q与C同前，F=`Q/all-first-calls-v1`。用户要求取消评估量门槛、等505五轮RL结束后补评404并替换报告。
最后明确退回每条轨迹每个skill的第一次调用；所有首调用全评估，同轨迹后续调用只作频次描述。
不是将C/P/D本身改成首调用读出，也不对实际训练advantages做新截断/重权/求和。
更大“每次调用锚点”方案未启动；不把讨论中的100044条续跑当作真实实验。

- 现有404已完成5轮训练、两端seen/unseen性能、旧60锚点×9×2=1080续跑、旧5技能读出及报告。
  当前505正在U2优化，1/5轮metrics落盘；606未启动。原105runtime源码仍全部hash匹配。
- 新`first_calls_support`从完整134条U0 unseen来源构造404个首调用锚点，25技能。
  `coverage.csv`保留37技能×5阶段，记录5558原始调用与5154后续调用，以及训练token/A/game/轨迹。
  18种有actual-batch读出输入，7种仅有效用输入，后者必须保留NA而非伪造零。
  取消的是数量准入；没有自然锚点的12种不能强行注入。
- `first_calls_protocol`复用原准备/模型/银行/router/种子/控制，只发布新覆盖协议和仅readout/evaluation许可。
  runtime/bank/profile/原生U5逐bit导出/模型inventory/原seal均绑定；旧report SHA先归档。
  `signals`三数量门槛设0，D阈值与排序方向不变，旧calibration原字节复用。
- `first_calls_measure`按新skill全集重分八片，原决策精确token/decision分数复用，仅补缺失技能前向。
  若新分片首行没有原witness，允许重复单状态前向做witness，必须与旧token signals逐值相同。
  原真实batch/OLD/optimizer/mask/A不重采，端点概率仍在相同输入teacher-forced计算。
- `first_calls_reuse`逐项比对run/model/router/payload/控制/锚点状态/prefix/采样参数/continuation seed，
  原seal和每文件SHA均检查。1080旧续跑复制到新目录，按新3636-job分区重分配索引，明确episode_reexecuted=False。
  每端点3636条中复用540条、补3096条，共7272完整记录／6192新增环境续跑。
- `first_calls_run`只做缺失readout→完整聚合→OLD补齐→固定评分→NEW补齐→报告/封存→发布；
  绝不调用训练、导出、完整性能重评，不回收旧概率。评分计算不读取新目录目标labels，
  同时显式披露seed404旧labels已存在，不能因此冒称研究重新成为prospective。
- `first_calls_report`沿用配对game/continuation bootstrap（10000），单game区间NA但保留点估计。
  O/P/N从首调用起对目标skill的后续自然调用仍生效；不是单次payload替换的孤立效应。
  完整性要求与数量筛选分开；全37 coverage、效用、readout、NA原因、排名和原性能均保留。
  性能CSV原字节复制；旧恢复/U0精度边界继承。所有新评估/配对/报告/seal完成后才允许report替换，
  发布前目标文件若被人工修改则fail-closed。原窗口seal及历史分析不覆盖。
- `first_calls_defer`按PID/start_ticks/cmdline hash冻结queue1201615/505supervisor1207419/training1207476。
  不给训练PID/worker/进程组发暂停信号。训练子进程退出0、durable stage成功、metrics1..5、
  checkpoint tracker5和8rank完整性成立后，才SIGSTOP两个调度PID。
  已经起动的CPU导出/评估子进程自然排空；全卡无compute PID才把GPU交给404。
  404新runner结束或异常均finally SIGCONT旧调度，不重试失败实验。
  待旧队列整体成功退出后，按同一提前冻结规则准备F/followup-s505及F/followup-s606并补齐。
  全三个seed同协议完成才重新发布当前Q/reports；混合口径的中间报告不能视为最终跨seed证据。

工程验证：新增31项初检PASS；七组740项完整回归PASS/145.04s；最后32项定向PASS/37.55s。
首轮完整测试为739PASS+1FAIL，原因是隐藏CUDA时旧CPU optimizer测试未mock GPU内存logger。
只在`tests.skillnet_cohort.first_calls_cpu_plugin`隔离这个单测试日志，其真实optimizer/math/assertions仍执行。
XML全保留在`C/artifacts/engineering/first-calls-20260920-*`；没有真实ALFWorld/GPU预检或训练重跑。
实际schema/授权/112源码/原seal/旧episode兼容/容量门槛CPU审计PASS；当前Q占用164904196543bytes，
空闲434009800704bytes；404新增记录预留36.40625GiB，另外80GiB仍保留，超过100GiB安全底线。

F/plan.json SHA=`e3521e2e66a02a5cc006905e8432d35e52a8c5b05b68b8610b8ea14e7cadfeb9`。
16:09启动deferred PID1259771；16:09:53状态WAITING_SEED505_RL，paused_pids=[]，gpu_experiment_started=false。
旧7个seed404报告与archive SHA相同，F/seed-404/launch.json与paused-coordinators.json均未创建。
当前只是后台等待已登记，不是404新评估完成、更不是三seed科学结论已验证。
状态入口F/deferred-status.json；原训练进度Q/seed-505/runtime-status.json；新阶段日志各variant的logs/。
若deferred进程被SIGKILL/掉电，finally无法执行：必须先核对paused-coordinators PID身份、新runner已退出与GPU空闲，
再人工SIGCONT两个单独调度PID；完整恢复操作说明见ALL-FIRST-CALLS-20260920-v1.md，禁止无核对宽泛杀进程。
无时间上限；磁盘保护/异常退出/无自动retry不变。未提交修改保留，无git提交、回滚、push。

## S. 2026-09-21：SQLite容量扫描竞态和全首调用补评显式恢复

F/Q/C同R，A=`F/recovery-v1`。旧补评PID1273023、deferred1259771均已退出；
原F/assessment.log:238报 `router.sqlite3-journal` 在枚举后、lstat前消失。
原F/seed-404/stopped.json SHA=`ddd9c56c1cba4462cbc79786418be293249748aff58fee4e8ff7d4ba2150e3b5`保留。
原分片U0计数237/235/230/234/233/233/234/235，共1871；完整JSON也是1871，无孤立文件。
8个读出分片/committed已完成；实际5512 OLD行、5511被覆盖决策，完整batch/mask/identity/witness审计PASS。
新增聚合features SHA=`14ac78af2c8923a56e7b1dff8f416bb0366321b0e0c552e81242610ad6bd4c30`；
token signals SHA=`b19379d6016fe0ad6846ec42fcb93aa9b5292b2168cf582ace4b096deacbd9fe`；恢复不重复这些计算。

核查期间旧queue1201615与505supervisor1207419自然结束，seed505/complete.json为complete。
两端seen/unseen分别140/134，U0 anchors134，旧两端效用540+540，全部保留复用。
A4/queue_finished.json为budget_stop、completed404/505，not_completed606；
A4/not-started-606.json明确remaining_shared_disk_budget，seed606目录不存在。
原606估算新增176.859117GiB+80预留+100最小空闲；磁盘约332.14GiB，cohort约225.62GiB。
因此是物理空闲准入不足，不是760GiB cohort上限/时间限制；当前至少差24.8GiB，补评也会增长。
未执行清理。原window_storage回收仍需每行独立逐bit再生证明，不能把无损解压等同于无需原文件再生。

新代码仅两个模块：
- `first_calls_storage`复制原容量语义、不开symlink；SQLite三个sidecar仅在对应普通DB仍存在时
  容忍缺失，存在则计大小；进度临时文件沿用旧白名单，原子episode JSON/完成receipt要求严格
  路径及其永久替代文件已存在。未知缺失/权限错误不吞，保护数值不变。
- `first_calls_recovery`新explicit attempt，不改原F plan/permit/protocol/source SHA。
  prepare要求已复核的特定失败、无旧补评进程、无U5/预测、786项CPU报告、逐条有效续跑及完整readout。
  绑定1989个既有文件与8份索引原前缀；未完成/孤立/坏身份的记录fail-closed，不自动覆盖。
  允许命令只有first_calls_measure、phase2.aggregate、phase2.evaluate；恢复404不调用前两者，
  不重复import U0；完整分片也不初始化policy，部分分片使用原evaluator的completed-ID跳过机制。
  新日志、watch和stopped都在A，不覆盖原失败。每60秒正确容量检查；没有超时/失败重试。

原runner规定505/606补评需全旧队列complete；本轮改为新入口逐seed真实完成核验，
只接受真实all-complete，或已审计的completed404/505且606因容量从未启动这一退出状态。
没有改科学协议或伪造旧queue_finished。404完成后新准备F/followup-s505，复用同一提前冻结覆盖规则；
505完成后写A/pending.json和BLOCKED_LEGACY_STORAGE，606需另行解决容量/正常RL准入。
在606缺失时不调用publish_cohort，不能把两seed结果称为跨三个seed已验证。
无旧PID/group信号；异常只停止该恢复器自己启动的评估子会话，旧RL/性能不重跑。

测试：首次target71PASS/1FAIL为新fixture过早monkeypatch协议函数导致旧reuse模块import绑定污染，
仅修正测试import隔离，target-v2 73PASS；随后增加调度/白名单/505路径测试，完整786PASS/156.93s。
XML全部保留于artifacts/engineering/first-calls-recovery-20260921-*。
其中真实SQLite事务提交在walk/lstat之间移除journal的测试PASS；永久证据、错误marker、孤立记录、
截断索引、错误seed/anchor/replay、索引前缀变化、缺失磁盘条件均拒绝；不执行真实RL/GPU预检。

A/plan.json SHA=`5847e38b1ab7e94a987a50a7fd943390afdeb4d614558a8df89043196a7d5dfd`。
114源码=112原冻结源码+2新增模块；新记录容量36.40625GiB+原80GiB预留，总reserve116.40625GiB，
初始332.140625GiB可用，因此本次补评准入通过，并未绕过606的更大训练准入要求。
04:14:42 UTC launch，PID1295037；04:15:51八个U0续跑子进程1295346–1295353启动。
首批有效新增episode的验收应另看A/active-monitor-handoff.json；此启动记录本身不声称完成。
最终报告替换仍须全部paired episodes/analysis/seal/原报告归档SHA通过，原七份404报告当前未变。
科学MaterialPassport仍UNVERIFIED；代码恢复PASS不构成效用变化/预测性或三seed泛化结论。

04:18 UTC实际验收追加（优先于上面的启动状态）：本次A/recovery-v1于04:17:07停止，
shard5 exit1；其他七片是恢复器按自身错误保护SIGTERM(-15)，不是七个独立模型错误。
新错误为RouterCacheError: Prior failed/incomplete local query; no automatic retry。
`router_cache.reserve_many_local`拒绝任何已有attempt却无decision的key；原故障SIGTERM留下4条
started reservation（id30851–30854，00:51:42），本次停机又留下id30856（04:17:06）。
SQLite quick_check=ok，30851成功attempt=30851 decision，0成功attempt缺decision，18382 hits；
backend=skillrl_embedding_state_batch，Qwen/Qwen3-Embedding-0.6B，无外部API。
数据库SHA=`cda4c3db81f9dc870f28de5080873b3d2b96e0a9d854941a9e48dd500f191f58`，306270208bytes。
原缓存未被诊断改动；不能通过删attempt、清空库、改key或放宽全局no-retry保护来掩盖旧账本。

新增有效episode仅1条：`895e14bbf45f74596887916e`，inventory-management/evidence/null，
prefix verified，continuation seed62011；04:16:56.194729写入，SHA
`d32f1d1ca92635fc0701aef56a07a25648ceecaa256c337e55bdd8d6e1c06531`。
U0现1872/3636，分片238/235/230/234/233/233/234/235，剩1764；无孤立文件/截断索引。
原1989文件与索引前缀、114源码、旧7报告再次PASS。U5及505补评均未开始；8卡2MiB/0%。
新失败shard log SHA=`08e2214244fad9d24afd3e6d9550c4c7b9f1db016177b7292f008e876fed4fd2`，
失败完整快照A/active-monitor-handoff.json包含5条reservation精确key/时间和其他hash。
786项CPU PASS不包含这类真实中断后孤立reservation的恢复准入；需补相应测试和GPU启动前检查。

按academic-research-suite/experiment-agent失败复核规则，已通过异步问题请求用户决定下一次：
保留原attempt账本/成功decision，为这5个明确未完成的本地key登记一次显式续算授权，
使用新版本入口/新attempt（不是再次执行A原命令），继续404/505，不重跑成功episode或RL。
尚未实现cache续算白名单/再次启动；不自动重试。606的额外24.8GiB以上空间问题仍单独待解决。

## T. 2026-09-21 04:44 UTC：五条本地query获批显式续算并通过实际接续验收

用户在S停点后批准“允许修复这5条未完成查询后继续”。A2=`F/recovery-v2`，不覆盖A1及更早失败。
两个新增模块`explicit_router_resume`、`first_calls_router_recovery`；没有改原114源码字节。
原failed/started查询的默认no-retry保护保留，只有新入口、正确模型/protocol/cache/seed404根/
U0或U5评估配置、精确5-key授权匹配时，局部cache factory才使用受限适配器。
API的RouterCache类不变，单次/API reserve在适配器中直接禁止；本地预算不增加。

prepare先核验特定旧失败、原SQLite SHA、5条id/key/time/status、无运行进程/sidecar，
原1872条轨迹和readout完整性、旧queue606仅磁盘停点、完整CPU测试，再复制整库快照（不删/修改原库）。
授权绑定每条原reservation完整row SHA；对实际可见输入/cache key再核对。
BEGIN IMMEDIATE内整batch先验、调用预算计原attempt+新增attempt，之后append。
既有原attempt永远不改key/status/data；新增attempt和成功decision携带授权SHA/原id及row SHA。
8线程争抢只允许1个续算，各key新尝试消耗后即便中断/失败也不再重试。
旧5条started会永久留档；是否已处理应join新decision/explicit_local_resume，不可数raw started误报活跃任务。

每个评估阶段GPU启动前，SQLite quick_check、协议、每条成功decision digest、未知未完成query、
以及原四表快照的全部旧行逐项比对。未知中断或任何旧行改变都fail-closed。
允许成功新记录/正常cache hit追加；snapshot SHA绑定在plan和授权中。
seed505正常新缓存不启用这5-key适配器；同样在阶段启动前检查缓存是否有未批准中断。
工作流只在该新恢复进程中临时替换assessment类，finally恢复，原文件/source hashes保持。

测试：初次专项30PASS/1FAIL是非法root先读文件才校验权限，调整校验顺序后专项36PASS/9.93s。
完整817PASS/154.81s，16条既有依赖告警，所有XML保留在artifacts/engineering/explicit-router-resume-20260921-*。
测试包含真实冻结router接口+CPU fake encoder、成功缓存命中、5次追加/全旧行保持、并发原子许可、
再中断拒绝、错输入/model/cache/预算/API/授权/快照拒绝、404/505子进程接线、原evaluate参数透传、
GPU启动前阻止未知reservation、工作流类异常后恢复；无新RL或额外GPU预检。

A2/plan.json SHA=`18eaaddd388dff9a2028be797f13834d700d97cb5e557b1e81daaf7e62fc54fe`，116源码、2009文件。
router-resume-authorization SHA=`4e04becb9c8f712ddc813ae0d62351a4f862bc5951b54ce5e27e415b956dd56c`。
快照A2/router-before-resume.sqlite3为306270208bytes，SHA
`cda4c3db81f9dc870f28de5080873b3d2b96e0a9d854941a9e48dd500f191f58`，与旧完整数据库原字节一致。
备份后空闲331.85546875GiB，cohort225.903492861GiB，补评reserve116.40625GiB通过100GiB底线。
606的356.859GiB准入不满足，仍不启动、无清理、无存储豁免。

04:40:08 UTC启动PID1302727（start_ticks473313688），04:41:25启动八个子进程1303116–1303123。
04:44:17现场5条原→新attempt：30851→30878、30852→30860、30853→30858、30854→30900、30856→30857，
新的5条均success、授权SHA匹配；旧5条仍started且原数据不变。
在只读SQLite事务中与快照比对旧protocol/decisions/attempts/cache_hits，改变行数分别0/0/0/0。
原2009文件及索引前缀、116源码hash重新PASS；原7份404报告与archive仍相同。
U0已1895/3636，八片241/238/232/237/236/235/238/238，比本attempt起点新增23条，
各片+3/+3/+2/+3/+3/+2/+4/+3；新trajectory身份、prefix replay和steps已逐条检查。
进程存活，无A2/stopped.json；U5及505新补评尚未开始，尚未发布新最终报告。
完整验收快照A2/active-monitor-handoff.json。agent主动检查到此结束，后台按既定顺序404→505，
之后606仍pending，不视为三seed完成；科学状态UNVERIFIED。无API/提交/回滚/push，无文件删除。

## U. 2026-09-21：505端口冲突和FileStore单rank显式恢复

A2后续完成404全部新口径效用、统计、seal与report-publication；用户询问时已只读核验
47份报告/汇总/索引SHA、7份正式report-publication和7份旧归档全部一致。404新版结果
在F/seed-404，正式报告视图在Q/seed-404/reports；此后不修改该已完成结果。

505补评路径R5=`F/followup-s505/seed-505`。A2在11:04:43.801 UTC退出，
日志A2/seed-505/logs/utility-u0000-shard7.log:64为DistNetworkError/EADDRINUSE/40669。
八个分片所报告端口为43787/47285/36107/44019/59545/53955/47371/40669，互不重复；
占用者身份没有证据，不能推断必然是其他分片抢占。确认本机vLLM0.22.0的UniProcExecutor
通过get_open_port执行bind探测后关socket，然后在另一步初始化TCPStore，有TOCTOU窗口。

用户明确授权修复并续跑505。本轮独立新增两个模块，原116源码字节不变：
- vllm_file_executor：继承安装UniProcExecutor，仅覆盖_distributed_args，严格要求
  single node、world_size/world_size_across_dp/TP/PP/DP均1、可见CUDA local_rank0。
  mkdtemp在/tmp创建新的0700目录和从未存在过的store路径，返回file://、rank0、local_rank0。
  audit receipt先记录engine PID、URI、transport、rank，FileStore自清理不在磁盘证据扫描树内。
  不重用或删除原路径，不探测TCP端口，不修改NCCL/Gloo/模型加载或调度/采样实现。
- first_calls_port_recovery：新A3prepare/execute/evaluate入口，评估子进程局部替换
  VLLMPolicy构造，但generate与generate_batch继承原实现。引擎kwargs和原build_engine
  唯一差异是executor类路径。process-local类和sys.argv在finally复原，不改原文件。

CPU专项42 PASS/15.62s；完整859 PASS/172.97s、16条既有告警，XML
`artifacts/engineering/seed505-port-recovery-20260921-{target,full}-v1.xml`。
完整XML SHA=`73a61d755ce4d33eaf54c965846cbea413937c6f6b33308617e0e92560276b0a`。
包含真实8个进程FileStore/Gloo初始化且MASTER_PORT被占用、实际vLLM executor类解析/
worker-method继承、新路径不复用、越权rank/device/root拒绝、同参数/同generate代码、
命令白名单、未知cache中断在GPU前拒绝、自有子进程错误退出不自动重试、505正确续跑顺序。
CPU测试没有使用GPU模型或生成新的ALFWorld轨迹。

准备时R5 U0完整540/3636、分片69/69/66/67/66/66/69/68。每条identity、原prefix replay、
continuation seed、完整steps、原anchor语义、SHA和索引原前缀核验。8个readout分片/
committed真实batch token/action/advantage/witness验证PASS；不重跑任何readout前向。
R5 router.sqlite3有正常protocol，但attempt=decision=0、unknown interruption=0；
无须且未赋予任何新的router查询续算授权。所有旧报告原SHA与归档均相同。

A3=`F/recovery-v3`，plan SHA=`9d319b2c7e9a4fb2d2025f2ec1537f272544b152dee6ecfcc609e959f92b2e16`。
绑定118运行源码和6份已安装vLLM/PyTorch实现文件SHA、8081保留文件以及8份U0索引原前缀。
保留范围包括404全部7361条sealed文件项及正式发布的7份报告、A2失败日志/计划、
505现有readout/support/manifest、540轨迹、两个seed旧报告归档等；不复制模型/原轨迹。
容量准入空闲326.699GiB、cohort231.005GiB、本次required116.406GiB，满足原100GiB底线、
80GiB检查点reserve和760GiB总量保护；没有为606降低其更高RL准入要求。

新进程PID1325542按用户授权只启动一次。执行阶段：505 U0补3096→lock_prediction→
复用U5旧540→补U5→report/seal→核对所有retained→publish505→complete505；然后
写pending606/BLOCKED_LEGACY_STORAGE，不调用publish_cohort/prepare606/任何训练或性能重采样。
完成的分片不启动policy，已完成trajectory_id在原evaluator里跳过。每阶段启动前检查缓存，
每60秒原容量保护、每30秒StageWatch；新失败只终止本attempt自有子进程session，无重试。

本段为prepare通过与启动记录，不把它当作实际生成完成。实际八卡接续验收会在
A3/active-monitor-handoff.json和本节后续追加；state/logs均在A3，A2 STOPPED保留历史意义。
无API、无Git提交/回滚/push，无旧报告/证据删除。科学MaterialPassport仍UNVERIFIED。

11:41:01.689588 UTC接续验收追加：supervisor1325542（start_ticks475801513，command SHA
`4c4b63768699442e341bfbb55c4d57b48c07ba5fcde7e3ff6bdcafe4a37a6433`）存活，八个评估器
1326115–1326122存活。引擎PID按shard0–7为1326316/1326340/1326298/1326301/
1326315/1326341/1326326/1326285，均实际使用独立file:///tmp/skillscope-eval-rdzv-*/store。
11:37:22–23日志记录FileStore/NCCL初始化，之后八份ENGINE_READY receipt均对应正确
plan和rendezvous SHA；原来的TCPStore端口分配路径没有使用。初次推理有Triton JIT告警，
不属于异常退出，未为此修改参数或重试。

U0当前563/3636，八片71/71/69/70/69/69/72/72；相对原540分别+2/+2/+3/+3/+3/+3/+3/+4。
这23条新增完整episode的预登记identity、分片、continuation seed、original_anchor语义、
prefix_replay_verified、完整steps与索引内容全部核对PASS；八片原索引字节前缀未变。
另于11:38:18再次核验118源码、8081保留文件、8前缀PASS，包括404完整seal/正式报告，
505未更新的旧7报告及归档。无A3/stopped，U5及505报告尚未打开，606仍未启动；
每GPU15217MiB，实际磁盘326.6367GiB。A3/active-monitor-handoff.json保存完整现场。
主动检查到此结束，后台无时间上限，但保留30秒遥测、60秒磁盘保护与失败不自动重试。

## V. 2026-09-21 11:57 UTC：505显式暂停与中心化数值审计边界

### Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run（用户暂停操作）及公式/源码只读核对；非实验重跑
- Origin Date: 2026-09-21
- Verification Status: ANALYZED（进程暂停已实测；数值修正和科学性能仍UNVERIFIED）
- Version Label: seed505_pause_20260921T115708Z_v1

用户最新指令“目前先暂停下505的评估”优先于附录U的后台继续。
本轮未启动新实验、未修改C/P/D、未替换报告、未打开API，未提交/回滚/推送或删除文件。
先校验A3/plan.json SHA、上次验收记录的supervisor身份及八个指定505 U0子进程；
用pidfd定向SIGSTOP，且发信号前再次检查start_ticks/command SHA，先停父调度器，
再停评估器及后代。17进程全部T，没有终止/退出，没有对其他任务发送信号。

暂停时间11:57:08.819266 UTC。父PID1325542，评估器1326115–1326122，引擎
1326285/1326298/1326301/1326315/1326316/1326326/1326340/1326341。
完整身份、信号前后状态在A3/pause-20260921T115708Z.json，后续观测在
A3/pause-verification-20260921T115708Z.json。后者两次观测11:57:49.896246和
11:58:50.396810 UTC：全部仍T，八份JSONL SHA/大小无增长、均为完整换行JSON记录。
U0索引810条，八片103/102/99/101/101/99/103/102，含原540条和恢复后新增270条。
此次计数不是重新逐一核验810条轨迹内容；前次23条的完整语义验收不外推到全部新轨迹。
U5新口径目录无索引，尚未导入旧540条；不等于旧U5证据不存在。404报告和606停点不变。
暂停后nvidia-smi八卡utilization均0，每卡memory.used=15217MiB，进程和显存现场保留。
原A3状态/遥测停留在运行态，不覆盖旧记录；读取当前状态应优先采用本暂停记录。
不要打开缓存写连接、整理journal或对正常in-flight查询添加续算许可；它们仍属于活的暂停进程。
未经用户恢复/终止指令不发SIGCONT/SIGTERM，恢复前核验身份、依赖及暂停时长影响，
不能把挂起时间当作计算耗时，也不能承诺任意长时间挂起后IPC一定无需额外验收。

数值核对范围：phase2/direction.py与aggregate.py及measure.py的数据保存路径。
direction.py先old_o.exp()，构造d，再做u/delta、乘积与平方，随后才sum(dtype=float64)；
因此输入为FP32时，这些前置步骤仍为FP32。现有C_upd_centered也使用该d，
并非独立高精度基准。aggregate_features会重新用原C、原delta_norm及阈值构造D门控，
以后即使导出中心化字段也须防止被汇总器重新覆盖。

精确算术下p归一化给出sum(d)=0，所以对每token的词表delta减均值不改变P，
不是对各技能汇总P减均值。中心化C一般改变分母/大小，tau_C=0时在有效非退化行
不改变符号；delta中心化范数及valid条件仍需单独检查，不声称任何阈值下D都必然相同。
用户引用的77,819 token、27门控差异、18技能排名不变和16-token FP64核验为既有审计，
本轮未重新执行这些计算，也未把门控替换结果当成完整稳定P/D修正结果。

建议（待用户确认，尚未编码）：保留原FP32版本；新增端到端FP64、规范化概率/
稳定零和投影的等价实现，校验平移不变性、中心化P一致性、动作概率近1/零advantage/
小范数边界；epsilon、tau_C、tau_delta、数学有效性定义、token/skill聚合和候选池不改。
raw/centered C和D可分列诊断，P raw/centered应按理论等价校验，不当作独立新预测信息。
如果另改归一化分母、门槛或支持规则，则是独立协议变体，不能混称数值修正。
旧效用轨迹与RL结果可复用；完整readout修正需要原向量，已存scalar parquet不足以
恢复全部高精度投影，measure.py仅为每分片首decision保存witness。必要时需在原batch上
重做端点/对照前向，但本轮未启动前向或任何重算。所有新旧结果须独立输出并保留前后对照，
不能依据已见404标签挑选更好版本后回写为原预登记结果。

## W. 2026-09-21：获批FP64零和读出修正与三版D对照

用户在附录V后确认数值稳定性修正＋原版对照，先重算404，再启动505及同规则修正；
随后再次明确“启动完整seed404重算”。这授权数值重算与条件恢复，不授权新RL/效用重采样/606。
原118运行源码全部SHA匹配，新增4个独立模块：stable_direction、numerical_readout、
numerical_readout_report、numerical_readout_run；不改正在挂起的505来源代码与缓存。
数学、字段、固定阈值、运行命令、输出路径完整记录于
`SkillRL/docs/experiments/phase12-independent-v4/NUMERICAL-READOUT-20260921-v1.md`。

固定legacy_recorded / stable_raw / stable_centered_gate三版。中心化D字段
`D_centered_contribution`已经实际接入逐token计算、技能汇总、同池AP/AUROC/top-k报告，
并非仅在文档中预留。centered P仅恒等式检查，两版稳定D使用同一P；不声称已改善性能。
epsilon=1e-12、tau_C=0、各seed已有tau_delta=1e-8，支持和聚合规则保持。
OLD方向概率用FP64 softmax规范化，d动作分量用尾概率和，投影用`A sum p_i(x_a-x_i)`；
避免共同分量与近1动作概率相消误差，保持四份原log-probability向量的u/delta定义。
FP64临时运算按16实际token分块，不截词表；原比较器仍按完整原token形状运行。

最终专项43 PASS/5.21s，完整898 PASS/162.01s，16条既有告警。完整XML SHA
`21219188250536eb48e90465fe6525bc6e7b0f0ef8ef0be43498dc44e67e2c86`。
八个旧witness共222 token×control CPU核验通过，P中心化最大差2.220446049250313e-16；
原404排序报告路径重构的keys一致，AP/AUROC/相关系数误差约1e-16（CSV往返）。
这些只验证工程与数值，不替代完整404重算或跨seed科学结论。

N=`Q/numerical-readout-v1`独立输出，原报告不替换。四worker各两GPU放原U0/U5 BF16模型，
八逻辑分片分两波；先算最长action并恢复原落盘行序，以尽早检验512-token内存峰值。
模型/temperature/SDPA/hidden层与原实现相同，仅设备放置不同。每decision的同次前向旧版
标量必须逐值等于原表；不满足即停止，不自动放宽或重试。witness用已有无覆盖.publish发布，
不引入旧.partial文件扫描竞态。启动时每卡至少15,000MiB空闲，不驱逐暂停505。

执行器先404全重算、汇总、同池报告与文件SHA验收，再505数值重算和commit，
之后才能核验17进程身份、暂停索引无推进、数值worker已释放GPU并通过pidfd发SIGCONT。
先恢复引擎，再评估器，再原调度器；原505继续剩余效用，不新建router续算许可或重新发环境命令。
等A3原505完整补评/原分析自然结束，再生成N中的505修正版对照报告。
三个版本、全部分层都报告；原效用compact标签/CI逐字节复用，不重采样bootstrap。
原source评分/报告不覆盖，OLD、batch、模型、所有轨迹保持。

30秒监测新worker、暂停505身份和旧索引；原100GiB空闲/80GiB保留/760GiB上限不变，
新产物另保留8GiB。取消时间预算的旧授权仍有效；606未启动。前置失败不恢复505，
任何阶段均无自动重试；SIGCONT意图和完成分别落盘，部分恢复不能误写为从未尝试。
此为最终测试与准备登记，实际plan SHA、PID、GPU最长动作验收及停点在后续追加。

### W1. 2026-09-21 12:44 UTC：GPU验收失败，完整重算当前已停止

N准备成功：122源码、111旧输入文件绑定；plan SHA
`ee6ec2cdee6333ed8ecf7589367b94f58a2e66ae808a097d6c660cfeef02dfa0`。
磁盘准入free326.441GiB、需保留88GiB（80检查点+8新产物），原保护不变。
launch时间12:36:29.071773 UTC，supervisor1342117；shard0..3
PID1342183/1342184/1342185/1342186。原U0/U5 BF16模型分别在各worker两卡加载，
其中shard1/2/3的最长动作第一次评分即发生CUDA OOM；shard0完成19个短动作后随组停止。
实际停止时间12:39:55.113956 UTC，N/stopped.json和runtime-status.json均为STOPPED。

堆栈：first_calls_measure.score_decision → numerical_readout.scoring_adapter →
stable_direction.token_signals → legacy_token_signals → direction.py:41
`base_u.square().sum(-1,dtype=torch.float64).sqrt()`。申请970MiB时余量约839MiB，
旧505引擎占14.85GiB、该新worker占15.67GiB。512-token原版比较器整段临时向量仍大；
只有FP64修正版按16-token分块，且本次尚未运行到长动作的修正版分块部分。
故不应把错误归因为“FP64分块仍无效”或训练/磁盘失败。15,000MiB启动准入没有覆盖原版峰值。

只读复核原表155638 token×control rows；已通过的首个row1008每control18 token，
不能外推512-token显存与数值验收。19个通过decision仅留进度日志，分片级产物尚未发布；
完整shard receipt为0、404新reports不存在、505数值任务未启动。
12:42–12:44 UTC再次核验原17进程身份/start_ticks/command SHA仍匹配且为T，
八个505 U0 JSONL SHA均未推进（810/3636）。N无resume-intent/resumed；新进程均已退出。
每GPU恢复为暂停505的15217MiB，利用率0。122源码仍匹配，原报告/原效用不动。

本次没有GPU自动重试，没有终止505以驱逐显存，没有更改原版阈值、候选池或丢弃长动作。
失败日志/plan/测试/attempt原样保留，后续修复须独立恢复版本并获得一次显式重启确认。
建议给原版对照也做逐token分块并保留逐值原标量一致性门禁，先用完整512-token实卡
验收后再继续全部404，成功后才按既定顺序修正/恢复505；该修复与第二次启动尚未执行。

## X. 2026-09-21：释放505后独占GPU，新的数值恢复与505磁盘接续

本节依据用户新指令替代W中的“保留17进程并SIGCONT”方案。用户允许不保存505内存状态，
404重算后再新启动505。release_paused_505完成8项CPU测试后，重验原505完整进程树和8个引擎GPU所有权，
预先打开pidfd、再次确认暂停身份，定向SIGKILL全部17进程，未SIGCONT或向其他任务发信号。
用户主动释放不计为又一次实验崩溃，不触发自动重试。原内存局内状态已丢弃，已落盘记录未删。

N2=`Q/numerical-readout-release-v1`中release-intent/release-505保留原授权、所有PID身份及实际时间。
原U0八份索引和cache字节SHA不变；释放前数据库备份在release-cache-snapshot/router.sqlite3，
SHA=`e7194869e3b202dcc34f4b4f2f73a20fbc07b08bfb4d4827d6ecb24496bc1b0e`。
没有SQLite journal/wal/shm侧文件。SQLite quick_check=ok，5831 attempts、5826 successful decisions；
5条未完成ID5826/5828/5829/5830/5831全为11:57:08暂停时的started。旧行、key和input用SHA绑定。
新启动505需要对这5条各一次显式续算才能恢复已丢弃的局内进度；其他中断及重复续算继续拒绝。
已有ExplicitLocalResumeCache和FileStoreUniProcExecutor按进程内适配复用，二者源码不改。
全部810条原505轨迹逐一检查身份、分片、steps、prefix_replay_verified、continuation_seed及anchor，
无孤立文件；剩余2826条U0，U5按既有规则导入旧540条后补缺，不重跑RL或完整seen/unseen性能。

原122份源码仍相同；新增3个独立模块（release_paused_505、numerical_released_run、released_505_evaluation）。
本次选择释放显存，不再修改原版比较器的token形状或分块；数值实现与W完全相同。
四worker各占一对GPU，U0/U5 BF16模型分卡，八逻辑分片分两波，启动最低空闲28000MiB。
先算最长真实动作，原版逐值比较与P中心化恒等式阈值不放宽；失败保留，不自动重试。
505在404完整报告验收前不得计算新版读出，在505新版commit前不得新建效用worker。
505新评估入口验证restart-intent绑定404 complete、505 commit及5条查询授权的SHA。
数值修正版报告仍只写N2，505原版新全首调用报告仅作为对照写到原未完成的followup目录；
不调用publish覆盖任何原训练报告。旧效用标签、轨迹、U0/U5、router协议和历史失败全部保留。

详细数学/指令/报告边界见
`SkillRL/docs/experiments/phase12-independent-v4/NUMERICAL-READOUT-RELEASE505-20260921-v1.md`。
当前准备阶段已通过111项综合专项；完整CPU回归、plan SHA、GPU512-token实测、启动PID另追加。
旧A3过期runtime/deferred记录仍显示running但进程已释放，不能据此判断运行或自动唤醒。
N2原始release记录是当前资源交接证据；前次N/stopped为保留失败，不覆盖。

### X1. 最终测试、plan与实际GPU启动验收（13:15 UTC）

最终935 PASS/173.19s，16条既有告警；完整XML SHA
`e08bc430b1c79562b053b37a06695b843b56b25219e3b4cb73df721fd1041845`。
N2新plan共125源码、118输入绑定，SHA
`4edd50b1afcbeb829a8e98721e8ab440e7ffdfd6ca9af713df16d5ef064380cf`。
四份数值源码SHA仍为W中版本；本次没有将原版比较器改为分块，采用用户选择的释放显存方案。
准入free326.417969GiB、used231.339725GiB、reserved88GiB，实际540旧效用复用与全部anchor口径不变。

N2/launch.json时间13:11:47.626301 UTC；父PID1348866、start_ticks476378188、command SHA
`0a01ae31a9a0aafe05b2806c3c57d823652eb792ef4d72ba86b6320a4b78e1d8`。
四worker PID1349405/1349406/1349407/1349408，start_ticks均476390346，父PID1348866，
物理GPU分别(0,1)/(2,3)/(4,5)/(6,7)。运行目录N2，模块numerical_released_run。
启动的证据/模型哈希和磁盘扫描阶段较慢；无GPU利用率时不能直接判为崩溃。

13:15:29实际进度22/11/10/11个decision，四片各689；尚不是完整分片或报告。
首个旧索引分别1008/17/1378/339；原标量表只读核实token数为18/512/512/512（每control）。
每个已完成decision均通过同次前向legacy标量与原表逐值一致，P中心化日志最大误差8.88e-16；
三个原失败分片均通过512-token。GPU显存观测最高20197MiB，未再发生OOM。
无N2/stopped，无505 restart-intent；旧505索引与释放收据再次PASS，125源码SHA未变。
细节保存在N2/active-monitor-handoff.json。自动队列仍在推进404，人工启动验收完成后结束本轮，
不承诺全部统计结果已产生，也不将一次成功前向当作整个流程已无故障。

## Y. 2026-09-22：seed404 reward-directed 候选全集与独立 CPU 统计

本节为新任务，替代“404尚在前向”的历史状态说明；本轮来源N2/seed-404已complete，
并按其provenance核验原稳定标量/效用标签/报告。用户仅要求在404探索，未在505/606试新变式。
研究边界和公式见`SkillRL/docs/experiments/phase12-independent-v4/REWARD-VARIANTS-SEED404-20260922-v1.md`。

入口均在C下，显式`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`：

~~~bash
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B -m skillnet_cohort.reward_variant_analysis prepare
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B -m skillnet_cohort.reward_variant_analysis run
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B -m scripts.verify_reward_variants_s404
~~~

这是已执行命令记录，不可在同一目录重跑；入口会拒绝重复attempt和覆盖。
V=`Q/reward-variants-s404-v1`；plan SHA=`7a767ce6a82133c5a4b333f3fecda73c0404823cb022ab338a365f4a76793b6b`。
registry列255个新分析分数（含reward、对应去reward、幅度），另6个原baseline，共261列。
39 reward公式×token/decision/game；game逐级平均token→decision→trajectory→game，未改首调用锚点口径。
所有新阈值只由pooled U0标量绝对值分位数确定，不按skill/gold选择；旧数学门控与分析主D完全保留。

单位advantage对照的P/C几何只能在原A!=0行重建；原A=0行仍留分母且几何为0，不能称为全集完全reward-free。
direct action类则有真正全行A=+1对照。符号负对照每条轨迹统一±1，C/P/gate一起重算；
保留局内advantage非恒定形状、两control身份与幅度，全部方法共用512组随机符号。
family-max AP用于显示择优风险，不具有确认性p值含义。C_upd与centered C是reward-directed参考，不归入无reward组。

2,000次bootstrap对技能共享game抽样、对每个抽中game共享continuation抽样，所有arm/endpoint已先做配对差。
每control 1,674次完整池draw、326次缺技能而整池NA；0pp AP 1,671次有效，另外3次无下降事件。
这是固定训练读出/候选池条件下的gold-label不确定性，既不覆盖训练seed方差，也不校正多公式选择。
旧效用与逐技能原区间不覆盖，完整coverage含37技能。18种readout共同池/25种自然锚点技能不混淆。

测试：283 PASS/16.96s，XML=`C/artifacts/code_checks/reward-variants-s404-20260922-v1/regression-tests.xml`，
SHA=`3ffb3ff286fa6ae128467061fa65b238f8d27c9b4aa6291d1672d80c92771bd6`。
initial/revised/complete-tests为开发阶段3次合成测试失败，修复枚举列、CSV的null误识别、NumPy整型JSON序列化；
final-tests为25项PASS，最后regression-tests为283项PASS，不能把旧失败标记写成真实RL/评估失败。
实际seed404分析一次成功、41.9985s，全部原输入/源码hash前后相同；没有重试或额外环境成本。

独立验证脚本逐行用sklearn AP/AUROC、scipy相关和手工方向混淆计数核对，5,220行通过；
与原报告9种已有分数各20行比较，最大误差约3.33e-16。V/direction-confusions.csv同时给出
原始准确率、平衡准确率、恒预测上升/下降基线和弃权，避免类别基率误读。

发布：只给R/phase2-complete-analysis.md追加第10节，前29,970字节仍与V/publication-original-phase2-complete-analysis.md逐字节相同；
旧SHA=`b13d3960b808af3c96b20bc5618c4e764da283146f75350e04704dc711ca88a4`。
另新增R/2026-09-22-seed404-reward-readout-variants-analysis.md，V/reports/phase2-results-expanded.md为旧数值正文+新扩展。
不能修改N2中的已封存report-provenance/complete以追溯混入本次探索。当前结果尚不支持稳健reward增益，
不得把最高候选AP=.693590等同跨seed验证或将新公式自动接入Phase3。所有未提交修改保留，无git提交/回滚/push。

### Y1. 常数保持修正与最终发布 v2（2026-09-22）

Y记载的v1实际运行与独立核验均成功，但随后数学负对照发现浮点伪排序：A固定+1的reward-only列
严格为−1，原加权reduce产生浮点末位技能差异，导致AP不等于事件基率。独立sklearn会忠实复现该伪排序，
因此仅有跨库数值一致性不足以替代理论恒等式测试。新增独立适配器和4项专项，不修改v1已封存源码。

`reward_variant_constant_fix.py`只在输入token值全部精确相等时直接返回该常数，所有非恒定列保留原运算；
trajectory-block符号负对照使用相同保护，512组符号与全部候选不变。没有选择新公式、重调标签或阈值。
详细规则见`C/docs/experiments/phase12-independent-v4/REWARD-VARIANTS-CONSTANT-20260922-v2.md`。
V2=`Q/reward-variants-s404-v2`，plan SHA
`ae5e7074a4736a0b8dede198dbd6a313d89c03b6410d26849a8de3ca67316136`。
plan保留19核心输入/125原运行源码，并额外绑定v1的5项结果及数值修正登记，合计25个输入；新分析源码6份。

已执行（同Y的CPU环境，非重跑指令）：

~~~bash
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B -m skillnet_cohort.reward_variant_constant_fix prepare
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B -m skillnet_cohort.reward_variant_constant_fix run
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B -m scripts.verify_reward_variants_s404 --output artifacts/phase12/skillnet37-independent-s404-505-606-v4/reward-variants-s404-v2
~~~

最终回归287 PASS/14.22s，XML为`C/artifacts/code_checks/reward-variants-s404-20260922-v1/constant-fixed-regression-tests.xml`。
v2约08:05:10→08:05:53 UTC一次完成；5220行统计、19836行预算，独立核验全部通过，9个原指标复现。
366个聚合单元被修正；主PLACEBO/all/0pp仅3个常数unsigned基线AP变化，117个reward分数AP均保持。
这3个常数列现在AP=.2777777778、AUROC=.5、Spearman=NA；主reward最高仍C_centered/decision=.6935897436。
符号family-max达到该AP的比例由v1的77.9296875%变为v2的77.734375%；仍不是确认性p值。

最终入口`V2/reports/phase2-results-expanded-v2.md`是数值修正封面加原自动扩展报告，保留该自动报告逐字节不动；
原模板内v1指公式族登记，不代表使用v1的常数伪排序。`V2/publication-verification.json`绑定最终人读报告、
修正收据/逐单元对照/方向混淆/测试/发布源码，验证第1–9节仍与29970字节历史备份完全相同。
V1新增`NUMERICAL-CONSTANT-WARNING.md`，原封存文件无覆盖；人工初版另存publication-initial-interpretation.md、
publication-initial-appended-report.md。新分析仍仅404，不改变任何既有其他seed队列或Phase3配置。

## Z. 2026-09-22：实际更新方向的reward校准扩展

协议与解释在 `C/docs/experiments/phase12-independent-v4/REALIZED-REWARD-SEED404-20260922-v1.md`。
新输出 `Q/realized-reward-s404-v2`，数学版本realized_reward_secant_v1；prepare-only v1目录
保留prelaunch-superseded.json和prepared-source-snapshot。此为启动前工程修正，不是失败GPU实验重跑。

新增四模块与 `tests/skillnet_cohort/test_realized_reward.py`，不编辑原冻结文件。
`vector_signals`用FP64/16 token分块直接保存 `||Hu||²`、`dot(Hu,Hdelta)`、系数、q、D及对照。
模型仍原BF16/SDPA，全词表；保留原U0现场OLD概率，U5和控制在相同实际输入上重放。
OLD逐行SHA对照原压缩账本，解压还验原始字节校验；U0/U5模型SHA对照旧数值评估清单。
逐决策新前向须精确复现旧稳定标量；不匹配即保留失败，不自动调整容差/重试。

科学约束：q=A*u[a]是固定优势的局部log-likelihood surrogate变化，不是clipped/KL总目标，
也不是逐步因果回报。r_real=q*Hu/(||Hu||²+epsilon)只是单秩代理；u出现在delta内存在机械相关。
新D完整保留正负贡献。A1在全部原loss token上设A=1，不继承零优势mask；abs(A)不称完全去reward。
主口径token，decision/game只作预先列出的敏感性。所有旧261列保留，新12列，共273列。
原始/中心化delta范数仅为排名对照，不把非负数值硬解释为下降方向；SIGNED原值与弃权另计。

统计复用v2固定pool及相同2000组game×continuation bootstrap、512组trajectory-block符号。
配对比较增加条件方向AUROC/Spearman，对同聚合幅度、去reward、q-only及旧D/P/C并列。
历史标签已见，只能事后探索；三聚合max对照不校正此前39公式搜索，不可改写为确认性p值。
效用锚点与gold不重采样；独立sklearn/scipy核验全分层指标，再核对原5220行旧指标。

工程测试证据：

- `C/artifacts/code_checks/realized-reward-s404-20260922-v1/initial-tests.xml`：33 PASS / 14.56s。
- 同目录`regression-tests.xml`：243 PASS / 89.63s；新增adapter恢复专项。
- 同目录`final-regression-tests.xml`：244 PASS / 94.90s；新增封存排除临时/活动文件专项。
- 同目录`archived-witness-check.json`：16 token/248320词表；PLACEBO误差3.39e-21，NULL 4.34e-19；
  仅归档样本恒等式检查，不读取gold，不是新GPU前向/效用实验。

实际命令入口（cwd=C，使用同一skillnet-vllm-20260918 Python；不是重跑指令）：

~~~bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B \
  -m skillnet_cohort.realized_reward_run prepare \
  --tests artifacts/code_checks/realized-reward-s404-20260922-v1/final-regression-tests.xml
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B \
  -m skillnet_cohort.realized_reward_run launch
~~~

后台只管理自己的worker、每轮生成不可覆盖心跳；空闲90秒仅告警，不强杀仍在工作的进程。
无总时间上限、无自动retry。磁盘仍min free100GiB、cohort max760GiB、checkpoint reserve80GiB，
另预留8GiB新输出；没有数据删除。原报告不覆盖；完整新报告与旧v2报告拼接为新的扩展副本。
前向全部结束后自动CPU聚合、统计、独立核验、provenance/complete封存；不继续505/606队列。

### Z1. 实际启动与首波验收

2026-09-22 09:29:45.886764 UTC后台启动PID1412373；v2 plan SHA
`dacb0ca075a11d743023582b858e49e5ff24ddba053bf68150f90aa3d62d3d6c`。
四个worker PID1412859/1412860/1412861/1412862，GPU pairs0,1 / 2,3 / 4,5 / 6,7。
首波前向收据时间09:37:25.940626 UTC，已完成115个决策：37/27/26/25，各分片总689。
首决策分片0为18token、分片1–3为512token；两control行数分别36/1024/1024/1024。
四分片均逐决策stable_exact=True，无Traceback/failed.json；显存约11821–20197MiB。
本收据不是实验完成或新指标更优的证据，最终以complete.json及配对统计为准。
v1没有run-intent或GPU启动；v2是唯一实际前向运行。未推送、提交、回滚、删除历史文件。

## AA. 2026-09-22：中心化幅度与reward定向因子的独立CPU验证

新增而不改冻结源码：`skillnet_cohort/factorized_reward.py`、`factorized_reward_analysis.py`、
`tests/skillnet_cohort/test_factorized_reward.py`及
`docs/experiments/phase12-independent-v4/FACTORIZED-REWARD-SEED404-20260922-v1.md`。
输入数值源仍N2/seed-404，比较源为Q/realized-reward-s404-v2；后者完整封存66项输出核验，
其actual完成时间为2026-09-22 11:04:50 UTC，不再是Z1的RUNNING历史状态。

新输出F=`Q/factorized-reward-s404-v1`。plan SHA
`f0d7a3a13bffbdc02532060730529952ca9576b2edfff78b3d581ef5be20b8e4`。
数学公式、解释和固定对照见协议；不换δ定义，H只作用于范数，b使用未中心化归一化动作log概率。
所有零A行留分母、A1覆盖全部原行、精确常数保持；无阈值搜索、门控或正部截断。
主D_factor符号在B>0时与D_action_adv相同，本次420个因子单元全部通过，epsilon弃权差异为0。

工程验收：initial-tests.xml 84 PASS/34.37s，补全端到端后final-regression-tests.xml 85 PASS/36.50s，
都位于`C/artifacts/code_checks/factorized-reward-s404-20260922-v1`。没有正式实验失败或自动重试。
以下为已执行记录，不可在原目录重跑；cwd=C、CPU环境CUDA_VISIBLE_DEVICES为空、OMP/OPENBLAS/MKL均1：

~~~bash
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B \
  -m skillnet_cohort.factorized_reward_analysis prepare \
  --tests artifacts/code_checks/factorized-reward-s404-20260922-v1/final-regression-tests.xml
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -B \
  -m skillnet_cohort.factorized_reward_analysis run
~~~

prepare于13:35:12 UTC完成；实际PID1425776，13:35:53.710开始，13:36:07.998封存全部285列，
之后才读取效用标签计算新比较（历史标签已经看过，不能称前瞻）。13:36:39.494完成，45.7858秒。
5700条指标独立sklearn/scipy核验；3660条signed方向混淆独立手算；原5460条比较指标复现。
420个因子单元独立math.fsum核验，B最大误差2.27e-13、D_factor最大2.84e-13；
与旧B/动作投影分子的最大浮点差1.14e-12，在冻结相对/绝对容差内，不更改任何旧分数。
137份旧源码、全部76项新计划绑定输入运行前后不变，新来源总141份。
完成后再次核验24项新封存文件；provenance SHA
`44ed26f77127c47b5691be22dd6322b125367c6e0368c71139af0a20a45ebb50`。

统计共18技能（5下降7上升6零点标签），方向AUC样本12。
主D_factor AUC=.6、rho=.101962、AP=.294949；B为.742857/.348982/.613651；
−R为.6/.097757/.294949。新组合与−R的AUC/AP相同但不是完整排序逐项相同。
新组合8/12准确、balanced=.6，下降只对1/5、上升对7/7；恒预测上升为7/12、balanced=.5。
top5新组合1/5下降，B为3/5下降。方向判断不变符合符号恒等式。
相对A1的AUC差+.171429但区间[-.566667,.583333]，相对B差−.142857、区间[-.7,.439236]。
512个trajectory-block符号对照31.6406%AUC不低于主结果，三聚合max参考44.5313%，不是校准p值。
所有all/0pp的control×聚合口径中新组合AUC点估计均低于同聚合B；不推广到全部阶段或所有reward方法。

完整自动报告F/reports/factorized-reward-analysis.md，扩展副本phase2-results-expanded.md不覆盖任何旧正文。
另新增R/2026-09-22-seed404-factorized-reward-validation.md为人读解释与入口；科学状态ANALYZED。
未启动/恢复505/606、RL、模型前向、环境效用、API或Phase3；未提交、回滚、推送或删除数据。

## AB. 2026-09-22：seed404八组gold效用精度扩展

新增代码为`skillnet_cohort/utility_precision.py`（严格扩展协议/复用/哈希准入）、
`utility_precision_run.py`（八卡FileStore worker/心跳/自动汇总）、
`utility_precision_analysis.py`（固定285读出的2/4/8及新增6组敏感性、重复分歧、配对bootstrap）。
仅增加gold重复，不修改run_branch、vLLM采样、router或旧数学定义；全部旧实现141份哈希保持。
新目录G=`Q/utility-precision-s404-v1`，设置与风险登记见同名20260922-v1协议文档。

新增gold为404、404100、404200、404300、404400、404500；保留63011、63021，evidence62011仍独立。
每个anchor×seed形成U0/U5×O/P/N六元配对；新base+step区间不重叠，原两组交叠的历史限制披露。
所有404锚点都增加同样6组，主结果8组；不按标签/读出表现决定采样量或停止。不是新的RL种子重复。
原每端点3636条逐一核验后复制到新分片，trajectory_id保留，trajectory_path指向新副本并附来源hash。
旧bank、模型、anchor、环境seed、evidence、285指标及candidate pool保持。新增每端点7272条，不重复旧轨迹。

CPU测试均在CUDA_VISIBLE_DEVICES为空，OMP/OPENBLAS/MKL=1的skillnet-vllm-20260918 Python中执行。
证据在C/artifacts/code_checks/utility-precision-s404-20260922-v1：initial-tests.xml31 PASS，
regression-tests.xml405 PASS/83.50s，final-regression-tests.xml407 PASS/72.02s，
final-protocol-tests.xml35 PASS/1.74s（含最后的历史统计守卫专项）。这些测试集有重叠，不能相加为独立测试数。
G/analysis-preflight.json是实际旧7272条记录上的CPU接口验收：176个效用点估计/区间复现，
5700行排名全部复现并独立sklearn/scipy核验，candidate pools完全一致。没有GPU预试重复或付费调用。

实际prepare命令（已执行一次，不可重复覆盖；cwd=C，CPU环境同上）：

~~~bash
/mnt/workspace/users/wangyifan/.venvs/skillnet-vllm-20260918/bin/python -u -B \
  -m skillnet_cohort.utility_precision_run prepare \
  --tests artifacts/code_checks/utility-precision-s404-20260922-v1/final-protocol-tests.xml
~~~

plan SHA=`81c7213909a3f884401d5aae016674d9af450b8a8219ac63d41fa195b45acc19`；
146源码、74核心输入，retained-inputs.json有7304条旧轨迹/索引绑定。
总21816续跑，7272复用，14544新增。初始磁盘free319.086GiB、cohort238.531GiB；
新增预留64.8125GiB加80GiB检查点保留量，共144.8125GiB。最低free100GiB、cohort760GiB仍执行。
原大模型/optimizer/概率、报告及全部未提交修改未删除；本环境没有git可执行程序，未安装或使用git变更命令。

launch为一次性`python -u -B -m skillnet_cohort.utility_precision_run launch`，本节写入时尚在容量准入。
实际PID/八卡新轨迹验收追加于AB1。后续成功以G/complete.json和provenance为准，不用旧目录complete当新结果。
G/logs/u0000-shard*.log、u0005-shard*.log记录逐条新增进度；workflow.log记录父队列。
失败写stopped.json并保留证据，不自动重试；只管理自身新建进程组，不操作505/606或外部任务。
全部完成后生成独立G/reports/phase1-results.md与phase2-results.md，不发布覆盖任何旧正文。

### AB1. 实际启动、八卡首批新轨迹验收

G/launch.json记录父PID1446810，start_ticks486001965，启动UTC2026-09-22 15:54:44；
command SHA=`c710f43a8c24dfe528a47f0fa154efa6ecc770ccdc478075a664c021dc70cd51`。
G/stages/utility-u0000.json记录八个worker1447876/1447877/1447878/1447879/1447880/1447881/1447882/1447883；
G/ready/u0000-shard*.json全8份，rendezvous为独立FileStore，不使用此前冲突的TCP探测端口。

16:03:38.106772 UTC首批验收：各片新增5/5/6/6/7/6/6/6条，共47/7272 U0新增续跑。
首条新trajectory的身份、continuation_seed、actual_continuation_seed、prefix_replay_verified、
全部ANCHOR_SEMANTIC_FIELDS逐项核验；文件SHA、worker身份、ready收据SHA见G/first-wave-acceptance.json。
每卡15217MiB，8片均有真实新轨迹，不只是模型加载成功。router本地attempt状态success774、started2、failed0；
started为现场正在编码的正常记录，不是新的中断恢复授权。外部API仍0，146源码/74输入hash复核通过。

当前科学状态RUNNING_NOT_COMPLETE，不能引用47条中间样本得出新效用/排名结论。
旧U5三组记录已经复制备查，但U5新增六组还未启动；等待全部U0新轨迹后自动推进。
启动验收后结束人工持续轮询；后台心跳及容量保护继续。metadata密集的cohort磁盘扫描可能延长心跳间隔，
只看某次GPU0%或日志短暂无增长不能判定崩溃。错误看stopped.json与各worker日志，无自动retry。
