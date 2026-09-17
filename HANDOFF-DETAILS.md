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
