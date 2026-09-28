## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-20T10:39:04.480771+00:00
- Verification Status: UNVERIFIED
- Version Label: phase12_readout_seed_v4

# Phase2：seed 404，U0→U5 预测性

实际第一轮训练 batch 提供固定状态、动作 token、mask 和 advantage；U0 与 U5 在相同输入上读出。gated D 为主，−P、C 及同源幅度指标为预登记诊断；不按 gold 选方向或窗口。readout 无额外环境 rollout，但验证标签来自独立 O/P/N 续跑，两者成本不混淆。

| context_id   | phase   | score                  |   threshold |   candidates |   declines |   average_precision |   auroc_decline_vs_rest |   spearman |    kendall |   P_sign_scored_units |   P_sign_agreement_on_nonzero_point_delta |   seed |
|:-------------|:--------|:-----------------------|------------:|-------------:|-----------:|--------------------:|------------------------:|-----------:|-----------:|----------------------:|------------------------------------------:|-------:|
| all_alfworld | all     | C_upd                  |           0 |            3 |          1 |            1        |                     1   |        0.5 |   0.333333 |                   nan |                                       nan |    404 |
| all_alfworld | all     | D_contribution         |           0 |            3 |          1 |            0.333333 |                     0   |       -1   |  -1        |                   nan |                                       nan |    404 |
| all_alfworld | all     | D_ungated_contribution |           0 |            3 |          1 |            0.333333 |                     0   |       -1   |  -1        |                   nan |                                       nan |    404 |
| all_alfworld | all     | P_int                  |           0 |            3 |          1 |            0.333333 |                     0   |       -0.5 |  -0.333333 |                     2 |                                         0 |    404 |
| all_alfworld | all     | activation_l16_norm    |           0 |            3 |          1 |            1        |                     1   |        0.5 |   0.333333 |                   nan |                                       nan |    404 |
| all_alfworld | all     | activation_l24_norm    |           0 |            3 |          1 |            1        |                     1   |        0.5 |   0.333333 |                   nan |                                       nan |    404 |
| all_alfworld | all     | activation_l32_norm    |           0 |            3 |          1 |            1        |                     1   |        0.5 |   0.333333 |                   nan |                                       nan |    404 |
| all_alfworld | all     | activation_l8_norm     |           0 |            3 |          1 |            1        |                     1   |        0.5 |   0.333333 |                   nan |                                       nan |    404 |
| all_alfworld | all     | delta_centered_norm    |           0 |            3 |          1 |            0.5      |                     0.5 |       -0.5 |  -0.333333 |                   nan |                                       nan |    404 |
| all_alfworld | all     | delta_norm             |           0 |            3 |          1 |            0.5      |                     0.5 |       -0.5 |  -0.333333 |                   nan |                                       nan |    404 |
| all_alfworld | all     | forward_kl_original    |           0 |            3 |          1 |            1        |                     1   |        1   |   1        |                   nan |                                       nan |    404 |
| all_alfworld | all     | js_original            |           0 |            3 |          1 |            1        |                     1   |        0.5 |   0.333333 |                   nan |                                       nan |    404 |
| all_alfworld | all     | old_margin             |           0 |            3 |          1 |            0.5      |                     0.5 |       -0.5 |  -0.333333 |                   nan |                                       nan |    404 |
| all_alfworld | all     | u_control_norm         |           0 |            3 |          1 |            0.333333 |                     0   |       -0.5 |  -0.333333 |                   nan |                                       nan |    404 |
| all_alfworld | all     | u_original_norm        |           0 |            3 |          1 |            0.333333 |                     0   |       -1   |  -1        |                   nan |                                       nan |    404 |
| all_alfworld | all     | random_expected        |           0 |            3 |          1 |            0.333333 |                     0.5 |      nan   | nan        |                   nan |                                       nan |    404 |

各分数使用相同的自然支持候选池；缺支持为弃权，零下降事件时 AP/Recall 不定义。D 是单侧下降风险，不能把 D=0 解释为稳定或上升；−P 的排序关联也不等于方向分类准确率。完整 top-k/比例预算、5pp 阈值、NULL、分阶段敏感性见 CSV/window 报告。一个 seed 不支持跨 seed 泛化结论，所有阶段分层和 continuation repeats 不是独立 RL seed。


恢复说明：seed404 在首轮 rollout 完成、首次 optimizer.step 之前发生进度文件扫描竞态。原128条轨迹及动作token被复用，未重新采样；已有OLD概率与恢复后的前向结果逐bit核对。vLLM/worker随机状态在登记seed上重新初始化，不声称与不中断运行逐bit等价。首轮遗失的vLLM采样logprob仅用于后端差异诊断，未伪造；训练使用重新完成的native OLD。累计30小时包含首次失败运行，扣除故障停机；详见recovery-v1及其保全证据。

第二次恢复补充：首次恢复已完成5512行native OLD和reference前向，随后Python bool/NumPy标量兼容错误在首次优化前中断。本次复用全部已存的实际trainer-chosen OLD概率；每rank一个原状态经native前向精确核验，并验证所有有效token在恢复native dtype后保持精确。reference因未持久化而重算。首轮OLD entropy日志缺失，未伪造；优化器内entropy正则及其余RL参数完全保留。U1没有重新采样环境轨迹；累计30小时计入此前两个attempt。详见recovery-v2。

第三次恢复补充：五轮RL、640条训练轨迹及每rank的204次Adam更新已完成，仅训练后导出子进程因本地verl导入路径中断。以模块入口修复并直接导出原U5，没有重跑训练、重采样训练轨迹或改变原生检查点。新导出在发布前与八rank原生FP32分片逐张量、逐bit核验；旧失败目录及记录保留。累计30小时计入此前所有实际运行，故障停机不计；详见recovery-v3。

运行时限修订：用户在2026-09-20明确取消三个seed累计时间上限；以上恢复说明中的30小时为历史规则，不再约束本次接续。磁盘保护、科学设置、404→505→606顺序不变，仍记录累计实际运行时间。
第四次恢复仅修复窗口汇总对逐轮NEW概率的过时依赖，复用原5轮RL、U5导出、U0全量评估/anchors/540条续跑及8个读出分片。窗口协议和C/P/D定义未改；起点OLD为实际训练概率，终点为相同输入的U5前向。U0 HF快照物理权重为BF16，不能把元数据的FP32当成原生master精度；参数范数诊断使用已登记原模型按原生加载步骤在CPU重建的FP32 B0，不将有舍入的U0导出上转后冒充原生master。无额外RL或环境rollout。原U0 BF16运行值核对为PASS，FP32导出一致性检查为FAIL，两者证据均保留。
