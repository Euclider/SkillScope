# Seed404：中心化交互幅度 × reward 定向因子

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run / validate
- Origin Date: 2026-09-22
- Verification Status: UNVERIFIED（运行前冻结；结果另写新目录）
- Version Label: factorized_centered_reward_v1
- 用户授权：在上一轮明确提出的单一分解方案上开始验证。

## 问题与科学边界

检验在保留中心化交互幅度的基础上，用已归档 reward/advantage 定向，能否改善技能效用变化方向的排序。
seed404 的历史标签和此前273列结果已知；这是事后探索，不是前瞻预注册或跨 seed 确认。
不依据此次结果改符号、阈值、聚合、技能池或公式，不自动启动后续变体。

固定状态、旧动作：delta = (log p5_skill - log p0_skill) - (log p5_control - log p0_control)。
H为每个预测位置的全词表均值消除，而不是跨技能中心化。

对于固定非负权重 w（总和为1）：

```
b_j = A_j * delta_j(a_j)
B_s = sum_j w_j * ||H delta_j||
R_s = sum_j w_j*b_j / (sum_j w_j*abs(b_j) + 1e-12)
D_factor(s) = -B_s * R_s
D_orientation(s) = -R_s
```

定向分子使用未中心化的、归一化动作 log 概率交互；H只用于B。
两种reward-free对照把所有原行的A设为1，包括原A=0行，重新计算R及D。
不增加abs(A)变式、阈值扫描、门控、正部截断或学习方向分类器。
零advantage行留在全部分母；精确恒定输入归约后仍保持精确恒定。

R是正负reward交互证据的净比例，不是校准置信度。A不是逐步因果credit。
B不是已校准的|效用变化|预测值。B>0时D_factor符号必与D_action_adv=-Agg(b)一致；
它能调整排序，但不能凭空修正后者的符号。B=0退化为零；数值epsilon弃权另行记录。
逐token的 ||Hdelta|| * cos(d,Hdelta) 本质回到原P，本次不是该恒等式的改名。

## 固定对照与统计

- 主分析：PLACEBO / all / token / 0pp，D_factor。
- 主排序指标：下降vs上升AUROC、Spearman(-delta_utility)。
- 方向判别：balanced accuracy（弃权计错）、原始准确率、coverage、混淆计数、恒预测类别基线。
- 次要：下降vs其余AP/AUROC、top-k命中与下降量覆盖；另报告B与|delta_utility|的Spearman。
- 既有273列全部保留；新D_factor、-R及各自A=1对照乘3种聚合，共12列，最终285列。
- 同聚合B、-R、A1、D_action_adv、旧D、-P、C_centered、q-only、D_real均列为比较对象；不挑赢家。
- token主口径不变；decision及game逐级等权只作敏感性，不根据标签选聚合。
- NULL、initial/early/middle/late阶段、5pp标签阈值均为敏感性，与历史固定池一致。
- 候选池、首次自然调用anchor、效用endpoint、gold seeds、标签全部复用，不新增筛选。
- 复用已有2000组paired game×continuation gold-label bootstrap；整池缺技能的draw标NA，不能按方法各自删技能。
- bootstrap只覆盖固定读出下的标签不确定性，不覆盖训练seed/读出抽样不确定性，不校正历史探索。
- 复用512组trajectory-block +/-1符号对照；相同轨迹内A整体翻号，幅度、零行、同轨迹A变动形状不变。
- 报告每个候选的随机参考分位数及三聚合max参考；不把参考比例称为校准p值。
- 方向AUC的提高与balanced accuracy提高分别报告；新组合符号恒等并不构成新的方向信用信息。

## 执行与保全

仅CPU，现有skillnet-vllm-20260918 Python，所有数学运算float64。
输出新目录 `artifacts/phase12/skillnet37-independent-s404-505-606-v4/factorized-reward-s404-v1`。
模块：`skillnet_cohort.factorized_reward_analysis`，先prepare后run；命令和环境写plan。
至少20项无skip合成CPU测试通过才prepare；真实分数先提交哈希，再打开效用标签。
对既有273列逐值保留，独立sklearn/scipy复核全部285列的分层指标，并手算方向混淆。
新增因子标量用独立math.fsum路径核验；与旧B/D_action_adv分子对照。
原137绑定代码、原输入和封存报告执行前后核验SHA；不编辑冻结代码。
原报告不覆盖，新报告单独发布；不提交、回滚、推送或删除任何历史材料。
不运行RL、模型前向、环境rollout、API、505/606或Phase3，不加载模型/OLD全词表张量。
记录PID、耗时、阶段进度及结束状态；沿用用户取消总预算限制的决定，无自动硬超时。
运行失败保留failed收据并停止，不自动重试。合成开发测试与真实实验失败分别记录。
