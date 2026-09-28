## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-23T15:36:26.878460+00:00
- Verification Status: UNVERIFIED
- Version Label: utility_precision_seed404_v1

# Phase2：冻结读出与提高精度后的效用标签

这是已见404/505旧标签后的精度扩展，不是新的独立RL seed或确认性检验。固定U0/U5、37技能库、0.6B逐状态top1路由、全部404个首次调用锚点及全部285列读出；原7,272条续跑复用，新增14,544条。仅gold重复数2→8，evidence仍独立且不用于gold标签。训练、seen/unseen完整成功率、读出模型前向不重跑。

所有arm/endpoint按相同anchor和续跑seed配对；环境seed和原前缀不变。seed编号等于404不代表更准确；新base间隔100避免新重复之间的50步seed区间重叠。旧63011/63021的base+step区间重叠是保留实现的边界，新增6组单独分析同时报告。

主结果固定8组；2/4/8曲线及added6-only均固定输出，不按结果选用哪组。点估计等权game；区间配对game/continuation重采样，保留单game区间NA。原点标签指标不删除；区间跨0标为方向未分辨，零宽区间不能视为充分精度。bootstrap比例不是后验真值概率；不确定性未覆盖训练随机性、状态分布差异或多公式选择。

| score                              |   candidates |   declines |   increases |   average_precision |   auroc_decline_vs_increase |   auroc_decline_vs_rest |    spearman |
|:-----------------------------------|-------------:|-----------:|------------:|--------------------:|----------------------------:|------------------------:|------------:|
| D_original::token::reward          |           18 |          7 |           7 |            0.591041 |                    0.44898  |                0.597403 |  0.00311208 |
| D_signed::token::reward            |           18 |          7 |           7 |            0.502406 |                    0.44898  |                0.532468 | -0.080914   |
| D_signed_gate::token::reward       |           18 |          7 |           7 |            0.515254 |                    0.44898  |                0.558442 | -0.116184   |
| C_centered::token::reward          |           18 |          7 |           7 |            0.535196 |                    0.489796 |                0.545455 | -0.00622415 |
| M_delta_centered::token::magnitude |           18 |          7 |           7 |            0.561369 |                    0.734694 |                0.649351 |  0.353739   |
| D_real::token::reward              |           18 |          7 |           7 |            0.426706 |                    0.571429 |                0.506494 |  0.0363075  |
| D_factor::token::reward            |           18 |          7 |           7 |            0.3484   |                    0.428571 |                0.350649 | -0.0643162  |
| D_orientation::token::reward       |           18 |          7 |           7 |            0.3484   |                    0.428571 |                0.350649 | -0.0663909  |

完整285指标及去reward对照、全部phase/control/0及5pp口径见CSV。历史公式搜索全部保留，不能从更新标签中选择赢家后称为预登记确认。配对指标差区间见paired_readout_gains.csv，不用两个边际区间重叠与否替代差值检验。

## 统计解释检查

覆盖11/11类：Simpson（保留分层）、生态谬误（不外推token/RL seed）、Berkson（共同池及完整NA覆盖）、collider（不按结果筛选）、基率（记录下降/上升数）、均值回归（不选极端技能追加）、幸存者偏差（必须全部完成）、多重检验（不宣布显著赢家）、分析路径（事后扩展披露）、相关/因果（不是编辑收益）、反向因果（冻结读出先于新标签）。
