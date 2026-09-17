# SkillScope

Skill-RL 的研究源码与实验分析快照，用于在新的 RTX 5090 服务器上重建环境、下载基础模型和数据，并追溯已有结论。

## 开始使用

在 5090 上将仓库放入用户目录：

```bash
git clone https://github.com/Euclider/SkillScope.git /mnt/workspace/users/wangyifan/skill-RL
cd /mnt/workspace/users/wangyifan/skill-RL
python3 deploy/5090/verify_project.py
```

然后按 [SETUP_5090.md](SETUP_5090.md) 创建环境、下载固定 revision 的 Qwen3.5-4B 与 ALFWorld 文本数据，并运行部署检查。说明中的安装方案尚未在目标 5090 验证；新训练需另设配置与输出目录。

## 研究与历史结果

- [交接说明](HANDOFF.md)与[技术附录](HANDOFF-DETAILS.md)。
- [Phase1 三 seed 效用结果](2026-09-04-qwen35-clean-all-skill-three-seed-utility-results.md)。
- [Phase2 synthesis-v2 完整分析](2026-09-14-phase2-complete-analysis.md)：报告及 11 个修订输入哈希已核验。
- [源码](SkillRL/)、[冻结配置](SkillRL/phase1/config/)、[Phase2 实现](SkillRL/phase2/)。
- [原工作区状态](deploy/5090/source-git-status.txt)、[未提交 tracked patch](deploy/5090/source-tracked.patch)与[原文件哈希清单](deploy/5090/project-original-files.jsonl)。

保留协议、统计 CSV/Parquet、排序审计、修订记录和完成记录。不包含基础模型、已安装环境、ALFWorld 原始下载、RL checkpoint、optimizer、完整逐轨迹记录或 token 张量；本仓库支持报告及统计输入核对，不能恢复或完整重放旧实验。

非日期文件 `phase2-complete-analysis.md` 与 v3 修订记录的报告哈希不一致，已原样保留并在部署说明中标记，未改写为“验证通过”。历史绝对路径和报告文字均保持原样。

## 来源

源代码基于 SkillRL/verl，保留了原文件中的版权声明和 [LICENSE](SkillRL/LICENSE)。源工作区 HEAD 为 `8e66726ed866a4e0a7f053586a41022798192e6c`；现有未提交源码内容已包含在此快照，原研究仓库没有被提交、回滚或清理。
