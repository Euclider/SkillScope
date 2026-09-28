# 发布与启动回执索引

Phase3已发布至 [Euclider/SkillScope-phase3](https://github.com/Euclider/SkillScope-phase3)，commit `47e4435a9989c88157f2e11f374a28a5e7530670`。865个远端blob与新导出目录逐项一致；未操作原工作区Git index，未提交或回滚原工作区文件。

- 启动说明：[仓库README](https://github.com/Euclider/SkillScope-phase3/blob/main/README.md)。
- 导出目录：`/mnt/workspace/users/wangyifan/SkillScope-phase3-export-20260918-v3`。
- 压缩包：`/mnt/workspace/users/wangyifan/SkillScope-phase3-20260918-v3.tar.gz`。
- 压缩包SHA-256：`426ad129a3fa5ad32d7bca849dd695b30c605d9677a364008c1c647b9bb3c801`。
- 发布回执：`/mnt/workspace/users/wangyifan/SkillScope-phase3-upload-20260918-v3.json`。
- 导出目录独立验证：core import确实来自export；SkillNet37哈希、全部文件hash/secret扫描、Python语法检查通过；可迁移测试275 passed in 10.23s。本地广泛回归541通过。
- 当前独立Python环境179包依赖检查通过；8卡合成原生验收证据见前一份设置留档。

本机Phase1–2新运行在确认发布后启动。run root：

`/mnt/workspace/users/wangyifan/skill-RL/SkillRL/artifacts/phase12/skillnet37-qwen35-s404-8gpu-20260918-v1`

启动时supervisor PID `839144`。这是启动回执，不声明实验完成；实时状态以进程及run目录为准。B0导出已完成，首段为u0→u5、8卡、seed404。`authorization.json`绑定原assets-v2、GitHub发布回执和100元router记账cap；密钥只经隐藏交互输入传入运行时环境，无key文件。

查看 `logs/train-u0000-u0005.log`、`metrics/`、`completed_windows/`、`router-cost-profile.sqlite3`；若出现 `stopped.json` 即未完成停止，不能重启同一block或提高预算来绕过。只有 `complete.json` 表示U150及预声明评价全部完成。

预算、磁盘和安全删除边界见 `2026-09-18-phase3-confirmed-setting-and-readout-plan.md`。当前未删除任何实验张量；没有再生证明的中间raw rows继续保留。未执行本机四条Phase3 RL。

启动观察补充：8个GPU worker均初始化完成，真实mini router请求已成功返回并写入共享费用账本；此观察仍不代表已完成一个RL更新。原有FLOPs估算器不支持Qwen3.5，日志MFU=0是未支持占位，不能作为真实硬件利用率或论文结果；应使用实际GPU监测、elapsed time与token计数，或另行验证估算器后再报告MFU。
