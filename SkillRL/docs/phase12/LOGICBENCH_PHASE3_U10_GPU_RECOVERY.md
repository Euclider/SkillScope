# LogicBench Phase3 U10 恢复与 GPU 等待（2026-10-01）

本次恢复保留 seed 707、D_sign_balance、50 次 RL 更新、每 5 次封存窗口与快照的已批准设置，继续使用登记的 RL speed v1 后端。

## 中断位置与修复

- 已完成 U1–U10 训练、U10 HF 导出、readout 和候选选择，未进入 U11。
- 中断于 U10 editor 的本地 token 计数：tiktoken 缺少 o200k_base 缓存，经继承代理下载失败；尚未发送 U10 编辑请求。
- o200k_base 已放到 `/home/wangyifan/.cache/skillrl-tiktoken`，校验官方 SHA256 `446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d`，并验证可断网加载。
- 恢复进程指定 `TIKTOKEN_CACHE_DIR` 并移除其继承的 HTTP/HTTPS/ALL/NO proxy 环境变量（含大小写变体）；不改变其他进程或系统设置。
- 编辑器沿用已批准的 600 秒 timeout、SDK 自动重试 0、原调用预算。队列不会自动重跑失败的训练或编辑请求。
- 按此前授权清理旧 ALF seed303 U29 checkpoint，约 49.39 GiB；文件清单及授权依据留在 run root 下 `recovery/legacy-alf-u29-cleanup-20261001.json`。LogicBench U5/U9/U10 恢复点保留。

## 后台队列

脚本：`scripts/watch_logicbench_phase3_resume.py`，以训练环境 Python 的 `-m scripts.watch_logicbench_phase3_resume` 方式执行。

- 从现有 `launch.json` 读取固定物理 GPU 1、2、3、4，不迁移原运行布局。
- 每 60 秒检查一次；四卡均无 compute PID、显存占用不超过 2048 MiB、利用率不超过 10%，连续两次通过后才启动。
- 启动前重新核验冻结源码/依赖/恢复凭证、tokenizer 缓存、磁盘预算和 GPU 状态。
- GPU 查询只是观察，并非集群级资源预约；最终训练 preflight 仍会检查占用，发生竞争则停止。
- 独立 flock 防止重复监控实例，锁描述符传给启动子进程。
- 通过 `scripts/run_logicbench_phase3_fast.sh --setting <root>/setting.json --root <root> --gpus 1,2,3,4 --execute` 恢复。现有端点和窗口文件使其跳过已完成训练/readout，从 U10 编辑/gate 继续。
- 运行失败只记录并退出，不自动重启。

运行目录：`artifacts/logicbench/phase3-dsign-s707-u50-cpu-v2`。

队列状态与日志在该目录的 `recovery/gpu-resume-queue/`：

- `status.json`：最近 GPU/磁盘观察、watcher PID、等待/运行/结束状态。
- `history.jsonl`：轮询和启动记录。
- `watcher.log`：监控器自身日志。
- `pipeline.log`：恢复启动器输出；具体阶段日志仍在原运行目录。

## 验证范围

新增测试覆盖：原卡被占用时不启动、连续空闲判定及重置、探测失败、磁盘不足、代理环境处理、重复队列锁、子进程失败后不重试。另运行现有 GPU queue、RL speed runtime、timeout recovery 和 LogicBench loop 测试。`--check-only` 在真实目录验证冻结恢复输入、缓存与磁盘，不发送编辑请求、不启动训练。
