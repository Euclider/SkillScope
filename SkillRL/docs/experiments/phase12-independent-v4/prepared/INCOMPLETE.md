## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: run
- Origin Date: 2026-09-18
- Verification Status: UNVERIFIED
- Version Label: independent_offline_preparation_incomplete_v1

仅离线资产准备，未启动模型、环境、Ray、RL或API。三个preparation/admission及sampling-precheck已写出，
最后生成共同cohort授权清单时，源码哈希列表误用了不存在的 `verl/utils/checkpoint/native_restore.py`。
实际恢复helper为 `skillnet_cohort/native_restore.py`，已经包含在目录哈希列表中。
本目录不完整（没有cohort.json），不得执行，不覆盖或删除。
修正清单路径后仅在新 `prepared-v2` 生成离线资产；这不是重跑失败实验。
