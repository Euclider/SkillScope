# SkillNet-37：ALFWorld 冻结初始库

状态：**bank 已构建，外部 router 与实验启动配置未完成**。此目录是独立、显式加载的资源，不改变历史 SkillRL bank、环境默认配置或已有报告。

## 来源与冻结身份

来源为 [SkillNet 固定 commit](https://github.com/zjunlp/SkillNet/tree/5c472b36d2a435001fdae3bc8439886d8050645a/experiments/src/skills/alfworld)，取完整 `experiments/src/skills/alfworld` 和仓库根 MIT `LICENSE`，不取运行器作为本项目执行入口。

| 项目 | 冻结值 |
|---|---|
| Bank ID | `skillnet-alfworld-37-5c472b36d2a4` |
| 上游 commit | `5c472b36d2a435001fdae3bc8439886d8050645a` |
| 文件 | 37 个 `SKILL.md` + 45 个附属文件 + 1 个 `LICENSE` |
| 原始文件总大小 | 152,902 bytes，含许可证 |
| manifest SHA-256 | `0767ff7578b1e997119a36b5f636fec6ebc1d0ac600ffb40b1e599065b7fe514` |
| 文件清单内容 SHA-256 | `a3fdd5f265f4a9926533686aef315163a6fb6b6f6ba5fc3f2153a9240dc662ab` |

`manifest.json` 记录下载归档的 URL/hash、每个原文件的路径/字节数/hash、每个技能的稳定 ID/原始描述/文件清单，以及渲染后的 payload hash。文件清单内容 hash 的输入为按上游相对路径排序的 `"{sha256}  {path}\n"`，不是 tar 文件或拼接正文的 hash。manifest 的外部固定 hash 在加载器和 `setting.json` 中另行锁定。

`upstream/` 下文件与固定 commit 逐字节一致，包括空白与无末尾换行的文件。不纠正文档、不去重、不补写技能、不合并 SkillRL-44。MIT 许可证原样保留；仓库来源不等于逐条轨迹的数据泄漏审计已经完成。

## 候选库与正文注入分开

- 六类 ALFWorld 任务使用同一组 37 个候选 ID，不按 task/game 硬过滤。task 类型仍可用于后续分层统计；这不意味着合并 train/validation/test 数据。
- 候选目录只给出 `skill_id`、原始 `name` 和 `description`，稳定按技能名排序。库本身不计算分数、不选择技能、不调用 API。
- 由未来独立 router 每步选择至多一个 ID，再加载该技能的完整文本包。不是把 37 份正文同时注入 policy。
- 渲染器 `skillrl.raw_skill_package.v1` 添加一个稳定技能标题，随后原样拼入 `SKILL.md`，再按相对路径排序拼入全部附属文件；只增加文件分隔标题，不改任何原正文。许可证不作为技能正文注入。
- 附属文件只作为文本，不执行；技能内的多步示例不自动变成环境宏动作。
- `selected_bundle(None)` 仅提供空选择的表示，尚未决定在线 router 是否允许弃权及其规则。
- ORIGINAL/PLACEBO/NULL 在选择后作用于 payload；不得删除候选 ID 或依据干预臂重路由。适配器通过已有 `phase1.first_invocation` 的载荷传递测试，不代表完整实验已接通。

完整包大小为 1,619–7,653 UTF-8 bytes，**不是 token 数**。尚未用 Qwen3.5 tokenizer 审核完整 prompt 的长度，也未验证旧 PLACEBO 固定文本流能匹配所有完整技能包。不能直接沿用旧 2,048-token prompt 预算后宣称没有截断。

## 离线使用

在代码根 `SkillRL/` 执行只读校验（不访问网络、不加载模型、不调用环境）：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -B -m agent_system.memory.frozen_skill_bank
```

显式加载示例；这里的 ID 是手工示范，不是 router 的选择结果：

```python
from agent_system.memory.frozen_skill_bank import (
    FrozenSkillBankMemory,
    load_skillnet37,
)

bank = load_skillnet37()
memory = FrozenSkillBankMemory(bank)
catalog = bank.router_catalog()  # 37 个原始描述，供未来 router 使用。
candidates = memory.retrieve("put a clean mug in a cabinet")
assert len(candidates["candidate_skill_ids"]) == 37
selected = memory.selected_bundle("skillnet:alfworld-clean-object")
payload = memory.format_for_prompt(selected)
```

`format_for_prompt(memory.retrieve(...))` 会报错，要求先完成单技能选择。`top_k < 37`、任务过滤参数、增加/删除/禁用/保存技能等更新接口均拒绝。读取时发现 manifest、文件集合、原文件字节或渲染载荷与固定身份不符即失败，不自动修复或回退旧库。

“只读/冻结”由固定 hash、加载校验、不可变技能对象及拒绝更新的 API 实现；不是修改文件系统权限，也不能阻止拥有写权限的人修改磁盘。磁盘修改会在下次校验/加载时被拒绝；已加载正文保存在不可变 bytes/string 中。

## 边界与后续

`setting.json` 是机器可读的 **bank-only 决策记录，不是可启动实验的配置**。旧 `env_manager`/launchers 不会自动改用此库。仍需确定并接入外部 router 的服务、模型快照、prompt/可见历史、解码、缓存与失败策略，完成 tokenizer/PLACEBO 审核，然后另行冻结新 seed、全任务 split/game 清单、窗口和预算。

复用同一库能控制初始技能资源差异，但不能据此声称完整复现 SkillNet 运行方式、保证公平性已全部满足、保证 37 个技能都有自然支持，或保证效果更好。路由覆盖应自然测量，不强制轮换技能。

详细决策、出处、论文附录草稿和验收边界见研究根 [设置留档](../../../../2026-09-17-skillnet37-frozen-bank-setting.md)；本次离线验证记录见 [verification.json](../../../docs/experiments/skillnet37/verification.json)。
