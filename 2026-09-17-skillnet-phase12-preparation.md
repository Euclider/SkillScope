# SkillNet-37 Phase1–2 准备修复与 seed404 登记

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: plan，含用户授权的代码修复和离线回归，不是实验 run
- Origin Date: 2026-09-17
- Verification Status: UNVERIFIED（真实训练、模型前向及环境评估尚未执行；软件核对另见 verification.json）
- Version Label: skillnet-phase12-preparation-v1，生效离线资产为 assets-v2

## 1. 实际停点和授权边界

用户确认新 cohort 采用 **seed=404、每5轮验证、固定5轮读出窗口**。训练沿用 SkillRL 公共脚本的采样设置，不以遍历训练池作为每轮完成条件，不根据性能挑窗口。本轮只修复准备问题，不启动 RL。旧 seeds101/202/303 与 synthesis-v2 属于已完成历史证据，不补跑、不重生成报告。

当前停在：独立训练环境、数据/模型指纹、37个控制文本、新128轨迹入口、全游戏性能评估器、自然锚点与窗口适配、容量/执行闸门及离线测试。没有产生 seed404 的真实性能、自然调用支持度、权重更新或 Phase2 预测效度结果。

代码/静态核对成功不等于四卡可运行，也不等于获准执行。尚需单独确定 API 总调用预算、存储预算、GPU 分配以及真实四卡前后向/保存恢复验收。Phase3 不是本轮开跑前置条件；之后仍允许共同外部编辑器新增、删除和修改，初始库同为37个。

## 2. 生效资产与旧文件保护

研究根目录现为 `/mnt/workspace/users/wangyifan/skill-RL`，代码目录为其下 `SkillRL/`。不改写历史文档中的旧主机路径。

- 生效准备清单：`SkillRL/docs/experiments/skillnet-phase12-preparation-v1/assets-v2/manifest.json`。
- 原始 SkillNet-37 bank 与 GPT-5.4-mini 请求协议未改，仍用原冻结 manifest/profile。
- `assets/` 是本轮较早生成、从未执行的准备版本，保留但不采用：只核对 payload 单独长度，拼接 prompt 时多1个 token。`assets-v2/` 修复了 BPE 边界；当前加载器拒绝旧版本。不是重跑旧实验，也不是覆盖旧报告。
- 本轮保护性目录哈希覆盖20,268个既有文件；核对前后相同。例外仅为预先登记的7个既有代码文件增量修改；完整排除项、前后 SHA256 见 `verification.json`。既有 rollout_loop.py 未修改。
- 旧公开脚本对齐 YAML/JSON、bank、router、报告、轨迹、权重以及其他既有未提交文件均保留。没有 git commit、reset、checkout、clean 或旧数据清理。

## 3. 固定设置与单位

| 项目 | 新 cohort 设置与解释 |
|---|---|
| B0 | 本地 Qwen3.5-4B 初始后训练模型；权重、config、tokenizer及chat template逐文件指纹，不取旧RL checkpoint |
| 初始库/路由 | 统一 SkillNet-37；六种任务共用完整候选库；独立 GPT-5.4-mini 每状态选1个；policy不决定路由 |
| RL | GRPO，学习率1e-6；16组×8轨迹=128条/全局迭代；最多150轮 |
| 优化 minibatch | 128个展开后的有效决策行，不是128条完整轨迹；actor microbatch/GPU=1，4GPU |
| 生成 microbatch | HF microbatch=1；新入口的32GB显存适配。会影响采样执行/RNG顺序，作为设置归档；不改变组大小/优化minibatch |
| 训练池 | 3553个合法可解game，覆盖六类任务；由SkillRL环境采样，不强制每轮/总训练完整遍历 |
| 训练时验证 | 每5轮64条 seen 监测轨迹，不冒充完整Seen140 |
| 性能评估 | 显式逐game遍历 Seen140 / Unseen134，分开报告成功率及任务分项；两种split不得合为一个主结果 |
| 轨迹限制 | 50步、prompt4096、response512、history2；训练temperature1，评估0.4；action-only、no-thinking |
| 保存 | 每5轮完整原生checkpoint，无自动清理；另有受控FP32分析导出入口 |
| 读出窗口 | 0→5、5→10、…、145→150，30个预登记窗口；不按结果选择窗口 |
| 读出定义 | 沿用冻结gated D及原有强基线，start-batch endpoint projection；不把5个单步D相加，不重新拟合模型 |
| 支持度 | 自然调用；训练读出仍要求20个非零优势决策、4个game、8条轨迹；不够就弃权，不能填零风险 |

150是全局迭代上限，不是150次optimizer.step，也不是150遍3553个game。名义训练轨迹上限为19,200条（不含监测），不保证唯一game覆盖。

以下是当前接口的**预先登记工程默认值**，不是用户逐项指定，也不是实测最优值：性能seed61001；锚点来源seed61011/61021；old-evidence续跑62011/62021；gold续跑63011/63021/63031/63041；技能锚点至少30次自然首次调用且来自至少10个game，上限50个、按game轮转。执行规模须在预算批准时核对；如调整，应建新版本，不能看过结果再调。

30个相邻窗口来自同一RL seed，共享端点，不能当作30个独立训练seed。Seen/Unseen划分仍为ALFWorld既有划分；不按task类别重新划分。task分项只用于报告，不限制技能候选。

## 4. 修复与证据范围

### 4.1 资产和环境

`skillnet_cohort.prepare` 不启动环境，仅按loader过滤规则建立3553/140/134清单并校验game/traj文件；为16/64行的训练/监测占位数据保存Parquet，避免下载不相关数据集。实际训练game来自环境采样，不来自这16个占位文本。

建立独立环境 `/mnt/workspace/users/wangyifan/.venvs/skillnet-phase12-20260917`，最终 `include-system-site-packages=false`。没有升级系统或旧router测试环境。核心版本与全部安装包见环境锁文件；新环境固定torch2.11.0、transformers5.10.4、openai3.14.1、ray2.43.0。历史依赖缺失在该新环境补齐，入口仅做import验证，没有调用trainer、Ray init、真实ALFWorld reset或Qwen模型加载。

37个PLACEBO均保持原首行wrapper和尾部空白，使用与任务无关的固定排版文字，孤立payload以及两种完整prompt模板长度匹配。payload长度402–1967 tokens。**内部Markdown结构并非逐行一致**，不能在论文中声称完全模板一致；该格式差异仍是控制设计的局限。未改变原技能正文。实际历史/状态可能更长：生成前严格检查完整chat prompt≤4096，超限报错，不截断或悄悄丢样本。合成状态检查不是全环境prompt覆盖证明。

### 4.2 新训练入口与资源

`skillnet_cohort.training` 复用此前冻结的SkillRL对齐模板，显式填入seed/model/data/root，启用逐轮Phase2实际batch、advantage及OLD/NEW全词表FP32分布捕获；补足driver与worker RNG。默认只打印plan，必须显式execute且有匹配准备清单的独立授权文件才能进入训练。原32行elastic launcher不适用于本128轨迹路径；禁止继承PHASE2_ELASTIC_TRAINING/PHASE2_CPU_ADAM实验覆盖项。

真实前后向/显存/恢复尚未测。本入口只允许全新run root，拒绝自动接续或覆盖已有目录；原生FSDP保存能力不能代替新设置的恢复验收。中断后不得直接重启同一路径掩盖半提交数据；须先检查checkpoint/证据，再设计显式恢复。这里未声称环境RNG支持逐bit续跑。

OLD+NEW概率行的未压缩体量为 `tokens × 248320 × 4 × 2` bytes，即每个response token约1.99MB。例如每轮10,000个有效response tokens，单轮约19.9GB；150轮约2.98TB，尚未计checkpoint/导出/轨迹。样例只是公式，不是实测平均。当前约994GiB可用并不能保证全程容纳。

捕获前检查run目录已用量、最低剩余空间及下一次checkpoint预留，预算不足即停止，不删证据、不自动放宽。没有沿用旧实验“忽略磁盘限制”的授权。预算/存储安排是正式执行前待解决项，不是已测可行性。

### 4.3 性能评估与Phase2连接

`skillnet_cohort.evaluate` 为每个game×seed产生唯一job；原子只增写结果；单写者锁避免并发重复执行；恢复时仅跑缺失项。完成声明需要job集合精确相等、身份/内容哈希一致，既不以行数代替完整性，也不允许异源结果混入。完成后的续评不加载policy或触发API。

`skillnet_cohort.support` 只从完整自然调用轨迹生成首次调用锚点：所有37个技能都进入支持清单，不够门槛的明确记为unsupported，不强迫调用。可支持的技能采用统一`all_alfworld`上下文，Seen/Unseen各自建独立support/window root。技能库统一，不再按clean/heat拆库。

窗口适配器把新bank、router、哈希绑定的自然锚点、原始/PLACEBO正文接入现有Phase2数学实现。O/P/N在相同可见状态先路由再替换payload；保留selected ID、router缓存协议及API元数据。NULL仅置空目标payload，不删除候选或关闭全库。

“冻结外部router”指冻结请求协议、模型别名、候选库和共享缓存；第三方API背后的权重快照不能仅凭`gpt-5.4-mini`别名证实。本轮没有新API调用，也没有扩充此前连接冒烟证据。

模型B0/端点FP32导出是后续受控步骤，本轮只登记原始B0指纹，未做导出。锚点来源checkpoint必须与读出起点权重/tokenizer一致。

顺序约束：起点自然锚点/三臂基线 → 实际训练batch和五轮端点读出 → commit features与prediction → 终点gold。终点评估会核对cohort、protocol、五轮窗口、features及checkpoint路径，防止用别的run预测文件放行。全游戏评估不替代首次自然调用的配对O/P/N效用评估。完全没有自然支持时读出停止并弃权，不得强迫某个技能进入实验。

## 5. 操作入口（本轮仅运行离线项）

工作目录为 `SkillRL/`，下列命令不启动训练/评估：

```bash
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/users/wangyifan/.venvs/skillnet-phase12-20260917/bin/python -B -m skillnet_cohort.preflight \
  --preparation docs/experiments/skillnet-phase12-preparation-v1/assets-v2/manifest.json \
  --run-root artifacts/skillnet37-seed404-phase12-v1

PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/users/wangyifan/.venvs/skillnet-phase12-20260917/bin/python -B -m skillnet_cohort.training \
  --preparation docs/experiments/skillnet-phase12-preparation-v1/assets-v2/manifest.json \
  --run-root artifacts/skillnet37-seed404-phase12-v1
```

`preflight --require-execution-ready` 故意返回非零：CPU测试不产生开跑授权。`checkpoints`、`evaluate`默认也只输出计划。

后续显式允许后才使用`checkpoints --execute`导出B0/端点，`evaluate --execute`运行全游戏performance或anchors，`support anchors/window`登记窗口，再调用既有`phase2.measure`、`aggregate`、`window_forecast`、`evaluate`和`window_report`。不要将旧`phase2.run_fast`/弹性launcher直接用于新cohort。所有输出必须放新目录，不能指向旧synthesis-v2根目录。

执行授权模板存于归档的`authorization.example.json`，其approved=false、API预算0且storage未填，不能开跑。密钥仅通过`SKILLNET_ROUTER_API_KEY`进程环境注入，不写模板、配置或归档。一次cohort应共享同一个router.sqlite3和总预算；Ray子进程凭据传递仍需在授权后的集成验收中确认。

## 6. 本轮验证与下一步

具体命令、软件回归数量、准备清单hash、文件保护hash、环境锁以及静态核对结果见 `SkillRL/docs/experiments/skillnet-phase12-preparation-v1/verification.json` 与 `preflight.json`。测试使用CPU、合成数据/假环境/假API；报告相关测试只写pytest临时目录，不重跑真实报告。

下一步不是直接150轮开跑：先确认API/存储额度和四张GPU，获准后做新设置的最小真实四卡前后向/保存恢复及完整prompt验收，再决定正式执行。现阶段不能宣称seed404性能、37个技能都充分被调用、150轮存储足够或readout预测增量已被验证。
