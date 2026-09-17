"""Post-hoc comparison of frozen observations; never refit or overwrite forecasts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from phase1.archive import atomic_write_json, sha256_file, utc_now
from phase2.signed_analysis import SIGNALS, correlation
from phase2.utilities import feature_frame


EXTRA_SIGNALS = {
    "C_upd_centered": "centered fidelity sensitivity",
    "D_decision_equal": "decision-equal gated opposition",
    "P_int_matched_backend": "matched-backend signed projection",
    "D_matched_backend": "matched-backend opposition",
}


def compare(frame):
    """Fixed orientation: -P is decline risk; all other raw values rank high first.

    The orientation of unsigned features is a descriptive high-magnitude-risk
    convention, NOT selected on test labels. Zeros stay in unconditional metrics.
    Down-vs-up AUROC is conditional and must disclose its smaller denominator.
    """
    rows = []
    for signal in dict(SIGNALS, **EXTRA_SIGNALS):
        q = frame.dropna(subset=[signal, "delta_utility"])
        score = -q[signal] if signal in ("P_int", "P_int_matched_backend") else q[signal]
        negative = q.delta_utility < -1e-12
        strong = q.delta_utility < -.05
        nonzero = q.delta_utility.abs() > 1e-12
        rows.append({
            "signal": signal, "risk_orientation": "-value" if signal.startswith("P_int") else "+value",
            "units": len(q), "updates": q.global_update.nunique(),
            "negative_points": int(negative.sum()), "nonzero_points": int(nonzero.sum()),
            "rho_raw_magnitude": correlation(q[signal], q.delta_utility.abs()),
            "rho_risk_decline": correlation(score, -q.delta_utility),
            "auroc_down_vs_up": float(roc_auc_score(negative[nonzero], score[nonzero])) if negative[nonzero].nunique() == 2 else None,
            "ap_any_decline": float(average_precision_score(negative, score)) if negative.any() else None,
            "ap_decline_gt_5pp": float(average_precision_score(strong, score)) if strong.any() else None,
        })
    return pd.DataFrame(rows)


def load_observations(root, max_update):
    pairs = pd.read_parquet(root / "metrics/signal_utility_pairs.parquet")
    pairs = pairs[(pairs.control == "placebo") & (pairs.phase == "all") & pairs.supported & (pairs.global_update <= max_update)].copy()
    margins = pd.read_parquet(root / "metrics/anchor_margins.parquet")
    features = pd.concat([feature_frame(root, int(u), margins) for u in sorted(pairs.global_update.unique())], ignore_index=True)
    keys = ["global_update", "skill_id", "phase", "control"]
    missing = [c for c in features if c not in pairs]
    result = pairs.merge(features[keys + missing], on=keys, validate="one_to_one")
    config = json.loads((root / "protocol.json").read_text())
    result["split"] = result.global_update.map(lambda u: "test" if u in config["test_updates"] else "dev" if u in config["development_updates"] else "boundary")
    result["unit"] = result.apply(lambda x: f"U{int(x.global_update)}/{x.skill_id}", axis=1)
    return result.sort_values(["global_update", "skill_id"]).reset_index(drop=True)


def detailed_tables(frame):
    q = frame.copy()
    for c in ["delta_utility", "old_margin", "old_margin_se", "train_success"]:
        q[c] *= 100
    q["CI_pp"] = frame.apply(lambda x: f"[{x.ci_low*100:+.2f}, {x.ci_high*100:+.2f}]", axis=1)
    groups = [
        ("A. 效用标签与 reward-directed 读出", ["unit", "split", "delta_utility", "CI_pp", "P_int", "D_contribution", "D_ungated_contribution", "C_upd", "C_upd_centered"]),
        ("B. 动作分布幅度与参数更新", ["unit", "u_original_norm", "u_control_norm", "delta_norm", "delta_centered_norm", "forward_kl_original", "js_original", "raw_parameter_delta_l2"]),
        ("C. Activation、旧效用与训练信息", ["unit", "activation_l8_norm", "activation_l16_norm", "activation_l24_norm", "activation_l32_norm", "old_margin", "old_margin_se", "train_success", "advantage"]),
        ("D. 支持量与聚合敏感性", ["unit", "token_count", "decision_count", "nonzero_advantage_decisions", "nonzero_advantage_games", "nonzero_advantage_trajectories", "direction_coverage", "gate_coverage", "D_decision_equal"]),
    ]
    return "\n\n".join(f"#### {title}\n\n" + q[columns].to_markdown(index=False, floatfmt=".6g") for title, columns in groups)


def render(frame, all_metrics, test_metrics, output):
    test = frame[frame.split == "test"]
    columns = ["signal", "risk_orientation", "rho_raw_magnitude", "rho_risk_decline", "auroc_down_vs_up", "ap_any_decline", "ap_decline_gt_5pp"]
    return f"""## 11. 追加分析：全部观测量的逐单元数值与 D 的方向排序对照

追加日期：2026-09-12。沿用 U31–U35 已完成结果，不新增 rollout、不改动既有标签、信号或锁定预测。以下比较是看过本批结果后的**探索性追加分析**，不是新增的预注册测试成功结论。7 个测试单元是 19 个有支持单元的子集，不能相加为 26 个样本。

### 11.1 如何读表

- `delta_utility`、`CI_pp`、`old_margin`、`old_margin_se` 均换算为 pp；`train_success` 为百分数。其他观测量保持原始尺度，不能横向直接比较不同指标的数值大小。
- `old_margin` 来自独立 evidence continuation，不等同于 gold 的 `utility_old`。
- `D_contribution` 是主 D；`D_ungated_contribution` 去掉 fidelity/magnitude gate；`D_decision_equal` 是 decision 等权敏感性。`P_int` 保留正负号。
- `delta_norm` 为原始 interaction norm；`delta_centered_norm` 为 centered interaction norm；`u_original_norm/u_control_norm` 分别为 ORIGINAL/PLACEBO 的更新幅度。activation 为第 8/16/24/32 层的交互范数。
- 支持比例 `direction_coverage/gate_coverage` 在 0–1 范围，非百分数；按 token 统计。完整 CSV 另含 matched-backend P/D、对齐误差、旧新各臂成功率等审计列，没有删去这些数据。

### 11.2 全部 {len(frame)} 个有支持单元

{detailed_tables(frame)}

### 11.3 单独列出的 {len(test)} 个时间外测试单元

以下保持时间/Skill 排序，不按 D 或其他指标事后重排以突出个别结果。

{detailed_tables(test)}

### 11.4 统一比较：幅度关联、下降排序与纯方向区分

所有方向比较都使用“越高越倾向下降”的 score：P 使用 **−P**；D 和无符号幅度使用原值。其余无自然方向的统计量也按原值列出，作为固定方向的描述性基线，**没有在测试集上择优翻转符号**。这不是比较每个幅度指标经过最优拟合之后的预测器。

- `rho_raw_magnitude`：原始观测量与 `|ΔM|` 的 Spearman 相关。
- `rho_risk_decline`：上述 risk score 与 `−ΔM` 的 Spearman 相关。
- `auroc_down_vs_up`：只在非零点估计中区分下降/上升；全体是 4 降+4 升，测试是 2 降+2 升。**它以“已知有变化”为条件，不能替代全样本表现**；零变化仍保留在其他指标及原 MAE 中。
- `ap_any_decline`：`ΔM<0` 对其余所有单元的 AP；全体事件率 4/19=0.211，测试 2/7=0.286。仅排除数值零误差（容差 1e-12），不是可靠方向标签。
- `ap_decline_gt_5pp`：保持原 §6.2 的 `ΔM<−5 pp` 定义；全体/测试均只有 1 个事件，事件率分别为 1/19 和 1/7。两种 AP 的任务不同，不能混用。

#### 全部 19 个有支持单元（含开发/隔离，仅探索性）

{all_metrics[columns].to_markdown(index=False, floatfmt='.4f')}

#### 7 个时间外测试单元（信号已锁定；本表比较口径为事后追加）

{test_metrics[columns].to_markdown(index=False, floatfmt='.4f')}

### 11.5 D 相比幅度指标，当前究竟有什么线索？

1. **对“任意下降”有更好的排序线索。** 测试集 D 将两个下降单元排在前两位，`AP_any_decline=1.000`；centered interaction norm 为 0.833，KL 为 0.700。测试下降/上升条件 AUROC 分别为 D=1.000、centered norm=0.750、KL=0.500。全体条件 AUROC 为 D=0.9375、centered norm=0.5625。D 与下降的 pooled 相关也强于 centered norm（全体 0.417 vs 0.062；测试 0.556 vs 0.185）。
2. **但不是所有定义下 D 都更好。** 对唯一的“下降超过 5 pp”事件，测试 D 的 AP=0.500，而 KL/JS 为 1.000：它们把这个事件排在第一。全体任意下降 AP 则是未门控 D=0.854，高于门控 D=0.799；不能声称 fidelity gate 已有确定增益。
3. **测试的完美二分类排序与 update 身份混杂。** 两个下降都来自 U35，两个上升都来自 U34；没有一个测试 update 同时包含上升和下降。因此无法从测试条件 AUROC=1 推出 D 已能在同一次更新内区分哪种 Skill 上升/下降，也不能视为跨 update 稳定性已经证实。
4. **D 的排序好不等于下降幅度预测准。** D 把 −1.67 pp 的 `cle_003@U35` 排在 −14.29 pp 的 `cle_004@U35` 前面；前者 CI 跨 0。方向排序、变化幅度回归、可靠下降识别是不同问题。
5. **本表不推翻预锁定模型未优于基线的结果。** 当前支持的是 reward-opposing 读出值得与无符号幅度作进一步比较，而不是已经确认奖励投影带来稳定增量。后续需固定比较口径、增加混合正负方向的独立窗口，并控制旧效用、Skill/context 及 update-level 变化。

可复核文件：[全部 19 单元完整数值 CSV]({output / 'all_supported.csv'})、[7 个测试单元 CSV]({output / 'heldout.csv'})、[全部对照指标 CSV]({output / 'comparisons.csv'})、[输入与代码 SHA-256]({output / 'manifest.json'})。数值表保留 6 位有效数字用于阅读，CSV 保留完整数值精度；matched-backend P/D 在本批与对应主读出一致。
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--max-update", type=int, default=35)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    root, output = a.root.resolve(), a.output.resolve()
    frame = load_observations(root, a.max_update)
    test = frame[frame.split == "test"]
    all_metrics, test_metrics = compare(frame), compare(test)
    output.mkdir(parents=True)
    frame.to_csv(output / "all_supported.csv", index=False)
    test.to_csv(output / "heldout.csv", index=False)
    comparison = pd.concat([all_metrics.assign(scope="all_exploratory"), test_metrics.assign(scope="heldout")], ignore_index=True)
    comparison.to_csv(output / "comparisons.csv", index=False)
    (output / "appendix.md").write_text(render(frame, all_metrics, test_metrics, output))
    sources = [root / "protocol.json", root / "metrics/signal_utility_pairs.parquet", root / "metrics/anchor_margins.parquet", Path(__file__), Path(__file__).with_name("utilities.py"), Path(__file__).with_name("signed_analysis.py")]
    for update in frame.global_update.unique():
        sources += [root / f"signals/u{update:04d}/skill_context_features.parquet", root / f"signals/u{update:04d}/parameter_delta.json", Path(__file__).resolve().parents[1] / f"artifacts/training_steps/phase2-s303-fast-u{update}/step-{update:06d}.json"]
    atomic_write_json(output / "manifest.json", {"created_at": utc_now(), "all_units": len(frame), "heldout_units": len(test), "test_is_subset": True, "post_hoc": True,
        "source_sha256": {str(path): sha256_file(path) for path in sources}, "outputs_sha256": {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}})
    print(output / "appendix.md")


if __name__ == "__main__":
    main()
