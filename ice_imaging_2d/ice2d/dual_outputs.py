"""Evaluation and figures isolated from the inversion's input interface."""

from pathlib import Path
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_dual_results(output, inverse, objective, result, truth, metadata, baseline_metrics):
    output = Path(output)
    x = np.linspace(inverse.control_x[0], inverse.control_x[-1], 201)
    bottom = result["geometry"].evaluate(x)
    true_bottom = truth.evaluate(x)
    top = result["geometry"].top
    thickness = top - bottom
    true_thickness = truth.top - true_bottom
    error = bottom - true_bottom
    metrics = dict(top_absolute_error_m=abs(top - truth.top), bottom_rmse_m=float(np.sqrt(np.mean(error ** 2))),
                   bottom_maximum_error_m=float(np.max(abs(error))),
                   bottom_relative_l2=float(np.linalg.norm(error) / max(np.linalg.norm(true_bottom), 1e-30)),
                   thickness_relative_l2=float(np.linalg.norm(thickness - true_thickness) / np.linalg.norm(true_thickness)),
                   thickness_rmse_m=float(np.sqrt(np.mean((thickness - true_thickness) ** 2))),
                   reference_bottom_rmse_m=float(np.sqrt(np.mean((inverse.reference_bottom - true_bottom) ** 2))),
                   reference_top_error_m=abs(inverse.reference_top - truth.top),
                   initial_residual=result["initial_norm"], final_residual=result["final_norm"],
                   stop_reason=result["status"], baselines=baseline_metrics)
    np.savez_compressed(output / "reconstruction.npz", x=x, bottom=bottom, top=top, thickness=thickness,
                        true_bottom=true_bottom, true_top=truth.top, true_thickness=true_thickness,
                        theta=result["theta"], control_x=inverse.control_x)
    arrays = {}
    for phase in ("initial", "final"):
        for window, values in result[f"{phase}_diagnostics"].items():
            for name, value in values.items():
                arrays[f"{phase}_{window}_{name}"] = value
    np.savez_compressed(output / "diagnostics.npz", **arrays)
    indicators = {name: arrays[name] for name in arrays if "indicator" in name}
    np.savez_compressed(output / "interface_indicators.npz", control_x=inverse.control_x, **indicators)
    np.savez_compressed(output / "time_windows.npz", time=objective.time, **objective.windows)
    np.savez_compressed(output / "data_predicted.npz", **result["predicted"])
    history = result["history"]
    with (output / "iteration_history.csv").open("w", newline="") as handle:
        keys = list(history[0]) if history else ["iteration", "residual_before", "residual_after", "accepted"]
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)
    current = result["final_diagnostics"][inverse.window]
    diagnostic_label = metadata.get("diagnostic_geometry", "final")
    validation = dict(finite_reconstruction=bool(np.isfinite(bottom).all() and np.isfinite(top)),
                      positive_thickness=bool(np.all(thickness > 0)),
                      residual_decreased=result["final_norm"] < result["initial_norm"],
                      full_parameter_numerical_rank=current["rank"], parameter_count=len(result["theta"]),
                      max_top_bottom_correlation=current["max_top_bottom_correlation"],
                      physical_convergence_checked=False, independent_grid=metadata["experiment"]["truth_grid_ratio"] > 1,
                      note="Run validation_checks.py separately for physical/linearization checks; these flags are not a resolution claim.")
    for filename, value in (("metadata.json", metadata), ("metrics.json", metrics), ("validation.json", validation)):
        (output / filename).write_text(json.dumps(value, ensure_ascii=False, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(x, true_bottom, "k--", label="True bottom")
    axes[0].plot(x, bottom, label="Estimated bottom")
    axes[0].axhline(truth.top, color="black", ls=":", label="True top")
    axes[0].axhline(top, color="tab:orange", label="Estimated top")
    axes[0].set(xlabel="x (m)", ylabel="z (m)", title="Coupled interface reconstruction")
    axes[0].legend(fontsize=8)
    axes[1].plot(x, true_thickness, "k--", label="Truth")
    axes[1].plot(x, thickness, label="Reconstruction")
    axes[1].set(xlabel="x (m)", ylabel="Thickness (m)")
    axes[1].legend()
    fig.savefig(output / "reconstruction.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for window in ("bottom", "internal", "full"):
        score = result["initial_diagnostics"][window]["indicator"]
        axes[0].scatter([window], [score[0]])
        axes[1].plot(inverse.control_x, score[1:], "o-", label=window)
    axes[0].set(ylabel="Normalized sensitivity correlation", title="Top parameter response")
    axes[1].set(xlabel="Bottom control x (m)", title="Bottom parameter responses")
    axes[1].legend()
    fig.savefig(output / "interface_indicators.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for window in ("bottom", "internal", "full"):
        axes[0].semilogy(result["final_diagnostics"][window]["normalized_singular_values"], "o-", label=window)
    axes[0].set(xlabel="Index", ylabel="Singular value", title="Column-normalized sensitivity")
    axes[0].legend()
    im = axes[1].imshow(current["correlation"], vmin=-1, vmax=1, cmap="coolwarm")
    axes[1].set(title=f"{diagnostic_label} correlation (0=top)", xlabel="Parameter", ylabel="Parameter")
    fig.colorbar(im, ax=axes[1])
    axes[2].plot([result["initial_norm"]] + [row["residual_after"] for row in history], "o-")
    axes[2].set(xlabel="Iteration", ylabel="Relative residual", title=result["status"])
    fig.savefig(output / "diagnostics.png", dpi=160)
    plt.close(fig)
    report = ["# 双界面数值灵敏度实验", "", "采用三介质/自由表面可配置正演及离散数值灵敏度；不是严格 RTM 或连续形状导数。",
              f"上边界模式：{metadata['physics']['upper_medium']}；材料界面：{metadata['physics']['interface_scheme']}。",
              "真值仅用于生成观测和最终评价，反演使用独立参考模型。", "",
              f"冰面绝对误差：{metrics['top_absolute_error_m']:.6g} m。",
              f"冰底 RMSE：{metrics['bottom_rmse_m']:.6g} m；参考平面 RMSE：{metrics['reference_bottom_rmse_m']:.6g} m。",
              f"冰厚相对 L2 误差：{metrics['thickness_relative_l2']:.6g}。",
              f"归一化残差：{result['initial_norm']:.6g} → {result['final_norm']:.6g}；停止原因：{result['status']}。",
              f"诊断位置：{diagnostic_label}。全参数数值秩：{current['rank']}/{len(result['theta'])}；最大上下界面列相关性：{current['max_top_bottom_correlation']:.6g}。",
              "", "成像指标是参数响应相关性，不是绝对高度或概率。只变单界面的实验也记录全参数诊断。",
              "当前结果不等于正式网格收敛、可辨识性或分辨率结论。空气强反差、界面过渡宽度和观测孔径均需进一步验证。",
              "本次运行不自动执行物理收敛验证；validation.json 明确记录验证范围。",
              "旧走时/DSM 对照是自由水域传播近似，不能将其结果归为耦合 RTM。",
              "", "参数说明：" + metadata["experiment"]["parameter_notes"],
              "", "依据：用户研究方案第 3、4.6、8、10 节；数值中心差分是本实现的离散近似。"]
    if metrics["bottom_rmse_m"] > metrics["reference_bottom_rmse_m"]:
        report.append("本次下界面误差高于参考平面，不能判定下界面恢复成功。")
    if metrics["top_absolute_error_m"] > metrics["reference_top_error_m"]:
        report.append("本次上界面误差高于初始参考高度，需检查参数串扰与模型误差。")
    (output / "RESULTS_zh.md").write_text("\n".join(report) + "\n")
