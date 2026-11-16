# 优化运行入口（2026-11-16）

默认现在只生成观测和参考数据，避免直接启动多轮反演。正式空气配置已纠正为中心15 kHz、带通10–20 kHz、网格2.5 mm；其计算量比旧配置更大，仍需网格验收。`dual_candidate.json` 是5 mm自由表面候选，尚未通过实际频段物理验收，不能直接当可信生产配置。

```bash
# 先估算，默认不运行反演
.venv/bin/python ice_imaging_2d/main.py --estimate
# 服务器：生成正式数据
.venv/bin/python ice_imaging_2d/main.py --stage data --workers 1
# 复用数据和一套导数，计算三个窗口的指标
.venv/bin/python ice_imaging_2d/main.py --stage indicator --workers 1
# 有预算的固定灵敏度更新（最多两步；默认总预算18个模型）
.venv/bin/python ice_imaging_2d/main.py --stage reconstruct --strategy frozen --no-baselines --workers 1
# 原逐次更新灵敏度的昂贵模式需显式指定
.venv/bin/python ice_imaging_2d/main.py --stage reconstruct --strategy full --workers 1
# 服务器有限步数计时，仅输出计时结果
.venv/bin/python -m ice_imaging_2d.experiments.benchmark --config ice_imaging_2d/configs/dual_candidate.json --steps 100 --output ice_outputs/benchmarks/candidate.json
# 服务器真实粗糙几何：空气/自由表面波形对照（耗时较大）
.venv/bin/python -m ice_imaging_2d.experiments.production_validation --reference-config ice_imaging_2d/configs/dual_short.json --candidate-config ice_imaging_2d/configs/dual_candidate.json --output ice_outputs/validation/air_free
```

数据保存在 `ice_outputs/datasets/<id>/`，正演及导数共用 `forward_cache/`，结果位于 `runs/<实验>/<阶段>/<策略>/<窗口>/`。`--compare-windows` 自动复用波场和导数。不同正则化试验请指定独立输出目录，避免同一 run 的配置混用。`--model-budget` 包含观测、参考和初始中心差分的模型等价成本；仅约束固定灵敏度路径，不含可选旧基线和物理验收，缓存读取也保守计入候选评价预算。

固定灵敏度重建只报告参考位置的 Gram/秩；兼容输出里的 `final_diagnostics` 仍存该参考诊断，`metadata.json` 的 `diagnostic_geometry=reference` 明确标识，不代表末点可辨识性。源数、接收数、冰厚和20 cm峰谷起伏保持现有定义。低频 `--quick` 仅用于软件检查。

当前实现完成缓存、阶段入口、预算更新及验证工具；自由边界误差、真实频段网格/相位收敛、边界污染和服务器并行调优尚待实测。对照工具的3%波形门槛通过也不自动授予生产验收，仍需逐道相位和导数检查。

以下为原方法说明；其中旧默认频段、默认执行行为及输出路径以本节为准。

# 二维海冰双界面成像

默认入口现在运行研究方案第 3、4.6、8、10 节对应的 **空气—弹性冰层—海水正演 + 离散数值灵敏度 + 正则化双界面联合重建**。未知量为平坦冰面高度和冰底局部线性控制点，材料参数已知。输出根目录为 `ice_outputs/`。

该方法用耦合正演响应包含 P/S 转换与冰内反射；中心差分是离散灵敏度近似，**没有实现严格连续形状导数、显式 Green 张量反传播、伴随 RTM、完整 FWI 或因子分解方法**。现有声学 DSM、走时成像和 P 回波扫描保留为对照。方法、离散与适用范围见 [数值说明](docs/dual_method.md)，旧实验说明见 [legacy](docs/legacy.md)。

## 环境与运行

从 `pyproj` 根目录执行，统一使用 Python 3.14 的 `.venv`；依赖为 NumPy、SciPy、Matplotlib、Numba，无神经网络。

```bash
.venv/bin/python -m pip install -r ice_imaging_2d/requirements.txt
# 默认研究规模仅估算，不运行或写文件
.venv/bin/python ice_imaging_2d/main.py --estimate
# 本地缩小低频验证：固定冰面、固定冰底、联合变化三类（含两种冰底）
.venv/bin/python ice_imaging_2d/main.py --quick --experiment all --workers 1 --out-dir ice_outputs/quick
# 物理、采集、灵敏度验证；自动清理临时波场，可选 --output 保存 JSON 摘要
.venv/bin/python -m ice_imaging_2d.experiments.validation_checks
# 正式短记录研究配置；计算量较大，应先估算，在服务器显式运行
.venv/bin/python ice_imaging_2d/main.py --experiment joint --workers 2
# 方案 1 s 长记录，先查看成本
.venv/bin/python ice_imaging_2d/main.py --config ice_imaging_2d/configs/dual_long.json --estimate
# 参考几何下固定三个不同时间窗，分别运行并比较
.venv/bin/python ice_imaging_2d/main.py --quick --experiment joint --compare-windows --out-dir ice_outputs/windows
# 旧水域核基线（旧配置文件仍可读）
.venv/bin/python ice_imaging_2d/main.py --mode legacy --case sine --out-dir ice_outputs/legacy
```

- `--mode dual` 为默认；`--mode legacy` 才运行原 `--case` 实验。双界面模式使用 `--experiment bottom_sine|bottom_bump|top|joint|all`，默认 `joint`。
- `--quick` 选择独立 `dual_smoke.json`，不能和 `--config` 同时使用。低频小域结果仅验证实现，不代表 10 kHz/10 m 的精度。
- `--window bottom|internal|full` 选择固定参考时间窗；默认 full。`--compare-windows` 分别运行三个窗，输出各自目录；并非自动选最佳窗。
- `--no-baselines` 跳过旧方法对照；`--skip-dsm` 仅跳过 DSM 对照，仍计算走时基线。启用合成压力噪声时不计算需要独立压力梯度噪声模型的 DSM 对照。
- `--reuse-data` 复用相同实验目录下的观测数据；配置、几何和求解器版本必须匹配。新算法不接受 v1 历史数据冒充当前三介质数据。旧档案可通过 legacy 路径复用。
- `--output` 与 `--out-dir` 同义，路径相对于运行时工作目录。双界面模式自动追加实验名及可选窗口名。

## 运行进度与服务器

默认 `dual_short` 也不是轻量任务：约 2754 万网格点、每炮 11.25 万步，单个模型 7 炮，灵敏度反演需要重复多次正演。只有 `--quick` 是本地缩小配置。`forward/` 是参考模型/扰动模型缓存目录，出现该目录不表示已完成一炮。

程序启动时打印规模；随后打印参考数据、观测数据、灵敏度阶段，以及逐炮建模、缓存命中、时间步进度和耗时。首次调用 Numba 还需要编译；建模和首次编译期间没有时间步进度是正常的。中断时只有已完整写入的炮可以复用。

当前 Numba 内核按单炮串行运行，通过 `--workers` 在 CPU 上并行不同炮，不使用 GPU。服务器的较大内存和多个 CPU 核心有帮助，但有效并行炮数不超过声源数；增加 workers 会成倍增加工作数组内存。正式运行前应以单炮实测耗时/峰值内存估算整个反演，不根据 CPU 核数直接承诺加速倍数。

## 配置

双界面 JSON 分为 `physics`（已知物理与离散）、`inverse`（独立参考模型/先验/求解参数）、`experiment`（仅用于合成观测和评价的真值设置）。未知字段报错。真值几何不会传入重建函数。

| 配置 | 收发深度 / 中心频率 | 空间步长 / 记录时间 | 用途 |
|---|---|---|---|
| `dual_short.json`（默认） | 10 m / 10 kHz | 0.004 m / 40 ms | 研究配置，覆盖首反射及冰内回波 |
| `dual_long.json` | 10 m / 10 kHz | 0.004 m / 1 s | 对应方案长记录，资源开销很大 |
| `dual_smoke.json` | 1 m / 1.2 kHz | 0.025 m / 4 ms | 本地链路检查，冰厚也缩为 0.5 m |

正式配置中平均冰厚 1.5 m、冰底振幅 0.1 m 来自方案第 10.3 节示例；7 个声源、81 个接收点、6–14 kHz 带宽、空气参数、40 ms 短窗、网格、正则化和初值属于明确给定的经验设置，非理论最优值。空气声速 343 m/s、密度 1.225 kg/m³ 是可修改的实验假设。

`inverse.control_x` 是局部线性冰底控制点；`reference_top/reference_bottom` 为独立参考平层。`difference_step` 单位 m，需做步长与网格检查。`regularization` 控制参数偏移惩罚，`smoothness` 控制底面相邻坡度变化；`max_step`、`top_bound`、`bottom_bound` 单位均为 m。优化只依据数据残差和先验，不依据真值误差选步长/停止。

`experiment.truth_grid_ratio>1` 使用更细网格生成观测，反演通过相同物理收发位置和公共时间区间对齐。`noise_snr_db` 是原始压力全数据范数的合成白噪声 SNR，与 legacy 首反射残差窗噪声约定不同。

## 输出

`ice_outputs/<experiment>/` 中主要文件：

| 文件 | 含义 |
|---|---|
| `metadata.json`、`config.json` | 模型、参考模型、参数来源、真实采集坐标、版本与耗时 |
| `data_observed/reference/predicted.npz` | 观测、参考、最终预测，压力形状 `(source, receiver, time)` |
| `reconstruction.npz/.png` | 平坦冰面、冰底曲线、冰厚及评价真值 |
| `interface_indicators.npz/.png` | 上界面参数与底面控制点的灵敏度相关性，不是绝对高度图 |
| `diagnostics.npz/.png` | 原始/列归一化奇异值、列范数、相关性和残差历史 |
| `time_windows.npz` | 固定时间窗、参考预测到时 |
| `metrics.json`、`iteration_history.csv` | 双界面/厚度误差、每步接受状态与目标函数 |
| `validation.json`、`RESULTS_zh.md` | 本次运行实际检查范围、停止原因及限制 |
| `baseline_images.npz`、`baseline_top_scan.npz` | 可选旧方法对照 |

正演按炮缓存完整几何和配置；中心差分按参数顺序计算，导数矩阵使用磁盘 memmap。每次迭代复用一个导数文件，保留初始与最终诊断。正演及处理目前仍需要一个完整炮集，`--estimate` 不代表包括所有进程、临时数组和缓存的内存上界；多进程会增加内存，长记录/大网格应先评估。

随包 `results/compact/`、`results/top_shift/` 是原自由表面/走时与 DSM 历史结果，保持原样；不能作为新耦合算法的性能证据。

## 文件职责

`ice2d/model.py`、`solver.py` 负责正演，`geometry.py` 负责几何与采集权重；`sensitivity.py` 只接收参考模型和观测，`reconstruction.py` 实现约束联合更新；`dual_outputs.py` 才接触评价真值。`experiments/dual_interface.py` 负责实验组织，`experiments/validation_checks.py` 调用公共数值检查，各实验脚本不互相导入。
