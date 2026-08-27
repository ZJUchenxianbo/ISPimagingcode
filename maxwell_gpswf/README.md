# maxwell_gpswf 实验说明

本子项目把 `06-005` 的低秩远场反演思路类推到三维各向异性 Maxwell 问题，主要研究 ball GPSWF 截断、极化恢复、频率与噪声影响，以及不同重构方法的比较。统一采用 Fourier 约定 `exp(-i C p·x)`。

## 代码结构

```text
maxwell_gpswf/
├── common/          # 配置、phantom、求积、极化恢复、GPSWF 与其他重构方法
├── forward/         # Analytic Born、Discrete VIE-Born 和 Full VIE 数据生成
├── nonlinear/       # 可复用的 BIM-GPSWF 非线性模块
├── experiments/     # exp1-exp5 正式实验
├── main.py          # 统一入口
├── output_guide.tex # 数值实验 LaTeX section
└── README.md
```

## 运行环境与依赖

项目使用根目录 `.venv/` 虚拟环境，所有 Python 命令通过 `.venv/bin/python` 执行。当前环境以 Python 3.14 为准，主要依赖：

- `numpy`：数组、复数线性代数和 NPZ 输出。
- `scipy`：GPSWF 本征问题、特殊函数、Lebedev/Gauss--Jacobi 求积、KDTree 和 VIE 线性求解。
- `matplotlib`：成像图和诊断曲线。
- `pandas`：可选；未安装时使用项目内轻量 CSV 实现。

`common/quadrature.py` 使用 `scipy.integrate.lebedev_rule`，需要支持该接口的 SciPy 版本。

## 运行

```bash
.venv/bin/python maxwell_gpswf/main.py --out-dir outputs
.venv/bin/python maxwell_gpswf/main.py --mode exp1 --out-dir outputs
.venv/bin/python maxwell_gpswf/main.py --mode exp3 --out-dir outputs
.venv/bin/python maxwell_gpswf/main.py --mode exp5 --out-dir outputs
.venv/bin/python maxwell_gpswf/main.py --mode all --quick --out-dir outputs_smoke
```

`--mode` 可取 `exp1`、`exp2`、`exp3`、`exp4`、`exp5` 或 `all`。由于
`exp1`--`exp4` 均包含稠密 Full VIE 求解，正式 `all` 计算量很大。
`all` 仍只运行 `exp1`--`exp4`；实验 5 必须用 `--mode exp5` 显式运行。

## 统一数据流程

正式实验只采用有限方向的 mock 流程：先固定 Lebedev 入射和观测方向，由

```text
p = (d - xhat) / 2
```

形成可测 Fourier 节点，再为每个 target quadrature node 选择附近且极化矩阵稳定的方向配置。底层显式方向对构造函数仍可供正向求解器内部使用，但不作为可选实验模式。

正向数据生成和反演严格分开：

```text
给定 Q(x)
-> 生成 far-field channels
-> 在远场通道上添加噪声
-> 极化恢复 Qhat(p)
-> GPSWF / Fourier / Bessel / DSM 重构 Q(x)
```

“解析”只描述用连续 phantom 的 Born 公式生成远场数据，不表示反演阶段直接已知 `Qhat(p)`。

## 正式实验

| 实验 | 目的 | 散射体 | 数据源 | 远场噪声 |
|------|------|--------|--------|----------|
| `exp1` | 比较 GPSWF 保留维数 `N` | 单立方体，`Q=0.2 Q0` | Full VIE | `0.2` |
| `exp2` | 比较噪声影响 | 最小间距 `0.20` 的三方块 | Full VIE | `0, 0.2, 0.4` |
| `exp3` | 比较波数、分辨率及 GPSWF/DSM | 最小间距 `0.20` 的三方块 | Full VIE | `0.2` |
| `exp4` | 比较 GPSWF/Fourier/Bessel | 最小间距 `0.20` 的三方块 | Full VIE | `0.2` |
| `exp5` | 比较正向数据源及其反演结果 | 最小间距 `0.20` 的三方块 | Analytic Born / Discrete VIE-Born / Full VIE | `0.2` |

实验 1 的六个截断维数按 `2×3` 排列：第一行为 `N=1, 21, 57`，第二行为 `N=71, 237, 496`。实验 3 的六个波数也按 `2×3` 排列：第一行为 `5, 6, 7`，第二行为 `8, 10, 15`；GPSWF、individual-scale 和两种 DSM 图均使用这一布局。

五个实验的完整公式、参数表和 individual-scale 图像见 [output_guide.tex](output_guide.tex)。
实验 2--5 共用同一组三方块几何，其最小边界间距为 `0.20`。
实验 3 额外输出 `exp3_frequency_dsm.png`，它与 GPSWF 主图共用同一份
Full VIE 远场、噪声和极化恢复后的 `Qhat_11`。
实验 1 使用较弱但仍保持非对称各向异性结构的介质对比度 `Q=0.2 Q0`，
正式 Full VIE 网格为 `n_per_axis=23`。实验同时用相同方向配置生成无噪声
解析 Born 数据，并输出 Full VIE/Born 的远场、`Qhat`、径向幅值和相位差
诊断；这些诊断不包含主成像使用的 `0.2` 随机噪声。
实验 2 在 `k=15` 时沿用实验 1 的 `requested_measure_dirs=974`、
`n_per_axis=23`、球求积和 GPSWF 参数；三个噪声水平使用同一个标准复噪声
样本按比例缩放，使比较只改变噪声幅度。

实验 3 的正式 Full VIE 网格与候选测量方向数随波数变化：

| `k` | `n_per_axis` | `requested_measure_dirs` |
|---:|---:|---:|
| 5 | 11 | 230 |
| 6 | 11 | 302 |
| 7 | 12 | 350 |
| 8 | 13 | 434 |
| 10 | 16 | 590 |
| 15 | 23 | 974 |

其中 Full VIE 网格以实验 1 的 `k=15, n_per_axis=23` 保持近似相同的
每波长网格密度；方向数以 `k=15, 974` 为锚点近似按 `k^(3/2)` 增长，
并向上取支持的正权 Lebedev 规则，以减弱高频 mock 节点相位误差。实验 3
在 `k=15` 时也沿用实验 1 的 `K=48`、`ell_max=12`、每个 `ell` 的 7 个
径向模态以及 `n_radial=12, n_angular=230`。

实验 4 在 `k=8,12,15` 时分别使用 `n_per_axis=13,19,23` 和
`requested_measure_dirs=434,770,974`；`k=15` 的 GPSWF 设置与实验 1
一致。每一行的 GPSWF、Fourier 和 Bessel 仍共享同一份 Full VIE
数据。
实验 5 固定 `k=12`，三类数据共用方向配置、极化矩阵、标准复高斯噪声样本、
目标求积节点和四种重构参数。正式 Full VIE 使用 `n_per_axis=14`，形成
1472 个体素和 4416 个电场未知量；代码按唯一入射方向复用两个极化总场，
但正式运行仍需要较多内存和时间。

## 极化恢复

每个 target Fourier 节点从 24 个近邻候选方向配置中选取 `polarimetric_J=6` 组。每组方向对使用两个正交入射极化并保留远场向量的三个分量，因此一般 full tensor 对应 `36 x 9` 联合极化矩阵。代码要求列秩为 9，并记录最小奇异值和条件数。

## 输出

```text
outputs/exp*/exp*_*.png                  # 成像图
outputs/exp*/exp*_diagnostics.csv        # 可读标量诊断
outputs/exp*/exp*_diagnostics_detail.npz # 详细压缩诊断
outputs/exp*/exp*_diagnostic_curves.png  # 稳定性与泄漏曲线
outputs/exp3/exp3_frequency_dsm.png       # 实验 3 DSM 波数对照图
outputs/exp3/exp3_dsm_diagnostics.csv    # 实验 3 DSM 诊断
outputs/exp3/exp3_frequency_farfield_dsm.png # 实验 3 原始远场 EM-DSM 对照图
outputs/exp3/exp3_farfield_dsm_diagnostics.csv # 原始远场 EM-DSM 诊断
outputs/exp1/exp1_full_vs_born_diagnostics.png
outputs/exp1/exp1_forward_model_diagnostics.csv
outputs/exp1/exp1_forward_model_diagnostics_detail.npz
```

实验 2 的成像图按 `Truth | noise=0 | noise=0.2 | noise=0.4` 单行排列。
实验 2--5 的三方块 phantom 统一使用实验 1 的弱对比度标准，即在原有三块相对复振幅不变的前提下整体乘以 `0.2`。
实验 3 同时输出两种 DSM：`exp3_frequency_dsm.png` 使用极化伪逆恢复后的 `Qhat_11`；`exp3_frequency_farfield_dsm.png` 直接计算 `M_i^* g_i`。两者使用同一份 Full VIE 远场数据和完全相同的噪声实现。

主要诊断包括 `retained_modes`、`target_nodes_per_retained_modes`、`mock_distance_mean/max/p95`、极化矩阵秩与条件数、GPSWF Gram 矩阵条件数、系数范数、目标区域幅值和背景 95% 分位幅值。实验 5 还记录相对 Analytic Born 的远场与极化恢复后 `Qhat` 误差，以及 Full VIE 的体素数、未知量数、唯一入射方向数、右端项数和抽样线性残差。

正式 Full VIE 的开销随体素数快速增长。实验 1、2 以及实验 3、4 的
`k=15` 行使用 `n_per_axis=23`，约有 6403 个体素和 19209 个复电场未知量，
单个稠密复矩阵约占 `5.50 GiB`；LU 分解还需要额外内存。实验 3 的
波数序列为 `5, 6, 7, 8, 10, 15`，不再包含原先内存压力最大的 `k=20`；
因此当前实验 3 的最大 VIE 网格也是 `k=15, n_per_axis=23`。正式运行仍需
为稠密 LU 分解预留额外内存。`--quick`
中的粗 VIE 网格和固定 110 个方向只验证代码链路，不用于评价正式 Full VIE
精度或成像效果。

实验输出的临时测试目录应在验证后删除。已有 `outputs/fig*` 和 `outputs2/fig*` 仅作为历史数值结果保留，不再对应活动实验脚本。
