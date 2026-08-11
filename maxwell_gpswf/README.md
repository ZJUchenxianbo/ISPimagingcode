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
`exp5` 包含稠密 Full VIE 求解，`all` 只运行 `exp1`--`exp4`；实验 5
必须用 `--mode exp5` 显式运行。

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
| `exp1` | 比较 GPSWF 保留维数 `N` | 单立方体 | Discrete VIE-Born | `0.2` |
| `exp2` | 比较噪声影响 | 三方块 | Discrete VIE-Born | `0, 0.2, 0.4` |
| `exp3` | 比较波数与近距离分辨率 | 最小间距 `0.20` 的三方块 | Analytic Born | `0.2` |
| `exp4` | 比较 GPSWF/Fourier/Bessel/DSM | 三方块 | Discrete VIE-Born | `0.2` |
| `exp5` | 比较正向数据源及其反演结果 | 三方块 | Analytic Born / Discrete VIE-Born / Full VIE | `0.2` |

五个实验的完整公式、参数表和 individual-scale 图像见 [output_guide.tex](output_guide.tex)。
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
```

主要诊断包括 `retained_modes`、`target_nodes_per_retained_modes`、`mock_distance_mean/max/p95`、极化矩阵秩与条件数、GPSWF Gram 矩阵条件数、系数范数、目标区域幅值和背景 95% 分位幅值。实验 5 还记录相对 Analytic Born 的远场与极化恢复后 `Qhat` 误差，以及 Full VIE 的体素数、未知量数、唯一入射方向数、右端项数和抽样线性残差。

实验输出的临时测试目录应在验证后删除。已有 `outputs/fig*` 和 `outputs2/fig*` 仅作为历史数值结果保留，不再对应活动实验脚本。
