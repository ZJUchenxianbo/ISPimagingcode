# maxwell_gpswf 实验说明

把 `06-005` 的标量远场反演方法类推到各向异性 Maxwell 问题。主方法：**ball GPSWF 函数截断 / alpha spectral cutoff**。Fourier 约定 `exp(-i C p·x)`。

## 代码结构

```
maxwell_gpswf/
├── common/          # config, phantom, 求积, 极化, GPSWF, Fourier/Bessel/DSM
├── forward/         # 解析 Born + VIE (含自作用项 L=-I/3)
├── experiments/     # 主图实验
├── main.py
└── README.md
```

## 运行环境与依赖

本项目在根目录 `.venv/` 虚拟环境中运行，当前本地环境为：

```text
Python 3.14.5
numpy 2.4.6
scipy 1.17.1
matplotlib 3.10.9
pandas 3.0.3
```

核心依赖：

- `numpy`：数组计算、复数线性代数、`.npz` 数据保存。
- `scipy`：GPSWF 三对角本征问题、球 Bessel/Jacobi 函数、Lebedev 球面积分节点、KDTree 匹配、GMRES/VIE 求解。
- `matplotlib`：主图和诊断曲线输出。
- `pandas`：可选依赖；若本地没有安装，代码会退回到轻量 CSV 表格实现。

注意：`maxwell_gpswf/common/quadrature.py` 使用 `scipy.integrate.lebedev_rule`，因此 SciPy 版本不能太旧。运行时统一使用项目根目录的虚拟环境：

```bash
.venv/bin/python maxwell_gpswf/main.py ...
```

## 运行

```bash
.venv/bin/python maxwell_gpswf/main.py --out-dir outputs
.venv/bin/python maxwell_gpswf/main.py --mode exp1 --data-mode mock --out-dir outputs
.venv/bin/python maxwell_gpswf/main.py --mode exp3 --data-mode ideal --out-dir outputs_ideal
.venv/bin/python maxwell_gpswf/main.py --mode all --quick --out-dir outputs/smoke
```

`--mode` 可取 `exp1`、`exp2`、`exp3`、`exp4`、`all`。默认数据模式为 `mock`。

每个主实验会在对应输出目录生成成像图和运行时诊断文件：

```text
exp*/exp*_*.png                  # 成像图
exp*/exp*_diagnostics.csv        # 标量诊断表
exp*/exp*_diagnostics_detail.npz # 标量诊断 NPZ 备份
exp*/exp*_diagnostic_curves.png  # 诊断曲线
```

当前 `exp1`--`exp4` 的数值设置、公式、图表和结果说明见
[output_guide.tex](output_guide.tex)。该文件是可由论文主文档直接
`\input` 的 LaTeX section 片段。

旧的论文表格型 `diagnostics/` 目录已删除。现在的诊断逻辑在 `common/diagnostics.py` 中，由各主图脚本在真实运行流程里调用。

---

## 数据源术语

本子项目中“解析”只描述**远场数据的生成方式**，不是说反演时直接把 `Q(x)` 交给算法。

三类数据源区分如下：

```text
Analytic Born far-field
    给定连续 phantom Q(x)，代入 Born 远场公式；
    对球、方块、高斯等形状，积分项有解析表达式。

Discrete VIE-Born far-field
    给定体素离散 Q_i，在 VIE 求解器框架内把总场替换为入射场 E_i；
    远场由体素求和得到，是离散 Born 数据。

Full VIE far-field
    给定体素离散 Q_i，先解 VIE 得到总场 E，再计算远场；
    数据含多重散射，不再等价于 Born 模型下的 Qhat(p)。
```

在 Born 近似下，远场通道满足 `g(p)=M(p)c(p)`，其中 `c(p)` 是 `Qhat(p)` 在张量基下的系数。极化恢复指的是从远场通道 `g(p)` 解出 `c(p)`，然后再进入 GPSWF / Fourier / Bessel / DSM 成像。

## 当前主实验

| 实验 | 目的 | 散射体 | 远场噪声 |
|------|------|--------|----------|
| `exp1` | 比较固定保留维数 `N` | 单立方体 | `0.2` |
| `exp2` | 比较噪声影响 | 三方块 | `0, 0.2, 0.4` |
| `exp3` | 比较波数与近距离分辨率 | 最小间距 `0.20` 的三方块 | `0.2` |
| `exp4` | 比较 GPSWF/Fourier/Bessel/DSM | 三方块 | `0.2` |

`exp1` 中的 `N` 是保留维数上限。模式按 `(ell,n)` 分组，同一组的
`m=-ell,...,ell` 必须完整保留；alpha 稳定平台内部按 Sturm-Liouville
特征值 `chi` 排序。因此诊断中的 `retained_N` 可能小于 `requested_N`。
当前三行维数依次为 `1, 5, 21`、`35, 57, 71` 和
`135, 237, 496`，均对应完整简并层的实际保留维数。
实验同时输出 `exp1_dimension.png` 和
`exp1_dimension_individual_scale.png`：前者统一使用真值色轴 `[0,1]`，
后者为每幅重构按自身球内 `min/max` 设置色轴并显示 colorbar，仅用于
观察结构细节，不用于比较不同 `N` 的绝对幅值。
实验一正式模式使用 `161×161` 成像网格，quick 模式保持 `51×51`。
正式模式使用 974 个 Lebedev 入射方向和同一组 974 个观测方向，quick
模式使用 110 个方向，以减小 mock Fourier 节点匹配误差对维数比较的影响。

实验二同时输出 `exp2_noise.png` 和
`exp2_noise_individual_scale.png`：前者对三个噪声水平统一使用真值色轴，
后者为每幅图按自身球内 `min/max` 设置色轴并显示 colorbar。实验二正式
模式同样使用 `161×161` 成像网格和 `bicubic` 显示插值，quick 模式保持
`51×51`。正式模式使用 974 个 Lebedev 入射方向和同一组 974 个观测
方向，quick 模式使用 110 个方向，避免方向离散误差形成固定误差底并
干扰噪声水平比较。

实验三同时输出 `exp3_frequency.png` 和
`exp3_frequency_individual_scale.png`：前者对不同波数统一使用真值色轴，
后者为每幅图按自身球内 `min/max` 设置色轴并显示 colorbar。面板标题显示
对应的 `k` 和实际保留维数 `N`。实验三正式模式使用 `161×161` 成像网格
和 `bicubic` 显示插值，quick 模式保持 `51×51`。

实验三使用 `k=4, 6, 8, 10, 15, 20`。正式模式的六个波数统一使用
974 个 Lebedev 入射方向和同一组 974 个观测方向，避免把方向密度变化
混入频率比较。三个长方体的最小边界间距为 `0.20`；其解析 Born 公式在
每组实际方向对应的 Fourier 节点生成远场，随后仍经过加噪声、极化恢复
和 GPSWF 重构。诊断中的 `C_mock_distance_mean/p95/max` 用于判断 mock
节点误差随波数产生的相位影响。上述方向数和 GPSWF 离散规模是经验参数。

实验四使用 `k=8, 12, 15` 比较 GPSWF、Cube Fourier、Ball Bessel 和
DSM。正式模式统一使用 974 个 Lebedev 入射方向和同一组 974 个观测
方向，以减小 mock Fourier 节点匹配误差；quick 模式使用 110 个方向。
实验同时输出 `exp4_basis.png` 和 `exp4_basis_individual_scale.png`：前者
统一使用真值色轴比较绝对幅值，后者按每个方法在自身有效支撑区域内的
`min/max` 设置色轴并显示 colorbar。正式模式使用 `161×161` 成像网格
和 `bicubic` 显示插值，quick 模式保持 `51×51`。

`exp3`、`exp4` 的 GPSWF 有效维数上限为：

```text
min(C^2/2, target_nodes/6, 512)
```

epsilon 筛选和上限截断同样不拆分 `(ell,n)` 简并层。两项实验与
`exp1`、`exp2` 统一使用 `viridis` 色图和真值幅值范围。
`exp4` 中 Fourier/Bessel 的候选模式数仍由 GPSWF 实际维数乘比例得到，
但最终保留数也统一限制为不超过 512。

四个实验采用相同的远场反演入口，但正向数据源不同：

```text
给定 Q(x)
-> exp1/exp2/exp4: Discrete VIE-Born 生成远场通道
   exp3: 长方体解析 Born 公式生成远场通道
-> 在远场通道上添加复高斯相对噪声
-> 极化恢复 Qhat(p)
-> GPSWF 或其他基函数重构 Q(x)
```

### mock 极化配置

mock 模式先固定有限 Lebedev 入射/观测方向，并形成实际测量节点
`p_ab=(d_a-xhat_b)/2`。对每个目标求积节点，从 24 个近邻候选中选取
`polarimetric_J=6` 个方向配置。选择顺序为联合矩阵秩、最小奇异值、
节点距离；一般 `full` 张量对应的联合极化矩阵形状为 `36 x 9`，代码要求
其列秩为 9，否则停止运行。

诊断表新增或统一记录：

- `polarimetric_J`
- `polarimetric_rank_min`
- `polarimetric_sigma_min_min/median`
- `polarimetric_condition_median/max`
- `candidate_count`
- `mock_distance_mean/max/p95`

---

## 历史 figure1-figure7 说明

以下内容对应保留在 `experiments/figure*.py` 中的旧实验脚本，不是当前
`main.py` 的 `exp1`-`exp4` 主入口。

---

## 图1: 噪声与截断维数

**目的**：固定 k=15，观察截断维数 N 和噪声水平对 Born 重建的影响。

**布局**：5 列 (truth + δ=0, 0.1, 0.2, 0.3) × 5 行 (N=5, 72, 144, 256, 512)

### 参数

| 参数 | full | quick | 说明 |
|------|------|-------|------|
| k | 15 | 15 | 波数 |
| C = 2k | 30 | 30 | 空间带宽积 |
| 入射/观测方向数 | 110 (Lebedev 17) | 38 (Lebedev 9) | mock 模式用 |
| n_radial | 12 | 6 | 径向 Gauss-Jacobi 点数 |
| 目标角向点数 | 230→302 (Lebedev 29) | 110 (Lebedev 9) | 实际取正权重最近规则 |
| grid_size | 81×81 | 51×51 | z=0 截面网格 |
| K | 60 | 40 | Jacobi 截断阶数 |
| ℓ_max | 18 | 10 | 球谐角向最大阶数 |
| n_modes_per_ℓ | 10 | 6 | 每个 ℓ 保留的径向模态数 |
| 总模式数 | ∑(2ℓ+1)·10 = 3610 | ∑(2ℓ+1)·6 = 486 | (ℓ_max+1)² × n_modes_per_ℓ |
| target_nodes | 12×302 = 3624 | 6×110 = 660 | n_radial × n_angular |
| ratio | 1.00 | 1.36 | nodes / modes |
| N 值 | 5, 72, 144, 256, 512 | 5, 20, 40, 60 | 按 |α| 排序取前 N 个 |
| 噪声 δ | 0, 0.1, 0.2, 0.3 | 同 | 相对噪声水平 |
| 截断方式 | N 固定 | N 固定 | 用于比较截断维数的影响 |
| Phantom | three_block_phantom("born") | 同 | 三方块 |
| 张量类型 | full (9 维) | 同 | |
| 数据分量 | Q_11 | 同 | tensor Fourier 第 0 分量 |
| α quad_order | 160 | 100 | alpha 估计求积阶数 |
| α r_eval_count | 120 | 80 | alpha 估计求值点数 |

### GPSWF 模式结构

对每个 ℓ = 0,...,ℓ_max：
- 径向：n = 0,...,n_modes_per_ℓ-1（三对角矩阵最小的 n_modes_per_ℓ 个本征对）
- 角向：m = -ℓ,...,ℓ（球谐函数 Y_ℓ^m）

---

## 图2: 频率与介质对比度

**目的**：观察不同波数和对比度下 Born 重建质量。GPSWF 参数随 k 联动。

**布局**：5 列 (truth + medium δ=0 + low + medium + high) × 6 行 (k=4, 6, 7, 8, 9, 10)

### GPSWF 参数联动

| k | C | ℓ_max | n_modes | K | n_radial | n_angular | modes | nodes | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 4 | 8 | 4 | 2 | 16 | 5 | 50 | 50 | 250 | 5.00 |
| 6 | 12 | 5 | 3 | 22 | 6 | 74 | 108 | 444 | 4.11 |
| 7 | 14 | 6 | 3 | 24 | 7 | 74 | 147 | 518 | 3.52 |
| 8 | 16 | 7 | 3 | 28 | 8 | 86 | 192 | 688 | 3.58 |
| 9 | 18 | 7 | 4 | 32 | 8 | 110 | 256 | 880 | 3.44 |
| 10 | 20 | 8 | 5 | 36 | 10 | 110 | 405 | 1100 | 2.72 |

### 截断

三层：GPSWF 参数 → ε=0.2 → N_cap = C²/2

### 参数

| 参数 | full | quick | 说明 |
|------|------|-------|------|
| k | 4, 6, 7, 8, 9, 10 | 4, 6, 8, 10 | |
| 入射/观测方向数 | 110 | 38 | |
| grid_size | 81×81 | 51×51 | |
| 噪声 δ | 0.2 | 同 | |
| 对比度 scale | 0.3, 1.0, 3.0 | 同 | low / medium / high |
| ε | 0.2 | 同 | `|α| > ε·max|α|` |
| N_cap | C²/2 (32~200) | — | 随 C 增长 |
| 张量类型 | full (9 维) | 同 | |
| 数据分量 | Q_11 | 同 | |
| α quad_order | 160 | 100 | |
| α r_eval_count | 120 | 80 | |

**截断**：三层 — GPSWF 参数 → ε=0.2 → N_cap = C²/2。

---

## 图3: 数据来源与散射体形状

**目的**：比较 Full VIE far-field / Discrete VIE-Born far-field / Analytic Born far-field 三种数据源在 5 种散射体上的反演。

**布局**：4 列 (truth + Full VIE + VIE-Born FF + Analytic Born FF) × 5 行

### 参数

| 参数 | full | quick | 说明 |
|------|------|-------|------|
| k | 15 | 15 | |
| C = 2k | 30 | 30 | |
| R | 1.0 | 1.0 | 成像区域（球）半径 |
| 入射/观测方向数 | 74 | 26 | |
| n_radial | 12 | 10 | |
| 目标角向点数 | 230→302 | 170 | |
| grid_size | 81×81 | 51×51 | |
| n_per_axis (VIE) | 19 (~3700 体素) | 5 (~81 体素) | VIE 网格分辨率 |
| K | 50 | 30 | |
| ℓ_max | 12 | 8 | |
| n_modes_per_ℓ | 7 | 5 | |
| 总模式数 | 7×13² = 1183 | 5×9² = 405 | n_modes × (ℓ_max+1)² |
| target_nodes | 12×302 = 3624 | 10×170 = 1700 | n_radial × n_angular |
| ratio | **3.06** | 4.20 | nodes / modes |
| ε | 0.1 | 同 | `|α| > ε·max|α|` |
| 张量类型 | isotropic | 同 | |
| α quad_order | 140 | 60 | |
| α r_eval_count | 100 | 50 | |

**截断**：三层 — GPSWF 参数 → ε=0.1 → N_cap = C²/2 (=450)。

### 5 种散射体

| 行 | 名称 | 描述 | 解析 Born 远场积分公式 |
|----|------|------|-------------------|
| 1 | sphere | 球，半径 0.25，中心原点 | 球 Bessel: 4π(sin(z)-z cos(z))/ξ³ |
| 2 | cube | 立方体，边长 0.4 | sinc 乘积 |
| 3 | two_spheres_cube | 双球 + 立方体组合 | sinc 乘积 |
| 4 | dispersed | 6 个分散小方块 | sinc 乘积 |
| 5 | inhomogeneous | 3 个高斯凸起，σ=0.08~0.14 | 高斯解析: (2π)^(3/2)σ³ exp(-σ²|ξ|²/2) |

### VIE 求解器

- 波数 k=15，立方体体素 h≈0.18
- 自作用项：退极化 dyadic `L = -I/3` + 辐射修正 `i k³V/(6π) I`
- GMRES 求解，rtol=1e-8
- VIE 远场经 `vie_to_fourier_convention()` 取共轭后统一约定

---

## 图4: 支撑半径缩放

**目的**：固定 k=8，改变先验支撑球半径 R，观察 C=2kR 增大时 GPSWF 重构的变化。这里 R 表示 `supp(Q) ⊂ B(0,R)` 的支撑外包球半径，真实散射体 Q(x) 的物理尺寸不随 R 改变。

**布局**：5 列 (truth + R=1.0/1.5/2.0/3.0) × 5 行（同图3 散射体）

**数据源**：Analytic Born far-field。所有列都使用固定物理坐标范围 `[-1,1]×[-1,1]` 显示。

尺度关系：

```text
x = R y,  y ∈ B(0,1)
C = 2 k R
f_R(y) = R^3 Q(Ry)
Q(x) = f_R(x/R) / R^3
```

因此图4中每个 R 列使用 `C=2kR` 构造单位球 GPSWF；重构时在物理网格 `x∈[-1,1]^2` 上取 `y=x/R` 进行模态求值，并除以 `R^3` 回到物理散射体 Q(x)。

### GPSWF 参数联动

| R | C | ℓ_max | n_modes | K | n_radial | n_angular | modes | nodes | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 16 | 7 | 3 | 28 | 8 | 86 | 192 | 688 | 3.58 |
| 1.5 | 24 | 10 | 5 | 40 | 10 | 170 | 605 | 1700 | 2.81 |
| 2.0 | 32 | 12 | 6 | 48 | 12 | 302 | 1014 | 3624 | 3.57 |
| 3.0 | 48 | 16 | 7 | 60 | 14 | 302 | 2023 | 4228 | 2.09 |

### 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| k | 8 | 固定波数 |
| R | 1.0, 1.5, 2.0, 3.0 | Q(x) 的先验支撑外包球半径 |
| ε | 0.2 | |
| N_cap | C²/2 (128~1152) | 随 C 增长 |
| 张量类型 | full (9 维) | |
| 数据分量 | Q_11 | |
| grid_size | 81×81 | 物理坐标范围固定为 `[-1,1]×[-1,1]` |
| 入射/观测方向数 | 110 | mock 模式固定 110 个 Lebedev 测量方向；ideal 模式直接构造 target node 对应方向对 |

**截断**：三层 — GPSWF 参数 → ε=0.2 → N_cap = C²/2。

---

## 图5: GPSWF、立方体 Fourier、球谐-Bessel、DSM 与 EM-DSM 对照

**目的**：用图2当前的波数配置，在 medium 对比度下比较三种重构空间和两个直接成像指标：

- GPSWF：单位球支撑，单频 Born 数据，`C=2k`。
- Cube Fourier：立方体支撑 `[-1,1]^3`，候选基函数满足 `|ξ_l| <= 2K_max`。
- Ball Bessel：单位球支撑，球谐-Bessel 候选基，径向零点满足 `ρ_{ℓn} <= 2K_max`。
- DSM：不求展开系数，直接对极化恢复后的 recovered Fourier 数据做相位回投指标。
- EM-DSM：不做极化伪逆恢复，直接用 Maxwell 远场通道算子的伴随 `M(p)^*g(p)` 做相位回投指标。

**布局**：6 列 (truth + GPSWF + Cube Fourier + Ball Bessel + DSM + EM-DSM) × 6 行 (k/Kmax=4, 6, 7, 8, 9, 10)。

GPSWF、Cube Fourier、Ball Bessel 和 DSM 使用同一套极化恢复前置流程：

```text
远场 Born 数据 -> 极化恢复 -> recovered Qhat(p)
```

Cube Fourier 使用标准立方体 Fourier 基：

```text
phi_l(x) = exp(i*pi*l.x),  l in Z^3,  x in [-1,1]^3
```

其系数由 recovered Fourier 数据解最小二乘方程：

```text
sum_l c_l int_cube phi_l(x) exp(-i C p_i.x) dx ~= Qhat(p_i)
```

Ball Bessel 使用单位球 Dirichlet 基：

```text
phi_{ell,n,m}(x) = N_{ell,n} j_ell(rho_{ell,n}|x|) Y_ell^m(theta, phi)
```

其系数同样由 recovered Fourier 数据解最小二乘方程：

```text
sum_j c_j int_ball phi_j(x) exp(-i C p_i.x) dx ~= Qhat(p_i)
```

为避免普通 Fourier/Bessel 数据方程在高频下严重放大噪声，图5会先生成候选模式，再使用经验稳定化截断。Fourier 和 Ball Bessel 的模式数比例分别由 `fourier_mode_fraction`、`bessel_mode_fraction` 控制，当前默认都为 `1.2`：

```text
Fourier modes <= max(12, fourier_mode_fraction * GPSWF retained modes)
Bessel modes  <= max(12, bessel_mode_fraction  * GPSWF retained modes)
```

图5当前最小二乘截断为 `rcond=1e-8`。

DSM 列使用同一批 recovered Fourier 数据：

```text
I_DSM(z) = |sum_i w_i Qhat(p_i) exp(i C p_i.z)|
```

图中显示的是归一化后的 DSM 指标，主要用于比较目标定位和旁瓣，不直接表示 `Q_11` 的对比度大小。

EM-DSM 列使用极化恢复之前的原始远场通道数据，但不是把通道向量当成标量直接相加，而是先作用 Maxwell 通道矩阵的伴随：

```text
I_EMDSM(z) = ||sum_i w_i M(p_i)^* g(p_i) exp(i C p_i.z)||_2
```

这里 `g(p_i)` 包含每个 Fourier 点下所有 admissible direction / incident polarization / observed vector component 通道，`M(p_i)^*` 对应完整电磁测试函数中的横向投影和入射极化结构。它不做 `pinv(M)`，因此仍是直接成像指标，而不是张量 Fourier 系数恢复。

### 候选模式数

| K_max | Fourier modes (`|ξ| <= 2K_max`) | Bessel modes (`ρ <= 2K_max`) |
|-------|----------------------------------|-------------------------------|
| 4 | 81 | 20 |
| 6 | 251 | 93 |
| 7 | 365 | 153 |
| 8 | 515 | 220 |
| 9 | 751 | 338 |
| 10 | 1045 | 456 |

---

## 图6: 不同张量散射体重构

**目的**：检查每个散射体具有不同 `3×3` 张量反差时，极化恢复后的 GPSWF、Cube Fourier、Ball Bessel、DSM，以及不做极化恢复的 EM-DSM 是否还能恢复目标位置。

图6 使用新的 `TensorBlock`：

```text
Q(x) = sum_b 1_{D_b}(x) T_b
```

其中三个 block 的位置沿用图5，但每个 block 使用不同的 full tensor `T_b`。这不再是图5中的 `Q(x)=q(x)T` 共享张量结构。

**布局**：6 列 (truth `||Q||_F` + GPSWF `||Q||_F` + Cube Fourier `||Q||_F` + Ball Bessel `||Q||_F` + DSM `||Q||_F` + EM-DSM) × 6 行 (k=4, 6, 7, 8, 9, 10)。

图6 完全沿用图5的成像流程，只是把共享张量 phantom 换成不同张量的 `TensorBlock` phantom：

```text
远场 Born 数据 -> 极化恢复 -> recovered c_r(p) -> 重构所有张量分量 -> 显示 ||Q(x)||_F
```

这里显示 Frobenius 范数，而不是单个 `Q_11` 分量，避免某个散射体在 `Q_11` 分量上弱或接近零时被误判为没有恢复。

Cube Fourier 和 Ball Bessel 的模式数截断同样分开控制，图6当前默认 `fourier_mode_fraction=1.1`、`bessel_mode_fraction=1.1`，最小二乘截断为 `rcond=1e-7`。

最后一列 EM-DSM 不恢复 `c_r(p)`，而是直接对原始 far-field channel vector 做伴随相位回投；它是定位指标，不是张量反差重构。

---

## 图7: BIM-GPSWF 多重散射修正

**目的**：沿用图5的不同入射波数设置，比较 Born-GPSWF 与 BIM-GPSWF 对 Full VIE 数据的成像效果，观察引入当前总场后是否改善分辨。

第一版采用标量反差模型：

```text
Q(x) = q(x) T0
```

其中 `T0` 为固定 isotropic tensor，未知量为标量函数 `q(x)`。BIM 更新量限制在 GPSWF 低秩空间：

```text
delta q(x) = sum_j a_j psi_j(x)
```

BIM-GPSWF 每次迭代使用当前总场 `E^n` 构造线性化响应：

```text
r^n = d_obs - d_pred^n
A_n(psi_j) = int P_xhat [psi_j(y) T0 E^n(y)] exp(-i k xhat.y) dy
min_a ||A_n a - r^n||^2 + lambda ||a||^2
q^{n+1} = q^n + step * sum_j a_j psi_j
```

**布局**：6 列 (truth + Analytic Born FF-GPSWF + Full VIE data-GPSWF + BIM iter 1 + BIM iter 2 + BIM iter 3) × 若干行。

full 模式使用：

```text
k = 4, 6, 7, 8, 9, 10
GPSWF 参数 = 图5同一 k 行参数
n_per_axis = 7
N_iter = 3
step = 0.2
lambda0 = 1e-2
epsilon = 0.2
phantom = three_block_phantom("born")
```

quick 模式使用：

```text
k = 4, 6, 8, 10
GPSWF 参数 = 图5 quick 同一 k 行参数
n_per_axis = 5
N_iter = 3
```

图7第一列数据源说明：

- `Analytic Born FF-GPSWF`：由连续三方块 phantom 的 Born 远场积分解析公式生成数据，再做 GPSWF 截断成像。
- `Full VIE data-GPSWF`：由体素离散 VIE 求出归一化 raw far-field channels，经极化恢复后再用 GPSWF 线性成像得到初值。
- `BIM iter`：以上一列 Full VIE data-GPSWF 为初值，在 retained GPSWF 空间中用同一批 raw far-field channels 的残差迭代修正。

VIE 远场原始相位对应 `exp(+i C p·x)`。图7为了与 GPSWF 投影使用的
`exp(-i C p·x)` 约定一致，生成 VIE / BIM 数据时使用 `-p` 的入射/观测方向对；
因此不再通过简单复共轭来转换 VIE 数据。这样对复值 `q(x)` 不会把反差错误共轭。

BIM 诊断中：

- `relative_data_residual_before_update`：本次 BIM 更新前的相对数据残差；
- `relative_data_residual`：应用本次更新并重新求解 VIE 后的相对数据残差；
- `bim_residual_space`：BIM 残差所在空间，当前主路径为 `raw_farfield_channel`；
- `mock_distance_mean`：Analytic Born far-field 路径的 mock 节点平均距离；
- `vie_mock_distance_mean`：VIE / BIM 使用 `-p` 方向对时的 mock 节点平均距离。

输出文件：

```text
figure7_bim_gpswf_frequency.png
figure7_diagnostics.csv
figure7_diagnostics_detail.npz
figure7_residual_curves.png
figure7_diagnostic_curves.png
```

---

## 实验性结论

### 高频平台模态与离散稳定性

当前图2的数值结果显示：当频率升高时，连续 GPSWF 算子的 effective rank 会增大，`|α_{ℓn}|` 可能出现更宽的稳定平台。此时仅用

```text
|α_{ℓn}| > ε · max |α|
```

作为截断条件，会把平台内的大量 `(ℓ,n,m)` 模式全部保留下来。连续意义上这些模式可能仍属于稳定平台，但当前离散系统未必能稳定承载全部平台模态。

这里需要区分：

```text
连续稳定模态 ≠ 离散稳定模态
```

离散实验中还受到以下因素限制：

- 球内 target quadrature 点数有限；
- mock far-field nodes 与 target nodes 不完全一致；
- 极化恢复引入有限配置误差；
- `A^H W A` 只是在离散意义下近似对角；
- 高 `ℓ` 模态对节点误差和求积误差更敏感。

因此，高频下不能只依赖 `α` 阈值截断。图2、图3应在 `α` 截断之外加入离散稳定性约束，例如：

```text
GPSWF 参数截断 → α cutoff → N_cap
```

其中 `N_cap` 直接限制展开后的实际成像模式数：

```text
ψ_{ℓnm},  m = -ℓ, ..., ℓ
```

这也是图1固定 `N` 时成像相对稳定的原因：固定 `N` 本质上给离散反演加入了硬维数约束。

### 后续调参原则

图2、图3后续调参时优先检查：

```text
retained_modes
retained_ell_max
target_nodes / retained_modes
background_p95_abs
coeff_norm
```

如果 `α` 曲线在高频下接近平坦，说明 `α cutoff` 不能有效筛选稳定模态，应优先调小 `N_cap` 或加入 `ℓ` 层级约束，而不是继续单独调 `ε`。

---

## 符号与约定

| 符号 | 含义 |
|------|------|
| `ψ_{ℓ n m}` → `Mode(ell, n, m)` | ball GPSWF 函数 |
| `χ_{ℓ n}` | 三对角矩阵本征值 |
| `β_{ℓ n}` | Jacobi 展开系数 (K 维向量) |
| `α_{ℓ n}` | restricted Fourier operator 本征值（不依赖 m） |
| `J_ε = {j: |α_j| > ε·max|α|}` | epsilon 截断集合 |
| `c_j = (A^H W d)_j / |α_j|²` | GPSWF 系数（求积投影） |
| `C = 2k` (R=1) | 空间带宽积，Fourier 球半径 |
| `exp(-i C p·x)` | 统一 Fourier 约定 |
| `p = (d - x̂)/2` | 远场方向对 → Fourier 球节点 |

## 数据模式

| 模式 | 说明 | mock error |
|------|------|-----------|
| `mock` | 有限方向 → 最近邻 `p ≈ (d-x̂)/2` | >0 |
| `ideal` | `admissible_farfield_pairs_from_nodes` 精确 `p = (d-x̂)/2` | 0 |
