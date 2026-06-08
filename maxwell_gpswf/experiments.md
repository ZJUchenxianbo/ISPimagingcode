# maxwell_gpswf 实验说明

把 `06-005` 的标量远场反演方法类推到各向异性 Maxwell 问题。主方法：**ball GPSWF 函数截断 / alpha spectral cutoff**。Fourier 约定 `exp(-i C p·x)`。

## 代码结构

```
maxwell_gpswf/
├── common/          # config, phantom, 求积, 极化, GPSWF
├── forward/         # 解析 Born + VIE (含自作用项 L=-I/3)
├── experiments/     # 三张主图
├── diagnostics/     # 模块诊断
├── main.py
└── experiments.md
```

## 运行

```bash
.venv/bin/python maxwell_gpswf/main.py --out-dir outputs                    # 全部
.venv/bin/python maxwell_gpswf/main.py --mode fig1 --data-mode ideal        # 单图+理想模式
.venv/bin/python maxwell_gpswf/main.py --quick --out-dir outputs/smoke      # 快速检查
```

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
| 总模式数 | ∑(2ℓ+1)·10 = 1352 | ∑(2ℓ+1)·6 = 486 | (ℓ_max+1)² × n_modes_per_ℓ |
| target_nodes | 12×302 = 3624 | 6×110 = 660 | n_radial × n_angular |
| ratio | **2.68** | 1.36 | nodes / modes |
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

**布局**：5 列 (truth + medium δ=0 + low + medium + high) × 4 行 (k=10, 20, 30, 40)

### GPSWF 参数联动

| k | C | ℓ_max | n_modes | K | n_radial | n_angular | modes | nodes | ratio |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 20 | 8 | 5 | 36 | 10 | 110 | 405 | 1100 | **2.72** |
| 15 | 30 | 10 | 5 | 44 | 12 | 170 | 605 | 2040 | **3.37** |
| 20 | 40 | 14 | 6 | 54 | 16 | 302 | 1350 | 4832 | **3.58** |
| 25 | 50 | 16 | 7 | 60 | 14 | 434 | 2023 | 6076 | **3.00** |

### 截断

三层：GPSWF 参数 → ε=0.2 → N_cap=1000

### 参数

| 参数 | full | quick | 说明 |
|------|------|-------|------|
| k | 10, 15, 20, 25 | 10, 15 | |
| 入射/观测方向数 | 110 | 38 | |
| grid_size | 81×81 | 51×51 | |
| 噪声 δ | 0.2 | 同 | |
| 对比度 scale | 0.3, 1.0, 3.0 | 同 | low / medium / high |
| ε | 0.2 | 同 | `|α| > ε·max|α|` |
| N_cap | 1000 | — | 第三层硬截断 |
| 张量类型 | full (9 维) | 同 | |
| 数据分量 | Q_11 | 同 | |
| α quad_order | 160 | 100 | |
| α r_eval_count | 120 | 80 | |

**截断**：三层 — GPSWF 参数 → ε=0.2 → N_cap=1000。

---

## 图3: 数据来源与散射体形状

**目的**：比较 Full VIE / VIE Born / Analytical Born 三种数据源在 5 种散射体上的反演。

**布局**：4 列 (truth + Full VIE + VIE Born + Analytical Born) × 5 行

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
| n_per_axis (VIE) | 11 (~739 体素) | 5 (~81 体素) | VIE 网格分辨率 |
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

**截断**：三层 — GPSWF 参数 → ε=0.1 → N_cap=600。

### 5 种散射体

| 行 | 名称 | 描述 | 解析 Fourier 公式 |
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
