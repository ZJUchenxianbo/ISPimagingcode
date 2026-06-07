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

**目的**：观察不同波数和对比度下 Born 重建质量。ε 截断随 C 自适应。

**布局**：4 列 (truth + low/medium/high) × 4 行 (k=10, 20, 30, 40)

### 参数

| 参数 | full | quick | 说明 |
|------|------|-------|------|
| k | 10, 20, 30, 40 | 10, 15 | 4 行对应 4 个波数 |
| C = 2k | 20, 40, 60, 80 | 20, 30 | |
| 入射/观测方向数 | 110 | 38 | |
| n_radial | 12 | 6 | |
| 目标角向点数 | 230→302 | 110 | |
| grid_size | 81×81 | 51×51 | |
| K | 60, 70, 80, 90 | 40 | 每行 K = 60 + row_idx×10 |
| ℓ_max | 18 | 12 | |
| n_modes_per_ℓ | 10 | 8 | |
| 噪声 δ | 0.2 | 同 | 固定 |
| 对比度 scale | 0.3, 1.0, 3.0 | 同 | low / medium / high |
| **截断方式** | **ε = 0.1** | 同 | `|α| > 0.1 max|α|` |
| 张量类型 | full (9 维) | 同 | |
| 数据分量 | Q_11 | 同 | |
| α quad_order | 160 | 100 | |
| α r_eval_count | 120 | 80 | |

**ε 截断效果**：保留模式数随 C³ 增长。k=10 (C=20) 约 250 个，k=40 (C=80) 约 1600 个。

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
| n_radial | 10 | 6 | |
| 目标角向点数 | 170 | 86 | |
| grid_size | 81×81 | 51×51 | |
| n_per_axis (VIE) | 11 (~739 体素) | 5 (~81 体素) | VIE 网格分辨率 |
| K | 50 | 30 | |
| ℓ_max | 12 | 8 | |
| n_modes_per_ℓ | 7 | 5 | |
| 总模式数 | ∑(2ℓ+1)·7 = 637 | ∑(2ℓ+1)·5 = 189 | |
| **截断方式** | **ε = 0.1** | 同 | `|α| > 0.1 max|α|` |
| 张量类型 | isotropic | 同 | |
| α quad_order | 140 | 60 | |
| α r_eval_count | 100 | 50 | |

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
