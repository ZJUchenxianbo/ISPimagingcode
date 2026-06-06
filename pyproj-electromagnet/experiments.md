# pyproj-electromagnet 实验说明

把 `06-005` 的标量远场反演方法类推到各向异性 Maxwell 问题。主方法：**ball GPSWF 函数截断 / alpha spectral cutoff**。Fourier 约定统一为 `exp(-i C p·x)`。

## 代码结构

```
pyproj-electromagnet/
├── common/          # 公共模块（phantom, 求积, 极化, GPSWF, 工具）
├── forward/         # 正向求解器（解析 Born + VIE 含自作用项）
├── experiments/     # 三张主图实验（一个文件一个实验）
├── diagnostics/     # 模块诊断
├── main.py          # 总入口
└── experiments.md
```

## 三张主图

### 图1: 噪声和截断维数的影响
`experiments/figure1_noise_dimension.py`

4行(N=5,72,144,256) × 5列(truth + 噪声0,0.1,0.2,0.3)。Born数据, k=15。

### 图2: 频率和介质对比度的影响
`experiments/figure2_frequency_contrast.py`

4行(k=10,20,30,40) × 4列(truth + low/medium/high contrast)。Born数据, 噪声0.2。

### 图3: 数据来源和散射体形状对比
`experiments/figure3_sources_shapes.py`

4行(球/立方体/双球+立方体/分散不规则体) × 4列(truth + Full VIE + VIE Born + Analytical Born)。k=15。

## 模块诊断

| 脚本 | 内容 |
|------|------|
| `diagnostics/polarimetric_conditioning.py` | M(p) rank、奇异值、条件数 |
| `diagnostics/noise_amplification.py` | pinv(M) 噪声放大 |
| `diagnostics/gpswf_residuals.py` | GPSWF 三对角残差、alpha 衰减 |
| `diagnostics/modal_cutoff.py` | Alpha spectral cutoff 稳定性 |

## 运行

```bash
# 全部
.venv/bin/python pyproj-electromagnet/main.py --out-dir outputs

# 单张图 / 仅诊断
.venv/bin/python pyproj-electromagnet/main.py --mode fig1 --out-dir outputs
.venv/bin/python pyproj-electromagnet/main.py --mode diagnostics --out-dir outputs

# Smoke test
.venv/bin/python pyproj-electromagnet/main.py --quick --out-dir outputs/smoke

# 单独运行
.venv/bin/python pyproj-electromagnet/experiments/figure1_noise_dimension.py --out-dir outputs/figures
```

## Fourier 约定
统一 `exp(-i C p·x)`。VIE 远场数据经 `vie_to_fourier_convention()`（取共轭）转换后送入 GPSWF 管道。
