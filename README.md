# ISPimagingcode

这是一个用于反散射成像与重建数值实验的 Python 项目。代码以可读、可复现实验脚本为主，覆盖 Helmholtz 声学散射、有限孔径数据、直接采样/正交采样、点散射体、小障碍物成像、迭代重建，以及 Maxwell-Born / VIE 相关实验。

## 环境

项目使用本地虚拟环境 `.venv/`，运行命令统一使用：

```bash
.venv/bin/python script_name.py
```

神经网络相关代码保留在项目中，但本地通常只做轻量检查，不默认运行训练或推理。

## 主要实验脚本

1. `point_scatterer_imaging.py`：点散射体解析远场、有限孔径 raw DSM 成像比较，以及可选 U-Net 后处理。常用参数：`--case`、`--no-unet`、`--aperture-half-widths`。
2. `obstacle_direct_imaging.py`：全孔径障碍物直接采样/正交采样成像，目标中心随机生成。
3. `obstacle_limited_aperture_imaging.py`：有限孔径障碍物直接采样成像。可用 `--cases small_target,small_cluster,large_target` 或 `--cases all` 选择目标案例。
4. `obstacle_joint_gn.py`：星形声软障碍物的联合 Gauss-Newton 参数重建实验。
5. `obstacle_prior_sensitivity.py`：测试初始先验信息对迭代重建结果的影响。
6. `obstacle_hybrid_imaging.py`：先用直接成像提取先验，再做迭代定量重建。
7. `apple_obstacle_imaging.py`：苹果形障碍物成像实验。
8. `limited_aperture_coherence.py`：有限孔径核函数层实验，包括单孔径宽度、双孔径 product coherence、stationary-sector 分类、Gram 条件数、噪声稳定性、孔径中心设计扫描和三维 spherical-cap 因子。

示例：

```bash
.venv/bin/python point_scatterer_imaging.py --no-unet
.venv/bin/python obstacle_limited_aperture_imaging.py --cases all
.venv/bin/python limited_aperture_coherence.py --experiments widths,gram,cap3d
```

## 公共模块

1. `scattering_common.py`：项目依赖根模块，提供常量、类型别名、方向向量、噪声工具、SNR 和命令行解析。
2. `forward_scattering.py`：统一正向散射求解器，包含障碍体 BEM、可穿透介质 Born/Lippmann-Schwinger 和点散射体模型。
3. `sampling_imaging.py`：直接采样/正交采样指标、有限孔径权重、分块网格计算和指标图绘制。
4. `target_cases.py`：合成目标案例，包括不可穿透障碍体、可穿透介质、点散射体，以及星形障碍物几何工具。
5. `obstacle_direct_sampling.py`：障碍物 clean/noisy 直接采样成像公共流程。
6. `obstacle_reconstruction.py`：障碍物定量重建算法，包括 MUSIC 指标、峰值选择、约束投影、Gauss-Newton 迭代和随机中心生成。
7. `unet_imaging.py`：点散射体成像脚本使用的 U-Net 工具。

## Maxwell 子项目

`pyproj-electromagnet/` 用 `06-005` 的低秩数值实验思路类推到各向异性 Maxwell 远场反演。主方法是 ball GPSWF 函数截断 / alpha spectral cutoff，Fourier 约定为 `exp(-i C p·x)`。

当前结构：

```text
pyproj-electromagnet/
├── common/       # phantom、求积、极化矩阵、GPSWF、工具函数
├── forward/      # 解析 Born 数据和 Maxwell VIE 正向原型
├── experiments/  # 三张主图实验
├── diagnostics/  # 极化、噪声放大、GPSWF、截断诊断
├── main.py       # 总入口
└── experiments.md
```

三张主图：

1. `figure1_noise_dimension.py`：噪声水平和 GPSWF 截断维数对重构的影响。
2. `figure2_frequency_contrast.py`：频率和介质对比度对重构的影响。
3. `figure3_sources_shapes.py`：不同数据来源和散射体形状对比，包括 Full VIE、VIE Born 和 Analytical Born。

运行示例：

```bash
# 运行全部 Maxwell 图和诊断
.venv/bin/python pyproj-electromagnet/main.py --out-dir outputs

# 只运行某张图
.venv/bin/python pyproj-electromagnet/main.py --mode fig1 --out-dir outputs
.venv/bin/python pyproj-electromagnet/main.py --mode fig2 --out-dir outputs
.venv/bin/python pyproj-electromagnet/main.py --mode fig3 --out-dir outputs

# 只运行模块诊断
.venv/bin/python pyproj-electromagnet/main.py --mode diagnostics --out-dir outputs

# 快速检查
.venv/bin/python pyproj-electromagnet/main.py --quick --out-dir outputs/smoke
```

更详细的 Maxwell 子项目说明见 `pyproj-electromagnet/experiments.md`。

## 输出

实验输出一般写入 `outputs*` 目录，常见文件包括：

- `.png`：成像图、误差图、诊断图。
- `.csv`：误差、条件数、截断维数等表格。
- `.npz`：节点、远场数据、重构数组等中间结果。
- `metadata.json`：部分实验的参数记录。
