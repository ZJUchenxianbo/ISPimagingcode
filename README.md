# ISPimagingcode

这是一个用于学习和实验反散射成像/重建问题的 Python 代码库。后续相关数值实验代码都会尽量放在这里，包括直接采样/正交采样、有限孔径成像、迭代重建、点散射体、障碍物散射，以及简单的神经网络后处理。

## 主要脚本

1. `obstacle_joint_gn.py`：星形声软障碍物的联合 Gauss-Newton 参数重建实验。
2. `obstacle_prior_sensitivity.py`：测试初始先验信息对迭代重建结果的影响。
3. `obstacle_direct_imaging.py`：全孔径直接采样/正交采样成像，目标中心随机生成。
4. `obstacle_limited_aperture_imaging.py`：有限孔径障碍物直接采样成像。可用 `--cases small_target,small_cluster,large_target` 或 `--cases all` 选择目标案例。
5. `point_scatterer_imaging.py`：点散射体解析远场、不同孔径 raw DSM 成像比较，以及可选的 U-Net 后处理。可用 `--case three_point_scatterers` 选择目标案例，用 `--no-unet` 跳过学习后处理，用 `--aperture-half-widths` 指定孔径半角列表。
6. `obstacle_hybrid_imaging.py`：先用直接成像提取先验，再做迭代定量重建。
7. `apple_obstacle_imaging.py`：苹果形障碍物成像实验。
8. `pyproj-electromagnet/`：用 `06-005` 的数值试验方法类推到各向异性 Maxwell-Born 远场反演问题，使用 Gauss-Jacobi 径向节点和 Lebedev 角向节点生成 mock-quadrature，重点生成极化恢复和 GPSWF 截断截面重构图像；另含小规模 Maxwell VIE 正向求解器原型和 VIE 数据反演试验。`main.py` 默认运行 GPSWF 主 pipeline，也可用 `--mode diagnostics`、`--mode forward`、`--mode vie-recon` 分别运行模块检查、VIE 原型和 VIE 数据反演。运行示例：`.venv/bin/python pyproj-electromagnet/main.py --out-dir pyproj-electromagnet/outputs_section8_diagnostics`。
9. `limited_aperture_coherence.py`：有限孔径论文的核函数层实验，计算单孔径宽度、双孔径 product coherence、stationary-sector 分类、Gram 条件数、噪声稳定性、孔径中心设计扫描和三维 spherical-cap 因子；输出 `.npz`、`.png` 和 `metadata.json`。

示例：

```bash
.venv/bin/python limited_aperture_coherence.py --experiments widths,gram,cap3d
```

## 公共模块

1. `scattering_common.py`：项目依赖根模块，提供数学常量（PI2）、类型别名（Array/CArray）、方向向量、噪声工具、SNR 和命令行解析。
2. `forward_scattering.py`：统一正向散射求解器，包含障碍体 BEM（单层/双层/组合场）、可穿透介质（Born/Lippmann-Schwinger）和点散射体（独立/Foldy-Lax）。
3. `sampling_imaging.py`：直接采样/正交采样指标、有限孔径权重、分块网格计算和指标图绘制。
4. `target_cases.py`：可复用的合成目标案例（不可穿透障碍体、可穿透介质、点散射体），以及星形障碍物几何参数化、边界离散化和绘图工具。
5. `unet_imaging.py`：点散射体成像脚本使用的 U-Net 工具（训练/推理/保存）。
6. `obstacle_direct_sampling.py`：障碍物 clean/noisy 直接采样成像的公共流程。
7. `obstacle_reconstruction.py`：共享的障碍物定量重建算法，包括 MUSIC 指标、峰值选择、约束投影、Gauss-Newton 迭代、中心距离评估和随机中心生成。
