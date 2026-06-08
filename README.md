# ISPimagingcode

反散射成像与重建数值实验项目，覆盖 Helmholtz 声学散射、有限孔径数据、直接采样/正交采样、点散射体、障碍物成像、迭代重建，以及 Maxwell-Born / VIE 电磁散射实验。

代码以可读、可复现实验脚本为主，每个子项目独立入口。

## 环境

```bash
.venv/bin/python <subproject>/main.py
```

神经网络代码保留在项目中，本地只做轻量检查，不默认运行训练或推理。

## 项目结构

```
pyproj/
├── common/                          # 共享库 (Layer 0-1)
│   ├── scattering.py                # 常量、类型、噪声、方向向量
│   ├── forward.py                   # BEM / Born / 点散射体求解器
│   ├── sampling.py                  # 直接采样指标、孔径工具
│   ├── targets.py                   # 合成目标案例
│   └── unet.py                      # U-Net 模型
├── obstacle_imaging/                # 障碍物成像
│   ├── lib/
│   │   ├── direct_sampling.py       # 直接采样公共流程
│   │   └── reconstruction.py        # Gauss-Newton / MUSIC 重建
│   ├── experiments/                 # 实验脚本
│   │   ├── direct_imaging.py
│   │   ├── hybrid_imaging.py
│   │   ├── joint_gn.py
│   │   ├── limited_aperture_imaging.py
│   │   ├── prior_sensitivity.py
│   │   └── apple_imaging.py
│   └── main.py
├── point_scatterer/                 # 点散射体成像
│   ├── experiments/imaging.py
│   └── main.py
├── limited_aperture/                # 有限孔径分析
│   ├── experiments/coherence.py
│   └── main.py
├── maxwell_gpswf/                  # Maxwell 远场反演 (GPSWF)
│   ├── common/                    # phantom、求积、极化、GPSWF、运行时诊断
│   ├── forward/                   # 解析 Born / VIE 正向求解器
│   ├── experiments/               # 主图实验
│   └── main.py
├── AGENTS.md
└── README.md
```

## 运行

```bash
# 障碍物成像
.venv/bin/python obstacle_imaging/main.py --mode all

# 点散射体
.venv/bin/python point_scatterer/main.py

# 有限孔径
.venv/bin/python limited_aperture/main.py

# Maxwell 实验
.venv/bin/python maxwell_gpswf/main.py --out-dir outputs
```

## 输出

实验输出写入 `outputs/` 目录，分子项目存放。常见文件：`.png`（图像）、`.csv`（表格）、`.npz`（中间数组）。Maxwell 主图实验会额外输出 `figure*_diagnostics.csv`、`figure*_diagnostics_detail.npz` 和 `figure*_diagnostic_curves.png`，用于定位成像异常来自节点、模态、投影、数据、系数还是图像阶段。
