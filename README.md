# ISPimagingcode

反散射成像与重建数值实验项目，覆盖 Helmholtz 声学散射、有限孔径数据、直接采样/正交采样、点散射体、障碍物成像、迭代重建，Maxwell-Born / VIE 电磁散射实验，以及二维海冰空气—弹性冰层—海水正演与双界面重建。

代码以可读、可复现实验脚本为主，每个子项目独立入口。

## 环境

```bash
.venv/bin/python <subproject>/main.py
```

本项目使用根目录 `.venv/` 虚拟环境运行。当前本地环境为：

```text
Python 3.14.5
numpy 2.4.6
scipy 1.17.1
matplotlib 3.10.9
pandas 3.0.3
numba 0.68.1（海冰子项目）
torch 未在当前本地环境安装
```

核心依赖：

- `numpy`：数组计算、复数线性代数、随机噪声、`.npz` 数据保存。
- `scipy`：特殊函数、数值积分、线性方程组、SVD/GMRES、KDTree、Lebedev 球面积分节点。
- `matplotlib`：实验图像和诊断曲线输出。
- `pandas`：可选依赖，用于保存表格；未安装时部分模块会退回到轻量 CSV 实现。
- `numba`：海冰子项目交错有限差分与走时成像内核加速；依赖见 `ice_imaging_2d/requirements.txt`。
- `torch`：神经网络模块依赖，用于 U-Net 数据接口、模型训练、推理和保存加载。

神经网络代码保留在项目中，但本地不默认运行训练或推理。服务器运行神经网络模块时需要额外安装 `torch`；如果使用 GPU，还应按服务器 CUDA 版本安装匹配的 PyTorch 构建。

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
├── ice_imaging_2d/                  # 二维海冰声–弹正演、DSM / 走时成像
│   ├── ice2d/                      # 模型、求解器、成像、处理与报告
│   ├── experiments/                # 算例与敏感性实验组织
│   ├── configs/                    # compact / top_check / field
│   ├── results/                    # 随包历史结果（原样归档）
│   └── main.py                     # 新输出默认写入 ice_outputs
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

# 海冰：查看方案实现状态，不执行正演
.venv/bin/python ice_imaging_2d/main.py --estimate
# 海冰：原灵敏度对照的软件检查（非方案直接方法）
.venv/bin/python ice_imaging_2d/main.py --mode sensitivity --quick --experiment joint --workers 1 --out-dir ice_outputs/quick
```

## 输出

海冰实验默认输出写入 `ice_outputs/`，其他实验输出写入 `outputs/`。常见文件：`.png`（图像）、`.csv`（表格）、`.npz`（中间数组）。Maxwell 主图实验会额外输出 `figure*_diagnostics.csv`、`figure*_diagnostics_detail.npz` 和 `figure*_diagnostic_curves.png`，用于定位成像异常来自节点、模态、投影、数据、系数还是图像阶段。

海冰子项目的依赖安装、配置、历史数据复用和物理假设见 [ice_imaging_2d/README.md](ice_imaging_2d/README.md)。新双界面实验默认保存 `metadata.json`、上下界面/厚度误差和灵敏度诊断；legacy 模式保留 `run_manifest.json`；随包 `results/` 保留原有结果，不作为默认输出位置。

海冰方案主线目前完成Cauchy离散反传播算子，耦合背景核、补偿场和上界面指标尚未完成；原数值灵敏度/Gauss–Newton已隔离为显式对照，详见子项目README。
