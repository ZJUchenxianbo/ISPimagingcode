# 正演核心来源与迁移边界

来源为用户提供的 `IceInvProblem-main.zip`，SHA-256：
`260b9be90385da30a0fa74179477386bbba0263a84f3393bff5d8bf08eb6f8c8`。
压缩包未包含 LICENSE；这里记录来源，不补写或推断外部许可证。

原 `solver/forward/solve.py` 的几何、耦合系统及 CPU/GPU 求解类提取至
`coupled.py`，删除原 CLI、文件锁、Linux 资源统计及绘图编排。
`water.py`、`ice.py`、`trace.py`、`condensed.py`、`uncondensed.py`、
`iterative.py` 和可选 MUMPS 接口来自原同名正演模块；`basis.py` 仅提取
`elastic.py` 的 Lagrange 基函数，`air.py` 仅保留平层弹性响应预条件工具。
`quadrature.py` 的 CQ 网格来自原 `receiver.py`，`frequency.py` 来自
`solver/experiments/frequency.py`。没有导入轴对称求解器或原反演算法。

`gpu/` 保留原二维正演 CUDA/C++ 实现及包装：BEM 装配、冰层面板 Schur
消元、cuDSS、弹性矩阵布局、水域耦合求解和常驻工作区。`build.sh` 默认
只编译 `bem schur water`，不要求形状导数核；架构必须通过
`PLANAR_CUDA_ARCH` 显式指定。备用 trace/cuDSS 包装保留其原构建函数。

本项目新增 `config.py`、`adapter.py`、`receiver.py`：材料参数显式传入
网格设计、PML 和耦合系统；接收点 +z 导数解析微分层势核，使用与声压相同
的远/近场求积。CPU/GPU 共用该接收评价；GPU 已解出的边界量直接复用，
导数评价目前在 CPU 进行。近场求积切换处仍应结合网格加密检查误差。

源采用 `(c_w^-2 d_tt - Laplacian)p=q(t)delta(x-y)` 的二维线源约定。
默认新增高斯调制短脉冲，同时保留原平滑启动载波。两者都由 CQ 阻尼、
解调和带内投影重建；`source_signal` 是相同重建链路下的有效脉冲。
Gaussian脉冲在启动前保留至少4-sigma余量，并检查带内投影相对原脉冲的
失真，超过5%则拒绝正演。CQ重建频带与后处理成像频带分开；否则复频
轮廓上的硬截断经指数补偿可能产生显著晚时振铃。正式实验配置进一步
要求源投影误差不超过1%，这不代表传播或成像误差。
不沿用旧有限差分应力率源的 `1e4`、时间导数或 `1/c_w^2` 转换因子。
TD-DSM 核使用记录中的有效源脉冲并做相同带通；CQ 离散色散、频带截断
与连续 Green 核之间的误差仍需时间步、带宽与网格收敛验证。

物理区别：下方为开域/PML，不是旧有限差分刚性海底。GPU 路径支持
`zero-traction`，显式空气耦合走 CPU 标量收发路径。CPU/GPU 数据均为
`(source, receiver, time)`；共址线源的总声压奇异，当前显式拒绝。
`direct/scattered` 两组声压和导数独立保存，避免重复扣除直达波。

小例检查记录只证明迁移和接口一致性，不是实际 10-20 kHz 成像精度证明。
GPU 必须在目标 Linux CUDA/cuDSS 环境编译后运行 `ice_imaging_2d.tests.check_forward --gpu`
进行独立验收；CPU 检查通过不意味着 GPU 已运行通过。
