# 旧声学 DSM / 走时基线说明

本文描述 legacy 模式及归档实验；默认入口现为双界面模式，见 [当前说明](../README.md)。

项目由 `main.py` 统一运行。默认计算声软界面对照、平坦弹性冰层、正弦粗糙弹性冰底，生成正演数据、反演图、界面曲线、误差、敏感性图和中文结果报告。随包历史结果原样保留在 `ice_imaging_2d/results/compact/` 和 `ice_imaging_2d/results/top_shift/`，它们不是本次迁移重新计算的结果。

## 安装与运行

从 `pyproj` 根目录运行，统一使用已有 `.venv/`（Python 3.14），不需要在子项目新建环境。

```bash
.venv/bin/python -m pip install -r ice_imaging_2d/requirements.txt
# 先查看规模，不执行正演或写入结果
.venv/bin/python ice_imaging_2d/main.py --mode legacy --estimate
# 完整紧凑实验（含加密检查，运行时间较长）
.venv/bin/python ice_imaging_2d/main.py --mode legacy --workers 2 --refine
```

本次迁移采用 Numba 0.68.1 / llvmlite 0.50.0 的 Python 3.14 安装包。
Numba 首次实际运行需要编译内核，之后可重用缓存。`--help` 和 `--estimate` 不加载数值内核。
项目不含神经网络。完整实验与 `--refine` 不属于本地轻量验证范围。

```bash
# 只运行正弦冰底，含 DSM、走时成像及敏感性实验
.venv/bin/python ice_imaging_2d/main.py --mode legacy --case sine --workers 2 --output ice_outputs/sine

# 重新处理本地此前生成的数据，配置必须和数据一致
.venv/bin/python ice_imaging_2d/main.py --mode legacy --case suite --reuse-data --output ice_outputs

# 快速查看全波数据上的走时成像
.venv/bin/python ice_imaging_2d/main.py --mode legacy --case sine --skip-dsm --output ice_outputs/sine_migration

# 局部凸起、多尺度起伏
.venv/bin/python ice_imaging_2d/main.py --mode legacy --case bump --output ice_outputs/bump
.venv/bin/python ice_imaging_2d/main.py --mode legacy --case multiscale --output ice_outputs/multiscale

# 冰面真值提高到 0.30 m，冰底形状相同
.venv/bin/python ice_imaging_2d/main.py --mode legacy --case top_shift --output ice_outputs/top_shift

# 重现随包提供的三炮冰面平移检查
.venv/bin/python ice_imaging_2d/main.py --mode legacy --config ice_imaging_2d/configs/top_check.json --case top_shift --skip-dsm --workers 1 --output ice_outputs/top_shift_check

# 平均厚度 4 m，相应移动水下阵列和海底
.venv/bin/python ice_imaging_2d/main.py --mode legacy --case thick --output ice_outputs/thick

# 10 m 深阵列的大规模参数，需要更长运行时间与更多内存
.venv/bin/python ice_imaging_2d/main.py --mode legacy --config ice_imaging_2d/configs/field.json --estimate
.venv/bin/python ice_imaging_2d/main.py --mode legacy --config ice_imaging_2d/configs/field.json --workers 1 --output ice_outputs/field
```

默认新输出为 `ice_outputs/`；`--out-dir` 是 `--output` 的同义参数，路径相对于运行时工作目录。配置文件路径同样相对于工作目录。

复用随包历史数据时，先复制到新的输出目录，避免覆盖归档结果：

```bash
mkdir -p ice_outputs/reimage_compact
cp ice_imaging_2d/results/compact/config.json ice_imaging_2d/results/compact/data_*.npz ice_outputs/reimage_compact/
.venv/bin/python ice_imaging_2d/main.py --mode legacy --config ice_imaging_2d/configs/compact.json --case sine --reuse-data --out-dir ice_outputs/reimage_compact
```

新运行同时保存 `metadata.json` 和兼容的 `run_manifest.json`，记录配置、请求算例、数据复用状态、依赖版本及耗时。原归档元数据不做修改。

同一输出目录只接受同一套配置。改变参数应指定新目录，以保持数据与记录的一致性。正演按炮保存缓存，意外中断后用原命令重跑，会重用已经完成的炮。`--reuse-data` 要求该算例和背景的 `data_*.npz` 已存在。

`--estimate` 会打印网格规模、实际时间采样、单算例原始数组体积与后处理数组的粗略内存估计。大规模设置的后处理需要数十 GB 内存，当前程序按完整炮集处理，需准备相应资源或减少源数和接收范围。紧凑算例适合先检查实现。

## 文件职责

历史迁移保留原模型；本轮仅在此基线上修正实际非均匀测线求积，并增加独立双界面算法。旧数据仍保留。`ice2d/` 是子项目公共数值层，`experiments/` 负责实验组织，不依赖其他子项目。

| 文件 | 作用 |
|---|---|
| `main.py` | 参数读取、配置变体、规模估计与统一入口 |
| `ice2d/config.py` | SI 单位参数、基本几何与稳定性约束 |
| `ice2d/model.py` | 真值几何、材料分布、错位网格、阵列和吸收层 |
| `ice2d/solver.py` | Numba 加速的二维声–弹速度应力正演、多炮与断点缓存 |
| `ice2d/processing.py` | 背景扣除、10–20 kHz 带通、预设首反射窗、可重复噪声 |
| `ice2d/imaging.py` | 两种走时叠加、Cauchy 数据 DSM、镜像补偿、冰面高度扫描 |
| `ice2d/metrics.py` | 不依赖真值的逐列取峰，随后比较误差 |
| `ice2d/plotting.py` | 模型、炮集、反演图、波场和敏感性图 |
| `ice2d/validation.py` | 解析直达波、平层到时、互易性、网格加密检查 |
| `experiments/run.py` | 算例、敏感性实验与运行调度，保存 NPZ、CSV、JSON |
| `ice2d/io.py`、`ice2d/reporting.py` | 数组读取与中文结果报告 |
| `configs/compact.json` | 已执行紧凑实验的完整参数 |
| `configs/field.json` | 接近前一轮讨论的较大计算区域 |

## 模型、边界与首轮假设

物理坐标为 $(x,z)$，$z$ 向上。冰域满足 $f(x)<z<a$，水域满足 $-D<z<f(x)$。默认 $a=0$，$D=5.5$ m。冰为二维平面应变各向同性弹性介质，保留面内 P、SV 波。海水用剪切模量为零的速度应力系统。

冰内本构参数为 $\mu_i=\rho_i c_S^2$、$\lambda_i=\rho_i(c_P^2-2c_S^2)$。连续方程为 $\rho v_t=\nabla\cdot\sigma$、$\sigma_t=\lambda(\nabla\cdot v)I+2\mu\varepsilon(v)$，海水中的声压为 $p=-\sigma_{xx}=-\sigma_{zz}$。水中各向同性应力率源加到两个法向应力分量。

冰水界面对应牵引连续、法向速度连续，流体侧切向牵引为零。离散采用错位网格上的密度平均和四角剪切模量调和平均。当一个角点接触流体时，该剪切模量取零。几何通过材料网格实现，界面存在阶梯误差。

**首轮采用冰面零牵引近似，空气域的声传播被省略。** 冰面以上的零质量、零刚度网格用于实现数值自由表面。这里用二阶时间、四阶空间交错有限差分替代上一轮建议的贴体谱元法，使整个项目可以由 Python/Numba 直接执行。自由表面和材料界面会降低局部离散精度，需要单独检查网格收敛。

海底是已知平坦刚性边界，法向速度为零。左右边界为三次增长的阻尼层。背景模型移除冰层，水域向上延伸到顶部吸收层。阻尼层近似截断水平无界域，属于有限区域数值实验。

首轮将“粗糙尺度 20 cm”解释为竖向峰谷差，水平周期取 4 m，以便在较小区域内观察一个波峰和一个波谷。大规模配置另取 20 m 周期。这一选择需要与实际粗糙尺度的定义核对。

| 参数 | 紧凑实验 | 大规模配置 |
|---|---:|---:|
| 平均冰厚 | 1.5 m | 1.5 m |
| 冰底正弦振幅 / 周期 | 0.10 m / 4 m | 0.10 m / 20 m |
| 水中声速 / 密度 | 1500 m/s / 1025 kg/m³ | 相同 |
| 冰中 P、S 波速 / 密度 | 3600、1800 m/s / 920 kg/m³ | 相同 |
| 阵列深度 / 海底深度 | 3 m / 5.5 m | 10 m / 25 m |
| 横向计算范围 | [-5,5] m | [-20,20] m |
| 发射数 | 7 | 49 |
| 发射横向范围 | [-1.8,1.8] m | [-12,12] m |
| 接收横向范围 / 间距 | [-2.4,2.4] m / 0.04 m | [-12,12] m / 0.03 m |
| 网格间距 | 0.01 m | 0.01 m |
| 记录时长 | 4.5 ms | 25 ms |
| 输出采样间隔 | 约 4 μs | 约 4 μs |
| 冰底扫描范围 | [-1.6,1.6] × [-1.9,-1.1] m² | [-6,6] × [-2,-1] m² |

实际收发位置按最近网格中心吸附，精确坐标保存在数据文件。内部时间步为 $\Delta t=0.32h/c_P$，紧凑实验约为 0.889 μs。输出步数取整数，所以应读取数据里的 `time`，不要假定恰好为 4 μs。

## 实验类别和回答的问题

| 算例 / 处理 | 目的 | 输出中应观察的量 |
|---|---|---|
| `soft` | 在粗糙声软面上检查原文型指标 | 峰谷位置、补偿前后差异、补偿本身的亮带 |
| `flat` | 检查声–弹正演和平面定位 | 冰底首反射、冰面 P 回波、平面深度误差 |
| `sine` | 恢复 20 cm 峰谷差 | 真值与提取曲线、RMSE、峰谷差、扫描边缘取峰比例 |
| `bump` | 单个局部突起 | 宽度、幅值、两侧伪影 |
| `multiscale` | 两种水平尺度叠加 | 小尺度起伏是否被抹平或产生伪峰 |
| `top_shift` | 冰面提高 30 cm | P 回波时延变化、冰面高度扫描 |
| `thick` | 平均 4 m 厚冰 | 较长冰内传播、多次反射对成像的影响 |
| 20、10 dB 噪声 | 固定残差数据的抗噪声能力 | 相同参数下 RMSE 与峰谷差 |
| 水声速 ±2% | 背景模型误差 | 绝对深度偏移及形状变化 |
| 局部孔径减半、接收抽稀 | 观测限制 | RMSE、连续界面和覆盖度 |
| 1 cm / 0.5 cm 网格 | 单炮正演离散误差 | 首反射波形差及到时差 |

`suite` 实际执行前三类，并在 `sine` 上计算全部敏感性处理。其余几何类别用上面的独立命令执行。每次运行报告只列出此次实际完成的结果。

## 正演数据来源和独立性

本项目的“观测数据”由 `solver.py` 数值求解波动方程生成。真实界面只在 `model.py` 中定义。冰内多次反射、模式转换和海底回波由离散方程自行产生，没有预先写入反射走时。全部材料无衰减，且数据为合成数据。

背景记录使用相同源、阵列、时间步、网格和海底。扣除背景后，平坦冰底自身的反射仍保留。首反射窗仅使用已知平均深度、收发几何和水声速，每道单独设置。窗中央平坦，边缘用余弦渐降。

真实界面不会传入 `migrate` 或 `dsm_energy`。评价阶段才计算真实曲线。保留平均深度、搜索区间、已知波速和局部平行假设这些先验。噪声实验在残差上加入确定种子的带限高斯噪声，并在首反射窗内控制全数据二范数 SNR。背景扣除误差尚未被模拟。

## 反演公式与实现含义

**走时叠加图。** 对候选点 $q$，计算 $\tau_{rs}(q)=(|x_s-q|+|x_r-q|)/c_w+t_0$。这里 $t_0$ 为源包络中心时刻。读取经过带通和选窗的记录在该时刻的值，并在局部收发孔径内加权叠加。包络版本取各道解析信号模值再叠加，相干版本先叠加解析信号再取模。两者都除以权重和，减小照明覆盖变化。成像网格与正演网格不同。

这给出一个可以检验全波记录是否支持冰底定位的参照方法。它依赖单次水中反射走时、已知水声速和局部孔径，在复杂模式重叠时可能偏移。

**Cauchy 数据反传播图。** `dsm_energy` 按原文式 (4.4) 的两个卷积项计算 $V_s$，再在正相关时延上积分 $|V_s|^2$。FFT 通过充分补零计算相关，二维核采用 Hankel 函数，使用与 NumPy 傅里叶符号一致的第二类 Hankel 函数。记录的梯度为 $+z$ 方向，计算下方测线时换成向下法向。

镜像测线关于预先给定的平均冰底高度反射，补偿按原文式 (5.1) 离散。图中同时输出无补偿、带补偿、只有补偿三项。使用已知源率对应的源谱归一化，以保留补偿与数据项的相对尺度。

`dsm_impedance` 额外检验近正入射的声阻抗修正。设已知半空间反射系数 $R_0=(\rho_i c_P-\rho_w c_w)/(\rho_i c_P+\rho_w c_w)$，把原文的 $V_s+C$ 改成 $V_s-R_0C$ 后计算能量。声软情形 $R_0=-1$ 时回到原文组合。该修正忽略斜入射反射系数、横波转换及有限厚度的频率响应，适合作为首反射窗内的候选对照。其结果单独保存在 `*_dsm_impedance.png`，两图分别归一化。

首反射时间窗、有限测线、参考平面与弹性冰层构成当前的数值迁移版本。原文定理针对声软界面。**只有结果中实际出现准确的界面峰，才可据此评价这一版本的数值表现。** 如声软算例也失败，应先分析公式离散、截断和补偿，再讨论弹性推广。

**上表面走时图。** 从成像得到冰底后，利用近零偏移记录中的 P 回波，在预设 $a\in[-0.4,0.6]$ m 上扫描。候选冰面回波时刻为 $t_b+2(a-\widehat f)/c_P$。它是局部平行近似下的 P 回波走时扫描。各道评分分别归一化后求平均，同时报告各道高度估计的离散程度。水中波速、冰中 P 波速与模式辨认误差都会影响结果。

## 怎样理解生成的文件

| 文件 | 内容及用途 |
|---|---|
| `sine_inversion.png` | 两种走时叠加反演图，青线是真值，白虚线是逐列取峰。右图直接比较界面深度 |
| `sine_dsm.png` | 原文型指标及镜像补偿的对照，三幅图共用能量色标 |
| `sine_records.png` | 中央炮总场、背景扣除后的数据、选窗和中央道回波 |
| `sine_wavefields.png` | 波在水、冰中的传播，以及冰中的剪切运动；界面附近也可能含离散误差 |
| `sine_top_scan.png` | 近零偏移 P 回波对候选冰面高度的评分 |
| `robustness.png` | 同一组粗糙冰底数据的噪声、波速、孔径敏感性 |
| `profile_sine.csv` | 各横坐标下的真实冰底和各方法提取值 |
| `metrics_sine.json` | RMSE、最大误差、偏差、峰谷差和边界取峰比例 |
| `validation.json` | 解析波形、到时、互易及可选网格检查结果 |
| `RESULTS_zh.md` | 本次运行的中文结果与限制说明 |

颜色表示成像泛函大小。图像独立归一化时，颜色不能用于跨方法比较实际幅度。DSM 对照图采用共享尺度。成像步长 1 cm 是搜索采样密度，测量精度应由误差和离散收敛决定。

逐列取最大值后使用三点二次插值，不进行横向平滑。真值只参与最终误差计算。若大部分列在扫描区域顶端或底端取峰，`boundary_fraction` 会较高，这通常表明该成像泛函没有在界面处形成所需的峰。

平面估计的对照误差 `constant_depth_rmse_m` 防止“恢复了一条近似平线”被误判成恢复了起伏。噪声实验应结合这个对照以及峰谷差一起看。

## 数据格式

```python
import numpy as np

data = np.load("ice_imaging_2d/results/compact/data_sine.npz", allow_pickle=False)
pressure = data["pressure"]      # shape (source, receiver, time)
gradient = data["gradient"]      # physical +z pressure derivative, same shape
time = data["time"]              # seconds
source_x = data["source_x"]      # metres
receiver_x = data["receiver_x"]  # metres
array_z = float(data["array_z"])

images = np.load("ice_imaging_2d/results/compact/images_sine.npz", allow_pickle=False)
image = images["migration"]     # shape (z sample, x sample)
x, z = images["x"], images["z"]
```

总场与背景是分别保存的，原始记录不因成像而改写。数据文件保存完整配置、算例名称和求解器版本。`cache/` 为可以删除并重建的中间缓存，归档包省略重复缓存。

## 需要继续检验的范围

首轮验证涉及较小二维区域、平海底和零牵引冰面。空气声学加载、冰的吸收与各向异性、粗糙海底、未知材料参数、现场声源和接收器响应需要扩展正演模型。4 m 厚冰在无衰减模型中的可见性不能推断为真实海冰中的信噪比。

当前网格加密检查只比较平冰层中央一道。粗糙界面的全炮集、侧边界距离、阻尼强度、记录长度以及空间采样还需系统收敛实验。`configs/field.json` 的运算规模显著增大，已有紧凑算例的结果不能替代它。

输出的法向压力梯度采用接收位置上下两点的二阶中心差分。声压主求解器使用四阶空间格式，Cauchy 数据中的梯度精度需要通过加密网格另行评估。

## 参考来源

Jean Virieux, “P-SV wave propagation in heterogeneous media: Velocity-stress finite-difference method,” *Geophysics*, 51(4), 889–901 (1986). DOI https://doi.org/10.1190/1.1442147 。用于速度应力方程与流固统一离散的依据。

Alan R. Levander, “Fourth-order finite-difference P-SV seismograms,” *Geophysics*, 53(11), 1425–1436 (1988). DOI https://doi.org/10.1190/1.1442422 。用于四阶空间、二阶时间交错格式的依据。

用户提供的 Y. Liu, *A Time-Domain Direct Sampling Method for Inverse Scattering Problems by Rough Surfaces*，第 4 节及式 (5.1)–(5.2)。用于 Cauchy 数据反传播及镜像补偿的数值对照。稿件出版状态未在本项目核实。

走时叠加与近零偏移厚度公式在本项目中作为明确给定的数值参照实现，其适用性由本次全波实验检验。
