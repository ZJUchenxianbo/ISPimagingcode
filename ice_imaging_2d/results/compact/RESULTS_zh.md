# 二维冰层数值实验结果

本报告由 main.py 根据本次实际计算结果自动生成。

## 模型与数据

采用二维平面应变声–弹速度应力方程。海水密度 1025 kg/m³，声速 1500 m/s。冰密度 920 kg/m³，纵、横波速度分别为 3600 和 1800 m/s。
冰面位于 0 m，平均冰底为 -1.5 m，正弦起伏振幅 0.1 m、周期 4 m。阵列深度 3 m，刚性平海底深度 5.5 m。
正演网格 0.01 m，源数 7。声源主频 15 kHz，处理频带 10–20 kHz。

空气加载采用冰面零牵引近似，外部用零质量、零刚度网格处理。界面是阶梯离散。侧面采用渐变阻尼层，海底施加法向速度零条件。全波数据包含纵横波转换和多次反射。材料采用无衰减、各向同性参数。

正演和成像分开实现。反演函数只接收记录、收发坐标、已知水声速和预先规定的搜索窗，真实冰底仅参与评价及绘图。背景数据由同一网格上的均匀水域和刚性海底计算，移除冰层并在顶部吸收。

## 实际误差

| 算例 | 方法 | RMSE / cm | 最大误差 / cm | 峰谷差 / cm | 扫描边界取峰比例 |
|---|---|---:|---:|---:|---:|
| flat | migration | 1.472 | 1.519 | 0.225 | 0.0% |
| flat | coherent | 0.471 | 0.536 | 0.155 | 0.0% |
| flat | dsm | 23.559 | 40.000 | 67.314 | 4.9% |
| flat | dsm_corrected | 1.772 | 5.136 | 5.424 | 0.0% |
| flat | correction_only | 28.169 | 40.000 | 80.000 | 24.7% |
| sine | migration | 1.889 | 2.551 | 19.715 | 0.0% |
| sine | coherent | 0.301 | 0.668 | 20.155 | 0.0% |
| sine | dsm | 33.457 | 49.921 | 80.000 | 27.2% |
| sine | dsm_corrected | 6.235 | 31.557 | 56.353 | 1.2% |
| sine | dsm_impedance | 9.884 | 49.298 | 71.147 | 1.2% |
| sine | correction_only | 28.724 | 49.921 | 80.000 | 24.7% |
| soft | migration | 1.719 | 2.110 | 19.594 | 0.0% |
| soft | coherent | 0.417 | 0.774 | 20.223 | 0.0% |
| soft | dsm | 35.153 | 50.000 | 80.000 | 30.9% |
| soft | dsm_corrected | 2.604 | 2.956 | 25.273 | 0.0% |
| soft | correction_only | 28.724 | 49.921 | 80.000 | 24.7% |

## 本次结果的含义

相干走时叠加的冰底 RMSE 为 0.301 cm，恒定平均深度估计的 RMSE 为 7.690 cm。提取峰谷差 20.155 cm，真值为 20.000 cm。该比较说明此次合成全波数据能够支持恢复界面起伏。
原文镜像补偿指标的 RMSE 为 6.235 cm，提取峰谷差为 56.353 cm。应对照图中的局部伪峰判读。走时成像的成功不能替代该指标在声–弹模型中的验证。
已知正入射声阻抗修正候选的 RMSE 为 9.884 cm。此结果用于检验简单常数反射系数补偿是否足够。
单炮网格加密的首反射波形相对差为 15.54%，到时差为 4.222 μs。当前几何误差与数值离散误差应共同考察，毫米级取峰误差尚不能作为实际分辨率结论。

migration 为包络走时叠加，coherent 为复解析信号相干走时叠加。dsm 为 Cauchy 数据反传播的正相关时延能量。dsm_corrected 加入原文参考平面镜像补偿。dsm_impedance 使用已知正入射反射系数修正补偿，属于半空间、近正入射近似下的候选。correction_only 将测量数据置零后单独计算补偿项。

图像颜色表示各自成像泛函的大小。它不直接表示冰的材料参数、概率或误差。亮带是否对应真实界面，应同时查看提取曲线、RMSE、峰谷差和扫描边界取峰比例。补偿项单独产生的亮带不能用作测量证据。

本程序对原文式 (4.4)、(5.1)、(5.2) 作有限测线、已知参考平面和首反射时间窗离散。声–弹模型中的聚焦性质以及此补偿的适用性仍需检验。原文数学模型为声软粗糙面。

## 图和数据文件

每个算例的 *_model.png 给出几何和源谱，*_records.png 给出炮集、时间窗和中央道，*_inversion.png 给出两种走时成像图及提取界面，*_dsm.png 比较原文型反传播能量与补偿，*_wavefields.png 给出平均法向应力及速度旋度的波场快照。速度旋度在冰中显示剪切运动，在阶梯界面附近还可能含离散误差。

data_*.npz 保存各炮总声压与 +z 法向梯度、时间轴、收发坐标、配置和求解器版本。images_*.npz 保存未经归一化的成像数组。profile_*.csv 保存真实及提取界面。metrics_*.json 保存误差。cache/ 支持正演断点续算。

## 敏感性实验

噪声在背景扣除后加入，先带通，再按固定首反射窗内的全体数据二范数标定 SNR。声压和梯度分别标定噪声。该试验代表合成残差数据的噪声，不包含现场背景误差。

| 条件 | 包络 RMSE / cm | 相干 RMSE / cm | 包络峰谷差 / cm |
|---|---:|---:|---:|
| Clean | 1.889 | 0.301 | 19.715 |
| 20 dB noise | 1.887 | 0.304 | 19.703 |
| 10 dB noise | 1.864 | 0.341 | 19.697 |
| Water speed -2% | 5.000 | 2.857 | 19.360 |
| Water speed +2% | 1.316 | 3.239 | 20.097 |
| Half local aperture | 1.396 | 0.289 | 19.830 |
| Every fourth receiver | 1.890 | 0.332 | 19.774 |

## 正演检查

```json
{
  "predicted_bottom_ms": 2.263333333333333,
  "measured_bottom_ms": 2.269333333333334,
  "predicted_top_ms": 3.0966666666666667,
  "measured_top_ms": 3.1013333333333337,
  "echo_separation_error_us": -1.3333333333336548,
  "direct_wave_relative_l2_error": 0.03772776630351784,
  "reciprocal_pairs": 0,
  "maximum_reciprocity_error": null,
  "arrival_check_pass": true,
  "direct_wave_check_pass": true,
  "mesh_refinement": {
    "coarse_spacing_m": 0.01,
    "fine_spacing_m": 0.005,
    "first_echo_relative_l2_difference": 0.155419833516228,
    "position_corrected_arrival_difference_us": 4.222222222221867,
    "scope": "One flat-model near-zero-offset first ice-bottom echo."
  },
  "dedicated_reciprocity": {
    "relative_l2_error": 1.0180639664625346e-09,
    "passed": true,
    "source_x_m": [
      -0.6049999999999995,
      0.5949999999999998
    ],
    "scope": "Dedicated two-shot flat-model reciprocal acquisition."
  }
}
```

网格检查仅针对指定单炮及时间窗。完整界面收敛、吸收边界距离收敛和全部炮次的细网格比较仍需追加计算。

## 结果的适用范围

此次结果针对已知波速、已知收发坐标、平坦自由冰面、平坦刚性海底和既定宽带源。平冰底和正弦冰底均为人工指定真值。没有使用实验室或现场数据。
首轮紧凑算例采用 3 m 深阵列和 4 m 水平周期。configs/field.json 提供 10 m 深阵列、20 m 周期、25 m 深海底的大规模设置。大规模设置只有在实际运行后才形成验证结果。
上表面高度和 4 m 厚冰提供独立运行入口，其结果应按各自输出目录判读。下表面成功成像不能代替上表面联合恢复的验证。

## 来源

Jean Virieux (1986), P-SV wave propagation in heterogeneous media: velocity-stress finite-difference method, Geophysics 51, 889–901. https://doi.org/10.1190/1.1442147
Alan R. Levander (1988), Fourth-order finite-difference P-SV seismograms, Geophysics 53, 1425–1436. https://doi.org/10.1190/1.1442422
用户提供的 Y. Liu, A Time-Domain Direct Sampling Method for Inverse Scattering Problems by Rough Surfaces，第 4 节及式 (5.1)–(5.2)。该稿的出版状态未在本项目中核实。

## 上表面走时扫描

*_top_scan.png 依据重建冰底和近零偏移 P 回波进行一维高度扫描。所有算例使用预设高度范围 [-0.4,0.6] m。此处采用局部平行界面近似，数值不直接构成双界面 TD-DSM 验证。

| 算例 | 估计冰面 / m | 真值 / m | 绝对误差 / cm | 各道估计标准差 / cm |
|---|---:|---:|---:|---:|
| flat | -0.015 | 0.000 | 1.50 | 0.00 |
| sine | 0.005 | 0.000 | 0.50 | 1.32 |