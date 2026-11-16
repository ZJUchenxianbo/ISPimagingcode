"""实验结果的中文报告。"""

import json
from pathlib import Path


def write_report(config, all_metrics, output, robustness=None, validation=None):
    output = Path(output)
    lines = ["# 二维冰层数值实验结果", "", "本报告由 main.py 根据本次实际计算结果自动生成。", "",
             "## 模型与数据", "",
             f"采用二维平面应变声–弹速度应力方程。海水密度 {config.water_density:g} kg/m³，声速 {config.water_speed:g} m/s。冰密度 {config.ice_density:g} kg/m³，纵、横波速度分别为 {config.ice_p_speed:g} 和 {config.ice_s_speed:g} m/s。",
             f"冰面位于 {config.ice_top:g} m，平均冰底为 {config.mean_bottom:g} m，正弦起伏振幅 {config.rough_amplitude:g} m、周期 {config.rough_period:g} m。阵列深度 {config.array_depth:g} m，刚性平海底深度 {config.seabed_depth:g} m。",
             f"正演网格 {config.grid_spacing:g} m，源数 {config.source_count}。声源主频 {config.source_frequency / 1000:g} kHz，处理频带 {config.band_low / 1000:g}–{config.band_high / 1000:g} kHz。",
             "", "该报告对应 legacy 水域传播核对照；实际空气/自由表面模式、界面表示和差分阶数以本次 config.json 为准。侧面采用渐变阻尼层，海底施加法向速度零条件。材料采用无衰减、各向同性参数。", "",
             "正演和成像分开实现。反演函数只接收记录、收发坐标、已知水声速和预先规定的搜索窗，真实冰底仅参与评价及绘图。背景数据由同一网格上的均匀水域和刚性海底计算，移除冰层并在顶部吸收。", "",
             "## 实际误差", "",
             "| 算例 | 方法 | RMSE / cm | 最大误差 / cm | 峰谷差 / cm | 扫描边界取峰比例 |",
             "|---|---|---:|---:|---:|---:|"]
    for case, metrics in all_metrics.items():
        for name, values in metrics.items():
            lines.append(f"| {case} | {name} | {100 * values['rmse_m']:.3f} | {100 * values['maximum_error_m']:.3f} | {100 * values['peak_to_valley_m']:.3f} | {values['boundary_fraction']:.1%} |")
    if "sine" in all_metrics:
        sine = all_metrics["sine"]
        coherent = sine["coherent"]
        lines.extend(["", "## 本次结果的含义", "",
                      f"相干走时叠加的冰底 RMSE 为 {100 * coherent['rmse_m']:.3f} cm，恒定平均深度估计的 RMSE 为 {100 * coherent['constant_depth_rmse_m']:.3f} cm。提取峰谷差 {100 * coherent['peak_to_valley_m']:.3f} cm，真值为 {100 * coherent['true_peak_to_valley_m']:.3f} cm。应结合恒深基线、峰谷差和离散误差判断是否恢复了起伏，不由单一误差数值推断分辨率。"])
        if "dsm_corrected" in sine:
            original = sine["dsm_corrected"]
            lines.append(f"原文镜像补偿指标的 RMSE 为 {100 * original['rmse_m']:.3f} cm，提取峰谷差为 {100 * original['peak_to_valley_m']:.3f} cm。应对照图中的局部伪峰判读。走时成像的成功不能替代该指标在声–弹模型中的验证。")
        if "dsm_impedance" in sine:
            candidate = sine["dsm_impedance"]
            lines.append(f"已知正入射声阻抗修正候选的 RMSE 为 {100 * candidate['rmse_m']:.3f} cm。此结果用于检验简单常数反射系数补偿是否足够。")
        if validation and "mesh_refinement" in validation:
            refinement = validation["mesh_refinement"]
            lines.append(f"单炮网格加密的首反射波形相对差为 {100 * refinement['first_echo_relative_l2_difference']:.2f}%，到时差为 {refinement['position_corrected_arrival_difference_us']:.3f} μs。当前几何误差与数值离散误差应共同考察，毫米级取峰误差尚不能作为实际分辨率结论。")
    lines.extend(["", "migration 为包络走时叠加，coherent 为复解析信号相干走时叠加。dsm 为 Cauchy 数据反传播的正相关时延能量。dsm_corrected 加入原文参考平面镜像补偿。dsm_impedance 使用已知正入射反射系数修正补偿，属于半空间、近正入射近似下的候选。correction_only 将测量数据置零后单独计算补偿项。", "",
                  "图像颜色表示各自成像泛函的大小。它不直接表示冰的材料参数、概率或误差。亮带是否对应真实界面，应同时查看提取曲线、RMSE、峰谷差和扫描边界取峰比例。补偿项单独产生的亮带不能用作测量证据。", "",
                  "本程序对原文式 (4.4)、(5.1)、(5.2) 作有限测线、已知参考平面和首反射时间窗离散。声–弹模型中的聚焦性质以及此补偿的适用性仍需检验。原文数学模型为声软粗糙面。", "",
                  "## 图和数据文件", "",
                  "每个算例的 *_model.png 给出几何和源谱，*_records.png 给出炮集、时间窗和中央道，*_inversion.png 给出两种走时成像图及提取界面，*_dsm.png 比较原文型反传播能量与补偿，*_wavefields.png 给出平均法向应力及速度旋度的波场快照。速度旋度在冰中显示剪切运动，在阶梯界面附近还可能含离散误差。", "",
                  "data_*.npz 保存各炮总声压与 +z 法向梯度、时间轴、收发坐标、配置和求解器版本。images_*.npz 保存未经归一化的成像数组。profile_*.csv 保存真实及提取界面。metrics_*.json 保存误差。cache/ 支持正演断点续算。", ""])
    if robustness:
        lines.extend(["## 敏感性实验", "", "噪声在背景扣除后加入，先带通，再按固定首反射窗内的全体数据二范数标定 SNR。声压和梯度分别标定噪声。该试验代表合成残差数据的噪声，不包含现场背景误差。", "",
                      "| 条件 | 包络 RMSE / cm | 相干 RMSE / cm | 包络峰谷差 / cm |", "|---|---:|---:|---:|"])
        for row in robustness:
            coherent_value = f"{100 * row['coherent_rmse_m']:.3f}" if "coherent_rmse_m" in row else "未计算"
            lines.append(f"| {row['experiment']} | {100 * row['rmse_m']:.3f} | {coherent_value} | {100 * row['peak_to_valley_m']:.3f} |")
    if validation:
        lines.extend(["", "## 正演检查", "", "```json", json.dumps(validation, ensure_ascii=False, indent=2), "```", "",
                      "网格检查仅针对指定单炮及时间窗。完整界面收敛、吸收边界距离收敛和全部炮次的细网格比较仍需追加计算。"])
    lines.extend(["", "## 结果的适用范围", "",
                  "此次结果针对已知波速、已知收发坐标、平坦自由冰面、平坦刚性海底和既定宽带源。平冰底和正弦冰底均为人工指定真值。没有使用实验室或现场数据。",
                  "首轮紧凑算例采用 3 m 深阵列和 4 m 水平周期。configs/field.json 提供 10 m 深阵列、20 m 周期、25 m 深海底的大规模设置。大规模设置只有在实际运行后才形成验证结果。",
                  "上表面高度和 4 m 厚冰提供独立运行入口，其结果应按各自输出目录判读。下表面成功成像不能代替上表面联合恢复的验证。", "",
                  "## 来源", "",
                  "Jean Virieux (1986), P-SV wave propagation in heterogeneous media: velocity-stress finite-difference method, Geophysics 51, 889–901. https://doi.org/10.1190/1.1442147",
                  "Alan R. Levander (1988), Fourth-order finite-difference P-SV seismograms, Geophysics 53, 1425–1436. https://doi.org/10.1190/1.1442422",
                  "用户提供的 Y. Liu, A Time-Domain Direct Sampling Method for Inverse Scattering Problems by Rough Surfaces，第 4 节及式 (5.1)–(5.2)。该稿的出版状态未在本项目中核实。", ""])
    lines.extend(["## 上表面走时扫描", "", "*_top_scan.png 依据重建冰底和近零偏移 P 回波进行一维高度扫描。所有算例使用预设高度范围 [-0.4,0.6] m。此处采用局部平行界面近似，数值不直接构成双界面 TD-DSM 验证。", "",
                  "| 算例 | 估计冰面 / m | 真值 / m | 绝对误差 / cm | 各道估计标准差 / cm |",
                  "|---|---:|---:|---:|---:|"])
    for case in all_metrics:
        path = output / f"top_metrics_{case}.json"
        if path.exists():
            values = json.loads(path.read_text())
            lines.append(f"| {case} | {values['estimated_top']:.3f} | {values['true_top_m']:.3f} | {100 * values['absolute_error_m']:.2f} | {100 * values['individual_std']:.2f} |")
    (output / "RESULTS_zh.md").write_text("\n".join(lines), encoding="utf-8")

