# AGENTS.md

本文件是 `maxwell_gpswf` 子项目的补充约定。参与本子项目的模型必须同时遵守父项目 `../AGENTS.md`；若规则冲突，以本文件中更具体的 Maxwell GPSWF 约定为准。

## 子项目定位

本子项目用于把 `06-005` 的标量远场反演数值试验方法类推到各向异性 Maxwell 问题，核心是 ball GPSWF 模态截断、极化恢复、mock/ideal Fourier 节点数据，以及 Born/VIE 数据源对比。

代码应优先服务于可解释的数值实验，不为了图像好看而改变物理流程或隐藏数值不稳定。

## 参数来源

- 重要实验参数必须说明来源，例如 `requested_measure_dirs`、`K`、`ell_max`、`n_modes_per_ell`、`epsilon`、`N_cap`、`n_radial`、`n_angular`。
- 若参数来自论文公式或文献设置，应在 README、注释或诊断说明中写清对应关系。
- 若参数是经验选择或临时调参，必须明确标注为经验参数，不得写成理论推导结果。
- 支撑半径 `R`、带宽 `C=2kR`、坐标缩放 `x=Ry` 等尺度关系修改时，应同步检查图像坐标轴、真值显示和 README 说明。

## 数据模式

- `mock` 模式表示先固定入射/观测方向网格，再由 `p=(d-xhat)/2` 形成候选 Fourier 节点，并用最近邻匹配 target quadrature nodes。
- `ideal` 模式表示先给定 target Fourier 节点 `p`，再直接构造满足 `p=(d-xhat)/2` 的 admissible direction pairs。
- 主图脚本必须读取 `ExperimentConfig.data_mode` 或命令行 `--data-mode`，不得在单个图中无说明地硬编码 `mock` 或 `ideal`。
- 分析输出目录前，必须先检查诊断表中的 `data_mode`，不能只根据目录名假设结果属于 mock 或 ideal。

## 正向数据与反演流程

- 本子项目必须严格区分两层流程：第一层是正向数据生成，第二层是由远场数据反演 `Q(x)`。
- 正向数据生成的输入是给定散射体 `Q(x)`，输出应是可测远场数据，例如 analytic Born far-field、discrete VIE-Born far-field 或 Full VIE far-field。
- 反演流程的输入应是远场数据，标准链路是 `far-field data -> polarimetric recovery Qhat(p) -> GPSWF/Fourier/Bessel/DSM reconstruction Q(x)`。
- 主实验不得把由 `Q(x)` 解析得到的 `Qhat(p)` 直接送入 GPSWF 作为最终评价流程；这种直接 Fourier 输入只能作为明确标注的调试或正向校验分支。
- “解析”只描述正向远场数据的生成方式，即通过 Born 远场公式和连续 phantom 的解析积分生成 far-field data；不表示反演时已知 `Qhat(p)`。
- 如果未来接入外部 Maxwell 正向求解器，应直接把其输出当作远场数据接入统一反演入口，而不是改变后续极化恢复和 GPSWF 重构流程。

## 方向对与极化配置

- `branch_count` 只表示 ideal 模式下每个 Fourier 点构造多少组方向对，不等同于论文中极化恢复的配置数。
- 极化恢复中的配置数应使用明确名称，例如 `polarimetric_J` 或 `polarimetric_config_count`，避免和 `branch_count` 混用。
- 对一般各向异性非对称 `full` 张量 `Q`，需要足够多且线性独立的极化测量方程；判断是否足够时应查看极化矩阵的奇异值和条件数，而不是只数方向对数量。
- 若修改 `build_geometries_from_p`、`build_polarimetric_matrix` 或 `recover_polarimetric_coefficients`，必须说明修改的是方向对几何、入射极化、观测投影，还是张量基。

## 诊断优先

- 分析成像异常前，先查看对应 `figure*_diagnostics.csv` 或 `figure*_diagnostics_detail.npz`。
- 至少检查 `retained_modes`、`target_nodes_per_retained_modes`、`mock_distance_mean/max/p95`、`gram_offdiag_ratio`、`gram_cond`、`coeff_norm`、`background_p95_abs`、`target_p95_abs`。
- 对 `R` 增大、`C` 增大或高频实验，必须同时关注测量方向密度、mock 匹配误差和保留模态数；不能只通过调色或显示范围判断算法效果。
- 如果诊断表缺少判断所需字段，应优先补诊断字段，而不是直接猜测原因。

## 输出与文档

- 子项目说明文档是 `maxwell_gpswf/README.md`。新增图、命令行参数、输出文件、诊断字段或重要实验假设时，应同步更新该文件。
- 输出文件名应能区分主图、诊断表、诊断曲线和缓存；不要让 mock/ideal 两类结果覆盖到同一目录。
