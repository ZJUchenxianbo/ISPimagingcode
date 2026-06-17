# Maxwell GPSWF 修改计划

本文件作为 `maxwell_gpswf` 子项目的持续计划文件。以后进入较大代码调整前，先在这里写清目标、问题和修改步骤，确认后再动代码。

这里的“调整”只指两类情况：

- 代码流程、数学约定或实现结构存在问题；
- 当前实现没有达到预期效果，导致结果难解释或和实验目标不一致。

## 1. 当前主流程

现阶段保留 B/C 两类主数据模式。

### B 模式：Analytic Born far-field

```text
Q(x)
-> analytic Fourier transform Qhat(p)
-> Born far-field channels g(p)=M(p)c(p)
-> polarimetric recovery Qhat_rec(p)
-> GPSWF / Fourier / Bessel / DSM reconstruction
```

### C 模式：Full VIE / external solver far-field

```text
Q(x)
-> Full VIE 或外部 Maxwell solver
-> measured far-field channels g_obs
-> polarimetric recovery Qhat_rec(p)
-> GPSWF initial reconstruction Q0
-> BIM-GPSWF nonlinear correction
```

BIM 属于 C 模式下的非线性反演分支，不应移出主流程。

## 2. 本轮已完成

- `forward/datasets.py` 支持显式传入 `incident_dirs / obs_dirs`。
- `FarfieldDataset.farfield_data` 统一采用 `g=M(p)c(p)` 型归一化数据。
- `analytic_born_farfield_dataset` 不再额外乘 `k^2/(4π)`。
- `discrete_vie_born_farfield_dataset` 和 `full_vie_farfield_dataset` 在写入 dataset 前移除 `k^2/(4π)` prefactor。
- VIE dataset 默认使用 `-p_nodes` 构造物理方向对，使 VIE 原始 `exp(+i C p·x)` 相位对应项目使用的 `exp(-i C p·x)` Fourier 约定。
- `figure3_sources_shapes.py` 的 VIE 路径在 mock 模式下改用 `-target_nodes` 匹配得到的方向对。
- `nonlinear/bim_gpswf.py` 新增 raw far-field channel 数据和 raw BIM 线性化：

```text
compute_raw_vie_farfield_data
compute_raw_bim_gpswf_linearization
```

- `figure7_bim_gpswf_frequency.py` 已切换为同一批 raw Full VIE far-field channels：

```text
g_obs -> polarimetric recovery -> GPSWF initial
g_obs - g(Q_n) -> raw BIM update
```

- 图 7 诊断表新增 `bim_residual_space=raw_farfield_channel`。
- `README.md` 已同步更新图 7 的 raw far-field BIM 说明。

## 3. 仍需调整的问题

### 问题 1：mock 模式下 full tensor 极化恢复可能欠定

mock 模式的 `generate_data_nodes` 目前每个 target Fourier node 只匹配一组方向对。对 `kind="full"` 的 9 个张量系数，一组方向对提供的通道数有限，极化矩阵可能欠定或病态。

需要检查：

- 图 3 中 `kind="full"` 的 Full VIE / VIE-Born 列是否因为极化恢复欠定而幅值异常；
- `farfield_dataset_to_qhat` 是否应输出或记录每个节点的 `rank / cond / sigma_min`；
- mock 模式是否需要为每个 target node 选取多个邻近 measured direction pairs，而不是只取最近的一个。

可能调整：

```text
generate_data_nodes(..., branch_count=J)
```

在 mock 模式下返回每个 target node 的 J 个近邻方向对，并保证 `FarfieldDataset` 按 branch 分组。

### 问题 2：图 3 的 VIE-Born 幅值需要进一步诊断

本轮 smoke test 中图 3 的 VIE-Born 列能跑通，但部分诊断显示幅值明显大于 analytic Born 和 Full VIE。需要判断这是：

- VIE 体素离散误差；
- mock 极化恢复欠定；
- VIE-Born 与 analytic phantom 几何不完全一致；
- 还是 normalization / phase 仍有遗漏。

需要补充诊断：

- polarimetric matrix rank / condition；
- `data_norm` 在 analytic Born / VIE-Born / Full VIE 三类数据源之间的对比；
- 同一 shape 下 VIE-Born 与 analytic Born 的相对数据误差。

### 问题 3：图 7 仍有公共样板代码未抽取

`common/reconstruction.py` 已经存在，但图 7 中仍保留：

- `_build_modes`
- 手写 alpha 截断；
- 手写 2D grid；
- 手写 GPSWF image reconstruction。

需要逐步改为：

```text
build_gpswf_modes
truncate_modes
make_xy_grid
gpswf_reconstruct_image
```

优先减少图 7 中和 BIM 无关的样板代码。

### 问题 4：raw BIM 只覆盖 scalar contrast q(x)T0

当前 raw BIM 已经在 raw far-field channel 上做残差，但未知量仍是标量函数 `q(x)`，张量方向 `T0` 已知。

这是当前图 7 的合理简化，但后续如果要处理不同张量块或各向异性 `Q(x)`，需要扩展为：

```text
delta Q(x) = sum_r sum_j a_{r,j} psi_j(x) T_r
```

该扩展暂不作为当前优先项。

## 4. 下一步建议

优先顺序：

1. 给 `farfield_dataset_to_qhat` 增加极化恢复诊断，至少记录 `rank`、`cond`、`sigma_min`。
2. 处理 mock 模式多方向对匹配，解决 full tensor 极化恢复欠定问题。
3. 用新增诊断重新检查图 3 的 VIE-Born 幅值异常。
4. 把图 7 接入 `common/reconstruction.py`，减少重复代码。

## 5. 验证命令

轻量检查：

```bash
.venv/bin/python -m py_compile $(find maxwell_gpswf -name '*.py' -not -path '*/__pycache__/*')
```

smoke test：

```bash
.venv/bin/python maxwell_gpswf/main.py --mode fig3 --quick --data-mode mock --out-dir outputs_smoke
.venv/bin/python maxwell_gpswf/main.py --mode fig7 --quick --data-mode mock --out-dir outputs_smoke
```

验证后删除：

```text
outputs_smoke
__pycache__
```

## 6. 当前不调整的内容

- 不删除 B 模式 analytic Born far-field。
- 不把 BIM 从主流程移出。
- 不为了图像更好看改变物理数据流程。
- 不在本阶段接入外部 Maxwell 正向求解器。
- 不一次性重写所有 figure 脚本。
