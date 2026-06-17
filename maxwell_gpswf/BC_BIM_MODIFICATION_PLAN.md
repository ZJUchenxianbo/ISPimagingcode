# Maxwell GPSWF 子项目修改方案

本文件记录下一阶段代码调整计划。目标是在现阶段保留 B/C 两类数据模式，并把 BIM 明确纳入 C 模式的非线性反演分支。

## 1. 总体目标

当前子项目保留两类主数据模式：

- B 模式：Analytic Born far-field
- C 模式：Full VIE far-field，后续可替换为外部 Maxwell 正向求解器数据

所有主实验应遵循以下分层：

```text
正向数据生成：
Q(x) -> far-field data

反演成像：
far-field data -> polarimetric recovery Qhat(p) -> reconstruction Q(x)
```

BIM 属于 C 模式下的非线性反演分支，不应被移出主流程。

## 2. B/C 模式定义

### B 模式：Analytic Born far-field

```text
Q(x)
-> analytic Fourier transform Qhat(p)
-> Born far-field channels g(p) = M(p) Qhat(p)
-> polarimetric recovery Qhat_rec(p)
-> GPSWF / Fourier / Bessel / DSM reconstruction
```

说明：

- 这是 Born 线性模型内部的解析远场数据。
- 它可以用于验证 GPSWF 截断、噪声、频率、mock 节点和不同基函数的影响。
- 它不是直接把已知 Qhat 输入反演；主实验中仍应经过远场通道和极化恢复。

### C 模式：Full VIE / external solver far-field

```text
Q(x)
-> Full VIE 或外部 Maxwell solver
-> measured far-field channels g_obs
-> polarimetric recovery Qhat_rec(p)
-> GPSWF initial reconstruction Q0
```

说明：

- 这是更接近真实正向数据的流程。
- Full VIE 只是当前内部正向求解器；后续外部正向求解器应作为同类 far-field data 接入。
- C 模式下可以继续做 BIM 非线性修正。

## 3. BIM 的正确位置

BIM 应作为 C 模式的非线性反演分支：

```text
Full VIE far-field g_obs
-> polarimetric recovery Qhat_rec
-> GPSWF initial Q0
-> BIM-GPSWF iterations
-> Q1, Q2, ...
```

每一步 BIM 迭代应使用同一批观测远场数据：

```text
r_n = g_obs - g(Q_n)
Dg[delta Q_n] ~= r_n
Q_{n+1} = Q_n + step * delta Q_n
```

更新量仍限制在 GPSWF 低秩空间：

```text
delta Q_n = sum_j a_j psi_j
```

## 4. 当前主要问题

### 4.1 mock 模式方向对不严格

当前 `forward/datasets.py` 中的 dataset 生成函数会重新从 `p_nodes` 构造 admissible direction pairs。  
这对 ideal 模式合理，但对 mock 模式不严格。

mock 模式应使用 `generate_data_nodes` 返回的真实匹配方向：

```text
target_nodes
-> generate_data_nodes(...)
-> p_nodes, matched_inc, matched_obs
-> 用 matched_inc / matched_obs 生成远场
```

### 4.2 far-field 尺度需要统一

需要明确统一约定：

```text
g(p) = M(p)c(p)
```

或：

```text
E_inf = k^2/(4pi) M(p)c(p)
```

建议在 `FarfieldDataset` 层统一归一到 `g(p)=M(p)c(p)`，这样 `farfield_dataset_to_qhat` 对所有数据源一致。

### 4.3 图 7 的 BIM 残差仍是 scalarized residual

当前图 7 的 Full VIE 初值已经接近：

```text
Full VIE far-field -> polarimetric recovery -> GPSWF
```

但 BIM 迭代仍使用 `compute_scalar_vie_data` 和 scalarized residual。  
后续应改为 raw far-field channel residual：

```text
g_obs - g(Q_n)
```

这样 BIM 才和 C 模式主流程一致。

### 4.4 figure 脚本重复较多

`figure5`、`figure6`、`figure7` 中重复了很多逻辑：

- GPSWF 模式构造
- alpha 缓存读取
- epsilon / N_cap 截断
- 2D 网格生成
- GPSWF component reconstruction
- 诊断行构造
- 图像显示函数

后续应逐步提取公共函数，避免继续堆叠。

## 5. 具体修改计划

### Step 1：统一 forward dataset 接口

修改文件：

- `maxwell_gpswf/forward/datasets.py`

计划：

- `FarfieldDataset` 保留以下字段：

```python
p_nodes
incident_dirs
obs_dirs
farfield_data
data_source
metadata
```

- 增加可选输入：

```python
incident_dirs: np.ndarray | None = None
obs_dirs: np.ndarray | None = None
```

- 若用户显式传入方向对，则必须使用这些方向对。
- 若没有传入方向对，才从 `p_nodes` 构造 ideal admissible direction pairs。
- `metadata` 中记录：

```text
data_mode
n_geometries
normalization
prefactor
```

### Step 2：统一 far-field 归一

修改文件：

- `maxwell_gpswf/forward/datasets.py`
- `maxwell_gpswf/common/polarimetric.py` 如有必要

计划：

- 明确 `farfield_dataset_to_qhat` 的输入必须满足 `g=M c`。
- analytic Born、VIE-Born、Full VIE 在写入 `FarfieldDataset.farfield_data` 前做同一尺度归一。
- 诊断表中加入 `farfield_normalization` 字段。

### Step 3：修正图 3 数据源对比

修改文件：

- `maxwell_gpswf/experiments/figure3_sources_shapes.py`

计划：

- 三个成像列统一为：

```text
FarfieldDataset -> farfield_dataset_to_qhat -> GPSWF
```

- mock 模式下传入 `matched_inc / matched_obs`。
- ideal 模式下允许 dataset 内部构造 admissible direction pairs。
- 保留当前布局：

```text
truth | Full VIE far-field | VIE-Born far-field | Analytic Born far-field
```

### Step 4：修正图 7 的 BIM 主流程

修改文件：

- `maxwell_gpswf/experiments/figure7_bim_gpswf_frequency.py`
- `maxwell_gpswf/nonlinear/bim_gpswf.py`

计划：

- 图 7 保持布局：

```text
truth
Analytic Born FF -> GPSWF
Full VIE FF -> GPSWF initial
Full VIE FF -> BIM iter 1
Full VIE FF -> BIM iter 2
Full VIE FF -> BIM iter 3
```

- 初值：

```text
g_obs -> polarimetric recovery Qhat_rec -> GPSWF -> Q0
```

- BIM 残差：

```text
r_n = g_obs - g(Q_n)
```

- BIM 线性化矩阵应对应 raw far-field channel residual，而不是 scalarized residual。
- 如果第一阶段暂时保留 scalarized BIM，需要在图标题、README 和诊断表明确写：

```text
scalarized BIM-GPSWF transitional implementation
```

但最终目标是 raw far-field BIM。

### Step 5：保留图 1/2/4/5/6 的 B 模式定位

修改文件：

- `maxwell_gpswf/experiments/figure1_noise_dimension.py`
- `maxwell_gpswf/experiments/figure2_frequency_contrast.py`
- `maxwell_gpswf/experiments/figure4_scale_scaling.py`
- `maxwell_gpswf/experiments/figure5_basis_comparison.py`
- `maxwell_gpswf/experiments/figure6_tensor_blocks.py`
- `maxwell_gpswf/README.md`

计划：

- 保留 analytic Born far-field 流程。
- 统一诊断字段：

```text
data_source = analytic_born_farfield
pipeline = B
```

- 避免使用容易误解的 `analytical`、`analytic_qhat`。

### Step 6：抽取公共重建工具

新增或修改文件：

- `maxwell_gpswf/common/reconstruction.py`

计划提取：

```python
build_gpswf_modes(...)
truncate_modes(...)
make_xy_grid(...)
reconstruct_gpswf_component(...)
reconstruct_gpswf_tensor_norm(...)
```

然后逐步让 figure 脚本只负责：

```text
设定参数 -> 调用公共函数 -> 保存图和诊断
```

## 6. 建议执行顺序

1. 先改 `forward/datasets.py`，解决方向对和尺度归一。
2. 再改图 3，验证 B/C 数据源对比流程。
3. 再改图 7，把 BIM 继续纳入 C 模式。
4. 更新 `README.md`，说明 B/C 模式和 BIM 分支。
5. 最后做公共函数抽取，降低 figure 脚本复杂度。

## 7. 验证计划

每轮修改后先做轻量检查：

```bash
.venv/bin/python -m py_compile $(find maxwell_gpswf -name '*.py' -not -path '*/__pycache__/*')
```

然后运行小规模 smoke test：

```bash
.venv/bin/python maxwell_gpswf/main.py --mode fig3 --quick --data-mode mock --out-dir outputs_smoke
.venv/bin/python maxwell_gpswf/main.py --mode fig7 --quick --data-mode mock --out-dir outputs_smoke
```

验证后删除 `outputs_smoke` 和 `__pycache__`。

## 8. 不做的事情

- 不把 BIM 从主流程移除。
- 不把 Full VIE 数据强行调到图像好看。
- 不用直接已知 Qhat 作为主图评价数据。
- 不在本阶段引入外部 Maxwell 正向求解器。
- 不重写全部 figure 脚本；按数据流优先逐步整理。
