# 冰层下表面首次反射 TD-DSM

正演采用用户提供项目中的水域 BEM、冰层 Q4-FEM 和 CQ，保留 CPU/CUDA 后端。反演使用首次反射 Cauchy 数据构建论文型虚拟回时波，比较无补偿、镜像补偿和仅补偿指标。上表面在这一阶段固定为平面。

## 配置与运行

从仓库根目录执行，唯一正式配置为 `configs/experiment.json`：15 kHz载频、10–20 kHz成像频带、1.5 m基准冰厚；所有局部厚度满足 `0 < H <= 4 m`，无冰脊。当前将20 cm粗糙尺度暂定义为竖向峰谷差0.20 m，即正弦振幅0.10 m；水平周期4 m是独立的模拟假设。更厚案例需同时调整阵列深度、采样区域、参考面和记录长度。

```bash
.venv/bin/python -m pip install -r ice_imaging_2d/requirements.txt
.venv/bin/python ice_imaging_2d/main.py --estimate
.venv/bin/python ice_imaging_2d/main.py -forward  # 仅正演
.venv/bin/python ice_imaging_2d/main.py -inv      # 仅读取已有数据并反演
.venv/bin/python ice_imaging_2d/main.py           # 依次执行两阶段
```

Windows使用 `.venv/Scripts/python.exe`。正式配置默认 `gpu`；CPU正演显式传 `--backend cpu --workers 2`。`-forward` 与 `-inv` 互斥。`-inv` 不接受 `--backend/--workers`，不导入求解器、构建网格或加载CUDA，可在CPU机器上处理GPU生成的数据。缺少数据直接报错。

```bash
.venv/bin/python ice_imaging_2d/main.py -forward --case sine --output ice_outputs/run01
.venv/bin/python ice_imaging_2d/main.py -inv --case sine --data ice_outputs/run01/data --output ice_outputs/run01
```

`--config` 指向修改后的完整实验JSON。`--data` 可为数据根目录、单案例目录或 `data_cq.npz` 文件；单案例数据需指定 `--case`。`--estimate` 只校验参数和报告规模，不求解；与 `-inv` 合用时也不导入正演实现。默认121源、160接收点、间距0.03 m，源接收错开半步。CQ重建频带0.5–35 kHz用于保持短脉冲，成像只使用10–20 kHz。

## 输出与恢复

默认输出根目录为 `ice_outputs/ice_bottom`：

- `data/<case>/data_cq.npz`：总场、直达场、散射场的声压及向上导数、有效源脉冲、时间/坐标、CQ传递函数、几何及配置；`metadata.json`记录求解器和执行环境。
- `data/<case>/cq_cache/<forward_id>/`：逐复频点原子保存，重新运行自动恢复；完整数据匹配时直接复用。物理参数、收发阵列或CQ改变后需使用新的输出目录，避免覆盖旧实验。
- `inversion/<case>/<inversion_id>/`：`images.npz`、`first_reflection.npz`、`metrics.json`、`window_sensitivity.json`、`metadata.json`，以及指标图、冰底曲线和波形窗口图。

正演标识不包含CPU/GPU执行选择与成像参数；反演标识包含输入文件SHA-256和成像设置。修改采样网格、时间窗或参考面可以复用原始数据，并保存到独立结果目录。首反射窗口根据先验参考面设置；真实边界仅在计算指标后用于误差评价和画图。

## Linux GPU 准备

目标服务器需要NVIDIA驱动、CUDA Toolkit、cuDSS开发库、C++编译环境及 `flock`。将下面的路径与架构改为实际环境；`sm_80` 仅为示例。更换架构或工具链后用 `build.sh --force` 重编译。

```bash
export CUDA_ROOT=/usr/local/cuda
export CUDSS_ROOT=/opt/nvidia/cudss
export PLANAR_CUDA_ARCH=sm_80
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDSS_ROOT/lib:$CUDSS_ROOT/lib64:${LD_LIBRARY_PATH:-}"
bash ice_imaging_2d/ice2d/forward_cq/gpu/build.sh
.venv/bin/python -m ice_imaging_2d.tests.check_forward --gpu
.venv/bin/python ice_imaging_2d/main.py --estimate
.venv/bin/python ice_imaging_2d/main.py
```

`forward.solver.gpu_devices` 与 `workers_per_device` 控制GPU进程；可用 `gpu_library_dir` 指定编译库位置。不会自动降频或回落CPU。接收点声压/法向导数评价目前在CPU进行，复用GPU求出的边界量。

## 轻量检查与科学边界

```bash
.venv/bin/python -m ice_imaging_2d.tests.check_td_dsm
.venv/bin/python -m ice_imaging_2d.tests.check_forward
.venv/bin/python -m ice_imaging_2d.tests.check_workflow
```

检查包括原ZIP响应对照、解析导数、多源、CQ、有限时间显式求和和分阶段CLI；临时结果自动清理，不运行神经网络。测试内部粗网格用于数值接口检查，不是正式精度案例。

时间门控近似提取首次反射，镜像补偿仍是允许透射情形下的候选方法，尚无聚焦定理或正式10–20 kHz精度结论。GPU代码已保留，但仍需目标服务器编译与运行验收。历史 `results/` 和已有低频输出保留归档。数学定义见 [docs/scheme_alignment.md](docs/scheme_alignment.md)，正演来源见 [SOURCE.md](ice2d/forward_cq/SOURCE.md)。
