# 冰层双界面时域直接成像

本项目应实现研究方案4.2–4.5节的耦合背景Green核、Cauchy数据反传播和双界面直接指标。**完整方案尚未实现**：目前完成第4.3节离散反传播算子；耦合核构建、补偿场和上界面专用指标仍有推导缺项。逐项对应、数据契约和验证边界见 [方案对齐说明](docs/scheme_alignment.md)。

此前的中心差分灵敏度/Gauss–Newton已移为显式对照模式，不能称为方案的直接成像方法。默认入口只显示真实实现状态，不启动昂贵正演。

从项目根目录运行，全部使用 `.venv/bin/python`：

```bash
# 显示已完成和缺少的部分，不求解正问题
.venv/bin/python ice_imaging_2d/main.py
# 检查第4.3节离散反传播算子（轻量，不运行正演或神经网络）
.venv/bin/python -m ice_imaging_2d.experiments.direct_checks
# 仅在已有经验证耦合核和Cauchy数据后执行反传播
.venv/bin/python ice_imaging_2d/main.py --stage backpropagate --observed observed.npz --reference-data reference.npz --green coupled_green.npz
# 原参数灵敏度对照：只估算成本
.venv/bin/python ice_imaging_2d/main.py --mode sensitivity --stage indicator --estimate
# 原低频软件链路检查，仅为对照，不代表实际条件验收
.venv/bin/python ice_imaging_2d/main.py --mode sensitivity --quick --stage data --workers 1 --out-dir ice_outputs/check
# 旧声学DSM/走时对照
.venv/bin/python ice_imaging_2d/main.py --mode legacy --case sine
```

`--mode direct`为默认。反传播需要声压和法向导数，缺少时拒绝运行；仓库尚无物理验收通过的耦合核，不能把上述命令理解为已有完整成像结果。反传播阶段不调用正演，不对候选界面做参数扰动。

输出位于 `ice_outputs/direct_backpropagation/`，包含 `backpropagated_field.npz`（source×sample×backprop_time）、坐标/有符号时间轴与 `metadata.json`。此场不是最终界面指标，没有自动添加补偿或输出伪造的冰厚图。

物理条件保留现有冰厚、10–20 kHz频段、源/接收阵列及粗糙几何。当前0.1 m正弦振幅对应20 cm峰谷差。显式空气配置 `dual_short.json` 为2.5 mm/40 ms，仍需实际频段收敛验证；`dual_candidate.json`为未验收的自由表面候选，不是方案默认物理模型。`dual_smoke.json`仅低频软件检查。

原对照支持 `--stage data|indicator|reconstruct`、`--strategy frozen|full`；只有显式 `--mode sensitivity` 才可运行。旧 `--mode dual`拒绝执行并说明方法区别。此前缓存、历史 `results/`、服务器日志与数据保留。原实现的数学与历史验证记录见 [对照方法说明](docs/dual_method.md)，旧声学方法见 [legacy](docs/legacy.md)。

依赖见 `requirements.txt`，无神经网络。本次纠偏不自动提交、推送或启动服务器任务。
