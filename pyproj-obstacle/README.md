# pyproj-obstacle — 障碍物成像

声软障碍物的直接采样成像、有限孔径成像、Gauss-Newton 定量重建和混合成像。

## 结构

```
pyproj-obstacle/
├── common/
│   ├── direct_sampling.py    # 直接/正交采样指标计算流程
│   └── reconstruction.py     # MUSIC、Gauss-Newton、峰值选择
├── experiments/
│   ├── direct_imaging.py     # 全孔径直接采样/正交采样成像
│   ├── limited_aperture_imaging.py  # 有限孔径直接采样成像
│   ├── joint_gn.py           # 星形障碍物联合 Gauss-Newton 重建
│   ├── hybrid_imaging.py     # 直接成像先验 + Gauss-Newton 迭代
│   ├── prior_sensitivity.py  # 初始先验对重建的影响
│   └── apple_imaging.py      # 苹果形障碍物 phaseless 成像
├── main.py
└── README.md
```

## 实验说明

| 脚本 | 实验内容 | 成像方法 |
|------|---------|---------|
| `direct_imaging.py` | 全孔径三障碍物定性成像 | 直接采样/正交采样 |
| `limited_aperture_imaging.py` | 有限孔径下的小/中/大障碍物成像 | 直接采样 |
| `joint_gn.py` | 星形障碍物定量重建 | 联合 Gauss-Newton |
| `hybrid_imaging.py` | 定性→定量混合重建 | DSM + GN |
| `prior_sensitivity.py` | 初始先验误差 vs 重建质量 | GN |
| `apple_imaging.py` | 苹果形障碍物 | 直接采样(phaseless) |

## 运行

```bash
.venv/bin/python pyproj-obstacle/main.py --mode all
.venv/bin/python pyproj-obstacle/main.py --mode direct
```
