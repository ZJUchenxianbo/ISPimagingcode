# pyproj-limited-aperture — 有限孔径分析

有限孔径核函数相干性、Gram 矩阵条件数和三维 spherical-cap 因子分析。

## 实验

| 脚本 | 内容 |
|------|------|
| `experiments/coherence.py` | 孔径宽度/双孔径 coherence、Gram 条件数、噪声稳定性、cap3d |

## 运行

```bash
.venv/bin/python pyproj-limited-aperture/main.py
.venv/bin/python pyproj-limited-aperture/experiments/coherence.py --experiments widths,gram,cap3d
```
