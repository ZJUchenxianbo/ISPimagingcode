# point_scatterer — 点散射体成像

各向同性点散射体的解析远场、直接采样成像和 U-Net 后处理。

## 实验

| 脚本 | 内容 |
|------|------|
| `experiments/imaging.py` | 点散射体解析远场、有限孔径 DSM、U-Net 增强 |

## 运行

```bash
.venv/bin/python point_scatterer/main.py
.venv/bin/python point_scatterer/experiments/imaging.py --no-unet
```
