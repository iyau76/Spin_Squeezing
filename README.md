# Spin Squeezing

这个仓库收集了若干自旋压缩（spin squeezing）相关的数值复现材料，包含：

- OAT（One-Axis Twisting）模型的大粒子数优化脚本。
- 动态合成 OAT / TAT 系统的并行仿真脚本。
- OAT、TAT、XYZ 以及脉冲序列等效模型的论文复现 Notebook。
- 已导出的图像与动画结果。

## 仓库结构

```text
.
├── Figure/        # 已生成的图片与 GIF 结果
├── notebooks/     # 论文复现与探索性分析 Notebook
├── scripts/       # 可直接运行的 Python 仿真脚本
└── README.md
```

## 文件说明

### `scripts/`

- `OAT模型大粒子数优化.py`：针对 OAT 模型的大粒子数扫描与最优压缩时间/方差计算。
- `合成OAT系统大粒子数优化.py`：动态合成 OAT 系统，与有效 OAT 理论做对比。
- `合成TAT系统大粒子数优化.py`：动态合成 TAT 系统，与有效 TAT 理论做对比。

### `notebooks/`

- `OAT模型论文复现.ipynb`：OAT 模型论文图像/结果复现。
- `TAT模型论文复现.ipynb`：TAT 模型论文图像/结果复现。
- `XYZ模型论文复现.ipynb`：XYZ 模型相关复现。
- `脉冲序列等效OAT和TAT论文复现.ipynb`：脉冲序列生成等效 OAT/TAT 的复现实验。

### `Figure/`

保存运行脚本或 Notebook 后导出的图片与动画，方便直接查看已有结果。

## 运行环境

建议使用 Python 3.10+，并安装以下常用依赖：

```bash
pip install numpy matplotlib scipy qutip tqdm
```

如果你使用 Jupyter Notebook，还需要：

```bash
pip install notebook
```

## 使用方式

### 运行脚本

在仓库根目录下执行：

```bash
python scripts/OAT模型大粒子数优化.py
python scripts/合成OAT系统大粒子数优化.py
python scripts/合成TAT系统大粒子数优化.py
```

这些脚本会进行数值演化、并行扫描，并弹出或生成相应图像结果。

### 打开 Notebook

```bash
jupyter notebook notebooks/
```

然后按需打开对应的 Notebook 进行复现和修改。

## 整理说明

本次整理主要做了以下调整：

- 将根目录下的 Python 脚本统一收纳到 `scripts/`。
- 将论文复现 Notebook 统一收纳到 `notebooks/`。
- 保留 `Figure/` 目录作为结果输出目录，避免影响已有图片资源引用。
- 新增本 README，方便快速理解项目内容与运行方式。

## 后续建议

如果你希望继续完善这个仓库，建议下一步可以补充：

1. `requirements.txt` 或 `environment.yml`，便于一键安装环境。
2. 对各脚本输出文件路径进行统一配置。
3. 为关键模型补充参数说明、公式来源和对应论文引用。
4. 将 Notebook 中稳定的核心逻辑逐步沉淀为可复用的 `.py` 模块。
