# HW1: Fashion-MNIST 三层 MLP 分类器

基于 NumPy 从零实现三层多层感知机（MLP），在 Fashion-MNIST 数据集上完成 10 类服装图像分类。不依赖任何支持自动微分的深度学习框架，所有前向传播、反向传播与参数更新均手动实现。


---

## 目录

- [环境依赖](#环境依赖)
- [项目结构](#项目结构)
- [网络结构](#网络结构)
- [快速开始](#快速开始)
- [超参数搜索](#超参数搜索)
- [可视化](#可视化)
- [模型权重](#模型权重)

---

## 环境依赖

**Python 3.10**，推荐使用 Conda 创建隔离环境：

```bash
conda env create -f environment.yml
conda activate fashion-mnist
```

主要依赖：

| 包 | 用途 |
|---|---|
| `numpy` | 矩阵运算（核心依赖） |
| `tensorflow` | 仅用于下载 Fashion-MNIST 数据集 |
| `scikit-learn` | 混淆矩阵辅助计算 |
| `matplotlib` / `seaborn` | 可视化 |
| `ipykernel` | Jupyter Notebook 支持 |

---

## 项目结构

```
HW1/
├── layers.py          # Linear 层（含权重初始化、前向/反向传播）
├── activations.py     # 激活函数：ReLU / Sigmoid / Tanh
├── loss.py            # Softmax + 交叉熵损失
├── optimizer.py       # SGD 优化器（含 L2 正则化）
├── scheduler.py       # StepLR 学习率衰减
├── model.py           # MLP 模型（784 → h1 → h2 → 10）
├── load_data.py       # 数据加载与预处理
├── train.py           # 训练脚本
├── evaluate.py        # 测试评估 + 混淆矩阵
├── search.py          # 网格搜索超参数
├── visualize.py       # 可视化模块（训练曲线、权重、错例等）
├── visualization.ipynb # 可视化展示 Notebook
├── environment.yml    # Conda 环境配置
├── save_models/
│   └── best_model.npz # 最优模型权重
└── figures/           # 自动生成的可视化图像
```

---

## 网络结构

```
输入 (784)
  └─ Linear(784 → hidden_dim1)
  └─ Activation (ReLU / Sigmoid / Tanh)
  └─ Linear(hidden_dim1 → hidden_dim2)
  └─ Activation
  └─ Linear(hidden_dim2 → 10)
  └─ Softmax + 交叉熵损失
```

- **激活函数**：通过 `activation` 参数在 `relu` / `sigmoid` / `tanh` 之间切换
- **优化器**：SGD + L2 Weight Decay
- **学习率策略**：StepLR（每 `step_size` 个 epoch 乘以 `gamma`）
- **最优权重保存**：按验证集准确率自动保存至 `save_models/best_model.npz`

---

## 快速开始

### 训练

```bash
python train.py
```

使用默认超参数（`hidden_dim1=256, hidden_dim2=128, lr=0.1, weight_decay=1e-4, epochs=10`）训练，训练过程实时打印每个 epoch 的 Loss 和验证集准确率，最优权重自动保存到 `save_models/best_model.npz`。

也可在代码中修改 `train()` 的参数，或直接调用：

```python
from train import train

best_val_acc, history = train(
    hidden_dim1=512,
    hidden_dim2=128,
    lr=0.1,
    weight_decay=0,
    epochs=20,
    activation='relu',
    step_size=5,
    gamma=0.5
)
```

### 测试评估

```bash
python evaluate.py
```

加载 `save_models/best_model.npz`，在独立测试集上输出分类准确率，并打印各类别混淆矩阵与逐类准确率。

示例输出：

```
测试集准确率: 0.8930

混淆矩阵 (行=真实, 列=预测)
...
各类别准确率:
  T-shirt/top  ████████████████░░░░ 82.0% (820/1000)
      Trouser  ████████████████████ 97.7% (977/1000)
      ...
```

---

## 超参数搜索

```bash
python search.py
```

对以下搜索空间执行网格搜索（共 81 种组合）：

| 超参数 | 候选值 |
|---|---|
| `lr` | 0.01 / 0.05 / 0.1 |
| `hidden_dim1` | 128 / 256 / 512 |
| `hidden_dim2` | 64 / 128 / 256 |
| `weight_decay` | 0 / 1e-4 / 1e-3 |

搜索结束后打印 Top 5 结果。**最优组合：**

```
#1  acc=0.8927  lr=0.1, h1=512, h2=128, wd=0
#2  acc=0.8925  lr=0.1, h1=512, h2=256, wd=0
#3  acc=0.8905  lr=0.1, h1=512, h2=64,  wd=0
```

---

## 可视化

运行可视化脚本（需先完成训练）：

```bash
python visualize.py
```

或打开 Notebook 查看完整报告展示：

```bash
jupyter notebook visualization.ipynb
```

生成的图像保存在 `figures/` 目录，包括：

| 图像 | 内容 |
|---|---|
| `training_curves.png` | 训练 Loss 曲线 + 验证集 Accuracy 曲线（含 LR Decay 标注） |
| `confusion_matrix.png` | 混淆矩阵热力图（计数 + 归一化双图） |
| `weight_visualization.png` | 第一隐藏层权重恢复为 28×28 图像（按 L2 范数排序） |
| `error_examples.png` | 测试集分类错误样本展示（含真实/预测类别及置信度） |
| `per_class_accuracy.png` | 各类别准确率水平条形图 |

---

## 模型权重

训练好的最优模型权重（`.npz` 格式）已上传至 Google Drive：

> **[Download](https://drive.google.com/)**（暂未上传，还需替换为实际链接）

下载后放置于 `save_models/best_model.npz` 即可直接运行 `evaluate.py` 或可视化脚本。
