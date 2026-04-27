# HW1: Fashion-MNIST 三层 MLP 分类器

基于 NumPy 从零实现三层多层感知机（MLP），在 Fashion-MNIST 数据集上完成 10 类服装图像分类。不依赖任何支持自动微分的深度学习框架（PyTorch / TensorFlow / JAX 等），所有前向传播、反向传播与参数更新均手动实现。

最终模型在独立测试集上取得约 **89.79%** 的分类准确率。

---

## 目录

- [环境依赖](#环境依赖)
- [项目结构](#项目结构)
- [网络结构](#网络结构)
- [快速开始](#快速开始)
- [超参数搜索](#超参数搜索)
- [实验结果](#实验结果)
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

> **说明**：TensorFlow 仅调用 `keras.datasets.fashion_mnist.load_data()` 下载数据，不参与模型训练。

---

## 项目结构

```
HW1/
├── layers.py            # Linear 层（He 初始化、前向/反向传播）
├── activations.py       # 激活函数：ReLU / Sigmoid / Tanh 及其导数
├── loss.py              # Softmax + 交叉熵损失（数值稳定实现）
├── optimizer.py         # SGD 优化器（含 L2 正则化）
├── scheduler.py         # StepLR 学习率衰减
├── model.py             # MLP 模型封装（784 → h1 → h2 → 10）
├── load_data.py         # 数据加载与预处理（展平、归一化、划分验证集）
├── train.py             # 训练脚本（含验证、最优权重自动保存）
├── evaluate.py          # 测试评估 + 混淆矩阵 + 逐类准确率
├── search.py            # 网格搜索超参数（162 种组合）
├── visualize.py         # 可视化模块（训练曲线、权重、错例等）
├── visualization.ipynb  # 可视化展示 Notebook
├── environment.yml      # Conda 环境配置
├── save_models/
│   └── best_model.npz   # 最优模型权重
└── figures/             # 自动生成的可视化图像
```

---

## 网络结构

```
Input (784)
  └─ Linear(784 → 1024)
  └─ ReLU
  └─ Linear(1024 → 256)
  └─ ReLU
  └─ Linear(256 → 10)
  └─ Softmax → Cross-Entropy Loss
```

- **权重初始化**：He 初始化 $ W \sim \mathcal{N}(0, \sqrt{2/n_{\text{in}}}) $，偏置初始化为零
- **激活函数**：支持 `relu` / `sigmoid` / `tanh` 切换（最优为 ReLU）
- **优化器**：SGD + 可选 L2 Weight Decay
- **学习率策略**：StepLR（每 `step_size` 个 epoch 乘以 `gamma`）
- **最优权重保存**：按验证集准确率自动保存至 `save_models/best_model.npz`

---

## 快速开始

### 训练

```bash
python train.py
```

默认使用搜索得到的最优超参数进行训练。也可在代码中自定义参数：

```python
from train import train

best_val_acc, history = train(
    hidden_dim1=1024,
    hidden_dim2=256,
    lr=0.2,
    weight_decay=0,
    epochs=30,
    activation='relu',
    step_size=5,
    gamma=0.5
)
```

训练过程实时打印每个 epoch 的 Train Loss、Val Loss 和 Val Accuracy，最优权重自动保存到 `save_models/best_model.npz`。

### 测试评估

```bash
python evaluate.py
```

加载 `save_models/best_model.npz`，在独立测试集上输出分类准确率，并打印混淆矩阵与逐类准确率。

---

## 超参数搜索

```bash
python search.py
```

对以下搜索空间执行网格搜索（共 **162** 种组合）：

| 超参数 | 候选值 |
|---|---|
| `lr` | 0.05 / 0.1 / 0.2 |
| `hidden_dim1` | 256 / 512 / 1024 |
| `hidden_dim2` | 64 / 128 / 256 |
| `weight_decay` | 0 / 1e-4 / 1e-3 |
| `activation` | relu / sigmoid |

每组配置固定训练 10 epochs（StepLR: step=5, γ=0.5），以验证集准确率为评价指标。

**Top-5 搜索结果：**

| 排名 | lr | h1 | h2 | wd | act | Val Acc |
|---|---|---|---|---|---|---|
| 1 | 0.2 | 1024 | 256 | 0 | relu | 89.84% |
| 2 | 0.2 | 1024 | 256 | 1e-4 | relu | 89.77% |
| 3 | 0.2 | 1024 | 128 | 1e-4 | relu | 89.55% |
| 4 | 0.2 | 1024 | 64 | 1e-4 | relu | 89.43% |
| 5 | 0.2 | 512 | 128 | 0 | relu | 89.33% |

---

## 实验结果

### 最优模型配置

| h_dim1 | h_dim2 | Activation | lr | step | γ | Weight Decay | Batch Size | Epochs |
|---|---|---|---|---|---|---|---|---|
| 1024 | 256 | ReLU | 0.2 | 5 | 0.5 | 0 | 64 | 30 |

### 测试集性能

- **Overall Accuracy**: ~89.86%
- **最易分类**：Trouser、Bag、Sneaker、Sandal（> 95%）
- **最难分类**：Shirt（常与 T-shirt / Pullover 混淆）

详细的混淆矩阵、逐类准确率、错例分析等见 `figures/` 目录或 `visualization.ipynb`。

---

## 可视化

运行可视化脚本（需先完成训练）：

```bash
python visualize.py
```

或打开 Notebook 查看完整图表：

```bash
jupyter notebook visualization.ipynb
```

生成的图像保存在 `figures/` 目录：

| 文件 | 内容 |
|---|---|
| `training_curves.png` | Train/Val Loss + Val Accuracy 曲线（含 LR Decay 标注） |
| `confusion_matrix.png` | 混淆矩阵热力图（计数 + 归一化双图） |
| `weight_visualization.png` | 第一隐藏层权重可视化（28×28，按 L2 范数排序） |
| `error_examples.png` | 测试集误分类样本展示（含真实/预测类别及置信度） |
| `per_class_accuracy.png` | 各类别准确率水平条形图 |

---

## 模型权重

训练好的最优模型权重（`.npz` 格式）已上传至 Google Drive：

> **[Google Drive 下载链接](https://drive.google.com/drive/folders/1DlvFEaEPOvmYdFyZQ1jncWVZxO9jsYgE?usp=drive_link)**


下载后放置于 `save_models/best_model.npz`，即可直接运行 `evaluate.py` 或可视化脚本。

