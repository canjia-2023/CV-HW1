import numpy as np
from model import MLP
from loss import SoftmaxCrossEntropy
from load_data import load_fashion_mnist

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat","Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def load_model(model, path='save_models/best_model.npz'):
    """
    1. 用 np.load(path) 加载 .npz 文件（得到一个字典-like 对象）
    2. 拿到模型的 linear 层列表（model.get_linear_layers()）
    3. 把 W0, b0, W1, b1, W2, b2 分别赋值回对应层的 .W 和 .b
    """
    layers = np.load(path)
    [linear1,linear2,linear3] = model.get_linear_layers()
    linear1.W , linear1.b = layers['W0'] , layers['b0']
    linear2.W , linear2.b = layers['W1'] , layers['b1']
    linear3.W , linear3.b = layers['W2'] , layers['b2']

    
def predict(model, X):
    """
    1. 前向传播 model.forward(X)
    2. 对输出取 argmax 得到预测类别
    3. 返回预测类别数组
    """
    out = model.forward(X)
    predictions = np.argmax(out, axis=1)
    return predictions

def confusion_matrix(y_true, y_pred, num_classes=10):
    """
    1. 创建一个 (num_classes, num_classes) 的零矩阵
    2. 遍历每对 (true, pred)，在矩阵对应位置 +1
    3. 返回矩阵
    """
    matrix = np.zeros([num_classes,num_classes])
    for i, j in zip(y_true, y_pred):
        matrix[i][j] += 1
    return matrix 

def print_confusion_matrix(matrix, class_names):
    num_classes = len(class_names)
    
    # 找到最长的类别名，用于对齐
    max_name_len = max(len(name) for name in class_names)
    cell_width = 6
    
    # 表头
    header = " " * (max_name_len + 2) + "".join(f"{i:>{cell_width}}" for i in range(num_classes))
    print("\n" + "=" * len(header))
    print("混淆矩阵 (行=真实, 列=预测)")
    print("=" * len(header))
    print(header)
    print(" " * (max_name_len + 2) + "-" * (cell_width * num_classes))
    
    # 每一行
    for i in range(num_classes):
        row_label = f"{class_names[i]:>{max_name_len}} |"
        row_data = "".join(f"{int(matrix[i][j]):>{cell_width}}" for j in range(num_classes))
        print(row_label + row_data)
    
    print()
    
    # 每类准确率
    print("各类别准确率:")
    print("-" * 40)
    for i in range(num_classes):
        total = int(np.sum(matrix[i]))
        correct = int(matrix[i][i])
        acc = correct / total if total > 0 else 0
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"  {class_names[i]:>{max_name_len}}  {bar} {acc:.1%} ({correct}/{total})")


def evaluate(class_names):
    '''
    1. 加载测试数据（和 train.py 里加载方式一样，但用测试集）
    2. 预处理（归一化等，和训练时一致）
    3. 创建模型（隐藏层大小要和训练时一致！）
    4. 调用 load_model 加载权重
    5. 调用 predict 得到预测结果
    6. 计算准确率
    7. 生成混淆矩阵并打印
    '''

    # 加载数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()
    # 创建模型
    model = MLP(hidden_dim1=256, hidden_dim2=128, activation='relu')
    # 加载权重
    load_model(model, path='save_models/best_model.npz')
    # 预测
    predictions = predict(model, X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"测试集准确率: {accuracy:.4f}")

    # 混淆矩阵
    matrix = confusion_matrix(y_test, predictions)
    print_confusion_matrix(matrix,class_names)

if __name__ == '__main__':

    evaluate(CLASS_NAMES)