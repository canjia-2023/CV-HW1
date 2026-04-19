import numpy as np
from tensorflow.keras.datasets import fashion_mnist

def load_fashion_mnist():
    """
    加载 Fashion-MNIST 数据集，返回预处理后的训练集、验证集和测试集。
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # step1:加载原始数据
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    
    # step2:拍平
    X_train = X_train.reshape(-1, 784)
    X_test = X_test.reshape(-1, 784)
    
    # step3:归一化
    X_train = X_train.astype(np.float32) / 255.0
    X_test = X_test.astype(np.float32) / 255.0

    # step4:划分验证集
    N_val = 10000
    X_val = X_train[:N_val]
    y_val = y_train[:N_val]
    X_train = X_train[N_val:]
    y_train = y_train[N_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test