import numpy as np

class SGD:
    def __init__(self, layers, lr=0.01, weight_decay=0):
        """
        参数：
            layers       - 所有有参数的层（Linear 层）
            lr           - 学习率
            weight_decay - L2 正则化系数（默认 0 = 不加正则）
        """
        self.layers = layers
        self.lr = lr
        self.weight_decay = weight_decay
    
    def step(self):
        """
        遍历每一层，用 SGD 公式更新 W 和 b：
            W -= lr * (grad_W + weight_decay * W)
            b -= lr * grad_b
        """
        for layer in self.layers: 
            layer.W -= self.lr * (layer.grad_W + self.weight_decay * layer.W)
            layer.b -= self.lr * layer.grad_b
