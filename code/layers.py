import numpy as np

class Linear:
    def __init__(self, D_in, D_out):
        """
        初始化权重和偏置
        - self.W: shape (D_in, D_out)，用 He 初始化
        - self.b: shape (D_out,)，初始化为 0
        - self.grad_W: 存权重梯度（先设为 None）
        - self.grad_b: 存偏置梯度（先设为 None）
        """
        # 初始化
        self.W = np.random.randn(D_in, D_out) * np.sqrt(2.0 / D_in)
        self.b = np.zeros(D_out)
        self.grad_W = np.zeros((D_in, D_out))
        self.grad_b = np.zeros(D_out)
    
    def forward(self, X):
        """
        前向传播：Y = X @ W + b
        - 缓存 X（backward 要用）
        - 返回 Y
        """
        Y = X @ self.W + self.b
        self.X = X
        return Y
    
    def backward(self, grad_output):
        """
        反向传播：
        1. 计算 self.grad_W（权重梯度）
        2. 计算 self.grad_b（偏置梯度）
        3. 计算并返回 grad_input（传给前一层）
        """
        self.grad_W = (self.X).T @ grad_output
        self.grad_b = np.sum(grad_output, axis=0)
        grad_input = grad_output @ (self.W).T
        return grad_input
