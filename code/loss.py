import numpy as np

class SoftmaxCrossEntropy:
    def forward(self, logits, y):
        """
        参数：
            logits - 网络最后一层的原始输出，shape (N, C)
            y - 真实标签，shape (N,)，每个值是 0~C-1 的整数
        返回：
            loss - 标量（一个数字）
        
        步骤：
        1. 数值稳定的 softmax → 得到 probs, shape (N, C)
        2. 计算 cross-entropy loss
        3. 缓存 probs 和 y（backward 要用）
        """
        # softmax
        logits_shifted = logits - np.max(logits, axis = 1, keepdims = True)
        exp_logits = np.exp(logits_shifted)
        probs = exp_logits / np.sum(exp_logits, axis = 1, keepdims = True)

        # cross-entropy loss
        N = y.shape[0]
        loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-8))
        
        self.probs = probs
        self.y = y
        self.N = N
        return loss
    
    def backward(self):
        """
        返回：grad - 损失对 logits 的梯度，shape (N, C)
        
        提示：就是 (probs - one_hot) / N
        """
        # one-hot 编码
        one_hot = np.zeros_like(self.probs)
        one_hot[np.arange(self.N), self.y] = 1  
        return (self.probs - one_hot) / self.N
