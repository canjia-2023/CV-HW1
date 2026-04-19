import numpy as np
from layers import Linear
from activations import ReLU, Sigmoid, Tanh

class MLP:
    def __init__(self, hidden_dim1=256, hidden_dim2=128, activation='relu'):
        """
        三层 MLP：784 → hidden_dim → hidden_dim2 → 10
        
        1. 根据 activation 参数选择激活函数类
        2. 创建三个 Linear 层和两个激活函数层
        3. 把所有层按顺序存到一个列表 self.layers 中
        """
        linear1 = Linear(784,hidden_dim1)
        linear2 = Linear(hidden_dim1,hidden_dim2)
        linear3 = Linear(hidden_dim2,10)
        if activation == 'relu':
            act_class = ReLU
        elif activation == 'sigmoid':
            act_class = Sigmoid
        elif activation == 'tanh':
            act_class = Tanh
        else:
            raise ValueError(f"不支持的激活函数: {activation}") 
        act1 = act_class()
        act2 = act_class()

        self.layers = [linear1, act1, linear2, act2, linear3]
    
    def forward(self, X):
        """
        遍历 self.layers，依次调用每层的 forward
        """
        out = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            out = layer.forward(out)
        return out
    
    def backward(self, grad):
        """
        反向遍历 self.layers，依次调用每层的 backward
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
    
    def get_linear_layers(self):
        """
        返回所有 Linear 层（给 optimizer 用）
        从 self.layers 中筛选出 Linear 类型的层
        """
        linear_list = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                linear_list.append(layer)
        return linear_list
