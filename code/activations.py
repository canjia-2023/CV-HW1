import numpy as np

class ReLU:
    def forward(self, x):
        """
        参数：x - 输入，shape 任意
        返回：output - 与 x 同 shape
        要求：把 x 缓存起来，backward 要用
        """
        self.x = x
        output = np.maximum(0,x)
        return output
    
    def backward(self, grad_output):
        """
        参数：grad_output - 从后一层传来的梯度，shape 与 forward 的输出相同
        返回：grad_input - 传给前一层的梯度，shape 与 forward 的输入相同
        """
        mask = (self.x > 0).astype(float)
        grad_input = grad_output * mask
        return grad_input

class Sigmoid:
    def forward(self, x):
        output = 1 / (1+np.exp(-x))
        self.output = output
        return output
    def backward(self, grad_output):
        mask = self.output*(1-self.output)
        grad_input = grad_output * mask
        return grad_input
    

class Tanh:
    def forward(self, x):
        output = np.tanh(x)
        self.output = output
        return output
    def backward(self, grad_output):
        mask = 1- self.output**2
        grad_input = grad_output * mask
        return grad_input