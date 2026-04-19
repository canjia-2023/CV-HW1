class StepLR:
    def __init__(self, optimizer, step_size, gamma=0.5):
        
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.epoch = 0

    def step(self):
        '''
        每个 epoch 结束后调用
        1. current_epoch += 1
        2. 如果到了衰减的时机 → optimizer.lr *= gamma
        3. (可选) 打印一句提示
        '''
        self.epoch += 1
        if self.epoch % self.step_size == 0:
            self.optimizer.lr *= self.gamma
            print(f"学习率衰减: lr = {self.optimizer.lr}")
