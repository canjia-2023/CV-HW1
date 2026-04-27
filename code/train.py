import numpy as np
from load_data import load_fashion_mnist
from model import MLP
from loss import SoftmaxCrossEntropy
from optimizer import SGD
from scheduler import StepLR
import os
os.makedirs('save_models', exist_ok=True)

def train(hidden_dim1=256,hidden_dim2=128,lr=0.1,weight_decay=1e-4,
          epochs=10,activation='relu',step_size=5,gamma=0.5):
    # 加载数据与处理数据
    X_train, y_train, X_val, y_val, X_test, y_test = load_fashion_mnist()

    # 搭建网络
    model = MLP(hidden_dim1,hidden_dim2,activation)
    criterion = SoftmaxCrossEntropy()
    optimizer = SGD(model.get_linear_layers(),lr,weight_decay)
    scheduler = StepLR(optimizer,step_size,gamma)

    # 训练
    batch_size = 64
    N = X_train.shape[0]
    best_val_acc = 0
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}


    for epoch in range(epochs):
        indices = np.arange(N)
        np.random.shuffle(indices)
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        total_loss = 0
        num_batches = 0

        for start in range(0,N,batch_size):
            end = min(start + batch_size, N)
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]

            # forward
            out = model.forward(X_batch)

            # loss
            loss = criterion.forward(out,y_batch)
            # backward
            grad = criterion.backward()
            grad = model.backward(grad)
            # update
            optimizer.step()

            total_loss += loss
            num_batches +=1

        avg_loss = total_loss/num_batches

        # lr update
        scheduler.step()

        # Acc in validation set
        out = model.forward(X_val)
        val_loss = criterion.forward(out, y_val)
        predictions = np.argmax(out, axis=1)
        acc = np.mean(predictions == y_val)

        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(acc)

        print(f"Epoch {epoch+1}/{epochs}  Loss: {avg_loss:.4f}  Val Loss: {val_loss:.4f}  Val Acc: {acc:.4f}")

        if acc > best_val_acc:
            best_val_acc = acc
            layers = model.get_linear_layers()
            np.savez('save_models/best_model.npz',
                    W0=layers[0].W, b0=layers[0].b,
                    W1=layers[1].W, b1=layers[1].b,
                    W2=layers[2].W, b2=layers[2].b)
            
    return best_val_acc, history

if __name__ == '__main__':
    train(hidden_dim1=1024,hidden_dim2=256,lr=0.2,        
        weight_decay=0,epochs=30,activation='relu',
        step_size=5,gamma=0.5)