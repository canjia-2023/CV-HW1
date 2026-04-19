import numpy as np
import itertools
from train import train

search_space = {
    'lr':           [0.01, 0.05, 0.1],
    'hidden_dim1':  [128, 256, 512],
    'hidden_dim2':  [64, 128, 256],
    'weight_decay': [0, 1e-4, 1e-3],
}

def grid_search(search_space, epochs=10):

    keys = list(search_space.keys())
    values = list(search_space.values())
    all_combos = list(itertools.product(*values))
    total = len(all_combos)

    results = []

    for i, combo in enumerate(all_combos):
        # 把 combo 和 keys 组成字典
        params = dict(zip(keys, combo))

        print(f"\n{'='*50}")
        print(f"[{i+1}/{total}] lr={params['lr']}, h1={params['hidden_dim1']}, h2={params['hidden_dim2']}, wd={params['weight_decay']}")
        print(f"{'='*50}")

        # 调用 train，记录结果
        best_val_acc, _ = train(
            hidden_dim1=params['hidden_dim1'],
            hidden_dim2=params['hidden_dim2'],
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            epochs=epochs
        )

        params['acc'] = best_val_acc
        results.append(params)
        print(f">>> 本组最佳验证准确率: {best_val_acc:.4f}")

    # 按准确率从高到低排序
    results.sort(key=lambda x: x['acc'], reverse=True)

    # 打印 Top 5
    print(f"\n{'='*50}")
    print("Top 5 超参数组合")
    print(f"{'='*50}")
    for rank, r in enumerate(results[:5]):
        print(f"  #{rank+1}  acc={r['acc']:.4f}  lr={r['lr']}, h1={r['hidden_dim1']}, h2={r['hidden_dim2']}, wd={r['weight_decay']}")

    return results

if __name__ == '__main__':
    '''
    ==================================================
    Top 5 超参数组合
    ==================================================
    #1  acc=0.8927  lr=0.1, h1=512, h2=128, wd=0
    #2  acc=0.8925  lr=0.1, h1=512, h2=256, wd=0
    #3  acc=0.8905  lr=0.1, h1=512, h2=64, wd=0
    #4  acc=0.8905  lr=0.1, h1=512, h2=128, wd=0.0001
    #5  acc=0.8880  lr=0.1, h1=512, h2=256, wd=0.0001
    '''
    results = grid_search(search_space, epochs=10)
