# import numpy as np
# import itertools
# from train import train

# search_space = {
#     'lr':           [0.01, 0.05, 0.1],
#     'hidden_dim1':  [128, 256, 512],
#     'hidden_dim2':  [64, 128, 256],
#     'weight_decay': [0, 1e-4, 1e-3],
# }

# def grid_search(search_space, epochs=10):

#     keys = list(search_space.keys())
#     values = list(search_space.values())
#     all_combos = list(itertools.product(*values))
#     total = len(all_combos)

#     results = []

#     for i, combo in enumerate(all_combos):
#         # 把 combo 和 keys 组成字典
#         params = dict(zip(keys, combo))

#         print(f"\n{'='*50}")
#         print(f"[{i+1}/{total}] lr={params['lr']}, h1={params['hidden_dim1']}, h2={params['hidden_dim2']}, wd={params['weight_decay']}")
#         print(f"{'='*50}")

#         # 调用 train，记录结果
#         best_val_acc, _ = train(
#             hidden_dim1=params['hidden_dim1'],
#             hidden_dim2=params['hidden_dim2'],
#             lr=params['lr'],
#             weight_decay=params['weight_decay'],
#             epochs=epochs
#         )

#         params['acc'] = best_val_acc
#         results.append(params)
#         print(f">>> 本组最佳验证准确率: {best_val_acc:.4f}")

#     # 按准确率从高到低排序
#     results.sort(key=lambda x: x['acc'], reverse=True)

#     # 打印 Top 5
#     print(f"\n{'='*50}")
#     print("Top 5 超参数组合")
#     print(f"{'='*50}")
#     for rank, r in enumerate(results[:5]):
#         print(f"  #{rank+1}  acc={r['acc']:.4f}  lr={r['lr']}, h1={r['hidden_dim1']}, h2={r['hidden_dim2']}, wd={r['weight_decay']}")

#     return results

# if __name__ == '__main__':
#     '''
#     ==================================================
#     Top 5 超参数组合
#     ==================================================
#     #1  acc=0.8927  lr=0.1, h1=512, h2=128, wd=0
#     #2  acc=0.8925  lr=0.1, h1=512, h2=256, wd=0
#     #3  acc=0.8905  lr=0.1, h1=512, h2=64, wd=0
#     #4  acc=0.8905  lr=0.1, h1=512, h2=128, wd=0.0001
#     #5  acc=0.8880  lr=0.1, h1=512, h2=256, wd=0.0001
#     '''
#     results = grid_search(search_space, epochs=10)


import numpy as np
import itertools
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from train import train

# ======================== 搜索空间 ========================
search_space = {
    'lr':           [0.05, 0.1, 0.2],          # 往上探到 0.2
    'hidden_dim1':  [256, 512, 1024],           # 往上探到 1024
    'hidden_dim2':  [64, 128, 256],
    'weight_decay': [0, 1e-4, 1e-3],
    'activation':   ['relu', 'sigmoid'],        # 作业要求对比
}

SEARCH_EPOCHS = 10
RESULT_DIR = 'search_results'


def grid_search(search_space, epochs=SEARCH_EPOCHS):
    os.makedirs(RESULT_DIR, exist_ok=True)

    keys = list(search_space.keys())
    values = list(search_space.values())
    all_combos = list(itertools.product(*values))
    total = len(all_combos)

    results = []

    for i, combo in enumerate(all_combos):
        params = dict(zip(keys, combo))

        print(f"\n{'='*60}")
        print(f"[{i+1}/{total}] lr={params['lr']}, h1={params['hidden_dim1']}, "
              f"h2={params['hidden_dim2']}, wd={params['weight_decay']}, "
              f"act={params['activation']}")
        print(f"{'='*60}")

        best_val_acc, history = train(
            hidden_dim1=params['hidden_dim1'],
            hidden_dim2=params['hidden_dim2'],
            lr=params['lr'],
            weight_decay=params['weight_decay'],
            activation=params['activation'],
            epochs=epochs,
        )

        params['best_val_acc'] = best_val_acc
        # 记录每个 epoch 的 loss 和 acc，方便后续画收敛曲线对比
        params['history_train_loss'] = [float(x) for x in history['train_loss']]
        params['history_val_acc'] = [float(x) for x in history['val_acc']]
        results.append(params)

        print(f">>> 本组最佳验证准确率: {best_val_acc:.4f}")

    # 按准确率排序
    results.sort(key=lambda x: x['best_val_acc'], reverse=True)

    # 保存全部结果到 JSON
    result_path = os.path.join(RESULT_DIR, 'grid_search_results.json')
    with open(result_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ 全部结果已保存到 {result_path}")

    # 打印 Top 10
    print(f"\n{'='*60}")
    print("Top 10 超参数组合")
    print(f"{'='*60}")
    for rank, r in enumerate(results[:10]):
        print(f"  #{rank+1}  acc={r['best_val_acc']:.4f}  "
              f"lr={r['lr']}, h1={r['hidden_dim1']}, h2={r['hidden_dim2']}, "
              f"wd={r['weight_decay']}, act={r['activation']}")

    return results


# ======================== 可视化 ========================

def load_results(path=None):
    if path is None:
        path = os.path.join(RESULT_DIR, 'grid_search_results.json')
    with open(path, 'r') as f:
        return json.load(f)


def plot_heatmap_h1_h2(results):
    """固定最优 lr/wd/activation，画 h1 × h2 热力图"""
    # 取 Top1 的 lr, wd, activation 作为固定值
    best = results[0]
    fix_lr, fix_wd, fix_act = best['lr'], best['weight_decay'], best['activation']

    filtered = [r for r in results
                if r['lr'] == fix_lr
                and r['weight_decay'] == fix_wd
                and r['activation'] == fix_act]

    h1_vals = sorted(set(r['hidden_dim1'] for r in filtered))
    h2_vals = sorted(set(r['hidden_dim2'] for r in filtered))

    matrix = np.zeros((len(h1_vals), len(h2_vals)))
    for r in filtered:
        i = h1_vals.index(r['hidden_dim1'])
        j = h2_vals.index(r['hidden_dim2'])
        matrix[i][j] = r['best_val_acc']

    plt.figure(figsize=(6, 4.5))
    sns.heatmap(matrix, annot=True, fmt='.4f', cmap='YlGnBu',
                xticklabels=h2_vals, yticklabels=h1_vals)
    plt.xlabel('Hidden Dim 2')
    plt.ylabel('Hidden Dim 1')
    plt.title(f'Val Accuracy Heatmap\n(lr={fix_lr}, wd={fix_wd}, act={fix_act})')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'heatmap_h1_h2.png'), dpi=150)
    plt.show()
    print("✅ heatmap_h1_h2.png saved")


def plot_sensitivity(results):
    """
    对每个超参数，固定其他参数取最优组合对应的值，
    画该超参数变化对 acc 的影响 —— 体现超参数敏感度
    """
    best = results[0]
    hp_names = ['lr', 'hidden_dim1', 'hidden_dim2', 'weight_decay', 'activation']

    fig, axes = plt.subplots(1, len(hp_names), figsize=(4 * len(hp_names), 4))

    for idx, hp in enumerate(hp_names):
        # 固定除 hp 之外的所有参数为 best 的值
        fixed = {k: best[k] for k in hp_names if k != hp}
        filtered = [r for r in results
                    if all(r[k] == fixed[k] for k in fixed)]

        xs = [r[hp] for r in filtered]
        ys = [r['best_val_acc'] for r in filtered]

        # 排序
        pairs = sorted(zip(xs, ys), key=lambda p: (p[0] if isinstance(p[0], (int, float)) else str(p[0])))
        xs, ys = zip(*pairs)

        ax = axes[idx]
        ax.bar(range(len(xs)), ys, color='steelblue', alpha=0.8)
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([str(x) for x in xs], rotation=30, fontsize=9)
        ax.set_xlabel(hp)
        ax.set_ylabel('Val Accuracy')
        ax.set_title(f'Sensitivity: {hp}')
        # y 轴范围只显示高区间，差异更明显
        all_accs = [r['best_val_acc'] for r in results]
        ax.set_ylim(min(all_accs) - 0.01, max(all_accs) + 0.005)

    plt.suptitle('Hyperparameter Sensitivity Analysis', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'sensitivity.png'), dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ sensitivity.png saved")


def plot_activation_comparison(results):
    """ReLU vs Sigmoid 的整体分布对比（box plot）"""
    relu_accs = [r['best_val_acc'] for r in results if r['activation'] == 'relu']
    sigmoid_accs = [r['best_val_acc'] for r in results if r['activation'] == 'sigmoid']

    plt.figure(figsize=(5, 4))
    plt.boxplot([relu_accs, sigmoid_accs], labels=['ReLU', 'Sigmoid'])
    plt.ylabel('Val Accuracy')
    plt.title('Activation Function Comparison (All Configs)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'activation_comparison.png'), dpi=150)
    plt.show()
    print("✅ activation_comparison.png saved")


def plot_top_k_curves(results, k=5):
    """Top-K 组合的训练收敛曲线对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    for rank in range(min(k, len(results))):
        r = results[rank]
        label = (f"#{rank+1} lr={r['lr']},h1={r['hidden_dim1']},"
                 f"h2={r['hidden_dim2']},wd={r['weight_decay']},{r['activation']}")
        epochs_range = range(1, len(r['history_train_loss']) + 1)

        ax1.plot(epochs_range, r['history_train_loss'], label=label)
        ax2.plot(epochs_range, r['history_val_acc'], label=label)

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Train Loss')
    ax1.set_title('Training Loss Curves (Top-K)')
    ax1.legend(fontsize=7)

    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Val Accuracy')
    ax2.set_title('Validation Accuracy Curves (Top-K)')
    ax2.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'top_k_curves.png'), dpi=150)
    plt.show()
    print("✅ top_k_curves.png saved")


# ======================== 主入口 ========================

if __name__ == '__main__':
    # Step 1: 跑搜索（如果已跑过，可以注释掉直接 load）
    results = grid_search(search_space, epochs=SEARCH_EPOCHS)

    # Step 2: 生成可视化（也可以单独跑，load 已有结果即可）
    # results = load_results()   # 如果之前跑过，直接加载

    plot_heatmap_h1_h2(results)
    plot_sensitivity(results)
    plot_activation_comparison(results)
    plot_top_k_curves(results, k=5)

'''
============================================================
Top 10 超参数组合
============================================================
  #1  acc=0.8984  lr=0.2, h1=1024, h2=256, wd=0, act=relu
  #2  acc=0.8977  lr=0.2, h1=1024, h2=256, wd=0.0001, act=relu
  #3  acc=0.8955  lr=0.2, h1=1024, h2=128, wd=0.0001, act=relu
  #4  acc=0.8943  lr=0.2, h1=1024, h2=64, wd=0.0001, act=relu
  #5  acc=0.8933  lr=0.2, h1=512, h2=128, wd=0, act=relu
  #6  acc=0.8927  lr=0.1, h1=512, h2=256, wd=0, act=relu
  #7  acc=0.8920  lr=0.2, h1=256, h2=128, wd=0, act=relu
  #8  acc=0.8919  lr=0.2, h1=512, h2=64, wd=0.0001, act=relu
  #9  acc=0.8916  lr=0.2, h1=512, h2=256, wd=0, act=relu
  #10  acc=0.8915  lr=0.1, h1=256, h2=128, wd=0.0001, act=relu
'''