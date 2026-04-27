import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from model import MLP
from load_data import load_fashion_mnist
import os

os.makedirs('figures', exist_ok=True)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

_BG    = '#F8FAFC'
_GRID  = '#E2E8F0'
_BLUE  = '#3B82F6'
_GREEN = '#10B981'
_AMBER = '#F59E0B'
_TEXT  = '#1E293B'


def _setup_ax(ax):
    ax.set_facecolor(_BG)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(color=_GRID, linewidth=0.8, zorder=0)


def _load_model(path='save_models/best_model.npz'):
    data = np.load(path)
    model = MLP(
        hidden_dim1=int(data['W0'].shape[1]),
        hidden_dim2=int(data['W1'].shape[1]),
        activation='relu'
    )
    layers = model.get_linear_layers()
    for i, (wk, bk) in enumerate([('W0', 'b0'), ('W1', 'b1'), ('W2', 'b2')]):
        layers[i].W, layers[i].b = data[wk], data[bk]
    return model


# def plot_training_curves(history, lr_decay_epochs=None,
#                          save_path='figures/training_curves.png'):
#     """
#     Train-loss (left) and val-accuracy (right) side by side.

#     Parameters
#     ----------
#     history : dict  {'train_loss': [...], 'val_acc': [...]}
#     lr_decay_epochs : list[int], optional  1-indexed epochs where LR was decayed
#     """
#     epochs     = list(range(1, len(history['train_loss']) + 1))
#     best_epoch = int(np.argmax(history['val_acc'])) + 1
#     best_acc   = float(max(history['val_acc']))

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=_BG)
#     _setup_ax(ax1)
#     _setup_ax(ax2)

#     # Loss
#     ax1.plot(epochs, history['train_loss'], color=_BLUE, linewidth=2,
#              marker='o', markersize=4, label='Train Loss')

#     # Accuracy
#     ax2.plot(epochs, history['val_acc'], color=_GREEN, linewidth=2,
#              marker='s', markersize=4, label='Val Accuracy')
#     ax2.scatter([best_epoch], [best_acc], color=_AMBER, s=120, zorder=5,
#                 label=f'Best: {best_acc:.2%} @ ep{best_epoch}')
#     ax2.annotate(f'{best_acc:.2%}',
#                  xy=(best_epoch, best_acc),
#                  xytext=(best_epoch + 0.4, best_acc - 0.012),
#                  fontsize=9, color=_AMBER, fontweight='bold')

#     if lr_decay_epochs:
#         labeled = False
#         for ax in (ax1, ax2):
#             for e in lr_decay_epochs:
#                 kw = dict(color=_AMBER, linewidth=1.2, linestyle='--', alpha=0.6, zorder=2)
#                 if not labeled:
#                     kw['label'] = 'LR Decay'
#                     labeled = True
#                 ax.axvline(e, **kw)

#     for ax, title, ylabel in [
#         (ax1, 'Training Loss',       'Loss'),
#         (ax2, 'Validation Accuracy', 'Accuracy'),
#     ]:
#         ax.set_xlabel('Epoch', fontsize=11)
#         ax.set_ylabel(ylabel,  fontsize=11)
#         ax.set_title(title, fontsize=13, fontweight='bold')
#         ax.legend(fontsize=10)

#     ax2.set_ylim(max(0.0, min(history['val_acc']) - 0.05), 1.02)
#     ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))

#     plt.suptitle('Training Curves', fontsize=15, fontweight='bold', y=1.02)
#     plt.tight_layout()
#     plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=_BG)
#     plt.show()
#     print(f'Saved: {save_path}')

def plot_training_curves(history, lr_decay_epochs=None,
                         save_path='figures/training_curves.png'):
    """
    Left:  Train Loss + Val Loss
    Right: Val Accuracy
    """
    epochs     = list(range(1, len(history['train_loss']) + 1))
    best_epoch = int(np.argmax(history['val_acc'])) + 1
    best_acc   = float(max(history['val_acc']))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), facecolor=_BG)
    _setup_ax(ax1)
    _setup_ax(ax2)

    # Left: Loss
    ax1.plot(epochs, history['train_loss'], color=_BLUE, linewidth=2,
             marker='o', markersize=4, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color='#EF4444', linewidth=2,
             marker='D', markersize=4, label='Val Loss')

    # Right: Accuracy
    ax2.plot(epochs, history['val_acc'], color=_GREEN, linewidth=2,
             marker='s', markersize=4, label='Val Accuracy')
    ax2.scatter([best_epoch], [best_acc], color=_AMBER, s=120, zorder=5,
                label=f'Best: {best_acc:.2%} @ ep{best_epoch}')
    ax2.annotate(f'{best_acc:.2%}',
                 xy=(best_epoch, best_acc),
                 xytext=(best_epoch + 0.4, best_acc - 0.012),
                 fontsize=9, color=_AMBER, fontweight='bold')

    # LR decay vertical lines
    if lr_decay_epochs:
        labeled = False
        for ax in (ax1, ax2):
            for e in lr_decay_epochs:
                kw = dict(color=_AMBER, linewidth=1.2, linestyle='--',
                          alpha=0.6, zorder=2)
                if not labeled:
                    kw['label'] = 'LR Decay'
                    labeled = True
                ax.axvline(e, **kw)

    for ax, title, ylabel in [
        (ax1, 'Training / Validation Loss', 'Loss'),
        (ax2, 'Validation Accuracy',        'Accuracy'),
    ]:
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel,  fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)

    ax2.set_ylim(max(0.0, min(history['val_acc']) - 0.05), 1.02)
    ax2.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda y, _: f'{y:.0%}'))

    plt.suptitle('Training Curves', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.show()
    print(f'Saved: {save_path}')


def plot_confusion_matrix_heatmap(y_true, y_pred,
                                  save_path='figures/confusion_matrix.png'):
    """Side-by-side confusion matrix: raw counts (left) + row-normalised (right)."""
    cm = np.zeros((10, 10), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7), facecolor=_BG)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.3, linecolor='#CBD5E1', ax=ax1)
    ax1.set_xlabel('Predicted', fontsize=11, labelpad=8)
    ax1.set_ylabel('True Label', fontsize=11, labelpad=8)
    ax1.set_title('Confusion Matrix (counts)', fontsize=13, fontweight='bold')
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax1.set_yticklabels(ax1.get_yticklabels(), rotation=0, fontsize=9)

    annot = np.array([[f'{v:.0%}' for v in row] for row in cm_norm])
    sns.heatmap(cm_norm, annot=annot, fmt='', cmap='RdYlGn', vmin=0, vmax=1,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                linewidths=0.3, linecolor='#CBD5E1', ax=ax2)
    ax2.set_xlabel('Predicted', fontsize=11, labelpad=8)
    ax2.set_ylabel('True Label', fontsize=11, labelpad=8)
    ax2.set_title('Confusion Matrix (normalised)', fontsize=13, fontweight='bold')
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha='right', fontsize=9)
    ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0, fontsize=9)

    overall = float(np.mean(y_true == y_pred))
    plt.suptitle(f'Confusion Matrix  |  Test Accuracy: {overall:.2%}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.show()
    print(f'Saved: {save_path}')


def plot_first_layer_weights(model_path='save_models/best_model.npz',
                             num_show=64,
                             save_path='figures/weight_visualization.png'):
    """
    Visualise first-layer weight filters on a dark background.
    Neurons are sorted by L2 norm (most active first).
    Each filter is independently normalised to [-vmax, vmax] for clarity.
    """
    data  = np.load(model_path)
    W0    = data['W0']                              # (784, hidden_dim1)
    n     = min(num_show, W0.shape[1])
    order = np.argsort(np.linalg.norm(W0, axis=0))[::-1][:n]

    cols = 8
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 1.7, rows * 1.7 + 0.5), facecolor='#1A1A2E')
    for plot_i, neuron_i in enumerate(order):
        ax   = fig.add_subplot(rows, cols, plot_i + 1)
        w    = W0[:, neuron_i].reshape(28, 28)
        vmax = max(float(np.abs(w).max()), 1e-6)
        ax.imshow(w, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
        ax.axis('off')

    plt.suptitle('First Hidden Layer Weights  (sorted by L2 norm, reshaped to 28×28)',
                 fontsize=11, color='white', y=1.005, fontweight='bold')
    plt.tight_layout(pad=0.25)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#1A1A2E')
    plt.show()
    print(f'Saved: {save_path}')


def plot_error_examples(num_show=12, model_path='save_models/best_model.npz',
                        save_path='figures/error_examples.png'):
    """
    Show randomly chosen misclassified test images.
    Each panel displays the true label (with true-class confidence) and
    predicted label (with predicted-class confidence).
    """
    _, _, _, _, X_test, y_test = load_fashion_mnist()
    model  = _load_model(model_path)
    logits = model.forward(X_test)
    exp    = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs  = exp / exp.sum(axis=1, keepdims=True)
    preds  = np.argmax(probs, axis=1)

    wrong = np.where(preds != y_test)[0]
    np.random.seed(42)
    show  = np.random.choice(wrong, size=min(num_show, len(wrong)), replace=False)

    cols = 4
    rows = (len(show) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 4), facecolor=_BG)
    axes = axes.flatten()

    for i, idx in enumerate(show):
        ax = axes[i]
        ax.imshow(X_test[idx].reshape(28, 28), cmap='gray_r', interpolation='bilinear')
        for sp in ax.spines.values():
            sp.set_edgecolor('#EF4444')
            sp.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"True:  {CLASS_NAMES[y_test[idx]]} ({probs[idx, y_test[idx]]:.1%})\n"
            f"Pred: {CLASS_NAMES[preds[idx]]} ({probs[idx, preds[idx]]:.1%})",
            fontsize=9, color=_TEXT, fontweight='bold', linespacing=1.5
        )

    for i in range(len(show), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f'Misclassified Examples  |  {len(wrong)}/{len(y_test)} errors  '
                 f'({len(wrong)/len(y_test):.1%} error rate)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.show()
    print(f'Saved: {save_path}')


def plot_per_class_accuracy(y_true, y_pred,
                            save_path='figures/per_class_accuracy.png'):
    """
    Horizontal bar chart of per-class accuracy, sorted ascending.
    A dashed line marks the overall accuracy.
    """
    accs    = [float(np.mean(y_pred[y_true == i] == y_true[y_true == i]))
               for i in range(10)]
    order   = np.argsort(accs)
    names   = [CLASS_NAMES[i] for i in order]
    vals    = [accs[i]        for i in order]
    overall = float(np.mean(y_true == y_pred))

    fig, ax = plt.subplots(figsize=(9, 6), facecolor=_BG)
    ax.set_facecolor(_BG)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)

    bars = ax.barh(names, vals, color=[plt.cm.RdYlGn(v) for v in vals],
                   edgecolor='white', linewidth=0.5, height=0.65)
    for bar, v in zip(bars, vals):
        ax.text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{v:.1%}', va='center', fontsize=10, fontweight='bold', color=_TEXT)

    ax.axvline(overall, color=_BLUE, linewidth=2, linestyle='--',
               label=f'Overall: {overall:.2%}')
    ax.set_xlim(0, 1.08)
    ax.set_xlabel('Accuracy', fontsize=11)
    ax.set_title('Per-Class Accuracy on Test Set', fontsize=14,
                 fontweight='bold', pad=12)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='x', color=_GRID, linewidth=0.8, zorder=0)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=_BG)
    plt.show()
    print(f'Saved: {save_path}')


if __name__ == '__main__':
    from train import train

    _, history = train(hidden_dim1=1024,hidden_dim2=256,lr=0.2,
          weight_decay=0,epochs=30,activation='relu',
          step_size=5,gamma=0.5
    )
    plot_training_curves(history, lr_decay_epochs=[5, 10, 15])

    _, _, _, _, X_test, y_test = load_fashion_mnist()
    model  = _load_model()
    logits = model.forward(X_test)
    exp    = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs  = exp / exp.sum(axis=1, keepdims=True)
    preds  = np.argmax(probs, axis=1)

    plot_confusion_matrix_heatmap(y_test, preds)
    plot_first_layer_weights()
    plot_error_examples()
    plot_per_class_accuracy(y_test, preds)
