# ==============================================
# 心衰AOP - KE2 最终优化版（均衡权重）
# ==============================================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import *
from sklearn.utils.class_weight import compute_class_weight
from lightning import pytorch as pl
from chemprop import data, featurizers, models, nn
import warnings

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
DATA_PATH = "E:/心衰AOP/预处理数据/KE2 去重.csv"
OUTPUT_DIR = Path("E:/心衰AOP/KE2_balanced")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SMILES_COL = "SMILES"
TARGET_COL = "label"

N_FOLDS = 5
RANDOM_SEED = 42
MAX_EPOCHS = 200
BATCH_SIZE = 32
EARLY_STOPPING_PATIENCE = 15
NUM_WORKERS = 0
D_H, DEPTH, DROPOUT = 300, 3, 0.3

print("=" * 60)
print("KE2 均衡版（温和权重 + 学习率调度）")
print("=" * 60)


# ==================== 数据加载 ====================
def load_data():
    from rdkit import Chem
    df = pd.read_csv(DATA_PATH)
    valid = []
    for i, row in df.iterrows():
        smi = str(row[SMILES_COL]).strip()
        if smi and smi != 'nan' and Chem.MolFromSmiles(smi):
            valid.append(i)
    df = df.iloc[valid].reset_index(drop=True)
    pos, neg = sum(df[TARGET_COL] == 1), len(df) - sum(df[TARGET_COL] == 1)
    print(f"数据: {len(df)} 分子, 阳性={pos}, 阴性={neg}")
    return df


def df_to_dps(df):
    return [data.MoleculeDatapoint.from_smi(str(row[SMILES_COL]), [float(row[TARGET_COL])])
            for _, row in df.iterrows()]


def build_model():
    mp = nn.BondMessagePassing(d_h=D_H, depth=DEPTH, dropout=DROPOUT)
    ffn = nn.BinaryClassificationFFN(n_tasks=1, input_dim=D_H, hidden_dim=D_H, n_layers=1, dropout=DROPOUT)
    return models.MPNN(message_passing=mp, agg=nn.MeanAggregation(), predictor=ffn, batch_norm=True)


# ==================== 加权模型包装器 ====================
class WeightedModel(pl.LightningModule):
    def __init__(self, model, class_weight):
        super().__init__()
        self.model = model
        self.class_weight = class_weight
        self.validation_step_outputs = []

    def forward(self, bmg):
        return self.model(bmg)

    def training_step(self, batch, batch_idx):
        y = batch.Y.float().view(-1)
        pred = self(batch.bmg).view(-1)
        w = torch.where(y == 1,
                        torch.tensor(self.class_weight[1], device=y.device),
                        torch.tensor(self.class_weight[0], device=y.device))
        loss = torch.nn.BCEWithLogitsLoss(weight=w)(pred, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y = batch.Y.float().view(-1)
        pred = self(batch.bmg).view(-1)
        loss = torch.nn.BCEWithLogitsLoss()(pred, y)
        self.log('val_loss', loss, prog_bar=True)
        prob = torch.sigmoid(pred)
        self.validation_step_outputs.append({'y_true': y.cpu(), 'y_pred': prob.cpu()})
        return {'y_true': y.cpu(), 'y_pred': prob.cpu()}

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        y_true = torch.cat([x['y_true'] for x in self.validation_step_outputs]).numpy()
        y_pred = torch.cat([x['y_pred'] for x in self.validation_step_outputs]).numpy()
        try:
            auc = roc_auc_score(y_true, y_pred)
            self.log('val_auc', auc, prog_bar=True, on_epoch=True)
        except:
            pass
        self.validation_step_outputs.clear()

    # ✅ 学习率调度
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=8, min_lr=1e-6
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_loss'}
        }


# ==================== 评估函数 ====================
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in loader:
            p = torch.sigmoid(model(batch.bmg)).view(-1)
            y_true.extend(batch.Y.numpy().flatten())
            y_pred.extend(p.numpy())
    y_true, y_pred = np.array(y_true), np.array(y_pred)

    best_t, best_score = 0.5, 0
    for t in np.arange(0.1, 0.9, 0.01):
        yc = (y_pred >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yc).ravel()
        se = tp / (tp + fn) if (tp + fn) else 0
        sp = tn / (tn + fp) if (tn + fp) else 0
        f1 = f1_score(y_true, yc, zero_division=0)
        score = (se + sp - 1) * 0.6 + f1 * 0.4
        if score > best_score:
            best_score, best_t = score, t

    yc = (y_pred >= best_t).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yc).ravel()
    return {
        'AUC': roc_auc_score(y_true, y_pred),
        'Accuracy': accuracy_score(y_true, yc),
        'Balanced_Accuracy': balanced_accuracy_score(y_true, yc),
        'F1': f1_score(y_true, yc, zero_division=0),
        'MCC': matthews_corrcoef(y_true, yc),
        'SE': tp / (tp + fn) if (tp + fn) else 0,
        'SP': tn / (tn + fp) if (tn + fp) else 0,
        'PPV': precision_score(y_true, yc, zero_division=0),
        'NPV': tn / (tn + fn) if (tn + fn) else 0,
        'Threshold': best_t,
        'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn)
    }, y_true, y_pred, yc


def print_metrics(m, title="结果"):
    print(f"\n{'=' * 50}\n{title}\n{'=' * 50}")
    print(f"  AUC                : {m['AUC']:.4f}")
    print(f"  Accuracy           : {m['Accuracy']:.4f}")
    print(f"  Balanced_Accuracy  : {m['Balanced_Accuracy']:.4f}")
    print(f"  F1                 : {m['F1']:.4f}")
    print(f"  MCC                : {m['MCC']:.4f}")
    print(f"  SE (敏感性)        : {m['SE']:.4f}")
    print(f"  SP (特异性)        : {m['SP']:.4f}")
    print(f"  SE-SP 平衡度       : {1 - abs(m['SE'] - m['SP']):.4f}")
    print(f"  PPV                : {m['PPV']:.4f}")
    print(f"  NPV                : {m['NPV']:.4f}")
    print(f"  Threshold          : {m['Threshold']:.3f}")
    print(f"\n  混淆矩阵: TP={m['TP']}, TN={m['TN']}, FP={m['FP']}, FN={m['FN']}")
    print("=" * 50)


# ==================== 绘制函数 ====================
def plot_training_curves(history, save_dir):
    if not history or len(history.get('epoch', [])) == 0:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['epoch'], history['train_loss'], 'b-', linewidth=2, label='Training Loss')
    axes[0].plot(history['epoch'], history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
    axes[0].set_xlabel('Epoch');
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend();
    axes[0].grid(True, alpha=0.3)
    if history.get('val_auc') and len(history['val_auc']) > 0:
        axes[1].plot(history['epoch'], history['val_auc'], 'g-', linewidth=2, label='Validation AUC')
        best_idx = np.argmax(history['val_auc'])
        axes[1].plot(history['epoch'][best_idx], history['val_auc'][best_idx], 'ro', markersize=10)
        axes[1].set_ylim(0, 1)
    axes[1].set_xlabel('Epoch');
    axes[1].set_ylabel('AUC')
    axes[1].set_title('Validation AUC Curve')
    axes[1].legend();
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "Training_Curves.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_pred, auc_score, save_dir, title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {auc_score:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.fill_between(fpr, tpr, alpha=0.15)
    plt.xlim([0.0, 1.0]);
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.title(title);
    plt.legend(loc='lower right');
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "ROC_Curve.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_class, save_dir, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_class)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted Label');
    plt.ylabel('True Label')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_dir / "Confusion_Matrix.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_comprehensive_analysis(level, cv_scores, test_metrics, y_true, y_pred, y_class):
    fig = plt.figure(figsize=(18, 12))
    ax1 = plt.subplot(2, 3, 1)
    folds = range(1, N_FOLDS + 1)
    for metric in ['AUC', 'F1', 'MCC']:
        if metric in cv_scores and cv_scores[metric]:
            ax1.plot(folds, cv_scores[metric], 'o-', label=metric, linewidth=2, markersize=8)
            ax1.axhline(np.mean(cv_scores[metric]), linestyle='--', alpha=0.5)
    ax1.set_xlabel('Fold');
    ax1.set_ylabel('Score')
    ax1.set_title(f'{level} {N_FOLDS}-Fold CV')
    ax1.legend();
    ax1.grid(True, alpha=0.3);
    ax1.set_ylim(0, 1)

    ax2 = plt.subplot(2, 3, 2)
    cm = confusion_matrix(y_true, y_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                xticklabels=['Non-toxic', 'Toxic'],
                yticklabels=['Non-toxic', 'Toxic'])
    ax2.set_title(f'Confusion Matrix (Threshold={test_metrics["Threshold"]:.2f})')
    ax2.set_ylabel('True Label');
    ax2.set_xlabel('Predicted Label')
    ax2.text(0.5, -0.15, f"SE={test_metrics['SE']:.3f}  SP={test_metrics['SP']:.3f}  MCC={test_metrics['MCC']:.3f}",
             transform=ax2.transAxes, ha='center', fontsize=11, fontweight='bold')

    ax3 = plt.subplot(2, 3, 3)
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ax3.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {test_metrics["AUC"]:.4f}')
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax3.fill_between(fpr, tpr, alpha=0.2)
    ax3.set_xlabel('False Positive Rate');
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve');
    ax3.legend(loc='lower right');
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1);
    ax3.set_ylim(0, 1)

    ax4 = plt.subplot(2, 3, 4)
    metric_names = ['Accuracy', 'Balanced_Acc', 'F1', 'MCC']
    metric_values = [test_metrics.get('Accuracy', 0), test_metrics.get('Balanced_Accuracy', 0),
                     test_metrics.get('F1', 0), test_metrics.get('MCC', 0)]
    bars = ax4.bar(metric_names, metric_values, color=['#2ecc71', '#3498db', '#e74c3c', '#9b59b6'])
    ax4.set_ylim(0, 1);
    ax4.set_ylabel('Score');
    ax4.set_title('Performance Metrics')
    ax4.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    for bar, val in zip(bars, metric_values):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom',
                 fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')

    ax5 = plt.subplot(2, 3, 5)
    se_sp_values = [test_metrics.get('SE', 0), test_metrics.get('SP', 0),
                    test_metrics.get('PPV', 0), test_metrics.get('NPV', 0)]
    bars = ax5.bar(['SE', 'SP', 'PPV', 'NPV'], se_sp_values, color=['#e74c3c', '#2ecc71', '#3498db', '#9b59b6'])
    ax5.set_ylim(0, 1);
    ax5.set_ylabel('Score');
    ax5.set_title('SE, SP, PPV, NPV')
    ax5.axhline(y=0.5, color='red', linestyle='--', alpha=0.3)
    for bar, val in zip(bars, se_sp_values):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f'{val:.3f}', ha='center', va='bottom',
                 fontsize=9)
    ax5.grid(True, alpha=0.3, axis='y')

    ax6 = plt.subplot(2, 3, 6)
    bins = np.linspace(0, 1, 21)
    ax6.hist(y_pred[y_true == 0], bins=bins, alpha=0.6, label='Non-toxic', color='blue')
    ax6.hist(y_pred[y_true == 1], bins=bins, alpha=0.6, label='Toxic', color='red')
    ax6.axvline(x=test_metrics["Threshold"], color='black', linestyle='--',
                label=f'Threshold={test_metrics["Threshold"]:.2f}')
    ax6.set_xlabel('Predicted Probability');
    ax6.set_ylabel('Frequency')
    ax6.set_title('Prediction Distribution');
    ax6.legend();
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'{level} - Complete Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_dir / f"{level}_Complete_Analysis.png", dpi=300, bbox_inches='tight')
    plt.close()


# ==================== 5折交叉验证 ====================
def cross_validate(train_df, class_weight):
    print(f"\n{N_FOLDS}折交叉验证...")
    labels = train_df[TARGET_COL].values
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    results = {k: [] for k in ['AUC', 'Accuracy', 'F1', 'MCC', 'SE', 'SP']}

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, labels)):
        print(f"\n  Fold {fold + 1}/{N_FOLDS}")

        train_dps = df_to_dps(train_df.iloc[train_idx])
        val_dps = df_to_dps(train_df.iloc[val_idx])

        train_loader = data.build_dataloader(
            data.MoleculeDataset(train_dps, featurizer),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
        val_loader = data.build_dataloader(
            data.MoleculeDataset(val_dps, featurizer),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

        model = WeightedModel(build_model(), class_weight)

        checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")
        early_stop = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=EARLY_STOPPING_PATIENCE)

        trainer = pl.Trainer(
            max_epochs=MAX_EPOCHS,
            accelerator="cpu",
            devices=1,
            callbacks=[early_stop, checkpoint],
            enable_progress_bar=True,
            log_every_n_steps=1,
            gradient_clip_val=0.5
        )
        trainer.fit(model, train_loader, val_loader)

        best = WeightedModel.load_from_checkpoint(
            checkpoint.best_model_path,
            model=build_model(),
            class_weight=class_weight
        )
        metrics, _, _, _ = evaluate(best, val_loader)
        for k in results:
            results[k].append(metrics[k])
        print(f"    AUC={metrics['AUC']:.4f}, F1={metrics['F1']:.4f}, SE={metrics['SE']:.4f}, SP={metrics['SP']:.4f}")

    print("\n  平均CV结果:")
    for k in results:
        print(f"    {k}: {np.mean(results[k]):.4f} ± {np.std(results[k]):.4f}")

    pd.DataFrame(results).to_csv(OUTPUT_DIR / "KE2_cv_results.csv", index=False)
    return results


# ==================== 主程序 ====================
def main():
    df = load_data()
    labels = df[TARGET_COL].values

    # ✅ 温和权重：平衡 SE 和 SP
    class_weight = {0: 1.5, 1: 0.8}  # 温和版
    print(f"类别权重: {class_weight}")

    train_idx, test_idx = train_test_split(
        range(len(df)), test_size=0.2,
        random_state=RANDOM_SEED, stratify=labels
    )
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    print(f"训练集: {len(train_df)}, 测试集: {len(test_df)}")

    cv_results = cross_validate(train_df, class_weight)

    print("\n训练最终模型...")
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

    all_dps = df_to_dps(train_df)
    train_loader = data.build_dataloader(
        data.MoleculeDataset(all_dps, featurizer),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    val_df = train_df.sample(frac=0.2, random_state=RANDOM_SEED)
    val_dps = df_to_dps(val_df)
    val_loader = data.build_dataloader(
        data.MoleculeDataset(val_dps, featurizer),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = WeightedModel(build_model(), class_weight)
    checkpoint = pl.callbacks.ModelCheckpoint(monitor="val_loss", mode="min")
    early_stop = pl.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=EARLY_STOPPING_PATIENCE)

    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="cpu",
        devices=1,
        callbacks=[early_stop, checkpoint],
        enable_progress_bar=True,
        log_every_n_steps=1,
        gradient_clip_val=0.5
    )
    trainer.fit(model, train_loader, val_loader)

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_auc': []}
    if hasattr(trainer.logger, 'history') and trainer.logger.history:
        log_keys = list(trainer.logger.history.keys())
        if 'train_loss' in log_keys:
            history['train_loss'] = trainer.logger.history['train_loss']
            history['epoch'] = list(range(len(history['train_loss'])))
        if 'val_loss' in log_keys:
            history['val_loss'] = trainer.logger.history['val_loss']
        if 'val_auc' in log_keys:
            history['val_auc'] = trainer.logger.history['val_auc']

    if history['epoch']:
        plot_training_curves(history, OUTPUT_DIR)
        pd.DataFrame(history).to_csv(OUTPUT_DIR / "Training_History.csv", index=False)

    best_model = WeightedModel.load_from_checkpoint(
        checkpoint.best_model_path,
        model=build_model(),
        class_weight=class_weight
    )
    torch.save(best_model.state_dict(), OUTPUT_DIR / "KE2_model.pt")

    test_dps = df_to_dps(test_df)
    test_loader = data.build_dataloader(
        data.MoleculeDataset(test_dps, featurizer),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    test_metrics, y_true, y_pred, y_class = evaluate(best_model, test_loader)
    print_metrics(test_metrics, "KE2 测试集结果")

    test_df['pred_prob'] = y_pred
    test_df['pred_class'] = y_class
    test_df.to_csv(OUTPUT_DIR / "KE2_predictions.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(OUTPUT_DIR / "KE2_test_metrics.csv", index=False)

    print("\n生成可视化图表...")
    plot_roc_curve(y_true, y_pred, test_metrics['AUC'], OUTPUT_DIR, "KE2 ROC Curve")
    plot_confusion_matrix(y_true, y_class, OUTPUT_DIR, "KE2 Confusion Matrix")
    plot_comprehensive_analysis("KE2", cv_results, test_metrics, y_true, y_pred, y_class)

    config = {
        'data_path': DATA_PATH,
        'n_folds': N_FOLDS,
        'max_epochs': MAX_EPOCHS,
        'batch_size': BATCH_SIZE,
        'class_weight': class_weight,
        'model_params': {'d_h': D_H, 'depth': DEPTH, 'dropout': DROPOUT},
        'test_metrics': {k: float(v) for k, v in test_metrics.items() if isinstance(v, (int, float, np.number))}
    }
    with open(OUTPUT_DIR / "KE2_config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n✅ 完成！所有结果保存在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()