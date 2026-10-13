"""
================================================================================
第二层堆叠模型 - XGBoost版
================================================================================
功能：
    1. 使用XGBoost作为元模型（替代逻辑回归/随机森林）
    2. 5折交叉验证评估
    3. 留一法（LOO）验证
    4. 用全部数据训练并保存元模型（meta_model_xgb.pkl）
    5. 生成可视化报告

输入：E:/心衰AOP/第一层集成/tier2_input.csv
输出：E:/心衰AOP/第二层堆叠/
      E:/心衰AOP/第二层堆叠/meta_model_xgb.pkl
================================================================================
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    precision_score,
    recall_score,
    brier_score_loss,
    roc_curve
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

INPUT_FILE = r"E:/心衰AOP/第一层集成/tier2_input.csv"
OUTPUT_DIR = r"E:/心衰AOP/第二层堆叠"
META_MODEL_SAVE_PATH = r"E:/心衰AOP/第二层堆叠/meta_model_xgb.pkl"

# XGBoost参数
XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': 42,
    'eval_metric': 'logloss'
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# 1. 加载数据
# ============================================================================

print("=" * 70)
print("第二层堆叠模型 - XGBoost版")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)
print(f"\n[1] 加载数据")
print(f"    样本总数: {len(df)}")

# 准备特征和标签
feature_cols = ['P_MIE', 'P_KE1', 'P_KE2', 'P_AO']
X = df[feature_cols].values
y = df['True_Label'].values

print(f"    阳性: {sum(y)} 个, 阴性: {len(y)-sum(y)} 个")
print(f"    特征数: {X.shape[1]}")

# ============================================================================
# 2. 5折交叉验证
# ============================================================================

print(f"\n[2] 5折交叉验证...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

# 计算类别权重（处理不平衡）
scale_pos_weight = len(y[y==0]) / len(y[y==1])
print(f"    类别权重: {scale_pos_weight:.2f}")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # 训练XGBoost
    xgb = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight
    )
    xgb.fit(X_train, y_train)

    prob = xgb.predict_proba(X_val)[:, 1]
    pred = (prob >= 0.5).astype(int)

    auc = roc_auc_score(y_val, prob)
    acc = accuracy_score(y_val, pred)
    f1 = f1_score(y_val, pred)
    mcc = matthews_corrcoef(y_val, pred)

    cv_results.append({'fold': fold+1, 'auc': auc, 'acc': acc, 'f1': f1, 'mcc': mcc})
    print(f"    Fold {fold+1}: AUC={auc:.4f}, Acc={acc:.4f}, F1={f1:.4f}, MCC={mcc:.4f}")

print(f"\n  平均性能:")
print(f"    AUC: {np.mean([r['auc'] for r in cv_results]):.4f} ± {np.std([r['auc'] for r in cv_results]):.4f}")
print(f"    Acc: {np.mean([r['acc'] for r in cv_results]):.4f} ± {np.std([r['acc'] for r in cv_results]):.4f}")
print(f"    F1:  {np.mean([r['f1'] for r in cv_results]):.4f} ± {np.std([r['f1'] for r in cv_results]):.4f}")
print(f"    MCC: {np.mean([r['mcc'] for r in cv_results]):.4f} ± {np.std([r['mcc'] for r in cv_results]):.4f}")

# ============================================================================
# 3. 留一法（LOO）验证
# ============================================================================

print(f"\n[3] 留一法（LOO）验证...")

loo = LeaveOneOut()
y_true_all = []
y_pred_all = []
y_prob_all = []
error_indices = []

fold = 0
for train_idx, test_idx in loo.split(X):
    fold += 1
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    xgb = XGBClassifier(
        **XGB_PARAMS,
        scale_pos_weight=scale_pos_weight
    )
    xgb.fit(X_train, y_train)

    prob = xgb.predict_proba(X_test)[:, 1][0]
    pred = int(prob >= 0.5)

    y_true_all.append(y_test[0])
    y_pred_all.append(pred)
    y_prob_all.append(prob)

    if pred != y_test[0]:
        error_indices.append(test_idx[0])

    if fold % 20 == 0:
        print(f"    已完成: {fold}/{len(df)} 次迭代")

print(f"    ✓ 留一法验证完成！")

y_true = np.array(y_true_all)
y_pred = np.array(y_pred_all)
y_prob = np.array(y_prob_all)

auc_loo = roc_auc_score(y_true, y_prob)
acc_loo = accuracy_score(y_true, y_pred)
f1_loo = f1_score(y_true, y_pred)
mcc_loo = matthews_corrcoef(y_true, y_pred)
precision_loo = precision_score(y_true, y_pred)
recall_loo = recall_score(y_true, y_pred)
brier_loo = brier_score_loss(y_true, y_prob)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print(f"\n{'─'*60}")
print(f"留一法（LOO）性能")
print(f"{'─'*60}")
print(f"  AUC:                 {auc_loo:.4f}")
print(f"  Accuracy:            {acc_loo:.4f}")
print(f"  F1:                  {f1_loo:.4f}")
print(f"  MCC:                 {mcc_loo:.4f}")
print(f"  Precision (精确率):   {precision_loo:.4f}")
print(f"  Recall (召回率):      {recall_loo:.4f}")
print(f"  Brier Score:         {brier_loo:.4f}")
print(f"\n  混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# ============================================================================
# 4. 错误样本分析
# ============================================================================

print(f"\n[4] 错误样本分析")
print(f"    总样本数: {len(df)}")
print(f"    预测错误数: {len(error_indices)}")
print(f"    错误率: {len(error_indices)/len(df)*100:.2f}%")

if len(error_indices) > 0:
    error_df = df.iloc[error_indices].copy()
    error_df['True_Label'] = y_true[error_indices]
    error_df['Predicted'] = y_pred[error_indices]
    error_df['Probability'] = y_prob[error_indices]
    error_df.to_csv(os.path.join(OUTPUT_DIR, 'loo_error_samples_xgb.csv'), index=False)
    print(f"    ✓ 错误样本已保存: loo_error_samples_xgb.csv")

# ============================================================================
# 5. ★★★ 用全部数据训练并保存XGBoost元模型 ★★★
# ============================================================================

print(f"\n[5] 用全部数据训练并保存元模型...")

final_model = XGBClassifier(
    **XGB_PARAMS,
    scale_pos_weight=scale_pos_weight
)
final_model.fit(X, y)

joblib.dump(final_model, META_MODEL_SAVE_PATH)
print(f"  ✓ 元模型已保存: {META_MODEL_SAVE_PATH}")

# 查看特征重要性
print(f"\n  特征重要性:")
importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)
for _, row in importance_df.iterrows():
    print(f"    {row['Feature']}: {row['Importance']:.4f}")

# ============================================================================
# 6. 可视化
# ============================================================================

print(f"\n[6] 生成可视化图表")

# 6.1 ROC曲线
plt.figure(figsize=(10, 8))
fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'LOO ROC (AUC = {auc_loo:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random (AUC = 0.5)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('留一法（LOO）ROC曲线 (XGBoost)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'loo_roc_curve_xgb.png'), dpi=300)
plt.close()
print(f"    ✓ ROC曲线已保存")

# 6.2 混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测阴性', '预测阳性'],
            yticklabels=['实际阴性', '实际阳性'])
plt.title('留一法（LOO）混淆矩阵 (XGBoost)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'loo_confusion_matrix_xgb.png'), dpi=300)
plt.close()
print(f"    ✓ 混淆矩阵已保存")

# 6.3 特征重要性
plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('重要性')
plt.title('XGBoost特征重要性')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance_xgb.png'), dpi=300)
plt.close()
print(f"    ✓ 特征重要性图已保存")

# ============================================================================
# 7. 保存预测结果
# ============================================================================

print(f"\n[7] 保存预测结果")

result_df = df.copy()
result_df['LOO_Prediction'] = y_pred
result_df['LOO_Probability'] = y_prob
result_df.to_csv(os.path.join(OUTPUT_DIR, 'loo_predictions_xgb.csv'), index=False)
print(f"    ✓ 预测结果已保存: loo_predictions_xgb.csv")

# ============================================================================
# 8. 最终报告
# ============================================================================

print("\n" + "=" * 70)
print("第二层堆叠模型完成! (XGBoost版)")
print("=" * 70)

print(f"\n{'─'*60}")
print(f"最终结论")
print(f"{'─'*60}")
print(f"  ✅ LOO AUC = {auc_loo:.4f}")
print(f"  错误样本数: {len(error_indices)} / {len(df)} ({len(error_indices)/len(df)*100:.2f}%)")

print(f"\n  特征重要性:")
print(importance_df.to_string(index=False))

print(f"\n📁 输出文件:")
print(f"  结果目录: {OUTPUT_DIR}")
print(f"  ★ 元模型: {META_MODEL_SAVE_PATH}")

print("\n" + "=" * 70)
print("运行完成!")
print("=" * 70)