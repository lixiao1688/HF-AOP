"""
第二层堆叠模型 - 随机森林版
功能：
    1. 使用随机森林作为元模型（避免负系数）
    2. 5折交叉验证 + 留一法验证
    3. 保存元模型
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
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
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 配置
# ============================================================================

INPUT_FILE = r"E:/心衰AOP/第一层集成/tier2_input.csv"
OUTPUT_DIR = r"E:/心衰AOP/第二层堆叠"
META_MODEL_SAVE_PATH = r"E:/心衰AOP/第二层堆叠/meta_model_rf.pkl"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 随机森林参数
RF_CONFIG = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'random_state': 42,
    'class_weight': 'balanced'
}

# ============================================================================
# 加载数据
# ============================================================================

print("=" * 70)
print("第二层堆叠模型 - 随机森林版")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)
print(f"\n[1] 加载数据")
print(f"    样本总数: {len(df)}")

feature_cols = ['P_MIE', 'P_KE1', 'P_KE2', 'P_AO']
X = df[feature_cols].values
y = df['True_Label'].values

print(f"    阳性: {sum(y)} 个, 阴性: {len(y)-sum(y)} 个")
print(f"    特征数: {X.shape[1]}")

# ============================================================================
# 5折交叉验证
# ============================================================================

print(f"\n[2] 5折交叉验证...")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    model = RandomForestClassifier(**RF_CONFIG)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_val)[:, 1]
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

# ============================================================================
# 留一法验证
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

    model = RandomForestClassifier(**RF_CONFIG)
    model.fit(X_train, y_train)

    prob = model.predict_proba(X_test)[:, 1][0]
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
# 特征重要性（全是正数！）
# ============================================================================

print(f"\n[4] 特征重要性分析")

final_model = RandomForestClassifier(**RF_CONFIG)
final_model.fit(X, y)

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': final_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n  特征重要性（所有特征均为正数）:")
for _, row in importance_df.iterrows():
    print(f"    {row['Feature']}: {row['Importance']:.4f}")

# 可视化
plt.figure(figsize=(8, 5))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='steelblue')
plt.xlabel('重要性')
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rf_feature_importance.png'), dpi=300)
plt.close()
print(f"    ✓ 特征重要性图已保存")

# ============================================================================
# ★★★ 用全部数据训练并保存元模型 ★★★
# ============================================================================

print(f"\n[5] 用全部数据训练并保存元模型...")

joblib.dump(final_model, META_MODEL_SAVE_PATH)
print(f"  ✓ 元模型已保存: {META_MODEL_SAVE_PATH}")
print(f"  模型类型: 随机森林")
print(f"  参数: {RF_CONFIG}")

# ============================================================================
# 保存预测结果
# ============================================================================

print(f"\n[6] 保存预测结果")

result_df = df.copy()
result_df['LOO_Prediction'] = y_pred
result_df['LOO_Probability'] = y_prob
result_df.to_csv(os.path.join(OUTPUT_DIR, 'loo_predictions_rf.csv'), index=False)
print(f"    ✓ 预测结果已保存: loo_predictions_rf.csv")

# 保存特征重要性
importance_df.to_csv(os.path.join(OUTPUT_DIR, 'feature_importance_rf.csv'), index=False)
print(f"    ✓ 特征重要性已保存: feature_importance_rf.csv")

# ============================================================================
# 最终报告
# ============================================================================

print("\n" + "=" * 70)
print("第二层堆叠模型完成! (随机森林版)")
print("=" * 70)

print(f"\n{'─'*60}")
print(f"最终结论")
print(f"{'─'*60}")
print(f"  ✅ LOO AUC = {auc_loo:.4f}")
print(f"  ✅ 所有特征重要性均为正数，无需解释负系数")
print(f"  错误样本数: {len(error_indices)} / {len(df)} ({len(error_indices)/len(df)*100:.2f}%)")

print(f"\n📁 输出文件:")
print(f"  ★ 元模型: {META_MODEL_SAVE_PATH}")
print(f"  特征重要性: feature_importance_rf.csv")

print("\n" + "=" * 70)
print("运行完成!")
print("=" * 70)