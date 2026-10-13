"""
外部验证 - XGBoost（无交互，4特征版）
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix

# ============================================================================
# ConsensusModel类（保持不变）
# ============================================================================

class ConsensusModel:
    def __init__(self, sub_models=None, model_names=None, level_name=None,
                 fingerprint=None, performance=None):
        self.sub_models = sub_models or []
        self.model_names = model_names or []
        self.level_name = level_name
        self.fingerprint = fingerprint
        self.performance = performance
        self.n_models = len(self.sub_models)

    def predict_proba(self, X):
        if self.n_models == 0:
            raise ValueError("没有子模型")
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        all_probs = []
        for model in self.sub_models:
            try:
                prob = model.predict_proba(X)[:, 1]
            except (AttributeError, TypeError):
                prob = model.predict(X)
                if hasattr(model, 'decision_function'):
                    prob = 1 / (1 + np.exp(-model.decision_function(X)))
                else:
                    prob = prob.astype(float)
            prob = np.array(prob).flatten()
            all_probs.append(prob)
        all_probs = np.column_stack(all_probs)
        return np.mean(all_probs, axis=1)


# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = r"E:/心衰AOP/外部验证结果"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = r"E:/心衰AOP/第一层模型"
META_MODEL_PATH = r"E:/心衰AOP/第二层堆叠/meta_model_xgb.pkl"

EXTERNAL_DATA_DIR = r"E:/心衰AOP/外部验证数据"
PUBFINGER_FILE = os.path.join(EXTERNAL_DATA_DIR, "fingerprints/external_pubchem_fingerprints.csv")
CHEMBERTA_FILE = os.path.join(EXTERNAL_DATA_DIR, "chem fingerprints/external_chemberta_fingerprints.csv")
CLEANED_LABEL_FILE = os.path.join(EXTERNAL_DATA_DIR, "external_validation_cleaned.csv")


print("=" * 70)
print("外部验证（XGBoost - 4特征版）")
print("=" * 70)

# 1. 加载数据
label_df = pd.read_csv(CLEANED_LABEL_FILE)
if 'True_Label' in label_df.columns:
    label_col = 'True_Label'
elif 'label' in label_df.columns:
    label_col = 'label'
else:
    print("❌ 未找到标签列")
    exit()

print(f"\n[1] 加载数据: {len(label_df)} 个 (阳性: {label_df[label_col].sum()})")

pubchem_df = pd.read_csv(PUBFINGER_FILE)
chemberta_df = pd.read_csv(CHEMBERTA_FILE)

pubchem_cols = [c for c in pubchem_df.columns if c.startswith('PubChem_')]
chemberta_cols = [c for c in chemberta_df.columns if c.startswith('ChemBERTa_')]

X_pubchem = pubchem_df[pubchem_cols].values.astype(np.float32)
X_chemberta = chemberta_df[chemberta_cols].values.astype(np.float32)

if 'label' in pubchem_df.columns:
    y_true = pubchem_df['label'].values
else:
    smile_to_label = dict(zip(label_df['SMILES'].astype(str).str.strip(), label_df[label_col]))
    pubchem_smiles = pubchem_df['SMILES'].astype(str).str.strip()
    y_true = np.array([smile_to_label.get(s, 0) for s in pubchem_smiles])

min_len = min(len(X_pubchem), len(X_chemberta), len(y_true))
X_pubchem = X_pubchem[:min_len]
X_chemberta = X_chemberta[:min_len]
y_true = y_true[:min_len]

print(f"  ✅ 最终验证集: {min_len} 个 (阳性: {sum(y_true)})")

# 2. 加载模型
print("\n[2] 加载模型...")
mie_model = joblib.load(os.path.join(MODEL_DIR, "MIE_consensus_model.pkl"))
ke1_model = joblib.load(os.path.join(MODEL_DIR, "KE1_consensus_model.pkl"))
ke2_model = joblib.load(os.path.join(MODEL_DIR, "KE2_consensus_model.pkl"))
ao_model = joblib.load(os.path.join(MODEL_DIR, "AO_consensus_model.pkl"))
meta_model = joblib.load(META_MODEL_PATH)
print("  ✅ 所有模型加载完成")

# 3. 第一层预测
print("\n[3] 第一层预测...")
P_MIE = mie_model.predict_proba(X_pubchem)
P_KE1 = ke1_model.predict_proba(X_pubchem)
P_KE2 = ke2_model.predict_proba(X_pubchem)
P_AO = ao_model.predict_proba(X_chemberta)

# 4. ★★★ 生成特征：仅4个基础概率（与训练时一致）★★★
print("\n[4] 生成特征...")
X_meta = np.column_stack([P_MIE, P_KE1, P_KE2, P_AO])
print(f"  ✅ 特征数: {X_meta.shape[1]} (P_MIE, P_KE1, P_KE2, P_AO)")

# 5. 第二层预测
print("\n[5] 第二层预测...")
y_prob = meta_model.predict_proba(X_meta)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)

# 6. 评估
auc = roc_auc_score(y_true, y_prob)
acc = accuracy_score(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

print(f"\n{'─'*60}")
print(f"外部验证结果 (4特征版)")
print(f"{'─'*60}")
print(f"  AUC:       {auc:.4f}")
print(f"  Accuracy:  {acc:.4f}")
print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")

# 7. 保存结果
result_df = pd.DataFrame({
    'P_MIE': P_MIE,
    'P_KE1': P_KE1,
    'P_KE2': P_KE2,
    'P_AO': P_AO,
    'True_Label': y_true,
    'Predicted_Label': y_pred,
    'Predicted_Probability': y_prob
})
result_df.to_csv(os.path.join(OUTPUT_DIR, 'external_predictions_4feat.csv'), index=False)
print(f"\n✅ 结果已保存: external_predictions_4feat.csv")

print("=" * 70)