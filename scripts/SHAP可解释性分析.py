import matplotlib

matplotlib.use('Agg')  # 必须在import pyplot之前设置
import shap
import pandas as pd
import numpy as np
import joblib  # 用于加载pkl文件
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings('ignore')

# 创建输出目录
output_dir = 'E:/心衰AOP/shap可解释性分析'
os.makedirs(output_dir, exist_ok=True)

# ============ 1. 加载模型（使用joblib加载pkl文件） ============
print("Loading Random Forest model...")
model_path = 'E:/心衰AOP/RF/AO_rf_model.pkl'
model = joblib.load(model_path)
print(f"Model loaded successfully! Model type: {type(model).__name__}")

# ============ 2. 加载数据 ============
print("\nLoading data...")
expl_data_raw = pd.read_csv(r'E:/心衰AOP/分子指纹/MIE/AO_X_test_maccs.csv', header=0)
background_data_raw = pd.read_csv(r'E:/心衰AOP/分子指纹/MIE/AO_X_train_maccs.csv', header=0)

print(f"Raw explain data shape: {expl_data_raw.shape}")
print(f"Raw background data shape: {background_data_raw.shape}")

# ============ 3. 数据预处理 ============
print("\nPreprocessing data...")

# 检查列名，智能移除不需要的列
print(f"First few columns: {expl_data_raw.columns[:5].tolist()}")

# 移除第一列（通常是SMILES或ID）
expl_data = expl_data_raw.iloc[:, 1:]
background_data = background_data_raw.iloc[:, 1:]

# 检查并移除任何包含'label'或'class'的列（通常在最后一列）
label_cols = [col for col in expl_data.columns if
              'label' in col.lower() or 'class' in col.lower() or 'activity' in col.lower()]
if label_cols:
    print(f"Removing label columns: {label_cols}")
    expl_data = expl_data.drop(columns=label_cols)
    background_data = background_data.drop(columns=label_cols)

# 如果数据形状和之前不同，可能是因为SMILES列被识别为第一列
print(f"After removing non-feature columns - Explain: {expl_data.shape}")
print(f"After removing non-feature columns - Background: {background_data.shape}")

# 处理数据类型
expl_data = expl_data.apply(pd.to_numeric, errors='coerce').fillna(0)
background_data = background_data.apply(pd.to_numeric, errors='coerce').fillna(0)

print(f"Final data shape - Explain: {expl_data.shape}, Background: {background_data.shape}")

# 特征名称
feature_names = expl_data.columns.tolist()
print(f"Number of features: {len(feature_names)}")

# 检查是否有零方差特征
zero_var_cols = []
for col in expl_data.columns:
    if expl_data[col].std() == 0:
        zero_var_cols.append(col)

if zero_var_cols:
    print(f"⚠️ Warning: {len(zero_var_cols)} features have zero variance (all same value)")
    if len(zero_var_cols) > 5:
        print(f"Examples: {zero_var_cols[:5]}")

# 采样（减少样本数加快计算）
background_sample = background_data.sample(n=min(100, len(background_data)), random_state=42)
sample_data = expl_data.sample(n=min(100, len(expl_data)), random_state=42)

print(f"Sampled shape - Background: {background_sample.shape}, Explain: {sample_data.shape}")

# 检查数据值分布
unique_vals = np.unique(sample_data.values)
print(f"Data value range: {sample_data.min().min()} to {sample_data.max().max()}")
print(f"Unique values count: {len(unique_vals)}")
if len(unique_vals) <= 10:
    print(f"Unique values: {unique_vals}")

# ============ 4. 创建SHAP解释器并计算 ============
print("\nCalculating SHAP values (this may take a few minutes)...")

# 对于随机森林，使用TreeExplainer（更快更精确）
try:
    print("Trying TreeExplainer (optimized for tree models)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample_data.values[:50])

    # TreeExplainer返回的格式处理
    if isinstance(shap_values, list):
        # 如果是二分类，取正类（索引1）
        if len(shap_values) == 2:
            shap_values_2d = shap_values[1]
        else:
            shap_values_2d = shap_values[0]
    else:
        shap_values_2d = shap_values

    # 处理3D数组
    if shap_values_2d.ndim == 3:
        if shap_values_2d.shape[-1] == 1:
            shap_values_2d = shap_values_2d.squeeze(-1)
        else:
            shap_values_2d = shap_values_2d[:, :, 0]

    method_used = "TreeExplainer"
    print(f"✅ TreeExplainer succeeded! Shape: {shap_values_2d.shape}")

except Exception as e:
    print(f"TreeExplainer failed: {str(e)[:100]}...")

    # 回退到KernelExplainer
    try:
        print("Trying KernelExplainer (fallback)...")


        def predict_fn(x):
            return model.predict_proba(x)[:, 1]


        background_small = background_sample.values[:30]
        explainer = shap.KernelExplainer(predict_fn, background_small)
        shap_values = explainer.shap_values(sample_data.values[:50], nsamples=100)

        # 处理KernelExplainer的返回格式
        if isinstance(shap_values, list):
            if len(shap_values) == 2:
                shap_values_2d = shap_values[1]
            else:
                shap_values_2d = shap_values[0]
        else:
            shap_values_2d = shap_values

        # 处理3D数组
        if shap_values_2d.ndim == 3:
            if shap_values_2d.shape[-1] == 1:
                shap_values_2d = shap_values_2d.squeeze(-1)
            else:
                shap_values_2d = shap_values_2d[:, :, 0]

        method_used = "KernelExplainer"
        print(f"✅ KernelExplainer succeeded! Shape: {shap_values_2d.shape}")

    except Exception as e2:
        print(f"All methods failed: {e2}")
        exit(1)

print(f"Final SHAP shape: {shap_values_2d.shape}")
print(f"Method used: {method_used}")

# 获取expected value
if hasattr(explainer, 'expected_value'):
    expected_val = explainer.expected_value
    if isinstance(expected_val, list):
        expected_val = expected_val[1] if len(expected_val) > 1 else expected_val[0]
    print(f"Expected value: {expected_val}")
else:
    expected_val = 0

# 设置美观的英文字体
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14

# ============ 5. 条形图（蓝色 #056eee） ============
print("\nGenerating bar plot...")
try:
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values_2d, sample_data.values[:shap_values_2d.shape[0]],
                      feature_names=feature_names[:shap_values_2d.shape[1]],
                      color="#056eee",
                      plot_type="bar",
                      show=False,
                      max_display=20)
    plt.gca().set_title('')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/AO_rf_maccs_shap_bar_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/AO_rf_maccs_shap_bar_plot.png")
except Exception as e:
    print(f"⚠️ Could not generate bar plot: {e}")

# ============ 6. 散点图/蜂窝图（默认红蓝） ============
print("Generating dot plot...")
try:
    plt.figure(figsize=(14, 10))
    shap.summary_plot(shap_values_2d, sample_data.values[:shap_values_2d.shape[0]],
                      feature_names=feature_names[:shap_values_2d.shape[1]],
                      plot_type="dot",
                      show=False,
                      max_display=20)
    plt.gca().set_title('')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/AO_rf_maccs_shap_dot_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/AO_rf_maccs_shap_dot_plot.png")
except Exception as e:
    print(f"⚠️ Could not generate dot plot: {e}")

# ============ 7. 保存SHAP值 ============
print("Saving SHAP values...")
shap_df = pd.DataFrame(shap_values_2d, columns=feature_names[:shap_values_2d.shape[1]])
shap_df.to_csv(f'{output_dir}/AO_rf_maccs_shap_values.csv', index=False)
print(f"✅ Saved: {output_dir}/AO_rf_maccs_shap_values.csv")

# ============ 8. 计算并保存特征重要性 ============
mean_shap = np.abs(shap_values_2d).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': feature_names[:len(mean_shap)],
    'mean_abs_shap': mean_shap,
}).sort_values('mean_abs_shap', ascending=False)
importance_df['rank'] = range(1, len(importance_df) + 1)
importance_df.to_csv(f'{output_dir}/AO_rf_maccs_feature_importance.csv', index=False)
print(f"✅ Saved: {output_dir}/AO_rf_maccs_feature_importance.csv")

# ============ 9. Top 20特征重要性条形图 ============
print("Generating feature importance bar chart (Top 20)...")
top_n = min(20, len(importance_df))
top_features = importance_df.head(top_n)

plt.figure(figsize=(12, 10))
colors = ['#056eee'] * top_n
bars = plt.barh(range(top_n), top_features['mean_abs_shap'].values, color=colors, edgecolor='black', alpha=0.85)
plt.yticks(range(top_n), top_features['feature'].values, fontsize=10)
plt.gca().set_title('')
plt.xlabel('Mean Absolute SHAP Value', fontsize=14)
plt.ylabel('Features', fontsize=14)
plt.gca().invert_yaxis()
plt.grid(axis='x', linestyle='--', alpha=0.3)

# 添加数值标签
for i, (bar, value) in enumerate(zip(bars, top_features['mean_abs_shap'].values)):
    plt.text(value + 0.0005, bar.get_y() + bar.get_height() / 2,
             f'{value:.4f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(f'{output_dir}/AO_rf_maccs_top20_features_bar.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {output_dir}/AO_rf_maccs_top20_features_bar.png")

# ============ 10. 蜂群图 ============
print("Generating beeswarm plot...")
try:
    # 创建Explanation对象
    explanation = shap.Explanation(
        values=shap_values_2d,
        base_values=expected_val,
        data=sample_data.values[:shap_values_2d.shape[0]],
        feature_names=feature_names[:shap_values_2d.shape[1]]
    )

    plt.figure(figsize=(14, 10))
    shap.plots.beeswarm(explanation, max_display=20, show=False)
    plt.gca().set_title('')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/AO_rf_maccs_shap_beeswarm_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {output_dir}/AO_rf_maccs_shap_beeswarm_plot.png")
except Exception as e:
    print(f"⚠️ Could not generate beeswarm plot: {e}")
    # 备用方案
    try:
        print("Using summary_plot as fallback for beeswarm...")
        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values_2d, sample_data.values[:shap_values_2d.shape[0]],
                          feature_names=feature_names[:shap_values_2d.shape[1]],
                          plot_type="dot",
                          show=False,
                          max_display=20)
        plt.gca().set_title('')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/AO_rf_maccs_shap_beeswarm_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Saved (fallback): {output_dir}/AO_rf_maccs_shap_beeswarm_plot.png")
    except Exception as e2:
        print(f"⚠️ Fallback also failed: {e2}")

# ============ 11. 输出结果 ============
print(f"\n{'=' * 60}")
print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
print(f"📁 Results saved to: {output_dir}")
print(f"🔧 SHAP Method: {method_used}")
print(f"📊 Samples analyzed: {shap_values_2d.shape[0]}")
print(f"📊 Features analyzed: {shap_values_2d.shape[1]}")
print(f"\n📁 Generated files:")
print(f"  1. hb_rf_maccs_shap_bar_plot.png         - SHAP bar plot")
print(f"  2. hb_rf_maccs_shap_dot_plot.png        - SHAP summary plot")
print(f"  3. hb_rf_maccs_shap_beeswarm_plot.png   - SHAP beeswarm plot")
print(f"  4. hb_rf_maccs_top20_features_bar.png   - Top 20 features bar chart")
print(f"  5. hb_rf_maccs_shap_values.csv          - All SHAP values")
print(f"  6. hb_rf_maccs_feature_importance.csv   - Feature importance ranking")
print(f"\n📊 Top {top_n} most important features:")
for idx, row in top_features.iterrows():
    print(f"  {row['rank']:2d}. {row['feature'][:25]:<25} : {row['mean_abs_shap']:.6f}")
print('=' * 60)