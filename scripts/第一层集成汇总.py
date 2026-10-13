"""
第一层共识模型构建代码 - 方案B（保存完整共识模型对象）
功能：
    1. 对每个AOP层级，对入选的单一模型进行简单平均集成
    2. 生成共识概率（.csv）
    3. ★★★ 保存完整的共识模型对象（包含子模型）★★★
    4. 生成第二层输入文件

集成策略：简单平均（Simple Averaging）- 无需权重，无参数
外部验证时：
    model = joblib.load("MIE_consensus_model.pkl")
    prob = model.predict_proba(fingerprint_array)  # 直接预测！
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# ==================== 第一步：配置区域（统一指纹策略） ======================
# ============================================================================

OUTPUT_DIR = r"E:/心衰AOP/第一层集成"
MODEL_OUTPUT_DIR = r"E:/心衰AOP/第一层模型"

# ---------- MIE 层级（统一 PubChem 指纹）----------
MIE_CONFIG = {
    'level_name': 'MIE',
    'fingerprint': 'PubChem',
    'test_file': r"E:/心衰AOP/分子指纹/chemberta/MIE_X_test_chemberta.csv",
    'label_file': r"E:/心衰AOP/分子指纹/MIE/MIE_y_test.csv",
    'label_col': 'True_Label',
    'id_col': None,
    'models': {
        'SVM_PubChem': r"E:/心衰AOP/SVM_Results pubchem/MIE pubchem_y_pred_proba.npy",
        'RF_PubChem': r"E:/心衰AOP/RF pubchem/y_pred_model_MIE_rf.npy",
        'DNN_PubChem': r"E:/心衰AOP/DNN pubchem/MIE_y_pred_prob.npy",
    },
    'selected_models': ['SVM_PubChem', 'RF_PubChem', 'DNN_PubChem'],
    'output_prob_col': 'P_MIE',
    'sub_model_pkls': {
        'SVM_PubChem': r"E:/心衰AOP/SVM_Results pubchem/MIE pubchem_svm_model.pkl",
        'RF_PubChem': r"E:/心衰AOP/RF pubchem/MIE_rf_model.pkl",
        'DNN_PubChem': r"E:/心衰AOP/DNN pubchem/DNN MIE.h5",
    }
}

# ---------- KE1 层级（统一 PubChem 指纹）----------
KE1_CONFIG = {
    'level_name': 'KE1',
    'fingerprint': 'PubChem',
    'test_file': r"E:/心衰AOP/分子指纹/MIE/KE1_X_test_pubchem.csv",
    'label_file': r"E:/心衰AOP/分子指纹/MIE/KE1_y_test.csv",
    'label_col': 'True_Label',
    'id_col': None,
    'models': {
        'RF_PubChem': r"E:/心衰AOP/RF pubchem/y_pred_model_KE1_rf.npy",
        'SVM_PubChem': r"E:/心衰AOP/SVM_Results pubchem/KE1 pubchem_y_pred_proba_sp.npy",
        'DNN_PubChem': r"E:/心衰AOP/DNN pubchem/KE1_y_pred_prob.npy",
    },
    'selected_models': ['RF_PubChem', 'SVM_PubChem', 'DNN_PubChem'],
    'output_prob_col': 'P_KE1',
    'sub_model_pkls': {
        'RF_PubChem': r"E:/心衰AOP/RF pubchem/KE1_rf_model.pkl",
        'SVM_PubChem': r"E:/心衰AOP/SVM_Results pubchem/KE1 pubchem_svm_model_sp.pkl",
        'DNN_PubChem': r"E:/心衰AOP/DNN pubchem/DNN_balanced.h5",
    }
}

# ---------- KE2 层级（统一 PubChem 指纹）----------
KE2_CONFIG = {
    'level_name': 'KE2',
    'fingerprint': 'PubChem',
    'test_file': r"E:/心衰AOP/分子指纹/MIE/KE2_X_test_pubchem.csv",
    'label_file': r"E:/心衰AOP/分子指纹/MIE/KE2_y_test.csv",
    'label_col': 'True_Label',
    'id_col': None,
    'models': {
        'SVM_PubChem': r"E:/心衰AOP/SVM_Results pubchem/KE2 pubchem_y_pred_proba_sp.npy",
        'RF_PubChem': r"E:/心衰AOP/RF pubchem/y_pred_model_KE2_rf.npy",
        'DNN_PubChem': r"E:/心衰AOP/DNN pubchem/KE2_y_pred_prob.npy",
    },
    'selected_models': ['SVM_PubChem', 'RF_PubChem', 'DNN_PubChem'],
    'output_prob_col': 'P_KE2',
    'sub_model_pkls': {
        'SVM_PubChem': r"E:/心衰AOP/SVM_Results pubchem/KE2 pubchem_svm_model_sp.pkl",
        'RF_PubChem': r"E:/心衰AOP/RF pubchem/KE2_rf_model.pkl",
        'DNN_PubChem': r"E:/心衰AOP/DNN pubchem/DNN KE2.h5",
    }
}

# ---------- AO 层级（统一 Chembreta 指纹）----------
AO_CONFIG = {
    'level_name': 'AO',
    'fingerprint': 'Chembreta',
    'test_file': r"E:/心衰AOP/分子指纹/chemberta/AO_X_test_chemberta.csv",
    'label_file': r"E:/心衰AOP/分子指纹/MIE/AO_y_test.csv",
    'label_col': 'True_Label',
    'id_col': None,
    'models': {
        'DNN_Chembreta': r"E:/心衰AOP/DNN chem/AO_y_pred_prob.npy",
        'SVM_Chembreta': r"E:/心衰AOP/SVM_Results CHEM/AO chem_y_pred_proba.npy",
    },
    'selected_models': ['DNN_Chembreta', 'SVM_Chembreta'],
    'output_prob_col': 'P_AO',
    'sub_model_pkls': {
        'DNN_Chembreta': r"E:/心衰AOP/DNN chem/models/DNN_optimized.h5",
        'SVM_Chembreta': r"E:/心衰AOP/SVM_Results CHEM/AO chem_svm_model.pkl",
    }
}

ALL_CONFIGS = [MIE_CONFIG, KE1_CONFIG, KE2_CONFIG, AO_CONFIG]


# ============================================================================
# ==================== 第二步：共识模型类 ====================================
# ============================================================================

class ConsensusModel:
    """
    ★★★ 完整的共识模型类 ★★★
    采用简单平均策略（无需权重）

    外部验证时：model = joblib.load("MIE_consensus_model.pkl")
                prob = model.predict_proba(fingerprint_array)
    """

    def __init__(self, sub_models, model_names=None, level_name=None,
                 fingerprint=None, performance=None):
        self.sub_models = sub_models
        self.model_names = model_names
        self.level_name = level_name
        self.fingerprint = fingerprint
        self.performance = performance
        self.n_models = len(sub_models)

    def predict_proba(self, X):
        """
        简单平均预测概率 (输入: 分子指纹数组)
        所有子模型权重相等
        """
        if self.n_models == 0:
            raise ValueError("没有子模型")

        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # 简单平均（等权）
        sum_probs = np.zeros(len(X))
        for model in self.sub_models:
            try:
                prob = model.predict_proba(X)[:, 1]
            except (AttributeError, TypeError):
                prob = model.predict(X)
                if hasattr(model, 'decision_function'):
                    prob = 1 / (1 + np.exp(-model.decision_function(X)))
                else:
                    prob = prob.astype(float)
            sum_probs += prob

        return sum_probs / self.n_models

    def predict(self, X, threshold=0.5):
        """预测类别"""
        prob = self.predict_proba(X)
        return (prob >= threshold).astype(int)

    def get_info(self):
        """返回模型信息"""
        return {
            'level_name': self.level_name,
            'fingerprint': self.fingerprint,
            'n_models': self.n_models,
            'model_names': self.model_names,
            'strategy': 'simple_average',
            'performance': self.performance
        }

    def __repr__(self):
        info = self.get_info()
        return (f"ConsensusModel(\n"
                f"  level='{info['level_name']}',\n"
                f"  fingerprint='{info['fingerprint']}',\n"
                f"  strategy='{info['strategy']}',\n"
                f"  n_models={info['n_models']},\n"
                f"  models={info['model_names']}\n"
                f")")


# ============================================================================
# ==================== 第三步：辅助函数 ======================================
# ============================================================================

def load_label_file(label_file, expected_len):
    """加载标签文件"""
    try:
        if label_file.endswith('.xls') or label_file.endswith('.xlsx'):
            df_temp = pd.read_excel(label_file, header=0)
        else:
            df_temp = pd.read_csv(label_file, header=0)
        raw_data = df_temp.iloc[:, 0].values.flatten()
    except Exception:
        try:
            if label_file.endswith('.xls') or label_file.endswith('.xlsx'):
                df_temp = pd.read_excel(label_file, header=None)
            else:
                df_temp = pd.read_csv(label_file, header=None)
            raw_data = df_temp.iloc[:, 0].values.flatten()
        except Exception as e:
            raise ValueError(f"无法读取标签文件: {e}")

    raw_data = [x for x in raw_data if pd.notna(x)]
    numeric_data = []
    for x in raw_data:
        try:
            numeric_data.append(float(x))
        except (ValueError, TypeError):
            continue

    if len(numeric_data) != expected_len:
        if len(numeric_data) > expected_len:
            numeric_data = numeric_data[:expected_len]
        else:
            while len(numeric_data) < expected_len:
                numeric_data.append(0)

    return np.array(numeric_data)


def load_model_file(file_path):
    """
    ★★★ 通用模型加载函数 ★★★
    自动识别文件格式（.pkl 或 .h5），返回模型对象
    """
    if not os.path.exists(file_path):
        return None, f"文件不存在: {file_path}"

    try:
        if file_path.endswith('.h5'):
            try:
                from tensorflow.keras.models import load_model as load_keras_model
                model = load_keras_model(file_path)
                return model, None
            except ImportError:
                try:
                    from keras.models import load_model as load_keras_model
                    model = load_keras_model(file_path)
                    return model, None
                except ImportError as e:
                    return None, f"无法加载.h5文件，请安装tensorflow或keras: {str(e)}"
        else:
            model = joblib.load(file_path)
            return model, None
    except Exception as e:
        return None, f"加载失败: {str(e)}"


# ============================================================================
# ==================== 第四步：核心函数 ======================================
# ============================================================================

def load_data(config):
    """加载测试集、标签和所有模型的预测概率（.npy）"""
    level = config['level_name']
    print(f"\n  ┌─ 加载数据: {level}")

    test_file = config['test_file']
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"✗ 测试集文件不存在: {test_file}")

    if test_file.endswith('.xls') or test_file.endswith('.xlsx'):
        df = pd.read_excel(test_file)
    else:
        df = pd.read_csv(test_file)

    print(f"  │  ✓ 测试集: {len(df)} 行, {len(df.columns)} 列")

    id_col = config.get('id_col', None)
    if id_col is None or id_col not in df.columns:
        df.insert(0, 'Compound_ID', range(1, len(df) + 1))
        print(f"  │  ✓ 已自动生成 Compound_ID (1~{len(df)})")

    label_col = config.get('label_col', 'True_Label')
    label_file = config.get('label_file', None)

    if label_file and os.path.exists(label_file):
        y_true = load_label_file(label_file, len(df))
        df[label_col] = y_true
        print(f"  │  ✓ 从标签文件加载: {os.path.basename(label_file)}")
    elif label_col in df.columns:
        print(f"  │  ✓ 从测试集找到标签列: {label_col}")
    else:
        raise ValueError(f"✗ 未找到标签: 请检查 label_file 路径或 label_col 列名")

    print(f"  │  ✓ 标签列: {label_col}")

    loaded_count = 0
    for model_name, prob_path in config['models'].items():
        if not os.path.exists(prob_path):
            print(f"  │  ⚠️ 文件不存在: {os.path.basename(prob_path)}")
            continue

        prob = np.load(prob_path).flatten()

        if len(prob) != len(df):
            print(f"  │  ⚠️ {model_name}: 长度({len(prob)})与测试集({len(df)})不一致")
            if len(prob) > len(df):
                prob = prob[:len(df)]
            else:
                prob = np.pad(prob, (0, len(df) - len(prob)),
                              constant_values=np.mean(prob) if len(prob) > 0 else 0.5)

        df[model_name] = prob
        loaded_count += 1
        print(f"  │  ✓ {model_name}: 加载完成")

    print(f"  └─ 共加载 {loaded_count}/{len(config['models'])} 个模型")
    return df


def simple_average_consensus(df, config):
    """
    简单平均集成（无权重，无参数）
    """
    level = config['level_name']
    selected_models = config['selected_models']

    available_models = [m for m in selected_models if m in df.columns]

    print(f"\n  ┌─ 构建共识模型: {level}")
    print(f"  │  入选模型: {len(selected_models)} 个")
    print(f"  │  集成策略: 简单平均（Simple Averaging）")
    print(f"  │  无需权重，无参数")

    if len(available_models) == 0:
        print(f"  │  ❌ 没有可用模型，跳过集成")
        return df

    missing_models = [m for m in selected_models if m not in available_models]
    if missing_models:
        print(f"  │  ⚠️ 缺失模型: {missing_models}")

    for m in available_models:
        print(f"  │    - {m}: 权重=1/{len(available_models):.0f}")

    # ★★★ 核心：简单平均（直接平均）★★★
    prob_col = config.get('output_prob_col', f'P_{level}')
    df[prob_col] = df[available_models].mean(axis=1)

    print(f"  │  ✓ 共识概率列: {prob_col}")
    print(f"  └─ 权重和: 1.0000")
    return df


def evaluate_model(df, config):
    """评估共识模型性能"""
    level = config['level_name']
    label_col = config.get('label_col', 'True_Label')
    prob_col = config.get('output_prob_col', f'P_{level}')

    if prob_col not in df.columns:
        print(f"\n  ┌─ 性能评估: {level}")
        print(f"  │  ❌ 共识概率列不存在: {prob_col}")
        return None

    y_true = df[label_col].values
    y_pred_prob = df[prob_col].values
    y_pred_class = (y_pred_prob >= 0.5).astype(int)

    auc_value = roc_auc_score(y_true, y_pred_prob)
    acc = accuracy_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class, zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred_class)

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_class).ravel()
    except Exception:
        tn = fp = fn = tp = 0

    results = {
        'Level': level,
        'AUC': auc_value,
        'Accuracy': acc,
        'F1': f1,
        'MCC': mcc,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn
    }

    print(f"\n  ┌─ 性能评估: {level}")
    print(f"  │  AUC:      {auc_value:.4f}")
    print(f"  │  Accuracy: {acc:.4f}")
    print(f"  │  F1:       {f1:.4f}")
    print(f"  │  MCC:      {mcc:.4f}")
    print(f"  └─ 混淆矩阵: TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    return results


def save_output(df, config, output_dir):
    """保存共识概率输出文件"""
    level = config['level_name']
    id_col = 'Compound_ID'
    label_col = config.get('label_col', 'True_Label')
    prob_col = config.get('output_prob_col', f'P_{level}')

    if prob_col not in df.columns:
        print(f"\n  ⚠️ 跳过保存: {level} 没有共识概率")
        return None

    output_cols = [id_col, label_col, prob_col]
    output_df = df[output_cols].copy()

    output_file = os.path.join(output_dir, f"{level}_consensus_probabilities.csv")
    output_df.to_csv(output_file, index=False)
    print(f"\n  ✓ 输出已保存: {os.path.basename(output_file)}")
    print(f"    列名: {list(output_df.columns)}")
    print(f"    行数: {len(output_df)}")

    return output_df


# ============================================================================
# ==================== ★★★ 核心：保存完整的共识模型对象 ★★★ ================
# ============================================================================

def build_and_save_consensus_model(config, selected_models, performance):
    """
    ★★★ 加载子模型，构建完整的共识模型对象并保存 ★★★
    使用简单平均，无需权重
    """
    level = config['level_name']
    sub_model_pkls = config.get('sub_model_pkls', {})

    print(f"\n  ┌─ 构建完整共识模型对象: {level}")
    print(f"  │  集成策略: 简单平均")

    # 1. 加载入选的子模型
    sub_models = []
    model_names = []

    for model_name in selected_models:
        if model_name not in sub_model_pkls:
            print(f"  │  ⚠️ {model_name}: 未找到模型路径")
            continue

        model_path = sub_model_pkls[model_name]
        model, error = load_model_file(model_path)

        if model is not None:
            sub_models.append(model)
            model_names.append(model_name)
            if model_path.endswith('.h5'):
                print(f"  │  ✓ {model_name}: 加载成功 (Keras .h5)")
            else:
                print(f"  │  ✓ {model_name}: 加载成功 (joblib .pkl)")
        else:
            print(f"  │  ✗ {model_name}: {error}")

    if len(sub_models) == 0:
        print(f"  │  ❌ 没有可用的子模型，无法构建完整共识模型")
        config_dict = {
            'type': 'simple_average',
            'level_name': level,
            'fingerprint': config.get('fingerprint', 'Unknown'),
            'selected_models': selected_models,
            'performance': performance,
            'has_sub_models': False,
            'warning': '此文件仅为配置，不含子模型对象，无法直接.predict_proba()'
        }
        save_path = os.path.join(MODEL_OUTPUT_DIR, f"{level}_consensus_model.pkl")
        joblib.dump(config_dict, save_path)
        print(f"  │  ✓ 配置文件已保存: {save_path}")
        return config_dict

    # 2. 构建共识模型对象（简单平均，无需权重）
    consensus_model = ConsensusModel(
        sub_models=sub_models,
        model_names=model_names,
        level_name=level,
        fingerprint=config.get('fingerprint', 'Unknown'),
        performance=performance
    )

    # 3. 保存为 .pkl
    save_path = os.path.join(MODEL_OUTPUT_DIR, f"{level}_consensus_model.pkl")
    joblib.dump(consensus_model, save_path)

    print(f"  │  ✓ ★★★ 完整共识模型已保存: {save_path}")
    print(f"  │     包含 {len(sub_models)} 个子模型")
    print(f"  │     集成策略: 简单平均（所有模型等权）")
    print(f"  └─ 指纹: {consensus_model.fingerprint}")

    return consensus_model


# ============================================================================
# ==================== 第五步：主函数 ========================================
# ============================================================================

def build_all_consensus_models():
    """为所有层级构建共识模型，保存.csv结果和完整的.pkl模型对象"""

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("第一层共识模型构建 - 简单平均版（无权重，无参数）")
    print("=" * 70)
    print(f"概率输出目录: {OUTPUT_DIR}")
    print(f"模型输出目录: {MODEL_OUTPUT_DIR}\n")

    all_results = []
    consensus_dfs = {}
    model_performances = {}

    for config in ALL_CONFIGS:
        level = config['level_name']
        print(f"\n{'─' * 60}")
        print(f"处理层级: {level} (指纹: {config.get('fingerprint', 'Unknown')})")
        print('─' * 60)

        try:
            # 1. 加载数据
            df = load_data(config)

            # 2. 简单平均集成（无权重）
            df = simple_average_consensus(df, config)

            # 3. 评估
            results = evaluate_model(df, config)
            if results is not None:
                all_results.append(results)
                model_performances[level] = results

            # 4. 保存共识概率
            output_df = save_output(df, config, OUTPUT_DIR)
            if output_df is not None:
                consensus_dfs[level] = output_df

            # 5. ★★★ 构建并保存完整的共识模型对象 ★★★
            if results is not None:
                build_and_save_consensus_model(
                    config=config,
                    selected_models=config['selected_models'],
                    performance=results
                )

        except Exception as e:
            print(f"\n  ✗ 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================================================
    # 汇总性能
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("汇总所有层级共识模型性能")
    print("=" * 70)

    if all_results:
        summary_df = pd.DataFrame(all_results)
        print("\n" + summary_df.to_string(index=False))
        summary_file = os.path.join(OUTPUT_DIR, "consensus_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"\n✓ 性能汇总已保存: {os.path.basename(summary_file)}")

    # ========================================================================
    # 生成Tier 2输入文件
    # ========================================================================
    required_levels = ['MIE', 'KE1', 'KE2', 'AO']
    existing_levels = [l for l in required_levels if l in consensus_dfs]

    if len(existing_levels) == 4:
        print(f"\n{'=' * 70}")
        print("生成Tier 2元模型输入文件")
        print("=" * 70)

        tier2_df = consensus_dfs['MIE'][['Compound_ID', 'True_Label']].copy()

        for level in required_levels:
            df_level = consensus_dfs[level]
            prob_col = f'P_{level}'
            tier2_df = tier2_df.merge(df_level[['Compound_ID', prob_col]], on='Compound_ID', how='inner')

        tier2_df.columns = ['Compound_ID', 'True_Label', 'P_MIE', 'P_KE1', 'P_KE2', 'P_AO']

        tier2_file = os.path.join(OUTPUT_DIR, "tier2_input.csv")
        tier2_df.to_csv(tier2_file, index=False)
        print(f"✓ Tier 2输入文件已保存: {os.path.basename(tier2_file)}")
        print(f"  包含化合物数: {len(tier2_df)}")
        print(f"  列名: {list(tier2_df.columns)}")
        print("\n  前5行预览:")
        print(tier2_df.head().to_string(index=False))
    else:
        print(f"\n⚠️ 已有层级: {existing_levels}")
        print(f"  缺少层级: {[l for l in required_levels if l not in existing_levels]}")

    # ========================================================================
    # 总结
    # ========================================================================
    print(f"\n{'=' * 70}")
    print("第一层共识模型构建完成! (简单平均版)")
    print("=" * 70)

    print(f"\n📁 输出文件汇总:")
    print(f"  CSV结果目录: {OUTPUT_DIR}")
    print(f"  模型目录: {MODEL_OUTPUT_DIR}")

    print("\n  CSV文件清单:")
    for level in required_levels:
        csv_file = os.path.join(OUTPUT_DIR, f"{level}_consensus_probabilities.csv")
        if os.path.exists(csv_file):
            print(f"    ✓ {level}_consensus_probabilities.csv")

    print("\n  ★ 完整共识模型文件清单:")
    for level in required_levels:
        pkl_file = os.path.join(MODEL_OUTPUT_DIR, f"{level}_consensus_model.pkl")
        if os.path.exists(pkl_file):
            print(f"    ✓ {level}_consensus_model.pkl")

    print(f"\n💡 集成策略: 简单平均（Simple Averaging）")
    print(f"   - 无需权重，无参数")
    print(f"   - 所有入选模型等权")
    print(f"   - 无数据泄露风险")

    print(f"\n💡 外部验证时使用方式:")
    print("  import joblib")
    print("  model = joblib.load('E:/心衰AOP/第一层模型/MIE_consensus_model.pkl')")
    print("  prob = model.predict_proba(fingerprint_array)  # 直接预测!")


# ============================================================================
# ==================== 程序入口 =============================================
# ============================================================================

if __name__ == "__main__":
    build_all_consensus_models()