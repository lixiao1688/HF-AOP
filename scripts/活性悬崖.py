"""
活性悬崖分析 - 完整版（含全部可视化图表）
生成内容：
    1. 汇总柱状图（活性悬崖数量 + 平均相似度）
    2. 各层级详细分布图（直方图 + 箱线图）
    3. 各层级散点图（相似度分布）
    4. 分子结构对比图（Top 3 活性悬崖对）
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import Draw
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 字体设置（确保在不同系统上正确显示）
# ============================================================================

def setup_chinese_font():
    """设置中文字体"""
    import platform
    system = platform.system()

    if system == 'Windows':
        font_names = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    elif system == 'Darwin':  # macOS
        font_names = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
    else:  # Linux
        font_names = ['WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']

    import matplotlib.font_manager as fm
    available = [f.name for f in fm.fontManager.ttflist]

    for font in font_names:
        if font in available:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            return font

    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    return None

setup_chinese_font()

# ============================================================================
# 配置
# ============================================================================

OUTPUT_DIR = r"E:/心衰AOP/活性悬崖"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 四个层级的训练集文件
LAYER_FILES = {
    'MIE': r"E:/心衰AOP/预处理数据/MIE 去重.csv",
    'KE1': r"E:/心衰AOP/预处理数据/KE1 去重.csv",
    'KE2': r"E:/心衰AOP/预处理数据/KE2 去重.csv",
    'AO': r"E:/心脏毒性数据/hf SMILES原始数据.csv",
}

SIMILARITY_THRESHOLD = 0.70
MAX_COMPOUNDS = 500


# ============================================================================
# 辅助函数
# ============================================================================

def read_smiles_and_labels(file_path):
    """读取SMILES和标签"""
    df = pd.read_csv(file_path)

    smiles_col = None
    for col in ['SMILES', 'smiles', 'Smiles', 'canonical_smiles']:
        if col in df.columns:
            smiles_col = col
            break
    if smiles_col is None:
        smiles_col = df.columns[0]

    label_col = None
    for col in ['label', 'Label', 'True_Label', 'active', 'Active']:
        if col in df.columns:
            label_col = col
            break

    smiles_list = df[smiles_col].dropna().astype(str).str.strip().tolist()

    if label_col:
        labels = df[label_col].values
        return smiles_list, labels

    return smiles_list, None


def generate_maccs_fingerprints(smiles_list):
    """生成MACCS指纹，同时记录有效索引"""
    fps = []
    valid_smiles = []
    valid_indices = []
    invalid_count = 0

    for idx, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(str(smiles).strip())
            if mol:
                fp = MACCSkeys.GenMACCSKeys(mol)
                fps.append(fp)
                valid_smiles.append(str(smiles).strip())
                valid_indices.append(idx)
            else:
                invalid_count += 1
        except:
            invalid_count += 1

    return fps, valid_smiles, valid_indices, invalid_count


def find_activity_cliffs(fps, labels, layer_name, sim_threshold=0.70, max_compounds=500):
    """寻找活性悬崖"""
    n = len(fps)

    # 采样
    if n > max_compounds:
        np.random.seed(42)
        pos_indices = [i for i, l in enumerate(labels) if l == 1]
        neg_indices = [i for i, l in enumerate(labels) if l == 0]

        n_pos = int(max_compounds * len(pos_indices) / n)
        n_neg = max_compounds - n_pos

        if len(pos_indices) > n_pos:
            pos_indices = np.random.choice(pos_indices, n_pos, replace=False)
        if len(neg_indices) > n_neg:
            neg_indices = np.random.choice(neg_indices, n_neg, replace=False)

        selected_indices = sorted(list(pos_indices) + list(neg_indices))
        fps_sampled = [fps[i] for i in selected_indices]
        labels_sampled = [labels[i] for i in selected_indices]
        original_indices = selected_indices
        n_comp = len(fps_sampled)
    else:
        fps_sampled = fps
        labels_sampled = labels
        original_indices = np.arange(n)
        n_comp = n

    cliffs = []
    pair_count = 0
    max_pairs = min(5000, n_comp * (n_comp - 1) // 2)

    for i in range(n_comp):
        for j in range(i + 1, n_comp):
            pair_count += 1
            if pair_count > max_pairs:
                break

            sim = TanimotoSimilarity(fps_sampled[i], fps_sampled[j])

            if sim >= sim_threshold:
                label_i = labels_sampled[i]
                label_j = labels_sampled[j]

                if label_i != label_j:
                    cliffs.append({
                        'Index1': original_indices[i],
                        'Index2': original_indices[j],
                        'Similarity': sim,
                        'Label1': int(label_i),
                        'Label2': int(label_j)
                    })

        if pair_count > max_pairs:
            break

    return pd.DataFrame(cliffs)


# ============================================================================
# 主程序
# ============================================================================

def main():
    print("=" * 70)
    print("活性悬崖分析 - 完整版")
    print("=" * 70)
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"相似性阈值: {SIMILARITY_THRESHOLD}")

    all_cliffs = {}
    layer_stats = {}
    layer_smiles = {}

    # ========================================================================
    # 分析各层级
    # ========================================================================

    for layer_name, file_path in LAYER_FILES.items():
        print(f"\n{'─'*50}")
        print(f"层级: {layer_name}")
        print(f"{'─'*50}")

        if not os.path.exists(file_path):
            print(f"  ❌ 文件不存在: {file_path}")
            continue

        smiles_list, labels = read_smiles_and_labels(file_path)
        print(f"  化合物数: {len(smiles_list)}")

        if labels is None:
            print(f"  ⚠️ 未找到标签列，跳过")
            continue

        print(f"    阳性: {sum(labels)}, 阴性: {len(labels) - sum(labels)}")

        fps, valid_smiles, valid_indices, invalid_count = generate_maccs_fingerprints(smiles_list)
        print(f"    有效: {len(fps)}, 无效: {invalid_count}")

        if len(fps) < 10:
            print(f"  ⚠️ 有效化合物太少，跳过")
            continue

        valid_labels = [labels[i] for i in valid_indices]

        cliffs_df = find_activity_cliffs(
            fps, valid_labels, layer_name,
            sim_threshold=SIMILARITY_THRESHOLD,
            max_compounds=MAX_COMPOUNDS
        )

        n_cliffs = len(cliffs_df)
        print(f"  活性悬崖对数: {n_cliffs}")

        if n_cliffs > 0:
            cliffs_df.to_csv(os.path.join(OUTPUT_DIR, f'activity_cliffs_{layer_name}.csv'), index=False)
            all_cliffs[layer_name] = cliffs_df
            layer_stats[layer_name] = {
                'n_cliffs': n_cliffs,
                'mean_sim': cliffs_df['Similarity'].mean(),
                'std_sim': cliffs_df['Similarity'].std(),
                'max_sim': cliffs_df['Similarity'].max(),
                'min_sim': cliffs_df['Similarity'].min()
            }
            layer_smiles[layer_name] = valid_smiles
        else:
            all_cliffs[layer_name] = pd.DataFrame()
            layer_stats[layer_name] = {
                'n_cliffs': 0,
                'mean_sim': 0,
                'std_sim': 0,
                'max_sim': 0,
                'min_sim': 0
            }
            layer_smiles[layer_name] = valid_smiles

    # ========================================================================
    # 保存汇总
    # ========================================================================

    summary_data = []
    for layer, stats in layer_stats.items():
        summary_data.append({
            'Layer': layer,
            'Cliffs_Count': stats['n_cliffs'],
            'Mean_Similarity': stats['mean_sim'],
            'Std_Similarity': stats['std_sim'],
            'Max_Similarity': stats['max_sim'],
            'Min_Similarity': stats['min_sim']
        })

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(OUTPUT_DIR, 'activity_cliffs_summary.csv'), index=False)
    print(f"\n✅ activity_cliffs_summary.csv")

    # ========================================================================
    # 图1：汇总柱状图（正文用）
    # ========================================================================

    print("\n[1] 生成汇总柱状图...")

    layers = list(layer_stats.keys())
    cliffs = [layer_stats[l]['n_cliffs'] for l in layers]
    sims = [layer_stats[l]['mean_sim'] for l in layers]
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax1 = axes[0]
    bars1 = ax1.bar(layers, cliffs, color=colors[:len(layers)], edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('AOP Layer', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Number of Activity Cliffs', fontsize=13, fontweight='bold')
    ax1.set_title('(a) Activity Cliff Counts by Layer', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, cliffs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{val}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2 = axes[1]
    bars2 = ax2.bar(layers, sims, color=colors[:len(layers)], edgecolor='black', linewidth=1.5)
    ax2.set_xlabel('AOP Layer', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Mean Tanimoto Similarity', fontsize=13, fontweight='bold')
    ax2.set_title('(b) Mean Similarity of Activity Cliffs', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars2, sims):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'activity_cliffs_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ activity_cliffs_summary.png")

    # ========================================================================
    # 图2：各层级详细分布图（直方图 + 箱线图）
    # ========================================================================

    print("\n[2] 生成各层级详细分布图...")

    for layer_name, cliffs_df in all_cliffs.items():
        if len(cliffs_df) == 0:
            print(f"  ⚠️ {layer_name}: 无活性悬崖，跳过")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 左图：相似度直方图
        ax1 = axes[0]
        ax1.hist(cliffs_df['Similarity'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.axvline(cliffs_df['Similarity'].mean(), color='red', linestyle='--',
                    linewidth=2, label=f"Mean: {cliffs_df['Similarity'].mean():.3f}")
        ax1.axvline(np.median(cliffs_df['Similarity']), color='orange', linestyle=':',
                    linewidth=2, label=f"Median: {np.median(cliffs_df['Similarity']):.3f}")
        ax1.set_xlabel('Tanimoto Similarity', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title(f'{layer_name} - Similarity Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右图：箱线图
        ax2 = axes[1]
        ax2.boxplot(cliffs_df['Similarity'], patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
        ax2.set_ylabel('Tanimoto Similarity', fontsize=12, fontweight='bold')
        ax2.set_title(f'{layer_name} - Similarity Boxplot (n={len(cliffs_df)})', fontsize=13, fontweight='bold')
        ax2.set_xticklabels(['Activity Cliffs'])
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.set_ylim(0.65, 1.0)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{layer_name}_distribution.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {layer_name}_distribution.png")

    # ========================================================================
    # 图3：各层级散点图
    # ========================================================================

    print("\n[3] 生成各层级散点图...")

    for layer_name, cliffs_df in all_cliffs.items():
        if len(cliffs_df) == 0:
            continue

        cliffs_sorted = cliffs_df.sort_values('Similarity', ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 6))

        ax.scatter(range(len(cliffs_sorted)), cliffs_sorted['Similarity'],
                   c='coral', s=40, alpha=0.7, edgecolors='black', linewidth=0.5)
        ax.axhline(y=SIMILARITY_THRESHOLD, color='gray', linestyle='--',
                   linewidth=1.5, label=f'Threshold ({SIMILARITY_THRESHOLD})')
        ax.axhline(y=cliffs_sorted['Similarity'].mean(), color='red', linestyle=':',
                   linewidth=2, label=f"Mean: {cliffs_sorted['Similarity'].mean():.3f}")

        ax.set_xlabel('Activity Cliff Pair Index (sorted by similarity)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Tanimoto Similarity', fontsize=12, fontweight='bold')
        ax.set_title(f'{layer_name} - Activity Cliffs Distribution (n={len(cliffs_sorted)})', fontsize=13, fontweight='bold')
        ax.set_ylim(0.60, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{layer_name}_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✅ {layer_name}_scatter.png")

    # ========================================================================
    # 图4：分子结构对比图（Top 3）
    # ========================================================================

    print("\n[4] 生成分子结构对比图...")

    for layer_name, cliffs_df in all_cliffs.items():
        if len(cliffs_df) == 0:
            continue

        smiles_list = layer_smiles.get(layer_name, [])
        if not smiles_list:
            continue

        top_cliffs = cliffs_df.nlargest(3, 'Similarity')

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        for idx, (_, row) in enumerate(top_cliffs.iterrows()):
            if idx >= 3:
                break

            i = int(row['Index1'])
            j = int(row['Index2'])

            mol1 = Chem.MolFromSmiles(str(smiles_list[i]).strip())
            mol2 = Chem.MolFromSmiles(str(smiles_list[j]).strip())

            if mol1 and mol2:
                img = Draw.MolsToGridImage(
                    [mol1, mol2],
                    molsPerRow=2,
                    subImgSize=(250, 250),
                    legends=[
                        f'Active={int(row["Label1"])}',
                        f'Active={int(row["Label2"])}'
                    ]
                )
                img.save(os.path.join(OUTPUT_DIR, f'{layer_name}_cliff_pair_{idx+1}.png'))
                print(f"  ✅ {layer_name}_cliff_pair_{idx+1}.png")

        plt.close()

    # ========================================================================
    # 图5：综合对比图（4层散点图拼图）
    # ========================================================================

    print("\n[5] 生成综合对比拼图...")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for ax, layer_name in zip(axes.flatten(), layers):
        cliffs_df = all_cliffs.get(layer_name, pd.DataFrame())

        if len(cliffs_df) > 0:
            cliffs_sorted = cliffs_df.sort_values('Similarity', ascending=False).reset_index(drop=True)
            ax.scatter(range(len(cliffs_sorted)), cliffs_sorted['Similarity'],
                       c='coral', s=30, alpha=0.6, edgecolors='black', linewidth=0.3)
            ax.axhline(y=SIMILARITY_THRESHOLD, color='gray', linestyle='--', linewidth=1.0)
            ax.set_title(f'{layer_name} (n={len(cliffs_sorted)})', fontsize=13, fontweight='bold')
        else:
            ax.text(0.5, 0.5, f'{layer_name}\nNo Activity Cliffs Found',
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(layer_name, fontsize=13, fontweight='bold')

        ax.set_xlabel('Cliff Pairs (sorted)', fontsize=11)
        ax.set_ylabel('Similarity', fontsize=11)
        ax.set_ylim(0.60, 1.0)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Activity Cliffs Across All AOP Layers', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_layers_cliffs_combined.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✅ all_layers_cliffs_combined.png")

    # ========================================================================
    # 最终报告
    # ========================================================================

    print("\n" + "=" * 70)
    print("活性悬崖分析完成!")
    print("=" * 70)

    print("\n汇总:")
    for layer, stats in layer_stats.items():
        print(f"  {layer}: {stats['n_cliffs']} 对活性悬崖 (平均相似性: {stats['mean_sim']:.3f})")

    print(f"\n输出目录: {OUTPUT_DIR}")
    print("\n生成的文件:")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.png') or f.endswith('.csv'):
            size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
            print(f"  ✅ {f} ({size/1024:.1f} KB)")
    print("=" * 70)


if __name__ == "__main__":
    main()