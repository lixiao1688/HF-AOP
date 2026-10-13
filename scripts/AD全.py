import warnings
import os
import gc
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 彻底解决matplotlib字体和编码问题
warnings.filterwarnings('ignore')

# 重置并设置matplotlib配置 - 确保高清晰度
matplotlib.rcParams.clear()
matplotlib.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'axes.unicode_minus': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
})

# 设置英文环境
os.environ['LANG'] = 'en_US.UTF-8'

# RDKit导入
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit import DataStructs

# ============= 配置 =============
OUTPUT_DIR = "E:/心衰AOP/AD分析"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def read_smiles(file_path):
    """读取SMILES文件"""
    try:
        df = pd.read_csv(file_path, encoding='utf-8')

        # 自动检测SMILES列
        smiles_col = None
        for col in ['SMILES', 'smiles', 'Smiles', 'SMILE']:
            if col in df.columns:
                smiles_col = col
                break
        if smiles_col is None:
            smiles_col = df.columns[0]

        smiles_list = df[smiles_col].dropna().astype(str).tolist()
        smiles_list = [s.strip() for s in smiles_list if s.strip()]
        print(f"读取 {len(smiles_list)} 个分子")
        return smiles_list
    except Exception as e:
        print(f"读取文件出错: {e}")
        return []


def calc_fingerprints(smiles_list):
    """计算MACCS指纹"""
    fps = []
    valid = []
    for smile in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                fps.append(MACCSkeys.GenMACCSKeys(mol))
                valid.append(smile)
        except:
            continue
    print(f"有效分子: {len(fps)}/{len(smiles_list)}")
    return fps, valid


def calc_sim_matrix(fps):
    """计算相似度矩阵"""
    n = len(fps)
    sim = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        sim[i][i] = 1.0
        for j in range(i + 1, n):
            s = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim[i][j] = s
            sim[j][i] = s
    return sim


def calc_sim_metrics(sim):
    """计算相似度统计指标"""
    n = sim.shape[0]
    metrics = []
    for i in range(n):
        sims = [sim[i][j] for j in range(n) if j != i]
        metrics.append({
            'mean': np.mean(sims),
            'max': np.max(sims),
            'min': np.min(sims),
            'std': np.std(sims),
            'median': np.median(sims)
        })
    return pd.DataFrame(metrics)


def calc_descriptors(smiles):
    """计算分子描述符"""
    desc = []
    for smile in smiles:
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol:
                desc.append([
                    mol.GetNumAtoms(),
                    mol.GetNumBonds(),
                    sum(a.GetMass() for a in mol.GetAtoms()),
                    mol.GetRingInfo().NumRings()
                ])
        except:
            continue
    return np.array(desc)


# ==================== 杠杆值AD分析 ====================

def leverage_analysis(descriptors, name):
    """杠杆值AD分析 - AD内点占60-70%区域，AD外点占30-40%区域"""
    if len(descriptors) < 10:
        return None

    X_train, X_test = train_test_split(descriptors, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    X_train_w = np.column_stack([np.ones(len(X_train_s)), X_train_s])
    X_test_w = np.column_stack([np.ones(len(X_test_s)), X_test_s])

    leverage = np.diag(X_test_w @ np.linalg.pinv(X_train_w.T @ X_train_w) @ X_test_w.T)

    n, p = X_train.shape
    threshold = 3 * (p + 1) / n

    outside = leverage > threshold
    pct_outside = outside.sum() / len(leverage) * 100

    print(f"  杠杆值AD: 阈值={threshold:.4f}, 超出={outside.sum()}/{len(leverage)} ({pct_outside:.2f}%)")

    # 计算AD内点的最大值
    inside_leverage = leverage[leverage <= threshold]
    if len(inside_leverage) > 0:
        inside_max = inside_leverage.max()
    else:
        inside_max = threshold

    # 计算所有点的最大值
    all_max = leverage.max()

    # 策略：让AD内点占纵坐标的约67%（y_max = inside_max * 1.5）
    y_max = inside_max * 1.5

    # 如果这样设置后AD外点显示不全，则调整，但保持AD内点至少占60%
    if y_max < all_max:
        # 需要显示AD外点，将y_max设置为all_max的1.05倍
        y_max = all_max * 1.05

    y_min = 0

    # 智能选择刻度间距
    if inside_max <= 0.05:
        tick_step = 0.005
    elif inside_max <= 0.1:
        tick_step = 0.01
    elif inside_max <= 0.2:
        tick_step = 0.02
    else:
        tick_step = 0.05

    # 生成刻度值
    y_ticks = np.arange(y_min, y_max + tick_step, tick_step)
    if len(y_ticks) > 15:
        y_ticks = y_ticks[::len(y_ticks) // 15 + 1]

    # 创建图形
    fig = plt.figure(figsize=(12, 6), dpi=300, facecolor='white')
    ax = fig.add_subplot(111)

    indices = np.arange(len(leverage))

    # AD内的点（蓝色）
    inside_mask = ~outside
    if inside_mask.any():
        ax.scatter(indices[inside_mask], leverage[inside_mask], alpha=0.6, s=25,
                   c='steelblue', label=f'Inside AD (n={inside_mask.sum()})',
                   edgecolors='none')

    # AD外的点（红色X）
    if outside.any():
        ax.scatter(indices[outside], leverage[outside], alpha=0.8, s=35,
                   c='red', marker='x', label=f'Outside AD (n={outside.sum()})',
                   linewidth=1.5)

    # 阈值线
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2.5,
               label=f'Warning Threshold ({threshold:.4f})')

    ax.set_xlabel('Sample Index', fontsize=13, fontweight='bold')
    ax.set_ylabel('Leverage Value', fontsize=13, fontweight='bold')
    ax.set_title(f'{name} - Leverage Distribution Plot', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11, frameon=True, fancybox=True, shadow=True)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)

    ax.set_yticks(y_ticks)
    y_labels = [f'{val:.3f}' for val in y_ticks]
    ax.set_yticklabels(y_labels, fontsize=10)
    ax.set_ylim(y_min, y_max)

    # 横坐标
    n_samples = len(leverage)
    if n_samples > 100:
        step = max(1, n_samples // 10)
        xticks = list(range(0, n_samples, step))
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks], fontsize=9)

    plt.tight_layout()

    # 保存图片
    png_path = os.path.join(OUTPUT_DIR, f'{name}_leverage.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')

    pdf_path = os.path.join(OUTPUT_DIR, f'{name}_leverage.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', format='pdf')

    plt.close(fig)

    # 保存数据
    data_df = pd.DataFrame({
        'Sample_Index': indices,
        'Leverage_Value': leverage,
        'Outside_AD': outside,
        'Threshold': threshold
    })
    data_path = os.path.join(OUTPUT_DIR, f'{name}_leverage_data.csv')
    data_df.to_csv(data_path, index=False)

    print(f"  纵坐标范围: [{y_min:.3f}, {y_max:.3f}]")
    print(f"  AD内点最大值: {inside_max:.4f} (占纵坐标的 {inside_max / y_max * 100:.1f}%)")
    print(f"  已保存: {png_path}")

    return png_path, pdf_path, data_path


def similarity_analysis(metrics, name):
    """相似度AD分析"""
    if metrics.empty:
        return []

    print(f"  相似度AD: 均值={metrics['mean'].mean():.3f}, 最大={metrics['max'].mean():.3f}")

    files = []

    # 1. 相似度分布直方图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=300, facecolor='white')
    fig.suptitle(f'{name} - Similarity Distribution Analysis', fontsize=16, fontweight='bold')

    config = [('mean', 'Mean Similarity', 'steelblue'),
              ('max', 'Maximum Similarity', 'coral'),
              ('min', 'Minimum Similarity', 'seagreen'),
              ('std', 'Std Deviation', 'orchid')]

    for i, (metric, title, color) in enumerate(config):
        ax = axes[i // 2, i % 2]
        ax.hist(metrics[metric], bins=50, edgecolor='black', alpha=0.7, color=color, linewidth=0.5)
        ax.axvline(metrics[metric].median(), color='red', linestyle='--', linewidth=2,
                   label=f'Median: {metrics[metric].median():.3f}')
        ax.axvline(metrics[metric].mean(), color='orange', linestyle=':', linewidth=2,
                   label=f'Mean: {metrics[metric].mean():.3f}')
        ax.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'{title} (n={len(metrics)})', fontsize=13)
        ax.legend(fontsize=10, frameon=True)
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, f'{name}_similarity_hist.png')
    plt.savefig(path1, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    files.append(path1)

    # 2. 相似度统计箱线图
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300, facecolor='white')
    data = [metrics['mean'], metrics['median'], metrics['std'], metrics['max'], metrics['min']]
    labels = ['Mean', 'Median', 'Std', 'Max', 'Min']
    colors = ['steelblue', 'coral', 'seagreen', 'orchid', 'goldenrod']

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.5)

    for whisker in bp['whiskers']:
        whisker.set_linewidth(1.5)
    for cap in bp['caps']:
        cap.set_linewidth(1.5)
    for median in bp['medians']:
        median.set_linewidth(2.5)
        median.set_color('black')

    ax.set_ylabel('Similarity Score', fontsize=13, fontweight='bold')
    ax.set_title(f'{name} - Similarity Statistics Boxplot', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y', linestyle='-', linewidth=0.8)

    ax.text(0.98, 0.97, f'n = {len(metrics)}', transform=ax.transAxes,
            fontsize=11, ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6, edgecolor='gray'))

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, f'{name}_similarity_box.png')
    plt.savefig(path2, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    files.append(path2)

    # 3. 阈值效应分析
    thresholds = np.arange(0.1, 0.95, 0.05)
    coverage = [(metrics['max'] >= t).sum() / len(metrics) * 100 for t in thresholds]

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300, facecolor='white')
    ax.plot(thresholds, coverage, 'b-o', linewidth=2.5, markersize=8, markerfacecolor='white',
            markeredgewidth=2, markeredgecolor='blue')
    ax.fill_between(thresholds, coverage, alpha=0.2, color='blue')
    ax.set_xlabel('Similarity Threshold', fontsize=13, fontweight='bold')
    ax.set_ylabel('Coverage (%)', fontsize=13, fontweight='bold')
    ax.set_title(f'{name} - AD Coverage vs Similarity Threshold', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8)
    ax.set_ylim(0, 105)
    ax.set_xlim(0.05, 0.95)

    plt.tight_layout()
    path3 = os.path.join(OUTPUT_DIR, f'{name}_threshold.png')
    plt.savefig(path3, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    files.append(path3)

    # 保存数据
    df = pd.DataFrame({
        'threshold': thresholds,
        'coverage_percent': coverage,
        'n_inside': [(metrics['max'] >= t).sum() for t in thresholds],
        'n_outside': [len(metrics) - (metrics['max'] >= t).sum() for t in thresholds]
    })
    path4 = os.path.join(OUTPUT_DIR, f'{name}_threshold_data.csv')
    df.to_csv(path4, index=False)
    files.append(path4)

    print(f"  已保存相似度分析图片 (300 DPI)")

    return files


# ==================== 主函数 ====================

def ad_analysis(smiles_list, dataset_name):
    """AD分析主函数"""
    print(f"\n{'=' * 50}")
    print(f"分析: {dataset_name}")
    print(f"{'=' * 50}")

    fps, valid = calc_fingerprints(smiles_list)
    if len(fps) < 2:
        print("有效分子不足")
        return []

    sim = calc_sim_matrix(fps)
    metrics = calc_sim_metrics(sim)
    files = similarity_analysis(metrics, dataset_name)

    desc = calc_descriptors(valid)
    if len(desc) > 0:
        leverage_png, leverage_pdf, leverage_data = leverage_analysis(desc, dataset_name)
        if leverage_png:
            files.extend([leverage_png, leverage_pdf, leverage_data])

    del fps, sim, metrics, desc
    gc.collect()

    return files


def batch_analysis(file_list, names):
    """批量分析"""
    all_files = []
    results = {}

    for file_path, name in zip(file_list, names):
        if not os.path.exists(file_path):
            print(f"\n文件不存在: {file_path}")
            continue

        print(f"\n{'=' * 60}")
        print(f"处理数据集: {name}")
        print(f"{'=' * 60}")

        smiles = read_smiles(file_path)
        if smiles:
            files = ad_analysis(smiles, name)
            results[name] = {'n': len(smiles), 'files': files}
            all_files.extend(files)

    return results, all_files


# ==================== 运行 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("AD适用域分析 - 300 DPI高清晰度版")
    print("=" * 60)
    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"图片DPI: 300")

    datasets = [
        ("E:/心衰AOP/预处理数据/MIE 去重.csv", "MIE"),
        ("E:/心衰AOP/预处理数据/KE1 去重.csv", "KE1"),
        ("E:/心衰AOP/预处理数据/KE2 去重.csv", "KE2"),
        ("E:/心脏毒性数据/hf SMILES原始数据.csv", "AO"),
    ]

    print(f"\n将要处理的数据集:")
    for file_path, name in datasets:
        exists = "✓" if os.path.exists(file_path) else "✗"
        print(f"  {exists} {name}: {file_path}")

    results, all_files = batch_analysis(
        [d[0] for d in datasets],
        [d[1] for d in datasets]
    )

    print("\n" + "=" * 60)
    print("分析完成!")
    print("=" * 60)

    for name, data in results.items():
        print(f"\n{name}: {data['n']} 个分子")
        print(f"  生成 {len(data['files'])} 个文件:")
        for f in data['files']:
            if os.path.exists(f):
                size = os.path.getsize(f)
                if size > 1024 * 1024:
                    size_str = f"{size / (1024 * 1024):.2f} MB"
                else:
                    size_str = f"{size / 1024:.2f} KB"
                print(f"    - {os.path.basename(f)} ({size_str})")

    print(f"\n输出目录: {OUTPUT_DIR}")
    print(f"共生成 {len(all_files)} 个文件")
    print("所有图片均为 300 DPI 高清晰度")