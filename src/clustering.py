"""
Wholesale Customer Segmentation Using K-Means
End-to-End Analysis, Modeling, Profiling, and Visualizations
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

FEATURE_COLS = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
CATEGORICAL_COLS = ['Channel', 'Region']

CHANNEL_MAP = {1: 'Horeca', 2: 'Retail'}
REGION_MAP = {1: 'Lisbon', 2: 'Oporto', 3: 'Other Region'}

CLUSTER_NAMES = {
    0: 'Broad-Category High Spenders',
    1: 'Fresh & Frozen / Horeca Focused',
    2: 'Grocery & Detergents / Retail Specialists'
}

def load_data(filepath='Wholesale_Customers.csv'):
    """Load wholesale customers dataset from path or fallback to dataset/."""
    if not os.path.exists(filepath):
        alt_path = os.path.join('dataset', filepath)
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath} or {alt_path}")
    df = pd.read_csv(filepath)
    return df

def audit_data(df, metadata_path='metadata.json'):
    """Audit dataset quality, check nulls, duplicates, metadata discrepancy."""
    audit = {}
    audit['shape'] = df.shape
    audit['columns'] = df.columns.tolist()
    audit['null_counts'] = df.isnull().sum().to_dict()
    audit['duplicate_rows'] = int(df.duplicated().sum())
    audit['negative_counts'] = (df[FEATURE_COLS] < 0).sum().to_dict()
    audit['zero_counts'] = (df[FEATURE_COLS] == 0).sum().to_dict()
    audit['min_values'] = df[FEATURE_COLS].min().to_dict()
    audit['max_values'] = df[FEATURE_COLS].max().to_dict()
    audit['channel_dist'] = df['Channel'].value_counts().to_dict()
    audit['region_dist'] = df['Region'].value_counts().to_dict()
    
    # Metadata comparison
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join('dataset', metadata_path)
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
            audit['metadata_num_features'] = meta.get('num_features')
            audit['metadata_target_col'] = meta.get('target_col')
    return audit

def run_model_comparison(df, random_state=42):
    """Evaluate K-Means, Agglomerative, and GMM across multiple preprocessing pipelines for k=2..10."""
    X_raw = df[FEATURE_COLS].values
    X_log = np.log1p(X_raw)

    scalers = {
        'Raw + StandardScaler': StandardScaler().fit_transform(X_raw),
        'Log1p + StandardScaler': StandardScaler().fit_transform(X_log),
        'Log1p + RobustScaler': RobustScaler().fit_transform(X_log)
    }

    records = []

    for prep_name, X in scalers.items():
        for k in range(2, 11):
            # K-Means
            km = KMeans(n_clusters=k, random_state=random_state, n_init=25)
            km_labels = km.fit_predict(X)
            records.append({
                'Preprocessing': prep_name,
                'Algorithm': 'K-Means',
                'k': k,
                'Inertia': km.inertia_,
                'Silhouette': silhouette_score(X, km_labels),
                'Davies_Bouldin': davies_bouldin_score(X, km_labels),
                'Calinski_Harabasz': calinski_harabasz_score(X, km_labels),
                'Cluster_Sizes': str(np.bincount(km_labels).tolist())
            })

            # Agglomerative (Ward linkage)
            agg = AgglomerativeClustering(n_clusters=k, linkage='ward')
            agg_labels = agg.fit_predict(X)
            records.append({
                'Preprocessing': prep_name,
                'Algorithm': 'Agglomerative (Ward)',
                'k': k,
                'Inertia': np.nan,
                'Silhouette': silhouette_score(X, agg_labels),
                'Davies_Bouldin': davies_bouldin_score(X, agg_labels),
                'Calinski_Harabasz': calinski_harabasz_score(X, agg_labels),
                'Cluster_Sizes': str(np.bincount(agg_labels).tolist())
            })

            # Gaussian Mixture
            gmm = GaussianMixture(n_components=k, random_state=random_state, n_init=10)
            gmm_labels = gmm.fit_predict(X)
            records.append({
                'Preprocessing': prep_name,
                'Algorithm': 'Gaussian Mixture',
                'k': k,
                'Inertia': np.nan,
                'Silhouette': silhouette_score(X, gmm_labels),
                'Davies_Bouldin': davies_bouldin_score(X, gmm_labels),
                'Calinski_Harabasz': calinski_harabasz_score(X, gmm_labels),
                'Cluster_Sizes': str(np.bincount(gmm_labels).tolist())
            })

    comparison_df = pd.DataFrame(records)
    return comparison_df

def fit_final_model(df, n_clusters=3, random_state=42):
    """
    Fit final K-Means model on Log1p + StandardScaler transformed data.
    Adds Cluster, Cluster_Name, PC1, PC2 to dataframe.
    """
    X_log = np.log1p(df[FEATURE_COLS])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)

    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=25)
    labels = km.fit_predict(X_scaled)

    # 2D PCA for visual projection and analysis
    pca = PCA(n_components=2, random_state=random_state)
    pca_coords = pca.fit_transform(X_scaled)

    df_clustered = df.copy()
    df_clustered['Cluster'] = labels
    df_clustered['Cluster_Name'] = df_clustered['Cluster'].map(CLUSTER_NAMES)
    df_clustered['PC1'] = pca_coords[:, 0]
    df_clustered['PC2'] = pca_coords[:, 1]
    df_clustered['Channel_Desc'] = df_clustered['Channel'].map(CHANNEL_MAP)
    df_clustered['Region_Desc'] = df_clustered['Region'].map(REGION_MAP)

    return df_clustered, km, scaler, pca, X_scaled

def compute_cluster_profiles(df_clustered):
    """Compute detailed summary statistics for each cluster on original raw scale."""
    profiles = []
    total_customers = len(df_clustered)

    for cluster_id in sorted(df_clustered['Cluster'].unique()):
        sub = df_clustered[df_clustered['Cluster'] == cluster_id]
        c_name = CLUSTER_NAMES.get(cluster_id, f"Cluster {cluster_id}")
        count = len(sub)
        pct = (count / total_customers) * 100

        # Channel proportions
        ch_counts = sub['Channel'].value_counts()
        horeca_count = ch_counts.get(1, 0)
        retail_count = ch_counts.get(2, 0)
        horeca_pct = (horeca_count / count) * 100
        retail_pct = (retail_count / count) * 100

        # Region proportions
        reg_counts = sub['Region'].value_counts()
        lisbon_count = reg_counts.get(1, 0)
        oporto_count = reg_counts.get(2, 0)
        other_count = reg_counts.get(3, 0)

        row = {
            'Cluster': cluster_id,
            'Cluster_Name': c_name,
            'Customer_Count': count,
            'Percentage': pct,
            'Horeca_Count': horeca_count,
            'Horeca_Pct': horeca_pct,
            'Retail_Count': retail_count,
            'Retail_Pct': retail_pct,
            'Lisbon_Count': lisbon_count,
            'Oporto_Count': oporto_count,
            'Other_Region_Count': other_count,
        }

        # Spending stats
        for col in FEATURE_COLS:
            row[f'{col}_Mean'] = sub[col].mean()
            row[f'{col}_Median'] = sub[col].median()
            row[f'{col}_Std'] = sub[col].std()
            row[f'{col}_Total'] = sub[col].sum()

        row['Total_Spending_Mean'] = sub[FEATURE_COLS].sum(axis=1).mean()
        row['Total_Spending_Median'] = sub[FEATURE_COLS].sum(axis=1).median()
        row['Total_Spending_Sum'] = sub[FEATURE_COLS].sum(axis=1).sum()

        profiles.append(row)

    return pd.DataFrame(profiles)

def generate_visualizations(df, df_clustered, comparison_df, km, scaler, pca, X_scaled, out_dir='outputs/figures'):
    """Generate and save all 11 required figures with high quality."""
    os.makedirs(out_dir, exist_ok=True)
    palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 1. spending_distributions.png
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for idx, col in enumerate(FEATURE_COLS):
        ax = axes[idx]
        sns.histplot(df[col], kde=True, ax=ax, color='#2b5c8f', bins=30)
        skew = df[col].skew()
        ax.set_title(f'{col} (Raw)\nSkewness: {skew:.2f}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Annual Spending (m.u.)', fontsize=10)
        ax.set_ylabel('Customer Count', fontsize=10)
    plt.suptitle('Distribution of Annual Spending Across Product Categories (Raw Scale)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'spending_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. spending_boxplots.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    sns.boxplot(data=df[FEATURE_COLS], ax=ax1, palette='Set2')
    ax1.set_title('Raw Spending Boxplots (Heavy Outliers)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Annual Spending (m.u.)', fontsize=11)
    ax1.tick_params(axis='x', rotation=30)

    df_log = np.log1p(df[FEATURE_COLS])
    sns.boxplot(data=df_log, ax=ax2, palette='Set2')
    ax2.set_title('Log-Transformed Spending Boxplots [log(1 + x)]', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Log Transformed Value', fontsize=11)
    ax2.tick_params(axis='x', rotation=30)

    plt.suptitle('Comparison of Spending Distributions Before and After Log Transformation', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'spending_boxplots.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. correlation_heatmap.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6.5))
    corr_raw = df[FEATURE_COLS].corr()
    corr_log = np.log1p(df[FEATURE_COLS]).corr()

    sns.heatmap(corr_raw, annot=True, fmt='.2f', cmap='Blues', vmin=-0.2, vmax=1.0, ax=ax1, square=True, cbar_kws={'shrink': 0.8})
    ax1.set_title('Raw Spending Pearson Correlation', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=30)

    sns.heatmap(corr_log, annot=True, fmt='.2f', cmap='Blues', vmin=-0.2, vmax=1.0, ax=ax2, square=True, cbar_kws={'shrink': 0.8})
    ax2.set_title('Log-Transformed Spending Pearson Correlation', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='x', rotation=30)

    plt.suptitle('Correlation Analysis of Wholesale Spending Categories', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'correlation_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. elbow_plot.png
    fig, ax = plt.subplots(figsize=(10, 6))
    km_res = comparison_df[comparison_df['Algorithm'] == 'K-Means']
    for prep in km_res['Preprocessing'].unique():
        sub = km_res[km_res['Preprocessing'] == prep]
        ax.plot(sub['k'], sub['Inertia'], marker='o', linewidth=2, label=prep)
    ax.axvline(x=2, color='gray', linestyle='--', alpha=0.7, label='Candidate k=2')
    ax.axvline(x=3, color='crimson', linestyle='--', alpha=0.7, label='Candidate k=3 (Selected)')
    ax.set_title('Elbow Plot for K-Means Clustering Across Preprocessing Strategies', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_xticks(range(2, 11))
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'elbow_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. silhouette_scores.png
    fig, ax = plt.subplots(figsize=(10, 6))
    for prep in comparison_df['Preprocessing'].unique():
        sub = comparison_df[(comparison_df['Preprocessing'] == prep) & (comparison_df['Algorithm'] == 'K-Means')]
        ax.plot(sub['k'], sub['Silhouette'], marker='s', linewidth=2, label=f'K-Means ({prep})')
    # Add Agglomerative on Log1p + StandardScaler
    sub_agg = comparison_df[(comparison_df['Preprocessing'] == 'Log1p + StandardScaler') & (comparison_df['Algorithm'] == 'Agglomerative (Ward)')]
    ax.plot(sub_agg['k'], sub_agg['Silhouette'], marker='^', linestyle=':', color='purple', linewidth=2, label='Agglomerative Ward (Log1p + Standard)')
    ax.axvline(x=3, color='crimson', linestyle='--', alpha=0.6, label='Selected k=3')
    ax.set_title('Silhouette Score by Number of Clusters (k)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_xticks(range(2, 11))
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'silhouette_scores.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 6. model_comparison.png
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5.5))
    prep_target = 'Log1p + StandardScaler'
    sub_log = comparison_df[comparison_df['Preprocessing'] == prep_target]

    for algo in sub_log['Algorithm'].unique():
        data_algo = sub_log[sub_log['Algorithm'] == algo]
        ax1.plot(data_algo['k'], data_algo['Silhouette'], marker='o', label=algo)
        ax2.plot(data_algo['k'], data_algo['Davies_Bouldin'], marker='s', label=algo)
        ax3.plot(data_algo['k'], data_algo['Calinski_Harabasz'], marker='^', label=algo)

    ax1.set_title('Silhouette Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('k', fontsize=11)
    ax1.set_xticks(range(2, 11))
    ax1.legend(fontsize=9)

    ax2.set_title('Davies-Bouldin Index (Lower is Better)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('k', fontsize=11)
    ax2.set_xticks(range(2, 11))
    ax2.legend(fontsize=9)

    ax3.set_title('Calinski-Harabasz Score (Higher is Better)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('k', fontsize=11)
    ax3.set_xticks(range(2, 11))
    ax3.legend(fontsize=9)

    plt.suptitle(f'Comprehensive Multi-Metric Model Comparison ({prep_target})', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 7. pca_clusters.png
    fig, ax = plt.subplots(figsize=(11, 7.5))
    var_exp = pca.explained_variance_ratio_ * 100
    cluster_colors = {0: '#2b5c8f', 1: '#2ca02c', 2: '#d62728'}

    for c in sorted(df_clustered['Cluster'].unique()):
        sub = df_clustered[df_clustered['Cluster'] == c]
        ax.scatter(sub['PC1'], sub['PC2'], color=cluster_colors[c], label=f'Cluster {c}: {CLUSTER_NAMES[c]} (n={len(sub)})',
                   alpha=0.7, edgecolors='white', s=70)

    # Plot cluster centers in PCA space
    centers_scaled = km.cluster_centers_
    centers_pca = pca.transform(centers_scaled)
    ax.scatter(centers_pca[:, 0], centers_pca[:, 1], c='black', marker='X', s=200, edgecolors='yellow', linewidth=1.5,
               label='Cluster Centroids', zorder=5)

    # Add feature loading vectors as biplot
    loadings = pca.components_.T * 3.5  # scale for visibility
    for i, feature in enumerate(FEATURE_COLS):
        ax.arrow(0, 0, loadings[i, 0], loadings[i, 1], color='#444444', alpha=0.8,
                 head_width=0.12, head_length=0.15, linewidth=1.5)
        ax.text(loadings[i, 0] * 1.15, loadings[i, 1] * 1.15, feature, color='#111111',
                fontsize=11, fontweight='bold', ha='center', va='center')

    ax.set_title(f'2D PCA Projection of Customer Clusters (Total Explained Variance: {var_exp.sum():.1f}%)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel(f'Principal Component 1 ({var_exp[0]:.2f}% Variance) - Pantry & Retail Goods', fontsize=12)
    ax.set_ylabel(f'Principal Component 2 ({var_exp[1]:.2f}% Variance) - Perishables & Horeca Goods', fontsize=12)
    ax.legend(fontsize=9.5, loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'pca_clusters.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 8. cluster_profile_heatmap.png
    fig, ax = plt.subplots(figsize=(10, 6))
    df_std = pd.DataFrame(X_scaled, columns=FEATURE_COLS)
    df_std['Cluster'] = df_clustered['Cluster'].values
    cluster_std_mean = df_std.groupby('Cluster').mean()
    cluster_std_mean.index = [f'Cluster {i}: {CLUSTER_NAMES[i]}' for i in cluster_std_mean.index]

    sns.heatmap(cluster_std_mean, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax,
                cbar_kws={'label': 'Standardized Mean Deviation (Z-score)'}, linewidths=0.5)
    ax.set_title('Cluster Profile Heatmap (Standardized Feature Means)', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=25)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cluster_profile_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 9. cluster_sizes.png
    fig, ax = plt.subplots(figsize=(9, 5.5))
    counts = df_clustered['Cluster'].value_counts().sort_index()
    pcts = (counts / len(df_clustered)) * 100
    labels = [f'Cluster {i}\n({CLUSTER_NAMES[i]})' for i in counts.index]
    bars = ax.bar(labels, counts.values, color=['#2b5c8f', '#2ca02c', '#d62728'], edgecolor='black', alpha=0.85, width=0.55)

    for bar, count, pct in zip(bars, counts.values, pcts.values):
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 5, f'{count} clients\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylim(0, max(counts.values) * 1.2)
    ax.set_title('Customer Distribution Across Segments (N = 440)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Wholesale Clients', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'cluster_sizes.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 10. spending_profiles_raw.png
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 11), sharex=True)
    mean_spends = df_clustered.groupby('Cluster')[FEATURE_COLS].mean()
    median_spends = df_clustered.groupby('Cluster')[FEATURE_COLS].median()

    x = np.arange(len(FEATURE_COLS))
    width = 0.25

    for c in sorted(df_clustered['Cluster'].unique()):
        offset = (c - 1) * width
        ax1.bar(x + offset, mean_spends.loc[c], width, label=f'Cluster {c}: {CLUSTER_NAMES[c]}',
                color=cluster_colors[c], alpha=0.85, edgecolor='black')
        ax2.bar(x + offset, median_spends.loc[c], width, label=f'Cluster {c}: {CLUSTER_NAMES[c]}',
                color=cluster_colors[c], alpha=0.85, edgecolor='black')

    ax1.set_title('Mean Annual Spending per Category by Cluster (Original Scale in m.u.)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Mean Spending (m.u.)', fontsize=11)
    ax1.legend(fontsize=10)

    ax2.set_title('Median Annual Spending per Category by Cluster (Original Scale in m.u.)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Median Spending (m.u.)', fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels(FEATURE_COLS, fontsize=11)
    ax2.legend(fontsize=10)

    plt.suptitle('Spending Profiles by Cluster: Mean vs Median Comparison', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'spending_profiles_raw.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 11. channel_region_profiles.png
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Channel crosstab
    ch_tab = pd.crosstab(df_clustered['Cluster'], df_clustered['Channel_Desc'], normalize='index') * 100
    ch_tab.index = [f'Cluster {i}\n({CLUSTER_NAMES[i][:15]}...)' for i in ch_tab.index]
    ch_tab[['Horeca', 'Retail']].plot(kind='bar', stacked=True, ax=ax1, color=['#3498db', '#e67e22'], edgecolor='black', alpha=0.85)
    ax1.set_title('Channel Composition per Cluster (%)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Proportion (%)', fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.tick_params(axis='x', rotation=15)
    for c in ax1.containers:
        ax1.bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontweight='bold', fontsize=10)

    # Region crosstab
    reg_tab = pd.crosstab(df_clustered['Cluster'], df_clustered['Region_Desc'], normalize='index') * 100
    reg_tab.index = [f'Cluster {i}\n({CLUSTER_NAMES[i][:15]}...)' for i in reg_tab.index]
    reg_cols = ['Lisbon', 'Oporto', 'Other Region']
    reg_tab = reg_tab[[c for c in reg_cols if c in reg_tab.columns]]
    reg_tab.plot(kind='bar', stacked=True, ax=ax2, color=['#9b59b6', '#1abc9c', '#34495e'], edgecolor='black', alpha=0.85)
    ax2.set_title('Region Composition per Cluster (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Proportion (%)', fontsize=11)
    ax2.set_ylim(0, 100)
    ax2.tick_params(axis='x', rotation=15)
    for c in ax2.containers:
        ax2.bar_label(c, fmt='%.1f%%', label_type='center', color='white', fontweight='bold', fontsize=9.5)

    plt.suptitle('Customer Channel and Geographic Distribution Across Segments', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'channel_region_profiles.png'), dpi=150, bbox_inches='tight')
    plt.close()

def main():
    print("=== Starting Wholesale Customer Segmentation Pipeline ===")
    df = load_data('Wholesale_Customers.csv')
    print(f"Data successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns.")

    print("\n--- Auditing Data ---")
    audit = audit_data(df)
    print(f"Null values: {sum(audit['null_counts'].values())}")
    print(f"Duplicate rows: {audit['duplicate_rows']}")
    print(f"Negative values: {sum(audit['negative_counts'].values())}")
    print(f"Zero values: {sum(audit['zero_counts'].values())}")
    print(f"Metadata feature count: {audit.get('metadata_num_features')} vs CSV: {len(audit['columns'])}")

    print("\n--- Running Multi-Model & Preprocessing Experiments ---")
    comparison_df = run_model_comparison(df, random_state=42)
    os.makedirs('outputs', exist_ok=True)
    comparison_df.to_csv('outputs/model_comparison.csv', index=False)
    print("Model comparison saved to outputs/model_comparison.csv")

    print("\n--- Fitting Final K-Means Model (k=3, Log1p + StandardScaler) ---")
    df_clustered, km, scaler, pca, X_scaled = fit_final_model(df, n_clusters=3, random_state=42)
    df_clustered.to_csv('outputs/clustered_customers.csv', index=False)
    print("Clustered customer data saved to outputs/clustered_customers.csv")

    print("\n--- Computing Segment Profiles ---")
    profiles_df = compute_cluster_profiles(df_clustered)
    profiles_df.to_csv('outputs/cluster_profiles.csv', index=False)
    print("Cluster profiles saved to outputs/cluster_profiles.csv")

    print("\n--- Generating Deliverable Visualizations ---")
    generate_visualizations(df, df_clustered, comparison_df, km, scaler, pca, X_scaled, out_dir='outputs/figures')
    print("All 11 figures successfully generated and saved to outputs/figures/")

    print("\n=== Pipeline Execution Completed Successfully ===")

if __name__ == '__main__':
    main()
